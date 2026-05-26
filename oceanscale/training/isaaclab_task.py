# mypy: ignore-errors
"""Isaac Lab 3 native DirectRLEnv for OceanScale underwater simulation.

Requires Isaac Sim 6 + Isaac Lab 3. For standalone use without Isaac Lab,
use OceanScaleDirectRLEnv from isaaclab_env.py instead.

Architecture::

    Isaac Lab 3 training loop (PPO / SAC / curriculum)
        └── OceanScaleTask(DirectRLEnv)
            ├── _setup_scene()     → Newton ROV body via InteractiveScene
            ├── _apply_action()    → Tier1 Fossen hydro → body_f
            ├── _get_observations  → body state → {"policy": tensor}
            ├── _get_rewards       → exp distance reward
            └── _get_dones         → OOB + truncation

    Newton physics step is managed by Isaac Lab's NewtonManager.
    OceanScale injects Fossen forces into body_f before each solver step.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import warp as wp

from oceanscale.vehicles.bluerov2 import BlueROV2Heavy

OBS_DIM = 20
ACT_DIM = 6

_HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None


if _HAS_ISAACLAB:
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.utils.configclass import configclass

    @configclass
    class OceanScaleTaskCfg(DirectRLEnvCfg):
        """Isaac Lab native configuration for OceanScaleTask."""

        num_envs: int = 64
        episode_length_s: float = 30.0
        decimation: int = 4
        physics_dt: float = 1 / 240
        device: str = "cuda:0"

        target_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
        init_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
        init_pos_noise: float = 0.5
        oob_distance: float = 5.0

        current_speed: float = 0.0
        current_direction: float = 0.0
        wave_height: float = 0.0
        wave_period: float = 8.0
        rho_water: float = 1025.0
        current_drag_coeff: float = 5.0

        reward_distance_scale: float = 1.0
        reward_velocity_weight: float = 0.1
        reward_action_weight: float = 0.05

        action_space: int = ACT_DIM
        observation_space: int = OBS_DIM
        state_space: int = 0

        sim: SimulationCfg = SimulationCfg(dt=1 / 240, render_interval=decimation)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=num_envs,
            env_spacing=8.0,
            replicate_physics=True,
            clone_in_fabric=True,
        )
        ui_window_class_type: type | str | None = None

        def __post_init__(self) -> None:
            self.sim.dt = self.physics_dt
            self.sim.render_interval = self.decimation
            self.scene.num_envs = self.num_envs

    class OceanScaleTask(DirectRLEnv):
        """Isaac Lab 3 native underwater ROV environment.

        Injects Fossen 6-DOF hydrodynamics into Newton's body_f array
        at each substep via _apply_action(). The physics stepping and
        scene management are handled by Isaac Lab's NewtonManager.
        """

        cfg: OceanScaleTaskCfg

        def __init__(self, cfg: OceanScaleTaskCfg, render_mode: str | None = None, **kwargs: Any):
            super().__init__(cfg, render_mode=render_mode, **kwargs)

            self._vehicle = BlueROV2Heavy()
            self._target = torch.tensor(
                cfg.target_pos, dtype=torch.float32, device=self.device
            )
            self._prev_action = torch.zeros(
                cfg.num_envs, ACT_DIM, dtype=torch.float32, device=self.device
            )

            self._init_tier1()

        def _init_tier1(self) -> None:
            from oceanscale.hydro.tier1 import Tier1
            from oceanscale.hydro.tier1_kernels import tier1_zero_wrench

            coeffs = self._vehicle.set_coeffs_kwargs()
            tier1_kw: dict[str, Any] = {"n_thrusters": self._vehicle.n_thrusters}
            if hasattr(self._vehicle, "tier1_kwargs"):
                tier1_kw.update(self._vehicle.tier1_kwargs())

            self._tier1 = Tier1(
                n_envs=self.num_envs,
                device=str(self.device),
                rho_water=self.cfg.rho_water,
                **tier1_kw,
            )
            self._tier1.set_coeffs(**coeffs)

            T_matrix = coeffs.get("T_matrix")
            self._t_pinv: torch.Tensor | None = None
            if T_matrix is not None:
                pinv = np.linalg.pinv(np.array(T_matrix, dtype=np.float64)).astype(np.float32)
                self._t_pinv = torch.from_numpy(pinv).to(self.device)

            self._u_cmd = wp.zeros(
                (self.num_envs, self._vehicle.n_thrusters),
                dtype=wp.float32, device=str(self.device),
            )
            self._quat_buf = wp.zeros(
                self.num_envs, dtype=wp.quatf, device=str(self.device),
            )
            self._zero_wrench_kernel = tier1_zero_wrench
            self._actions_buf: torch.Tensor | None = None

        def _setup_scene(self) -> None:
            pass

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            actions = actions.clamp(-1.0, 1.0)
            self._prev_action.copy_(actions)

            if self._t_pinv is not None:
                u_cmd = (actions @ self._t_pinv.T).clamp(-1.0, 1.0)
            else:
                u_cmd = torch.zeros(
                    self.num_envs, self._vehicle.n_thrusters,
                    device=self.device, dtype=torch.float32,
                )
                for i in range(min(ACT_DIM, self._vehicle.n_thrusters)):
                    u_cmd[:, i] = actions[:, i]

            self._u_cmd = wp.from_torch(u_cmd.contiguous(), dtype=wp.float32)

        def _apply_action(self) -> None:
            body_f = self.sim.physics_sim_view.body_f
            body_qd = self.sim.physics_sim_view.body_qd

            wp.launch(
                self._zero_wrench_kernel,
                dim=self.num_envs,
                inputs=[body_f],
                device=str(self.device),
            )

            body_q_t = wp.to_torch(self.sim.physics_sim_view.body_q)
            quat_dst = wp.to_torch(self._quat_buf)
            quat_dst.copy_(body_q_t[:, 3:7])

            self._tier1.compute_wrench(
                nu=body_qd,
                quat=self._quat_buf,
                u_cmd=self._u_cmd,
                dt=self.cfg.physics_dt,
            )
            self._tier1.write_to_body_f(body_f)

        def _get_observations(self) -> dict[str, torch.Tensor]:
            body_q = wp.to_torch(self.sim.physics_sim_view.body_q)
            body_qd = wp.to_torch(self.sim.physics_sim_view.body_qd)

            pos = body_q[:, :3]
            quat = body_q[:, 3:7]
            lin_vel = body_qd[:, 3:6]
            ang_vel = body_qd[:, :3]
            pos_error = self._target - pos
            depth = pos[:, 2:3]

            obs = torch.cat(
                [pos_error, quat, lin_vel, ang_vel, depth, self._prev_action],
                dim=-1,
            )
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            obs = self._get_observations()["policy"]
            pos_error = obs[:, :3]
            lin_vel = obs[:, 7:10]

            dist = torch.linalg.norm(pos_error, dim=-1)
            r_dist = torch.exp(-dist / self.cfg.reward_distance_scale)
            r_vel = -self.cfg.reward_velocity_weight * torch.linalg.norm(lin_vel, dim=-1)
            r_act = -self.cfg.reward_action_weight * torch.linalg.norm(self._prev_action, dim=-1)
            return r_dist + r_vel + r_act

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            body_q = wp.to_torch(self.sim.physics_sim_view.body_q)
            pos = body_q[:, :3]
            dist = torch.linalg.norm(self._target - pos, dim=-1)
            terminated = dist > self.cfg.oob_distance
            truncated = self.episode_length_buf >= self.max_episode_length
            return terminated, truncated

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            super()._reset_idx(env_ids)
            self._prev_action[env_ids] = 0.0

            if self.cfg.init_pos_noise > 0:
                noise = self.cfg.init_pos_noise
                body_q = wp.to_torch(self.sim.physics_sim_view.body_q)
                for i in env_ids.tolist():
                    body_q[i, 0] += torch.empty(1, device=self.device).uniform_(-noise, noise).item()
                    body_q[i, 1] += torch.empty(1, device=self.device).uniform_(-noise, noise).item()
                    body_q[i, 2] += torch.empty(1, device=self.device).uniform_(-noise, noise).item()

else:

    @dataclass
    class OceanScaleTaskCfg:
        """Portable configuration stub used when Isaac Lab is unavailable."""

        num_envs: int = 64
        episode_length_s: float = 30.0
        decimation: int = 4
        physics_dt: float = 1 / 240
        device: str = "cuda:0"

        target_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
        init_pos: tuple[float, float, float] = (0.0, 0.0, -5.0)
        init_pos_noise: float = 0.5
        oob_distance: float = 5.0

        current_speed: float = 0.0
        current_direction: float = 0.0
        wave_height: float = 0.0
        wave_period: float = 8.0
        rho_water: float = 1025.0
        current_drag_coeff: float = 5.0

        reward_distance_scale: float = 1.0
        reward_velocity_weight: float = 0.1
        reward_action_weight: float = 0.05

        action_space: int = ACT_DIM
        observation_space: int = OBS_DIM

        sim: Any = None
        scene: Any = None

    class OceanScaleTask:
        """Placeholder — Isaac Lab 3 is not installed.

        Use OceanScaleDirectRLEnv from isaaclab_env.py for standalone
        training without Isaac Lab.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "OceanScaleTask requires Isaac Sim 6 + Isaac Lab 3. "
                "Install Isaac Lab or use OceanScaleDirectRLEnv from "
                "oceanscale.training.isaaclab_env for standalone training."
            )
