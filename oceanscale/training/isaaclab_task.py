# mypy: ignore-errors
"""Isaac Lab 3 native DirectRLEnv for OceanScale underwater simulation.

Requires Isaac Sim 6 + Isaac Lab 3. For standalone use without Isaac Lab,
use OceanScaleDirectRLEnv from isaaclab_env.py instead.

Architecture::

    Isaac Lab 3 training loop (PPO / SAC / curriculum)
        └── OceanScaleTask(DirectRLEnv)
            ├── _setup_scene()         → RigidObject ROV + terrain via InteractiveScene
            ├── _pre_physics_step()    → Tier1 Fossen hydro → wrench buffers
            ├── _apply_action()        → wrench composer → body forces
            ├── _get_observations()    → body state → {"policy": tensor}
            ├── _get_rewards()         → JIT exp distance reward
            └── _get_dones()           → OOB + truncation
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


# ---------------------------------------------------------------------------
# Standalone JIT reward — no self, no Isaac Lab dependency
# ---------------------------------------------------------------------------

@torch.jit.script
def _compute_reward_jit(
    pos_error: torch.Tensor,
    lin_vel: torch.Tensor,
    prev_action: torch.Tensor,
    dist_scale: float,
    vel_weight: float,
    act_weight: float,
) -> torch.Tensor:
    dist = torch.linalg.norm(pos_error, dim=-1)
    r_dist = torch.exp(-dist / dist_scale)
    r_vel = -vel_weight * torch.linalg.norm(lin_vel, dim=-1)
    r_act = -act_weight * torch.linalg.norm(prev_action, dim=-1)
    return r_dist + r_vel + r_act


# ---------------------------------------------------------------------------
# Isaac Lab 3 native implementation
# ---------------------------------------------------------------------------

if _HAS_ISAACLAB:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
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

        robot_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.CuboidCfg(
                size=(0.457, 0.338, 0.254),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, -5.0),
            ),
        )

        ui_window_class_type: type | str | None = None

        def __post_init__(self) -> None:
            self.sim.dt = self.physics_dt
            self.sim.render_interval = self.decimation
            self.scene.num_envs = self.num_envs

    class OceanScaleTask(DirectRLEnv):
        """Isaac Lab 3 native underwater ROV environment.

        Uses the official wrench composer API to inject Fossen 6-DOF
        hydrodynamics into the simulation. Tier1 forces are computed in
        _pre_physics_step and applied via permanent_wrench_composer in
        _apply_action, matching the Quadcopter pattern.
        """

        cfg: OceanScaleTaskCfg

        def __init__(self, cfg: OceanScaleTaskCfg, render_mode: str | None = None, **kwargs: Any):
            super().__init__(cfg, render_mode=render_mode, **kwargs)

            self._body_ids, _ = self._robot.find_bodies(".*")
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
            # Pre-allocate force/torque buffers for wrench composer
            self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
            self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        def _setup_scene(self) -> None:
            self._robot = RigidObject(self.cfg.robot_cfg)
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])
            self.scene.rigid_objects["robot"] = self._robot

            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
            light_cfg.func("/World/Light", light_cfg)

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

            # Compute Tier1 hydro forces (drag, buoyancy, added mass, thrust)
            root_vel_w = self._robot.data.root_vel_w.torch
            body_qd = wp.from_torch(root_vel_w.contiguous(), dtype=wp.spatial_vectorf)

            wp.launch(
                self._zero_wrench_kernel,
                dim=self.num_envs,
                inputs=[self._tier1.wrench_buf],
                device=str(self.device),
            )

            body_q_t = self._robot.data.root_quat_w.torch
            quat_dst = wp.to_torch(self._quat_buf)
            quat_dst[:, :3].copy_(body_q_t[:, 1:4])
            quat_dst[:, 3].copy_(body_q_t[:, 0])

            self._tier1.compute_wrench(
                nu=body_qd,
                quat=self._quat_buf,
                u_cmd=self._u_cmd,
                dt=self.cfg.physics_dt,
            )

            # Extract force/torque for wrench composer
            wrench_t = wp.to_torch(self._tier1.wrench_buf).reshape(self.num_envs, 2, 3)
            self._thrust.copy_(wrench_t[:, 0:1, :])
            self._moment.copy_(wrench_t[:, 1:2, :])

        def _apply_action(self) -> None:
            self._robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=self._thrust,
                torques=self._moment,
                body_ids=self._body_ids,
                env_ids=None,
            )

        def _get_observations(self) -> dict[str, torch.Tensor]:
            pos = self._robot.data.root_pos_w.torch - self.scene.env_origins
            quat = self._robot.data.root_quat_w.torch
            root_vel = self._robot.data.root_vel_w.torch
            lin_vel = root_vel[:, :3]
            ang_vel = root_vel[:, 3:6]
            pos_error = self._target - pos
            depth = pos[:, 2:3]

            obs = torch.cat(
                [pos_error, quat, lin_vel, ang_vel, depth, self._prev_action],
                dim=-1,
            )
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            obs = self._get_observations()["policy"]
            return _compute_reward_jit(
                pos_error=obs[:, :3],
                lin_vel=obs[:, 7:10],
                prev_action=self._prev_action,
                dist_scale=self.cfg.reward_distance_scale,
                vel_weight=self.cfg.reward_velocity_weight,
                act_weight=self.cfg.reward_action_weight,
            )

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            pos = self._robot.data.root_pos_w.torch - self.scene.env_origins
            dist = torch.linalg.norm(self._target - pos, dim=-1)
            terminated = dist > self.cfg.oob_distance
            truncated = self.episode_length_buf >= self.max_episode_length
            return terminated, truncated

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            super()._reset_idx(env_ids)
            self._prev_action[env_ids] = 0.0

            root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
            root_vel = self._robot.data.default_root_vel.torch[env_ids].clone()
            root_pose[:, :3] += self.scene.env_origins[env_ids]
            if self.cfg.init_pos_noise > 0:
                noise = self.cfg.init_pos_noise
                root_pose[:, :3] += torch.empty(
                    root_pose[:, :3].shape,
                    device=self.device,
                    dtype=root_pose.dtype,
                ).uniform_(-noise, noise)

            self._robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
            self._robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)

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
        """Placeholder -- Isaac Lab 3 is not installed.

        Use OceanScaleDirectRLEnv from isaaclab_env.py for standalone
        training without Isaac Lab.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "OceanScaleTask requires Isaac Sim 6 + Isaac Lab 3. "
                "Install Isaac Lab or use OceanScaleDirectRLEnv from "
                "oceanscale.training.isaaclab_env for standalone training."
            )
