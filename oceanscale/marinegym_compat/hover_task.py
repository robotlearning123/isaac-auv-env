"""MarineGym Hover task ported to Isaac Lab 3 DirectRLEnv.

Reimplements the BlueROV hover task from MarineGym's single/hover.py using
Isaac Lab 3's DirectRLEnv base class and OceanScale's Tier1 Fossen
hydrodynamics + Newton/Warp physics.

Supports MarineGym features:
- Fossen 6-DOF hydrodynamic forces (via Tier1)
- T200 thruster dynamics
- Payload disturbance (random mass on tether joint)
- Flow disturbance (ocean current)
- Time encoding in observations
- Heading alignment + action smoothness reward

Requires Isaac Sim 6 + Isaac Lab 3. When Isaac Lab is not installed, provides
a portable configuration stub.

Ported from marinegym/envs/single/hover.py (MIT License,
Copyright (c) 2023 Botian Xu, Tsinghua University).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from oceanscale.vehicles.bluerov2 import BlueROV2MarineGym

_HAS_ISAACLAB = importlib.util.find_spec("isaaclab") is not None

OBS_DIM = 32  # rpos(3) + quat(4) + lin_vel(3) + ang_vel(3) + heading(3) + up(3) + throttle(6) + rheading(3) + time(4)
ACT_DIM = 6


@dataclass
class MarineGymHoverCfg:
    """Configuration for MarineGym Hover task."""

    num_envs: int = 64
    episode_length_s: float = 10.0
    decimation: int = 1
    physics_dt: float = 1 / 60
    device: str = "cuda:0"
    seed: int = 42

    # Vehicle
    vehicle: str = "BlueROV"  # BlueROV or BlueROVHeavy

    # Reward weights
    reward_distance_scale: float = 1.2
    reward_effort_weight: float = 0.1
    reward_action_smoothness_weight: float = 0.0

    # Time encoding
    time_encoding: bool = True
    time_encoding_dim: int = 4

    # Disturbances
    enable_payload: bool = False
    payload_mass_range: tuple[float, float] = (0.01, 0.2)
    payload_z_range: tuple[float, float] = (-0.1, 0.1)
    enable_flow: bool = False
    max_flow_velocity: list[float] = field(
        default_factory=lambda: [0.5, 0.5, 0.5, 0.0, 0.0, 0.0]
    )
    flow_noise: list[float] = field(
        default_factory=lambda: [0.1, 0.1, 0.1, 0.0, 0.0, 0.0]
    )

    # Initialization bounds
    init_pos_range: tuple[float, float] = (-2.5, 2.5)
    init_depth_range: tuple[float, float] = (1.5, 2.5)
    init_rpy_range: tuple[float, float] = (-0.2, 0.2)
    target_pos: tuple[float, float, float] = (0.0, 0.0, 2.0)

    # Termination
    min_depth: float = 0.2
    max_distance: float = 4.0

    # Ocean physics
    rho_water: float = 1025.0
    current_drag_coeff: float = 5.0

    # Isaac Lab 3 fields (only used when Isaac Lab is available)
    action_space: int = ACT_DIM
    observation_space: int = OBS_DIM
    state_space: int = 0

    # Smoothing
    alpha: float = 0.8

    def compute_obs_dim(self) -> int:
        base = 3 + 4 + 3 + 3 + 3 + 3 + ACT_DIM + 3  # rpos + quat + vel + heading + up + throttle + rheading
        if self.time_encoding:
            base += self.time_encoding_dim
        return base


if _HAS_ISAACLAB:
    import warp as wp

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.utils.configclass import configclass

    @configclass
    class MarineGymHoverIsaacCfg(DirectRLEnvCfg):
        """Isaac Lab 3 native configuration for MarineGym Hover task."""

        num_envs: int = 64
        episode_length_s: float = 10.0
        decimation: int = 1
        physics_dt: float = 1 / 60
        device: str = "cuda:0"

        action_space: int = ACT_DIM
        observation_space: int = OBS_DIM
        state_space: int = 0

        sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=1)
        scene: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=64, env_spacing=8.0, replicate_physics=True, clone_in_fabric=True
        )

        robot_cfg: RigidObjectCfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Robot",
            spawn=sim_utils.CuboidCfg(
                size=(0.457, 0.338, 0.254),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
        )

        ui_window_class_type: type | str | None = None

        def __post_init__(self) -> None:
            self.sim.dt = self.physics_dt
            self.sim.render_interval = self.decimation
            self.scene.num_envs = self.num_envs

    class MarineGymHoverTask(DirectRLEnv):
        """MarineGym Hover task as Isaac Lab 3 DirectRLEnv.

        BlueROV hover-to-target task with Fossen 6-DOF hydrodynamics,
        T200 thruster dynamics, and optional disturbances.
        """

        cfg: MarineGymHoverIsaacCfg

        def __init__(
            self,
            cfg: MarineGymHoverIsaacCfg | None = None,
            hover_cfg: MarineGymHoverCfg | None = None,
            render_mode: str | None = None,
            **kwargs: Any,
        ) -> None:
            if cfg is None:
                cfg = MarineGymHoverIsaacCfg()
            super().__init__(cfg, render_mode=render_mode, **kwargs)

            self._hover_cfg = hover_cfg or MarineGymHoverCfg()
            self._body_ids, _ = self._robot.find_bodies(".*")
            self._vehicle = BlueROV2MarineGym()
            self._target = torch.tensor(
                self._hover_cfg.target_pos, dtype=torch.float32, device=self.device
            )
            self._prev_action = torch.zeros(
                self.num_envs, ACT_DIM, dtype=torch.float32, device=self.device
            )
            self._effort = torch.zeros(
                self.num_envs, ACT_DIM, dtype=torch.float32, device=self.device
            )
            self._throttle_difference = torch.zeros(
                self.num_envs, dtype=torch.float32, device=self.device
            )

            # Heading tracking
            self._target_heading = torch.zeros(
                self.num_envs, 1, 3, device=self.device
            )

            # Stats (exponential moving average)
            self._stats = torch.zeros(
                self.num_envs, 5, dtype=torch.float32, device=self.device
            )
            # [return, pos_error, heading_alignment, uprightness, action_smoothness]

            self._init_tier1()

        def _init_tier1(self) -> None:
            from oceanscale.hydro.tier1 import Tier1
            from oceanscale.hydro.tier1_kernels import tier1_zero_wrench

            coeffs = self._vehicle.set_coeffs_kwargs()
            tier1_kw: dict[str, Any] = {"n_thrusters": self._vehicle.n_thrusters}
            tier1_kw.update(self._vehicle.tier1_kwargs())

            self._tier1 = Tier1(
                n_envs=self.num_envs,
                device=str(self.device),
                rho_water=self._hover_cfg.rho_water,
                **tier1_kw,
            )
            self._tier1.set_coeffs(**coeffs)

            T_matrix = coeffs.get("T_matrix")
            self._t_pinv: torch.Tensor | None = None
            if T_matrix is not None:
                pinv = np.linalg.pinv(np.array(T_matrix, dtype=np.float64)).astype(
                    np.float32
                )
                self._t_pinv = torch.from_numpy(pinv).to(self.device)

            self._u_cmd = wp.zeros(
                (self.num_envs, self._vehicle.n_thrusters),
                dtype=wp.float32,
                device=str(self.device),
            )
            self._quat_buf = wp.zeros(
                self.num_envs, dtype=wp.quatf, device=str(self.device)
            )
            self._zero_wrench_kernel = tier1_zero_wrench
            self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
            self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)

        def _setup_scene(self) -> None:
            self._robot = RigidObject(self.cfg.robot_cfg)
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.scene.filter_collisions(global_prim_paths=[])
            self.scene.rigid_objects["robot"] = self._robot

            light_cfg = sim_utils.DomeLightCfg(
                intensity=2000.0, color=(0.75, 0.75, 0.75)
            )
            light_cfg.func("/World/Light", light_cfg)

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            actions = actions.clamp(-1.0, 1.0)
            last_action = self._prev_action.clone()
            self._prev_action.copy_(actions)

            if self._t_pinv is not None:
                u_cmd = (actions @ self._t_pinv.T).clamp(-1.0, 1.0)
            else:
                u_cmd = torch.zeros(
                    self.num_envs,
                    self._vehicle.n_thrusters,
                    device=self.device,
                    dtype=torch.float32,
                )
                for i in range(min(ACT_DIM, self._vehicle.n_thrusters)):
                    u_cmd[:, i] = actions[:, i]

            self._u_cmd = wp.from_torch(u_cmd.contiguous(), dtype=wp.float32)
            self._effort = actions.abs()

            root_vel_w = self._robot.data.root_vel_w.torch
            quat_w = self._robot.data.root_quat_w.torch  # (w,x,y,z)

            from oceanscale.marinegym_compat.math import quat_rotate_inverse
            lin_vel_b = quat_rotate_inverse(quat_w, root_vel_w[:, :3])
            ang_vel_b = quat_rotate_inverse(quat_w, root_vel_w[:, 3:6])
            body_vel = torch.cat([ang_vel_b, lin_vel_b], dim=-1)  # wp.spatial_vectorf: [ω; v]
            body_qd = wp.from_torch(body_vel.contiguous(), dtype=wp.spatial_vectorf)

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

            wrench_t = wp.to_torch(self._tier1.wrench_buf).reshape(
                self.num_envs, 2, 3
            )
            self._thrust.copy_(wrench_t[:, 0:1, :])
            self._moment.copy_(wrench_t[:, 1:2, :])
            self._throttle_difference = torch.norm(
                actions - last_action, dim=-1
            )

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

            from oceanscale.marinegym_compat.math import quat_axis

            heading = quat_axis(quat, axis=0)
            up = quat_axis(quat, axis=2)

            rpos = self._target.to(self.device) - pos
            rheading = self._target_heading.squeeze(1) - heading

            obs_parts = [rpos, quat, lin_vel, ang_vel, heading, up, self._prev_action, rheading]
            if self._hover_cfg.time_encoding:
                t = (self.episode_length_buf / self.max_episode_length).unsqueeze(-1)
                time_enc = t.expand(-1, self._hover_cfg.time_encoding_dim).unsqueeze(
                    1
                )
                obs_parts.append(time_enc.squeeze(1))

            obs = torch.cat(obs_parts, dim=-1)
            return {"policy": obs}

        def _get_rewards(self) -> torch.Tensor:
            pos = self._robot.data.root_pos_w.torch - self.scene.env_origins
            quat = self._robot.data.root_quat_w.torch

            from oceanscale.marinegym_compat.math import quat_axis

            heading = quat_axis(quat, axis=0)
            up = quat_axis(quat, axis=2)

            rpos = self._target.to(self.device) - pos
            rheading = self._target_heading.squeeze(1) - heading

            pos_error = torch.norm(rpos, dim=-1)
            heading_alignment = torch.sum(heading * self._target_heading.squeeze(1), dim=-1)
            distance = torch.norm(torch.cat([rpos, rheading], dim=-1), dim=-1)

            reward_pose = 0.5 / (1.0 + torch.square(
                self._hover_cfg.reward_distance_scale * distance
            ))
            root_vel_w = self._robot.data.root_vel_w.torch
            reward_up = torch.square((up[..., 2] + 1) / 2)
            spinnage = torch.square(root_vel_w[:, 5])
            reward_spin = 1.0 / (1.0 + spinnage)

            reward_effort = self._hover_cfg.reward_effort_weight * torch.exp(
                -self._effort.sum(-1)
            )
            reward_smoothness = (
                self._hover_cfg.reward_action_smoothness_weight
                * torch.exp(-self._throttle_difference)
            )

            reward = (
                reward_pose
                + reward_pose * (reward_up + reward_spin)
                + reward_effort
                + reward_smoothness
            )

            # Update stats
            a = self._hover_cfg.alpha
            self._stats[:, 0] += reward  # return
            self._stats[:, 1].lerp_(pos_error, 1 - a)  # pos_error
            self._stats[:, 2].lerp_(heading_alignment, 1 - a)  # heading
            self._stats[:, 3].lerp_(up[..., 2], 1 - a)  # uprightness
            self._stats[:, 4].lerp_(-self._throttle_difference, 1 - a)  # smoothness

            return reward

        def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
            pos = self._robot.data.root_pos_w.torch - self.scene.env_origins
            quat = self._robot.data.root_quat_w.torch

            from oceanscale.marinegym_compat.math import quat_axis

            heading = quat_axis(quat, axis=0)
            rpos = self._target.to(self.device) - pos
            rheading = self._target_heading.squeeze(1) - heading
            distance = torch.norm(
                torch.cat([rpos, rheading], dim=-1),
                dim=-1,
            )
            misbehave = (pos[:, 2] < self._hover_cfg.min_depth) | (
                distance > self._hover_cfg.max_distance
            )
            truncated = self.episode_length_buf >= self.max_episode_length
            return misbehave, truncated

        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            super()._reset_idx(env_ids)
            self._prev_action[env_ids] = 0.0
            self._stats[env_ids] = 0.0

            from oceanscale.marinegym_compat.math import euler_to_quaternion

            n = len(env_ids)
            lo_p, hi_p = self._hover_cfg.init_pos_range
            lo_d, hi_d = self._hover_cfg.init_depth_range

            root_pose = self._robot.data.default_root_pose.torch[env_ids].clone()
            root_vel = self._robot.data.default_root_vel.torch[env_ids].clone()

            # Randomize position
            root_pose[:, 0] = torch.empty(n, device=self.device).uniform_(lo_p, hi_p)
            root_pose[:, 1] = torch.empty(n, device=self.device).uniform_(lo_p, hi_p)
            root_pose[:, 2] = torch.empty(n, device=self.device).uniform_(lo_d, hi_d)
            root_pose[:, :3] += self.scene.env_origins[env_ids]

            # Randomize orientation
            lo_rpy, hi_rpy = self._hover_cfg.init_rpy_range
            rpy = torch.empty(n, 3, device=self.device).uniform_(lo_rpy, hi_rpy)
            rpy[:, 2] = torch.empty(n, device=self.device).uniform_(0, 2 * torch.pi)
            rot = euler_to_quaternion(rpy)
            # MarineGym uses (w,x,y,z), Isaac Lab uses (w,x,y,z) for root_quat_w
            root_pose[:, 3:7] = rot

            # Randomize target heading
            target_rpy = torch.zeros(n, 3, device=self.device)
            target_rpy[:, 2] = torch.empty(n, device=self.device).uniform_(
                0, 2 * torch.pi
            )
            from oceanscale.marinegym_compat.math import quat_axis as qa
            target_rot = euler_to_quaternion(target_rpy)
            self._target_heading[env_ids, 0] = qa(target_rot, axis=0)

            self._robot.write_root_pose_to_sim_index(
                root_pose=root_pose, env_ids=env_ids
            )
            self._robot.write_root_velocity_to_sim_index(
                root_velocity=root_vel, env_ids=env_ids
            )

else:

    @dataclass
    class MarineGymHoverIsaacCfg:
        """Stub config for when Isaac Lab is not installed."""

        num_envs: int = 64
        episode_length_s: float = 10.0
        decimation: int = 1
        physics_dt: float = 1 / 60
        device: str = "cuda:0"
        action_space: int = ACT_DIM
        observation_space: int = OBS_DIM
        state_space: int = 0
        sim: Any = None
        scene: Any = None

    class MarineGymHoverTask:
        """Placeholder -- Isaac Lab 3 is not installed."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "MarineGymHoverTask requires Isaac Sim 6 + Isaac Lab 3. "
                "Install Isaac Lab or use OceanScaleDirectRLEnv from "
                "oceanscale.training.isaaclab_env for standalone training."
            )
