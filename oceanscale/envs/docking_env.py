"""DockingApproachEnv — Precision docking approach task for ROV.

Uses OceanSim for physics (Newton rigid-body + Tier1 Fossen hydrodynamics).
The ROV must approach a fixed docking station from 2-4m away and arrive
within 0.1m with velocity <0.05 m/s and heading error <15 degrees.

Observation (31-dim):
    [0:3]   relative position to dock (world frame)
    [3:7]   quaternion (xyzw)
    [7:10]  angular velocity (body frame)
    [10:13] linear velocity (body frame)
    [13]    depth difference
    [14]    roll angle
    [15]    pitch angle
    [16:22] previous action (6-dim)
    [22:26] previous wrench [fz, tx, ty, tz]
    [26]    range to dock (scalar distance)
    [27:30] bearing to dock (unit vector, body frame)
    [30]    approach speed (velocity component along bearing)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp

from oceanscale.sim import OceanConfig, OceanSim, OceanSimConfig
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy

OBS_DIM = 31
ACT_DIM = 6


def _rotate_to_body_torch(quat: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from world to body frame using quaternion (xyzw).

    Computes q* ⊗ v ⊗ q (conjugate sandwich) for inverse rotation.
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack(
        [
            vx - qw * t0 + qy * t2 - qz * t1,
            vy - qw * t1 + qz * t0 - qx * t2,
            vz - qw * t2 + qx * t1 - qy * t0,
        ],
        dim=-1,
    )


def _rotate_to_world_torch(quat: torch.Tensor, vec_body: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from body to world frame using quaternion (xyzw).

    Computes q ⊗ v ⊗ q* for forward rotation.
    """
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_body[:, 0], vec_body[:, 1], vec_body[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack(
        [
            vx + qw * t0 + qy * t2 - qz * t1,
            vy + qw * t1 + qz * t0 - qx * t2,
            vz + qw * t2 + qx * t1 - qy * t0,
        ],
        dim=-1,
    )


@dataclass
class DockingApproachEnvCfg:
    """Configuration for DockingApproachEnv."""

    n_envs: int = 64
    device: str = "cuda:0"
    dt: float = 1.0 / 240.0
    decimation: int = 1
    max_episode_steps: int = 7200
    dock_pos: tuple[float, float, float] = (0.0, 0.0, -1.5)
    sensor_noise_std: float = 0.02
    use_domain_randomization: bool = False
    dock_offset_range: float = 1.0
    init_range_min: float = 2.0
    init_range_max: float = 4.0
    init_bearing_v_max: float = 20.0
    init_vel_max: float = 0.2
    init_heading_offset_max: float = 30.0
    current_speed_max: float = 0.3
    current_coupling: float = 5.0


class DockingApproachEnv(gym.Env):
    """Precision docking approach environment backed by OceanSim.

    The ROV must decelerate as it approaches the dock to achieve a soft
    landing. Physics via OceanSim.step_torch(); docking-specific obs,
    reward, and termination computed on GPU with torch tensors.
    """

    is_vector_env = True
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, cfg: DockingApproachEnvCfg | None = None) -> None:
        if cfg is None:
            cfg = DockingApproachEnvCfg()
        self.cfg = cfg
        self.num_envs = cfg.n_envs
        self._device = torch.device(cfg.device)

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32
        )

        self.sim = OceanSim(
            OceanSimConfig(
                vehicle=BlueROV2Heavy(),
                ocean=OceanConfig(),
                n_envs=cfg.n_envs,
                dt=cfg.dt,
                device=cfg.device,
                init_pos=cfg.dock_pos,
            )
        )

        self._target = torch.tensor(
            cfg.dock_pos, dtype=torch.float32, device=self._device
        )
        self._step_count = torch.zeros(
            cfg.n_envs, dtype=torch.int64, device=self._device
        )
        self._prev_action = torch.zeros(
            cfg.n_envs, ACT_DIM, dtype=torch.float32, device=self._device
        )
        self._prev_wrench = torch.zeros(
            cfg.n_envs, 4, dtype=torch.float32, device=self._device
        )
        self._dock_positions = (
            self._target.unsqueeze(0).expand(cfg.n_envs, -1).clone()
        )
        self._min_range = torch.full(
            (cfg.n_envs,), float("inf"), dtype=torch.float32, device=self._device
        )
        self._current_force = torch.zeros(
            cfg.n_envs, 3, dtype=torch.float32, device=self._device
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        super().reset(seed=seed)
        if env_ids is None:
            self._reset_indices(torch.arange(self.num_envs, device=self._device))
        else:
            ids_t = torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
            self._reset_indices(ids_t)

        obs = self._get_observations()
        info: dict[str, Any] = {"min_range": self._min_range.clone()}
        if self.num_envs == 1:
            return obs.squeeze(0), info
        return obs, info

    def _reset_indices(self, env_ids: torch.Tensor) -> None:
        import newton as nt

        from oceanscale.hydro.tier1 import RandomizationRanges

        n = len(env_ids)
        dev = self._device
        cfg = self.cfg
        ids_np = env_ids.cpu().numpy()
        wp_dev = str(dev)

        # Dock position with optional offset
        if cfg.dock_offset_range > 0:
            offset = (torch.rand(n, 3, device=dev) * 2 - 1) * cfg.dock_offset_range
            self._dock_positions[env_ids] = self._target.unsqueeze(0) + offset
        else:
            self._dock_positions[env_ids] = self._target.unsqueeze(0)

        dock_pos = self._dock_positions[env_ids]

        # Random start position on sphere around dock
        init_range = torch.empty(n, device=dev).uniform_(
            cfg.init_range_min, cfg.init_range_max
        )
        bearing_h = torch.empty(n, device=dev).uniform_(0, 2 * np.pi)
        bearing_v = torch.empty(n, device=dev).uniform_(
            -np.deg2rad(cfg.init_bearing_v_max),
            np.deg2rad(cfg.init_bearing_v_max),
        )
        dx = torch.cos(bearing_h) * torch.cos(bearing_v)
        dy = torch.sin(bearing_h) * torch.cos(bearing_v)
        dz = torch.sin(bearing_v)
        direction = torch.stack([dx, dy, dz], dim=-1)
        start_pos = dock_pos + init_range.unsqueeze(-1) * direction

        # Heading: face dock with random offset
        yaw_to_dock = torch.atan2(-dy, -dx)
        heading_offset = torch.empty(n, device=dev).uniform_(
            -np.deg2rad(cfg.init_heading_offset_max),
            np.deg2rad(cfg.init_heading_offset_max),
        )
        yaw = yaw_to_dock + heading_offset
        half_yaw = yaw * 0.5

        # Random initial velocity
        init_vel = torch.empty(n, device=dev).uniform_(0, cfg.init_vel_max)
        vel_h = torch.empty(n, device=dev).uniform_(0, 2 * np.pi)
        vel_v = torch.empty(n, device=dev).uniform_(-0.5, 0.5)
        vx = init_vel * torch.cos(vel_h) * torch.cos(vel_v)
        vy = init_vel * torch.sin(vel_h) * torch.cos(vel_v)
        vz = init_vel * torch.sin(vel_v)

        # Write to Newton model
        q = self.sim.model.joint_q.numpy()
        qd = self.sim.model.joint_qd.numpy()
        for k, idx in enumerate(env_ids.tolist()):
            bq = idx * 7
            q[bq : bq + 3] = [
                start_pos[k, 0].item(),
                start_pos[k, 1].item(),
                start_pos[k, 2].item(),
            ]
            q[bq + 3 : bq + 7] = [
                0.0,
                0.0,
                torch.sin(half_yaw[k]).item(),
                torch.cos(half_yaw[k]).item(),
            ]
            bqd = idx * 6
            qd[bqd : bqd + 6] = [
                vx[k].item(),
                vy[k].item(),
                vz[k].item(),
                0.0,
                0.0,
                0.0,
            ]
        self.sim.model.joint_q.assign(q)
        self.sim.model.joint_qd.assign(qd)
        nt.eval_fk(
            self.sim.model,
            self.sim.model.joint_q,
            self.sim.model.joint_qd,
            self.sim.state_curr,
        )

        # Reset tier1 internal state
        nu_prev = self.sim.tier1.nu_prev.numpy()
        nu_dot_prev = self.sim.tier1.nu_dot_prev.numpy()
        u_eff_prev = self.sim.tier1.u_eff_prev.numpy()
        nu_prev[ids_np] = 0.0
        nu_dot_prev[ids_np] = 0.0
        u_eff_prev[ids_np] = 0.0
        wp.copy(
            self.sim.tier1.nu_prev,
            wp.array(nu_prev, dtype=wp.spatial_vectorf, device=wp_dev),
        )
        wp.copy(
            self.sim.tier1.nu_dot_prev,
            wp.array(nu_dot_prev, dtype=wp.spatial_vectorf, device=wp_dev),
        )
        wp.copy(
            self.sim.tier1.u_eff_prev,
            wp.array(u_eff_prev, dtype=wp.float32, device=wp_dev),
        )
        self.sim.tier1.zero_wrench()
        wp.synchronize()

        # Domain randomization
        if cfg.use_domain_randomization:
            docking_ranges = RandomizationRanges(
                mass=0.15, added_mass=0.20, d_lin=0.20, d_quad=0.20
            )
            self.sim.tier1.randomize_coeffs(env_ids=ids_np, ranges=docking_ranges)

        # Random current per env
        current_speed = torch.empty(n, device=dev).uniform_(0, cfg.current_speed_max)
        current_dir = torch.empty(n, device=dev).uniform_(0, 2 * np.pi)
        self._current_force[env_ids, 0] = (
            cfg.current_coupling * current_speed * torch.cos(current_dir)
        )
        self._current_force[env_ids, 1] = (
            cfg.current_coupling * current_speed * torch.sin(current_dir)
        )
        self._current_force[env_ids, 2] = 0.0

        # Reset tracking state
        self._step_count[env_ids] = 0
        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._min_range[env_ids] = float("inf")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self, action: torch.Tensor | np.ndarray
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | float,
        torch.Tensor | bool,
        torch.Tensor | bool,
        dict[str, Any],
    ]:
        action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.num_envs, -1).clone()
        action = action.clamp(-1.0, 1.0)
        self._prev_action.copy_(action)

        for _ in range(self.cfg.decimation):
            self.sim.step_torch(action)
            self._apply_docking_current_impulse()

        # Capture wrench from last substep
        wrench = self.sim.tier1.wrench_buf.numpy()
        self._prev_wrench.copy_(
            torch.tensor(wrench[:, 2:6], dtype=torch.float32, device=self._device)
        )

        self._step_count += 1

        # State from OceanSim
        state = self.sim.observe_torch()
        pos = state["position"]
        quat = state["orientation"]
        lin_vel = state["linear_velocity"]
        ang_vel = state["angular_velocity"]

        # Docking geometry
        diff = self._dock_positions - pos
        range_to_dock = torch.linalg.norm(diff, dim=-1)
        vel_norm = torch.linalg.norm(lin_vel, dim=-1)

        self._min_range = torch.minimum(self._min_range, range_to_dock)

        bearing_world = diff / torch.clamp(range_to_dock.unsqueeze(-1), min=1e-8)
        heading_err = self._heading_error_to_dock(quat, bearing_world, range_to_dock)

        # Reward
        reward, info = self._compute_reward(range_to_dock, vel_norm, heading_err)

        # Termination
        success = (
            (range_to_dock < 0.1) & (vel_norm < 0.05) & (heading_err < np.deg2rad(15))
        )
        hard_contact = (range_to_dock < 0.05) & (vel_norm > 0.3)
        drift = range_to_dock > 5.0

        reward = torch.where(success, reward + 5.0, reward)
        reward = torch.where(hard_contact, reward - 5.0, reward)

        terminated = hard_contact | drift
        truncated = self._step_count >= self.cfg.max_episode_steps

        info["success"] = success
        info["min_range"] = self._min_range.clone()
        info["final_range"] = range_to_dock
        info["hard_contact"] = hard_contact
        info["range_to_dock"] = range_to_dock

        # Auto-reset divergent envs
        divergent = ~torch.isfinite(reward) | drift
        if divergent.any():
            div_ids = divergent.nonzero(as_tuple=True)[0]
            self._reset_indices(div_ids)
            reward[div_ids] = 0.0
            terminated[div_ids] = False
            truncated[div_ids] = False

        obs = self._get_observations()

        if self.num_envs == 1:
            return (
                obs.squeeze(0),
                float(reward.item()),
                bool(terminated.item()),
                bool(truncated.item()),
                info,
            )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_observations(self) -> torch.Tensor:
        state = self.sim.observe_torch()
        pos = state["position"]
        quat = state["orientation"]
        lin_vel = state["linear_velocity"]
        ang_vel = state["angular_velocity"]

        diff = self._dock_positions - pos
        range_to_dock = torch.linalg.norm(diff, dim=-1)
        bearing_world = diff / torch.clamp(range_to_dock.unsqueeze(-1), min=1e-8)
        bearing_body = _rotate_to_body_torch(quat, bearing_world)

        obs = torch.zeros(
            self.num_envs, OBS_DIM, dtype=torch.float32, device=self._device
        )
        obs[:, 0:3] = diff
        obs[:, 3:7] = quat
        obs[:, 7:10] = ang_vel
        obs[:, 10:13] = lin_vel
        obs[:, 13] = diff[:, 2]
        obs[:, 14] = 2.0 * torch.atan2(quat[:, 2], quat[:, 3])
        obs[:, 15] = 2.0 * torch.arcsin(
            torch.clamp(-quat[:, 0] * quat[:, 2] + quat[:, 1] * quat[:, 3], -1, 1)
        )
        obs[:, 16:22] = self._prev_action
        obs[:, 22:26] = self._prev_wrench
        obs[:, 26] = range_to_dock
        obs[:, 27:30] = bearing_body
        obs[:, 30] = torch.sum(lin_vel * bearing_body, dim=-1)

        if self.cfg.sensor_noise_std > 0:
            obs += torch.randn_like(obs) * self.cfg.sensor_noise_std

        obs.nan_to_num_(nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        range_to_dock: torch.Tensor,
        vel_norm: torch.Tensor,
        heading_err: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        act_norm = torch.linalg.norm(self._prev_action, dim=-1)

        r_range = 0.4 * torch.exp(-range_to_dock / 2.0)
        r_vel = 0.3 * torch.exp(-vel_norm / 0.5)
        r_heading = 0.2 * torch.exp(-heading_err / 0.5)
        r_act = 0.1 * torch.exp(-act_norm / 1.0)

        reward = (r_range + r_vel + r_heading + r_act).to(torch.float32)

        info = {
            "reward_range": float(r_range.mean()),
            "reward_velocity": float(r_vel.mean()),
            "reward_heading": float(r_heading.mean()),
            "reward_action": float(r_act.mean()),
        }
        return reward, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _heading_error_to_dock(
        self,
        quat: torch.Tensor,
        bearing_world: torch.Tensor,
        range_to_dock: torch.Tensor,
    ) -> torch.Tensor:
        body_x = torch.zeros(len(quat), 3, dtype=torch.float32, device=self._device)
        body_x[:, 0] = 1.0
        body_x_world = _rotate_to_world_torch(quat, body_x)
        cos_angle = torch.clamp(
            torch.sum(body_x_world * bearing_world, dim=-1), -1, 1
        )
        heading_err = torch.arccos(cos_angle)
        return torch.where(
            range_to_dock < 0.15, torch.zeros_like(heading_err), heading_err
        )

    def _apply_docking_current_impulse(self) -> None:
        if self._current_force.abs().max() < 1e-8:
            return
        quat = self.sim.observe_torch()["orientation"]
        F_body = _rotate_to_body_torch(quat, self._current_force)
        mass = self.sim.cfg.vehicle.mass
        dv = F_body * (self.sim.dt / mass)
        body_qd_t = wp.to_torch(self.sim.state_curr.body_qd)
        body_qd_t[:, 3:6] += dv

    def close(self) -> None:
        self.sim.close()
