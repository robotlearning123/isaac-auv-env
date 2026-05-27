"""WaypointFollowingEnv -- sequential waypoint navigation using OceanSim backend.

Composes OceanSim (Newton + Tier1 hydro + ocean) with waypoint task logic.
Each env has N waypoints in sequence; advance on proximity; bonus on reach.
Observation (29-dim): base 26 + 3D relative waypoint position (body frame).
Action (6-dim): wrench in [-1, 1], mapped to thrusters via T_matrix.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import torch
import warp as wp
from gymnasium import spaces

from oceanscale.sim import OceanConfig, OceanSim, OceanSimConfig
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


def _rotate_to_body_torch(quat: torch.Tensor, vec_world: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from world to body frame using quaternion (xyzw) on GPU."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack(
        [
            vx - qw * t0 - qy * t2 + qz * t1,
            vy - qw * t1 - qz * t0 + qx * t2,
            vz - qw * t2 - qx * t1 + qy * t0,
        ],
        dim=-1,
    )


class WaypointFollowingEnv(gym.Env[Any, Any]):
    """GPU-batched waypoint following environment backed by OceanSim.

    Each env has N waypoints in sequence within a configurable cube.
    ROV starts near the first waypoint (within 0.3m noise).
    Advance to next when within wp_tolerance.
    Success: all waypoints visited within time budget.

    Observation (29-dim): base ROV 26-dim + 3D body-frame relative waypoint.
    Action (6-dim): wrench command in [-1, 1].
    Reward: exponential shaping + wp_bonus on waypoint reach.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        n_envs: int = 64,
        n_waypoints: int = 5,
        wp_tolerance: float = 1.0,
        wp_bonus: float = 10.0,
        time_limit: float = 120.0,
        min_wp_dist: float = 1.5,
        max_wp_dist: float = 8.0,
        cube_half_size: float = 5.0,
        depth_range: tuple[float, float] = (-5.0, -0.5),
        decimation: int = 1,
        sensor_noise_std: float = 0.02,
        device: str = "cuda:0",
        dt: float = 1.0 / 240.0,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.n_envs = n_envs
        self.n_waypoints = n_waypoints
        self.wp_tolerance = wp_tolerance
        self.wp_bonus = wp_bonus
        self.min_wp_dist = min_wp_dist
        self.max_wp_dist = max_wp_dist
        self.cube_half_size = cube_half_size
        self.depth_range = depth_range
        self.decimation = decimation
        self.sensor_noise_std = sensor_noise_std
        self.device = device
        self.dt = dt
        self.max_episode_steps = int(time_limit / (dt * decimation))

        # OceanSim physics backend
        vehicle = BlueROV2Heavy()
        self._sim_cfg = OceanSimConfig(
            vehicle=vehicle,
            ocean=OceanConfig(),
            n_envs=n_envs,
            dt=dt,
            device=device,
        )
        self.sim = OceanSim(self._sim_cfg)

        # Waypoint state (GPU tensors)
        self._waypoints = torch.zeros(
            n_envs, n_waypoints, 3, device=device, dtype=torch.float32
        )
        self._current_wp_idx = torch.zeros(
            n_envs, device=device, dtype=torch.int64
        )
        self._waypoints_reached = torch.zeros(
            n_envs, n_waypoints, device=device, dtype=torch.bool
        )
        self._wp_just_reached = torch.zeros(
            n_envs, device=device, dtype=torch.bool
        )

        # Tracking tensors (GPU)
        self._prev_action = torch.zeros(
            n_envs, 6, device=device, dtype=torch.float32
        )
        self._prev_wrench = torch.zeros(
            n_envs, 6, device=device, dtype=torch.float32
        )
        self._step_count = torch.zeros(
            n_envs, device=device, dtype=torch.int64
        )
        self._done = torch.zeros(n_envs, device=device, dtype=torch.bool)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    # Waypoint generation
    # ------------------------------------------------------------------

    def _generate_waypoints(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Generate random waypoints for specified envs."""
        n = len(env_ids)
        wps = torch.zeros(
            n, self.n_waypoints, 3, device=self.device, dtype=torch.float32
        )

        for i in range(n):
            for j in range(self.n_waypoints):
                placed = False
                candidate = torch.zeros(3, device=self.device, dtype=torch.float32)
                for _ in range(100):
                    x = self.np_random.uniform(-self.cube_half_size, self.cube_half_size)
                    y = self.np_random.uniform(-self.cube_half_size, self.cube_half_size)
                    z = self.np_random.uniform(*self.depth_range)
                    candidate = torch.tensor(
                        [x, y, z], device=self.device, dtype=torch.float32
                    )
                    if j == 0:
                        placed = True
                        break
                    dist = torch.linalg.norm(candidate - wps[i, j - 1])
                    if self.min_wp_dist <= dist.item() <= self.max_wp_dist:
                        placed = True
                        break
                wps[i, j] = candidate
        return wps

    def _current_targets(self) -> torch.Tensor:
        """Current waypoint target per env. Shape: (n_envs, 3)."""
        idx = self._current_wp_idx.clamp(0, self.n_waypoints - 1)
        return self._waypoints[
            torch.arange(self.n_envs, device=self.device), idx
        ]

    # ------------------------------------------------------------------
    # Body state helpers
    # ------------------------------------------------------------------

    def _get_body_state(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (position, quaternion, velocity) from OceanSim state.

        pos: (n_envs, 3), quat: (n_envs, 4), vel: (n_envs, 6).
        """
        body_q = wp.to_torch(self.sim.state_curr.body_q)
        body_qd = wp.to_torch(self.sim.state_curr.body_qd)
        return body_q[:, :3], body_q[:, 3:7], body_qd[:, :6]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        gym.Env.reset(self, seed=seed)

        if env_ids is not None:
            env_ids_t = torch.as_tensor(
                np.asarray(env_ids), device=self.device, dtype=torch.int64
            )
        else:
            env_ids_t = torch.arange(
                self.n_envs, device=self.device, dtype=torch.int64
            )

        # Generate new waypoints
        self._waypoints[env_ids_t] = self._generate_waypoints(env_ids_t)
        self._current_wp_idx[env_ids_t] = 0
        self._waypoints_reached[env_ids_t] = False
        self._wp_just_reached[env_ids_t] = False

        # Reset sim positions near first waypoint
        self._reset_env_positions(env_ids_t)

        obs = self._get_obs()
        obs_np = obs.cpu().numpy()
        if self.n_envs == 1:
            obs_np = obs_np.squeeze(0)

        info: dict[str, Any] = {
            "waypoints": self._waypoints.cpu().numpy(),
            "current_wp_idx": self._current_wp_idx.cpu().numpy(),
            "waypoints_reached": self._waypoints_reached.cpu().numpy(),
        }
        return obs_np, info

    def _reset_env_positions(self, env_ids: torch.Tensor) -> None:
        """Reset OceanSim state to positions near first waypoint."""
        import newton as nt

        from oceanscale.hydro.tier1_kernels import tier1_zero_wrench

        n = len(env_ids)
        targets = self._waypoints[env_ids, 0]
        noise = (torch.rand(n, 3, device=self.device, dtype=torch.float32) - 0.5) * 0.6
        start_pos = targets + noise

        ids_np = env_ids.cpu().numpy()
        joint_q = self.sim.model.joint_q.numpy()
        joint_qd = self.sim.model.joint_qd.numpy()

        for k, idx in enumerate(ids_np):
            base = idx * 7
            joint_q[base : base + 7] = [
                start_pos[k, 0].item(),
                start_pos[k, 1].item(),
                start_pos[k, 2].item(),
                0.0,
                0.0,
                0.0,
                1.0,
            ]
        for idx in ids_np:
            base = idx * 6
            joint_qd[base : base + 6] = 0.0

        self.sim.model.joint_q = wp.array(
            joint_q, dtype=wp.float32, device=self.sim.device
        )
        self.sim.model.joint_qd = wp.array(
            joint_qd, dtype=wp.float32, device=self.sim.device
        )

        nt.eval_fk(
            self.sim.model,
            self.sim.model.joint_q,
            self.sim.model.joint_qd,
            self.sim.state_curr,
        )

        wp.launch(
            tier1_zero_wrench,
            dim=self.n_envs,
            inputs=[self.sim.state_curr.body_f],
            device=self.sim.device,
        )

        # Reset Tier1 internal buffers
        nu_prev = self.sim.tier1.nu_prev.numpy()
        nu_dot_prev = self.sim.tier1.nu_dot_prev.numpy()
        u_eff_prev = self.sim.tier1.u_eff_prev.numpy()
        nu_prev[ids_np] = 0.0
        nu_dot_prev[ids_np] = 0.0
        u_eff_prev[ids_np] = 0.0
        wp.copy(
            self.sim.tier1.nu_prev,
            wp.array(nu_prev, dtype=wp.spatial_vectorf, device=self.sim.device),
        )
        wp.copy(
            self.sim.tier1.nu_dot_prev,
            wp.array(nu_dot_prev, dtype=wp.spatial_vectorf, device=self.sim.device),
        )
        wp.copy(
            self.sim.tier1.u_eff_prev,
            wp.array(u_eff_prev, dtype=wp.float32, device=self.sim.device),
        )
        self.sim.tier1.zero_wrench()
        wp.synchronize()

        # Reset tracking
        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False

    def step(  # type: ignore[override]
        self, action: np.ndarray,
    ) -> tuple[
        np.ndarray | Any,
        np.ndarray | float,
        np.ndarray | bool,
        np.ndarray | bool,
        dict[str, Any],
    ]:
        """Step all envs with decimation.

        action: (n_envs, 6) or (6,) wrench in [-1, 1].
        Returns (obs, reward, terminated, truncated, info).
        """
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float32)
        if action_t.ndim == 1:
            action_t = action_t.unsqueeze(0).expand(self.n_envs, -1).clone()
        action_t = action_t.clamp(-1.0, 1.0)
        self._prev_action.copy_(action_t)

        # Physics substeps with zero-order hold
        for _ in range(self.decimation):
            self.sim.step_torch(action_t)

        # Capture wrench from Tier1
        wrench_t = wp.to_torch(self.sim.tier1.wrench_buf)
        self._prev_wrench.copy_(wrench_t.reshape_as(self._prev_wrench))
        self._step_count += 1

        # -- Waypoint advancement (vectorized) --
        pos, _quat, vel = self._get_body_state()
        targets = self._current_targets()

        self._wp_just_reached[:] = False
        active_mask = self._current_wp_idx < self.n_waypoints
        if active_mask.any():
            active_ids = torch.where(active_mask)[0]
            wp_idx_clamped = self._current_wp_idx[active_ids].clamp(
                0, self.n_waypoints - 1
            )
            active_targets = self._waypoints[active_ids, wp_idx_clamped]
            dists = torch.linalg.norm(pos[active_ids] - active_targets, dim=-1)
            reached = dists < self.wp_tolerance

            if reached.any():
                reached_ids = active_ids[reached]
                reached_wp_idx = self._current_wp_idx[reached_ids]
                self._waypoints_reached[reached_ids, reached_wp_idx] = True
                self._current_wp_idx[reached_ids] += 1
                self._wp_just_reached[reached_ids] = True

        # -- Reward --
        reward, reward_components = self._compute_reward(pos, vel, targets)

        # -- Termination --
        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        all_visited = self._current_wp_idx >= self.n_waypoints
        terminated |= all_visited

        # OOB: >10m from current waypoint (vectorized)
        oob_active = self._current_wp_idx < self.n_waypoints
        if oob_active.any():
            oob_ids = torch.where(oob_active)[0]
            oob_wp_idx = self._current_wp_idx[oob_ids].clamp(0, self.n_waypoints - 1)
            oob_targets = self._waypoints[oob_ids, oob_wp_idx]
            oob_dist = torch.linalg.norm(pos[oob_ids] - oob_targets, dim=-1)
            terminated[oob_ids[oob_dist > 10.0]] = True

        truncated = self._step_count >= self.max_episode_steps
        self._done = terminated | truncated

        obs = self._get_obs()
        obs_np = obs.cpu().numpy()

        info: dict[str, Any] = reward_components
        info["waypoints_reached"] = self._waypoints_reached.sum(dim=1).cpu().numpy()
        info["current_wp_idx"] = self._current_wp_idx.cpu().numpy()
        info["success"] = all_visited.cpu().numpy()

        if self.n_envs == 1:
            return (
                obs_np.squeeze(0),
                float(reward.item()),
                bool(terminated.item()),
                bool(truncated.item()),
                info,
            )
        return obs_np, reward.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info

    # ------------------------------------------------------------------
    # Observation & Reward
    # ------------------------------------------------------------------

    def _get_obs(self) -> torch.Tensor:
        """Compute 29-dim observation: base 26 + relative waypoint (body frame)."""
        pos, quat, vel = self._get_body_state()
        targets = self._current_targets()

        obs = torch.zeros(
            self.n_envs, 29, device=self.device, dtype=torch.float32
        )

        obs[:, 0:3] = targets - pos
        obs[:, 3:7] = quat
        obs[:, 7:10] = vel[:, 0:3]
        obs[:, 10:13] = vel[:, 3:6]
        obs[:, 13] = targets[:, 2] - pos[:, 2]

        q = quat
        obs[:, 14] = 2.0 * torch.atan2(q[:, 2], q[:, 3])
        obs[:, 15] = 2.0 * torch.asin(
            torch.clamp(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1)
        )

        obs[:, 16:22] = self._prev_action
        obs[:, 22:26] = self._prev_wrench[:, 2:6]

        rel_world = targets - pos
        obs[:, 26:29] = _rotate_to_body_torch(quat, rel_world)

        if self.sensor_noise_std > 0:
            obs += torch.randn_like(obs) * self.sensor_noise_std

        return torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

    def _compute_reward(
        self,
        pos: torch.Tensor,
        vel: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exponential reward: dense positive signal + milestone bonus."""
        dist_to_wp = torch.linalg.norm(pos - targets, dim=-1).float()
        vel_norm = torch.linalg.norm(vel[:, :3], dim=-1).float().clamp(max=5.0)
        act_norm = torch.linalg.norm(self._prev_action, dim=-1).float()

        r_dist = 0.5 * torch.exp(-dist_to_wp / 3.0)
        r_vel = 0.3 * torch.exp(-vel_norm / 1.0)
        r_act = 0.2 * torch.exp(-act_norm / 1.0)
        reward = r_dist + r_vel + r_act

        wp_bonus = torch.zeros(self.n_envs, device=self.device, dtype=torch.float32)
        wp_bonus[self._wp_just_reached] = self.wp_bonus
        reward += wp_bonus

        info = {
            "reward_distance": float(r_dist.mean().item()),
            "reward_velocity": float(r_vel.mean().item()),
            "reward_action": float(r_act.mean().item()),
            "reward_waypoint_bonus": float(wp_bonus.mean().item()),
            "distance_to_wp": dist_to_wp.cpu().numpy(),
        }
        return reward, info

    def close(self) -> None:
        self.sim.close()
