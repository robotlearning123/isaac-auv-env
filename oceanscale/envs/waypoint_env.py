"""WaypointFollowingEnv -- sequential waypoint navigation for ROV.

Subclasses ROVEnv: 5 waypoints per env, advance on proximity, bonus reward on reach.
Observation (29-dim): base 26 + 3D relative waypoint position (body frame).
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import warp as wp
from gymnasium import spaces

from oceanscale.hydro.tier1_kernels import tier1_zero_wrench
from oceanscale.rov_env import ROVEnv


class WaypointFollowingEnv(ROVEnv):
    """GPU-batched waypoint following environment.

    Each env has 5 waypoints in sequence within a 10m cube.
    ROV starts near the first waypoint (within 0.3m).
    Advance to next when within 1.0m tolerance.
    Success: all 5 visited within time budget (120s).

    Observation (29-dim): base ROV 26-dim + 3D relative position to current waypoint (body frame).
    Action (6-dim): same as ROVEnv.
    Reward: -distance_to_current_waypoint + 10.0 * waypoint_reached_bonus.
    """

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
        **kwargs,
    ) -> None:
        dt = kwargs.get("dt", 1.0 / 240.0)
        max_steps = int(time_limit / dt)
        kwargs["max_episode_steps"] = max_steps
        kwargs.setdefault("target_pos", np.array([0.0, 0.0, -1.5], dtype=np.float32))
        kwargs.setdefault("sensor_noise_std", 0.02)
        kwargs.setdefault("init_pos_noise_std", 0.3)

        super().__init__(n_envs=n_envs, **kwargs)

        self.n_waypoints = n_waypoints
        self.wp_tolerance = wp_tolerance
        self.wp_bonus = wp_bonus
        self.min_wp_dist = min_wp_dist
        self.max_wp_dist = max_wp_dist
        self.cube_half_size = cube_half_size
        self.depth_range = depth_range

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(29,), dtype=np.float32
        )

        self._waypoints = np.zeros((n_envs, n_waypoints, 3), dtype=np.float32)
        self._current_wp_idx = np.zeros(n_envs, dtype=np.int32)
        self._waypoints_reached = np.zeros((n_envs, n_waypoints), dtype=bool)
        self._wp_just_reached = np.zeros(n_envs, dtype=bool)

    def _generate_waypoints(self, n: int) -> np.ndarray:
        """Generate n sets of random waypoints. Shape: (n, n_waypoints, 3)."""
        wps = np.zeros((n, self.n_waypoints, 3), dtype=np.float32)
        for i in range(n):
            for j in range(self.n_waypoints):
                placed = False
                for _ in range(100):
                    x = self.np_random.uniform(-self.cube_half_size, self.cube_half_size)
                    y = self.np_random.uniform(-self.cube_half_size, self.cube_half_size)
                    z = self.np_random.uniform(*self.depth_range)
                    candidate = np.array([x, y, z], dtype=np.float32)
                    if j == 0:
                        wps[i, j] = candidate
                        placed = True
                        break
                    dist = np.linalg.norm(candidate - wps[i, j - 1])
                    if self.min_wp_dist <= dist <= self.max_wp_dist:
                        wps[i, j] = candidate
                        placed = True
                        break
                if not placed:
                    wps[i, j] = candidate
        return wps

    def _current_targets(self, env_ids: np.ndarray) -> np.ndarray:
        """Get current waypoint target per env. Shape: (len(env_ids), 3)."""
        idx = np.clip(self._current_wp_idx[env_ids], 0, self.n_waypoints - 1)
        return self._waypoints[env_ids, idx]

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
        if not self._built:
            self._build()

        if env_ids is not None:
            env_ids = np.asarray(env_ids, dtype=np.int32)
        else:
            env_ids = np.arange(self.n_envs)

        self._waypoints[env_ids] = self._generate_waypoints(len(env_ids))
        self._current_wp_idx[env_ids] = 0
        self._waypoints_reached[env_ids] = False
        self._wp_just_reached[env_ids] = False

        self._reset_indices(env_ids)
        obs = self._get_flat_obs()
        if self.n_envs == 1:
            obs = obs.squeeze(0)

        info: dict[str, Any] = {
            "waypoints": self._waypoints.copy(),
            "current_wp_idx": self._current_wp_idx.copy(),
            "waypoints_reached": self._waypoints_reached.copy(),
        }
        return obs, info

    def _reset_indices(self, env_ids: np.ndarray) -> None:
        """Reset positions near first waypoint with +-0.3m noise."""
        import newton

        n = len(env_ids)
        targets = self._waypoints[env_ids, 0]
        noise = self.np_random.uniform(-0.3, 0.3, (n, 3)).astype(np.float32)
        start_pos = targets + noise

        joint_q = self.model.joint_q.numpy()
        for k, idx in enumerate(env_ids):
            base = idx * 7
            joint_q[base:base + 7] = [
                start_pos[k, 0], start_pos[k, 1], start_pos[k, 2],
                0.0, 0.0, 0.0, 1.0,
            ]
        self.model.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.device)

        joint_qd = self.model.joint_qd.numpy()
        for idx in env_ids:
            base = idx * 6
            joint_qd[base:base + 6] = 0.0
        self.model.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=self.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_curr)

        wp.launch(
            tier1_zero_wrench, dim=self.n_envs,
            inputs=[self.state_curr.body_f], device=self.device,
        )

        nu_prev = self.tier1.nu_prev.numpy()
        nu_dot_prev = self.tier1.nu_dot_prev.numpy()
        u_eff_prev = self.tier1.u_eff_prev.numpy()
        nu_prev[env_ids] = 0.0
        nu_dot_prev[env_ids] = 0.0
        u_eff_prev[env_ids] = 0.0
        wp.copy(self.tier1.nu_prev, wp.array(nu_prev, dtype=wp.spatial_vectorf, device=self.device))
        wp.copy(self.tier1.nu_dot_prev, wp.array(nu_dot_prev, dtype=wp.spatial_vectorf, device=self.device))
        wp.copy(self.tier1.u_eff_prev, wp.array(u_eff_prev, dtype=wp.float32, device=self.device))

        self.tier1.zero_wrench()
        wp.synchronize()

        if self.use_domain_randomization:
            self.tier1.randomize_coeffs(env_ids=env_ids, ranges=self.randomization_ranges)

        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if not self._built:
            raise RuntimeError("Call reset() first")

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = np.broadcast_to(action, (self.n_envs, 6)).copy()
        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action.copy()

        if self._t_pinv is not None:
            u_cmd = np.clip(action @ self._t_pinv.T, -1.0, 1.0).astype(np.float32)
        else:
            u_cmd = np.zeros((self.n_envs, self.n_thrusters), dtype=np.float32)
            for i in range(min(6, self.n_thrusters)):
                u_cmd[:, i] = action[:, i]
        self._u_cmd = wp.array(u_cmd, dtype=wp.float32, device=self.device)

        wp.launch(
            tier1_zero_wrench, dim=self.n_envs,
            inputs=[self.state_curr.body_f], device=self.device,
        )

        self.tier1.compute_wrench(
            nu=self.state_curr.body_qd,
            quat=self._extract_quat(),
            u_cmd=self._u_cmd,
            dt=self.dt,
        )
        self.tier1.write_to_body_f(self.state_curr.body_f)

        self.solver.step(
            self.state_curr, self.state_next, self.control, None, self.dt
        )
        wp.synchronize()
        self.state_curr, self.state_next = self.state_next, self.state_curr

        self._prev_wrench = self.tier1.wrench_buf.numpy().copy()
        self._step_count += 1

        # -- Waypoint advancement --
        pos, _, _ = self._get_body_state()
        self._wp_just_reached[:] = False

        for i in range(self.n_envs):
            wp_idx = self._current_wp_idx[i]
            if wp_idx >= self.n_waypoints:
                continue
            dist = np.linalg.norm(pos[i] - self._waypoints[i, wp_idx])
            if dist < self.wp_tolerance:
                self._waypoints_reached[i, wp_idx] = True
                self._current_wp_idx[i] += 1
                self._wp_just_reached[i] = True

        # -- Reward --
        reward, reward_components = self._compute_reward_with_components()

        # -- Termination --
        terminated = np.zeros(self.n_envs, dtype=bool)
        all_visited = self._current_wp_idx >= self.n_waypoints
        terminated |= all_visited

        # OOB: >10m from current waypoint
        for i in range(self.n_envs):
            wp_idx = self._current_wp_idx[i]
            if wp_idx < self.n_waypoints:
                if np.linalg.norm(pos[i] - self._waypoints[i, wp_idx]) > 10.0:
                    terminated[i] = True

        truncated = self._step_count >= self.max_episode_steps
        self._done = terminated | truncated

        obs = self._get_flat_obs()

        info: dict[str, Any] = reward_components
        info["waypoints_reached"] = self._waypoints_reached.sum(axis=1)
        info["current_wp_idx"] = self._current_wp_idx.copy()
        info["success"] = all_visited.copy()

        if self.n_envs == 1:
            obs = obs.squeeze(0)
            reward = float(reward.item())
            terminated = bool(terminated.item())
            truncated = bool(truncated.item())
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation & Reward overrides
    # ------------------------------------------------------------------

    def _get_obs_for(self, env_ids: np.ndarray) -> np.ndarray:
        """Compute 29-dim observation: base 26 + relative waypoint position (body frame)."""
        pos, quat, vel = self._get_body_state()
        n = len(env_ids)

        obs = np.zeros((n, 29), dtype=np.float32)
        targets = self._current_targets(env_ids)

        obs[:, 0:3] = targets - pos[env_ids]
        obs[:, 3:7] = quat[env_ids]
        obs[:, 7:10] = vel[env_ids, 0:3]
        obs[:, 10:13] = vel[env_ids, 3:6]
        obs[:, 13] = targets[:, 2] - pos[env_ids, 2]

        q = quat[env_ids]
        obs[:, 14] = 2.0 * np.arctan2(q[:, 2], q[:, 3])
        obs[:, 15] = 2.0 * np.arcsin(np.clip(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1))

        obs[:, 16:22] = self._prev_action[env_ids]
        obs[:, 22:26] = self._prev_wrench[env_ids, 2:6]

        # Relative waypoint position in body frame
        rel_world = targets - pos[env_ids]
        obs[:, 26:29] = self._rotate_to_body(quat[env_ids], rel_world)

        if self.sensor_noise_std > 0:
            obs += self.np_random.normal(0, self.sensor_noise_std, obs.shape).astype(np.float32)

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _compute_reward_with_components(
        self,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Exponential reward: dense positive signal + milestone bonus on waypoint reach."""
        pos, _, vel = self._get_body_state()
        targets = self._current_targets(np.arange(self.n_envs))
        dist_to_wp = np.linalg.norm(pos - targets, axis=-1).astype(np.float32)
        vel_norm = np.clip(np.linalg.norm(vel[:, :3], axis=-1), 0, 5.0).astype(np.float32)
        act_norm = np.linalg.norm(self._prev_action, axis=-1).astype(np.float32)

        r_dist = 0.5 * np.exp(-dist_to_wp / 3.0)
        r_vel = 0.3 * np.exp(-vel_norm / 1.0)
        r_act = 0.2 * np.exp(-act_norm / 1.0)
        reward = r_dist + r_vel + r_act

        wp_bonus = np.zeros(self.n_envs, dtype=np.float32)
        wp_bonus[self._wp_just_reached] = self.wp_bonus
        reward += wp_bonus

        info = {
            "reward_distance": float(np.mean(r_dist)),
            "reward_velocity": float(np.mean(r_vel)),
            "reward_action": float(np.mean(r_act)),
            "reward_waypoint_bonus": float(np.mean(wp_bonus)),
            "distance_to_wp": dist_to_wp.copy(),
        }
        return reward, info
