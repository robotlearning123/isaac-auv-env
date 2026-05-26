"""DockingApproachEnv — Precision docking approach task for ROV.

The ROV must approach a fixed docking station from 2-4m away and arrive
within 0.1m with velocity <0.05 m/s and heading error <15 degrees.
Requires a deceleration profile enforced by a speed-limit schedule.

Observation (31-dim): extends ROVEnv 26-dim with:
    [26]    range to dock (scalar distance)
    [27:30] bearing to dock (unit vector, body frame)
    [30]    approach speed (velocity component along bearing)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp
from gymnasium import spaces

from oceanscale.hydro.tier1 import DEFAULT_DT, RandomizationRanges
from oceanscale.hydro.tier1_kernels import tier1_zero_wrench
from oceanscale.rov_env import ROVEnv


class DockingApproachEnv(ROVEnv):
    """Precision docking approach environment.

    Extends ROVEnv with docking-specific observation, reward, and termination.
    The ROV must decelerate as it approaches the dock to achieve a soft landing.
    """

    def __init__(
        self,
        n_envs: int = 64,
        n_thrusters: int = 8,
        device: str = "cuda",
        dt: float = DEFAULT_DT,
        dock_pos: np.ndarray | None = None,
        coeffs: dict[str, Any] | None = None,
        sensor_noise_std: float = 0.02,
        use_domain_randomization: bool = False,
        randomization_ranges: RandomizationRanges | None = None,
        dock_offset_range: float = 1.0,
    ) -> None:
        if dock_pos is None:
            dock_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)

        super().__init__(
            n_envs=n_envs,
            n_thrusters=n_thrusters,
            device=device,
            dt=dt,
            max_episode_steps=7200,
            target_pos=dock_pos,
            coeffs=coeffs,
            sensor_noise_std=sensor_noise_std,
            use_domain_randomization=use_domain_randomization,
            randomization_ranges=randomization_ranges,
            init_pos_noise_std=0.0,
            init_yaw_noise_std=0.0,
            use_fluid=False,
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32
        )

        self._w_range = 1.0
        self._w_vel_dock = 2.0
        self._w_align = 0.5
        self._w_soft = 3.0
        self._w_dock = 100.0
        self._w_crash = 50.0
        self._w_act_dock = 0.01

        self._init_range_min = 2.0
        self._init_range_max = 4.0
        self._init_bearing_v_max = np.deg2rad(20.0)
        self._init_vel_max = 0.2
        self._init_heading_offset_max = np.deg2rad(30.0)
        self._current_speed_max = 0.3
        self._current_coupling = 5.0
        self._dock_offset_range = dock_offset_range

        self._dock_positions: np.ndarray | None = None
        self._min_range: np.ndarray | None = None
        self._current_force: np.ndarray | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = super().reset(seed=seed, options=options, env_ids=env_ids)
        if self.n_envs == 1:
            obs = obs.squeeze(0)
        return obs, info

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self) -> None:
        super()._build()
        self._dock_positions = np.tile(
            self.target_pos, (self.n_envs, 1)
        ).astype(np.float32)
        self._min_range = np.full(self.n_envs, np.inf, dtype=np.float32)
        self._current_force = np.zeros((self.n_envs, 3), dtype=np.float32)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

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
            u_cmd = np.clip(
                action @ self._t_pinv.T, -1.0, 1.0
            ).astype(np.float32)
        else:
            u_cmd = np.zeros((self.n_envs, self.n_thrusters), dtype=np.float32)
            for i in range(min(6, self.n_thrusters)):
                u_cmd[:, i] = action[:, i]
        self._u_cmd = wp.array(u_cmd, dtype=wp.float32, device=self.device)

        wp.launch(
            tier1_zero_wrench,
            dim=self.n_envs,
            inputs=[self.state_curr.body_f],
            device=self.device,
        )

        self.tier1.compute_wrench(
            nu=self.state_curr.body_qd,
            quat=self._extract_quat(),
            u_cmd=self._u_cmd,
            dt=self.dt,
        )
        self.tier1.write_to_body_f(self.state_curr.body_f)

        self._apply_docking_current()

        self.solver.step(
            self.state_curr, self.state_next, self.control, None, self.dt
        )
        wp.synchronize()
        self.state_curr, self.state_next = self.state_next, self.state_curr

        self._prev_wrench = self.tier1.wrench_buf.numpy().copy()
        self._step_count += 1

        reward, info = self._compute_reward_with_components()

        pos, quat, vel = self._get_body_state()
        diff = self._dock_positions - pos
        range_to_dock = np.linalg.norm(diff, axis=-1)
        vel_norm = np.linalg.norm(vel[:, :3], axis=-1)

        self._min_range = np.minimum(self._min_range, range_to_dock)

        bearing_world = diff / np.maximum(range_to_dock[:, None], 1e-8)
        heading_err = self._heading_error_to_dock(quat, bearing_world, range_to_dock)

        success = (
            (range_to_dock < 0.1)
            & (vel_norm < 0.05)
            & (heading_err < np.deg2rad(15))
        )
        hard_contact = (range_to_dock < 0.05) & (vel_norm > 0.3)
        drift = range_to_dock > 5.0

        reward = np.where(success, reward + 5.0, reward)
        reward = np.where(hard_contact, reward - 5.0, reward)

        terminated = hard_contact | drift
        truncated = self._step_count >= self.max_episode_steps
        self._done = terminated | truncated | success

        info["success"] = success
        info["min_range"] = self._min_range.copy()
        info["final_range"] = range_to_dock
        info["hard_contact"] = hard_contact
        info["range_to_dock"] = range_to_dock

        divergent = ~np.isfinite(reward) | drift
        if np.any(divergent):
            div_ids = np.where(divergent)[0]
            self._reset_indices(div_ids)
            reward[div_ids] = 0.0
            terminated[div_ids] = False
            truncated[div_ids] = False

        obs = self._get_flat_obs()
        if self.n_envs == 1:
            obs = obs.squeeze(0)
            reward = float(reward.item())
            terminated = bool(terminated.item())
            truncated = bool(truncated.item())
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs_for(self, env_ids: np.ndarray) -> np.ndarray:
        pos, quat, vel = self._get_body_state()
        dock_pos = self._dock_positions[env_ids]

        obs = np.zeros((len(env_ids), 31), dtype=np.float32)

        obs[:, 0:3] = dock_pos - pos[env_ids]
        obs[:, 3:7] = quat[env_ids]
        obs[:, 7:10] = vel[env_ids, 0:3]
        obs[:, 10:13] = vel[env_ids, 3:6]
        obs[:, 13] = dock_pos[:, 2] - pos[env_ids, 2]

        q = quat[env_ids]
        obs[:, 14] = 2.0 * np.arctan2(q[:, 2], q[:, 3])
        obs[:, 15] = 2.0 * np.arcsin(
            np.clip(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1)
        )
        obs[:, 16:22] = self._prev_action[env_ids]
        obs[:, 22:26] = self._prev_wrench[env_ids, 2:6]

        diff = dock_pos - pos[env_ids]
        range_to_dock = np.linalg.norm(diff, axis=-1)
        bearing_world = diff / np.maximum(range_to_dock[:, None], 1e-8)
        bearing_body = self._rotate_to_body(quat[env_ids], bearing_world)

        obs[:, 26] = range_to_dock
        obs[:, 27:30] = bearing_body
        obs[:, 30] = np.sum(vel[env_ids, 0:3] * bearing_body, axis=-1)

        if self.sensor_noise_std > 0:
            obs += self.np_random.normal(
                0, self.sensor_noise_std, obs.shape
            ).astype(np.float32)

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward_with_components(
        self,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        pos, quat, vel = self._get_body_state()
        dock_pos = self._dock_positions

        diff = dock_pos - pos
        range_to_dock = np.linalg.norm(diff, axis=-1)
        bearing_world = diff / np.maximum(range_to_dock[:, None], 1e-8)
        bearing_body = self._rotate_to_body(quat, bearing_world)

        vel_body = vel[:, 0:3]
        vel_norm = np.linalg.norm(vel_body, axis=-1)

        body_x = np.zeros((len(quat), 3), dtype=np.float32)
        body_x[:, 0] = 1.0
        body_x_world = self._rotate_to_world(quat, body_x)
        cos_heading = np.sum(body_x_world * bearing_world, axis=-1)
        heading_err = np.arccos(np.clip(cos_heading, -1, 1))

        act_norm = np.linalg.norm(self._prev_action, axis=-1)

        r_range = 0.4 * np.exp(-range_to_dock / 2.0)
        r_vel = 0.3 * np.exp(-vel_norm / 0.5)
        r_heading = 0.2 * np.exp(-heading_err / 0.5)
        r_act = 0.1 * np.exp(-act_norm / 1.0)

        reward = (r_range + r_vel + r_heading + r_act).astype(np.float32)

        info = {
            "reward_range": float(np.mean(r_range)),
            "reward_velocity": float(np.mean(r_vel)),
            "reward_heading": float(np.mean(r_heading)),
            "reward_action": float(np.mean(r_act)),
        }
        return reward, info

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _reset_indices(self, env_ids: np.ndarray) -> None:
        import newton

        n_reset = len(env_ids)

        if self._dock_offset_range > 0:
            dock_offset = self.np_random.uniform(
                -self._dock_offset_range, self._dock_offset_range, (n_reset, 3)
            ).astype(np.float32)
            self._dock_positions[env_ids] = self.target_pos + dock_offset
        else:
            self._dock_positions[env_ids] = self.target_pos

        dock_pos = self._dock_positions[env_ids]

        init_range = self.np_random.uniform(
            self._init_range_min, self._init_range_max, n_reset
        ).astype(np.float32)
        bearing_h = self.np_random.uniform(
            0, 2 * np.pi, n_reset
        ).astype(np.float32)
        bearing_v = self.np_random.uniform(
            -self._init_bearing_v_max, self._init_bearing_v_max, n_reset
        ).astype(np.float32)

        dx = np.cos(bearing_h) * np.cos(bearing_v)
        dy = np.sin(bearing_h) * np.cos(bearing_v)
        dz = np.sin(bearing_v)
        direction = np.stack([dx, dy, dz], axis=-1)

        start_pos = dock_pos + init_range[:, None] * direction

        yaw_to_dock = np.arctan2(-dy, -dx)
        heading_offset = self.np_random.uniform(
            -self._init_heading_offset_max, self._init_heading_offset_max, n_reset
        ).astype(np.float32)
        yaw = yaw_to_dock + heading_offset
        half_yaw = yaw * 0.5

        joint_q = self.model.joint_q.numpy()
        for k, idx in enumerate(env_ids):
            base = idx * 7
            joint_q[base : base + 7] = [
                float(start_pos[k, 0]),
                float(start_pos[k, 1]),
                float(start_pos[k, 2]),
                0.0,
                0.0,
                float(np.sin(half_yaw[k])),
                float(np.cos(half_yaw[k])),
            ]
        self.model.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.device)

        init_vel_mag = self.np_random.uniform(
            0, self._init_vel_max, n_reset
        ).astype(np.float32)
        vel_dir_h = self.np_random.uniform(
            0, 2 * np.pi, n_reset
        ).astype(np.float32)
        vel_dir_v = self.np_random.uniform(-0.5, 0.5, n_reset).astype(np.float32)
        vx = init_vel_mag * np.cos(vel_dir_h) * np.cos(vel_dir_v)
        vy = init_vel_mag * np.sin(vel_dir_h) * np.cos(vel_dir_v)
        vz = init_vel_mag * np.sin(vel_dir_v)

        joint_qd = self.model.joint_qd.numpy()
        for k, idx in enumerate(env_ids):
            base = idx * 6
            joint_qd[base : base + 6] = [
                float(vx[k]),
                float(vy[k]),
                float(vz[k]),
                0.0,
                0.0,
                0.0,
            ]
        self.model.joint_qd = wp.array(
            joint_qd, dtype=wp.float32, device=self.device
        )

        newton.eval_fk(
            self.model, self.model.joint_q, self.model.joint_qd, self.state_curr
        )

        wp.launch(
            tier1_zero_wrench,
            dim=self.n_envs,
            inputs=[self.state_curr.body_f],
            device=self.device,
        )

        nu_prev = self.tier1.nu_prev.numpy()
        nu_dot_prev = self.tier1.nu_dot_prev.numpy()
        u_eff_prev = self.tier1.u_eff_prev.numpy()
        nu_prev[env_ids] = 0.0
        nu_dot_prev[env_ids] = 0.0
        u_eff_prev[env_ids] = 0.0
        wp.copy(
            self.tier1.nu_prev,
            wp.array(nu_prev, dtype=wp.spatial_vectorf, device=self.device),
        )
        wp.copy(
            self.tier1.nu_dot_prev,
            wp.array(nu_dot_prev, dtype=wp.spatial_vectorf, device=self.device),
        )
        wp.copy(
            self.tier1.u_eff_prev,
            wp.array(u_eff_prev, dtype=wp.float32, device=self.device),
        )

        self.tier1.zero_wrench()
        wp.synchronize()

        if self.use_domain_randomization:
            docking_ranges = RandomizationRanges(
                mass=0.15, m_added=0.20, d_lin=0.20, d_quad=0.20
            )
            self.tier1.randomize_coeffs(env_ids=env_ids, ranges=docking_ranges)

        current_speed = self.np_random.uniform(
            0, self._current_speed_max, n_reset
        ).astype(np.float32)
        current_dir = self.np_random.uniform(
            0, 2 * np.pi, n_reset
        ).astype(np.float32)
        self._current_force[env_ids, 0] = (
            self._current_coupling * current_speed * np.cos(current_dir)
        )
        self._current_force[env_ids, 1] = (
            self._current_coupling * current_speed * np.sin(current_dir)
        )
        self._current_force[env_ids, 2] = 0.0

        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
        self._min_range[env_ids] = np.inf

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _apply_docking_current(self) -> None:
        if self._current_force is None:
            return
        quat = self.state_curr.body_q.numpy()[:, 3:7]
        force_body = self._rotate_to_body(quat, self._current_force)
        body_f = self.state_curr.body_f.numpy()
        body_f[:, 0:3] += force_body
        wp.copy(
            self.state_curr.body_f,
            wp.array(body_f, dtype=wp.float32, device=self.device),
        )

    @staticmethod
    def _rotate_to_body(
        quat: np.ndarray, vec_world: np.ndarray
    ) -> np.ndarray:
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]
        t0 = 2.0 * (qy * vz - qz * vy)
        t1 = 2.0 * (qz * vx - qx * vz)
        t2 = 2.0 * (qx * vy - qy * vx)
        result = np.stack([
            vx - qw * t0 + qy * t2 - qz * t1,
            vy - qw * t1 + qz * t0 - qx * t2,
            vz - qw * t2 + qx * t1 - qy * t0,
        ], axis=-1)
        return result.astype(np.float32)

    def _heading_error_to_dock(
        self, quat: np.ndarray, bearing_world: np.ndarray, range_to_dock: np.ndarray
    ) -> np.ndarray:
        body_x = np.zeros((len(quat), 3), dtype=np.float32)
        body_x[:, 0] = 1.0
        body_x_world = self._rotate_to_world(quat, body_x)
        cos_angle = np.clip(
            np.sum(body_x_world * bearing_world, axis=-1), -1, 1
        )
        heading_err = np.arccos(cos_angle)
        heading_err = np.where(range_to_dock < 0.15, 0.0, heading_err)
        return heading_err
