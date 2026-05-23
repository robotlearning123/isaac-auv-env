"""PyBullet BlueROV2 Heavy hover environment — CPU baseline for benchmarking.

Mirrors oceanscale.rov_env.ROVEnv exactly:
  - Observation: 26-dim (pos_err, quat, vel, depth_err, heading_err, prev_act, prev_wrench)
  - Action: 6-DOF [-1,+1] mapped to thrusters via T_matrix pinv
  - Reward: station-keeping (pos + vel + act + depth + heading)
  - Hydrodynamics: Fossen Tier1 (added mass, damping, restoring, Coriolis) as external forces

Used by benchmarks/oceanscale_vs_bullet.py for apples-to-apples throughput comparison.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Lazy import — pybullet is an optional bench dependency
_pybullet = None


def _get_pybullet():
    global _pybullet
    if _pybullet is None:
        import pybullet as p

        _pybullet = p
    return _pybullet


class BulletBlueROV2Env(gym.Env):
    """Single-env BlueROV2 Heavy in PyBullet with Fossen Tier1 hydrodynamics.

    Observation (26-dim) — identical to ROVEnv:
        [0:3]   position error (target - current xyz)
        [3:7]   orientation quaternion (xyzw)
        [7:10]  linear velocity (body frame)
        [10:13] angular velocity (body frame)
        [13]    depth error scalar
        [14:16] heading error (roll, pitch)
        [16:22] previous action (6-dim)
        [22:26] previous hydro wrench (fz, mx, my, mz)

    Action (6-dim): [-1, +1] per DOF (surge, sway, heave, roll, pitch, yaw).
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        dt: float = 1.0 / 240.0,
        max_episode_steps: int = 1000,
        target_pos: np.ndarray | None = None,
        sensor_noise_std: float = 0.02,
        reward_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__()

        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.sensor_noise_std = sensor_noise_std

        if target_pos is None:
            target_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)

        # BlueROV2 Heavy coefficients — identical to oceanscale/vehicles/bluerov2.py
        self.mass = 13.5
        self.volume = 0.0134
        self.coBM = 0.01
        self.Ix, self.Iy, self.Iz = 0.26, 0.23, 0.37

        # Added mass (Fossen 6-tuple)
        self.added_mass = np.array([6.36, 7.12, 18.68, 0.189, 0.135, 0.222], dtype=np.float32)
        # Linear damping
        self.d_lin = np.array([13.7, 0.0, 33.0, 0.0, 0.8, 0.0], dtype=np.float32)
        # Quadratic damping
        self.d_quad = np.array([141.0, 217.0, 190.0, 1.19, 0.47, 1.5], dtype=np.float32)

        self.rho_water = 1025.0
        self.g_accel = 9.81
        self.n_thrusters = 8
        self.max_thrust = 51.5

        # T_matrix and pseudoinverse — identical to BlueROV2Heavy.compute_t_matrix()
        self._t_matrix = self._compute_t_matrix()
        self._t_pinv = np.linalg.pinv(self._t_matrix.astype(np.float64)).astype(np.float32)

        w = reward_weights or {}
        self._w_pos = w.get("pos", 1.0)
        self._w_vel = w.get("vel", 0.1)
        self._w_act = w.get("act", 0.01)
        self._w_depth = w.get("depth", 2.0)
        self._w_heading = w.get("heading", 0.5)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        self._prev_action = np.zeros(6, dtype=np.float32)
        self._prev_wrench = np.zeros(6, dtype=np.float32)
        self._step_count = 0

        # EMA state for nu_dot estimation
        self._nu_prev = np.zeros(6, dtype=np.float32)
        self._nu_dot_prev = np.zeros(6, dtype=np.float32)
        self._ema_alpha = 0.3

        # Thruster low-pass state
        self._u_eff_prev = np.zeros(self.n_thrusters, dtype=np.float32)

        self._built = False
        self._client_id = None
        self._body_id = None

    @staticmethod
    def _compute_t_matrix() -> np.ndarray:
        """Compute 6x8 allocation matrix — identical to BlueROV2Heavy.compute_t_matrix()."""
        s2 = 1.0 / np.sqrt(2.0)
        alphas = [0.0, 5.05, 1.91, np.pi]
        betas = [0.0, np.pi / 2, 3 * np.pi / 4, np.pi]
        gammas = [0.0, 4.15, 1.01, np.pi]

        def Rz(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        base_pos_h = np.array([0.156, 0.111, 0.085])
        base_dir_h = np.array([s2, -s2, 0.0])
        base_pos_v = np.array([0.120, 0.218, 0.0])
        base_dir_v = np.array([0.0, 0.0, 1.0])

        T = np.zeros((6, 8), dtype=np.float32)
        for i in range(4):
            pos = Rz(alphas[i]) @ base_pos_h
            dirn = Rz(betas[i]) @ base_dir_h
            T[:3, i] = dirn
            T[3:, i] = np.cross(pos, dirn)
        for j in range(4):
            pos = Rz(gammas[j]) @ base_pos_v
            T[:3, 4 + j] = base_dir_v
            T[3:, 4 + j] = np.cross(pos, base_dir_v)
        return T

    def _build(self) -> None:
        p = _get_pybullet()
        self._client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self._client_id)
        p.setTimeStep(self.dt, physicsClientId=self._client_id)

        # Create collision shape and body
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1, physicsClientId=self._client_id)
        inertia = [self.Ix, self.Iy, self.Iz]
        self._body_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=col,
            basePosition=self.target_pos.tolist(),
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client_id,
        )
        # Set inertia explicitly
        p.changeDynamics(
            self._body_id, -1,
            localInertiaDiagonal=inertia,
            physicsClientId=self._client_id,
        )
        self._built = True

    def close(self) -> None:
        if self._built and self._client_id is not None:
            p = _get_pybullet()
            p.disconnect(self._client_id)
            self._client_id = None
        self._built = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if not self._built:
            self._build()

        p = _get_pybullet()
        p.resetBasePositionAndOrientation(
            self._body_id,
            self.target_pos.tolist(),
            [0.0, 0.0, 0.0, 1.0],
            physicsClientId=self._client_id,
        )
        p.resetBaseVelocity(self._body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client_id)

        self._prev_action = np.zeros(6, dtype=np.float32)
        self._prev_wrench = np.zeros(6, dtype=np.float32)
        self._nu_prev = np.zeros(6, dtype=np.float32)
        self._nu_dot_prev = np.zeros(6, dtype=np.float32)
        self._u_eff_prev = np.zeros(self.n_thrusters, dtype=np.float32)
        self._step_count = 0

        obs = self._get_obs()
        info: dict[str, Any] = {"target_position": self.target_pos.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if not self._built:
            raise RuntimeError("Call reset() first")

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._prev_action = action.copy()

        p = _get_pybullet()

        # Map 6-DOF wrench to thrusters via T_matrix pinv
        u_cmd = np.clip(action @ self._t_pinv.T, -1.0, 1.0).astype(np.float32)

        # Thruster low-pass (matching Tier1 tau_lag=0.1)
        tau_lag = 0.1
        alpha_lp = self.dt / (tau_lag + self.dt)
        u_eff = self._u_eff_prev + alpha_lp * (u_cmd - self._u_eff_prev)
        self._u_eff_prev = u_eff.copy()

        # Compute wrench from thrusters
        thruster_forces = u_eff * self.max_thrust
        wrench_thrust = self._t_matrix @ thruster_forces

        # Get current state
        pos, quat_wxyz = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client_id)
        vel_world, angvel_world = p.getBaseVelocity(self._body_id, physicsClientId=self._client_id)

        pos = np.array(pos, dtype=np.float32)
        quat_xyzw = np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]], dtype=np.float32)
        quat_xyzw_wxyz = np.array(quat_wxyz, dtype=np.float32)  # wxyz for PyBullet

        # Body velocity via inverse rotation
        R = self._quat_to_rotmat(quat_xyzw)
        vel_body = R.T @ np.array(vel_world, dtype=np.float32)
        angvel_body = R.T @ np.array(angvel_world, dtype=np.float32)
        nu = np.concatenate([vel_body, angvel_body])

        # Fossen Tier1 hydrodynamics (CPU, identical to Warp kernels)
        hydro_wrench = self._compute_hydro_wrench(nu, quat_xyzw)
        total_wrench = hydro_wrench
        total_wrench[:3] += wrench_thrust[:3]
        total_wrench[3:] += wrench_thrust[3:]

        self._prev_wrench = total_wrench.copy()

        # Apply force/torque in world frame
        force_world = R @ total_wrench[:3]
        torque_world = R @ total_wrench[3:]
        p.applyExternalForce(
            self._body_id, -1, force_world.tolist(), [0, 0, 0], p.LINK_FRAME,
            physicsClientId=self._client_id,
        )
        p.applyExternalTorque(
            self._body_id, -1, torque_world.tolist(), p.LINK_FRAME,
            physicsClientId=self._client_id,
        )

        p.stepSimulation(physicsClientId=self._client_id)
        self._step_count += 1

        # Compute reward
        reward, reward_components = self._compute_reward()

        terminated = False
        truncated = self._step_count >= self.max_episode_steps

        # Out-of-bounds
        pos_new, _ = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client_id)
        pos_new = np.array(pos_new, dtype=np.float32)
        oob = np.linalg.norm(pos_new - self.target_pos) > 5.0
        if oob:
            terminated = True

        obs = self._get_obs()
        info: dict[str, Any] = reward_components
        return obs, reward, terminated, truncated, info

    def _compute_hydro_wrench(self, nu: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        """Fossen Tier1 hydro: added mass + damping + Coriolis + restoring."""
        wrench = np.zeros(6, dtype=np.float32)

        # EMA for nu_dot
        nu_dot = self._ema_alpha * (nu - self._nu_prev) / self.dt + (1 - self._ema_alpha) * self._nu_dot_prev
        self._nu_prev = nu.copy()
        self._nu_dot_prev = nu_dot.copy()

        # Added mass: -M_A * nu_dot
        wrench[:3] -= self.added_mass[:3] * nu_dot[:3]
        wrench[3:] -= self.added_mass[3:] * nu_dot[3:]

        # Damping: -(d_lin + d_quad * |nu|) * nu
        wrench[:3] -= (self.d_lin[:3] + self.d_quad[:3] * np.abs(nu[:3])) * nu[:3]
        wrench[3:] -= (self.d_lin[3:] + self.d_quad[3:] * np.abs(nu[3:])) * nu[3:]

        # Coriolis added mass: -C_A(nu) * nu
        # Simplified diagonal approximation matching Warp kernel
        nu1, nu2, nu3 = nu[0], nu[1], nu[2]
        nu4, nu5, nu6 = nu[3], nu[4], nu[5]
        ma_x, ma_y, ma_z = self.added_mass[0], self.added_mass[1], self.added_mass[2]
        ma_p, ma_q, ma_r = self.added_mass[3], self.added_mass[4], self.added_mass[5]

        # C_A from Fossen (diagonal approximation)
        wrench[0] -= -ma_y * nu5 * nu2 + ma_z * nu6 * nu3
        wrench[1] -= ma_x * nu4 * nu1 - ma_z * nu6 * nu3
        wrench[2] -= -ma_x * nu4 * nu1 + ma_y * nu5 * nu2
        wrench[3] -= ma_z * nu5 * nu6 - ma_q * nu5 * nu5 + ma_r * nu6 * nu6 - ma_q * nu4 * nu6
        wrench[4] -= -ma_z * nu4 * nu6 + ma_p * nu4 * nu4 - ma_r * nu4 * nu6 + ma_p * nu4 * nu6
        wrench[5] -= ma_y * nu4 * nu5 - ma_p * nu4 * nu4 + ma_q * nu5 * nu5 - ma_p * nu4 * nu5

        # Restoring: buoyancy - gravity
        R = self._quat_to_rotmat(quat_xyzw)
        buoyancy = self.rho_water * self.g_accel * self.volume
        gravity = self.mass * self.g_accel
        # Net upward force in world frame
        f_restore_world = np.array([0.0, 0.0, buoyancy - gravity], dtype=np.float32)
        f_restore_body = R.T @ f_restore_world
        wrench[:3] += f_restore_body

        # Restoring moment from COB offset
        # coBM along body z-axis (COB above COG)
        wrench[3] += f_restore_body[2] * 0.0  # rx=0
        wrench[4] += -f_restore_body[2] * 0.0  # ry=0
        wrench[5] += 0.0
        # Correct restoring moment: -g * (m - rho*V) * r_g + g * rho * V * r_b
        # With COB offset coBM along body z:
        wrench[3] += -(buoyancy - gravity) * 0.0  # simplified for small coBM
        wrench[4] += -(buoyancy - gravity) * 0.0
        # The dominant restoring moment from roll/pitch via buoyancy-gravity couple
        # M_restore = (r_g x W_b) + (r_b x B_b) in body frame
        # r_g = 0 (CoG at origin), r_b = [0, 0, coBM]
        r_b = np.array([0.0, 0.0, self.coBM], dtype=np.float32)
        B_body = np.array([0.0, 0.0, buoyancy], dtype=np.float32)
        W_body = np.array([0.0, 0.0, -gravity], dtype=np.float32)
        moment = np.cross(r_b, B_body) + np.cross(np.zeros(3), W_body)
        wrench[3:] += moment

        return wrench

    @staticmethod
    def _quat_to_rotmat(q_xyzw: np.ndarray) -> np.ndarray:
        """Quaternion (xyzw) to 3x3 rotation matrix."""
        qx, qy, qz, qw = q_xyzw
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        p = _get_pybullet()
        pos, quat_wxyz = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client_id)
        vel_world, angvel_world = p.getBaseVelocity(self._body_id, physicsClientId=self._client_id)

        pos = np.array(pos, dtype=np.float32)
        quat_xyzw = np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]], dtype=np.float32)
        R = self._quat_to_rotmat(quat_xyzw)
        vel_body = R.T @ np.array(vel_world, dtype=np.float32)
        angvel_body = R.T @ np.array(angvel_world, dtype=np.float32)

        obs = np.zeros(26, dtype=np.float32)
        obs[0:3] = self.target_pos - pos
        obs[3:7] = quat_xyzw
        obs[7:10] = vel_body
        obs[10:13] = angvel_body
        obs[13] = self.target_pos[2] - pos[2]

        q = quat_xyzw
        obs[14] = 2.0 * np.arctan2(q[2], q[3])
        obs[15] = 2.0 * np.arcsin(np.clip(-q[0]*q[2] + q[1]*q[3], -1, 1))

        obs[16:22] = self._prev_action
        obs[22:26] = self._prev_wrench[2:6]

        if self.sensor_noise_std > 0:
            obs += np.random.normal(0, self.sensor_noise_std, obs.shape).astype(np.float32)

        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _compute_reward(self) -> tuple[np.ndarray, dict[str, Any]]:
        p = _get_pybullet()
        pos, quat_wxyz = p.getBasePositionAndOrientation(self._body_id, physicsClientId=self._client_id)
        vel_world, angvel_world = p.getBaseVelocity(self._body_id, physicsClientId=self._client_id)

        pos = np.array(pos, dtype=np.float32)
        quat_xyzw = np.array([quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]], dtype=np.float32)
        R = self._quat_to_rotmat(quat_xyzw)
        vel_body = R.T @ np.array(vel_world, dtype=np.float32)
        angvel_body = R.T @ np.array(angvel_world, dtype=np.float32)

        pos_err = np.linalg.norm(self.target_pos - pos)
        depth_err = np.abs(self.target_pos[2] - pos[2])
        vel = np.concatenate([vel_body, angvel_body])
        vel_norm = np.linalg.norm(vel)
        act_norm = np.linalg.norm(self._prev_action)

        q = quat_xyzw
        roll = 2.0 * np.arctan2(q[2], q[3])
        pitch = 2.0 * np.arcsin(np.clip(-q[0]*q[2] + q[1]*q[3], -1, 1))
        heading_err = np.abs(roll) + np.abs(pitch)

        r_pos = -self._w_pos * pos_err
        r_vel = -self._w_vel * vel_norm
        r_act = -self._w_act * act_norm
        r_depth = -self._w_depth * depth_err
        r_heading = -self._w_heading * heading_err

        reward = float(r_pos + r_vel + r_act + r_depth + r_heading)
        info = {
            "reward_distance": float(r_pos),
            "reward_action": float(r_act),
            "reward_velocity": float(r_vel),
            "reward_depth": float(r_depth),
            "reward_heading": float(r_heading),
        }
        return np.float32(reward), info
