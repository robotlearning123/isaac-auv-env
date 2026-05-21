"""ROVEnv — Gymnasium wrapper for GPU-batched ROV simulation with Newton + Tier1 hydro.

Vectorized env: n_envs parallel worlds on a single GPU.
Observation: 26-dim per env.
Action: 6-dim (surge, sway, heave, roll, pitch, yaw) in [-1, +1].
Reward: station-keeping (penalize position error + velocity + action effort).
Supports partial reset of individual env indices without rebuilding the model.

Usage:
    env = ROVEnv(n_envs=64, device="cuda")
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import warp as wp
from gymnasium import spaces

from oceanscale.fluid.grid import GridFluidSolver
from oceanscale.hydro.tier1 import DEFAULT_DT, RandomizationRanges, Tier1
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


def _default_bluerov_coeffs() -> dict[str, Any]:
    """BlueROV2 Heavy coefficients from von Benzon et al. 2022."""
    vehicle = BlueROV2Heavy()
    return vehicle.set_coeffs_kwargs()


class ROVEnv(gym.Env):
    """GPU-batched ROV environment with Tier-1 Fossen hydrodynamics.

    Each of n_envs worlds contains a single free-body sphere (ROV proxy)
    with 6-DOF hydrodynamic forces computed by Tier1 kernels.

    Observation (26-dim):
        [0:3]   position error (target - current xyz)
        [3:7]   orientation quaternion (xyzw)
        [7:10]  linear velocity (body frame)
        [10:13] angular velocity (body frame)
        [13]    depth error scalar
        [14:16] heading error (roll, pitch)
        [16:22] previous action (6-dim)
        [22:26] previous hydro wrench (force_z, torque_x, torque_y, torque_z)

    Action (6-dim): [-1, +1] per DOF (surge, sway, heave, roll, pitch, yaw).
    Mapped to n_thrusters via diagonal T_matrix allocation.

    n_envs=1: returns squeezed single-env shapes (obs=(26,), reward=float).
    n_envs>1: returns batched shapes (obs=(n,26), reward=(n,)).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        n_envs: int = 64,
        n_thrusters: int = 8,
        device: str = "cuda",
        dt: float = DEFAULT_DT,
        max_episode_steps: int = 1000,
        target_pos: np.ndarray | None = None,
        coeffs: dict[str, Any] | None = None,
        sensor_noise_std: float = 0.02,
        reward_weights: dict[str, float] | None = None,
        use_fluid: bool = False,
        fluid_config: dict[str, Any] | None = None,
        current_velocity: np.ndarray | None = None,
        use_domain_randomization: bool = False,
        randomization_ranges: RandomizationRanges | None = None,
    ) -> None:
        super().__init__()

        self.n_envs = n_envs
        self.n_thrusters = n_thrusters
        self.device = device
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self._single = n_envs == 1
        self.sensor_noise_std = sensor_noise_std
        self.use_fluid = use_fluid
        self.fluid_config = fluid_config or {}
        self.current_velocity = (
            np.asarray(current_velocity, dtype=np.float32)
            if current_velocity is not None
            else np.array([0.15, 0.0, 0.0], dtype=np.float32)
        )
        self.use_domain_randomization = use_domain_randomization
        self.randomization_ranges = randomization_ranges or RandomizationRanges()

        if target_pos is None:
            target_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)

        if coeffs is None:
            coeffs = _default_bluerov_coeffs()
        self.coeffs = coeffs

        # Extract T_matrix from coeffs if provided (BlueROV2 Heavy path)
        self._t_matrix = coeffs.get("T_matrix")

        w = reward_weights or {}
        self._w_pos = w.get("pos", 1.0)
        self._w_vel = w.get("vel", 0.1)
        self._w_act = w.get("act", 0.01)
        self._w_depth = w.get("depth", 2.0)
        self._w_heading = w.get("heading", 0.5)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._built = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Lazy-build Newton model + Tier1 hydro on first reset."""
        import newton

        wp.init()

        template = newton.ModelBuilder()
        template.gravity = 0.0  # Tier1 restoring kernel handles buoyancy/gravity
        body = template.add_body(mass=self.coeffs["mass"])
        template.add_shape_sphere(body, radius=0.1)
        template.joint_q = [
            self.target_pos[0], self.target_pos[1], self.target_pos[2],
            0.0, 0.0, 0.0, 1.0,
        ]
        template.joint_qd = [0.0] * 6

        scene = newton.ModelBuilder()
        scene.replicate(template, world_count=self.n_envs, spacing=(0.0, 0.0, 0.0))
        self.model = scene.finalize(device=self.device)

        self.state_curr = self.model.state()
        self.state_next = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_curr)

        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        self.tier1 = Tier1(
            n_envs=self.n_envs,
            n_thrusters=self.n_thrusters,
            device=self.device,
        )
        tier1_kwargs = dict(self.coeffs)
        if self._t_matrix is not None:
            tier1_kwargs["T_matrix"] = self._t_matrix
        self.tier1.set_coeffs(**tier1_kwargs)

        self._u_cmd = wp.zeros(
            (self.n_envs, self.n_thrusters), dtype=wp.float32, device=self.device
        )

        self._prev_action = np.zeros((self.n_envs, 6), dtype=np.float32)
        self._prev_wrench = np.zeros((self.n_envs, 6), dtype=np.float32)
        self._step_count = np.zeros(self.n_envs, dtype=np.int32)
        self._done = np.zeros(self.n_envs, dtype=bool)
        self._last_info: dict[str, Any] = {}

        # Fluid solver for ocean current FSI coupling
        self._fluid: GridFluidSolver | None = None
        self._fluid_positions: wp.array | None = None
        if self.use_fluid:
            self._build_fluid_solver()

        self._built = True

    def close(self) -> None:
        """Release GPU resources."""
        self._built = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset envs. Returns squeezed (26,) for single, (n,26) for batched."""
        super().reset(seed=seed)
        if not self._built:
            self._build()

        info: dict[str, Any] = {"target_position": self.target_pos.copy()}

        if options is not None and "target_position" in options:
            new_target = np.asarray(options["target_position"], dtype=np.float32)
            self.target_pos = new_target
            info["target_position"] = new_target.copy()

        if env_ids is not None:
            env_ids = np.asarray(env_ids, dtype=np.int32)
            self._reset_indices(env_ids)
            obs = self._get_flat_obs()
            return obs, info

        self._reset_indices(np.arange(self.n_envs))
        obs = self._get_flat_obs()
        return obs, info

    def reset_envs(self, env_ids: list[int] | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Partial reset: reset only specified env indices."""
        return self.reset(env_ids=env_ids)

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Step all envs. Always returns batched shapes."""
        if not self._built:
            raise RuntimeError("Call reset() first")

        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 1:
            action = np.broadcast_to(action, (self.n_envs, 6)).copy()

        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action.copy()

        # Map 6-DOF action to n_thrusters via diagonal allocation
        u_cmd = np.zeros((self.n_envs, self.n_thrusters), dtype=np.float32)
        for i in range(min(6, self.n_thrusters)):
            u_cmd[:, i] = action[:, i]
        self._u_cmd = wp.array(u_cmd, dtype=wp.float32, device=self.device)

        # Tier1 hydro wrench
        self.tier1.compute_wrench(
            nu=self.state_curr.body_qd,
            quat=self._extract_quat(),
            u_cmd=self._u_cmd,
            dt=self.dt,
        )
        self.tier1.write_to_body_f(self.state_curr.body_f)

        # FSI: ocean current drag force
        if self._fluid is not None:
            self._apply_ocean_current()

        self.solver.step(
            self.state_curr, self.state_next, self.control, None, self.dt
        )
        wp.synchronize()

        self.state_curr, self.state_next = self.state_next, self.state_curr

        self._prev_wrench = self.tier1.wrench_buf.numpy().copy()
        self._step_count += 1

        # Compute rewards and done flags
        reward, reward_components = self._compute_reward_with_components()
        terminated = np.zeros(self.n_envs, dtype=bool)
        truncated = self._step_count >= self.max_episode_steps

        # Out-of-bounds termination (>5m from target)
        pos, _, _ = self._get_body_state()
        oob = np.linalg.norm(pos - self.target_pos, axis=-1) > 5.0
        terminated = terminated | oob

        self._done = terminated | truncated

        # Auto-reset divergent or out-of-bounds envs
        divergent = ~np.isfinite(reward) | oob
        if np.any(divergent):
            div_ids = np.where(divergent)[0]
            self._reset_indices(div_ids)
            reward[div_ids] = 0.0
            terminated[div_ids] = False
            truncated[div_ids] = False

        obs = self._get_flat_obs()
        info: dict[str, Any] = reward_components
        return obs, reward, terminated, truncated, info

    def action_space_sample(self) -> np.ndarray:
        """Sample actions for all envs. Shape: (n_envs, 6)."""
        return np.stack([self.action_space.sample() for _ in range(self.n_envs)])

    def _build_fluid_solver(self) -> None:
        """Initialize GridFluidSolver and pre-compute steady-state current field."""
        grid_res = self.fluid_config.get("grid_res", 32)
        domain_size = self.fluid_config.get("domain_size", 10.0)
        viscosity = self.fluid_config.get("viscosity", 0.001)

        self._fluid = GridFluidSolver(
            grid_res=grid_res,
            domain_size=domain_size,
            viscosity=viscosity,
            device=self.device,
        )

        # Inject source once and warm up to establish steady-state field
        src_pos = (grid_res / 2.0, grid_res / 2.0, grid_res / 2.0)
        src_vel = tuple(self.current_velocity.tolist())
        src_radius = float(grid_res / 3)
        self._fluid.add_source(
            position=src_pos,
            velocity=src_vel,
            radius=src_radius,
        )
        for _ in range(10):
            self._fluid.step(dt=self.dt)

        # Pre-compute mean field velocity for force scaling
        vel_field = self._fluid.get_velocity_field()
        self._fluid_mean_vel = np.mean(vel_field.numpy(), axis=0).astype(np.float32)

        self._fluid_positions = wp.zeros(
            self.n_envs, dtype=wp.vec3, device=self.device
        )

    def _apply_ocean_current(self) -> None:
        """Sample pre-computed fluid velocity at ROV positions and apply drag force."""
        body_q = self.state_curr.body_q.numpy()
        positions = body_q[:, 0:3].copy()

        # Shift positions into fluid grid space (grid covers [0, domain_size])
        domain_size = self._fluid.domain_size
        half = domain_size / 2.0
        positions += half  # center domain at origin

        self._fluid_positions = wp.array(positions, dtype=wp.vec3, device=self.device)

        # Sample pre-computed velocity field (no stepping)
        fluid_vel = self._fluid.sample_velocity_at(self._fluid_positions)
        fluid_vel_np = fluid_vel.numpy()  # (n_envs, 3) world frame

        # Body velocity in world frame
        body_qd = self.state_curr.body_qd.numpy()
        body_vel_body = body_qd[:, 0:3]  # linear velocity in body frame

        # Rotate body velocity to world frame using quaternion
        quat = body_q[:, 3:7]  # xyzw
        body_vel_world = self._rotate_to_world(quat, body_vel_body)

        # Relative velocity (fluid - body) in world frame
        v_rel = fluid_vel_np - body_vel_world

        # FSI drag: F = coupling_strength * v_rel (Morison linear drag)
        # Independent of vehicle damping — models current force on submerged body
        coupling_strength = self.fluid_config.get("coupling_strength", 8.0)
        force_world = coupling_strength * v_rel
        force_body = self._rotate_to_body(quat, force_world)

        # Apply to body_f (additive)
        body_f = self.state_curr.body_f.numpy()
        body_f[:, 0:3] += force_body
        wp.copy(
            self.state_curr.body_f,
            wp.array(body_f, dtype=wp.float32, device=self.device),
        )

    @staticmethod
    def _rotate_to_world(
        quat: np.ndarray, vec_body: np.ndarray
    ) -> np.ndarray:
        """Rotate vectors from body to world frame using quaternion (xyzw)."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec_body[:, 0], vec_body[:, 1], vec_body[:, 2]

        # q * v * q_conj
        t0 = 2.0 * (qy * vz - qz * vy)
        t1 = 2.0 * (qz * vx - qx * vz)
        t2 = 2.0 * (qx * vy - qy * vx)

        result = np.stack([
            vx + qw * t0 + qy * t2 - qz * t1,
            vy + qw * t1 + qz * t0 - qx * t2,
            vz + qw * t2 + qx * t1 - qy * t0,
        ], axis=-1)
        return result.astype(np.float32)

    @staticmethod
    def _rotate_to_body(
        quat: np.ndarray, vec_world: np.ndarray
    ) -> np.ndarray:
        """Rotate vectors from world to body frame using quaternion (xyzw)."""
        qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]

        # q_conj * v * q
        t0 = 2.0 * (qy * vz - qz * vy)
        t1 = 2.0 * (qz * vx - qx * vz)
        t2 = 2.0 * (qx * vy - qy * vx)

        result = np.stack([
            vx - qw * t0 - qy * t2 + qz * t1,
            vy - qw * t1 - qz * t0 + qx * t2,
            vz - qw * t2 - qx * t1 + qy * t0,
        ], axis=-1)
        return result.astype(np.float32)

    def _get_flat_obs(self) -> np.ndarray:
        """Get observation for all envs. Always returns (n_envs, 26)."""
        return self._get_obs_for(np.arange(self.n_envs))

    def _get_obs(self) -> np.ndarray:
        """Alias for _get_flat_obs()."""
        return self._get_flat_obs()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _extract_quat(self) -> wp.array:
        """Extract quaternion from body_q (transform: xyz + quat_xyzw)."""
        body_q = self.state_curr.body_q.numpy()
        quat = body_q[:, 3:7].copy()
        return wp.array(quat, dtype=wp.quatf, device=self.device)

    def _get_body_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, quaternion, velocity) arrays, each (n_envs, ...)."""
        body_q = self.state_curr.body_q.numpy()
        body_qd = self.state_curr.body_qd.numpy()
        pos = body_q[:, 0:3]
        quat = body_q[:, 3:7]
        vel = body_qd[:, 0:6]
        return pos, quat, vel

    def _get_obs_for(self, env_ids: np.ndarray) -> np.ndarray:
        """Compute observations for specified envs. Shape: (len(env_ids), 26)."""
        pos, quat, vel = self._get_body_state()

        obs = np.zeros((len(env_ids), 26), dtype=np.float32)

        # Position error (target - current)
        obs[:, 0:3] = self.target_pos - pos[env_ids]

        # Orientation quaternion (xyzw)
        obs[:, 3:7] = quat[env_ids]

        # Body velocity: linear + angular
        obs[:, 7:10] = vel[env_ids, 0:3]
        obs[:, 10:13] = vel[env_ids, 3:6]

        # Depth error
        depth_err = self.target_pos[2] - pos[env_ids, 2]
        obs[:, 13] = depth_err

        # Heading error (roll, pitch from quaternion, small-angle approximation)
        q = quat[env_ids]
        roll_err = 2.0 * np.arctan2(q[:, 2], q[:, 3])
        pitch_err = 2.0 * np.arcsin(np.clip(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1))
        obs[:, 14] = roll_err
        obs[:, 15] = pitch_err

        # Previous action
        obs[:, 16:22] = self._prev_action[env_ids]

        # Previous wrench summary (fz, mx, my, mz)
        obs[:, 22:26] = self._prev_wrench[env_ids, 2:6]

        # Sensor noise
        if self.sensor_noise_std > 0:
            obs += self.np_random.normal(0, self.sensor_noise_std, obs.shape).astype(np.float32)

        # Clamp non-finite values
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _compute_reward_with_components(
        self,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Station-keeping reward with component breakdown."""
        pos, _, vel = self._get_body_state()

        pos_err = np.linalg.norm(self.target_pos - pos, axis=-1)
        depth_err = np.abs(self.target_pos[2] - pos[:, 2])
        vel_norm = np.linalg.norm(vel, axis=-1)
        act_norm = np.linalg.norm(self._prev_action, axis=-1)

        q = self.state_curr.body_q.numpy()[:, 3:7]
        roll = 2.0 * np.arctan2(q[:, 2], q[:, 3])
        pitch = 2.0 * np.arcsin(np.clip(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1))
        heading_err = np.abs(roll) + np.abs(pitch)

        r_pos = -self._w_pos * pos_err
        r_vel = -self._w_vel * vel_norm
        r_act = -self._w_act * act_norm
        r_depth = -self._w_depth * depth_err
        r_heading = -self._w_heading * heading_err

        reward = (r_pos + r_vel + r_act + r_depth + r_heading).astype(np.float32)

        info = {
            "reward_distance": float(np.mean(r_pos)),
            "reward_action": float(np.mean(r_act)),
            "reward_velocity": float(np.mean(r_vel)),
            "reward_depth": float(np.mean(r_depth)),
            "reward_heading": float(np.mean(r_heading)),
        }
        return reward, info

    def _reset_indices(self, env_ids: np.ndarray) -> None:
        """Reset specified env indices to initial state."""
        import newton

        target = self.target_pos

        joint_q = self.model.joint_q.numpy()
        for idx in env_ids:
            base = idx * 7
            joint_q[base:base + 7] = [target[0], target[1], target[2], 0.0, 0.0, 0.0, 1.0]
        self.model.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.device)

        joint_qd = self.model.joint_qd.numpy()
        for idx in env_ids:
            base = idx * 6
            joint_qd[base:base + 6] = 0.0
        self.model.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=self.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_curr)

        # Reset Tier1 internal buffers for specified envs
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

        # Domain randomization: perturb hydrodynamic coefficients per env
        if self.use_domain_randomization and self._built:
            self.tier1.randomize_coeffs(
                env_ids=env_ids,
                ranges=self.randomization_ranges,
            )

        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
