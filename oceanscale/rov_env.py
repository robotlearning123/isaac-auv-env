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
import torch
import warp as wp
from gymnasium import spaces

from oceanscale.fluid.grid import GridFluidSolver
from oceanscale.hydro.tier1 import DEFAULT_DT, RandomizationRanges, Tier1
from oceanscale.hydro.tier1_kernels import tier1_zero_wrench
from oceanscale.vehicles.bluerov2 import BlueROV2Heavy


def _default_bluerov_coeffs() -> dict[str, Any]:
    """BlueROV2 Heavy coefficients from von Benzon et al. 2022."""
    vehicle = BlueROV2Heavy()
    return vehicle.set_coeffs_kwargs()


def _rotate_to_world_torch(
    quat: torch.Tensor, vec_body: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors from body to world frame using quaternion (xyzw) on GPU."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_body[:, 0], vec_body[:, 1], vec_body[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack([
        vx + qw * t0 + qy * t2 - qz * t1,
        vy + qw * t1 + qz * t0 - qx * t2,
        vz + qw * t2 + qx * t1 - qy * t0,
    ], dim=-1)


def _rotate_to_body_torch(
    quat: torch.Tensor, vec_world: torch.Tensor
) -> torch.Tensor:
    """Rotate vectors from world to body frame using quaternion (xyzw) on GPU."""
    qx, qy, qz, qw = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    vx, vy, vz = vec_world[:, 0], vec_world[:, 1], vec_world[:, 2]
    t0 = 2.0 * (qy * vz - qz * vy)
    t1 = 2.0 * (qz * vx - qx * vz)
    t2 = 2.0 * (qx * vy - qy * vx)
    return torch.stack([
        vx - qw * t0 - qy * t2 + qz * t1,
        vy - qw * t1 - qz * t0 + qx * t2,
        vz - qw * t2 - qx * t1 + qy * t0,
    ], dim=-1)


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
    Mapped to n_thrusters via T_matrix pseudoinverse (Moore-Penrose) allocation.

    `init_pos_noise_std` and `init_yaw_noise_std` are uniform half-ranges, not
    Gaussian stddev: reset samples uniformly in [-std, +std].

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
        init_pos_noise_std: float = 0.5,
        init_yaw_noise_std: float = 0.5,
        current_drag_coeff: float = 5.0,
        gpu_obs: bool = False,
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
        self.init_pos_noise_std = init_pos_noise_std
        self.init_yaw_noise_std = init_yaw_noise_std
        self.current_drag_coeff = current_drag_coeff
        self.gpu_obs = gpu_obs

        if target_pos is None:
            target_pos = np.array([0.0, 0.0, -1.5], dtype=np.float32)
        self.target_pos = np.asarray(target_pos, dtype=np.float32)

        if coeffs is None:
            coeffs = _default_bluerov_coeffs()
        self.coeffs = coeffs

        # Extract T_matrix from coeffs if provided (BlueROV2 Heavy path)
        self._t_matrix = coeffs.get("T_matrix")

        # Precompute pseudoinverse for wrench→thruster allocation
        if self._t_matrix is not None:
            self._t_pinv = np.linalg.pinv(np.array(self._t_matrix, dtype=np.float64)).astype(np.float32)
        else:
            self._t_pinv = None

        # reward_weights kept for API compat but unused (exponential shaping is fixed)
        self._w_pos = 0.5
        self._w_vel = 0.3
        self._w_act = 0.2

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        self._target_pos_torch = torch.from_numpy(self.target_pos).to(self.device)

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
        mass = self.coeffs["mass"]
        I_max = max(self.coeffs.get("Ix", 0.26), self.coeffs.get("Iy", 0.23), self.coeffs.get("Iz", 0.37))
        radius = (5.0 * I_max / (2.0 * mass)) ** 0.5
        body = template.add_body(mass=mass)
        template.add_shape_sphere(body, radius=radius)
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
        tier1_kwargs.pop("Ix", None)
        tier1_kwargs.pop("Iy", None)
        tier1_kwargs.pop("Iz", None)
        self.tier1.set_coeffs(**tier1_kwargs)

        self._u_cmd = wp.zeros(
            (self.n_envs, self.n_thrusters), dtype=wp.float32, device=self.device
        )

        self._thruster_gain = np.ones(self.n_envs, dtype=np.float32)
        self._thruster_gain_gpu = torch.ones(self.n_envs, device=self.device, dtype=torch.float32)

        self._prev_action = np.zeros((self.n_envs, 6), dtype=np.float32)
        self._prev_wrench = np.zeros((self.n_envs, 6), dtype=np.float32)
        self._step_count = np.zeros(self.n_envs, dtype=np.int32)
        self._done = np.zeros(self.n_envs, dtype=bool)
        self._last_info: dict[str, Any] = {}

        # GPU-side persistent tensors (zero-copy via wp.to_torch)
        self._prev_action_gpu = torch.zeros(
            self.n_envs, 6, device=self.device, dtype=torch.float32
        )
        self._prev_wrench_gpu = torch.zeros(
            self.n_envs, 6, device=self.device, dtype=torch.float32
        )
        self._quat_buf = wp.zeros(self.n_envs, dtype=wp.quatf, device=self.device)

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
            self._target_pos_torch = torch.from_numpy(new_target).to(self.device)
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

    def reset_torch(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        env_ids: list[int] | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """GPU-native reset: returns CUDA tensor obs, no CPU transfer."""
        super().reset(seed=seed)
        if not self._built:
            self._build()

        info: dict[str, Any] = {"target_position": self.target_pos.copy()}

        if options is not None and "target_position" in options:
            new_target = np.asarray(options["target_position"], dtype=np.float32)
            self.target_pos = new_target
            self._target_pos_torch = torch.from_numpy(new_target).to(self.device)
            info["target_position"] = new_target.copy()

        if env_ids is not None:
            self._reset_indices(np.asarray(env_ids, dtype=np.int32))
        else:
            self._reset_indices(np.arange(self.n_envs))

        obs = self._get_obs_torch()
        return obs, info

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
        self._prev_action_gpu = torch.tensor(
            action, device=self.device, dtype=torch.float32
        )

        # Map 6-DOF wrench action to n_thrusters via T_matrix pseudoinverse
        if self._t_pinv is not None:
            u_cmd = np.clip(action @ self._t_pinv.T, -1.0, 1.0).astype(np.float32)
        else:
            u_cmd = np.zeros((self.n_envs, self.n_thrusters), dtype=np.float32)
            for i in range(min(6, self.n_thrusters)):
                u_cmd[:, i] = action[:, i]
        u_cmd *= self._thruster_gain[:, np.newaxis]
        self._u_cmd = wp.array(u_cmd, dtype=wp.float32, device=self.device)

        # Zero body forces before applying new hydro wrench
        wp.launch(
            tier1_zero_wrench, dim=self.n_envs,
            inputs=[self.state_curr.body_f], device=self.device,
        )

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

        # Domain-randomized ocean current force
        if self.use_domain_randomization and self.tier1.current_vec_arr is not None:
            self._apply_dr_current()

        self.solver.step(
            self.state_curr, self.state_next, self.control, None, self.dt
        )
        wp.synchronize()

        self.state_curr, self.state_next = self.state_next, self.state_curr

        # GPU-side wrench snapshot (no CPU transfer)
        wrench_t = wp.to_torch(self.tier1.wrench_buf)
        self._prev_wrench_gpu.copy_(wrench_t.reshape_as(self._prev_wrench_gpu))
        self._step_count += 1

        # Compute rewards and done flags (GPU torch)
        reward, reward_components = self._compute_reward_with_components()

        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        step_t = torch.from_numpy(self._step_count).to(self.device)
        truncated = step_t >= self.max_episode_steps

        # Out-of-bounds termination (>5m from target)
        pos, _, _ = self._get_body_state_torch()
        oob = torch.linalg.norm(pos - self._target_pos_torch, dim=-1) > 5.0
        terminated = terminated | oob

        self._done = (terminated | truncated).cpu().numpy()

        # Auto-reset divergent or out-of-bounds envs
        divergent = ~torch.isfinite(reward) | oob
        if torch.any(divergent):
            div_ids = torch.where(divergent)[0].cpu().numpy()
            self._reset_indices(div_ids)
            reward[div_ids] = 0.0
            terminated[div_ids] = False
            truncated[div_ids] = False

        obs = self._get_flat_obs()
        info: dict[str, Any] = reward_components
        if self.gpu_obs:
            return obs, reward, terminated, truncated, info
        return obs, reward.cpu().numpy(), terminated.cpu().numpy(), truncated.cpu().numpy(), info

    def step_torch(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """GPU-native step: accepts and returns CUDA tensors. No CPU transfer."""
        if not self._built:
            raise RuntimeError("Call reset() or reset_torch() first")

        action = action.to(device=self.device, dtype=torch.float32)
        if action.ndim == 1:
            action = action.unsqueeze(0).expand(self.n_envs, -1).clone()
        action = action.clamp(-1.0, 1.0)

        self._prev_action_gpu.copy_(action)
        self._prev_action = action.cpu().numpy()

        # Thruster allocation on GPU (avoids numpy round-trip)
        if self._t_pinv is not None:
            if not hasattr(self, "_t_pinv_torch"):
                self._t_pinv_torch = torch.from_numpy(self._t_pinv).to(self.device)
            u_cmd = (action @ self._t_pinv_torch.T).clamp(-1.0, 1.0)
        else:
            u_cmd = torch.zeros(self.n_envs, self.n_thrusters, device=self.device, dtype=torch.float32)
            for i in range(min(6, self.n_thrusters)):
                u_cmd[:, i] = action[:, i]
        u_cmd *= self._thruster_gain_gpu.unsqueeze(1)
        self._u_cmd = wp.from_torch(u_cmd.contiguous(), dtype=wp.float32)

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

        if self._fluid is not None:
            self._apply_ocean_current()

        # Domain-randomized ocean current force
        if self.use_domain_randomization and self.tier1.current_vec_arr is not None:
            self._apply_dr_current()

        self.solver.step(
            self.state_curr, self.state_next, self.control, None, self.dt
        )
        wp.synchronize()

        self.state_curr, self.state_next = self.state_next, self.state_curr

        wrench_t = wp.to_torch(self.tier1.wrench_buf)
        self._prev_wrench_gpu.copy_(wrench_t.reshape_as(self._prev_wrench_gpu))
        self._step_count += 1

        reward, info = self._compute_reward_with_components()

        terminated = torch.zeros(self.n_envs, dtype=torch.bool, device=self.device)
        step_t = torch.from_numpy(self._step_count).to(self.device)
        truncated = step_t >= self.max_episode_steps

        pos, _, _ = self._get_body_state_torch()
        oob = torch.linalg.norm(pos - self._target_pos_torch, dim=-1) > 5.0
        terminated = terminated | oob

        self._done = (terminated | truncated).cpu().numpy()

        divergent = ~torch.isfinite(reward) | oob
        if torch.any(divergent):
            div_ids = torch.where(divergent)[0].cpu().numpy()
            self._reset_indices(div_ids)
            reward[div_ids] = 0.0
            terminated[div_ids] = False
            truncated[div_ids] = False

        obs = self._get_obs_torch()

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
        """Sample fluid velocity at ROV positions and apply drag force.

        All computation stays on GPU via wp.to_torch + torch ops.
        """
        body_q = wp.to_torch(self.state_curr.body_q)
        positions = body_q[:, 0:3].clone()

        domain_size = self._fluid.domain_size
        half = domain_size / 2.0
        positions += half

        self._fluid_positions = wp.from_torch(positions.contiguous(), dtype=wp.vec3)

        fluid_vel = self._fluid.sample_velocity_at(self._fluid_positions)
        fluid_vel_t = wp.to_torch(fluid_vel)

        body_qd = wp.to_torch(self.state_curr.body_qd)
        body_vel_body = body_qd[:, 0:3]

        quat = body_q[:, 3:7]
        body_vel_world = _rotate_to_world_torch(quat, body_vel_body)

        v_rel = fluid_vel_t - body_vel_world
        coupling_strength = self.fluid_config.get("coupling_strength", 8.0)
        force_world = coupling_strength * v_rel
        force_body = _rotate_to_body_torch(quat, force_world)

        body_f = wp.to_torch(self.state_curr.body_f)
        body_f[:, 0:3] += force_body

    def _apply_dr_current(self) -> None:
        """Apply domain-randomized ocean current as drag force (GPU-only)."""
        current_vec_t = wp.to_torch(self.tier1.current_vec_arr)  # (n, 3) world frame
        body_q = wp.to_torch(self.state_curr.body_q)
        body_qd = wp.to_torch(self.state_curr.body_qd)
        quat = body_q[:, 3:7]
        body_vel_world = _rotate_to_world_torch(quat, body_qd[:, 0:3])
        v_rel = current_vec_t - body_vel_world
        force_world = self.current_drag_coeff * v_rel
        force_body = _rotate_to_body_torch(quat, force_world)
        body_f = wp.to_torch(self.state_curr.body_f)
        body_f[:, 0:3] += force_body

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
        """Extract quaternion from body_q without CPU transfer."""
        body_q_t = wp.to_torch(self.state_curr.body_q)
        quat_dst = wp.to_torch(self._quat_buf)
        quat_dst.copy_(body_q_t[:, 3:7])
        return self._quat_buf

    def _get_body_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (position, quaternion, velocity) arrays, each (n_envs, ...)."""
        body_q = self.state_curr.body_q.numpy()
        body_qd = self.state_curr.body_qd.numpy()
        pos = body_q[:, 0:3]
        quat = body_q[:, 3:7]
        vel = body_qd[:, 0:6]
        return pos, quat, vel

    def _get_body_state_torch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GPU-only body state via wp.to_torch (zero-copy, no CPU transfer)."""
        body_q = wp.to_torch(self.state_curr.body_q)
        body_qd = wp.to_torch(self.state_curr.body_qd)
        return body_q[:, :3], body_q[:, 3:7], body_qd[:, :6]

    def _get_obs_for(self, env_ids: np.ndarray) -> np.ndarray:
        """Compute observations on GPU. Shape: (len(env_ids), 26)."""
        pos, quat, vel = self._get_body_state_torch()
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        n = len(env_ids)

        obs = torch.zeros(n, 26, device=self.device, dtype=torch.float32)

        # Position error (target - current)
        obs[:, 0:3] = self._target_pos_torch - pos[env_ids_t]

        # Orientation quaternion (xyzw)
        obs[:, 3:7] = quat[env_ids_t]

        # Body velocity: linear + angular
        obs[:, 7:10] = vel[env_ids_t, 0:3]
        obs[:, 10:13] = vel[env_ids_t, 3:6]

        # Depth error
        obs[:, 13] = self._target_pos_torch[2] - pos[env_ids_t, 2]

        # Heading error (roll, pitch from quaternion)
        q = quat[env_ids_t]
        obs[:, 14] = 2.0 * torch.atan2(q[:, 2], q[:, 3])
        obs[:, 15] = 2.0 * torch.asin(
            torch.clamp(-q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3], -1, 1)
        )

        # Previous action
        obs[:, 16:22] = self._prev_action_gpu[env_ids_t]

        # Previous wrench summary (fz, mx, my, mz)
        obs[:, 22:26] = self._prev_wrench_gpu[env_ids_t, 2:6]

        # Sensor noise
        if self.sensor_noise_std > 0:
            obs += torch.randn_like(obs) * self.sensor_noise_std

        # Clamp non-finite values
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        if self.gpu_obs:
            return obs
        return obs.cpu().numpy()

    def _get_obs_torch(self) -> torch.Tensor:
        """Observations as CUDA tensor. Shape: (n_envs, 26). No CPU transfer."""
        pos, quat, vel = self._get_body_state_torch()

        obs = torch.zeros(self.n_envs, 26, device=self.device, dtype=torch.float32)
        obs[:, 0:3] = self._target_pos_torch - pos
        obs[:, 3:7] = quat
        obs[:, 7:10] = vel[:, 0:3]
        obs[:, 10:13] = vel[:, 3:6]
        obs[:, 13] = self._target_pos_torch[2] - pos[:, 2]

        obs[:, 14] = 2.0 * torch.atan2(quat[:, 2], quat[:, 3])
        obs[:, 15] = 2.0 * torch.asin(
            torch.clamp(-quat[:, 0] * quat[:, 2] + quat[:, 1] * quat[:, 3], -1, 1)
        )

        obs[:, 16:22] = self._prev_action_gpu
        obs[:, 22:26] = self._prev_wrench_gpu[:, 2:6]

        if self.sensor_noise_std > 0:
            obs += torch.randn_like(obs) * self.sensor_noise_std

        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _compute_reward_with_components(
        self,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Exponential shaping reward: r = exp(-error/scale), always (0, 1]."""
        pos, quat, vel = self._get_body_state_torch()
        target = self._target_pos_torch

        pos_err = torch.linalg.norm(target - pos, dim=-1)
        vel_norm = torch.linalg.norm(vel, dim=-1)
        act_norm = torch.linalg.norm(self._prev_action_gpu, dim=-1)

        r_pos = torch.exp(-pos_err / 1.0)
        r_vel = torch.exp(-vel_norm / 0.5)
        r_act = torch.exp(-act_norm / 1.0)

        reward = 0.5 * r_pos + 0.3 * r_vel + 0.2 * r_act

        info = {
            "reward_distance": r_pos.mean().item(),
            "reward_velocity": r_vel.mean().item(),
            "reward_action": r_act.mean().item(),
        }
        return reward, info

    def _reset_indices(self, env_ids: np.ndarray) -> None:
        """Reset specified env indices to initial state with optional perturbation."""
        import newton

        target = self.target_pos
        n_reset = len(env_ids)

        # Sample perturbation offsets
        if self.init_pos_noise_std > 0 or self.init_yaw_noise_std > 0:
            pos_noise = self.np_random.uniform(-self.init_pos_noise_std, self.init_pos_noise_std, (n_reset, 3)).astype(np.float32)
            yaw_offsets = self.np_random.uniform(-self.init_yaw_noise_std, self.init_yaw_noise_std, n_reset).astype(np.float32)
        else:
            pos_noise = np.zeros((n_reset, 3), dtype=np.float32)
            yaw_offsets = np.zeros(n_reset, dtype=np.float32)

        joint_q = self.model.joint_q.numpy()
        for k, idx in enumerate(env_ids):
            base = idx * 7
            px = target[0] + pos_noise[k, 0]
            py = target[1] + pos_noise[k, 1]
            pz = target[2] + pos_noise[k, 2]
            half_yaw = yaw_offsets[k] * 0.5
            joint_q[base:base + 7] = [px, py, pz, 0.0, 0.0, float(np.sin(half_yaw)), float(np.cos(half_yaw))]
        self.model.joint_q = wp.array(joint_q, dtype=wp.float32, device=self.device)

        joint_qd = self.model.joint_qd.numpy()
        for idx in env_ids:
            base = idx * 6
            joint_qd[base:base + 6] = 0.0
        self.model.joint_qd = wp.array(joint_qd, dtype=wp.float32, device=self.device)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_curr)

        # Zero body forces to prevent accumulation from previous step
        wp.launch(
            tier1_zero_wrench, dim=self.n_envs,
            inputs=[self.state_curr.body_f], device=self.device,
        )

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
            # Per-env thruster gain: scale max thrust by ±thruster_gain range
            gain_range = self.randomization_ranges.thruster_gain
            self._thruster_gain[env_ids] = 1.0 + self.np_random.uniform(
                -gain_range, gain_range, n_reset
            ).astype(np.float32)
            self._thruster_gain_gpu = torch.tensor(
                self._thruster_gain, device=self.device, dtype=torch.float32
            )

        # Zero GPU-side prev tensors for reset envs
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        self._prev_action_gpu[env_ids_t] = 0.0
        self._prev_wrench_gpu[env_ids_t] = 0.0

        self._prev_action[env_ids] = 0.0
        self._prev_wrench[env_ids] = 0.0
        self._step_count[env_ids] = 0
        self._done[env_ids] = False
