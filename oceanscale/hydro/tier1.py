"""Tier-1 Fossen orchestrator — owns per-env buffers, calls kernels in order,
writes wrench to Newton's state.body_f.

Usage pattern (env loop):
    tier1 = Tier1(n_envs, n_thrusters=8, device="cuda")
    tier1.set_coeffs(added_mass=..., d_lin=..., d_quad=..., volume=..., coBM=...)
    # ... per step:
    tier1.compute_wrench(state_curr, u_thruster_cmd, dt=1/240)
    tier1.write_to_body_f(state_curr)   # adds wrench to state_curr.body_f
    solver.step(state_curr, state_next, control, None, dt)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import warp as wp
from numpy.typing import NDArray

from oceanscale.hydro.tier1_kernels import (
    tier1_accumulate_to_body_f,
    tier1_added_mass,
    tier1_coriolis_a,
    tier1_damping,
    tier1_restoring,
    tier1_thruster_alloc,
    tier1_update_nu_dot_ema,
    tier1_zero_wrench,
)

DEFAULT_RHO = 1025.0  # salt water (kg/m³)
DEFAULT_G = 9.81
DEFAULT_DT = 1.0 / 240.0
DEFAULT_EMA_ALPHA = 0.3  # per MarineGym L230

FloatArray = NDArray[np.float32]


def _wp_numpy(array: Any) -> NDArray[Any]:
    return cast(NDArray[Any], array.numpy())


def _require_value(value: Any, name: str) -> Any:
    if value is None:
        raise RuntimeError(f"Tier1 missing {name}; call set_coeffs() first")
    return value


@dataclass
class RandomizationRanges:
    """Per-coefficient multiplicative variation range (fraction of base value).

    Each field is a float in [0, 1]. The actual per-env coefficient is sampled as:
        coeff_i = base * (1 + uniform(-range, +range))
    """

    mass: float = 0.20
    added_mass: float = 0.25
    d_lin: float = 0.30
    d_quad: float = 0.30
    volume: float = 0.15
    coBM: float = 0.50  # noqa: N815
    thruster_gain: float = 0.20
    current_speed_max: float = 0.5


class Tier1:
    """Tier-1 Fossen hydro for n_envs worlds with shared kernel launches.

    Per-env state buffers (all on GPU):
      - wrench_buf: spatial_vectorf, accumulated each step
      - nu_prev, nu_dot_prev: for added-mass ν̇ estimation
      - u_eff_prev: thruster low-pass state

    Per-env coefficient arrays (set via .set_coeffs):
      - M_A_lin (vec3f), M_A_ang (vec3f) per env
      - d_lin_lin, d_lin_ang, d_quad_lin, d_quad_ang (vec3f each)
      - mass (f32), volume (f32), coBM (f32) per env
      - T_matrix (n_envs × 6 × n_thrusters)
    """

    def __init__(
        self,
        n_envs: int,
        n_thrusters: int = 8,
        device: str = "cuda",
        rho_water: float = DEFAULT_RHO,
        g_accel: float = DEFAULT_G,
        max_thrust: float = 51.5,  # BlueROV2 T200 max ~5.25 kgf ≈ 51.5 N at full throttle
        deadband: float = 0.05,
        tau_lag: float = 0.1,  # MarineGym uses 0.01; we plan 0.1 conservative
        ema_alpha: float = DEFAULT_EMA_ALPHA,
    ) -> None:
        self.n_envs = n_envs
        self.n_thrusters = n_thrusters
        self.device = device
        self.rho_water = float(rho_water)
        self.g_accel = float(g_accel)
        self.max_thrust = float(max_thrust)
        self.deadband = float(deadband)
        self.tau_lag = float(tau_lag)
        self.ema_alpha = float(ema_alpha)

        # Output wrench buffer
        self.wrench_buf = wp.zeros(n_envs, dtype=wp.spatial_vectorf, device=device)
        # Previous state for ν̇ EMA
        self.nu_prev = wp.zeros(n_envs, dtype=wp.spatial_vectorf, device=device)
        self.nu_dot_prev = wp.zeros(n_envs, dtype=wp.spatial_vectorf, device=device)
        # Thruster low-pass state
        self.u_eff_prev = wp.zeros((n_envs, n_thrusters), dtype=wp.float32, device=device)
        self.u_eff_out = wp.zeros((n_envs, n_thrusters), dtype=wp.float32, device=device)

        # Coefficient slots (filled by set_coeffs)
        self.M_A_lin: Any | None = None
        self.M_A_ang: Any | None = None
        self.d_lin_lin: Any | None = None
        self.d_lin_ang: Any | None = None
        self.d_quad_lin: Any | None = None
        self.d_quad_ang: Any | None = None
        self.mass_arr: Any | None = None
        self.volume_arr: Any | None = None
        self.coBM_arr: Any | None = None
        self.current_vec_arr: Any | None = None
        self.T_matrix: Any | None = None

    def set_coeffs(
        self,
        added_mass: tuple[float, float, float, float, float, float],
        d_lin: tuple[float, float, float, float, float, float],
        d_quad: tuple[float, float, float, float, float, float],
        mass: float,
        volume: float,
        coBM: float,
        T_matrix: list[list[float]] | None = None,
    ) -> None:
        """Broadcast scalar/6-tuple coefs to all envs.

        Tuples follow Fossen ordering: (surge, sway, heave, roll, pitch, yaw).
        """
        import numpy as np

        n = self.n_envs
        ma_lin_np = np.tile(np.array(added_mass[:3], dtype=np.float32), (n, 1))
        ma_ang_np = np.tile(np.array(added_mass[3:], dtype=np.float32), (n, 1))
        self.M_A_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device=self.device)
        self.M_A_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device=self.device)

        dll_np = np.tile(np.array(d_lin[:3], dtype=np.float32), (n, 1))
        dla_np = np.tile(np.array(d_lin[3:], dtype=np.float32), (n, 1))
        dql_np = np.tile(np.array(d_quad[:3], dtype=np.float32), (n, 1))
        dqa_np = np.tile(np.array(d_quad[3:], dtype=np.float32), (n, 1))
        self.d_lin_lin = wp.array(dll_np, dtype=wp.vec3f, device=self.device)
        self.d_lin_ang = wp.array(dla_np, dtype=wp.vec3f, device=self.device)
        self.d_quad_lin = wp.array(dql_np, dtype=wp.vec3f, device=self.device)
        self.d_quad_ang = wp.array(dqa_np, dtype=wp.vec3f, device=self.device)

        self.mass_arr = wp.array(
            np.full(n, mass, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.volume_arr = wp.array(
            np.full(n, volume, dtype=np.float32), dtype=wp.float32, device=self.device
        )
        self.coBM_arr = wp.array(
            np.full(n, coBM, dtype=np.float32), dtype=wp.float32, device=self.device
        )

        if T_matrix is None:
            # Default identity-like allocation: first 6 thrusters = unit axis;
            # remaining thrusters zero. For unit tests only.
            T = np.zeros((n, 6, self.n_thrusters), dtype=np.float32)
            for k in range(min(6, self.n_thrusters)):
                T[:, k, k] = 1.0
        else:
            T_arr = np.array(T_matrix, dtype=np.float32)
            assert T_arr.shape == (6, self.n_thrusters), (
                f"T_matrix shape {T_arr.shape} != (6, {self.n_thrusters})"
            )
            T = np.tile(T_arr, (n, 1, 1))
        self.T_matrix = wp.array(T, dtype=wp.float32, device=self.device)

        # Store base values for domain randomization
        self._base_coeffs: dict[str, Any] = {
            "added_mass": np.array(added_mass, dtype=np.float32),
            "d_lin": np.array(d_lin, dtype=np.float32),
            "d_quad": np.array(d_quad, dtype=np.float32),
            "mass": float(mass),
            "volume": float(volume),
            "coBM": float(coBM),
        }

    def randomize_coeffs(
        self,
        env_ids: np.ndarray | list[int] | None = None,
        ranges: RandomizationRanges | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Sample per-env coefficient variation around base values.

        Modifies GPU coefficient arrays in-place for the specified envs.
        Must be called after set_coeffs().

        Args:
            env_ids: Env indices to randomize. None = all envs.
            ranges: Variation fractions. Uses defaults if None.
            rng: NumPy RNG for reproducibility. Fresh default if None.
        """
        assert hasattr(self, "_base_coeffs"), "Call set_coeffs() before randomize_coeffs()"
        if ranges is None:
            ranges = RandomizationRanges()
        if rng is None:
            rng = np.random.default_rng()
        if env_ids is None:
            env_ids = np.arange(self.n_envs)
        ids = np.asarray(env_ids)
        n = len(ids)
        base = self._base_coeffs

        # Added mass (6-DOF): per-env, per-axis
        ma_scale = 1.0 + rng.uniform(-ranges.added_mass, ranges.added_mass, (n, 6)).astype(
            np.float32
        )
        ma_rand = cast(FloatArray, base["added_mass"])[np.newaxis, :] * ma_scale
        ma_lin_np = _wp_numpy(_require_value(self.M_A_lin, "M_A_lin"))
        ma_ang_np = _wp_numpy(_require_value(self.M_A_ang, "M_A_ang"))
        ma_lin_np[ids] = ma_rand[:, :3]
        ma_ang_np[ids] = ma_rand[:, 3:]
        self.M_A_lin = wp.array(ma_lin_np, dtype=wp.vec3f, device=self.device)
        self.M_A_ang = wp.array(ma_ang_np, dtype=wp.vec3f, device=self.device)

        # Damping linear (6-DOF)
        dl_scale = 1.0 + rng.uniform(-ranges.d_lin, ranges.d_lin, (n, 6)).astype(np.float32)
        dl_rand = cast(FloatArray, base["d_lin"])[np.newaxis, :] * dl_scale
        dll_np = _wp_numpy(_require_value(self.d_lin_lin, "d_lin_lin"))
        dla_np = _wp_numpy(_require_value(self.d_lin_ang, "d_lin_ang"))
        dll_np[ids] = dl_rand[:, :3]
        dla_np[ids] = dl_rand[:, 3:]
        self.d_lin_lin = wp.array(dll_np, dtype=wp.vec3f, device=self.device)
        self.d_lin_ang = wp.array(dla_np, dtype=wp.vec3f, device=self.device)

        # Damping quadratic (6-DOF)
        dq_scale = 1.0 + rng.uniform(-ranges.d_quad, ranges.d_quad, (n, 6)).astype(np.float32)
        dq_rand = cast(FloatArray, base["d_quad"])[np.newaxis, :] * dq_scale
        dql_np = _wp_numpy(_require_value(self.d_quad_lin, "d_quad_lin"))
        dqa_np = _wp_numpy(_require_value(self.d_quad_ang, "d_quad_ang"))
        dql_np[ids] = dq_rand[:, :3]
        dqa_np[ids] = dq_rand[:, 3:]
        self.d_quad_lin = wp.array(dql_np, dtype=wp.vec3f, device=self.device)
        self.d_quad_ang = wp.array(dqa_np, dtype=wp.vec3f, device=self.device)

        # Mass, volume, coBM (scalar per env)
        m_np = _wp_numpy(_require_value(self.mass_arr, "mass_arr"))
        m_np[ids] = float(base["mass"]) * (
            1.0 + rng.uniform(-ranges.mass, ranges.mass, n).astype(np.float32)
        )
        self.mass_arr = wp.array(m_np, dtype=wp.float32, device=self.device)

        v_np = _wp_numpy(_require_value(self.volume_arr, "volume_arr"))
        v_np[ids] = float(base["volume"]) * (
            1.0 + rng.uniform(-ranges.volume, ranges.volume, n).astype(np.float32)
        )
        self.volume_arr = wp.array(v_np, dtype=wp.float32, device=self.device)

        c_np = _wp_numpy(_require_value(self.coBM_arr, "coBM_arr"))
        c_np[ids] = float(base["coBM"]) * (
            1.0 + rng.uniform(-ranges.coBM, ranges.coBM, n).astype(np.float32)
        )
        self.coBM_arr = wp.array(c_np, dtype=wp.float32, device=self.device)

        # Ocean current: speed in [0, max] × random unit direction
        if ranges.current_speed_max > 0:
            speeds = rng.uniform(0, ranges.current_speed_max, n).astype(np.float32)
            theta = rng.uniform(0, 2 * np.pi, n).astype(np.float32)
            cos_phi = rng.uniform(-1, 1, n).astype(np.float32)
            sin_phi = np.sqrt(np.clip(1.0 - cos_phi**2, 0, None)).astype(np.float32)
            current_vec = np.stack(
                [
                    speeds * sin_phi * np.cos(theta),
                    speeds * sin_phi * np.sin(theta),
                    speeds * cos_phi,
                ],
                axis=-1,
            ).astype(np.float32)
            cv_np = (
                _wp_numpy(self.current_vec_arr).copy()
                if self.current_vec_arr is not None
                else np.zeros((self.n_envs, 3), dtype=np.float32)
            )
            cv_np[ids] = current_vec
            self.current_vec_arr = wp.array(cv_np, dtype=wp.vec3f, device=self.device)

    def zero_wrench(self) -> None:
        wp.launch(tier1_zero_wrench, dim=self.n_envs, inputs=[self.wrench_buf], device=self.device)

    def compute_wrench(
        self,
        nu: Any,  # current body velocity (n_envs,) spatial_vectorf
        quat: Any,  # body orientation (n_envs,) quatf
        u_cmd: Any,  # (n_envs, n_thrusters)
        dt: float = DEFAULT_DT,
    ) -> None:
        """Launch all 5 hydro kernels + thruster, accumulating into wrench_buf.

        Caller is responsible for ensuring set_coeffs() was called first.
        """
        assert self.M_A_lin is not None, "Call set_coeffs() before compute_wrench()"
        self.zero_wrench()

        # Update nu_dot via EMA (needed for added-mass)
        wp.launch(
            tier1_update_nu_dot_ema,
            dim=self.n_envs,
            inputs=[nu, self.nu_prev, self.nu_dot_prev, self.ema_alpha, dt],
            device=self.device,
        )

        # Added mass: -M_A · ν̇
        wp.launch(
            tier1_added_mass,
            dim=self.n_envs,
            inputs=[self.nu_dot_prev, self.M_A_lin, self.M_A_ang, self.wrench_buf],
            device=self.device,
        )

        # Damping
        wp.launch(
            tier1_damping,
            dim=self.n_envs,
            inputs=[
                nu,
                self.d_lin_lin,
                self.d_lin_ang,
                self.d_quad_lin,
                self.d_quad_ang,
                self.wrench_buf,
            ],
            device=self.device,
        )

        # Coriolis C_A
        wp.launch(
            tier1_coriolis_a,
            dim=self.n_envs,
            inputs=[nu, self.M_A_lin, self.M_A_ang, self.wrench_buf],
            device=self.device,
        )

        # Restoring
        wp.launch(
            tier1_restoring,
            dim=self.n_envs,
            inputs=[
                quat,
                self.mass_arr,
                self.volume_arr,
                self.coBM_arr,
                self.rho_water,
                self.g_accel,
                self.wrench_buf,
            ],
            device=self.device,
        )

        # Thruster allocation
        wp.launch(
            tier1_thruster_alloc,
            dim=self.n_envs,
            inputs=[
                u_cmd,
                self.u_eff_prev,
                self.u_eff_out,
                self.T_matrix,
                self.max_thrust,
                self.deadband,
                self.tau_lag,
                dt,
                self.wrench_buf,
            ],
            device=self.device,
        )
        # swap thruster low-pass state
        self.u_eff_prev, self.u_eff_out = self.u_eff_out, self.u_eff_prev

    def write_to_body_f(self, body_f: Any) -> None:
        """Add accumulated hydro wrench into Newton's state.body_f."""
        wp.launch(
            tier1_accumulate_to_body_f,
            dim=self.n_envs,
            inputs=[self.wrench_buf, body_f],
            device=self.device,
        )
