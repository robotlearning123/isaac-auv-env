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

import warp as wp

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
        self.M_A_lin: wp.array | None = None
        self.M_A_ang: wp.array | None = None
        self.d_lin_lin: wp.array | None = None
        self.d_lin_ang: wp.array | None = None
        self.d_quad_lin: wp.array | None = None
        self.d_quad_ang: wp.array | None = None
        self.mass_arr: wp.array | None = None
        self.volume_arr: wp.array | None = None
        self.coBM_arr: wp.array | None = None
        self.T_matrix: wp.array | None = None

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

    def zero_wrench(self) -> None:
        wp.launch(tier1_zero_wrench, dim=self.n_envs, inputs=[self.wrench_buf], device=self.device)

    def compute_wrench(
        self,
        nu: wp.array,  # current body velocity (n_envs,) spatial_vectorf
        quat: wp.array,  # body orientation (n_envs,) quatf
        u_cmd: wp.array,  # (n_envs, n_thrusters)
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

    def write_to_body_f(self, body_f: wp.array) -> None:
        """Add accumulated hydro wrench into Newton's state.body_f."""
        wp.launch(
            tier1_accumulate_to_body_f,
            dim=self.n_envs,
            inputs=[self.wrench_buf, body_f],
            device=self.device,
        )
