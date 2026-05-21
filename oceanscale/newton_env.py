"""Newton-based underwater robotics environment.

Bridges Newton physics with OceanScale Tier-1 hydrodynamics.
Supports multiple solver backends and batched simulation via Newton's
begin_world / end_world multi-world API.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.solvers

from oceanscale.hydro.tier1 import DEFAULT_DT, Tier1


# Default underwater vehicle: box 0.5 x 0.3 x 0.3 m, mass 10 kg
_VEHICLE_HX = 0.25
_VEHICLE_HY = 0.15
_VEHICLE_HZ = 0.15
_VEHICLE_MASS = 10.0
_VEHICLE_VOLUME = 2 * _VEHICLE_HX * 2 * _VEHICLE_HY * 2 * _VEHICLE_HZ  # 0.045 m^3

# Approximate added-mass coefficients for a box-shaped AUV (Fossen Table 2.2)
_DEFAULT_ADDED_MASS = (5.0, 12.0, 12.0, 0.01, 0.01, 0.01)
_DEFAULT_D_LIN = (10.0, 20.0, 20.0, 0.5, 0.5, 0.5)
_DEFAULT_D_QUAD = (5.0, 10.0, 10.0, 0.2, 0.2, 0.2)
# COB above COG: slightly positive for self-righting
_DEFAULT_COBM = 0.01


def _build_model(n_envs: int, device: str) -> newton.Model:
    # Gravity = 0 because Tier1 restoring forces handle weight + buoyancy.
    # Newton's solver would otherwise double-count gravity.
    builder = newton.ModelBuilder(gravity=0.0)
    for i in range(n_envs):
        builder.begin_world(label=f"env_{i}")
        body_idx = builder.add_body(
            mass=_VEHICLE_MASS,
            com=(0.0, 0.0, 0.0),
        )
        builder.add_shape_box(
            body=body_idx,
            hx=_VEHICLE_HX,
            hy=_VEHICLE_HY,
            hz=_VEHICLE_HZ,
        )
        builder.end_world()
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _extract_quat(body_q: wp.array) -> wp.array:
    q_np = body_q.numpy()[:, 3:7].copy()
    return wp.array(q_np, dtype=wp.quatf, device=body_q.device)


class NewtonEnv:
    """Newton-based underwater robotics environment.

    Bridges Newton physics with OceanScale hydrodynamics (Tier-1 Fossen model).
    Supports multiple solver backends and batched multi-world simulation.
    """

    def __init__(
        self,
        n_envs: int = 1,
        solver_type: str = "semi_implicit",
        dt: float = DEFAULT_DT,
        device: str = "cuda",
        domain_rand: bool = False,
    ) -> None:
        self._n_envs = n_envs
        self._solver_type = solver_type
        self._dt = dt
        self._device = device
        self._domain_rand = domain_rand

        self.model = _build_model(n_envs, device)
        self.state_in = self.model.state()
        self.state_out = self.model.state()
        self.control = self.model.control()

        self.solver = self._create_solver(solver_type)

        self.tier1 = Tier1(n_envs=n_envs, n_thrusters=6, device=device)
        self.tier1.set_coeffs(
            added_mass=_DEFAULT_ADDED_MASS,
            d_lin=_DEFAULT_D_LIN,
            d_quad=_DEFAULT_D_QUAD,
            mass=_VEHICLE_MASS,
            volume=_VEHICLE_VOLUME,
            coBM=_DEFAULT_COBM,
        )

        self._u_cmd = wp.zeros((n_envs, 6), dtype=wp.float32, device=device)

    def _create_solver(
        self, solver_type: str
    ) -> newton.solvers.SolverSemiImplicit | newton.solvers.SolverMuJoCo:
        if solver_type == "semi_implicit":
            return newton.solvers.SolverSemiImplicit(self.model)
        elif solver_type == "mujoco":
            return newton.solvers.SolverMuJoCo(self.model)
        else:
            raise ValueError(f"Unknown solver_type: {solver_type!r}")

    def step(self, u_cmd: wp.array | None = None, dt: float | None = None) -> None:
        if dt is None:
            dt = self._dt

        if u_cmd is not None:
            self._u_cmd = u_cmd

        nu = self.state_in.body_qd
        quat = _extract_quat(self.state_in.body_q)

        self.state_in.clear_forces()
        self.tier1.compute_wrench(nu, quat, self._u_cmd, dt)
        self.tier1.write_to_body_f(self.state_in.body_f)

        self.solver.step(self.state_in, self.state_out, self.control, None, dt)
        self.state_in, self.state_out = self.state_out, self.state_in

    def _reset_tier1_buffers(self, env_ids: list[int] | np.ndarray | None = None) -> None:
        if env_ids is None:
            self.tier1.nu_prev.zero_()
            self.tier1.nu_dot_prev.zero_()
            self.tier1.u_eff_prev.zero_()
            return

        ids = np.asarray(env_ids)
        nu_np = self.tier1.nu_prev.numpy()
        nu_np[ids] = 0.0
        wp.copy(self.tier1.nu_prev, wp.array(nu_np, dtype=wp.spatial_vectorf, device=self._device))
        nud_np = self.tier1.nu_dot_prev.numpy()
        nud_np[ids] = 0.0
        wp.copy(self.tier1.nu_dot_prev, wp.array(nud_np, dtype=wp.spatial_vectorf, device=self._device))
        u_prev_np = self.tier1.u_eff_prev.numpy()
        u_prev_np[ids] = 0.0
        wp.copy(self.tier1.u_eff_prev, wp.array(u_prev_np, dtype=wp.float32, device=self._device))

    def reset(self, env_ids: list[int] | np.ndarray | None = None) -> None:
        import numpy as np

        q_np = self.state_in.body_q.numpy()
        qd_np = self.state_in.body_qd.numpy()

        if env_ids is None:
            env_ids = np.arange(self._n_envs)

        for idx in env_ids:
            q_np[idx, :3] = [0.0, 0.0, 0.0]
            q_np[idx, 3:] = [0.0, 0.0, 0.0, 1.0]
            qd_np[idx, :] = 0.0

        wp.copy(self.state_in.body_q, wp.array(q_np, dtype=wp.transformf, device=self._device))
        wp.copy(self.state_in.body_qd, wp.array(qd_np, dtype=wp.spatial_vectorf, device=self._device))
        self._u_cmd = wp.zeros((self._n_envs, 6), dtype=wp.float32, device=self._device)
        self._reset_tier1_buffers(env_ids)

        if self._domain_rand:
            self.tier1.randomize_coeffs(env_ids=env_ids)

    def get_state(self) -> dict[str, np.ndarray]:
        q_np = self.state_in.body_q.numpy()
        qd_np = self.state_in.body_qd.numpy()
        return {
            "position": q_np[:, :3].copy(),
            "orientation": q_np[:, 3:7].copy(),
            "velocity": qd_np[:, :3].copy(),
            "angular_velocity": qd_np[:, 3:6].copy(),
        }

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def device(self) -> str:
        return self._device
