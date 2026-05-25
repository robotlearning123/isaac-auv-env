"""MPM two-way coupling between fluid particles and Newton rigid bodies."""

from __future__ import annotations

from typing import Any, cast

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverImplicitMPM


@wp.kernel
def _apply_collider_forces(
    dt: float,
    collider_ids: wp.array(dtype=int),
    collider_impulses: wp.array(dtype=wp.vec3),
    collider_impulse_pos: wp.array(dtype=wp.vec3),
    body_ids: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    i = wp.tid()
    cid = collider_ids[i]
    if cid < 0 or cid >= body_ids.shape[0]:
        return
    body_index = body_ids[cid]
    if body_index == -1:
        return
    f_world = collider_impulses[i] / dt
    r = collider_impulse_pos[i] - wp.transform_point(body_q[body_index], body_com[body_index])
    wp.atomic_add(body_f, body_index, wp.spatial_vector(f_world, wp.cross(r, f_world)))


class MPMRigidCoupling:
    """Two-way force coupling between MPM fluid and Newton rigid bodies.

    Uses the pattern from Newton's example_mpm_twoway_coupling:
    separate rigid and fluid models, collider impulses transferred via kernel.
    """

    def __init__(
        self,
        rigid_model: newton.Model,
        fluid_model: newton.Model,
        mpm_solver: SolverImplicitMPM,
        rigid_solver: Any,
        device: str = "cuda:0",
    ) -> None:
        self.rigid_model = rigid_model
        self.fluid_model = fluid_model
        self.mpm_solver = mpm_solver
        self.rigid_solver = rigid_solver
        self.device = device

        self.collider_body_id = mpm_solver.collider_body_index

        max_nodes = 1 << 18
        self._impulses = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._impulse_pos = wp.zeros(max_nodes, dtype=wp.vec3, device=device)
        self._impulse_ids = wp.full(max_nodes, value=-1, dtype=int, device=device)

        self._body_forces = wp.zeros_like(
            rigid_model.state().body_f, device=device
        )

    def step(
        self,
        rigid_state_in: Any,
        rigid_state_out: Any,
        fluid_state: Any,
        dt: float,
        contacts: Any = None,
        control: Any = None,
    ) -> None:
        rigid_state_in.clear_forces()

        wp.launch(
            _apply_collider_forces,
            dim=self._impulse_ids.shape[0],
            inputs=[
                dt,
                self._impulse_ids,
                self._impulses,
                self._impulse_pos,
                self.collider_body_id,
                rigid_state_in.body_q,
                self.rigid_model.body_com,
                rigid_state_in.body_f,
            ],
            device=self.device,
        )

        self._body_forces.assign(rigid_state_in.body_f)

        if contacts is not None:
            self.rigid_model.collide(rigid_state_in, contacts)
        cast(Any, self.rigid_solver).step(
            rigid_state_in, rigid_state_out, control, contacts, dt
        )

        if fluid_state.body_q is not None:
            wp.copy(fluid_state.body_q, rigid_state_out.body_q)
            wp.copy(fluid_state.body_qd, rigid_state_out.body_qd)

        cast(Any, self.mpm_solver).step(fluid_state, fluid_state, None, None, dt)

        self._collect_impulses(fluid_state)

    def _collect_impulses(self, fluid_state: Any) -> None:
        impulses, pos, ids = self.mpm_solver.collect_collider_impulses(fluid_state)
        self._impulse_ids.fill_(-1)
        n = min(impulses.shape[0], self._impulses.shape[0])
        if n > 0:
            self._impulses[:n].assign(impulses[:n])
            self._impulse_pos[:n].assign(pos[:n])
            self._impulse_ids[:n].assign(ids[:n])

    def get_fluid_forces_on_body(self, body_idx: int) -> np.ndarray:
        f = self._body_forces.numpy()
        if body_idx < f.shape[0]:
            return f[body_idx]
        return np.zeros(6, dtype=np.float32)
# mypy: ignore-errors
