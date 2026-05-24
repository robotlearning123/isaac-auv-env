"""Mesh-based fluid boundary using wp.Mesh queries.

Replaces simplified sphere boundaries with true geometry for FSI coupling.
Uses mesh_query_point_sign_normal for inside/outside detection and
mesh_eval_velocity for no-slip moving boundary conditions.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel
def _mesh_boundary_sph_kernel(
    mesh_id: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    damping: wp.float32,
):
    i = wp.tid()
    p = pos[i]
    query = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist, 1.0e-5)
    if not query.result:
        return
    if query.sign >= 0.0:
        return
    cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    n = wp.mesh_eval_face_normal(mesh_id, query.face)
    nl = wp.length(n)
    if nl < 1.0e-8:
        return
    n = n / nl
    # Ensure outward normal (away from interior) using sign
    n = n * (-query.sign)
    wall_vel = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)
    pos[i] = cp + n * 0.001
    v = vel[i]
    vn = wp.dot(v - wall_vel, n)
    if vn < 0.0:
        vel[i] = v - n * vn * (1.0 + damping) + wall_vel * (1.0 - damping)


@wp.kernel
def _mesh_boundary_grid_kernel(
    mesh_id: wp.uint64,
    u_vel: wp.array(dtype=wp.float32),
    v_vel: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx: wp.float32,
    max_dist: wp.float32,
):
    idx = wp.tid()
    nxy = nx * ny
    iz = idx / nxy
    rem = idx - iz * nxy
    iy = rem / nx
    ix = rem - iy * nx
    p = wp.vec3(float(ix) * dx, float(iy) * dx, float(iz) * dx)
    query = wp.mesh_query_point_sign_normal(mesh_id, p, max_dist, 1.0e-5)
    if not query.result:
        return
    if query.sign >= 0.0:
        return
    wv = wp.mesh_eval_velocity(mesh_id, query.face, query.u, query.v)
    u_vel[idx] = wv[0]
    v_vel[idx] = wv[1]
    w_vel[idx] = wv[2]


class MeshBoundary:
    """Mesh-based fluid boundary using wp.Mesh queries."""

    def __init__(
        self,
        vertices: np.ndarray,
        indices: np.ndarray,
        velocities: np.ndarray | None = None,
        device: str = "cuda:0",
    ):
        self.device = device
        verts = np.asarray(vertices, dtype=np.float32)
        inds = np.asarray(indices, dtype=np.int32).ravel()
        self._points = wp.array(verts, dtype=wp.vec3, device=device)
        self._indices = wp.array(inds, dtype=wp.int32, device=device)
        if velocities is not None:
            v = np.asarray(velocities, dtype=np.float32)
            self._velocities = wp.array(v, dtype=wp.vec3, device=device)
        else:
            self._velocities = wp.zeros(len(verts), dtype=wp.vec3, device=device)
        self.mesh = wp.Mesh(
            points=self._points,
            indices=self._indices,
            velocities=self._velocities,
        )

    def update_positions(self, vertices: np.ndarray):
        verts = np.asarray(vertices, dtype=np.float32)
        new_pts = wp.array(verts, dtype=wp.vec3, device=self.device)
        self.mesh.points = new_pts
        self._points = new_pts
        self.mesh.refit()

    def update_velocities(self, velocities: np.ndarray):
        v = np.asarray(velocities, dtype=np.float32)
        new_v = wp.array(v, dtype=wp.vec3, device=self.device)
        self.mesh.velocities = new_v
        self._velocities = new_v

    def apply_boundary_sph(
        self,
        positions: wp.array,
        velocities: wp.array,
        max_dist: float = 5.0,
        damping: float = 0.5,
    ):
        wp.launch(
            _mesh_boundary_sph_kernel,
            dim=positions.shape[0],
            inputs=[self.mesh.id, positions, velocities, float(max_dist), float(damping)],
            device=self.device,
        )

    def apply_boundary_grid(
        self,
        u: wp.array,
        v: wp.array,
        w: wp.array,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        max_dist: float = 5.0,
    ):
        wp.launch(
            _mesh_boundary_grid_kernel,
            dim=nx * ny * nz,
            inputs=[self.mesh.id, u, v, w, nx, ny, nz, float(dx), float(max_dist)],
            device=self.device,
        )


def make_box_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    half_extents: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> tuple[np.ndarray, np.ndarray]:
    cx, cy, cz = center
    hx, hy, hz = half_extents
    verts = np.array([
        [cx - hx, cy - hy, cz - hz],
        [cx + hx, cy - hy, cz - hz],
        [cx + hx, cy + hy, cz - hz],
        [cx - hx, cy + hy, cz - hz],
        [cx - hx, cy - hy, cz + hz],
        [cx + hx, cy - hy, cz + hz],
        [cx + hx, cy + hy, cz + hz],
        [cx - hx, cy + hy, cz + hz],
    ], dtype=np.float32)
    indices = np.array([
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        2, 3, 7, 2, 7, 6,
        0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ], dtype=np.int32)
    return verts, indices
