# mypy: ignore-errors
"""Eulerian grid-based and SPH Lagrangian fluid solvers using Warp kernels.

GridFluidSolver: Chorin projection method on a uniform 3D grid.
SPHSolver: Smoothed Particle Hydrodynamics with spatial hash grid.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# Grid kernels — flat index convention: idx = ix + iy*nx + iz*nx*ny
# ---------------------------------------------------------------------------


@wp.kernel
def _advect_field(
    field: wp.array(dtype=wp.float32),
    field_new: wp.array(dtype=wp.float32),
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx: wp.float32,
    dt: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        field_new[idx] = field[idx]
        return

    # backtrace
    fx = float(ix)
    fy = float(iy)
    fz = float(iz)

    xb = fx - u[idx] * dt / dx
    yb = fy - v[idx] * dt / dx
    zb = fz - w_vel[idx] * dt / dx

    fnx = float(nx)
    fny = float(ny)
    fnz = float(nz)

    xb = wp.max(0.5, wp.min(fnx - 1.5, xb))
    yb = wp.max(0.5, wp.min(fny - 1.5, yb))
    zb = wp.max(0.5, wp.min(fnz - 1.5, zb))

    x0 = wp.int32(wp.floor(xb))
    y0 = wp.int32(wp.floor(yb))
    z0 = wp.int32(wp.floor(zb))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    sx = xb - float(x0)
    sy = yb - float(y0)
    sz = zb - float(z0)
    sx = wp.max(0.0, wp.min(1.0, sx))
    sy = wp.max(0.0, wp.min(1.0, sy))
    sz = wp.max(0.0, wp.min(1.0, sz))

    sx1 = 1.0 - sx
    sy1 = 1.0 - sy
    sz1 = 1.0 - sz

    c000 = field[x0 + y0 * nx + z0 * nx * ny]
    c100 = field[x1 + y0 * nx + z0 * nx * ny]
    c010 = field[x0 + y1 * nx + z0 * nx * ny]
    c110 = field[x1 + y1 * nx + z0 * nx * ny]
    c001 = field[x0 + y0 * nx + z1 * nx * ny]
    c101 = field[x1 + y0 * nx + z1 * nx * ny]
    c011 = field[x0 + y1 * nx + z1 * nx * ny]
    c111 = field[x1 + y1 * nx + z1 * nx * ny]

    field_new[idx] = (
        (c000 * sx1 + c100 * sx) * (sy1 * sz1)
        + (c010 * sx1 + c110 * sx) * (sy * sz1)
        + (c001 * sx1 + c101 * sx) * (sy1 * sz)
        + (c011 * sx1 + c111 * sx) * (sy * sz)
    )


@wp.kernel
def _diffuse_jacobi(
    phi: wp.array(dtype=wp.float32),
    phi_new: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    alpha: wp.float32,
    beta: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        phi_new[idx] = phi[idx]
        return

    c = phi[idx]
    l = phi[idx - 1]
    r = phi[idx + 1]
    d = phi[idx - nx]
    u_val = phi[idx + nx]
    b = phi[idx - nx * ny]
    t = phi[idx + nx * ny]

    phi_new[idx] = (c + alpha * (l + r + d + u_val + b + t)) * beta


@wp.kernel
def _compute_divergence(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    div: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        div[idx] = 0.0
        return

    dudx = (u[idx + 1] - u[idx - 1]) / (2.0 * dx)
    dvdy = (v[idx + nx] - v[idx - nx]) / (2.0 * dx)
    dwdz = (w_vel[idx + nx * ny] - w_vel[idx - nx * ny]) / (2.0 * dx)
    div[idx] = dudx + dvdy + dwdz


@wp.kernel
def _pressure_jacobi(
    p: wp.array(dtype=wp.float32),
    p_new: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx2: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        p_new[idx] = 0.0
        return

    l = p[idx - 1]
    r = p[idx + 1]
    d = p[idx - nx]
    u_val = p[idx + nx]
    b = p[idx - nx * ny]
    t = p[idx + nx * ny]

    p_new[idx] = (l + r + d + u_val + b + t - dx2 * rhs[idx]) / 6.0


@wp.kernel
def _correct_velocity(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    p: wp.array(dtype=wp.float32),
    u_new: wp.array(dtype=wp.float32),
    v_new: wp.array(dtype=wp.float32),
    w_new: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx: wp.float32,
    dt: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        u_new[idx] = u[idx]
        v_new[idx] = v[idx]
        w_new[idx] = w_vel[idx]
        return

    dpdx = (p[idx + 1] - p[idx - 1]) / (2.0 * dx)
    dpdy = (p[idx + nx] - p[idx - nx]) / (2.0 * dx)
    dpdz = (p[idx + nx * ny] - p[idx - nx * ny]) / (2.0 * dx)

    u_new[idx] = u[idx] - dt * dpdx
    v_new[idx] = v[idx] - dt * dpdy
    w_new[idx] = w_vel[idx] - dt * dpdz


@wp.kernel
def _scale_array(
    arr: wp.array(dtype=wp.float32),
    scale: wp.float32,
):
    i = wp.tid()
    arr[i] = arr[i] * scale


@wp.kernel
def _add_velocity_source_kernel(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    cx: wp.float32,
    cy: wp.float32,
    cz: wp.float32,
    vx: wp.float32,
    vy: wp.float32,
    vz: wp.float32,
    radius: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    idx = wp.tid()
    nxy = nx * ny
    iz = idx / nxy
    rem = idx - iz * nxy
    iy = rem / nx
    ix = rem - iy * nx

    fx = float(ix)
    fy = float(iy)
    fz = float(iz)

    dx = cx - fx
    dy = cy - fy
    dz = cz - fz
    dist2 = dx * dx + dy * dy + dz * dz
    r2 = radius * radius
    if dist2 < r2:
        weight = 1.0 - wp.sqrt(dist2) / radius
        wp.atomic_add(u, idx, vx * weight)
        wp.atomic_add(v, idx, vy * weight)
        wp.atomic_add(w_vel, idx, vz * weight)


@wp.kernel
def _apply_boundary_sphere(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    cx: wp.float32,
    cy: wp.float32,
    cz: wp.float32,
    radius: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    idx = wp.tid()
    nxy = nx * ny
    iz = idx / nxy
    rem = idx - iz * nxy
    iy = rem / nx
    ix = rem - iy * nx

    fx = float(ix)
    fy = float(iy)
    fz = float(iz)

    dx = cx - fx
    dy = cy - fy
    dz = cz - fz
    dist2 = dx * dx + dy * dy + dz * dz
    r2 = radius * radius
    if dist2 < r2:
        u[idx] = 0.0
        v[idx] = 0.0
        w_vel[idx] = 0.0


@wp.kernel
def _sample_velocity_kernel(
    positions: wp.array(dtype=wp.vec3),
    out_vel: wp.array(dtype=wp.vec3),
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx: wp.float32,
):
    i = wp.tid()
    pos = positions[i]

    gx = pos[0] / dx
    gy = pos[1] / dx
    gz = pos[2] / dx

    fnx = float(nx)
    fny = float(ny)
    fnz = float(nz)

    gx = wp.max(0.5, wp.min(fnx - 1.5, gx))
    gy = wp.max(0.5, wp.min(fny - 1.5, gy))
    gz = wp.max(0.5, wp.min(fnz - 1.5, gz))

    x0 = wp.int32(wp.floor(gx))
    y0 = wp.int32(wp.floor(gy))
    z0 = wp.int32(wp.floor(gz))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    sx = gx - float(x0)
    sy = gy - float(y0)
    sz = gz - float(z0)
    sx = wp.max(0.0, wp.min(1.0, sx))
    sy = wp.max(0.0, wp.min(1.0, sy))
    sz = wp.max(0.0, wp.min(1.0, sz))

    sx1 = 1.0 - sx
    sy1 = 1.0 - sy
    sz1 = 1.0 - sz

    nxy = nx * ny

    # interpolate u
    c000 = u[x0 + y0 * nx + z0 * nxy]
    c100 = u[x1 + y0 * nx + z0 * nxy]
    c010 = u[x0 + y1 * nx + z0 * nxy]
    c110 = u[x1 + y1 * nx + z0 * nxy]
    c001 = u[x0 + y0 * nx + z1 * nxy]
    c101 = u[x1 + y0 * nx + z1 * nxy]
    c011 = u[x0 + y1 * nx + z1 * nxy]
    c111 = u[x1 + y1 * nx + z1 * nxy]
    u_val = (
        (c000 * sx1 + c100 * sx) * (sy1 * sz1)
        + (c010 * sx1 + c110 * sx) * (sy * sz1)
        + (c001 * sx1 + c101 * sx) * (sy1 * sz)
        + (c011 * sx1 + c111 * sx) * (sy * sz)
    )

    # interpolate v
    c000 = v[x0 + y0 * nx + z0 * nxy]
    c100 = v[x1 + y0 * nx + z0 * nxy]
    c010 = v[x0 + y1 * nx + z0 * nxy]
    c110 = v[x1 + y1 * nx + z0 * nxy]
    c001 = v[x0 + y0 * nx + z1 * nxy]
    c101 = v[x1 + y0 * nx + z1 * nxy]
    c011 = v[x0 + y1 * nx + z1 * nxy]
    c111 = v[x1 + y1 * nx + z1 * nxy]
    v_val = (
        (c000 * sx1 + c100 * sx) * (sy1 * sz1)
        + (c010 * sx1 + c110 * sx) * (sy * sz1)
        + (c001 * sx1 + c101 * sx) * (sy1 * sz)
        + (c011 * sx1 + c111 * sx) * (sy * sz)
    )

    # interpolate w
    c000 = w_vel[x0 + y0 * nx + z0 * nxy]
    c100 = w_vel[x1 + y0 * nx + z0 * nxy]
    c010 = w_vel[x0 + y1 * nx + z0 * nxy]
    c110 = w_vel[x1 + y1 * nx + z0 * nxy]
    c001 = w_vel[x0 + y0 * nx + z1 * nxy]
    c101 = w_vel[x1 + y0 * nx + z1 * nxy]
    c011 = w_vel[x0 + y1 * nx + z1 * nxy]
    c111 = w_vel[x1 + y1 * nx + z1 * nxy]
    w_val = (
        (c000 * sx1 + c100 * sx) * (sy1 * sz1)
        + (c010 * sx1 + c110 * sx) * (sy * sz1)
        + (c001 * sx1 + c101 * sx) * (sy1 * sz)
        + (c011 * sx1 + c111 * sx) * (sy * sz)
    )

    out_vel[i] = wp.vec3(u_val, v_val, w_val)


# ---------------------------------------------------------------------------
# GridFluidSolver
# ---------------------------------------------------------------------------


class GridFluidSolver:
    """Eulerian grid-based fluid solver using Warp kernels.

    Supports: projection method (Chorin), stable fluids (Stam).
    Can output velocity field for coupling with Newton rigid bodies.
    """

    def __init__(
        self,
        grid_res: int = 64,
        domain_size: float = 1.0,
        viscosity: float = 0.001,
        device: str = "cuda",
    ):
        self.nx = grid_res
        self.ny = grid_res
        self.nz = grid_res
        self.n = grid_res * grid_res * grid_res
        self.domain_size = domain_size
        self.dx = domain_size / float(grid_res)
        self.viscosity = viscosity
        self.device = device

        n = self.n
        self.u = wp.zeros(n, dtype=wp.float32, device=device)
        self.v = wp.zeros(n, dtype=wp.float32, device=device)
        self.w = wp.zeros(n, dtype=wp.float32, device=device)
        self.p = wp.zeros(n, dtype=wp.float32, device=device)
        self.density = wp.zeros(n, dtype=wp.float32, device=device)

        self.u_tmp = wp.zeros(n, dtype=wp.float32, device=device)
        self.v_tmp = wp.zeros(n, dtype=wp.float32, device=device)
        self.w_tmp = wp.zeros(n, dtype=wp.float32, device=device)
        self.p_tmp = wp.zeros(n, dtype=wp.float32, device=device)
        self.density_tmp = wp.zeros(n, dtype=wp.float32, device=device)

        self.div = wp.zeros(n, dtype=wp.float32, device=device)

        self._diffusion_iters = 20
        self._pressure_iters = 50

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.n, inputs=inputs, device=self.device)

    def advect(self, dt: float):
        """Semi-Lagrangian advection of velocity and density."""
        self._launch(
            _advect_field,
            [self.u, self.u_tmp, self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, dt],
        )
        self._launch(
            _advect_field,
            [self.v, self.v_tmp, self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, dt],
        )
        self._launch(
            _advect_field,
            [self.w, self.w_tmp, self.u, self.v, self.w, self.nx, self.ny, self.nz, self.dx, dt],
        )
        self._launch(
            _advect_field,
            [
                self.density,
                self.density_tmp,
                self.u,
                self.v,
                self.w,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                dt,
            ],
        )
        self.u, self.u_tmp = self.u_tmp, self.u
        self.v, self.v_tmp = self.v_tmp, self.v
        self.w, self.w_tmp = self.w_tmp, self.w
        self.density, self.density_tmp = self.density_tmp, self.density

    def diffuse(self, dt: float, iterations: int = 20):
        """Implicit diffusion via Jacobi iteration."""
        alpha = self.viscosity * dt / (self.dx * self.dx)
        beta = 1.0 / (1.0 + 6.0 * alpha)

        for field, field_tmp in [
            (self.u, self.u_tmp),
            (self.v, self.v_tmp),
            (self.w, self.w_tmp),
            (self.density, self.density_tmp),
        ]:
            for it in range(iterations):
                if it % 2 == 0:
                    self._launch(
                        _diffuse_jacobi,
                        [field, field_tmp, self.nx, self.ny, self.nz, alpha, beta],
                    )
                else:
                    self._launch(
                        _diffuse_jacobi,
                        [field_tmp, field, self.nx, self.ny, self.nz, alpha, beta],
                    )
            # ensure primary buffer holds latest result
            if iterations % 2 == 1:
                wp.copy(field, field_tmp)

    def pressure_solve(self, iterations: int = 50):
        """Poisson pressure solve via Jacobi iteration."""
        self._launch(
            _compute_divergence,
            [self.u, self.v, self.w, self.div, self.nx, self.ny, self.nz, self.dx],
        )
        dx2 = self.dx * self.dx

        for it in range(iterations):
            if it % 2 == 0:
                self._launch(
                    _pressure_jacobi,
                    [self.p, self.p_tmp, self.div, self.nx, self.ny, self.nz, dx2],
                )
            else:
                self._launch(
                    _pressure_jacobi,
                    [self.p_tmp, self.p, self.div, self.nx, self.ny, self.nz, dx2],
                )

        if iterations % 2 == 1:
            self.p, self.p_tmp = self.p_tmp, self.p

    def project(self):
        """Correct velocity to be divergence-free using pressure gradient."""
        self._launch(
            _correct_velocity,
            [
                self.u,
                self.v,
                self.w,
                self.p,
                self.u_tmp,
                self.v_tmp,
                self.w_tmp,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
                1.0,
            ],
        )
        self.u, self.u_tmp = self.u_tmp, self.u
        self.v, self.v_tmp = self.v_tmp, self.v
        self.w, self.w_tmp = self.w_tmp, self.w

    def step(self, dt: float = 0.01):
        """One timestep: advect -> diffuse -> project -> correct."""
        self.advect(dt)
        self.diffuse(dt, iterations=self._diffusion_iters)
        self.pressure_solve(iterations=self._pressure_iters)
        self.project()

    def add_source(self, position: tuple, velocity: tuple, radius: float = 3.0):
        """Add a velocity source at grid position (gx, gy, gz)."""
        gx, gy, gz = position
        vx, vy, vz = velocity
        self._launch(
            _add_velocity_source_kernel,
            [
                self.u,
                self.v,
                self.w,
                float(gx),
                float(gy),
                float(gz),
                float(vx),
                float(vy),
                float(vz),
                float(radius),
                self.nx,
                self.ny,
                self.nz,
            ],
        )

    def add_boundary_sphere(self, center: tuple, radius: float):
        """Zero velocity inside a sphere for solid boundaries."""
        cx, cy, cz = center
        self._launch(
            _apply_boundary_sphere,
            [
                self.u,
                self.v,
                self.w,
                float(cx),
                float(cy),
                float(cz),
                float(radius),
                self.nx,
                self.ny,
                self.nz,
            ],
        )

    def sample_velocity_at(self, positions: wp.array) -> wp.array:
        """Sample fluid velocity at arbitrary positions (for FSI coupling)."""
        n_pts = positions.shape[0]
        out = wp.zeros(n_pts, dtype=wp.vec3, device=self.device)
        wp.launch(
            _sample_velocity_kernel,
            dim=n_pts,
            inputs=[
                positions,
                out,
                self.u,
                self.v,
                self.w,
                self.nx,
                self.ny,
                self.nz,
                self.dx,
            ],
            device=self.device,
        )
        return out

    def get_velocity_field(self) -> wp.array:
        """Return combined velocity as wp.array of vec3."""
        u_np = self.u.numpy()
        v_np = self.v.numpy()
        w_np = self.w.numpy()
        vel_np = np.stack([u_np, v_np, w_np], axis=-1).astype(np.float32)
        return wp.array(vel_np, dtype=wp.vec3, device=self.device)

    def get_pressure_field(self) -> wp.array:
        return self.p


# ---------------------------------------------------------------------------
# SPH kernels — wp.HashGrid accelerated neighbor search
# ---------------------------------------------------------------------------


@wp.kernel
def _sph_compute_density(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=wp.float32),
    n: wp.int32,
    h: wp.float32,
    mass: wp.float32,
):
    tid = wp.tid()
    if tid >= n:
        return
    i = wp.hash_grid_point_id(grid, tid)
    pi = pos[i]
    h2 = h * h
    rho = float(0.0)  # noqa: UP018
    h9 = h * h * h * h * h * h * h * h * h
    coeff = 315.0 / (64.0 * 3.14159265 * h9)

    query = wp.hash_grid_query(grid, pi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        d = pos[j] - pi
        dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        if dist2 < h2:
            q = h2 - dist2
            rho = rho + mass * coeff * q * q * q

    density[i] = rho


@wp.kernel
def _sph_compute_forces(
    grid: wp.uint64,
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=wp.float32),
    force: wp.array(dtype=wp.vec3),
    n: wp.int32,
    h: wp.float32,
    mass: wp.float32,
    viscosity: wp.float32,
    stiffness: wp.float32,
    rest_density: wp.float32,
    gravity: wp.float32,
):
    tid = wp.tid()
    if tid >= n:
        return
    i = wp.hash_grid_point_id(grid, tid)
    pi = pos[i]
    vi = vel[i]
    rho_i = density[i]

    p_i = stiffness * (rho_i - rest_density)

    h2 = h * h
    fx = float(0.0)  # noqa: UP018
    fy = float(0.0)  # noqa: UP018
    fz = float(0.0)  # noqa: UP018

    h6 = h * h * h * h * h * h
    spiky_coeff = -45.0 / (3.14159265 * h6)
    visc_coeff = 45.0 / (3.14159265 * h6)

    query = wp.hash_grid_query(grid, pi, h)
    j = int(0)
    while wp.hash_grid_query_next(query, j):
        if i == j:
            continue
        d = pos[j] - pi
        dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
        if dist2 < h2 and dist2 > 1e-12:
            dist = wp.sqrt(dist2)
            rho_j = density[j]
            p_j = stiffness * (rho_j - rest_density)

            p_term = -(p_i + p_j) / (2.0 * rho_j)
            q = h - dist
            grad_mag = spiky_coeff * q * q / dist
            fx = fx + mass * p_term * grad_mag * d[0] / dist
            fy = fy + mass * p_term * grad_mag * d[1] / dist
            fz = fz + mass * p_term * grad_mag * d[2] / dist

            vj = vel[j]
            visc_term = viscosity * mass / rho_j * visc_coeff * (h - dist)
            fx = fx + visc_term * (vj[0] - vi[0])
            fy = fy + visc_term * (vj[1] - vi[1])
            fz = fz + visc_term * (vj[2] - vi[2])

    fy = fy + rho_i * gravity

    force[i] = wp.vec3(fx, fy, fz)


@wp.kernel
def _sph_integrate(
    pos: wp.array(dtype=wp.vec3),
    vel: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    density: wp.array(dtype=wp.float32),
    n: wp.int32,
    dt: wp.float32,
    domain_x: wp.float32,
    domain_y: wp.float32,
    domain_z: wp.float32,
    damping: wp.float32,
):
    i = wp.tid()
    if i >= n:
        return
    rho = density[i]
    if rho < 1e-6:
        return

    inv_rho = 1.0 / rho
    f = force[i]
    ax = f[0] * inv_rho
    ay = f[1] * inv_rho
    az = f[2] * inv_rho

    v = vel[i]
    vx = v[0] + ax * dt
    vy = v[1] + ay * dt
    vz = v[2] + az * dt

    p = pos[i]
    px = p[0] + vx * dt
    py = p[1] + vy * dt
    pz = p[2] + vz * dt

    if px < 0.0:
        px = -px
        vx = -vx * damping
    if px > domain_x:
        px = 2.0 * domain_x - px
        vx = -vx * damping
    if py < 0.0:
        py = -py
        vy = -vy * damping
    if py > domain_y:
        py = 2.0 * domain_y - py
        vy = -vy * damping
    if pz < 0.0:
        pz = -pz
        vz = -vz * damping
    if pz > domain_z:
        pz = 2.0 * domain_z - pz
        vz = -vz * damping

    pos[i] = wp.vec3(px, py, pz)
    vel[i] = wp.vec3(vx, vy, vz)


# ---------------------------------------------------------------------------
# SPHSolver
# ---------------------------------------------------------------------------


class SPHSolver:
    """SPH fluid solver with wp.HashGrid accelerated neighbor search.

    For Lagrangian fluid simulation, better for free-surface flows.
    Uses NVIDIA Warp's native HashGrid for O(N) neighbor queries.
    """

    def __init__(
        self,
        n_particles: int,
        smoothing_length: float = 0.1,
        domain: tuple = (1.0, 1.0, 1.0),
        rest_density: float = 1000.0,
        stiffness: float = 2000.0,
        viscosity: float = 200.0,
        gravity: float = -9.81,
        device: str = "cuda",
    ):
        self.n = n_particles
        self.h = smoothing_length
        self.domain = domain
        self.rest_density = rest_density
        self.stiffness = stiffness
        self.viscosity = viscosity
        self.gravity = gravity
        self.device = device

        vol = domain[0] * domain[1] * domain[2]
        self.mass = rest_density * vol / float(n_particles)

        self.pos = wp.zeros(n_particles, dtype=wp.vec3, device=device)
        self.vel = wp.zeros(n_particles, dtype=wp.vec3, device=device)
        self.density = wp.zeros(n_particles, dtype=wp.float32, device=device)
        self.force = wp.zeros(n_particles, dtype=wp.vec3, device=device)

        grid_dim = max(int(math.ceil(max(domain) / smoothing_length)), 4)
        self.grid = wp.HashGrid(grid_dim, grid_dim, grid_dim, device=device)

    def build_spatial_hash(self):
        """Rebuild HashGrid from current particle positions."""
        self.grid.build(self.pos, self.h)

    def compute_density(self):
        """Compute density at each particle using SPH kernels."""
        wp.launch(
            _sph_compute_density,
            dim=self.n,
            inputs=[
                self.grid.id,
                self.pos,
                self.density,
                self.n,
                self.h,
                float(self.mass),
            ],
            device=self.device,
        )

    def compute_forces(self):
        """Compute pressure + viscosity + gravity forces."""
        wp.launch(
            _sph_compute_forces,
            dim=self.n,
            inputs=[
                self.grid.id,
                self.pos,
                self.vel,
                self.density,
                self.force,
                self.n,
                self.h,
                float(self.mass),
                float(self.viscosity),
                float(self.stiffness),
                float(self.rest_density),
                float(self.gravity),
            ],
            device=self.device,
        )

    def integrate(self, dt: float):
        """Symplectic Euler integration with boundary reflection."""
        wp.launch(
            _sph_integrate,
            dim=self.n,
            inputs=[
                self.pos,
                self.vel,
                self.force,
                self.density,
                self.n,
                float(dt),
                float(self.domain[0]),
                float(self.domain[1]),
                float(self.domain[2]),
                0.5,
            ],
            device=self.device,
        )

    def step(self, dt: float = 0.001):
        """build_hash -> density -> forces -> integrate."""
        self.build_spatial_hash()
        self.compute_density()
        self.compute_forces()
        self.integrate(dt)
