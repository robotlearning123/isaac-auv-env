"""Full 3D incompressible Navier-Stokes solver benchmark on RTX 5090.

Projection method (Chorin's splitting):
  1. Advection (semi-Lagrangian, trilinear interpolation)
  2. Diffusion (implicit Jacobi iteration)
  3. Pressure projection (Poisson solve via Jacobi)
  4. Velocity correction (subtract pressure gradient)

Test cases:
  - Taylor-Green vortex: validated against analytical solution
  - Lid-driven cavity: Re=100, Re=1000 centerline profiles

Grid sizes: 32^3, 64^3, 128^3, 256^3 (if memory allows)
"""

from __future__ import annotations

import subprocess
import time

import numpy as np

import warp as wp

wp.init()

DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_mem_mb() -> tuple[int, int]:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    used, total = out.split(",")
    return int(used.strip()), int(total.strip())


def field_bytes(n: int) -> int:
    # velocity (3 components) + pressure + divergence + temporaries
    # ~8 arrays of float32
    return 8 * n * 4


def max_grid_for_mem(gb_limit: float = 28.0) -> int:
    _, total_mb = gpu_mem_mb()
    limit_bytes = gb_limit * 1024**3
    n = int((limit_bytes / 32) ** (1.0 / 3.0))
    # round down to power of 2
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


# ---------------------------------------------------------------------------
# Kernels — Index convention: flat index = ix + iy*nx + iz*nx*ny
# ---------------------------------------------------------------------------

@wp.kernel
def init_taylor_green(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    p: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    L: wp.float32,
    V0: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    dx = L / wp.float32(nx)
    x = (wp.float32(ix) + 0.5) * dx
    y = (wp.float32(iy) + 0.5) * dx
    z = (wp.float32(iz) + 0.5) * dx

    twopi = 2.0 * 3.141592653589793
    k = twopi / L
    # Taylor-Green vortex: u = V0*cos(kx)*sin(ky)*sin(kz)
    u[idx] = V0 * wp.cos(k * x) * wp.sin(k * y) * wp.sin(k * z)
    v[idx] = -V0 * wp.sin(k * x) * wp.cos(k * y) * wp.sin(k * z)
    w_vel[idx] = 0.0
    # analytical pressure for incompressible NS at t=0:
    # p = rho*V0^2/16 * (cos(2kx) + cos(2ky)) * (cos(2kz) + 2)
    p[idx] = (V0 * V0 / 16.0) * (wp.cos(2.0 * k * x) + wp.cos(2.0 * k * y)) * (wp.cos(2.0 * k * z) + 2.0)


@wp.kernel
def init_cavity(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    p: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    lid_vel: wp.float32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    u[idx] = 0.0
    v[idx] = 0.0
    w_vel[idx] = 0.0
    p[idx] = 0.0

    # lid at z = nz-1, moves in x-direction
    if iz == nz - 1 and iy > 0 and iy < ny - 1 and ix > 0 and ix < nx - 1:
        u[idx] = lid_vel


@wp.kernel
def apply_boundary_noslip(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    lid_vel: wp.float32,
    is_cavity: wp.int32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    on_boundary = ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1
    if not on_boundary:
        return

    u[idx] = 0.0
    v[idx] = 0.0
    w_vel[idx] = 0.0

    if is_cavity == 1 and iz == nz - 1:
        u[idx] = lid_vel


@wp.kernel
def advect_semi_lagrangian(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
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

    fx = wp.float32(ix)
    fy = wp.float32(iy)
    fz = wp.float32(iz)

    # backtrace
    xb = fx - u[idx] * dt / dx
    yb = fy - v[idx] * dt / dx
    zb = fz - w_vel[idx] * dt / dx

    # clamp
    xb = wp.max(0.5, wp.min(wp.float32(nx - 1) - 0.5, xb))
    yb = wp.max(0.5, wp.min(wp.float32(ny - 1) - 0.5, yb))
    zb = wp.max(0.5, wp.min(wp.float32(nz - 1) - 0.5, zb))

    x0 = wp.int32(wp.floor(xb))
    y0 = wp.int32(wp.floor(yb))
    z0 = wp.int32(wp.floor(zb))
    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    sx = xb - wp.float32(x0)
    sy = yb - wp.float32(y0)
    sz = zb - wp.float32(z0)
    sx = wp.max(0.0, wp.min(1.0, sx))
    sy = wp.max(0.0, wp.min(1.0, sy))
    sz = wp.max(0.0, wp.min(1.0, sz))

    sx1 = 1.0 - sx
    sy1 = 1.0 - sy
    sz1 = 1.0 - sz

    # helper to trilinear-interp a scalar field
    # u
    c000 = u[x0 + y0 * nx + z0 * nx * ny]
    c100 = u[x1 + y0 * nx + z0 * nx * ny]
    c010 = u[x0 + y1 * nx + z0 * nx * ny]
    c110 = u[x1 + y1 * nx + z0 * nx * ny]
    c001 = u[x0 + y0 * nx + z1 * nx * ny]
    c101 = u[x1 + y0 * nx + z1 * nx * ny]
    c011 = u[x0 + y1 * nx + z1 * nx * ny]
    c111 = u[x1 + y1 * nx + z1 * nx * ny]
    u_new[idx] = (c000*sx1 + c100*sx)*(sy1*sz1) + (c010*sx1 + c110*sx)*(sy*sz1) + (c001*sx1 + c101*sx)*(sy1*sz) + (c011*sx1 + c111*sx)*(sy*sz)

    # v
    c000 = v[x0 + y0 * nx + z0 * nx * ny]
    c100 = v[x1 + y0 * nx + z0 * nx * ny]
    c010 = v[x0 + y1 * nx + z0 * nx * ny]
    c110 = v[x1 + y1 * nx + z0 * nx * ny]
    c001 = v[x0 + y0 * nx + z1 * nx * ny]
    c101 = v[x1 + y0 * nx + z1 * nx * ny]
    c011 = v[x0 + y1 * nx + z1 * nx * ny]
    c111 = v[x1 + y1 * nx + z1 * nx * ny]
    v_new[idx] = (c000*sx1 + c100*sx)*(sy1*sz1) + (c010*sx1 + c110*sx)*(sy*sz1) + (c001*sx1 + c101*sx)*(sy1*sz) + (c011*sx1 + c111*sx)*(sy*sz)

    # w
    c000 = w_vel[x0 + y0 * nx + z0 * nx * ny]
    c100 = w_vel[x1 + y0 * nx + z0 * nx * ny]
    c010 = w_vel[x0 + y1 * nx + z0 * nx * ny]
    c110 = w_vel[x1 + y1 * nx + z0 * nx * ny]
    c001 = w_vel[x0 + y0 * nx + z1 * nx * ny]
    c101 = w_vel[x1 + y0 * nx + z1 * nx * ny]
    c011 = w_vel[x0 + y1 * nx + z1 * nx * ny]
    c111 = w_vel[x1 + y1 * nx + z1 * nx * ny]
    w_new[idx] = (c000*sx1 + c100*sx)*(sy1*sz1) + (c010*sx1 + c110*sx)*(sy*sz1) + (c001*sx1 + c101*sx)*(sy1*sz) + (c011*sx1 + c111*sx)*(sy*sz)


@wp.kernel
def diffuse_jacobi(
    phi: wp.array(dtype=wp.float32),
    phi_new: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    alpha: wp.float32,
    beta: wp.float32,
):
    """One Jacobi iteration for implicit diffusion: (1 + alpha*6)*phi_new = phi + alpha*lap(phi_new).
    alpha = nu*dt/dx^2, beta = 1/(1+6*alpha).
    """
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
    u = phi[idx + nx]
    b = phi[idx - nx * ny]
    t = phi[idx + nx * ny]

    phi_new[idx] = (c + alpha * (l + r + d + u + b + t)) * beta


@wp.kernel
def compute_divergence(
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
def pressure_jacobi(
    p: wp.array(dtype=wp.float32),
    p_new: wp.array(dtype=wp.float32),
    div: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    dx2: wp.float32,
):
    """Jacobi iteration for pressure Poisson: lap(p) = div/dt.
    p_new = (neighbors - dx2*div/dt) / 6.
    We absorb 1/dt into the RHS computed before calling.
    """
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
    u = p[idx + nx]
    b = p[idx - nx * ny]
    t = p[idx + nx * ny]

    p_new[idx] = (l + r + d + u + b + t - dx2 * div[idx]) / 6.0


@wp.kernel
def correct_velocity(
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
def scale_divergence(
    div: wp.array(dtype=wp.float32),
    dt: wp.float32,
    n: wp.int32,
):
    i = wp.tid()
    div[i] = div[i] / dt


@wp.kernel
def compute_l2_sq(
    u: wp.array(dtype=wp.float32),
    v: wp.array(dtype=wp.float32),
    w_vel: wp.array(dtype=wp.float32),
    u_exact: wp.array(dtype=wp.float32),
    v_exact: wp.array(dtype=wp.float32),
    w_exact: wp.array(dtype=wp.float32),
    l2_out: wp.array(dtype=wp.float32),
    n: wp.int32,
):
    i = wp.tid()
    du = u[i] - u_exact[i]
    dv = v[i] - v_exact[i]
    dw = w_vel[i] - w_exact[i]
    err = du * du + dv * dv + dw * dw
    wp.atomic_add(l2_out, 0, err)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class NavierStokesSolver3D:
    def __init__(self, nx: int, ny: int, nz: int, L: float, nu: float, dt: float):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.n = nx * ny * nz
        self.L = L
        self.nu = nu
        self.dt = dt
        self.dx = L / float(nx)
        self.device = DEVICE

        n = self.n
        self.u = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.v = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.w = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.p = wp.zeros(n, dtype=wp.float32, device=self.device)

        self.u_tmp = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.v_tmp = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.w_tmp = wp.zeros(n, dtype=wp.float32, device=self.device)
        self.p_tmp = wp.zeros(n, dtype=wp.float32, device=self.device)

        self.div = wp.zeros(n, dtype=wp.float32, device=self.device)

        self.diffusion_iters = 20
        self.pressure_iters = 50

    def _launch(self, kernel, inputs):
        wp.launch(kernel, dim=self.n, inputs=inputs, device=self.device)

    def init_taylor_green(self, V0: float = 1.0):
        self._launch(init_taylor_green, [
            self.u, self.v, self.w, self.p,
            self.nx, self.ny, self.nz, self.L, V0,
        ])

    def init_cavity(self, lid_vel: float):
        self._launch(init_cavity, [
            self.u, self.v, self.w, self.p,
            self.nx, self.ny, self.nz, lid_vel,
        ])

    def apply_bc(self, lid_vel: float = 0.0, is_cavity: bool = False):
        self._launch(apply_boundary_noslip, [
            self.u, self.v, self.w,
            self.nx, self.ny, self.nz, lid_vel,
            1 if is_cavity else 0,
        ])

    def step(self, lid_vel: float = 0.0, is_cavity: bool = False) -> dict:
        timings = {}
        dx = self.dx
        dt = self.dt
        nu = self.nu

        # 1. Advection
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self._launch(advect_semi_lagrangian, [
            self.u, self.v, self.w, self.u_tmp, self.v_tmp, self.w_tmp,
            self.nx, self.ny, self.nz, dx, dt,
        ])
        wp.synchronize_device(self.device)
        timings["advection"] = time.perf_counter() - t0
        self.u, self.u_tmp = self.u_tmp, self.u
        self.v, self.v_tmp = self.v_tmp, self.v
        self.w, self.w_tmp = self.w_tmp, self.w

        # 2. Diffusion (Jacobi)
        alpha = nu * dt / (dx * dx)
        beta = 1.0 / (1.0 + 6.0 * alpha)
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        for it in range(self.diffusion_iters):
            if it % 2 == 0:
                self._launch(diffuse_jacobi, [
                    self.u, self.u_tmp, self.nx, self.ny, self.nz, alpha, beta,
                ])
                self._launch(diffuse_jacobi, [
                    self.v, self.v_tmp, self.nx, self.ny, self.nz, alpha, beta,
                ])
                self._launch(diffuse_jacobi, [
                    self.w, self.w_tmp, self.nx, self.ny, self.nz, alpha, beta,
                ])
            else:
                self._launch(diffuse_jacobi, [
                    self.u_tmp, self.u, self.nx, self.ny, self.nz, alpha, beta,
                ])
                self._launch(diffuse_jacobi, [
                    self.v_tmp, self.v, self.nx, self.ny, self.nz, alpha, beta,
                ])
                self._launch(diffuse_jacobi, [
                    self.w_tmp, self.w, self.nx, self.ny, self.nz, alpha, beta,
                ])
        wp.synchronize_device(self.device)
        timings["diffusion"] = time.perf_counter() - t0
        # ensure u,v,w hold latest
        if self.diffusion_iters % 2 == 1:
            self.u, self.u_tmp = self.u_tmp, self.u
            self.v, self.v_tmp = self.v_tmp, self.v
            self.w, self.w_tmp = self.w_tmp, self.w

        # boundary
        self.apply_bc(lid_vel, is_cavity)

        # 3. Divergence + Pressure projection
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self._launch(compute_divergence, [
            self.u, self.v, self.w, self.div,
            self.nx, self.ny, self.nz, dx,
        ])
        # scale divergence by 1/dt for Poisson RHS
        self._launch(scale_divergence, [self.div, dt, self.n])

        dx2 = dx * dx
        for it in range(self.pressure_iters):
            if it % 2 == 0:
                self._launch(pressure_jacobi, [
                    self.p, self.p_tmp, self.div, self.nx, self.ny, self.nz, dx2,
                ])
            else:
                self._launch(pressure_jacobi, [
                    self.p_tmp, self.p, self.div, self.nx, self.ny, self.nz, dx2,
                ])
        wp.synchronize_device(self.device)
        timings["projection"] = time.perf_counter() - t0
        if self.pressure_iters % 2 == 1:
            self.p, self.p_tmp = self.p_tmp, self.p

        # 4. Velocity correction
        wp.synchronize_device(self.device)
        t0 = time.perf_counter()
        self._launch(correct_velocity, [
            self.u, self.v, self.w, self.p,
            self.u_tmp, self.v_tmp, self.w_tmp,
            self.nx, self.ny, self.nz, dx, dt,
        ])
        wp.synchronize_device(self.device)
        timings["correction"] = time.perf_counter() - t0
        self.u, self.u_tmp = self.u_tmp, self.u
        self.v, self.v_tmp = self.v_tmp, self.v
        self.w, self.w_tmp = self.w_tmp, self.w

        # boundary
        self.apply_bc(lid_vel, is_cavity)

        return timings

    def compute_l2_error(self, t: float, V0: float) -> float:
        """L2 error norm against analytical Taylor-Green vortex solution."""
        n = self.n
        u_exact = wp.zeros(n, dtype=wp.float32, device=self.device)
        v_exact = wp.zeros(n, dtype=wp.float32, device=self.device)
        w_exact = wp.zeros(n, dtype=wp.float32, device=self.device)

        # compute analytical solution on host, upload
        nx, ny, nz = self.nx, self.ny, self.nz
        L = self.L
        dx = self.dx
        twopi = 2.0 * np.pi
        k = twopi / L
        nu = self.nu

        u_h = np.zeros(n, dtype=np.float32)
        v_h = np.zeros(n, dtype=np.float32)
        w_h = np.zeros(n, dtype=np.float32)

        for iz in range(nz):
            for iy in range(ny):
                for ix in range(nx):
                    idx = ix + iy * nx + iz * nx * ny
                    x = (ix + 0.5) * dx
                    y = (iy + 0.5) * dx
                    z = (iz + 0.5) * dx
                    # analytical decaying Taylor-Green
                    decay = np.exp(-3.0 * nu * k * k * t)
                    u_h[idx] = V0 * np.cos(k * x) * np.sin(k * y) * np.sin(k * z) * decay
                    v_h[idx] = -V0 * np.sin(k * x) * np.cos(k * y) * np.sin(k * z) * decay
                    # for incompressibility: dw/dz = -(du/dx + dv/dy)
                    # analytical w = 0 for the standard 3D Taylor-Green with kz=0 component
                    # but with our initial condition w=0, the exact solution has w=0 throughout
                    w_h[idx] = 0.0

        wp.copy(u_exact, wp.array(u_h, dtype=wp.float32, device=self.device))
        wp.copy(v_exact, wp.array(v_h, dtype=wp.float32, device=self.device))
        wp.copy(w_exact, wp.array(w_h, dtype=wp.float32, device=self.device))

        l2_buf = wp.zeros(1, dtype=wp.float32, device=self.device)
        wp.launch(compute_l2_sq, dim=n, inputs=[
            self.u, self.v, self.w,
            u_exact, v_exact, w_exact,
            l2_buf, n,
        ], device=self.device)
        wp.synchronize_device(self.device)

        l2_sq = float(l2_buf.numpy()[0])
        # exclude boundary cells for fair comparison
        interior = max(1, (nx - 2) * (ny - 2) * (nz - 2))
        return np.sqrt(l2_sq / interior)

    def get_centerline_u(self, axis: str = "z") -> np.ndarray:
        """Get centerline u-velocity profile for cavity benchmark."""
        u_h = self.u.numpy()
        nx, ny, nz = self.nx, self.ny, self.nz
        mid_y = ny // 2
        mid_x = nx // 2

        if axis == "z":
            # vertical centerline at (nx//2, ny//2, :)
            profile = np.zeros(nz, dtype=np.float32)
            for iz in range(nz):
                idx = mid_x + mid_y * nx + iz * nx * ny
                profile[iz] = u_h[idx]
            return profile
        return np.array([])


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_taylor_green(grid_sizes: list[int], n_steps: int = 100) -> list[dict]:
    results = []
    L = 2.0 * np.pi
    V0 = 1.0
    nu = 0.01

    for gs in grid_sizes:
        n = gs * gs * gs
        mem_needed = field_bytes(n)
        used_mb, total_mb = gpu_mem_mb()
        if mem_needed > (total_mb - used_mb) * 1024 * 1024:
            print(f"  {gs}^3: SKIP (need {mem_needed/1024/1024:.0f} MB, only {(total_mb-used_mb)*1024*1024 - mem_needed:.0f} MB free)")
            continue

        dx = L / gs
        dt = 0.5 * dx / V0  # CFL ~0.5
        dt = min(dt, 0.25 * dx * dx / nu)  # diffusion stability

        solver = NavierStokesSolver3D(gs, gs, gs, L, nu, dt)
        solver.init_taylor_green(V0)

        # warmup 5 steps
        for _ in range(5):
            solver.step()
        wp.synchronize_device(DEVICE)

        # measure per-step breakdown
        step_times = {"advection": 0.0, "diffusion": 0.0, "projection": 0.0, "correction": 0.0}
        wp.synchronize_device(DEVICE)
        t_total_start = time.perf_counter()

        for s in range(n_steps):
            timings = solver.step()
            for k in step_times:
                step_times[k] += timings[k]

        wp.synchronize_device(DEVICE)
        t_total = time.perf_counter() - t_total_start

        # validation at t_final
        t_final = n_steps * dt
        l2_err = solver.compute_l2_error(t_final, V0)

        # also validate at intermediate times using fresh solver
        l2_t01 = 0.0
        l2_t05 = 0.0
        l2_t10 = l2_err

        # t=0.1
        n_01 = max(1, int(0.1 / dt))
        solver01 = NavierStokesSolver3D(gs, gs, gs, L, nu, dt)
        solver01.init_taylor_green(V0)
        for _ in range(n_01):
            solver01.step()
        l2_t01 = solver01.compute_l2_error(n_01 * dt, V0)

        # t=0.5
        n_05 = max(1, int(0.5 / dt))
        solver05 = NavierStokesSolver3D(gs, gs, gs, L, nu, dt)
        solver05.init_taylor_green(V0)
        for _ in range(n_05):
            solver05.step()
        l2_t05 = solver05.compute_l2_error(n_05 * dt, V0)

        cells_per_sec = n * n_steps / t_total
        result = {
            "test": "Taylor-Green vortex",
            "grid": f"{gs}^3",
            "cells": n,
            "dt": dt,
            "steps": n_steps,
            "total_s": t_total,
            "ms_per_step": t_total / n_steps * 1000,
            "cells_per_sec": cells_per_sec,
            "advection_ms": step_times["advection"] / n_steps * 1000,
            "diffusion_ms": step_times["diffusion"] / n_steps * 1000,
            "projection_ms": step_times["projection"] / n_steps * 1000,
            "correction_ms": step_times["correction"] / n_steps * 1000,
            "L2_t0.1": l2_t01,
            "L2_t0.5": l2_t05,
            "L2_t1.0": l2_t10,
        }
        results.append(result)
        print(f"  {gs}^3 ({n:>9,d} cells) | {t_total/n_steps*1000:>8.2f} ms/step | {cells_per_sec:>14,.0f} cells/s | dt={dt:.5f}")
        print(f"    L2 err: t=0.1={l2_t01:.6f}  t=0.5={l2_t05:.6f}  t=1.0={l2_t10:.6f}")
        print(f"    Breakdown: adv={step_times['advection']/n_steps*1000:.2f}ms diff={step_times['diffusion']/n_steps*1000:.2f}ms proj={step_times['projection']/n_steps*1000:.2f}ms corr={step_times['correction']/n_steps*1000:.2f}ms")

    return results


def bench_cavity(Re_list: list[int], grid_sizes: list[int], n_steps: int = 200) -> list[dict]:
    results = []
    L = 1.0
    lid_vel = 1.0

    for Re in Re_list:
        nu = lid_vel * L / Re
        print(f"\n  === Re = {Re} (nu = {nu:.6f}) ===")

        for gs in grid_sizes:
            n = gs * gs * gs
            mem_needed = field_bytes(n)
            used_mb, total_mb = gpu_mem_mb()
            if mem_needed > (total_mb - used_mb) * 1024 * 1024:
                print(f"    {gs}^3: SKIP (insufficient memory)")
                continue

            dx = L / gs
            dt = 0.25 * dx / lid_vel  # CFL ~0.25
            dt = min(dt, 0.2 * dx * dx / nu)  # diffusive stability

            solver = NavierStokesSolver3D(gs, gs, gs, L, nu, dt)
            solver.init_cavity(lid_vel)

            # warmup
            for _ in range(10):
                solver.step(lid_vel=lid_vel, is_cavity=True)
            wp.synchronize_device(DEVICE)

            wp.synchronize_device(DEVICE)
            t0 = time.perf_counter()
            for _ in range(n_steps):
                solver.step(lid_vel=lid_vel, is_cavity=True)
            wp.synchronize_device(DEVICE)
            t_total = time.perf_counter() - t0

            profile = solver.get_centerline_u(axis="z")
            cells_per_sec = n * n_steps / t_total

            result = {
                "test": f"Lid-driven cavity Re={Re}",
                "grid": f"{gs}^3",
                "cells": n,
                "Re": Re,
                "nu": nu,
                "dt": dt,
                "steps": n_steps,
                "total_s": t_total,
                "ms_per_step": t_total / n_steps * 1000,
                "cells_per_sec": cells_per_sec,
                "u_centerline_z": profile.tolist(),
            }
            results.append(result)
            print(f"    {gs}^3 ({n:>9,d} cells) | {t_total/n_steps*1000:>8.2f} ms/step | {cells_per_sec:>14,.0f} cells/s | dt={dt:.6f}")
            print(f"    Centerline u(z): min={profile.min():.4f} max={profile.max():.4f}")

    return results


def bench_throughput(grid_sizes: list[int], n_steps: int = 100) -> list[dict]:
    """Pure throughput benchmark with random IC, no validation."""
    results = []
    L = 1.0
    nu = 0.001
    dt_base = 0.001

    for gs in grid_sizes:
        n = gs * gs * gs
        mem_needed = field_bytes(n)
        used_mb, total_mb = gpu_mem_mb()
        if mem_needed > (total_mb - used_mb) * 1024 * 1024:
            print(f"  {gs}^3: SKIP (insufficient memory)")
            continue

        dx = L / gs
        dt = min(dt_base, 0.25 * dx, 0.2 * dx * dx / nu)

        solver = NavierStokesSolver3D(gs, gs, gs, L, nu, dt)
        # random init
        u_np = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        v_np = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        w_np = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
        wp.copy(solver.u, wp.array(u_np, dtype=wp.float32, device=DEVICE))
        wp.copy(solver.v, wp.array(v_np, dtype=wp.float32, device=DEVICE))
        wp.copy(solver.w, wp.array(w_np, dtype=wp.float32, device=DEVICE))

        # warmup
        for _ in range(5):
            solver.step()
        wp.synchronize_device(DEVICE)

        # measure per-step breakdown
        step_times = {"advection": 0.0, "diffusion": 0.0, "projection": 0.0, "correction": 0.0}
        wp.synchronize_device(DEVICE)
        t_total_start = time.perf_counter()

        for _ in range(n_steps):
            timings = solver.step()
            for k in step_times:
                step_times[k] += timings[k]

        wp.synchronize_device(DEVICE)
        t_total = time.perf_counter() - t_total_start
        cells_per_sec = n * n_steps / t_total

        result = {
            "test": "Throughput (random IC)",
            "grid": f"{gs}^3",
            "cells": n,
            "dt": dt,
            "steps": n_steps,
            "total_s": t_total,
            "ms_per_step": t_total / n_steps * 1000,
            "cells_per_sec": cells_per_sec,
            "advection_ms": step_times["advection"] / n_steps * 1000,
            "diffusion_ms": step_times["diffusion"] / n_steps * 1000,
            "projection_ms": step_times["projection"] / n_steps * 1000,
            "correction_ms": step_times["correction"] / n_steps * 1000,
        }
        results.append(result)
        print(f"  {gs}^3 ({n:>9,d} cells) | {t_total/n_steps*1000:>8.2f} ms/step | {cells_per_sec:>14,.0f} cells/s")
        print(f"    adv={step_times['advection']/n_steps*1000:.2f}ms diff={step_times['diffusion']/n_steps*1000:.2f}ms proj={step_times['projection']/n_steps*1000:.2f}ms corr={step_times['correction']/n_steps*1000:.2f}ms")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 90)
    print("3D Incompressible Navier-Stokes Solver Benchmark — RTX 5090 (sm_120, CUDA 12.8)")
    print("Solver: Chorin projection (semi-Lagrangian advection + Jacobi diffusion + Poisson)")
    print("=" * 90)

    used_before, total_before = gpu_mem_mb()
    print(f"\nGPU memory BEFORE: {used_before} / {total_before} MiB")

    # determine max grid size
    max_gs = max_grid_for_mem(26.0)
    print(f"Max grid size for ~26 GB budget: {max_gs}^3")
    all_grids = [gs for gs in [32, 64, 128, 256] if gs <= max_gs]
    print(f"Grid sizes to test: {['{gs}^3' for gs in all_grids]}")

    # ---- 1. Throughput benchmark ----
    print(f"\n{'='*90}")
    print("1. THROUGHPUT BENCHMARK (100 steps, random IC)")
    print(f"{'='*90}")
    tp_results = bench_throughput(all_grids, n_steps=100)

    # ---- 2. Taylor-Green vortex ----
    print(f"\n{'='*90}")
    print("2. TAYLOR-GREEN VORTEX VALIDATION (100 steps)")
    print(f"{'='*90}")
    tg_results = bench_taylor_green(all_grids, n_steps=100)

    # ---- 3. Lid-driven cavity ----
    print(f"\n{'='*90}")
    print("3. LID-DRIVEN CAVITY (200 steps per Re)")
    print(f"{'='*90}")
    cavity_grids = [gs for gs in all_grids if gs <= 128]
    cav_results = bench_cavity([100, 1000], cavity_grids, n_steps=200)

    used_after, _ = gpu_mem_mb()

    # ---- Summary ----
    print(f"\n{'='*90}")
    print("SUMMARY — ALL RAW NUMBERS")
    print(f"{'='*90}")

    print("\n[1. Throughput — Random IC, 100 steps]")
    print(f"  {'Grid':>8s} | {'cells':>10s} | {'ms/step':>9s} | {'cells/s':>14s} | {'adv':>7s} | {'diff':>7s} | {'proj':>7s} | {'corr':>7s}")
    for r in tp_results:
        print(f"  {r['grid']:>8s} | {r['cells']:>10,d} | {r['ms_per_step']:>9.2f} | {r['cells_per_sec']:>14,.0f} | {r['advection_ms']:>7.2f} | {r['diffusion_ms']:>7.2f} | {r['projection_ms']:>7.2f} | {r['correction_ms']:>7.2f}")

    print("\n[2. Taylor-Green Vortex — Validation]")
    print(f"  {'Grid':>8s} | {'cells':>10s} | {'dt':>10s} | {'ms/step':>9s} | {'cells/s':>14s} | {'L2@t=0.1':>10s} | {'L2@t=0.5':>10s} | {'L2@t=1.0':>10s}")
    for r in tg_results:
        print(f"  {r['grid']:>8s} | {r['cells']:>10,d} | {r['dt']:>10.5f} | {r['ms_per_step']:>9.2f} | {r['cells_per_sec']:>14,.0f} | {r['L2_t0.1']:>10.6f} | {r['L2_t0.5']:>10.6f} | {r['L2_t1.0']:>10.6f}")

    print("\n[3. Lid-Driven Cavity]")
    print(f"  {'Re':>6s} | {'Grid':>8s} | {'cells':>10s} | {'dt':>10s} | {'ms/step':>9s} | {'cells/s':>14s} | {'u_min':>8s} | {'u_max':>8s}")
    for r in cav_results:
        profile = r["u_centerline_z"]
        print(f"  {r['Re']:>6d} | {r['grid']:>8s} | {r['cells']:>10,d} | {r['dt']:>10.6f} | {r['ms_per_step']:>9.2f} | {r['cells_per_sec']:>14,.0f} | {min(profile):>8.4f} | {max(profile):>8.4f}")

    print(f"\nGPU memory: {used_before} -> {used_after} MiB (delta: {used_after - used_before} MiB)")
    print(f"\nSolver config: diffusion_jacobi_iters={20}, pressure_jacobi_iters={50}")


if __name__ == "__main__":
    main()
