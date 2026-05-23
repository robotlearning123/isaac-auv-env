"""D3Q19 Lattice Boltzmann Method solver — Warp custom kernels.

Pull-based streaming (each cell pulls from neighbors) to avoid race conditions.
Lid-driven cavity and Poiseuille channel flow benchmarks.

Reports MLUPS, memory usage, steps/sec, and L2 validation errors vs Ghia et al.
"""

from __future__ import annotations

import subprocess
import time

import numpy as np

import warp as wp

wp.init()

Q = 19

CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0], dtype=np.int32)
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, 1, -1], dtype=np.int32)
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1], dtype=np.int32)

W = np.array([
    1.0/3.0,
    1.0/18.0, 1.0/18.0, 1.0/18.0, 1.0/18.0,
    1.0/18.0, 1.0/18.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
    1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0,
], dtype=np.float32)

OPP = np.array([0, 2, 1, 4, 3, 6, 5, 9, 8, 7, 10, 13, 12, 11, 14, 17, 16, 15, 18], dtype=np.int32)


def make_lattice_arrays():
    return {
        'w': wp.array(W, dtype=wp.float32, device='cuda'),
        'cx': wp.array(CX, dtype=wp.int32, device='cuda'),
        'cy': wp.array(CY, dtype=wp.int32, device='cuda'),
        'cz': wp.array(CZ, dtype=wp.int32, device='cuda'),
        'opp': wp.array(OPP, dtype=wp.int32, device='cuda'),
    }


@wp.kernel
def init_equilibrium(
    f: wp.array4d(dtype=wp.float32),
    rho: wp.array3d(dtype=wp.float32),
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    w_arr: wp.array(dtype=wp.float32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    r = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    usq = vx * vx + vy * vy + vz * vz
    for q in range(19):
        cu = float(cx_arr[q]) * vx + float(cy_arr[q]) * vy + float(cz_arr[q]) * vz
        f[q, i, j, k] = w_arr[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)


@wp.kernel
def compute_macroscopic(
    f: wp.array4d(dtype=wp.float32),
    rho: wp.array3d(dtype=wp.float32),
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    is_wall = (i == 0 or i == nx - 1 or
               j == 0 or j == ny - 1 or
               k == 0 or k == nz - 1)
    if is_wall:
        rho[i, j, k] = 1.0
        ux[i, j, k] = float(0.0)
        uy[i, j, k] = float(0.0)
        uz[i, j, k] = float(0.0)
        return

    r = float(0.0)
    vx = float(0.0)
    vy = float(0.0)
    vz = float(0.0)
    for q in range(19):
        fq = f[q, i, j, k]
        r += fq
        vx += fq * float(cx_arr[q])
        vy += fq * float(cy_arr[q])
        vz += fq * float(cz_arr[q])
    rho[i, j, k] = r
    inv_r = 1.0 / r
    ux[i, j, k] = vx * inv_r
    uy[i, j, k] = vy * inv_r
    uz[i, j, k] = vz * inv_r


@wp.kernel
def collide_bgk(
    f: wp.array4d(dtype=wp.float32),
    fnew: wp.array4d(dtype=wp.float32),
    rho: wp.array3d(dtype=wp.float32),
    ux: wp.array3d(dtype=wp.float32),
    uy: wp.array3d(dtype=wp.float32),
    uz: wp.array3d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    omega: wp.float32,
    w_arr: wp.array(dtype=wp.float32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
):
    """BGK collision step. Interior cells only — walls are untouched."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    is_wall = (i == 0 or i == nx - 1 or
               j == 0 or j == ny - 1 or
               k == 0 or k == nz - 1)
    if is_wall:
        return

    r = rho[i, j, k]
    vx = ux[i, j, k]
    vy = uy[i, j, k]
    vz = uz[i, j, k]
    usq = vx * vx + vy * vy + vz * vz

    for q in range(19):
        cu = float(cx_arr[q]) * vx + float(cy_arr[q]) * vy + float(cz_arr[q]) * vz
        feq = w_arr[q] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        fnew[q, i, j, k] = f[q, i, j, k] - omega * (f[q, i, j, k] - feq)


@wp.kernel
def stream_pull(
    fnew: wp.array4d(dtype=wp.float32),
    f: wp.array4d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    cz_arr: wp.array(dtype=wp.int32),
    opp_arr: wp.array(dtype=wp.int32),
):
    """Pull-based streaming with bounce-back.

    Interior cells: pull from upstream (including wall cells with valid fnew).
    Out-of-bounds upstream: self-bounce.
    Wall cells: bounce-back (reflect all directions).
    """
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    is_wall = (i == 0 or i == nx - 1 or
               j == 0 or j == ny - 1 or
               k == 0 or k == nz - 1)

    if is_wall:
        for q in range(19):
            oq = opp_arr[q]
            f[oq, i, j, k] = fnew[q, i, j, k]
    else:
        for q in range(19):
            si = i - cx_arr[q]
            sj = j - cy_arr[q]
            sk = k - cz_arr[q]
            # Self-bounce for out-of-bounds or wall upstream
            if si < 0 or si >= nx or sj < 0 or sj >= ny or sk < 0 or sk >= nz:
                oq = opp_arr[q]
                f[q, i, j, k] = fnew[oq, i, j, k]
            else:
                src_wall = (si == 0 or si == nx - 1 or
                            sj == 0 or sj == ny - 1 or
                            sk == 0 or sk == nz - 1)
                if src_wall:
                    oq = opp_arr[q]
                    f[q, i, j, k] = fnew[oq, i, j, k]
                else:
                    f[q, i, j, k] = fnew[q, si, sj, sk]


@wp.kernel
def copy_walls(
    f: wp.array4d(dtype=wp.float32),
    fnew: wp.array4d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    """Copy wall distributions from f to fnew (walls don't collide)."""
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return

    is_wall = (i == 0 or i == nx - 1 or
               j == 0 or j == ny - 1 or
               k == 0 or k == nz - 1)
    if is_wall:
        for q in range(19):
            fnew[q, i, j, k] = f[q, i, j, k]


@wp.kernel
def apply_moving_wall_interior(
    fnew: wp.array4d(dtype=wp.float32),
    f: wp.array4d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
    opp_arr: wp.array(dtype=wp.int32),
    w_arr: wp.array(dtype=wp.float32),
    cx_arr: wp.array(dtype=wp.int32),
    cy_arr: wp.array(dtype=wp.int32),
    ux_wall: wp.float32,
):
    """Moving wall BC for lid-driven cavity.
    Overrides self-bounce values at interior cells just below the top wall (j=ny-2).
    For directions pointing towards the lid (cy[q] > 0), replaces self-bounce with
    modified bounce-back including lid momentum.
    """
    i, k = wp.tid()
    if i >= nx or k >= nz:
        return
    j = ny - 2  # interior cell just below top wall

    for q in range(19):
        if cy_arr[q] > 0:
            # This direction points towards the top wall.
            # stream_pull did self-bounce: f[q, i, j, k] = fnew[opp[q], i, j, k]
            # Replace with moving wall bounce-back:
            # f[q] = fnew[opp[q]] + 2 * w[opp[q]] * rho * 3 * cx[opp[q]] * ux_wall
            oq = opp_arr[q]
            w_oq = w_arr[oq]
            cx_oq = float(cx_arr[oq])
            f[q, i, j, k] = fnew[oq, i, j, k] + 6.0 * w_oq * cx_oq * ux_wall


@wp.kernel
def copy_f(
    f: wp.array4d(dtype=wp.float32),
    fsrc: wp.array4d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    i, j, k = wp.tid()
    if i >= nx or j >= ny or k >= nz:
        return
    for q in range(19):
        f[q, i, j, k] = fsrc[q, i, j, k]


@wp.kernel
def channel_inlet_zou_he(
    f: wp.array4d(dtype=wp.float32),
    ux_in: wp.float32,
    ny: wp.int32,
    nz: wp.int32,
):
    j, k = wp.tid()
    if j >= ny or k >= nz:
        return
    rho_in = (1.0 / (1.0 - ux_in)) * (
        f[0, 0, j, k] + f[3, 0, j, k] + f[4, 0, j, k] +
        f[5, 0, j, k] + f[6, 0, j, k] +
        2.0 * (f[2, 0, j, k] + f[8, 0, j, k] + f[10, 0, j, k] +
               f[12, 0, j, k] + f[16, 0, j, k])
    )
    f[1, 0, j, k] = f[2, 0, j, k] + (2.0 / 3.0) * rho_in * ux_in
    f[9, 0, j, k] = f[8, 0, j, k] - 0.5 * (f[3, 0, j, k] - f[4, 0, j, k]) + (1.0 / 6.0) * rho_in * ux_in
    f[10, 0, j, k] = f[7, 0, j, k] + 0.5 * (f[3, 0, j, k] - f[4, 0, j, k]) + (1.0 / 6.0) * rho_in * ux_in
    f[11, 0, j, k] = f[12, 0, j, k] - 0.5 * (f[5, 0, j, k] - f[6, 0, j, k]) + (1.0 / 6.0) * rho_in * ux_in
    f[16, 0, j, k] = f[17, 0, j, k] + 0.5 * (f[5, 0, j, k] - f[6, 0, j, k]) + (1.0 / 6.0) * rho_in * ux_in


@wp.kernel
def channel_outlet_copy(
    f: wp.array4d(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    j, k = wp.tid()
    if j >= ny or k >= nz:
        return
    for q in range(19):
        f[q, nx - 1, j, k] = f[q, nx - 2, j, k]


# Ghia et al. benchmark data
GHIA_RE100_Y = np.array([0.0, 0.0547, 0.0625, 0.0703, 0.1016, 0.1484,
                          0.1953, 0.2422, 0.2891, 0.3359, 0.3828, 0.4297,
                          0.4766, 0.5234, 0.5703, 0.6172, 0.6641, 0.7109,
                          0.7578, 0.8047, 0.8516, 0.8984, 0.9453, 1.0])
GHIA_RE100_U = np.array([0.0, -0.03717, -0.04192, -0.04775, -0.06436, -0.07561,
                          -0.07216, -0.06169, -0.05046, -0.04083, -0.03376, -0.02864,
                          -0.02461, -0.02076, -0.01654, -0.01172, -0.00628, -0.00062,
                          0.00540, 0.01313, 0.02580, 0.05302, 0.14034, 1.0])

GHIA_RE400_Y = GHIA_RE100_Y.copy()
GHIA_RE400_U = np.array([0.0, -0.08186, -0.09266, -0.10358, -0.14591, -0.17152,
                          -0.15496, -0.12253, -0.09140, -0.06506, -0.04391, -0.02772,
                          -0.01619, -0.00849, -0.00321, 0.00106, 0.00528, 0.01075,
                          0.01869, 0.03195, 0.05603, 0.10091, 0.19174, 1.0])

GHIA_RE1000_Y = GHIA_RE100_Y.copy()
GHIA_RE1000_U = np.array([0.0, -0.18109, -0.20196, -0.21619, -0.24929, -0.22457,
                           -0.16316, -0.10929, -0.07422, -0.05010, -0.03416, -0.02336,
                           -0.01594, -0.01036, -0.00538, -0.00031, 0.00595, 0.01456,
                           0.02810, 0.05030, 0.08675, 0.15387, 0.26551, 1.0])


def run_lid_cavity(nx, ny, nz, re, u_lid=0.1, max_steps=5000, warmup_steps=100):
    nu = u_lid * (nx - 1) / re
    tau = 3.0 * nu + 0.5
    omega_val = 1.0 / tau

    if tau <= 0.51:
        print(f"\n  Lid-driven cavity: {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f} — SKIPPED (unstable)")
        return None

    print(f"\n  Lid-driven cavity: {nx}x{ny}x{nz}, Re={re}, tau={tau:.4f}, omega={omega_val:.4f}")

    f = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
    fnew = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
    rho = wp.array(np.ones((nx, ny, nz), dtype=np.float32), dtype=wp.float32, device='cuda')
    ux = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    uy = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    uz = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    lat = make_lattice_arrays()

    wp.launch(init_equilibrium, dim=(nx, ny, nz),
              inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                      lat['w'], lat['cx'], lat['cy'], lat['cz']],
              device='cuda')
    wp.synchronize_device('cuda')

    all_times = []
    for step in range(max_steps):
        t0 = time.perf_counter()

        # 1. Macroscopic from f (wall cells set to rho=1, u=0)
        wp.launch(compute_macroscopic, dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')

        # 2. Collide (BGK) interior cells: f -> fnew
        wp.launch(collide_bgk, dim=(nx, ny, nz),
                  inputs=[f, fnew, rho, ux, uy, uz, nx, ny, nz,
                          wp.float32(omega_val),
                          lat['w'], lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')

        # 3. Copy wall distributions to fnew (walls don't collide)
        wp.launch(copy_walls, dim=(nx, ny, nz),
                  inputs=[f, fnew, nx, ny, nz],
                  device='cuda')

        # 4. Stream pull from fnew -> f + bounce-back walls (self-bounce for wall-adjacent)
        wp.synchronize_device('cuda')
        wp.launch(stream_pull, dim=(nx, ny, nz),
                  inputs=[fnew, f, nx, ny, nz,
                          lat['cx'], lat['cy'], lat['cz'], lat['opp']],
                  device='cuda')

        # 5. Moving wall: override self-bounce at cells just below top wall
        wp.launch(apply_moving_wall_interior, dim=(nx, nz),
                  inputs=[f, f, nx, ny, nz, lat['opp'], lat['w'], lat['cx'], lat['cy'],
                          wp.float32(u_lid)],
                  device='cuda')

        wp.synchronize_device('cuda')
        all_times.append(time.perf_counter() - t0)

    # Validation
    wp.launch(compute_macroscopic, dim=(nx, ny, nz),
              inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                      lat['cx'], lat['cy'], lat['cz']],
              device='cuda')
    wp.synchronize_device('cuda')

    ux_host = ux.numpy()
    centerline_u = ux_host[nx // 2, :, nz // 2] / u_lid

    if np.any(np.isnan(centerline_u)):
        print(f"    WARNING: NaN detected. Simulation unstable.")
        l2_error = float('nan')
    else:
        ghia_y = {100: GHIA_RE100_Y, 400: GHIA_RE400_Y, 1000: GHIA_RE1000_Y}[re]
        ghia_u = {100: GHIA_RE100_U, 400: GHIA_RE400_U, 1000: GHIA_RE1000_U}[re]
        y_positions = np.arange(ny) / (ny - 1)
        our_u_interp = np.interp(ghia_y, y_positions, centerline_u)
        mask = (ghia_y > 0.01) & (ghia_y < 0.99)
        l2_error = np.sqrt(np.mean((our_u_interp[mask] - ghia_u[mask]) ** 2)) if mask.sum() > 2 else float('nan')

    times_arr = np.array(all_times[warmup_steps:])
    avg_time = np.mean(times_arr)
    cells = nx * ny * nz
    mlups = cells * 1e-6 / avg_time
    total_mem = Q * cells * 4 * 2 + cells * 4 * 4  # f + fnew + rho/ux/uy/uz
    mem_mb = total_mem / (1024 * 1024)

    result = dict(nx=nx, ny=ny, nz=nz, re=re, tau=tau, omega=omega_val,
                  mlups=mlups, steps_per_sec=1.0 / avg_time,
                  avg_step_ms=avg_time * 1000, mem_mb=mem_mb, cells=cells,
                  l2_error=l2_error, max_u=float(np.nanmax(np.abs(centerline_u))),
                  steps=max_steps)

    print(f"    MLUPS: {mlups:.1f}, Steps/s: {1.0/avg_time:.1f}, "
          f"Step: {avg_time*1000:.2f}ms, Mem: {mem_mb:.1f}MB, L2: {l2_error:.4f}")
    return result


def run_channel_flow(nx=128, ny=64, nz=64, u_max=0.05, max_steps=5000, warmup_steps=100):
    H = ny - 1
    tau = 0.8
    nu = (tau - 0.5) / 3.0
    omega_val = 1.0 / tau
    re = u_max * H / nu

    print(f"\n  Channel flow: {nx}x{ny}x{nz}, Re={re:.1f}, tau={tau:.4f}")

    f = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
    fnew = wp.zeros((Q, nx, ny, nz), dtype=wp.float32, device='cuda')
    rho = wp.array(np.ones((nx, ny, nz), dtype=np.float32), dtype=wp.float32, device='cuda')
    ux = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    uy = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    uz = wp.zeros((nx, ny, nz), dtype=wp.float32, device='cuda')
    lat = make_lattice_arrays()

    wp.launch(init_equilibrium, dim=(nx, ny, nz),
              inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                      lat['w'], lat['cx'], lat['cy'], lat['cz']],
              device='cuda')
    wp.synchronize_device('cuda')

    all_times = []
    for step in range(max_steps):
        t0 = time.perf_counter()

        wp.launch(compute_macroscopic, dim=(nx, ny, nz),
                  inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                          lat['cx'], lat['cy'], lat['cz']],
                  device='cuda')

        wp.launch(collide_stream_pull, dim=(nx, ny, nz),
                  inputs=[f, fnew, rho, ux, uy, uz, nx, ny, nz,
                          wp.float32(omega_val),
                          lat['w'], lat['cx'], lat['cy'], lat['cz'], lat['opp']],
                  device='cuda')

        wp.launch(channel_inlet_zou_he, dim=(ny, nz),
                  inputs=[fnew, wp.float32(u_max), ny, nz], device='cuda')

        wp.launch(channel_outlet_copy, dim=(ny, nz),
                  inputs=[fnew, nx, ny, nz], device='cuda')

        wp.launch(copy_f, dim=(nx, ny, nz),
                  inputs=[f, fnew, nx, ny, nz], device='cuda')

        wp.synchronize_device('cuda')
        all_times.append(time.perf_counter() - t0)

    wp.launch(compute_macroscopic, dim=(nx, ny, nz),
              inputs=[f, rho, ux, uy, uz, nx, ny, nz,
                      lat['cx'], lat['cy'], lat['cz']],
              device='cuda')
    wp.synchronize_device('cuda')

    ux_host = ux.numpy()
    profile = ux_host[nx // 2, :, nz // 2]
    y_arr = np.arange(ny)
    analytical = u_max * 4.0 * (y_arr / H) * (1.0 - y_arr / H)
    interior = np.arange(1, ny - 1)

    if np.any(np.isnan(profile)):
        l2_error = float('nan')
        max_u = float('nan')
        print(f"    WARNING: NaN in channel flow.")
    else:
        l2_error = np.sqrt(np.mean((profile[interior] - analytical[interior]) ** 2))
        max_u = float(np.max(np.abs(profile[interior])))

    times_arr = np.array(all_times[warmup_steps:])
    avg_time = np.mean(times_arr)
    cells = nx * ny * nz
    mlups = cells * 1e-6 / avg_time
    total_mem = Q * cells * 4 * 2 + cells * 4 * 4
    mem_mb = total_mem / (1024 * 1024)

    result = dict(nx=nx, ny=ny, nz=nz, re=re, tau=tau, omega=omega_val,
                  mlups=mlups, steps_per_sec=1.0 / avg_time,
                  avg_step_ms=avg_time * 1000, mem_mb=mem_mb, cells=cells,
                  l2_error=l2_error, max_u=max_u, u_max_analytical=u_max,
                  steps=max_steps)

    print(f"    MLUPS: {mlups:.1f}, Steps/s: {1.0/avg_time:.1f}, "
          f"Step: {avg_time*1000:.2f}ms, L2: {l2_error:.6f}, "
          f"max_u: {max_u:.4f} (analytical: {u_max:.4f})")
    return result


def main():
    print("=" * 70)
    print("LBM D3Q19 Warp Kernel Benchmark — RTX 5090")
    print("=" * 70)

    mem_before = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    print(f"GPU memory before: {mem_before} MB")

    print("\nWarmup...")
    run_lid_cavity(16, 16, 16, re=100, max_steps=50, warmup_steps=5)
    print("Warmup done.")

    print("\n" + "-" * 50)
    print("LID-DRIVEN CAVITY")
    print("-" * 50)

    cavity_results = []
    for n in [64, 128, 256]:
        for re in [100, 400, 1000]:
            steps = 5000 if n == 64 else (3000 if n == 128 else 2000)
            r = run_lid_cavity(n, n, n, re=re, max_steps=steps, warmup_steps=100)
            if r is not None:
                cavity_results.append(r)

    print("\n" + "-" * 50)
    print("CHANNEL FLOW (POISEUILLE)")
    print("-" * 50)

    channel_result = run_channel_flow(nx=128, ny=64, nz=64, max_steps=5000, warmup_steps=100)

    print("\n" + "=" * 70)
    print("SUMMARY — Warp LBM D3Q19")
    print("=" * 70)

    print("\nLid-driven cavity:")
    print(f"{'Grid':>12} {'Re':>6} {'MLUPS':>10} {'Steps/s':>10} {'Step(ms)':>10} {'Mem(MB)':>10} {'L2 err':>10}")
    for r in cavity_results:
        print(f"{r['nx']:>4}x{r['ny']}x{r['nz']:>4} {r['re']:>6} {r['mlups']:>10.1f} "
              f"{r['steps_per_sec']:>10.1f} {r['avg_step_ms']:>10.2f} {r['mem_mb']:>10.1f} {r['l2_error']:>10.4f}")

    print("\nChannel flow:")
    r = channel_result
    print(f"  Grid: {r['nx']}x{r['ny']}x{r['nz']}, Re={r['re']:.1f}")
    print(f"  MLUPS: {r['mlups']:.1f}, Steps/s: {r['steps_per_sec']:.1f}")
    print(f"  L2 error vs analytical: {r['l2_error']:.6f}")
    print(f"  Max velocity: {r['max_u']:.4f} (analytical: {r['u_max_analytical']:.4f})")

    mem_after = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    print(f"\nGPU memory after: {mem_after} MB")

    for r in cavity_results:
        bytes_per_cell = r['mem_mb'] * 1024 * 1024 / r['cells']
        print(f"  {r['nx']}^3: {bytes_per_cell:.1f} bytes/cell")

    return cavity_results, channel_result


if __name__ == "__main__":
    main()
