"""Warp GPU fluid simulation kernel benchmarks on RTX 5090.

Tests:
1. SPH neighbor search — hash-grid neighbor finding for N particles
2. Grid-based pressure solve — Jacobi iteration on 3D grid
3. Velocity advection — Semi-Lagrangian on 3D grid
4. Kernel launch overhead — trivial identity kernel timing
"""

from __future__ import annotations

import subprocess
import time

import numpy as np

import warp as wp

wp.init()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gpu_mem_usage() -> str:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    ).strip()
    return out  # "used_mb, total_mb"


def warmup_kernel(kernel, dim, inputs, device="cuda", iters=10):
    for _ in range(iters):
        wp.launch(kernel, dim=dim, inputs=inputs, device=device)
    wp.synchronize_device(device)


# ---------------------------------------------------------------------------
# 1. SPH Neighbor Search
# ---------------------------------------------------------------------------

@wp.kernel
def sph_count_neighbors(
    pos: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
    n: wp.int32,
    h: wp.float32,
):
    i = wp.tid()
    pi = pos[i]
    c = wp.int32(0)
    h2 = h * h
    for j in range(n):
        if i != j:
            d = pos[j] - pi
            dist2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
            if dist2 < h2:
                c += 1
    counts[i] = c


def bench_sph_neighbor_search(particle_counts: list[int]) -> list[dict]:
    results = []
    for n in particle_counts:
        h = 0.1  # search radius in a [0,1]^3 box
        positions_np = np.random.uniform(0, 1, (n, 3)).astype(np.float32)
        pos = wp.array(positions_np, dtype=wp.vec3, device="cuda")
        counts = wp.zeros(n, dtype=wp.int32, device="cuda")

        # warmup (use small dim for warmup speed)
        warmup_kernel(sph_count_neighbors, dim=min(n, 1024), inputs=[pos, counts, n, h])

        # benchmark
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        wp.launch(sph_count_neighbors, dim=n, inputs=[pos, counts, n, h], device="cuda")
        wp.synchronize_device("cuda")
        dt = time.perf_counter() - t0

        ms = dt * 1000
        pps = n / dt
        results.append({
            "test": "SPH neighbor search",
            "N": n,
            "ms_per_step": ms,
            "particles_per_sec": pps,
        })
        print(f"  N={n:>7,d} | {ms:>10.2f} ms | {pps:>14,.0f} p/s")

    return results


# ---------------------------------------------------------------------------
# 2. Grid-based Pressure Solve (Jacobi)
# ---------------------------------------------------------------------------

@wp.kernel
def jacobi_step(
    p: wp.array(dtype=wp.float32),
    p_new: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    nx: wp.int32,
    ny: wp.int32,
    nz: wp.int32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx

    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        p_new[idx] = 0.0
        return

    c = p[idx]
    l = p[idx - 1]
    r = p[idx + 1]
    d = p[idx - nx]
    u = p[idx + nx]
    b = p[idx - nx * ny]
    t = p[idx + nx * ny]

    p_new[idx] = (l + r + d + u + b + t - rhs[idx]) * 0.16666667  # 1/6


def bench_pressure_solve(grid_sizes: list[int], jacobi_iters: int = 100) -> list[dict]:
    results = []
    for gs in grid_sizes:
        total = gs * gs * gs
        p = wp.zeros(total, dtype=wp.float32, device="cuda")
        p_new = wp.zeros(total, dtype=wp.float32, device="cuda")
        rhs_np = np.random.uniform(-1, 1, total).astype(np.float32)
        rhs = wp.array(rhs_np, dtype=wp.float32, device="cuda")

        # warmup
        for _ in range(5):
            wp.launch(jacobi_step, dim=total, inputs=[p, p_new, rhs, gs, gs, gs], device="cuda")
        wp.synchronize_device("cuda")

        # benchmark: ping-pong jacobi iterations
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        for it in range(jacobi_iters):
            if it % 2 == 0:
                wp.launch(jacobi_step, dim=total, inputs=[p, p_new, rhs, gs, gs, gs], device="cuda")
            else:
                wp.launch(jacobi_step, dim=total, inputs=[p_new, p, rhs, gs, gs, gs], device="cuda")
        wp.synchronize_device("cuda")
        dt = time.perf_counter() - t0

        ms = dt * 1000
        cells_per_sec = total * jacobi_iters / dt
        results.append({
            "test": "Pressure solve (Jacobi)",
            "grid": f"{gs}^3",
            "cells": total,
            "iters": jacobi_iters,
            "ms_total": ms,
            "ms_per_iter": ms / jacobi_iters,
            "cells_per_sec": cells_per_sec,
        })
        print(f"  {gs}^3 = {total:>9,d} cells | {ms:>8.2f} ms total | {ms/jacobi_iters:>7.3f} ms/iter | {cells_per_sec:>14,.0f} cells/s")

    return results


# ---------------------------------------------------------------------------
# 3. Velocity Advection (Semi-Lagrangian)
# ---------------------------------------------------------------------------

@wp.kernel
def semi_lagrangian_advect(
    vel: wp.array(dtype=wp.vec3),
    vel_new: wp.array(dtype=wp.vec3),
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

    # backtrace
    v = vel[idx]
    x_back = wp.float32(ix) - v[0] * dt / dx
    y_back = wp.float32(iy) - v[1] * dt / dx
    z_back = wp.float32(iz) - v[2] * dt / dx

    # clamp to grid bounds
    x0 = wp.int32(wp.max(0.0, wp.min(wp.float32(nx - 2), wp.floor(x_back))))
    y0 = wp.int32(wp.max(0.0, wp.min(wp.float32(ny - 2), wp.floor(y_back))))
    z0 = wp.int32(wp.max(0.0, wp.min(wp.float32(nz - 2), wp.floor(z_back))))

    x1 = wp.min(x0 + 1, nx - 1)
    y1 = wp.min(y0 + 1, ny - 1)
    z1 = wp.min(z0 + 1, nz - 1)

    # fractional parts
    sx = wp.min(wp.max(x_back - wp.float32(x0), 0.0), 1.0)
    sy = wp.min(wp.max(y_back - wp.float32(y0), 0.0), 1.0)
    sz = wp.min(wp.max(z_back - wp.float32(z0), 0.0), 1.0)

    # trilinear interpolation (inlined — warp does not support nested def)
    c000 = vel[x0 + y0 * nx + z0 * nx * ny]
    c100 = vel[x1 + y0 * nx + z0 * nx * ny]
    c010 = vel[x0 + y1 * nx + z0 * nx * ny]
    c110 = vel[x1 + y1 * nx + z0 * nx * ny]
    c001 = vel[x0 + y0 * nx + z1 * nx * ny]
    c101 = vel[x1 + y0 * nx + z1 * nx * ny]
    c011 = vel[x0 + y1 * nx + z1 * nx * ny]
    c111 = vel[x1 + y1 * nx + z1 * nx * ny]

    sx1 = 1.0 - sx
    sy1 = 1.0 - sy
    sz1 = 1.0 - sz

    c00 = c000 * sx1 + c100 * sx
    c10 = c010 * sx1 + c110 * sx
    c01 = c001 * sx1 + c101 * sx
    c11 = c011 * sx1 + c111 * sx

    c0 = c00 * sy1 + c10 * sy
    c1 = c01 * sy1 + c11 * sy

    vel_new[idx] = c0 * sz1 + c1 * sz


def bench_velocity_advection(grid_sizes: list[int]) -> list[dict]:
    results = []
    dx = 1.0 / 64.0
    dt_sim = 0.01

    for gs in grid_sizes:
        total = gs * gs * gs
        vel_np = np.random.uniform(-0.5, 0.5, (total, 3)).astype(np.float32)
        vel = wp.array(vel_np, dtype=wp.vec3, device="cuda")
        vel_new = wp.zeros(total, dtype=wp.vec3, device="cuda")

        # warmup
        warmup_kernel(semi_lagrangian_advect, dim=total, inputs=[vel, vel_new, gs, gs, gs, dx, dt_sim])

        # benchmark
        n_steps = 50
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        for s in range(n_steps):
            if s % 2 == 0:
                wp.launch(semi_lagrangian_advect, dim=total, inputs=[vel, vel_new, gs, gs, gs, dx, dt_sim], device="cuda")
            else:
                wp.launch(semi_lagrangian_advect, dim=total, inputs=[vel_new, vel, gs, gs, gs, dx, dt_sim], device="cuda")
        wp.synchronize_device("cuda")
        elapsed = time.perf_counter() - t0

        ms_per = elapsed / n_steps * 1000
        cells_per_sec = total * n_steps / elapsed
        results.append({
            "test": "Velocity advection",
            "grid": f"{gs}^3",
            "cells": total,
            "ms_per_step": ms_per,
            "cells_per_sec": cells_per_sec,
        })
        print(f"  {gs}^3 = {total:>9,d} cells | {ms_per:>8.3f} ms/step | {cells_per_sec:>14,.0f} cells/s")

    return results


# ---------------------------------------------------------------------------
# 4. Kernel Launch Overhead
# ---------------------------------------------------------------------------

@wp.kernel
def identity_kernel(data: wp.array(dtype=wp.float32)):
    i = wp.tid()
    data[i] = data[i] * 1.0


def bench_launch_overhead(launch_counts: list[int]) -> list[dict]:
    results = []
    data = wp.zeros(1, dtype=wp.float32, device="cuda")

    # warmup
    for _ in range(100):
        wp.launch(identity_kernel, dim=1, inputs=[data], device="cuda")
    wp.synchronize_device("cuda")

    for count in launch_counts:
        wp.synchronize_device("cuda")
        t0 = time.perf_counter()
        for _ in range(count):
            wp.launch(identity_kernel, dim=1, inputs=[data], device="cuda")
        wp.synchronize_device("cuda")
        dt = time.perf_counter() - t0

        us_per_launch = dt / count * 1e6
        results.append({
            "test": "Kernel launch overhead",
            "launches": count,
            "total_ms": dt * 1000,
            "us_per_launch": us_per_launch,
        })
        print(f"  {count:>6,d} launches | {dt*1000:>8.2f} ms total | {us_per_launch:>8.2f} us/launch")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("Warp Fluid Kernel Benchmarks — RTX 5090 (sm_120, CUDA 12.8)")
    print("=" * 80)

    mem_before = gpu_mem_usage()
    print(f"\nGPU memory BEFORE: {mem_before} MiB")

    # ---- 1. SPH Neighbor Search ----
    print("\n--- 1. SPH Neighbor Search (brute-force, radius h=0.1) ---")
    print(f"  {'N':>9s} | {'ms/step':>10s} | {'particles/s':>14s}")
    sph_results = bench_sph_neighbor_search([10_000, 50_000, 100_000, 500_000])

    # ---- 2. Pressure Solve ----
    print("\n--- 2. Grid Pressure Solve (Jacobi, 100 iters) ---")
    print(f"  {'Grid':>12s} | {'ms total':>10s} | {'ms/iter':>9s} | {'cells/s':>14s}")
    ps_results = bench_pressure_solve([32, 64, 128], jacobi_iters=100)

    # ---- 3. Velocity Advection ----
    print("\n--- 3. Semi-Lagrangian Velocity Advection ---")
    print(f"  {'Grid':>12s} | {'ms/step':>10s} | {'cells/s':>14s}")
    va_results = bench_velocity_advection([32, 64, 128])

    # ---- 4. Launch Overhead ----
    print("\n--- 4. Kernel Launch Overhead (identity, dim=1) ---")
    print(f"  {'Launches':>9s} | {'total ms':>10s} | {'us/launch':>10s}")
    lo_results = bench_launch_overhead([100, 1_000, 10_000])

    mem_after = gpu_mem_usage()
    print(f"\nGPU memory AFTER:  {mem_after} MiB")

    # ---- Summary Table ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n[SPH Neighbor Search]")
    for r in sph_results:
        print(f"  N={r['N']:>7,d}  |  {r['ms_per_step']:.2f} ms  |  {r['particles_per_sec']:,.0f} p/s")

    print("\n[Pressure Solve — Jacobi x100]")
    for r in ps_results:
        print(f"  {r['grid']:>6s} ({r['cells']:>9,d} cells)  |  {r['ms_total']:.2f} ms  |  {r['ms_per_iter']:.3f} ms/iter  |  {r['cells_per_sec']:,.0f} cells/s")

    print("\n[Velocity Advection — Semi-Lagrangian]")
    for r in va_results:
        print(f"  {r['grid']:>6s} ({r['cells']:>9,d} cells)  |  {r['ms_per_step']:.3f} ms  |  {r['cells_per_sec']:,.0f} cells/s")

    print("\n[Kernel Launch Overhead]")
    for r in lo_results:
        print(f"  {r['launches']:>6,d} launches  |  {r['total_ms']:.2f} ms  |  {r['us_per_launch']:.2f} us/launch")

    print(f"\nGPU memory: BEFORE {mem_before} MiB  |  AFTER {mem_after} MiB")


if __name__ == "__main__":
    main()
