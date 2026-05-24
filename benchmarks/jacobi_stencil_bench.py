"""Unified Jacobi stencil benchmark: 2D & 3D pressure solve across 5 GPU frameworks.

Consolidates cross_framework_fluid_bench.py, framework_fluid_compare.py,
and triton_cfd_bench.py into a single parametric benchmark.

Frameworks: CuPy, JAX, PyTorch (eager + compile), Warp, Triton
Dimensions: 2D (5-point stencil) and 3D (7-point stencil)
"""

from __future__ import annotations

import argparse
import gc
import time

import numpy as np

ALL_FRAMEWORKS = ["warp", "pytorch", "pytorch-compile", "cupy", "cupy-raw", "jax", "triton"]
DEFAULT_SIZES_2D = [256, 512, 1024, 2048]
DEFAULT_SIZES_3D = [32, 64, 128]

_CONFIG = {"iters_2d": 1000, "iters_3d": 100}
WARMUP = 5


def _sync_torch():
    import torch
    torch.cuda.synchronize()


def _reset_torch_mem():
    import torch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


# =========================================================================
# 2D Jacobi — 5-point stencil: p_new = (L + R + U + D - rhs) * 0.25
# =========================================================================

def bench_2d_cupy_raw(n: int) -> dict:
    import cupy as cp

    kernel = cp.RawKernel(r"""
extern "C" __global__
void jacobi2d(const float* p, const float* rhs, float* p_new, int nx, int ny) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;
    int idx = j * nx + i;
    p_new[idx] = (p[idx-1] + p[idx+1] + p[idx-nx] + p[idx+nx] - rhs[idx]) * 0.25f;
}
""", "jacobi2d")

    p = cp.zeros((n, n), dtype=cp.float32)
    rhs = cp.random.randn(n, n, dtype=cp.float32) * 0.01
    p_new = cp.empty_like(p)
    threads = (16, 16)
    blocks = ((n + 15) // 16, (n + 15) // 16)

    for _ in range(WARMUP):
        kernel(blocks, threads, (p, rhs, p_new, np.int32(n), np.int32(n)))
        p, p_new = p_new, p
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_2d"]):
        kernel(blocks, threads, (p, rhs, p_new, np.int32(n), np.int32(n)))
        p, p_new = p_new, p
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    cells_s = n * n * _CONFIG["iters_2d"] / dt
    mem = cp.get_default_memory_pool().used_bytes() / 1e9
    return {"framework": "CuPy-Raw", "dim": "2D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_2d"] * 1000, "cells_s": cells_s, "mem_gb": mem}


def bench_2d_cupy(n: int) -> dict:
    import cupy as cp

    p = cp.zeros((n, n), dtype=cp.float32)
    rhs = cp.random.randn(n, n, dtype=cp.float32) * 0.01
    p_new = cp.empty_like(p)

    for _ in range(WARMUP):
        p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                              p[1:-1, :-2] + p[1:-1, 2:] - rhs[1:-1, 1:-1]) * 0.25
        p, p_new = p_new, p
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_2d"]):
        p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                              p[1:-1, :-2] + p[1:-1, 2:] - rhs[1:-1, 1:-1]) * 0.25
        p, p_new = p_new, p
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    cells_s = n * n * _CONFIG["iters_2d"] / dt
    mem = cp.get_default_memory_pool().used_bytes() / 1e9
    return {"framework": "CuPy", "dim": "2D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_2d"] * 1000, "cells_s": cells_s, "mem_gb": mem}


def bench_2d_jax(n: int) -> dict:
    import jax
    import jax.numpy as jnp

    @jax.jit
    def step(p, rhs):
        p_new = jnp.zeros_like(p)
        return p_new.at[1:-1, 1:-1].set(
            (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - rhs[1:-1, 1:-1]) * 0.25)

    def scan_jacobi(p, rhs, n_iter):
        def body(carry, _):
            return step(carry, rhs), None
        p, _ = jax.lax.scan(body, p, None, length=n_iter)
        return p

    scan_jit = jax.jit(scan_jacobi, static_argnums=(2,))

    p = jnp.zeros((n, n), dtype=jnp.float32)
    rhs = jax.random.normal(jax.random.key(42), (n, n), dtype=jnp.float32) * 0.01

    _ = scan_jit(p, rhs, _CONFIG["iters_2d"]).block_until_ready()

    t0 = time.perf_counter()
    _ = scan_jit(p, rhs, _CONFIG["iters_2d"]).block_until_ready()
    dt = time.perf_counter() - t0

    cells_s = n * n * _CONFIG["iters_2d"] / dt
    return {"framework": "JAX", "dim": "2D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_2d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


def bench_2d_pytorch(n: int, compile_mode: bool = False) -> dict:
    import torch

    p = torch.zeros(n, n, device="cuda", dtype=torch.float32)
    rhs = torch.randn(n, n, device="cuda", dtype=torch.float32) * 0.01

    def step(p_in):
        p_new = torch.zeros_like(p_in)
        p_new[1:-1, 1:-1] = (p_in[:-2, 1:-1] + p_in[2:, 1:-1] +
                              p_in[1:-1, :-2] + p_in[1:-1, 2:] - rhs[1:-1, 1:-1]) * 0.25
        return p_new

    fn = torch.compile(step, mode="default") if compile_mode else step

    for _ in range(WARMUP):
        p = fn(p)
    _sync_torch()

    _sync_torch()
    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_2d"]):
        p = fn(p)
    _sync_torch()
    dt = time.perf_counter() - t0

    cells_s = n * n * _CONFIG["iters_2d"] / dt
    label = "PyTorch-compile" if compile_mode else "PyTorch"
    return {"framework": label, "dim": "2D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_2d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


# =========================================================================
# 3D Jacobi — 7-point stencil: p_new = (6 neighbors - rhs) / 6
# =========================================================================

import warp as wp

wp.init()


@wp.kernel
def _warp_jacobi3d(
    p: wp.array(dtype=wp.float32),
    p_new: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    nx: wp.int32, ny: wp.int32, nz: wp.int32,
):
    idx = wp.tid()
    iz = idx / (nx * ny)
    rem = idx - iz * nx * ny
    iy = rem / nx
    ix = rem - iy * nx
    if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1 or iz == 0 or iz == nz - 1:
        p_new[idx] = 0.0
        return
    p_new[idx] = (p[idx - 1] + p[idx + 1] + p[idx - nx] + p[idx + nx] +
                  p[idx - nx * ny] + p[idx + nx * ny] - rhs[idx]) * 0.16666667


def bench_3d_warp(n: int) -> dict:
    total = n ** 3
    p = wp.zeros(total, dtype=wp.float32, device="cuda")
    p_new = wp.zeros(total, dtype=wp.float32, device="cuda")
    rhs = wp.array(np.random.uniform(-1, 1, total).astype(np.float32), dtype=wp.float32, device="cuda")

    for _ in range(WARMUP):
        wp.launch(_warp_jacobi3d, dim=total, inputs=[p, p_new, rhs, n, n, n], device="cuda")
    wp.synchronize_device("cuda")

    t0 = time.perf_counter()
    for it in range(_CONFIG["iters_3d"]):
        if it % 2 == 0:
            wp.launch(_warp_jacobi3d, dim=total, inputs=[p, p_new, rhs, n, n, n], device="cuda")
        else:
            wp.launch(_warp_jacobi3d, dim=total, inputs=[p_new, p, rhs, n, n, n], device="cuda")
    wp.synchronize_device("cuda")
    dt = time.perf_counter() - t0

    cells_s = total * _CONFIG["iters_3d"] / dt
    return {"framework": "Warp", "dim": "3D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_3d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


def bench_3d_pytorch(n: int, compile_mode: bool = False) -> dict:
    import torch

    p = torch.zeros(n, n, n, device="cuda", dtype=torch.float32)
    p_new = torch.zeros_like(p)
    rhs = torch.rand(n, n, n, device="cuda", dtype=torch.float32) * 2 - 1

    def step(p_in):
        out = torch.zeros_like(p_in)
        out[1:-1, 1:-1, 1:-1] = (
            p_in[:-2, 1:-1, 1:-1] + p_in[2:, 1:-1, 1:-1] +
            p_in[1:-1, :-2, 1:-1] + p_in[1:-1, 2:, 1:-1] +
            p_in[1:-1, 1:-1, :-2] + p_in[1:-1, 1:-1, 2:] -
            rhs[1:-1, 1:-1, 1:-1]
        ) / 6.0
        return out

    fn = torch.compile(step, mode="max-autotune") if compile_mode else step

    for _ in range(WARMUP):
        p = fn(p)
    _sync_torch()

    _sync_torch()
    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_3d"]):
        p = fn(p)
    _sync_torch()
    dt = time.perf_counter() - t0

    total = n ** 3
    cells_s = total * _CONFIG["iters_3d"] / dt
    label = "PyTorch-compile" if compile_mode else "PyTorch"
    return {"framework": label, "dim": "3D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_3d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


def bench_3d_cupy(n: int) -> dict:
    import cupy as cp

    p = cp.zeros((n, n, n), dtype=cp.float32)
    p_new = cp.zeros_like(p)
    rhs = cp.random.uniform(-1, 1, (n, n, n)).astype(cp.float32)

    for _ in range(WARMUP):
        p_new[1:-1, 1:-1, 1:-1] = (
            p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +
            p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +
            p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:] - rhs[1:-1, 1:-1, 1:-1]
        ) / 6.0
        p, p_new = p_new, p
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for it in range(_CONFIG["iters_3d"]):
        if it % 2 == 0:
            p_new[1:-1, 1:-1, 1:-1] = (
                p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +
                p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +
                p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:] - rhs[1:-1, 1:-1, 1:-1]
            ) / 6.0
        else:
            p[1:-1, 1:-1, 1:-1] = (
                p_new[:-2, 1:-1, 1:-1] + p_new[2:, 1:-1, 1:-1] +
                p_new[1:-1, :-2, 1:-1] + p_new[1:-1, 2:, 1:-1] +
                p_new[1:-1, 1:-1, :-2] + p_new[1:-1, 1:-1, 2:] - rhs[1:-1, 1:-1, 1:-1]
            ) / 6.0
    cp.cuda.Stream.null.synchronize()
    dt = time.perf_counter() - t0

    total = n ** 3
    cells_s = total * _CONFIG["iters_3d"] / dt
    return {"framework": "CuPy", "dim": "3D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_3d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


def bench_3d_jax(n: int) -> dict:
    import jax
    import jax.numpy as jnp

    key = jax.random.PRNGKey(42)
    rhs = jax.random.uniform(key, (n, n, n), minval=-1.0, maxval=1.0, dtype=jnp.float32)

    @jax.jit
    def step(p):
        return p.at[1:-1, 1:-1, 1:-1].set(
            (p[:-2, 1:-1, 1:-1] + p[2:, 1:-1, 1:-1] +
             p[1:-1, :-2, 1:-1] + p[1:-1, 2:, 1:-1] +
             p[1:-1, 1:-1, :-2] + p[1:-1, 1:-1, 2:] - rhs[1:-1, 1:-1, 1:-1]) / 6.0)

    p = jnp.zeros((n, n, n), dtype=jnp.float32)
    for _ in range(10):
        p = step(p)
    p.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_3d"]):
        p = step(p)
    p.block_until_ready()
    dt = time.perf_counter() - t0

    total = n ** 3
    cells_s = total * _CONFIG["iters_3d"] / dt
    return {"framework": "JAX", "dim": "3D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_3d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


def bench_3d_triton(n: int) -> dict:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def _jacobi3d_triton(
        ptr_in, ptr_out,
        NX: tl.constexpr, NY: tl.constexpr, NLAST: tl.constexpr,
    ):
        pid = tl.program_id(0)
        ny = pid // NX
        nx = pid % NX
        if nx == 0 or nx >= NX - 1 or ny == 0 or ny >= NY - 1:
            return
        nz = tl.arange(0, NLAST)
        mask = (nz > 0) & (nz < NLAST - 1)
        idx = (ny * NX + nx) * NLAST + nz
        xp = tl.load(ptr_in + idx + NLAST, mask=mask, other=0.0)
        xm = tl.load(ptr_in + idx - NLAST, mask=mask, other=0.0)
        yp = tl.load(ptr_in + idx + NX * NLAST, mask=mask, other=0.0)
        ym = tl.load(ptr_in + idx - NX * NLAST, mask=mask, other=0.0)
        zp = tl.load(ptr_in + idx + 1, mask=mask, other=0.0)
        zm = tl.load(ptr_in + idx - 1, mask=mask, other=0.0)
        out = (xp + xm + yp + ym + zp + zm) / 6.0
        tl.store(ptr_out + idx, out, mask=mask)

    x = torch.randn(n, n, n, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)

    for _ in range(WARMUP):
        _jacobi3d_triton[(n * n,)](x, y, n, n, n, num_warps=8)
    torch.cuda.synchronize()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(_CONFIG["iters_3d"]):
        _jacobi3d_triton[(n * n,)](x, y, n, n, n, num_warps=8)
        x, y = y, x
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    total = n ** 3
    cells_s = total * _CONFIG["iters_3d"] / dt
    return {"framework": "Triton", "dim": "3D", "grid": n, "ms_total": dt * 1000,
            "ms_iter": dt / _CONFIG["iters_3d"] * 1000, "cells_s": cells_s, "mem_gb": 0.0}


# =========================================================================
# Dispatch tables
# =========================================================================

BENCH_2D = {
    "warp": None,  # no 2D Warp stencil in originals
    "pytorch": lambda n: bench_2d_pytorch(n, compile_mode=False),
    "pytorch-compile": lambda n: bench_2d_pytorch(n, compile_mode=True),
    "cupy": bench_2d_cupy,
    "cupy-raw": bench_2d_cupy_raw,
    "jax": bench_2d_jax,
    "triton": None,  # no 2D Triton in originals
}

BENCH_3D = {
    "warp": bench_3d_warp,
    "pytorch": lambda n: bench_3d_pytorch(n, compile_mode=False),
    "pytorch-compile": lambda n: bench_3d_pytorch(n, compile_mode=True),
    "cupy": bench_3d_cupy,
    "cupy-raw": None,  # 3D raw kernel not in originals
    "jax": bench_3d_jax,
    "triton": bench_3d_triton,
}


def print_table(results: list[dict]):
    if not results:
        return
    hdr = f"  {'Framework':<20s} | {'Dim':>3s} | {'Grid':>8s} | {'ms total':>10s} | {'ms/iter':>9s} | {'cells/s':>16s}"
    sep = f"  {'-'*20}-+-{'-'*3}-+-{'-'*8}-+-{'-'*10}-+-{'-'*9}-+-{'-'*16}"
    print(hdr)
    print(sep)
    for r in results:
        g = f"{r['grid']}^{2 if r['dim']=='2D' else 3}"
        print(f"  {r['framework']:<20s} | {r['dim']:>3s} | {g:>8s} | {r['ms_total']:>9.2f}ms | {r['ms_iter']:>8.4f}ms | {r['cells_s']:>15,.0f}")


def main():
    parser = argparse.ArgumentParser(description="Jacobi stencil benchmark across GPU frameworks")
    parser.add_argument("--frameworks", nargs="+", default=ALL_FRAMEWORKS,
                        choices=ALL_FRAMEWORKS, help="Frameworks to benchmark")
    parser.add_argument("--sizes-2d", nargs="+", type=int, default=DEFAULT_SIZES_2D,
                        help="Grid sizes for 2D benchmarks")
    parser.add_argument("--sizes-3d", nargs="+", type=int, default=DEFAULT_SIZES_3D,
                        help="Grid sizes for 3D benchmarks")
    parser.add_argument("--dims", nargs="+", default=["2d", "3d"],
                        choices=["2d", "3d"], help="Dimensions to run")
    parser.add_argument("--iters-2d", type=int, default=_CONFIG["iters_2d"])
    parser.add_argument("--iters-3d", type=int, default=_CONFIG["iters_3d"])
    args = parser.parse_args()

    _CONFIG["iters_2d"] = args.iters_2d
    _CONFIG["iters_3d"] = args.iters_3d

    import torch
    print("=" * 95)
    print("Jacobi Stencil Benchmark — Unified Cross-Framework Comparison")
    print(f"Device: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    print(f"Frameworks: {', '.join(args.frameworks)}")
    print(f"Dims: {', '.join(args.dims)}")
    print("=" * 95)

    results = []

    if "2d" in args.dims:
        print(f"\n--- 2D Jacobi (5-point stencil, {_CONFIG['iters_2d']} iters) ---")
        for n in args.sizes_2d:
            for fw in args.frameworks:
                fn = BENCH_2D.get(fw)
                if fn is None:
                    continue
                try:
                    _reset_torch_mem()
                    print(f"  {fw} {n}x{n} ...", end=" ", flush=True)
                    r = fn(n)
                    results.append(r)
                    print(f"{r['ms_iter']:.4f} ms/iter  {r['cells_s']:,.0f} cells/s")
                except Exception as e:
                    print(f"ERROR: {e}")

    if "3d" in args.dims:
        print(f"\n--- 3D Jacobi (7-point stencil, {_CONFIG['iters_3d']} iters) ---")
        for n in args.sizes_3d:
            for fw in args.frameworks:
                fn = BENCH_3D.get(fw)
                if fn is None:
                    continue
                try:
                    _reset_torch_mem()
                    print(f"  {fw} {n}^3 ...", end=" ", flush=True)
                    r = fn(n)
                    results.append(r)
                    print(f"{r['ms_iter']:.4f} ms/iter  {r['cells_s']:,.0f} cells/s")
                except Exception as e:
                    print(f"ERROR: {e}")

    print(f"\n{'='*95}")
    print("RESULTS")
    print(f"{'='*95}")
    print_table(results)

    if len(results) > 1:
        print(f"\n{'='*95}")
        print("SPEEDUP vs PyTorch eager (per grid size)")
        print(f"{'='*95}")
        for dim in ["2D", "3D"]:
            dim_results = [r for r in results if r["dim"] == dim]
            grids = sorted(set(r["grid"] for r in dim_results))
            for g in grids:
                grid_results = [r for r in dim_results if r["grid"] == g]
                base = next((r for r in grid_results if r["framework"] == "PyTorch"), None)
                if not base:
                    continue
                gstr = f"{g}^{2 if dim == '2D' else 3}"
                print(f"\n  {dim} {gstr}:")
                for r in sorted(grid_results, key=lambda x: x["ms_iter"]):
                    speedup = base["ms_iter"] / r["ms_iter"]
                    print(f"    {r['framework']:<20s}  {r['ms_iter']:>8.4f} ms/iter  ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
