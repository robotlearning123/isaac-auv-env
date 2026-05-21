"""
GPU Fluid Simulation Primitive Comparison: CuPy vs JAX vs PyTorch
2D Jacobi Pressure Solve on RTX 5090 (32GB, CUDA 12.8)

Measures cold start (incl. JIT), warm run (avg 5), peak memory, throughput.
Grid sizes: 256x256, 512x512, 1024x1024, 2048x2048, 1000 Jacobi iterations.
"""

import gc
import sys
import time
import traceback

import numpy as np
import torch

RESULTS = []


def gpu_mem_gb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def reset_mem():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# CuPy Jacobi solver
# ---------------------------------------------------------------------------
def bench_cupy(grid_sizes, n_iter=1000, n_warm=5):
    try:
        import cupy as cp
    except Exception as e:
        print(f"[CuPy] IMPORT FAILED: {e}")
        for gs in grid_sizes:
            RESULTS.append(("CuPy-RawKernel", gs, "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL"))
            RESULTS.append(("CuPy-cuBLAS", gs, "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL"))
        return

    print(f"[CuPy] version={cp.__version__}")

    # Raw kernel: one Jacobi step
    jacobi_kernel = cp.ElementwiseKernel(
        "raw float32 p, raw float32 rhs",
        "raw float32 p_new",
        """
        int nx = _ind.get()[1];  // this won't work for raw; use manual indexing
        """,
        name="jacobi_raw",
    )

    # Better: use RawKernel with explicit grid
    jacobi_raw = cp.RawKernel(
        r"""
extern "C" __global__
void jacobi_step(const float* p, const float* rhs, float* p_new,
                 int nx, int ny, float dx2_inv) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1) return;
    int idx = j * nx + i;
    p_new[idx] = (p[idx - 1] + p[idx + 1] + p[idx - nx] + p[idx + nx]
                  - rhs[idx]) * (0.25f * dx2_inv / dx2_inv);  // simplifies to avg - rhs/4
    // Actually: p_new = (p_left + p_right + p_top + p_bottom - rhs * dx^2) / 4
    // For dx=1: p_new = (neighbors_sum - rhs) / 4
    p_new[idx] = (p[idx - 1] + p[idx + 1] + p[idx - nx] + p[idx + nx]
                  - rhs[idx]) * 0.25f;
}
""",
        "jacobi_step",
    )

    # Boundary kernel: set edges to 0
    boundary_kernel = cp.RawKernel(
        r"""
extern "C" __global__
void set_boundary(float* p, int nx, int ny, float val) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nx && i >= ny) return;
    if (i < nx) { p[i] = val; p[(ny-1)*nx + i] = val; }
    if (i < ny) { p[i * nx] = val; p[i * nx + nx - 1] = val; }
}
""",
        "set_boundary",
    )

    def cupy_jacobi_raw(nx, ny, n_iter):
        p = cp.zeros((ny, nx), dtype=cp.float32)
        rhs = cp.random.randn(ny, nx, dtype=cp.float32) * 0.01
        p_new = cp.empty_like(p)

        threads = (16, 16)
        blocks_inner = ((nx + 15) // 16, (ny + 15) // 16)
        blocks_bnd = ((max(nx, ny) + 255) // 256,)

        # Cold: include kernel compilation on first call
        t0 = time.perf_counter()
        for _ in range(n_iter):
            jacobi_raw(blocks_inner, threads, (p, rhs, p_new, np.int32(nx), np.int32(ny), np.float32(1.0)))
            boundary_kernel(blocks_bnd, (256,), (p_new, np.int32(nx), np.int32(ny), np.float32(0.0)))
            p, p_new = p_new, p
        cp.cuda.Stream.null.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm runs
        warm_times = []
        for _ in range(n_warm):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                jacobi_raw(blocks_inner, threads, (p, rhs, p_new, np.int32(nx), np.int32(ny), np.float32(1.0)))
                boundary_kernel(blocks_bnd, (256,), (p_new, np.int32(nx), np.int32(ny), np.float32(0.0)))
                p, p_new = p_new, p
            cp.cuda.Stream.null.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.mean(warm_times)
        mem = cp.get_default_memory_pool().used_bytes() / 1e9
        cells_sec = nx * ny * n_iter / (warm_ms / 1000)
        return cold_ms, warm_ms, mem, cells_sec

    # CuPy high-level (array ops, no raw kernel)
    def cupy_jacobi_hl(nx, ny, n_iter):
        p = cp.zeros((ny, nx), dtype=cp.float32)
        rhs = cp.random.randn(ny, nx, dtype=cp.float32) * 0.01

        # Cold
        t0 = time.perf_counter()
        for _ in range(n_iter):
            p_new = cp.zeros_like(p)
            p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                                  p[1:-1, :-2] + p[1:-1, 2:] -
                                  rhs[1:-1, 1:-1]) * 0.25
            p = p_new
        cp.cuda.Stream.null.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm
        warm_times = []
        for _ in range(n_warm):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                p_new = cp.zeros_like(p)
                p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                                      p[1:-1, :-2] + p[1:-1, 2:] -
                                      rhs[1:-1, 1:-1]) * 0.25
                p = p_new
            cp.cuda.Stream.null.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.mean(warm_times)
        mem = cp.get_default_memory_pool().used_bytes() / 1e9
        cells_sec = nx * ny * n_iter / (warm_ms / 1000)
        return cold_ms, warm_ms, mem, cells_sec

    for (nx, ny) in grid_sizes:
        print(f"  CuPy RawKernel {nx}x{ny} ...", end=" ", flush=True)
        try:
            cp.get_default_memory_pool().free_all_blocks()
            c, w, m, t = cupy_jacobi_raw(nx, ny, n_iter)
            RESULTS.append(("CuPy-RawKernel", f"{nx}x{ny}", c, w, m, t))
            print(f"cold={c:.1f}ms warm={w:.1f}ms mem={m:.2f}GB cells/s={t:.2e}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            RESULTS.append(("CuPy-RawKernel", f"{nx}x{ny}", "ERROR", "ERROR", "ERROR", "ERROR"))

        print(f"  CuPy HighLevel {nx}x{ny} ...", end=" ", flush=True)
        try:
            cp.get_default_memory_pool().free_all_blocks()
            c, w, m, t = cupy_jacobi_hl(nx, ny, n_iter)
            RESULTS.append(("CuPy-HighLevel", f"{nx}x{ny}", c, w, m, t))
            print(f"cold={c:.1f}ms warm={w:.1f}ms mem={m:.2f}GB cells/s={t:.2e}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            RESULTS.append(("CuPy-HighLevel", f"{nx}x{ny}", "ERROR", "ERROR", "ERROR", "ERROR"))


# ---------------------------------------------------------------------------
# JAX Jacobi solver
# ---------------------------------------------------------------------------
def bench_jax(grid_sizes, n_iter=1000, n_warm=5):
    try:
        import jax
        import jax.numpy as jnp
    except Exception as e:
        print(f"[JAX] IMPORT FAILED: {e}")
        for gs in grid_sizes:
            RESULTS.append(("JAX-jit", gs, "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL", "IMPORT_FAIL"))
        return

    print(f"[JAX] version={jax.__version__}, devices={jax.devices()}")

    @jax.jit
    def jacobi_step_jax(p, rhs):
        p_new = jnp.zeros_like(p)
        p_new = p_new.at[1:-1, 1:-1].set(
            (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] -
             rhs[1:-1, 1:-1]) * 0.25
        )
        return p_new

    def scan_jacobi(p, rhs, n_iter):
        def body(carry, _):
            p = carry
            p = jacobi_step_jax(p, rhs)
            return p, None
        p, _ = jax.lax.scan(body, p, None, length=n_iter)
        return p

    scan_jit = jax.jit(scan_jacobi, static_argnums=(2,))

    for (nx, ny) in grid_sizes:
        print(f"  JAX {nx}x{ny} ...", end=" ", flush=True)
        try:
            p = jnp.zeros((ny, nx), dtype=jnp.float32)
            rhs = jax.random.normal(jax.random.key(42), (ny, nx), dtype=jnp.float32) * 0.01

            # Cold: first call triggers XLA compilation + execution
            t0 = time.perf_counter()
            _ = scan_jit(p, rhs, n_iter).block_until_ready()
            cold_ms = (time.perf_counter() - t0) * 1000

            # Warm: subsequent calls
            warm_times = []
            for _ in range(n_warm):
                t0 = time.perf_counter()
                _ = scan_jit(p, rhs, n_iter).block_until_ready()
                warm_times.append((time.perf_counter() - t0) * 1000)

            warm_ms = np.mean(warm_times)

            # Memory: estimate from JAX buffers
            mem = (nx * ny * 4 * 4) / 1e9  # rough: p, rhs, p_new, scratch
            try:
                mem_stats = jax.device_memory_stats()
                if mem_stats:
                    mem = mem_stats.get("peak_bytes_in_use", 0) / 1e9
            except Exception:
                pass

            cells_sec = nx * ny * n_iter / (warm_ms / 1000)
            RESULTS.append(("JAX-jit-scan", f"{nx}x{ny}", cold_ms, warm_ms, mem, cells_sec))
            print(f"cold={cold_ms:.1f}ms warm={warm_ms:.1f}ms mem={mem:.2f}GB cells/s={cells_sec:.2e}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            RESULTS.append(("JAX-jit-scan", f"{nx}x{ny}", "ERROR", "ERROR", "ERROR", "ERROR"))


# ---------------------------------------------------------------------------
# PyTorch Jacobi solver
# ---------------------------------------------------------------------------
def bench_pytorch(grid_sizes, n_iter=1000, n_warm=5):
    print(f"[PyTorch] version={torch.__version__}, CUDA={torch.version.cuda}")

    def torch_jacobi_eager(nx, ny, n_iter):
        p = torch.zeros(ny, nx, device="cuda", dtype=torch.float32)
        rhs = torch.randn(ny, nx, device="cuda", dtype=torch.float32) * 0.01

        def step(p):
            p_new = torch.zeros_like(p)
            p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                                  p[1:-1, :-2] + p[1:-1, 2:] -
                                  rhs[1:-1, 1:-1]) * 0.25
            return p_new

        # Cold
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            p = step(p)
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm
        warm_times = []
        for _ in range(n_warm):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                p = step(p)
            torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.mean(warm_times)
        mem = gpu_mem_gb()
        cells_sec = nx * ny * n_iter / (warm_ms / 1000)
        return cold_ms, warm_ms, mem, cells_sec

    def torch_jacobi_compile(nx, ny, n_iter):
        p = torch.zeros(ny, nx, device="cuda", dtype=torch.float32)
        rhs = torch.randn(ny, nx, device="cuda", dtype=torch.float32) * 0.01

        # Use mode="default" (no CUDA Graphs) to avoid feedback-loop issues
        @torch.compile(mode="default")
        def step_compile(p):
            p_new = torch.zeros_like(p)
            p_new[1:-1, 1:-1] = (p[:-2, 1:-1] + p[2:, 1:-1] +
                                  p[1:-1, :-2] + p[1:-1, 2:] -
                                  rhs[1:-1, 1:-1]) * 0.25
            return p_new

        # Cold (includes torch.compile Inductor warmup)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iter):
            p = step_compile(p)
        torch.cuda.synchronize()
        cold_ms = (time.perf_counter() - t0) * 1000

        # Warm
        warm_times = []
        for _ in range(n_warm):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_iter):
                p = step_compile(p)
            torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - t0) * 1000)

        warm_ms = np.mean(warm_times)
        mem = gpu_mem_gb()
        cells_sec = nx * ny * n_iter / (warm_ms / 1000)
        return cold_ms, warm_ms, mem, cells_sec

    for (nx, ny) in grid_sizes:
        reset_mem()
        print(f"  PyTorch-eager {nx}x{ny} ...", end=" ", flush=True)
        try:
            c, w, m, t = torch_jacobi_eager(nx, ny, n_iter)
            RESULTS.append(("PyTorch-eager", f"{nx}x{ny}", c, w, m, t))
            print(f"cold={c:.1f}ms warm={w:.1f}ms mem={m:.2f}GB cells/s={t:.2e}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            RESULTS.append(("PyTorch-eager", f"{nx}x{ny}", "ERROR", "ERROR", "ERROR", "ERROR"))

        reset_mem()
        print(f"  PyTorch-compile {nx}x{ny} ...", end=" ", flush=True)
        try:
            c, w, m, t = torch_jacobi_compile(nx, ny, n_iter)
            RESULTS.append(("PyTorch-compile", f"{nx}x{ny}", c, w, m, t))
            print(f"cold={c:.1f}ms warm={w:.1f}ms mem={m:.2f}GB cells/s={t:.2e}")
        except Exception as e:
            print(f"ERROR: {e}")
            traceback.print_exc()
            RESULTS.append(("PyTorch-compile", f"{nx}x{ny}", "ERROR", "ERROR", "ERROR", "ERROR"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    grid_sizes = [(256, 256), (512, 512), (1024, 1024), (2048, 2048)]
    n_iter = 1000
    n_warm = 5

    print("=" * 80)
    print("GPU Fluid Kernel Comparison: 2D Jacobi Pressure Solve")
    print(f"Grid sizes: {grid_sizes}, iterations: {n_iter}, warm runs: {n_warm}")
    print(f"Device: {torch.cuda.get_device_name(0)}, CUDA {torch.version.cuda}")
    print("=" * 80)

    bench_cupy(grid_sizes, n_iter, n_warm)
    bench_jax(grid_sizes, n_iter, n_warm)
    bench_pytorch(grid_sizes, n_iter, n_warm)

    # Print comparison table
    print("\n" + "=" * 100)
    print(f"{'Framework':<22} {'Grid':<12} {'Cold(ms)':>12} {'Warm(ms)':>12} {'Mem(GB)':>10} {'Cells/sec':>15}")
    print("-" * 100)
    for r in RESULTS:
        if isinstance(r[2], str):
            print(f"{r[0]:<22} {r[1]:<12} {r[2]:>12} {r[3]:>12} {r[4]:>10} {r[5]:>15}")
        else:
            print(f"{r[0]:<22} {r[1]:<12} {r[2]:>12.1f} {r[3]:>12.1f} {r[4]:>10.2f} {r[5]:>15.2e}")
    print("=" * 100)


if __name__ == "__main__":
    main()
