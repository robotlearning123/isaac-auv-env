"""Sparse Linear Solve Benchmark (Part 3).

Sparse Laplacian (5-point stencil) on NxN grid.
Solvers: CuPy sparse direct, PyTorch sparse, SciPy (CPU baseline).
Grids: 64x64, 128x128, 256x256, 512x512.
"""

import time
import numpy as np
import cupy as cp
import torch
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve as scipy_spsolve

N_WARMUP = 2
N_TRIALS = 5


def build_laplacian_2d(N):
    """Build 5-point Laplacian on NxN grid with Dirichlet BC (zero)."""
    n_total = N * N
    h = 1.0 / (N + 1)
    h2_inv = 1.0 / (h * h)

    # Main diagonal: -4/h^2
    diag = -4.0 * h2_inv * np.ones(n_total)
    # Off-diagonal: 1/h^2
    off_x = h2_inv * np.ones(n_total - 1)
    off_y = h2_inv * np.ones(n_total - N)

    # Zero out connections across row boundaries for x-stencil
    for i in range(1, N):
        off_x[i * N - 1] = 0.0

    A = sp.diags(
        [diag, off_x, off_x, off_y, off_y],
        [0, 1, -1, N, -N],
        shape=(n_total, n_total),
        format="csr",
    )
    return A


def rhs_2d(N):
    """RHS from random RHS (realistic pressure Poisson scenario)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(N * N)


def bench_scipy(A, b):
    """SciPy CPU sparse direct solve."""
    for _ in range(N_WARMUP):
        _ = scipy_spsolve(A, b)

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        x = scipy_spsolve(A, b)
        times.append(time.perf_counter() - t0)

    avg = np.mean(times)
    resid = np.linalg.norm(A @ x - b)
    return avg, resid


def bench_cupy_sparse(A, b):
    """CuPy GPU sparse direct solve."""
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csplg

    A_gpu = csp.csr_matrix(A)
    b_gpu = cp.asarray(b)

    # Warmup
    for _ in range(N_WARMUP):
        _ = csplg.spsolve(A_gpu, b_gpu)
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        x = csplg.spsolve(A_gpu, b_gpu)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)

    avg = np.mean(times)
    resid = cp.linalg.norm(A_gpu @ x - b_gpu).get()
    del A_gpu, b_gpu, x
    cp.get_default_memory_pool().free_all_blocks()
    return avg, resid


def bench_cupy_cg(A, b):
    """CuPy sparse CG solve."""
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as csplg

    A_gpu = csp.csr_matrix(A)
    b_gpu = cp.asarray(b)

    # Warmup
    for _ in range(N_WARMUP):
        _ = csplg.cg(A_gpu, b_gpu, atol=1e-10, rtol=1e-10)
    cp.cuda.Stream.null.synchronize()

    times = []
    iters_list = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        x, info = csplg.cg(A_gpu, b_gpu, atol=1e-10, rtol=1e-10)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
        iters_list.append(info)

    avg = np.mean(times)
    avg_iters = np.mean(iters_list)
    resid = cp.linalg.norm(A_gpu @ x - b_gpu).get()
    del A_gpu, b_gpu, x
    cp.get_default_memory_pool().free_all_blocks()
    return avg, resid, avg_iters


def bench_pytorch_sparse(A, b):
    """PyTorch sparse solve (via sparse-dense matmul iterative approach)."""
    # PyTorch doesn't have a direct sparse solver, so we implement CG
    A_coo = A.tocoo()
    indices = torch.tensor(np.vstack([A_coo.row, A_coo.col]), dtype=torch.int64)
    values = torch.tensor(A_coo.data, dtype=torch.float64)
    A_th = torch.sparse_coo_tensor(indices, values, size=A.shape, device="cuda").to_sparse_csr()

    b_th = torch.tensor(b, dtype=torch.float64, device="cuda")

    # CG solver
    def torch_sparse_cg(A, b, max_iter=1000, tol=1e-8):
        x = torch.zeros_like(b)
        r = b - torch.sparse.mm(A, x.unsqueeze(1)).squeeze(1)
        p = r.clone()
        rs_old = torch.dot(r, r)
        b_norm = torch.linalg.norm(b)
        for i in range(max_iter):
            Ap = torch.sparse.mm(A, p.unsqueeze(1)).squeeze(1)
            alpha = rs_old / torch.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) / b_norm < tol:
                return x, i + 1
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x, max_iter

    # Warmup
    for _ in range(N_WARMUP):
        _ = torch_sparse_cg(A_th, b_th)
    torch.cuda.synchronize()

    times = []
    iters_list = []
    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        x, n_iters = torch_sparse_cg(A_th, b_th)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        iters_list.append(n_iters)

    avg = np.mean(times)
    avg_iters = np.mean(iters_list)
    resid = torch.linalg.norm(
        torch.sparse.mm(A_th, x.unsqueeze(1)).squeeze(1) - b_th
    ).item()
    del A_th, b_th, x
    torch.cuda.empty_cache()
    return avg, resid, avg_iters


def bench_setup_time(N):
    """Measure matrix construction time."""
    t0 = time.perf_counter()
    A = build_laplacian_2d(N)
    t_build = time.perf_counter() - t0

    import cupyx.scipy.sparse as csp

    t0 = time.perf_counter()
    A_gpu = csp.csr_matrix(A)
    cp.cuda.Stream.null.synchronize()
    t_transfer = time.perf_counter() - t0

    del A_gpu
    cp.get_default_memory_pool().free_all_blocks()
    return t_build, t_transfer, A


def main():
    print("=" * 90)
    print("PART 3: Sparse Linear Solve Benchmark (Pressure Poisson Equation)")
    print("=" * 90)
    print(f"CuPy {cp.__version__}, PyTorch {torch.__version__}")
    print(f"5-point Laplacian stencil, Dirichlet BC")
    print()

    grids = [64, 128, 256, 512]

    # Setup times
    print(f"{'Grid':>8} {'Unknowns':>10} {'NNZ':>12} {'Build(s)':>10} {'GPU transfer':>14}")
    print("-" * 60)
    for N in grids:
        t_build, t_transfer, A = bench_setup_time(N)
        print(f"{N:>8} {N*N:>10} {A.nnz:>12} {t_build:>10.4f} {t_transfer:>13.4f}s")

    print()

    # Solve benchmarks
    print(f"{'Grid':>8} | {'SciPy Direct':>30} | {'CuPy Direct':>30} | {'CuPy CG':>36} | {'PyTorch CG':>36}")
    print(f"{'':>8} | {'Time':>12} {'Residual':>14} | {'Time':>12} {'Residual':>14} | "
          f"{'Time':>12} {'Residual':>14} {'Iters':>6} | {'Time':>12} {'Residual':>14} {'Iters':>6}")
    print("-" * 160)

    for N in grids:
        A = build_laplacian_2d(N)
        b = rhs_2d(N)

        # SciPy
        t_scipy, res_scipy = bench_scipy(A, b)

        # CuPy direct
        t_cupy, res_cupy = bench_cupy_sparse(A, b)

        # CuPy CG
        t_cupy_cg, res_cupy_cg, iters_cupy_cg = bench_cupy_cg(A, b)

        # PyTorch CG
        t_torch, res_torch, iters_torch = bench_pytorch_sparse(A, b)

        print(f"{N:>8} | {t_scipy:>11.4f}s {res_scipy:>13.3e} | "
              f"{t_cupy:>11.4f}s {res_cupy:>13.3e} | "
              f"{t_cupy_cg:>11.4f}s {res_cupy_cg:>13.3e} {int(iters_cupy_cg):>5} | "
              f"{t_torch:>11.4f}s {res_torch:>13.3e} {int(iters_torch):>5}")

    print()
    print("Benchmark complete.")


if __name__ == "__main__":
    main()
