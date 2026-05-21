"""CuPy RawKernel D2Q9 LBM benchmark.

Grid sizes: 512, 1024, 2048, 4096
Compares with PyTorch LBM.
"""

import time
import numpy as np
import cupy as cp

# D2Q9 lattice velocities and weights
CX = cp.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=cp.float32)
CY = cp.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=cp.float32)
W = cp.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=cp.float32)

OPP = [0, 3, 4, 1, 2, 7, 8, 5, 6]  # opposite direction indices

ITERS = 200
WARMUP = 20

# CuPy RawKernel: collide + stream (fused)
lbm_kernel_code = r"""
extern "C" __global__
void lbm_collide_stream(
    const float* f_in,
    float* f_out,
    float* rho,
    float* ux,
    float* uy,
    const float* cx,
    const float* cy,
    const float* w,
    const int nx,
    const int ny,
    const float omega
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    // Compute macroscopic quantities
    float _rho = 0.0f, _ux = 0.0f, _uy = 0.0f;
    for (int q = 0; q < 9; q++) {
        float fq = f_in[q * nx * ny + y * nx + x];
        _rho += fq;
        _ux += cx[q] * fq;
        _uy += cy[q] * fq;
    }
    _ux /= _rho;
    _uy /= _rho;

    rho[y * nx + x] = _rho;
    ux[y * nx + x] = _ux;
    uy[y * nx + x] = _uy;

    // Collide (BGK)
    float feq[9];
    float usq = _ux * _ux + _uy * _uy;
    for (int q = 0; q < 9; q++) {
        float cu = cx[q] * _ux + cy[q] * _uy;
        feq[q] = w[q] * _rho * (1.0f + 3.0f * cu + 4.5f * cu * cu - 1.5f * usq);
    }

    // Stream (pull scheme with bounce-back on boundaries)
    int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    for (int q = 0; q < 9; q++) {
        int sx = x - (int)cx[q];
        int sy = y - (int)cy[q];
        float fnew = f_in[q * nx * ny + y * nx + x] + omega * (feq[q] - f_in[q * nx * ny + y * nx + x]);

        if (sx >= 0 && sx < nx && sy >= 0 && sy < ny) {
            f_out[q * nx * ny + sy * nx + sx] = fnew;
        } else {
            // Bounce-back
            f_out[opp[q] * nx * ny + y * nx + x] = fnew;
        }
    }
}
"""

lbm_kernel = cp.RawKernel(lbm_kernel_code, "lbm_collide_stream")


def bench_cupy_lbm(nx, ny):
    """Benchmark CuPy RawKernel D2Q9 LBM."""
    f = cp.random.uniform(0.1, 0.2, (9, ny, nx), dtype=cp.float32)
    # Initialize to near-equilibrium
    rho_arr = cp.ones((ny, nx), dtype=cp.float32)
    ux_arr = cp.zeros((ny, nx), dtype=cp.float32)
    uy_arr = cp.zeros((ny, nx), dtype=cp.float32)
    for q in range(9):
        f[q] = W[q].item()

    f_out = cp.zeros_like(f)
    omega = 1.8

    block = (16, 16, 1)
    grid = ((nx + 15) // 16, (ny + 15) // 16, 1)

    # Warmup
    for _ in range(WARMUP):
        lbm_kernel(grid, block, (f, f_out, rho_arr, ux_arr, uy_arr,
                                  CX, CY, W, nx, ny, omega))
        f, f_out = f_out, f
    cp.cuda.Stream.null.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        lbm_kernel(grid, block, (f, f_out, rho_arr, ux_arr, uy_arr,
                                  CX, CY, W, nx, ny, omega))
        f, f_out = f_out, f
    cp.cuda.Stream.null.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000

    # MLUPS = million lattice updates per second
    mlups = (nx * ny * ITERS) / (time.perf_counter() - t0 + (ms * ITERS / 1000)) / 1e6
    # Recalculate properly
    total_lattice_updates = nx * ny * ITERS
    elapsed_s = ms * ITERS / 1000
    mlups = total_lattice_updates / elapsed_s / 1e6
    return ms, mlups


def bench_pytorch_lbm(nx, ny):
    """PyTorch D2Q9 LBM baseline."""
    import torch

    device = "cuda"
    f = torch.zeros(9, ny, nx, device=device, dtype=torch.float32)
    cx_t = torch.tensor([0, 1, 0, -1, 0, 1, -1, -1, 1], device=device, dtype=torch.float32)
    cy_t = torch.tensor([0, 0, 1, 0, -1, 1, 1, -1, -1], device=device, dtype=torch.float32)
    w_t = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36],
                       device=device, dtype=torch.float32)
    omega = 1.8

    # Initialize equilibrium
    rho = torch.ones(ny, nx, device=device, dtype=torch.float32)
    for q in range(9):
        f[q] = w_t[q]

    def lbm_step(f):
        rho = f.sum(dim=0)
        ux = (f * cx_t.view(9, 1, 1)).sum(dim=0) / rho
        uy = (f * cy_t.view(9, 1, 1)).sum(dim=0) / rho

        usq = ux * ux + uy * uy
        feq = torch.zeros_like(f)
        for q in range(9):
            cu = cx_t[q] * ux + cy_t[q] * uy
            feq[q] = w_t[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)

        f_post = f - omega * (f - feq)

        # Stream
        f_new = torch.zeros_like(f)
        for q in range(9):
            sx = int(cx_t[q].item())
            sy = int(cy_t[q].item())
            f_new[q] = torch.roll(torch.roll(f_post[q], sx, dims=1), sy, dims=0)

        return f_new

    for _ in range(WARMUP):
        f = lbm_step(f)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        f = lbm_step(f)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000

    total_lattice_updates = nx * ny * ITERS
    elapsed_s = ms * ITERS / 1000
    mlups = total_lattice_updates / elapsed_s / 1e6
    return ms, mlups


def main():
    sizes = [512, 1024, 2048, 4096]

    print("=== D2Q9 LBM Benchmark (CuPy RawKernel vs PyTorch) ===")
    print(f"Iters: {ITERS}, Warmup: {WARMUP}")
    print()

    results = []

    for n in sizes:
        print(f"--- Grid {n}x{n} ---")
        print(f"  CuPy RawKernel ...", end=" ", flush=True)
        cupy_ms, cupy_mlups = bench_cupy_lbm(n, n)
        print(f"{cupy_ms:.3f} ms/iter, {cupy_mlups:.0f} MLUPS")

        print(f"  PyTorch eager   ...", end=" ", flush=True)
        torch_ms, torch_mlups = bench_pytorch_lbm(n, n)
        print(f"{torch_ms:.3f} ms/iter, {torch_mlups:.0f} MLUPS")

        speedup = torch_ms / cupy_ms
        print(f"  CuPy speedup: {speedup:.2f}x")
        results.append((n, cupy_ms, cupy_mlups, torch_ms, torch_mlups, speedup))
        print()

    print("=== Summary ===")
    print(f"  {'Grid':>8s} | {'CuPy ms':>10s} | {'CuPy MLUPS':>10s} | {'PyTorch ms':>10s} | {'PyTorch MLUPS':>12s} | {'Speedup':>8s}")
    print(f"  {'-'*8} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*12} | {'-'*8}")
    for n, c_ms, c_ml, t_ms, t_ml, sp in results:
        print(f"  {n:>8d} | {c_ms:>10.3f} | {c_ml:>10.0f} | {t_ms:>10.3f} | {t_ml:>12.0f} | {sp:>7.2f}x")


if __name__ == "__main__":
    main()
