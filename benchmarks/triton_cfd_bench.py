"""Triton vs Warp vs PyTorch 3D Jacobi stencil benchmark (128^3)."""

import time
import numpy as np
import torch
import triton
import triton.language as tl
import warp as wp

wp.init()

N = 128
ITERS = 20
WARMUP = 5


# ---- Triton kernel ----
@triton.jit
def jacobi3d_triton(
    ptr_in,
    ptr_out,
    NX: tl.constexpr,
    NY: tl.constexpr,
    NLAST: tl.constexpr,
):
    pid = tl.program_id(0)
    ny = pid // NX
    nx = pid % NX
    if nx == 0 or nx >= NX - 1 or ny == 0 or ny >= NY - 1:
        return
    nz = tl.arange(0, NLAST)
    mask = (nz > 0) & (nz < NLAST - 1)
    idx = (ny * NX + nx) * NLAST + nz
    c = tl.load(ptr_in + idx, mask=mask, other=0.0)
    xp = tl.load(ptr_in + idx + NLAST, mask=mask, other=0.0)
    xm = tl.load(ptr_in + idx - NLAST, mask=mask, other=0.0)
    yp = tl.load(ptr_in + idx + NX * NLAST, mask=mask, other=0.0)
    ym = tl.load(ptr_in + idx - NX * NLAST, mask=mask, other=0.0)
    zp = tl.load(ptr_in + idx + 1, mask=mask, other=0.0)
    zm = tl.load(ptr_in + idx - 1, mask=mask, other=0.0)
    out = (xp + xm + yp + ym + zp + zm + c) / 7.0
    tl.store(ptr_out + idx, out, mask=mask)


def bench_triton():
    x = torch.randn(N, N, N, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)

    for _ in range(WARMUP):
        jacobi3d_triton[(N * N,)](x, y, N, N, N, num_warps=8)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        jacobi3d_triton[(N * N,)](x, y, N, N, N, num_warps=8)
        x, y = y, x
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    return ms


# ---- PyTorch eager ----
def jacobi3d_torch(x: torch.Tensor) -> torch.Tensor:
    return (
        x[2:, 1:-1, 1:-1]
        + x[:-2, 1:-1, 1:-1]
        + x[1:-1, 2:, 1:-1]
        + x[1:-1, :-2, 1:-1]
        + x[1:-1, 1:-1, 2:]
        + x[1:-1, 1:-1, :-2]
        + x[1:-1, 1:-1, 1:-1]
    ) / 7.0


def bench_torch_eager():
    x = torch.randn(N, N, N, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    for _ in range(WARMUP):
        y[1:-1, 1:-1, 1:-1] = jacobi3d_torch(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        y[1:-1, 1:-1, 1:-1] = jacobi3d_torch(x)
        x, y = y, x
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    return ms


# ---- torch.compile ----
_compiled_jacobi = None


def bench_torch_compile():
    global _compiled_jacobi
    if _compiled_jacobi is None:
        _compiled_jacobi = torch.compile(jacobi3d_torch, mode="max-autotune")
    x = torch.randn(N, N, N, device="cuda", dtype=torch.float32)
    y = torch.zeros_like(x)
    for _ in range(WARMUP):
        y[1:-1, 1:-1, 1:-1] = _compiled_jacobi(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        y[1:-1, 1:-1, 1:-1] = _compiled_jacobi(x)
        x, y = y, x
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    return ms


# ---- Warp kernel ----
@wp.kernel
def jacobi3d_warp(
    x: wp.array3d(dtype=wp.float32),
    y: wp.array3d(dtype=wp.float32),
    nx: int,
    ny: int,
    nz: int,
):
    i, j, k = wp.tid()
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1 or k < 1 or k >= nz - 1:
        return
    y[i, j, k] = (
        x[i + 1, j, k]
        + x[i - 1, j, k]
        + x[i, j + 1, k]
        + x[i, j - 1, k]
        + x[i, j, k + 1]
        + x[i, j, k - 1]
        + x[i, j, k]
    ) / 7.0


def bench_warp():
    x_np = np.random.randn(N, N, N).astype(np.float32)
    y_np = np.zeros((N, N, N), dtype=np.float32)
    x_wp = wp.from_numpy(x_np, dtype=wp.float32)
    y_wp = wp.from_numpy(y_np, dtype=wp.float32)

    for _ in range(WARMUP):
        wp.launch(jacobi3d_warp, dim=(N, N, N), inputs=[x_wp, y_wp, N, N, N])
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(ITERS):
        wp.launch(jacobi3d_warp, dim=(N, N, N), inputs=[x_wp, y_wp, N, N, N])
        x_wp, y_wp = y_wp, x_wp
    wp.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    return ms


# ---- Modulus FNO inference (physics-ML baseline) ----
def bench_modulus_fno():
    from modulus.models.fno.fno import FNO
    from modulus.models.module import Module
    from modulus.models.meta import ModelMetaData

    class Dec(Module):
        def __init__(self):
            super().__init__(meta=ModelMetaData(var_dim=3))
            self.c1 = torch.nn.Conv2d(32, 16, 1)
            self.c2 = torch.nn.Conv2d(16, 3, 1)
            self.a = torch.nn.GELU()

        def forward(self, x):
            return self.c2(self.a(self.c1(x)))

    model = FNO(
        decoder_net=Dec(),
        in_channels=3,
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=16,
    ).cuda()

    x = torch.randn(2, 3, 64, 64, device="cuda")
    for _ in range(WARMUP):
        model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        model(x)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / ITERS * 1000
    params = sum(p.numel() for p in model.parameters())
    return ms, params


def main():
    print(f"=== 3D Jacobi Stencil Benchmark (N={N}, {ITERS} iters) ===")
    print()

    results = {}

    print("Triton 3.6 ...", end=" ", flush=True)
    results["Triton"] = bench_triton()
    print(f"{results['Triton']:.3f} ms/iter")

    print("PyTorch eager ...", end=" ", flush=True)
    results["PyTorch_eager"] = bench_torch_eager()
    print(f"{results['PyTorch_eager']:.3f} ms/iter")

    print("torch.compile ...", end=" ", flush=True)
    results["torch_compile"] = bench_torch_compile()
    print(f"{results['torch_compile']:.3f} ms/iter")

    print("Warp 1.9 ...", end=" ", flush=True)
    results["Warp"] = bench_warp()
    print(f"{results['Warp']:.3f} ms/iter")

    print()
    print("=== Modulus FNO (physics-ML baseline) ===")
    try:
        ms, params = bench_modulus_fno()
        print(f"Modulus FNO inference: {ms:.3f} ms/iter (4.2M params, batch=2, 64x64)")
    except Exception as e:
        print(f"Modulus FNO: FAILED ({e})")

    print()
    print("=== Speedup vs PyTorch eager ===")
    base = results["PyTorch_eager"]
    for name, ms in results.items():
        speedup = base / ms
        print(f"  {name:20s}: {ms:8.3f} ms/iter  ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
