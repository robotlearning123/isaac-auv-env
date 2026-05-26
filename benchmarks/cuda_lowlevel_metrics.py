"""L0: CUDA low-level kernel metrics — RTX 5090."""
import time

import torch
import warp as wp

if not torch.cuda.is_available():
    raise SystemExit("CUDA required — no GPU detected.")
wp.init()

N_ELEM = 1_000_000
N_LAUNCH = 10000
N_ITER = 1000


@wp.kernel
def empty_kernel(x: wp.array(dtype=float)):
    pass


@wp.kernel
def add_kernel(
    a: wp.array(dtype=float),
    b: wp.array(dtype=float),
    c: wp.array(dtype=float),
):
    i = wp.tid()
    c[i] = a[i] + b[i]


def bench_empty_launch():
    x = wp.zeros(1, dtype=float, device="cuda:0")
    for _ in range(100):
        wp.launch(empty_kernel, dim=1, inputs=[x], device="cuda:0")
    wp.synchronize()

    t0 = time.perf_counter()
    for _ in range(N_LAUNCH):
        wp.launch(empty_kernel, dim=1, inputs=[x], device="cuda:0")
    wp.synchronize()
    us = (time.perf_counter() - t0) / N_LAUNCH * 1e6
    return us


def bench_elementwise():
    wa = wp.array(range(N_ELEM), dtype=float, device="cuda:0")
    wb = wp.array(range(N_ELEM), dtype=float, device="cuda:0")
    wc = wp.zeros(N_ELEM, dtype=float, device="cuda:0")
    ta = torch.randn(N_ELEM, device="cuda:0")
    tb = torch.randn(N_ELEM, device="cuda:0")

    # Warp warmup + bench
    for _ in range(10):
        wp.launch(add_kernel, dim=N_ELEM, inputs=[wa, wb, wc], device="cuda:0")
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        wp.launch(add_kernel, dim=N_ELEM, inputs=[wa, wb, wc], device="cuda:0")
    wp.synchronize()
    warp_us = (time.perf_counter() - t0) / N_ITER * 1e6

    # PyTorch warmup + bench
    for _ in range(10):
        _ = ta + tb
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = ta + tb
    torch.cuda.synchronize()
    torch_us = (time.perf_counter() - t0) / N_ITER * 1e6

    return warp_us, torch_us


def bench_d2d_bandwidth():
    results = []
    for size_mb in [1, 10, 100]:
        n_floats = size_mb * 1024 * 1024 // 4
        a = torch.randn(n_floats, device="cuda:0")
        b = torch.empty_like(a)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            b.copy_(a)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / 100
        bw = size_mb * 2 / elapsed / 1024  # GB/s (read + write)
        results.append((size_mb, bw))
    return results


if __name__ == "__main__":
    print("L0: CUDA Kernel Metrics — RTX 5090")
    print("=" * 60)

    launch_us = bench_empty_launch()
    print(f"  Empty kernel launch: {launch_us:.2f} us")

    warp_us, torch_us = bench_elementwise()
    print(
        f"  Elementwise add (1M): Warp={warp_us:.1f}us, "
        f"PyTorch={torch_us:.1f}us, ratio={warp_us / torch_us:.2f}x"
    )

    for size_mb, bw in bench_d2d_bandwidth():
        print(f"  D2D copy {size_mb:>3d}MB: {bw:.1f} GB/s")

    print("=" * 60)
