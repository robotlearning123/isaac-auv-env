"""RTX 5090 GPU baseline benchmarks — memory bandwidth, compute, kernel launch, allocation speed."""

import time
import subprocess

import cupy as cp
import numpy as np
import torch
import warp as wp

wp.init()


def _separator(n: int = 80) -> None:
    print("-" * n)


# ---------------------------------------------------------------------------
# 0. GPU specs
# ---------------------------------------------------------------------------
def gpu_specs() -> dict:
    props = torch.cuda.get_device_properties(0)

    # Clock rate from torch (in kHz -> MHz)
    clock_mhz = props.clock_rate / 1000.0
    mem_clock_mhz = props.memory_clock_rate / 1000.0

    specs = {
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "sm_count": props.multi_processor_count,
        "gpu_clock_mhz": clock_mhz,
        "mem_clock_mhz": mem_clock_mhz,
        "memory_bus_width_bits": props.memory_bus_width,
        "total_memory_gb": props.total_memory / (1024**3),
        "l2_cache_mb": props.L2_cache_size / (1024**2),
    }
    # Power limit from nvidia-smi
    try:
        smi_power = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.default_limit", "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        specs["power_limit_w"] = float(smi_power)
    except Exception:
        specs["power_limit_w"] = "N/A"

    # Theoretical bandwidth: bus_width_bytes * mem_clock * 2 (DDR)
    bus_bytes = props.memory_bus_width // 8
    specs["theoretical_bw_gbs"] = bus_bytes * mem_clock_mhz * 2 * 1e6 / 1e9

    # Theoretical bandwidth: bus_width/8 * 2 (DDR) * clock
    # But we'll just report measured vs theoretical from nvidia-smi
    return specs


# ---------------------------------------------------------------------------
# 1. Memory bandwidth (CuPy)
# ---------------------------------------------------------------------------
def benchmark_memory_bandwidth() -> dict:
    size = 256 * 1024 * 1024  # 256M elements = 1 GB for float32
    dtype = cp.float32
    elem_bytes = 4

    # Warmup
    _ = cp.zeros(size, dtype=dtype)
    cp.cuda.runtime.deviceSynchronize()

    host_arr = np.random.randn(size).astype(np.float32)

    results = {}

    # host -> device
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    d_arr = cp.asarray(host_arr)
    end.record()
    end.synchronize()
    h2d_ms = cp.cuda.get_elapsed_time(start, end)
    h2d_gb = (size * elem_bytes) / (h2d_ms / 1000) / 1e9
    results["h2d_ms"] = h2d_ms
    results["h2d_gbs"] = h2d_gb

    # device -> host
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    _ = d_arr.get()
    end.record()
    end.synchronize()
    d2h_ms = cp.cuda.get_elapsed_time(start, end)
    d2h_gb = (size * elem_bytes) / (d2h_ms / 1000) / 1e9
    results["d2h_ms"] = d2h_ms
    results["d2h_gbs"] = d2h_gb

    # device -> device
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    d_dst = d_arr.copy()
    end.record()
    end.synchronize()
    d2d_ms = cp.cuda.get_elapsed_time(start, end)
    d2d_gb = (size * elem_bytes) / (d2d_ms / 1000) / 1e9
    results["d2d_ms"] = d2d_ms
    results["d2d_gbs"] = d2d_gb

    del d_arr, d_dst
    return results


# ---------------------------------------------------------------------------
# 2. Compute throughput (PyTorch)
# ---------------------------------------------------------------------------
def benchmark_compute() -> dict:
    results = {}
    device = torch.device("cuda")

    matmul_configs = [
        ("FP32", torch.float32, False),
        ("TF32", torch.float32, True),
        ("FP16", torch.float16, False),
    ]
    for dtype_name, torch_dtype, use_tf32 in matmul_configs:
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        torch.backends.cudnn.allow_tf32 = use_tf32

        # Warmup
        _ = torch.randn(512, 512, dtype=torch_dtype, device=device) @ torch.randn(512, 512, dtype=torch_dtype, device=device)
        torch.cuda.synchronize()

        n = 4096
        a = torch.randn(n, n, dtype=torch_dtype, device=device)
        b = torch.randn(n, n, dtype=torch_dtype, device=device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        iters = 20
        start.record()
        for _ in range(iters):
            c = a @ b
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end) / iters
        flops = 2 * (n ** 3)
        tflops = flops / (elapsed_ms / 1000) / 1e12

        results[f"{dtype_name}_ms"] = elapsed_ms
        results[f"{dtype_name}_tflops"] = tflops

        torch.backends.cuda.matmul.allow_tf32 = prev_tf32
        del a, b, c

    return results


# ---------------------------------------------------------------------------
# 3. Kernel launch overhead (Warp)
# ---------------------------------------------------------------------------
def benchmark_kernel_launch() -> dict:
    @wp.kernel
    def trivial_add(arr: wp.array(dtype=wp.float32)):
        arr[0] = arr[0] + 1.0

    arr = wp.zeros(1, dtype=wp.float32)
    # Warmup
    for _ in range(100):
        wp.launch(trivial_add, dim=1, inputs=[arr])
    wp.synchronize_device()

    n_launches = 10000
    start = time.perf_counter()
    for _ in range(n_launches):
        wp.launch(trivial_add, dim=1, inputs=[arr])
    wp.synchronize_device()
    elapsed = time.perf_counter() - start

    us_per_launch = (elapsed / n_launches) * 1e6
    return {"total_s": elapsed, "n_launches": n_launches, "us_per_launch": us_per_launch}


# ---------------------------------------------------------------------------
# 4. Memory allocation speed
# ---------------------------------------------------------------------------
def benchmark_allocation() -> dict:
    results = {}
    n_allocs = 1000
    alloc_bytes = 1 * 1024 * 1024  # 1 MB
    n_elements = alloc_bytes // 4   # float32

    # CuPy
    cp.cuda.runtime.deviceSynchronize()
    start = time.perf_counter()
    for _ in range(n_allocs):
        a = cp.empty(n_elements, dtype=cp.float32)
        cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - start
    results["cupy_us"] = (elapsed / n_allocs) * 1e6
    del a

    # PyTorch
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_allocs):
        a = torch.empty(n_elements, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    results["torch_us"] = (elapsed / n_allocs) * 1e6
    del a

    # Warp
    wp.synchronize_device()
    start = time.perf_counter()
    for _ in range(n_allocs):
        a = wp.empty(n=n_elements, dtype=wp.float32)
        wp.synchronize_device()
    elapsed = time.perf_counter() - start
    results["warp_us"] = (elapsed / n_allocs) * 1e6
    del a

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("RTX 5090 GPU Baseline Benchmarks")
    print("=" * 80)

    # 0. GPU specs
    print("\n[0] GPU Specifications")
    _separator()
    specs = gpu_specs()
    for k, v in specs.items():
        print(f"  {k:30s} {v}")

    # 1. Memory bandwidth
    print("\n[1] Memory Bandwidth (1 GB arrays, float32)")
    _separator()
    bw = benchmark_memory_bandwidth()
    print(f"  Host   -> Device   : {bw['h2d_ms']:8.2f} ms  | {bw['h2d_gbs']:8.2f} GB/s")
    print(f"  Device -> Host     : {bw['d2h_ms']:8.2f} ms  | {bw['d2h_gbs']:8.2f} GB/s")
    print(f"  Device -> Device   : {bw['d2d_ms']:8.2f} ms  | {bw['d2d_gbs']:8.2f} GB/s")

    # 2. Compute
    print("\n[2] Compute Throughput (4096x4096 matmul)")
    _separator()
    comp = benchmark_compute()
    for dtype_name in ["FP32", "TF32", "FP16"]:
        ms_key = f"{dtype_name}_ms"
        tf_key = f"{dtype_name}_tflops"
        print(f"  {dtype_name}: {comp[ms_key]:8.2f} ms  | {comp[tf_key]:8.2f} TFLOPS")

    # 3. Kernel launch
    print("\n[3] Kernel Launch Overhead (Warp, trivial kernel)")
    _separator()
    kl = benchmark_kernel_launch()
    print(f"  Total launches  : {kl['n_launches']}")
    print(f"  Total time      : {kl['total_s']:.3f} s")
    print(f"  Per launch      : {kl['us_per_launch']:.2f} us")

    # 4. Allocation
    print("\n[4] Memory Allocation Speed (1000x 1MB, float32)")
    _separator()
    alloc = benchmark_allocation()
    print(f"  CuPy  : {alloc['cupy_us']:8.2f} us/alloc")
    print(f"  Torch : {alloc['torch_us']:8.2f} us/alloc")
    print(f"  Warp  : {alloc['warp_us']:8.2f} us/alloc")

    print("\n" + "=" * 80)
    print("Done.")
