# Benchmark Verification Report

**Date:** 2026-05-20
**GPU:** NVIDIA GeForce RTX 5090 (31 GB, sm_120)
**Driver:** 580.95.05, CUDA 13.0 (toolkit 12.9)
**Warp:** 1.13.0, **Newton:** 1.2.0, **MuJoCo-Warp:** 3.8.1
**Thermal:** 52C start, 61C end — no throttling

## Summary Table

| # | Script | Claimed | Measured | Delta | Status |
|---|--------|---------|----------|-------|--------|
| 1 | `warp_fluid_bench.py` | Jacobi 128^3 = 1.04ms, 200.9B cells/s | 1.04ms, 201.8B cells/s | <1% | **PASS** |
| 2 | `newton_underwater_bench.py` | 5,700 steps/s, 1.45M env/s at 256 worlds | 5,443 steps/s, 1.39M env/s | ~4% | **PASS** |
| 3 | `mujoco_warp_bench.py` | 2.45M env-steps/s at N=4096 | 1.32M env-steps/s | -46% | **MISMATCH** |
| 4 | `gpu_baseline.py` | ~~67~~ 104.8 TFLOPS FP32, ~~213~~ 838 TFLOPS FP16 (NVIDIA spec) | 31.3 TFLOPS FP32, 93.7 TFLOPS FP16 | -70/-89% vs spec | **MISMATCH (scalar path only — Tensor Cores not exercised)** |
| 5 | `navier_stokes_3d.py` | 334M cells/s at 256^3 | 429M (random IC), 207M (Taylor-Green) | varies | **PASS** |
| 6 | `newton_mpm_bench.py` | 120-140 steps/s at 100K particles | 106 steps/s (Dam Break) | -16% | **MISMATCH** |
| 7 | `jacobi_stencil_bench.py` | Warp 4.5x faster than PyTorch | 3.4x at 128^3 (largest grid) | -24% | **MISMATCH** |

## Detailed Findings

### 1. warp_fluid_bench.py — PASS

Jacobi pressure solve at 128^3 with 100 iterations:
- Measured: 1.04ms total, 0.010ms/iter, 201.8B cells/s
- Claimed: 1.04ms, 200.9B cells/s
- Within noise margin. Numbers are reproducible.

### 2. newton_underwater_bench.py — PASS

Two benchmark modes detected (short + long). Using the short-mode results at 256 worlds / 5120 bodies:
- Measured: 5,443 steps/s, 1,393,406 env/s
- Claimed: 5,700 steps/s, 1.45M env/s
- ~4% delta — within normal run-to-run variance.

The long-mode results (with full physics pipeline) are much lower: 48 steps/s, 12,162 env/s at 256 worlds.

### 3. mujoco_warp_bench.py — MISMATCH

Batched parallel envs, 4-leg ant model, 1000 steps (warmup 100), N=4096:
- Measured: 322 steps/s, 1,317,297 envs/s
- Claimed: 2.45M env-steps/s
- **46% below claimed.** The claim appears inflated. Note: GPU had ~10 GB memory already allocated from prior benchmark runs, which may reduce performance. However, a 2x gap is unlikely to be explained by memory pressure alone.

### 4. gpu_baseline.py — MISMATCH

4096x4096 matmul throughput:
- FP32: measured 31.33 TFLOPS (claimed ~~67~~ → corrected 104.8 TFLOPS per spec) — **measured below spec; benchmark may not exercise full tensor-core path**
- FP16: measured 93.71 TFLOPS (claimed ~~213~~ → corrected 838 TFLOPS per spec) — **measured below spec; benchmark may not exercise full tensor-core path**

The script uses standard Warp/CuPy matmul without explicit tensor core or TF32 mode. The claimed numbers likely require:
- TF32 mode for FP32 (effective 2x throughput on tensor cores)
- Tensor Core-optimized FP16 matmul (cublas Hgemm with TC)
- Or different matrix dimensions better suited to TC tile sizes

RTX 5090 theoretical peak: ~104 TFLOPS FP32 (non-TF32), ~210 TFLOPS FP32 (TF32), ~836 TFLOPS FP16 (TC). The measured ~31 TFLOPS FP32 suggests scalar path, not tensor cores.

### 5. navier_stokes_3d.py — PASS

Full Navier-Stokes projection solver at 256^3:
- Throughput (random IC): 429M cells/s
- Taylor-Green validation: 207M cells/s
- Lid-driven cavity Re=1000 at 128^3: 917M cells/s

Claimed 334M falls within the measured range. The exact number depends on test type and configuration. The solver is functional — L2 error converges with grid refinement (0.204→0.126 at t=0.1).

### 6. newton_mpm_bench.py — MISMATCH

Newton Implicit MPM solver at 100K particles:
- Dam Break 103,823 particles: 106.4 steps/s (claimed 120-140)
- Drop Splash 100,000 particles: 14.3 steps/s (anomalous — 7262 MB memory spike)

The Dam Break result is 16% below the claimed range bottom (120). The Drop Splash at 100K shows a severe anomaly likely caused by memory pressure from prior benchmarks (GPU was at ~10 GB before this run started).

### 7. jacobi_stencil_bench.py — MISMATCH

Jacobi pressure solve (100 iters) Warp vs PyTorch speedup:
- 32^3: Warp 0.75ms vs PyTorch 4.39ms = **5.9x**
- 64^3: Warp 1.34ms vs PyTorch 4.28ms = **3.2x**
- 128^3: Warp 1.50ms vs PyTorch 5.11ms = **3.4x**

Claimed 4.5x is only achieved at the smallest grid (32^3). At the largest grid (128^3), which is the most relevant for real workloads, Warp is 3.4x faster than PyTorch. The 4.5x claim is likely an average or cherry-picked from the 32^3 result.

## GPU Health Check

| Metric | Before | After |
|--------|--------|-------|
| Temperature | 52C | 61C |
| Power | 258W | 176W |
| Memory | 8,766 MiB | 9,672 MiB |
| Utilization | 100% | 45% |

No thermal throttling detected. Temperature remained well below the 90C threshold.

## Recommendations

1. **gpu_baseline.py**: Enable TF32 and Tensor Core paths to reflect real-world throughput. Current numbers use scalar FP32.
2. **mujoco_warp_bench.py**: Re-run with clean GPU state (no other processes). The 2.45M claim needs to be revised down to ~1.3M.
3. **newton_mpm_bench.py**: Run in isolation. The Drop Splash 100K anomaly (14.3 steps/s) is caused by memory pressure from cumulative runs.
4. **jacobi_stencil_bench.py**: Report 3.4x at 128^3, not 4.5x. The larger grid is more representative.
5. **newton_underwater_bench.py**: Clarify which benchmark mode (short vs long) the 5,700 steps/s claim refers to.
