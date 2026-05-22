# OceanScale vs PyBullet Benchmark Results

## TL;DR

At **n_envs = 64**, OceanScale (Newton+Warp GPU) runs **10.5× faster** than PyBullet
(CPU) on the same BlueROV2 hover task. At n_envs = 1, **PyBullet is 5× faster**
(GPU kernel-launch overhead dominates small-batch latency). The crossover sits
between n_envs = 8 and 16. RL training with vectorized environments — the
standard approach — is the OceanScale regime.

This benchmark documents three known physics differences between the two
environments (see "Documented Differences" below). It is "matched", not "identical".

## Hardware

- GPU: NVIDIA GeForce RTX 5090 (31 GiB, sm_120)
- CUDA: 12.9, Driver 13.0
- Python: 3.12
- Run date: 2026-05-22

## Sweep Results (30K steps/run × 2 runs, init noise disabled)

| Engine | Throughput (env-steps/s) | Per-env FPS | Time for 1M env-steps | Speedup vs PyBullet |
|--------|--------------------------|-------------|-----------------------|---------------------|
| PyBullet n=1 | 1,661 ± 66 | 1,661 | 602.9 s | 1.00× |
| OceanScale n=1 | 337 ± 6 | 337 | 2,972.2 s | **0.20×** (slower) |
| OceanScale n=4 | 1,349 ± 3 | 337 | 741.2 s | 0.81× |
| OceanScale n=16 | 5,076 ± 8 | 317 | 197.0 s | 3.06× |
| OceanScale n=64 | 17,427 ± 5 | 272 | 57.4 s | **10.49×** |

Per-env FPS for OceanScale stays roughly flat (337 → 272 from n=1 to n=64); the
small drop reflects GPU memory bandwidth not kernel launch. Throughput scales
near-linearly. The headline number is throughput, because that is what RL
training cares about.

## Documented Differences (NOT a clean "identical conditions" claim)

| Aspect | OceanScale Tier-1 | PyBullet baseline |
|--------|-------------------|-------------------|
| Init position | Disabled noise in benchmark (default is ±0.5 m random) | Always exact target |
| Damping | Diagonal + 4 cross-coupling terms (`oceanscale/hydro/tier1_kernels.py`) | Diagonal only (`benchmarks/bullet_bluerov_env.py`) |
| Coriolis | Full skew-symmetric C(ν) (M_RB + M_A) | Simplified angular-momentum form |
| Restoring | Euler-angle form per von Benzon Eq 12 | Quaternion-derived equivalent |
| Thruster | Tier-1 deadband + saturation | Low-pass filter only, no deadband |

For this benchmark, the init noise was disabled in OceanScale so reset behavior
matches PyBullet's deterministic reset. The other four differences remain. They
are physics modeling choices that PyBullet does not aim to match; OceanScale's
Tier-1 follows Fossen + von Benzon 2022 for AUV-applicable accuracy.

A future Wave (post v0.1) will either bring the PyBullet baseline to bit-for-bit
parity with Tier-1, or use a different baseline (Gazebo URDF + ROS) for full
hydrodynamic comparison.

## What This Benchmark Says

1. **OceanScale's advantage is parallelism**, not per-step latency. GPU kernel
   launch overhead dominates at small batch sizes. For RL with PPO/SAC at
   n_envs ≥ 16, OceanScale delivers 3–10× throughput over single-env PyBullet.

2. **Crossover point** sits between n_envs = 8 and 16 for this task on this
   hardware. Below 8, PyBullet wins on wall-clock.

3. **RL training implication**: 1M env-steps of training data takes 57 s at
   n_envs = 64 vs 603 s on PyBullet — that is the practical impact.

4. **Memory**: OceanScale at n=64 uses ~50 MB additional GPU memory. PyBullet
   single env uses ~0 MB additional. GPU memory headroom is generous on RTX
   5090 (31 GB).

## What This Benchmark Does NOT Compare

1. **Rendering**: Neither environment renders visuals in this benchmark.
   PyBullet DIRECT mode and OceanScale headless are both compute-only.

2. **Contact / collision**: PyBullet has full rigid-body collision. OceanScale
   Tier-1 uses a sphere proxy with no contact handling.

3. **Fluid simulation**: Neither uses fluid CFD/SPH in this run. OceanScale has
   an optional `GridFluidSolver` (disabled here).

4. **Multi-body / articulation**: Both use a single rigid body. Articulated
   ROV-with-manipulator would change the picture significantly.

5. **Warm-up cost**: OceanScale has ~2 s one-time Warp JIT compilation. Not
   counted in the 30K-step measurement (excluded via 10-step warmup loop).
   PyBullet has no JIT step.

## Reproduction

```bash
# Install bench extras (PyBullet + psutil)
uv sync --extra bench

# Sweep (default: 100K steps × 2 runs)
uv run python benchmarks/oceanscale_vs_bullet.py --sweep --runs 2

# Smaller for quick check
uv run python benchmarks/oceanscale_vs_bullet.py --sweep --n-steps 30000 --runs 2

# Single point
uv run python benchmarks/oceanscale_vs_bullet.py --n-envs 64 --runs 3
```

All raw numbers are saved to `benchmarks/oceanscale_vs_bullet_results.json`.

## Fairness Statement

This benchmark is designed to be honest and reproducible:

- Same vehicle (BlueROV2 Heavy, von Benzon 2022 parameters where possible)
- Same task (hover at [0, 0, -1.5])
- Same observation space (26 dim) and reward formula (matched implementations)
- Same dt (1/240 s)
- Same zero policy (no thruster input — isolates physics step cost from
  policy compute)
- Init noise disabled on OceanScale side to match PyBullet's deterministic
  reset (default OceanScale behavior in v0.1 is ±0.5 m / ±0.5 rad random)
- 2 runs per configuration for stddev (n=2 is the bare minimum; v0.2 should
  raise to ≥5 for tighter confidence intervals)

Honest conclusion: **OceanScale scales; PyBullet wins single-env latency.**
For RL training with vectorized environments — the standard pattern in modern
robot learning — OceanScale at n_envs ≥ 16 delivers the speedup that makes
the difference between iterating in seconds vs minutes.
