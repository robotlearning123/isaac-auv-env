# benchmarks

GPU throughput benchmarks for the OceanScale simulation stack. OceanScale vs PyBullet head-to-head, Newton solver scaling, fluid kernel comparisons, and ecosystem surveys.

## What is here

**OceanScale vs PyBullet:**
- `oceanscale_vs_bullet.py` — BlueROV2 hover task, apples-to-apples throughput comparison
- `tier1_throughput.py` — Tier-1 hydrodynamics kernel throughput at varying batch sizes

**Newton + Warp kernels:**
- `newton_underwater_bench.py` — Newton solver + underwater dynamics (rigid, contact, multiworld, FSI, solver comparison)
- `newton_mpm_bench.py` — Newton MPM (material point method)
- `newton_worlds_throughput.py` — Multi-world throughput scaling
- `kernel_throughput.py` — Warp kernel launch overhead and throughput
- `gpu_baseline.py` — GPU compute baseline (CUDA occupancy)

**Fluid dynamics:**
- `warp_fluid_bench.py` — Warp fluid solver
- `navier_stokes_3d.py` — 3D Navier-Stokes
- `sph_advanced_bench.py` — Smoothed particle hydrodynamics
- `spectral_cfd_bench.py` — Spectral CFD methods
- `vortex_particle_bench.py` — Vortex particle methods
- `lbm_d3q19_bench.py` — D3Q19 Lattice Boltzmann (Warp + PyTorch backends)
- `fsi_underwater_bench.py` — Fluid-structure interaction

**Cross-framework comparisons:**
- `jacobi_stencil_bench.py` — Jacobi pressure stencil across CuPy/JAX/PyTorch/Warp/Triton (2D + 3D)
- `mujoco_warp_bench.py` — MuJoCo-Warp throughput
- `taichi_sph_bench.py` / `taichi_euler_bench.py` — Taichi framework
- `jax_cfd_bench.py` — JAX CFD
- `cupy_lbm_bench.py` — CuPy LBM
- `kamino_solver_bench.py` — Kamino solver
- `sparse_solve_bench.py` — Sparse linear solver
- `xlb_bench.py` — XLB LBM framework
- `gpu_optimization_bench.py` — GPU optimization sweep

**PyBullet reference:**
- `bullet_bluerov_env.py` — PyBullet BlueROV2 environment (for comparison)

**Surveys and results:**
- `RESULTS.md` — Benchmark results and methodology
- `ECOSYSTEM_MATRIX.md` — Ecosystem framework comparison matrix
- `VERIFICATION_REPORT.md` — Verification report
- `isaac_lab_survey.md` — Isaac Lab feature survey
- `official_community_survey.md` — Community survey results

## How to run

```bash
# Requires CUDA 12.8+ GPU and uv
uv sync --extra dev

# OceanScale vs PyBullet head-to-head
uv run python benchmarks/oceanscale_vs_bullet.py

# Tier-1 throughput sweep
uv run python benchmarks/tier1_throughput.py

# Newton underwater benchmark (all scenarios)
uv run python benchmarks/newton_underwater_bench.py --scenario all

# Full kernel throughput
uv run python benchmarks/kernel_throughput.py
```

## Current results

See [`RESULTS.md`](RESULTS.md) for full methodology, hardware config, and numbers. Headline: 17K env-steps/s at n=64 on a single RTX 5090, 10.5x over PyBullet n=1.

## Further reading

- [`../AGENTS.md`](../AGENTS.md) — agent configuration and build commands
- [`../POSITIONING.md`](../POSITIONING.md) — messaging law
