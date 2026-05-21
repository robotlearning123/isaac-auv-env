# NVIDIA Official & Community Fluid Examples Survey — RTX 5090

Date: 2026-05-20
Hardware: NVIDIA GeForce RTX 5090 (31 GiB, sm_120)
CUDA: 12.8 / 12.9 (driver 13.0)
Python: 3.13 (system Warp) / 3.12 (project venv)

---

## Part 1: NVIDIA Warp Official Examples (Warp 1.9.0 / 1.13.0)

### 1.1 Core Examples

| Example | Path | Status | Grid/Particles | Avg Step | Notes |
|---------|------|--------|----------------|----------|-------|
| **example_fluid** | `warp/examples/core/example_fluid.py` | PASS | 256x128 | 0.10 ms | 2D Stable Fluids solver. Extremely fast. ~3,000 FPS equivalent. |
| **example_sph** | `warp/examples/core/example_sph.py` | PASS | ~15K particles | 2.7 ms | 3D SPH with USD renderer. Requires `usd-core` (`pip install usd-core`). |
| **example_wave** | `warp/examples/core/example_wave.py` | PASS | 128x128 | 0.33 ms | 2D wave equation. Very fast. Requires `usd-core`. |

### 1.2 FEM Examples

| Example | Path | Status | Solver | Notes |
|---------|------|--------|--------|-------|
| **example_apic_fluid** | `warp/examples/fem/example_apic_fluid.py` | PASS | APIC + CR | APIC fluid with CR linear solver. ~10 ms/frame including render. Converged in ~55 CR iterations. |
| **example_navier_stokes** | `warp/examples/fem/example_navier_stokes.py` | PASS | CG | 2D Navier-Stokes FEM. CG solver converged in 17-70 iterations per frame. |
| **example_stokes** | `warp/examples/fem/example_stokes.py` | PASS | CG | Stokes flow FEM. CG converged in ~810 iterations. |
| **example_convection_diffusion** | `warp/examples/fem/example_convection_diffusion.py` | PASS | CG | Convection-diffusion. CG converged in ~10 iterations. |
| **example_convection_diffusion_dg** | `warp/examples/fem/example_convection_diffusion_dg.py` | PASS | BiCGSTAB | DG variant. BiCGSTAB converged in ~5 iterations. |
| **example_burgers** | `warp/examples/fem/example_burgers.py` | PASS | Implicit | Burgers equation. |
| **example_adaptive_grid** | `warp/examples/fem/example_adaptive_grid.py` | PASS | CR | Adaptive grid FEM. CR converged in ~258 iterations. |
| **example_streamlines** | `warp/examples/fem/example_streamlines.py` | PASS | CG | Streamline visualization for Stokes flow. |

### 1.3 Sim Examples

| Example | Path | Status | Avg Step | Notes |
|---------|------|--------|----------|-------|
| **example_soft_body** | `warp/examples/sim/example_soft_body.py` | PASS | 0.13 ms | Soft body simulation. Extremely fast. |
| **example_granular** | `warp/examples/sim/example_granular.py` | PASS | 0.03 ms | Granular particle sim. Lightning fast. |

### 1.4 Optimization Examples

| Example | Path | Status | Notes |
|---------|------|--------|-------|
| **example_fluid_checkpoint** | `warp/examples/optim/example_fluid_checkpoint.py` | PASS | Differentiable fluid with Adam optimizer. Loss decreased from 0.52 to 0.34 over 50 iterations. |

### Warp Summary

All Warp fluid/FEM examples passed on RTX 5090 with CUDA 12.8. Key findings:

- **2D Stable Fluids**: 0.10 ms/step (256x128) — ~10,000 steps/sec
- **3D SPH**: 2.7 ms/step (~15K particles) — ~370 steps/sec
- **2D Wave**: 0.33 ms/step (128x128) — ~3,000 steps/sec
- **FEM APIC Fluid**: 10.6 ms/frame (including USD render overhead)
- **Soft body**: 0.13 ms/step
- **Granular**: 0.03 ms/step

Note: Warp examples must be run from system Python 3.13 (where Warp 1.9.0 was installed at time of survey), not from the project venv (Warp 1.13.0, pinned via pyproject.toml). Examples requiring USD rendering need `pip install usd-core`.

---

## Part 2: Newton Physics Examples (Newton 1.2.0)

Newton runs in the project venv (`uv run`), uses Warp 1.13.0.

### 2.1 MPM Examples

| Example | Path | Status | Benchmark | Notes |
|---------|------|--------|-----------|-------|
| **example_mpm_granular** | `newton/examples/mpm/example_mpm_granular.py` | PASS | **4.61 FPS** (97 frames in 21.03s) | Implicit MPM with rheology solver. Granular material. |
| **example_mpm_viscous** | `newton/examples/mpm/example_mpm_viscous.py` | PASS | **4.73 FPS** (97 frames in 20.50s) | Viscous fluid MPM. |
| **example_mpm_twoway_coupling** | `newton/examples/mpm/example_mpm_twoway_coupling.py` | PASS | **25.5 FPS** (97 frames in 3.81s) | Two-way coupling (fluid-rigid). Faster due to simpler material model. |
| **example_mpm_snow_ball** | `newton/examples/mpm/example_mpm_snow_ball.py` | PASS | Completed (no benchmark output captured) | Snow simulation. |
| **example_mpm_multi_material** | `newton/examples/mpm/example_mpm_multi_material.py` | PASS | Completed (no benchmark output captured) | Multi-material MPM. |

### 2.2 Soft Body Examples

| Example | Path | Status | Benchmark | Notes |
|---------|------|--------|-----------|-------|
| **example_softbody_hanging** | `newton/examples/softbody/example_softbody_hanging.py` | PASS | **16.1 FPS** (97 frames in 6.01s) | VBD solver. JIT compilation took ~11s (7250ms + 3574ms for two modules). |

### Newton Summary

- Newton MPM examples run with `--viewer null --num-frames 100 --benchmark`
- MPM granular/viscous: ~4.6 FPS (implicit solver with rheology — heavy computation)
- MPM two-way coupling: 25.5 FPS (lighter material model)
- Soft body (VBD): 16.1 FPS after warmup JIT
- All examples verified working on RTX 5090 with CUDA 12.9

### Other Newton Examples (not benchmarked, cataloged)

Newton also includes:
- `mpm/example_mpm_anymal.py` — Anymal robot on MPM terrain
- `mpm/example_mpm_beam_twist.py` — Beam twist MPM
- `mpm/example_mpm_grain_rendering.py` — Grain rendering
- `robot/` — 10+ robot examples (anymal, cartpole, H1, G1, panda, UR10, allegro hand)
- `cloth/` — 8 cloth examples
- `cable/` — 4 cable examples
- `diffsim/` — 6 differentiable simulation examples
- `sensors/` — IMU, contact sensor, tiled camera
- `multiphysics/` — softbody dropping to cloth, gift wrapping

---

## Part 3: Community Packages

### 3.1 XLB (Autodesk LBM) — v0.3.1

**Install**: `uv pip install xlb` — installed successfully (14 packages including VTK, pyvista)

**Result: FAILED** — Incompatible with Warp 1.13.0

Errors encountered:
1. `ImportError: cannot import name 'ScopedTimer' from 'warp.utils'` — Warp 1.13.0 removed `ScopedTimer`
2. After patching that: `AttributeError: module 'warp' has no attribute 'mat'` — Warp 1.13.0 removed `wp.mat`

XLB 0.3.1 targets Warp ~1.7.x or earlier. It requires:
- `warp.utils.ScopedTimer` (removed in Warp 1.13)
- `warp.mat` (removed/restructured in Warp 1.13)

**Verdict**: Cannot use with current Warp. Requires either downgrading Warp to 1.7.x or waiting for XLB update.

### 3.2 FluidX3D

**Result: SKIPPED** — FluidX3D is C++/OpenCL based, not a Python package. Requires building from source and OpenCL runtime. Not applicable to Warp/CUDA pipeline. No pip package available.

### 3.3 Lettuce (LBM)

**Result: FAILED** — The PyPI package `lettuce` (v0.2.23) is a BDD testing framework, NOT an LBM solver. The actual LBM lettuce project (https://github.com/ae2405/lettuce) is not published on PyPI and would need manual install from GitHub. Not tested further.

### 3.4 Taichi — v1.7.4

**Install**: `uv pip install taichi` — installed successfully

**Result: PASS** — Works on RTX 5090 with CUDA

Benchmark results:
- **Taichi 2D Euler Fluid** (512x512 grid, 40 Jacobi pressure iterations):
  - Avg step time: **1.58 ms**
  - Throughput: **165.8 Mcell/s**
  - Min: 0.76 ms, Max: 8.12 ms
- **Taichi CUDA init**: Successful, recognizes RTX 5090

Note: Taichi requires code to be in a file (cannot use `python -c` for `@ti.kernel` decorated functions due to source inspection limitations).

### Community Summary

| Package | Version | Install | Import | CUDA | Benchmark |
|---------|---------|---------|--------|------|-----------|
| XLB | 0.3.1 | PASS | FAIL (Warp API incompat) | N/A | N/A |
| FluidX3D | N/A | N/A (C++/OpenCL) | N/A | N/A | N/A |
| Lettuce | 0.2.23 | PASS | FAIL (wrong package) | N/A | N/A |
| **Taichi** | 1.7.4 | PASS | PASS | PASS | **165.8 Mcell/s** (512x512 Euler) |

---

## Part 4: Performance Comparison Matrix

| Solver | Grid/Particles | Avg Step | Throughput | Backend |
|--------|---------------|----------|------------|---------|
| Warp Stable Fluids | 256x128 (32K) | 0.10 ms | ~330M cell/s | Warp CUDA |
| Warp SPH 3D | ~15K particles | 2.7 ms | ~5.6K part/s | Warp CUDA |
| Warp Wave 2D | 128x128 (16K) | 0.33 ms | ~50M cell/s | Warp CUDA |
| Warp Soft Body | ~1K verts | 0.13 ms | N/A | Warp CUDA |
| Warp Granular | ~5K particles | 0.03 ms | ~167K part/s | Warp CUDA |
| Newton MPM Granular | ~10K particles | ~217 ms | ~4.6 FPS | Warp CUDA |
| Newton MPM Viscous | ~10K particles | ~211 ms | ~4.7 FPS | Warp CUDA |
| Newton MPM Two-way | ~5K particles | ~39 ms | 25.5 FPS | Warp CUDA |
| Newton Soft Body (VBD) | ~2K verts | ~62 ms | 16.1 FPS | Warp CUDA |
| Taichi Euler 2D | 512x512 (262K) | 1.58 ms | 165.8 Mcell/s | Taichi CUDA |

---

## Files Generated

- `/home/robot/workspace/46-marine/benchmarks/taichi_euler_bench.py` — Taichi 2D Euler benchmark script
- `/home/robot/workspace/46-marine/benchmarks/taichi_sph_bench.py` — Taichi SPH benchmark script (O(N^2))
- `/home/robot/workspace/46-marine/benchmarks/xlb_bench.py` — XLB benchmark (failed due to API incompat)
- Various `.usd` files in project root from Warp/Newton USD renders

## Key Takeaways

1. **Warp is production-ready** for fluid simulation on RTX 5090. All examples passed without modification.
2. **Newton MPM** is the most relevant for underwater simulation — supports granular, viscous, snow, multi-material, and two-way coupling.
3. **XLB is dead** on current Warp — API incompatibility with Warp 1.13+ is a hard blocker.
4. **Taichi works** and is competitive (165.8 Mcell/s for 512x512 Euler), but is a separate ecosystem from Warp/Newton.
5. The **Warp FEM examples** (Navier-Stokes, Stokes, APIC fluid) are the most directly useful reference for OceanScale's fluid solver development.
