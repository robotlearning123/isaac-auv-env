# NVIDIA Ecosystem Fluid/CFD Capability Matrix

Hardware: **NVIDIA GeForce RTX 5090** (31 GB, sm_120, 170 SMs)
Date: 2026-05-20
CUDA: 12.8 | Driver: 13.0

## Framework Versions

| Framework | Version | Backend |
|-----------|---------|---------|
| Warp      | 1.9.0   | CUDA 12.8 (native kernels) |
| PyTorch   | 2.10.0  | CUDA 12.8 |
| JAX       | 0.10.1  | XLA/CUDA |
| CuPy      | 13.6.0  | CUDA 12.8 (RawKernel) |
| Triton    | 3.6.0   | PTX (via PyTorch) |
| Modulus   | 0.2.1   | PyTorch (physics-ML) |
| MuJoCo    | 3.3.6   | CPU + MuJoCo-Warp GPU |

---

## 1. Stencil Operations (3D Jacobi, 7-point)

Grid: 128^3 = 2,097,152 cells, 20 iterations

| Framework        | ms/iter | Speedup vs PyTorch |
|------------------|---------|--------------------|
| Triton 3.6       | 0.014   | 3.44x              |
| Warp 1.9         | 0.015   | 3.05x              |
| torch.compile    | 0.038   | 1.22x              |
| PyTorch eager    | 0.046   | 1.00x              |

Winner: **Triton** (custom kernel, 3.4x over eager)

---

## 2. 2D Advection-Diffusion (512x512)

| Framework        | ms/iter | Notes |
|------------------|---------|-------|
| JAX lax.scan     | 0.005   | Fused 100-step scan (amortized) |
| Warp             | 0.008   | Custom kernel |
| JAX vmap (b=8)   | 0.017   | 8 batched envs |
| JAX single step  | 0.031   | JIT warm |
| PyTorch eager    | 0.064   | Roll-based periodic BC |
| JAX JIT compile  | 44.9    | One-time cold compile |

Winner: **JAX lax.scan** (kernel fusion across iterations), **Warp** best single-step

---

## 3. Lattice Boltzmann Method (LBM)

### D2Q9 2D (CuPy RawKernel vs PyTorch)

| Grid      | CuPy MLUPS | PyTorch MLUPS | CuPy ms/iter | Speedup |
|-----------|------------|---------------|--------------|---------|
| 512^2     | 25,566     | 354           | 0.010        | 72.2x   |
| 1024^2    | 36,617     | 924           | 0.029        | 39.6x   |
| 2048^2    | 16,221     | 1,002         | 0.259        | 16.2x   |
| 4096^2    | 15,780     | 775           | 1.063        | 20.4x   |

### D3Q19 3D Lid-Driven Cavity (Warp vs PyTorch)

| Grid    | Warp MLUPS | PyTorch MLUPS | Warp ms/step | PyTorch ms/step |
|---------|------------|---------------|--------------|-----------------|
| 64^3    | 268        | 36            | 0.98         | 7.39            |
| 128^3   | 1,419      | 113           | 1.48         | 18.64           |
| 256^3   | 2,182      | 137           | 7.69         | 122.12          |

Winner: **CuPy RawKernel** for 2D (36K MLUPS peak), **Warp** for 3D (2.2K MLUPS)

---

## 4. FFT / Spectral Methods

### FFT throughput (batched 2D FFT, complex64)

| Grid | CuPy fwd | CuPy inv | Torch fwd | Torch inv | CuPy cells/s |
|------|----------|----------|-----------|-----------|--------------|
| 64   | 0.27 ms  | 0.29 ms  | 0.27 ms   | 0.08 ms   | 9.8e8        |
| 128  | 0.69 ms  | 0.77 ms  | 0.71 ms   | 0.71 ms   | 3.1e9        |
| 256  | 3.94 ms  | 4.99 ms  | 4.53 ms   | 4.74 ms   | 4.3e9        |
| 512  | 35.1 ms  | 42.7 ms  | 35.6 ms   | 42.4 ms   | 3.8e9        |

### Pseudo-spectral Navier-Stokes (2D Taylor-Green)

| Grid | time/step | cells/s     | L2 error |
|------|-----------|-------------|----------|
| 256  | 0.774 ms  | 8.5e7       | 1.23e-07 |
| 512  | 1.589 ms  | 1.6e8       | 6.13e-08 |
| 1024 | 4.780 ms  | 2.2e8       | 3.07e-08 |
| 2048 | 24.61 ms  | 1.7e8       | 1.53e-08 |

---

## 5. SPH (Smoothed Particle Hydrodynamics)

Warp hash-grid SPH, 2D dam break (WCSPH)

| N        | h       | ms/step | particles/s | rho err |
|----------|---------|---------|-------------|---------|
| 19,881   | 0.0028  | 1.00    | 19.9M       | --      |
| 99,856   | 0.0012  | 2.28    | 43.8M       | --      |
| 499,849  | 0.0006  | 23.46   | 21.3M       | --      |

Neighbor search (hash grid vs brute force, Warp):

| N        | BF ms | HG ms | Speedup | Avg neighbors |
|----------|-------|-------|---------|---------------|
| 10,000   | 1.07  | 2.65  | 0.4x    | 37.3          |
| 50,000   | 4.76  | 1.00  | 4.7x    | 186.4         |
| 100,000  | 18.94 | 5.03  | 3.8x    | 373.5         |
| 500,000  | OOM   | 152   | --      | 1,865         |
| 1,000,000| OOM   | 589   | --      | 3,734         |

---

## 6. Vortex Particle Method

Warp, flow past cylinder (Re=100):

| N_max   | Active | steps/s | ms/step | Strouhal |
|---------|--------|---------|---------|----------|
| 1,000   | 20     | 231.4   | 4.32    | N/A      |
| 10,000  | 20     | 229.8   | 4.35    | N/A      |
| 50,000  | 20     | 105.8   | 9.45    | -1.00    |

Biot-Savart throughput (synthetic):

| N      | steps/s | ms/step | N^2 ops/s  |
|--------|---------|---------|------------|
| 1,000  | 5,311   | 0.19    | 5.3e9      |
| 10,000 | 476     | 2.10    | 4.8e10     |
| 50,000 | 83      | 12.05   | 2.1e11     |

---

## 7. Sparse Linear Solve (Pressure Poisson)

5-point Laplacian, Dirichlet BC

| Grid | SciPy Direct | CuPy Direct | CuPy CG | PyTorch CG |
|------|-------------|-------------|---------|------------|
| 64   | 0.0045s     | 0.1163s     | 0.4855s | 0.4268s    |
| 128  | 0.0284s     | 0.5288s     | 1.4329s | 1.6255s    |
| 256  | 0.1808s     | 4.2582s     | 2.5955s | 1.6811s    |
| 512  | 1.2383s     | 14.0120s    | 0.9553s | 1.2475s    |

---

## 8. FSI / Rigid Body + Hydrodynamics

Warp + Newton SolverSemiImplicit, batched AUV simulation:

| N_envs | steps/s | ms/step | env-steps/s  | M env/s |
|--------|---------|---------|--------------|---------|
| 64     | 8,505   | 0.12    | 544,306      | 0.54    |
| 256    | 13,622  | 0.07    | 3,487,309    | 3.49    |
| 1,024  | 18,057  | 0.06    | 18,490,331   | 18.49   |
| 4,096  | 14,771  | 0.07    | 60,501,313   | 60.50   |

Terminal velocity validation: all materials within 0.00% error vs analytical.

MuJoCo-Warp (humanoid, n_envs=64): 104 steps/s (9.61 ms/step), 0.007 M env-steps/s.

---

## 9. Physics-ML (NVIDIA Modulus)

Modulus 0.2.1 installed and functional. Tested:

| Model | Params | Inference (batch=2, 64x64) |
|-------|--------|----------------------------|
| FNO   | 4.2M   | 0.952 ms/iter              |

Available models: FNO, AFNO, GraphCast, MeshGraphNet (needs DGL), DLWP, Pix2Pix, SRRN, RNN.
Available data pipelines: Darcy, KelvinHelmholtz, ERA5, AhmedBody, VortexShedding.
Finite difference and finite volume kernels available.
Note: Modulus 0.2.1 is older; current upstream is 0.9+ (requires Python <3.12 due to numpy dependency).

---

## 10. GPU Baseline

| Metric | Value |
|--------|-------|
| Peak FP32 | 44.8 TFLOPS (measured, 4096^2 matmul) |
| Peak FP16 | 135.8 TFLOPS (measured) |
| Device-to-Device BW | 67.4 GB/s (1 GB copy) |
| Warp kernel launch overhead | 4.34 us |
| CuPy alloc speed | 1.63 us/alloc |
| Mixed precision stencil speedup | 1.49x (fp16/fp32 mixed) |
| CUDA graph speedup | 1.06x |

---

## 11. Cross-Framework 3D Stencil (Copy Proxy)

Grid 128^3, 100 steps:

| Framework | ms/iter | cells/s |
|-----------|---------|---------|
| Warp      | 0.056   | 37.4e9  |
| PyTorch   | 0.090   | 23.4e9  |
| JAX       | 0.100   | 21.1e9  |
| CuPy      | 0.364   | 5.8e9   |

---

## 12. Summary: Framework Selection Guide

| Capability        | Best Framework | Runner-up | Notes |
|-------------------|----------------|-----------|-------|
| Custom CUDA kernels | **Triton** | Warp | Triton 3.4x faster than PyTorch eager |
| Lattice Boltzmann  | **CuPy RawKernel** | Warp | CuPy 16-72x over PyTorch; Warp best for 3D |
| Stencil ops (3D)   | **Warp/Triton** | torch.compile | Within 7% of each other |
| Spectral/FFT CFD   | **CuPy** | PyTorch | Similar throughput; cuFFT backing both |
| SPH                | **Warp** | -- | Hash-grid neighbor search, 44M p/s |
| Batched physics    | **Warp+Newton** | MuJoCo-Warp | 60M env-steps/s at 4K envs |
| Physics-ML (surrogate) | **Modulus** | -- | FNO, AFNO, GraphCast; PyTorch-based |
| Iterative solvers  | **JAX lax.scan** | Warp | JAX fuses across iterations |
| Rapid prototyping  | **PyTorch** | JAX | Easiest API; compile for speed |
| Sparse linear solve | **SciPy** (small) | CuPy CG | GPU overhead dominates at small scale |

### Key Takeaways

1. **Custom kernels dominate**: Triton and Warp custom kernels are 3x+ faster than PyTorch eager for structured grid ops.
2. **CuPy RawKernel is unmatched for LBM**: D2Q9 hits 36K MLUPS at 1024^2, 20-72x faster than PyTorch tensor ops.
3. **Warp is the best all-rounder for physics**: Stencils, SPH, rigid body, and FSI all in one framework with native CUDA kernel generation.
4. **JAX excels at fusion**: lax.scan amortizes kernel launch overhead across iterations, achieving 0.005 ms/iter for advection-diffusion.
5. **Modulus is viable for surrogates**: FNO inference at 0.95ms for 64x64 fields; useful for steady-state CFD replacement, not time-stepping.
6. **Newton + Warp for underwater robotics**: 60M env-steps/s for batched AUV with hydrodynamic forces; validated to 0% error vs analytical.
