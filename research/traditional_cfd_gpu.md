# Traditional CFD Solvers on GPU for Underwater/Ocean Engineering

Research conducted 2026-05-20. Focus: GPU-accelerated traditional CFD solvers relevant to underwater/ocean engineering and potential integration with NVIDIA Isaac Sim / Newton ecosystem.

---

## 1. OpenFOAM on GPU

### Current Status

OpenFOAM remains the most widely used open-source CFD code, but as of 2025 it **does not have a standard, widely-adopted GPU port**. The core codebase uses OpenMPI for CPU parallelism only.

Source: [Neil Ashton, LinkedIn](https://www.linkedin.com/posts/neilashton_cae-gpus-openfoam-activity-7259556369580572673-M4Ao), [Reddit r/OpenFOAM discussion](https://www.reddit.com/r/OpenFOAM/comments/1r6iyvj/openfoam_gpu_acceleration_status/)

### Active GPU Porting Efforts

| Approach | Status | Notes |
|----------|--------|-------|
| **RapidCFD** | Experimental | CUDA port of OpenFOAM; historically showed 2-20x speedup depending on solver and mesh. Not all solvers/boundary conditions ported. [GitHub: xulong-cf/RapidCFD](https://github.com/xulong-cf/RapidCFD) |
| **AmgX integration** | Active | NVIDIA's AmgX library used to accelerate linear solvers within OpenFOAM. Presented at NVIDIA GTC 2025. [YouTube: NVIDIA Developer](https://www.youtube.com/watch?v=1d1DVURNz4E) |
| **PETSc4FOAM** | Research | GPU-accelerated via PETSc backend; supports NVIDIA (CUDA), AMD (HIP), Intel discrete GPUs. [KAUST presentation (PDF)](https://wiki.openfoam.com/images/c/cd/OpenFOAM_2020_KAUST_Zampini.pdf) |
| **ArXiv 2025 PoC** | Recent | 2025 study demonstrating GPU offloading for `laplacianFoam`. [arXiv:2507.18268](https://arxiv.org/html/2507.18268v1) |

Source: [MDPI paper on GPU acceleration of OpenFOAM](https://www.mdpi.com/2226-4310/10/9/792), [CFD Online Forums](https://www.cfd-online.com/Forums/openfoam-programming-development/214050-gpu-parallelization-openfoam.html)

### Can OpenFOAM Couple with Newton/Isaac Sim?

No direct coupling exists today. However, several coupling pathways are possible:

1. **OpenFOAM + Chrono::Physics** -- A 2025 paper demonstrates coupling OpenFOAM with Chrono (multibody dynamics) for wave-structure interaction in marine environments. This is the closest analog to an OpenFOAM-Newton coupling.
   Source: [ResearchGate: Coupling OpenFOAM with Chrono](https://www.researchgate.net/publication/397606621_A_new_CFD-MBD_wave-structure_interaction_model_Coupling_OpenFOAM_with_Chrono)

2. **preCICE coupling library** -- General-purpose partitioned coupling library that can connect OpenFOAM to arbitrary physics solvers (including custom rigid-body dynamics engines).
   Source: [TUM preCICE + OpenFOAM (PDF)](https://mediatum.ub.tum.de/doc/1577072/1577072.pdf)

3. **HOS-OpenFOAM-MoorDyn** -- Coupling strategy for underwater hydrodynamic loads using High-Order Spectral + OpenFOAM + mooring dynamics.
   Source: [HAL Science (PDF)](https://hal.science/hal-04491026v1/file/Revision2_Elsevier_s_Hydrodynamic_Coupling_RedMark.pdf)

### Performance on RTX 5090 / H100

No published benchmarks specifically for RTX 5090 or H100 with OpenFOAM GPU ports as of 2026-05. Historical RapidCFD benchmarks show 2-20x speedup on older GPU architectures. The main bottleneck is VRAM capacity -- large maritime meshes (millions of cells) exceed consumer GPU memory.

---

## 2. ANSYS Fluent / CFX on GPU

### GPU Acceleration Support

ANSYS Fluent has the most mature GPU CFD solver among commercial tools:

- **Fluent 2025 R1**: GPU solver requires CUDA 11.8+
- **Fluent 2025 R2**: CUDA 12.8 required; full VOF (Volume of Fluid) and species transport support on GPU
- **Fluent 2026 R1**: Continued GPU solver improvements

Source: [ANSYS Fluent GPU Hardware Buying Guide](https://innovationspace.ansys.com/knowledge/forums/topic/fluent-gpu-solver-hardware-buying-guide/), [ANSYS Blog Part 3](https://www.ansys.com/blog/unleashing-full-power-gpus-ansys-fluent-software-part-3)

### NVIDIA Partnership

ANSYS has a deepening partnership with NVIDIA spanning GPU acceleration, Omniverse integration, and AI-powered simulation:

- **Omniverse integration**: Announced Q3 2025, NVIDIA Omniverse visualization and data processing capabilities will be available directly within Fluent.
  Source: [ANSYS News Center](https://www.ansys.com/news-center/press-releases/8-12-25-ansys-oems-omniverse), [LinkedIn: ANSYS](https://www.linkedin.com/posts/ansys-inc_ansys-to-integrate-nvidia-omniverse-activity-7308487870439149586-2tTX)

- **Digital twins**: ANSYS + NVIDIA leveraging Omniverse Blueprint, CUDA, and PhysicsNeMo for AI-driven digital twins.
  Source: [NVIDIA GTC 2025 Session](https://www.nvidia.com/en-us/on-demand/session/gtc25-s73028/)

- **Licensing**: New "CFD HPC Ultimate" license in 2025 R1 simplifies enterprise GPU usage.
  Source: [EDRMedeso](https://edrmedeso.com/article/accelerating-cfd-with-gpus-how-ansys-discovery-and-fluent-unlock-10x-faster-simulations/)

### Relevance to Underwater/Ocean Engineering

VOF support on GPU (2025 R2) is directly relevant to free-surface and multiphase flows common in marine/ocean engineering. The Omniverse integration pathway provides a potential route to couple Fluent CFD with Isaac Sim for underwater robotics simulation.

---

## 3. STAR-CCM+ GPU

### Siemens GPU Support

Simcenter STAR-CCM+ has been adding GPU capabilities since 2022:

- **2022.1**: Initial GPU support via CUDA for segregated flow solver (constant density) with most turbulence models
- **Version 2406**: Expanded GPU solver capabilities
- **Version 2602** (2026): **GPU-native VOF and MMP (Multiphase) solvers** -- major milestone for multiphase simulations on GPU

Source: [Siemens Blog: STAR-CCM+ 2602](https://blogs.sw.siemens.com/simcenter/simcenter-star-ccm-2602-released/), [Volupe: GPGPU enhancements in 2602](https://volupe.com/simcenter-star-ccm/gpgpu-enhancements-in-simcenter-star-ccm-2602/)

### Performance

Up to **6.2x speedup** over CPU-only in benchmarks with NVIDIA RTX 6000 Ada GPUs.
Source: [Exxact Corporation benchmarks](https://www.exxactcorp.com/blog/engineering-mpd/star-ccm-cpu-vs-gpu-runtime-benchmarks-with-mayahtt)

### Coupling with Isaac Sim / Omniverse

Siemens has an active partnership with NVIDIA for Omniverse integration:

- **NVIDIA Omniverse libraries** are being integrated into Simcenter STAR-CCM+ for physics-based digital twins.
  Source: [NVIDIA Customer Stories: Siemens](https://www.nvidia.com/en-us/case-studies/siemens-accelerates-product-development-and-innovation-with-industrial-ai/)

- Siemens launched **Digital Twin Composer** leveraging Omniverse for physically accurate, real-time 3D simulation.
  Source: [Engineering.com](https://www.engineering.com/siemens-introduces-digital-twin-composer-for-lifecycle-modeling/), [PR Newswire](https://www.prnewswire.com/news-releases/siemens-brings-the-industrial-metaverse-to-life-with-digital-twin-composer-302654058.html)

- Joint solution of Siemens STAR-CCM+ + NVIDIA + Lenovo provides optimized CAE environment.
  Source: [Connection.com (PDF)](https://www.connection.com/media/nlabjth4/lenovo-nvidia-mfg.pdf)

### Licensing

Requires "Power Session Plus" license for unlimited GPU/CPU usage.
Source: [Siemens Community](https://community.sw.siemens.com/s/question/0D54O00007LKrHoSAL/fact-sheet-on-gpgpu-acceleration-in-simcenter-starccm)

---

## 4. NVIDIA AmgX

### What Is AmgX?

AmgX is NVIDIA's **GPU-accelerated Algebraic Multigrid (AMG)** library for solving large sparse linear systems. It provides the core linear solver component that dominates CFD computation time.

- **2-10x speedup** over competitive CPU implementations for linear solving
- Open-source on GitHub: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX)
- Provides drop-in GPU acceleration for distributed AMG solvers

Source: [SIAM foundational paper](https://epubs.siam.org/doi/10.1137/140980260), [GitHub: NVIDIA/AMGX](https://github.com/NVIDIA/AMGX)

### Can It Accelerate CFD Linear Solvers?

Yes. AmgX is specifically designed to accelerate the linear solver portion of CFD and other PDE simulations:

- **TRUST CFD platform** (2025) implements iterative solvers and AMG preconditioners optimized for NVIDIA GPUs using AmgX.
  Source: [ScienceDirect: TRUST CFD Platform](https://www.sciencedirect.com/org/science/article/pii/S2491929225000561)

- **COMSOL Multiphysics** uses AMG for large CFD simulations since version 5.3a.
  Source: [COMSOL Blog](https://www.comsol.com/blogs/using-the-algebraic-multigrid-amg-method-for-large-cfd-simulations)

- Active use in 2025 publications (CERFACS Sparse Days, Ginkgo comparisons).
  Source: [CERFACS Sparse Days 2025 (PDF)](https://sparsedays.cerfacs.fr/wp-content/uploads/sites/72/2025/07/2025_SparseDay_YHMTsai_Ginkgo.pdf)

### Current Version and Capabilities

- Actively maintained by NVIDIA; supports multi-GPU and distributed solving
- Supports mixed-precision approaches for further acceleration
- C API with bindings available for integration into custom codes

### Integration with Custom Solvers

AmgX can be integrated into custom CFD codes via its C API. The typical pattern:
1. Assemble sparse matrix on CPU (or GPU)
2. Transfer to AmgX solver object
3. Configure AMG preconditioner + Krylov solver (CG, GMRES, BiCGStab)
4. Solve on GPU
5. Return solution vector

This is relevant for building a custom GPU CFD solver within the Newton/Isaac ecosystem -- AmgX could handle the linear algebra while custom code manages mesh, discretization, and boundary conditions.

---

## 5. NVIDIA cuSPARSE / cuSOLVER / cuFFT / cuDSS

### Math Libraries for Building Custom CFD on GPU

| Library | Purpose | Key Capability |
|---------|---------|---------------|
| **cuSPARSE** | Sparse matrix operations | SpMV, sparse triangular solve, sparse matrix arithmetic |
| **cuSOLVER** | Dense and sparse solvers | LU/QR/Cholesky factorization, eigenvalue problems |
| **cuFFT** | Fast Fourier Transform | 1D/2D/3D FFT, now with Link-Time Optimization for kernel fusion |
| **cuDSS** | Direct Sparse Solver (new) | GPU-accelerated direct solver for very sparse matrices |

Source: [NVIDIA cuSOLVER docs](https://docs.nvidia.com/cuda/cusolver/index.html), [NVIDIA cuDSS docs](https://docs.nvidia.com/cuda/cudss/), [NVIDIA CUDALibrarySamples](https://github.com/nvidia/cudalibrarysamples)

### cuDSS (2025 -- New and Significant)

NVIDIA cuDSS is a first-generation GPU-accelerated **Direct Sparse Solver** library. Key performance claims:

- **11x speedup** for matrix solver in ANSYS HFSS
- **13.8x speedup** for BLR-enabled GPU solver vs CPU exact solver
- **COMSOL 6.4** integrates cuDSS for GPU acceleration across all physics
- **Altair OptiStruct** has also adopted cuDSS
- Latest version 0.7.0 focuses on single-node GPU performance and multi-GPU cluster scaling

Source: [NVIDIA Blog: CUDA-X](https://blogs.nvidia.com/blog/cuda-x-grace-hopper-blackwell/), [COMSOL + cuDSS](https://www.digitalengineering247.com/article/comsol-expand-gpu-acceleration-with-nvidia-direct-sparse-solver), [GTC 2026 session](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81824/)

### Can These Be Used from Python/Warp?

- **CuPy** provides Python access to cuSPARSE, cuSOLVER, cuFFT operations
- **scikit-cuda** offers Python wrappers for CUDA libraries
- **NVIDIA Warp** (Python JIT framework) can call CUDA libraries and provides auto-differentiation. Warp is being used for production CFD by Autodesk Research.
  Source: [NVIDIA Warp developer page](https://developer.nvidia.com/warp-python), [GitHub: NVIDIA/Warp](https://github.com/nvidia/warp)

- **NVIDIA Warp + XLB** (Autodesk's Lattice Boltzmann solver) is a concrete example of Warp being used for GPU-accelerated CFD.
  Source: [GitHub: Autodesk/XLB](https://github.com/Autodesk/XLB), [NVIDIA Developer Blog: Autodesk Warp CFD](https://developer.nvidia.com/blog/autodesk-research-brings-warp-speed-to-computational-fluid-dynamics-on-nvidia-gh200/)

### Performance vs CPU Sparse Solvers

GPU sparse solvers can outperform CPU (SciPy/Intel MKL Pardiso) for large problems, but small problems may be slower due to GPU transfer overhead. A forum thread reports cuSOLVER being slower than SciPy Sparse for some workloads.
Source: [NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/help-improving-performance-using-cusolver-cusparse-routines/276250)

---

## 6. Lattice Boltzmann on GPU

### Overview

Lattice Boltzmann Method (LBM) is highly amenable to GPU parallelization due to its local, stencil-based computation pattern. This makes it one of the most promising approaches for real-time fluid simulation.

### Open-Source GPU LBM Solvers

| Solver | Language | GPU Support | Notes |
|--------|----------|-------------|-------|
| **OpenLB** | C++ | CUDA (NVIDIA), AMD, preliminary Intel | Multi-GPU via CUDA-aware MPI. Release 1.9 added AMD support. |
| **Palabos** | C++ | Community forks with CUDA/OpenCL | No official GPU support as of 2025 |
| **XLB** (Autodesk) | Python/JAX/Warp | JAX or Warp backend | **Fully differentiable**, ML-ready. Best candidate for RL integration. |
| **Sailfish** | Python | CUDA | Open-source GPU LBM solver |
| **gLBM** | C++/CUDA | CUDA | GPU-enabled LBM library |

Source: [OpenLB Performance](https://www.openlb.net/performance/), [CloudHPC OpenLB article (Feb 2025)](https://cloudhpc.cloud/2025/02/11/openlb-harnessing-the-power-of-lattice-boltzmann-for-cfd-and-accelerating-it-with-gpus-on-cloudhpc/), [GitHub: Autodesk/XLB](https://github.com/Autodesk/XLB)

### NVIDIA's LBM Work

NVIDIA does not have a first-party LBM solver, but:
- **NVIDIA Warp** is the framework being used for LBM (via XLB)
- Autodesk Research demonstrated Warp-accelerated LBM on NVIDIA GH200 Grace Hopper
  Source: [NVIDIA Developer Blog: Autodesk Warp CFD](https://developer.nvidia.com/blog/autodesk-research-brings-warp-speed-to-computational-fluid-dynamics-on-nvidia-gh200/)

### Real-Time Capable LBM for RL Training?

LBM is the most promising CFD method for RL coupling because:

1. **XLB + Warp/JAX** provides a fully differentiable LBM solver that can run on GPU with automatic differentiation -- ideal for policy gradient methods
2. **Computation is local** -- no global linear system solve required, making it inherently parallel and fast on GPU
3. **Real-time performance** is achievable for moderate grid sizes (millions of cells at 60+ FPS on modern GPUs)
4. **Multi-GPU scaling** via CUDA-aware MPI (OpenLB) or JAX pmap (XLB)

Key caveat: LBM struggles with high-Mach-number compressible flows. For underwater applications (low Mach, incompressible), LBM is well-suited.

---

## 7. Spectral Methods on GPU

### Pseudo-Spectral Navier-Stokes Solvers

| Solver | Backend | Key Feature | Source |
|--------|---------|-------------|--------|
| **JAX-CFD** | JAX | Experimental ML+CFD with auto-diff on GPU/TPU | [GitHub: google/jax-cfd](https://github.com/google/jax-cfd) |
| **PhiFlow** | PyTorch/TF/JAX/NumPy | Differentiable simulations, multiple NS solver variants | [PhiFlow paper (PDF)](https://raw.githubusercontent.com/mlresearch/v235/main/assets/holl24a/holl24a.pdf) |
| **JAX-Fluids** | JAX | Fully-differentiable compressible CFD | [TUM JAX-Fluids](https://www.mep.tum.de/en/mep/scicohub/jax-fluids/) |

### Performance for Ocean-Scale Simulation

**Dion Haefner's work** (2021) demonstrated supercharged high-resolution ocean simulation with JAX, showing significant GPU acceleration for spectral ocean models.
Source: [Dion Haefner blog post](https://dionhaefner.github.io/2021/12/supercharged-high-resolution-ocean-simulation-with-jax/)

**Veros** (Versatile Ocean Simulator) is a full-fledged ocean general circulation model written in Python with JAX backend, enabling GPU acceleration without manual CUDA code.
Source: [GitHub: team-ocean/voros](https://github.com/team-ocean/voros), [HackMD compilation](https://hackmd.io/@noisyoscillator/Syng3eV8q)

JAX-based spectral solvers benefit from:
- JIT compilation to GPU
- Automatic differentiation for gradient-based optimization
- `vmap` and `pmap` for batch and distributed computation
- Seamless integration with ML frameworks (Flax, Haiku, Optax)

### Limitations for Underwater Robotics

Spectral methods require regular grids and periodic or simple boundary conditions. Complex underwater geometries (submarine hulls, propellers, manipulator arms) are poorly suited to spectral methods. These methods are better for large-scale ocean dynamics than vehicle-scale simulation.

---

## 8. Reynolds-Averaged Navier-Stokes (RANS) on GPU

### Real-Time GPU RANS Solvers

Several GPU-accelerated RANS implementations exist:

| Solver | Type | GPU Support | Source |
|--------|------|-------------|--------|
| **ANSYS Fluent GPU** | Commercial | Full GPU RANS since 2025 R1 | [ANSYS Blog](https://www.ansys.com/blog/unleashing-full-power-gpus-ansys-fluent-software-part-3) |
| **STAR-CCM+ GPGPU** | Commercial | GPU-native solvers since 2022.1 | [Siemens Blog](https://blogs.sw.siemens.com/simcenter/simcenter-star-ccm-2602-released/) |
| **ITP Aero edge-based RANS** | Research | GPU edge-based CFD for RANS on unstructured grids | [NVIDIA GPU Applications Catalog (PDF)](https://www.nvidia.cn/content/gpu-applications/PDF/gpu-applications-catalog.pdf) |
| **TU Delft implicit RANS** | Research | GPU RANS with implicit time-stepping for turbomachinery | [TU Delft (PDF)](https://diamhomes.ewi.tudelft.nl/~kvuik/numanal/aissa.pdf) |
| **CaLES** | Research (LES focus) | GPU-accelerated finite-difference, incompressible wall-bounded flows | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010465525000499) |

Source: [Reddit r/CFD: RANS solvers on GPUs](https://www.reddit.com/r/CFD/comments/1lhfp9t/rans_cfd_solvers_that_run_on_gpus/)

### Can RANS Provide Force Coefficients for RL Training?

Yes, but with important caveats:

- **RANS provides steady-state force coefficients** (drag, lift, moment) for given configurations. These are commonly used as reward signals or state observations in RL-based flow control.
- **Speed**: GPU RANS can be 5-10x faster than CPU RANS, but still typically requires seconds to minutes per configuration -- too slow for millions of RL episodes.
- **Fidelity**: RANS with turbulence models (k-omega SST, SA) provides reasonable force predictions but misses transient flow features important for control.

### Surrogate Modeling Approach (Offline CFD -> Online Lookup)

The dominant strategy for combining CFD with RL:

1. **Offline phase**: Run many RANS (or higher-fidelity) CFD simulations across a parameter space (Reynolds number, angle of attack, control surface deflections, etc.)
2. **Train a surrogate model**: Neural network learns the mapping from parameters to force coefficients
3. **Online phase**: RL agent queries the surrogate model (microseconds) instead of running CFD (seconds to hours)

This is the standard approach in marine robotics for hydrodynamic coefficient prediction. The surrogate model becomes a fast "lookup table" with interpolation capability.

---

## 9. CFD-RL Coupling Strategies

### How to Couple Traditional CFD with RL Training

| Strategy | Description | Speed | Fidelity | Best For |
|----------|-------------|-------|----------|----------|
| **Full CFD-in-the-loop** | RL agent directly interacts with CFD solver each step | Slow (sec/step) | High | Active flow control research |
| **Surrogate model** | NN replaces CFD; trained offline on CFD data | Fast (us/step) | Medium | Policy training, optimization |
| **Pre-computed tables** | Lookup tables from CFD sweeps | Fastest | Low-Medium | Simple parametric studies |
| **Hybrid (DL-MBRL)** | Alternate between surrogate and real CFD | Medium | High | Active control with accuracy |

Source: [arXiv:2408.14232 -- DL-MBRL approach](https://arxiv.org/html/2408.14232v1), [HAL thesis: Coupling DRL and CFD](https://pastel.hal.science/tel-04043187/document)

### Pre-Computed Force Coefficient Tables

Standard approach in naval architecture:
1. Run CFD sweeps over angle of attack, sideslip, control surface deflection, Reynolds number
2. Store Cd, Cl, Cm in multi-dimensional tables
3. Interpolate during simulation

Limitations: no transient effects, no flow-history dependence, no active flow control.

### Surrogate Models (Neural Network Replacing CFD)

**NVIDIA PhysicsNeMo** (formerly Modulus) is the leading framework for building CFD surrogate models:

- Open-source Python framework for physics AI models
- Supports **PINNs, Fourier Neural Operators (FNOs), MeshGraphNets, DoMINO**
- Latest release: **PhysicsNeMo 25.08** (August 2025) with large deformation and full waveform inversion recipes
- **Fourier Neural Operators** can learn the entire CFD solution operator, providing 1000x+ speedup over traditional solvers

Source: [NVIDIA PhysicsNeMo developer page](https://developer.nvidia.com/physicsnemo), [GitHub: NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo), [NVIDIA Blog: Transforming CFD with ML](https://developer.nvidia.com/blog/transforming-cfd-simulations-with-ml-using-nvidia-physicsnemo/), [PhysicsNeMo 25.08 release](https://nvidia.github.io/physicsnemo/blog/2025/08/27/physicsnemo-release-25-08/)

### CFD-RL Frameworks

| Framework | CFD Solver | Coupling | Key Feature | Source |
|-----------|-----------|----------|-------------|--------|
| **SmartFlow** | Solver-agnostic | Non-intrusive | Multi-agent DRL, HPC/MPI-compatible | [arXiv:2508.00645](https://arxiv.org/pdf/2508.00645) |
| **DRLinFluids** | Multiple | Python package | Flexible, easy to use | [GitHub](https://github.com/venturi123/DRLinFluids) |
| **DRLFluent** | ANSYS-Fluent | Non-intrusive | Standardized interfaces | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1877750323002314) |
| **Intrusive DRL-OpenFOAM** | OpenFOAM | Intrusive (embedded) | Eliminates external communication bottleneck | [Chalmers University](https://research.chalmers.se/en/publication/541542) |

Source: [MDPI: Rapid CFD Prediction Based on ML Surrogate](https://www.mdpi.com/2311-5521/10/8/193), [ResearchGate: Surrogate-Based Pressure-Velocity Coupling](https://www.researchgate.net/publication/394002762_Surrogate-Based_Pressure-Velocity_Coupling_Accelerating_Incompressible_Cfd_Flow_Solvers_with_Machine_Learning)

---

## 10. Ocean-Specific CFD on GPU

### ROMS (Regional Ocean Modeling System) on GPU

- No full native GPU port of ROMS exists as of 2025
- NVIDIA's HPC SDK v25.7 simplifies GPU porting of ocean models using **OpenACC and Unified Memory**, targeting models like ROMS
- Community discussion on ROMS forum about running Python-based ocean models on GPUs, achieving better performance than traditional CPU-based parallel simulations

Source: [NVIDIA Blog: Simplify Ocean Modeling on GPUs](https://developer.nvidia.com/blog/less-coding-more-science-simplify-ocean-modeling-on-gpus-with-openacc-and-unified-memory/), [ROMS Forum](https://www.myroms.org/forum/viewtopic.php?t=6104)

### MITgcm GPU Port

- No full native GPU port as of 2025
- Referenced in 2025 GPU ocean modeling review papers; GPU strategies discussed but not implemented
- MITgcm remains primarily Fortran + MPI

Source: [Copernicus: GPU Technologies for Ocean Forecasting (PDF)](https://sp.copernicus.org/articles/5-opsr/23/2025/sp-5-opsr-23-2025.pdf), [NSF PDF](https://par.nsf.gov/servlets/purl/10635289)

### Oceananigans.jl (Leading GPU-Native Ocean Model)

**Oceananigans.jl** is the most advanced GPU-native ocean model available:

- Written in pure **Julia** with first-class GPU support via CUDA.jl and AMDGPU.jl
- Developed by **CliMA (Climate Modeling Alliance)** at Caltech
- Achieves "breakthrough resolution, memory efficiency, and energy usage" vs traditional Fortran models
- Enables **decade-long ocean simulations in a single day** on GPU
- **Breeze.jl** extends Oceananigans to atmospheric modeling

Source: [GitHub: CliMA/Oceananigans.jl](https://github.com/clima/Oceananigans.jl), [arXiv companion paper](https://arxiv.org/html/2309.06662v2), [Caltech/CliMA blog (April 2025)](https://clima.caltech.edu/2025/04/30/high-level-high-resolution-ocean-modeling-at-all-scales-with-oceananigans/), [Official benchmarks (v0.36.0)](https://clima.github.io/OceananigansDocumentation/v0.36.0/benchmarks/)

### GPU Ocean (SINTEF)

GPU-based ensemble ocean forecasting framework running shallow-water simulations on GPUs for short-term uncertainty prediction.
Source: [SINTEF GPU Ocean](https://www.sintef.no/en/software/gpu-ocean/)

### Veros (JAX Ocean Model)

Full ocean general circulation model in Python/JAX with GPU acceleration.
Source: [GitHub: team-ocean/voros](https://github.com/team-ocean/voros)

### AI Surrogate for Ocean Circulation

A fully GPU-accelerated workflow using AI-based surrogate models for coastal ocean circulation, optimized on NVIDIA DGX-2 A100 GPUs.
Source: [arXiv:2410.14952](https://arxiv.org/html/2410.14952v2)

---

## Summary: Integration Pathway for OceanScale/Newton

### Most Promising Approaches (Ranked)

1. **XLB (Lattice Boltzmann) + NVIDIA Warp** -- Fully differentiable, GPU-native, Python-based, auto-diff compatible with RL. Best candidate for real-time underwater fluid simulation in Newton. Limitation: low-Mach only (acceptable for underwater).

2. **NVIDIA PhysicsNeMo Surrogate Models** -- Train FNOs/PINNs on offline CFD data (OpenFOAM/Fluent) to create fast surrogate models that run inside Newton/Isaac Sim. 1000x+ speedup over CFD. Best for when you need force coefficients but not full flow field.

3. **AmgX + Custom Solver via Warp** -- Build a minimal RANS solver using Warp for GPU execution and AmgX for linear algebra. Provides a middle ground between full CFD fidelity and surrogate model speed.

4. **OpenFOAM + preCICE Coupling** -- Use OpenFOAM as the offline CFD engine coupled to Newton via preCICE. Full fidelity but not real-time. Best for generating training data.

5. **Oceananigans.jl** -- For large-scale ocean environment simulation (currents, waves, thermocline). Not suitable for vehicle-scale simulation but excellent for generating ocean boundary conditions.

### Recommended Architecture

```
[Large-scale ocean environment]
  Oceananigans.jl (GPU) -> boundary conditions -> Newton scene

[Vehicle-scale hydrodynamics]
  Option A: XLB + Warp (real-time LBM in Newton)
  Option B: PhysicsNeMo FNO surrogate (pre-trained on OpenFOAM/Fluent data)
  Option C: AmgX-accelerated custom RANS in Warp

[RL Training Loop]
  Isaac Lab -> Newton physics -> fluid force query -> reward -> policy update
  Fluid force query = surrogate model (fast) or XLB LBM (moderate) or full CFD (slow, offline only)
```
