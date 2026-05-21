# SOTA GPU Fluid/CFD/Physics Simulation — Comprehensive Survey 2024-2026

Date: 2026-05-20
Scope: Exhaustive landscape scan covering NVIDIA ecosystem, academic SOTA, Disney Research, Google DeepMind, industrial/commercial, and open-source rising stars.

---

## Table of Contents

1. [NVIDIA Ecosystem](#1-nvidia-ecosystem)
2. [Academic SOTA (2024-2026)](#2-academic-sota-2024-2026)
3. [Disney Research](#3-disney-research)
4. [Google DeepMind](#4-google-deepmind)
5. [Industrial / Commercial](#5-industrial--commercial)
6. [Open Source Rising Stars](#6-open-source-rising-stars)
7. [Underwater-Specific Simulation](#7-underwater-specific-simulation)
8. [Hardware: NVIDIA Grace Blackwell](#8-hardware-nvidia-grace-blackwell)
9. [Cross-Cutting Trends](#9-cross-cutting-trends)

---

## 1. NVIDIA Ecosystem

### 1.1 NVIDIA Warp

| Field | Value |
|-------|-------|
| Name | NVIDIA Warp |
| URL | https://nvidia.github.io/warp/ |
| Year | Active 2022-2026 |
| Type | Open-source Python framework for GPU-accelerated differentiable simulation |

Key contributions:
- Python-first framework that compiles simulation kernels to efficient GPU code
- Auto-differentiable simulation kernels — no separate C++/CUDA codebase needed
- Rich primitives for physics simulation, robotics, geometry processing
- Integrates with Isaac Sim, Omniverse, and ML pipelines
- GTC 2026 session: "How to Use NVIDIA Warp to Build GPU-Accelerated Computational Physics" — https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81837/
- NERSC workshop (May 2025): "Building GPU-Accelerated Differentiable Simulations with NVIDIA Warp" — https://www.nersc.gov/news-and-events/calendar-of-events/nvidia-warp-python-may2025/
- Tutorial (Aug 2025): "Kinematics and Optimization Using NVIDIA Warp" — https://ckrapu.github.io/blog/2025/kinematics-and-optimization-warp/
- GitHub: https://github.com/NVIDIA/warp

### 1.2 NVIDIA Newton Physics Engine

| Field | Value |
|-------|-------|
| Name | Newton Physics Engine |
| URL | https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/ |
| Year | Announced March 2025 (GTC 2025), Newton 1.0 at GTC 2026 |
| Type | Open-source, GPU-accelerated physics engine for robotics |

Key contributions:
- Co-developed by NVIDIA, Google DeepMind, Disney Research
- Managed under the Linux Foundation
- Built on NVIDIA Warp
- Integrates MuJoCo Warp for parallel simulation (5-10x speedup over CPU MuJoCo)
- Available in NVIDIA Isaac Lab
- OpenUSD pipeline for robot learning
- GTC 2026 session: "An Introduction to the Newton Physics Engine for Robotics" — https://www.nvidia.com/en-us/on-demand/session/gtc26-s81613/
- SIGGRAPH 2025 session: "How Disney Droids Come to Life with Physics Simulation" — https://www.nvidia.com/en-us/on-demand/session/siggraph25-s12/
- CoRL 2025 press release: https://investor.nvidia.com/news/press-release-details/2025/NVIDIA-Accelerates-Robotics-Research-and-Development-With-New-Open-Models-and-Simulation-Libraries/default.aspx

### 1.3 NVIDIA PhysX 5 + Flow

| Field | Value |
|-------|-------|
| Name | PhysX 5 + Flow |
| URL | https://developer.nvidia.com/physx-sdk |
| Year | Fully open-sourced April 2025 |
| Type | Real-time physics + GPU fluid simulation |

Key contributions:
- PhysX 5 SDK fully open-source under same license as PhysX 4
- PhysX Flow: GPU-accelerated volumetric fluid simulation (smoke, fire, combustible fluid) — now fully open-source
- Fluid simulation features primarily GPU-accelerated (not in open-source CPU code)
- NVIDIA blog: https://developer.nvidia.com/blog/open-source-simulation-expands-with-nvidia-physx-5-release/
- CG Channel coverage: https://www.cgchannel.com/2025/04/nvidia-open-sources-physxs-gpu-simulation-code/

### 1.4 NVIDIA Isaac Sim

| Field | Value |
|-------|-------|
| Name | Isaac Sim 5.0 |
| URL | https://developer.nvidia.com/isaac-sim |
| Year | Active development 2025-2026 |
| Type | Robotics simulation platform |

Key contributions:
- Isaac Sim 5.0 supports simulating liquid spills, physical events in warehouse environments
- Fluid simulation via community efforts and extensions
- Open-source underwater sim example: https://github.com/leonlime/isaac_underwater
- Forum discussions on fluid/water simulation: https://forums.developer.nvidia.com/t/fluid-simulation-in-isaac-sim/364380
- PTC Onshape integration announced at GTC 2026: https://www.ptc.com/en/news/2026/ptc-announces-onshape-nvidia-isaac-sim-workflow

### 1.5 NVIDIA Modulus (PhysicsNeMo)

| Field | Value |
|-------|-------|
| Name | NVIDIA Modulus |
| URL | https://www.nvidia.com/en-us/developer/modulus/ |
| GitHub | https://github.com/NVIDIA/modulus |
| Year | Active development 2024-2026 |
| Type | Physics-informed machine learning framework |

Key contributions:
- Physics-Informed Neural Networks (PINNs) for solving PDEs
- Fourier Neural Operator (FNO) and neural operator architectures
- Modulus Sym — symbolic API for model development
- CFD applications: laminar/turbulent flow, conjugate heat transfer
- Integration with Omniverse for visualization
- Multi-GPU / multi-node training support
- Used in Omniverse Blueprint for CAE digital twins (see Section 1.6)

### 1.6 NVIDIA Omniverse Blueprint for CFD

| Field | Value |
|-------|-------|
| Name | Omniverse Blueprint for CAE / Digital Twins for Fluid Simulation |
| URL | https://github.com/NVIDIA-Omniverse-blueprints/digital-twins-for-fluid-simulation |
| Year | 2025 |
| Type | Reference workflow for real-time CFD digital twins |

Key contributions:
- Comprehensive reference workflow for external aerodynamic CFD simulations
- Combines CUDA, Modulus (PhysicsNeMo), and Omniverse
- Ansys is first ISV to adopt — applying to Ansys Fluent for accelerated CFD
- Up to 50x faster simulation showcased at GTC 2025
- Ansys OEMing Omniverse technology starting with CFD and autonomy (Aug 2025): https://www.ansys.com/news-center/press-releases/8-12-25-ansys-oems-omniverse
- NVIDIA news: https://nvidianews.nvidia.com/news/nvidia-announces-omniverse-real-time-physics-digital-twins-with-industry-software-leaders
- GTC 2025 session: https://www.nvidia.com/en-us/on-demand/session/gtc25-s73028/

---

## 2. Academic SOTA (2024-2026)

### 2.1 Differentiable Fluid Simulation

#### Adjoint Method for Differentiable Fluid Simulation on Flow Maps
- URL: https://dl.acm.org/doi/10.1145/3757377.3763903
- Year: 2025
- Venue: ACM SIGGRAPH / ACM Transactions on Graphics
- Contribution: Novel adjoint solver for differentiable fluid simulation based on bidirectional flow maps. Enables efficient gradient computation for fluid optimization and control tasks.

#### Hierarchical Differentiable Fluid Simulation
- URL: https://onlinelibrary.wiley.com/doi/10.1111/cgf.70226
- Year: 2025
- Venue: Computer Graphics Forum (Eurographics)
- Contribution: Two-step algorithm that significantly reduces memory usage for differentiable fluid simulation. Addresses high memory consumption in grid-based differentiable simulation.

#### PICT — Differentiable GPU-Accelerated Multi-Block PISO Solver
- URL: https://arxiv.org/abs/2505.16992
- Year: 2025
- Venue: arXiv preprint
- Contribution: Differentiable pressure-implicit solver coded in PyTorch with GPU support. Designed for simulation-coupled learning tasks in fluid dynamics.

#### NeuralFluid (NeurIPS 2024)
- URL: https://neurips.cc/virtual/2024/poster/95600
- Year: 2024
- Venue: NeurIPS 2024
- Contribution: Neural control and design of complex fluidic systems with dynamic solid boundaries via differentiable simulation.

#### Diff-FlowFSI — GPU-Optimized Differentiable CFD Platform
- URL: https://arxiv.org/html/2505.23940v1
- Year: 2025
- Venue: arXiv
- Contribution: GPU-accelerated, fully differentiable CFD platform for high-fidelity turbulence and fluid-structure interaction (FSI).

### 2.2 Neural Fluid Simulation / Deep Learning for CFD

#### Physics-Aware Neural Operator for High-Fidelity Fluid Dynamics
- URL: https://pubs.aip.org/aip/pof/article/37/11/115111/3371210/Physics-aware-neural-operator-for-high-fidelity
- Year: 2025
- Venue: Physics of Fluids (AIP)
- Contribution: Physics-aware extensions of FNO using spectral convolutions for high-fidelity fluid dynamics.

#### HUFNO — Hybrid U-Net + FNO for Large Eddy Simulation
- URL: https://arxiv.org/html/2504.13126v1
- Year: 2025
- Venue: arXiv
- Contribution: Hybrid architecture combining U-Net and FNO for mixed periodic and non-periodic boundary conditions in LES.

#### CFDONEval — Comprehensive Evaluation of Operator-Learning Models for CFD
- URL: https://www.ijcai.org/proceedings/2025/0640.pdf
- Year: 2025
- Venue: IJCAI 2025
- Contribution: Benchmark evaluating 12 operator-learning neural network models across 7 fluid simulation tasks.

#### torch-cfd — Neural Operator-Assisted Fluid Simulation (ICLR 2025)
- URL: https://github.com/scaomath/torch-cfd
- Year: 2025
- Venue: ICLR 2025
- Contribution: Spatiotemporal FNO training and evaluation via neural operator-assisted fluid simulation pipelines.

#### 3D Fluid Reconstruction from Single Video (CVPR 2025)
- URL: https://cvpr.thecvf.com/virtual/2025/poster/33449
- Year: 2025
- Venue: CVPR 2025
- Contribution: Reconstructing and predicting 3D fluid appearance and velocity from single video input.

### 2.3 Physics-Informed Neural Networks (PINNs) for Fluid

#### PINNs for Complex Fluids (Springer 2025)
- URL: https://link.springer.com/article/10.1007/s13367-025-00140-6
- Year: 2025
- Venue: Springer Nature
- Contribution: PINNs embedding physical laws into neural networks for complex fluid behaviors (forward and inverse problems).

#### Multi-Domain PINN (MDPINN) — NeurIPS 2025
- URL: https://neurips.cc/virtual/2025/poster/115781
- Year: 2025
- Venue: NeurIPS 2025
- Contribution: Multi-domain decomposition framework addressing scalability and generalization in large-scale PINN problems.

#### PINNs with Re-Initialization Strategy
- URL: https://www.nature.com/articles/s41598-025-99354-5
- Year: 2025
- Venue: Nature Scientific Reports
- Contribution: Novel "re-initialization" training strategy for improved fluid flow analysis with PINNs.

#### Enhancement of PINNs for Fluid Applications
- URL: https://pubs.aip.org/aip/pof/article/37/5/057130/3347333/Enhancement-of-physics-informed-neural-networks-in
- Year: 2025
- Venue: Physics of Fluids (AIP)
- Contribution: Novel deep learning algorithm within PINN architecture for enhanced fluid dynamics accuracy.

### 2.4 Fourier Neural Operator (FNO) for CFD

#### FNO for Fluid-Structure Interaction (FSI)
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0167278924000964
- Year: 2024
- Venue: Journal of Computational Physics
- Contribution: FNO-based FSI solver combining neural operator speed with FSI physics.

#### FNO for High-Resolution Fluid Flow Simulation
- URL: https://link.springer.com/article/10.1007/s13131-024-2453-1
- Year: 2024
- Venue: Journal of Hydrodynamics (Springer)
- Contribution: FNO as cost-effective, high-precision model for high-resolution fluid flow.

#### FNO for Real-Time 3D Urban Microclimate
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0360132323010909
- Year: 2024
- Venue: Building and Environment (ScienceDirect)
- Contribution: FNO achieving 25x faster simulation than traditional numerical solvers for 3D urban CFD.

### 2.5 Graph Neural Networks (GNNs) for Fluid

#### Masked GNN Pre-Training for CFD
- URL: https://arxiv.org/abs/2501.08738
- Year: 2025
- Venue: OpenReview / arXiv
- Contribution: Novel masked pre-training technique for GNNs applied to CFD problems.

#### Mesh-Based Super-Resolution with Multiscale GNN
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0045782525003445
- Year: 2025
- Venue: Computer Methods in Applied Mechanics and Engineering
- Contribution: GNN approach enabling mesh-based 3D super-resolution of fluid flows.

#### GNN Framework for Unsteady Fluid Flows
- URL: https://open.library.ubc.ca/media/download/pdf/24/1.0450307/4
- Year: 2025
- Venue: UBC thesis
- Contribution: Fast modeling of fluid flow for marine vessel design optimization using GNNs.

### 2.6 Lattice Boltzmann Method (LBM) on GPU

#### CooLBM — Collaborative Open-Source Reactive LBM
- URL: https://www.sciencedirect.com/science/article/pii/S0010465525002139
- Year: 2025
- Venue: Computer Physics Communications
- Contribution: Multi-CPU/GPU code for single and reactive flow simulations using LBM.

#### JAX-LaB — Differentiable Multiphase LBM Library
- URL: https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025MS005313
- Year: 2025
- Venue: Journal of Advances in Modeling Earth Systems (AGU)
- Contribution: JAX-based differentiable multiphase LBM library scalable across CPUs, GPUs, and distributed systems.

#### Multi-GPU Palabos LBM
- URL: https://arxiv.org/html/2506.09242
- Year: 2025
- Venue: arXiv
- Contribution: GPU port of Palabos Lattice Boltzmann library with multi-GPU scaling analysis.

#### NVIDIA Hybrid LBM-ML Solver (GTC 2024)
- URL: https://www.nvidia.com/en-us/on-demand/session/gtc24-s62237/
- Year: 2024
- Venue: NVIDIA GTC 2024
- Contribution: Hybrid Lattice-Boltzmann algorithm where ML model replaces parts of the algorithm. Fully differentiable.

### 2.7 SPH (Smoothed Particle Hydrodynamics) on GPU

#### Large-Scale Particle-Based Fluid Simulation (MDPI 2025)
- URL: https://www.mdpi.com/2076-3417/15/17/9706
- Year: 2025
- Venue: Applied Sciences (MDPI)
- Contribution: Modern GPU-accelerated SPH enabling millions of particles in real-time or near-real-time.

#### Enhancement of GPU-Accelerated SPH
- URL: https://www.sciencedirect.com/science/article/pii/S2590123025028634
- Year: 2025
- Venue: ScienceDirect
- Contribution: Physically accurate, large-scale, real-time fluid animations requiring million-scale particle systems.

#### WebGPU Fluid Simulations (Codrops Feb 2025)
- URL: https://tympanus.net/codrops/2025/02/26/webgpu-fluid-simulations-high-performance-real-time-rendering/
- Year: 2025
- Contribution: Browser-based high-performance fluid simulation using WebGPU; 180K particles at 2W.

### 2.8 MPM (Material Point Method) on GPU

#### Sparse-Memory-Encoding GPU-MPM Framework
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0266352X2500062X
- Year: 2025
- Venue: Computers and Geotechnics (ScienceDirect)
- Contribution: GPU-MPM with sparse memory encoding for large-scale granular material simulation.

#### GPUMPM — Open-Source CUDA MPM
- URL: https://github.com/kuiwuchn/GPUMPM
- Year: Ongoing
- Contribution: Full pipeline of MPM on GPU with sparse grid structure (particle-to-grid and grid-to-particle transfers).

#### Multi-GPU MPM for Debris-Fluid Simulation (Wiley)
- URL: https://onlinelibrary.wiley.com/doi/10.1002/nme.70210
- Year: 2025
- Venue: International Journal for Numerical Methods in Engineering
- Contribution: Multi-GPU MPM validation for debris-fluid scenarios with benchmark data.

#### JAX-MPM — Differentiable Meshfree Framework (2025)
- URL: https://arxiv.org/html/2507.04192v1
- Year: 2025
- Venue: arXiv
- Contribution: Differentiable Material Point Method in JAX; validated through 2D/3D dam-break and granular collapse benchmarks.

---

## 3. Disney Research

### 3.1 Kamino — GPU-Based Massively Parallel Solver

| Field | Value |
|-------|-------|
| Name | Kamino |
| URL | https://disneyresearch.github.io/kamino/ |
| Paper | https://arxiv.org/html/2603.16536v1 |
| Year | 2025-2026 |
| Type | GPU-accelerated physics solver for complex mechanical systems |

Key contributions:
- GPU-based physics solver for massively parallel simulation of heterogeneous, highly-coupled mechanical systems
- Specifically handles kinematic loops (notoriously hard for traditional solvers)
- Used to power Disney's robotic Olaf character and other droids
- Featured in Jensen Huang's GTC 2025 and GTC 2026 keynotes
- Integrated with Newton for reinforcement learning-based control of complex mechanical systems
- SIGGRAPH 2025 session with NVIDIA: https://www.nvidia.com/en-us/on-demand/session/siggraph25-s12/

### 3.2 Disney-NVIDIA-DeepMind Newton Collaboration

| Field | Value |
|-------|-------|
| Name | Newton (Disney Research contribution) |
| URL | https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning |
| Year | 2025 |

Key contributions:
- Disney Research co-develops Newton with NVIDIA and Google DeepMind
- Kamino simulator powers Disney Imagineering's robotic characters
- Focus on sim-to-real transfer for animatronic robots
- Moritz Baecher (Disney Research) leads the Kamino effort

---

## 4. Google DeepMind

### 4.1 MuJoCo Warp (MJWarp)

| Field | Value |
|-------|-------|
| Name | MuJoCo Warp |
| URL | https://github.com/google-deepmind/mujoco_warp |
| Docs | https://mujoco.readthedocs.io/en/latest/mjwarp/ |
| Year | 2025 |
| Type | GPU-optimized MuJoCo rewrite using NVIDIA Warp |

Key contributions:
- GPU-optimized implementation of MuJoCo physics simulator
- Written in NVIDIA Warp, optimized for NVIDIA hardware
- Achieves 5-10x speedup over CPU-based MuJoCo
- Designed for massively parallel simulation
- Foundation for Newton physics engine
- GTC 2025 session: https://www.nvidia.com/en-us/on-demand/session/gtc25-s72709/
- MuJoCo Playground integration: https://github.com/google-deepmind/mujoco_playground/discussions/197

### 4.2 MuJoCo (General)

| Field | Value |
|-------|-------|
| Name | MuJoCo |
| URL | https://mujoco.org/ |
| Year | Free and open-source since 2021 |
| Type | Advanced physics simulation for robotics research |

- DeepMind blog: https://deepmind.google/blog/opening-up-a-physics-simulator-for-robotics/
- Contact model accurately captures complex interactions for robotics
- MJX (JAX backend) and MJWarp (Warp backend) now available

---

## 5. Industrial / Commercial

### 5.1 Ansys Fluent GPU Solver

| Field | Value |
|-------|-------|
| Name | Ansys Fluent GPU Solver |
| URL | https://www.ansys.com/blog/new-era-ansys-fluent-computations |
| Year | 2025 (R1 and R2) |
| Type | Commercial GPU-accelerated CFD solver |

Key contributions:
- 2025 R1: GPU solver extended to combustion, acoustics, free surface
- 2025 R2: GPU solver now supports VOF and species transport
- Up to 8x speedup vs. dual-socket AMD EPYC 7543 CPU (Reddit user report)
- 110x acceleration on GH200 for large-scale CFD (4 weeks to 6 hours): https://investors.ansys.com/news-releases/news-release-details/ansys-accelerates-cfd-simulation-110x-nvidia-gh200-grace-hopper/
- 0.6 billion-cell external aerodynamics in 14 hours on 20 NVIDIA L40 GPUs
- Official hardware guide: https://innovationspace.ansys.com/knowledge/forums/topic/fluent-gpu-solver-hardware-buying-guide/
- OEMs NVIDIA Omniverse (Aug 2025): https://www.ansys.com/news-center/press-releases/8-12-25-ansys-oems-omniverse

### 5.2 Siemens Simcenter STAR-CCM+

| Field | Value |
|-------|-------|
| Name | Simcenter STAR-CCM+ |
| URL | https://developer.nvidia.com/blog/computational-fluid-dynamics-revolution-driven-by-gpu-acceleration/ |
| Year | 2022+ (ongoing GPU acceleration) |
| Type | Commercial GPU-accelerated CFD |

Key contributions:
- 20x speedup on NVIDIA A100 GPUs (STAR-CCM+ 2022.1 benchmark)
- GPU-accelerated solver for industrial CFD

### 5.3 Concepts NREC + ADS CFD

| Field | Value |
|-------|-------|
| Name | ADS CFD (GPU-accelerated) |
| URL | https://www.conceptsnrec.com/news/concepts-nrec-and-ads-cfd-announce-strategic-partnership-to-bring-blazing-fast-gpu-accelerated-cfd-to-turbomachinery-design |
| Year | 2025 |
| Type | Commercial GPU CFD for turbomachinery |

Key contributions:
- 15-120x faster performance vs. traditional CPU-based CFD
- Validated accuracy maintained

### 5.4 Cadence Fidelity CFD Platform

| Field | Value |
|-------|-------|
| Name | Cadence Fidelity CFD |
| URL | https://community.cadence.com/cadence_blogs_8/b/corporate-news/posts/nvidia-accelerated-compute-blackwell-collaboration-2025 |
| Year | 2025 |
| Type | Commercial GPU CFD |

Key contributions:
- Multi-billion cell simulations completed in under 24 hours on NVIDIA Grace Blackwell GB200 GPUs

### 5.5 NVIDIA Omniverse Blueprint (see Section 1.6)

---

## 6. Open Source Rising Stars

### 6.1 XLB — Accelerated Lattice Boltzmann (Autodesk Research)

| Field | Value |
|-------|-------|
| Name | XLB |
| URL | https://github.com/Autodesk/XLB |
| Paper | https://arxiv.org/html/2311.16080v3 |
| Year | 2024-2025 |
| Type | Differentiable, massively parallel LBM library |

Key contributions:
- **Warp backend** provides state-of-the-art single-GPU LBM performance
- **Neon backend** extends to multi-GPU (single-resolution)
- ~8x speedup on GH200 Grace Hopper Superchip vs. prior GPU baseline
- GTC 2025: Scaling up to **100 billion cells** — https://www.nvidia.com/en-us/on-demand/session/gtc25-s72057/
- NVIDIA developer blog: https://developer.nvidia.com/blog/autodesk-research-brings-warp-speed-to-computational-fluid-dynamics-on-nvidia-gh200/
- Differentiable LBM for physics-based ML
- Showcased at Supercomputing 2024

### 6.2 JAX-Fluids (TUM)

| Field | Value |
|-------|-------|
| Name | JAX-Fluids |
| URL | https://github.com/tumaer/JAXFLUIDS |
| Year | 2022-2025 |
| Type | Fully-differentiable CFD solver for compressible two-phase flows |

Key contributions:
- Fully-differentiable CFD solver written entirely in JAX
- 3D compressible single-phase and two-phase flows
- High-order Godunov-type schemes
- Automatic differentiation through entire simulation pipeline
- Used in TUM aerodynamics research

### 6.3 JAX-CFD (Google)

| Field | Value |
|-------|-------|
| Name | JAX-CFD |
| URL | https://github.com/google/jax-cfd |
| Year | 2021+ (experimental) |
| Type | Experimental research CFD framework |

Key contributions:
- Google research project for ML + automatic differentiation + GPU/TPU for CFD
- Foundation for JAX-Fluids and other differentiable CFD efforts

### 6.4 JANC — Differentiable Compressible Reacting Flow Solver

| Field | Value |
|-------|-------|
| Name | JANC |
| URL | https://www.sciencedirect.com/science/article/abs/pii/S0010465525004163 |
| Year | 2025 |
| Type | Differentiable solver for compressible reacting flows |

Key contributions:
- Extends JAX-Fluids to species transport equations and detailed thermodynamics
- Fills gaps in JAX-Fluids (which lacks species transport and detailed thermo)

### 6.5 Taichi Lang

| Field | Value |
|-------|-------|
| Name | Taichi Lang |
| URL | https://www.taichi-lang.org/ |
| Year | 2019-2025 |
| Type | Domain-specific language for high-performance parallel programming in Python |

Key contributions:
- GPU-accelerated DSL embedded in Python
- Now documented on AMD ROCm platform (growing GPU vendor support): https://rocm.docs.amd.com/projects/taichi/en/docs-25.11/what-is-taichi.html
- GeoTaichi: high-performance numerical simulator for multiscale geophysical problems — https://www.sciencedirect.com/science/article/abs/pii/S0010465524001425
- Taichi-LBM3D: GPU-accelerated Lattice Boltzmann in 3D
- Used at ETH Zurich for physically-based simulation course
- Alternatives compared: JAX, Warp, Phi-Flow, MLX

### 6.6 DiffTaichi

| Field | Value |
|-------|-------|
| Name | DiffTaichi |
| URL | https://github.com/taichi-dev/difftaichi |
| Paper | https://openreview.net/forum?id=B1eB5xSFvr |
| Year | 2020 (original), still relevant in 2025 |
| Type | Differentiable programming for physical simulation |

Key contributions:
- DiffSim GPU is 1.9x faster than JAX GPU on smoke simulation benchmark
- Code is 4.2x shorter than equivalent CUDA
- 10 differentiable physical simulators included
- Still used as baseline in 2025 differentiable simulation papers

### 6.7 FluidX3D

| Field | Value |
|-------|-------|
| Name | FluidX3D |
| URL | https://github.com/ProjectPhysX/FluidX3D |
| Year | Ongoing (active 2024-2025) |
| Type | Open-source Lattice Boltzmann CFD solver |

Key contributions:
- Single-GPU and CPU benchmarks measured in MLUPs/s
- Supports both single (float) and double precision
- Multi-GPU support with domain decomposition
- Cross-vendor GPU support (NVIDIA CUDA, AMD, Intel via OpenCL)

### 6.8 OpenFOAM GPU

| Field | Value |
|-------|-------|
| Name | OpenFOAM GPU (via SPUMA / AmgX) |
| URL | https://www.sciencedirect.com/science/article/abs/pii/S0010465525005107 |
| Year | 2025 |
| Type | Minimally invasive GPU porting of OpenFOAM |

Key contributions:
- SPUMA approach: >=90% efficiency with NVIDIA AmgX linear algebra solver
- Single A100 GPU competitive with multi-CPU setups
- Avoids full OpenFOAM rewrite
- PISO + PETSc + AMG + CGc shows best scalability
- Hivenet overview: https://www.hivenet.com/post/openfoam-gpu-state-of-play
- Active backends: AmgX, PETSc, Ginkgo for GPU offloading

### 6.9 GeoTaichi

| Field | Value |
|-------|-------|
| Name | GeoTaichi |
| URL | https://www.sciencedirect.com/science/article/abs/pii/S0010465524001425 |
| Year | 2024 |
| Type | Taichi-powered high-performance numerical simulator |

Key contributions:
- Open-source, multiscale geophysical problems
- GPU-accelerated computing for large-scale simulations

---

## 7. Underwater-Specific Simulation

### 7.1 OceanSim — GPU-Accelerated Underwater Robot Perception Simulator

| Field | Value |
|-------|-------|
| Name | OceanSim |
| URL | https://github.com/umfieldrobotics/OceanSim |
| Paper | https://arxiv.org/html/2503.01074v2 |
| Project | https://umfieldrobotics.github.io/OceanSim/ |
| Year | 2025 |
| Type | High-fidelity underwater simulation built on NVIDIA Isaac Sim |

Key contributions:
- GPU-accelerated real-time ray tracing for underwater perception
- Physics-based rendering of light attenuation, scattering, turbidity, caustics, color absorption
- Synthetic training data generation for AUV/ROV perception
- Built on NVIDIA Isaac Sim
- Open-source from University of Michigan Field Robotics group

### 7.2 isaac_underwater — Isaac Sim Underwater Examples

| Field | Value |
|-------|-------|
| Name | isaac_underwater |
| URL | https://github.com/leonlime/isaac_underwater |
| Year | 2025 |
| Type | Open-source underwater simulation examples for Isaac Sim |

Key contributions:
- Water and underwater simulation examples
- Floating box simulations and underwater physics tests

### 7.3 NVIDIA Omniverse for Ocean Digital Twins

| Field | Value |
|-------|-------|
| Name | Digital Ocean Collaboration (Omniverse) |
| URL | https://www.nvidia.com/en-us/on-demand/session/gtc25-s73656/ |
| Year | 2025 |
| Venue | GTC 2025 |

Key contributions:
- Using NVIDIA Omniverse to enhance digital collaboration in ocean exploration, research, and conservation
- Digital twin technology for marine environments

---

## 8. Hardware: NVIDIA Grace Blackwell

### 8.1 Blackwell for CFD/CAE

| Field | Value |
|-------|-------|
| Name | NVIDIA Grace Blackwell |
| URL | https://nvidianews.nvidia.com/news/nvidia-blackwell-accelerates-computer-aided-engineering-software-by-orders-of-magnitude-for-real-time-digital-twins |
| Year | 2025 |
| Type | GPU architecture for simulation acceleration |

Key performance claims:
- Up to **50x faster simulation** on Blackwell GPUs for CAE workloads
- Cadence Fidelity CFD: Multi-billion cell simulations in under 24 hours on GB200
- Ansys Fluent: Blackwell support incoming
- Blackwell system specs (GTC 2025 keynote): 2,592 Grace CPU cores, 72 Blackwell GPUs, 130 TB/s bandwidth, 1.1 EF FP4
- ~5.2M Blackwell GPUs shipped in 2025; Rubin (next-gen) coming 2026
- RTX PRO Blackwell workstations from Dell/Lenovo/HP at GTC 2026
- NVIDIA blog: https://www.engineering.com/nvidia-boasts-50x-faster-simulation-at-gtc-2025/

### 8.2 GH200 Grace Hopper Superchip

Key performance claims:
- Ansys Fluent: 110x acceleration on GH200 (4 weeks to 6 hours)
- Autodesk XLB + Warp: ~8x speedup on GH200 vs. prior GPU baseline

---

## 9. Cross-Cutting Trends

### 9.1 Differentiable Everything
Every major simulation framework now supports automatic differentiation:
- NVIDIA Warp (native)
- JAX-Fluids, JAX-CFD, JAX-LaB, JAX-MPM (JAX ecosystem)
- DiffTaichi (Taichi ecosystem)
- PICT (PyTorch-based)
- Diff-FlowFSI (2025)

### 9.2 GPU-First Architectures
- Blackwell GPUs delivering 50x+ speedups for CFD
- All new solvers designed GPU-first (not CPU-ported)
- Multi-GPU scaling now standard (XLB, OpenFOAM, Palabos)
- 100 billion cell simulations demonstrated (XLB at GTC 2025)

### 9.3 Neural Operators + Physics Fusion
- FNO, GNN, PINN approaches maturing rapidly
- Hybrid architectures (HUFNO = U-Net + FNO)
- Comprehensive benchmarks (CFDONEval at IJCAI 2025)
- 10-25x speedup over traditional solvers while maintaining fidelity

### 9.4 Open-Source Explosion
- PhysX 5 + Flow fully open-source (April 2025)
- Newton open-source under Linux Foundation
- XLB, FluidX3D, Taichi, JAX-Fluids all open-source
- OpenFOAM GPU porting via AmgX

### 9.5 Robotics-Driven Physics
- Newton ecosystem (NVIDIA + DeepMind + Disney) driving physics simulation for robotics
- MuJoCo Warp enables 5-10x parallel simulation speedup
- Kamino handles kinematic loops for complex robot mechanisms
- Sim-to-real transfer becoming mainstream

### 9.6 Underwater Simulation Gap
- OceanSim is the only significant GPU-native underwater sim (built on Isaac Sim)
- No dedicated GPU fluid solver specifically optimized for underwater/hydrostatic scenarios
- Major gap: underwater digital twins still rely on generic CFD tools
- Opportunity for OceanScale to fill this gap

---

## Performance Comparison Matrix

| Solver | Method | GPU Perf | Scale | Differentiable | Year |
|--------|--------|----------|-------|----------------|------|
| Ansys Fluent GPU | Finite Volume | 8-110x vs CPU | 0.6B cells | No | 2025 |
| Siemens STAR-CCM+ | Finite Volume | 20x on A100 | Industrial | No | 2022+ |
| XLB (Warp) | Lattice Boltzmann | ~8x on GH200 | 100B cells | Yes | 2025 |
| JAX-Fluids | Finite Volume (Godunov) | GPU-accelerated | 3D two-phase | Yes | 2025 |
| PhysX Flow | Eulerian/Grid | Real-time | VFX scale | No | 2025 |
| FluidX3D | Lattice Boltzmann | Multi-GPU | Research | No | 2025 |
| OpenFOAM+AmgX | Finite Volume | >=90% eff. on A100 | HPC | No | 2025 |
| Taichi/GeoTaichi | Multi-method | GPU-accelerated | Geophysical | Yes | 2024 |
| DiffTaichi | Multi-method | 1.9x vs JAX | Research | Yes | 2020+ |
| CooLBM | Lattice Boltzmann | Multi-CPU/GPU | Reactive flows | No | 2025 |
| JAX-LaB | Lattice Boltzmann | Multi-GPU/distributed | Multiphase | Yes | 2025 |
| FNO-based | Neural Operator | 10-25x vs traditional | Varies | Yes | 2025 |
| GNN-based | Neural Operator | GPU-accelerated | Mesh-based | Varies | 2025 |
| Newton/MuJoCo Warp | Rigid body + contact | 5-10x vs CPU | Robotics | Partial | 2025 |
| Kamino | Rigid body + kinematic loops | GPU-parallel | Robotics | Partial | 2025 |

---

*Sources: All URLs verified via web search on 2026-05-20. No fabricated URLs.*
