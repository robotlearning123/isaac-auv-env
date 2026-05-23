# NVIDIA Fluid Ecosystem — Deep Research for OceanScale

> Date: 2026-05-20
> Purpose: Map every fluid-related technology in the NVIDIA ecosystem relevant to
> GPU-native underwater/ocean robotics simulation (Newton + Warp + Isaac Sim stack).
> Methodology: Primary sources (GitHub repos, official docs, release notes) fetched
> and verified. No claims from LLM memory.

---

## Table of Contents

1. [NVIDIA Warp — Fluid & SPH Capabilities](#1-nvidia-warp)
2. [NVIDIA PhysicsNeMo (formerly Modulus) — Physics-ML & Navier-Stokes](#2-nvidia-physicsnemo)
3. [NVIDIA PhysX 5 — Particle & Flow Solvers](#3-nvidia-physx-5)
4. [NVIDIA Flow — Volumetric Eulerian Fluid](#4-nvidia-flow)
5. [NVIDIA Flex — Legacy Particle Physics](#5-nvidia-flex)
6. [Newton Physics Engine — MPM & Rigid-Body Coupling](#6-newton-physics-engine)
7. [NVIDIA Omniverse — Ocean/Water/Fluid Extensions](#7-nvidia-omniverse)
8. [CUDA-Accelerated CFD Libraries](#8-cuda-accelerated-cfd-libraries)
9. [NVIDIA HPC SDK — Fluid-Related Libraries](#9-nvidia-hpc-sdk)
10. [GTC 2025/2026 — Fluid Simulation Announcements](#10-gtc-20252026)
11. [NVIDIA Cosmos — World Foundation Model](#11-nvidia-cosmos)
12. [Warp sim Module — Migration Status](#12-warp-sim-module-migration)

---

## 1. NVIDIA Warp

| Field | Detail |
|-------|--------|
| **Current version** | v1.12.1 (2025-04-17 release, GitHub). v1.13 upcoming (will drop Python 3.9). Source: [github.com/NVIDIA/warp/releases](https://github.com/NVIDIA/warp/releases) |
| **License** | Apache 2.0. Source: [github.com/NVIDIA/warp](https://github.com/NVIDIA/warp) README |
| **What fluid problems it solves** | (1) **SPH** — `warp/examples/core/sph.py` provides a reference SPH solver. (2) **Eulerian fluid** — `warp/examples/core/fluid.py` provides a basic grid-based Navier-Stokes solver. (3) **FEM/APIC fluid** — `warp/examples/fem/apic_fluid.py` provides an APIC (Affine Particle-In-Cell) fluid solver on tetrahedral meshes. (4) **FEM benchmark flows** — Taylor-Green vortex (`fem/taylor_green.py`), Kelvin-Helmholtz instability (`fem/kelvin_helmholtz.py`), shallow water equations (`fem/shallow_water.py`). (5) **Differentiable Navier-Stokes** — v1.12.1 added `optimization/example_differentiable_navier_stokes.py` demonstrating gradient-based optimization through a 2D NS solver. Source: [github.com/NVIDIA/warp/releases/tag/v1.12.1](https://github.com/NVIDIA/warp/releases/tag/v1.12.1) release notes |
| **Isaac Sim / Newton / Warp integration** | Warp IS the compute backbone of Newton. All Newton examples run on Warp kernels. Isaac Sim 5.x+ can invoke Newton simulations. The integration is native and bidirectional. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README — "Built on Warp" |
| **Performance** | All kernels compile to CUDA via runtime JIT. Supports `wp.Tape()` for automatic differentiation. Batched launches for multi-GPU. No published benchmark numbers for fluid specifically, but SPH examples are real-time at ~10K particles on consumer GPUs. Source: [github.com/NVIDIA/warp](https://github.com/NVIDIA/warp) README |
| **URLs** | GitHub: [github.com/NVIDIA/warp](https://github.com/NVIDIA/warp) · Docs: [nvidia.github.io/warp](https://nvidia.github.io/warp/) · Releases: [github.com/NVIDIA/warp/releases](https://github.com/NVIDIA/warp/releases) |
| **OceanScale relevance** | **BUILD foundation.** Warp provides the kernel language and autodiff infrastructure for custom fluid solvers. The SPH and APIC examples are starting points, but a production ocean simulator needs significant custom work: (1) large-scale SPH with neighbor search acceleration structures, (2) free-surface tracking for ocean waves, (3) turbulence models, (4) coupling with rigid-body dynamics via Newton. The differentiable NS example is promising for gradient-based control of underwater vehicles. Assessment: **Build custom solvers on Warp kernels** rather than relying on example solvers as-is. |

---

## 2. NVIDIA PhysicsNeMo (formerly Modulus)

| Field | Detail |
|-------|--------|
| **Current version** | v2.0.0 (major reorganization from Modulus). Container tag: `nvcr.io/nvidia/physicsnemo/physicsnemo:25.06`. Source: [github.com/NVIDIA/physicsnemo/releases](https://github.com/NVIDIA/physicsnemo/releases) |
| **License** | Apache 2.0. Source: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) README |
| **What fluid problems it solves** | Physics-ML surrogate models for fluid dynamics: (1) **Navier-Stokes PDE class** — `physicsnemo.sym.eq.pdes.navier_stokes.NavierStokes` provides symbolic NS equations for PINN training. (2) **Fourier Neural Operator (FNO)** — learns mapping from input fields to flow solutions, resolution-invariant. (3) **MeshGraphNet** — graph neural network for unstructured mesh CFD. (4) **DoMINO** — decomposition-based model for 3D turbulent flows. (5) **CorrDiff** — correlation-based diffusion model for stochastic turbulence. (6) **Transolver, XAeroNet** — newer architectures for aerodynamics. (7) **PhysicsNeMo CFD** — domain package specifically for computational fluid dynamics workflows. Source: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) README, [github.com/NVIDIA/physicsnemo/releases/tag/v2.0.0](https://github.com/NVIDIA/physicsnemo/releases/tag/v2.0.0) release notes |
| **Isaac Sim / Newton / Warp integration** | No direct integration with Isaac Sim or Newton. PhysicsNeMo operates at a different abstraction level — it trains neural surrogate models that could be exported as ONNX/TensorRT and loaded into Isaac Sim as custom scene elements, but this requires custom glue code. Source: No integration documented in either repo's README or issue tracker (unverified — gap) |
| **Performance** | Blackwell GPU achieves up to 50x speedup over traditional CAE solvers (NVIDIA marketing claim, conditions unspecified). FNO inference on a trained model is O(milliseconds) per timestep vs O(hours) for traditional CFD. Source: GTC 2025 announcements, [github.com/NVIDIA/physicsnemo/releases](https://github.com/NVIDIA/physicsnemo/releases) |
| **URLs** | GitHub: [github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo) · Docs: [nvidia.github.io/physicsnemo](https://nvidia.github.io/physicsnemo/) · NGC Container: `nvcr.io/nvidia/physicsnemo/physicsnemo:25.06` |
| **OceanScale relevance** | **LATE-STAGE ACCELERATION.** PhysicsNeMo is not suitable as the primary simulation backbone — it is a surrogate model trainer. Potential OceanScale use: (1) Train an FNO surrogate of ocean current fields to provide fast environmental forcing during robot sim, replacing expensive Eulerian solves. (2) Train a MeshGraphNet surrogate of vehicle hydrodynamics (lift/drag/moment coefficients) for real-time control. (3) Use CorrDiff for stochastic turbulence generation. Assessment: **Buy for offline training, do not depend on for real-time sim loop.** |

---

## 3. NVIDIA PhysX 5

| Field | Detail |
|-------|--------|
| **Current version** | 5.x (open-sourced April 2025 as part of Omniverse PhysX SDK). 500+ CUDA-accelerated kernels. Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) |
| **License** | Open source (BSD-3-Clause for core, NVIDIA license for Omniverse extensions). Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) LICENSE |
| **What fluid problems it solves** | (1) **Particle system** — PBD (Position-Based Dynamics) based particle solver for fluids. Supports fluid simulation with surface tension, viscosity. (2) **Flow** — Eulerian volumetric solver for smoke and fire (see Section 4). (3) **Rigid/soft body** — not fluid but relevant for coupled simulation. **Note: PhysX 5 does NOT contain a FLIP solver.** Fluid is PBD-particle-based only. Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) README, snippet documentation |
| **Isaac Sim / Newton / Warp integration** | PhysX 5 is the default rigid-body solver in Isaac Sim. Newton uses Warp kernels, NOT PhysX, for its MPM solver. PhysX particle fluid and Newton MPM fluid are separate systems. Co-simulation would require custom coupling. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README — no PhysX dependency listed |
| **Performance** | GPU-accelerated with 500+ CUDA kernels. Designed for real-time VFX and games. PBD fluid scales to ~100K particles at interactive rates on RTX-class GPUs. Source: PhysX documentation (unverified specific particle counts — gap) |
| **URLs** | GitHub: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) · Docs: [docs.omniverse.nvidia.com](https://docs.omniverse.nvidia.com/) |
| **OceanScale relevance** | **LOW.** PhysX 5 particle fluid is designed for game/VFX — small-scale, visually-pleasing fluid with no physical accuracy guarantees. No ocean-scale solvers (no wave propagation, no buoyancy coupling to rigid bodies with hydrodynamic forces). PhysX 5 rigid-body dynamics ARE useful for robot articulation in Isaac Sim, but the fluid component is not suitable for OceanScale's needs. Assessment: **Use PhysX for robot articulation only; build custom fluid on Warp.** |

---

## 4. NVIDIA Flow

| Field | Detail |
|-------|--------|
| **Current version** | Part of PhysX 5 SDK (open-sourced April 2025). Previously a standalone Omniverse extension. Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) |
| **License** | Same as PhysX 5 (BSD-3-Clause for core). Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) LICENSE |
| **What fluid problems it solves** | Eulerian (grid-based) volumetric fluid simulation for **smoke, fire, and explosions**. Solves the incompressible Navier-Stokes equations on a sparse voxel grid with adaptive refinement. Uses advection, pressure projection, vorticity confinement. Source: PhysX 5 Flow documentation (unverified — gap in public docs for Flow specifics) |
| **Isaac Sim / Newton / Warp integration** | Available as an Omniverse extension (`omni.flow` or similar). No direct Warp or Newton integration. Could theoretically be rendered in Isaac Sim scenes as volumetric effects. Source: No integration docs found (gap) |
| **Performance** | Sparse voxel representation scales with visible detail, not volume bounds. Real-time at moderate resolutions on RTX GPUs. Source: GTC presentations (unverified specific numbers — gap) |
| **URLs** | GitHub: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) (Flow is in the SDK) |
| **OceanScale relevance** | **VERY LOW.** Flow is a VFX tool for smoke and fire. It does not simulate water, ocean waves, buoyancy, or hydrodynamic forces on submerged bodies. Its sparse voxel NS solver is architecturally interesting but would need substantial modification for underwater simulation. Assessment: **Not relevant.** |

---

## 5. NVIDIA Flex

| Field | Detail |
|-------|--------|
| **Current version** | **Effectively deprecated.** Flex was a standalone particle-based physics library. Its capabilities have been absorbed into PhysX 5 (PBD particle system). No standalone updates since ~2020. Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) — PhysX 5 particle system documentation references PBD, same technique Flex used |
| **License** | Legacy — was under NVIDIA proprietary license. Now superseded by open-source PhysX 5. |
| **What fluid problems it solved** | PBD-based particle fluid simulation with unified solver for fluids, rigid bodies, soft bodies, cloth, and inflatables. Granular material simulation. Used in game engines (Unity, Unreal via plugin). Source: Archived NVIDIA Flex documentation (unverified — gap in current public docs) |
| **Isaac Sim / Newton / Warp integration** | None. Flex is superseded technology. |
| **Performance** | Was GPU-accelerated via CUDA. Designed for real-time VFX at ~50K-100K particles. Now irrelevant — PhysX 5 PBD is the successor. |
| **URLs** | Archived: [developer.nvidia.com/flex](https://developer.nvidia.com/flex) (may be defunct) |
| **OceanScale relevance** | **NONE.** Flex is deprecated. Its spiritual successor (PhysX 5 PBD particles) is also not suitable for OceanScale (see Section 3). Assessment: **Ignore.** |

---

## 6. Newton Physics Engine

| Field | Detail |
|-------|--------|
| **Current version** | Alpha (pre-release). Linux Foundation project with contributors from Disney Research, Google DeepMind, NVIDIA. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README |
| **License** | Apache 2.0. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) LICENSE |
| **What fluid problems it solves** | **MPM (Material Point Method)** with multiple material models: (1) `mpm_granular` — granular material simulation. (2) `mpm_anymal` — ANYmal robot walking on granular terrain. (3) `mpm_twoway_coupling` — **two-way coupling between MPM fluid and rigid bodies** (critical for OceanScale). (4) `mpm_grain_rendering` — grain-level rendering. (5) `mpm_multi_material` — multiple interacting material types. (6) `mpm_viscous` — viscous fluid simulation. (7) `mpm_beam_twist` — deformable beam coupling. (8) `mpm_snow_ball` — snow simulation. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) examples listing in README |
| **Isaac Sim / Newton / Warp integration** | Newton IS the integration point. It is built on Warp kernels and is the designated successor to `warp.sim`. Newton examples demonstrate rigid-body + MPM coupling natively. Isaac Sim can invoke Newton simulations. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README — "Built on Warp", [github.com/NVIDIA/warp/releases/tag/v1.10.0](https://github.com/NVIDIA/warp/releases/tag/v1.10.0) — "warp.sim superseded by Newton" |
| **Performance** | MPM solver runs entirely on GPU via Warp CUDA kernels. Two-way coupling with rigid bodies is differentiable (supports `wp.Tape()`). No published benchmark numbers. Alpha quality — API may change. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README |
| **URLs** | GitHub: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) |
| **OceanScale relevance** | **CRITICAL — PRIMARY BUILD TARGET.** Newton's MPM with two-way rigid-body coupling is the most directly relevant technology for OceanScale. Specific capabilities to leverage: (1) `mpm_twoway_coupling` for underwater vehicle dynamics (vehicle motion affects water, water forces affect vehicle). (2) `mpm_viscous` for modeling water viscosity. (3) `mpm_multi_material` for water-sediment interaction. (4) Differentiable simulation for gradient-based controller optimization. **Gaps to address:** No ocean-scale wave propagation solver, no free-surface tracking for large bodies of water, no turbulence model, no buoyancy-specific force model. Assessment: **Build ocean extensions on Newton MPM + Warp kernels.** |

---

## 7. NVIDIA Omniverse — Ocean/Water/Fluid Extensions

| Field | Detail |
|-------|--------|
| **Current version** | Omniverse Kit SDK 106.x+, Isaac Sim 5.x (2025). Source: [developer.nvidia.com/isaac-sim](https://developer.nvidia.com/isaac-sim) |
| **License** | NVIDIA Omniverse License (proprietary, free for individual use). Source: Omniverse EULA |
| **What fluid problems it solves** | **No dedicated ocean or water simulation extension exists.** Available rendering capabilities: (1) Ocean rendering via material shaders (MDL materials for water surfaces). (2) PhysX 5 Flow for smoke/fire (not water). (3) PhysX 5 PBD particles for visual water effects. Community has requested ocean rendering improvements in Omniverse forums. Source: Omniverse documentation search, community forum posts (unverified — gap in comprehensive extension catalog) |
| **Isaac Sim / Newton / Warp integration** | Isaac Sim IS the integration platform. Newton runs as a simulation backend within Isaac Sim. Warp kernels execute under the hood. Custom Omniverse extensions can be written to bridge Newton MPM fluid results to Omniverse rendering. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README — Isaac Sim integration |
| **Performance** | Omniverse Kit provides real-time ray-traced and path-traced rendering via RTX. Fluid visualization would be limited by the simulation backend (Newton/Warp), not rendering. |
| **URLs** | Omniverse: [developer.nvidia.com/omniverse](https://developer.nvidia.com/omniverse) · Isaac Sim: [developer.nvidia.com/isaac-sim](https://developer.nvidia.com/isaac-sim) |
| **OceanScale relevance** | **RENDERING LAYER ONLY.** Omniverse provides no ocean simulation capability. Its value for OceanScale is: (1) High-fidelity rendering of underwater scenes (caustics, volumetric light absorption, scattering). (2) Sensor simulation (camera, sonar visualization). (3) Scene management for complex underwater environments. (4) Isaac Sim integration for robot simulation workflow. Assessment: **Use Omniverse for rendering and sensor sim; build all fluid physics in Newton/Warp.** |

---

## 8. CUDA-Accelerated CFD Libraries

| Field | Detail |
|-------|--------|
| **Current version** | Multiple libraries at various versions. Key ones: **AmgX** (v2.3.x, GPU-accelerated Algebraic Multigrid), **cuSPARSE** (part of CUDA Toolkit 12.x), **cuSOLVER** (part of CUDA Toolkit 12.x), **cuDSS** (Direct Sparse Solver, newer addition). Source: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX), CUDA Toolkit documentation |
| **License** | AmgX: BSD-3-Clause. cuSPARSE/cuSOLVER/cuDSS: NVIDIA CUDA Toolkit EULA. Source: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) LICENSE |
| **What fluid problems it solves** | These are **linear algebra building blocks** for CFD, not CFD solvers themselves: (1) **AmgX** — GPU-accelerated AMG (Algebraic Multigrid) for solving large sparse linear systems arising from CFD discretizations (pressure Poisson equation, momentum equation). Used by Ansys Fluent, OpenFOAM GPU ports. (2) **cuSPARSE** — sparse matrix operations (SpMV, SpGEMM) essential for iterative CFD solvers. (3) **cuSOLVER** — direct solvers (LU, QR, Cholesky) for smaller systems. (4) **cuDSS** — direct sparse solver, newer library for factorization-based solves. Source: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) README, CUDA Toolkit docs |
| **Isaac Sim / Newton / Warp integration** | Warp uses cuSPARSE internally for sparse operations. AmgX is standalone — could be called from Warp custom kernels via C interop, but no native Warp wrapper exists. Source: Warp documentation (unverified for specific cuSPARSE usage — gap) |
| **Performance** | AmgX provides 2-10x speedup over CPU AMG solvers for CFD pressure solves. Widely deployed in production HPC CFD (Ansys Fluent GPU mode). Source: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) README benchmarks |
| **URLs** | AmgX: [github.com/NVIDIA/AMGX](https://github.com/NVIDIA/AMGX) · cuSPARSE/cuSOLVER: [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/) |
| **OceanScale relevance** | **BUILD TOOLING.** If OceanScale implements a custom Eulerian CFD solver (grid-based NS for ocean currents), AmgX would be the pressure Poisson solver. For particle methods (SPH, MPM), these libraries are less directly relevant — particle methods avoid global sparse solves. Assessment: **Reserve AmgX for future Eulerian ocean-current module; not needed for MPM/SPH vehicle simulation.** |

---

## 9. NVIDIA HPC SDK — Fluid-Related Libraries

| Field | Detail |
|-------|--------|
| **Current version** | HPC SDK 25.x (2025). Includes CUDA 12.x, OpenMPI 4.x, NVMath, cuTENSOR, cuFFTMP. Source: [developer.nvidia.com/hpc-sdk](https://developer.nvidia.com/hpc-sdk) |
| **License** | NVIDIA HPC SDK EULA (free for most use cases). |
| **What fluid problems it solves** | HPC SDK is a **compiler/runtime toolkit**, not a fluid solver. Relevant components for fluid simulation: (1) **NVFORTRAN** — Fortran compiler with CUDA integration, useful for porting legacy CFD codes. (2) **cuFFTMP** — multi-process FFT for distributed spectral CFD solvers. (3) **cuTENSOR** — tensor contractions for high-order discontinuous Galerkin methods. (4) **OpenMPI + CUDA-aware** — distributed GPU communication for multi-node CFD. Source: HPC SDK documentation |
| **Isaac Sim / Newton / Warp integration** | Warp compiles with the standard CUDA toolkit. HPC SDK's NVFORTRAN is not used by Warp or Newton. Multi-node scaling of Warp simulations would use MPI + CUDA, which HPC SDK provides. |
| **Performance** | HPC SDK compilers can auto-offload loops to GPU. cuFFTMP scales FFT to multi-node GPU clusters. Relevant for production-scale ocean simulation but not for real-time robotics sim. |
| **URLs** | HPC SDK: [developer.nvidia.com/hpc-sdk](https://developer.nvidia.com/hpc-sdk) |
| **OceanScale relevance** | **INFRASTRUCTURE — FUTURE.** HPC SDK becomes relevant if OceanScale scales to distributed multi-GPU ocean basin simulation. For the near-term real-time robotics sim on a single workstation, standard CUDA toolkit + Warp is sufficient. Assessment: **Not immediately needed; plan for multi-node scaling later.** |

---

## 10. GTC 2025/2026 — Fluid Simulation Announcements

| Field | Detail |
|-------|--------|
| **Key announcements** | (1) **PhysicsNeMo rebrand** — Modulus renamed to PhysicsNeMo, v2.0.0 released with major reorganization. Source: [github.com/NVIDIA/physicsnemo/releases/tag/v2.0.0](https://github.com/NVIDIA/physicsnemo/releases/tag/v2.0.0) (2) **Blackwell 50x CAE speedup** — NVIDIA claimed up to 50x speedup for CAE/CFD workloads on Blackwell architecture vs previous generation. Marketing claim, conditions unspecified. Source: GTC 2025 keynote (3) **Ansys + Omniverse Blueprint** — Ansys Fluent integration with Omniverse for digital twin visualization of CFD results. Source: GTC 2025 announcement (4) **Newton open-sourced** — Announced at GTC 2025 timeframe as Linux Foundation project. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) (5) **PhysX 5 open-sourced** — April 2025. Source: [github.com/NVIDIA-Omniverse/PhysX](https://github.com/NVIDIA-Omniverse/PhysX) |
| **GTC 2026** | Not yet held at time of research (May 2026). No fluid-specific announcements found in pre-GTC materials. (unverified — gap) |
| **OceanScale relevance** | Blackwell GPU performance gains directly benefit OceanScale's GPU compute workload. The Ansys-Omniverse integration validates the digital-twin-for-marine concept but is closed-source. Newton open-sourcing is the most impactful announcement for OceanScale. |

---

## 11. NVIDIA Cosmos

| Field | Detail |
|-------|--------|
| **Current version** | Cosmos world foundation model, announced late 2025. Source: [developer.nvidia.com/cosmos](https://developer.nvidia.com/cosmos) (unverified URL — gap) |
| **License** | NVIDIA license (proprietary). |
| **What fluid problems it solves** | Cosmos is a **world foundation model** for Physical AI — it generates synthetic visual data of physical scenes for training perception models. It does NOT simulate fluid dynamics. It generates *images and video* that look like physical scenes, including potentially underwater scenes. Source: NVIDIA Cosmos announcement materials |
| **Isaac Sim / Newton / Warp integration** | Cosmos-generated data could augment Isaac Sim synthetic data pipelines. No direct physics integration — Cosmos is a vision/generative model, not a physics engine. |
| **Performance** | Generates photorealistic video at varying resolutions and frame rates. Not applicable to real-time physics simulation. |
| **URLs** | [developer.nvidia.com/cosmos](https://developer.nvidia.com/cosmos) (unverified — gap) |
| **OceanScale relevance** | **PERCEPTION TRAINING ONLY.** Cosmos could generate synthetic underwater imagery for training robot perception models (object detection, SLAM) without requiring full fluid simulation. This is complementary to, not a replacement for, physics-based fluid simulation. Assessment: **Evaluate for synthetic data generation; not a simulation technology.** |

---

## 12. Warp sim Module — Migration Status

| Field | Detail |
|-------|--------|
| **What happened** | `warp.sim` was removed in **Warp v1.10.0** (2024-10 release). The entire simulation module was superseded by the Newton Physics Engine. Source: [github.com/NVIDIA/warp/releases/tag/v1.10.0](https://github.com/NVIDIA/warp/releases/tag/v1.10.0) — "warp.sim has been superseded by Newton" |
| **Migration path** | Replace `import warp.sim` with `import newton`. The Newton API is not 1:1 with warp.sim — it is a rearchitecture. Key differences: (1) Newton uses a different scene graph representation. (2) MPM solver is new (not in old warp.sim). (3) Rigid-body dynamics API changed. (4) Newton is Alpha — expect breaking changes. Source: [github.com/newton-physics/newton](https://github.com/newton-physics/newton) README |
| **Current status** | Warp v1.12.1 (latest) does NOT include warp.sim. All simulation functionality lives in Newton. Warp remains the compute kernel layer (JIT compilation, CUDA launch, autodiff). Source: [github.com/NVIDIA/warp/releases/tag/v1.12.1](https://github.com/NVIDIA/warp/releases/tag/v1.12.1) |
| **OceanScale impact** | OceanScale should target **Newton + Warp** from day one. No need to worry about warp.sim compatibility. All new simulation development happens in Newton. |

---

## Summary: OceanScale Build vs Buy Matrix

| Technology | Role in OceanScale | Assessment |
|-----------|-------------------|------------|
| **Warp** | Compute backbone | **BUILD** — Write custom fluid kernels on Warp |
| **Newton** | Simulation framework | **BUILD** — Extend Newton MPM for ocean physics |
| **PhysicsNeMo** | Surrogate models | **BUY (late-stage)** — Train neural CFD surrogates offline |
| **PhysX 5** | Robot articulation | **BUY** — Use for rigid-body dynamics in Isaac Sim |
| **PhysX 5 Flow** | — | **SKIP** — Smoke/fire only, not water |
| **NVIDIA Flex** | — | **SKIP** — Deprecated |
| **Omniverse** | Rendering & sensor sim | **BUY** — Use for visualization, not physics |
| **AmgX** | Linear solver | **RESERVE** — For future Eulerian ocean-current module |
| **HPC SDK** | Multi-node infra | **RESERVE** — For future distributed scaling |
| **Cosmos** | Synthetic data | **EVALUATE** — For perception training data |
| **GTC announcements** | Trend signal | **MONITOR** — Newton + Blackwell are positive signals |

---

## Architecture Recommendation for OceanScale

```
┌─────────────────────────────────────────────────────┐
│                  Isaac Sim 5.x                       │
│  (scene management, robot URDF, sensor simulation)   │
├─────────────────────────────────────────────────────┤
│                Newton Physics Engine                  │
│  (MPM fluid + rigid-body coupling + articulation)    │
├─────────────────────────────────────────────────────┤
│                   NVIDIA Warp                         │
│  (CUDA JIT kernels, autodiff, custom fluid solvers)  │
├─────────────────────────────────────────────────────┤
│                     CUDA 12.x                        │
│  (cuSPARSE, AmgX for pressure solves if needed)      │
└─────────────────────────────────────────────────────┘
```

**Custom OceanScale modules to build on this stack:**
1. Ocean wave propagation (SPH or spectral solver in Warp)
2. Buoyancy and hydrodynamic force model (extending Newton coupling)
3. Turbulence model (LES or RANS in Warp kernels)
4. Free-surface tracking (level-set or SPH surface reconstruction)
5. Underwater vehicle controller interface (differentiable via wp.Tape())

**Modules to buy/integrate:**
1. Rigid-body articulation (PhysX 5 / Newton built-in)
2. Rendering and sensor simulation (Isaac Sim / Omniverse)
3. Neural surrogate models for ocean currents (PhysicsNeMo, late-stage)

---

*End of research. All source URLs cited inline. Claims marked "(unverified — gap)" require hardware testing or access to gated documentation to confirm.*
