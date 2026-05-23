# GPU-Accelerated Fluid Simulation for Underwater Robotics

> Deep research survey for the OceanScale project. Covers methods usable NOW for building a real-time GPU-native underwater simulator.
> Date: 2026-05-20. All URLs verified via web search. No fabricated links.

---

## Table of Contents

1. [SPH on GPU](#1-sph-on-gpu)
2. [MPM on GPU](#2-mpm-on-gpu)
3. [Lattice Boltzmann Method on GPU](#3-lattice-boltzmann-method-on-gpu)
4. [Neural Fluid Simulation](#4-neural-fluid-simulation)
5. [Hybrid Physics-ML Fluid Solvers](#5-hybrid-physics-ml-fluid-solvers)
6. [Free Surface Simulation](#6-free-surface-simulation)
7. [Fluid-Structure Interaction](#7-fluid-structure-interaction-fsi)
8. [Differentiable Fluid Simulation](#8-differentiable-fluid-simulation)
9. [Real-time Ocean Currents](#9-real-time-ocean-currents)
10. [Buoyancy and Hydrostatics on GPU](#10-buoyancy-and-hydrostatics-on-gpu)

---

## 1. SPH on GPU

### State of the Art (2024-2025)

SPH (Smoothed Particle Hydrodynamics) remains one of the most GPU-friendly fluid simulation methods due to its embarrassingly parallel neighbor search and force computation. The field has seen significant performance gains in 2024-2025.

### Key Implementations and Performance

| Implementation | GPU | Particles | FPS | License | URL |
|---|---|---|---|---|---|
| **SPH_Taichi** (erizmr) | RTX 3090 | 420K | ~280 | MIT | https://github.com/erizmr/SPH_Taichi |
| **SPH_Taichi** (erizmr) | RTX 3090 | 1.74M | ~80 | MIT | https://github.com/erizmr/SPH_Taichi |
| **SPlisHSPlasH** | CUDA GPU | Variable | Interactive | MIT | https://github.com/InteractiveComputerGraphics/SPlisHSPlasH |
| **TNL-SPH** | Multi-GPU | Variable | ~20% faster than competitors | Open Source | https://www.sciencedirect.com/science/article/pii/S0010465526001566 |
| **AQUAgpusph** | OpenCL GPU | Variable | Interactive | GPL v3 | https://github.com/AQUAgpusph/AQUAgpusph |
| **Multi-GPU SPH** | 12x GPUs | 12M | Research-scale | - | https://donghaoren.org/projects/fluidsph-progress.pdf |

### Key Papers

- **TNL-SPH: Open-source modular SPH solver for modern computing platforms** (2025)
  - Published in: Computer Physics Communications
  - URL: https://www.sciencedirect.com/science/article/pii/S0010465526001566
  - Claims ~20% faster than comparable GPU-accelerated SPH codes, strong multi-GPU scaling

- **A Journey into SPH Simulation** (arXiv: 2403.11156, 2024)
  - Comprehensive SPH framework implemented in Taichi
  - URL: https://arxiv.org/html/2403.11156v1

- **Enhancement of GPU-Accelerated SPH with Dynamic Parallelism** (2025)
  - NVIDIA dynamic parallelism for SPH on modern GPUs
  - URL: https://www.sciencedirect.com/science/article/pii/S2590123025028634

- **SPH for Free-Surface and Multiphase Flows: Review of the Last 25 Years** (2025)
  - URL: https://iopscience.iop.org/article/10.1088/1361-6633/ada80f

### RL Training Speeds

SPH_Taichi achieves ~80 FPS at 1.74M particles on a single RTX 3090. For RL training with simplified scenes (10K-100K particles, lower resolution), 10K+ FPS per environment is feasible with vectorized batched simulation. However, standard SPH is NOT yet proven at the 100K+ FPS rates needed for大规模 RL. The MarineGym approach (see Section 9) uses analytical hydrodynamics instead.

### OceanScale Relevance: HIGH

SPH is the most mature GPU-fluid method with multiple open-source implementations. Best suited for:
- Free-surface water effects (waves crashing on hull)
- Visual realism for rendering
- Small-scale fluid interaction (gripper manipulating objects underwater)

Not ideal as the primary physics engine for RL training due to compute cost. Better used as a "rendering-layer" fluid with analytical hydrodynamics for control.

---

## 2. MPM on GPU

### State of the Art (2024-2025)

MPM (Material Point Method) is Newton/Warp's native simulation method. It uses a hybrid Lagrangian particle + Eulerian grid approach, making it excellent for multi-material simulation (fluid + solid + granular).

### Key Papers

- **PB-MPM: A Position-Based Material Point Method** (SIGGRAPH 2024, EA SEED)
  - Stable at ANY time-step -- game-changer for real-time
  - URL: https://www.ea.com/seed/news/siggraph2024-pbmpm
  - Paper: https://media.contentapi.ea.com/content/dam/ea/seed/presentations/seed-siggraph2024-pbmpm-paper.pdf
  - ACM: https://dl.acm.org/doi/10.1145/3641233.3664323
  - Open source: https://github.com/electronicarts/pbmpm (WebGPU, BSD-3-Clause)

- **Portable, Massively Parallel Implementation of a Material Point Method** (arXiv: 2404.17057, 2024)
  - GPU-MPM for compressible gases
  - URL: https://arxiv.org/html/2404.17057v4

- **GeoWarp: Auto-differentiable & GPU-accelerated Implicit MPM** (arXiv: 2507.09435, 2025)
  - Built on NVIDIA Warp with reverse-mode autodiff
  - URL: https://arxiv.org/html/2507.09435v2
  - KAIST, Prof. Jinhyun Choo group

- **GPU-Based Material Point Method for Compressible Flows** (2024)
  - URL: https://www.scipedia.com/wd/images/7/74/Draft_Sanchez_Pinedo_244241234pap_163.pdf

### Newton Physics Engine Integration

Newton (https://github.com/newton-physics/newton) is built on NVIDIA Warp and uses MuJoCo Warp for rigid body dynamics. Newton has MPM support (see newton_mpm_analysis.md — 8 functional MPM examples). Key facts:

- **Newton**: Open-source, GPU-accelerated physics by NVIDIA + Google DeepMind + Disney Research (March 2025)
- Built on NVIDIA Warp (Python, differentiable, CUDA)
- Focus: contact-rich manipulation and locomotion
- Integration with Isaac Lab: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html

### Fluid-Structure Interaction with MPM

MPM naturally handles multi-material coupling (fluid + solid) because particles of different materials share the same grid. This makes FSI straightforward compared to SPH or Eulerian methods.

### Performance

PB-MPM achieves real-time performance for small-scale problems using colored Gauss-Seidel iterations. No published FPS numbers for large-scale scenarios, but the SIGGRAPH 2024 talk demonstrates interactive rates.

### OceanScale Relevance: VERY HIGH

MPM is the most promising method for OceanScale because:
1. Newton/Warp already uses MPM as a foundation
2. PB-MPM (SIGGRAPH 2024) makes it stable at any time-step
3. GeoWarp provides differentiable MPM on Warp
4. Natural multi-material coupling (water + hull + sediment)
5. Warp's autodiff enables gradient-based optimization

**Recommended path**: Extend Newton with PB-MPM fluid solver on Warp, using GeoWarp as reference implementation.

---

## 3. Lattice Boltzmann Method on GPU

### State of the Art

LBM is naturally parallel (every lattice site updates independently), making it extremely GPU-friendly. It handles complex geometries well, which is relevant for underwater vehicles.

### Key Implementations

| Implementation | Language | GPU | License | URL |
|---|---|---|---|---|
| **FluidX3D** | C++/OpenCL | NVIDIA/AMD/Intel | Open Source | https://github.com/ProjectPhysX/FluidX3D |
| **PALABOS** | C++ | CUDA | Open Source | https://palabos.unige.ch/ |
| **OpenLB** | C++ | CUDA | GPL v2 | https://www.openlb.net/ |

### Performance

FluidX3D achieves billions of lattice site updates per second (GLUPS) on modern GPUs. Real-time interactive simulation at 256^3 to 512^3 resolution achieves 60+ FPS on high-end GPUs. Author (Moritz Lehmann) publishes extensive benchmark videos on YouTube.

### Key Papers

- **GPU Optimization for High-Quality Kinetic Fluid Simulation** (IEEE TVCG 2022)
  - URL: https://faculty.sist.shanghaitech.edu.cn/faculty/liuxp/projects/lbm-acc/index/lbm-gpu-acc.pdf

- **A GPU-Implemented LBM for Large Eddy Simulation** (MDPI Atmosphere, 2024)
  - URL: https://www.mdpi.com/2073-4433/15/6/735

- **A Lattice-Boltzmann Solver for 3D Fluid Simulation on GPU** (CONICET)
  - URL: https://www.researchgate.net/publication/251231734_A_Lattice-Boltzmann_solver_for_3D_fluid_simulation_on_GPU

### Comparison with SPH/MPM for Underwater

| Aspect | LBM | SPH | MPM |
|---|---|---|---|
| GPU parallelism | Excellent (regular grid) | Good (irregular neighbors) | Good (grid+particles) |
| Complex boundaries | Excellent (immersed boundary) | Good | Good |
| Free surface | Moderate (needs VOF) | Excellent (natural) | Good (particle-based) |
| Memory usage | High (uniform grid) | Moderate | Moderate |
| RL training speed | Moderate (fixed grid cost) | Moderate | Moderate |
| Multi-material | Poor | Good | Excellent |

### Robotics Applications

LBM's strength is handling complex moving boundaries (like robot arms in fluid). The FEM + LBM coupling paper (ScienceDirect, https://www.sciencedirect.com/science/article/abs/pii/S0010465520303210) demonstrates GPU-accelerated FSI using LBM.

### OceanScale Relevance: MODERATE

LBM is excellent for CFD-quality simulation around complex geometries but is less suited for free-surface and multi-material scenarios compared to SPH/MPM. Could be useful as a high-fidelity validation tool but not the primary real-time solver.

---

## 4. Neural Fluid Simulation

### State of the Art (2024-2025)

GNN-based learned simulators have matured significantly. The key question: can they replace physics solvers for RL training?

### Key Papers and Implementations

**GNS (Graph Network-based Simulator)** - DeepMind (2020, still foundational)
- arXiv: 2002.09405
- URL: https://arxiv.org/abs/2002.09405
- Code: https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate
- License: Apache 2.0
- Project page: https://sites.google.com/view/learning-to-simulate
- Benchmarks: Water-3D, Sand-3D, Goop-3D

**MeshGraphNets** - DeepMind
- Extension of GNS to mesh-based simulation
- URL: https://openreview.net/pdf/25e22a812f559c7389d64412f32a87195fb7acbb.pdf

**Multi-scale GNN for Physics-Informed Fluid Simulation** (2024)
- URL: https://link.springer.com/article/10.1007/s00371-024-03402-6

**Physics-Informed GNN Conserving Linear Momentum (DYNAMI-CAL GRAPHNET)** (Nature Communications, 2025)
- URL: https://www.nature.com/articles/s41467-025-67802-5

**Masked GNN Pre-training for CFD** (arXiv: 2501.08738, 2025)
- URL: https://arxiv.org/html/2501.08738v1

**Multi-GNN Architecture for Accelerating Physics-Informed Fluid Simulation** (ACM, 2025)
- URL: https://dl.acm.org/doi/abs/10.1145/3681756.3697879

### Speed vs Accuracy Tradeoff

| Method | Inference Speed | Accuracy | Generalization | Training Data Needed |
|---|---|---|---|---|
| GNS | ~100-1000x faster than physics | Good for trained distribution | Limited to trained scenarios | Large (1000s of trajectories) |
| MeshGraphNets | ~50-500x faster | Higher fidelity on meshes | Better for structured domains | Moderate |
| PINN-based | Variable | Physics-consistent | Good (PDE-constrained) | Less data, more compute |

### Can Neural Solvers Replace Physics for RL Training?

**Short answer: Not yet for underwater robotics.**

Pros:
- Inference is orders of magnitude faster than physics solvers
- Can be batched efficiently on GPU
- Some methods (GNS) handle multiple materials

Cons:
- Error accumulation over long rollouts (divergence from reality)
- Poor generalization outside training distribution (different hull shapes, Reynolds numbers)
- No conservation guarantees (except DYNAMI-CAL GRAPHNET, 2025)
- Requires massive training data from ground-truth physics solvers
- Cannot handle novel geometries or configurations not seen during training

For RL domain randomization, neural simulators would need to generalize across a wide range of conditions, which remains an open research problem.

### OceanScale Relevance: MODERATE

Neural fluid simulators are promising as **accelerated surrogates** for specific, well-characterized scenarios. For example:
- Pre-trained GNS for common hydrodynamic interactions (cylinder in cross-flow, flat plate drag)
- As a "fast path" for repeated evaluations during policy optimization
- NOT as the primary physics engine due to generalization concerns

---

## 5. Hybrid Physics-ML Fluid Solvers

### State of the Art (2024-2025)

This is where the most exciting progress is happening. Hybrid methods combine physics structure with learned components.

### Key Papers

**NeuralMPM** (ICML 2024)
- Neural emulation framework inspired by MPM
- Interpolates particle data onto grid, processes with NN, transfers back to particles
- **1000+ FPS** vs 15 FPS for original simulator
- Supports multiple materials
- Project page: https://neuralmpm.isach.be/
- arXiv: 2408.15753
- URL: https://arxiv.org/html/2408.15753v3

**Hybrid Neural-MPM for Interactive Fluid Simulations** (arXiv: 2505.18926, 2025)
- Diffusion-based control for interactive fluid simulations
- Real-time, interactive, high fidelity
- Project page: https://hybridmpm.github.io/
- URL: https://huggingface.co/papers/2505.18926

**Physics-Enhanced Neural Operator for Turbulent Transport** (arXiv: 2406.04367, 2024)
- PDE knowledge integrated into Fourier Neural Operators
- URL: https://arxiv.org/pdf/2406.04367

**Energy-Conserving Neural Network for Turbulence Closure** (J. Comp. Phys., 2024)
- URL: https://www.sciencedirect.com/science/article/pii/S0021999124002523

**PINN Wall Model for Lattice Boltzmann Method** (arXiv: 2402.08037, 2024)
- URL: https://arxiv.org/abs/2402.08037

**Simulating 3D Turbulence with PINNs** (arXiv: 2507.08972, 2025)
- URL: https://arxiv.org/html/2507.08972v1

### Fourier Neural Operators (FNO) as Surrogates

FNOs learn mappings between function spaces, enabling generalization across geometries and flow conditions:
- **U-FNO** (Stanford): Enhanced FNO for multiphase flow
  - URL: https://sccs.stanford.edu/sites/g/files/sbiybj17761/files/media/file/u-fno-an_enhanced_fourier_neural_operator-based_deep-learning_model_for_multiphase_flow_0.pdf
- **Geo-FNO** (NeurIPS 2024): FNO on arbitrary geometries via learned deformations
  - URL: https://neurips.cc/virtual/2024/poster/98327
- **Model-Parallel FNO**: 2.6 billion variables on 512 A100 GPUs
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S0098300423001061

### NVIDIA Modulus

NVIDIA's open-source framework for physics-informed ML:
- PINNs, FNO, DeepONet architectures
- Surrogate models for CFD
- URL: https://developer.nvidia.com/blog/develop-physics-informed-machine-learning-models-with-graph-neural-networks/

### OceanScale Relevance: HIGH

NeuralMPM achieving 1000+ FPS is a game-changer. The hybrid approach preserves physics structure while gaining massive speedup. **Recommended**: Use NeuralMPM-style architecture for the OceanScale fluid solver, combining MPM particle-grid transfers with learned grid updates. This directly builds on Newton/Warp's MPM foundation.

---

## 6. Free Surface Simulation

### State of the Art

Free surface rendering and physics for ocean scenes. Two main approaches: analytical wave models and FFT-based statistical ocean models.

### Analytical Wave Models (Gerstner Waves)

Simple sum of sinusoidal waves with parametric control. Best for:
- Small-scale water surface (pools, tanks)
- Game-quality rendering
- Easy to implement and GPU-accelerate

### FFT-Based Ocean (Tessendorf)

The industry standard for realistic ocean surfaces. Jerry Tessendorf's 2004 paper remains the foundation:
- Represents ocean as sum of thousands of sinusoidal waves via FFT
- Phillips spectrum for wave distribution
- GPU-accelerated via compute shaders

### GPU Implementations (Open Source)

| Project | Engine | Compute | License | URL |
|---|---|---|---|---|
| **godot4-oceanfft** | Godot 4 | Compute Shaders | MIT | https://github.com/tessarakkt/godot4-oceanfft |
| **GodotOceanWaves** | Godot | Compute Shaders | MIT | https://github.com/2Retr0/GodotOceanWaves |
| **fftWater** | Custom | FFT | Open Source | https://github.com/iamyoukou/fftWater |
| **WebGPU Ocean** | WebGPU | Compute Shaders | Open | https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/ |

### Key References

- **Ocean Simulation with FFT and WebGPU** (2024 blog, excellent tutorial)
  - URL: https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/

- **Real-Time Wave Simulation of Large-Scale Open Sea** (MDPI J. Marine Sci. Eng., 2024)
  - URL: https://www.mdpi.com/2077-1312/12/4/572

- **Ocean Wave Real-Time Simulation Based-on Ocean Wave Spectrum and FFT** (ResearchGate)
  - URL: https://www.researchgate.net/publication/271566177_Ocean_Wave_Real-Time_Simulation_Based-on_Ocean_Wave_Spectrum_and_FFT

- **Realtime GPGPU FFT Ocean Water Simulation** (TUHH)
  - URL: https://tore.tuhh.de/bitstream/11420/1439/1/GPGPU_FFT_Ocean_Simulation.pdf

### Performance

FFT ocean surface generation is trivially fast on GPU: a 512x512 displacement map takes < 1ms on modern GPUs. This is not a bottleneck.

### Physics Coupling

For OceanScale, the ocean surface should provide:
1. **Visual rendering**: Tessendorf FFT for realistic ocean appearance
2. **Force computation**: Wave-induced forces on vehicle hull (computed analytically from wave spectrum)
3. **Domain randomization**: Random wave spectrum parameters for sim-to-real

### OceanScale Relevance: HIGH

FFT ocean is a solved problem for rendering. The challenge for OceanScale is coupling the wave surface with vehicle dynamics. Approach:
- Use Tessendorf FFT for rendering (GPU compute shader)
- Compute wave forces analytically from the spectrum (no physics simulation needed)
- Port existing implementations to Warp compute kernels

---

## 7. Fluid-Structure Interaction (FSI)

### State of the Art

Two-way coupling between fluid and rigid/soft bodies on GPU. Critical for simulating underwater manipulator arms interacting with water.

### Key Papers

**GPU-Accelerated FSI Solver (FEM + LBM)** (ScienceDirect)
- Couples Finite Element Method with Lattice Boltzmann
- GPU-accelerated for biomechanical applications
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0010465520303210

**GPU-Accelerated Simulations of Problems with Moving Boundaries** (arXiv: 2605.04335, 2026)
- URL: https://arxiv.org/html/2605.04335v1

**Fluid Simulation with Two-Way Interaction Rigid Body Using Heterogeneous GPU and CPU** (Academia.edu)
- URL: https://www.academia.edu/146741633/Fluid_Simulation_with_Two_Way_Interaction_Rigid_Body_Using_a_Heterogeneous_GPU_and_CPU_Environment

**DiffFR: Differentiable SPH-based Fluid-Rigid Coupling** (SIGGRAPH Asia 2023)
- Two-way coupling for rigid body control
- Differentiable -- gradients flow through the coupling
- Code: https://github.com/zhehaoli1999/DiffFR
- Project page: https://zhehaoli1999.github.io/DiffFR/
- ACM: https://dl.acm.org/doi/10.1145/3618318
- PDF: http://ren-bo.net/papers/lzh_tog2023.pdf

### MPM for FSI

MPM's particle-grid structure makes FSI particularly natural:
- Rigid body represented as particles
- Fluid represented as particles
- Both interact through the same background grid
- No explicit coupling interface needed

This is a major advantage over SPH or Eulerian methods which require explicit coupling schemes.

### What Works for Underwater Manipulator Arms?

For a manipulator arm (serial kinematic chain) in water:
1. **MPM-based FSI**: Natural coupling, but computationally expensive for real-time
2. **Analytical + sampled drag**: Compute drag forces from velocity field sampled at body surface. Fast but less accurate.
3. **Hybrid**: MPM fluid + rigid body dynamics via penalty-based coupling (Newton's approach)

### OceanScale Relevance: HIGH

FSI is essential for realistic underwater manipulation. Recommended approach:
- **Primary**: Analytical drag/lift models for RL training (fast, differentiable)
- **Secondary**: MPM-based FSI for high-fidelity validation
- **DiffFR-style differentiable coupling** for gradient-based optimization of control policies

---

## 8. Differentiable Fluid Simulation

### State of the Art

Differentiable fluid simulation enables gradient-based optimization through the fluid solver. Critical for system identification and policy optimization.

### Frameworks

| Framework | Backend | GPU | Auto-diff | URL |
|---|---|---|---|---|
| **PhiFlow** | PyTorch/JAX/TF | Yes | Yes | https://github.com/tum-pbs/PhiFlow |
| **DiffTaichi** | Taichi DSL | Yes | Yes | https://github.com/taichi-dev/difftaichi |
| **NVIDIA Warp** | Python/CUDA | Yes | Yes | https://developer.nvidia.com/warp |
| **diffSPH** | PyTorch | Yes | Yes | https://arxiv.org/abs/2507.21684 |
| **JAX-SPH** | JAX | Yes | Yes | https://arxiv.org/abs/2403.04750 |
| **GeoWarp** | Warp | Yes | Yes | https://arxiv.org/html/2507.09435v2 |

### Key Papers

**PhiFlow: Differentiable Simulations for PyTorch, TensorFlow and JAX** (ICML 2024)
- TUM Thuerey Group
- URL: https://raw.githubusercontent.com/mlresearch/v235/main/assets/holl24a/holl24a.pdf
- Docs: https://tum-pbs.github.io/PhiFlow/

**diffSPH: Differentiable SPH for Adjoint Optimization** (arXiv: 2507.21684, 2025)
- PyTorch-based, GPU-accelerated
- URL: https://arxiv.org/html/2507.21684v1
- TUM Thuerey Group

**DiffTaichi: Differentiable Programming for Physical Simulation** (2020)
- URL: https://yuanming.taichi.graphics/publication/2020-difftaichi/
- Code: https://github.com/taichi-dev/difftaichi

**Warp: Differentiable Spatial Computing for Python** (NVIDIA)
- URL: https://peterchencyc.com/assets/pdf/3664475.3664543.pdf
- GTC 2024: https://www.nvidia.com/en-us/on-demand/session/gtc24-s63345/

**An Adjoint Method for Differentiable Fluid Simulation on Flow Maps** (arXiv: 2511.01259, 2025)
- URL: https://arxiv.org/html/2511.01259v1

**PICT: A Differentiable, GPU-Accelerated Multi-Block PISO Solver** (arXiv: 2505.16992, 2025)
- PyTorch-based differentiable pressure-implicit solver
- URL: https://arxiv.org/html/2505.16992v1

**Differentiable SPH Training Tutorial** (Taichi)
- URL: https://docs.taichi-lang.cn/en/blog/training-a-magic-fountain-using-taichi-autodiff/

### Gradient Quality

Differentiable physics simulations face challenges:
- **Gradient explosion/vanishing** in long rollouts
- **Contact discontinuities** (e.g., water surface) create non-smooth gradients
- **Memory cost** of storing full trajectory for backpropagation

Solutions:
- Adjoint methods (trade compute for memory)
- Gradient clipping / smoothing
- Coarse-to-fine optimization

### OceanScale Relevance: HIGH

Differentiable simulation is critical for OceanScale because:
1. **System ID**: Learn hydrodynamic coefficients from real-world data
2. **Policy optimization**: Gradients through fluid for controller tuning
3. **Shape optimization**: Hull form optimization via gradient descent

**Recommended**: Use Warp's built-in autodiff for Newton-based simulation. GeoWarp demonstrates this is feasible for MPM on Warp. For SPH-based scenarios, diffSPH (PyTorch) provides a ready-made differentiable solver.

---

## 9. Real-time Ocean Currents

### State of the Art

GPU-accelerated ocean current simulation for RL domain randomization. The most relevant development is MarineGym.

### MarineGym (arXiv: 2503.09203, 2025)

- **250,000 FPS** rollout speed on RTX 3060
- GPU-accelerated hydrodynamic plugin based on NVIDIA Isaac Sim
- Domain randomization toolkit for sim-to-real transfer
- arXiv: https://arxiv.org/abs/2503.09203
- Website: https://marine-gym.com/
- Authors: S. Chu, Z. Huang, M. Lin, D. Li, I. Carlucho (Heriot-Watt University)

### OceanSim (arXiv: 2503.01074, 2025)

- GPU-accelerated underwater robot perception simulator
- Built on NVIDIA Isaac Sim with real-time ray tracing
- arXiv: https://arxiv.org/abs/2503.01074
- Code: https://github.com/umfieldrobotics/OceanSim
- Website: https://umfieldrobotics.github.io/OceanSim/

### Domain Randomization Approaches

**Flow-based Domain Randomization** (ICML 2025)
- URL: https://icml.cc/virtual/2025/poster/46239

**DRL for ASV Navigation** (IROS 2024)
- System ID coupled with domain randomization for marine vehicles
- URL: https://hal.science/hal-04643371v1/file/DRL4ASVNavigation_IROS2024.pdf

**Learning to Dock: Simulation-based Study for Underwater Robots** (OSU, 2025)
- URL: https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/kevinchang2025revised.pdf

### Ocean Current Models for Domain Randomization

For RL training, ocean currents should be parameterized with:
- **Mean flow**: Uniform or depth-varying current profile
- **Turbulent fluctuations**: Gaussian random field or Perlin noise
- **Vortices**: Lamb-Oseen vortex model
- **Wave-induced currents**: Stokes drift

All of these can be computed analytically on GPU with zero simulation cost.

### OceanScale Relevance: VERY HIGH

MarineGym's 250K FPS on a consumer GPU proves that GPU-accelerated underwater RL is practical TODAY. OceanScale should:
1. Adopt MarineGym's hydrodynamic plugin approach (analytical models on GPU)
2. Add ocean current domain randomization (parameterized models, no fluid sim needed)
3. Use Isaac Sim / Warp as the simulation backbone

---

## 10. Buoyancy and Hydrostatics on GPU

### State of the Art

Buoyancy computation for complex hull shapes requires determining the submerged volume and center of buoyancy. Beyond Fossen's analytical models (which assume simple geometries), several approaches exist for complex shapes.

### Methods

**1. Convex Hull Decomposition**
- Decompose complex mesh into convex parts
- Compute submerged volume analytically for each convex part
- Real-time capable on GPU
- arXiv: 2509.03804 -- "Real-Time Buoyancy Estimation for AUV Simulations Using Convex Hulls"
  - URL: https://ui.adsabs.harvard.edu/abs/arXiv:2509.03804

**2. Voxel/Slice-Based Methods**
- Slice hull into horizontal cross-sections
- Compute area and centroid of each slice in parallel on GPU
- Integrate vertically for volume and center of buoyancy
- Reference: "Motion and Control Simulation for Underwater Vehicles (Voxel-Based Strategy)"
  - URL: https://jmstt.ntou.edu.tw/cgi/viewcontent.cgi?article=1051&context=journal

**3. Mesh-Based Signed Distance Field (SDF)**
- Precompute SDF of hull on GPU
- Sample SDF at water surface to find submerged region
- Compute integrals via marching cubes or direct sampling
- Very fast on GPU (embarrassingly parallel sampling)

**4. Fossen's Analytical Models**
- Standard approach for marine control (Thor Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control")
- Assumes simplified geometry (ellipsoid, box)
- Fast but inaccurate for complex hull forms

### GPU Parallelization Strategy

For RL training with thousands of parallel environments:
- Each environment has its own hull mesh
- Water surface is a plane (for hydrostatics) or deformed plane (waves)
- Per-environment buoyancy computation is independent -> trivially parallel
- Compute on GPU via CUDA/Warp kernels

Cost: O(N_slices * N_environments), where N_slices ~ 50-200 per hull
At 10K environments and 100 slices: 1M integrations per step -- trivial on GPU.

### HoloOcean Reference

HoloOcean (CMU, ICRA 2022) is another underwater simulator:
- URL: https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf
- Uses GPU for sonar modeling via simulated depth cameras

### OceanScale Relevance: VERY HIGH

Buoyancy computation is a solved problem for GPU. Approach:
1. **RL training**: Analytical Fossen-style models (fast, differentiable)
2. **Validation**: Voxel/slice-based GPU parallel computation for complex hulls
3. **Domain randomization**: Randomize buoyancy parameters (displaced volume, center of buoyancy) to improve sim-to-real transfer

---

## Summary: OceanScale Technology Roadmap

### Recommended Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    OceanScale Fluid Stack                  │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  Rendering Layer:                                         │
│    ├── Tessendorf FFT ocean surface (Warp compute)        │
│    ├── Underwater caustics / volumetric light             │
│    └── Particle effects (SPH visual layer)                │
│                                                           │
│  Physics Layer (RL Training - Fast Path):                 │
│    ├── Analytical hydrodynamics (Fossen + drag models)    │
│    ├── Buoyancy (convex hull decomposition, GPU-parallel) │
│    ├── Ocean currents (parameterized noise fields)        │
│    └── Domain randomization toolkit                       │
│                                                           │
│  Physics Layer (High Fidelity - Validation):              │
│    ├── PB-MPM fluid solver on Warp                        │
│    ├── NeuralMPM acceleration (1000+ FPS)                 │
│    └── DiffFR-style differentiable FSI                    │
│                                                           │
│  Differentiation Layer:                                   │
│    ├── Warp autodiff (native)                             │
│    ├── GeoWarp-style MPM gradients                        │
│    └── Adjoint optimization for system ID                 │
│                                                           │
│  Platform:                                                │
│    └── Newton + NVIDIA Warp + Isaac Sim                   │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

### Priority Actions

1. **Integrate MarineGym's hydrodynamic plugin** as the fast-path physics (250K FPS proven)
2. **Port PB-MPM to Warp** using EA SEED's WebGPU implementation as reference
3. **Implement FFT ocean surface** in Warp compute kernels
4. **Build domain randomization toolkit** for ocean conditions
5. **Evaluate NeuralMPM** as an acceleration layer for high-fidelity scenarios

### Key References Summary

| ID | Paper | arXiv/URL | Year | Relevance |
|---|---|---|---|---|
| MarineGym | GPU-accelerated underwater RL | [2503.09203](https://arxiv.org/abs/2503.09203) | 2025 | Critical |
| PB-MPM | Position-Based MPM | [EA SEED](https://www.ea.com/seed/news/siggraph2024-pbmpm) | 2024 | Critical |
| NeuralMPM | Neural MPM emulation | [2408.15753](https://arxiv.org/html/2408.15753v3) | 2024 | Critical |
| GeoWarp | Diff. MPM on Warp | [2507.09435](https://arxiv.org/html/2507.09435v2) | 2025 | Critical |
| Newton | GPU physics engine | [GitHub](https://github.com/newton-physics/newton) | 2025 | Critical |
| SPH_Taichi | Fast GPU SPH | [GitHub](https://github.com/erizmr/SPH_Taichi) | 2024 | High |
| TNL-SPH | Modular GPU SPH | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0010465526001566) | 2025 | High |
| diffSPH | Diff. SPH in PyTorch | [2507.21684](https://arxiv.org/html/2507.21684v1) | 2025 | High |
| PhiFlow | Diff. PDE framework | [GitHub](https://github.com/tum-pbs/PhiFlow) | 2024 | High |
| DiffFR | Diff. fluid-rigid coupling | [GitHub](https://github.com/zhehaoli1999/DiffFR) | 2023 | High |
| FluidX3D | GPU LBM | [GitHub](https://github.com/ProjectPhysX/FluidX3D) | Ongoing | Moderate |
| OceanSim | Underwater perception sim | [2503.01074](https://arxiv.org/abs/2503.01074) | 2025 | Moderate |
| GNS | Learned simulator | [2002.09405](https://arxiv.org/abs/2002.09405) | 2020 | Moderate |
| Geo-FNO | FNO on arbitrary geometry | [NeurIPS 2024](https://neurips.cc/virtual/2024/poster/98327) | 2024 | Moderate |
