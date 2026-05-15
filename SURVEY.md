# Underwater Robot Simulator Landscape & Next-Gen Design

**Date:** 2026-05-15  
**Scope:** Survey of underwater/marine robot simulators (May 2026) and design proposal for a next-generation GPU-scale underwater simulator.

---

## 1. Executive Summary

The underwater simulation field bifurcated in 2025-2026:

- **Classic ROS/Unreal/Unity simulators** (HoloOcean 2.x, Stonefish 1.6, DAVE, MARUS) — high physics & sensor fidelity, mature, but **CPU-bound or rendering-only GPU**; cap at ~800 FPS / 100× real-time. Unsuitable for large-scale parallel RL.
- **GPU-native research simulators** (MarineGym, OceanSim, JaxLrauv, "Learning to Swim") — achieve 250k+ FPS / 10,000× real-time, but each is **incomplete in a different dimension** (no sensors, weak dynamics, single-vehicle, no acoustic comms).

Meanwhile the broader physics-engine stack has consolidated:

- **NVIDIA Newton 1.0 GA** (GTC 2026, Apache-2.0, Linux Foundation) is now the open-source GPU robotics physics standard, built on **Warp + OpenUSD**, with **MuJoCo Warp** as the primary rigid-body solver and Disney's **Kamino VBD** for deformables.
- **Isaac Sim 6.0** + **Isaac Lab 3.0 Beta** introduced multi-backend (PhysX ↔ Newton) and Warp-native data pipelines.
- **MuJoCo 3.5** + **MJWarp** ship with phenomenological fluid forces (Kutta lift, Magnus, ellipsoid drag) — adequate for AUV-scale dynamics, not for free-surface / wakes.
- Newton, MJWarp, MJX, and Isaac Lab all **lack native underwater fluid coupling**. This is the open gap.

**Strategic conclusion:** Build a Newton-native underwater extension that combines (a) GPU-parallel Fossen + free-surface hydrodynamics, (b) Warp-kernel sonar/DVL/optical sensors, (c) acoustic-comms-aware multi-agent envs, (d) a first-class Isaac Lab/skrl RL frontend, all on a single OpenUSD scene graph.

---

## 2. State-of-the-Art Versions (verified, May 2026)

| Component | Version | Date | Notes |
|---|---|---|---|
| MuJoCo | 3.5.0 | 2026-02-13 | MJWarp officially shipped; system identification toolbox |
| mujoco-mjx | latest | 2026-04-24 | Fully differentiable via JAX |
| MuJoCo Warp (MJWarp) | shipped | 2026 | Now primary Newton rigid-body backend |
| NVIDIA Newton | 1.2.0 | 2026-05-12 | 1.0 GA at GTC 2026 (Mar 18); Apache-2.0 |
| NVIDIA Warp | 1.13.0 | 2026 | `warp.sim` deprecated → migrated into Newton |
| Isaac Sim | 5.1 GA / 6.0 Early Dev | 6.0 at GTC 2026 | 6.0 brings multi-backend, Warp-native, MuJoCo-importer |
| Isaac Lab | 2.3.x stable / 3.0 Beta | 2026 | 3.0 factory-based multi-backend, Warp data, kit-less |
| PhysX | 5.x in Isaac 5.1+ | 2026 | Direct-GPU API, +70% throughput latest |
| HoloOcean | 2.3.0 (preview 2.0) | 2025-10 | UE 5.3, raytraced sonar, Fossen 6-DOF, ROS2 bridge |
| Stonefish / `stonefish_ros2` | 1.6.0 | 2025 (ICRA) | Bullet+OpenGL, event/thermal cameras, manual stepping for RL |
| DAVE | 4.0.0 + ROS2 port (GSoC '25) | 2024-2025 | Gazebo Harmonic + ROS 2 Jazzy |
| MARUS | core updated | 2026-03 | Unity HDRP, gRPC ROS bridge |
| MarineGym | arXiv 2503.09203 | 2025 (IROS '25) | Isaac Sim + PhysX + custom GPU Fossen plugin; 250k FPS |
| OceanSim | arXiv 2503.01074 | 2025 (IROS '25) | Isaac Sim + Warp; sonar/DVL/water-column; weak dynamics |
| SMaRCSim | arXiv 2506.07781 | 2025 | KTH multi-vehicle (UUV/USV/UAV) |
| Genesis | active | 2026 | SPH/MPM/PBD/Stable-Fluid; marine **not built-in** (Issue #682) |
| rl_games | 1.6.5 | 2026-02 | Default RL backend for Isaac Lab |
| skrl | 2.0.0 | 2026 | PyTorch + JAX + **Warp** backends |
| RSL-RL | v4-v5 | 2025-09 (arXiv 2509.10771) | Minimal PPO+DAgger; ~90k FPS Spot |
| DreamerV3 | Nature 2025 | 2025 | World-model, 150+ tasks single config |

URLs collected per source in §7.

---

## 3. Physics & Hydrodynamics Landscape

### 3.1 Built-in Fluid Models
- **MuJoCo fluid forces** (stateless): inertia-based + ellipsoid-based; 5 tunable params/geom; Kutta + Magnus lift, blunt/slender/angular drag. Validated for fish-swimming (SRL-ETHZ fishsim) and fruit-fly flight. Works in MJX and MJWarp.
- **PhysX 5**: rigid-body + soft-body + cloth + fluid (FLIP/PBD); not directly aimed at AUV.
- **Newton 1.0**: no native fluid; MPM with two-way rigid-coupling is the closest primitive (`mpm_twoway_coupling` example). Hydroelastic contact (Drake-inherited) is for solids, not water.
- **Genesis**: SPH + MPM + Stable Fluid solvers, but **no AUV hydrodynamics models** (Issue #682 open).

### 3.2 Underwater-specific Hydro
- **Fossen 6-DOF** (rigid body + added mass + Coriolis + quadratic damping + restoring) remains the analytic baseline. Open code: MSS (MATLAB), `cybergalactic/FossenHandbook`.
- **MarineGym** implements Fossen as GPU PyTorch tensors on PhysX rigid bodies → 250k FPS.
- **HoloOcean 2.0** custom Fossen Python/C++ (240 Hz substep), validated <2% vs REMUS 100 data.
- **arXiv 2603.07939 (2026)** + **arXiv 2602.23283 (2026)** identify hydrodynamic parameters in MuJoCo via CMA-ES; show stateless fluid model generalizes across actuation frequencies for tendon-driven fish robots.
- **arXiv 2512.13359 (Dec 2025)** trains 6-DOF AUV in JAX/MJX, 4096 envs on RTX 4060, zero-shot sim-to-real, beats MPC.

### 3.3 CFD / Particle Coupling for RL
Full Navier-Stokes is too slow for RL rollout speed. Current frontier:
- **Hybrid Neural-MPM** (arXiv 2505.18926) — GNN low-res + MPM fallback.
- **MPM-Async** convex-contact coupling.
- **SPH for AUVs** (Angelidis 2025, Adv Robotics 185) — dedicated but not RL-scaled.
- **No public combination of real-time CFD + GPU-parallel underwater RL exists as of May 2026.**

### 3.4 Free Surface / Waves
- MuJoCo open issue #1691 — partial submersion / amphibious works around shallow water with hacks.
- HoloOcean 2.0 roadmap: UE5 Gerstner/FFT waves + buoyancy via wave-height sampling.
- Stonefish: lower-order current/wave models, marked as limitation.
- **Open gap across all major sims.**

---

## 4. Sensor Modeling

### 4.1 Sonar
- **HoloOcean 2.0** (UE5.3 + RTX ray tracing): 0.012 s/tick single-beam (vs 0.055 s octree). Sidescan, FLS, profiling, bathymetric. **Real-time for dynamic scenes**.
- **OceanSim**: Warp-kernel ray-traced imaging sonar, faster than HoloOcean 1.x.
- **Stonefish 1.6**: OpenGL multibeam, FLS, side-scan, mechanical scanning.
- **DAVE/Gazebo**: CPU ray-based multibeam (slow).

### 4.2 Optics (Underwater Camera)
- **Jaffe-McGlamery IFM** + Beer-Lambert canonical. Embedded in OceanSim's water-column image formation (multiple Jerlov water types).
- Differentiable underwater rendering: **Neural Reflectance Fields underwater** (arXiv 2304.03384), **MD-Net** (Sci. Rep. 2025), **SUCRe** (arXiv 2212.09129).
- HoloOcean 2.0 UE5 Lumen + caustics for high visual fidelity.

### 4.3 DVL / IMU / Pressure
- DVL bottom-lock vs water-track distinction modeled in OceanSim with range-dependent dropout.
- arXiv 2510.21215 (Oct 2025) — graph-based VIA-D SLAM with **DVL velocity-bias preintegration**; reference for noise modeling.
- Industry units: Nortek DVL1000, WaterLinked DVL-A50.

### 4.4 Acoustic Communication
- **BELLHOP** Gaussian beam tracing (canonical), wrapped by **WOSS** for network sims, **VirTEX** for time-varying channels.
- **Q-D model** (arXiv 2501.04238) hybridizes BELLHOP + GBSM stochastic.
- HoloOcean has built-in acoustic comms; MarineGym/OceanSim do not.

---

## 5. GPU RL Training Stack (verified)

| Stack | Role | Underwater-ready? |
|---|---|---|
| **NVIDIA Warp 1.13** | Kernel-level GPU primitives, differentiable | Kernels for any custom physics |
| **Newton 1.x** | GPU rigid-body + diff sim, OpenUSD | No native hydro — extension point |
| **MJX / MJWarp** | MuJoCo on GPU; ellipsoid fluid forces built-in | **Yes for analytic AUVs** |
| **Isaac Lab 3.0 Beta** | Multi-backend env wrapper, ≥150k FPS | Bring-your-own hydro plugin |
| **Genesis** | Universal solvers (SPH/MPM/PBD) | Solvers exist, marine models don't |
| **rl_games 1.6.5** | Default PPO for Isaac Lab; multi-node, ONNX | Env-agnostic |
| **skrl 2.0** | PyTorch/JAX/Warp; works with Isaac Lab + Playground | Env-agnostic, best for multi-paradigm |
| **RSL-RL v5** | Minimal PPO+DAgger (ETH style) | Env-agnostic |
| **PureJaxRL / Stoix** | End-to-end JAX (env+agent) | Pair with MJX/Brax envs |
| **DreamerV3 / TD-MPC2** | World models; sample-efficient sim-to-real | Best pair with high-fidelity envs |

### Throughput References
- MuJoCo Warp vs MJX: **252× locomotion, 475× manipulation** on RTX PRO 6000 Blackwell.
- Newton in Isaac Lab: +65% vs PhysX on in-hand manipulation.
- Isaac Lab classic MDPs: **up to 2M steps/s** on 8× RTX Pro 6000.
- MarineGym: **250k FPS on RTX 3060** with 8,000 parallel UUVs.
- Genesis: 43M FPS for Franka on RTX 4090.
- PureJaxRL: 1000-4000× over PyTorch RL baselines.

---

## 6. Gap Analysis

| Capability | Best existing | Gap |
|---|---|---|
| GPU-parallel Fossen hydro | MarineGym | Coupled with Newton/USD scene graph |
| Free-surface waves (real-time) | HoloOcean 2.0 (in progress) | None production-grade for RL |
| Ray-traced sonar (dynamic) | HoloOcean 2.0 / OceanSim | Not parallel-env scalable |
| Jaffe-McGlamery rendering | OceanSim | Single-vehicle, perception-only |
| Acoustic comms (BELLHOP) | HoloOcean | Not in GPU-parallel sim |
| Multi-vehicle (AUV+USV+UAV) | SMaRCSim, HoloOcean | Not GPU-parallel |
| CFD-RL coupling | none | Open research |
| Diff. underwater dynamics | MJX (analytic only) | No diff. CFD |
| Standardized RL benchmark | MarineGym (tasks: hover, dock, follow) | No community-wide benchmark suite |
| Sim-to-real validated zero-shot | "Learning to Swim", EasyUUV, HoloOcean | Robust DR toolkit + real-data calibration |

**No single simulator today combines:** GPU-parallel hydro + ray-traced sonar + Jaffe-McGlamery vision + Fossen dynamics + acoustic comms + multi-agent + USD scene + Isaac Lab RL integration.

---

## 7. Source URLs (verified during research)

**MuJoCo / MJX / MJWarp**
- https://github.com/google-deepmind/mujoco
- https://mujoco.readthedocs.io/en/stable/mjx.html
- https://mujoco.readthedocs.io/en/stable/computation/fluid.html
- https://mujoco.readthedocs.io/en/stable/OpenUSD/index.html
- https://github.com/srl-ethz/fishsim
- https://arxiv.org/html/2512.13359v1

**Newton**
- https://github.com/newton-physics/newton
- https://developer.nvidia.com/newton-physics
- https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning
- https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html

**Isaac Sim / Lab**
- https://github.com/isaac-sim/IsaacSim
- https://github.com/isaac-sim/IsaacLab
- https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709
- https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta
- https://docs.isaacsim.omniverse.nvidia.com/6.0.0/physics/index.html

**Underwater simulators**
- https://arxiv.org/abs/2503.01074 (OceanSim)
- https://github.com/umfieldrobotics/OceanSim
- https://arxiv.org/abs/2503.09203 (MarineGym main)
- https://arxiv.org/abs/2410.14117 (MarineGym preview)
- https://github.com/Marine-RL/MarineGym
- https://arxiv.org/html/2510.06160v1 (HoloOcean 2.0 preview)
- https://byu-holoocean.github.io/holoocean-docs/develop/index.html
- https://github.com/patrykcieslak/stonefish
- https://github.com/patrykcieslak/stonefish_ros2
- https://arxiv.org/html/2502.11887v2 (Stonefish ICRA 2025)
- https://field-robotics-lab.github.io/dave.doc/
- https://github.com/Liquid-ai/Plankton
- https://github.com/MARUSimulator
- https://github.com/open-airlab/UNav-Sim
- https://arxiv.org/abs/2504.06245 (Underwater Sim Review 2025)
- https://arxiv.org/html/2506.07781v1 (SMaRCSim)
- https://arxiv.org/html/2505.08222v2 (JaxLrauv MARL)
- https://arxiv.org/html/2410.00120v1 (Learning to Swim)
- https://arxiv.org/html/2509.03804v1 (Real-time buoyancy convex-hull)
- https://arxiv.org/html/2510.22126v1 (EasyUUV)

**Physics / Sensors**
- https://arxiv.org/html/2304.03384 (Underwater NeRF)
- https://arxiv.org/abs/2501.04238 (Q-D acoustic channel)
- https://arxiv.org/html/2510.21215v1 (VIA-D SLAM, DVL preintegration)
- https://arxiv.org/html/2505.18926v1 (Hybrid Neural-MPM)
- https://github.com/cybergalactic/MSS

**RL stacks**
- https://github.com/NVIDIA/warp
- https://github.com/Denys88/rl_games
- https://github.com/Toni-SM/skrl
- https://github.com/leggedrobotics/rsl_rl
- https://arxiv.org/html/2509.10771v1
- https://github.com/Genesis-Embodied-AI/genesis-world
- https://github.com/Genesis-Embodied-AI/Genesis/issues/682
- https://github.com/luchris429/purejaxrl
- https://github.com/EdanToledo/Stoix
- https://www.nature.com/articles/s41586-025-08744-2 (DreamerV3)
- https://openreview.net/forum?id=Oxh5CstDJU (TD-MPC2)
