# OceanScale Competitive Analysis

**Date:** 2026-05-20
**Author:** Research Agent
**Status:** Verified (sources cited inline)

---

## 1. Direct Competitors — GPU-Native Underwater Simulators

### 1.1 OceanSim (University of Michigan)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim) |
| **Affiliation** | University of Michigan Field Robotics Group |
| **Founded** | 2025 (repo created 2025-02-17) |
| **Funding** | Academic research (no commercial funding disclosed) |
| **GitHub Stars** | 446 (as of 2026-05-20) |
| **License** | BSD-3-Clause |
| **Tech Stack** | NVIDIA Isaac Sim, Omniverse, PhysX, GPU ray tracing |
| **Last Push** | 2025-09-26 |

**Features:**
- GPU-accelerated real-time ray tracing for underwater perception
- Physics-based rendering for visual and acoustic sensors
- Imaging sonar rendering (real-time)
- Synthetic data generation for ML training
- Digital twin support for field deployments
- Accepted at IROS 2025 ([arxiv.org/html/2503.01074v2](https://arxiv.org/html/2503.01074v2))

**Strengths:**
- Built on Isaac Sim — inherits NVIDIA ecosystem (PhysX, RTX rendering, USD)
- First-mover in GPU-accelerated underwater sensor simulation
- Strong academic credibility (UMich, NVIDIA Robotics endorsement)
- Real-time sonar rendering is novel

**Weaknesses:**
- Perception-focused only — no RL training, no control loop, no robot dynamics
- No hydrodynamics / fluid simulation
- Requires Isaac Sim (heavy dependency, NVIDIA GPU mandatory)
- Academic project — unclear long-term maintenance

**Source:** [github.com/umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim), [arxiv.org/html/2503.01074v2](https://arxiv.org/html/2503.01074v2), [marktechpost.com](https://www.marktechpost.com/2025/04/07/university-of-michigan-researchers-introduce-oceansim-a-high-performance-gpu-accelerated-underwater-simulator-for-advanced-marine-robotics/)

---

### 1.2 MarineGym (Heriot-Watt / Zhejiang University)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/Marine-RL/MarineGym](https://github.com/Marine-RL/MarineGym) |
| **Affiliation** | Heriot-Watt University / Zhejiang University |
| **Founded** | 2025 (repo created 2025-10-18) |
| **Funding** | Academic research |
| **GitHub Stars** | 165 (as of 2026-05-20) |
| **License** | MIT |
| **Tech Stack** | NVIDIA Isaac Sim, OmniDrones, custom GPU hydrodynamic plugin |
| **Last Push** | 2026-01-27 |

**Features:**
- GPU-accelerated hydrodynamic plugin for Isaac Sim
- Up to 250,000 FPS on RTX 3060 ([arxiv.org/html/2503.09203v1](https://arxiv.org/html/2503.09203v1))
- ~107 simulation steps per second
- Batched RL training environments
- UUV (unmanned underwater vehicle) support
- Accepted at IROS 2025
- Companion benchmark: URoBench

**Strengths:**
- Only other project with GPU-accelerated hydrodynamics + batched RL
- Impressive raw throughput (250K FPS)
- MIT license — very permissive
- Comes with URoBench benchmarking framework
- Active research group (Heriot-Watt + Zhejiang)

**Weaknesses:**
- Hydrodynamics are force-based (simplified), not full Navier-Stokes fluid
- No real-time fluid visualization / rendering
- Tightly coupled to Isaac Sim stack
- No sonar / acoustic sensor simulation
- New project — small community (165 stars)
- No sensor simulation beyond basic proprioception

**Source:** [github.com/Marine-RL/MarineGym](https://github.com/Marine-RL/MarineGym), [arxiv.org/html/2503.09203v1](https://arxiv.org/html/2503.09203v1), [marine-gym.com](https://marine-gym.com/)

---

### 1.3 HoloOcean (BYU FRoStLab)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/byu-holoocean](https://github.com/byu-holoocean) (source on [Bitbucket](https://bitbucket.org/frostlab/holoocean)) |
| **Affiliation** | Brigham Young University, FRoSt Lab |
| **Founded** | 2022 |
| **Funding** | Academic (BYU, research grants) |
| **License** | Open source (specific license tied to UE EULA restrictions) |
| **Tech Stack** | Unreal Engine 4 (v1.x) → UE 5.3 (HoloOcean 2.0), Holodeck |

**Features:**
- Multi-agent underwater simulation
- Rich sensor suite: DVL, IMU, depth sensor, RGB camera (29 params in 2.0), depth camera
- Sonar: imaging, sidescan, profiling (octree-based ray tracing)
- Fossen hydrodynamic models (v2.0)
- ROS1 and ROS2 bridge
- Hardware-in-the-loop (HIL) support
- HoloOcean 2.0 migrates to UE 5.3 ([arxiv.org/html/2510.06160v1](https://arxiv.org/html/2510.06160v1))

**Strengths:**
- Most mature underwater simulator (ICRA 2022, IROS 2022)
- Best sonar simulation in open-source space
- UE5 delivers photorealistic rendering
- Active development (2.0 in progress)
- ROS2 bridge for real robot integration
- Multi-agent support

**Weaknesses:**
- UE-based — NOT GPU-native in the batched simulation sense (no massive parallelism)
- No batched RL training (single-instance simulation)
- CPU-bound physics (UE Chaos physics, not GPU-accelerated)
- UE EULA restrictions limit distribution (no PyPI release)
- Not designed for ML training at scale
- No fluid dynamics simulation (visual only)

**Source:** [byu-holoocean.github.io/holoocean-docs](https://byu-holoocean.github.io/holoocean-docs/), [arxiv.org/html/2510.06160v1](https://arxiv.org/html/2510.06160v1), [semanticscholar.org](https://www.semanticscholar.org/paper/HoloOcean%3A-An-Underwater-Robotics-Simulator-Potokar-Lay/e9d05d4d116bc6bd56b0b92ad31a7fd80290482d)

---

### 1.4 DAVE — DAVE Aquatic Virtual Environment (Naval Postgraduate School)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/field-robotics-lab/dave](https://github.com/field-robotics-lab/dave) |
| **Affiliation** | Naval Postgraduate School (NPS) |
| **Founded** | 2019 |
| **Funding** | US Navy / DoD research |
| **GitHub Stars** | 286 |
| **License** | Apache-2.0 |
| **Tech Stack** | ROS 1/2, Gazebo (Classic → Harmonic migration in progress) |
| **Last Push** | 2024-08-19 |

**Features:**
- Open-source simulation for underwater robots, sensors, environments
- Hydrodynamic vehicle dynamics
- Multibeam sonar plugin
- Gazebo-based physics (ODE/Bullet)
- ROS 2 migration underway (GSoC 2024)
- Sensor simulation (sonar, cameras)

**Strengths:**
- Backed by US Navy — stable funding
- Apache-2.0 license — commercial-friendly
- Comprehensive sensor and environment models
- Active ROS integration

**Weaknesses:**
- Gazebo-based — CPU-only physics, no GPU acceleration
- No batched RL training
- ROS 2 migration still in progress
- Development pace slowing (last push Aug 2024)
- No fluid dynamics simulation
- Single-instance simulation only

**Source:** [field-robotics-lab.github.io/dave.doc](https://field-robotics-lab.github.io/dave.doc/), [discourse.openrobotics.org](https://discourse.openrobotics.org/t/gsoc-2024-migration-of-project-dave-to-ros-2-and-harmonic-launch-files-robot-models-and-sensor-plugins/39321)

---

### 1.5 Stonefish (Patryk Cieslak)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish) |
| **Affiliation** | Independent researcher (University of Gdansk background) |
| **Founded** | 2019 |
| **Funding** | Self-funded / academic collaboration |
| **GitHub Stars** | 253 |
| **License** | GPL-3.0 |
| **Tech Stack** | C++, Bullet Physics (extended), custom lightweight GPU rendering |
| **Last Push** | 2025-12-04 |

**Features:**
- Custom physics engine extending Bullet Physics for marine hydrodynamics
- Lightweight GPU rendering pipeline
- Sensor suite: RGB camera, depth camera, event camera, thermal camera, optical flow camera
- ROS interface
- Intervention AUV (I-AUV) support
- Version 1.6.0

**Strengths:**
- Purpose-built for marine robotics
- Custom physics engine with realistic hydrodynamics
- Novel sensor types (event camera, thermal camera)
- Lightweight — runs without heavy engine dependency
- Actively maintained (v1.6.0, push Dec 2025)

**Weaknesses:**
- GPL-3.0 license — restricts commercial use
- CPU-based physics (Bullet), no GPU batch simulation
- No batched RL training
- Single-developer project — bus factor = 1
- No sonar / acoustic simulation
- Small community

**Source:** [github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish), [arxiv.org/html/2502.11887v1](https://arxiv.org/html/2502.11887v1), [stonefish.readthedocs.io](https://stonefish.readthedocs.io/)

---

### 1.6 UUV Simulator (EU SWARMs Project)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/uuvsimulator/uuv_simulator](https://github.com/uuvsimulator/uuv_simulator) |
| **Affiliation** | EU ECSEL Project SWARMs |
| **Founded** | 2016 |
| **Funding** | EU research grant (completed) |
| **GitHub Stars** | 876 |
| **License** | Apache-2.0 (source: GitHub) |
| **Tech Stack** | ROS 1, Gazebo Classic |
| **Last Push** | 2023-08-08 (archived/stale) |

**Features:**
- Gazebo plugins for underwater vehicle simulation
- Hydrodynamic models, thruster models, ocean currents
- Multi-robot support
- Sensor simulation
- Vehicle models: LAUV, Desistek SAGA ROV

**Strengths:**
- Highest star count in underwater sim space (876)
- Mature and well-documented
- Apache-2.0 — commercial-friendly
- Established ROS ecosystem integration

**Weaknesses:**
- **Abandoned** — last push August 2023
- ROS 1 / Gazebo Classic only — no ROS 2
- CPU-only physics
- No GPU acceleration
- No batched RL training
- No fluid dynamics
- Legacy codebase

**Source:** [github.com/uuvsimulator/uuv_simulator](https://github.com/uuvsimulator/uuv_simulator), [uuvsimulator.github.io](https://uuvsimulator.github.io/)

---

### 1.7 Commercial Simulators (PaleBlue, GRi VROV)

| Simulator | URL | Model |
|-----------|-----|-------|
| **PaleBlue ROV Trainer** | [pale.blue](https://pale.blue/) | SaaS / license (~$9,800/year for base) |
| **GRi VROV** | [grisim.com](https://grisim.com/) | Enterprise license (quote-based) |

**PaleBlue:**
- VR/XR-based ROV pilot training
- Launched May 2025 ([pale.blue](https://pale.blue/2025/05/01/paleblue-publicly-launches-rov-training-simulator/))
- Delivered to French Navy Dec 2025
- Focus: pilot training, NOT robotics development / RL
- No GPU batched simulation, no fluid dynamics, no open source

**GRi VROV:**
- Professional-grade ROV simulation for offshore industry
- Sonar simulation (VSonar module), OEM topside integration
- AUV/ROV launch and recovery simulation
- Enterprise pricing (not public)
- Focus: subsea engineering, NOT ML/RL research
- Proprietary, closed-source

**Relevance to OceanScale:** These are training tools, not development platforms. Different market segment. OceanScale targets RL researchers and robotics developers, not ROV pilot training.

---

### 1.8 Mecatron UnitySim (NTU Singapore)

| Attribute | Detail |
|-----------|--------|
| **URL** | [github.com/org-arl/UWRoboticsSimulator](https://github.com/org-arl/UWRoboticsSimulator) |
| **Affiliation** | Nanyang Technological University (NTU) Mecatron team |
| **GitHub Stars** | 34 |
| **License** | MIT |
| **Tech Stack** | Unity3D, ROS, ROSBridge |
| **Last Push** | 2023-11-27 |

**Features:**
- Unity3D-based underwater simulation
- ROS integration via ROSBridge
- AUV validation and testing

**Strengths:** Lightweight, MIT license, Unity engine provides decent rendering.

**Weaknesses:** Very small community, stale (last push Nov 2023), no GPU batched sim, no RL training, no fluid dynamics.

**Source:** [discourse.openrobotics.org](https://discourse.openrobotics.org/t/maritime-community-group-meeting-feb-2025-mecatron-unitysim-for-underwater-robotics/41857)

---

## 2. GPU Fluid Simulation Landscape

### 2.1 NVIDIA Warp + Newton (OceanScale's Stack)

**NVIDIA Warp** ([github.com/nvidia/warp](https://github.com/nvidia/warp)):
- 6,672 stars, Apache-2.0, actively developed (pushed 2026-05-20)
- Python framework for GPU-accelerated simulation and ML
- JIT-compiles Python to GPU kernels
- Reverse-mode automatic differentiation (differentiable simulation)
- Includes fluid simulation primitives (SPH, Eulerian grid solvers)
- Foundation layer for Newton physics engine

**NVIDIA Newton** ([github.com/newton-physics/newton](https://github.com/newton-physics/newton)):
- 4,943 stars, Apache-2.0, actively developed (pushed 2026-05-20)
- Open-source GPU-accelerated physics engine for robotics
- Built by NVIDIA, Google DeepMind, Disney Research
- Managed by Linux Foundation
- Beta at CoRL 2025, full open-source release Q4 2025
- GTC 2026 session planned ([nvidia.com/on-demand/session/gtc26-s81613](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81613/))
- Built on Warp — custom physics solvers are first-class citizens

**OceanScale's positioning:** The ONLY project writing custom Warp fluid kernels on top of Newton for underwater-specific dynamics. No one else does this.

### 2.2 NVIDIA PhysX Flow (Open Source since April 2025)

- GPU-accelerated fluid simulation engine
- Went open source April 2025 ([digitalproduction.com](https://digitalproduction.com/2025/04/15/physx-goes-open-fluid-dynamics-unleashed/))
- Position-Based Dynamics (PBD) particle system
- Integrated into Omniverse as Omni PhysX particles
- Focus: VFX / visual effects, NOT robotics
- No differentiable simulation, no RL integration

**Relevance:** PhysX Flow is a rendering/VFX tool. OceanScale uses Warp for differentiable, RL-compatible GPU fluid kernels. Different purpose.

### 2.3 Unity DOTS / ECS Fluid

- Community SPH implementations on Unity ECS/Job System ([medium.com](https://medium.com/@leomontes_60748/how-to-implement-a-fluid-simulation-on-the-cpu-with-unity-ecs-job-system-bf90a0f2724f))
- GPU compute shader architecture for particle fluids
- Used in robotics arm simulators ([discussions.unity.com](https://discussions.unity.com/t/using-dots-ecs-in-robotic-arm-simulator/903724))
- ECS maturing but still debated for production readiness

**Relevance:** Niche, hobbyist/research use. No marine robotics application. No batched RL.

### 2.4 Custom CUDA Solvers in Robotics Labs

- GPUSPH: first fully-GPU weakly-compressible SPH solver ([arxiv.org/abs/2207.11328](https://arxiv.org/abs/2207.11328))
- CUDA-SPH-Solver: open-source CUDA SPH ([github.com/WerenskjoldH/CUDA-SPH-Solver](https://github.com/WerenskjoldH/CUDA-SPH-Solver))
- FLIP GPU with spatial hashing ([MDPI 2025](https://www.mdpi.com/2673-3951/7/1/27))
- Various lab-specific solvers (NTNU, INRIA)

**Relevance:** Research prototypes, not integrated platforms. No robotics integration, no RL training, no Newton bridge. OceanScale wraps this class of solver into a production framework.

### 2.5 Siemens Simcenter (STAR-CCM+)

- Enterprise CFD suite for marine simulation ([siemens.com](https://www.siemens.com/en-us/products/simcenter/simulation-test/computational-fluid-dynamics/))
- Ship hull optimization, tow tank simulation
- SUBOFF underwater vehicle benchmarking
- Closed-source, expensive license

**Relevance:** Traditional CFD, not real-time, not RL-compatible, not open source. Different market (naval architecture vs. robotics). No "Flowster" product found — likely confused with Simcenter FLOEFD or Flomaster.

---

## 3. Differentiation Matrix

| Feature | **OceanScale** | **OceanSim** | **MarineGym** | **HoloOcean** | **DAVE** | **Stonefish** | **UUV Sim** |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| GPU-native fluid dynamics | **Yes (Warp)** | No | Partial (hydro forces) | No | No | No | No |
| Batched RL training | **Yes** | No | Yes | No | No | No | No |
| Newton physics bridge | **Yes** | No | No (Isaac Sim) | No | No | No | No |
| Real-time sonar sim | Planned | Yes | No | Yes (best) | Partial | No | Partial |
| Differentiable sim | **Yes (Warp AD)** | No | No | No | No | No | No |
| Custom GPU fluid kernels | **Yes** | No | No | No | No | No | No |
| Multi-world parallelism | **Yes** | No | Yes | No | No | No | No |
| Open source | **Apache-2.0** | BSD-3 | MIT | OS (UE EULA) | Apache-2.0 | GPL-3.0 | Apache-2.0 |
| Sensor diversity | Growing | Good (sonar) | Basic | Best | Good | Good (novel) | Basic |
| ROS integration | Planned | Via Isaac | Via Isaac | Yes (ROS2) | Yes (ROS2) | Yes | Yes (ROS1) |
| Underwater-specific dynamics | **Yes** | No | Yes | Yes (Fossen) | Yes | Yes | Yes |
| Active development | **Yes** | Slow | Moderate | Active (2.0) | Slow | Active | **Abandoned** |
| Rendering quality | Moderate | High (RTX) | Moderate | High (UE5) | Low | Moderate | Low |
| Sim-to-real pipeline | **Planned** | No | No | HIL support | No | No | No |

### Key Differentiators (OceanScale-unique)

1. **Custom Warp fluid kernels** — No other project writes custom GPU fluid solvers for underwater dynamics
2. **Newton integration** — Only project building on Newton (NVIDIA/DeepMind/Disney) for physics
3. **Differentiable underwater simulation** — Warp AD enables gradient-based optimization of robot controllers through fluid
4. **Batched multi-world underwater dynamics** — MarineGym has batched hydrodynamics but not full fluid simulation

---

## 4. Moat Analysis

### What's Hard to Copy (Defensibility)

| Moat | Difficulty to Replicate | Why |
|------|:-----------------------:|-----|
| Newton integration depth | **High** | Newton API is evolving rapidly; deep integration requires following upstream changes. OceanScale builds custom solvers ON Newton, not just using it. Requires understanding Warp kernel programming + Newton articulation API. |
| Custom Warp fluid kernels | **Very High** | Writing production-quality GPU fluid kernels (SPH, grid-based) requires CFD expertise + CUDA/Warp fluency + robotics domain knowledge. This intersection is extremely rare. |
| Differentiable underwater dynamics | **Very High** | Warp's autodiff + fluid + underwater dynamics = unique combination. No other framework offers this. Would require starting from scratch on a differentiable physics framework. |
| Batched multi-world underwater | **High** | MarineGym proves it's possible for hydro forces, but full fluid dynamics in batched mode is harder. Memory management for N parallel fluid fields on GPU is non-trivial. |
| Sensor simulation (sonar) | **Medium** | HoloOcean and OceanSim already do sonar well. Catching up is feasible but requires significant effort. |
| Ecosystem lock-in (Newton/Warp) | **Medium** | Competing on Isaac Sim (like MarineGym/OceanSim) is a different ecosystem. Switching costs are high once committed. |

### What's Easy to Copy

| Feature | Difficulty | Note |
|---------|:----------:|------|
| Basic underwater dynamics | Low | Fossen models are published, well-understood |
| ROS integration | Low | Standard ROS packages |
| Visual rendering | Low | UE5 / Isaac Sim provide this out-of-box |
| RL gym wrapper | Low | Gymnasium API is trivial to implement |

### Sustainable Competitive Advantages

1. **Stack depth** — OceanScale is the ONLY project that owns the full stack: custom GPU fluid kernels → Newton physics → RL training → (planned) sensor simulation. Competitors pick 1-2 layers.

2. **Timing** — Newton is pre-1.0 (beta since CoRL 2025). Building deep integration NOW means OceanScale shapes the upstream API for underwater use cases. Latecomers face a mature API they don't control.

3. **Differentiation through physics** — All competitors use simplified hydrodynamic force models. OceanScale simulates the actual fluid. This is a physics-quality gap that matters for sim-to-real transfer.

4. **Apache-2.0** — Most permissive license among GPU-native underwater sims. MarineGym (MIT) is also permissive, but OceanSim (BSD-3), Stonefish (GPL-3.0), HoloOcean (UE EULA) have restrictions.

### Risks to Moat

1. **MarineGym could add fluid simulation** — They already have GPU hydrodynamics on Isaac Sim. If they add Navier-Stokes or SPH, they become a direct competitor.
2. **NVIDIA could add underwater physics** — If NVIDIA builds underwater dynamics into Newton/Isaac Sim natively, OceanScale's custom kernels become less differentiated.
3. **HoloOcean 2.0 could add batched sim** — UE5's Chaos physics supports some parallelism. If HoloOcean adds ML training support, their superior rendering becomes a major draw.
4. **Market size** — Underwater simulation is niche. Limited TAM means limited resources for defensibility investment.

---

## 5. Competitive Positioning Summary

```
                    GPU Fluid Quality
                         ^
                         |
            OceanScale * |  
                         |     * MarineGym
                         |
                         |
   ──────────────────────┼────────────────────> ML/RL Capability
                         |
          * OceanSim     |
                         |  * Stonefish
        * HoloOcean      |
     * DAVE              |              * UUV Sim
                         |
```

**OceanScale occupies the unique position of combining GPU fluid simulation with ML/RL training capability.** No other project sits in this quadrant.

**Closest competitor:** MarineGym (batched RL + simplified hydrodynamics), but they lack:
- Full fluid dynamics (they use force models, not fluid solvers)
- Differentiable simulation
- Custom physics engine integration (they use Isaac Sim, not Newton)
- Sonar / sensor simulation

---

## Sources

- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim) — 446 stars, BSD-3-Clause
- [OceanSim arXiv paper](https://arxiv.org/html/2503.01074v2)
- [OceanSim MarkTechPost](https://www.marktechpost.com/2025/04/07/university-of-michigan-researchers-introduce-oceansim-a-high-performance-gpu-accelerated-underwater-simulator-for-advanced-marine-robotics/)
- [MarineGym GitHub](https://github.com/Marine-RL/MarineGym) — 165 stars, MIT
- [MarineGym arXiv paper](https://arxiv.org/html/2503.09203v1)
- [MarineGym website](https://marine-gym.com/)
- [HoloOcean docs](https://byu-holoocean.github.io/holoocean-docs/)
- [HoloOcean 2.0 preview](https://arxiv.org/html/2510.06160v1)
- [DAVE documentation](https://field-robotics-lab.github.io/dave.doc/)
- [DAVE GitHub](https://github.com/field-robotics-lab/dave) — 286 stars, Apache-2.0
- [DAVE ROS2 migration](https://discourse.openrobotics.org/t/gsoc-2024-migration-of-project-dave-to-ros-2-and-harmonic-launch-files-robot-models-and-sensor-plugins/39321)
- [Stonefish GitHub](https://github.com/patrykcieslak/stonefish) — 253 stars, GPL-3.0
- [Stonefish arXiv paper](https://arxiv.org/html/2502.11887v1)
- [UUV Simulator GitHub](https://github.com/uuvsimulator/uuv_simulator) — 876 stars, Apache-2.0 (stale)
- [UWRoboticsSimulator GitHub](https://github.com/org-arl/UWRoboticsSimulator) — 34 stars, MIT
- [Mecatron UnitySim presentation](https://discourse.openrobotics.org/t/maritime-community-group-meeting-feb-2025-mecatron-unitysim-for-underwater-robotics/41857)
- [PaleBlue ROV launch](https://pale.blue/2025/05/01/paleblue-publicly-launches-rov-training-simulator/)
- [GRi VROV](https://grisim.com/products/vrov-virtual-remotely-operated-vehicle/)
- [NVIDIA Warp GitHub](https://github.com/nvidia/warp) — 6,672 stars, Apache-2.0
- [NVIDIA Newton GitHub](https://github.com/newton-physics/newton) — 4,943 stars, Apache-2.0
- [Newton NVIDIA blog](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)
- [Newton GTC 2026](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81613/)
- [Newton Robot Report](https://www.therobotreport.com/nvidia-launches-newton-physics-engine-gr00t-ai-corl-2025/)
- [PhysX Flow open source](https://digitalproduction.com/2025/04/15/physx-goes-open-fluid-dynamics-unleashed/)
- [URoBench paper](https://www.researchgate.net/publication/384302614_URoBench_Comparative_Analyses_of_Underwater_Robotics_Simulators_from_Reinforcement_Learning_Perspective)
- [Underwater simulators review 2025](https://arxiv.org/html/2504.06245v1)
- [GPU FLIP spatial hashing](https://www.mdpi.com/2673-3951/7/1/27)
- [Siemens Simcenter CFD](https://www.siemens.com/en-us/products/simcenter/simulation-test/computational-fluid-dynamics/)

All GitHub star counts verified via GitHub API on 2026-05-20.
