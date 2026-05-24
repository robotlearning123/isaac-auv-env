# OceanScale — Competitive Matrix

**Date:** 2026-05-24
**Sources:** SURVEY.md, REFERENCES.md, official repos/papers (cited inline). Claims not verified against running code are marked `(unverified)`.

---

## Overview

Seven simulators targeting underwater robotics RL, compared across eight axes. No single simulator combines GPU-parallel hydrodynamics + sensor fidelity + RL throughput + open license. OceanScale's gap: GPU-parallel Fossen on Newton + open license + Gymnasium RL frontend — but currently single-vehicle, stub-level IMU/DVL/pressure only, no sonar/camera/acoustic comms, and early-stage product packaging.

---

## Matrix

| Axis | OceanScale | MarineGym | OceanSim | HoloOcean | Stonefish | DAVE | Isaac AUV |
|------|-----------|-----------|----------|-----------|-----------|------|-----------|
| **Physics engine** | Newton 1.2 (GPU, Warp kernels) + custom Warp hydro | Isaac Sim 4.1 + PhysX 5 (GPU rigid body) + custom GPU Fossen plugin | Isaac Sim + Warp kernels (ray tracing for sensors) | Unreal Engine 5.3 + custom Fossen C++/Python (CPU, 240 Hz substep) | Bullet + OpenGL (CPU) | Gazebo Harmonic (CPU, ODE/DART) | Isaac Lab 3.0 Beta + rsl_rl (GPU, PhysX or Newton backend) |
| **Hydro model** | Fossen 6-DOF: added mass, Coriolis, linear + quadratic damping, restoring. Warp kernels. Validated vs von Benzon 2022 | Fossen 6-DOF: PyTorch tensors for added mass, damping, Coriolis, buoyancy on PhysX rigid bodies [arXiv 2503.09203 §IV] | Weak dynamics — no dedicated hydrodynamic model; focuses on sensor simulation [arXiv 2503.01074] | Fossen 6-DOF custom C++/Python, 240 Hz substep. Validated <2% trajectory error vs REMUS 100 field data [arXiv 2510.06160] | Fossen-style rigid-body hydro in C++ (Bullet). Lower-order current/wave models, marked as limitation [arXiv 2502.11887] | Basic buoyancy + drag plugin (uuv_simulator legacy). No added mass or Coriolis | MJX/MuJoCo ellipsoid fluid forces (stateless: Kutta lift, Magnus, drag). Not Fossen-tier — no added mass matrix |
| **Sensor support** | Headless core with IMU, DVL, and depth/pressure stubs. MP4 demo rendering exists, but no physical camera model, sonar, or acoustic comms yet | No dedicated sensor models. Perception via Isaac Sim camera API only | Imaging sonar (Warp ray-traced, Rayleigh + Gaussian speckle), underwater camera (Akkaynak-Treibitz attenuation + backscatter), DVL (4-beam Janus, range-dependent dropout), barometer [arXiv 2503.01074] | Ray-traced sonar (UE5.3 RTX): single-beam, sidescan, FLS, profiling, bathymetric. UE5 Lumen camera + caustics. Acoustic comms. DVL, IMU [arXiv 2510.06160] | Multibeam sonar (OpenGL), FLS, side-scan, mechanical scanning. Event camera, thermal camera. DVL, IMU [arXiv 2502.11887] | CPU ray-based multibeam sonar (slow). Standard Gazebo camera, IMU. No DVL model | No dedicated sensor models beyond Isaac Lab defaults. Perception via Isaac Sim camera API |
| **RL integration** | Gymnasium-compatible `ROVEnv`. Stable-Baselines3 PPO. Custom `BatchedVecEnv` on Newton tensors. VecNormalize. EV ~0.9 at 1M hover steps | TorchRL + OmniDrones wrapper. Isaac Sim ArticulationView. PPO. 8,000 parallel UUVs [arXiv 2503.09203] | Isaac Sim Replicator-based. No standalone Gymnasium env. Sensor data output only, no RL training loop in paper | No native Gym/Gymnasium wrapper. ROS 2 bridge for external RL. Manual episode management | No native Gym wrapper. ROS 2 integration via `stonefish_ros2`. Manual stepping API for RL | ROS 2 (Jazzy) integration. No native Gym wrapper. External RL via ROS topics | Isaac Lab env wrapper (MDP). rsl_rl PPO (ETH minimal). 17-D obs, 6-D action. Zero-shot sim-to-real on BlueROV2-class vehicle [arXiv 2410.00120] |
| **GPU parallelism** | 85,824 env-steps/s at n=64 and 4,588,922 env-steps/s at n=4096 on RTX 5090 (`benchmarks/competitive/results.json`, 200-step standardized launch bench). BatchedVecEnv shape (n_envs, ...) | 250k FPS on RTX 3060, 8,000 parallel envs [arXiv 2503.09203] | Single-vehicle only. Sensor kernels are GPU-accelerated but env stepping is not parallelized | Single-env. 240 Hz physics substep on CPU. ~800 FPS / ~100x real-time cap (UE5 rendering bottleneck) | Single-env. CPU-bound. Manual stepping for RL | Single-env. CPU-bound (Gazebo/ODE) | GPU-parallel via Isaac Lab. ~90k FPS on Spot locomotion (RSL-RL v5 benchmark) `(unverified for AUV task)` |
| **Fidelity evidence** | von Benzon 2022 6-DOF BlueROV2 tank-test cross-validation (ARCHITECTURE.md). PPO hover converges EV ~0.9 | Paper reports 10-20% translational error, 30-100% rotational error in hydro coefficients vs reference (Eidsvik method). No sim-to-real validation in paper | No dynamics validation (sensor-only paper). Sonar image realism qualitative | <2% trajectory error vs REMUS 100 field data. Most rigorous sim-to-real validation in the field [arXiv 2510.06160] | Validated in multiple ICRA/IROS papers against pool trials. C++ physics is mature | Limited validation. Community-tested via DARPA competitions and university groups | Zero-shot sim-to-real on BlueROV2-class vehicle, beats hand-tuned PID [arXiv 2410.00120]. Strongest RL transfer evidence |
| **License** | Apache-2.0 | MIT | BSD-3-Clause | MIT `(unverified — SURVEY.md says "verify")` | GPL-3.0 (incompatible with Apache-2.0 outbound) | BSD-3-Clause (Gazebo + ROS 2 stack) | BSD-3-Clause (inherited from Isaac Lab) |
| **Maintenance** | Active (0.1.0a0 package, 2026-05). Solo dev | Last push 2026-01 (4 mo stale). 163 stars, 9 open issues [github.com/Marine-RL/MarineGym] | Last push 2025-09 (8 mo stale). 446 stars, 7 open issues [github.com/umfieldrobotics/OceanSim] | Active (2.3.0 preview). BYU Field Robotics lab, continuous development | Last major release 2025 (ICRA). Single maintainer (Cieślak). GPL limits community | Gazebo Harmonic port + ROS 2 Jazzy (GSoC '25). Community-maintained | Last push 2025-08 (9 mo stale). 57 stars, 1 issue. WARPLab [github.com/warplab/isaac-auv-env] |
| **Setup friction** | `uv sync --extra dev` (Newton + Warp + CUDA 12.8). Requires NVIDIA GPU. ~5 min on RTX 5090 | Isaac Sim 4.1 install (~30 GB, omniverse launcher). PhysX plugin. Python 3.10. Heavy dep tree | Isaac Sim + Replicator + Omniverse extensions. Heavy dep tree. Requires NVIDIA GPU | Unreal Engine 5.3 build (~50 GB source). Epic Games launcher or source build. C++ compilation. ~1-2 hr first build | CMake build from source. ROS 2 integration optional. Light deps. Cross-platform | Gazebo Harmonic + ROS 2 Jazzy. `apt install` on Ubuntu. Moderate deps | Isaac Lab 3.0 Beta + Isaac Sim 6.0 (~40 GB). Heavy dep tree. Requires NVIDIA GPU |

---

## Comparison by Axis

### Physics Engine: GPU-native vs CPU-legacy

| Simulator | GPU physics? | Rigid body | Fluid coupling | Differentiable? |
|-----------|-------------|-----------|---------------|----------------|
| OceanScale | Yes (Newton + Warp) | Yes (MuJoCo Warp solver) | Custom Fossen Warp kernels | Partial (Warp autograd on hydro forces) |
| MarineGym | Yes (PhysX 5 GPU) | Yes | Custom GPU Fossen on PhysX | No (PyTorch tensors, no autograd through physics) |
| OceanSim | Yes (Warp sensor kernels) | Yes (PhysX) | None | No |
| HoloOcean | No (UE5 CPU substep) | Yes | Custom C++ Fossen | No |
| Stonefish | No (Bullet CPU) | Yes | C++ Fossen | No |
| DAVE | No (Gazebo ODE/DART CPU) | Yes | Basic drag/buoyancy only | No |
| Isaac AUV | Yes (PhysX or Newton) | Yes | MJX ellipsoid fluid (stateless) | Yes (MJX fully differentiable via JAX) |

### RL Throughput Hierarchy

```
MarineGym      250k FPS   (8,000 envs, RTX 3060)  ← throughput king
Isaac AUV       ~90k FPS  (Isaac Lab benchmark)    (unverified for AUV)
OceanScale      85.8k FPS (64 envs, RTX 5090)      ← measured launch bench
HoloOcean         ~800 FPS (single env, CPU)
Stonefish         ~500 FPS (single env, CPU)        (estimated, not measured)
DAVE              ~200 FPS (single env, CPU Gazebo)
```

### Where Each Simulator Leads

| Simulator | Leads at | Trade-off |
|-----------|----------|-----------|
| **OceanScale** | Open GPU-native Fossen on Newton (Apache-2.0). Warp-native hydro kernels. Clean Gymnasium API. | Single vehicle, stub-level IMU/DVL/pressure only, no sonar/camera/acoustic comms. |
| **MarineGym** | RL throughput (250k FPS, 8k envs). Most GPU-parallel underwater RL platform. | Stale (4 mo). Isaac Sim heavy. No sensor models. No sim-to-real. |
| **OceanSim** | Sensor fidelity (sonar + DVL + underwater camera). Only sim with Akkaynak-Treibitz rendering. | No hydrodynamics. No RL training loop. Stale (8 mo). Single vehicle. |
| **HoloOcean** | Sim-to-real validation (<2% vs REMUS). Best sensor suite (ray-traced sonar, acoustic comms, caustics). | CPU-bound. No GPU parallelism. UE5 build friction. Not designed for RL. |
| **Stonefish** | Mature C++ physics. Best sensor variety (event camera, thermal). ROS 2 native. | GPL-3.0 (license blocker). CPU-bound. No GPU parallelism. Single maintainer. |
| **DAVE** | Lowest setup friction (apt install). Gazebo + ROS 2 standard. Community-tested. | Weakest physics (no added mass, no Coriolis). CPU-bound. |
| **Isaac AUV** | Strongest RL transfer evidence (zero-shot, beats PID). Isaac Lab ecosystem. Differentiable (MJX). | Stale (9 mo). No Fossen-tier hydro. Isaac Sim dependency. Not open to extensions. |

---

## Gap Analysis

No simulator combines all six critical capabilities for underwater robotics RL at scale:

| Capability | OceanScale | MarineGym | OceanSim | HoloOcean | Stonefish | DAVE | Isaac AUV |
|-----------|:---------:|:---------:|:--------:|:---------:|:---------:|:----:|:---------:|
| GPU-parallel Fossen hydro | Yes | Yes | No | No | No | No | Partial (ellipsoid) |
| Ray-traced sonar | No | No | Yes | Yes | Yes | Minimal | No |
| Underwater camera model | No | No | Yes (Akkaynak-Treibitz) | Yes (Lumen + caustics) | No | No | No |
| Acoustic comms | No | No | No | Yes | No | No | No |
| GPU-parallel RL (>10k envs) | Planned | Yes (8k) | No | No | No | No | Yes |
| Open license (Apache/MIT/BSD) | Apache-2.0 | MIT | BSD-3 | MIT | GPL-3.0 (no) | BSD-3 | BSD-3 |
| Sim-to-real validated | Planned | No | No | Yes (<2%) | Yes (pool) | Limited | Yes (zero-shot) |

---

## Source Citations

All claims sourced from:

1. **SURVEY.md** — simulator landscape, versions, throughput figures, gap analysis (§§1-6)
2. **REFERENCES.md** — license verification, code structure, activity tiers, vendor/port status (§§1-11)
3. **ARCHITECTURE.md** — OceanScale current shipping stack (Newton 1.2 + Warp 1.13 + SB3)
4. **benchmarks/competitive/results.json** — OceanScale standardized launch benchmark (85,824 env-steps/s at n=64, RTX 5090)
5. **benchmarks/oceanscale_vs_bullet_results.json** — historical PyBullet comparison sweep
6. **pyproject.toml** — OceanScale dependencies and version pins

### Paper references

- MarineGym: Chu et al., arXiv 2503.09203, IROS 2025
- OceanSim: Song et al., arXiv 2503.01074, IROS 2025
- HoloOcean 2.0: Potokar et al., arXiv 2510.06160, 2025
- Stonefish: Cieślak et al., arXiv 2502.11887, ICRA 2025
- Isaac AUV ("Learning to Swim"): Cai et al., arXiv 2410.00120, ICRA 2025
- DAVE: field-robotics-lab.github.io/dave.doc/

### Unverified claims

- HoloOcean license listed as MIT per SURVEY.md but marked "(verify)" in REFERENCES.md — **unverified**
- Isaac AUV throughput for AUV task specifically — only RSL-RL Spot locomotion benchmark cited (~90k FPS), not AUV dynamics
- Stonefish FPS estimated, not measured in SURVEY.md
- DAVE FPS estimated from Gazebo ODE typical performance
