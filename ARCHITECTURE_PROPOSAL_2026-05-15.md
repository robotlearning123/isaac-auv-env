# Next-Gen Underwater Robot Simulator — Architecture Proposal (v0.1.0)

**Date:** 2026-05-15  
**Status:** Draft for discussion  
**Companion:** `SURVEY.md` (state-of-the-art landscape, May 2026)

---

## 1. Mission

Build the **first GPU-native underwater robot simulator** that simultaneously delivers:

1. **Throughput** — ≥250k FPS / single H100 or RTX 5090 with full hydrodynamics (matches MarineGym, scales further on Blackwell).
2. **Sensor fidelity** — ray-traced sonar, Jaffe-McGlamery vision, DVL, IMU, pressure, acoustic comms — all GPU-parallelized.
3. **Multi-agent** — heterogeneous fleets (AUV + ROV + USV + UAV) sharing one OpenUSD scene.
4. **Validated dynamics** — Fossen 6-DOF baseline, free-surface waves, currents, fluid-structure coupling for soft manipulators.
5. **First-class RL** — Isaac Lab 3.0 envs + skrl/rl_games/RSL-RL/DreamerV3 backends, ONNX export, domain-randomization toolkit, sim-to-real benchmarks.
6. **Differentiable** — gradients through analytic hydro for system ID and policy gradient methods.

---

## 2. Why Now

- **Newton 1.0 GA + Isaac Lab 3.0 multi-backend** finally make USD-native, Warp-based, swappable-physics robotics simulation production-ready.
- **MuJoCo Warp** ships GPU-parallel fluid forces *plus* differentiable autograd — the analytic-hydro half is "free."
- **The community gap** (every existing underwater sim is incomplete in a distinct dimension) is wide enough that even a v0.2 unification would be a meaningful contribution.
- **Compute** — RTX 5090 (32 GB GDDR7, 21,760 CUDA cores, Blackwell) and B200 in cloud put 250k+ FPS underwater RL within reach for single-lab labs.

---

## 3. Tech Stack (decided)

| Layer | Choice | Rationale |
|---|---|---|
| Scene graph | **OpenUSD** | Single source of truth; native in Newton + Isaac Sim 6; tooling mature. |
| Physics core | **Newton 1.x** (MuJoCo-Warp primary solver, Kamino VBD secondary) | GA, Apache-2.0, GPU-native, differentiable, USD-native, multi-engine fallback. |
| Underwater hydro | **Custom Warp kernels** + **MuJoCo ellipsoid fluid model** (built-in fallback) | Best of both: analytic Fossen at AUV scale; Warp SPH for wakes/manipulator FSI. |
| Sensors | **Warp ray-tracing kernels** (sonar, vision physics), **PhysX RTX renderer** for visual sim-to-real | Match OceanSim's sensor approach + HoloOcean's RTX fidelity. |
| Renderer | **Omniverse RTX** in Isaac Sim 6.0 | Lumen-class lighting, caustics, Gaussian-Splat NuRec for digital twins. |
| Acoustic comms | **BELLHOP** wrapper + **Q-D model** as GPU kernel | Real-time enough for RL, calibrated against industry. |
| RL frontend | **Isaac Lab 3.0** env + **skrl 2.0** (default), **rl_games** (throughput), **RSL-RL** (sim-to-real PPO) | All three are env-agnostic; skrl spans PyTorch/JAX/Warp. |
| World model option | **DreamerV3** or **TD-MPC2** | Sample-efficient sim-to-real on sparse-reward AUV tasks. |
| Language | Python 3.12 API; Warp Python→CUDA for kernels; C++ for hot loops only | Match Isaac Lab 3.0 conventions. |
| License | **Apache-2.0** | Matches Newton, Isaac, OceanSim; enables industry adoption. |

---

## 4. Architecture Diagram (logical)

```
┌─────────────────────────────────────────────────────────────────┐
│                  RL FRONTEND (Isaac Lab 3.0 envs)               │
│   skrl • rl_games • RSL-RL • DreamerV3 • PureJaxRL adapter      │
└──────────────────────────────┬──────────────────────────────────┘
                               │ Warp arrays (zero-copy ↔ torch/jax)
┌──────────────────────────────▼──────────────────────────────────┐
│                 OCEANSCALE CORE (this project)                  │
│  ┌────────────┐ ┌───────────────┐ ┌──────────┐ ┌─────────────┐  │
│  │ Hydro      │ │ Sensors       │ │ Comms    │ │ Scenarios   │  │
│  │ • Fossen   │ │ • Sonar (RT)  │ │ BELLHOP  │ │ benchmark   │  │
│  │ • SPH wake │ │ • Vision JMG  │ │ Q-D      │ │ tasks +     │  │
│  │ • Waves    │ │ • DVL,IMU,P   │ │ acoustic │ │ DR toolkit  │  │
│  │ • Currents │ │ • Optical comm│ │          │ │             │  │
│  └─────┬──────┘ └────────┬──────┘ └────┬─────┘ └──────┬──────┘  │
│        │ (Warp kernels) all interface via Warp arrays            │
└────────┼─────────────────┼─────────────┼──────────────┼─────────┘
         ▼                 ▼             ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│  NEWTON 1.x (Warp + MuJoCo-Warp + Kamino + MPM + SDF contact)   │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│   OPENUSD SCENE  (vehicles, terrain, water, tether, payload)    │
│   Isaac Sim 6.0 / Omniverse RTX rendering / NuRec digital twin  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Hydrodynamics Design

### 5.1 Tiered Fluid Model
| Tier | Model | Use case | Cost |
|---|---|---|---|
| 0 | MuJoCo ellipsoid fluid (built-in) | Default AUV/ROV at cruise | free |
| 1 | Custom Warp Fossen kernel (added mass, drag, Coriolis, restoring) | Standard AUV control RL | ~few µs/vehicle/step |
| 2 | Warp SPH localized around manipulator/thruster | Soft manipulator FSI, thruster wash | bounded-domain only |
| 3 | Warp Stable-Fluid Eulerian for surface | Wave-coupled USV/buoy | 2D-coupled |
| 4 | (Future) External CFD pass (OpenFOAM) for offline calibration | Coefficient identification | offline |

### 5.2 Free Surface & Waves
- Gerstner/FFT wave field as Warp kernel on a 2D grid co-aligned with USD `MeshPrim`.
- Submersion ratio per geom computed via Warp signed-distance to wave surface (extends MJC fluid model from global to local).
- Currents as time-varying Warp velocity field; sampled at body COG and CFD-lite control surfaces.

### 5.3 Differentiability
- Tier-0 and Tier-1 inherit MJX/MJWarp autograd → gradients for system ID and direct policy gradient.
- Tier-2/3 differentiable per Warp's existing diff-SPH examples.

---

## 6. Sensor Design

### 6.1 Sonar (highest priority differentiator)
- Warp BVH ray cast → per-bin energy aggregation (matches OceanSim approach).
- Three modalities at v0.2: forward-looking imaging, side-scan, multibeam.
- Speckle noise + bottom reverberation per Lambertian + Lommel-Seeliger BRDF.
- Dynamic scene support out of the box (no octree precompute, like HoloOcean 2.0).
- Target: ≥100 Hz per vehicle, ≥256 parallel vehicles per RTX 5090.

### 6.2 Vision (Jaffe-McGlamery)
- Omniverse RTX path-traced RGB as base.
- Post-pass Warp kernel applies: absorption (Beer-Lambert per Jerlov water type), forward-scatter (Gaussian PSF), backscatter (Henyey-Greenstein), caustics injection.
- Differentiable on demand for inverse rendering / image-enhancement training.

### 6.3 DVL / IMU / Pressure / GPS
- DVL: 4-beam Doppler; bottom-lock vs water-track logic; range-adaptive dropout (per OceanSim).
- IMU: Allan-variance-calibrated noise, axis misalignment, bias instability.
- Pressure: depth-derived + sensor noise; used for buoyancy depth feedback.
- GPS: surface-only with cold-start mask underwater.

### 6.4 Acoustic Modem
- BELLHOP normal-mode precompute → Warp lookup at runtime per (Tx, Rx) pair.
- Q-D model overlay for time-varying / Doppler.
- Bitrate / packet-loss simulation for MARL benchmarks.

---

## 7. Multi-Agent & Scenarios

### 7.1 Heterogeneous Fleet
- Single OpenUSD scene supports AUV (BlueROV2, BlueROV2 Heavy, IVER3, REMUS-class, CoUG-UV), ROV (manipulator-equipped), USV (WAM-V class), small UAV (for cross-domain experiments).
- Per-vehicle physics tier configurable.

### 7.2 Benchmark Tasks (v0.2 ship list)
1. **Station-keeping under current+wave** (single AUV, baseline from MarineGym).
2. **Dynamic positioning** (ROV with manipulator load).
3. **Pipe / cable following** (forward-looking sonar).
4. **Docking** (visual + sonar, multi-stage).
5. **Acoustic-localization MARL** (multiple AUV tracking a target, BELLHOP comms).
6. **Sample collection** (manipulator, FSI tier-2).
7. **Bathymetric survey** (sidescan, side-by-side coverage planning).
8. **Tethered ROV operations** (tether constraint solver via Newton VBD).

### 7.3 Domain Randomization
- Drop-in DR module: mass/inertia, COB-COM offset, thruster gain & deadband, current speed/direction, wave height/period, water turbidity (Jerlov type), DVL dropout, sonar SNR, IMU bias, payload mass, hull damage.
- DDR (Data-Informed) hooks for closed-loop calibration with real trajectory data.

---

## 8. RL Frontend

- **Default backend:** skrl 2.0 (PyTorch + Warp), exposes `gym.Env`-compatible vectorized API on top of Newton tensors.
- **High-throughput PPO:** rl_games 1.6.5, multi-node via NCCL.
- **Sim-to-real PPO:** RSL-RL v5 with symmetry augmentation + RND.
- **World model:** DreamerV3 reference implementation; TD-MPC2 alternative.
- **ONNX export** required for every benchmark policy.
- **Asymmetric actor-critic** with privileged info (ground-truth currents, ground-truth obstacles) standard.

### Throughput targets (v0.5)
| Task | Single H100 | Single RTX 5090 |
|---|---|---|
| Station-keep, tier-0 | ≥1M FPS | ≥600k FPS |
| Pipe-follow, tier-1 + sonar | ≥150k FPS | ≥80k FPS |
| Multi-AUV MARL, tier-1 + comms | ≥50k FPS | ≥25k FPS |
| Manipulator FSI, tier-2 | ≥10k FPS | ≥5k FPS |

---

## 9. Roadmap (v0.x.x)

| Version | Scope | Effort |
|---|---|---|
| **v0.1.0** | Project scaffold, Newton + Isaac Sim 6 dev container, USD scene loader, one AUV (BlueROV2 Heavy), Tier-0 fluid, IMU/pressure, Isaac Lab env wrapper, station-keep task | 4–6 weeks |
| **v0.2.0** | Tier-1 Fossen Warp kernel, DVL, sonar v1 (single-beam + multibeam), DR toolkit, MarineGym task parity | 6–8 weeks |
| **v0.3.0** | Free-surface waves, currents, Jaffe-McGlamery vision pass, sidescan/FLS sonars, ONNX export, sim-to-real benchmark with BlueROV2 | 8–10 weeks |
| **v0.4.0** | Multi-agent + BELLHOP acoustic comms, MARL benchmarks, tether constraints, manipulator FSI tier-2 | 8–10 weeks |
| **v0.5.0** | Public release + paper; throughput targets met; Isaac Lab 3.0 GA compatible | 4–6 weeks |
| **v1.0** | Differentiable everywhere, world-model integration, gauntlet of zero-shot sim-to-real demos | post-paper |

---

## 10. Open Risks & Decisions Needed

1. **Tier-1 vs Tier-2 default cost** — Fossen-only is fast but misses thruster wash & manipulator wakes; SPH everywhere kills throughput. Default to Tier-1 with opt-in Tier-2 around manipulators?
2. **Newton vs MJX direct** — Newton is "right" long-term but Isaac Lab 3.0 Newton path is experimental ("breaking changes expected"). MJX direct (à la `arXiv 2512.13359`) is more stable today. Recommendation: dual-track — keep MJX backend until Newton+IsaacLab GA stabilizes.
3. **Acoustic comms granularity** — full BELLHOP runs are ~10 ms; cache vs runtime trade-off must be benchmarked.
4. **Real hardware validation partner** — BlueROV2 Heavy (cheap, ubiquitous), or partner with BYU FRoST (CoUG-UV)? Both are practical.
5. **OceanSim/MarineGym integration vs greenfield** — license-compatible (Apache-2.0 / MIT). Recommend wrapping MarineGym's hydro plugin as our Tier-1 reference and adopting OceanSim's image-formation as our Jaffe-McGlamery pass — credit, do not reinvent.
6. **License of trained policies and benchmark scenes** — recommend CC-BY-4.0 for scenes, Apache-2.0 for code, MIT for policies.
7. **Compute target** — single-GPU first (RTX 5090 lab default), then multi-node H100/B200 once GA.

---

## 11. What this proposal is NOT
- Not a CFD solver. CFD is offline-calibration only.
- Not a replacement for Stonefish / HoloOcean for high-fidelity single-vehicle ROS pipelines — those remain the gold standard for ROS2 production.
- Not a generative world model — that is a v1.0+ stretch, post-paper.

---

## 12. Immediate Next Steps (Week 1)

1. Stand up an Isaac Sim 6.0 + Isaac Lab 3.0 dev container; confirm Newton GPU path works on the lab's RTX 5090.
2. Reproduce **MarineGym station-keeping** as the throughput baseline (250k FPS RTX 3060 → expect ≥700k on RTX 5090 with same dynamics).
3. Reproduce **OceanSim sonar** in a Warp kernel atop Newton's USD scene.
4. Write the **Tier-1 Fossen Warp kernel** with autograd; gradient-check against analytic Fossen.
5. Open the public repo skeleton + CI; write the **v0.1.0** task spec.
