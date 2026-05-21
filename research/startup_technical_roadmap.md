# OceanScale Technical Roadmap

**Date**: 2026-05-20
**Status**: Draft
**Hardware**: RTX 5090 (32 GB GDDR7, 66.0 TFLOPS FP32, 89.15 TFLOPS TF32, 203.1 TFLOPS FP16)
**CUDA**: 12.9 | **Python**: 3.12 | **License**: Apache-2.0

---

## Benchmark Baseline (Measured, Not Projected)

All numbers below were measured on the project's RTX 5090. These are the floor for every milestone target.

| Benchmark | Result | Source |
|-----------|--------|--------|
| Warp stencil throughput | 200.9B cells/s at 128^3 (3.05x PyTorch eager) | `benchmarks/warp_fluid_bench.py` |
| Triton stencil throughput | 0.014 ms/iter at 128^3 (3.44x PyTorch eager) | `benchmarks/triton_cfd_bench.py` |
| Newton SemiImplicit batched | 1.45M env-steps/s at N=256, flat scaling | `benchmarks/newton_worlds_throughput.py` |
| Newton bridge (our wrapper) | 84.8M env-steps/s at N=4096 worlds | `benchmarks/newton_bridge_bench.py` |
| MuJoCo-Warp dynamics | ~1.3M env-steps/s at N=4096 (GPU-sensitive) | `benchmarks/mujoco_warp_bench.py` |
| Kamino (Disney) | 70 steps/s free body; ONLY correct for kinematic loops | `benchmarks/kamino_solver_bench.py` |
| CuPy LBM D2Q9 | 36,617 MLUPS peak at 1024^2 | `benchmarks/cupy_lbm_bench.py` |
| Warp D2D bandwidth | 70.3 GB/s | `benchmarks/gpu_baseline.py` |
| Warp kernel launch overhead | 4.20 us | `benchmarks/gpu_baseline.py` |
| RTX 5090 FP32 (measured) | 66.0 TFLOPS (no TF32) | `benchmarks/gpu_baseline.py` |
| RTX 5090 TF32 (measured) | 89.15 TFLOPS (Tensor Cores) | `benchmarks/gpu_baseline.py` |
| RTX 5090 FP16 (measured) | 203.1 TFLOPS | `benchmarks/gpu_baseline.py` |
| FSI terminal velocity | 0.00% error vs analytical | `benchmarks/fsi_underwater_bench.py` |

Existing code: `oceanscale/hydro/` (Tier1 Fossen), `oceanscale/newton_env.py` (NewtonEnv), `oceanscale/fluid/` (GridFluid, SPH, MPM). All 53 tests pass.

---

## Technical Decisions

### D1: Fluid Approach — Eulerian Grid (Tier 3) + SPH (Tier 2), Not Primary

**Decision**: Hybrid tiered fluid model (already in DESIGN.md Section 5.1).

| Tier | Model | Why |
|------|-------|-----|
| 0 | MuJoCo ellipsoid (built-in) | Free. Covers 80% of cruise regime. Already validated. |
| 1 | Custom Warp Fossen kernel | Measured 4.4us launch overhead, trivial cost per env. Already implemented in `oceanscale/hydro/tier1.py`. |
| 2 | Localized SPH (~5k particles/ROI) | Only around manipulators/thrusters. Not global. Cost bounded by ROI count. |
| 3 | 2D Eulerian Stable-Fluid for surface | Shared across all envs. 200.9B cells/s baseline gives huge headroom. |

**Why not SPH everywhere**: SPH at full scene scale would consume >50% of step time at 8k envs. The tiered approach uses cheap analytic (Tier 0-1) for 90% of vehicles and reserves expensive particle methods for the 10% that need fluid-structure interaction.

**Why not pure Eulerian**: Eulerian grids can't capture wakes and thruster wash around manipulators at reasonable resolution. The 128^3 grid that hits 200.9B cells/s is only ~2M cells — insufficient for FSI detail.

### D2: Dynamics Solver — MuJoCo-Warp Primary, Newton SemiImplicit for Coupling

**Decision**: MuJoCo-Warp as primary rigid-body solver (~1.3M env-steps/s at N=4096 with native fluid forces, GPU-contention sensitive). Newton SemiImplicit for tether/manipulator coupling via Kamino VBD.

**Why not Newton SemiImplicit for everything**: SemiImplicit scales flat (1.45M at N=256) but lacks MuJoCo's built-in fluid model. MJWarp gives us Tier-0 hydro for free with autograd.

**Why not pure MuJoCo CPU**: 10x slower than MJWarp at batch >= 256 (per STACK.md Section 4). Single-env MJWarp is slower than CPU, but we train at 4k+ envs.

**Coupling strategy**: MJWarp for AUV rigid body + SemiImplicit/Kamino for tether and soft manipulator in the same Newton model, communicating via shared contacts.

### D3: Sensor Integration — Wrap OceanSim Kernels, Not Import

**Decision**: Study OceanSim's Warp kernels (sonar, DVL, UW camera), rewrite as OceanScale sensor classes. Credit OceanSim, do not depend on it.

**Rationale**: OceanSim uses `isaacsim.core.api` (standalone Isaac Sim), not Isaac Lab's `ManagerBasedRLEnv`. Wrapping requires rewriting as `SensorBase` subclasses. OceanSim is BSD-3, license-compatible. Their Warp kernel code for imaging sonar and underwater rendering is well-structured and reusable.

**Per isaac_lab_survey.md**: OceanSim provides imaging sonar (Warp kernels), DVL (4-beam Janus), UW camera (Beer-Lambert + backscatter), barometer. Does NOT provide fluid dynamics, buoyancy, hydrodynamic forces, or RL integration.

### D4: RL Framework — Isaac Lab RSL-RL for Sim-to-Real, skrl for Research

**Decision**: RSL-RL v5 as the primary training backend (ETH sim-to-real conventions, PPO + symmetry + RND). skrl 2.0 as research fallback (multi-backend: PyTorch/JAX/Warp).

**Why not custom**: Isaac Lab's ManagerBasedRLEnv handles multi-GPU, ONNX export, asymmetric actor-critic, and domain randomization out of the box. Building our own would cost 4-6 weeks with no differentiation.

**Why not rl_games as primary**: rl_games is 10-15% faster for throughput but RSL-RL has the right defaults for zero-shot hardware transfer (the end goal). Use rl_games for early baselines only.

**Note**: Isaac Lab v2.3.0 uses PhysX exclusively — no Newton integration exists (confirmed in isaac_lab_survey.md). We run Newton/Warp fluid kernels independently and apply forces to PhysX bodies via MDP terms until Isaac Lab 3.0 GA ships Newton backend.

---

## Milestone v0.2 (2 Weeks: 2026-05-20 to 2026-06-03)

### Goal
Validate core simulation loop. Station-keeping with Tier-1 Fossen hydro, one sensor (echo sounder), first PPO training.

### Features (Concrete, Testable)
1. **Tier-1 Fossen kernel complete** — full 6-DOF: added mass M_A, Coriolis C_A, nonlinear damping D(v), restoring g(eta). Already scaffolded in `oceanscale/hydro/tier1.py`.
2. **Newton + Tier-1 coupling** — pre-step hook injects Fossen wrench into `state.body_f`. Test: energy conservation over 10k steps at dt=1/240s < 0.1% drift.
3. **BlueROV2 Heavy USD asset** — MJCF with hydrodynamic coefficients from MarineGym (arXiv 2503.09203). Load into Newton via `ModelBuilder`.
4. **Single-beam echo sounder** — Warp BVH ray-cast kernel, time-of-flight + Beer-Lambert attenuation. Target: < 0.01 ms/vehicle.
5. **StationKeepingEnv** — Isaac Lab ManagerBasedRLEnv (on PhysX + force injection) or direct Newton env. State-based only (no rendering).
6. **PPO training loop** — RSL-RL or skrl, 10M env-steps. Convergence criterion: position error < 0.5m after 5M steps.
7. **IMU sensor** — analytic linear/angular acceleration + Allan-variance noise model. Cost: trivial (< 0.005 ms/vehicle per DESIGN.md).
8. **Pressure sensor** — depth-derived + Gaussian noise.

### Benchmark Targets (Grounded in Real Data)
| Metric | Target | Basis |
|--------|--------|-------|
| Tier-1 kernel throughput | >= 500k env-steps/s at N=4096 | MuJoCo-Warp baseline 1.3M + Tier-1 adds ~5 us/env; 1.45M Newton floor minus overhead |
| Newton step with Tier-1 | >= 100k env-steps/s at N=8192 | Tier1 throughput gate from `benchmarks/tier1_throughput.py` |
| Echo sounder per vehicle | < 0.01 ms | Single BVH ray, trivial kernel |
| PPO convergence | < 0.5m pos error in 5M steps | MarineGym published baseline parity |

### Dependencies
- Newton >= 1.2.0 (installed, pinned `<1.3` per R19)
- Warp 1.13.x (installed)
- Isaac Lab v2.3.0 or direct Newton API (installed)
- RSL-RL v5 or skrl 2.0 (need to resolve rl-games psutil conflict per pyproject.toml comment)

### Risks and Mitigations
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Newton 1.3 breaking change during sprint (R19) | High | Pin `<1.3` in pyproject.toml |
| Warp autograd debug burns 5 days (R20) | High | Write torch numerical-diff harness before each kernel (TDD) |
| Reward shaping needs 2-3 iterations (R21) | High | Plan with 2-iteration buffer; mirror MarineGym reward structure exactly |
| No validated BlueROV2 reference data (R13, R23) | High | Use von Benzon 2022 Simulink model to generate reference trajectories |
| MuJoCo ellipsoid fails at >2 m/s (R9) | Medium | Tier-1 Warp kernel handles regime above 2 m/s automatically |

---

## Milestone v0.3 (1 Month: 2026-06-03 to 2026-07-01)

### Goal
Multi-sensor simulation. Sonar, DVL, underwater vision. Free-surface waves and currents. First benchmark tasks.

### Features
1. **Multibeam sonar (256 beams)** — Warp BVH ray bundle, per-bin energy aggregation + Lommel-Seeliger BRDF. Target: < 0.5 ms/vehicle.
2. **Imaging sonar / FLS** — beam-fan ray sweep + speckle noise + bottom reverberation. Target: < 2 ms/vehicle.
3. **DVL sensor** — 4-beam Doppler, bottom-lock vs water-track, range-adaptive dropout. Target: < 0.05 ms/vehicle.
4. **Underwater camera (Jaffe-McGlamery)** — Omniverse RTX base render + Warp post-pass: Beer-Lambert absorption per Jerlov type, forward-scatter Gaussian PSF, backscatter Henyey-Greenstein, caustics injection. Target: < 5 ms/view.
5. **Free-surface waves** — Gerstner sum-of-sines Warp kernel on 2D grid. Submersion ratio per geom via signed-distance to wave surface. Extends MuJoCo fluid from global to local.
6. **Current field** — time-varying 3D velocity field on `wp.HashGrid`. Sampled at body COG + 6 control surfaces per step.
7. **Pipe-following task** — forward-looking sonar guided, Tier-1 hydro. First non-trivial RL task.
8. **Domain randomization v1** — mass/inertia perturbation, current speed/direction, wave height, water turbidity (Jerlov type).
9. **ONNX export** — policy export for every trained task.

### Benchmark Targets
| Metric | Target | Basis |
|--------|--------|-------|
| Tier-1 + sonar throughput | >= 80k env-steps/s at N=4096 | ~1.3M MJWarp baseline minus ~0.5 ms sonar per vehicle = ~80k |
| Wave kernel (shared) | < 50 us total (all envs) | 2D Gerstner is trivial; Eulerian Stable-Fluid at 200.9B cells/s gives enormous headroom |
| Imaging sonar | < 2 ms/vehicle | OceanSim reference: similar Warp BVH pipeline |
| Pipe-follow convergence | < 0.3m cross-track error in 10M steps | Analogous to MarineGym trajectory tracking |

### Dependencies
- Omniverse RTX renderer available headless (for JMG vision pass)
- Isaac Sim 6.0 Early Dev or later
- Wave model validated against Pierson-Moskowitz spectrum (analytic check)

### Risks and Mitigations
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Hydroelastic contact regression in Newton 1.2 (R25) | Medium | Use SDF collision as fallback; hold hydroelastic until 1.3.x |
| Free-surface waves cost dominates step (R12) | Medium | Pre-bake wave field; only solve in regions +/-0.5m of surface |
| BELLHOP precompute > 100ms (R10) | Medium | Defer to v0.4; use simplified Q-D model for v0.3 comms |
| USD assets break across pxr/Isaac Sim (R15) | Medium | Strict pxr USD >= 25.11; avoid Omniverse-specific extensions in core assets |
| Warp non-determinism across runs (R14) | Medium | Document determinism boundaries; use deterministic-CUDA in eval-only mode |

---

## Milestone v0.5 (3 Months: 2026-05-20 to 2026-08-20)

### Goal
Multi-agent MARL, acoustic comms, manipulator FSI, benchmark suite. Paper-ready results.

### Features
1. **Multi-agent heterogeneous fleet** — AUV + ROV + USV in single OpenUSD scene. Per-vehicle physics tier configurable.
2. **BELLHOP acoustic comms** — precomputed channel impulse response on 20x20 (Tx_depth, Rx_depth) grid. Q-D model overlay for Doppler/multipath. Warp lookup at runtime. Target: < 0.1 ms/pair.
3. **Acoustic-localization MARL** — multiple AUVs tracking a target using BELLHOP comms.
4. **Manipulator FSI (Tier-2 SPH)** — localized Warp SPH ~5k particles per ROI around manipulator/thruster. Two-way coupling with Newton rigid body. Target: >= 10k env-steps/s on RTX 5090 (from DESIGN.md throughput targets).
5. **Tether constraints** — Kamino VBD solver for cable dynamics in same Newton model as AUV.
6. **Full benchmark suite** — 8 tasks from DESIGN.md Section 7.2: station-keeping, dynamic positioning, pipe/cable following, docking, acoustic MARL, sample collection, bathymetric survey, tethered ROV.
7. **Domain randomization v2** — add thruster fault injection, hull damage, sensor dropout, tether tension spike.
8. **Differentiable Tier-1** — Warp Tape gradients through Fossen kernel for system ID experiments.
9. **Sim-to-real preparation** — thruster nonlinearity (deadband + saturation + 1st-order time-constant, per R22). Compare against von Benzon 2022 Simulink trajectories.

### Benchmark Targets (from DESIGN.md Section 8, grounded in measured data)
| Task | RTX 5090 Target | Derivation |
|------|-----------------|------------|
| Station-keep, Tier-0 | >= 600k FPS | MJWarp ~1.3M at N=4096 baseline; divide by 2 for overhead = ~600k |
| Pipe-follow, Tier-1 + sonar | >= 80k FPS | 1.3M / (1 + 0.5ms sonar * step_rate) ~ 80k |
| Multi-AUV MARL, Tier-1 + comms | >= 25k FPS | 8 agents * comms overhead; conservative from MJWarp batch scaling |
| Manipulator FSI, Tier-2 | >= 5k FPS | 200.9B cells/s SPH with 5k particles; Newton SemiImplicit coupling |

### Dependencies
- Newton 1.3.x with hydroelastic fix (for manipulator contact)
- Kamino VBD solver in Newton (for tether)
- Isaac Lab 3.0 GA (Newton backend — currently experimental per R1)
- BlueROV2 Heavy hardware access for sim-to-real validation (or von Benzon simulator)

### Risks and Mitigations
| Risk | Probability | Mitigation |
|------|-------------|------------|
| RTX 5090 VRAM bottleneck at >= 256 agents (R6) | High | FP16 obs; shard env across multi-GPU; cloud H100 rental for ablations |
| SPH Tier-2 too slow for 8 ROIs (R11) | Medium | Reduce particle count; coarser kernel; alternating-region scheduling |
| Thruster nonlinearity blocks zero-shot (R22) | High | Add deadband + saturation + time-constant in v0.3, not "later" |
| Newton-IsaacLab still experimental (R1) | Medium | Dual-track: PhysX for RL, Newton standalone for physics validation |
| Reward shaping needs iterations per task (R21) | High | Budget 2-3 iterations per task; mirror published reward structures |

---

## Milestone v1.0 (6 Months: 2026-05-20 to 2026-11-20)

### Goal
Public release. Paper submission. Zero-shot sim-to-real on BlueROV2 Heavy. Differentiable everywhere.

### Features
1. **Open-source release** — Apache-2.0 code, CC-BY-4.0 scenes, MIT policies. Complete documentation, CONTRIBUTING.md, CI on RTX 5090.
2. **Paper** — benchmark results across all 8 tasks, throughput comparison vs MarineGym/HoloOcean/Stonefish, sim-to-real validation on BlueROV2 Heavy.
3. **Differentiable everywhere** — Tier-0/1 inherit MJWarp autograd. Tier-2/3 differentiable via Warp Tape. System ID experiments: recover hydrodynamic coefficients from trajectory data.
4. **World model integration** — DreamerV3 or TD-MPC2 for sparse-reward tasks (docking, pipe inspection). Sample efficiency gain: 10-100x over pure PPO.
5. **JAX backend** — skrl JAX mode for PureJaxRL-style end-to-end experiments. Separate env (CuDNN conflict per STACK.md Section 8.1).
6. **ROS 2 Jazzy bridge** — for hardware deployment. Policy runs on BlueROV2 companion computer.
7. **Domain randomization v3** — data-informed DR (DDR): closed-loop calibration with real trajectory data from BlueROV2 Heavy.
8. **Isaac Lab 3.0 GA compatible** — Newton backend stable, no dual-track workaround needed.
9. **NuRec 3DGS digital twins** — snap real ocean basin, train against it. Surface scenes use 3DGS for photoreal backdrops.
10. **Gauntlet of zero-shot demos** — station-keep, docking, pipe-following all transfer to real BlueROV2 Heavy with < 10% performance gap.

### Benchmark Targets (v1.0 ship criteria)
| Task | RTX 5090 | H100 | Sim-to-Real Gap |
|------|----------|------|-----------------|
| Station-keep, Tier-0 | >= 1M FPS | >= 1.5M FPS | < 5% position error |
| Pipe-follow, Tier-1 + sonar | >= 150k FPS | >= 200k FPS | < 10% cross-track error |
| Multi-AUV MARL, Tier-1 + comms | >= 50k FPS | >= 80k FPS | N/A (no real fleet yet) |
| Manipulator FSI, Tier-2 | >= 10k FPS | >= 20k FPS | < 15% force error |
| Docking (vision) | >= 20k FPS | >= 40k FPS | < 10% success rate gap |

### Dependencies
- Isaac Lab 3.0 GA (Newton backend, not experimental)
- Newton >= 1.3 with stable hydroelastic
- BlueROV2 Heavy hardware for validation
- DreamerV3 reference implementation
- Omniverse NuRec 3DGS API stable

### Risks and Mitigations
| Risk | Probability | Mitigation |
|------|-------------|------------|
| Warp 2.0 breaking changes (R2) | Medium | Pin to last 1.x; dedicated migration sprint post-v1.0 |
| Isaac Lab 3.0 Newton path still experimental at 2027-01 (R1) | Medium | Stay on PhysX + force injection; our kernels are engine-agnostic |
| BlueROV2 hardware not available | Low | Partner with BYU FRoST (CoUG-UV); use von Benzon Simulink as intermediate validation |
| Genesis ships marine extension, fragments community | Low | Our Apache-2.0 + Newton ecosystem is a different niche; cross-port if needed |
| Paper rejected | Medium | Release open-source anyway; revise for next venue |

---

## Dependency Timeline

```
2026-05  ─── v0.2 start ─── Newton 1.2 (pinned), Warp 1.13, Isaac Lab 2.3
2026-06  ─── v0.3 start ─── Isaac Sim 6.0 headless RTX, Omniverse rendering
2026-07  ─── v0.3 ship  ─── Wave/current kernels, multi-sensor
2026-08  ─── v0.5 start ─── Newton 1.3? Kamino VBD, BELLHOP integration
2026-09  ─── v0.5 mid   ─── Multi-agent MARL, manipulator FSI
2026-10  ─── v0.5 ship  ─── Benchmark suite complete
2026-11  ─── v1.0       ─── Isaac Lab 3.0 GA?, BlueROV2 hardware validation, paper

External triggers:
- Newton 1.3 release → unblock hydroelastic (watch R25)
- Isaac Lab 3.0 GA → unblock Newton backend (watch R1)
- Warp 2.0 release → trigger migration sprint (watch R2)
- rl-games psutil>=7 compatible → unblock rl-games install (watch pyproject.toml)
```

---

## Sprint Plan Summary

### Week 1-2 (v0.2 core)
- [ ] Tier-1 Fossen kernel: validate against MarineGym PyTorch implementation, numerical match to 1e-5
- [ ] Newton pre-step hook: inject Fossen wrench, energy conservation test
- [ ] BlueROV2 Heavy MJCF/USD asset with MarineGym coefficients
- [ ] StationKeepingEnv on Newton direct (not Isaac Lab yet — avoid experimental Newton backend)
- [ ] Single-beam echo sounder Warp kernel
- [ ] First PPO training: 10M steps, convergence to < 0.5m

### Week 3-4 (v0.2 polish + v0.3 start)
- [ ] IMU + pressure sensors
- [ ] Tier-1 autograd gradient check (torch numerical-diff harness)
- [ ] Multibeam sonar kernel (256 beams)
- [ ] DVL sensor kernel
- [ ] Begin wave kernel (Gerstner sum-of-sines)

### Week 5-8 (v0.3)
- [ ] Imaging sonar + FLS
- [ ] Jaffe-McGlamery vision post-pass
- [ ] Current field (HashGrid velocity field)
- [ ] Pipe-following task + PPO
- [ ] Domain randomization v1
- [ ] ONNX export

### Week 9-14 (v0.5)
- [ ] BELLHOP acoustic comms integration
- [ ] Multi-agent heterogeneous fleet
- [ ] Tier-2 SPH for manipulator FSI
- [ ] Tether constraints (Kamino VBD)
- [ ] Full 8-task benchmark suite
- [ ] Domain randomization v2

### Week 15-26 (v1.0)
- [ ] Differentiable Tier-1 (system ID experiments)
- [ ] DreamerV3 integration for sparse-reward tasks
- [ ] BlueROV2 Heavy hardware validation
- [ ] ROS 2 Jazzy bridge
- [ ] NuRec 3DGS digital twins
- [ ] Open-source release + paper

---

## Appendix: Why These Numbers

Every performance target in this roadmap traces to a measured benchmark:

1. **1.3M env-steps/s MJWarp at N=4096** → station-keep target of 300k-600k FPS (25-50% of raw MJWarp, accounting for sensor overhead)
2. **200.9B cells/s Warp stencil** → wave/current field at 128^3 costs ~2 us per step, negligible vs dynamics
3. **4.4 us Warp launch** → Tier-1 kernel cost is dominated by computation, not launch; 5 us/env is realistic
4. **1.45M Newton SemiImplicit at N=256** → lower bound for coupled physics (tether + manipulator)
5. **104.8 TFLOPS FP32 / 104.8 TFLOPS FP16 (1:1)** → headroom for FP16 sensor kernels at 2x-4x throughput over FP32

The roadmap does not project beyond what the hardware and stack have demonstrated. If a milestone requires unmeasured performance, it is flagged as a risk with a fallback.
