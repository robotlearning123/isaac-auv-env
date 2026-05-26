# OceanScale — Architecture

**Status:** Current shipping stack. Last revised 2026-05-26.
**Authority:** This file describes what OceanScale ACTUALLY ships today. For the original architectural ambitions, see ARCHITECTURE_PROPOSAL_2026-05-15.md (archived). For positioning law, see POSITIONING.md.

---

## Stack (current v0.0.2)

| Layer | Choice | Version |
|-------|--------|---------|
| Physics core | Newton (GPU-native, USD-native) | >=1.2.0,<1.3 |
| GPU kernels | NVIDIA Warp (Python-to-CUDA, autograd) | >=1.13.0,<2.0 |
| Simulation orchestrator | OceanSim — single step() for physics + hydro + ocean + sensors | (internal) |
| Fluid simulation | Multi-fidelity ladder: Grid / SPH (WCSPH) / MPM / Volume | (internal) |
| RL framework | Stable-Baselines3 (PPO baseline) | >=2.5 |
| Isaac Lab integration | DirectRLEnv 3 native + standalone fallback | >=1.2.3 |
| Vectorization | Custom BatchedVecEnv on top of Newton tensors | (internal) |
| Renderer (current) | matplotlib Agg + imageio-ffmpeg (headless MP4) | (internal) |
| Hydrodynamics | Fossen 6-DOF + MuJoCo drag/lift + distributed drag + partial submersion | (internal) |
| Sensors | Ray-DVL, Ray-Sonar, ImagingSonar, UW Camera, Multibeam, 9 real DVL configs | (internal) |
| Rendering | Physics-based underwater (10 Jerlov water types, Warp GPU) | (internal) |
| Controllers | PID, NL-PID, Lee geometric, sliding mode | (internal) |
| Validation reference | von Benzon 2022 6-DOF BlueROV2 port | (internal) |
| Benchmarks | GPU kernel throughput suite | (internal) |
| Python | CPython | 3.12 or 3.13 |
| CUDA | NVIDIA | 12.8+ |
| GPU | Reference platform | RTX 5090 |
| License | Apache-2.0 | — |

**Explicitly NOT yet shipping**: Isaac Sim 6.0 rendering layer (installing, blocked on GA), Omniverse RTX camera capture, BELLHOP, Gaussian-Splat scene capture, world-model layer. These are documented in ARCHITECTURE_PROPOSAL_2026-05-15.md as future ambitions.

---

## Two-layer architecture (per POSITIONING.md §4)

OceanScale is structured as a tiered hierarchy. Only Tier 1 ships today.

| Tier | Concept | Status |
|------|---------|--------|
| 1 | Ocean simulator (GPU-native Newton + Warp + Fossen hydrodynamics) | Shipping |
| 2 | Ocean world model (learned dynamics layer) | Future; not in v0.0.2 |
| 3 | Ocean foundation model (data + model + distribution leverage) | Research target; not in v0.0.2 |

---

## Core: OceanSim orchestrator

`oceanscale/sim.py` — `OceanSim` class. Single `step()` call composes:

1. Newton rigid-body physics step
2. Tier-1 Fossen hydrodynamics (added mass, Coriolis, damping, restoring)
3. Ocean state (water column properties, currents, waves)
4. Sensor updates (DVL, sonar, camera, IMU, pressure)
5. Domain randomization (if enabled)

Replaces the three parallel integration paths (ROVEnv, OceanWorld, UnifiedDemo) with one shared orchestration loop. All components share the same physical world and simulation time.

Configuration via `OceanSimConfig` dataclass: vehicle, ocean params, sensor mounts, n_envs, dt, device, seabed mesh, domain randomization ranges.

---

## What ships now

### Vehicles (10 underwater robots)
- BlueROV2 Heavy (8 thrusters, Fossen + von Benzon 2022) — from bluerov2_gz
- BlueROV MarineGym (6 thrusters, MarineGym hydro YAML) — from MarineGym
- WarpAUV (6 thrusters, MIT) — from isaac-auv-env
- RexROV (8 thrusters, 1863 kg) — from UUV Simulator
- Slocum Glider, Wave Glider, WHOI Hybrid Glider — from DAVE
- eROV — from DAVE
- TendonFish (biomimetic, experimentally validated fluid coefficients) — from fishsim

### Hydrodynamics (5 models)
- Tier-1 Fossen 6-DOF: added mass, Coriolis, damping, restoring (Warp kernels)
- MuJoCo-style drag: geometry-inferred quadratic drag + Kutta lift + Magnus lift (Warp kernels, fishsim-validated coefficients)
- Distributed drag: per-link independent drag for articulated bodies (Warp kernel)
- Partial submersion: linear buoyancy transition at free surface for amphibious sim (Warp kernel)
- CMA-ES system identification reference framework

### Fluid simulation (multi-fidelity ladder)

`oceanscale/fluid/` — 5 fidelity levels, all GPU-native on Warp:

| Level | Method | Use case |
|-------|--------|----------|
| 0 | None | Fossen analytical only (no fluid mesh) |
| 1 | Grid Eulerian | Chorin projection, spatially varying currents |
| 2 | SPH (WCSPH) | Lagrangian particles, fluid-body interaction |
| 3 | MPM | Material Point Method via Newton, deformable terrain |
| 4 | Volume | wp.Volume NanoVDB sparse grid, ocean-scale domains |

WCSPH solver (`fluid/sph.py`): cubic spline kernel (Monaghan 1992), Warp HashGrid neighbor queries, Tait equation of state, seawater defaults (ρ=1025 kg/m³).

Additional fluid modules: `wave.py` (ocean wave field), `wave_fft.py` (FFT spectral waves), `grid.py` (Eulerian solver), `mpm.py` (Newton MPM), `volume_solver.py` (NanoVDB), `surface_extractor.py`, `mesh_boundary.py`, `adaptive_grid.py`, `differentiable.py` (autograd-compatible), `mpm_coupling.py`.

### Propulsion
- Generic propeller thruster allocation matrix
- T200 thruster model (MarineGym, Warp kernel): deadband, time constants, quadratic thrust curve
- Buoyancy engine (glider-style)
- Flapping fin locomotion

### Sensors
- Ray-DVL (Warp mesh raycasting, 4-beam Janus)
- Ray-Sonar (Warp fan-beam scan)
- ImagingSonar (5 Warp kernels, Oculus M370 defaults, polar-binned 2D image) — from OceanSim
- Underwater Camera (`sensors/underwater_camera.py`) — GPU pinhole ray casting + Beer-Lambert attenuation, Jerlov water presets, synthetic RGB + depth output
- Multibeam echo sounder (`sensors/multibeam.py`) — fan-shaped beam array, bathymetric depth profiles, Kongsberg EM-2040 defaults
- 9 real-world DVL configs (Nortek DVL1000/500, Teledyne Explorer/Pathfinder/Pioneer, Sonardyne Syrinx, LinkQuest NavQuest, Rowe SeaPilot, RDI WorkHorse) — from DAVE
- IMU, pressure, magnetometer, USBL, acoustic modem, sidescan sonar

### Rendering
- Physics-based underwater rendering (10 Jerlov water types, Beer-Lambert + backscatter, Warp GPU) — from OceanSim
- Caustic overlay, vignette, film grain
- HUD overlay with mission telemetry
- Video exporter (MP4)

### Controllers
- PID (position, depth, heading hold) with per-vehicle presets
- NL-PID (nonlinear PID for underwater vehicles)
- Sliding mode control
- Lee geometric position controller
- Vehicle-specific config YAMLs (BlueROV, RexROV)

### Task environments
- `envs/DockingApproachEnv` — approach and dock with a station
- `envs/CurrentStationKeepingEnv` — hold position against ocean currents
- `envs/WaypointFollowingEnv` — navigate through waypoints
- All environments integrate with OceanSim orchestrator

### Environments (7 underwater scenes)
- Shipwreck, Munkholmen, BOP panel — from UUV Simulator
- Santorini terrain — from DAVE
- Pipeline, rock formations — from SMARC
- Sand heightmap + seabed — from bluerov2_gz

### Training
- Vectorized environments (BatchedVecEnv shape (n_envs, ...))
- Domain randomization (volume, COM-COB offset, drag, thruster noise) — from isaac-auv-env
- PPO hover training (converges, EV ~0.9 at 1M steps)
- Isaac Lab 3 native DirectRLEnv (`training/isaaclab_task.py`):
  - Wrench composer: converts Fossen hydro forces to Newton body forces
  - JIT reward: `torch.jit.script` exponential distance reward
  - Scene setup: RigidObject ROV + terrain via InteractiveScene
  - Observations, rewards, dones: GPU-batched, zero for-loop
- Standalone Isaac Lab env (`training/isaaclab_env.py`) — runs without Isaac Sim
- WarpAUV pretrained checkpoint
- CLI: `oceanscale demo bluerov2-hover`, `oceanscale demo underwater-mvp`, `oceanscale train bluerov2-hover`
- 1066 tests passing on RTX 5090

### Benchmarks
- `oceanscale/benchmarks/suite.py` — GPU kernel throughput benchmarking framework

---

## What does NOT ship in v0.0.2

Per ADR-007 (`DECISIONS.md`): no Isaac Sim / Isaac Lab hard dependency (both are optional). Per POSITIONING.md §4 promotion thresholds: no world-model layer claim (Tier 2) and no foundation-model claim (Tier 3) in public surfaces.

Future tier promotions are gated on:
- Tier 2 → homepage: shipped learned-dynamics demo with public benchmark vs classical baseline (`POSITIONING.md` §9.2)
- Tier 3 → headline: data + parameter-scale + distribution-leverage evidence (`POSITIONING.md` §4)

---

## Cross-references

- `POSITIONING.md` — brand law (anchor, scope, lexicon)
- `DESIGN.md` — visual brand law
- `STACK.md` — tech stack decisions and rationale
- `pyproject.toml` — authoritative dependency manifest
- `DECISIONS.md` — ADRs (especially ADR-007 on Isaac Sim removal)
- `ARCHITECTURE_PROPOSAL_2026-05-15.md` — original aspirational architecture (archived)

---

## Changelog

- **2026-05-23 v1** — Initial current-state architecture. Replaces the 2026-05-15 architecture proposal which was archived to ARCHITECTURE_PROPOSAL_2026-05-15.md.
- **2026-05-25 v2** — Massive integration session: 10 robots, 7 environments, 9 DVL configs, ImagingSonar, UW rendering, T200 thruster, MuJoCo drag+lift, distributed drag, partial submersion, controllers, domain randomization. Sources: MarineGym, OceanSim, isaac-auv-env, UUV Simulator, DAVE, SMARC, fishsim, bluerov2_gz. Tests: 730→922.
- **2026-05-26 v3** — OceanSim unified orchestrator, Isaac Lab 3 DirectRLEnv native, WCSPH fluid solver, underwater camera sensor, multibeam echo sounder, task environments (docking/station-keeping/waypoint), fluid fidelity ladder (5 levels), benchmarks suite. Tests: 922→1066.
