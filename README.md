# OceanScale

The ocean simulator for underwater robotics.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem: Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch. It provides GPU-native ocean physics, underwater environments, and sensor models for AUVs and ROVs.

## Why OceanScale?

- **NVIDIA-native ocean physics** — Newton + Warp CUDA kernels, not CPU loops.
- **Isaac 6 ecosystem baseline** — Isaac Sim 6 and Isaac Lab 3 are the main simulation and training-integration targets.
- **Multi-fidelity fluid** — 5-level ladder: Fossen analytical → Grid Eulerian → SPH (WCSPH) → MPM → Volume (NanoVDB).
- **14+ underwater sensors** — DVL (ray-based + 9 real-world configs), imaging sonar (Oculus M370), multibeam (Kongsberg EM-2040), sidescan sonar, underwater camera (Beer-Lambert), IMU, pressure, magnetometer, USBL, acoustic modem.
- **10 vehicle models** — BlueROV2 Heavy, BlueROV MarineGym, WarpAUV, RexROV, Slocum/Wave/WHOI gliders, eROV, TendonFish biomimetic.
- **RL-ready** — Gymnasium-compatible environments with `BatchedVecEnv`; PPO training via skrl, Isaac Lab 3 DirectRLEnv integration.
- **Validated** — Tier-1 hydrodynamics matched against von Benzon 2022 BlueROV2 reference model (0.013% position error, 0.0025 deg attitude RMS).

## Quickstart

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv python pin 3.12
uv sync --extra dev
uv run oceanscale --help
```

Run the BlueROV2 hover demo:

```bash
# CPU
uv run oceanscale demo bluerov2-hover --device cpu

# CUDA with MP4 output
uv run oceanscale demo bluerov2-hover --device cuda --render-mp4 demo.mp4
```

Train a policy:

```bash
uv run oceanscale train bluerov2-hover --total 1000000 --n_envs 4 --device cuda
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

## Architecture

`OceanSim` (`oceanscale/sim.py`) is the unified orchestrator. A single `step()` call composes:

1. Newton rigid-body physics step
2. Hydrodynamics (Fossen 6-DOF, MuJoCo drag, distributed drag, partial submersion)
3. Ocean state (water column properties, currents, waves)
4. Sensor updates (DVL, sonar, camera, IMU, pressure)
5. Domain randomization (if enabled)

Configuration via `OceanSimConfig` dataclass: vehicle, ocean params, sensor mounts, n_envs, dt, device, seabed mesh, domain randomization ranges.

### Hydrodynamics

| Model | Description |
|-------|-------------|
| Tier-1 Fossen 6-DOF | Added mass, Coriolis/centripetal, linear + quadratic damping, restoring forces (Warp kernels) |
| MuJoCo-style drag | Geometry-inferred quadratic drag + Kutta lift + Magnus lift (fishsim-validated coefficients) |
| Distributed drag | Per-link independent drag for articulated bodies (Warp kernel) |
| Partial submersion | Linear buoyancy transition at free surface for amphibious sim (Warp kernel) |
| CMA-ES sysid | Reference framework for fluid parameter identification |

### Fluid Fidelity Ladder

| Level | Solver | Use Case |
|-------|--------|----------|
| 0 | None (Fossen analytical) | Baseline RL training, fast iteration |
| 1 | Grid Eulerian (Chorin projection) | Moderate-fidelity flow interaction |
| 2 | SPH (WCSPH, Monaghan 1992) | Free-surface, splashing, sloshing |
| 3 | MPM (Material Point Method) | Fluid-structure interaction |
| 4 | Volume (NanoVDB sparse grid) | High-fidelity volumetric |

### Vehicles

10 underwater robot models with assets (USD, SDF, DAE, STL):

- **BlueROV2 Heavy** — 8 thrusters, Fossen + von Benzon 2022 parameters
- **BlueROV MarineGym** — 6 thrusters, MarineGym hydro YAML
- **WarpAUV** — 6 thrusters, MIT isaac-auv-env
- **RexROV** — 8 thrusters, 1863 kg, UUV Simulator
- **Slocum / Wave / WHOI Gliders** — DAVE assets
- **eROV** — DAVE assets
- **TendonFish** — Biomimetic, experimentally validated fluid coefficients

### Sensors

14+ sensor models with real-world hardware defaults:

- **DVL** — Ray-based (Warp BVH) + 9 real-world configs (Nortek, Teledyne, Sonardyne, LinkQuest, Rowe, RDI)
- **Imaging sonar** — Oculus M370, polar-binned 2D sonar image (5 Warp kernels)
- **Multibeam** — Kongsberg EM-2040, fan-shaped beam array
- **Underwater camera** — GPU pinhole ray casting + Beer-Lambert attenuation, 10 Jerlov water types
- **Sidescan sonar**, **IMU**, **pressure**, **magnetometer**, **USBL**, **acoustic modem**

### Task Environments

- **DockingApproachEnv** — Precision docking task
- **CurrentStationKeepingEnv** — Hold position in ocean currents
- **WaypointFollowingEnv** — Follow waypoints through current

7 underwater scene assets: shipwreck, Munkholmen, BOP panel, pipeline, rock, Santorini terrain, seabed.

## Benchmark

Launch-standardized OceanScale throughput on RTX 5090. Full methodology in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Envs | Env-steps/s |
| ---: | ---: |
| 1 | 1,326 |
| 64 | 85,824 |
| 256 | 303,206 |
| 1024 | 1,272,106 |
| 4096 | 4,588,922 |

**PyBullet comparison** (separate benchmark, 2026-05-22): At n=64, OceanScale reached 17,427 env-steps/s, 10.49x over single-env PyBullet on the matched BlueROV2 hover task. Crossover between n=8 and n=16. See [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) for documented physics differences.

## NVIDIA Isaac 6 Ecosystem

OceanScale is fully based on the NVIDIA Isaac 6 ecosystem for simulation and training integration. The canonical local setup keeps the Isaac validation environment side-by-side with the OceanScale core development environment so Isaac package pins do not overwrite the Newton/Warp stack used by OceanScale core.

- **Isaac Lab 3 DirectRLEnv** — Native integration via `training/isaaclab_task.py` with wrench composer and JIT reward
- **Standalone fallback** — `training/isaaclab_env.py` runs without Isaac Sim dependency

Use [docs/isaac6-isaaclab3-install.md](docs/isaac6-isaaclab3-install.md) for the full install procedure.

## Verification

Use [docs/verification.md](docs/verification.md) as the customer/new-user runbook for proving the source install, CLI, first demo, focused tests, and Isaac Sim 6 / Isaac Lab 3 validation lane.

## Status (v0.1.0-alpha)

**Works:**
- 10 vehicle models with full asset pipelines (USD, SDF, DAE)
- 5 hydrodynamics models (Fossen, MuJoCo drag, distributed drag, submersion, sysid)
- 5-level fluid fidelity ladder (Fossen → Grid → SPH → MPM → Volume)
- 14+ underwater sensor models with real hardware defaults
- 7 underwater scenes with object assets
- 3 task environments (docking, station-keeping, waypoint)
- Vectorized environments (BatchedVecEnv + VecNormalize)
- PPO training via skrl + Isaac Lab 3 DirectRLEnv
- Domain randomization (volume, COM-COB offset, drag, thruster noise)
- Headless MP4 export (simple + cinematic modes)
- Source-installable CLI + Colab notebook
- Isaac Sim 6 / Isaac Lab 3 validation lane passing locally
- Physics-based underwater rendering (10 Jerlov water types, Beer-Lambert)

**Known limitations:**
- Bundled hover demo is a CLI smoke and checkpoint-loading path, not a policy-quality release gate
- Contact, docking, station-keeping, waypoint, and multi-vehicle code paths exist but are not yet long-horizon release gates
- Cross-coupling damping disabled in v0.1 (coefficients not independently identified)
- Omniverse RTX rendering is not a public OceanScale release surface yet
- No sim-to-real transfer validation
- PyPI publishing not yet verified end-to-end

## Repo Layout

```
oceanscale/              Python GPU simulation package
  sim.py                 OceanSim unified orchestrator
  cli.py                 CLI entry point
  rov_env.py             Gymnasium ROV environment
  vec_env.py             BatchedVecEnv (Newton tensors)
  hydro/                 Hydrodynamics kernels (Fossen, MuJoCo drag, distributed, submersion, sysid)
  fluid/                 Multi-fidelity fluid (Grid, SPH, MPM, Volume, waves, adaptive grid)
  sensors/               14+ underwater sensor models + 9 DVL configs
  vehicles/              10 vehicle models + fleet registry
  controllers/           PID, Lee geometric, YAML configs
  propulsion/            T200 thruster, buoyancy engine, flapping
  rendering/             Underwater rendering (Jerlov types) + MP4 video export
  training/              skrl PPO, Isaac Lab 3 DirectRLEnv, domain randomization
  envs/                  Task environments (docking, station-keeping, waypoint)
  environments/          Environment presets
  worlds/                Ocean world
  validation/            Von Benzon 2022 reference model
  assets/                Vehicle meshes, scene assets, sensor configs
website/                 Astro 6 + Tailwind v4 landing page (Cloudflare Pages)
benchmarks/              Performance benchmarks (40+ scripts)
notebooks/               Colab notebooks
tests/                   Test suite (80+ test files)
docs/                    Technical documentation
```

## Website

Landing page at [oceanscale-web.pages.dev](https://oceanscale-web.pages.dev). Built with Astro 6 + Tailwind v4, bilingual (EN/ZH), deployed to Cloudflare Pages via tag-driven releases.

```bash
cd website && pnpm install && pnpm dev    # localhost:4321
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{oceanscale2026,
  title = {OceanScale: GPU-Native Underwater Robotics Simulation},
  author = {OceanScale Team},
  year = {2026},
  url = {https://github.com/robotlearning123/oceanscale}
}
```
