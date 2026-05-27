# Session Handoff — 2026-05-26 (session 6)

## Current addendum — Isaac 6 docs and verification

The current new-user path is documented in `docs/README.md`, `docs/getting-started.md`, `docs/verification.md`, and `docs/isaac6-isaaclab3-install.md`. OceanScale is using Isaac Sim 6 / Isaac Lab 3 as the main validation lane, with a separate Isaac validation venv so Isaac package pins do not overwrite the OceanScale Newton/Warp core environment. Public PyPI publishing is not verified; use the source install path.

Latest local Isaac baseline verifier evidence: `/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-26-125621/summary.json`.

## What shipped since last handoff

Four PRs merged (#55–#59):

| PR | Title | Key additions |
|----|-------|---------------|
| #55 | MarineGym + OceanSim integration | OceanSim orchestrator, Isaac Lab DirectRLEnv, unified demo |
| #56 | Isaac Lab 3 official pattern | Wrench composer, JIT reward, scene setup, DirectRLEnv 3 native |
| #58 | RL envs → OceanSim + camera | Task envs (docking/waypoint/station-keeping), underwater camera sensor |
| #59 | WCSPH fluid solver | Warp HashGrid + Tait EOS + cubic spline, fluid fidelity ladder |

Tests: 922 → 1066 (+144), 0 regressions.

## Current codebase state

### Core architecture

- **OceanSim** (`sim.py`) — single `step()` orchestrator: Newton physics + Fossen hydro + ocean state + sensors
- **Fluid ladder** (`fluid/`) — 5 levels: None → Grid → SPH (WCSPH) → MPM → Volume (NanoVDB)
- **Isaac Lab** (`training/isaaclab_task.py`) — DirectRLEnv 3 native with JIT reward, wrench composer
- Standalone env (`training/isaaclab_env.py`) — runs without Isaac Sim

### All modules (oceanscale/)

| Module | Files | Engine |
|--------|-------|--------|
| sim.py | OceanSim, OceanSimConfig, OceanConfig | Warp + Newton |
| fluid/ | sph (WCSPH), grid, mpm, volume_solver, wave, wave_fft, surface_extractor, mesh_boundary, adaptive_grid, differentiable, mpm_coupling | Warp |
| hydro/ | tier1 (Fossen 6-DOF), tier1_kernels, mujoco_drag, distributed_drag, submersion, sysid | Warp |
| sensors/ | ray_dvl, ray_sonar, imaging_sonar, underwater_camera, multibeam, dvl, dvl_configs, imu, pressure, magnetometer, usbl, acoustic_modem, sidescan_sonar | Warp |
| envs/ | docking_env, waypoint_env, station_keeping_env | Newton |
| training/ | isaaclab_task, isaaclab_env, domain_rand, train_isaaclab, skrl_trainer | PyTorch |
| vehicles/ | bluerov2, commercial, fish, fleet | dataclass |
| controllers/ | pid, lee_position | numpy |
| rendering/ | underwater (Jerlov), video | Warp |
| propulsion/ | t200, propeller, buoyancy_engine, flapping | Warp |
| benchmarks/ | suite | Warp |

### Uncommitted changes (working tree)

```
M  AGENTS.md
M  artifacts/isaacsim/isaacsim6_official_setup_log_2026-05-25.md
M  artifacts/isaacsim/official_suite_coverage.json
M  artifacts/isaacsim/official_suite_coverage.md
M  artifacts/isaacsim/run_isaacsim_standalone_smokes.py
M  oceanscale/fluid/sph.py
M  oceanscale/hydro/mujoco_drag.py
M  oceanscale/sensors/multibeam.py
M  oceanscale/sim.py
M  oceanscale/training/isaaclab_task.py
?? artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md
?? artifacts/isaacsim/oceanscale_latest_isaac_baseline_2026-05-25.md
?? artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
?? oceanscale/benchmarks/
?? tests/test_benchmarks.py
?? tests/test_ocean_sim_domain_rand.py
?? tmp/
```

### Isaac Sim status

- **6.0**: Installed at `/mnt/storage/isaacsim-6.0-official/`. Kit runtime 4.2GB downloaded. Standheadless smoke tests partially passing. Camera/SyntheticData segfaults on RTX 5090 + Driver 580 remain.
- **5.1**: Swapchain capture works (1440×900). MarineGym/OceanSim partially run but blocked by API breaks.
- **4.5 pip**: Stub, no Kit runtime.

## Key decisions this session

1. OceanSim = single orchestrator replacing three parallel env paths
2. Isaac Lab 3 official DirectRLEnv pattern adopted (wrench composer + JIT reward)
3. Fluid fidelity ladder: 5 levels, WCSPH at level 2
4. Task envs (docking/waypoint/station-keeping) migrated to OceanSim backend
5. Underwater camera: pinhole ray casting + Beer-Lambert (no Isaac Sim dep)

## Next priorities

1. **Commit uncommitted changes** — several modified files (sph.py, sim.py, isaaclab_task.py, etc.) need branching + atomic commits + PR
2. **Isaac Sim 6.0 rendering** — resolve Camera segfault on RTX 5090; if works, build RT2 rendering pipeline
3. **Hover task port to OceanSim** — migrate PPO hover reward/obs to use OceanSim orchestrator
4. **Benchmarks** — commit and run the new benchmark suite (`oceanscale/benchmarks/`)
5. **FARMS CPG** — amphibious locomotion controllers (v0.2)
6. **Website update** — update demo section to reflect OceanSim + fluid ladder
