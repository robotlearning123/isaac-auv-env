# Session Handoff — 2026-05-25 (session 5)

## What shipped

PR #54 merged (cleanup/organize WIP). Massive multi-project integration: 8 open-source underwater sim projects merged into OceanScale. Tests: 730 → 922 (+192), 0 regressions.

## Integration summary

| Source | License | What was merged |
|--------|---------|----------------|
| MarineGym | MIT | BlueROV USD + hydro YAML + T200 thruster (Warp kernel) |
| OceanSim | Apache-2.0 | ImagingSonar (5 Warp kernels) + UW rendering (10 Jerlov water types) |
| isaac-auv-env | BSD-3 | MuJoCo drag + domain randomization + WarpAUV USD |
| bluerov2_gz | MIT | BlueROV2 Heavy high-fidelity Collada mesh (24MB) + seabed terrain |
| UUV Simulator | Apache-2.0 | RexROV (mesh + hydro + TAM) + 3 environments (shipwreck, Munkholmen, BOP) + PID/NL-PID/sliding mode controllers |
| DAVE | Apache-2.0 | 4 vehicles (Slocum, Wave Glider, WHOI, eROV) + 9 real DVL configs + Santorini terrain + objects |
| SMARC | BSD-3 | Pipeline + shipwreck + rock environments |
| fishsim | MIT | Kutta lift + Magnus lift (Warp kernels, experimentally validated) + TendonFish + CMA-ES sysid |

## New modules (13)

| Module | Path | Engine |
|--------|------|--------|
| T200 thruster | `propulsion/t200.py` | Warp |
| ImagingSonar | `sensors/imaging_sonar.py` | Warp |
| DVL configs (9 models) | `sensors/dvl_configs.py` | dataclass |
| UW rendering | `rendering/underwater.py` | Warp |
| MuJoCo drag + lift | `hydro/mujoco_drag.py` | Warp |
| Distributed drag | `hydro/distributed_drag.py` | Warp |
| Partial submersion | `hydro/submersion.py` | Warp |
| System identification | `hydro/sysid.py` | dataclass |
| PID controller | `controllers/pid.py` | numpy |
| Lee geometric ctrl | `controllers/lee_position.py` | numpy |
| Domain randomization | `training/domain_rand.py` | numpy |
| Fish vehicle | `vehicles/fish.py` | dataclass |
| Vehicle fleet registry | `vehicles/fleet.py` | dataclass |

## 10 underwater robots supported

BlueROV2 Heavy, BlueROV MarineGym, WarpAUV, RexROV, Slocum Glider, Wave Glider, WHOI Hybrid, eROV, TendonFish. Assets in `oceanscale/assets/`.

## 7 environments

Shipwreck, Munkholmen, BOP panel, Santorini terrain, pipeline, rock, sand heightmap. In `oceanscale/assets/environments/`.

## Isaac Sim status

- **5.1**: Camera/SyntheticData segfaults on RTX 5090 + Driver 580. Swapchain capture works (1440×900). MarineGym/OceanSim partially run but blocked by API breaks.
- **6.0**: Installing at `/mnt/storage/isaacsim-6.0-official/` (downloading 4.2GB Kit runtime via `isaacsim[all,extscache]`). Not finished.
- **4.5 pip**: Stub, no Kit runtime.

## Demo videos produced

- Warp 3D + OceanSim physics water FX: 240 frames, 8s, 1920×1080
- Isaac Sim RTX + MarineGym BlueROV mesh: 120 frames, 4s, 1440×900
- Isaac Sim RTX raw (no postfx): 120 frames, comparison

## Key decisions this session

1. OceanScale = THE unified underwater robot simulator (absorb all open-source projects)
2. All merged code has provenance headers (URL + license + paper + modifications)
3. Isaac Sim 6.0 is the target rendering platform (Newton schema, RT2)
4. Development-phase: license compliance deferred to pre-release audit
5. FARMS swim-to-walk physics implemented independently (partial submersion + distributed drag)

## Next priorities

1. **Isaac Sim 6.0**: Complete installation → test headless Camera → if works, build rendering pipeline
2. **Commit + PR**: All integration work is uncommitted on main — create branch, atomic commits, PR
3. **Hover task port**: Port MarineGym Hover reward/obs to OceanScale Newton env (P0c deferred)
4. **FARMS CPG**: Integrate amphibious locomotion controllers (v0.2)
5. **Demo video**: Re-render with Isaac Sim 6.0 RT2 once available
6. **MSS validation**: Extract Fossen reference trajectories from cybergalactic/MSS for hydro kernel numerical verification
