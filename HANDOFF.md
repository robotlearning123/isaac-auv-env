# Session Handoff — 2026-05-24 (session 4)

## What shipped

7 PRs merged (#47-53), 693 tests, ~14,000 LOC. Zero uncommitted changes.

| PR | Content | Tests |
|----|---------|-------|
| 47 | Fluid upgrades (HashGrid SPH, wave, fidelity API) + sim/website/CI | 391 |
| 48 | wp.Volume solver, CUDA Graph, Newton tether | 423 |
| 49 | All NVIDIA features: Mesh FSI, ray sensors, diff NS, FFT, MPM, MarchingCubes, cloth, adaptive grid | 505 |
| 50 | End-to-end integration pipeline (fluid→sensor→training) | 516 |
| 51 | Isaac Lab DirectRLEnv training layer | 534 |
| 52 | OceanWorld + UnifiedDemo | 592 |
| 53 | Ocean physics: water column, propulsion, currents, acoustics | 693 |

## 12 NVIDIA features integrated

wp.Volume, CUDA Graph, Newton rod, wp.Mesh FSI, mesh_query_ray sensors, differentiable NS (wp.Tape), Tile FFT, MPM coupling, MarchingCubes, Newton cloth, AdaptiveNanogrid, Isaac Lab DirectRLEnv.

## 4 self-developed ocean physics modules

WaterColumn (UNESCO EOS-80), PropellerThruster/FlappingFin/BuoyancyEngine, OceanCurrentField (M2 tides + Ekman), AcousticPropagation (Snell + Thorp + Wenz).

## Performance baselines (RTX 5090)

FFT Wave: 4,983 steps/s · CUDA Graph: 5.5x · RayDVL: 8,765 Hz · Pipeline: 194 steps/s · Isaac Lab 16 envs: 80 env-steps/s · GPU: 3%.

## Key decisions

1. Official features first — max NVIDIA, then custom ocean physics
2. No official ocean sim = our market window
3. Fossen = baseline; real fidelity = Warp CFD + Newton
4. OpenUSD = scene format (Newton `builder.add_usd()` ready)
5. Ocean core, NVIDIA decides robot scope
6. 12 ocean robot types mapped to Newton solvers

## Known issues

1. `cosh` overflow wave.py T<5s deep water → deep-water approx k*d>20
2. VBD cloth ≥15×15 NaN → increase iterations or reduce dt
3. DVL beam direction on complex terrain → check orientation
4. Pipeline ROV not advancing → force coupling chain
5. UnifiedDemo ROV falls → gravity/buoyancy tuning

## Next priorities

1. Fix 5 known issues
2. Wire ocean physics into UnifiedDemo (WaterColumn + currents + acoustics + propellers)
3. USD scene import (`builder.add_usd()` for BlueROV2)
4. AUV vehicle model (torpedo hydro + rudder joints)
5. ROV + manipulator arm (Featherstone + URDF)
6. Build the virtual ocean — all modules as one realistic system
7. Soft robot PoC (XPBD + FSI for bio-inspired)
