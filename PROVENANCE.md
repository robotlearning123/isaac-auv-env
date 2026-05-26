# OceanScale Provenance

Source traceability for all external references used in OceanScale.

**Outbound license:** Apache-2.0
**Last updated:** 2026-05-25

---

## Provenance Table

| Source | Type | URL / DOI / Reference | Date Retrieved | License | Purpose | Cited In |
|--------|------|-----------------------|----------------|---------|---------|----------|
| von Benzon et al. 2022 — BlueROV2 6-DOF platform | Paper + Simulink | DOI [10.3390/jmse10121898](https://doi.org/10.3390/jmse10121898) | 2026-05-15 | CC-BY-4.0 | Gold validation reference for Tier-1 hydrodynamics; tank-validated complete Fossen model | REFERENCES.md §6, ARCHITECTURE.md §5, benchmarks/RESULTS.md |
| Fossen 2021 — Handbook of Marine Craft Hydrodynamics and Motion Control (2nd ed.) | Textbook | ISBN 978-1-119-57526-4, Wiley | 2026-05-15 | n/a (citation) | Core 6-DOF hydrodynamic equations: Coriolis (§3.3), added-mass (§6.2), damping (§7.1), restoring (§8.2), thruster allocation (§10.4) | REFERENCES.md §4, ARCHITECTURE.md §5 |
| OceanSim (Song et al. 2025) | Code + Paper | https://github.com/umfieldrobotics/OceanSim · arXiv [2503.01074](https://arxiv.org/abs/2503.01074) | 2026-05-15 | Apache-2.0 | **Integrated 2026-05-25:** UW_render Warp kernel → `oceanscale/rendering/underwater.py`; ImagingSonar Warp kernels → `oceanscale/sensors/imaging_sonar.py`. Original: `isaacsim/oceansim/utils/UWrenderer_utils.py`, `sensors/ImagingSonarSensor.py` | REFERENCES.md §2 (REF-OCEANSIM), ARCHITECTURE.md §6 |
| MarineGym (Chu et al. 2025) | Code + Paper | https://github.com/Marine-RL/MarineGym · arXiv [2503.09203](https://arxiv.org/abs/2503.09203) | 2026-05-15 | MIT | **Integrated 2026-05-25:** BlueROV USD mesh + hydro YAML → `oceanscale/assets/bluerov_marinegym/`; T200 thruster model → `oceanscale/propulsion/t200.py`; vehicle config → `BlueROV2MarineGym` class. Original: `marinegym/robots/assets/usd/BlueROV/`, `marinegym/actuators/t200.py` | REFERENCES.md §2 (REF-MARINEGYM), ARCHITECTURE.md §5 |
| HoloOcean 2.0 (Potokar et al. 2025) | Code + Paper | https://github.com/byu-holoocean · arXiv [2510.06160](https://arxiv.org/abs/2510.06160) | 2026-05-15 | MIT (verify) | RTX sonar validation methodology; REMUS-validated <2% trajectory benchmark | REFERENCES.md §4, §6, ARCHITECTURE.md §6.1 |
| Newton 1.x physics engine | Code | https://github.com/newton-physics/newton | 2026-05-12 | Apache-2.0 | GPU-native physics core; `ModelBuilder`, `replicate()`, solvers; pinned `>=1.2.0,<1.3` | REFERENCES.md §2 (REF-NEWTON), ARCHITECTURE.md §3 |
| Warp (NVIDIA) | Code | https://github.com/NVIDIA/warp | 2026-05-12 | Apache-2.0 | GPU kernel JIT for all Tier-1 hydrodynamics and sensor kernels; `@wp.kernel`, `wp.launch`, `wp.Tape` | REFERENCES.md §2 (REF-WARP), ARCHITECTURE.md §3 |
| MuJoCo Warp (via Newton) | Code | Bundled in newton-physics/newton at `_src/solvers/mujoco/` | 2026-05-12 | Apache-2.0 | Primary physics solver option; GPU-parallel fluid forces + differentiable autograd | ARCHITECTURE.md §3, §5.3 |
| PyBullet | Code + Benchmark | https://github.com/bulletphysics/bullet3 | 2026-05-22 | zlib (compatible) | CPU baseline for apples-to-apples throughput benchmark; 10.5× slower at n_envs=64 | benchmarks/RESULTS.md, REFERENCES.md §5 |
| BELLHOP acoustic ray-tracing | Software | (unverified — no canonical GitHub URL confirmed) | (unverified) | (unverified) | Acoustic ray-tracing for underwater comms simulation; referenced as future v0.4 integration | ARCHITECTURE.md §6.4, REFERENCES.md (via REF-ISAACAUV) |
| BlueROV2 Heavy (BlueRobotics) | Hardware datasheet | https://bluerobotics.com/store/rov/bluerov2/ | 2026-05-15 | n/a (factual) | Mass, dimensions, thrust-per-motor specs; geometry sanity check for USD/URDF models | REFERENCES.md §3, §6 |
| Isaac AUV Env (Cai et al. 2024 — Learning to Swim) | Code + Paper | https://github.com/warplab/isaac-auv-env · arXiv [2410.00120](https://arxiv.org/abs/2410.00120) | 2026-05-15 | BSD-3-Clause (inherited Isaac Lab) | **Integrated 2026-05-25:** MuJoCo drag model → `oceanscale/hydro/mujoco_drag.py`; domain rand → `oceanscale/training/domain_rand.py`; WarpAUV USD → `oceanscale/assets/warpauv/` | REFERENCES.md §2 (REF-ISAACAUV), ARCHITECTURE.md §8 |
| bluerov2_gz (McQueen) | Code | https://github.com/clydemcqueen/bluerov2_gz (commit 661264b) | 2026-05-25 | MIT (in package.xml) | **Integrated 2026-05-25:** BlueROV2 Heavy high-fidelity Collada mesh → `oceanscale/assets/bluerov2_heavy_gz/`; sand heightmap → `oceanscale/assets/sand_heightmap/` | REFERENCES.md §3 (REF-BLUEROV2GZ) |
| UUV Simulator (Manhães et al.) | Code | https://github.com/uuvsimulator/uuv_simulator | 2026-05-25 | Apache-2.0 | **Integrated 2026-05-25:** RexROV mesh + hydro + TAM → `oceanscale/assets/rexrov/`; environments (shipwreck, Munkholmen, BOP) → `oceanscale/assets/environments/`; PID/sliding mode controllers → `oceanscale/controllers/` | REFERENCES.md §2 |
| DAVE (Field Robotics Lab) | Code | https://github.com/Field-Robotics-Lab/dave | 2026-05-25 | Apache-2.0 | **Integrated 2026-05-25:** Slocum/Wave/WHOI gliders + eROV → `oceanscale/assets/gliders/`, `oceanscale/assets/erov/`; 9 DVL configs → `oceanscale/sensors/dvl_configs.py`; Santorini terrain → `oceanscale/assets/environments/` | REFERENCES.md §2 |
| SMARC (KTH) | Code | https://github.com/smarc-project/smarc_stonefish_worlds | 2026-05-25 | BSD-3-Clause | **Integrated 2026-05-25:** Pipeline + shipwreck + rock environments → `oceanscale/assets/environments/` | REFERENCES.md §2 |
| fishsim (SRL-ETH Zürich) | Code + Paper | https://github.com/srl-ethz/fishsim · arXiv [2602.23283](https://arxiv.org/abs/2602.23283) | 2026-05-25 | MIT | **Integrated 2026-05-25:** Kutta lift + Magnus lift (validated coefficients) → `oceanscale/hydro/mujoco_drag.py`; TendonFish vehicle → `oceanscale/vehicles/fish.py`; CMA-ES sysid → `oceanscale/hydro/sysid.py` | REFERENCES.md §2 |
| Akkaynak & Treibitz 2018 — Revised underwater image formation | Paper | CVPR 2018 | 2026-05-15 | n/a (citation) | Per-channel underwater image formation model; replaces Jaffe-McGlamery per DR decision | REFERENCES.md §4, §10 decision log |
| Chaffre et al. IJRR 2025 — Adaptive DR for BlueROV | Paper | IJRR vol. 44 no. 3, pp. 407–430 | 2026-05-15 | n/a (citation) | SAC+BIER adaptive domain randomization methodology; tank-test validation | REFERENCES.md §4 |
| Hong et al. 2026 — Marine Robot DRL Survey | Paper | arXiv [2506.21063](https://arxiv.org/abs/2506.21063) | 2026-05-15 | n/a (citation) | Sim platform comparison taxonomy | REFERENCES.md §4 |

---

## Notes

- **"Date Retrieved"** is when the reference was first admitted into REFERENCES.md or first used in code, whichever is earlier.
- **(unverified)** marks fields not confirmed by direct inspection of the source. These must be resolved before the related implementation task begins.
- **BELLHOP**: referenced in ARCHITECTURE.md as the acoustic comms engine for v0.4. No canonical open-source URL has been verified in this repo yet. The Portenal/CMRE implementation is the de facto standard but its license and URL have not been audited per REFERENCES.md §0.
- **Fossen 2021**: textbook equations are not copyrightable; numerical parameter values from worked examples are factual. Cited equations are standard 6-DOF marine craft hydrodynamics.
- **BlueROV2 Heavy specs**: BlueRobotics publishes mass, dimensions, and thrust curves as product specifications. These are factual data, not subject to copyright.
- Most entries follow the admission criteria in REFERENCES.md §0 (license verified, compatibility checked, activity tier assigned, use case defined, risk noted). Entries with `(unverified)` fields or conditional license notes (e.g. `MIT (verify)`) MUST be resolved before code or dependency use.

---

## What This File Is NOT

- Not a replacement for REFERENCES.md — that file is the single source of truth for license compliance and vendor/port decisions.
- Not a survey — see SURVEY.md.
- Not a list of all papers ever cited — only external references that materially influence code, assets, or benchmarks.
