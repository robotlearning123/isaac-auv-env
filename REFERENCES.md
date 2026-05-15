# References — Verified, Versioned, Reusable

**Date:** 2026-05-15  
**Maintainer:** This file is the single source of truth for every external project, paper, asset, or dataset OceanScale depends on. Every entry must have a verified license, pinned version, intended use, and risk note. New entries pass the §0 admission criteria.

**Outbound license:** Apache-2.0  
**Compatible inbound:** MIT, BSD-2/3-Clause, Apache-2.0, CC-BY-4.0, ISC, Zlib  
**Incompatible:** GPL-2/3/AGPL/SSPL (taints derivative works), unlicensed (no permission to redistribute)

---

## Table of contents

- [§0 Admission criteria](#0-admission-criteria)
- [§1 Master matrix](#1-master-matrix)
- [§2 Code references](#2-code-references) — repositories we vendor / port / study
- [§3 Asset references](#3-asset-references) — vehicle USD/MJCF/URDF sources
- [§4 Paper references](#4-paper-references) — algorithm and validation sources
- [§5 Anti-references](#5-anti-references) — projects we explicitly do NOT use
- [§6 Validation data](#6-validation-data) — published BlueROV2 tank tests
- [§7 License compliance plan](#7-license-compliance-plan)
- [§8 Quick-fetch commands](#8-quick-fetch-commands)
- [§9 BibTeX bibliography](#9-bibtex-bibliography)
- [§10 Decision log](#10-decision-log)
- [§11 Pending verifications](#11-pending-verifications)

---

## §0 Admission criteria

A new external reference is admitted only if it passes **all** of the following:

1. **License verified** by direct LICENSE file inspection (not just GitHub metadata), recorded with SPDX identifier and quote of first 5 lines.
2. **Compatible** with Apache-2.0 outbound per §7. GPL/AGPL → Anti-references (§5).
3. **Activity tier** assessed (S = actively maintained < 6 mo, A = < 18 mo, B = stale, C = abandoned).
4. **Use case** is one of: Vendor (copy in with NOTICE) / Port (rewrite, cite) / Study (read only) / Cite (paper only).
5. **Pinned anchor** chosen: commit SHA for vendor/port, paper version for cite.
6. **Risk note**: at least one sentence on what could go wrong.

Anything failing 1-3 → §5 Anti-references.

---

## §1 Master matrix

| Ref ID | Project | License | Use | Activity | Subgoal | Risk |
|---|---|---|---|---|---|---|
| `REF-NEWTON` | newton-physics/newton | Apache-2.0 ✓ | Vendor (Apache) + cite | S (active daily) | G2-G5 | API still 1.x, breaking changes possible |
| `REF-WARP` | NVIDIA/warp | Apache-2.0 ✓ | Vendor (Apache) | S | G2-G5 | 1.x→2.x breaking changes flagged |
| `REF-MARINEGYM` | Marine-RL/MarineGym | MIT ✓ | Vendor + Port | A (Jan 2026 last commit) | G3, G4 | Single-lab, may stall |
| `REF-OCEANSIM` | umfieldrobotics/OceanSim | BSD-3-Clause ✓ | Vendor + Port | A (Sept 2025 last commit) | v0.2 sensors | Replicator-coupled; needs rewrite |
| `REF-ISAACAUV` | warplab/isaac-auv-env | BSD-3-Clause ✓ (inherited Isaac Lab) | Port + Study | A (Aug 2025 last commit) | G5 | Tied to Isaac Lab; design transfers, plumbing doesn't |
| `REF-BLUEROV2GZ` | clydemcqueen/bluerov2_gz | **NO LICENSE** ❌ | **Study only** (cannot vendor) | S (Dec 2025 last commit) | G4 | **License blocker — code unredistributable** |
| `REF-BLUEROV2GYM` | gokulp01/bluerov2_gym | **NO LICENSE** ❌ | **Study only** | A (Apr 2025) | G4 | License blocker |
| `REF-FOSSEN21` | Fossen 2021 Handbook | Textbook (citation only) | Cite (math equations) | n/a | G3, T1.3 | Equations not copyrightable; numerical examples are |
| `REF-MSS` | cybergalactic/MSS | MIT (verify) | Study (MATLAB reference impl) | A | G3 | Different language; numerical cross-check only |
| `REF-LEARNTOSWIM` | arXiv 2410.00120 | Paper | Cite + design | 2024 | G5 | Paper, not code (code = REF-ISAACAUV) |
| `REF-FASTAUVMJX` | arXiv 2512.13359 | Paper | Cite + design | 2025-12 | G5 | No public code |
| `REF-EASYUUV` | arXiv 2510.22126 | Paper | Cite (DR strategy) | 2026-02 | G5 | No clear public code |
| `REF-CHAFFRE25` | Chaffre IJRR 2025 | Paper + closed code | Cite | 2025 | G5 | DR ranges + hardware methodology |
| `REF-HOLOOCEAN` | byu-holoocean | MIT (verify) | Cite (validation method) | A | G3 | UE5-coupled; not vendorable as code |
| `REF-WU2018` | Wu 2018 Flinders **MS thesis** | Thesis (no peer review) | **Demoted to caveat-cite** | 2018 | G4 | Author never received the BlueROV2 Heavy hardware; values are extrapolated from non-Heavy specs |
| `REF-VONBENZON22` | von Benzon et al. 2022 JMSE 10(12):1898 | CC-BY-4.0 paper + Simulink sim | **GOLD validation reference** | 2022 | G4, R23 | Tank-validated complete simulator. Supersedes Wu 2018. |
| `REF-SIM2SWIM` | Tunçay et al. *Sim2Swim* arXiv 2512.08656 | Paper | Cite + benchmark target | 2025-12 | G5 | 3-min training claim for 6-DOF zero-shot — direct G5 competitor |
| `REF-LEARNTODOCK` | Singh et al. arXiv 2506.17823 | Paper | Cite + mirror obs/reward design | 2025-06 | G5 | Only published BlueROV2-Heavy Isaac Sim RL with DR ablations |
| `REF-DIFFSPH` | diffSPH arXiv 2507.21684 | OSS (license check) | v0.3 watchlist | 2025-07 | v0.3 manipulator wake | Differentiable SPH, PyTorch+GPU, weakly compressible + free surface |
| `REF-CENTRALENANTES` | CentraleNantesROV/bluerov2 GitHub | License check | Vendor rosbag schema | 2024+ | G4 validation logging | ROS2+Gazebo BlueROV2; exact `thruster_cmd → pose_gt` topic schema |
| `REF-OSLOMET` | OsloMet-OceanLab/BlueROV2 GitHub | License check | Study (Fossen+Wu 2018 Simulink alt) | 2024+ | G4 cross-check | Simulink 6-DOF Fossen model — alternative coefs |
| `REF-PRIVDREAMER` | PrivilegedDreamer ICRA 2025 | Paper | Cite | 2025 | G5 + R23 | Hidden-parameter MDP world model; sim-to-real on AUV/marine |

---

## §2 Code references

### REF-NEWTON — Physics engine ⭐⭐⭐⭐⭐

| Field | Value |
|---|---|
| **URL** | https://github.com/newton-physics/newton |
| **License** | Apache-2.0 (verified) |
| **SPDX** | `Apache-2.0` |
| **Activity** | **S** — last pushed 2026-05-15 18:35 UTC (same-day), 4,909 stars / 533 forks / 248 open issues |
| **Authors** | NVIDIA + Google DeepMind + Disney Research; Linux Foundation governance |
| **Version installed** | 1.2.0 (May 12 2026 PyPI release) |
| **Pinned anchor** | `newton==1.2.0` in pyproject.toml; upper-bound `<2.0` |

**What we use**:
- `newton.ModelBuilder` + `add_body` + `add_shape_*` (G3, G4)
- `newton.ModelBuilder.replicate(template, world_count)` for parallel envs (G2)
- `newton.solvers.SolverSemiImplicit` (v0.1 default), `SolverMuJoCo` (alternative)
- `newton.eval_fk` for state init from joint coords
- USD interop via `add_usd` and `usd_schemas`
- Local install path: `.venv/lib/python3.12/site-packages/newton/`

**Reference files** in the installed package (read for canonical patterns):
| File | What it teaches |
|---|---|
| `examples/robot/example_robot_cartpole.py:36-58` | N-world replicate, SolverMuJoCo step |
| `tests/test_multiworld_body_properties.py::_build_model_with_per_world_com` | Per-world parameter variation |
| `tests/test_control_force.py::test_floating_body` | Free body + eval_fk + step pattern |
| `_src/sim/builder.py:2116` | `replicate()` implementation |
| `_src/solvers/mujoco/solver_mujoco.py` | Where `separate_worlds = ... model.world_count > 1` lives |

**Risk**: Newton is 1.x; major version bumps in next 12 months likely break our pre-step hook patterns. Mitigation: keep `<2.0` upper bound; maintain a CI run against `main` branch nightly.

---

### REF-WARP — GPU kernel JIT ⭐⭐⭐⭐⭐

| Field | Value |
|---|---|
| **URL** | https://github.com/NVIDIA/warp |
| **License** | Apache-2.0 |
| **Version installed** | 1.13.0 (May 2026) |
| **Pinned anchor** | `warp-lang[torch-cu12]>=1.13.0,<2.0` |

**What we use**: `@wp.kernel`, `wp.array`, `wp.launch`, `wp.Tape` for autograd, `wp.spatial_vectorf`, `wp.mat33f`, `wp.quatf`. All Tier-1 hydrodynamics kernels and sensor kernels are Warp-native.

**Risk**: `warp.sim` already deprecated (moved into Newton). 2.x→ may continue API churn. Pin lower bound; track upstream issue tracker.

---

### REF-MARINEGYM — Fossen reference + BlueROV2 asset ⭐⭐⭐⭐⭐

| Field | Value |
|---|---|
| **URL** | https://github.com/Marine-RL/MarineGym |
| **License** | **MIT** ✓ verified via GitHub API + spdx |
| **Activity** | **A** — last pushed 2026-01-27 (3+ months ago), 163 stars / 23 forks / 9 open issues |
| **Authors** | Shuguang Chu et al., Zhejiang U + Heriot-Watt; IROS 2025 |
| **Paper** | [arXiv 2503.09203](https://arxiv.org/abs/2503.09203) |
| **Stack** | Isaac Sim 4.1 + PhysX + Python 3.10 + TorchRL + OmniDrones |
| **Pinned anchor** | **TODO** — fetch latest main commit SHA at T3.3 |

**Vendor candidates**:
| File | Destination | Why |
|---|---|---|
| `marinegym/robots/assets/usd/BlueROV/*.usd` | `assets/vehicles/bluerov2_heavy/` | Calibrated USD asset, MIT-vendorable |
| `marinegym/robots/drone/BlueROV.py` (constants) | `assets/vehicles/bluerov2.hydro.yaml` | Hydro coefficients |
| `marinegym/robots/drone/BlueROVHeavy.py` (constants) | `assets/vehicles/bluerov2_heavy.hydro.yaml` | Heavy-variant coefficients |

**Port candidates**:
| File | Destination | Why |
|---|---|---|
| `marinegym/robots/drone/underwaterVehicle.py` | `oceanscale/hydro/tier1_kernels.py` | PyTorch Fossen → Warp |

**Code structure verified by inspection** of `underwaterVehicle.py` lines 1-80:
- Class: `UnderwaterVehicle(RobotBase)`
- Methods: `apply_hydrodynamic_forces()`, `calculate_added_mass()`, `calculate_damping()`, `calculate_corilis()` *(sic — typo in source)*, `calculate_buoyancy()`
- Attributes: `self.added_mass_matrix`, `self.linear_damping_matrix`, `self.quadratic_damping_matrix`

**Mapping to our Warp kernels**:
| MarineGym method | Our kernel | Notes |
|---|---|---|
| `calculate_added_mass()` | `tier1_added_mass_kernel` | M_A · ν̇ (T2.2) |
| `calculate_damping()` | `tier1_damping_kernel` | -(D_lin + D_quad·|v|)·v (T2.3) |
| `calculate_corilis()` | `tier1_coriolis_kernel` | C_RB(ν) + C_A(ν) (T2.4) |
| `calculate_buoyancy()` | `tier1_restoring_kernel` | g(η) (T2.5) |
| `apply_hydrodynamic_forces()` | pre-step hook on `state.body_f` | Force injection (T2.1) |

**Risk**: MarineGym pins Isaac Sim 4.1; their PhysX `apply_forces_and_torques_at_pos` does not exist in Newton. The **math transfers cleanly**; the **force-injection plumbing** must be rewritten on `state.body_f` pre-step hook.

**Skip**:
- `marinegym/learning/ppo/` — TorchRL implementation; we use stable-baselines3 (v0.1) / skrl (v0.2)
- `marinegym/envs/isaac_env.py` — assumes Isaac Sim ArticulationView; design transfers, code doesn't
- `scripts/train.py` — Hydra entrypoint, ref-only

---

### REF-OCEANSIM — Sensor kernels (v0.2) ⭐⭐⭐⭐

| Field | Value |
|---|---|
| **URL** | https://github.com/umfieldrobotics/OceanSim |
| **License** | **BSD-3-Clause** ✓ verified via LICENSE file content ("Copyright (c) 2024-2025, The OceanSim Project Developers... SPDX-License-Identifier: BSD-3-Clause") |
| **Activity** | **A** — last pushed 2025-09-26 (8 mo), 446 stars / 61 forks / 7 open issues |
| **Authors** | Jingyu Song, Haoyu Ma et al., UMich Field Robotics; IROS 2025 |
| **Paper** | [arXiv 2503.01074](https://arxiv.org/abs/2503.01074) |
| **Pinned anchor** | TODO at v0.2 sprint |

**Vendor candidates (BSD-3 with attribution, into `oceanscale/sensors/_oceansim/`)**:
| File | Destination | What |
|---|---|---|
| `isaacsim/oceansim/utils/ImagingSonar_kernels.py` | `oceanscale/sensors/_oceansim/sonar_kernels.py` | Warp kernels: bin accumulation, Rayleigh + Gaussian speckle |
| `isaacsim/oceansim/utils/UWrenderer_utils.py` | `oceanscale/sensors/_oceansim/camera_kernels.py` | **Akkaynak-Treibitz** RGB attenuation + backscatter |
| `isaacsim/oceansim/utils/MultivariateNormal.py` | `oceanscale/randomize/_oceansim/mvnormal.py` | DR sampler |
| `isaacsim/oceansim/utils/MultivariateUniform.py` | `oceanscale/randomize/_oceansim/mvuniform.py` | DR sampler |

**Port (rewrite to remove Replicator dep)**:
| File | Destination | Why rewrite |
|---|---|---|
| `sensors/ImagingSonarSensor.py` | `oceanscale/sensors/sonar.py` | Replace Omniverse Replicator point-cloud annotator with our **Warp BVH over Newton scene mesh** |
| `sensors/UW_Camera.py` | `oceanscale/sensors/camera.py` | Compose with Omniverse RTX rendering when available (v0.3) |
| `sensors/DVLsensor.py` | `oceanscale/sensors/dvl.py` | 4-beam Janus + adaptive dropout |
| `sensors/BarometerSensor.py` | `oceanscale/sensors/pressure.py` | Direct port |

**Math correction**: STACK.md §5.2 says "Jaffe-McGlamery"; OceanSim actually uses **Akkaynak-Treibitz 2019** (per channel: `I_c = J·exp(-β_attn·d) + B_∞·(1 - exp(-β_bs·d))`). Patch in T1.4.

**Gap inherited** (we still must build):
- Caustics rendering (no Warp surface kernel in OceanSim)
- Jerlov water-type catalog (parameters tunable but no catalog)
- IMU model (inherited from Isaac Sim native; we need standalone)
- Acoustic comms (BELLHOP not in OceanSim)

---

### REF-ISAACAUV — Station-keeping design template ⭐⭐⭐⭐⭐

| Field | Value |
|---|---|
| **URL** | https://github.com/warplab/isaac-auv-env |
| **License** | **BSD-3-Clause** (inherited from Isaac Lab; LICENSE attributes "Isaac Lab Project Developers") |
| **Activity** | **A** — last pushed 2025-08-25 (8+ mo); 57 stars / 5 forks / 1 issue |
| **Authors** | Cai et al., WARPLab; ICRA 2025 |
| **Paper** | [arXiv 2410.00120](https://arxiv.org/abs/2410.00120) "Learning to Swim" |
| **Tag** | `v0.1-icra2025-warpauv` (2025-02-24) |
| **Stack** | Isaac Lab + rsl_rl (ETH minimalist PPO) + Isaac Sim |
| **Obs space** | `Box(shape=(17,), float64)` |
| **Action space** | `Box(low=-1, high=1, shape=(6,), float64)` |
| **Sim-to-real** | BlueROV2-class, **zero-shot deployed, beats hand-tuned PID** |

**Reuse plan**:
- **v0.1**: lift 17-D obs design + reward shaping + PPO hyperparameters; **write our own SB3 wrapper** (not Isaac Lab dependent)
- **v0.2**: port their Isaac Lab env config 1:1 once we integrate Isaac Lab
- **v0.3**: use as benchmark target — compare our sim-to-real to theirs

**Caveat on action dimension**: They use 6-D direct body-frame thrust commands, not 8 per-thruster commands. We will test both at T4.1 — 8-thruster gives policy more freedom but harder credit assignment; 6-D is easier to learn but requires a fixed allocation matrix (acceptable since allocation is well-known).

**Risk**: Inherited Isaac Lab BSD-3-Clause means our port carries upstream copyright attribution. Keep NOTICE.

---

## §3 Asset references

### Decision: BlueROV2 Heavy primary source

**Picked**: **MarineGym USD** (MIT, vendorable) for both geometry and dynamics.

**Why not** clydemcqueen/bluerov2_gz: **No LICENSE file** in repo + README has no licensing statement. Without explicit license, default copyright applies → we cannot legally vendor or modify. Best we can do is read the source for design ideas (allowed under fair-use scholarship).

**Plan B** if MarineGym USD asset proves inadequate:
1. Write our own URDF from scratch using BlueRobotics public datasheet specs (specs are factual, not copyrightable)
2. Use [orca4/orca_description](https://github.com/clydemcqueen/orca4) — sibling repo, **need to verify its LICENSE first** (may also be missing)
3. Issue PR to clydemcqueen/bluerov2_gz asking to add a LICENSE file (community good)

**Validation source**: cross-check geometry/inertia against Wu 2018 published tank-test parameters (§6).

---

## §4 Paper references

| Citation | Year | What we use |
|---|---|---|
| Fossen 2021 *Handbook of Marine Craft Hydrodynamics and Motion Control 2nd ed.* | 2021 | §3.3 Coriolis, §6.2 added-mass, §6.3 C_A, §7.1 damping, §8.2 restoring, §10.4 thruster allocation |
| Cai et al. *Learning to Swim* arXiv 2410.00120 | 2024 | 17-D obs + 6-D action + DR + zero-shot methodology |
| Chu et al. *MarineGym* arXiv 2503.09203 | 2025 | Fossen tensor implementation pattern; 8k env throughput claim |
| Song et al. *OceanSim* arXiv 2503.01074 | 2025 | Sonar intensity formula; Akkaynak-Treibitz parameters |
| Tunçay et al. *Fast Policy Learning 6-DOF* arXiv 2512.13359 | 2025 | 12-D body-frame obs design; 4096-env RTX 4060 result |
| Chaffre et al. IJRR 2025 *BlueROV adaptive DR* | 2025 | SAC+BIER; tank-test validation methodology |
| Potokar et al. *HoloOcean 2.0* arXiv 2510.06160 | 2025 | REMUS-validated <2% trajectory; 240 Hz substep methodology |
| Akkaynak & Treibitz *A revised underwater image formation model* CVPR 2018 | 2018 | Per-channel underwater image formation (camera model) |
| Wu 2018 *Modeling and control of BlueROV2 Heavy* (paper to identify) | 2018 | Published tank-test parameters for BlueROV2 Heavy |
| von Benzon et al. 2022 *An open-source BlueROV2 platform...* | 2022 | Alternative parameter set for validation |
| Hong et al. *Marine Robot DRL Survey* arXiv 2506.21063 | 2026 | Sim platform comparison taxonomy |

---

## §5 Anti-references — do NOT use

| Project | Reason | What we lose |
|---|---|---|
| **patrykcieslak/stonefish** | **GPL-3.0** — incompatible with Apache-2.0 outbound; viral license would force us to GPL the entire OceanScale | Validated C++ Fossen + sonar reference. We can **cite the paper** [Cieślak 2019 ICRA, arXiv 2502.11887] but not touch the code. |
| **uuvsimulator/uuv_simulator** | NOASSERTION + stale (Aug 2023 last push) + ROS 1 only | Historical Fossen Gazebo plugin reference; superseded by MarineGym |
| **clydemcqueen/bluerov2_gz** | **NO LICENSE FILE** — default copyright, unvendorable | Cleanest BlueROV2 Heavy SDF geometry. We can **read for design** (fair use); cannot copy/modify. |
| **gokulp01/bluerov2_gym** | **NO LICENSE** | Single-env gymnasium baseline; small loss |
| **patrickelectric/bluerov_ros_playground** | License unspecified | Incomplete BlueROV2 (no Heavy variant) |
| **Genesis-Embodied-AI/genesis-world** | Apache-2.0 compatible, but **NOT in main stack** per STACK.md §17 lock — single-vendor, marine not built-in (issue #682). Watchlist only. | Universal solvers; defer to post-v1.0 |
| **discoverse-dev/gs_playground** | MIT-compatible but tied to MotrixSim engine (single-lab, no LF backing); rigid-only RLGK; 3DGS bakes lighting (unfit for underwater). Watchlist. | 3DGS rendering speed; replicate via NVIDIA NuRec inside Isaac Sim instead |
| **Lightwheel proprietary SimReady assets** | Commercial; B2B | Marketing reference only, see project_strategy_framing memory (parked) |

---

## §6 Validation data

Sources for BlueROV2 Heavy parameter cross-checks. Numerical values from these papers/datasheets are factual (not copyrightable) and may be quoted with citation.

**Per DR review F2 (2026-05-15)**: previous primary "Wu 2018" turns out to be a Flinders MS thesis where the author never received the BlueROV2 Heavy hardware (thesis explicitly states this). Demoted. **Gold reference is now von Benzon 2022 JMSE 10(12):1898 (CC-BY-4.0)**.

| Source | License | Parameters available | Use |
|---|---|---|---|
| **von Benzon et al. 2022** [JMSE 10(12):1898, DOI 10.3390/jmse10121898](https://doi.org/10.3390/jmse10121898) | CC-BY-4.0 | Complete Fossen + thruster + tether model; tank-validated; published Simulink simulator | **GOLD primary** — generate reference trajectories for our Tier-1 unit tests |
| **BlueRobotics datasheet** (https://bluerobotics.com/store/rov/bluerov2/) | n/a (factual) | Mass, dimensions, thrust per motor, depth rating | Geometry sanity check |
| **Wu 2018 Flinders MS thesis** (PDF link in §9) | n/a | Approximate Fossen coefs **extrapolated from non-Heavy BlueROV** (thesis states tank tests not done) | **Caveat-cite only**; cross-check against von Benzon |
| **MarineGym `BlueROVHeavy.py` constants** | MIT | Eidsvik-method M_A + drag coefs (10-20% trans err, 30-100% rot err per their paper) | Cross-check; flagged for porting at T2.2 |
| **Manhães et al. 2016** (UUV Simulator paper) | Apache-2.0 | Generic UUV parameters | Sanity bounds |
| **Eidsvik 2015** (NTNU MS thesis) | n/a | Modeling reference | Cross-validation |
| **HoloOcean 2.0 paper** (arXiv 2510.06160) | MIT | REMUS-validated <2% methodology | Validation procedure |
| **FloWave tethered tank tests** (Heriot-Watt, ResearchGate 352310776) | Variable | 8-tether load-cell measurements, 1 m/s currents + regular waves | Concrete tank-test reference |
| **Cerrada et al. 2025** (Jornadas de Automática) | OA | BlueROV2 Heavy Depth-Hold mode least-squares ID with on-board sensors | Recent identification source |
| **MDPI 8(9):688 OpenFOAM + experimental hybrid** | OA | RANS surge/sway/heave/yaw + tank cross-validation | CFD-level cross-check |
| **CentraleNantesROV/bluerov2** (GitHub) | License TBV | Exact `thruster_cmd → pose_gt` rosbag schema | Future-hardware data logging template |

**Open dataset reality**: no packaged open-access (thruster_cmd → pose) dataset for BlueROV2 Heavy exists. Fallback: generate from von Benzon 2022 Simulink simulator.

We will record our final BlueROV2 Heavy parameters in `assets/vehicles/bluerov2_heavy.hydro.yaml` with citations per coefficient.

---

## §7 License compliance plan

For every vendored file:

1. **Preserve the original LICENSE** at `oceanscale/_vendored/<project>/LICENSE`.
2. **Top-of-file attribution** — every ported/vendored `.py` starts with:
   ```python
   # Adapted from <project> <commit_sha>
   # Originally licensed under <SPDX-ID>, copyright (c) <year> <author>
   # See oceanscale/_vendored/<project>/LICENSE
   ```
3. **`THIRD_PARTY_NOTICES.md`** at repo root — one row per vendored project with name, URL, license, files used, attribution line.
4. **`pyproject.toml` license-files** field lists all NOTICE files.
5. **In paper**: bibtex per §9 entry.

### Per-license requirements

| License | Must do |
|---|---|
| **MIT** | Preserve copyright + permission notices (the LICENSE file body) |
| **BSD-3-Clause** | (1) source redist: keep notice; (2) binary redist: keep notice in docs; (3) no using authors' names to endorse |
| **Apache-2.0** | (1) keep LICENSE + NOTICE files; (2) state significant changes; (3) keep attributions in derivative works |
| **CC-BY-4.0** (assets) | Credit creator, link to license, indicate changes |

---

## §8 Quick-fetch commands

```bash
# One-off scratch clone of all references (NOT committed):
mkdir -p /tmp/oceanscale-refs && cd /tmp/oceanscale-refs

# OK to vendor
git clone --depth 1 https://github.com/Marine-RL/MarineGym                  # MIT
git clone --depth 1 https://github.com/umfieldrobotics/OceanSim             # BSD-3
git clone --depth 1 https://github.com/warplab/isaac-auv-env                # BSD-3 (inherited)
git clone --depth 1 https://github.com/newton-physics/newton                # Apache-2.0
git clone --depth 1 https://github.com/NVIDIA/warp                          # Apache-2.0
git clone --depth 1 https://github.com/cybergalactic/MSS                    # MATLAB ref

# Read-only / study (no vendor allowed)
git clone --depth 1 https://github.com/clydemcqueen/bluerov2_gz             # NO LICENSE — DO NOT COPY
git clone --depth 1 https://github.com/clydemcqueen/orca4                   # license unverified
git clone --depth 1 https://github.com/gokulp01/bluerov2_gym                # NO LICENSE — DO NOT COPY

# Verify licenses
for d in */; do
  echo "=== $d ==="
  for f in LICENSE LICENSE.txt LICENSE.md COPYING; do
    [ -f "$d/$f" ] && head -5 "$d/$f" && break
  done
done
```

---

## §9 BibTeX bibliography

```bibtex
@book{fossen2021handbook,
  title={Handbook of Marine Craft Hydrodynamics and Motion Control},
  author={Fossen, Thor I.},
  year={2021},
  edition={2nd},
  publisher={John Wiley \& Sons}
}

@article{cai2024learning,
  title={Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven Autonomous Underwater Vehicles},
  author={Cai et al.},
  journal={arXiv preprint arXiv:2410.00120},
  year={2024}
}

@article{chu2025marinegym,
  title={MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics},
  author={Chu, Shuguang and others},
  journal={arXiv preprint arXiv:2503.09203},
  year={2025},
  note={IROS 2025}
}

@article{song2025oceansim,
  title={OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework},
  author={Song, Jingyu and Ma, Haoyu and others},
  journal={arXiv preprint arXiv:2503.01074},
  year={2025},
  note={IROS 2025}
}

@article{tuncay2025fast,
  title={Fast Policy Learning for 6-DOF Position Control of Underwater Vehicles},
  author={Tun{\c{c}}ay, et al.},
  journal={arXiv preprint arXiv:2512.13359},
  year={2025}
}

@article{xie2026easyuuv,
  title={EasyUUV: Domain-Randomized Underwater Robotic Manipulation with LLMs},
  author={Xie et al.},
  journal={arXiv preprint arXiv:2510.22126},
  year={2026}
}

@article{chaffre2025adaptive,
  title={Adaptive Domain Randomization for BlueROV Station-Keeping},
  author={Chaffre et al.},
  journal={International Journal of Robotics Research},
  volume={44},
  number={3},
  pages={407--430},
  year={2025}
}

@article{potokar2025holoocean2,
  title={A Preview of HoloOcean 2.0},
  author={Potokar et al.},
  journal={arXiv preprint arXiv:2510.06160},
  year={2025}
}

@inproceedings{akkaynak2018revised,
  title={A Revised Underwater Image Formation Model},
  author={Akkaynak, Derya and Treibitz, Tali},
  booktitle={CVPR},
  year={2018}
}

@article{hong2026marine,
  title={Control of Marine Robots in the Era of Data-Driven Intelligence},
  author={Hong et al.},
  journal={arXiv preprint arXiv:2506.21063},
  year={2026}
}

@misc{newton2026,
  title={Newton: An Open-Source GPU-Accelerated Physics Engine for Robotics Simulation},
  howpublished={\url{https://github.com/newton-physics/newton}},
  year={2026},
  note={Apache-2.0; Linux Foundation. v1.0 GA at GTC 2026.}
}
```

---

## §10 Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-15 | BlueROV2 source = **MarineGym USD** (not bluerov2_gz) | License-compatible (MIT vs none) |
| 2026-05-15 | Tier-1 math source = **MarineGym `underwaterVehicle.py`** | MIT, IROS 2025 paper-backed, matches our regime |
| 2026-05-15 | Sonar reference = **OceanSim** BSD-3 (defer to v0.2) | Compatible license, BVH replace needed |
| 2026-05-15 | Camera model = **Akkaynak-Treibitz** (NOT Jaffe-McGlamery) | OceanSim uses AK-T; STACK.md §5.2 needs patch |
| 2026-05-15 | RL framework v0.1 = **stable-baselines3** | rl_games has psutil<6 vs warp>=7.1 conflict |
| 2026-05-15 | Station-keep design = **REF-ISAACAUV** (17-D obs, 6-D action) | Closest stack match + real-hardware-validated |
| 2026-05-15 | Anti-reference Stonefish (GPL-3) | License taint risk |
| 2026-05-15 | Anti-reference bluerov2_gz | No LICENSE = cannot vendor; study-only |
| 2026-05-15 (DR-F2) | **Gold validation = von Benzon 2022** (DOI 10.3390/jmse10121898); Wu 2018 demoted | Wu 2018 is Flinders MS thesis; author never received hardware; von Benzon is tank-validated + CC-BY |
| 2026-05-15 (DR-F3) | Newton pin tightened to `>=1.2.0,<1.3` | 3 minors in 30 days; sub-minor breaking changes documented; hydroelastic 30× regression in 1.2 |
| 2026-05-15 (DR-F6) | Added 6 refs: Sim2Swim, Learning-to-Dock, diffSPH, CentraleNantesROV, OsloMet-OceanLab, PrivilegedDreamer | Closes gaps surfaced by DR review |

---

## §11 Pending verifications

These must be resolved before the related T-task can start:

| Pending | Blocks | Action | Owner |
|---|---|---|---|
| MarineGym latest commit SHA → pin | T3.3 vendor | `cd refs/MarineGym && git rev-parse HEAD` | T3.1 |
| OceanSim latest commit SHA → pin | v0.2 vendor | same pattern | v0.2 sprint |
| Newton 1.x → 2.x deprecation timeline | R2 risk | watch upstream | continuous |
| MarineGym USD asset quality (does it render in Newton viewer?) | T3.2 | visual inspection | T3.1 |
| orca4 LICENSE verification (fallback if MarineGym USD inadequate) | T3.1 fallback | fetch LICENSE | T3.1 |
| `bluerov2_gz` license — file PR asking for LICENSE? | future-vendor | community contribution | optional |
| Wu 2018 paper exact DOI / title | T3.3 validation | google-scholar | T3.3 |
| `cybergalactic/MSS` LICENSE verification | T1.3 ref | fetch LICENSE | T1.3 |
| `warplab/isaac-auv-env` reward function (need raw file) | T4.1 design | fetch env file | T4.1 |

---

## §12 What this file is NOT

- Not a survey of the field — see `SURVEY.md`
- Not an architecture proposal — see `DESIGN.md` and `STACK.md`
- Not an implementation plan — see `IMPLEMENTATION_PLAN.md`
- Not a place to add aspirational references — every entry must be admitted via §0
- Not a tutorial — links to upstream docs, doesn't reproduce them

When in doubt about adding a new entry: **does it pass §0?** If not, it goes to §5 Anti-references with a reason, not into the main matrix.
