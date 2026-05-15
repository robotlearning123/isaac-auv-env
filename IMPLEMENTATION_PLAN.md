# Implementation Plan — v0.1.0 (post-scaffold → ship)

**Date:** 2026-05-15  
**Branch:** `feat/v0.1.0-scaffold` (current) → future `feat/v0.1.0-fossen`, etc.  
**Goal:** Train a station-keeping PPO policy for BlueROV2 Heavy on OceanScale, with Tier-1 GPU-parallel Fossen hydrodynamics, ≥100 k env-steps/s at 8192 envs on RTX 5090.

**Ship definition** (must be true to call v0.1.0 done):
1. ≥ 8192 parallel envs run BlueROV2 Heavy under Fossen Tier-1 hydrodynamics on Newton.
2. PPO policy trained from random init reaches station-keep error ≤ 0.1 m within 10 M env-steps.
3. Wall-clock training time ≤ 30 minutes on a single RTX 5090.
4. Full test suite green; benchmark numbers recorded.
5. Tier-1 Fossen kernel has unit test parity against an analytic reference within 1e-4 relative error.
6. Autograd through the kernel matches torch numerical-grad within 1e-3.
7. Tagged release `v0.1.0` on `feat/v0.1.0-station-keep` branch ready for merge into `main`.

**Timeline acknowledgement (DR review F1, 2026-05-15)**: 4 weeks solo is **a stretch goal**. DR review found realistic timeline = 6-9 weeks for this scope (Cai=3-author lab, Chu=7-author team, Chaffre=PhD-scale). Decision: keep 4w as **aggressive target**, treat W4 slip into W5-W6 as expected, not failure. If on-track at W2 gate, hold 4w; if not, slip ship date, not scope.

**TDD rule (DR review F1 → R20)**: every Tier-1 kernel (T2.2-T2.6) must be written **AFTER** a torch numerical-diff harness for it exists in `tests/hydro/test_*_autograd.py`. The harness is the spec; the kernel passes it.

**Thruster nonlinearity rule (DR review F4 → R22)**: T2.6 must include saturation, deadband, and 1st-order time-constant. NOT just `τ = T · u`. MarineGym's likely-linear-only thrust model is insufficient for zero-shot sim-to-real.

---

## Iron principles for this sprint

1. **No from-scratch.** Every implementation step starts with "which existing project do we vendor / port / cite?" See `REFERENCES.md` (in progress) for the candidate map.
2. **Plan → branch → code → test → commit → next.** Each task is one short-lived branch and one logical commit.
3. **Verify on hardware, not on faith.** No conclusion accepted without a passing test or recorded benchmark in `install_log/`.
4. **Empirical truth beats spec.** When the installed API contradicts STACK.md, the installed API wins; patch the doc.
5. **Stay tight on scope.** Anything not in §6 (Ship Definition) is v0.2+. No exploration during ship sprint.

---

## Sprint structure — 4 weeks, 5 subgoals (G1–G5)

```
W1: G2  Newton-worlds investigation + Tier-1 math sketch
W2: G3  Tier-1 Fossen Warp kernel + autograd tests
W3: G4  BlueROV2 Heavy asset + scene + multi-env smoke
W4: G5  StationKeeping env + PPO + ship gate
        + buffer for slips
```

Doc patches and reference compilation are continuous, not sprint-gated.

---

## Week 1 — G2: Worlds API + math sketch

Goal: resolve the parallel-env architecture gap discovered in `STATUS.md`. Decide on Newton API pattern for 8192 parallel envs. Sketch the 6-DOF Fossen math.

### T1.1  Identify Newton worlds / batched-env API
- **What**: read Newton source + bundled examples; find canonical multi-env pattern
- **Verify**: write a 3-line PoC creating N independent envs and stepping them
- **Reference**: agent `newton-worlds-api` (in flight)
- **Output**: `install_log/40_newton_worlds_poc.txt`
- **Gate to next**: PoC works at N=64 minimum

### T1.2  Throughput re-benchmark with proper worlds
- **What**: rerun `benchmarks/kernel_throughput.py` using the worlds pattern at N ∈ {64, 256, 1024, 4096, 8192}, both SolverSemiImplicit and SolverMuJoCo
- **Verify**: SolverMuJoCo no longer OOMs at N=8192
- **Output**: `STATUS.md` updated benchmark table, `benchmarks/kernel_throughput_worlds.py`
- **Gate**: choose v0.1 default solver based on results

### T1.3  6-DOF Fossen math sketch
- **What**: write `docs/math/fossen.md` (informal) — the equations we will implement, variable names, units, sign conventions, reference axis (NED vs ENU)
- **Verify**: review against Fossen 2021 + MarineGym source
- **Reference**: agent `marinegym-deep` (in flight)
- **Output**: `docs/math/fossen.md`

### T1.4  Docs patch
- **What**: update STACK.md / DESIGN.md / RISKS.md with empirical findings from scaffold sprint:
  - spatial-vector layout (linear_xyz, angular_xyz)
  - `add_body()` implicit free joint
  - shape density overrides body mass
  - SolverMuJoCo CPU OOM behavior (or its resolution after T1.2)
- **Verify**: diff review; all 4 gotchas patched
- **Output**: commit `docs(empirical): record 4 Newton 1.2 API gotchas from scaffold sprint`

### Week-1 gate
- Worlds API working, throughput re-measured, default solver chosen.
- Math sketch + docs patched.
- If T1.1 finds no clean worlds API → **architectural decision required**: stay on SolverSemiImplicit at N=8192 (proven 220 M env-steps/s) for v0.1, defer SolverMuJoCo to v0.2.

---

## Week 2 — G3: Tier-1 Fossen Warp kernel (the core IP)

Goal: differentiable 6-DOF Fossen dynamics as Warp kernels, integrating into Newton's body_f via pre-step hook.

### T2.1  Tier-1 kernel skeleton + force injection wiring
- **What**: write `oceanscale/hydro/tier1_kernels.py` with placeholder kernels; wire up via Newton pre-step hook (or whatever the worlds API provides) so forces reach body_f
- **Verify**: kernel runs, body_f reads back nonzero; unit test asserts wiring works with zero force first
- **Reference**: study how MarineGym injects (via PhysX FloatingBaseAsset wrappers; we need Newton equivalent)
- **Output**: failing test; pass once wiring works

### T2.2  Added-mass M_A
- **What**: 6×6 added-mass matrix as a per-env Warp array; multiply by ν̇ (estimated from previous step) → force
- **Verify**: unit test against analytic Fossen handbook example values (e.g., a known sphere, prolate spheroid)
- **Reference**: MarineGym PyTorch implementation; Fossen 2021 §8.2
- **Output**: 1 of the 5 forces wired

### T2.3  Quadratic damping D(ν)ν
- **What**: nonlinear drag, both linear and quadratic terms; component-wise diag default
- **Verify**: matches MarineGym's reference values for BlueROV2 Heavy at unit velocity in each axis
- **Output**: 2 of 5

### T2.4  Coriolis C_RB(ν), C_A(ν)
- **What**: rigid-body and added-mass Coriolis cross-terms
- **Verify**: angular maneuver test — yaw input gives correct centripetal coupling
- **Output**: 3 of 5

### T2.5  Restoring g(η)
- **What**: buoyancy + weight + COB offset → restoring force (depends on quaternion)
- **Verify**: stable hover at neutral buoyancy; tilts back to upright when displaced
- **Output**: 4 of 5

### T2.6  Thruster mapping τ
- **What**: 8 → 6 thruster matrix for BlueROV2 Heavy; saturate, deadband
- **Verify**: identity command (all thrusters off) gives τ=0; symmetric forward/back gives only surge
- **Output**: 5 of 5

### T2.7  Autograd correctness test
- **What**: wp.Tape gradient through full Tier-1 vs torch autograd numerical-diff (finite differences) at random states
- **Verify**: max relative error < 1e-3 across 100 random samples
- **Output**: `tests/hydro/test_tier1_autograd.py` passes

### T2.8  Performance test
- **What**: bench Tier-1 + Newton step at N ∈ {1024, 4096, 8192}
- **Verify**: ≥ 100 k env-steps/s at N=8192 on RTX 5090
- **Output**: `STATUS.md` row added

### Week-2 gate
- All 5 forces wired with unit tests.
- Autograd correctness verified.
- Throughput ≥ 100 k env-steps/s at N=8192.
- If throughput fails → identify hot kernel via profiler before W3.

---

## Week 3 — G4: BlueROV2 Heavy asset + scene

Goal: realistic BlueROV2 Heavy in OceanScale, ready for RL.

### T3.1  Pick canonical BlueROV2 Heavy source
- **What**: review `REFERENCES.md` candidates (MarineGym, HoloOcean, Stonefish, uuv_simulator, Orca4, bluerov2_gym)
- **Verify**: pick one based on license + completeness + format
- **Reference**: agent `bluerov2-assets` (in flight)
- **Output**: decision recorded in `REFERENCES.md`

### T3.2  Asset port → MJCF or USD
- **What**: port the chosen source to MJCF (preferred — Newton has best MJCF importer) or USD
- **Verify**: visual inspection in Newton viewer; mass + inertia + dimensions match published BlueROV2 specs (mass 13.6 kg, dim 0.46×0.575×0.254 m)
- **Output**: `assets/vehicles/bluerov2_heavy.xml` (MJCF) or `.usd`

### T3.3  Hydrodynamic coefficient YAML
- **What**: `assets/vehicles/bluerov2_heavy.hydro.yaml` — M_A diagonal, D_lin, D_quad, COB offset, thruster positions+directions, all from published BlueROV2 Heavy values (Wu 2018 + BlueRobotics datasheet)
- **Verify**: cross-check with MarineGym hydro params
- **Output**: 1 file, ~30 lines

### T3.4  Python builder
- **What**: `oceanscale/scene/vehicles/bluerov2_heavy.py` exposing `build(num_envs, init_pose) → (model, hydro_cfg)`
- **Verify**: builds N=8192 envs without OOM; finalize succeeds; mass = 13.6 ± 0.5 kg per env
- **Output**: passes `tests/scene/test_bluerov2_heavy.py`

### T3.5  End-to-end smoke
- **What**: 8192 BlueROV2 envs + Tier-1 hydro + Newton solver, step 1 second of sim time
- **Verify**: bodies remain stable (no NaN, no escaping the domain); rest at neutral buoyancy
- **Output**: `install_log/45_bluerov_smoke.txt`

### Week-3 gate
- BlueROV2 Heavy alive in OceanScale at 8192 envs with realistic mass/inertia/hydro.
- Stable at rest, responsive to thruster inputs.

---

## Week 4 — G5: StationKeepingEnv + PPO + ship

Goal: train a station-keeping policy from scratch to ≤ 0.1 m error in ≤ 30 minutes.

### T4.1  Observation / action / reward spec
- **What**: write `oceanscale/tasks/station_keeping.py` with explicit ObsCfg + ActCfg + RewCfg
- **Reference**: MarineGym's station_keeping.py; arXiv 2410.00120 Learning-to-Swim
- **Verify**: pyright-style review; dimensionality matches BlueROV2 sensor suite

### T4.2  Gymnasium adapter
- **What**: `oceanscale/rl/gym_env.py` — vectorized `gym.Env` wrapping Newton state
- **Verify**: passes `gymnasium` env_checker

### T4.3  PPO baseline framework choice
- **What**: pick — stable-baselines3 (easiest) vs custom PPO (clean) vs skrl 2.0 (powerful) vs PureJaxRL pattern (fastest)
- **Constraint**: rl_games still has psutil<6 vs warp>=7.1 conflict, so NOT rl_games yet
- **Decision**: provisional **stable-baselines3** for v0.1 (battle-tested, no conflict); revisit at v0.2 when migrating to Isaac Lab
- **Output**: decision recorded

### T4.4  Training script
- **What**: `examples/04_station_keeping_train.py` — train PPO, log to tensorboard
- **Verify**: runs without error; logs reward + episode length

### T4.5  Train to convergence
- **What**: train 10 M env-steps; capture learning curve
- **Verify**: final policy mean position error ≤ 0.1 m; training wall-clock ≤ 30 min
- **Output**: `results/v0.1.0_station_keep/` directory with logs + ckpt + curve PNG

### T4.6  Eval + ONNX export
- **What**: load ckpt, run 100 deterministic eval episodes, export ONNX
- **Verify**: ONNX inference matches PyTorch within fp16 tolerance
- **Output**: `results/v0.1.0_station_keep/policy.onnx`

### T4.7  Tag and merge
- **What**: bump VERSION, write CHANGELOG entry, tag `v0.1.0`, open PR to `main`
- **Verify**: all tests green, all gates passed, branch ready
- **Output**: PR merged to main; v0.1.0 released

### Week-4 gate
- Trained policy meets all 7 ship criteria from top of doc.
- Tagged + merged.

---

## Continuous (across all weeks)

- `REFERENCES.md` grows with each new candidate or decision
- `RISKS.md` reviewed weekly Friday; states updated
- `install_log/` keeps growing — every meaningful command logged
- `STATUS.md` updated whenever a benchmark or empirical finding changes
- `STACK.md` patched when API contradicts expectations
- `pyproject.toml` updated only for verified-needed deps; document why in PR

---

## Out of scope for v0.1.0 (do NOT pull in)

- Tier-2 SPH / Tier-3 waves — v0.3
- Sonar, vision, DVL, IMU sensor kernels — v0.2
- Isaac Sim / Isaac Lab integration — v0.2
- Multi-agent / acoustic comms — v0.4
- Genesis or Moore Threads backend — never (per main-stack lock)
- 3DGS rendering — v0.3 spike, not v0.1
- Real-hardware validation on BlueROV2 — v0.5
- Docs site (Sphinx/MkDocs) — v0.3

If something on this list appears tempting mid-sprint, **add it to the v0.2 backlog and keep moving**.

---

## KPI dashboard (one row per Friday)

| Date | W | Tasks done | Tests | Throughput | Notes |
|---|---|---|---|---|---|
| 2026-05-15 | 0 | scaffold | 18/18 | 220 M env-steps/s (no hydro, free body) | install + smoke complete |
| 2026-05-22 | 1 | … | … | … | … |
| 2026-05-29 | 2 | … | … | … | … |
| 2026-06-05 | 3 | … | … | … | … |
| 2026-06-12 | 4 | ≤ 0.1 m | green | ≥ 100 k @ 8192 envs Tier-1 | SHIP |

---

## Decision log (filled live)

- 2026-05-15  v0.1 default solver = **SolverSemiImplicit** unless T1.1 finds a clean worlds API for SolverMuJoCo.
- 2026-05-15  v0.1 RL framework = **stable-baselines3** unless T4.3 finds a compatible alternative without psutil conflict.
- 2026-05-15  v0.1 BlueROV2 source = **TBD** pending agent `bluerov2-assets` (in flight).
- 2026-05-15  v0.1 asset format = **MJCF preferred** for fastest Newton import; USD reserved for v0.2.

---

## What this plan is NOT

- Not a research plan — sim-to-real, multi-agent, novel kernels = v0.2+
- Not a paper outline — paper draft is its own deliverable post-v0.1
- Not a hiring plan — solo developer
- Not a benchmark sweep — one number per task is enough; comprehensive benchmarking is its own sprint
