# /goal Plan — W2: Tier-1 Fossen Warp Kernels

**Status**: ready to fire `/goal` against this plan.  
**Date**: 2026-05-15  
**Branch**: new `feat/v0.1.0-tier1-fossen` (off `feat/v0.1.0-worlds-poc`).  
**Estimated wall-clock**: 12-20 hours focused work (8 subtasks T2.1-T2.8).  
**Risk**: HIGH per DR review F1 + R20 (Warp autograd debug 5d budget).

---

## 0. Goal statement (single sentence)

Implement Tier-1 6-DOF Fossen hydrodynamics as differentiable Warp kernels (added-mass, Coriolis, damping, restoring, thruster allocation), integrated into Newton's pre-step force injection on replicated worlds, with TDD-style unit tests and autograd correctness verified to ≤ 1e-3 relative error vs torch numerical-diff at ≥ 100k env-steps/s on 8192-env scale.

---

## 1. `/goal` condition string (copy-paste ready)

```
W2 done: tests/hydro/ green (8 tests), tier1_throughput ≥100k env-steps/s @ 8192 envs, autograd rel-err < 1e-3, ruff+mypy clean, commits on feat/v0.1.0-tier1-fossen, all T2.1-T2.8 marked completed
```

**How /goal verifies**: command sequence

```bash
.venv/bin/python -m pytest tests/hydro/ -v
.venv/bin/python benchmarks/tier1_throughput.py
.venv/bin/ruff check oceanscale tests
.venv/bin/mypy oceanscale/hydro/ --strict
git log --oneline feat/v0.1.0-tier1-fossen -8
```

All five must succeed for /goal to release.

---

## 2. Success criteria (binary; the goal-driven loop checks these)

| # | Criterion | Verification command | Pass = |
|---|---|---|---|
| 1 | T2.1 injection wiring works | `pytest tests/hydro/test_injection_wiring.py -v` | `8 passed` or > 0 passed |
| 2 | T2.2 added-mass kernel correct | `pytest tests/hydro/test_added_mass.py -v` | all green |
| 3 | T2.3 damping kernel correct | `pytest tests/hydro/test_damping.py -v` | all green |
| 4 | T2.4 Coriolis kernels correct | `pytest tests/hydro/test_coriolis.py -v` | all green |
| 5 | T2.5 restoring kernel correct | `pytest tests/hydro/test_restoring.py -v` | all green |
| 6 | T2.6 thruster allocation correct | `pytest tests/hydro/test_thruster.py -v` | all green |
| 7 | T2.7 full-pipeline autograd correct | `pytest tests/hydro/test_tier1_autograd.py -v` | rel-err < 1e-3 |
| 8 | T2.8 throughput target met | `python benchmarks/tier1_throughput.py` | reports ≥ 100,000 env-steps/s at N=8192 |
| 9 | Lint + type clean | `ruff check oceanscale tests && mypy oceanscale/hydro/ --strict` | exit 0 |
| 10 | Commit hygiene | one commit per T2.* on `feat/v0.1.0-tier1-fossen` | 8 commits visible |

---

## 3. Preconditions (must hold before /goal fires)

- [x] `feat/v0.1.0-worlds-poc` has T1.1 + T1.2 + T1.3 committed
- [x] `docs/math/fossen.md` exists (T1.3) — implementation reference
- [x] `REFERENCES.md` lists MarineGym, von Benzon 2022, Fossen 2021
- [x] `newton.ModelBuilder.replicate(world_count=N)` verified working (T1.1)
- [x] `SolverSemiImplicit` confirmed as v0.1 default (T1.2, locked)
- [ ] T1.4 docs patches landed (NOT a strict precondition; can run in parallel)
- [ ] `feat/v0.1.0-tier1-fossen` branch created off worlds-poc
- [ ] MarineGym repo cloned to scratch for reference: `/tmp/oceanscale-refs/MarineGym/`

---

## 4. References — read BEFORE each subtask

| For | Read | Why |
|---|---|---|
| All T2.* | `docs/math/fossen.md` (sections 4-8) | Variable names, sign conventions, kernel→T-task mapping |
| All T2.* | `REFERENCES.md` §2 REF-MARINEGYM | What we vendor + port; attribution |
| T2.1 wiring | `tests/test_worlds_poc.py` (existing) | Newton replicate() + state.body_f pattern |
| T2.2 M_A | `marinegym/robots/drone/underwaterVehicle.py::calculate_added_mass` | Source impl (PyTorch tensors) |
| T2.3 damping | same `::calculate_damping` | Source impl |
| T2.4 Coriolis | same `::calculate_corilis` (sic) | Source impl |
| T2.5 restoring | same `::calculate_buoyancy` | Source impl |
| T2.6 thruster | clydemcqueen `bluerov2_gz` SDF (read only, no copy) + Fossen §10.4 | Thruster geometry + allocation matrix |
| T2.7 autograd | `tests/test_warp_smoke.py::test_quad_drag_autograd_matches_analytic` | wp.Tape pattern |
| T2.8 perf | `benchmarks/newton_worlds_throughput.py` | Existing benchmark template |

---

## 5. Per-subtask spec (detailed)

### T2.1 — Tier-1 kernel skeleton + Newton pre-step force injection

**Files to create**:
```
oceanscale/hydro/__init__.py                # exports: Tier1
oceanscale/hydro/tier1_kernels.py           # @wp.kernel stubs (no real math yet)
oceanscale/hydro/injector.py                # pre-step force injection wiring
tests/hydro/__init__.py
tests/hydro/test_injection_wiring.py        # TDD spec — write FIRST
```

**TDD spec (test_injection_wiring.py)**:
```
test_inject_zero_force_no_change_in_velocity:
  Build N=64 worlds, attach Tier1 with all kernels returning 0,
  step once, assert state.body_qd unchanged (within machine epsilon).

test_inject_constant_world_force:
  Inject a constant downward force F = -m·g in body frame for each world,
  step once, assert net acceleration ≈ 0 (cancels gravity).

test_inject_per_world_different_force:
  Inject F_world[i] = i*1.0 N in surge for each world,
  step once, assert v_surge[i] proportional to i.
```

**Implementation**:
- `Tier1` is a dataclass holding per-env wp.arrays for hydro state
- `injector.attach(model, tier1)` registers a pre-step callback
- Callback launches kernels, accumulates into a `wp.spatial_vectorf wrench_buffer`, then a final kernel copies wrench_buffer → state.body_f

**Verification**:
```bash
.venv/bin/python -m pytest tests/hydro/test_injection_wiring.py -v
```
Pass = 3/3.

**Commit message**:
```
feat(hydro): T2.1 Tier-1 skeleton + Newton pre-step force injection wiring
```

---

### T2.2 — Added-mass M_A Warp kernel

**Files**:
```
oceanscale/hydro/tier1_kernels.py   # add added_mass_kernel
tests/hydro/test_added_mass.py      # TDD first
assets/vehicles/bluerov2_heavy.hydro.yaml  # placeholder hydro coefs (BlueROV2 Heavy)
```

**TDD spec**:
```
test_unit_surge_derivative:
  ν̇ = [1, 0, 0, 0, 0, 0], M_A = diag(5.5, 12.7, 14.57, 0.12, 0.12, 0.12)
  expected wrench[0] = -5.5 N (Fossen sign: F_A = -M_A · ν̇)
  tolerance: 1e-5

test_unit_yaw_derivative:
  ν̇ = [0, 0, 0, 0, 0, 1]
  expected wrench[5] = -0.12 N·m
  tolerance: 1e-5

test_diag_only_no_cross_coupling:
  ν̇ random, M_A diag → wrench[i] = -M_A[i,i] · ν̇[i]
  100 random samples, all rel-err < 1e-5

test_autograd_dwrench_dnudot:
  wp.Tape vs analytic: dF_A/dν̇ = -M_A (constant matrix)
  100 random states, max rel-err < 1e-4
```

**Kernel signature (Warp pseudocode)**:
```python
@wp.kernel
def tier1_added_mass(
    nu_dot: wp.array(dtype=wp.spatial_vectorf),   # (n_envs,) per-env accel
    M_A: wp.array(dtype=wp.vec6f),                # (n_envs,) per-env diag M_A
    wrench: wp.array(dtype=wp.spatial_vectorf),   # (n_envs,) output, ACCUMULATED
):
    i = wp.tid()
    nu_dot_i = nu_dot[i]
    m_a_i = M_A[i]
    # element-wise: F_A = -diag(M_A) · ν̇
    f_lin = -wp.vec3f(m_a_i[0]*nu_dot_i[0], m_a_i[1]*nu_dot_i[1], m_a_i[2]*nu_dot_i[2])
    f_ang = -wp.vec3f(m_a_i[3]*nu_dot_i[3], m_a_i[4]*nu_dot_i[4], m_a_i[5]*nu_dot_i[5])
    wp.atomic_add(wrench, i, wp.spatial_vector(f_lin, f_ang))  # ACCUMULATE (other kernels also write here)
```

**ν̇ estimation**: explicit added-mass scheme — `ν̇ ≈ (ν[k] - ν[k-1]) / dt`. Use a `nu_prev` buffer.

**Verification**:
```bash
.venv/bin/python -m pytest tests/hydro/test_added_mass.py -v
```

**Commit message**:
```
feat(hydro): T2.2 added-mass M_A Warp kernel with autograd
```

---

### T2.3 — Damping D(ν)·ν Warp kernel

**TDD spec**:
```
test_unit_surge_velocity:
  ν = [1, 0, ..., 0], D_lin = diag(-4.03,...), D_quad = diag(-18.18,...)
  expected wrench[0] = -(D_lin[0] + D_quad[0]·1)·1 = -(-4.03) + -(-18.18) = 22.21 N
  Wait sign: F = -D·ν; F = -(D_lin·ν + D_quad·|ν|·ν) = -(-4.03·1 + -18.18·|1|·1) = -(-22.21) = +22.21
  Hmm, drag opposes motion so should be NEGATIVE.
  D_lin coefs in Fossen are typically negative; D · ν gives a positive number; -D · ν gives the drag force (negative, opposing motion).
  RE-VERIFY in MarineGym source at port-time; bake the sign convention into the test.

test_zero_velocity_zero_force:
  ν = 0 → wrench = 0  (exact)

test_negative_velocity_negative_force:
  ν = [-1, 0, ...] → wrench[0] sign opposite from +1 case

test_quadratic_dominates_at_high_velocity:
  ν = [10, 0, ...] → quadratic term should be ~100× larger than linear

test_autograd_dwrench_dnu_at_zero:
  At ν = 0, dF/dν = -D_lin (constant), no quadratic contribution
  Verify wp.Tape returns -D_lin within 1e-5

test_autograd_dwrench_dnu_random:
  100 random ν, autograd vs torch finite-diff, rel-err < 1e-3
```

**Kernel**:
```python
@wp.kernel
def tier1_damping(
    nu: wp.array(dtype=wp.spatial_vectorf),
    d_lin: wp.array(dtype=wp.vec6f),
    d_quad: wp.array(dtype=wp.vec6f),
    wrench: wp.array(dtype=wp.spatial_vectorf),
):
    i = wp.tid()
    v = nu[i]
    d_l = d_lin[i]
    d_q = d_quad[i]
    f = wp.spatial_vector(...)  # F = -(D_lin + D_quad·|v|)·v componentwise
    wp.atomic_add(wrench, i, f)
```

**Commit**: `feat(hydro): T2.3 damping D(ν)ν Warp kernel with autograd`

---

### T2.4 — Coriolis C_RB(ν) + C_A(ν) Warp kernels

**TDD spec**:
```
test_zero_velocity_zero_coriolis:
  ν = 0 → wrench = 0 exact

test_pure_yaw_no_force:
  ν = [0,0,0, 0,0,r] → linear-component wrench = 0 (yaw alone doesn't create surge/sway force in body frame)

test_surge_plus_yaw_centripetal:
  ν = [u, 0,0, 0,0,r] with u, r > 0
  C(ν)·ν gives a side force in -y direction (sway), magnitude m·u·r
  Sign: F_centripetal points TOWARD the center of rotation; for yaw r > 0 turning left,
        centripetal points to body +y or -y depending on convention. Resolve via MarineGym source.

test_diagonal_M_A_simplification_consistent:
  Both rigid-body and added-mass forms simplify; the combined wrench should equal manual computation.

test_autograd_random:
  100 random ν, max rel-err < 1e-3 vs torch finite-diff
```

**Implementation note**: per `docs/math/fossen.md §5.3` use block form with diagonal M_A:
```
C_RB(ν)·ν has terms like (m·ω_b × v_b), (I·ω_b × ω_b)
C_A(ν)·ν has terms like (M_A_lin·v_b) × ω_b
```

**Commit**: `feat(hydro): T2.4 Coriolis C_RB + C_A Warp kernels with autograd`

---

### T2.5 — Restoring g(η) Warp kernel

**TDD spec**:
```
test_neutral_buoyancy_zero_wrench_when_upright:
  q = identity, W = B (neutral), r_b_b = (0,0,0.01)
  expected wrench ≈ 0 except small moment from COB offset...
  actually wait — if W = B then NET force = 0, but moment = r_b_b × (-W·ẑ) − r_b_b × (-B·ẑ) = r_b_b × (B-W)·ẑ = 0 too.
  So neutral + upright → wrench = 0.

test_tilt_30deg_about_x_restoring_moment:
  q = quat(30°, x-axis), W = B, r_b_b z > 0 (COB above COG)
  Restoring moment should point such that it RIGHTS the vehicle (negative roll moment)

test_negative_buoyancy_sinks:
  W > B → net force component in body frame in +z direction (downward in NED)

test_autograd_quaternion_smooth:
  Restoring depends on q via R(q); verify autograd through this chain
  100 random (q, r_b_b, W, B), max rel-err < 1e-3
```

**Implementation**: per Fossen §6.5 + `docs/math/fossen.md §5.5`.

**Commit**: `feat(hydro): T2.5 restoring g(η) Warp kernel with autograd`

---

### T2.6 — Thruster mapping τ = T · u Warp kernel

**TDD spec**:
```
test_all_zero_input:
  u = 0_8 → τ = 0_6 exact

test_pure_surge_thrust:
  All 4 horizontal thrusters at +1.0 (signs per BlueROV2 Heavy geometry)
  → τ should have only surge (+X) component, sway/heave/yaw/pitch/roll all ≈ 0

test_pure_heave_thrust:
  All 4 vertical thrusters at +1.0 → τ should have only heave (+Z) component

test_saturation:
  u outside [-1, 1] saturated to ±1 before allocation

test_deadband:
  |u_i| < deadband → u_i_effective = 0 (per R22 nonlinearity requirement)

test_time_constant_first_order:
  u_command step → u_effective rises with τ_lag time-constant (R22)
  At t = τ_lag, u_eff ≈ 0.63·u_command

test_autograd_through_T:
  d(τ)/d(u) = T (constant matrix when in linear region); verify
```

**T matrix (8×6) for BlueROV2 Heavy**: from clydemcqueen `bluerov2_gz/models/bluerov2_heavy/model.sdf` thruster `<pose>` entries. Read-only inspection (no vendor — bluerov2_gz has no LICENSE).

**Commit**: `feat(hydro): T2.6 thruster allocation with saturation + deadband + time-constant`

---

### T2.7 — Full-pipeline autograd correctness

**TDD spec**:
```
test_full_pipeline_forward_consistent:
  Hand-compute wrench from random (η, ν, ν̇, u) using numpy reference;
  compare vs Warp pipeline output within 1e-4.

test_full_pipeline_autograd_vs_finite_diff:
  100 random states (η, ν, ν̇, u);
  for each: compute wp.Tape gradient of sum(wrench) wrt ν;
  compare against torch finite-diff (h = 1e-3) of same scalar wrt ν;
  max rel-err < 1e-3 across all 100 samples.

test_autograd_through_full_step:
  Build N=64 worlds, step solver under Tier-1, take final position;
  compute dP_final / dnu_initial via wp.Tape;
  verify gradient is finite (no NaN) and matches a hand-derived approximation.
```

**Critical R20 mitigation**: this test must be writable BEFORE we write the kernels. Write it now (in T2.1 effectively) as the spec for what each kernel must satisfy.

**Commit**: `test(hydro): T2.7 full-pipeline autograd correctness vs torch finite-diff`

---

### T2.8 — Performance test ≥ 100k env-steps/s @ 8192 envs

**Files**:
```
benchmarks/tier1_throughput.py
```

**Spec**: replicate(8192), Newton SolverSemiImplicit, attach Tier1 with all 6 kernels, 200 steps, report env-steps/s.

**Gate**: ≥ 100,000 env-steps/s (≥ 1/2200 of bare-Newton 221M number — leaves >99% budget for hydro overhead).

**If fails**:
1. Profile via `WARP_KERNEL_TIMING=1` env var
2. Identify hottest kernel
3. Try fusing kernels (single launch instead of 6 launches)
4. Last resort: revert to torch tensor implementation for hot path (deferred performance)

**Commit**: `bench(hydro): T2.8 Tier-1 full pipeline throughput at 8192 envs`

---

## 6. Failure modes + recovery (R20-R24 from RISKS.md)

| Failure | Recovery |
|---|---|
| **Warp autograd test fails** (R20) | Step 1: re-read `tests/test_warp_smoke.py::test_quad_drag_autograd_matches_analytic` for working pattern. Step 2: simplify kernel to scalar version, get gradient match, scale up. Step 3: file Warp GitHub issue if intractable; fall back to torch primary + Warp port at v0.2 (cost: throughput drops to ~10M env-steps/s instead of 100M, still meets target). |
| **Newton pre-step hook doesn't exist or doesn't work** | Use post-step `state.body_f` write between kernel and solver.step. Document. |
| **Numerical instability at dt=1/240** | Tighten to dt=1/500. Update STATUS.md. |
| **Throughput < 100k @ 8192** | Per T2.8 fallback plan. |
| **MarineGym coefficient values differ from clydemcqueen by 3×** | Cross-validate against von Benzon 2022 Simulink simulator (R23). Pick the published value with most-recent citation chain. |
| **Sign convention error caught in restoring test** | Document in `docs/math/fossen.md` errata; update unit tests; commit fix. |

---

## 7. Out of scope for /goal (will NOT do)

- Tier-2 SPH localized fluid (v0.3)
- Tier-3 Gerstner/Stable-Fluid waves (v0.3)
- Currents / disturbances → separate task in W3 (T3.5 or later)
- BlueROV2 asset porting → W3 (T3.*)
- Station-keeping env → W4 (T4.*)
- ONNX export → W4
- Isaac Lab integration → v0.2
- Sensors (sonar, camera, DVL) → v0.2

If `/goal` execution drifts into any of these, treat as scope leak and refocus.

---

## 8. Commit hygiene (8 commits expected)

```
feat(hydro): T2.1 Tier-1 skeleton + Newton pre-step force injection wiring
feat(hydro): T2.2 added-mass M_A Warp kernel with autograd
feat(hydro): T2.3 damping D(ν)ν Warp kernel with autograd
feat(hydro): T2.4 Coriolis C_RB + C_A Warp kernels with autograd
feat(hydro): T2.5 restoring g(η) Warp kernel with autograd
feat(hydro): T2.6 thruster allocation with saturation + deadband + time-constant
test(hydro): T2.7 full-pipeline autograd correctness vs torch finite-diff
bench(hydro): T2.8 Tier-1 full pipeline throughput at 8192 envs
```

Each commit: TDD test file added or expanded + minimal kernel impl + passing test demonstration in commit message. Use `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` footer per CLAUDE.md.

---

## 9. Final acceptance (single command, single output)

```bash
cd /home/robot/workspace/46-marine
git checkout feat/v0.1.0-tier1-fossen
.venv/bin/python -m pytest tests/hydro/ -v && \
.venv/bin/python benchmarks/tier1_throughput.py && \
.venv/bin/ruff check oceanscale tests && \
.venv/bin/mypy oceanscale/hydro/ --strict && \
echo "W2 PASS"
```

Output must contain `W2 PASS` on last line. Goal-driven loop checks for this string.

---

## 10. Quota / time budget

| Resource | Budget | Notes |
|---|---|---|
| Wall-clock | 12-20 hours focused work (R21 reward iteration N/A here, this is hydro only) | Per DR review, Warp autograd debug = 5 days budget if stuck |
| GPU memory | < 8 GB peak (Tier-1 pipeline at 8192 envs) | We have 31 GB free |
| Disk | < 1 GB new artifacts | Cache + test outputs |
| DR / WebSearch calls | 0 (planning done) | All references already collected |
| Risk activation | R20, R22 likely to trigger; R23 unlikely | Mitigations documented above |

---

## 11. Pre-flight checklist (run before `/goal`)

```bash
# branch check
git checkout feat/v0.1.0-worlds-poc
git pull --ff-only
git checkout -b feat/v0.1.0-tier1-fossen

# refs scratch clone
mkdir -p /tmp/oceanscale-refs && cd /tmp/oceanscale-refs
[ -d MarineGym ] || git clone --depth 1 https://github.com/Marine-RL/MarineGym
cd /home/robot/workspace/46-marine

# environment check
.venv/bin/python -c "import newton, warp; print(newton.__version__, warp.__version__)"
# expect: 1.2.0 1.13.0

# task list check
# all T2.* tasks pending (#23, 26, 27, 30, 32, 36, 41, 45)

# test baseline
.venv/bin/python -m pytest tests/ -v --tb=short
# expect: 22 passed (10 env + 4 warp + 4 newton + 4 worlds_poc)
```

---

## 12. Suggested /goal invocation (after pre-flight clean)

```
/goal W2 done: tests/hydro/ green (8 tests), tier1_throughput ≥100k env-steps/s @ 8192 envs, autograd rel-err < 1e-3, ruff+mypy clean, commits on feat/v0.1.0-tier1-fossen, all T2.1-T2.8 marked completed
```

The goal-driven loop will then:
1. Check pre-flight (§11)
2. Iterate T2.1 → T2.8 in order (some can parallelize: T2.2-T2.5 are independent)
3. Run §9 final acceptance every N iterations
4. Release when §9 prints `W2 PASS`
