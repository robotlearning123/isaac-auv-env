# 6-DOF Fossen Hydrodynamics — Math Sketch (T1.3)

**Date:** 2026-05-15  
**Status:** Reference for T2.1-T2.7 Warp-kernel implementation.  
**Source:** Fossen 2021 (2nd ed.) §3-8, §10; MarineGym `underwaterVehicle.py` for variable naming; HoloOcean 2.0 paper for BlueROV2 Heavy regime.

This file defines the equations, sign conventions, and unit/frame choices we implement. Mapping to our Warp kernels is in §8 below. **This document is normative for v0.1.0.**

---

## 1. Frames and axis convention

Two frames:

- **`{n}` north-east-down (NED)** inertial / world frame. Z **down**. Origin: world origin of the Newton scene.
- **`{b}` body** frame, attached to the AUV CG. x-forward (surge), y-starboard (sway), z-down (heave).

> **Sanity check vs Newton/MJC**: Newton defaults to **Z-up** with gravity `(0, 0, -9.81)`. We will rotate body frame so heave maps to `-z` of world. Specifically: when we say "the AUV heaves downward", body-frame +z aligns with world-frame `-z`. Document this in `oceanscale/core/frames.py` and verify in unit tests.

Rotation `{n}` → `{b}`: unit quaternion `q = (x, y, z, w)`.

---

## 2. State variables (Fossen 2021 §2.2)

| Symbol | Shape | Meaning | Units |
|---|---|---|---|
| `η` | (7,) world-frame pose | position `p_n = (x, y, z)` + attitude `q = (qx, qy, qz, qw)` | m, dimensionless |
| `ν` | (6,) body-frame twist | linear `v_b = (u, v, w)` + angular `ω_b = (p, q, r)` | m/s, rad/s |
| `ν̇` | (6,) body-frame accel | `(u̇, v̇, ẇ, ṗ, q̇, ṙ)` | m/s², rad/s² |
| `τ` | (6,) body-frame wrench | force `F_b = (X, Y, Z)` + torque `M_b = (K, M, N)` | N, N·m |

**Layout in Warp:** `wp.spatial_vectorf` = `(linear_xyz, angular_xyz)` per empirical verification in STATUS.md. So `ν[0:3] = v_b`, `ν[3:6] = ω_b`. Wrenches `τ` use the same layout. **Important** — Fossen books write `ν = [u, v, w, p, q, r]ᵀ` which is the same ordering; matches Warp natively.

### Why body-frame `ν` (not world-frame velocity)?

Hydrodynamic forces are functions of velocity **relative to the surrounding fluid**, which is most natural in the body frame. Currents are added as `ν_r = ν - J(η)⁻¹ ν_c`, where `ν_c` is the current velocity expressed in body frame via the rotation matrix `R(q)`.

---

## 3. Kinematic relation (Fossen §2.4)

```
η̇ = J(η) · ν                                              (1)
```

where `J(η)` is the 7×6 kinematics-transform Jacobian (mixed position-quaternion form):

```
J(η) = [ R(q)        0        ]                            (2)
       [ 0    0.5·Q(q)        ]
```

- `R(q)` (3×3): body-to-world rotation matrix from quaternion `q`.
- `Q(q)` (4×3): quaternion-rate matrix s.t. `q̇ = 0.5 · Q(q) · ω_b`.

For our v0.1, **Newton handles the kinematic integration internally** via its semi-implicit solver — we don't write `J(η)`. We compute forces in body frame, Newton integrates `ν → η`.

---

## 4. Equation of motion (Fossen §6.7)

The full 6-DOF rigid-body + hydrodynamic equation of motion in body frame:

```
M · ν̇  +  C(ν) · ν  +  D(ν) · ν  +  g(η)  =  τ_thrust + τ_disturbance     (3)
```

where the lumped matrices decompose as:

```
M     =  M_RB  +  M_A                          (4)   total inertia (6×6)
C(ν)  =  C_RB(ν) + C_A(ν)                       (5)   total Coriolis (6×6)
D(ν)  =  D_lin + D_quad(ν)                      (6)   total damping (6×6)
g(η)  =  buoyancy + weight terms in body frame  (7)   restoring (6,)
τ_thrust = T · u_thrusters                      (8)   from actuators (6,)
```

**Our Tier-1 kernels** (§8 below) implement each addend separately on the GPU.

---

## 5. Term-by-term definitions

### 5.1 M_RB — rigid-body mass matrix (Fossen §3.3.1)

```
M_RB = [ m·I₃     -m·S(r_g_b)  ]                            (9)
       [ m·S(r_g_b)    I_g     ]
```

- `m`: scalar mass (kg). BlueROV2 Heavy: **m ≈ 13.6 kg** (BlueRobotics datasheet).
- `r_g_b ∈ ℝ³`: vector from body origin to center of gravity in `{b}`. Often `r_g_b = 0` if body origin is at CG.
- `I_g ∈ ℝ^{3×3}`: inertia tensor about CG.
- `S(·)`: skew-symmetric matrix operator: `S(a) = [[0,-a3,a2],[a3,0,-a1],[-a2,a1,0]]`.

For BlueROV2 Heavy, body origin = CG, so `r_g_b = 0` and `M_RB = diag(m, m, m, Ixx, Iyy, Izz)` (approximately diagonal, off-diagonal Ixy/Ixz/Iyz negligible per clydemcqueen SDF: Ixx=0.26, Iyy=0.23, Izz=0.37 kg·m²).

**Newton already builds `M_RB`** from `add_body(mass=m)` + shape inertia. We do NOT compute `M_RB` ourselves; it's in `model.body_inv_mass` already.

### 5.2 M_A — added-mass matrix (Fossen §6.2)

The fluid is accelerated when the body accelerates; the reactive force is parameterized by `M_A`:

```
F_A = -M_A · ν̇                                            (10)
```

**6×6 diagonal approximation** (standard for AUVs of BlueROV2 form):

```
M_A = -diag( X_u̇, Y_v̇, Z_ẇ, K_ṗ, M_q̇, N_ṙ )                (11)
```

(Note: signs follow Fossen — coefficients X_u̇ etc. are negative, so `M_A` entries are positive added-mass values.)

Source values (BlueROV2 Heavy, from MarineGym `BlueROVHeavy.py`):

| Coeff | Value | Units |
|---|---|---|
| X_u̇  | -5.5 | kg |
| Y_v̇  | -12.7 | kg |
| Z_ẇ  | -14.57 | kg |
| K_ṗ  | -0.12 | kg·m² |
| M_q̇  | -0.12 | kg·m² |
| N_ṙ  | -0.12 | kg·m² |

(These values are **to be verified against Wu 2018** at T3.3.)

### 5.3 C_RB(ν), C_A(ν) — Coriolis-centripetal (Fossen §3.3.2, §6.3)

Both have the same skew-block structure:

```
C(ν) = [        0           -S(M_11·v_1 + M_12·v_2) ]      (12)
       [ -S(M_11·v_1 + M_12·v_2)   -S(M_21·v_1 + M_22·v_2) ]
```

where `M_ij` are the 3×3 blocks of `M_RB` or `M_A`, and `v_1 = v_b`, `v_2 = ω_b`.

For our **diagonal M_A** simplification with `r_g_b = 0`:

```
C_RB(ν) = [ 0_3          -m·S(v_b) ]                       (13a)
          [ -m·S(v_b)    -S(I_g·ω_b) ]

C_A(ν)  = [ 0_3          -S(M_A_11 · v_b) ]                (13b)
          [ -S(M_A_11·v_b)  -S(M_A_22 · ω_b) ]
```

These are bilinear in `ν` and must be re-evaluated every step.

### 5.4 D(ν) — damping (Fossen §6.4)

Decomposes into linear + quadratic drag:

```
D(ν) · ν = D_lin · ν + D_quad(ν) · ν                       (14)

D_lin = diag( X_u, Y_v, Z_w, K_p, M_q, N_r )               (15)

D_quad(ν) = diag( X_{u|u|}·|u|, Y_{v|v|}·|v|, Z_{w|w|}·|w|,
                  K_{p|p|}·|p|, M_{q|q|}·|q|, N_{r|r|}·|r| )  (16)
```

Source values (BlueROV2 Heavy, MarineGym + clydemcqueen):

| Coeff | Value | Notes |
|---|---|---|
| X_u            | -4.03  | linear surge |
| X_{u\|u\|}     | -18.18 | quadratic surge (or -58.42 per clydemcqueen — discrepancy to resolve T3.3) |
| Y_v            | -6.22  | linear sway |
| Y_{v\|v\|}     | -21.66 | quadratic sway |
| Z_w            | -5.18  | linear heave |
| Z_{w\|w\|}     | -36.99 | quadratic heave (or -124.82 per clydemcqueen) |
| K_p            | -0.07  | linear roll rate |
| K_{p\|p\|}     | -1.55  | quadratic roll rate |
| M_q, M_{q\|q\|}| similar | pitch rate |
| N_r, N_{r\|r\|}| similar | yaw rate |

> **Discrepancy alert**: MarineGym and clydemcqueen quadratic drag coefficients differ by ~3×. This is **not** unit conversion (both are SI). Resolution at T3.3 by cross-checking against Wu 2018 published values.

### 5.5 g(η) — restoring force (Fossen §6.5)

Buoyancy and weight, expressed in body frame:

```
g(η) = [ R(q)ᵀ · (W - B) · ẑ_n              ]              (17)
       [ r_g_b × R(q)ᵀ · W · ẑ_n  -  r_b_b × R(q)ᵀ · B · ẑ_n ]
```

where:
- `W = m · g` (weight, scalar, N)
- `B = ρ · ∇ · g` (buoyancy, scalar, N) with `ρ ≈ 1025 kg/m³` salt water, `∇` = displaced volume (m³)
- `ẑ_n`: world-frame down unit vector
- `r_g_b`: COG offset (body frame), often zero
- `r_b_b`: **COB offset** (body frame). For BlueROV2 Heavy, COB is **above** COG (typically z = -0.011 m in NED).

For near-neutral-buoyancy BlueROV2 with `W ≈ B`, the force balance is near zero but the **moment** `r_b_b × B · ẑ_n` is non-zero and provides the restoring torque (rights the vehicle when tilted). This is what gives the BlueROV2 its passive stability.

### 5.6 τ_thrust — thruster wrench (Fossen §10.4)

```
τ_thrust = T · u                                           (18)
```

- `u ∈ ℝ^8`: per-thruster command, each `u_i ∈ [-u_max, u_max]` newtons (or `u_i ∈ [-1, 1]` normalized, with saturation/deadband mapping).
- `T ∈ ℝ^{6×8}`: **thruster allocation matrix** mapping per-thruster forces to body-frame wrench. Each column is `(f_i_b, r_i_b × f_i_b)` where `f_i_b` is the unit thrust direction in body frame and `r_i_b` is the thruster position relative to body origin.

BlueROV2 Heavy: 8 vectored thrusters — 4 vectored at 45° in the horizontal plane for surge/sway/yaw, 4 vertical for heave/roll/pitch. Exact `T` from clydemcqueen SDF (publicly-available geometry).

### 5.7 Currents and disturbances

Modeled as relative-velocity perturbation:

```
ν_r = ν - R(q)ᵀ · v_c                                       (19)
```

where `v_c ∈ ℝ³` is current velocity in world frame. All hydrodynamic terms (`M_A·ν̇`, `C_A·ν`, `D·ν`) use `ν_r` instead of `ν`.

For v0.1.0: **constant uniform current**, randomized per env at reset. Time-varying spatial currents = v0.3 with Tier-3 surface waves.

---

## 6. Discretization (semi-implicit Euler)

Newton's SolverSemiImplicit handles this for us. Conceptually:

```
ν[k+1] = ν[k] + dt · M⁻¹ · ( τ - C(ν[k]) · ν[k] - D(ν[k]) · ν[k] - g(η[k]) )    (20)
η[k+1] = η[k] + dt · J(η[k]) · ν[k+1]                                            (21)
```

For BlueROV2 Heavy at 240 Hz: `dt = 1/240 ≈ 4.17 ms`. Stability for full Fossen at this rate is well-established (HoloOcean 2.0 uses same rate).

For our Warp force-injection: we compute the wrench `τ_total = τ_thrust - C·ν - D·ν - g`, write it to `state.body_f`, and the solver does the `M⁻¹ · τ_total` step.

> **M_A is the tricky one**: Newton's `M = M_RB` only. `M_A · ν̇` is the **reactive** force from fluid; we add it to the wrench using the **PREVIOUS-step** `ν̇`. This is the explicit added-mass scheme (per MarineGym). The implicit scheme would require modifying Newton's mass matrix, which is non-trivial — defer to v0.3 if explicit proves unstable.

---

## 7. Sign conventions (one place to look up)

| Quantity | Positive direction |
|---|---|
| Body-frame surge `u` | forward (nose direction) |
| Body-frame sway `v` | starboard (right when looking from above) |
| Body-frame heave `w` | down |
| Body-frame roll `p` | right hand around +x |
| Body-frame pitch `q` | right hand around +y |
| Body-frame yaw `r` | right hand around +z (turning right viewed from above) |
| World-frame depth `z` | down (NED) |
| Buoyancy `B` | force pointing up (so `B · ẑ_n` in NED actually means upward in world) |

> **Cross-check at T2.5**: gravity-only test should give negative `u_b` … no wait, depends on orientation. Just unit-test: drop from rest, body should accumulate `w_b > 0` (downward heave) in NED. In Newton's Z-up world, this means body-frame +z aligned with world -z, gravity pulls body in body-frame +z direction. **Write this verification explicitly in `tests/hydro/test_signs.py`** at T2.5.

---

## 8. Mapping to our Warp kernels (T2.1 - T2.6)

| Term | Kernel | T-task | LOC est | Sourced from |
|---|---|---|---|---|
| `M_RB · ν̇` | (Newton built-in) | — | 0 | `model.body_inv_mass` |
| `M_A · ν̇` | `oceanscale.hydro.tier1_kernels.added_mass` | T2.2 | ~25 | MarineGym `calculate_added_mass()` |
| `C_RB(ν) · ν` | `oceanscale.hydro.tier1_kernels.coriolis_rb` | T2.4 | ~30 | MarineGym `calculate_corilis()` |
| `C_A(ν) · ν` | `oceanscale.hydro.tier1_kernels.coriolis_a` | T2.4 | ~30 | MarineGym `calculate_corilis()` |
| `D(ν) · ν` | `oceanscale.hydro.tier1_kernels.damping` | T2.3 | ~25 | MarineGym `calculate_damping()` |
| `g(η)` | `oceanscale.hydro.tier1_kernels.restoring` | T2.5 | ~30 | MarineGym `calculate_buoyancy()` |
| `T · u` | `oceanscale.hydro.tier1_kernels.thruster_alloc` | T2.6 | ~20 | clydemcqueen SDF thruster geometry |
| Force inject | `oceanscale.hydro.injector.pre_step` | T2.1 | ~40 | wires the kernels into Newton |

All accumulate into a per-env `wp.spatial_vectorf wrench_buffer`, then a single `wp.launch` copies to `state.body_f` before `solver.step`.

---

## 9. Unit-test plan (T2.7)

`tests/hydro/test_tier1_*.py`:

| Test | Pass criterion |
|---|---|
| `added_mass_diag_only` | `ν̇ = [1, 0, 0, 0, 0, 0]` → wrench `≈ X_u̇` only in surge |
| `damping_at_unit_velocity` | each axis unit velocity gives `(X_u + X_{u\|u\|}) · 1` |
| `coriolis_yaw_centripetal` | `r=1, u=1, others=0` → side force `−m·u·r` (correct sign) |
| `restoring_neutral` | `q = identity, W = B` → wrench ≈ 0 (`< 1e-3 N`) |
| `restoring_tilt` | tilt 30° about x-axis → restoring torque points toward upright |
| `thruster_zero` | all `u = 0` → `τ = 0` |
| `thruster_symmetric_surge` | all 4 horiz thrusters +1 → only X component |
| `autograd_full` | wp.Tape gradient through full pipeline vs torch numerical-diff at 100 random samples; max rel-err < 1e-3 |

---

## 10. What this doc does NOT cover

- **CFD-level effects**: vortex shedding, thruster wash, hull boundary layer. These are residuals; our error budget assumes they're <10% at BlueROV2 station-keep speeds (0-2 m/s). To verify against HoloOcean 2.0 / Tunçay 2025 numbers at T2.8.
- **Free-surface waves**: deferred to v0.3.
- **Soft-body coupling** (tether, cable, manipulator): v0.4.
- **Actuator dynamics** (thruster lag, ESC ramp): linear first-order lumped into `u → u_eff` is enough for v0.1; full thruster model at v0.2.

---

## References

- Fossen, T. I. (2021). *Handbook of Marine Craft Hydrodynamics and Motion Control* (2nd ed.). Wiley. ISBN 978-1119575054.
- Potokar et al. (2025). *A Preview of HoloOcean 2.0*. arXiv:2510.06160.
- Chu, S. et al. (2025). *MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics*. arXiv:2503.09203.
- BlueRobotics. *BlueROV2 Heavy Configuration Datasheet*. https://bluerobotics.com/store/rov/bluerov2/

To be verified at T3.3:
- Wu, T. (2018). *Modeling and Control of BlueROV2 Heavy* — exact citation needed.
- von Benzon et al. (2022). *An Open-Source Plug-and-Play BlueROV2 Modeling Platform*. (Frontiers in Robotics?)

