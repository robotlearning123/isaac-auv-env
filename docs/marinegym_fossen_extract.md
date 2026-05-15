# MarineGym Source Extraction — Fossen Impl, BlueROV Coefs, Hover Task

**Source**: https://github.com/Marine-RL/MarineGym  
**Commit pinned**: `ebdca1bf` (2026-01-21)  
**License**: MIT (full LICENSE file, Copyright (c) 2025 Shuguang Chu, Zhejiang University)  
**Extracted**: 2026-05-15 (R1.4 of W1.5)  
**Read-only**: NO code vendored to oceanscale yet; this doc is preparation for T2.2-T2.6 + T3.3 + T4.1.

---

## Critical correction — what MarineGym actually ships

**Previously assumed (REFERENCES.md, docs/math/fossen.md)**: MarineGym ships BlueROV2 **Heavy** USD asset and hydro coefs.

**Actual finding (R1.4)**: MarineGym ships **BlueROV (basic, 6 rotors)** USD only:
- `marinegym/robots/assets/usd/BlueROV/BlueROV.usd` ✓ exists
- `marinegym/robots/assets/usd/BlueROVHeavy/` — **directory DOES NOT EXIST in the repo**
- `BlueROVHeavy.py:8` references `usd/BlueROVHeavy/BlueROVHeavy.usd` but the file is missing

→ The hydro coefs we quoted in `docs/math/fossen.md §5.2` as "BlueROV2 Heavy from MarineGym" are actually **BlueROV (basic) coefs**. Heavy variant is **NOT in MarineGym**.

→ **Plan correction for T3.3**: BlueROV2 Heavy parameters must come from **von Benzon 2022 (JMSE 10(12):1898)** or BlueRobotics datasheet, not MarineGym.

---

## §1 — Fossen implementation in `marinegym/robots/drone/underwaterVehicle.py`

15,603 bytes, 362 LOC. Class `UnderwaterVehicle(RobotBase)`. Uses PyTorch tensors on Isaac Sim PhysX bodies.

### Mapping Fossen terms → MarineGym methods → our Warp kernels

| Fossen term | MarineGym method | LOC range | Our kernel (T2.*) |
|---|---|---|---|
| M_RB · ν̇ | (delegated to Isaac Sim PhysX) | — | (delegated to Newton MJWarp) |
| **M_A · ν̇** | `calculate_added_mass()` | L250-254 | T2.2 `tier1_added_mass` |
| C_RB(ν) · ν | (delegated to Isaac Sim PhysX) | — | (delegated to Newton MJWarp) |
| **C_A(ν) · ν** | `calculate_corilis()` *(sic)* | L256-264 | T2.4 `tier1_coriolis_a` |
| **D(ν) · ν** | `calculate_damping()` | L238-248 | T2.3 `tier1_damping` |
| **g(η)** | `calculate_buoyancy()` | L266-277 | T2.5 `tier1_restoring` |
| τ thruster | `apply_action()` + RotorGroup + T200 | L172-201 | T2.6 `tier1_thruster_alloc` |
| ν̇ estimate | `calculate_acc()` | L229-236 | T2.1 wiring (`prev_body_vels` buffer) |
| Inject forces | `apply_hydrodynamic_forces()` → `base_link.apply_forces_and_torques_at_pos` | L203-227 + L195-199 | T2.1 `state.body_f` write |

### Key implementation details

**Damping has cross-coupling** (L239-244):
```python
maintained_body_vels = torch.diag_embed(body_vels)
maintained_body_vels[:, 1, 5] = body_vels[:, 5]   # sway-yaw coupling
maintained_body_vels[:, 2, 4] = body_vels[:, 4]   # heave-pitch coupling
maintained_body_vels[:, 4, 2] = body_vels[:, 2]   # pitch-heave (symmetric)
maintained_body_vels[:, 5, 1] = body_vels[:, 1]   # yaw-sway (symmetric)
damping_matrix = self.linear_damping_matrix + self.quadratic_damping_matrix * torch.abs(maintained_body_vels)
damping = damping_matrix @ body_vels.unsqueeze(2)
```
**T2.3 update needed**: our Warp kernel must include these 4 off-diagonal cross-coupling terms. Our `docs/math/fossen.md §5.4` had pure diagonal — patch.

**Added mass is purely diagonal** (L250-254): `added_mass_matrix @ body_acc`. Stored at L165 as `torch.diag(added_mass)`.

**Coriolis only computes C_A (added-mass)**, not C_RB (L256-264):
```python
ab = self.added_mass_matrix @ body_vels.unsqueeze(2)  # ab = M_A · ν
coriolis[:, 0:3] = -cross(ab[:, 0:3], body_vels[:, 3:6])              # F_lin = -ab_lin × ω
coriolis[:, 3:6] = -(cross(ab[:, 0:3], body_vels[:, 0:3]) +
                     cross(ab[:, 3:6], body_vels[:, 3:6]))            # M_ang = -(ab_lin × v + ab_ang × ω)
```
This is the **standard Fossen 2021 §6.3 cross-product form** with diagonal M_A simplification.

**Restoring uses RPY (Euler), not quaternion** (L266-277):
```python
buoyancyForce = 997 * 9.8 * volume   # ρ=997 (kg/m³), g=9.8 (NOT 9.81), V
dis = coBM                            # COG-COB distance
buoyancy[:, 0] =  bF · sin(pitch)
buoyancy[:, 1] = -bF · sin(roll) · cos(pitch)
buoyancy[:, 2] = -bF · cos(roll) · cos(pitch)
buoyancy[:, 3] = -dis·bF · cos(pitch) · sin(roll)   # roll moment
buoyancy[:, 4] = -dis·bF · sin(pitch)               # pitch moment
# buoyancy[:, 5] (yaw moment) = 0
```
**Note**: uses `g = 9.8` not `9.81`. **Note**: needs quaternion → RPY conversion. **T2.5 plan**: implement directly with quaternion to avoid singularities at ±90° pitch.

**Added-mass ν̇ estimation is filtered** (L229-236):
```python
alpha = 0.3
acc_raw = (body_vels - prev_body_vels) / dt
filteredAcc = (1 - alpha) * prev_body_acc + alpha * acc_raw
```
**T2.2 plan addition**: include `α = 0.3` low-pass filter on `ν̇` for stability (Fossen 2021 recommends similar smoothing).

**Frame correction sign flip** (L212-213, 222-223):
```python
body_vels[..., [1,2,4,5]] *= -1
body_rpy[..., [1,2]] *= -1
# ... compute hydro ...
hydro[:, [1, 2, 4, 5]] *= -1
buoyancy[:, [1, 2, 4, 5]] *= -1
```
This is Isaac Sim ↔ "marine NED" frame fix. **For Newton**: investigate at T2.1 whether Newton's z-up convention requires a similar fix. Likely yes for components [1, 2] (y, z signs).

### Anti-patterns spotted

1. **Print statement debug leak**: `print(self._view.dof_names)` at L86, `print(self._view._dof_indices)` at L87. Production code smell.
2. **Functional-call boilerplate**: `make_functional` + `vmap` + manual param expansion is verbose; modern `torch.func` API is cleaner.
3. **All computation single-precision implicit** (no `dtype=torch.float64` option) — fine for RL, but document.

---

## §2 — BlueROV (basic, NOT Heavy) hydro coefficients

Source: `marinegym/robots/assets/usd/BlueROV/BlueROV.yaml`.

```yaml
name: BlueROV
drag_coef: 0.3            # unused in Fossen path; legacy?
volume: 0.0113459          # m³ (= 11.35 L)
coBM: 0.01                 # m (COG-COB distance = 1 cm)
hydro_coef:
  added_mass:              # 6-vector for M_A diag (X_u̇, Y_v̇, Z_ẇ, K_ṗ, M_q̇, N_ṙ)
    [5.5, 12.7, 14.57, 0.12, 0.12, 0.12]
  linear_damping:          # 6-vector for D_lin diag
    [4.03, 6.22, 5.18, 0.07, 0.07, 0.07]
  quadratic_damping:       # 6-vector for D_quad diag (NOT cross-coupling values)
    [18.18, 21.66, 36.99, 1.55, 1.55, 1.55]
rotor_configuration:
  num_rotors:              6              # ← BlueROV basic has 6, BlueROV2 Heavy has 8
  directions:              [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
  time_constants:          [0.01] * 6
  force_constants:         [4.4e-07] * 6
  max_rotation_velocities: [3900] * 6
  moment_constants:        [1.3677728816219314e-09] * 6
```

### For BlueROV2 Heavy (8 thrusters), we must obtain coefs elsewhere

Sources to investigate at T3.3 (R23 mitigation):
1. **von Benzon 2022** [JMSE 10(12):1898](https://doi.org/10.3390/jmse10121898) — primary
2. **BlueRobotics datasheet** — geometry
3. **Wu 2018 Flinders MS thesis** — caveat-cite (no tank data)
4. **clydemcqueen/bluerov2_gz** SDF — geometry only, license blocker

---

## §3 — Hover task (= our station-keep template)

`marinegym/envs/single/hover.py` (260 LOC) + `cfg/task/Hover.yaml`.

### Task config (Hover.yaml)
```yaml
env:
  num_envs: 64               # NOT 8192; our v0.1 target is 8192
  env_spacing: 6             # m
  max_episode_length: 200    # steps (NOT seconds)
drone_model:
  name: BlueROV              # NOT BlueROV2 Heavy (see §1 above)
  controller: LeePositionController
reward_effort_weight: 0.1
reward_action_smoothness_weight: 0.0
reward_distance_scale: 1.2
time_encoding: true
```

### Initial state distributions (hover.py L57-66)
```python
init_pos_dist  = Uniform([-2.5, -2.5, 1.5],  [2.5, 2.5, 2.5])    # x, y, z in m
init_rpy_dist  = Uniform([-0.2π, -0.2π, 0],   [0.2π, 0.2π, 2π])  # roll, pitch, yaw
target_pos     = [0, 0, 2]                                       # m
target_rpy_dist = Uniform([0, 0, 0],          [0, 0, 2π])        # only yaw randomized
```

### What we use from this for T4.1 (StationKeepingEnv)
- **Initial pose distributions** — same ranges, mirror exactly
- **Target = [0, 0, 2]** — natural depth-hold task
- **Reward weights** (effort=0.1, distance_scale=1.2, action_smoothness=0.0) — initial values
- **Max episode length 200 steps** — at our 240 Hz × 0.1 dt = 20 s episode

### What we DON'T copy
- Their `Hover.__init__` ties into `IsaacEnv` base class — Newton-incompatible, reimplement
- `attach_payload` mechanism — payload DR happens via Newton mass modification
- `LeePositionController` — not needed; we go raw thruster control

---

## §4 — Domain randomization (R22 + R23 driver)

`cfg/task/randomization.yaml` + `cfg/task/disturbances.yaml`:

### Body parameter randomization (apply at reset)
| Parameter | Train scale | Evaluate scale | Map to our DR (T4.* + R22) |
|---|---|---|---|
| mass | [0.8, 1.2] | [0.8, 1.2] | ✓ mirror |
| volume | [0.9, 1.1] | [0.8, 1.2] | ✓ mirror |
| coBM (COG-COB cm) | [0.5, 1.5] | [0.5, 1.5] | ✓ mirror — KEY for restoring force DR |
| inertia | [0.8, 1.2] | [0.8, 1.2] | ✓ mirror |
| added_mass | [0.5, 1.0] | [0.5, 1.0] | ✓ note: ASYMMETRIC (scales down only) |
| linear_damping | [0.5, 1.0] | [0.5, 1.0] | ✓ note: ASYMMETRIC (less damping = harder task) |
| quadratic_damping | [0.5, 1.0] | [0.5, 1.0] | ✓ asymmetric |

### Rotor randomization
| Parameter | Scale | Map to T2.6 thruster |
|---|---|---|
| time_constants_scale | [0.8, 1.2] | direct — this is the τ_lag in our thruster model |
| force_constants_scale | [0.8, 1.2] | direct — thrust gain |

### Disturbance (current + payload)
```yaml
flow:
  max_flow_velocity: [0.5, 0.5, 0.5, 0, 0, 0]   # m/s in body frame xyz
  flow_velocity_gaussian_noise: [0.1, 0.1, 0.1, 0, 0, 0]
payload:
  mass: [0.01, 0.2]   # as scale of body mass
  z: [-0.1, 0.1]      # m offset
```

**Train vs evaluate**: enable_randomization is FALSE in train, TRUE in evaluate — they do "stationary training, randomized evaluation" pattern. This is **unusual** — most DR practice is randomize during training too. Verify if this is intentional or a config bug.

**For our v0.1 DR** (T4.* + R22):
- Mirror all 7 body randomization parameters
- Mirror the 2 rotor randomization parameters
- Add **history-conditioned policy** (Chaffre 2025 finding) — beyond MarineGym

---

## §5 — What we'll change vs MarineGym (when porting to Warp)

| MarineGym | Our oceanscale |
|---|---|
| PyTorch tensors on Isaac Sim PhysX | Warp kernels on Newton |
| `apply_forces_and_torques_at_pos` (PhysX API) | direct write to `state.body_f` (Newton) |
| Diagonal added-mass matrix | same — diagonal is the standard |
| Damping with 4 cross-coupling terms | **same** — include cross-coupling per L239-244 (currently NOT in our docs/math/fossen.md §5.4) |
| Coriolis C_A only (delegate C_RB to engine) | same — let MJWarp handle C_RB |
| Restoring via RPY (Euler) | use quaternion directly (avoid gimbal lock) |
| ν̇ filtered with α=0.3 EMA | **same** — add to T2.2 |
| Frame correction `[1,2,4,5] *= -1` | investigate at T2.1 — likely yes for Newton too |
| `g = 9.8` | use Newton's `g = 9.81` (consistent with our setup) |
| ρ_water = 997 (fresh water) | parameterize — default 1025 (salt), allow override to 997 (fresh) |
| 64 parallel envs default | 8192 parallel envs target via Newton replicate() |
| `num_rotors = 6` (basic BlueROV) | `num_rotors = 8` (BlueROV2 Heavy) |
| YAML config | YAML config (mirror their structure for compatibility) |
| TorchRL training loop | stable-baselines3 (v0.1) then skrl/rl_games (v0.2) |

---

## §6 — Updates needed to existing oceanscale docs

| File | Section | Update |
|---|---|---|
| `docs/math/fossen.md` | §5.2 added-mass | Note "values from MarineGym **BlueROV basic** (NOT Heavy); Heavy comes from von Benzon 2022 at T3.3" |
| `docs/math/fossen.md` | §5.4 damping | Add cross-coupling terms: `D[1,5] = D[5,1] = quad·\|r\|` etc. per MarineGym L239-244 |
| `docs/math/fossen.md` | §5.5 restoring | Add note: MarineGym uses RPY+9.8+997; we use quat+9.81+1025; document precision difference |
| `goals/W2_tier1_fossen.md` | T2.2 | Add α=0.3 EMA filter on ν̇ estimate |
| `goals/W2_tier1_fossen.md` | T2.3 | Update kernel to include 4 cross-coupling terms |
| `goals/W2_tier1_fossen.md` | T3.3 | Source: von Benzon 2022, NOT MarineGym (Heavy missing) |
| `REFERENCES.md` §1 master matrix | REF-MARINEGYM | Update "ships BlueROV2 Heavy USD" claim to "ships basic BlueROV USD" |

---

## Citation (BibTeX)

```bibtex
@inproceedings{chu2025marinegym,
  title={MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics},
  author={Chu, Shuguang and Huang, Hao and Li, Yongqi and Lin, Yuan and Li, Daoqi and Carlucho, Ignacio and Petillot, Yvan and Yang, Yu},
  booktitle={IROS},
  year={2025},
  eprint={2503.09203},
  url={https://github.com/Marine-RL/MarineGym},
  note={MIT, commit ebdca1bf, accessed 2026-05-15}
}
```
