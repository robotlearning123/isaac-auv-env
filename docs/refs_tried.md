# Reference Projects: Run/Read Tried (W1.5)

**Date started**: 2026-05-15  
**Branch**: `feat/v0.1.0-run-references`  
**Goal**: install/run/read 5 reference simulators on RTX 5090 + Python 3.12 host, capture real numbers + patterns + gotchas.

**Per-row format**:
```
status:       RAN | READ | FAILED | PARTIAL
license:      verified SPDX or NOASSERTION
install_time: minutes
runs_at:      single-env FPS, or N parallel envs FPS
key_files:    paths inspected
reusable_pattern: ...
anti_pattern / gotcha: ...
citation: URL + commit SHA + date tested
```

---

## REF-BLUEROV2GYM — `gokulp01/bluerov2_gym`

- **status:** RAN
- **commit tested:** `fd0ee80` (2025-04-02)
- **url:** https://github.com/gokulp01/bluerov2_gym
- **license:** **MIT declared in `pyproject.toml`** (`license = {text = "MIT"}`); **NO `LICENSE` file in repo**. SPDX status: MIT-declared, license-text-missing. → Earlier `REFERENCES.md` §5 anti-reference status updated: cite + study OK; vendor with caveat.
- **install_time:** 4 m 12 s (uv pip install -e .); resolved 84 packages, installed 171 in venv.
- **runs_at:** **11,435 FPS single-env, CPU, no rendering** (5000 steps, 0.437 s, Python 3.12 venv, AMD Ryzen / Intel i9-14900K class)
- **assets:** `bluerov2_gym/assets/BlueRov2.dae` (Collada mesh, MIT-declared)
- **action_space:** `Box(low=-1.0, high=1.0, shape=(4,), dtype=float32)` — **4-D control**, not 8-thruster
- **obs_space:** **Dict(8 keys)**: x/y/z/theta + vx/vy/vz/omega, each `Box((1,), float32)`
- **reward:** `-(1.0·||pos||₂ + 0.1·||vel||₂ + 0.5·|θ|)` (rewards.py, 26 LOC)
- **termination:** `|z| > 10` or `|x| > 15` or `|y| > 15`
- **dt:** 0.1 s (= 10 Hz; way slower than HoloOcean's 240 Hz)

### key_files_inspected
- `bluerov2_gym/envs/core/dynamics.py` (102 LOC) — 4-DOF planar dynamics, system-identified compound coefficients (a_vx, a_vx2, a_vyw, b_vx, ...) not Fossen 6-DOF
- `bluerov2_gym/envs/core/rewards.py` (26 LOC) — station-keep reward
- `bluerov2_gym/envs/bluerov_env.py` (96 LOC) — gym wrapper
- `pyproject.toml` — license + deps

### critical finding (not what we assumed)
**bluerov2_gym is NOT a Fossen 6-DOF reference.** It implements a simplified **4-DOF planar model** (surge/sway/heave/yaw, no roll/pitch, no quaternion) with **identified compound coefficients** (not decomposed M_RB/M_A/C/D matrices). The disturbance Gaussian (mean + 4×4 covariance) appears to come from real BlueROV2 tank-trace fitting.

→ **Update REFERENCES.md**: anti-reference label was technically wrong; reframe as "4-DOF identified-coefficient baseline, MIT-declared, study-only for reward shaping + disturbance modeling".

### reusable_pattern
1. **Disturbance Gaussian sampler** (`scipy.stats.multivariate_normal` with pre-computed mean + cov) — direct candidate for v0.1 DR toolkit
2. **Reward weighting** (1.0 / 0.1 / 0.5 = pos_err / vel_pen / orient_err) — useful initial values for our T4.1 StationKeepingEnv
3. **Termination thresholds** (10 m vertical, 15 m horizontal) — sane defaults for AUV station-keep
4. **`render_mode=None` for headless** — clean gym convention; we should match

### anti_pattern / gotcha
1. **Dict obs space** — slow vs `Box(8,)`; for 8k parallel envs use a packed tensor
2. **dt = 0.1 (10 Hz)** — too coarse for accurate AUV simulation; HoloOcean uses 240 Hz, MarineGym uses 100+ Hz
3. **`print(action)` debug statement in `dynamics.py:46`** — production code smell; spams stdout
4. **Meshcat viewer starts even with `render_mode=None`** — daemon leak at 127.0.0.1:7000
5. **Single-env design** — `self.state` is a dict, not a tensor; **cannot batch to 8192** without rewrite
6. **Numpy + scipy CPU only** — no GPU path; no autograd
7. **License text missing** — MIT declared but file absent; vendor only with caveat (declare + cite)

### what we DON'T use from this
- Their 4-DOF dynamics math (we need 6-DOF Fossen)
- Their `dt=0.1` timestep
- Their per-step dict state
- Anything from this codebase directly into our `oceanscale/` (cite + reference reward design only)

### what we DO use
- Their **reward weights** as initial values for T4.1
- Their **disturbance Gaussian pattern** as v0.1 DR baseline
- Their **termination thresholds** as default for `StationKeepingEnv`
- Their **`gym.make()` API surface** as our gymnasium wrapper template

### citation
```bibtex
@misc{bluerov2gym2025,
  title={BlueROV2 Gymnasium Environment},
  author={gokulp01 and contributors},
  url={https://github.com/gokulp01/bluerov2_gym},
  note={MIT-declared in pyproject.toml, no LICENSE file. Commit fd0ee80, 2025-04-02.},
  year={2025}
}
```

---

## REF-MARINEGYM — `Marine-RL/MarineGym`

- **status:** READ (per plan: Isaac Sim 4.1 stack too stale to install on RTX 5090 Blackwell)
- **commit pinned:** `ebdca1bf` (2026-01-21)
- **url:** https://github.com/Marine-RL/MarineGym
- **license:** **MIT (full LICENSE file)**, Copyright (c) 2025 Shuguang Chu, Zhejiang University ✓
- **install_attempted:** NO — read-only per plan §5 R1.4
- **runs_at:** N/A (paper reports 250k FPS @ 8000 envs on RTX 3060; not verified locally)

### what they actually ship
- ✓ `marinegym/robots/assets/usd/BlueROV/BlueROV.usd` (basic BlueROV, **6 rotors**)
- ❌ `marinegym/robots/assets/usd/BlueROVHeavy/` — **directory DOES NOT EXIST** despite `BlueROVHeavy.py:8` referencing it. **Critical correction** vs prior REFERENCES.md claims.

### key_files_inspected (778 LOC total across 8 files)
| File | LOC | Purpose |
|---|---|---|
| `marinegym/robots/drone/underwaterVehicle.py` | 361 | Full Fossen 6-DOF impl in PyTorch tensors |
| `marinegym/robots/drone/BlueROV.py` | 9 | Points to BlueROV.usd + yaml |
| `marinegym/robots/drone/BlueROVHeavy.py` | 9 | Points to MISSING BlueROVHeavy.usd |
| `marinegym/envs/single/hover.py` | 260 | Station-keep env design |
| `cfg/task/Hover.yaml` | 25 | Task config |
| `cfg/task/randomization.yaml` | 30 | DR scale ranges |
| `cfg/task/disturbances.yaml` | 20 | Flow + payload disturbances |
| `marinegym/robots/assets/usd/BlueROV/BlueROV.yaml` | 64 | Hydro coefs |

### extraction → `docs/marinegym_fossen_extract.md` (~12 KB)
Per-Fossen-term method mapping (calculate_added_mass → T2.2, etc.),  
BlueROV (basic) numerical hydro coefs,  
Hover task obs/action/reward spec,  
9 DR parameter ranges,  
5+ implementation gotchas.

### reusable_pattern
1. **Fossen impl structure** — 6 functions per term, M_RB/C_RB delegated to physics engine
2. **YAML hydro coef schema** (`added_mass`, `linear_damping`, `quadratic_damping` 6-vectors + volume + coBM) — mirror exactly
3. **DR ranges** (9 body + 2 rotor params) — directly applicable to T4.*
4. **α=0.3 EMA filter on ν̇** — stability trick we missed in our T2.2 plan
5. **Damping cross-coupling** — 4 off-diagonal terms (sway-yaw, heave-pitch); our `docs/math/fossen.md §5.4` had pure diagonal → patch
6. **Coriolis = C_A only** — let physics engine handle C_RB
7. **Hover initial pose distributions** — Uniform([-2.5, 2.5]) xy, [1.5, 2.5] z; RPY ±0.2π; yaw 0-2π

### anti_pattern / gotcha
1. **BlueROV2 Heavy USD missing** despite Python class referencing it
2. **`print(view.dof_names)` debug at L86-87** — production code smell
3. **g=9.8, ρ=997 (fresh water)** — we use g=9.81, ρ=1025 (salt); be explicit
4. **DR disabled in TRAIN by default**, enabled in EVALUATE — backwards from common practice
5. **RPY-based restoring** — uses Euler; we'll do quaternion (gimbal-lock safe)
6. **Frame correction `[1,2,4,5] *= -1`** — Isaac Sim ↔ marine NED axis fix
7. **TorchRL trainer** — different stack from our stable-baselines3 v0.1 plan
8. **functorch + make_functional** — older PyTorch idiom; `torch.func` is cleaner

### what we DO use
- **Fossen impl as porting source** for T2.2-T2.5 (torch → Warp; same equations)
- **Hydro coef YAML schema** (mirror exactly, fill with our Heavy values from von Benzon 2022)
- **DR ranges** for T4.* DR toolkit
- **Hover task obs/action/reward** as T4.1 baseline
- **Damping cross-coupling pattern** (update `docs/math/fossen.md §5.4`)
- **α=0.3 EMA filter** (update `goals/W2_tier1_fossen.md T2.2`)

### what we DON'T use
- BlueROV basic (6-rotor) parameters — we need Heavy (8-rotor)
- Isaac Sim 4.1 / PhysX-specific force injection API
- TorchRL trainer
- LeePositionController
- `print()` debug

### updates required (T1.4 patch list)
- `docs/math/fossen.md §5.2`: "MarineGym coefs are BlueROV basic, not Heavy"
- `docs/math/fossen.md §5.4`: add 4 damping cross-coupling terms
- `goals/W2_tier1_fossen.md T2.2`: add α=0.3 EMA filter
- `goals/W2_tier1_fossen.md T2.3`: kernel must include cross-coupling
- `goals/W2_tier1_fossen.md T3.3`: source = von Benzon 2022, NOT MarineGym
- `REFERENCES.md §1`: REF-MARINEGYM "ships Heavy USD" → "ships BlueROV basic only"

### citation
See `docs/marinegym_fossen_extract.md` for full BibTeX (Chu et al. 2025 IROS).

---
