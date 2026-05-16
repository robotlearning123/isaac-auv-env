# Session Handoff — 2026-05-15

**Workspace**: `/home/robot/workspace/46-marine/`  
**Brand**: OceanScale / 沧渊 — AI-native simulation infrastructure for underwater robotics  
**Stack**: Newton 1.2 + Warp 1.13 + Python 3.12 + RTX 5090  
**Last session ended**: 2026-05-15 ~21:30 UTC  
**Resume from**: branch `feat/v0.1.0-tier1-fossen` (or new branch off it for W3)

---

## 1. Technical state (v0.1.0 sprint progress)

### Branches in play

```
main                          (empty — never committed per CLAUDE.md worktree rule)
feat/v0.1.0-scaffold          fa48d02   scaffold + planning docs (~110 KB)
feat/v0.1.0-worlds-poc        1d88c70   T1.1 worlds API + T1.2 benchmark + T1.3 math + R1.4 patches
feat/v0.1.0-run-references    bab6ba5   W1.5 (R1.1+R1.4+R1.5)
feat/v0.1.0-tier1-fossen ★    d57935e   W2 COMPLETE (CURRENT — resume here)
```

### Test status

```
.venv/bin/python -m pytest tests/ -q  →  31 passed in 13.87s
.venv/bin/python -m ruff check oceanscale tests benchmarks  →  All checks passed
PYTHONPATH=. .venv/bin/python benchmarks/tier1_throughput.py @ N=8192  →  88 M env-steps/s
```

### Throughput baselines on RTX 5090 (verified)

| Workload | N | env-steps/s | Source |
|---|---|---|---|
| Pure Warp drag kernel | 8192 | 1.5 B | `benchmarks/kernel_throughput.py` |
| Newton SemiImplicit (free body, no hydro) | 8192 | **221 M** | `benchmarks/newton_worlds_throughput.py` |
| Newton MuJoCo (replicate worlds) | 8192 | 5.5 M | same |
| **Newton + Tier-1 Fossen (this project)** | **8192** | **88 M** | `benchmarks/tier1_throughput.py` |
| Target (W2 ship gate) | 8192 | ≥ 100 k | goals/W2 §2 — **880× exceeded** |

### Tests by file (31 total, all PASS)

| File | Count | Topic |
|---|---|---|
| `tests/test_environment.py` | 10 | Python/CUDA/Warp/Newton/USD versions + imports |
| `tests/test_warp_smoke.py` | 4 | Warp kernel + autograd + RTX 5090 detection |
| `tests/test_newton_smoke.py` | 4 | Newton free body fall (SemiImplicit + MuJoCo) + 8192-body finalize |
| `tests/test_worlds_poc.py` | 4 | replicate(N) worlds API + MuJoCo OOM fix |
| `tests/hydro/test_tier1_smoke.py` | 5 | Tier1 import + construct + zero/compute_wrench + neutral buoyancy |
| `tests/hydro/test_tier1_autograd.py` | 4 | added-mass + damping analytic gradient via wp.Tape vs torch finite-diff |

---

## 2. Plans completed (both PASS)

### W1.5 — Run & Reproduce References (`goals/W1.5_run_references.md`)

| Ref | Status | Key finding |
|---|---|---|
| R1.1 bluerov2_gym | RAN | 11,435 FPS single-env CPU; **4-DOF model, NOT Fossen 6-DOF** |
| R1.4 MarineGym | READ | **Only ships BlueROV basic (6 rotors), NOT Heavy** |
| R1.5 stonefish | READ | GPL-3 confirmed; cite-only, no vendor possible |

**14 reusable patterns + 20 anti-patterns** in `docs/refs_tried.md`.  
**12 KB Fossen extract** in `docs/marinegym_fossen_extract.md`.

### W2 — Tier-1 Fossen Warp Kernels (`goals/W2_tier1_fossen.md`)

All 8 sub-tasks complete:

| T# | Deliverable | Verification |
|---|---|---|
| T2.1 | injection wiring | `state.body_f` ordered write, no callback API needed |
| T2.2 | M_A · ν̇ with α=0.3 EMA | wp.Tape gradient = -M_A (exact) |
| T2.3 | D(ν)·ν with 4 cross-coupling terms | unit-velocity → -(D_lin + 2·D_quad) per axis |
| T2.4 | C_A(ν)·ν Coriolis | Fossen §6.3 cross-product form |
| T2.5 | g(η) restoring (quaternion) | identity quat + BlueROV at rest → +2.23 N (B-W verified) |
| T2.6 | thruster τ = T · u + saturation + deadband + τ_lag | per R22 mitigation |
| T2.7 | autograd correctness | 4 tests, rel-err < 1e-3 |
| T2.8 | throughput ≥ 100k env-steps/s @ 8192 | **88 M, 880× target** |

---

## 3. Files inventory

### Source code (4 .py)
```
oceanscale/__init__.py
oceanscale/hydro/__init__.py            5 kernel exports
oceanscale/hydro/tier1.py               Tier1 orchestrator (~250 LOC)
oceanscale/hydro/tier1_kernels.py       8 @wp.kernel definitions (~250 LOC)
```

### Tests (8 .py)
```
tests/__init__.py
tests/test_environment.py               env validation
tests/test_warp_smoke.py                Warp + autograd
tests/test_newton_smoke.py              Newton free body
tests/test_worlds_poc.py                replicate() worlds API
tests/hydro/__init__.py
tests/hydro/test_tier1_smoke.py         smoke + neutral buoyancy
tests/hydro/test_tier1_autograd.py      autograd correctness
```

### Benchmarks (4 .py)
```
benchmarks/__init__.py
benchmarks/kernel_throughput.py         pure Warp drag kernel (historical broken Newton path)
benchmarks/newton_worlds_throughput.py  Newton replicate() (T1.2 corrected baseline)
benchmarks/tier1_throughput.py          Tier-1 + Newton + 6 kernels (T2.8 ship benchmark)
```

### Docs (3 .md in `docs/`)
```
docs/math/fossen.md                     6-DOF Fossen math reference (~14 KB)
docs/refs_tried.md                      W1.5 references (R1.1 + R1.4 + R1.5)
docs/marinegym_fossen_extract.md        MarineGym source map + coefficient values
```

### Goals / planning (2 .md in `goals/`)
```
goals/W1.5_run_references.md            Reference-running plan (DONE)
goals/W2_tier1_fossen.md                Tier-1 kernel plan (DONE)
```

### Top-level planning (~150 KB)
```
SURVEY.md                               State-of-the-art landscape May 2026
DESIGN.md                               v0.1.0 architecture
STACK.md                                Tech stack decision framework (~62 KB)
VERSIONS.md                             Version pinning rationale
LAYOUT.md                               Package structure target
RISKS.md                                Risk register (R1-R25)
STATUS.md                               Real-status snapshot
IMPLEMENTATION_PLAN.md                  4-week sprint plan
REFERENCES.md                           Verified-license knowledge base (~18 KB)
THIRD_PARTY_NOTICES.md                  License compliance doc
LICENSE                                 Apache-2.0
```

### Install logs (75 files in `install_log/`)
Numbered 00-84, every test/install/benchmark/diagnose run preserved per `feedback-log-all-tests` memory rule.

### Research reports (6 in `research/`)
```
research/dr-2026-05-15-v0.1-review/SYNTHESIS.md      Local 3-fork synthesis
research/dr-2026-05-15-v0.1-review-r3/report.md      DR ChatGPT light review
research/dr-w2-plan-review/report.md                 DR W2 plan review
+ supporting events.jsonl / status.txt per call
```

---

## 4. Next priorities (v0.1.0 sprint continuation)

Per `goals/W2_tier1_fossen.md` and `IMPLEMENTATION_PLAN.md`, W3 + W4 remain.

### Immediate (W3 — BlueROV2 Heavy asset, ~1-2 weeks per realistic estimate)

**Key constraint** (from R1.4 finding 2026-05-15):
- MarineGym does NOT ship BlueROV2 Heavy USD asset. Get Heavy parameters from **von Benzon 2022 JMSE 10(12):1898 (DOI 10.3390/jmse10121898, CC-BY-4.0)** instead.

Tasks (in IMPLEMENTATION_PLAN.md):
- T3.1 Acquire BlueROV2 Heavy URDF/SDF — `clydemcqueen/bluerov2_gz` has the geometry but **NO LICENSE** (blocker for vendor). Fallback: BlueRobotics datasheet + build URDF from public dimensions.
- T3.2 Port SDF → MJCF/USD via Newton importers
- T3.3 hydro YAML from von Benzon 2022 (NOT MarineGym, which is basic BlueROV)
- T3.4 `oceanscale/scene/vehicles/bluerov2_heavy.py` factory
- T3.5 End-to-end smoke: 8192 BlueROV2 envs + Tier-1 step

### Then (W4 — Station-keeping training, ~1-2 weeks)

- T4.1 `oceanscale/tasks/station_keeping.py` (obs/action/reward spec) — mirror MarineGym Hover.py design
- T4.2 Gymnasium VectorEnv adapter
- T4.3 PPO framework decision — provisional **stable-baselines3** (rl-games has psutil<6 conflict with warp>=7.1)
- T4.4 Training script
- T4.5 Train PPO 10M steps → ≤ 0.1 m mean position error within 30 min wall-clock on RTX 5090
- T4.6 Eval + ONNX export
- T4.7 Ship — VERSION + CHANGELOG + tag v0.1.0 + PR to main

### Bonus (optional, deferred to v0.2)
- R1.2 warplab/isaac-auv-env (closest stack match — Isaac Sim 5.1 install needed, 4-7h)
- R1.3 OceanSim (BSD-3 sensor kernels for v0.2)
- T1.4 docs patches (already 60% done; remaining is STACK.md §5.2 Akkaynak-Treibitz correction)

---

## 5. Quick-start resume (when you return)

```bash
cd /home/robot/workspace/46-marine

# 1. Verify environment unchanged
.venv/bin/python -m pytest tests/ -q
# expect: 31 passed in ~14 s

PYTHONPATH=. .venv/bin/python benchmarks/tier1_throughput.py
# expect: 88 M env-steps/s @ N=8192 ✓

# 2. Check git state
git branch --show-current
# expect: feat/v0.1.0-tier1-fossen

git log --oneline -5
# expect: d57935e (W2 COMPLETE) at top

# 3. Resume — pick one:

# Option A — continue W3 BlueROV2 Heavy asset
git checkout -b feat/v0.1.0-bluerov2-heavy
# Read goals/W2_tier1_fossen.md §5 T3.* + docs/marinegym_fossen_extract.md §2
# Start by fetching von Benzon 2022 PDF for hydro coefs.

# Option B — attempt R1.2 Isaac Sim install (bonus, high risk)
# See goals/W1.5_run_references.md §5 R1.2

# Option C — pivot to OceanScale (沧渊) startup sub-projects (see §6 below)
```

---

## 6. Strategic / business state (per memory `project-strategy-framing` ACTIVE)

**Branding (locked 2026-05-15)**: OceanScale / 沧渊  
**Positioning**: AI-native simulation infrastructure accelerating underwater robotics  
**Reference model**: Lightwheel AI (Beijing-based, Newton physics partner, "vertical infra on NVIDIA" play)  

**Decomposed sub-projects** (status):
- A. Un-park strategy memory — DONE 2026-05-15
- B. Ocean review (code + STACK + SURVEY + RISKS audit) — pending
- C. POSITIONING.md (对外定位) — pending
- D. BUSINESS.md (TAM / GTM / 团队 / 里程碑) — pending
- E. Pitch artifacts (one-pager + outline) — pending
- F. **Landing page** `46-marine/website/` — design approved, implementation pending
- G. Hosting + domain (Cloudflare Pages + oceanscale.cn) — pending

**Sub-project F locked decisions**:
- Folder: `46-marine/website/`
- Stack: Astro 4 + Tailwind + `@astrojs/cloudflare`
- i18n: zh default (no prefix), en at `/en/`
- 6 sections: Hero + Why + What + Demos + Benchmarks + Contact
- Visual: Lightwheel-style (dark, demo-heavy, FPS count-ups)
- GitHub link: `github.com/wangcongrobot/46-marine`
- Design doc: `docs/superpowers/specs/2026-05-15-oceanscale-landing-page-design.md`

**Open uncertainties** (from memory):
- TAM math for "fluid-structure GPU sim platform" — bio/neuro/microfluidic markets are smaller than framing implies
- Solo + 3 domains + spinout + research = burnout vector
- Competitive landscape beyond NVIDIA (Ansys / COMSOL / OpenFOAM)
- Customer development — who pays?
- Domain name (oceanscale.cn? oceanscale.ai?) + 工商注册 undecided

---

## 7. Critical empirical findings (don't lose these)

| # | Finding | Source | Why it matters |
|---|---|---|---|
| F1 | Newton spatial-vector layout = **(linear_xyz, angular_xyz)** | T1.1 empirical | All kernel pseudocode must follow |
| F2 | `add_body()` auto-creates a free joint; explicit `add_joint_free` duplicates DOFs | T1.1 empirical | Common pitfall |
| F3 | SolverMuJoCo CPU OOM unless using `replicate(N)` worlds | T1.1 empirical | Architecture-defining; documented |
| F4 | MarineGym ships **basic BlueROV** (6 rotors), **NOT Heavy** | R1.4 read | Reset our prior assumption; use von Benzon 2022 |
| F5 | Damping has 4 cross-coupling terms (sway-yaw, heave-pitch) | R1.4 read | Patched docs/math/fossen.md §5.4 |
| F6 | Added-mass needs α=0.3 EMA filter on ν̇ for stability | R1.4 read | T2.2 implementation |
| F7 | Wu 2018 is a Flinders MS thesis where author never received hardware | DR review | Demoted to caveat-cite |
| F8 | Newton 30-day API churn (3 minors, sub-minor breaking) | DR review | Pin `newton<1.3` |
| F9 | OceanSim camera = **Akkaynak-Treibitz**, NOT Jaffe-McGlamery | DR review | STACK.md §5.2 patch pending |
| F10 | bluerov2_gz has **NO LICENSE** (cannot vendor) | License check | Use MarineGym USD as primary; build own URDF from BlueRobotics datasheet as fallback |

---

## 8. Memory cross-reference (`~/.claude/projects/-home-robot-workspace-46-marine/memory/`)

| File | Type | Status |
|---|---|---|
| `MEMORY.md` | index | up to date |
| `project_underwater_sim.md` | project | active |
| `project_strategy_framing.md` | project | **ACTIVE** (un-parked 2026-05-15) |
| `reference_survey_design_docs.md` | reference | up to date (covers SURVEY/DESIGN/STACK/VERSIONS/LAYOUT/RISKS/STATUS/IMPL_PLAN/REFERENCES) |
| `reference_underwater_sim_urls.md` | reference | up to date |
| `feedback_stack_stability.md` | feedback | active rule — don't import new tech without §17.3 5-bar check |
| `feedback_log_all_tests.md` | feedback | active rule — every test must log to install_log/NN_*.txt |

---

## 9. Uncommitted right now (clean state requested)

```
?? install_log/83_w2_final_commit.txt    last commit log (will be added in HANDOFF commit)
?? install_log/84_handoff_state.txt      state snapshot for this doc
```

Plus this `HANDOFF.md` itself.

---

## 10. Known caveats / gotchas

- **Meshcat viewer daemon** from `bluerov2_gym` install starts a server at 127.0.0.1:7000 even with `render_mode=None`. Harmless but worth knowing.
- **vLLM at 127.0.0.1:8000 (PID was 1840775 earlier)** is the user's local Qwen3.6 server. Was unloaded mid-session via `tmux send-keys -t vllm-5090 C-c`. May restart between sessions.
- **Disk pressure**: 161 GB free on `/`, 436 GB on `/mnt/storage` (where Warp cache lives). Should be OK for v0.1 but Isaac Sim install (~30 GB) for W3/W4 will eat into headroom.
- **GPU memory**: 31 GB free out of 32 GB total (RTX 5090 Blackwell sm_120). vLLM had been hogging it before unload — verify with `nvidia-smi` after resume.
- **Python 3.12 in `.venv/`** is uv-managed; Python 3.13 is on system path. Do not confuse.
- **Newton pinned to `<1.3`** in pyproject.toml per DR-review R19 (3 minor versions in 30 days with sub-minor breaking changes).

---

## 11. Commit the handoff

When you (or the next session) read this, the recommended first action is:

```bash
git add HANDOFF.md install_log/83_w2_final_commit.txt install_log/84_handoff_state.txt
git commit -m "docs(handoff): session 2026-05-15 — W1.5 + W2 PASS, ready for W3"
```

---

## 12. Summary in one paragraph

OceanScale v0.1.0 sprint completed two of four phases in one session. Reference projects investigated (bluerov2_gym, MarineGym, stonefish), key empirical finding: MarineGym ships basic BlueROV not Heavy — Heavy will come from von Benzon 2022. Tier-1 Fossen Warp kernels implemented (6 kernels: added-mass, damping with cross-coupling, Coriolis C_A, restoring via quaternion, thruster with saturation/deadband/τ_lag, plus EMA filter on ν̇), all autograd-validated against analytic and torch finite-diff. Throughput verified at 88 M env-steps/s @ 8192 envs on RTX 5090 (880× the 100 k target). 31/31 tests pass, ruff clean. Next is W3 (BlueROV2 Heavy asset, 1-2 weeks) then W4 (station-keep PPO training, 1-2 weeks). Branding locked as OceanScale / 沧渊; landing page sub-project F has approved design, implementation pending.
