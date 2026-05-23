# Real-Status Snapshot — 2026-05-15

Hardware: **RTX 5090 32 GB Blackwell (sm_120)** + Intel i9-14900K + 62 GB RAM, Ubuntu 24.04, driver 580.95.05, CUDA toolkit 12.8.

Stack installed: warp-lang 1.13.0, newton 1.2.0, torch 2.11.0+cu128, mujoco-warp 3.8.1, usd-core 26.3, numpy 2.4.4, scipy 1.17.1. **140 packages total, install time 16:14.**

---

## 4-Dimensional Review — 2026-05-21

Review completed across 4 dimensions: tech stack, website, roadmap, staff.

**Reports**: `/tmp/oceanscale_review_tech_stack.md`, `/tmp/oceanscale_review_roadmap.md`, `/tmp/oceanscale_review_staff.md`

### Founder decisions (2026-05-21)

1. **Brand english-only** — 沧渊 retired; wordmark = OceanScale on both zh/en. New Chinese name TBD.
2. **Staff no-hurry** — solo dev continues; 6-9 week v0.1 timeline accepted; no hiring push.
3. **Website concept-mainly** — drop specific unverified throughput numbers; keep vision-level framing.

### Next gates

1. **v0.6.0 ship** — concept-mainly website edits land + tag (in progress, uncommitted — see git status)
2. **W2 autograd gate** — `tests/hydro/test_tier1_autograd.py` must pass (kernels written, autograd test pending)
3. **von Benzon reference scaffold** — `validation/vonbenzon_reference.py` (R13 unblocker, unmitigated per tech review Top Risk #1)

### Active workers

- Website content rewrite (en + zh content collections) — uncommitted in working tree
- Hydro Tier-1 kernels (`oceanscale/hydro/tier1*.py`) — modified in working tree

### Review summary — tech stack

- **Confirmed**: 25 items (Warp, Newton, CUDA 12.8 benchmarks all verified on RTX 5090)
- **Risky**: 15 items (MuJoCo-Warp throughput -46%, Newton API instability R19, Isaac Lab Newton gap)
- **Unverified**: 18 items (Tier-1 hydro throughput, sensor kernels, sim-to-real claims)
- **Top risk**: No validated AUV reference data — cannot confirm physics correctness (R13, unmitigated)

### Review summary — roadmap

- Website v0.5.0 live; v0.6.0 concept-mainly rewrite in progress
- Python v0.1.0 scaffold: 18/18 tests pass, RL pipeline at 108 tests
- No customer discovery, funding plan, or GTM strategy in any reviewed document
- Roadmap reads as excellent research project plan, not yet a startup roadmap

### Review summary — staff

- Solo developer validated GPU stack + Newton scaffolding
- Zero domain-specific code (hydro, sensors, RL training) shipped yet
- P0 gaps: Fossen 6-DOF implementation, Warp autograd debugging, RL reward shaping
- Verdict: significantly understaffed for ambition, but solo continues per founder decision

---

## Test status (`pytest tests/ -v`)

**18/18 PASS in 13.39s** (after Warp + Newton JIT compilation primed).

| Test file | Count | Notes |
|---|---|---|
| `test_environment.py` | 10 | All imports + CUDA detection + version pins |
| `test_warp_smoke.py` | 4 | Forward kernel, autograd vs analytic, determinism, device |
| `test_newton_smoke.py` | 4 | Free-body falls (SemiImplicit + MuJoCo), 8192-body finalize |

---

## Real performance numbers (`benchmarks/kernel_throughput.py`)

### 1. Pure Warp kernel (1D quadratic drag, no physics overhead)
| n_envs | kernel-launches/s | M env-steps/s |
|---|---|---|
| 1024 | 183 k | 187 |
| 8192 | 185 k | **1 518** |
| 65 536 | 188 k | 12 322 |
| 262 144 | 181 k | 47 317 |

Pure GPU ceiling: **~47 B env-steps/s at 256 k envs**. Launch overhead dominates below 8 k envs.

### 2. Newton **SolverSemiImplicit** — free bodies in vacuum (no contacts)
| n_envs | steps/s | ms/step | M env-steps/s |
|---|---|---|---|
| 64 | 26.4 k | 0.04 | 1.7 |
| 256 | 25.6 k | 0.04 | 6.6 |
| 1024 | 26.0 k | 0.04 | 26.6 |
| 4096 | 26.4 k | 0.04 | 108.0 |
| 8192 | **26.9 k** | **0.04** | **220.1** |

Step rate ~27 kHz **regardless of batch size** — perfectly parallel. vs real-time 240 Hz that's **112× faster per env, 8192 envs concurrent**. Beats MarineGym's RTX 3060 number (250 k FPS) by ~880× before any hydrodynamics.

### 3. Newton **SolverMuJoCo** (MuJoCo-Warp) — historical (broken pattern, flat-list bodies)
| n_envs | result |
|---|---|
| 64 | 262 steps/s, 3.82 ms/step, 0.017 M env-steps/s |
| 256 | ❌ `IndexError: index 19 is out of bounds for axis 0 with size 19` |
| 1024 | ❌ `mj_stackAlloc: out of memory, MakeHessian, line 1695` |
| 4096 | ❌ `mj_stackAlloc: out of memory, PrimalAllocate, line 1106` |
| 8192 | ❌ same, requesting 19 GB host RAM |

### Architectural fix applied (T1.1 + T1.2, 2026-05-15)

The 256+ OOM was because all bodies lived in `world=-1` → MuJoCo treated them as one giant articulation. **Fix**: `newton.ModelBuilder.replicate(template, world_count=N)` distributes the bodies into N independent worlds. MuJoCo's `separate_worlds` path then handles them correctly.

### 4. Newton throughput with `replicate(world_count=N)` — CORRECTED (T1.2)

`benchmarks/newton_worlds_throughput.py`, RTX 5090, free spheres, no contacts.

| N | SolverSemiImplicit | SolverMuJoCo (now works) |
|---|---|---|
| 64 | 1.72 M env-steps/s (26.9 k Hz, 0.04 ms/step) | 0.043 M env-steps/s (678 Hz, 1.47 ms/step) |
| 256 | 6.92 M | 0.176 M (was OOM) |
| 1024 | 27.6 M | 0.707 M (was OOM) |
| 4096 | 111.9 M | 2.84 M (was OOM) |
| 8192 | **221 M** | **5.5 M** (was OOM) |

**Key observations**:
- **SolverMuJoCo no longer OOMs** at any tested N. Architectural fix verified.
- **SolverSemiImplicit ≈ 40× faster** than SolverMuJoCo on this workload (free spheres, no contacts). MuJoCo has ~1.5 ms host-side per-step overhead that dominates at this batch size.
- The "475× faster than MJX" marketing claim is for a single complex robot, not N parallel free bodies.
- **For our v0.1 station-keep target (≥100k env-steps/s at 8192 envs)**, both solvers have headroom:
  - SemiImplicit: 221 M / 100 k = **2200× margin** before Tier-1 hydro overhead
  - MuJoCo: 5.5 M / 100 k = **55× margin**

### Decision (LOCKED 2026-05-15 by T1.2 evidence)

**v0.1 default solver = SolverSemiImplicit.** SolverMuJoCo viable as v0.2 alternative when richer physics (contacts, joints, articulations) actually warrants the overhead.

---

## Stack health summary

✅ **Working & validated**
- Python 3.12 + uv-managed venv
- CUDA driver 580 + Toolkit 12.8
- PyTorch 2.11 + cu128 detects RTX 5090
- Warp 1.13 — JIT compile, kernel launch, autograd, multi-device-detect
- Newton 1.2 — ModelBuilder, finalize on CUDA, eval_fk, SolverSemiImplicit step
- Newton 1.2 — 8192 body finalize on RTX 5090 succeeds (mass + state alloc OK)
- USD-core 26.3 import
- MuJoCo-Warp 3.8.1 (bundled with Newton)
- pytest, ruff, mypy, pre-commit toolchain

⚠️ **Found gotchas (worth recording for STACK.md)**
- Newton spatial vector layout = **(linear_xyz, angular_xyz)**, NOT `(angular, linear)`. Empirically verified.
- `add_body()` automatically creates a free joint with 7 joint_q (xyz + quat-xyzw) and 6 joint_qd. **Do NOT add an explicit `add_joint_free`** — duplicates DOFs.
- `model.collide(state)` is optional — passing `None` for contacts to `solver.step` is supported.
- Shape default density = 1000 → shape mass overrides `add_body(mass=...)` argument.
- `mujoco-usd-converter` not currently bundled with newton 1.2.0 (we may need it for asset import).
- SolverMuJoCo has CPU memory ceiling at ~hundreds of independent bodies (host-side `mj_stackAlloc`).
- Newton JIT compile takes ~30 s on first cold load (lots of kernels); cached compilation < 100 ms after.

❌ **Not yet tested / unknown**
- Isaac Lab 3.0 — deferred to v0.2 (source build)
- Isaac Sim 5.1 — deferred (large install)
- USD scene authoring beyond pxr import — deferred to v0.1.x
- BVH-based sonar ray casting (Warp) — v0.2
- Newton worlds / per-env replication API — needs investigation
- ROS 2 Jazzy bridge — deferred

---

## Disk + memory accounting

- uv cache: 18 GB (in `/mnt/storage` NVMe, 436 GB free remaining)
- venv `.venv/`: ~5 GB (PyTorch CUDA libs dominate)
- GPU memory free after vLLM unload: 31.4 GB / 32.6 GB
- Host RAM: 42 GB free / 62 GB total

---

## What changes from `STACK.md` & `VERSIONS.md` based on real status

1. **STACK.md §11** — bump CUDA toolkit baseline from 12.4 → 12.8 (already done in VERSIONS.md)
2. **STACK.md §3.4** — note that "475× MJX manipulation" is a single-robot benchmark; for parallel-env RL, **SolverSemiImplicit is the right primary** until we map the worlds API
3. **STACK.md §9.1** — when writing Tier-1 Fossen kernel, use **(linear_xyz, angular_xyz)** layout (was ambiguous)
4. **ARCHITECTURE.md §3** — Tier-0/1 should explicitly call SolverSemiImplicit, not "Newton default"
5. **RISKS.md** — add R19: "SolverMuJoCo CPU host-side OOM for many-body workloads; mitigation = SolverSemiImplicit until worlds API understood"

---

## Next steps (proposed)

1. ✅ Commit current state (18 tests + benchmark + scaffold)
2. Learn Newton's `worlds` / replication mechanism — investigate via `newton.examples` source
3. Re-run benchmark with proper per-env worlds, expect SolverMuJoCo to recover
4. Write Tier-1 Fossen Warp kernel (the actual project starting point)
5. Patch STACK/DESIGN/RISKS docs with empirical findings above
