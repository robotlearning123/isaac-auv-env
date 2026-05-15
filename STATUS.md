# Real-Status Snapshot — 2026-05-15

Hardware: **RTX 5090 32 GB Blackwell (sm_120)** + Intel i9-14900K + 62 GB RAM, Ubuntu 24.04, driver 580.95.05, CUDA toolkit 12.8.

Stack installed: warp-lang 1.13.0, newton 1.2.0, torch 2.11.0+cu128, mujoco-warp 3.8.1, usd-core 26.3, numpy 2.4.4, scipy 1.17.1. **140 packages total, install time 16:14.**

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

### 3. Newton **SolverMuJoCo** (MuJoCo-Warp, the "primary" backend)
| n_envs | result |
|---|---|
| 64 | 262 steps/s, **3.82 ms/step**, 0.017 M env-steps/s |
| 256 | ❌ `IndexError: index 19 is out of bounds for axis 0 with size 19` |
| 1024 | ❌ `mj_stackAlloc: out of memory, MakeHessian, line 1695` |
| 4096 | ❌ `mj_stackAlloc: out of memory, PrimalAllocate, line 1106` |
| 8192 | ❌ same, requesting 19 GB host RAM |

### Interpretation
- SolverMuJoCo is **~100× slower per step** than SolverSemiImplicit at low n (3.82 ms vs 0.04 ms)
- SolverMuJoCo **CPU-side stack-allocs** scale O(n × constraint-count) — it treats 8192 free spheres as ONE giant articulation
- The "475× faster than MJX" claim in NVIDIA marketing is for a **single multi-DOF robot**, not for parallel-RL many-body workloads
- **Implication**: for parallel-env RL, our path is either:
  - **(a)** Use **SolverSemiImplicit** as the Tier-0/1 default (proven 220 M env-steps/s at 8 k envs)
  - **(b)** Use Newton's `world` mechanism — replicate per-env models with proper isolation, so MuJoCo sees N independent worlds, not 1 mega-world
  - **(c)** Lift our scene model so each env is a single 6-DOF body (this is the typical Isaac Lab pattern; we just need to figure out Newton's equivalent)

### Decision (provisional)
**v0.1 uses SolverSemiImplicit as primary**, defers SolverMuJoCo to v0.2 once we learn the worlds API. SemiImplicit gives us 200× our STACK.md throughput target without optimization. We can swap later.

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
4. **DESIGN.md §3** — Tier-0/1 should explicitly call SolverSemiImplicit, not "Newton default"
5. **RISKS.md** — add R19: "SolverMuJoCo CPU host-side OOM for many-body workloads; mitigation = SolverSemiImplicit until worlds API understood"

---

## Next steps (proposed)

1. ✅ Commit current state (18 tests + benchmark + scaffold)
2. Learn Newton's `worlds` / replication mechanism — investigate via `newton.examples` source
3. Re-run benchmark with proper per-env worlds, expect SolverMuJoCo to recover
4. Write Tier-1 Fossen Warp kernel (the actual project starting point)
5. Patch STACK/DESIGN/RISKS docs with empirical findings above
