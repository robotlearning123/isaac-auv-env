# Technical Risk Register — v0.1.0

**Date:** 2026-05-15  
**Scope:** Expands `STACK.md §12` — for each risk: trigger conditions, monitoring metric, fallback plan, decision deadline.

Reviewed bi-weekly (Fridays). Anything in 🔴 needs immediate attention; 🟡 watch; 🟢 mitigated/accepted.

---

## 1. Stack-level risks

### R1 🟡 Newton in Isaac Lab is "experimental, breaking changes expected"
| Field | Value |
|---|---|
| Trigger | Isaac Lab `develop` branch breaks our Newton env in a sync |
| Monitor | Weekly `git pull` of IsaacLab → run our env smoke test |
| Probability | High (it's explicitly marked experimental) |
| Impact | Medium — Isaac Lab integration deferred to v0.2, so v0.1 doesn't depend on it yet |
| Fallback | Use Newton standalone (Python API, no Isaac Lab wrapper) until 3.0 GA. Build our own gymnasium wrapper if needed. |
| Decision deadline | v0.2 sprint start (estimated 2026-08) |
| Watchlist link | `STACK.md §17.2` |

### R2 🟡 Warp 1.x → 2.x breaking changes
| Field | Value |
|---|---|
| Trigger | Warp ships 2.0 with API breaks (`warp.sim` already deprecated/removed in Newton) |
| Monitor | Subscribe to GitHub releases; CI on `warp-lang>=2,<3` nightly |
| Probability | Medium — semver should protect us but 1.x has had subtle behavior changes |
| Impact | High — our kernels are the differentiation; rewriting them costs weeks |
| Fallback | Pin to last known-good 1.x in `pyproject.toml`; defer 2.x migration to a dedicated sprint |
| Decision deadline | When Warp 2.0 ships (estimated 2026 Q4) |

### R3 🟢 CUDA driver compatibility
| Field | Value |
|---|---|
| Trigger | OS update bumps driver beyond what wheels support |
| Monitor | `nvidia-smi` driver version in CI |
| Probability | Low — driver 580 with CUDA 13 support is forward-compat |
| Impact | High if it breaks |
| Fallback | Hold OS updates; pin driver via apt hold |
| Decision deadline | Continuous |
| Notes | Currently 580.95.05 + CUDA toolkit 12.8 — confirmed working |

### R4 🟡 PyTorch wheel for CUDA 12.8 disappears
| Field | Value |
|---|---|
| Trigger | PyTorch drops cu128 wheels in favor of cu129 / cu13 |
| Monitor | https://download.pytorch.org/whl/cu128/ index changes |
| Probability | Medium (12-18 months) |
| Impact | Medium — we'd migrate to cu129/13 wheels |
| Fallback | Bump to newer cu wheel + retest stack |
| Decision deadline | 2027 Q1 review |

### R5 🟢 Newton package name change (`newton-physics` → `newton`)
| Field | Value |
|---|---|
| Trigger | Already happened (Feb 2026) |
| Monitor | n/a — captured in `pyproject.toml` |
| Probability | n/a |
| Impact | n/a |
| Fallback | n/a |
| Decision deadline | n/a (mitigated) |

---

## 2. Hardware / compute risks

### R6 🟡 RTX 5090 single-GPU bottleneck for v0.4 multi-agent
| Field | Value |
|---|---|
| Trigger | Multi-AUV MARL benchmarks exceed 32 GB VRAM at ≥256 agents |
| Monitor | `oceanscale bench --task acoustic_marl --vehicles N` memory headroom |
| Probability | High when we reach v0.4 |
| Impact | High — caps benchmark scale |
| Fallback | (a) Cloud H100/H200 rental for ablations; (b) lower precision (FP16 obs); (c) shard env across multi-GPU |
| Decision deadline | v0.4 sprint start (~2027 Q1) |

### R7 🟢 vLLM (or other) co-resident workloads steal GPU memory
| Field | Value |
|---|---|
| Trigger | Other user workload allocates VRAM during training |
| Monitor | `nvidia-smi` start-of-run check in CLI |
| Probability | High in dev (saw 30 GB vLLM today) |
| Impact | Training OOM |
| Fallback | CLI checks `>= 28 GB free` before launch; warn user |
| Decision deadline | Add check in v0.1.1 |

### R8 🟢 Disk pressure
| Field | Value |
|---|---|
| Trigger | uv cache + asset library grow past 200 GB |
| Monitor | `df -h /home/robot` in CI |
| Probability | Medium — already at 82% |
| Impact | Hard failure mid-install |
| Fallback | Use `UV_CACHE_DIR` to move cache to larger disk; or LFS for assets |
| Decision deadline | Continuous, alert if `< 50 GB free` |
| Notes | Today: 171 GB free; uv cache 17 GB → projected 25 GB after install |

---

## 3. Physics / domain risks

### R9 🔴 MuJoCo ellipsoid fluid model accuracy at high-speed AUV
| Field | Value |
|---|---|
| Trigger | BlueROV2 Heavy at >2 m/s shows >5% trajectory error vs Fossen analytic |
| Monitor | `tests/hydro/test_tier1_fossen.py::test_marinegym_parity` |
| Probability | Medium — model is phenomenological, regime-dependent |
| Impact | High — invalidates Tier-0 default; forces Tier-1 always-on (+ overhead) |
| Fallback | Already planned — Tier-1 Warp kernel ready by v0.2 |
| Decision deadline | v0.1 validation gate |
| Notes | Validate against MarineGym + HoloOcean published REMUS data |

### R10 🟡 BELLHOP precompute > 100 ms makes acoustic comms unusable for RL
| Field | Value |
|---|---|
| Trigger | One Tx-Rx IR computation exceeds 100 ms wall |
| Monitor | `benchmarks/sensor_throughput.py::bellhop` |
| Probability | Medium — BELLHOP is CPU |
| Impact | High for v0.4 MARL benchmark |
| Fallback | Coarser grid, lookup-table interpolation, GPU-port a subset |
| Decision deadline | v0.4 |

### R11 🟡 SPH Tier-2 too slow for 8 manipulator ROIs in parallel
| Field | Value |
|---|---|
| Trigger | Tier-2 kernel < 100 Hz at 8 simultaneous 5k-particle regions |
| Monitor | `benchmarks/kernel_throughput.py::tier2_sph` |
| Probability | Medium — SPH at multi-region |
| Impact | Medium — manipulator FSI is a v0.4 feature |
| Fallback | Reduce particle count; coarser kernel; alternating-region scheduling |
| Decision deadline | v0.4 |

### R12 🟡 Free-surface waves cost dominates step at near-surface tasks
| Field | Value |
|---|---|
| Trigger | Tier-3 surface wave > 30% of step time |
| Monitor | profile during station-keeping near surface |
| Probability | Medium |
| Impact | Medium — caps near-surface task throughput |
| Fallback | Pre-bake wave field; only solve in regions ±0.5 m of surface |
| Decision deadline | v0.3 |

---

## 4. Reproducibility / validation risks

### R13 🔴 No validated AUV reference data
| Field | Value |
|---|---|
| Trigger | Cannot validate Tier-1 Fossen kernel without ground-truth |
| Monitor | Identify ≥1 published REMUS / BlueROV2 dataset by v0.2 |
| Probability | We don't have data today |
| Impact | High — can't claim sim-to-real until validated |
| Fallback | (a) HoloOcean 2.0 paper REMUS data; (b) MarineGym ablation traces; (c) collect own data with real BlueROV2 |
| Decision deadline | v0.2 sprint start |

### R14 🟡 Warp non-determinism across runs
| Field | Value |
|---|---|
| Trigger | Same seed, same input, different output across runs |
| Monitor | dedicated `tests/repro/` set, pinned seeds + CUDA deterministic flags |
| Probability | Medium — atomic ops in kernels are inherently non-deterministic |
| Impact | Medium for debug; high for RL evaluation |
| Fallback | Document determinism boundaries; use deterministic-CUDA in eval-only mode |
| Decision deadline | v0.2 |

### R15 🟡 USD assets break across pxr / Omniverse / Isaac Sim 6 versions
| Field | Value |
|---|---|
| Trigger | A scene loads correctly in usd-core but not in Isaac Sim 6 (or vice versa) |
| Monitor | Cross-tool validation in CI when both available |
| Probability | High — USD schema fragmentation is real |
| Impact | High at v0.2 when we add Isaac Sim |
| Fallback | Strict pxr USD ≥25.11 + avoid Omniverse-specific extensions in core assets |
| Decision deadline | v0.2 |

---

## 5. Schedule / scope risks

### R16 🟡 Newton solver bugs in Tier-1 + custom kernel interaction
| Field | Value |
|---|---|
| Trigger | Forces inject correctly via pre-step hook but Newton integrator misbehaves |
| Monitor | Energy conservation check + analytic-trajectory match |
| Probability | Medium — pre-step hooks are well-defined but combination is novel |
| Impact | High — blocks v0.1 |
| Fallback | File issue + workaround via post-step force application; engage Newton consortium |
| Decision deadline | First custom kernel landing (~v0.1.0-w2) |

### R17 🟢 Disk full during install
| Field | Value |
|---|---|
| Trigger | uv cache + venv + assets > available |
| Monitor | `df -h` pre-flight |
| Probability | Low today (171 GB free) |
| Impact | Aborts install |
| Fallback | Move uv cache to bigger disk via env var |
| Decision deadline | Now — `df` check is the pre-flight |

### R18 🟡 Python 3.12 vs ecosystem lag
| Field | Value |
|---|---|
| Trigger | A v0.2 dependency (Isaac Lab, skrl, etc.) requires 3.11 |
| Monitor | Pre-add dep version check |
| Probability | Low — 3.12 well-supported by May 2026 |
| Impact | Medium — would force 3.11 downgrade |
| Fallback | Maintain `python-3.11` group via uv extras |
| Decision deadline | When adding each new dep |

---

## 6. Decision triggers (read this column when reviewing)

| Risk | Triggers a major architecture re-think if... |
|---|---|
| R1 | Newton-IsaacLab still experimental at 2027-01 |
| R2 | Warp 2.0 ships with non-trivial breaking changes |
| R6 | RTX 5090 is consistently the bottleneck for our target tasks |
| R9 | Tier-0 ellipsoid model fails BlueROV2 validation at any common speed |
| R13 | Cannot procure validated reference data by v0.2 |
| R16 | Newton + custom pre-step hooks have a fundamental incompatibility |

If any of the above ⇒ open §17 main-stack-lock review (per `STACK.md §17.3`).

---

## 7. Out of scope for this register

- Business / market / funding risks — out of scope (project is research-mode)
- IP / legal risks — see HMS/MGB Innovation when relevant; not active topic
- Personnel / hiring risks — solo developer for now
- Cross-domain (bio / neuro) risks — that's a "Beijing" track concern, not current scope

---

## 8. Update history

- 2026-05-15 — register created, 18 risks logged, all initial states.
