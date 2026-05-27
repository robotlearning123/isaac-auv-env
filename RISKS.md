# Technical Risk Register

Reviewed regularly. Anything in RED needs immediate attention; YELLOW watch; GREEN mitigated/accepted.

---

## 1. Stack-level risks

### R1 YELLOW — Newton in Isaac Lab is "experimental, breaking changes expected"
| Field | Value |
|---|---|
| Trigger | Isaac Lab `develop` branch breaks our Newton env in a sync |
| Monitor | Re-run `python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py` before changing the Isaac Lab source pin |
| Probability | High (it's explicitly marked experimental) |
| Impact | High -- Isaac Lab 3 is the main robot-learning integration lane for v0.1 validation |
| Fallback | Keep the standalone Newton/OceanScale env as the development fallback while the Isaac lane is repaired. Do not silently switch Isaac Lab source heads without rerunning the verifier. |

### R2 YELLOW — Warp 1.x -> 2.x breaking changes
| Field | Value |
|---|---|
| Trigger | Warp ships 2.0 with API breaks |
| Monitor | Subscribe to GitHub releases; CI on `warp-lang>=2,<3` nightly |
| Probability | Medium -- semver should protect us but 1.x has had subtle behavior changes |
| Impact | High -- our kernels are the differentiation; rewriting them costs weeks |
| Fallback | Pin to last known-good 1.x in `pyproject.toml`; defer 2.x migration to a dedicated sprint |

### R3 GREEN — CUDA driver compatibility
| Field | Value |
|---|---|
| Trigger | OS update bumps driver beyond what wheels support |
| Monitor | `nvidia-smi` driver version in CI |
| Probability | Low -- driver 580 with CUDA 13 support is forward-compat |
| Impact | High if it breaks |
| Fallback | Hold OS updates; pin driver via apt hold |

### R4 YELLOW — PyTorch wheel for CUDA 12.8 disappears
| Field | Value |
|---|---|
| Trigger | PyTorch drops cu128 wheels in favor of cu129 / cu13 |
| Monitor | PyTorch download index changes |
| Probability | Medium (12-18 months) |
| Impact | Medium -- we'd migrate to newer cu wheel |
| Fallback | Bump to newer cu wheel + retest stack |

### R5 GREEN — Newton package name change
| Field | Value |
|---|---|
| Trigger | Already happened (Feb 2026) |
| Impact | Mitigated -- captured in `pyproject.toml` |

---

## 2. Hardware / compute risks

### R6 YELLOW — Single-GPU bottleneck for multi-agent
| Field | Value |
|---|---|
| Trigger | Multi-AUV MARL benchmarks exceed 32 GB VRAM at >=256 agents |
| Monitor | `oceanscale bench --task acoustic_marl --vehicles N` memory headroom |
| Probability | High when we reach multi-agent features |
| Impact | High -- caps benchmark scale |
| Fallback | (a) Cloud H100/H200 rental for ablations; (b) lower precision (FP16 obs); (c) shard env across multi-GPU |

### R7 GREEN — Co-resident workloads steal GPU memory
| Field | Value |
|---|---|
| Trigger | Other workload allocates VRAM during training |
| Monitor | `nvidia-smi` start-of-run check in CLI |
| Probability | High in dev |
| Impact | Training OOM |
| Fallback | CLI checks `>= 28 GB free` before launch; warn user |

### R8 GREEN — Disk pressure
| Field | Value |
|---|---|
| Trigger | uv cache + asset library grow past available disk |
| Monitor | `df -h` in CI |
| Impact | Hard failure mid-install |
| Fallback | Use `UV_CACHE_DIR` to move cache to larger disk; or LFS for assets |

---

## 3. Physics / domain risks

### R9 RED — MuJoCo ellipsoid fluid model accuracy at high-speed AUV
| Field | Value |
|---|---|
| Trigger | BlueROV2 Heavy at >2 m/s shows >5% trajectory error vs Fossen analytic |
| Monitor | `tests/hydro/test_tier1_fossen.py::test_marinegym_parity` |
| Probability | Medium -- model is phenomenological, regime-dependent |
| Impact | High -- invalidates Tier-0 default; forces Tier-1 always-on |
| Fallback | Tier-1 Warp kernel ready by v0.2 |
| Notes | Validate against MarineGym + HoloOcean published REMUS data |

### R10 YELLOW — BELLHOP precompute > 100 ms makes acoustic comms unusable for RL
| Field | Value |
|---|---|
| Trigger | One Tx-Rx IR computation exceeds 100 ms wall |
| Monitor | `benchmarks/sensor_throughput.py::bellhop` |
| Probability | Medium -- BELLHOP is CPU |
| Impact | High for MARL benchmark |
| Fallback | Coarser grid, lookup-table interpolation, GPU-port a subset |

### R11 YELLOW — SPH Tier-2 too slow for 8 manipulator ROIs in parallel
| Field | Value |
|---|---|
| Trigger | Tier-2 kernel < 100 Hz at 8 simultaneous 5k-particle regions |
| Monitor | `benchmarks/kernel_throughput.py::tier2_sph` |
| Probability | Medium |
| Impact | Medium -- manipulator FSI is a later feature |
| Fallback | Reduce particle count; coarser kernel; alternating-region scheduling |

### R12 YELLOW — Free-surface waves cost dominates step at near-surface tasks
| Field | Value |
|---|---|
| Trigger | Tier-3 surface wave > 30% of step time |
| Monitor | profile during station-keeping near surface |
| Probability | Medium |
| Impact | Medium -- caps near-surface task throughput |
| Fallback | Pre-bake wave field; only solve in regions +/- 0.5 m of surface |

---

## 4. Reproducibility / validation risks

### R13 RED — No validated AUV reference data
| Field | Value |
|---|---|
| Trigger | Cannot validate Tier-1 Fossen kernel without ground-truth |
| Monitor | Identify >=1 published REMUS / BlueROV2 dataset |
| Probability | We don't have data today |
| Impact | High -- can't claim sim-to-real until validated |
| Fallback | (a) HoloOcean 2.0 paper REMUS data; (b) MarineGym ablation traces; (c) collect own data with real BlueROV2 |

### R14 YELLOW — Warp non-determinism across runs
| Field | Value |
|---|---|
| Trigger | Same seed, same input, different output across runs |
| Monitor | dedicated `tests/repro/` set, pinned seeds + CUDA deterministic flags |
| Probability | Medium -- atomic ops in kernels are inherently non-deterministic |
| Impact | Medium for debug; high for RL evaluation |
| Fallback | Document determinism boundaries; use deterministic-CUDA in eval-only mode |

### R15 YELLOW — USD assets break across pxr / Omniverse / Isaac Sim 6 versions
| Field | Value |
|---|---|
| Trigger | A scene loads correctly in usd-core but not in Isaac Sim 6 (or vice versa) |
| Monitor | Isaac baseline verifier plus cross-tool validation when both runtimes are available |
| Probability | High -- USD schema fragmentation is real |
| Impact | High -- Isaac Sim 6 is the main NVIDIA ecosystem target even though core assets stay dependency-isolated |
| Fallback | Strict pxr USD >=25.11 + avoid Omniverse-specific extensions in core assets |

---

## 5. Technical debt risks

### R16 YELLOW — Newton solver bugs in Tier-1 + custom kernel interaction
| Field | Value |
|---|---|
| Trigger | Forces inject correctly via pre-step hook but Newton integrator misbehaves |
| Monitor | Energy conservation check + analytic-trajectory match |
| Probability | Medium -- pre-step hooks are well-defined but combination is novel |
| Impact | High -- blocks v0.1 |
| Fallback | File issue + workaround via post-step force application; engage Newton consortium |

### R17 GREEN — Disk full during install
| Field | Value |
|---|---|
| Trigger | uv cache + venv + assets > available |
| Monitor | `df -h` pre-flight |
| Probability | Low |
| Impact | Aborts install |
| Fallback | Move uv cache to bigger disk via env var |

### R18 YELLOW — Python 3.12 vs ecosystem lag
| Field | Value |
|---|---|
| Trigger | A dependency requires a Python version outside the Isaac Sim 6 lane |
| Monitor | Pre-add dependency version check and Isaac baseline verifier |
| Probability | Medium -- core-only experiments can run newer Python, but Isaac Sim 6 is Python 3.12-first |
| Impact | Medium -- broken env resolution or contaminated imports |
| Fallback | Keep Python 3.12 pinned for the Isaac validation env; use separate core-only envs for experiments. |

---

## 6. Newton ecosystem risks

### R19 RED — Newton sub-minor breaking changes faster than expected
| Field | Value |
|---|---|
| Trigger | Newton 1.3 ships breaking-change in a function we depend on |
| Monitor | Newton release notes; CI pin |
| Probability | High -- 3 minors in 30 days, documented breaks 1.1->1.2 |
| Impact | High -- could block development mid-sprint |
| Fallback | Pin `<1.3` until next sprint absorbs known-good 1.3.x |

### R20 RED — Warp autograd debugging complexity
| Field | Value |
|---|---|
| Trigger | Autograd correctness test fails on kernel ports |
| Monitor | Warp issue tracker; daily progress |
| Probability | High -- Warp issues archive shows ~5 days typical for nontrivial autograd debug |
| Impact | High -- directly blocks development |
| Fallback | Write torch numerical-diff harness BEFORE writing each kernel (TDD); fall back to torch primary + Warp port if intractable |

### R21 RED — Reward shaping needs multiple iterations
| Field | Value |
|---|---|
| Trigger | First-pass PPO does not converge to <=0.1m within 10M env-steps |
| Monitor | Learning curve dashboard |
| Probability | High -- literature notes 17-D obs reward shaping is nontrivial |
| Impact | High -- could push schedule |
| Fallback | Plan with 2-iteration buffer; mirror reference implementation reward exactly for first attempt |

### R22 RED — Thruster nonlinearity required for zero-shot
| Field | Value |
|---|---|
| Trigger | Trained policy fails on BlueROV2 Heavy hardware due to thrust curve mismatch |
| Monitor | Inspect MarineGym `BlueROVHeavy.py` thrust function |
| Probability | High -- empirical sim2real residual taxonomy shows 30% of failures are thrust allocation |
| Impact | High -- blocks real-hardware claims |
| Fallback | Add deadband + saturation + 1st-order time-constant explicitly, not "later" |

### R23 RED — No open packaged tank-test dataset for BlueROV2 Heavy
| Field | Value |
|---|---|
| Trigger | We need to validate Tier-1 coefficients but cannot find ground-truth trajectories |
| Monitor | n/a -- already true |
| Probability | n/a -- confirmed |
| Impact | Medium -- can't claim sim-to-real until hardware arrives |
| Fallback | Use **von Benzon 2022 (DOI 10.3390/jmse10121898) Simulink simulator** to generate reference trajectories |

### R24 YELLOW — MarineGym frozen on Isaac Sim 4.1
| Field | Value |
|---|---|
| Trigger | We vendor their tensor Fossen and bug-fix it ourselves indefinitely |
| Monitor | Watch upstream commits |
| Probability | High -- last commit 2026-01-27 |
| Impact | Low for v0.1 (we port to Warp; don't depend on their code at runtime) |
| Fallback | Our Warp port becomes the maintained version. Credit MarineGym in NOTICE. |

### R25 YELLOW — Hydroelastic contact regression in Newton 1.2.0
| Field | Value |
|---|---|
| Trigger | Manipulator + cable tasks need hydroelastic; 1.2.0 is 30x too weak |
| Monitor | Newton issue tracker for fix |
| Probability | Documented in open Newton issue |
| Impact | Medium -- affects later versions only |
| Fallback | Hold hydroelastic until Newton 1.3.x ships with fix; or use SDF collision as alternative |

---

## 7. Decision triggers

| Risk | Triggers a major architecture re-think if... |
|---|---|
| R1 | Newton-IsaacLab still experimental at GA |
| R2 | Warp 2.0 ships with non-trivial breaking changes |
| R6 | Single GPU is consistently the bottleneck for target tasks |
| R9 | Tier-0 ellipsoid model fails BlueROV2 validation at any common speed |
| R13 | Cannot procure validated reference data |
| R16 | Newton + custom pre-step hooks have a fundamental incompatibility |
