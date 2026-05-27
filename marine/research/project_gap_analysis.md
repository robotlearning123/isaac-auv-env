# OceanScale Project Gap Analysis

**Date:** 2026-05-20
**Analyst:** Top-down review of all project documentation, code, and website content.
**Scope:** Everything in `/home/robot/workspace/46-marine/` against stated project goals.

---

## 1. Current State Summary

### What exists

**Documentation (extensive, high quality):**
- `STACK.md` (~1013 lines) -- deep technical analysis of every stack layer, decision framework, GS-Playground evaluation, main-stack lock policy
- `DESIGN.md` (209 lines) -- architecture proposal, tiered hydro model, sensor design, roadmap
- `SURVEY.md` (219 lines) -- SOTA landscape with verified versions and source URLs
- `REFERENCES.md` (499 lines) -- comprehensive reference tracking with admission criteria, license verification, BibTeX
- `RISKS.md` (314 lines) -- 25-item risk register (R1-R25) with triggers, monitors, fallbacks
- `LAYOUT.md` (301 lines) -- full package layout plan with module responsibilities and import discipline
- `IMPLEMENTATION_PLAN.md` (274 lines) -- 4-week sprint plan with T1.1-T4.7 tasks and ship criteria
- `VERSIONS.md` (257 lines) -- version pinning rationale and tiered install plan
- `STATUS.md` (141 lines) -- real benchmark numbers, stack health, gotchas
- `VERSION` -- `0.5.0` (website)
- `CHANGELOG.md` -- website release history v0.1.0 through v0.5.0

**Code (minimal, scaffold stage):**
- `oceanscale/__init__.py` -- version `0.1.0`
- `oceanscale/hydro/__init__.py`, `tier1_kernels.py` (245 lines), `tier1.py` (247 lines) -- all 5 Fossen kernels + orchestrator class implemented
- `tests/` -- 8 test files: `test_environment.py`, `test_warp_smoke.py`, `test_newton_smoke.py`, `test_worlds_poc.py`, `hydro/test_tier1_smoke.py`, `hydro/test_tier1_autograd.py`
- `benchmarks/` -- 3 files: `kernel_throughput.py`, `newton_worlds_throughput.py`, `tier1_throughput.py`
- `pyproject.toml` -- dependencies pinned and resolving
- `THIRD_PARTY_NOTICES.md` -- structure ready, no entries yet

**Website (deployed, functional):**
- Astro 6 + Tailwind v4 landing page at `oceanscale-web.pages.dev`
- Full EN/ZH bilingual content: hero, manifesto, why, what, built-on, demos, benchmarks, stats, contact
- Tag-driven Cloudflare Pages deployment via GitHub Actions

### What is planned but not built

Per `IMPLEMENTATION_PLAN.md`, the project is at **Week 1 / G2 stage** (Newton-worlds investigation). The following major items remain:

| Item | Status | Target |
|---|---|---|
| BlueROV2 Heavy USD/MJCF asset | Not started | T3.2 (W3) |
| Hydro coefficient YAML | Not started | T3.3 (W3) |
| StationKeepingEnv (RL task) | Not started | T4.1 (W4) |
| PPO training pipeline | Not started | T4.4 (W4) |
| All sensor kernels (sonar, DVL, IMU, camera) | Not started | v0.2 |
| Isaac Lab integration | Not started | v0.2 |
| Free-surface waves (Tier 3) | Not started | v0.3 |
| Acoustic comms (BELLHOP) | Not started | v0.4 |
| Multi-agent MARL | Not started | v0.4 |
| Public release + paper | Not started | v0.5 |

**Directories from LAYOUT.md that do NOT exist:**
- `oceanscale/core/`, `sensors/`, `scene/`, `tasks/`, `randomize/`, `rl/`, `viz/`, `cli/`
- `examples/`, `assets/`, `configs/`, `scripts/`, `docs/`

---

## 2. Documentation Gaps

### 2.1 Outdated information in STACK.md

**GAP-D1: CUDA version mismatch (STACK.md line 728, 942 vs VERSIONS.md lines 50-52, STATUS.md line 2)**

`STACK.md` section 11 (line 728) states "CUDA 12.4" as the runtime. `VERSIONS.md` section 7 (line 249) already corrected this to CUDA 12.8 with a note: "this document supersedes with verified May 2026 facts." `STATUS.md` (line 2) confirms CUDA toolkit 12.8. The main stack lock table at line 942 also says "CUDA 12.4." These two lines in STACK.md were never patched despite the explicit intent to do so recorded in VERSIONS.md line 256 ("I'll patch STACK.md with these updates after this planning doc is approved").

**GAP-D2: PyTorch version mismatch (STACK.md line 729 vs STATUS.md line 5)**

`STACK.md` line 729 says "PyTorch 2.7 (CuDNN 9.7)". `STATUS.md` line 5 reports `torch 2.11.0+cu128` installed. The VERSIONS.md pin says `>=2.7.0,<3` which is compatible but the STACK.md text is stale.

**GAP-D3: Newton pin tighter than STACK.md states (pyproject.toml line 11 vs STACK.md line 929)**

`STACK.md` line 929 locks Newton at `>=1.2, pin tag`. `pyproject.toml` line 11 actually pins `>=1.2.0,<1.3` (tight upper bound per R19). `VERSIONS.md` line 62 also says `>=1.2.0,<2.0`. Three different pin ranges across three documents. The actual `pyproject.toml` is the truth; the docs disagree with each other.

**GAP-D4: DESIGN.md roadmap version labels (DESIGN.md line 178)**

DESIGN.md section 9 roadmap says "v0.5.0 -- Public release + paper" but VERSION is at 0.5.0 for the *website*, while the simulator package is at 0.1.0. This creates ambiguity about what "v0.5.0" refers to. The VERSION file tracks the website; `pyproject.toml` tracks the simulator. The roadmap in DESIGN.md conflates the two versioning streams.

### 2.2 Missing sections

**GAP-D5: No CONTRIBUTING.md (LAYOUT.md line 16 lists it)**

`LAYOUT.md` line 16 says `CONTRIBUTING.md -- added when repo opens for contributors`. It does not exist yet. Since the repo is not yet public, this is expected but should be tracked.

**GAP-D6: No `docs/math/fossen.md` (referenced in IMPLEMENTATION_PLAN.md T1.3, line 69)**

Task T1.3 says to write `docs/math/fossen.md` as the math reference for the Tier-1 kernel. The `docs/` directory exists but contains only `marinegym_fossen_extract.md` and `refs_tried.md`. The canonical math document is missing.

**GAP-D7: No `docs/` sphinx site (LAYOUT.md line 25)**

Planned for v0.3 per LAYOUT.md. Noted as expected gap.

**GAP-D8: HANDOFF.md exists but is not listed in CLAUDE.md, LAYOUT.md, or README.md**

`HANDOFF.md` is a session handoff document from 2026-05-19 but is not referenced in any of the documentation index files. It contains deployment workflow details that duplicate and extend CLAUDE.md.

### 2.3 Inconsistencies across documents

**GAP-D9: Camera model naming (STACK.md line 673-684 vs REFERENCES.md line 197, 461)**

`STACK.md` section 9.3 calls the camera model "Jaffe-McGlamery" throughout. `REFERENCES.md` section REF-OCEANSIM (line 197) contains an explicit correction: "STACK.md says 'Jaffe-McGlamery'; OceanSim actually uses **Akkaynak-Treibitz 2019**." The decision log at line 461 records the patch decision, but STACK.md was never updated. The sensor kernels in LAYOUT.md are named `vision_jmg.py` (line 59), perpetuating the wrong name.

**GAP-D10: Solver recommendation differs between DESIGN.md and STATUS.md**

`DESIGN.md` section 3 (line 36) lists "MuJoCo-Warp primary solver, Kamino VBD secondary." `STATUS.md` line 79 locks "v0.1 default solver = SolverSemiImplicit" based on benchmark evidence that SolverSemiImplicit is 40x faster for parallel free bodies. `IMPLEMENTATION_PLAN.md` line 261 agrees with STATUS.md. The DESIGN.md text was never patched.

**GAP-D11: RL framework choice differs across documents**

- `STACK.md` line 933: "skrl 2.0" as default RL backend
- `DESIGN.md` line 41: "skrl 2.0 (default), rl_games (throughput)"
- `IMPLEMENTATION_PLAN.md` line 193: "stable-baselines3 for v0.1" (due to rl_games psutil conflict)
- `REFERENCES.md` line 462: confirms "RL framework v0.1 = stable-baselines3"
- `pyproject.toml`: neither skrl, rl_games, nor sb3 is in dependencies (rl is commented out at line 33)

The docs acknowledge the shift (psutil conflict) but the main stack lock in STACK.md still lists skrl 2.0 as default without noting the v0.1 sb3 override.

**GAP-D12: Website what.ts claims incorrect Newton co-maintainers**

`website/src/content/en/what.ts` line 25 says Newton is "co-maintained by Anthropic, NVIDIA, Lightwheel, and Apple." `REFERENCES.md` line 83 and STACK.md line 309 say "NVIDIA + Google DeepMind + Disney Research; Linux Foundation governance." The website lists Anthropic (not DeepMind) and Apple (not Disney) and adds "Lightwheel" which is not mentioned anywhere in the technical docs as a Newton contributor. This is a factual error on the live website.

**GAP-D13: Website what.ts claims "90 M env-steps/s" but STATUS.md shows 221 M**

`website/src/content/en/what.ts` line 12 claims "90 M env-steps/s @ 8192 envs." `STATUS.md` line 67 shows the actual benchmark at 221 M env-steps/s for SolverSemiImplicit at 8192 envs. The website under-reports by 2.5x. This may be intentional (conservative claim with Tier-1 hydro overhead), but no such caveat is documented.

**GAP-D14: Website what.ts claims "11,435 FPS single-env" -- source untraceable**

The metric "11,435 FPS single-env BlueROV2 (RTX 5090)" in `what.ts` line 7 does not appear in STATUS.md or any benchmark file. STATUS.md reports 26.9 k steps/s at 8192 envs, which translates to ~220 M total but a different per-env metric. The origin of the 11,435 number is not documented.

---

## 3. Tech Stack Gaps

### 3.1 Referenced in DESIGN.md but not in STACK.md

**GAP-T1: "Kamino VBD secondary" (DESIGN.md line 36) has no dedicated section in STACK.md**

STACK.md section 3 mentions Kamino briefly (line 313) as "Disney's closed-loop chains" but provides no "What/Why/How/Where/When/Alternatives/Gotchas" analysis comparable to the other stack components. Given it is listed as a secondary solver in the architecture, it deserves at least a subsection.

**GAP-T2: BELLHOP (DESIGN.md line 40, STACK.md line 686-698) lacks implementation analysis**

BELLHOP is referenced as the acoustic comms model but there is no evaluation of which BELLHOP wrapper to use (the original Fortran, `arlpyp`, or a custom port), no license analysis for the BELLHOP source code itself, and no decision recorded in REFERENCES.md. The REFERENCES.md pending-verifications table does not mention BELLHOP licensing.

**GAP-T3: OpenUSD schema definitions (DESIGN.md line 92, STACK.md line 208) not designed**

Both documents mention custom schemas (`oceanscale:Hydrodynamics`, `oceanscale:Sonar`, `oceanscale:Current`) but no schema definition file, schema design document, or USD schema authoring plan exists. `LAYOUT.md` lists `hydro/schemas.py` and `sensors/schemas.py` but no design for what they contain.

### 3.2 Version mismatches between VERSIONS.md and actual state

**GAP-T4: VERSIONS.md section 2.1 says Python 3.12, but pyproject.toml allows 3.13**

`VERSIONS.md` line 43 specifies Python 3.12 via `uv venv --python 3.12`. `pyproject.toml` line 7 says `requires-python = ">=3.12,<3.14"`, which allows 3.13. This is a minor discrepancy -- the lower bound is the real constraint, and allowing 3.13 is forward-compatible, but the versioning rationale document should be updated to reflect the pyproject.toml range.

**GAP-T5: VERSIONS.md section 2.6 lists rl_games as v0.1 RL backend; pyproject.toml has it commented out**

`VERSIONS.md` line 74-76 says "rl_games >=1.6.0,<2.0 -- Battle-tested default for Isaac Lab." But `pyproject.toml` line 33-36 has the rl extra commented out entirely with a note about the psutil conflict. The actual v0.1 RL choice is stable-baselines3 (per IMPLEMENTATION_PLAN.md and REFERENCES.md decision log), which is not mentioned in VERSIONS.md at all.

**GAP-T6: `warp-lang` installed version (1.13.0) matches pins; but VERSIONS.md line 28 records user system at 1.9.0**

The venv has 1.13.0 (confirmed by STATUS.md line 4), which matches all pin specs. The VERSIONS.md note about the user's system Python having 1.9.0 is historical context, not a gap, but could be confusing.

---

## 4. Implementation Gaps

### 4.1 Items in IMPLEMENTATION_PLAN.md not yet done

**GAP-I1: Week 1 tasks partially complete**

Per STATUS.md, the following W1 tasks are done:
- T1.1 (Newton worlds API): done -- `test_worlds_poc.py` exists, `newton_worlds_throughput.py` benchmarked
- T1.2 (Throughput re-benchmark): done -- STATUS.md section 4 has corrected numbers
- T1.3 (6-DOF Fossen math sketch): unclear -- `docs/marinegym_fossen_extract.md` exists but the canonical `docs/math/fossen.md` referenced in T1.3 does not
- T1.4 (Docs patch): partially done -- STATUS.md lists 5 patches needed (lines 125-131), but several STACK.md inconsistencies identified in this analysis remain unpatched

**GAP-I2: Week 2 tasks (T2.1-T2.8) -- Tier-1 kernels are written but integration incomplete**

The Warp kernels in `oceanscale/hydro/tier1_kernels.py` implement all 5 forces (added-mass, damping, Coriolis, restoring, thruster alloc) plus helper kernels. `tier1.py` provides the orchestrator class. Tests exist (`test_tier1_smoke.py`, `test_tier1_autograd.py`). However:
- No integration with Newton's `state.body_f` via the actual pre-step hook mechanism is visible in test results
- `benchmarks/tier1_throughput.py` exists but its results are not recorded in STATUS.md
- The Week-2 gate criteria ("throughput >= 100k env-steps/s at N=8192") has not been recorded as met or failed

**GAP-I3: Week 3 tasks (T3.1-T3.5) -- not started**

No BlueROV2 Heavy asset, no hydro coefficient YAML, no scene builder, no end-to-end smoke test. Expected -- the project is between W1 and W2.

**GAP-I4: Week 4 tasks (T4.1-T4.7) -- not started**

No station-keeping env, no PPO training, no ONNX export. Expected.

### 4.2 Items in LAYOUT.md not yet created

**GAP-I5: Most package directories do not exist**

Per LAYOUT.md section 5, the v0.1.0 minimal scaffold should include:
- `oceanscale/core/` with `types.py`, `tape.py` -- MISSING
- `oceanscale/scene/vehicles/bluerov2_heavy.py` -- MISSING
- `tests/core/test_types.py` -- MISSING
- `tests/hydro/test_tier1_fossen.py` (numerical match vs MarineGym) -- MISSING (only smoke and autograd tests exist)
- `examples/01_hello_warp.py` -- MISSING

The `core/types.py` module is listed as a dependency of the hydro module in the import discipline (LAYOUT.md line 203: "core < depended on by all"). Its absence means the kernels operate directly on Warp primitives without the planned Pose/Twist/Wrench dataclasses.

**GAP-I6: benchmarks/compare_marinegym.py (LAYOUT.md line 149) not created**

Planned for v0.2 per LAYOUT.md but the file is listed in the tree. Minor -- not blocking.

**GAP-I7: No GitHub Actions CI workflows (LAYOUT.md lines 21-24)**

LAYOUT.md lists `.github/workflows/lint.yml`, `test-cpu.yml`, `test-gpu.yml` as "added when CI configured (v0.2)." A `.github/workflows/release.yml` exists for website deployment but no CI for the Python package.

---

## 5. Risk Gaps

### 5.1 Risks in RISKS.md needing updates

**GAP-R1: R3 (CUDA driver compatibility) -- driver version is stale**

RISKS.md line 42 notes "Currently 580.95.05 + CUDA toolkit 12.8 -- confirmed working." This is from 2026-05-15. Five days have passed; if any driver update occurred, this note should reflect it. Minor.

**GAP-R2: R9 (MuJoCo ellipsoid fluid accuracy) -- still at initial status**

R9 is marked red. No validation progress recorded since creation. The fallback notes say "Already planned -- Tier-1 Warp kernel ready by v0.2." Since Tier-1 kernels are now written (though not yet integrated), the risk mitigation status could be updated to reflect that the Warp kernels exist, even if not yet validated.

**GAP-R3: R19 (Newton sub-minor breaking changes) -- pin should be reflected in STATUS.md**

R19 (line 246) records the risk from Newton's rapid minor-version churn. The tight `<1.3` pin in pyproject.toml is the mitigation. However, STATUS.md section "What changes from STACK.md" (line 131) says to add R19 to RISKS.md, which was done, but STATUS.md itself was not updated to reflect the tight pin as a locked decision.

**GAP-R4: No risk covering the website content accuracy**

The live website contains factual errors about Newton co-maintainers (GAP-D12) and potentially stale performance numbers (GAP-D13-D14). There is no risk entry for "public-facing claims diverge from verified technical facts."

**GAP-R5: R20 (Warp autograd debugging) -- tests exist now, risk should be updated**

R20 says autograd correctness may burn 5 days. `tests/hydro/test_tier1_autograd.py` now exists and presumably passes (test count in STATUS.md is from before this file was added; git status shows it as a new file). The risk status should be updated if the autograd tests pass.

### 5.2 Missing risks given latest NVIDIA releases

**GAP-R6: No risk for Newton 1.3+ API changes to `replicate()` or `separate_worlds`**

STATUS.md section 4 (line 53-79) documents that the key architectural fix -- using `replicate(world_count=N)` to distribute bodies into independent worlds -- was essential to make SolverMuJoCo work. If Newton's `replicate()` API changes in 1.3+, this is a critical regression point. R19 covers general Newton breaking changes but does not specifically flag `replicate()` as load-bearing.

**GAP-R7: No risk for Isaac Sim 6.0 GA timeline slip affecting v0.2**

STACK.md section 6 (line 514) notes Isaac Sim 6.0 is "Early Developer Release (May 2026) -- GA expected later in 2026." If GA slips into 2027, the v0.2 Isaac Lab integration may need to proceed on the beta or fall back to Isaac Sim 5.1. No specific risk entry tracks this.

---

## 6. Reference Gaps

### 6.1 Projects/papers that should be in REFERENCES.md but are not

**GAP-REF1: Sim2Swim (arXiv 2512.08656) -- present, good**

Already in REFERENCES.md as REF-SIM2SWIM (line 64). No gap.

**GAP-REF2: OceanSim -- present, good**

Already in REFERENCES.md as REF-OCEANSIM (line 170-205). No gap.

**GAP-REF3: URoBench -- MISSING**

URoBench (Underwater Robot Benchmark) is a standardized benchmark suite for underwater perception and navigation tasks. It is not mentioned in SURVEY.md, REFERENCES.md, or DESIGN.md. Given that DESIGN.md section 7.2 proposes 8 benchmark tasks and SURVEY.md section 6 identifies "No community-wide benchmark suite" as a gap, URoBench (if it exists in 2026) should at minimum appear in the SURVEY.md landscape or REFERENCES.md pending-verifications. If it does not exist, the "open gap" claim is strengthened and should be explicitly confirmed.

**GAP-REF4: AegirJAX -- MISSING**

AegirJAX (or similar JAX-native underwater simulation projects emerging in 2025-2026) are not referenced. SURVEY.md lists JaxLrauv (line 194) but the broader category of JAX-native marine simulators may have new entrants. SURVEY.md should acknowledge whether a systematic search for JAX-native underwater simulators was done beyond JaxLrauv.

**GAP-REF5: No reference to NVIDIA Cosmos or other world-model-based simulation**

NVIDIA Cosmos (world foundation model for physical AI) was announced at CES 2025 and could be relevant as a future watchlist item for generating synthetic underwater training data. Not mentioned in STACK.md watchlist or REFERENCES.md.

**GAP-REF6: No reference to PlatoSim or other academic underwater simulators**

SURVEY.md's survey is comprehensive for the major GPU-native and ROS-based simulators but may miss smaller academic projects. The underwater simulation review paper cited (arXiv 2504.06245, line 193) would be the authoritative source for completeness; its bibliography should be cross-checked.

**GAP-REF7: MuJoCo 3.5 fluid model documentation -- cited but not verified in REFERENCES.md**

SURVEY.md line 19 and STACK.md line 367 reference MuJoCo's stateless fluid model with "Kutta lift, Magnus, ellipsoid drag" and "5 tunable params/geom." REFERENCES.md does not have a dedicated entry for MuJoCo itself (it is bundled with Newton but the fluid model is MuJoCo-specific). The fluid model documentation URL is listed in SURVEY.md line 159 but not in REFERENCES.md with a license/activity check.

**GAP-REF8: No reference to Open Robotics Gazebo Harmonic (post-classic)**

DAVE is mentioned as using "Gazebo Harmonic + ROS 2 Jazzy" (SURVEY.md line 39), but Gazebo Harmonic itself (the GPU-renovated successor to Gazebo Classic) is not evaluated as a potential physics backend or watchlist item. Given the ecosystem gravity toward USD, this may be correctly excluded, but the explicit why-not belongs in STACK.md alternatives or REFERENCES.md anti-references.

### 6.2 Reference verification gaps

**GAP-REF9: Multiple "TODO" pins in REFERENCES.md**

- REF-MARINEGYM pinned anchor: "TODO -- fetch latest main commit SHA at T3.3" (line 133)
- REF-OCEANSIM pinned anchor: "TODO at v0.2 sprint" (line 179)
- REF-BLUEROV2GZ: listed as "NO LICENSE" but no PR filed to request one (line 484)
- REF-MSS: "MIT (verify)" -- license not yet verified (line 58)
- REF-HOLOOCEAN: "MIT (verify)" -- license not yet verified (line 62)

These are tracked in the pending-verifications table (lines 475-487) so this is an acknowledged backlog, not an oversight.

---

## 7. Website / Content Gaps

### 7.1 Accuracy issues

**GAP-W1: Newton co-maintainers are wrong (en/what.ts line 25, zh/what.ts line 25)**

English version: "co-maintained by Anthropic, NVIDIA, Lightwheel, and Apple"
Chinese version: "Newton 物理由 Anthropic、NVIDIA 联合维护"

The correct list per REFERENCES.md is: **NVIDIA + Google DeepMind + Disney Research**, under Linux Foundation governance. "Anthropic" should be "Google DeepMind," "Apple" should be "Disney Research," and "Lightwheel" does not appear in any technical document as a Newton contributor. Both language versions need correction.

**GAP-W2: Performance claims may be stale or unverifiable (en/what.ts lines 7, 12)**

- "11,435 FPS single-env BlueROV2 (RTX 5090)" -- no benchmark in the repo produces this exact number. STATUS.md shows 26.9k steps/s at 8192 envs for free bodies, and MarineGym's published 250k FPS is on RTX 3060. The 11,435 number's origin is unclear.
- "90 M env-steps/s @ 8192 envs" -- STATUS.md shows 221 M for SolverSemiImplicit at 8192 envs (free bodies, no hydro). With Tier-1 hydro overhead, 90 M may be realistic but has not been benchmarked yet (`benchmarks/tier1_throughput.py` exists but results are not in STATUS.md).

**GAP-W3: Website implies capability that does not exist yet**

The landing page presents OceanScale as if the simulator is operational: "Full RL training pipeline," "8192 parallel environments," "6-DOF Custom Newton + SPH." In reality:
- No RL training has been run (PPO training is T4.4, not started)
- The 8192-env throughput is for free bodies without hydrodynamics (the Tier-1 kernels are written but not integrated into a Newton step loop at 8192 envs)
- SPH (Tier-2) is not implemented at all
- No sensor kernels exist (sonar, DVL, IMU, camera are all v0.2)

The website version (0.5.0) is far ahead of the simulator version (0.1.0 scaffold). This is a strategic choice (marketing site precedes product) but the performance claims should either be caveated or deferred until verified.

### 7.2 Content completeness

**GAP-W4: No "How it works" technical detail on the website**

The website has hero, manifesto, why, what, built-on, demos, benchmarks, stats, and contact sections. There is no "How it works" or "Architecture" section that explains the tiered hydro model, sensor pipeline, or RL integration at a technical level for the target audience (researchers and engineers). The `what.ts` capability cards are the closest but are metric-driven rather than explanatory.

**GAP-W5: Benchmarks section content not verified**

`website/src/content/en/benchmarks.md` and `benchmarks.ts` exist but were not read in this analysis. Given the accuracy issues in `what.ts`, these should be cross-checked against STATUS.md numbers.

**GAP-W6: No "Getting Started" or documentation link on the website**

For a project targeting researchers and engineers, the landing page has no link to documentation, GitHub repo, or getting-started guide. This is expected pre-release but should be planned for v0.5 (public release) or v0.2 (first usable version).

---

## 8. Priority Recommendations

### Top 5 things to fix/update immediately

**P1. Fix the website's Newton co-maintainer claim (GAP-D12, GAP-W1)**

Files: `website/src/content/en/what.ts` line 25, `website/src/content/zh/what.ts` line 25-26
Action: Change "Anthropic, NVIDIA, Lightwheel, and Apple" to "NVIDIA, Google DeepMind, and Disney Research" in EN. Change "Anthropic、NVIDIA" to "NVIDIA、Google DeepMind、Disney Research" in ZH. This is a live factual error on a deployed website.

**P2. Patch STACK.md CUDA version and solver text (GAP-D1, GAP-D2, GAP-D3, GAP-D10)**

Files: `STACK.md` lines 728, 729, 929, 942
Action: Update "CUDA 12.4" to "CUDA 12.8" at lines 728 and 942. Update "PyTorch 2.7" to "PyTorch >=2.7" at line 729. Add a note that v0.1 uses SolverSemiImplicit (not MuJoCo-Warp) as primary based on benchmark evidence. Update Newton pin to `>=1.2.0,<1.3` to match pyproject.toml. These were intended per VERSIONS.md line 256 but never executed.

**P3. Record Tier-1 throughput benchmarks in STATUS.md (GAP-I2, GAP-W2)**

File: `STATUS.md`
Action: Run `benchmarks/tier1_throughput.py` and record the results. This closes the Week-2 gate criterion ("throughput >= 100k env-steps/s at N=8192"). Also provides the basis for either validating or correcting the "90 M env-steps/s" website claim.

**P4. Update camera model naming from "Jaffe-McGlamery" to "Akkaynak-Treibitz" (GAP-D9)**

Files: `STACK.md` lines 673-684, `LAYOUT.md` line 59, `oceanscale/sensors/vision_jmg.py` (not yet created)
Action: The decision was made on 2026-05-15 (REFERENCES.md line 461) but never propagated to STACK.md. Correct the naming in STACK.md section 9.3 and update LAYOUT.md to name the file `vision_akt.py` or `vision_underwater.py` instead of `vision_jmg.py`.

**P5. Update RISKS.md R20 and R9 status to reflect Tier-1 kernel progress (GAP-R2, GAP-R5)**

File: `RISKS.md`
Action: R20 (Warp autograd debugging) -- if `test_tier1_autograd.py` passes, downgrade from red to yellow with a note. R9 (MuJoCo ellipsoid accuracy) -- note that Tier-1 Warp kernels now exist as the mitigation path, even if not yet validated. These updates keep the risk register honest and current.

---

## Appendix: Files Analyzed

| File | Lines | Status |
|---|---|---|
| `STACK.md` | ~1013 | Read in full |
| `DESIGN.md` | 209 | Read in full |
| `SURVEY.md` | 219 | Read in full |
| `REFERENCES.md` | 499 | Read in full |
| `RISKS.md` | 314 | Read in full |
| `LAYOUT.md` | 301 | Read in full |
| `VERSIONS.md` | 257 | Read in full |
| `STATUS.md` | 141 | Read in full |
| `IMPLEMENTATION_PLAN.md` | 274 | Read in full |
| `CLAUDE.md` | 35 | Read in full |
| `README.md` | 32 | Read in full |
| `CHANGELOG.md` | 36 | Read in full |
| `HANDOFF.md` | 39 | Read in full |
| `THIRD_PARTY_NOTICES.md` | 76 | Read in full |
| `pyproject.toml` | 78 | Read in full |
| `VERSION` | 1 | Read in full |
| `oceanscale/__init__.py` | 3 | Read in full |
| `oceanscale/hydro/tier1_kernels.py` | 245 | Read in full |
| `oceanscale/hydro/tier1.py` | 247 | Read in full |
| `website/src/content/en/hero.ts` | 5 | Read in full |
| `website/src/content/en/manifesto.ts` | 9 | Read in full |
| `website/src/content/en/what.ts` | 42 | Read in full |
| `website/src/content/en/why.ts` | 25 | Read in full |
| `website/src/content/zh/hero.ts` | 7 | Read in full |
| `website/src/content/zh/manifesto.ts` | 9 | Read in full |
| `website/src/content/zh/what.ts` | 42 | Read in full |
| `website/src/content/zh/why.ts` | 25 | Read in full |
| `website/src/pages/en/index.astro` | 27 | Read in full |

**Gaps identified: 39** (14 documentation, 6 tech stack, 7 implementation, 7 risk, 8 reference, 6 website)
**Priority actions: 5**
