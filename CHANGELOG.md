# Changelog

OceanScale website releases. Tag a version (`git tag v0.0.2`) to trigger production deploy.

## Unreleased

Wave A–D features are merged to main but unreleased. Next tag is v0.0.2.

### Added
- **Hydro Tier-1 Fossen 6-DOF kernels** (`oceanscale/hydro/`) with W2-gate tests: autograd vs torch numerical-diff (<1e-3), throughput, determinism.
- **Von Benzon 2022 reference port** (`oceanscale/validation/vonbenzon_reference.py`) — 6-DOF Fossen dynamics + RK4 integration + 6 regression tests against frozen snapshot. R13 (no AUV ground-truth reference) now partially unblocked.
- **NVIDIA ecosystem benchmark matrix** (`benchmarks/`) — 28 scripts + ECOSYSTEM_MATRIX.md + VERIFICATION_REPORT.md.
- **RL training pipeline** (`oceanscale/{newton_env,rov_env,vec_env}.py`, `train.py`, `oceanscale/vehicles/`). PPO hover converges with T_matrix.pinv() allocation + init perturbation + VecNormalize: explained_variance 0→0.901 at 1M steps, depth target hit to 0.7mm minimum (drifts ~0.41m laterally over a 33s eval — known v0.2 work).
- **OceanScale vs PyBullet benchmark** (`benchmarks/{bullet_bluerov_env,oceanscale_vs_bullet}.py` + RESULTS.md) — sweep n_envs ∈ {1,4,16,64} × 2 runs. 10.49× throughput at n=64 on RTX 5090, crossover at n≈8, 5 documented physics differences (no "identical conditions" overclaim).
- **CLI entry point** (`oceanscale/cli.py`) — `oceanscale demo bluerov2-hover`, `oceanscale train ...`, `oceanscale --version`. Pretrained checkpoint bundled at `oceanscale/data/bluerov2_station_keep_final.zip` (1M-step PPO) + `vec_normalize.pkl`.
- **Headless MP4 renderer** (`oceanscale/rendering.py`) — matplotlib Agg + imageio-ffmpeg, no GUI dep. Simple single-panel mode (default) + cinematic 4-panel mode (`--cinematic`: side view, top view, depth-error chart, thruster bars, HUD overlay).
- **Colab T4 notebook** (`notebooks/bluerov2_hover_colab.{py,ipynb}`) — install, smoke, train, eval. Real-T4 install path documented but not yet verified end-to-end (Newton/Warp on Colab).
- **PyPI publish workflow** (`.github/workflows/publish-pypi.yml`) — tag → TestPyPI → manual gate → PyPI. Tag/version match check.
- **Demo landing section** (`website/src/components/Demos.astro` + `content/{en,zh}/demo.ts`) — install command terminal, Colab badge, benchmark table, drift caveat. Bilingual. Placeholder video panel until R2 upload at release tag.
- **README.md** rewritten — quickstart, benchmark table, honest "works / known limitations" status.
- **2-page operator/VC brief** (`docs/v0.1_brief.md`) — pain → wedge → before/after → roadmap → ask. All claims cite measured data; cost numbers marked as public industry estimates.
- **Deep research report** (`deep-research-report.md`) — OceanSim Lab → Simulation Factory → Ocean AI Infrastructure (Chinese, 513 lines).
- **AGENTS.md** + **OCEAN_TECH_REFERENCE.md** — cross-tool agent instructions, consolidated tech reference.

### Changed
- **pyproject.toml / uv.lock** — base deps now include SB3 + imageio + matplotlib so the demo command runs after default `pip install` (no `[rl]` extra required). `[bench]` extra adds pybullet + psutil. PyPI metadata: name, classifiers, scripts entry point, package_data for bundled checkpoint.
- **.gitignore** — exceptions for `tests/**/data/` (regression snapshots) and `oceanscale/data/` (bundled checkpoint).
- **STATUS.md** + **HANDOFF.md** — 2026-05-21 review (brand english-only, staff no-hurry, website concept-mainly).
- **CLAUDE.md** — orchestration / model strategy refresh.

### Fixed
- **R20** (Warp quaternion autograd) — `_quat_rotate` torch reference rewritten as element-wise so autograd sees qw² dependency; `test_restoring_autograd_vs_torch` no longer skipped.
- **PPO hover** — three independent bugs (action mapping ignored T_matrix, passive buoyancy local optimum, reward scale crushed value head). Hover now actually learns. 150/150 tests pass.

## v0.0.1 — 2026-05-22

**Identity reset.** Positioning locked: OceanScale is AI-native simulation infrastructure for underwater robotics. Anchor: "The ocean simulator for underwater robotics." Six canonical voice versions in `POSITIONING.md` Appendix A. Cross-model approved (Opus 4.7 + cx GPT-5.x).

Version line restarted at `0.0.x` to signal pre-product concept stage. Earlier `v0.5.x` tags reflect prior website iterations and remain in git history for reference.

### Added
- `POSITIONING.md` — canonical brand/scope/voice/lexicon law (single source of truth for messaging).
- Repo renamed on GitHub: `46-marine` → `oceanscale`. Local checkout path unchanged.

### Changed
- `AGENTS.md` Project Overview aligned with positioning (underwater robotics, Newton + Warp, no Isaac Sim dep). Added Messaging Authority section pointing to POSITIONING.md.
- `pyproject.toml`: `version` reset to `0.0.1`, GitHub URLs updated.
- `VERSION` reset to `0.0.1`.
- `README.md`, `notebooks/bluerov2_hover_colab.py`, `website/src/content/{en,zh}/demo.ts`, `HANDOFF.md`, `LAYOUT.md`: GitHub URLs updated to `robotlearning123/oceanscale`.

## v0.5.3 — 2026-05-20

High-tech visual pass: rebuild the landing page around a stronger first screen, simulation cockpit treatment, architecture strips, benchmark proof band, and tighter AI infrastructure copy.

## v0.5.2 — 2026-05-20

Top-down website rewrite: reduce page flow to hero, challenge, platform, benchmarks, and contact; remove defense wording, stale stack claims, manifesto copy, demo detours, and repeated metrics.

## v0.5.1 — 2026-05-20

Remove the retired Chinese brand name from public website copy.

## v0.5.0 — 2026-05-16

Lightwheel-voice rewrite + UI/UX bento overhaul.

## v0.4.1 — 2026-05-16

Codex review fixes: ownable H1 + live telemetry ticker + full-screen sections + Built with.

## v0.4.0 — 2026-05-16

Cool factor: Hero video + cursor glow + card lift glow.

## v0.3.3 — 2026-05-16

Polish: renumber + Stats opacity + brighter sonar + Contact CTA + scroll-reveal.

## v0.3.2 — 2026-05-16

Every section becomes a full-screen poster with unique bg image (6 gpt-image-2 gens).

## v0.3.1 — 2026-05-16

Remove Beijing + each section becomes full-screen poster.

## v0.2.0 — 2026-05-16

Brand suite: logo, section bgs, rewritten copy.

## v0.1.0 — 2026-05-15

Initial landing page scaffold (Astro 6 + Tailwind v4).
