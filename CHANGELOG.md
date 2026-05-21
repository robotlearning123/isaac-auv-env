# Changelog

OceanScale website releases. Tag a version (`git tag v0.5.3`) to trigger production deploy.

## Unreleased

Wave 2 of v0.6 prep (engineering buildup; website unchanged from v0.5.3).

### Added
- **Hydro Tier-1 Fossen 6-DOF kernels** (`oceanscale/hydro/`) with W2-gate tests: autograd vs torch numerical-diff (<1e-3), throughput (≥100k env-steps/s @ 8192 envs target), determinism.
- **Von Benzon 2022 reference scaffold** (`oceanscale/validation/`) — Python interface + parity-test skeleton; unblocks R13 (no AUV ground-truth data) once Simulink port lands.
- **NVIDIA ecosystem benchmark matrix** (`benchmarks/`) — 28 scripts + ECOSYSTEM_MATRIX.md + VERIFICATION_REPORT.md across Warp / Newton / MuJoCo-Warp / CuPy / JAX / Taichi / Triton / XLB on RTX 5090.
- **RL training pipeline scaffold** (`oceanscale/{newton_env,rov_env,vec_env}.py`, `train.py`, `oceanscale/vehicles/` + tests).
- **Fluid module scaffold** (`oceanscale/fluid/{grid,mpm}.py` + tests).
- **Strategy / market / competitive / tech research docs** (`research/`, 19 files).
- **AGENTS.md** — cross-tool agent instructions (Claude Code, Codex, Gemini).
- **OCEAN_TECH_REFERENCE.md** — consolidated underwater sim tech reference.

### Changed
- **STATUS.md** + **HANDOFF.md** — 2026-05-21 review section: decisions (brand english-only, staff no-hurry, website concept-mainly) + next gates (W2 autograd, von Benzon scaffold).
- **CLAUDE.md** — orchestration / model strategy refresh.
- **pyproject.toml / uv.lock** — deps for hydro + RL stack.
- **.gitignore** — exclude tool state, build artifacts, large USD outputs.

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
