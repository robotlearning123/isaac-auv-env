# OceanScale Progress

Historical snapshot. The 2026-05-22 content below is retained for context and is not the current source of truth.

Current source-of-truth docs:

- `README.md` for the source quickstart.
- `docs/README.md` for the documentation map.
- `docs/verification.md` for the customer/new-user verification runbook.
- `docs/isaac6-isaaclab3-install.md` for the Isaac Sim 6 / Isaac Lab 3 lane.
- `DECISIONS.md` ADR-013 for the current Isaac 6 validation-lane decision.
- `STATUS.md` current addendum for the latest status boundary.

Current version identity: `0.1.0-alpha` in `VERSION` and `pyproject.toml`.
Public PyPI publishing is not verified end-to-end yet; use the source install path.

---

## 1. Identity

- **Version**: `0.0.1` (pre-product concept stage; `VERSION`, `pyproject.toml`)
- **Branch policy**: single `main`; short-lived feature branches
- **Positioning**: `POSITIONING.md` v1.1 — brand/scope/voice/lexicon law, locked
- **Visual brand**: `DESIGN.md` v1 — visual identity law, locked
- **Architecture**: `ARCHITECTURE.md` (renamed from prior `DESIGN.md`)
- **Repo**: `robotlearning123/oceanscale` on GitHub

## 2. Active branch

```
feat/v0.6-net-new-2026-05-21
```

## 3. Most recent commit

```
44334c3 feat(v0.1): cinematic renderer + retrained checkpoint + landing demo + brief — Wave-D
```

Full recent history (5 commits):

| SHA | Message |
|---|---|
| `44334c3` | Wave-D — cinematic renderer + checkpoint + landing demo + brief |
| `a2e7920` | Wave-C — OceanScale vs PyBullet benchmark |
| `39f8aa4` | Wave-B — pip-installable + CLI + Colab + bundled checkpoint |
| `e723e11` | Wave-A2 — PPO hover converges (T_matrix + VecNormalize fix) |
| `0d9961d` | Wave-A.A3 — von Benzon 2022 reference model port |

## 4. In flight / uncommitted

Grouped from `git status` on `feat/v0.6-net-new-2026-05-21`.

### 4a. Positioning & brand reset
- `POSITIONING.md` (new) — canonical brand law
- `DESIGN.md` (new, replaces old → `ARCHITECTURE.md`) — visual brand law
- `GLOSSARY.md` (new) — project-specific terms
- `DESIGN.md → ARCHITECTURE.md` rename

### 4b. Agent-native monorepo configs
- `.codex/`, `.context/`, `.cursor/`, `.windsurfrules` — per-tool agent configs
- `.github/copilot-instructions.md`, `.github/dependabot.yml`
- `.github/ISSUE_TEMPLATE/`, `.github/PULL_REQUEST_TEMPLATE.md`
- Per-package `AGENTS.md` + `README.md` (website, benchmarks, notebooks, oceanscale)

### 4c. Python package (Wave A–D, v0.1 feature work)
- `oceanscale/{cli,rov_env,vec_env}.py` — CLI entry point + env fixes
- `oceanscale/data/vec_normalize.pkl` → `.npz` — checkpoint format migration
- `pyproject.toml`, `tests/test_rov_env.py` — packaging + tests
- `benchmarks/RESULTS.md`, `notebooks/bluerov2_hover_colab.{py,ipynb}` — benchmark + Colab
- `scripts/` (new) — utility scripts

### 4d. Project governance
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md` (all new)
- `CHANGELOG.md`, `HANDOFF.md`, `STATUS.md`, `IMPLEMENTATION_PLAN.md` — updated
- `README.md`, `REFERENCES.md`, `STACK.md`, `OCEAN_TECH_REFERENCE.md`, `LAYOUT.md` — aligned to positioning
- `VERSION` — reset to `0.0.1`

### 4e. Website content
- `website/src/content/{en,zh}/demo.ts` — bilingual demo section
- `.github/workflows/release.yml` — tag-driven deploy pipeline

### 4f. Artifacts
- `artifacts/pr2-review-synthesis-2026-05-22.html` — cross-model PR review

## 5. Blockers

None currently.

**Top risk** (per STATUS.md tech review): no validated AUV reference data — cannot confirm physics correctness (R13). Partially unblocked by von Benzon 2022 reference port (`oceanscale/validation/`).

**Open questions**: `POSITIONING.md` §9 — Chinese name TBD (沧渊 retired, no replacement selected).

## 6. Next priorities

1. **Commit + push current branch** — massive uncommitted set across all 4b–4e categories; needs structured commit(s) before further work
2. **Next website production tag** — `VERSION` is `0.0.1` post identity reset; next semver-appropriate tag (`v0.0.2`+) triggers production deploy via `.github/workflows/release.yml`
3. **W2 autograd gate** — `tests/hydro/test_tier1_autograd.py` must pass before W3; kernels written, test pending
4. **von Benzon reference dataset** — scaffold exists at `oceanscale/validation/`; needs regression ground-truth generation to unblock R13
5. **Colab T4 verification** — notebook written but not end-to-end verified on Colab (Newton/Warp availability on T4 unconfirmed)

## Cross-references

- `STATUS.md` — historical status + hardware specs + benchmark numbers
- `ROADMAP.md` — future plan (if present)
- `CHANGELOG.md` — released versions + unreleased Wave A–D changelog
- `HANDOFF.md` — session handoff notes (last: 2026-05-19)
- `POSITIONING.md` — brand law (single source of truth for messaging)
