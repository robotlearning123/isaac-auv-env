# OceanScale — Agent Instructions

Cross-tool configuration for AI coding agents. Read by: Claude Code (via @AGENTS.md in CLAUDE.md), OpenAI Codex (native), Gemini CLI (via .gemini/settings.json).

## Project Overview

OceanScale is the ocean simulator for underwater robotics. It is the ocean layer in the NVIDIA robotics simulation ecosystem: Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch.

- Monorepo: `oceanscale/` (Python GPU sim), `website/` (Astro 6 landing page), `benchmarks/` (perf tests), `notebooks/` (Colab notebooks), `tests/`, `docs/`
- Each sub-package has its own `AGENTS.md` — read it before working in that directory
- Docs: `docs/README.md` (documentation map), `docs/getting-started.md` (source install), `docs/verification.md` (customer/new-user verification runbook), `docs/isaac6-isaaclab3-install.md` (Isaac 6 lane), `POSITIONING.md` (brand law), `DESIGN.md` (visual brand law), `ARCHITECTURE.md` (technical architecture), `STACK.md`, `RISKS.md`, `GLOSSARY.md`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`
- `VERSION` tracks project identity (currently `0.1.0-alpha`). Python package metadata is in `pyproject.toml`; website package metadata is in `website/package.json`
- Agent configs: `.codex/AGENTS.md`, `.gemini/{settings.json,GEMINI.md}`, `.cursor/rules/oceanscale.mdc`, `.windsurfrules`, `.github/copilot-instructions.md` — all inherit from this root `AGENTS.md`

## Messaging Authority

`POSITIONING.md` is the canonical source for category, scope, anchor sentence, concept hierarchy, enemy framing, lexicon, and banned words. Any agent generating user-facing copy reads `POSITIONING.md` §§1–8 before drafting. Banned words list is enforced. Anchor sentence (§3) and enemy line (§5) are quoted verbatim.

## Build & Run Commands

```bash
# Website
cd website && pnpm install && pnpm dev        # dev server at localhost:4321
cd website && pnpm build                       # production build

# Python package
uv python pin 3.12
uv sync --extra dev                              # install all deps (CUDA 12.8)
uv run pytest tests/ -v                          # run test suite
uv run python benchmarks/kernel_throughput.py    # GPU benchmarks
```

## Code Style

- Python: type hints on public APIs, no unnecessary comments, `ruff format`
- TypeScript/ Astro: strict mode, no emoji unless explicitly requested
- Commit messages: Conventional Commits (`feat(scope):`, `fix(scope):`, `test:`, `ci:`, `docs:`, `chore:`)
- One logical change per commit

## Architecture Constraints

- Website: Astro 6 + Tailwind v4, bilingual (zh/en) via content collections in `src/content/`
- Python: Newton 1.2.0 + Warp (CUDA 12.8), Python 3.12, `uv` for package management
- Isaac lane: Isaac Sim 6 / Isaac Lab 3 validation env remains separate from the OceanScale `.venv`; use `docs/isaac6-isaaclab3-install.md`
- GPU: targets RTX 5090, CUDA 12.8+
- Deployment: Cloudflare Pages, tag-driven (`v*.*.*` tags trigger production deploy)

## Testing Rules

- Run tests after every code change. Fix before committing.
- Bug fix: write failing test FIRST, then fix.
- Test pyramid: unit → BDD → integration

## Safety

- Never force push to main
- Never commit credentials, tokens, or .env files
- All development in short-lived branches off main
- Never delete data files without confirmation

## Data Integrity

- Never fabricate URLs, version strings, API signatures, or file paths
- Verify claims against actual code/docs before stating as fact
- Report real measurements only — no interpolation between data points

## What NOT To Do

- Do not add features, refactor, or introduce abstractions beyond what the task requires
- Do not add comments explaining WHAT the code does — well-named identifiers suffice
- Do not create documentation files unless explicitly requested
- Do not expand scope beyond the assigned task

<claude-mem-context>
# Memory Context

# [46-marine] recent context, 2026-05-26 1:22pm EDT

No previous sessions found.
</claude-mem-context>
