# OceanScale (沧渊) — Agent Instructions

Cross-tool configuration for AI coding agents. Read by: Claude Code (via @AGENTS.md in CLAUDE.md), OpenAI Codex (native), Gemini CLI (via .gemini/settings.json).

## Project Overview

OceanScale is an AI-native marine robotics infrastructure platform. Greenfield GPU-native underwater simulator built on Newton + Warp + Isaac Sim.

- Monorepo: `website/` (Astro 6 landing page), `oceanscale/` (Python GPU sim), `benchmarks/` (perf tests)
- Docs: `STACK.md`, `DESIGN.md`, `SURVEY.md`, `REFERENCES.md`, `RISKS.md`
- `VERSION` file tracks the **website** version (semver `v0.x.x`)
- Python package version is in `pyproject.toml`

## Build & Run Commands

```bash
# Website
cd website && npm install && npm run dev        # dev server at localhost:4321
cd website && npm run build                      # production build

# Python package
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

# [46-marine] recent context, 2026-05-20 11:37pm EDT

No previous sessions found.
</claude-mem-context>