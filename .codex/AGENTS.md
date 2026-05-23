# OceanScale — Codex Instructions

Codex CLI reads this file when invoked in this directory tree.

**Inherits root `AGENTS.md`.** Read in this order:

- `../AGENTS.md` — cross-tool agent instructions
- `../POSITIONING.md` — brand / scope / voice / lexicon law (banned words §8 enforced)
- `../DESIGN.md` — visual brand law (banned patterns §2 enforced)
- `../ARCHITECTURE.md` — technical architecture
- `../GLOSSARY.md` — terminology

## Codex-specific defaults

- Reasoning effort: `high` for review / design tasks, `medium` for implementation.
- Sandbox: read-only by default. Switch to `workspace-write` only when explicitly editing.
- Web search: `web_search_cached` enabled for up-to-date docs.
- Boundary: do not read `~/.claude/`, `.claude/skills/`, or `agents/openai.yaml` — those are Claude Code skill definitions for a different AI system. Stay focused on repository code.

## Project-specific reminders

- Newton + Warp, CUDA 12.8+, Python 3.12.
- Monorepo: `oceanscale/` (Python), `website/` (Astro), `benchmarks/`, `notebooks/`.
- Each sub-package may have its own `AGENTS.md` — read it before working in that directory.
- POSITIONING.md is messaging law. Banned words enforced.

## Useful commands

```bash
# Python package
uv sync --extra dev
uv run pytest tests/ -v
uv run ruff format oceanscale/ tests/

# Website
cd website && pnpm dev      # localhost:4321
cd website && pnpm build
```
