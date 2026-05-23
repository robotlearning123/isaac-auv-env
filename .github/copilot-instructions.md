# GitHub Copilot Instructions for OceanScale

This repo is configured to be agent-native. Read these files before suggesting code or copy.

## Authority

| File | Purpose |
|------|---------|
| `POSITIONING.md` | Brand law. Banned words list (§8) is enforced. Anchor sentence (§3) quoted verbatim. |
| `DESIGN.md` | Visual brand law. Banned visual patterns (§2) enforced. |
| `AGENTS.md` | Cross-tool agent instructions. |
| `ARCHITECTURE.md` | Technical architecture. |
| `GLOSSARY.md` | Project terminology. |

## Code style

- Python 3.12+, type hints on public APIs, `ruff format`. No needless comments.
- TypeScript / Astro: strict mode. Tailwind v4.
- Conventional Commits.
- One logical change per commit.
- Tests after every code change.

## Architecture context

- Newton + Warp (no Isaac Sim dependency).
- GPU-native, targets CUDA 12.8+, RTX 5090 reference.
- Bilingual zh/en website via Astro content collections.

## Repo layout (monorepo)

```
oceanscale/       Python GPU sim package (the simulator)
website/          Astro 6 landing page
benchmarks/       perf tests
notebooks/        Colab notebooks
tests/            pytest test suite
docs/             contributor docs
```

Each sub-package has its own `AGENTS.md` — read it before working in that directory.

## Don'ts

- Do not introduce defense / military framing.
- Do not add buzzwords without backing evidence.
- Do not paraphrase the anchor sentence or enemy line.
- Do not bypass the banned-words check on user-facing copy.
- Do not add features beyond the assigned task.
