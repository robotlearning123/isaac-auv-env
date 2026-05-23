# Contributing to OceanScale

OceanScale is open source AI-native simulation infrastructure for underwater robotics. Both human and AI agent contributions are welcome.

## Read first

Before any contribution:

| File | Purpose |
|------|---------|
| `POSITIONING.md` | Brand law. Banned words list (§8) is enforced. |
| `DESIGN.md` | Visual brand law. Banned patterns (§2) are enforced. |
| `AGENTS.md` | Cross-tool AI agent instructions. |
| `ARCHITECTURE.md` | System architecture. |
| `GLOSSARY.md` | Project terminology. |

## Quick start

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv sync --extra dev          # Python deps (CUDA 12.8+)
cd website && pnpm install   # website deps
```

## Branching

- Single `main` branch.
- Short-lived feature branches off `main`. Open a PR when ready.
- Never force-push `main`. Never commit credentials, tokens, or `.env` files.

## Commit messages

Conventional Commits, one logical change per commit:

```
feat(scope): short summary
fix(scope): short summary
docs(scope): short summary
test: short summary
chore: short summary
refactor(scope): short summary
```

## Testing

- Run tests after every code change. Fix failures before committing.
- Bug fix: failing test first, then fix.
- Python: `uv run pytest tests/ -v`
- Website: `cd website && pnpm build` (build-time check)

## Pull request checklist

- [ ] Tests pass locally
- [ ] `CHANGELOG.md` updated under `Unreleased`
- [ ] Copy contributions: words on positioning, no banned terms (`POSITIONING.md` §8)
- [ ] Visual contributions: visuals on brand, no banned patterns (`DESIGN.md` §2)
- [ ] No defense / military framing on public surfaces

## AI agents contributing

- Read `AGENTS.md` first. Cross-tool instructions live there.
- Read `POSITIONING.md` and `DESIGN.md` before generating any user-facing copy or visuals.
- Enforce the banned-words list (`POSITIONING.md` §8) and banned-visuals list (`DESIGN.md` §2) before committing.
- This is a monorepo: sub-packages (`oceanscale/`, `website/`, `benchmarks/`, `notebooks/`) each have their own `AGENTS.md`. Read the relevant one before working in that directory.

## Reporting issues

- **Bug:** use the `Bug` issue template.
- **Feature:** use the `Feature request` issue template.
- **Question / discussion:** open a GitHub Discussion.

## Security

See `SECURITY.md`. Do not file security issues in public GitHub issues.

## Code of Conduct

See `CODE_OF_CONDUCT.md`.

## License

Apache-2.0. By contributing, you agree your contributions are licensed under the same.
