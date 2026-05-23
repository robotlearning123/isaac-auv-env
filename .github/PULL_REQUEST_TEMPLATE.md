## Summary

<!-- 1-3 bullets: what changed and why -->

## Type

- [ ] feat (new functionality)
- [ ] fix (bug fix)
- [ ] docs (documentation only)
- [ ] refactor (no functional change)
- [ ] test (adds or fixes tests)
- [ ] chore (build, CI, repo config)

## Scope

- [ ] `oceanscale/` (Python package)
- [ ] `website/` (Astro app)
- [ ] `benchmarks/`
- [ ] `notebooks/`
- [ ] Repo-wide / docs / config

## Test plan

- [ ] `uv run pytest tests/ -v` passes
- [ ] (if website) `cd website && pnpm build` succeeds
- [ ] (if copy / visual) banned-words / banned-patterns checked

## Positioning impact

- [ ] No user-facing copy changes
- [ ] User-facing copy: derives from `POSITIONING.md` §§1–8, banned terms checked
- [ ] User-facing visuals: derives from `DESIGN.md` §§1–10, banned patterns checked

## Checklist

- [ ] `CHANGELOG.md` updated under `Unreleased`
- [ ] No credentials / tokens / `.env` committed
- [ ] One logical change per commit
