@AGENTS.md

# OceanScale — Claude Instructions

## Repo layout (monorepo)

- `website/` — Astro 6 + Tailwind v4 landing page (the deployed product)
- `oceanscale/` — Python GPU simulation package (Newton + Warp)
- `benchmarks/` — perf tests
- Docs: `STACK.md`, `DESIGN.md`, `SURVEY.md`, `REFERENCES.md`, `RISKS.md`

`VERSION` tracks the **website** version. The Python package has its own version in `pyproject.toml`.

## Deployment (tag-driven)

Cloudflare Pages production deploys are **disabled on auto-push**. Production only deploys on `v*.*.*` tag push via `.github/workflows/release.yml`.

```
# preview (push to any branch, including main)
git push origin main           # → preview URL only, production unchanged

# release
git tag v0.5.1 && git push --tags   # → GH Action → CF API → live in ~30s
```

Cloudflare project: `oceanscale-web` · production branch: `main` · monorepo watch path: `website/*`. Changes outside `website/` do not trigger builds.

Secret: `CLOUDFLARE_API_TOKEN` is set on the GitHub repo (sourced from 1Password `op://Dev/Cloudflare Account Master Token/credential`).

## Branch policy

Single `main` branch. Feature work via short-lived branches off main → preview URL → merge.

## Versioning

Semver `v0.x.x` for the website. Bump in `VERSION` + add CHANGELOG entry before tagging.
