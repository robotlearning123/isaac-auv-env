# Session Handoff — 2026-05-19

**Workspace**: `/home/robot/workspace/46-marine/`
**Branch**: `main`
**This session shipped**: **deploy infrastructure** — Cloudflare Pages migrated to Git integration, tag-driven release pipeline, account-wide CF API token in 1Password.

## What changed

1. **Git** — branch consolidation. 5 legacy branches deleted; `main` is now the single source of truth, both locally and on origin. GitHub default branch flipped to `main`.

2. **Cloudflare Pages** — `oceanscale-web` migrated from direct-upload to Git-integration with `robotlearning123/46-marine @ main`. Monorepo watch path set to `website/*` so non-website commits don't trigger builds. Production auto-deploy is **disabled**; tags trigger production via GitHub Actions.

3. **GitHub Actions** — `.github/workflows/release.yml` triggers on `v*.*.*` tag push. Reads `CLOUDFLARE_API_TOKEN` from repo secret, calls CF Pages API, polls deployment, verifies live URL. ~30s end-to-end.

4. **1Password** — account-wide CF API token (18 permissions) saved to Dev vault: `op://Dev/Cloudflare Account Master Token/credential`. Use in any session.

5. **Docs** — `CLAUDE.md`, `CHANGELOG.md`, `VERSION` created. `README.md` updated with deployment contract.

## Deploy workflow

```
git push origin main                # preview only (no production change)
git tag v0.5.1 && git push --tags   # → GH Action → CF API → live in ~30s
```

## v0.5.1 polish queued (from prior session §4)

1. EN eyebrow wrap (mobile)
2. BuiltOn region logos
3. Demos featured card styling
4. Hero CTAs above the fold
5. Mobile pass

## Reference

- **Live**: https://oceanscale-web.pages.dev/
- **Full session report (CN)**: `artifacts/infra-migration-2026-05-19.html`
- **CF token retrieval**: `CF_TOKEN=$(op read 'op://Dev/Cloudflare Account Master Token/credential')`
