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

## Next Priorities — 2026-05-21

1. **v0.6.0 ship** — concept-mainly website edits landing (content collections rewrite in progress, uncommitted — see git status); tag when shipped
2. **W2 autograd gate** — `tests/hydro/test_tier1_autograd.py` must pass before W3 start (kernels in `oceanscale/hydro/tier1*.py` written, autograd test pending)
3. **von Benzon reference dataset** — `validation/vonbenzon_reference.py` scaffold; generates ground-truth BlueROV2 trajectories from von Benzon 2022 Simulink model (R13 unblocker, blocks physics validation)
4. **GitHub full repo update** — push current working tree state to origin; many modified files across website + hydro

### Carried from prior session (v0.5.1 polish)

1. EN eyebrow wrap (mobile)
2. BuiltOn region logos
3. Demos featured card styling
4. Hero CTAs above the fold
5. Mobile pass

## Reference

- **Live**: https://oceanscale-web.pages.dev/
- **Full session report (CN)**: `artifacts/infra-migration-2026-05-19.html`
- **CF token retrieval**: `CF_TOKEN=$(op read 'op://Dev/Cloudflare Account Master Token/credential')`
