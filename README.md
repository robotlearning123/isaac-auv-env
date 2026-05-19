# OceanScale

GPU-native fluid-structure simulation for underwater robotics.

**Status:** website v0.5.0 live at [oceanscale-web.pages.dev](https://oceanscale-web.pages.dev/) · simulator at v0.1.0 (early scaffold).

## Layout

- `website/` — Astro 6 landing page (deployed)
- `oceanscale/` — Python simulation package
- `benchmarks/` — perf tests
- `STACK.md`, `DESIGN.md`, `SURVEY.md`, `VERSIONS.md` — design docs

## Deployment

Cloudflare Pages, **tag-driven**:

```
git push origin main                # → preview URL only
git tag v0.5.1 && git push --tags   # → production deploy
```

See `CLAUDE.md` for the full deploy contract.

## Develop website locally

```
cd website
npm install
npm run dev    # http://127.0.0.1:4321
```
