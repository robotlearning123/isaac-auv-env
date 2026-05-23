# `website/` Astro Landing Page — Agent Instructions

This sub-package is the OceanScale bilingual landing page (Cloudflare Pages production). Working here? Read this AGENTS.md *and* the root files:

- `../AGENTS.md` — cross-tool agent law
- `../POSITIONING.md` — brand law (anchor, scope, lexicon, banned words)
- `../DESIGN.md` — visual brand law (banned patterns, required patterns, wordmark rules)
- `../GLOSSARY.md` — terminology

## Stack

- Astro 6 + Tailwind v4
- Bilingual (zh / en) via content collections in `src/content/`
- Production deploy: Cloudflare Pages, tag-driven via `.github/workflows/release.yml`
- Preview deploys auto on every push to main

## Conventions

- **Content collections** live at `src/content/{en,zh}/`. Both languages are canonical; neither is a translation of the other (`../POSITIONING.md` §6).
- **Hero copy** uses Appendix A3 from POSITIONING.md (verbatim or close adaptation within §6 derivation rules).
- **About page** uses A4. Long-form vision page uses A6.
- **Anchor sentence** (`POSITIONING.md` §3) appears as H1; quote verbatim.
- **Enemy line** (`POSITIONING.md` §5) opens the vision narrative; quote verbatim.

## Bilingual hygiene

- CJK content uses `word-break: keep-all` to prevent compound-split.
- Insert ZWSP at compound boundaries in CJK headlines (e.g. `海洋机[ZWSP]器人`).
- Half-width Arabic digits in tables; full-width punctuation in body prose where culturally expected.
- Every EN string has a paired ZH string (release-blocking gap if missing).

## Visual rules (from DESIGN.md)

Banned on this surface (`../DESIGN.md` §2):
- Cinematic ocean depth imagery as hero
- Submarine silhouettes against blue gradients
- "Editorial deep ocean abyss" theme or similar romance aesthetics
- AI-generated marine imagery
- Defense / military iconography

Required (`../DESIGN.md` §3):
- Technical diagrams, terminal screenshots, benchmark tables
- Grid-locked layout, generous breathing room
- One accent color, monochrome base, no body-content gradients

## Build & deploy

```bash
pnpm install      # install deps
pnpm dev          # dev server at localhost:4321
pnpm build        # production build → dist/
pnpm preview      # preview production build locally
```

Production deploys are **tag-driven**. Push to `main` = preview only. Tag `v0.x.x` + push triggers production via `.github/workflows/release.yml`.

```bash
git tag v0.0.2 && git push --tags    # triggers production deploy
```

## Cloudflare Pages project

- Project: `oceanscale-web`
- Monorepo watch path: `website/*` (changes outside don't trigger builds)
- Account: see `op://Dev/Cloudflare Account Master Token/credential`
- Production URL: https://oceanscale-web.pages.dev/

## What NOT to do

- Do not introduce ocean romance imagery (`../DESIGN.md` §2 banned patterns).
- Do not break bilingual symmetry (every EN string needs a paired ZH).
- Do not paraphrase the anchor sentence or enemy line — quote verbatim.
- Do not add the `沧渊` Chinese wordmark (retired; brand is English-only).
- Do not add `next-gen` / `下一代` more than once per page.

## Cross-references

- `../POSITIONING.md` Appendix A — voice versions A1–A6 for surface mapping.
- `../DESIGN.md` — visual brand law.
- `wrangler.jsonc` — Cloudflare Pages config.
- `.github/workflows/release.yml` — tag-driven production deploy.
