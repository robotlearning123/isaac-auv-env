# Brand Contamination Audit — OceanScale Website

Generated: 2026-05-23. Measures current `website/` against root `POSITIONING.md` v1.1 and `DESIGN.md` v1.

---

## Banned Copy (file:line — exact text)

| File | Line | Violation |
|------|------|-----------|
| `website/src/content/en/what.ts` | 19 | `metricLabel: "03 / Marine Physics"` — "Marine Physics" banned scope noun |
| `website/src/content/zh/what.ts` | 19 | `metricLabel: "03 / Marine Physics"` — same (ZH content file) |
| `website/src/content/en/what.md` | 15 | `title: "Marine physics"` — banned scope noun |
| `website/src/components/Why.astro` | 20 | `Iteration cost / ocean robotics AI` — "ocean robotics" banned |
| `website/src/i18n/ui.ts` | 14 | `'hero.sonarLabel': '基准 · RTX 5090'` — key name uses "sonar" |
| `website/src/i18n/ui.ts` | 25 | `'hero.sonarLabel': 'BENCHMARK · RTX 5090'` — key name uses "sonar" |
| `website/src/components/Hero.astro` | 72 | `{t('hero.sonarLabel')}` — references sonar-named key |
| `website/src/components/Manifesto.astro` | 14 | `Decorative sonar-trace gridlines` — sonar reference in comment |
| `website/DESIGN.md` | 18 | `Deep-Sea Sonar + SpaceX-style mission energy` — entire visual direction banned |
| `website/DESIGN.md` | 24 | `sonar ping and the pupil of an ocean creature` — ocean creature imagery |
| `website/DESIGN.md` | 159 | `Mission-driven (SpaceX/Anthropic style)` — "mission" framing |

## Banned Visual Tokens (file:line — CSS property)

### Color Token System (global.css — ALL need replacement)

| File | Line | Token | Issue |
|------|------|-------|-------|
| `global.css` | 4 | `--color-abyss-0: #02050C` | Ocean-themed name |
| `global.css` | 5 | `--color-abyss-1: #071019` | Ocean-themed name |
| `global.css` | 6 | `--color-abyss-2: #101C27` | Ocean-themed name |
| `global.css` | 10 | `--color-acid: #6EFFD1` | Cyan/teal accent — needs monochrome replacement |
| `global.css` | 11 | `--color-lumen: #6EFFD1` | Duplicate of acid, ocean-themed name |
| `global.css` | 12 | `--color-ember: #FF7A38` | Second accent — DESIGN.md allows ONE accent only |

### Body/Section Gradients (ALL violate "no gradients on body/bg")

| File | Line | Pattern |
|------|------|---------|
| `global.css` | 35-37 | Body `background`: radial cyan + radial ember + linear gradient |
| `global.css` | 47-48 | `tech-grid::before`: cyan gridlines |
| `global.css` | 115-117 | `sim-cockpit::before`: cyan + ember diagonal gradients |
| `global.css` | 126-127 | `sim-cockpit`: radial cyan gradient |
| `global.css` | 148-158 | `.depth-map`: full sonar depth map with cyan/ember gradients |
| `global.css` | 177-180 | `.depth-map__scan`: cyan scan line animation |
| `global.css` | 194-197 | `.route-a`, `.route-b`: cyan + ember route lines |
| `global.css` | 269 | Additional section gradient |
| `Hero.astro` | 32 | Radial cyan gradient overlay |
| `Hero.astro` | 34 | `bg-gradient-to-t` from abyss |
| `Why.astro` | 13 | `bg-gradient-to-br` from abyss to steel |
| `What.astro` | 15 | Linear + radial ember gradient |
| `Benchmarks.astro` | 18 | `bg-gradient-to-br` from abyss to steel |
| `Demos.astro` | 12 | `bg-gradient-to-b` from abyss |
| `Demos.astro` | 22 | `bg-gradient-to-br` from abyss |
| `Contact.astro` | 13 | Linear + radial cyan gradient |

### Sonar/HUD Elements (ALL banned — ocean romance)

| File | Line | Element |
|------|------|---------|
| `global.css` | 148-160 | `.depth-map` — sonar depth map |
| `global.css` | 162-175 | `.depth-map::before/after` — crosshair overlays |
| `global.css` | 177-182 | `.depth-map__scan` — scanning line animation |
| `global.css` | 188-197 | `.route-a`, `.route-b` — sonar route lines |
| `global.css` | 199-218 | `.vehicle` — diamond vehicle glyph + glow |
| `global.css` | 219-229 | `.ping` — sonar ping dots |
| `Hero.astro` | 76-83 | `depth-map`, `vehicle`, `ping` HTML elements |
| `Hero.astro` | 89 | Acid glow `shadow-[0_0_16px_var(--color-acid)]` |

### Glow Effects (banned per "no filled icons, no glow")

| File | Line | Pattern |
|------|------|---------|
| `global.css` | 80 | `.btn-lumen:hover` `box-shadow: 0 8px 32px -8px var(--color-acid)` |
| `global.css` | 216 | `.vehicle::after` `box-shadow: 0 0 20px var(--color-acid)` |
| `Hero.astro` | 89 | `shadow-[0_0_16px_var(--color-acid)]` |

**Total: 64 lines in global.css alone contain banned visual patterns.**

## Assets Inventory (file — status)

| File | Status | Reason |
|------|--------|--------|
| `public/favicon.ico` | **REPLACE** | Likely sonar-themed per stale DESIGN.md |
| `public/favicon.svg` | **REPLACE** | Stale DESIGN.md says "sonar rings, cyan" |
| `public/images/hero-bg.jpg` | **DELETE** | Cinematic ocean background — banned imagery |
| `public/images/what-bg.jpg` | **DELETE** | Section ocean background |
| `public/images/why-bg.jpg` | **DELETE** | Section ocean background |
| `public/images/benchmarks-bg.jpg` | **DELETE** | Section ocean background |
| `public/images/demos-bg.jpg` | **DELETE** | Section ocean background |
| `public/images/contact-bg.jpg` | **DELETE** | Section ocean background |
| `public/images/builton-bg.jpg` | **DELETE** | Section ocean background (dead component) |
| `public/images/manifesto-bg.jpg` | **DELETE** | Section ocean background (dead component) |
| `public/images/stats-bg.jpg` | **DELETE** | Section ocean background (dead component) |
| `public/images/logo.png` | **REPLACE** | Likely sonar-themed mark |
| `public/og-image.jpg` | **REPLACE** | Social preview — needs infrastructure treatment |
| `public/videos/hero-ambient.mp4` | **DELETE** | Cinematic ocean hero video — banned |
| `public/videos/auv-demo.mp4` | **KEEP** | Actual demo content |
| `public/videos/bluerov2-demo.mp4` | **KEEP** | Actual demo content |
| `public/videos/sph-demo.mp4` | **KEEP** | Actual demo content |

**Summary: 11 DELETE, 3 REPLACE, 3 KEEP**

## Dead Components (component — usage)

| Component | Status | Action |
|-----------|--------|--------|
| `BuiltOn.astro` | **DEAD** — not imported by any page | DELETE |
| `Manifesto.astro` | **DEAD** — not imported by any page | DELETE |
| `Stats.astro` | **DEAD** — not imported by any page | DELETE |
| `Nav.astro` | INDIRECT — imported by layout only | KEEP (rebuild) |
| `Footer.astro` | INDIRECT — imported by layout only | KEEP (rebuild) |

## Metadata Violations (file:line — current value)

| File | Line | Field | Current Value | Issue |
|------|------|-------|---------------|-------|
| `BaseLayout.astro` | 23 | `og:image` | `/og-image.jpg` | Points to ocean-themed OG image |
| `en/index.astro` | 12 | `description` | `"OceanScale builds GPU-native simulation..."` | OK but should use A2 canonical form |
| `index.astro` | 12 | `description` | `"OceanScale 构建面向水下机器人的..."` | OK but should use A2 canonical form |

Page titles are compliant — they use the anchor sentence.

## Stale website/DESIGN.md Conflicts

| Line | Stale Doc Says | Root DESIGN.md Says |
|------|---------------|---------------------|
| 18 | Visual direction: "Deep-Sea Sonar + SpaceX-style mission energy" | Infrastructure-first, not ocean romance |
| 24 | Mark: "sonar ping and ocean creature pupil" | Line-only technical glyphs (gear, terminal, GPU chip) |
| 28 | Generated by gpt-image-2, cyan-on-transparent | No AI-generated marine imagery |
| 34 | "bioluminescent cyan #4DEEEA only" | One accent color, no cinematic ocean blues |
| 49-51 | `--color-abyss-*` token names | Monochrome base, no ocean-themed naming |
| 53 | `--color-lumen: #4DEEEA` — "bioluminescent cyan" | One accent, infrastructure tone |
| 59 | "gradient evokes depth + bioluminescence" | No gradients on body/bg |
| 61 | "Cyan chosen for biological feel (real sea-life)" | Infrastructure, not bio/ocean romance |
| 82 | "Geist 800 + SpaceX impact" | Inter or IBM Plex Sans |
| 94 | "Hero min-height: 88vh (cinematic)" | Not cinematic |
| 120-124 | `.btn-lumen`, `.sonar-ping` class names | Infrastructure nomenclature |
| 133 | Imagery: "Cinematic deep ocean abyss" | Hardware close-ups, terminal screenshots, benchmark tables |
| 153 | Section background: "deep ocean abyss with bioluminescent particles" | No cinematic ocean photos |
| 159 | Tone: "Mission-driven" | Infrastructure tone (Lightwheel, Modal Labs, Anthropic) |
| 210 | favicon.svg: "sonar rings, cyan" | Line-only technical mark |

**Verdict: `website/DESIGN.md` conflicts with root `DESIGN.md` on virtually every design decision. It is the primary source of old-brand contamination and must be deleted.**

---

## Summary

| Category | Count |
|----------|-------|
| Banned copy violations | 11 |
| Banned visual token lines (global.css) | 64 |
| Banned visual patterns in components | 30+ |
| Assets to delete | 11 |
| Assets to replace | 3 |
| Dead components | 3 |
| Stale DESIGN.md conflicts | 15+ |

**The old "Deep-Sea Sonar" identity is embedded in every layer: CSS tokens, component markup, content files, i18n keys, public assets, metadata, and the stale local design doc. A CSS-only patch will not fix this. Clean-sheet presentation rebuild required.**
