# OceanScale — Visual Design

**Status:** Canonical v1. Last revised 2026-05-22.
**Authority:** Visual companion to `POSITIONING.md`. Together they form OceanScale's brand law — POSITIONING governs the words, DESIGN governs the visuals. Downstream artifacts (website, deck, README badges, social cards) derive from these two files.
**Maintenance:** Changes happen here first. Downstream follows. Edits to identity / banned patterns / required patterns require a new Changelog entry.

This doc is design law, not a stylesheet. Implementation tokens (colors, fonts, spacing) live in `website/src/styles/global.css`.

---

## 1. Identity Anchor

OceanScale visuals are **infrastructure-first**. Not ocean romance.

**Voice models** (study for tone, not imitation):
- **Lightwheel** — robotics infrastructure. Grid-locked layout, restrained color, fixed-width type for technical accents.
- **Modal Labs** — compute infrastructure. High-contrast monochrome base, one accent color.
- **Anthropic** — AI infrastructure. Plain typography, generous whitespace, no marketing flourish.

**Anti-models** (do not echo):
- Cinematic ocean photography (sunset on the sea, submarine silhouette in deep blue)
- Adventure / dive aesthetic (GoPro, Garmin, dive-watch brands)
- Generic SaaS gradient + glow
- AI-slop renders (impossible-anatomy robots, dreamlike marine life)

---

## 2. Banned Visual Patterns

These ship reputation risk or category drift.

- Submarine silhouettes against blue gradients
- Cinematic ocean depth imagery as the hero
- Stock-photo divers, ROVs in dramatic lighting
- AI-generated marine imagery (octopuses, sharks, abstract ocean)
- Defense / military aesthetics — camo patterns, target reticles, naval iconography, sonar-blip motifs used decoratively
- Mystery / exploration animations (slow zooms into deep water)
- Decorative gradient meshes on body content
- Emoji decoration in UI chrome or marketing copy

---

## 3. Required Visual Patterns

These reinforce category as infrastructure.

- Technical diagrams (architecture, data flow, dimensional drawings)
- Terminal screenshots (CLI output, code blocks)
- Benchmark tables and charts (axis-labeled, no decoration)
- Component close-ups (one specific thing, photographed flat)
- Code as content (syntax-highlighted, monospace)
- Schematic illustrations (line-only, no painting)

---

## 4. Layout Principles

- **Grid-locked.** Predictable rhythm. Vertical stack on mobile, two/three columns on desktop.
- **Reading measure capped at ~70ch.** Body content does not span the viewport.
- **Generous vertical breathing room.** Section breaks are obvious without resorting to graphics.
- **Visible structure on cards / panels.** Borders, not just spacing — readers should see the system.
- **No center-aligned body paragraphs.** Left-aligned (LTR) or culturally appropriate.

---

## 5. Color

Implementation tokens: `website/src/styles/global.css`.

- One accent color at a time. Accent reserved for state (link, CTA, highlight).
- Monochrome base — high-contrast text on near-black or near-white surface.
- No gradients on body text or background. Solid surfaces only.
- Ocean-themed blues that match cinematic ocean photography aesthetics are off-palette.

---

## 6. Typography

Implementation: `website/src/styles/global.css`.

- Sans-serif body (geometric workhorse — Inter, IBM Plex Sans, or similar).
- Monospace for code, numbers, technical metadata.
- One display weight on a page maximum.
- ZWSP markers in CJK headlines to prevent compound-split (e.g. `海洋机[ZWSP]器人` must not break as `机|器人`).

---

## 7. Iconography

- Line icons only, single weight (1.5px or 2px).
- Technical glyphs: gear, terminal, package, graph, GPU chip, file tree.
- **Banned:** filled cartoon icons, character mascots, illustration sets, emoji.

---

## 8. Wordmark Rules

- `OceanScale` is one word, two caps: `O` and `S`.
- Never split (`Ocean Scale`, `OCEAN SCALE`). Lowercase `oceanscale` allowed only in URLs / repo names / package names.
- Never abbreviate (`OS`, `O/S`).
- Monospace wordmark allowed in technical contexts (code blocks, terminal headers).
- Sans-serif wordmark default everywhere else.

---

## 9. Photography / Imagery

**Allowed:**
- Photographed hardware close-ups (single ROV, AUV, thruster, sensor module)
- Lab GPU rigs, terminal screenshots, simulator UI captures
- Engineering diagrams (CAD-style, dimensional, schematic)

**Banned:**
- Cinematic ocean photography (open ocean, sunsets, depth scenes)
- Divers, salvage scenes, adventure aesthetic
- AI-generated marine imagery

---

## 10. Bilingual / CJK Considerations

- CJK content uses `word-break: keep-all` so compound words do not split mid-line.
- Insert ZWSP at compound boundaries in headlines.
- Number alignment in tables: half-width Arabic digits; full-width punctuation where culturally expected.
- Font fallback: latin font primary, system CJK fallback.

---

## 11. Cross-References

- `POSITIONING.md` — verbal brand law (anchor sentence, scope, lexicon, banned words).
- `website/src/styles/global.css` — implementation tokens.
- `website/src/components/` — component patterns.

---

## Changelog

- **2026-05-22 v1** — Initial visual brand law. Companion to `POSITIONING.md` v1.1. Root architecture proposal (formerly named `DESIGN.md`) renamed to `ARCHITECTURE.md`; this file (`DESIGN.md`) is the new visual brand canon.
