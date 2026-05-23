# OceanScale — Architecture Decision Records (ADRs)

ADR log. Each record: date, status, context, options considered, decision, rationale, consequences.

---

## ADR-001 — Brand positioning: Infrastructure category

**Date:** 2026-05-22
**Status:** Accepted

**Context:** OceanScale needed a category that anchors messaging and differentiates from existing underwater-sim and AI-lab competitors.

**Options considered:**
1. AI lab (Wayve / 1X / Physical Intelligence pattern)
2. Generic simulation tool (Gazebo / Stonefish / HoloOcean pattern)
3. Foundation-model company
4. Infrastructure company

**Decision:** OceanScale is an **infrastructure** company. Specifically: AI-native simulation infrastructure for underwater robotics.

**Rationale:** Source: `POSITIONING.md` §1. Voice models are Lightwheel, Modal Labs, Anthropic — compress, do not accumulate. An infrastructure framing owns the category rather than competing within a simulation-tool market, and avoids conflating the product with end-user AI capabilities.

**Consequences:** All external docs name the category as infrastructure. Downstream copy may not drift to "AI lab" or "simulation tool" framings.

---

## ADR-002 — Scope: underwater robotics (not marine robotics)

**Date:** 2026-05-22
**Status:** Accepted

**Context:** The domain scope needed precise boundaries to prevent positioning dilution.

**Options considered:**
1. Marine robotics (broader, includes surface vessels)
2. Ocean robotics
3. Underwater robotics (AUVs, ROVs, underwater vehicles only)

**Decision:** Scope is **underwater robotics**: AUVs, ROVs, underwater vehicles. Excludes surface vessels, USVs, autonomous ships, generic marine engineering, and defense/military.

**Rationale:** Source: `POSITIONING.md` §2. Test: if a sentence reads naturally with "marine engineering" or "marine robotics" replacing "underwater robotics", rewrite it. Broader scope dilutes the product story before the core is validated.

**Consequences:** "Marine robotics" is banned as a primary scope noun in headlines and canonical lines. Expansion to marine robotics is gated on three conditions in `POSITIONING.md` §9.1.

---

## ADR-003 — Concept hierarchy: simulator > world model > foundation model

**Date:** 2026-05-22
**Status:** Accepted

**Context:** OceanScale's architecture has multiple layers. Public messaging needed a tiered promotion system to avoid overclaiming.

**Options considered:**
1. Lead with "foundation model" (maximum hype)
2. Lead with "world model" (research credibility)
3. Tiered hierarchy with evidence-gated promotion

**Decision:** Three-tier hierarchy. Tier 1 (ocean simulator) = always public. Tier 2 (ocean world model) = promoted to homepage only when a shipped demo shows learned-layer improvement with a public benchmark. Tier 3 (ocean foundation model) = research notes only, promoted with data-scale + parameter-scale + distribution-leverage evidence.

**Rationale:** Source: `POSITIONING.md` §4. A simulator is a concrete product primitive. A world model is a deeper technical claim. A foundation model is a status claim. Promote each tier only when evidence catches up.

**Consequences:** Homepage and public copy default to "ocean simulator." World model appears one click below homepage. "Foundation model" is banned from headlines per `POSITIONING.md` §8.

---

## ADR-004 — Bilingual zh/en, both canonical

**Date:** 2026-05-22
**Status:** Accepted

**Context:** OceanScale targets a bilingual audience (Chinese researchers + international developers). Translation policy needed to prevent drift between language versions.

**Options considered:**
1. English canonical, Chinese derivative
2. Chinese canonical, English derivative
3. Both canonical, neither is a translation of the other

**Decision:** EN and ZH are both canonical. Neither is a translation of the other. Both are written for native readers and must convey the same category, scope, anchor, and enemy.

**Rationale:** Source: `POSITIONING.md` §6 "Translation rule." Bilingual asymmetry is a release-blocking bug. Audience priority per surface is mapped in §6.

**Consequences:** Every content-collection edit must verify both language versions. The `POSITIONING.md` Appendix A provides six canonical voice versions as reference implementations.

---

## ADR-005 — Anchor sentence verbatim policy

**Date:** 2026-05-22
**Status:** Accepted

**Context:** The anchor sentence appears across many surfaces (site title, homepage H1, deck cover, README, social bios). Risk of drift is high.

**Options considered:**
1. Freshen the anchor per surface for variety
2. Lock a single verbatim sentence across all surfaces

**Decision:** The anchor sentence is repeated verbatim everywhere it appears. Repetition is the point. Do not freshen for each surface.

**Rationale:** Source: `POSITIONING.md` §3. EN: "The ocean simulator for underwater robotics." ZH: "面向水下机器人的海洋仿真基础设施。" Brand recognition comes from consistency, not variety.

**Consequences:** Any surface that paraphrases the anchor is a bug. Derivation rules in §6 allow adaptation of surrounding copy, not the anchor itself.

---

## ADR-006 — No defense framing in public surfaces

**Date:** Pre-2026-05-22 (policy enforced since 2026-05-20)
**Status:** Accepted

**Context:** Defense terminology appeared in website copy during early drafts and was explicitly removed. It resurfaced during a later edit, causing user pushback.

**Options considered:**
1. Include defense as a target vertical
2. Mention defense only in private materials
3. Ban defense framing entirely from public-facing surfaces and fundraising materials

**Decision:** Defense / military framing is banned from all public-facing surfaces and fundraising materials. Banned terms: defense, military, naval, 防务, 国防, 军事, 军用, naval autonomy, maritime security, surveillance, dual-use.

**Rationale:** Source: `POSITIONING.md` §8 "Reputation / legal" section. Defense framing creates issues with export-controls, public PR, partner trust, and brand precision. (Rationale reinforced by feedback_no_defense_framing memory: user removed defense wording in commit `6576833` and pushed back when it resurfaced.)

**Consequences:** Every content-collection edit must audit for defense terms before committing. Applies to all `website/src/content/{en,zh}/*` and inline copy in `*.astro` components.

---

## ADR-007 — Newton + Warp stack, no Isaac Sim dependency

**Date:** Pre-2026-05-22 (decision crystallized 2026-05-15, confirmed 2026-05-21)
**Status:** Accepted

**Context:** OceanScale needs a GPU-native physics stack. NVIDIA's ecosystem offers Newton/Warp (lightweight) and Isaac Sim (heavyweight, full-featured).

**Options considered:**
1. Isaac Sim + Isaac Lab as primary stack (full NVIDIA ecosystem)
2. Newton + Warp only (lightweight, pip-installable)
3. MuJoCo / MJX direct
4. Dual-track: Newton primary, Isaac Sim deferred

**Decision:** Newton 1.x + Warp as the physics core. No Isaac Sim dependency for v0.1. SolverSemiImplicit is the default solver (40× faster than SolverMuJoCo for parallel envs on RTX 5090, per `STATUS.md` §4).

**Rationale:** Source: `ARCHITECTURE.md` §3 (tech stack rationale), `STATUS.md` §Decision (LOCKED 2026-05-15). SolverSemiImplicit achieves 221 M env-steps/s at 8192 envs on RTX 5090 — 2200× margin over the 100k target. Isaac Sim install friction is a known v0.1 risk (project_v01_product_shape memory: "Isaac Sim install friction → v0.1 must NOT depend on Isaac Sim").

**Consequences:** `pip install oceanscale` pulls Newton + Warp + SB3 only. Isaac Sim / Isaac Lab integration deferred to v0.2+. MuJoCo-Warp remains available as an alternative solver when richer physics is needed.

---

## ADR-008 — Version 0.0.x identity reset

**Date:** 2026-05-22
**Status:** Accepted

**Context:** Prior version tags (v0.1.0 through v0.5.3) tracked website iterations. The positioning reset required a clean version slate to signal pre-product concept stage.

**Options considered:**
1. Continue v0.5.x numbering from prior website work
2. Reset to v0.0.x to signal concept stage
3. Reset to v0.1.0 directly (skip 0.0.x)

**Decision:** Version line restarted at `0.0.1`. Earlier v0.5.x tags remain in git history for reference.

**Rationale:** Source: `CHANGELOG.md` v0.0.1 entry. The identity reset aligns version semantics with project state: POSITIONING.md was just locked as brand law, and the Python package is still in scaffold phase. A 0.0.x version honestly communicates pre-product status.

**Consequences:** `VERSION` file reset to `0.0.1`. `pyproject.toml` version reset to `0.0.1`. All GitHub URLs updated to `robotlearning123/oceanscale`. Prior website tags (v0.1.0–v0.5.3) are historical, not product versions.

---

## ADR-009 — Repo rename: 46-marine to oceanscale

**Date:** 2026-05-22
**Status:** Accepted

**Context:** The GitHub repository was named `46-marine` (a project number). A public-facing repo needed a name matching the brand.

**Options considered:**
1. Keep `46-marine` (internal naming convention)
2. Rename to `oceanscale` (brand-aligned)

**Decision:** Repository renamed on GitHub from `46-marine` to `oceanscale`. Local checkout path unchanged.

**Rationale:** Source: `CHANGELOG.md` v0.0.1 entry. The repo name is a public surface — it appears in PyPI metadata, Colab notebooks, README, and website demo links. A brand-aligned name reduces cognitive friction for first-time visitors.

**Consequences:** All GitHub URLs across the codebase updated to `robotlearning123/oceanscale`. Local checkout path remains `46-marine` (no action needed). Git remote updated automatically by GitHub on rename.

---

## ADR-010 — Agent-native monorepo with per-tool configs

**Date:** 2026-05-22
**Status:** Accepted

**Context:** OceanScale is developed with AI coding agents (Claude Code, OpenAI Codex, Gemini CLI, Cursor, Windsurf). Each tool has different config conventions.

**Options considered:**
1. Single CLAUDE.md only (Claude-centric)
2. Tool-agnostic AGENTS.md at root with per-tool adapters
3. No agent config (rely on code comments)

**Decision:** Agent-native monorepo. Root `AGENTS.md` is the single source of truth for project overview, build commands, code style, architecture constraints, testing rules, safety, and data integrity. Per-tool configs (`.codex/AGENTS.md`, `.gemini/{settings.json,GEMINI.md}`, `.cursor/rules/oceanscale.mdc`, `.windsurfrules`, `.github/copilot-instructions.md`) inherit from the root AGENTS.md.

**Rationale:** Source: `AGENTS.md` header ("Cross-tool configuration for AI coding agents"). A single source of truth avoids drift between agent configs. Per-tool adapters handle syntax differences without duplicating policy.

**Consequences:** Any policy change goes into root `AGENTS.md` first, then per-tool configs are updated. Each sub-package (`website/`, `oceanscale/`, `benchmarks/`, `notebooks/`) has its own `AGENTS.md` read before working in that directory.

---

## ADR-011 — Cross-model review mandatory for major artifacts

**Date:** 2026-05-22 (policy established 2026-05-21)
**Status:** Accepted

**Context:** Single-model review misses framing errors, factual inconsistencies, and structural issues. A notable incident occurred when one model's output passed self-review but failed user scrutiny.

**Options considered:**
1. Single-model self-review (faster)
2. Cross-model review only on user request
3. Mandatory cross-model discussion before every wave decision

**Decision:** Before every wave-level decision (prioritization, what to ship, what to defer), Opus and cx (GPT-5.x) must discuss together, synthesize agreement/disagreement, and present a joint view to the user.

**Rationale:** Source: feedback_opus_cx_discuss_always memory. User mandate: "let opus 4.7 & gpt 5.5 to discuss always, together, to decide, wave by wave." Cross-verify catches bad framing earlier when correction is free. Incident on 2026-05-21 where skipping cx led to user pushback ("fuck, cx is must").

**Consequences:** Every wave decision requires an cx consult before execution. Cost is ~30s–1min per wave. Override only when user explicitly says "skip cx" — ambiguous signals like "go" or "ship" do NOT override.

---

## ADR-012 — Doc split: verbal POSITIONING + visual DESIGN + technical ARCHITECTURE

**Date:** 2026-05-22
**Status:** Accepted

**Context:** The original `DESIGN.md` served triple duty: brand voice, visual identity, and technical architecture. This made it unclear which sections were brand law vs. technical reference.

**Options considered:**
1. Keep single DESIGN.md with labeled sections
2. Split into three documents with clear ownership

**Decision:** Split into three documents:
- `POSITIONING.md` — verbal brand law (category, scope, anchor, hierarchy, lexicon, banned words)
- `DESIGN.md` — visual brand law (colors, typography, layout, imagery rules)
- `ARCHITECTURE.md` — technical architecture (stack, module design, data flow)

**Rationale:** (Rationale inferred from file rename in git status: `R DESIGN.md -> ARCHITECTURE.md`, plus new `DESIGN.md` and `POSITIONING.md` as separate files.) Each document has a distinct authority level and audience. POSITIONING.md is brand law — downstream artifacts must follow it. DESIGN.md governs visual identity. ARCHITECTURE.md is a living technical reference.

**Consequences:** Each document has its own maintenance policy. POSITIONING.md changes require a changelog entry. DESIGN.md changes must stay consistent with POSITIONING.md's "infrastructure-first" visual identity rule (§6). ARCHITECTURE.md evolves with the codebase.

---

*End of ADR log. To add a new ADR, follow the format above: date, status, context, options considered, decision, rationale (with source citation), consequences.*
