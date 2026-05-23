# OceanScale Roadmap

**Status:** Active. Version 0.0.x identity-reset baseline. Last revised 2026-05-22.
**Scheduling:** All dates are tentative. Items marked *exploratory* are not yet committed and may change or be dropped.
**Authority:** This file owns scheduling and sequencing. See cross-reference table at the end for which doc owns what.

---

## v0.0.x — Near-term (Q3–Q4 2026, tentative)

Identity-reset baseline. Ship a credible v0.1 SDK with one vehicle, one task, and honest benchmarks.

### SDK and tooling

- [x] Agent-native repo (per-tool configs: `.codex/`, `.gemini/`, `.cursor/`, `.github/copilot-instructions.md`)
- [x] `POSITIONING.md` v1.1 locked as brand law
- [x] `ARCHITECTURE.md` technical proposal
- [x] Tag-driven Cloudflare Pages deployment pipeline
- [ ] PyPI publish workflow verified end-to-end (`.github/workflows/publish-pypi.yml` exists, untested at release)
- [ ] Newton + Warp install path verified on Colab T4 (documented, not yet confirmed)

### BlueROV2 stabilization

- [x] Hydro Tier-1 Fossen 6-DOF Warp kernels with autograd gate tests
- [x] Von Benzon 2022 6-DOF reference port with regression tests
- [x] PPO hover policy converges (explained_variance 0.901 at 1M steps)
- [x] Headless MP4 renderer (single-panel + cinematic 4-panel)
- [x] CLI entry point: `oceanscale demo bluerov2-hover`
- [ ] Lateral drift in hover eval (~0.41m over 33s) — v0.2 target

### Benchmarks and validation

- [x] OceanScale vs PyBullet apples-to-apples benchmark (10.49x at n=64, RTX 5090)
- [x] NVIDIA ecosystem benchmark matrix (28 scripts)
- [ ] Sim-to-real validation with physical BlueROV2 (needs hardware partner)

### Demo and website

- [x] Landing page (Astro 6 + Tailwind v4, bilingual zh/en)
- [x] Demo section with install command, Colab badge, benchmark table
- [ ] Replace placeholder video with real render at release tag

### Customer discovery

- [ ] 5–10 customer interviews with underwater robotics operators
- [ ] 2-page operator/VC brief shipped (`docs/v0.1_brief.md` exists, needs interview input)

### Shipping criteria for v0.1

1. `pip install oceanscale && oceanscale demo bluerov2-hover` works on RTX 5090
2. Benchmark table on website matches reproducible `benchmarks/RESULTS.md`
3. At least 3 customer interviews completed
4. CHANGELOG entry + version tag

---

## v0.x — Medium-term (2027, tentative)

Expand from one vehicle and one task to a multi-vehicle, multi-environment platform. Items here derive from `ARCHITECTURE.md` §9 roadmap and `POSITIONING.md` §4 tier hierarchy.

### Vehicles (ARCHITECTURE.md §7.1)

- [ ] BlueROV2 Heavy (exploratory)
- [ ] IVER3-class AUV (exploratory)
- [ ] REMUS-class AUV (exploratory)

### Environments and physics (ARCHITECTURE.md §5)

- [ ] Tier-1 Fossen stabilization (lateral drift fix, thruster wash coupling)
- [ ] Tier-2 SPH fluid model for manipulator/thruster wakes (exploratory, gated on throughput)
- [ ] Free-surface waves (Gerstner/FFT Warp kernel)
- [ ] Time-varying current fields
- [ ] Domain randomization toolkit (mass, inertia, currents, turbidity, sensor noise)

### Sensors (ARCHITECTURE.md §6)

- [ ] Forward-looking sonar (Warp BVH ray cast)
- [ ] Sidescan and multibeam sonar
- [ ] Jaffe-McGlamery underwater vision pass
- [ ] DVL (4-beam Doppler, bottom-lock / water-track)
- [ ] Acoustic modem (BELLHOP wrapper)

### RL and benchmark tasks (ARCHITECTURE.md §7.2, §8)

- [ ] Station-keeping under current + wave
- [ ] Pipe / cable following with sonar
- [ ] Docking (visual + sonar, multi-stage)
- [ ] Multi-agent acoustic-localization MARL
- [ ] Bathymetric survey with sidescan
- [ ] ONNX export for every benchmark policy

### Throughput targets (ARCHITECTURE.md §8)

| Task | Target (single RTX 5090) | Status |
|------|--------------------------|--------|
| Station-keep, tier-0 | >=600k FPS | exploratory |
| Pipe-follow, tier-1 + sonar | >=80k FPS | exploratory |
| Multi-AUV MARL, tier-1 + comms | >=25k FPS | exploratory |
| Manipulator FSI, tier-2 | >=5k FPS | exploratory |

### Release milestones (from ARCHITECTURE.md §9)

| Version | Scope | Effort estimate |
|---------|-------|-----------------|
| v0.2 | Tier-1 Fossen stabilization, DVL, sonar v1, DR toolkit, MarineGym task parity | 6–8 weeks |
| v0.3 | Free-surface waves, currents, vision pass, sidescan, ONNX export | 8–10 weeks |
| v0.4 | Multi-agent, acoustic comms, MARL benchmarks, tether, manipulator FSI tier-2 | 8–10 weeks |
| v0.5 | Public release + paper; throughput targets met | 4–6 weeks |

---

## vN — Long-term (>2027, exploratory)

Directions that depend on evidence not yet available. Promotion gates from `POSITIONING.md` §4 and §9 apply.

### Tier-2: ocean world model (POSITIONING.md §4, §9.2)

**Promotion gate:** a shipped demo shows the learned layer improving simulator fidelity or transfer, with a public benchmark against a relevant non-learned baseline.

- [ ] Evidence: learned layer demonstrably improves fidelity or sim-to-real transfer
- [ ] Evidence: public benchmark vs non-learned baseline
- [ ] Promote "ocean world model" to homepage (POSITIONING.md §4 tier-2 row)

Until the gate is met, the world model stays "one click below the homepage" (POSITIONING.md §4). Technical exploration is fine; public positioning as a world-model company is not.

### Tier-3: ocean foundation model (POSITIONING.md §4)

**Promotion gate:** (a) data scale evidence, (b) model trained at meaningful parameter scale, (c) distribution leverage.

- [ ] Evidence: data pipeline at scale
- [ ] Evidence: model trained and evaluated
- [ ] Evidence: distribution leverage demonstrated

Until all three hold, "ocean foundation model" is research-note / private-deck only, with explicit distinction from tier-2 (POSITIONING.md §4, §8).

### Scope expansion (POSITIONING.md §9.1)

Current scope: underwater robotics (AUVs, ROVs). Expansion to broader marine robotics revisited when all three hold:

1. The same simulator primitive clearly serves underwater and surface systems
2. At least two serious commercial conversations pull toward USVs or broader autonomy
3. The category can expand without reviving banned language (POSITIONING.md §8)

---

## Cross-reference table

| Domain | Owner | Content |
|--------|-------|---------|
| Vision, category, scope, voice, lexicon, banned words, tier hierarchy, promotion gates | `POSITIONING.md` | Brand law. Downstream docs derive from it. |
| Technical architecture, physics tiers, sensor design, RL frontend, throughput targets | `ARCHITECTURE.md` | Engineering proposal. §9 has version-level scope. |
| Scheduling, sequencing, shipping criteria, status tracking | `ROADMAP.md` (this file) | What ships when. Marks tentative vs committed. |
| Version history, per-release changes | `CHANGELOG.md` | Keep-a-Changelog format. Ground truth for what shipped. |
| Current priorities, blockers, session state | `HANDOFF.md` | Operational. Updated each session. |
| Visual identity, color, typography | `DESIGN.md` | Visual brand law. |

---

## Changelog

- **2026-05-22** — Initial draft. Near-term items from `CHANGELOG.md` (unreleased + v0.0.1). Medium-term from `ARCHITECTURE.md` §9. Long-term from `POSITIONING.md` §4 and §9. All dates tentative.
