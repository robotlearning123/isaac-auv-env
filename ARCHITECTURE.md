# OceanScale — Architecture

**Status:** Current v0.0.1 shipping stack. Last revised 2026-05-23.
**Authority:** This file describes what OceanScale ACTUALLY ships today. For the original architectural ambitions, see ARCHITECTURE_PROPOSAL_2026-05-15.md (archived). For positioning law, see POSITIONING.md.

---

## Stack (current v0.0.1)

| Layer | Choice | Version |
|-------|--------|---------|
| Physics core | Newton (GPU-native, USD-native) | >=1.2.0,<1.3 |
| GPU kernels | NVIDIA Warp (Python-to-CUDA, autograd) | >=1.13.0,<2.0 |
| RL framework | Stable-Baselines3 (PPO baseline) | >=2.5 |
| Vectorization | Custom BatchedVecEnv on top of Newton tensors | (internal) |
| Renderer (current) | matplotlib Agg + imageio-ffmpeg (headless MP4) | (internal) |
| Hydrodynamics | Custom Warp Tier-1 Fossen 6-DOF kernels | (internal) |
| RL env API | Gymnasium-compatible ROVEnv | >=1.2.3 |
| Validation reference | von Benzon 2022 6-DOF BlueROV2 port | (internal) |
| Python | CPython | 3.12 or 3.13 |
| CUDA | NVIDIA | 12.8+ |
| GPU | Reference platform | RTX 5090 |
| License | Apache-2.0 | — |

**Explicitly NOT in v0.0.1**: Isaac Sim, Isaac Lab, Omniverse RTX, BELLHOP, USD-native scenes, Gaussian-Splat scene capture, ray-traced sonar, world-model layer. These are documented in ARCHITECTURE_PROPOSAL_2026-05-15.md as future ambitions, not current capability.

---

## Two-layer architecture (per POSITIONING.md §4)

OceanScale is structured as a tiered hierarchy. Only Tier 1 ships today.

| Tier | Concept | Status |
|------|---------|--------|
| 1 | Ocean simulator (GPU-native Newton + Warp + Fossen hydrodynamics) | Shipping |
| 2 | Ocean world model (learned dynamics layer) | Future; not in v0.0.1 |
| 3 | Ocean foundation model (data + model + distribution leverage) | Research target; not in v0.0.1 |

---

## What ships in v0.0.1

- BlueROV2 Heavy 6-DOF dynamics (Fossen + von Benzon 2022 validation)
- Tier-1 Warp hydrodynamics: added mass, Coriolis, damping, restoring
- Vectorized environments (BatchedVecEnv shape (n_envs, ...))
- PPO hover training (converges, EV ~0.9 at 1M steps)
- Headless MP4 rendering (simple + cinematic 4-panel)
- CLI: `oceanscale demo bluerov2-hover`, `oceanscale train bluerov2-hover`
- Vectorized normalization (VecNormalize)
- Bundled pretrained hover checkpoint
- 150+ tests passing on RTX 5090

---

## What does NOT ship in v0.0.1

Per ADR-007 (`DECISIONS.md`): no Isaac Sim / Isaac Lab dependency. Per POSITIONING.md §4 promotion thresholds: no world-model layer claim (Tier 2) and no foundation-model claim (Tier 3) in public surfaces.

Future tier promotions are gated on:
- Tier 2 → homepage: shipped learned-dynamics demo with public benchmark vs classical baseline (`POSITIONING.md` §9.2)
- Tier 3 → headline: data + parameter-scale + distribution-leverage evidence (`POSITIONING.md` §4)

---

## Cross-references

- `POSITIONING.md` — brand law (anchor, scope, lexicon)
- `DESIGN.md` — visual brand law
- `STACK.md` — tech stack decisions and rationale
- `pyproject.toml` — authoritative dependency manifest
- `DECISIONS.md` — ADRs (especially ADR-007 on Isaac Sim removal)
- `ARCHITECTURE_PROPOSAL_2026-05-15.md` — original aspirational architecture (archived)

---

## Changelog

- **2026-05-23 v1** — Initial current-state architecture. Replaces the 2026-05-15 architecture proposal which was archived to ARCHITECTURE_PROPOSAL_2026-05-15.md.
