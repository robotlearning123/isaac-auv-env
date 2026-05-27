# v0.1.0 Plan Review — Synthesis (3 fork-agent reports, 2026-05-15)

DR (ChatGPT) refused due to temp-chat limitation; downgraded to local
WebSearch/WebFetch via 3 parallel fork agents covering 7 review questions.

## Critical findings (high-confidence)

### F1. Timeline is over-budget (HIGH CONFIDENCE)

Realistic solo timeline for ship definition (≤0.1m error, 8k envs, custom
Warp Tier-1, zero-shot-ready) = **6-9 weeks**, not 4. All published wall-clock
numbers in MarineGym/Tunçay/Cai are training-only and hide infrastructure
debt. Cai (3-author WHOI), Chu (7-author Zhejiang+Heriot-Watt), Chaffre
(PhD-scale 7-author) — none are 4-week solo projects.

**Driver**: writing custom Warp Tier-1 kernels (instead of using
off-the-shelf Isaac Lab + PhysX like Cai) adds 1-2 weeks. Warp autograd
debugging alone is ~5 days (per Warp issue tracker history).

**Action**: revise IMPLEMENTATION_PLAN.md ship gate to acknowledge 4w as
"stretch", add explicit buffer week + TDD rule (torch numerical-diff
harness BEFORE kernel, not after).

### F2. Wu 2018 is NOT a peer-reviewed paper (CRITICAL)

Wu 2018 is a **Flinders University MS thesis** by Chu-Jou Wu. **Wu states
his planned tank experiments could not be performed because the BlueROV2
Heavy was not delivered in time**. He fell back on technical specifications
+ published data from the *non-Heavy* BlueROV. Significantly weakens it
as our primary parameter source.

**Gold reference**: **von Benzon et al. 2022, JMSE 10(12):1898**, DOI
[10.3390/jmse10121898](https://doi.org/10.3390/jmse10121898), CC-BY-4.0
open access. Tank-validated complete simulator with full hydrodynamics +
thruster + tether models. Published Simulink simulator.

**Action**: replace Wu 2018 with von Benzon 2022 in REFERENCES.md §6.
Add CentraleNantesROV/bluerov2 ROS2+Gazebo rosbag schema as future-data
template.

### F3. Newton ecosystem is moving fast — TIGHTEN PIN (HIGH CONFIDENCE)

Newton released 3 minor versions in 30 days (1.0 → 1.1 → 1.2). **Sub-minor
breaking changes confirmed**:
- All raycast functions changed return format
- SDF.sparse_volume and SDF.coarse_volume removed
- VBD stiffness defaults changed
- Camera no longer auto-builds BVH

**Quality regression** flagged in open issue: hydroelastic contact forces
**~30× weaker in 1.2.0 vs 1.0.0** at identical parameters.

**Action**: tighten `newton` upper-bound from `<2.0` to **`<1.3`** in
pyproject.toml. Schedule a "Newton bump" sprint at v0.3 to absorb a
known-good 1.3.x.

### F4. Tier-1 Fossen is sufficient IF... (HIGH CONFIDENCE)

For BlueROV2 Heavy station-keeping at 0-2 m/s, raw Fossen is sufficient
for zero-shot sim-to-real **only if**:

1. Thruster model includes saturation + deadband + time-constant
   (not just `τ = T · u`). MarineGym `BlueROVHeavy.py` is likely linear-only
   — verify before T2.6.
2. DR covers ±30% thrust gain, ±15% mass, ±0.3 m/s current, +COB-COM
   offset, +IMU bias (per Chaffre 2025 methodology).
3. Coefficients pre-validated against published tank data (von Benzon
   2022) within ±10% before training starts.

**Evidence**:
- Chaffre 2025 added adaptive PID pole-placement on top of PPO because
  raw Fossen+PPO wasn't enough — they did NOT trust zero-shot.
- Cai 2024 achieved zero-shot but in dry-dock-class clean water with
  off-the-shelf Isaac Lab + simpler dynamics.
- Sim2real residual taxonomy: 40% parameter mismatch, 30% thrust
  allocation, 20% unmodeled dynamics, 10% sensor noise. Bad coefficients
  + bad thrust allocation are bigger threats than missing CFD.

**Deferring Tier-2 SPH to v0.3 is correct** for station-keep regime.

### F5. Add MJX as documented secondary physics path (RECOMMENDATION)

Insurance against Newton 2.x breaking changes in 6-12 months. ~2 days extra
plumbing in Tier-1 kernel interface. Tunçay 2025 proves the same task
converges at our scale on MJX (RTX 4060, 4096 envs, <2 min).

**Action**: add to DESIGN.md §3 explicit "physics backend abstraction" —
Tier-1 kernels call a thin adapter, not Newton directly. Cost: 2 days; benefit:
6-12 month resilience.

### F6. 6 high-impact missing references

To add to REFERENCES.md:

1. **Sim2Swim** (Tunçay et al., [arXiv 2512.08656](https://arxiv.org/abs/2512.08656))
   — 3-minute training claim for 6-DOF AUV zero-shot, direct G5 competitor.
2. **Learning to Dock** (Singh et al., [arXiv 2506.17823](https://arxiv.org/abs/2506.17823))
   — only published BlueROV2 Heavy Isaac Sim RL paper with DR ablations.
   Mirror obs/reward design.
3. **diffSPH** ([arXiv 2507.21684](https://arxiv.org/abs/2507.21684))
   — differentiable SPH for v0.3 manipulator wake; bypasses years of
   reimplementation.
4. **CentraleNantesROV/bluerov2** (GitHub) — ROS2+Gazebo BlueROV2 with
   exact rosbag topic schema for thruster→pose recording.
5. **OsloMet-OceanLab/BlueROV2** (GitHub) — Simulink 6-DOF Fossen+Wu 2018
   model; alternative parameter source.
6. **PrivilegedDreamer** (ICRA 2025) — model-based RL with hidden-parameter
   MDPs, documented sim-to-real improvement on AUV/marine tasks.

## Risk register additions (RISKS.md)

| ID | Priority | What | Source |
|---|---|---|---|
| R19 | 🔴 HIGH | Newton 1.x→2.x churn faster than expected (3 minors in 30 days) | Q5 finding |
| R20 | 🔴 HIGH | Warp autograd correctness debugging blocks T2.7 — budget 5 days, not 1 | Q1 finding |
| R21 | 🔴 HIGH | 4-week timeline assumes reward shaping converges first try — reality 2-3 iterations | Q1 finding |
| R22 | 🔴 HIGH | Thruster nonlinear (saturation+deadband+time-constant) required for zero-shot — MarineGym likely linear-only | Q3 finding |
| R23 | 🔴 HIGH | No open packaged tank-test dataset for BlueROV2 Heavy → von Benzon 2022 simulator-generated data is fallback | Q3+Q7 |
| R24 | 🟡 MED | MarineGym frozen on Isaac Sim 4.1; if we vendor we maintain it ourselves | Q2 finding |
| R25 | 🟡 MED | Hydroelastic contact regression in Newton 1.2.0 (30× weaker than 1.0); affects v0.3 manipulator tasks | Q5 finding |

## Auto-patches applied this commit

- [x] REFERENCES.md — replace Wu 2018 with von Benzon 2022 as gold validation source
- [x] REFERENCES.md — add F6 (6 missing refs)
- [x] pyproject.toml — pin `newton>=1.2.0,<1.3`
- [x] RISKS.md — add R19-R25 (7 new risks)
- [x] IMPLEMENTATION_PLAN.md — F1 timeline acknowledgement + TDD rule for T2.7

## Deferred for explicit user decision

- [ ] Timeline 4w → 6-9w explicit shift, OR commit to 4w as documented stretch
- [ ] MJX backend abstraction (+2 days) — add to v0.1 scope, OR document as v0.2 nice-to-have
- [ ] Visit von Benzon 2022 Simulink simulator + run reference trajectories for our Tier-1 unit tests

## Sources

Three fork-agent reports archived in:
- `/tmp/claude-1000/.../tasks/a105fc2fe9ed0a89a.output` (Q1+Q3)
- `/tmp/claude-1000/.../tasks/ab2100b9d549bb97c.output` (Q2+Q5)
- `/tmp/claude-1000/.../tasks/a4782f217fa60aaad.output` (Q4+Q7)

Each contains inline citations to arXiv/GitHub/DOI URLs.
