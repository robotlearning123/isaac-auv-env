# OceanScale Documentation

The ocean simulator for underwater robotics.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem. These docs are organized for a new user who wants to install the repo, understand the NVIDIA Isaac 6 baseline, run the first demo, and then inspect the technical architecture.

## Start Here

| Document | Use it for |
| --- | --- |
| [`../README.md`](../README.md) | Project overview, positioning, quickstart, benchmark summary |
| [`getting-started.md`](getting-started.md) | Source install, first demo, first training command, basic tests |
| [`verification.md`](verification.md) | Customer/new-user verification checklist and expected evidence |
| [`isaac6-isaaclab3-install.md`](isaac6-isaaclab3-install.md) | Canonical NVIDIA Isaac Sim 6 / Isaac Lab 3 / Newton setup and verifier |

## Technical Docs

| Document | Use it for |
| --- | --- |
| [`../ARCHITECTURE.md`](../ARCHITECTURE.md) | System architecture, physics tiers, sensors, training integration |
| [`../STACK.md`](../STACK.md) | Stack rationale and dependency decision framework |
| [`math/fossen.md`](math/fossen.md) | Fossen 6-DOF hydrodynamics notes |
| [`marinegym_fossen_extract.md`](marinegym_fossen_extract.md) | MarineGym reference extraction and comparison notes |
| [`refs_tried.md`](refs_tried.md) | Reference simulator install/read results and gotchas |
| [`v0.1_brief.md`](v0.1_brief.md) | v0.1 product/technical brief |

## Project Governance

| Document | Use it for |
| --- | --- |
| [`../POSITIONING.md`](../POSITIONING.md) | Canonical category, scope, wording, and banned framings |
| [`../DESIGN.md`](../DESIGN.md) | Visual brand law |
| [`../ROADMAP.md`](../ROADMAP.md) | Planned milestones and gates |
| [`../RISKS.md`](../RISKS.md) | Technical and execution risks |
| [`../DECISIONS.md`](../DECISIONS.md) | Architecture decision records, including the Isaac 6 validation-lane decision |
| [`../STATUS.md`](../STATUS.md) | Current status addendum plus older historical snapshots |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Development workflow |
| [`../AGENTS.md`](../AGENTS.md) | Agent/tool instructions for this repo |

## Evidence And Artifacts

The reproducibility evidence for the current Isaac 6 work is under [`../artifacts/isaacsim/`](../artifacts/isaacsim/). Start with:

| Artifact | Use it for |
| --- | --- |
| [`../artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md`](../artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md) | Final Isaac 6 / Isaac Lab 3 summary |
| [`../artifacts/isaacsim/oceanscale_latest_isaac_baseline_2026-05-25.md`](../artifacts/isaacsim/oceanscale_latest_isaac_baseline_2026-05-25.md) | Baseline verifier and environment details |
| [`../artifacts/isaacsim/isaacsim6_official_setup_log_2026-05-25.md`](../artifacts/isaacsim/isaacsim6_official_setup_log_2026-05-25.md) | Full install/setup log |
| [`../artifacts/isaacsim/official_suite_coverage.md`](../artifacts/isaacsim/official_suite_coverage.md) | Official demo/test coverage snapshot |

## New-User Path

1. Read [`../README.md`](../README.md).
2. Run the source install in [`getting-started.md`](getting-started.md).
3. Run `uv run oceanscale demo bluerov2-hover --device cpu`.
4. Follow the customer checklist in [`verification.md`](verification.md).
5. Install the NVIDIA Isaac 6 lane with [`isaac6-isaaclab3-install.md`](isaac6-isaaclab3-install.md).
6. Run `export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official` and then `python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py --isaac-root "$ISAAC_ROOT"`.
