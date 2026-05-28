# OceanScale Evaluation Protocol

OceanScale treats simulation first as an evaluation surface for underwater robotics. The goal is to run controlled, repeatable checks before field testing, then report only the measurements that were actually produced.

## Quick Start

The evaluation workflow has three steps: generate a profile, run evaluations, then produce a report.

**Step 1. Generate the robustness profile.**

```bash
uv run oceanscale eval robustness-profile \
  --output-json reports/underwater_robustness_profile.json \
  --output-md reports/underwater_robustness_profile.md
```

This produces a definition file (not a result). It contains one baseline case plus perturbation cases across four categories:

| Category | Examples |
| --- | --- |
| ocean | current speed, current direction, water density |
| vehicle | mass scale, drag scale, thruster efficiency |
| sensor | visibility, DVL dropout, IMU gyro bias |
| task | target offset, obstacle count |

Each case records a deterministic seed and the number of episodes planned. The profile is a protocol, not a benchmark result.

**Step 2. Run evaluations.**

```bash
uv run oceanscale eval robustness-run \
  --case-limit 4 --steps-per-episode 10 \
  --action-mode zero --device cpu \
  --output-jsonl reports/underwater_robustness_records.jsonl \
  --summary-json reports/underwater_robustness_summary.json
```

Supported tasks: `station-keeping` (default), `docking`, `waypoint`. Action modes: `zero`, `proportional`, `damped`.

**Step 3. Generate the report.**

```bash
uv run oceanscale eval robustness-report \
  --input-jsonl reports/underwater_robustness_records.jsonl \
  --output-json reports/underwater_robustness_report.json \
  --output-md reports/underwater_robustness_report.md
```

## Output Formats

The evaluation pipeline produces four output types. Each serves a different purpose and they build on each other:

| Format | Produced by | Content | Purpose |
| --- | --- | --- | --- |
| JSONL records | `robustness-run` | One JSON object per case per line | Raw per-case data. Source of truth. |
| Summary JSON | `robustness-run` | Aggregated metrics, environment configs, run metadata | Quick overview of a single run. |
| Report JSON | `robustness-report` | Full report with metric distributions and failed cases | Machine-readable release artifact. |
| Report Markdown | `robustness-report` | Human-readable summary of the JSON report | Inspection and review. |

The JSONL file is the canonical record. Summary, report, and markdown are all derived from it and can be regenerated at any time.

For batch runs across multiple task/action-mode combinations, use `robustness-suite`:

```bash
uv run oceanscale eval robustness-suite \
  --tasks station-keeping docking waypoint \
  --action-modes zero proportional damped \
  --case-limit 4 --steps-per-episode 10 --device cpu \
  --output-dir reports/underwater_robustness_suite
```

This writes one set of outputs (JSONL, summary JSON, report JSON, report Markdown) per task/action-mode pair, plus a `manifest.json` listing every artifact.

## Action Modes

The runner provides three built-in action modes. These are evaluation baselines, not trained policies.

| Mode | Behavior | Use case |
| --- | --- | --- |
| `zero` | No control input (all-zero actions) | Drift baseline. Measures how the vehicle behaves with no controller. |
| `proportional` | Proportional target-seeking using the first 3 observation coordinates | Simple reactive baseline. Catches whether the task is solvable at all. |
| `damped` | Target-seeking with velocity damping (position error minus a velocity term) | Mission heuristic baseline. A slightly smarter controller that resists overshoot. |

These modes exist to produce evidence that the evaluation pipeline runs end-to-end. Trained-policy evaluators will be added in a future release.

## Genesis World Reference Boundary

Genesis World 1.0 is a method reference for evaluation-first robotics simulation:

Source: [The Role of Simulation in Scalable Robotics, Genesis World 1.0, and the Path Forward](https://www.genesis.ai/blog/the-role-of-simulation-in-scalable-robotics-genesis-world-10-and-the-path-forward).

- Build repeatable closed-loop evaluation before claiming data-generation or transfer gains.
- Separate training and evaluation paths so policies do not only optimize against one simulator setup.
- Report metrics with fixed seeds, case counts, and hardware context.

OceanScale does not add Genesis World, Quadrants, or Nyx as runtime dependencies. The core runtime remains Newton + Warp.

## Required Metrics

Every rollout report derived from this profile includes:

- success_rate
- episode_return
- mean_position_error_m
- max_position_error_m
- energy_proxy
- constraint_violations
- wall_time_s

Do not publish aggregate claims unless the raw case-level results, action mode, environment configuration snapshots, run metadata, and the command that generated them are available.

## Public Claim Rules

Allowed now:

- OceanScale provides a repeatable underwater robustness evaluation profile.
- OceanScale evaluates task rollouts across ocean, vehicle, sensor, and task perturbation axes.
- OceanScale uses Genesis World as a method reference, not as a dependency.

Not allowed yet:

- Zero-shot sim-to-real transfer.
- A correlation number between simulation and hardware.
- Any Genesis-style numeric result borrowed from Genesis.
- OceanScale as an ocean foundation model or public homepage world-model claim.

Public OceanScale claims must follow POSITIONING.md: no sim-to-real transfer claim without a real underwater robot experiment and quantified transfer gap.

## Next Implementation Gate

The next step is to expand rollout execution beyond smoke heuristics:

1. Add trained-policy evaluators for station-keeping, docking, and waypoint environments.
2. Produce full-profile public-readiness artifacts across all supported tasks once trained-policy runners land.
