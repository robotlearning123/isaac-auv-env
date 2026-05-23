# OceanScale Glossary

Project-specific terms. Updated as the project evolves. For broader infrastructure terminology, see `STACK.md` and `ARCHITECTURE.md`.

## Brand & Positioning

| Term | Meaning |
|------|---------|
| **OceanScale** | The project / company. One word, two caps (`O`, `S`). |
| **Ocean simulator** | Tier-1 concept (`POSITIONING.md` §4). Public default. |
| **Ocean world model** | Tier-2 concept. One click below the homepage. |
| **Ocean foundation model** | Tier-3 concept. Research / deck depth only. |
| **Field testing as bottleneck** | Canonical enemy framing (`POSITIONING.md` §5). |
| **Anchor sentence** | The single sentence everything radiates from (`POSITIONING.md` §3). |
| **Enemy line** | The canonical "current world" line that creates narrative tension (`POSITIONING.md` §5). |

## Technical (underwater)

| Term | Meaning |
|------|---------|
| **AUV** | Autonomous Underwater Vehicle. |
| **ROV** | Remotely Operated Vehicle. |
| **USV** | Unmanned Surface Vehicle. Out of current scope (`POSITIONING.md` §2). |
| **6-DOF** | Six degrees of freedom (X, Y, Z, roll, pitch, yaw). |
| **Fossen 6-DOF** | Thor I. Fossen's 6-DOF hydrodynamic model for marine vehicles. |
| **BlueROV2 / BlueROV2 Heavy** | Open-source ROV platforms used as reference hardware. |
| **Tier-0 / Tier-1 / Tier-2 / Tier-3 fluid** | Hydrodynamics fidelity tiers (`ARCHITECTURE.md` §5.1). |
| **JMG / Jaffe-McGlamery** | Underwater image formation model. |
| **DVL** | Doppler Velocity Log. |
| **IMU** | Inertial Measurement Unit. |
| **BELLHOP** | Acoustic ray-tracing reference solver. |

## Infrastructure

| Term | Meaning |
|------|---------|
| **Newton** | Open-source physics engine (Newton 1.x, GPU-native). |
| **Warp** | NVIDIA GPU programming language for differentiable physics kernels. |
| **MuJoCo Warp** | MuJoCo physics engine ported to Warp (`mujoco-warp`). |
| **OpenUSD** | Pixar's universal scene description format. |
| **Isaac Sim / Isaac Lab** | NVIDIA's simulation platform. Referenced for context only; not a dependency. |

## Process

| Term | Meaning |
|------|---------|
| **Wave** | A coordinated batch of related changes (e.g., Wave A through D of a release). |
| **Tier promotion** | Moving a concept (e.g., world model) from internal-only to public-facing once evidence catches up (`POSITIONING.md` §4). |
| **Cross-model review** | Opus + cx (or similar pair) review of a substantial artifact before lock. |

## Cross-references

- `POSITIONING.md` — brand terminology canon.
- `ARCHITECTURE.md` — technical terminology source.
- `STACK.md` — tech stack and decision rationale.
- `AGENTS.md` — AI agent terminology.
