# BlueROV MarineGym Assets — Provenance

**Source**: [MarineGym](https://github.com/Marine-RL/MarineGym)
**Original path**: `marinegym/robots/assets/usd/BlueROV/`
**License**: MIT
**Paper**: Chu et al., "MarineGym: A high-performance reinforcement learning platform for underwater robotics", IROS 2025
**Retrieved**: 2026-05-25

## Contents

| File | Description |
|------|-------------|
| `BlueROV.usd` | BlueROV2 USD mesh with rotor joints, collision geometry, visual mesh |
| `BlueROV.yaml` | 6-DOF hydrodynamic coefficients (added mass, damping, rotor config) |
| `config.yaml` | USD loading configuration |
| `Props/` | Instanced mesh components |
| `fixed_usd/` | Alternative USD with fixed joint configuration |

## Modifications

None — files are verbatim copies of MarineGym originals. Loaded by
`oceanscale.vehicles.BlueROV2MarineGym` which maps the YAML format to
OceanScale's Tier1 Fossen hydrodynamics interface.
