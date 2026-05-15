# Third-Party Notices

OceanScale incorporates code, assets, or substantial design ideas from the following projects. Each entry lists license, attribution, and what we use.

This file is the human-readable companion to per-file attribution headers and `oceanscale/_vendored/<project>/LICENSE` copies. Updated whenever a new dependency is vendored or ported.

---

## License compatibility

OceanScale outbound license: **Apache-2.0**. All inbound dependencies must be license-compatible (MIT, BSD-2/3, Apache-2.0, CC-BY-4.0, ISC, Zlib). Incompatible inbound (GPL-2/3/AGPL/SSPL) → not vendored. See `REFERENCES.md §7`.

---

## Vendored / ported code

*(none yet — entries added as T2.* and v0.2 sprints land)*

When MarineGym files land:
```
### MarineGym
URL: https://github.com/Marine-RL/MarineGym
License: MIT
Copyright: (c) 2025 Shuguang Chu, Zhejiang University
Files we use:
  - oceanscale/hydro/tier1_kernels.py — adapted from marinegym/robots/drone/underwaterVehicle.py
  - assets/vehicles/bluerov2_heavy/ — vendored marinegym/robots/assets/usd/BlueROV/*.usd
  - assets/vehicles/bluerov2_heavy.hydro.yaml — extracted constants from BlueROVHeavy.py
Pinned commit: TBD at T3.3
```

When OceanSim files land (v0.2):
```
### OceanSim
URL: https://github.com/umfieldrobotics/OceanSim
License: BSD-3-Clause
Copyright: (c) 2024-2025, The OceanSim Project Developers
Files we use:
  - oceanscale/sensors/_oceansim/sonar_kernels.py — from utils/ImagingSonar_kernels.py
  - oceanscale/sensors/_oceansim/camera_kernels.py — from utils/UWrenderer_utils.py
  - oceanscale/randomize/_oceansim/mvnormal.py — from utils/MultivariateNormal.py
  - oceanscale/randomize/_oceansim/mvuniform.py — from utils/MultivariateUniform.py
Pinned commit: TBD at v0.2 sprint
```

---

## Cited but not vendored

These projects are cited as references in our papers/docs/code comments but no file is copied or adapted.

| Project | Citation | Where |
|---|---|---|
| Newton (used as dependency, not vendored) | https://github.com/newton-physics/newton | `pyproject.toml` |
| NVIDIA Warp (used as dependency) | https://github.com/NVIDIA/warp | `pyproject.toml` |
| Fossen 2021 Handbook | Wiley | `docs/math/fossen.md`, paper |
| warplab/isaac-auv-env "Learning to Swim" | https://github.com/warplab/isaac-auv-env | `oceanscale/tasks/station_keeping.py` design notes |
| Akkaynak & Treibitz 2018 image model | CVPR 2018 | underwater camera math doc (v0.2) |
| HoloOcean 2.0 (validation methodology only) | arXiv 2510.06160 | validation reports |

---

## Explicitly NOT used (and why)

See `REFERENCES.md §5` for the full anti-reference list. Most notable:

- **Stonefish (GPL-3.0)** — incompatible license; we cite the paper only
- **clydemcqueen/bluerov2_gz** — no LICENSE file in repo (default copyright; cannot vendor)
- **gokulp01/bluerov2_gym** — no LICENSE file

---

## Authority for this file

Maintained by the OceanScale maintainers. Every PR that adds or modifies a vendored file must include a matching entry here in the same PR. Reviewed in each release as part of the v0.x.0 release checklist.
