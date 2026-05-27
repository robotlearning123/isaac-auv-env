# OceanScale

The ocean simulator for underwater robotics.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem: Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch. Lightwheel is the reference peer for robots on land; OceanScale builds the ocean physics, environments, and underwater sensor layer for AUVs and ROVs.

## Why OceanScale?

- **NVIDIA-native ocean physics** — Newton + Warp CUDA kernels, not CPU loops.
- **Isaac 6 ecosystem baseline** — Isaac Sim 6 and Isaac Lab 3 are the main simulation and training-integration targets.
- **RL-ready** — Gymnasium-compatible `ROVEnv` with `BatchedVecEnv`; PPO runs from the repo.
- **Validated** — Tier-1 hydrodynamics matched against von Benzon 2022 BlueROV2 reference model; benchmark docs state the current cross-coupling boundary explicitly.

## NVIDIA Isaac 6 Ecosystem

OceanScale is fully based on the NVIDIA Isaac 6 ecosystem for simulation and training integration. The canonical local setup keeps the Isaac validation environment side-by-side with the OceanScale core development environment so Isaac package pins do not overwrite the Newton/Warp stack used by OceanScale core.

Use [docs/isaac6-isaaclab3-install.md](docs/isaac6-isaaclab3-install.md) for the full install, source checkout, verification, and reproduction procedure.

## Quickstart From Source

PyPI publishing is not verified yet. For a new user, use the source path:

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv python pin 3.12
uv sync --extra dev
uv run oceanscale --help
uv run oceanscale demo bluerov2-hover --device cpu
```

For a CUDA run with MP4 output:

```bash
uv run oceanscale demo bluerov2-hover --device cuda --render-mp4 demo.mp4
```

Train a policy:

```bash
uv run oceanscale train bluerov2-hover --total 1000000 --n_envs 4 --device cuda
```

## Verification

Use [docs/verification.md](docs/verification.md) as the customer/new-user runbook for proving the source install, CLI, first demo, focused tests, and Isaac Sim 6 / Isaac Lab 3 validation lane.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

## Benchmark

Current launch-standardized OceanScale-only benchmark on RTX 5090. Full methodology in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Envs | Env-steps/s | Bench steps |
| ---: | ---: | ---: |
| 1 | 1,326 | 200 |
| 64 | 85,824 | 200 |
| 256 | 303,206 | 200 |
| 1024 | 1,272,106 | 200 |
| 4096 | 4,588,922 | 200 |

The older PyBullet comparison is also documented in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md): OceanScale at n=64 reached 17,427 env-steps/s on the matched BlueROV2 hover benchmark, 10.49x over single-env PyBullet. Keep these two benchmark tables separate.

## Status (v0.1 alpha)

**Works:**
- BlueROV2 6-DOF dynamics (Fossen + von Benzon 2022)
- Bundled BlueROV2 hover demo and PPO training path
- Vectorized environments (BatchedVecEnv + VecNormalize)
- Headless MP4 export (simple + cinematic modes)
- Source-installable CLI + Colab notebook
- Isaac Sim 6 / Isaac Lab 3 validation lane passing locally

**Known limitations:**
- Bundled hover demo is a CLI smoke and checkpoint-loading path, not a policy-quality release gate
- Contact, docking, station-keeping, waypoint, and multi-vehicle code paths exist but are not yet long-horizon release gates
- Isaac Sim 6 / Isaac Lab 3 validation is passing locally, but the full official Isaac demo sweep is not required for OceanScale readiness
- Omniverse RTX rendering is not a public OceanScale release surface yet
- No sim-to-real transfer validation

## Setup (developer)

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv sync --extra dev
uv run pytest tests/ -v
```

## Repo layout

```
oceanscale/          Python GPU simulation package
  hydro/             Fossen 6-DOF dynamics + Warp kernels
  vehicles/          BlueROV2 model
  envs/              Gymnasium environments
  sensors/           DVL, IMU, pressure
  training/          skrl PPO pipeline
  validation/        Von Benzon 2022 reference
website/             Astro 6 + Tailwind v4 landing page (Cloudflare Pages)
benchmarks/          Performance benchmarks
notebooks/           Colab notebooks
tests/               Test suite
docs/                Technical documentation
```

## Website

Landing page at [oceanscale-web.pages.dev](https://oceanscale-web.pages.dev). Built with Astro 6 + Tailwind v4, deployed to Cloudflare Pages via tag-driven releases.

```bash
cd website && pnpm install && pnpm dev    # localhost:4321
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{oceanscale2026,
  title = {OceanScale: GPU-Native Underwater Robotics Simulation},
  author = {OceanScale Team},
  year = {2026},
  url = {https://github.com/robotlearning123/oceanscale}
}
```
