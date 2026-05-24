# OceanScale

GPU-native ocean simulation infrastructure for underwater robotics. Fossen 6-DOF dynamics on Newton + Warp, vectorized RL training, reproducible benchmarks on commodity GPU.

## Why OceanScale?

- **GPU-native physics** — Newton + Warp CUDA kernels, not CPU loops. 17K env-steps/s at n=64 on a single RTX 5090.
- **RL-ready** — Gymnasium-compatible `ROVEnv` with `BatchedVecEnv`, PPO converges out of the box.
- **Validated** — Tier-1 hydrodynamics matched against von Benzon 2022 BlueROV2 reference model (6-DOF, cross-coupling damping, full Coriolis).

## Quickstart

```bash
pip install oceanscale
oceanscale demo bluerov2-hover --render-mp4 demo.mp4
```

Train a policy:

```bash
oceanscale train bluerov2-hover --total 1000000 --n_envs 4
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

## Benchmark: OceanScale vs PyBullet

BlueROV2 hover task, 30K steps, RTX 5090, zero policy. Full methodology in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

| Config | Throughput (steps/s) | 1M steps | Speedup |
|--------|---------------------|----------|---------|
| PyBullet n=1 | 1,661 | 603 s | 1.0x |
| OceanScale n=16 | 5,076 | 197 s | 3.1x |
| OceanScale n=64 | 17,427 | 57 s | **10.5x** |

OceanScale's advantage is parallelism, not per-step latency. Below n=8, PyBullet wins on wall-clock. For RL with vectorized environments, OceanScale at n>=16 delivers the speedup.

## Status (v0.0.2 alpha)

**Works:**
- BlueROV2 6-DOF dynamics (Fossen + von Benzon 2022)
- PPO hover training (1M steps, EV ~0.9, depth target to 0.7mm)
- Vectorized environments (BatchedVecEnv + VecNormalize)
- Headless MP4 export (simple + cinematic modes)
- pip-installable CLI + Colab notebook

**Known limitations:**
- Hover policy drifts horizontally over a full episode (~0.41m after 33s)
- No contact / collision handling
- No multi-vehicle scenarios
- No docking / manipulation tasks
- Visual rendering is matplotlib-based 2D
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
