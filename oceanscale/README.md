# oceanscale

OceanScale builds AI-native simulation infrastructure where underwater robots learn, test, and validate before deployment.

GPU-native fluid-structure simulation on Newton + Warp. BlueROV2 6-DOF dynamics (Fossen + von Benzon 2022). Vectorized Gym environments. PPO converges out of the box.

## Install

Requires CUDA 12.8+ and an NVIDIA GPU.

```bash
pip install oceanscale --extra-index-url https://download.pytorch.org/whl/cu128
```

Developer setup:

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv sync --extra dev
```

## Quick demo

```bash
oceanscale demo bluerov2-hover --render-mp4 demo.mp4
oceanscale demo bluerov2-hover --render-mp4 demo.mp4 --cinematic  # 4-panel
```

Train a policy:

```bash
oceanscale train bluerov2-hover --total 1000000 --n_envs 4
```

Or try it in your browser: [Open in Colab](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb).

## Architecture

Two-layer design: a high-fidelity ocean simulator running sensor-accurate vehicle dynamics and fluid physics on GPU (Newton + Warp CUDA kernels), with a learned ocean world model beneath it. The simulator handles Fossen 6-DOF rigid-body dynamics, hydrodynamic damping (cross-coupling), and Coriolis forces. Vectorized environments (`BatchedVecEnv`) step thousands of parallel envs per kernel launch.

Full technical spec: [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Status

v0.0.1 alpha. Works: BlueROV2 6-DOF dynamics, PPO hover training (EV ~0.9), vectorized envs, headless MP4 export, pip-installable CLI. Known gaps: no contact handling, no multi-vehicle, no sim-to-real validation.

## Testing

```bash
uv run pytest tests/ -v                      # full suite (150+ tests)
uv run pytest tests/ -v -m "not gpu"         # CPU-only
uv run pytest tests/hydro/test_tier1_autograd.py  # single test
```

Bug fix workflow: write a failing test first, then fix the code.

## Further reading

- [`../AGENTS.md`](../AGENTS.md) — cross-tool agent configuration
- [`../POSITIONING.md`](../POSITIONING.md) — messaging law (required reading for any copy)
- [`../DESIGN.md`](../DESIGN.md) — visual brand law
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — system architecture
- [`../README.md`](../README.md) — project overview
