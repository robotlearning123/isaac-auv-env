# OceanScale Python Package

The ocean simulator for underwater robotics.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem: Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch.

This package contains the core Python simulation code: GPU-native ocean physics on Newton + Warp, BlueROV2 6-DOF dynamics, vectorized Gymnasium environments, sensors, and training helpers. The canonical Isaac Sim 6 / Isaac Lab 3 environment is documented in [`../docs/isaac6-isaaclab3-install.md`](../docs/isaac6-isaaclab3-install.md).

## Install

PyPI publishing is not verified yet. For a new user, install from source:

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv python pin 3.12
uv sync --extra dev
uv run oceanscale --help
```

For the full NVIDIA Isaac 6 ecosystem setup, continue with:

- [`../docs/isaac6-isaaclab3-install.md`](../docs/isaac6-isaaclab3-install.md)

## Quick Demo

```bash
uv run oceanscale demo bluerov2-hover --device cpu
```

For CUDA with MP4 output:

```bash
uv run oceanscale demo bluerov2-hover --device cuda --render-mp4 demo.mp4
```

Train a policy:

```bash
uv run oceanscale train bluerov2-hover --total 1000000 --n_envs 4 --device cuda
```

Or try it in your browser: [Open in Colab](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb).

## Architecture

Two-layer design: a high-fidelity ocean simulator running sensor-accurate vehicle dynamics and fluid physics on GPU (Newton + Warp CUDA kernels), with a learned ocean world model beneath it. The simulator handles Fossen 6-DOF rigid-body dynamics, hydrodynamic damping, and Coriolis forces. Vectorized environments (`BatchedVecEnv`) step thousands of parallel envs per kernel launch.

Full technical spec: [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Status

v0.1 alpha. Works: BlueROV2 6-DOF dynamics, PPO hover training, vectorized envs, source-installable CLI, headless MP4 export, and a passing Isaac Sim 6 / Isaac Lab 3 validation lane. Known gaps: no public PyPI release yet, no full official Isaac demo sweep, no long-horizon IsaacLab RL training gate, and no sim-to-real validation.

## Testing

```bash
uv run pytest tests/ -v                      # full suite
uv run pytest tests/ -v -m "not gpu"         # CPU-only
uv run pytest tests/hydro/test_tier1_autograd.py  # single test
```

Bug fix workflow: write a failing test first, then fix the code.

## Further Reading

- [`../README.md`](../README.md) — project overview
- [`../docs/README.md`](../docs/README.md) — documentation map
- [`../docs/getting-started.md`](../docs/getting-started.md) — first install and demo
- [`../docs/isaac6-isaaclab3-install.md`](../docs/isaac6-isaaclab3-install.md) — NVIDIA Isaac 6 setup
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — system architecture
- [`../POSITIONING.md`](../POSITIONING.md) — messaging law
- [`../AGENTS.md`](../AGENTS.md) — agent/tool instructions
