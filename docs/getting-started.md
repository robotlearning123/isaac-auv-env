# Getting Started with OceanScale

GPU-native underwater robotics simulation. Run BlueROV2 dynamics on Newton+Warp, train RL policies with vectorized environments, all on a single NVIDIA GPU.

## Requirements

| Dependency | Minimum | Notes |
|-----------|---------|-------|
| NVIDIA GPU | Any CUDA-capable | Tested on RTX 5090 |
| CUDA runtime | 12.8+ | Required by Warp and PyTorch |
| Python | 3.12–3.13 | `>=3.12,<3.14` |
| [uv](https://docs.astral.sh/uv/) | latest | Recommended for dev setup |

## Install

### From PyPI (users)

```bash
pip install oceanscale --extra-index-url https://download.pytorch.org/whl/cu128
```

> **Note:** PyPI publish pipeline (`.github/workflows/publish-pypi.yml`) is staged but not yet verified end-to-end. Until the first published release lands, use the from-source path below. See `ROADMAP.md`.

### From source (developers)

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv sync --extra dev          # installs all deps + dev tools (pytest, ruff, mypy)
```

### Colab (no local GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

Target: free Colab T4 GPU. Newton + Warp install path on T4 is documented but not yet verified end-to-end (`ROADMAP.md`). Tested on Colab paid runtimes.

## First demo

Run the pretrained BlueROV2 hover policy:

```bash
oceanscale demo bluerov2-hover --render-mp4 demo.mp4
```

For a cinematic 4-panel rendering:

```bash
oceanscale demo bluerov2-hover --render-mp4 demo.mp4 --cinematic
```

## First training

Train a PPO hover policy from scratch:

```bash
oceanscale train bluerov2-hover --total 50000 --n_envs 4
```

Flags: `--total` (timesteps), `--n_envs` (parallel environments), `--device` (default: `cuda`), `--seed` (default: `42`). Checkpoints save to `./checkpoints/` by default.

## Run the tests

```bash
uv run pytest tests/ -v              # full suite (150+ tests)
uv run pytest tests/ -v -m "not gpu" # CPU-only subset
```

## Where to read next

| Document | What it covers |
|----------|---------------|
| [`README.md`](../README.md) | Project overview, benchmark results, status |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Technical architecture, physics model, sensor design |
| [`AGENTS.md`](../AGENTS.md) | AI tool configuration (Claude, Codex, Gemini) |
| [`POSITIONING.md`](../POSITIONING.md) | Messaging and brand law (required for any copy) |
| [`DESIGN.md`](../DESIGN.md) | Visual brand law |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Clone, setup, test, and PR workflow |

## Troubleshooting

**`CUDA not available`** — verify your driver supports CUDA 12.8+ (`nvidia-smi`). Install PyTorch with the cu128 index:

```bash
pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
```

**`Pretrained model not found`** — the bundled checkpoint ships with the package. If missing, train one: `oceanscale train bluerov2-hover --total 1000000`.

**Install fails on Python 3.14+** — OceanScale requires Python 3.12 or 3.13. Use `uv python pin 3.12` to set the version.
