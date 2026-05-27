# Getting Started with OceanScale

The ocean simulator for underwater robotics.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem. Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch are the main development and validation baseline.

## Requirements

| Dependency | Minimum | Notes |
|-----------|---------|-------|
| NVIDIA GPU | Any CUDA-capable | Tested on RTX 5090 |
| CUDA runtime | 12.8+ | Required by Warp and PyTorch |
| Python | 3.12 | Canonical Isaac Sim 6 path; Python 3.13 is core-only, outside the Isaac lane |
| [uv](https://docs.astral.sh/uv/) | latest | Recommended for dev setup |

## Install

### From source (current supported path)

```bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv python pin 3.12
uv sync --extra dev          # installs deps + dev tools
uv run oceanscale --help
```

### NVIDIA Isaac 6 ecosystem setup

For the canonical Isaac Sim 6 / Isaac Lab 3 setup, continue with:

- [Isaac Sim 6 and Isaac Lab 3 install guide](isaac6-isaaclab3-install.md)
- [Verification runbook](verification.md)

### PyPI (future)

PyPI publishing is not verified end-to-end yet. Until the first published release lands, use the source path above. See `ROADMAP.md`.

### Colab (no local GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

Target: free Colab T4 GPU. Newton + Warp install path on T4 is documented but not yet verified end-to-end (`ROADMAP.md`). Tested on Colab paid runtimes.

## First demo

Run the bundled BlueROV2 hover checkpoint as a CLI smoke test:

```bash
uv run oceanscale demo bluerov2-hover --device cpu
```

For a cinematic 4-panel rendering:

```bash
uv run oceanscale demo bluerov2-hover --device cuda --render-mp4 demo.mp4 --cinematic
```

## First training

Train a PPO hover policy from scratch:

```bash
uv run oceanscale train bluerov2-hover --total 50000 --n_envs 4 --device cuda
```

Flags: `--total` (timesteps), `--n_envs` (parallel environments), `--device` (default: `cuda`), `--seed` (default: `42`). Checkpoints save to `./checkpoints/` by default.

## Run the tests

```bash
uv run pytest tests/test_isaacsim_underwater_demo.py -q
uv run pytest tests/test_isaaclab_task.py tests/test_isaaclab_training.py -q
uv run pytest tests/ -q
```

Use [`verification.md`](verification.md) for the complete customer/new-user checklist and expected outputs.

## Where to read next

| Document | What it covers |
|----------|---------------|
| [`README.md`](README.md) | Documentation map |
| [`README.md`](../README.md) | Project overview, benchmark results, status |
| [`verification.md`](verification.md) | Customer/new-user verification checklist |
| [`isaac6-isaaclab3-install.md`](isaac6-isaaclab3-install.md) | Canonical NVIDIA Isaac Sim 6 / Isaac Lab 3 ecosystem install and validation lane |
| [`ARCHITECTURE.md`](../ARCHITECTURE.md) | Technical architecture, physics model, sensor design |
| [`AGENTS.md`](../AGENTS.md) | AI tool configuration (Claude, Codex, Gemini) |
| [`POSITIONING.md`](../POSITIONING.md) | Messaging and brand law (required for any copy) |
| [`DESIGN.md`](../DESIGN.md) | Visual brand law |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | Clone, setup, test, and PR workflow |

## Troubleshooting

**`CUDA not available`** — verify your driver supports CUDA 12.8+ (`nvidia-smi`). Install PyTorch with the cu128 index:

```bash
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cu128
```

**`Pretrained model not found`** — the bundled checkpoint ships with the package. If missing, train one: `uv run oceanscale train bluerov2-hover --total 1000000`.

**Install fails on Python 3.13+ in the Isaac lane** — use Python 3.12 for the canonical Isaac Sim 6 setup. Python 3.13 is only for core OceanScale experiments outside Isaac. Use `uv python pin 3.12` to set the version.
