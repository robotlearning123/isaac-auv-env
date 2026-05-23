# notebooks

Colab tutorials for OceanScale. Run BlueROV2 hover training in your browser, no local GPU required.

## Available notebooks

- [`bluerov2_hover_colab.ipynb`](bluerov2_hover_colab.ipynb) — BlueROV2 6-DOF hover task: install `oceanscale`, train a PPO policy, render the result. Self-contained.
- [`bluerov2_hover_colab.py`](bluerov2_hover_colab.py) — Same notebook as a plain Python script (for `jupytext` or local execution).

## Colab usage

Open directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/robotlearning123/oceanscale/blob/main/notebooks/bluerov2_hover_colab.ipynb)

The notebook installs `oceanscale` and its dependencies in a Colab runtime with a free T4 GPU. No local setup needed.

## Local usage

```bash
uv sync --extra dev
uv run jupyter lab notebooks/bluerov2_hover_colab.ipynb
```

Or run the Python script directly:

```bash
uv run python notebooks/bluerov2_hover_colab.py
```

## Further reading

- [`../AGENTS.md`](../AGENTS.md) — project-level agent configuration
- [`../oceanscale/AGENTS.md`](../oceanscale/AGENTS.md) — package conventions and CLI surface
- [`../README.md`](../README.md) — project overview and quickstart
