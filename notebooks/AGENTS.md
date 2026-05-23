# `notebooks/` — Agent Instructions

Jupyter notebooks for tutorials and Colab demos.

Read root `../AGENTS.md` first.

## Conventions

- Notebooks are authored as `.py` in jupytext percent format (`# %%` cell markers), then exported to `.ipynb`.
- Commit both `.py` and `.ipynb` versions.
- Target the **free Colab T4** runtime where possible (note: Newton + Warp CUDA install on T4 is documented but not yet end-to-end verified).
- Use `pip install oceanscale[rl]` in install cells (the `[rl]` extra is a legacy alias for back-compat; base install also works in v0.0.x+).

## Cross-references

- `../POSITIONING.md` — brand law for any markdown copy in notebooks.
- `../oceanscale/AGENTS.md` — Python package conventions.
