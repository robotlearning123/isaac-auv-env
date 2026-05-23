# `oceanscale/` Python Package — Agent Instructions

This sub-package is the Python GPU simulator. Working here? Read this AGENTS.md *and* the root files:
- `../AGENTS.md` — cross-tool agent law
- `../POSITIONING.md` — brand / scope / voice / banned words
- `../ARCHITECTURE.md` — system architecture (tiered fluid model, sensor design)
- `../STACK.md` — tech stack rationale
- `../GLOSSARY.md` — terminology

## Package layout

See `../LAYOUT.md` §1 for the full directory tree.

## Dependencies

- Newton 1.x (Apache-2.0, USD-native, GPU-native)
- Warp >= 1.13 (NVIDIA, Python-to-CUDA)
- PyTorch >= 2.7 (CUDA 12.8 wheels)
- NumPy >= 1.26, SciPy >= 1.13
- Gymnasium >= 1.2

**No Isaac Sim dependency.** Isaac Sim and Isaac Lab are referenced for context, not required for any code path.

## Conventions

- **Warp kernels** declared with `@wp.kernel`. Avoid heavy Python-side branching inside.
- **Newton state**: prefer `state.body_f` for force injection (do not modify `state.body_q` outside the integrator).
- **Vectorized envs**: use shape `(n_envs, ...)`. All env stepping is GPU-vectorized.
- **Type hints required** on public APIs. Use `wp.array[...]` types where Warp arrays are passed.
- **No `print()`** — use `rich.print` or structured logging.

## Testing

```bash
uv run pytest tests/ -v                          # full test suite
uv run pytest tests/ -v -m "not gpu"             # CPU-only subset
uv run pytest tests/hydro/test_tier1_autograd.py # specific test
```

Bug fix workflow: failing test first, then fix.

## Linting

```bash
uv run ruff format oceanscale/ tests/
uv run ruff check oceanscale/ tests/
uv run mypy oceanscale/
```

## CLI surface

```bash
oceanscale demo bluerov2-hover --render-mp4 demo.mp4
oceanscale train bluerov2-hover --total 1000000 --n_envs 4
oceanscale --version
```

## What NOT to do

- Do not add Isaac Sim as a dependency.
- Do not modify `state.body_q` directly outside the integrator.
- Do not add `print()` statements.
- Do not add features beyond the assigned task.
- Do not add buzzword comments — well-named identifiers suffice.
