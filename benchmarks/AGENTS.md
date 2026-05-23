# `benchmarks/` — Agent Instructions

Performance benchmarks for OceanScale vs alternatives (PyBullet, MuJoCo, etc.).

Read root `../AGENTS.md` first.

## Benchmarking discipline

- **Real measurements only.** Never interpolate between datapoints. Never invent numbers.
- **Report the exact command used.** Reproducibility is the value here.
- **Mark caveats explicitly.** Hardware variance, single-run vs averaged, env setup differences.
- **No "identical conditions" overclaim** when implementations differ — document the differences.

## Existing benchmarks

- `kernel_throughput.py` — per-kernel FPS on RTX 5090
- `bullet_bluerov_env.py` + `oceanscale_vs_bullet.py` — head-to-head BlueROV2 hover task
- See `RESULTS.md` for current numbers.

## Adding a new benchmark

1. Use a fixed seed.
2. Sweep `n_envs ∈ {1, 4, 16, 64}` minimum.
3. Run 2+ times, report median + range.
4. Document hardware (GPU, CUDA, kernel version).
5. Update `RESULTS.md` with the new entry.

## Cross-references

- `../ARCHITECTURE.md` §8 — throughput targets.
- `../POSITIONING.md` — when writing summary copy, no overclaim.
