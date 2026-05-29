# Goal: A real water tank in OceanScale (FloWave) that can simulate wave generation

Branch target: `goal/flowave-real-verify` (work) — never commit to `main`.
Repo: `/home/robot/workspace/46-marine` (OceanScale, Isaac Sim 6 / Newton / Warp).

## What "can sim wave generation" means here

A wavemaker (FloWave flap-paddle array) drives a **physically-correct, time-varying
free surface** in the tank, runnable **headless** in this env (warp/numpy/pxr; no
omni/carb), authored to an animated USD loadable later in Isaac Sim, and **proven**
correct by tests: realized wavelength obeys the dispersion relation, realized height
matches the Biésel wavemaker transfer, realized period matches the command.

Design (disclosed): deterministic **spectral wavemaking** — Biésel-correct paddle
commands + Airy superposition surface, consistent by construction. This is the
standard approach for a calibration tank. A *causal* paddle-forced CFD solve
(grid/SPH/SWE forced by the paddle boundary) is a larger, separate goal.

## Criteria (verify commands)
- [ ] C1 physics-green — `pytest tests/test_wave.py tests/test_wave_fft.py tests/test_paddle_array_biesel.py` → 33 passed
- [ ] C2 hero-integration-green — `pytest tests/test_virtual_flowave_hero.py` → 0 failed
- [ ] C3 regular-wave-synth — `FlapPaddleArray().synthesize_regular(H,T,depth)` returns paddle_commands (168,) + eta_field; no exception
- [ ] C4 wavegen-headless-run — `python3 -m oceanscale.cli flowave wavegen --wave regular --height 0.1 --period 2.0 --frames 60 --out /tmp/wavegen` → animated `.usda` + `wave_probes.npz` exist, non-empty
- [ ] C5 dispersion+Biésel acceptance — `pytest tests/test_flowave_wavegen_acceptance.py` → wavelength within 5% of L=2π/k(ω); height within 10% of Biésel-commanded H; period within 5% of T
- [ ] C6 no-regression-broad — `pytest tests/ -k 'flowave or wave or paddle or fluid or tank or coupling'` → no NEW failures vs baseline

All criteria use: `-o addopts="" -p no:cacheprovider --continue-on-collection-errors`.

## Build tasks (ordered)
1. T1 confirm hero tests pass headless (C2) — **DONE at baseline** (18 passed, 1 xpassed)
2. T2 `synthesize_regular(H,T,depth,direction)` + 2D-grid eta convenience — `facilities/flowave/paddle_array.py`
3. T3 `wave_probe.py` WaveProbe + optional probe sampling in `coupling.py` — depends T2
4. T4 `oceanscale flowave wavegen` CLI subcommand + mission preset — `cli.py`, `missions.py`, hero — depends T2,T3
5. T5 `tests/test_flowave_wavegen_acceptance.py` — the core proof — depends T2,T3
6. T6 run C1–C6, fix regressions (no weakening), docs — `README.md`, `docs/flowave_wavegen.md`

## Baseline (before changes) — verified 2026-05-29
- C1: PASS (33 passed)
- C2: PASS (18 passed, 1 xpassed) — hero pipeline already runs headless
- can_sim_wavegen_now: TRUE (scripts/virtual_flowave_hero.py runs paddle→surface→animated USD headless)
- Gaps to "verified product-grade": no acceptance proof (G1), no wave probe (G2), no CLI (G3), no regular-wave mode (G4)
- Only suite error: tests/test_marinegym_compat.py carb collection error (omni absent; unrelated; bypassed)

## Risks
- Analytic spectral wavemaking, not causal CFD (disclosed; standard for calibration tank).
- RTX-quality render needs Isaac Sim app (omni/carb) — ABSENT here; only USD authoring + headless sim verifiable.
- Wavelength FFT on 256×256 over [-12,12]² (dx≈0.094 m) limits short waves → use T≥1.5 s (L≳3 m) for the dispersion check.
- synthesize_regular must not break the 6 passing irregular paddle tests (shared dispersion/transfer helpers).
