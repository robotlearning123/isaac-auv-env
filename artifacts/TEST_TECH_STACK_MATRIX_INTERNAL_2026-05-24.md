# OceanScale Internal Test And Tech Stack Matrix

日期：2026-05-24
范围：内部 alpha readiness；不作为公开发布材料。

## Current Test Evidence

| Check | Command | Result |
|---|---|---|
| Test collection | `uv run pytest tests/ --collect-only -q` | 378 tests collected in 2.39s |
| Full pytest | `uv run pytest tests/ -q` | 373 passed, 5 xfailed in 23.11s |
| Test modules | `find tests -type f -name '*.py'` + collection grouping | 33 modules with collected tests; 35 Python test files including package markers |
| Test code size | `find tests -type f -name '*.py' -print0 | xargs -0 wc -l` | 6,239 total test lines including package markers |

Configured pytest markers from `pyproject.toml`:

- `gpu`: requires CUDA GPU.
- `slow`: takes more than 10 seconds.
- `benchmark`: performance benchmark test.

## Current Tech Stack Evidence

| Layer | Current version / evidence | Status |
|---|---|---|
| Python package | `oceanscale=0.1.0a0` | GREEN locally |
| Warp | `warp-lang=1.13.0`; CUDA Toolkit 12.9; `cuda:0` arch 120 | GREEN locally |
| Newton | `newton=1.2.0` | GREEN locally |
| Torch | `torch=2.11.0+cu128`; Torch CUDA 12.8 | GREEN locally |
| GPU | NVIDIA GeForce RTX 5090; driver 580.95.05; 32607 MiB | GREEN locally |
| NumPy / SciPy | `numpy=2.4.4`; `scipy=1.17.1` | GREEN locally |
| Gymnasium | `gymnasium=1.2.3` | GREEN locally |
| SB3 | `stable-baselines3=2.8.0` | GREEN locally |
| skrl | `skrl=2.1.0`; `from oceanscale.training import train_skrl_ppo` import smoke passes | GREEN for import, not full training convergence |
| Tooling | `pytest=8.4.2`; `ruff=0.15.13`; `mypy=1.20.2` | GREEN locally |
| CLI | `uv run oceanscale --help` shows `demo` and `train` subcommands | GREEN locally |

## Test Coverage By Launch Area

| Launch area | Test files | Count | What it proves | Launch interpretation |
|---|---|---:|---|---|
| Stack / compatibility / version identity | `tests/test_environment.py`, `tests/test_version_compat.py`, `tests/test_version_identity.py`, `tests/test_warp_smoke.py`, `tests/test_newton_smoke.py` | 36 | Core imports, CUDA availability, Warp/Newton smoke, cross-framework compatibility checks, pyproject/metadata/`__version__` package-internal consistency | Local stack is real on this machine; release/tag identity remains a separate blocker |
| Newton / world simulation | `tests/test_newton_env.py`, `tests/test_worlds_poc.py`, `tests/test_energy_conservation.py` | 15 | Basic Newton env shapes, stepping, buoyancy, free-body/world replication, energy sanity | Supports internal simulator demo |
| Hydro Tier-1 / physics | `tests/hydro/test_action_allocation.py`, `test_domain_randomization.py`, `test_init_perturbation.py`, `test_tier1_autograd.py`, `test_tier1_determinism.py`, `test_tier1_smoke.py`, `test_tier1_throughput.py`, `test_vonbenzon_regression.py`, plus `tests/test_trajectory_parity.py`, `tests/test_param_fidelity.py` | 76 | Thruster allocation, domain randomization, init perturbation, autograd, determinism, throughput, von Benzon regression, CPU/GPU parity, parameter fidelity | Strong internal physics evidence, with known xfail caveats below |
| Product envs / Gym API | `tests/test_rov_env.py`, `tests/test_station_keeping.py`, `tests/test_docking_env.py`, `tests/test_waypoint_env.py`, `tests/test_vec_env.py` | 192 | Gym spaces, reset/step API, batched behavior, task rewards, partial reset, SB3-compatible VecEnv wrappers, render_mode warning guard | Core product API is covered and is the strongest test area |
| Sensors | `tests/test_sensors.py` | 14 | IMU, pressure/depth, DVL shapes, deterministic/noisy behavior, DVL identity-frame behavior | Valid for current sensor stubs only |
| Fluid / coupling | `tests/test_grid_fluid.py`, `tests/test_mpm_fluid.py`, `tests/test_sph_fluid.py`, `tests/test_rov_fluid_coupling.py` | 28 | Grid/MPM/SPH smoke and conservation checks, ROV fluid current coupling, training-style loop with fluid | Useful internal evidence, but SPH has known xfail bugs |
| CLI / product surface | `tests/test_cli.py` | 4 | Parser surface and hover checkpoint fallback resolution | Guards bundled SB3 demo fallback |
| Package/data policy | `tests/test_package_data.py` | 4 | Supported data artifacts exist; package-data includes zip/npz and excludes pickle/wildcard packaging | Guards data artifact boundary |
| Internal launch suite verifier | `tests/test_launch_suite_verifier.py` + `scripts/verify_internal_launch_suite.py` | 2 | Manifest paths, not-public boundary, benchmark truth, pipeline blocker, approval boundaries, and completion-audit scope | Turns the internal report suite into an executable consistency check |
| External pipeline state | `tests/test_external_launch_state.py` + `scripts/refresh_external_launch_state.py` | 4 | Evaluates GitHub run state, PyPI availability, release/PyPI deferral policy, and workflow runner-policy violations without mutating external systems | Turns pipeline evidence into a refreshable JSON artifact |
| Package artifact state | `tests/test_package_artifact_state.py` + `scripts/refresh_package_artifact_state.py` | 3 | Evaluates wheel/sdist data boundaries, entry point, twine status, and no-deps install smoke | Turns package/data evidence into a refreshable JSON artifact |

Total: 378 collected tests.

## Current Xfail Inventory

The 5 expected failures are not hidden; they document known model or solver gaps.

| Area | Tests | Reason | Launch impact |
|---|---|---|---|
| BlueROV2 inertia fidelity | `tests/test_param_fidelity.py::TestInertiaMatchesBlueROV2::test_ix_matches`, `test_iy_matches` | Current Newton sphere geometry gives isotropic inertia, while von Benzon reference uses anisotropic inertia | Do not claim exact anisotropic inertia fidelity in current Newton geometry |
| Water density fidelity | `tests/test_param_fidelity.py::TestWaterDensityConsistent::test_rho_matches` | Tier1 default is salt water rho 1025; von Benzon reference uses fresh water rho 1000 | Be explicit about water-density assumptions in fidelity claims |
| SPH force symmetry / momentum | `tests/test_sph_fluid.py::test_sph_force_symmetry`, `test_sph_momentum_conservation` | Known SPH pressure asymmetry and extra distance factor | Do not present SPH as production-grade validated fluid solver |

## Warning Inventory

Full pytest currently reports no warnings. The prior Stable-Baselines3 `render_mode` warning was removed by making `BatchedVecEnv` expose `render_mode=None` through `get_attr("render_mode")`.

## What Pytest Does Not Prove

| Requirement | Current evidence outside pytest | Status |
|---|---|---|
| Standard benchmark throughput | `benchmarks/competitive/results.json` and `uv run python benchmarks/competitive/oceanscale_standard_bench.py` | Covered by benchmark artifact, not by full pytest |
| Package publishing | `uv build` + `twine check` local pass; public PyPI/TestPyPI publish disabled by current policy | DEFERRED externally |
| Package-internal version consistency | `uv run pytest tests/test_version_identity.py -q` | GREEN for pyproject/installed metadata/`oceanscale.__version__`; does not solve release tag or `VERSION` mismatch |
| PyPI installability | `uvx --from pip pip index versions oceanscale` returns no matching distribution | RED externally |
| CI runner correctness | `.github/workflows/*` inspection shows all `runs-on` entries use `oceanscale-arc` | GREEN |
| Public website claim alignment | Live site stale from prior verification; user said not public | Deferred |
| skrl training convergence | Import smoke passes; benchmark/trainer code exists | Not covered by full pytest convergence test |
| Product demo command | `oceanscale demo bluerov2-hover --device cuda` now exits 0 via bundled SB3 fallback; final position error 1.2721m in latest smoke | YELLOW product-quality caveat |
| Sonar / camera / acoustic comms | Benchmark feature truth marks them false | Not implemented, no tests expected |
| Data packaging boundary | package-data includes `.zip` and `.npz`; `.pkl` tracked but not packaged | YELLOW, owner decision needed |

## Internal Launch Interpretation

- Internal alpha technical review: GO from test/stack perspective.
- Design-partner preview prep: CONDITIONAL GO, only with explicit limits on PyPI, public website, sensor scope, SPH maturity, and xfail fidelity caveats.
- Public launch: NO-GO until release identity, PyPI, CI runner, data boundary, and public surface are resolved.

## Next Test Hardening Candidates

These are useful after owner approval on the release/data tracks, but they are not blockers for continuing internal readiness work:

1. Add an installed-wheel package-data smoke in CI if runtime budget allows.
2. Extend CLI smoke coverage to run `oceanscale demo bluerov2-hover --device cuda` as an integration test if runtime budget allows.
3. Add a minimal skrl trainer smoke with very small timesteps or mark it slow/gpu.
4. Add a public-surface Playwright/browser test only if the work switches back to public launch.
