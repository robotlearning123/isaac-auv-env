# OceanScale Internal Product Capability Evidence

日期：2026-05-24
范围：内部 alpha readiness；不作为公开发布材料。

## Purpose

本文件验证当前产品/API surface 的真实可用状态：CLI、core env、task env、sensors、VecEnv wrapper、training smoke、demo command。它不替代 full pytest；它是面向 startup/product readiness 的 smoke evidence。

## CLI Evidence

| Command | Result | Interpretation |
|---|---|---|
| `uv run oceanscale --version` | `oceanscale 0.1.0a0` | CLI entry point works in repo env |
| `uv run oceanscale --help` | shows `demo` and `train` subcommands | Root CLI surface exists |
| `uv run oceanscale demo --help` | shows `bluerov2-hover`, `bluerov2-dock` | Demo command surface exists |
| `uv run oceanscale train --help` | shows `bluerov2-hover` plus train flags | Train command surface exists |

## Core Product API Smoke

Command: a direct Python smoke using CUDA and `n_envs=4`.

| Component | Import path / class | Reset shape | Step shape | Reward shape | Result |
|---|---|---:|---:|---:|---|
| Core hover env | `oceanscale.rov_env.ROVEnv` | `(4, 26)` | `(4, 26)` | `(4,)` | PASS, finite reward |
| Station keeping task | `oceanscale.envs.station_keeping_env.CurrentStationKeepingEnv` | `(4, 29)` | `(4, 29)` | `(4,)` | PASS, finite reward |
| Docking task | `oceanscale.envs.docking_env.DockingApproachEnv` | `(4, 31)` | `(4, 31)` | `(4,)` | PASS, finite reward |
| Waypoint task | `oceanscale.envs.waypoint_env.WaypointFollowingEnv` | `(4, 29)` | `(4, 29)` | `(4,)` | PASS, finite reward |

Naming note: the task classes are not named `StationKeepingEnv`, `DockingEnv`, or `WaypointEnv`; the real exported class names are `CurrentStationKeepingEnv`, `DockingApproachEnv`, and `WaypointFollowingEnv`. `oceanscale.envs.__init__` currently does not re-export them.

## Sensor Smoke

| Component | Class | Output shape | Result |
|---|---|---:|---|
| IMU stub | `IMUSensor(n_envs=4).read(...)` | `(4, 6)` | PASS, finite tensor |
| Pressure/depth stub | `PressureSensor(n_envs=4).read(...)` | `(4, 1)` | PASS, finite tensor |
| DVL stub | `DVLSensor(n_envs=4).read(...)` | `(4, 3)` | PASS, finite tensor |

Sensor scope remains stub-level only. This does not prove sonar, physical camera, or acoustic comms.

## Vectorized Environment Smoke

| Component | Result | Interpretation |
|---|---|---|
| `BatchedVecEnv(ROVEnv(n_envs=4))` reset | `(4, 26)` | SB3-compatible wrapper reset works |
| `BatchedVecEnv.step_wait()` | obs `(4, 26)`, rewards `(4,)`, dones `(4,)`, infos length `4` | SB3-style stepping works |
| `OceanScaleVecEnv(n_envs=4)` reset | `(4, 19)` | Gymnasium vector env wrapper reset works |
| `OceanScaleVecEnv.step()` | obs `(4, 19)`, reward `(4,)`, terminated `(4,)`, truncated `(4,)` | Gymnasium vector stepping works |

`BatchedVecEnv` now exposes `render_mode=None`, which removes the prior Stable-Baselines3 `render_mode` warning from full pytest.

## Training Smoke

Command:

```bash
uv run oceanscale train bluerov2-hover --total 8 --n_envs 4 --device cuda --checkpoint-dir /tmp/oceanscale-train-smoke-20260524
```

Result: PASS.

Output files:

| File | Size |
|---|---:|
| `/tmp/oceanscale-train-smoke-20260524/bluerov2_skrl_policy.pt` | 86522 bytes |
| `/tmp/oceanscale-train-smoke-20260524/bluerov2_skrl_value.pt` | 83685 bytes |

Interpretation: the skrl training command can start and save policy/value artifacts. This is a smoke only; it does not prove training convergence or policy quality.

## Demo Surface Check

Command:

```bash
uv run oceanscale demo bluerov2-hover --device cuda
```

Result: PASS after CLI fallback fix.

Observed output:

```text
Using bundled SB3 checkpoint: /home/robot/workspace/46-marine/oceanscale/data/bluerov2_station_keep_final.zip
Demo complete: 1000 steps, total_reward=271.56
Final position: [-0.07194088  0.58327055 -2.6282218 ]
Target position: [ 0.   0.  -1.5]
Position error: 1.2721m, Depth error: 1.1282m
```

Fix evidence:

- `oceanscale/cli.py` now prefers `bluerov2_skrl_policy.pt` when present and falls back to bundled `bluerov2_station_keep_final.zip`.
- `bluerov2_station_keep_final.zip` contains SB3-style files: `data`, `pytorch_variables.pth`, `policy.pth`, `policy.optimizer.pth`, `_stable_baselines3_version`, `system_info.txt`.
- Focused CLI tests cover checkpoint resolution and parser surface.

Interpretation: core API, training surface, and out-of-box hover demo command are executable. The bundled SB3 policy quality remains weak for hover accuracy in this run, so do not claim high-quality convergence from this demo.

## Fix Validation

| Check | Result |
|---|---|
| `uv run ruff check oceanscale/cli.py tests/test_cli.py` | PASS |
| `uv run mypy oceanscale/cli.py --show-error-codes --no-error-summary` | PASS |
| `uv run pytest tests/test_cli.py -q` | PASS: 4 passed |
| `uv run pytest tests/ -q` | PASS: 373 passed, 5 xfailed, no warnings in 23.11s |
| `uv run oceanscale demo bluerov2-hover --device cuda` | PASS: 1000 steps, exit 0 |
| `uv build --out-dir /tmp/oceanscale-warning-fix-build-20260524` + `twine check` | PASS |

## Product Readiness Classification

| Area | Status | Reason |
|---|---|---|
| CLI metadata/help | GREEN | version and help commands work |
| Core env API | GREEN | ROVEnv reset/step smoke passes |
| Task env API | GREEN | station keeping, docking, waypoint smoke passes |
| Sensor stubs | GREEN for current scope | IMU/DVL/pressure stubs return finite outputs |
| VecEnv wrappers | GREEN | reset/step works; SB3 render_mode warning removed |
| Training command | GREEN for smoke | 8-step skrl train saves policy/value files |
| Hover demo command | YELLOW | executable via bundled SB3 fallback; quality caveat remains |
| Public product claim | NO-GO | public pipeline and demo-quality claims are not ready |

## Recommended Internal Next Steps

1. Decide whether the bundled SB3 demo quality is acceptable for internal demos or whether to generate and bundle a stronger skrl `.pt` policy.
2. Consider adding an integration CLI smoke for `oceanscale demo bluerov2-hover --device cuda` if runtime budget allows.
3. If bundling a skrl policy is approved, generate it through the training path, add it to package-data, and re-run package/data evidence.
4. Consider re-exporting task env classes from `oceanscale.envs.__init__` if they are intended public API.
5. Keep sonar/camera/acoustic comms out of product claims until implemented and tested.
