# OceanScale Verification Runbook

This runbook is the customer/new-user checklist for proving that OceanScale is installed, runnable, and aligned with the NVIDIA Isaac 6 ecosystem docs.

Scope:

- Verifies OceanScale from source.
- Verifies the documented CLI and first demo.
- Verifies the Isaac Sim 6 / Isaac Lab 3 validation lane after that lane is installed.
- Records where the reproducibility evidence is written.

Non-scope:

- This is not a full sweep of every upstream Isaac Sim or Isaac Lab demo.
- This is not an Isaac Sim source build.
- This is not a public PyPI release gate; PyPI publishing is not verified yet.

## 1. Source Install

Use the source path. Public PyPI publishing is not verified end-to-end yet.

~~~bash
git clone https://github.com/robotlearning123/oceanscale.git
cd oceanscale
uv python pin 3.12
uv sync --extra dev
~~~

Expected result: `uv sync` completes and creates the OceanScale development environment.

## 2. CLI Smoke

~~~bash
uv run oceanscale --help
uv run oceanscale --version
~~~

Expected result:

~~~text
oceanscale 0.1.0a0
~~~

The repo identity in `VERSION` and package metadata in `pyproject.toml` is `0.1.0-alpha`; Python normalizes that version as `0.1.0a0`.

## 3. First Demo

Run the CPU demo first because it does not require Isaac Sim or Isaac Lab:

~~~bash
uv run oceanscale demo bluerov2-hover --device cpu
~~~

Expected result: the command exits 0 and prints `Demo complete`.

For a CUDA MP4 run:

~~~bash
uv run oceanscale demo bluerov2-hover --device cuda --render-mp4 demo.mp4
~~~

## 4. Core Test Gates

Fast focused gates:

~~~bash
uv run pytest tests/test_isaacsim_underwater_demo.py -q
uv run pytest tests/test_isaaclab_task.py tests/test_isaaclab_training.py -q
~~~

Full test suite:

~~~bash
uv run pytest tests/ -q
~~~

If the full suite is too slow for a customer smoke check, run the focused gates above plus the Isaac baseline verifier below.

## 5. Isaac Sim 6 / Isaac Lab 3 Lane

Install the Isaac lane with:

- [`isaac6-isaaclab3-install.md`](isaac6-isaaclab3-install.md)

The Isaac validation environment must stay separate from the OceanScale development `.venv`. This prevents Isaac package pins from overwriting the Newton/Warp stack used by OceanScale core.

After the Isaac lane is installed, run from the OceanScale repo root:

~~~bash
export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py --isaac-root "$ISAAC_ROOT"
~~~

Expected result:

~~~text
{"passed": true, "summary": ".../summary.json"}
~~~

The verifier writes a timestamped directory under `$ISAAC_ROOT/logs/`. The most recent local verified summary at the time of this doc update was:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-26-125621/summary.json
~~~

## 6. ROS2 Bridge Verification

The ROS2 bridge requires a system ROS2 installation. It is optional — OceanScale core works without it.

### Install ROS2 Jazzy (Ubuntu 24.04)

~~~bash
sudo apt install -y ros-jazzy-ros-base ros-jazzy-rclpy
source /opt/ros/jazzy/setup.bash
~~~

### Verify rclpy

~~~bash
python3.12 -c "import rclpy; rclpy.init(); rclpy.shutdown(); print('rclpy OK')"
~~~

### Run mock-based tests (no ROS2 required)

~~~bash
uv run pytest tests/test_ros2_bridge.py -v
~~~

Expected: 54/54 pass.

### Verify with real ROS2 types

~~~bash
source /opt/ros/jazzy/setup.bash
PYTHONPATH=. python3.12 -c "
from oceanscale.ros2.messages import pose_from_obs, pressure_from_obs
import numpy as np
obs = {'position': np.array([0,0,-5], dtype=np.float32), 'orientation': np.array([0,0,0,1], dtype=np.float32)}
print('pose:', pose_from_obs(obs).pose.position.z)
print('pressure:', round(pressure_from_obs(obs).fluid_pressure, 0), 'Pa')
print('ROS2 bridge OK')
"
~~~

## 7. Documentation Checks

For documentation-only edits, run:

~~~bash
git diff --check
~~~

For link checks, use a Markdown link checker if available. The current docs were checked with a local path-existence scan over the root README, docs index, getting-started guide, Isaac install guide, package README, decision/status docs, and Isaac final summary.

## 7. Evidence Files

Keep these files together when handing off the environment:

| File | Purpose |
| --- | --- |
| [`README.md`](../README.md) | Project overview and source quickstart |
| [`docs/README.md`](README.md) | Documentation map |
| [`docs/getting-started.md`](getting-started.md) | First install, demo, training, and basic tests |
| [`docs/isaac6-isaaclab3-install.md`](isaac6-isaaclab3-install.md) | Isaac Sim 6 / Isaac Lab 3 install and validation lane |
| [`../artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md`](../artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md) | Latest recorded Isaac 6 evidence summary |
| [`../artifacts/isaacsim/official_suite_coverage.md`](../artifacts/isaacsim/official_suite_coverage.md) | Upstream official demo coverage snapshot |

## 8. Completion Standard

Call the local OceanScale environment ready only when:

- Source install completes.
- CLI help and version work.
- The CPU BlueROV2 demo completes.
- Focused OceanScale/Isaac-facing tests pass.
- The Isaac baseline verifier exits with `passed: true` for the Isaac lane.
- Any failed upstream official demos are documented as upstream/non-gate items, not hidden.
