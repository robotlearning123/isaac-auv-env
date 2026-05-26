# OceanScale Isaac 6 Final Summary - 2026-05-26

Generated at: 2026-05-26T00:00:50-04:00

## Verdict

OceanScale has a usable local Isaac Sim 6 / IsaacLab 3 validation and training-integration environment.

This is complete for the current OceanScale development gate:

- OceanScale itself is the focus.
- Isaac Sim 6.0 is the main Isaac target.
- IsaacLab 3.x is the IsaacLab target.
- Older Isaac/IsaacLab installs are not overwritten.
- Source checkouts are available for official reference.
- The final verifier passes.

This is not a claim that every official Isaac Sim / IsaacLab demo has been swept, nor that Isaac Sim has been built from source.

## Latest Repro Command

Run from the repository root:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Latest result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-235642/summary.json"}
~~~

## Final Verifier Evidence

Summary JSON:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-235642/summary.json
~~~

Key result:

| Check | Result |
| --- | --- |
| Overall verifier | pass |
| Errors | none |
| Warning | OceanScaleDirectRLEnv is a Gymnasium vector adapter, not an Isaac Lab DirectRLEnv subclass |
| Source heads and clean source status | pass |
| OceanScale development env versions | pass |
| Isaac validation env versions | pass |
| OceanScale core tests | 146 passed, 3 skipped, 1 warning in 18.27s |
| Isaac validation adapter/native cfg tests | 27 passed, 3 skipped, 41 warnings in 3.32s |
| OceanScale native IsaacLab stress | pass |

Native IsaacLab stress result:

| Field | Value |
| --- | --- |
| App launcher | IsaacLab AppLauncher, headless, cuda:0 |
| Environment class | oceanscale.training.isaaclab_task.OceanScaleTask |
| Config class | OceanScaleTaskCfg, DirectRLEnvCfg-ready |
| Number of envs | 16 |
| Steps | 64 |
| Actions | random, scale 0.35 |
| Observation shape | [16, 20] |
| Reward shape | [16] |
| obs_abs_max | 5.070893287658691 |
| reward_min | -0.04258129373192787 |
| reward_max | 0.9089276790618896 |
| terminated_count | 0 |
| truncated_count | 0 |
| returncode | 0 |

## Installed Runtime Layout

Isaac root:

~~~text
/mnt/storage/isaacsim-6.0-official
~~~

Important envs:

| Env | Path | Role |
| --- | --- | --- |
| OceanScale development env | /home/robot/workspace/46-marine/.venv | Core OceanScale Newton/Warp development |
| Isaac validation env | /mnt/storage/isaacsim-6.0-official/venv | Isaac Sim 6 / IsaacLab 3 integration validation |
| Strict Isaac Sim pip env | /mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict | Isaac Sim standalone/runtime smoke lane |

OceanScale development env versions:

| Package | Version |
| --- | --- |
| oceanscale | 0.1.0a0 |
| warp-lang | 1.13.0 |
| newton | 1.2.0 |
| torch | 2.11.0+cu128 |
| gymnasium | 1.2.3 |
| isaacsim | not installed |
| isaaclab | not installed |
| isaaclab_newton | not installed |

Isaac validation env versions:

| Package | Version |
| --- | --- |
| isaacsim | 6.0.0.0 |
| isaacsim-app | 6.0.0.0 |
| isaacsim-kernel | 6.0.0.0 |
| isaaclab | 6.0.0 |
| isaaclab_tasks | 1.10.0 |
| isaaclab_assets | 0.3.4 |
| isaaclab_newton | 0.12.0 |
| isaaclab_experimental | 0.0.5 |
| warp-lang | 1.13.0 |
| newton | 1.0.0 |
| torch | 2.10.0+cu128 |

The Isaac validation env intentionally remains separate from the OceanScale development env because the IsaacLab package lane currently carries newton 1.0.0 while OceanScale core is on newton 1.2.0.

## Source Code Baseline

All source trees are side-by-side under the Isaac root and should not overwrite older installs.

| Component | Path | Branch / head | Role |
| --- | --- | --- | --- |
| Isaac Sim source | /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop | develop / f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc | Official source reference |
| IsaacLab latest develop | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-develop | develop / f39f5b231b84daab09c1a1ecb1b7efa4bd56340f | Latest 3.x/Newton source baseline |
| IsaacLab 3.0 beta2 | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2 | release/3.0.0-beta2 / 9fe080c1a1c73e8f8a7a8f971f98517876a99861 | Stable beta reference used by validation env |
| IsaacLab main | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-main-2026-05-25 | main / 54a65ea830c6002e17dc18c77831fa60e43937bc | Non-Newton main reference |
| Newton source | /mnt/storage/isaacsim-6.0-official/sources/newton-v1.2.0 | a886e3fb411137d8a6ff370a1f3da427eccefbed | OceanScale-preferred Newton reference |

## OceanScale Code State Verified

The final verifier validates the working tree's current Isaac-facing implementation:

| Layer | Status |
| --- | --- |
| OceanScale core package | Runs without Isaac Sim / IsaacLab installed |
| OceanScaleDirectRLEnv | Lightweight Gymnasium vector adapter |
| OceanScaleTask | Native IsaacLab DirectRLEnv subclass |
| OceanScaleTaskCfg | DirectRLEnvCfg-ready config with sim and scene config |
| Native hooks | _pre_physics_step, _apply_action, _get_observations, _get_rewards, _get_dones, _reset_idx |
| Native reset/step | Tested headless through IsaacLab AppLauncher |
| Underwater camera export | Concrete underwater_camera module present in current tree |

## Official Demo / Runtime Coverage Snapshot

The full official demo sweep was intentionally not used as the OceanScale finish gate.

Current official coverage index:

| Scope | Passed | Failed | Uncovered |
| --- | ---: | ---: | ---: |
| all | 144 | 32 | 1923 |
| isaacsim | 110 | 16 | 81 |
| isaaclab | 34 | 16 | 1842 |

Runtime basics smoke lane:

| Case | Result |
| --- | --- |
| simulation_app_headless_no_rendering | pass |
| validation_extension_count_headless | pass |
| python_sh_import_sys | pass |
| python_sh_path_length | pass |
| simulation_app_ovd | fail, needs OVD recording configuration |
| python_sh_import_scipy | fail, source script expects source-build archive path |
| python_sh_import_torch | fail, source script expects source-build archive path |

Interpretation:

- The pip install lane is valid for Isaac Sim app/runtime tests and OceanScale validation.
- Some source-tree python_sh tests are not valid against the pip-only lane because the source checkout has no built python.sh archive layout.
- Failing official demos are not currently OceanScale blockers unless they touch OceanScale's own integration path.

## Repro Commands

Primary final verifier:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Core OceanScale test slice used by the verifier:

~~~bash
uv run pytest \
  tests/test_environment.py \
  tests/test_newton_smoke.py \
  tests/test_isaaclab_task.py \
  tests/test_isaaclab_training.py \
  tests/test_ocean_sim.py \
  tests/test_rov_env.py \
  tests/test_version_compat.py \
  tests/hydro/test_tier1_smoke.py \
  -q
~~~

Isaac validation adapter/native cfg tests:

~~~bash
PYTHONNOUSERSITE=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH=/home/robot/workspace/46-marine \
  /mnt/storage/isaacsim-6.0-official/venv/bin/python -m pytest \
  tests/test_isaaclab_task.py \
  tests/test_isaaclab_training.py \
  -q
~~~

Runtime basics smoke:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r31-runtime-basics-recheck \
  --case simulation_app_headless_no_rendering \
  --case simulation_app_ovd \
  --case validation_extension_count_headless \
  --case python_sh_import_sys \
  --case python_sh_import_scipy \
  --case python_sh_import_torch \
  --case python_sh_path_length
~~~

Coverage rebuild:

~~~bash
python3 artifacts/isaacsim/build_official_suite_inventory.py
python3 artifacts/isaacsim/build_official_suite_coverage.py
~~~

Hygiene checks from final turn:

~~~bash
python3 -m py_compile \
  artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py \
  artifacts/isaacsim/run_isaacsim_standalone_smokes.py

git diff --check
~~~

Both passed.

## Open Risks And Non-Goals

These remain explicitly outside the current finish gate:

- Full Isaac Sim standalone example sweep.
- Full Isaac Lab tutorial/demo sweep.
- Full external ROS2 graph validation.
- Long-horizon RL training stability.
- Isaac Sim source build from IsaacSim-develop.
- Isaac Lab optional mimic install.
- Asset/Nucleus-heavy demos that require extra assets, LFS blobs, or long runtimes.
- Merging OceanScale Newton 1.2.0 and IsaacLab packaged newton 1.0.0 into one shared env.

## Current Worktree Note

The working tree is not clean. Current status at final logging time included existing modified and untracked files under:

- AGENTS.md
- artifacts/isaacsim/
- oceanscale/
- tests/
- tmp/

Do not reset or overwrite these changes without explicit confirmation.

