# OceanScale Latest Isaac Ecosystem Baseline - 2026-05-25

## Scope

OceanScale development is based on the latest Isaac ecosystem, but OceanScale core remains independent of Isaac Sim as a base dependency.

The current validation scope is OceanScale itself:

- OceanScale core runs in the project development environment with Newton 1.2, Warp 1.13, Torch cu128.
- Isaac Sim 6 / IsaacLab 3 are maintained as side-by-side external validation and training-integration environments.
- Full official demo sweep is no longer a release gate for OceanScale. Official demos remain useful for dependency health checks and failure triage.

## Source Baseline

| Component | Path | Branch / Head | Role |
| --- | --- | --- | --- |
| Isaac Sim source | /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop | develop / f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc | source reference |
| IsaacLab latest develop | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-develop | develop / f39f5b231b84daab09c1a1ecb1b7efa4bd56340f | latest 3.x/Newton source baseline |
| IsaacLab 3.0 beta2 | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2 | release/3.0.0-beta2 / 9fe080c1a1c73e8f8a7a8f971f98517876a99861 | stable beta reference |
| IsaacLab main | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-main-2026-05-25 | main / 54a65ea830c6002e17dc18c77831fa60e43937bc | non-Newton main reference |
| Newton source | /mnt/storage/isaacsim-6.0-official/sources/newton-v1.2.0 | a886e3fb411137d8a6ff370a1f3da427eccefbed | OceanScale-preferred Newton source reference |

All source trees above are side-by-side. Do not overwrite older installs or source trees.

## Runtime Baseline

### OceanScale development venv

Command:

~~~bash
uv run python - <<'PY'
import importlib.metadata as md
for dist in ["oceanscale", "warp-lang", "newton", "torch", "gymnasium", "isaacsim", "isaaclab", "isaaclab_newton"]:
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist}=NOT_INSTALLED")
PY
~~~

Result:

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

Interpretation: this is the correct OceanScale core environment. Isaac is not a base dependency.

### Isaac 6 / IsaacLab 3 validation venv

Command:

~~~bash
PYTHONNOUSERSITE=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
/mnt/storage/isaacsim-6.0-official/venv/bin/python - <<'PY'
import importlib.metadata as md
for dist in ["isaacsim", "isaacsim-app", "isaacsim-kernel", "isaaclab", "isaaclab_tasks", "isaaclab_assets", "isaaclab_newton", "isaaclab_experimental", "warp-lang", "newton", "torch"]:
    try:
        print(f"{dist}=={md.version(dist)}")
    except md.PackageNotFoundError:
        print(f"{dist}=NOT_INSTALLED")
PY
~~~

Result:

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

Interpretation: this venv is for IsaacLab validation. Do not mutate it to force OceanScale's Newton 1.2 preference unless we intentionally create a new combined validation lane.

## Official Documentation Basis

Context7 IsaacLab official docs checked:

- Newton integration is on IsaacLab develop as part of the IsaacLab 3.0 beta line.
- Newton setup docs use Python 3.12, Isaac Sim 6.0.0, Torch 2.10.0 cu128, then ./isaaclab.sh -i.
- IsaacLab 3.0 beta introduces isaaclab_newton, isaaclab_experimental, and isaaclab_task_experimental.

## Current Validation

### OceanScale core lane

Command:

~~~bash
uv run pytest \
  tests/test_environment.py \
  tests/test_newton_smoke.py \
  tests/test_isaaclab_training.py \
  tests/test_ocean_sim.py \
  tests/test_rov_env.py \
  tests/test_version_compat.py \
  tests/hydro/test_tier1_smoke.py \
  -q
~~~

Result:

~~~text
141 passed in 17.62s
~~~

### OceanScale IsaacLab integration lane

Command:

~~~bash
PYTHONPATH=/home/robot/workspace/46-marine \
PYTHONNOUSERSITE=1 \
ACCEPT_EULA=Y \
OMNI_KIT_ACCEPT_EULA=YES \
/mnt/storage/isaacsim-6.0-official/venv/bin/python -m pytest tests/test_isaaclab_training.py -q
~~~

Result:

~~~text
21 passed, 1 skipped in 2.41s
~~~

### IsaacLab latest-develop core health lane

Command:

~~~bash
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --lab-root /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-develop \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaaclab-develop-official-smokes-2026-05-25-r37-core \
  --case tutorial_00_create_empty \
  --case env_zero_cartpole_direct \
  --case demo_deformables_newton
~~~

Result:

| Case | Result |
| --- | --- |
| tutorial_00_create_empty | pass |
| env_zero_cartpole_direct | pass |
| demo_deformables_newton | pass |

### Reproducible OceanScale baseline verifier

Command:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-225406/summary.json"}
~~~

Summary evidence:

| Check | Result |
| --- | --- |
| Source heads and clean status | pass |
| OceanScale development venv versions | pass |
| Isaac 6 / IsaacLab 3 validation venv versions | pass |
| OceanScale core tests | 141 passed in 17.31s |
| OceanScale IsaacLab adapter tests | 21 passed, 1 skipped in 2.29s |

Latest rerun after scope clarification:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-230639/summary.json"}
~~~

Summary evidence:

| Check | Result |
| --- | --- |
| Source heads and clean status | pass |
| OceanScale development venv versions | pass |
| Isaac 6 / IsaacLab 3 validation venv versions | pass |
| OceanScale core tests | 141 passed in 16.96s |
| OceanScale IsaacLab adapter tests | 21 passed, 1 skipped in 2.36s |

Latest rerun with IsaacLab contract probe:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-231348/summary.json"}
~~~

Summary evidence:

| Check | Result |
| --- | --- |
| Source heads and clean status | pass |
| OceanScale development venv versions | pass |
| Isaac 6 / IsaacLab 3 validation venv versions | pass |
| IsaacLab contract probe | pass with warning: gym_vector_adapter, not native DirectRLEnv subclass |
| OceanScale core tests | 141 passed in 17.86s |
| OceanScale IsaacLab adapter tests | 21 passed, 1 skipped in 2.34s |

Latest rerun with native task readiness probe:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-231821/summary.json"}
~~~

Summary evidence:

| Check | Result |
| --- | --- |
| Source heads and clean status | pass |
| OceanScale development venv versions | pass |
| Isaac 6 / IsaacLab 3 validation venv versions | pass |
| OceanScale core + Isaac contract tests | 146 passed, 2 skipped in 18.22s |
| Isaac validation env adapter + native scaffold tests | 26 passed, 3 skipped, 38 warnings in 3.31s |

Latest rerun with native DirectRLEnv random-action stress:

~~~bash
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py
~~~

Result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-25-235642/summary.json"}
~~~

Summary evidence:

| Check | Result |
| --- | --- |
| Source heads and clean status | pass |
| OceanScale development venv versions | pass |
| Isaac 6 / IsaacLab 3 validation venv versions | pass |
| OceanScale native DirectRLEnv random-action stress | pass; 16 envs created, reset, 64 random-action steps; obs shape [16, 20]; obs_abs_max 5.070893; reward range [-0.042581, 0.908928]; terminated 0; truncated 0 |
| OceanScale core + Isaac contract tests | 146 passed, 3 skipped, 1 warning in 18.27s |
| Isaac validation env adapter + native cfg tests | 27 passed, 3 skipped, 41 warnings in 3.32s |

Current IsaacLab integration layers:

| Layer | Status |
| --- | --- |
| OceanScaleDirectRLEnv | ready Gymnasium vector adapter with DirectRLEnv-style data surface |
| OceanScaleTask | IsaacLab DirectRLEnv subclass; headless native random-action stress passes |
| OceanScaleTaskCfg | DirectRLEnvCfg-ready config with validate(), sim, and scene config |

Verifier warnings:

- OceanScaleDirectRLEnv is a Gymnasium vector adapter, not an Isaac Lab DirectRLEnv subclass.

IsaacLab contract boundary:

| Probe | Result |
| --- | --- |
| isaaclab.envs.DirectRLEnv importable | yes |
| OceanScaleDirectRLEnv is Gymnasium env | yes |
| OceanScaleDirectRLEnv is IsaacLab DirectRLEnv | no |
| OceanScaleTask is IsaacLab DirectRLEnv | yes |
| Native hooks currently implemented | _pre_physics_step, _apply_action, _get_observations, _get_rewards, _get_dones, _reset_idx |
| Native hooks still missing | none |
| OceanScaleTaskCfg is DirectRLEnvCfg | yes |
| OceanScaleTaskCfg validate/sim/scene present | yes |
| Native env instantiation tested | yes |
| Native reset/step tested | yes, 16 envs and 64 random-action steps |

## Project-Side Compatibility Fix

oceanscale.training now lazy-loads skrl only when train_skrl_ppo is called.

Reason: importing oceanscale.training.isaaclab_env should not require skrl in the IsaacLab validation venv. This preserves the public train_skrl_ppo entrypoint while keeping backend dependencies isolated.

oceanscale.sensors now has a concrete underwater_camera module for its exported camera API.

Reason: an earlier native verifier rerun exposed a core-test failure where importing a magnetometer mount loaded oceanscale.sensors.__init__, which referenced underwater_camera before the module was present in the working tree. The targeted regression check is:

~~~bash
uv run pytest tests/test_ocean_sim.py::TestSensors::test_magnetometer_attached -q
~~~

Result:

~~~text
1 passed in 2.56s
~~~

## Open Risks

- IsaacLab validation venv has newton 1.0.0, while OceanScale's preferred core stack uses newton 1.2.0. Keep these lanes separate unless creating a deliberate combined env.
- OceanScaleDirectRLEnv is currently a Gymnasium vector adapter with a DirectRLEnv-style data surface. It is not yet an IsaacLab-native DirectRLEnv subclass.
- OceanScaleTask now passes a 16-env, 64-step native IsaacLab random-action stress, but long-horizon and actual RL training stability are not tested yet.
- Some official IsaacLab PhysX collection/deformable demos still fail on missing omni.physics.tensors.api. This is not currently an OceanScale release gate.
- External ROS2 graph validation was not performed.
- Isaac Sim source build was not performed.
