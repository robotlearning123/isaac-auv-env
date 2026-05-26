# NVIDIA Isaac Sim 6 and Isaac Lab 3 Install Guide

This is the canonical NVIDIA Isaac 6 ecosystem setup for OceanScale development.

OceanScale is the ocean layer in the NVIDIA robotics simulation ecosystem. Isaac Sim 6, Isaac Lab 3, Newton, Warp, CUDA, and PyTorch are the main development and validation baseline. Lightwheel is the reference peer for robots on land; OceanScale builds the ocean physics, environments, and underwater sensor layer for AUVs and ROVs.

The repository is NVIDIA ecosystem-first. Dependency isolation is only for reproducibility: the OceanScale core development environment and the Isaac validation environment stay separate so Isaac package pins do not accidentally override the Newton/Warp stack used by OceanScale core.

## Current Status

The local OceanScale Isaac 6 gate is passing.

Run from the repository root:

~~~bash
export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py --isaac-root "$ISAAC_ROOT"
~~~

Latest recorded result:

~~~text
{"passed": true, "summary": "/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-26-125621/summary.json"}
~~~

On a new machine, the verifier creates a new timestamped summary under your chosen `$ISAAC_ROOT/logs/`. The verifier defaults to `$ISAAC_ROOT` when that environment variable is set, and also accepts `--isaac-root` explicitly.

The verifier creates OceanScale's native Isaac Lab task through Isaac Lab `AppLauncher` in headless `cuda:0` mode and runs a 16-env, 64-step random-action stress check.

| Check | Result |
| --- | --- |
| OceanScale native Isaac Lab task | pass |
| Parallel envs | 16 |
| Random-action steps | 64 |
| Observation shape | [16, 20] |
| Reward shape | [16] |
| terminated_count | 788 |
| truncated_count | 0 |
| OceanScale core tests | 146 passed, 3 skipped |
| Isaac validation adapter/native cfg tests | 27 passed, 3 skipped |

Random actions can terminate episodes; the pass condition is that the native task launches, resets, steps with finite observations/rewards, and the verifier exits with `passed: true`.

## Official References

Use official docs and source as the source of truth before changing this file:

- Isaac Sim 6 docs index: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html
- Isaac Sim 6 Python install: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html
- Isaac Sim 6 requirements: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
- Isaac Sim source: https://github.com/isaac-sim/IsaacSim
- Isaac Lab source: https://github.com/isaac-sim/IsaacLab
- Newton source: https://github.com/newton-physics/newton
- Local Isaac Lab 3 install docs: `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2/docs/source/setup/installation/index.rst`
- Local Isaac Lab 3 verification docs: `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2/docs/source/setup/installation/include/src_verify_isaaclab.rst`

Official Isaac Lab 3 docs describe several installation modes. For OceanScale, the main NVIDIA ecosystem lane is Isaac Sim pip runtime plus Isaac Lab source install, with official source trees retained side by side.

## Install Policy

- Isaac Sim 6.0 is the main simulator target for this repo.
- Isaac Lab 3.x is the main robot-learning integration target.
- Newton 1.2.0 and Warp are the OceanScale core physics source baseline.
- CUDA and PyTorch cu128 are the GPU runtime baseline.
- Do not overwrite older Isaac Sim or Isaac Lab installs.
- Do not install Isaac Sim or Isaac Lab into the normal OceanScale `.venv` unless deliberately creating a combined experiment.
- Keep the Isaac validation venv separate from the OceanScale development venv.
- Record exact commands, versions, source heads, logs, and verifier summaries.

## Tested Directory Layout

Current local layout:

~~~text
/mnt/storage/isaacsim-6.0-official/
  venv/
  venv-isaacsim-strict/
  logs/
  sources/
    IsaacSim-develop/
    IsaacLab-develop/
    IsaacLab-release-3.0.0-beta2/
    IsaacLab-main-2026-05-25/
    newton-v1.2.0/
~~~

Use the same shape on a new machine:

~~~bash
export OCEANSCALE_ROOT=/path/to/oceanscale
export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official
mkdir -p $ISAAC_ROOT/logs $ISAAC_ROOT/sources
~~~

## Prerequisites

| Requirement | Guidance |
| --- | --- |
| OS | Ubuntu Linux x64; Ubuntu 24.04.2 was tested locally |
| Python | 3.12 for Isaac Sim 6.x |
| GPU | NVIDIA GPU; RTX 5090 was tested locally |
| Driver | 580.95.05 was tested locally; use a current production driver |
| Package tool | `uv` preferred; `pip` can be used with equivalent commands |
| Disk | Plan for tens of GB for packages, caches, logs, and sources |
| Source build only | GCC/G++ 11; current Isaac Sim source README says GCC/G++ 12+ is unsupported for this path |

Check basics:

~~~bash
nvidia-smi
python3.12 --version
uv --version
git --version
git lfs version
~~~

## 1. Create The Isaac Validation Venv

~~~bash
export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official
export OCEANSCALE_ROOT=/path/to/oceanscale
mkdir -p $ISAAC_ROOT/logs $ISAAC_ROOT/sources
uv venv --python 3.12 --seed $ISAAC_ROOT/venv
source $ISAAC_ROOT/venv/bin/activate
uv pip install --upgrade pip
~~~

If not using `uv`, create a Python 3.12 venv and replace `uv pip` with `python -m pip`.

## 2. Install Isaac Sim 6

Install Isaac Sim from NVIDIA's package index:

~~~bash
UV_HTTP_TIMEOUT=300 uv pip install --prerelease=allow \
  "isaacsim[all,extscache]==6.0.0" \
  --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee $ISAAC_ROOT/logs/isaacsim6-pip-install.log
~~~

Install the official CUDA 12.8 PyTorch wheel for this Isaac lane:

~~~bash
UV_HTTP_TIMEOUT=300 uv pip install --no-deps -U \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128 \
  2>&1 | tee $ISAAC_ROOT/logs/isaacsim6-torch-install.log
~~~

The `--no-deps` form is the local strict lane because it preserves Isaac Sim package pins while replacing Torch with the official cu128 wheel.

## 3. Set Non-Interactive Runtime Variables

~~~bash
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONNOUSERSITE=1
~~~

`PYTHONNOUSERSITE=1` prevents user-site packages from contaminating Isaac imports.

## 4. Verify Isaac Sim

~~~bash
$ISAAC_ROOT/venv/bin/python - <<'PY'
import importlib.metadata as md
for package in ['isaacsim', 'isaacsim-app', 'isaacsim-kernel', 'torch']:
    print(package, md.version(package))
PY

OMNI_KIT_ACCEPT_EULA=YES $ISAAC_ROOT/venv/bin/isaacsim --help
~~~

Expected local versions from the final verifier:

| Package | Version |
| --- | --- |
| isaacsim | 6.0.0.0 |
| isaacsim-app | 6.0.0.0 |
| isaacsim-kernel | 6.0.0.0 |
| torch | 2.10.0+cu128 |

## 5. Clone Isaac Lab 3 Source

Use the pinned Isaac Lab 3 checkout for the validation lane:

~~~bash
cd $ISAAC_ROOT/sources
git clone --filter=blob:none https://github.com/isaac-sim/IsaacLab.git IsaacLab-release-3.0.0-beta2
cd IsaacLab-release-3.0.0-beta2
git checkout 9fe080c1a1c73e8f8a7a8f971f98517876a99861
~~~

That local checkout is `release/3.0.0-beta2`. Keep newer Isaac Lab source checkouts for reference, but do not silently switch the validation env without rerunning the OceanScale verifier.

## 6. Install Isaac Lab Core

Run from the Isaac Lab source checkout with the Isaac validation venv active:

~~~bash
cd $ISAAC_ROOT/sources/IsaacLab-release-3.0.0-beta2
source $ISAAC_ROOT/venv/bin/activate
PYTHONNOUSERSITE=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -i none \
  2>&1 | tee $ISAAC_ROOT/logs/isaaclab3-core-install.log
~~~

The `-i none` core install is the OceanScale-tested baseline. Install optional RL frameworks only when needed and record the extra command and result.

## 7. Verify Isaac Lab

Official docs show GUI verification with `--viz kit`. For headless OceanScale validation use `--viz none`:

~~~bash
cd $ISAAC_ROOT/sources/IsaacLab-release-3.0.0-beta2
PYTHONNOUSERSITE=1 ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES \
  $ISAAC_ROOT/venv/bin/python scripts/tutorials/00_sim/create_empty.py --viz none
~~~

Then run the OceanScale verifier:

~~~bash
cd $OCEANSCALE_ROOT
python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py --isaac-root "$ISAAC_ROOT"
~~~

Do not call the Isaac lane ready if the verifier reports errors.

## OceanScale Development Environment

OceanScale development remains its own environment:

~~~bash
cd $OCEANSCALE_ROOT
uv sync --extra dev
uv run pytest tests/ -v
~~~

Expected final verifier versions for the OceanScale dev env:

| Package | Version |
| --- | --- |
| oceanscale | 0.1.0a0 |
| warp-lang | 1.13.0 |
| newton | 1.2.0 |
| torch | 2.11.0+cu128 |
| gymnasium | 1.2.3 |
| isaacsim | installed in the Isaac validation env, not the core `.venv` |
| isaaclab | installed in the Isaac validation env, not the core `.venv` |
| isaaclab_newton | installed in the Isaac validation env, not the core `.venv` |

This separation is intentional: the Isaac validation env currently carries `newton==1.0.0`, while OceanScale core is on `newton==1.2.0`.

## Source Code Baseline

All source trees are side by side and must not overwrite older installs.

| Component | Path | Branch / head |
| --- | --- | --- |
| Isaac Sim source | `/mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop` | `develop` / `f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc` |
| Isaac Lab latest develop | `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-develop` | `develop` / `f39f5b231b84daab09c1a1ecb1b7efa4bd56340f` |
| Isaac Lab 3.0 beta2 | `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2` | `release/3.0.0-beta2` / `9fe080c1a1c73e8f8a7a8f971f98517876a99861` |
| Isaac Lab main reference | `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-main-2026-05-25` | `main` / `54a65ea830c6002e17dc18c77831fa60e43937bc` |
| Newton source | `/mnt/storage/isaacsim-6.0-official/sources/newton-v1.2.0` | `v1.2.0` / `a886e3fb411137d8a6ff370a1f3da427eccefbed` |

Source reference clone pattern:

~~~bash
cd $ISAAC_ROOT/sources
GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none \
  -b develop https://github.com/isaac-sim/IsaacSim.git IsaacSim-develop

git clone --filter=blob:none \
  https://github.com/newton-physics/newton.git newton-v1.2.0
cd newton-v1.2.0
git checkout a886e3fb411137d8a6ff370a1f3da427eccefbed
~~~

Use `GIT_LFS_SKIP_SMUDGE=1` for source reference checkouts until large LFS assets are deliberately needed.

## Pip Runtime vs Source Build

| Path | Role | Status |
| --- | --- | --- |
| Isaac Sim pip install | Main runnable Isaac Sim 6 runtime | tested and passing |
| Isaac Lab source install | Main Isaac Lab 3 validation lane | tested and passing |
| Isaac Sim source checkout | Official reference and future source-build base | checked out |
| Isaac Sim source build | Only if modifying/building Isaac Sim itself | not part of current gate |

Pip is the main runtime because it is the official Python install route for Isaac Sim 6 and is faster to reproduce for OceanScale validation. Source is retained because OceanScale is based on the NVIDIA Isaac 6 ecosystem and needs upstream code for API inspection, patch triage, and future source builds.

## Optional: Strict Isaac Sim Runtime Smoke Venv

This optional venv supports `artifacts/isaacsim/run_isaacsim_standalone_smokes.py`. It is not required for the OceanScale Isaac baseline verifier.

~~~bash
export ISAAC_ROOT=/mnt/storage/isaacsim-6.0-official
uv venv --python 3.12 --seed $ISAAC_ROOT/venv-isaacsim-strict
source $ISAAC_ROOT/venv-isaacsim-strict/bin/activate
uv pip install --upgrade pip
UV_HTTP_TIMEOUT=300 uv pip install --prerelease=allow \
  'isaacsim[all,extscache]==6.0.0.0' \
  --extra-index-url https://pypi.nvidia.com
UV_HTTP_TIMEOUT=300 uv pip install --no-deps -U \
  torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
~~~

Run a bounded runtime smoke with explicit paths:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --root "$ISAAC_ROOT" \
  --sim-root "$ISAAC_ROOT/sources/IsaacSim-develop" \
  --venv "$ISAAC_ROOT/venv-isaacsim-strict" \
  --logs-dir "$ISAAC_ROOT/logs/isaacsim-standalone-smokes-manual" \
  --case simulation_app_headless_no_rendering \
  --case validation_extension_count_headless \
  --case python_sh_import_sys \
  --case python_sh_path_length
~~~

## Advanced: Isaac Sim Source Build

Only use this path when deliberately building Isaac Sim itself.

Official Isaac Sim source README says:

- clone `isaac-sim/IsaacSim` branch `develop`
- run `git lfs install` and `git lfs pull` before a real build
- on Ubuntu 24.04, use GCC/G++ 11
- build with `./build.sh`
- launch from `_build/linux-x86_64/release/isaac-sim.sh`

Skeleton:

~~~bash
cd $ISAAC_ROOT/sources/IsaacSim-develop
git lfs install
git lfs pull
gcc --version
g++ --version
./build.sh
cd _build/linux-x86_64/release
./isaac-sim.sh
~~~

This is not the normal OceanScale verification path.

## Official Demo Coverage Boundary

The full official demo sweep is not the current OceanScale install gate.

Current coverage snapshot:

| Scope | Passed | Failed | Uncovered |
| --- | ---: | ---: | ---: |
| all | 144 | 32 | 1923 |
| isaacsim | 110 | 16 | 81 |
| isaaclab | 34 | 16 | 1842 |

Interpretation:

- The Isaac Sim pip lane is valid for runtime tests and OceanScale validation.
- Some source-tree tests assume a source-build `python.sh` layout and are not valid pip-lane failures.
- Several Isaac Lab 3 beta2 demo failures appear to be upstream beta/API drift, not OceanScale failures.
- Full official demo sweep, external ROS2 graph validation, long-horizon RL training, and source build remain separate gates.

## Reproducibility Checklist

1. Confirm `nvidia-smi`, Python 3.12, `uv`, `git`, and `git lfs`.
2. Create an isolated Isaac root and venv.
3. Install Isaac Sim 6 from `https://pypi.nvidia.com`.
4. Install official cu128 Torch for the Isaac lane.
5. Clone the pinned Isaac Lab 3 source checkout.
6. Run `./isaaclab.sh -i none`.
7. Run `python3 artifacts/isaacsim/verify_oceanscale_latest_isaac_baseline.py --isaac-root "$ISAAC_ROOT"` from `$OCEANSCALE_ROOT`.
8. Save the verifier `summary.json` path in `artifacts/isaacsim/`.
9. Do not call the environment ready if the verifier has errors.

## Troubleshooting

### Isaac import fails with user-site or NumPy errors

Use:

~~~bash
export PYTHONNOUSERSITE=1
~~~

Then rerun from the intended venv.

### Isaac Lab headless warnings

Use `--viz none` for bounded headless smoke tests unless a GUI is required.

### Source-tree `python_sh` tests fail under pip install

Those tests may assert source-build archive paths such as `omni.pip.compute` or `omni.isaac.ml_archive`. Treat them as source-layout mismatches unless the same failure appears in the OceanScale verifier.

### Optional Isaac Lab install fails

Keep the core install as the baseline:

~~~bash
./isaaclab.sh -i none
~~~

Install optional frameworks only when needed and record each extra command.

### Isaac Sim source build wants the wrong compiler

Set up GCC/G++ 11 before source build. Do not use the default Ubuntu 24.04 G++ 13 path for the current Isaac Sim source README build path.

## Evidence Files

- `artifacts/isaacsim/oceanscale_isaac6_final_summary_2026-05-26.md`
- `artifacts/isaacsim/oceanscale_latest_isaac_baseline_2026-05-25.md`
- `artifacts/isaacsim/isaacsim6_official_setup_log_2026-05-25.md`
- `/mnt/storage/isaacsim-6.0-official/logs/oceanscale-latest-isaac-baseline-2026-05-26-125621/summary.json`
