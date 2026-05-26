# Isaac Sim 6 Official Local Setup Log - 2026-05-25

Status: local Isaac Sim 6 runtime installed and smoke-tested; Isaac Lab 3 core source install completed; source trees cloned; first Isaac Lab official smoke batches passed; Newton/MJWarp and PhysX bounded Cartpole runtime passed; Isaac Sim standalone smoke runner added with broad official smoke coverage including Replicator/SDG, ROS2 internal bridge/clock/camera timestamp, core API, contacts, camera/pre-ISP/CameraView, RTX Lidar/Radar GMO, StableIdMap object IDs, wheeled robot, cloner, cloth, deformable stress, experimental control, and motion generation; coverage index now tracks 87 clean passed, 18 clean failed, and 1994 uncovered official Python entrypoints; full source build and full official demo sweep not run yet.

## User decisions

- Isaac Sim 6.0 is the main local Isaac Sim target.
- Older Isaac Sim / Isaac Lab installs must remain available side-by-side and must not be overwritten.
- Use official docs/repos as source of truth for Isaac Sim, Isaac Lab, Newton, PyTorch/CUDA, and ROS references.
- Log commands, paths, version decisions, blockers, and verification evidence for future reproduction.

## Official references consulted

- Isaac Sim 6.0 docs index: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html
- Isaac Sim 6.0 release notes: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html
- Isaac Sim 6.0 Python install: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html
- Isaac Sim 6.0 requirements: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
- Isaac Sim 6.0 download page: https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/download.html
- Official Isaac Sim source repo README: https://github.com/isaac-sim/IsaacSim/tree/develop
- Isaac Lab 3 source install docs from /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2/docs/source/setup/installation/index.rst.
- Isaac Lab 3 verification docs from /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2/docs/source/setup/installation/include/src_verify_isaaclab.rst.

## Official facts captured

- Isaac Sim 6.0 docs banner says: Isaac Sim 6.0 Early Developer Release; docs are incomplete; source build from isaac-sim/IsaacSim develop is the documented early developer path; binaries, pip packages, pre-built containers, and other artifacts are expected with GA.
- NVIDIA package index also currently resolves isaacsim as 6.0.0.0; the Python install page lists: pip install isaacsim[all,extscache]==6.0.0 --extra-index-url https://pypi.nvidia.com.
- Isaac Sim 6.0 requires Python 3.12.
- Isaac Sim 6.0 release notes: Kit 110.0, experimental Newton backend, Newton/MuJoCo-Warp solver backend, ROS 2 modularization, URDF/MJCF Importer 3.0, Asset Structure 3.0.
- Official PyTorch command from Isaac Sim 6.0 Python install page for CUDA 12: pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128.
- Official Isaac Sim source README says source build quickstart is git clone -b develop, confirm GCC/G++ 11 on Linux, then run ./build.sh and launch from _build/linux-x86_64/release.
- Official Isaac Sim source README says Ubuntu 24.04 builds require GCC/G++ 11; GCC/G++ 12+ is not supported for that source build path.

## Local system facts captured

- CWD: /home/robot/workspace/46-marine.
- GPU: NVIDIA GeForce RTX 5090, driver 580.95.05, VRAM 32607 MiB.
- python3.12 --version: Python 3.12.13.
- Project .venv/bin/python --version: Python 3.12.13.
- python3 --version: Python 3.13.5.
- git-lfs: /usr/bin/git-lfs.
- gcc --version | head -1: gcc (Ubuntu 11.4.0-9ubuntu1) 11.4.0.
- g++ --version | head -1: g++ (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0.
- Official source build on Ubuntu 24.04 wants GCC/G++ 11; local g++ default is 13, so source-build would need explicit G++ 11 setup before use.
- Disk check before install: /mnt/storage had about 404G available; root /home/robot had about 73G available.
- Existing old Isaac-related tree found at /home/robot/workspace/08-isaaclab/isaacsim; do not overwrite.

## Side-by-side install layout

- New main Isaac Sim 6 path: /mnt/storage/isaacsim-6.0-official.
- New main venv: /mnt/storage/isaacsim-6.0-official/venv.
- Pip cache: /mnt/storage/isaacsim-6.0-official/pip-cache.
- Temporary directory: /mnt/storage/isaacsim-6.0-official/tmp.
- Command logs directory: /mnt/storage/isaacsim-6.0-official/logs.
- Existing older installs remain untouched.

## Commands run so far

~~~bash
curl -L --max-time 25 -s https://docs.isaacsim.omniverse.nvidia.com/6.0.0/index.html
curl -L --max-time 25 -s https://docs.isaacsim.omniverse.nvidia.com/6.0.0/overview/release_notes.html
curl -L --max-time 25 -s https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/install_python.html
curl -L --max-time 25 -s https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html
curl -L --max-time 25 -s https://raw.githubusercontent.com/isaac-sim/IsaacSim/develop/README.md
~~~

~~~bash
which isaacsim || true
python3 --version
python3.12 --version || true
python3.12 -m pip index versions isaacsim --extra-index-url https://pypi.nvidia.com
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
find /home/robot /mnt/storage -maxdepth 3 \( -iname "*isaac*sim*" -o -iname "IsaacSim" -o -name "isaac-sim.sh" \) 2>/dev/null | head -100
~~~

~~~bash
test -e /mnt/storage/isaacsim-6.0-official && echo EXISTS || echo MISSING
mkdir -p /mnt/storage/isaacsim-6.0-official
mkdir -p /mnt/storage/isaacsim-6.0-official/pip-cache /mnt/storage/isaacsim-6.0-official/tmp
python3.12 -m venv /mnt/storage/isaacsim-6.0-official/venv
/mnt/storage/isaacsim-6.0-official/venv/bin/python --version
~~~

The first venv was recreated cleanly before the successful torch install:

~~~bash
rm -rf /mnt/storage/isaacsim-6.0-official/venv /mnt/storage/isaacsim-6.0-official/tmp
mkdir -p /mnt/storage/isaacsim-6.0-official/tmp /mnt/storage/isaacsim-6.0-official/pip-cache
python3.12 -m venv /mnt/storage/isaacsim-6.0-official/venv
PIP_CACHE_DIR=/mnt/storage/isaacsim-6.0-official/pip-cache \
TMPDIR=/mnt/storage/isaacsim-6.0-official/tmp \
/mnt/storage/isaacsim-6.0-official/venv/bin/python -m pip install --upgrade pip
PIP_CACHE_DIR=/mnt/storage/isaacsim-6.0-official/pip-cache \
TMPDIR=/mnt/storage/isaacsim-6.0-official/tmp \
/mnt/storage/isaacsim-6.0-official/venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
~~~

## Completed install evidence

- Torch install completed successfully in the isolated 6.0 venv.
- Installed torch package: torch-2.10.0+cu128.
- Installed CUDA support packages included nvidia-cublas-cu12-12.8.4.1, nvidia-cudnn-cu12-9.10.2.21, nvidia-cusolver-cu12-11.7.3.90, nvidia-cusparse-cu12-12.5.8.93, nvidia-nccl-cu12-2.27.5, and triton-3.6.0.
- Isaac Sim 6.0 pip install completed successfully from the official NVIDIA package index into /mnt/storage/isaacsim-6.0-official/venv.
- Full Isaac Sim install log: /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-pip-install-2026-05-25.log.
- Pip freeze snapshot: /mnt/storage/isaacsim-6.0-official/logs/pip-freeze-2026-05-25.txt.
- pip show verified:
  - isaacsim 6.0.0.0
  - isaacsim-core 6.0.0.0
  - newton 1.0.0
  - mujoco 3.5.0
  - mujoco-warp 3.5.0.2
  - warp-lang 1.15.0.dev20260525
- Isaac Sim CLI verification passed with OMNI_KIT_ACCEPT_EULA=YES /mnt/storage/isaacsim-6.0-official/venv/bin/isaacsim --help; it reported Kit Version 110.0.0+feature.276876.4a5123f4.gl.
- Python import smoke passed:
  - torch 2.10.0+cu128, CUDA 12.8, CUDA available true
  - device: NVIDIA GeForce RTX 5090
  - newton 1.0.0
  - mujoco 3.5.0
  - warp 1.15.0.dev20260525
- The official Isaac Sim 6.0 pip dependency set pins Newton to 1.0.0 inside the Isaac Sim runtime. Any newer standalone Newton line should be treated as a separate source/runtime decision, not assumed to be Isaac Sim's bundled Newton version.
- Current installed 6.0 runtime footprint observed: about 23G under /mnt/storage/isaacsim-6.0-official.

## Next commands to run

~~~bash
mkdir -p /mnt/storage/isaacsim-6.0-official/logs
PIP_CACHE_DIR=/mnt/storage/isaacsim-6.0-official/pip-cache \
TMPDIR=/mnt/storage/isaacsim-6.0-official/tmp \
/mnt/storage/isaacsim-6.0-official/venv/bin/python -m pip install "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-pip-install-2026-05-25.log
~~~

After install:

~~~bash
/mnt/storage/isaacsim-6.0-official/venv/bin/python -m pip show isaacsim
/mnt/storage/isaacsim-6.0-official/venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
~~~

## Source-code requirement

User requested source code in addition to the pip/runtime install.

Planned source root:

~~~bash
/mnt/storage/isaacsim-6.0-official/sources
~~~

Planned official source checkouts:

- Isaac Sim official source: https://github.com/isaac-sim/IsaacSim.git, branch develop, matching the Isaac Sim 6.0 early developer README.
- Isaac Lab official source: verify latest 3.x route from https://github.com/isaac-sim/IsaacLab before checkout.
- Newton official source: verify latest official route from https://github.com/newton-physics/newton before checkout.

For Isaac Sim source, start with source checkout only using GIT_LFS_SKIP_SMUDGE=1 to avoid implicitly pulling large binary/LFS artifacts. If a source build is needed later, follow the official README and run git lfs pull deliberately.

## Source checkouts completed

Source root:

~~~bash
/mnt/storage/isaacsim-6.0-official/sources
~~~

Completed source trees:

| Source | Path | Ref | Commit |
| --- | --- | --- | --- |
| Isaac Sim | /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop | develop | f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc |
| Isaac Lab | /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2 | release/3.0.0-beta2 | 9fe080c1a1c73e8f8a7a8f971f98517876a99861 |
| Newton runtime match | /mnt/storage/isaacsim-6.0-official/sources/newton-v1.0.0 | v1.0.0 | d6046f187f1f6c6b8f8da98c5d0f93b8944eb5f0 |
| Newton latest tag observed | /mnt/storage/isaacsim-6.0-official/sources/newton-v1.2.0 | v1.2.0 | a886e3fb411137d8a6ff370a1f3da427eccefbed |
| MuJoCo Warp main | /mnt/storage/isaacsim-6.0-official/sources/mujoco_warp-main | main | cb350933f2013be9d506493b1cfa6f01f2830950 |
| MuJoCo Warp runtime match | /mnt/storage/isaacsim-6.0-official/sources/mujoco_warp-v3.5.0.2 | v3.5.0.2 | 1166e75230ec8c1f16f9484ee03ad74098d3201a |

Source checkout commands used:

~~~bash
mkdir -p /mnt/storage/isaacsim-6.0-official/sources

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none -b develop \
  https://github.com/isaac-sim/IsaacSim.git \
  /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none -b release/3.0.0-beta2 \
  https://github.com/isaac-sim/IsaacLab.git \
  /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none -b v1.0.0 \
  https://github.com/newton-physics/newton.git \
  /mnt/storage/isaacsim-6.0-official/sources/newton-v1.0.0

GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none -b v1.2.0 \
  https://github.com/newton-physics/newton.git \
  /mnt/storage/isaacsim-6.0-official/sources/newton-v1.2.0

git clone --depth 1 --filter=blob:none \
  https://github.com/google-deepmind/mujoco_warp.git \
  /mnt/storage/isaacsim-6.0-official/sources/mujoco_warp-main

git clone --depth 1 --filter=blob:none -b v3.5.0.2 \
  https://github.com/google-deepmind/mujoco_warp.git \
  /mnt/storage/isaacsim-6.0-official/sources/mujoco_warp-v3.5.0.2
~~~

Official source notes captured:

- Isaac Lab release/3.0.0-beta2 README says it is a development branch for Isaac Sim 6.0.
- Isaac Lab README warns that a recent Isaac Lab develop branch breaking change is not compatible with Isaac Sim develop, and recommends Isaac Lab commit f0234a82e432e2a0b0f0a26ca3c5b59e527ddaaa or earlier, or the v3.0.0-beta tag for Isaac Sim GitHub develop.
- Isaac Lab README dependency table says develop branch maps to Isaac Sim 6.0, while main and v2.3.x map to Isaac Sim 4.5 / 5.0 / 5.1.
- Newton v1.0.0 README says Python 3.10+, NVIDIA GPU driver 545 or newer for CUDA 12, no local CUDA toolkit required, and quickstart is pip install "newton[examples]".
- MJWarp README says MJWarp is maintained by Google DeepMind and NVIDIA as part of the Newton project and is available as pip install mujoco-warp.

## Local Isaac Sim 6 smoke test

Command:

~~~bash
OMNI_KIT_ACCEPT_EULA=YES /mnt/storage/isaacsim-6.0-official/venv/bin/python - <<'PY' 2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-simulationapp-smoke-2026-05-25.log
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("SIMULATION_APP_STARTED")
app.close()
print("SIMULATION_APP_CLOSED")
PY
~~~

Evidence from /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-simulationapp-smoke-2026-05-25.log:

- Driver Version: 580.95.05, Graphics API: Vulkan.
- GPU detected: NVIDIA GeForce RTX 5090, 32607 MB.
- omni.usd.schema.newton-1.1.0 started.
- isaacsim.asset.importer.urdf-3.2.1 started.
- isaacsim.sensors.experimental.physics-2.2.0 started.
- Simulation App Startup Complete at 7.559s.
- Simulation App Shutting Down at 7.925s.

## Isaac Lab 3 source install

Official source path:

~~~bash
/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
~~~

Commands:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH ./isaaclab.sh --help

VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
  ./isaaclab.sh -i 2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-source-install-2026-05-25.log

VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
  ./isaaclab.sh -i none 2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-core-install-2026-05-25.log
~~~

Install result:

- ./isaaclab.sh --help worked in the Isaac Sim 6 venv.
- Default ./isaaclab.sh -i attempted optional extras and failed on isaaclab_mimic through egl-probe==1.0.2 with a CMake compatibility error: Compatibility with CMake < 3.5 has been removed from CMake.
- The official selective core install ./isaaclab.sh -i none completed successfully.
- The core install warned that _isaac_sim was not found inside the Isaac Lab source checkout, so VS Code settings were generated without Isaac Sim extra paths. This is expected for the pip Isaac Sim layout rather than a bundled binary checkout.
- Isaac Lab changed the runtime dependency set: warp-lang became 1.13.0 after core install. Isaac Sim was smoke-tested again after this change.

Installed package versions after Isaac Lab core install:

~~~text
isaacsim==6.0.0.0
isaaclab==6.0.0
isaaclab-newton==0.12.0
newton==1.0.0
mujoco==3.5.0
mujoco-warp==3.5.0.2
warp-lang==1.13.0
torch==2.10.0+cu128
~~~

Pip freeze after Isaac Lab:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/pip-freeze-after-isaaclab-2026-05-25.txt
~~~

## Isaac Lab 3 verification attempt

Official verification script from Isaac Lab docs:

~~~bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz kit
~~~

Headless local attempts:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
OMNI_KIT_ACCEPT_EULA=YES VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
  timeout --signal=INT 60s ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-create-empty-headless-2026-05-25.log

OMNI_KIT_ACCEPT_EULA=YES VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
  timeout --signal=INT 120s ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --viz none \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-create-empty-viz-none-2026-05-25.log
~~~

Result:

- Both attempts reached AppLauncher initialization and detected RTX 5090 through Vulkan.
- Both attempts exited with timeout code 124 after the bounded runtime.
- No traceback or fatal error appeared in either log.
- The tutorial did not reach its script-level marker [INFO]: Setup complete..., so this is recorded as a partial verification only, not a clean pass.
- The first command used --headless and Isaac Lab warned that --headless is deprecated; --viz none is the newer headless route.

## Post-Isaac-Lab Isaac Sim smoke test

Command:

~~~bash
OMNI_KIT_ACCEPT_EULA=YES /mnt/storage/isaacsim-6.0-official/venv/bin/python - <<'PY' 2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-post-isaaclab-smoke-2026-05-25.log
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})
print("POST_ISAACLAB_SIMULATION_APP_STARTED")
app.close()
print("POST_ISAACLAB_SIMULATION_APP_CLOSED")
PY
~~~

Evidence from /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-post-isaaclab-smoke-2026-05-25.log:

- Driver Version: 580.95.05, Graphics API: Vulkan.
- GPU detected: NVIDIA GeForce RTX 5090, 32607 MB.
- omni.usd.schema.newton-1.1.0 started.
- isaacsim.asset.importer.urdf-3.2.1 started.
- isaacsim.sensors.experimental.physics-2.2.0 started.
- Simulation App Startup Complete at 8.351s.
- Simulation App Shutting Down at 8.824s.
- Command exit code: 0.

## Pip install vs build from source decision

Current main local environment:

- Use official pip install as the runnable Isaac Sim 6 runtime.
- Keep official source trees beside it for audit, reference, future patching, and source builds.
- Do not overwrite older Isaac Sim / Isaac Lab installs.

Why pip install is the main runtime now:

- It is the official Python install command in the Isaac Sim 6.0 docs.
- It installed into an isolated venv under /mnt/storage/isaacsim-6.0-official/venv.
- It already passed CLI, import, GPU, and SimulationApp startup/shutdown smoke tests on the local RTX 5090.
- It is faster and more reproducible for day-to-day Isaac Lab / OceanScale work than a full source build.

Why source build is still kept:

- Isaac Sim 6.0 docs identify the GitHub source path as the early developer source route.
- Source is needed if we must patch Isaac Sim extensions, debug source-level behavior, package a custom binary, or align with upstream development commits before GA artifacts settle.
- The source trees are cloned but large LFS assets were not pulled automatically.

Source build not run yet:

- Official source build requires explicit GCC/G++ 11 use on Ubuntu 24.04; local gcc is 11.4.0 but default g++ is 13.3.0.
- A real source build should first set up g++-11 explicitly, then intentionally run git lfs pull if needed.
- Source build command per official README is ./build.sh, then launch from _build/linux-x86_64/release.
- Local dry probe of `./build.sh --help` does not show help until the source EULA has been accepted.
- `ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ./build.sh --help` still exits before help; the source script checks only `.eula_accepted` or interactive `yes`.

## Official smoke runner added

Local reproducibility helpers added in this repo:

~~~text
/home/robot/workspace/46-marine/artifacts/isaacsim/run_isaaclab_official_smokes.py
/home/robot/workspace/46-marine/artifacts/isaacsim/isaaclab_bounded_random_agent.py
/home/robot/workspace/46-marine/artifacts/isaacsim/build_official_suite_inventory.py
/home/robot/workspace/46-marine/artifacts/isaacsim/run_isaaclab_official_next_smokes.py
~~~

The smoke runner keeps official Isaac Lab commands bounded and records per-case logs plus a JSON summary. The bounded random-agent helper mirrors the official Isaac Lab random-agent launch path but adds a fixed step count so solver smoke tests can exit cleanly without killing an infinite demo loop.

Smoke runner command:

~~~bash
cd /home/robot/workspace/46-marine
python3 artifacts/isaacsim/run_isaaclab_official_smokes.py
~~~

Smoke runner output:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/official-smokes-2026-05-25/summary.json
~~~

Summary:

| Case | Result | Evidence |
| --- | --- | --- |
| list_envs_cartpole | passed | markers: Available Environments in Isaac Lab, Isaac-Cartpole-Direct-v0 |
| list_envs_cartpole_presets | passed | markers: physics: newton_mjwarp, renderer: isaacsim_rtx_renderer |
| tutorial_00_create_empty | passed as expected-timeout demo | reached [INFO]: Setup complete..., then SIGINT at 90s because tutorial is designed to keep running |
| tutorial_00_launch_app | passed as expected-timeout demo | reached [INFO]: Setup complete..., then SIGINT at 90s because tutorial is designed to keep running |
| pytest_core_light | passed | 36 passed |
| pytest_sim_context_headless | passed | 15 passed |

Important correction:

- Earlier create_empty runs were marked partial because stdout was buffered and the setup marker did not appear before timeout. Re-running with PYTHONUNBUFFERED=1 showed [INFO]: Setup complete..., so the official create_empty smoke should be treated as passed under bounded expected-timeout semantics.

## Isaac Sim 6 standalone examples/tests

Commands and logs:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacSim-develop

OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout --signal=INT 180s /mnt/storage/isaacsim-6.0-official/venv/bin/python \
  source/standalone_examples/testing/isaacsim.simulation_app/test_headless_no_rendering.py \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-standalone-test-headless-no-rendering-2026-05-25.log

OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
  timeout --signal=INT 240s /mnt/storage/isaacsim-6.0-official/venv/bin/python \
  source/standalone_examples/testing/isaacsim.core.api/test_hello_world.py --test \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaacsim6-standalone-core-test-hello-world-2026-05-25.log
~~~

Results:

- source/standalone_examples/testing/isaacsim.simulation_app/test_headless_no_rendering.py passed with exit 0.
- The first run of test_headless_no_rendering.py synced missing Kit 110 extensions from the Omniverse extension registry into the local Omniverse cache. This first-run registry/cache state should be considered part of reproduction.
- source/standalone_examples/testing/isaacsim.core.api/test_hello_world.py --test passed with exit 0 and reached Simulation App Startup Complete / Simulation App Shutting Down.
- source/standalone_examples/api/isaacsim.simulation_app/hello_world.py reached Simulation App Startup Complete and app ready, but did not exit in this no-DISPLAY/headless environment before manual SIGINT. Record this as partial for no-display automation, not a clean headless pass.

## Isaac Lab Newton/MJWarp and PhysX bounded runtime

Newton/MJWarp command:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
timeout --signal=INT 180s ./isaaclab.sh -p /home/robot/workspace/46-marine/artifacts/isaacsim/isaaclab_bounded_random_agent.py \
  --task Isaac-Cartpole-Direct-v0 --num_envs 4 --steps 16 --headless physics=newton_mjwarp \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-bounded-cartpole-newton-mjwarp-2026-05-25.log
~~~

Newton/MJWarp result:

- Passed with exit 0.
- Created Isaac-Cartpole-Direct-v0 with 4 envs on cuda:0.
- Initialized Newton manager, finalized builder, initialized solver, created CUDA graph.
- Ran 16 random-action steps and printed step 1 / step 16 reward summaries.

PhysX command:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
timeout --signal=INT 180s ./isaaclab.sh -p /home/robot/workspace/46-marine/artifacts/isaacsim/isaaclab_bounded_random_agent.py \
  --task Isaac-Cartpole-Direct-v0 --num_envs 4 --steps 16 --headless physics=physx \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-bounded-cartpole-physx-2026-05-25.log
~~~

PhysX result:

- Passed with exit 0.
- Created Isaac-Cartpole-Direct-v0 with 4 envs on cuda:0.
- Ran 16 random-action steps and printed step 1 / step 16 reward summaries.

Command-level caveat:

- Official docs say --headless is deprecated and --viz none is the newer visualizer-disabling route.
- For scripts/environments/random_agent.py with physics=newton_mjwarp, --viz none failed locally with RuntimeError: Explicitly requested visualizer(s) ['none'] could not be configured. Valid types: 'newton', 'rerun', 'viser', 'kit'.
- The same Newton/MJWarp task passed when using --headless, so use --headless for current automated solver smoke until this 3.0 beta visualizer edge is resolved or explained upstream.

## Official suite inventory generated

Command:

~~~bash
cd /home/robot/workspace/46-marine
python3 -m py_compile artifacts/isaacsim/build_official_suite_inventory.py
python3 artifacts/isaacsim/build_official_suite_inventory.py
~~~

Generated files:

- /home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_inventory.json
- /home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_inventory.md

Inventory result:

| Area | Count |
| --- | ---: |
| Total official Python entrypoints inventoried | 2099 |
| Isaac Sim source/standalone_examples entrypoints | 207 |
| Isaac Lab scripts/source entrypoints | 1892 |
| Classified headless candidates | 1340 |
| Classified bounded expected-timeout entrypoints | 50 |
| Classified GUI/default-display entrypoints | 93 |
| Classified long-runtime entrypoints | 247 |
| Classified asset/network-dependent entrypoints | 160 |
| Classified interactive/device-dependent entrypoints | 127 |
| Classified optional mimic/imitation blocked-or-long entrypoints | 63 |
| Classified ROS 2 entrypoints | 19 |

## Isaac Lab next official tutorial/demo batch

Official docs/source references checked for this batch:

- docs/source/overview/developer-guide/repo_structure.rst defines scripts/demos as demo applications and scripts/tutorials as step-by-step tutorials.
- docs/source/api/lab/isaaclab.app.rst documents HEADLESS/LIVESTREAM/ENABLE_CAMERAS behavior.
- source/isaaclab/isaaclab/app/app_launcher.py documents --headless as deprecated and --viz none as the explicit visualizer-disable path.

Runner command:

~~~bash
cd /home/robot/workspace/46-marine
python3 -m py_compile artifacts/isaacsim/run_isaaclab_official_next_smokes.py
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py
~~~

Runner output:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25/summary.json
/mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-demo-quadcopter-viznone/summary.json
~~~

Summary:

| Case | Result | Evidence |
| --- | --- | --- |
| tutorial_00_log_time | passed as expected-timeout demo | reached [INFO]: Setup complete... and [INFO] Logging experiment to directory: |
| tutorial_01_run_rigid_object | passed as expected-timeout demo | reached [INFO]: Setup complete..., [INFO]: Resetting object state..., and CUDA root-position tensor output |
| tutorial_01_run_articulation | passed as expected-timeout demo | reached [INFO]: Setup complete... and repeated [INFO]: Resetting robot state... |
| tutorial_02_create_scene | passed as expected-timeout demo | reached [INFO]: Setup complete... and repeated [INFO]: Resetting robot state... |
| tutorial_03_create_cartpole_base_env | passed as expected-timeout demo | created 4 cuda:0 envs, printed manager tables, reached [INFO]: Resetting environment... and [Env 0]: Pole joint: |
| tutorial_03_run_cartpole_rl_env | passed as expected-timeout demo | created 4 cuda:0 envs, printed command/event/action/observation/termination/reward managers, reached [INFO]: Resetting environment... and [Env 0]: Pole joint: |
| demo_quadcopter | failed official demo | --headless path still tried the script's default kit visualizer and failed before setup; --viz none bypassed visualizer but failed at robot.root_view.get_masses().sum() with AttributeError: 'array' object has no attribute 'sum' |

Additional quadcopter diagnostic:

~~~bash
cd /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1 \
VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv PATH=/mnt/storage/isaacsim-6.0-official/venv/bin:$PATH \
timeout --signal=INT 120s ./isaaclab.sh -p scripts/demos/quadcopter.py --viz none \
  2>&1 | tee /mnt/storage/isaacsim-6.0-official/logs/isaaclab3-demo-quadcopter-viz-none-2026-05-25.log
~~~

Quadcopter failure details:

- Local release/3.0.0-beta2 checkout: 9fe080c1a1c73e8f8a7a8f971f98517876a99861.
- Remote main checked with git fetch only, without changing the local worktree: 54a65ea830c6002e17dc18c77831fa60e43937bc.
- release/3.0.0-beta2 uses robot.root_view.get_masses().sum().
- remote main uses robot.root_physx_view.get_masses().sum() in the same demo.
- Treat this as an official Isaac Lab demo incompatibility/bug in the currently installed beta2 source, not as an OceanScale failure.

Runner self-test after marker-driven stop update:

~~~bash
cd /home/robot/workspace/46-marine
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --case tutorial_00_log_time \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-runner-selftest
~~~

Result:

- Passed in 11.117s with marker_stopped=true, instead of waiting for the full 75s timeout.
- This runner behavior is now preferred for future bounded official demo sweeps.

## Isaac Lab next official tutorial/demo batch 2

Command:

~~~bash
cd /home/robot/workspace/46-marine
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --case tutorial_01_run_deformable_object_physx \
  --case tutorial_01_run_deformable_object_newton \
  --case tutorial_01_run_surface_gripper_cpu \
  --case tutorial_03_create_cube_base_env \
  --case tutorial_04_run_ray_caster \
  --case tutorial_04_run_frame_transformer \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-batch2
~~~

Runner output:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-batch2/summary.json
~~~

Summary:

| Case | Result | Evidence |
| --- | --- | --- |
| tutorial_01_run_deformable_object_physx | failed official tutorial | ValueError: Could not import module for backend 'physx' for factory DeformableObject; original error: No module named 'omni.physics.tensors.api' |
| tutorial_01_run_deformable_object_newton | passed as marker-stopped demo | reached [INFO]: Setup complete... and [INFO]: Resetting object state... |
| tutorial_01_run_surface_gripper_cpu | passed as marker-stopped demo | reached [INFO]: Setup complete... and [INFO]: Resetting robot state... |
| tutorial_03_create_cube_base_env | failed official tutorial | RuntimeError: Scene replication is enabled, which may affect USD-level randomization; local beta2 comment says replicate_physics should be False but code sets replicate_physics=True |
| tutorial_04_run_ray_caster | passed as marker-stopped demo | reached [INFO]: Setup complete... |
| tutorial_04_run_frame_transformer | failed official tutorial | ModuleNotFoundError: No module named 'isaacsim.util' |

Additional official-source comparison:

- Local release/3.0.0-beta2 create_cube_base_env.py sets MySceneCfg(..., replicate_physics=True) even though the adjacent comment says False is required.
- Remote main create_cube_base_env.py sets MySceneCfg(..., replicate_physics=False), matching the comment and runtime guard.
- Remote main run_frame_transformer.py still imports isaacsim.util.debug_draw._debug_draw, so that issue is not obviously fixed by the fetched main branch.

## Isaac Lab next official tutorial/demo batch 3

Command:

~~~bash
cd /home/robot/workspace/46-marine
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --case tutorial_00_spawn_prims \
  --case tutorial_00_set_rendering_mode \
  --case tutorial_01_add_new_robot \
  --case tutorial_03_create_quadruped_base_env \
  --case tutorial_04_add_sensors_on_robot \
  --case tutorial_04_run_ray_caster_camera \
  --case tutorial_04_run_usd_camera \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-batch3
~~~

Runner output:

~~~text
/mnt/storage/isaacsim-6.0-official/logs/official-next-smokes-2026-05-25-batch3/summary.json
~~~

Summary:

| Case | Result | Evidence |
| --- | --- | --- |
| tutorial_00_spawn_prims | passed as marker-stopped demo | reached [INFO]: Setup complete... after Nucleus/USD asset path |
| tutorial_00_set_rendering_mode | failed/timeout before setup | timed out at 180s after omni.client.python blocking warning while loading hospital scene; no traceback |
| tutorial_01_add_new_robot | failed after setup/reset | reached [INFO]: Setup complete... and [INFO]: Resetting Jetbot and Dofbot state..., then Warp RuntimeError because kernel expected array(ndim=2, dtype=float32) but received Tensor |
| tutorial_03_create_quadruped_base_env | failed official tutorial | ImportError: cannot import name AdditiveUniformNoiseCfg from isaaclab.utils.noise |
| tutorial_04_add_sensors_on_robot | failed/timeout before setup | camera/Replicator import path hit AttributeError: module 'warp' has no attribute 'context' |
| tutorial_04_run_ray_caster_camera | failed official tutorial | ModuleNotFoundError: No module named 'omni.replicator' |
| tutorial_04_run_usd_camera | failed official tutorial | omni.replicator.core import hit AttributeError: module 'warp' has no attribute 'context' |

Warp/Replicator diagnostic:

~~~bash
/mnt/storage/isaacsim-6.0-official/venv/bin/python - <<'PY'
import warp as wp
print('warp', getattr(wp, '__version__', None), 'has_context', hasattr(wp, 'context'))
print('warp_file', getattr(wp, '__file__', None))
PY
~~~

Result:

- warp-lang version is 1.13.0.
- import warp resolves to /mnt/storage/isaacsim-6.0-official/venv/lib/python3.12/site-packages/warp/__init__.py.
- hasattr(warp, 'context') is False, while omni.replicator.core annotators reference wp.context.Kernel.
- venv user site is disabled: ENABLE_USER_SITE=False and /home/robot/.local/lib/python3.12/site-packages is not in sys.path.

## Official docs/source recheck and clean Isaac Sim venv diagnostics

Official documentation references checked:

- Context7 official Isaac Lab main docs for Newton integration point to
  https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/installation.html.
- The documented Newton/Isaac Sim 6.0 install sequence is: create a Python 3.12 uv venv, install
  `isaacsim[all,extscache]==6.0.0` from `https://pypi.nvidia.com`, install
  `torch==2.10.0 torchvision==0.25.0` from the PyTorch cu128 index, then run `./isaaclab.sh -i`.
- The same docs say direct `pip` can replace `uv pip`, but the 3.0/Newton path is source-based; the stable pip-install docs still refer to Isaac Lab 2.3.x / Isaac Sim 5.1.

Official source branch reality on 2026-05-25:

~~~text
git ls-remote --heads --tags https://github.com/isaac-sim/IsaacLab.git
54a65ea830c6002e17dc18c77831fa60e43937bc refs/heads/main
9fe080c1a1c73e8f8a7a8f971f98517876a99861 refs/heads/release/3.0.0-beta2
a4a7602f29e755e2673fe0022ea35566df6dd7d5 refs/tags/v3.0.0-beta
~~~

- The latest official 3.x/6.0 branch currently visible is `release/3.0.0-beta2`; no final `v3.0.0` tag was visible.
- A side-by-side latest-main clone was created at
  `/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-main-2026-05-25`, HEAD
  `54a65ea830c6002e17dc18c77831fa60e43937bc`.
- That `main` checkout contains useful fixes for several beta2 tutorial/demo failures, but its setup.py classifiers and dependencies still target Isaac Sim 5.1-era pins such as `numpy<2` and `pillow==11.3.0`. Do not use `main` as the 6.0/3.x install baseline without a separate upstream confirmation.
- `python3.12 -m pip index versions isaaclab --extra-index-url https://pypi.nvidia.com` and with `--index-url https://pypi.nvidia.com` both returned `ERROR: No matching distribution found for isaaclab` on this machine, so the current 3.x lane remains source-based here.

Clean Isaac Sim official-order venv:

~~~bash
uv venv --python 3.12 --seed /mnt/storage/isaacsim-6.0-official/venv-isaacsim-only
source /mnt/storage/isaacsim-6.0-official/venv-isaacsim-only/bin/activate
uv pip install --upgrade pip
UV_HTTP_TIMEOUT=300 uv pip install --prerelease=allow "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
UV_HTTP_TIMEOUT=300 uv pip install -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
~~~

Observed install details:

- The exact docs command without `--prerelease=allow` failed under uv 0.9.18 because `isaacsim-core==6.0.0.0` requires `tinyobjloader==2.0.0rc13`.
- Increasing `UV_HTTP_TIMEOUT=300` was needed for large CUDA / extscache wheel downloads.
- After the official torch upgrade, `pip check` found Isaac Sim pin drift:
  `filelock==3.20.0`, `fsspec==2025.12.0`, `Jinja2==3.1.5`, `numpy==2.3.1`, `Pillow==12.1.1`, and `typing_extensions==4.12.2` were upgraded away by dependency resolution.
- Despite that pin drift, this venv passed a headless `SimulationApp` + `omni.replicator.core` import + `app.update()` smoke:
  `/mnt/storage/isaacsim-6.0-official/logs/isaacsim6-clean-replicator-smoke-2026-05-25.log`.

Strict Isaac Sim 6.0 baseline venv:

~~~bash
uv venv --python 3.12 --seed /mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict
source /mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict/bin/activate
uv pip install --upgrade pip
UV_HTTP_TIMEOUT=300 uv pip install --prerelease=allow "isaacsim[all,extscache]==6.0.0" --extra-index-url https://pypi.nvidia.com
UV_HTTP_TIMEOUT=300 uv pip install --no-deps -U torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip check
~~~

Strict baseline result:

~~~text
coverage==7.4.4
filelock==3.20.0
fsspec==2025.12.0
isaacsim==6.0.0.0
Jinja2==3.1.5
llvmlite==0.46.0
numpy==2.3.1
packaging==26.0
pillow==12.1.1
torch==2.10.0+cu128
torchvision==0.25.0+cu128
typing_extensions==4.12.2
warp-lang==1.15.0.dev20260525
No broken requirements found.
torch 2.10.0+cu128 12.8 True
~~~

Strict baseline smoke:

- `/mnt/storage/isaacsim-6.0-official/logs/isaacsim6-strict-replicator-smoke-2026-05-25.log`
- Result: `simulation_app_started`, `replicator_import_ok`, `simulation_app_update_ok`, and clean shutdown.
- This is the best current Isaac Sim 6.0-only base for OceanScale work because it preserves Isaac Sim exact pins and still uses the official cu128 torch wheel.

Module/API parity diagnostics:

- Log: `/mnt/storage/isaacsim-6.0-official/logs/isaacsim6-module-parity-2026-05-25.log`.
- Both the clean Sim-only venv and the main IsaacLab beta2 venv can import `omni.replicator.core`, `isaacsim.sensors.camera`, and `isaacsim.util` after `SimulationApp({"headless": True})` starts.
- Both venvs fail direct import of `omni.physics.tensors.api` and `omni.physx.tensors` as Python modules.
- The Isaac Sim 6.0 extension file that exists is `omni/physics/tensors/impl/api.py`, and its bundled docs use `.. automodule:: omni.physics.tensors.impl.api`.
- Local `release/3.0.0-beta2` Isaac Lab PhysX code imports `omni.physics.tensors.api`; the fetched `main` checkout changed the deformable object import back to `omni.physics.tensors.impl.api`.
- This makes the PhysX deformable tutorial failure a source/API mismatch in official beta2, not a missing extension-cache install.

## Isaac Lab beta2 side-by-side smoke patch probe

Purpose:

- Preserve the official `release/3.0.0-beta2` checkout and its as-is failure evidence.
- Create a separate compatibility-probe worktree to test whether known failures are shallow source/API mismatches or deeper Isaac Sim 6.0 runtime problems.

Worktree:

~~~bash
git -C /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2 \
  worktree add -b oceanscale-smoke-patches \
  /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2-oceanscale-smoke-patches \
  release/3.0.0-beta2
~~~

Patch artifact:

~~~text
/home/robot/workspace/46-marine/artifacts/isaacsim/isaaclab_beta2_oceanscale_smoke_patch.diff
~~~

Current artifact size after camera/Replicator compatibility probes: 488 lines, regenerated from the side-by-side worktree with:

~~~bash
git -C /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2-oceanscale-smoke-patches diff \
  > /home/robot/workspace/46-marine/artifacts/isaacsim/isaaclab_beta2_oceanscale_smoke_patch.diff
~~~

Patch summary:

- Move Isaac Lab PhysX imports from `omni.physics.tensors.api` to `omni.physics.tensors.impl.api`, matching the installed Isaac Sim 6.0 extension docs and fetched main deformable-object code.
- Set `create_cube_base_env.py` `replicate_physics=False`, matching its own comment and fetched main.
- Add `AdditiveUniformNoiseCfg = UniformNoiseCfg` compatibility alias for quadruped tutorial.
- Enable `isaacsim.util.debug_draw` before frame-transformer debug draw import.
- Convert quadcopter mass array to a torch tensor before summing.
- Add temporary Replicator/Warp compatibility shims before Replicator first import in camera examples and before enabling Replicator in USD color/texture randomization paths:
  - `wp.context = types.SimpleNamespace(Kernel=wp.Kernel)` when `wp.context` is missing.
  - `wp.types.array = wp.array` when `wp.types.array` is missing.
  - `wp.types.warp_type_to_np_dtype` and `wp.types.np_dtype_to_warp_type` reconstructed for scalar dtypes when missing.

Patched smoke command pattern:

~~~bash
PATCH=/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2-oceanscale-smoke-patches
PATCH_PYTHONPATH=$(find "$PATCH/source" -mindepth 1 -maxdepth 1 -type d | sort | paste -sd:)
export PYTHONPATH="$PATCH_PYTHONPATH:${PYTHONPATH:-}"
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py --lab-root "$PATCH" ...
~~~

Patched smoke results:

| Case | Official beta2 as-is | Smoke patch result | Evidence |
| --- | --- | --- | --- |
| tutorial_01_run_deformable_object_physx | failed on `omni.physics.tensors.api` | passed marker smoke | `official-next-smokes-2026-05-25-smoke-patches/tutorial_01_run_deformable_object_physx.log` reached setup/reset with no traceback |
| tutorial_03_create_quadruped_base_env | failed missing `AdditiveUniformNoiseCfg` | passed marker smoke | `official-next-smokes-2026-05-25-smoke-patches/tutorial_03_create_quadruped_base_env.log` reached reset |
| tutorial_04_run_frame_transformer | failed missing `isaacsim.util` | passed marker smoke after debug_draw enable | `official-next-smokes-2026-05-25-smoke-patches-r2/tutorial_04_run_frame_transformer.log` reached setup |
| demo_quadcopter | failed on mass array `.sum()` | passed marker smoke after tensor conversion | `official-next-smokes-2026-05-25-smoke-patches-r2/demo_quadcopter.log` reached setup and reset |
| tutorial_04_run_ray_caster_camera | failed missing `omni.replicator` in as-is batch3 | passed marker smoke | `official-next-smokes-2026-05-25-smoke-patches-r6/tutorial_04_run_ray_caster_camera.log` reached depth image shape with no traceback |
| tutorial_04_add_sensors_on_robot | failed on Replicator/Warp camera path | passed marker smoke after pre-AppLauncher Warp shim and dtype maps | `official-next-smokes-2026-05-25-smoke-patches-camera-r4/tutorial_04_add_sensors_on_robot.log` reached RGB image shape with no traceback |
| tutorial_04_run_usd_camera | failed on Replicator/Warp camera path | passed marker smoke after pre-AppLauncher Warp shim and dtype maps | `official-next-smokes-2026-05-25-smoke-patches-camera-r5/tutorial_04_run_usd_camera.log` reached RGB image shape with no traceback |
| tutorial_03_create_cube_base_env | failed replicate flag, then Replicator/Warp import order | functionally ran and printed steps after patch, but log still contains a nonfatal Replicator OmniGraph traceback | `official-next-smokes-2026-05-25-smoke-patches-r6/tutorial_03_create_cube_base_env.log` reached reset and Step 230; runner still marks failed because it saw `Traceback` from `/Replicator/SDGPipeline` graph creation |

Interpretation:

- Several Isaac Lab 3.0 beta2 failures are shallow and fixable with small compatibility patches.
- The camera tutorial failures were not missing camera/Replicator capability in Isaac Sim 6.0; they were caused by Isaac Lab beta2 plus the installed Warp dependency exposing newer Warp names than Replicator expected during import/data extraction.
- The patched cube tutorial is not clean enough to count as fully passed because Replicator still emits an OmniGraph traceback during startup, even though the environment runs.
- This supports using Isaac Sim 6.0 as the OceanScale base, but Isaac Lab 3.0 beta2 should be treated as beta-quality and wrapped with an OceanScale compatibility layer or tracked patch branch rather than treated as a fully clean upstream baseline.

## Isaac Sim standalone smoke runner

Purpose:

- Expand verification beyond Isaac Lab into official Isaac Sim `source/standalone_examples`.
- Keep the Isaac Sim standalone checks on the strict Isaac Sim 6.0 venv, separate from the Isaac Lab beta2 venv dependency conflicts.
- Preserve per-case logs and a JSON summary so future agents can reproduce exactly which official examples were run.

Runner:

~~~text
/home/robot/workspace/46-marine/artifacts/isaacsim/run_isaacsim_standalone_smokes.py
~~~

Official behavior used:

- Context7/official Isaac Sim docs confirm `SimulationApp` is the standalone entry point and must be initialized before Omniverse/USD module imports.
- Context7/official Isaac Lab docs confirm `isaaclab.sh -p` as the source-install verification path; this runner is intentionally separate because it targets Isaac Sim source examples directly.

Command:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r5
~~~

Environment set by runner:

~~~text
VIRTUAL_ENV=/mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict
PYTHONNOUSERSITE=1
OMNI_KIT_ACCEPT_EULA=YES
ACCEPT_EULA=Y
EXP_PATH=/mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict/lib/python3.12/site-packages/isaacsim/apps
~~~

Result:

~~~text
Summary: /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r5/summary.json
Overall: passed=true, 20/20 cases passed
~~~

Passed cases:

| Case | Official source entrypoint | Evidence |
| --- | --- | --- |
| asset_importer_mjcf_test | `source/standalone_examples/api/isaacsim.asset.importer.mjcf/mjcf_import.py --test` | marker `MJCF import successful.` |
| asset_importer_urdf_test | `source/standalone_examples/api/isaacsim.asset.importer.urdf/urdf_import.py --test` | marker `URDF import successful.` |
| asset_transformer_test | `source/standalone_examples/api/isaacsim.asset.transformer/run_asset_transformer.py --test` | marker `Asset transformation completed successfully.` |
| simulation_app_hello_world | `source/standalone_examples/api/isaacsim.simulation_app/hello_world.py --headless` | marker `Hello World!` |
| simulation_app_fetch_results | `source/standalone_examples/testing/isaacsim.simulation_app/test_fetch_results.py` | markers `Start test`, `Fetch results`, `Finish Test` |
| simulation_app_ogn | `source/standalone_examples/testing/isaacsim.simulation_app/test_ogn.py` | markers `Hello`, `Goodbye` |
| simulation_app_syntheticdata | `source/standalone_examples/testing/isaacsim.simulation_app/test_syntheticdata.py` | marker `3686400` |
| core_api_omnigraph_triggers | `source/standalone_examples/api/isaacsim.core.api/omnigraph_triggers.py` | markers `Starting just the app.`, `separate rendering and physics stepping.` |
| core_experimental_omnigraph_triggers | `source/standalone_examples/api/isaacsim.core.experimental.api/omnigraph_triggers.py --test` | markers `Starting just the app.`, `Separate rendering and physics stepping.` |
| replicator_sdg_getting_started_01 | `source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_01.py --headless` | markers `Output directory:`, `Step 2`; wrote 13 files including RGB PNG, bbox NPY, and JSON labels |
| replicator_sdg_getting_started_02 | `source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_02.py --headless` | markers `Output directory:`, `Step 2`, `[Annotator][Top][2]`; wrote 19 pose writer files |
| replicator_sdg_getting_started_03 | `source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_03.py --headless` | markers `Output directory:`, `Step 2` |
| replicator_sdg_getting_started_04 | `source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_04.py --headless` | markers `Output directory:`, `Step 0;` |
| replicator_sdg_getting_started_05 | `source/standalone_examples/api/isaacsim.replicator.examples/sdg_getting_started_05.py --headless` | markers `[SDG] Running with wait_for_render=True`, `[SDG] Avg randomization:` |
| replicator_multi_camera | `source/standalone_examples/api/isaacsim.replicator.examples/multi_camera.py --headless` | markers `Writing writer data to`, `Writing annotator data to`, `Step 4`; wrote 31 PNG/metadata files |
| replicator_custom_event_and_write | `source/standalone_examples/api/isaacsim.replicator.examples/custom_event_and_write.py --headless` | markers `Writing data to`, `Moving large cube position` |
| replicator_custom_fps_writer_annotator | `source/standalone_examples/api/isaacsim.replicator.examples/custom_fps_writer_annotator.py --headless` | markers `Writer data will be written to:`, `Capturing frame 5`; wrote 6 RGB frames plus metadata |
| replicator_simulation_get_data | `source/standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py --headless` | markers `Outputting data to`, `Cube_0 stopped moving`; wrote 31 files including Cube_0 through Cube_4 RGB/semantic outputs |
| sensors_physics_contact | `source/standalone_examples/testing/isaacsim.sensors.physics/contact_sensor_test.py` | marker `cube pose` |
| sensors_experimental_physics_contact | `source/standalone_examples/testing/isaacsim.sensors.experimental.physics/contact_sensor_test.py` | marker `cube pose` |

Notes:

- `hello_world.py` first timed out without explicit headless args because it defaults to `SimulationApp()` and GUI startup. Adding `--headless` let the official script run unchanged and pass.
- Several Replicator examples hard-code `SimulationApp(launch_config={"headless": False})`, but official Kit/SimulationApp args still allowed `--headless` to run them in this environment without editing NVIDIA source.
- Runner executes each case from an isolated `logs_dir/work/<case>` directory so SDG writers do not write generated PNG/JSON/NPY outputs into the official Isaac Sim source checkout.

## Isaac Sim asset, ROS2, and official-test follow-up

Runner updates:

- `run_isaacsim_standalone_smokes.py` now supports additional asset-heavy official cases and per-case environment variables.
- This was needed for the official ROS2 internal-library fallback path, where the first run printed the required environment variables.

### Asset-heavy standalone examples

Command:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r6-assets \
  --case core_api_simulation_callbacks \
  --case core_api_time_stepping \
  --case core_experimental_simulation_callbacks \
  --case simulation_app_load_stage_franka \
  --case simulation_app_async_call \
  --case testing_core_api_articulation \
  --case testing_core_api_time_stepping
~~~

Result:

| Case | Result | Evidence |
| --- | --- | --- |
| core_api_simulation_callbacks | pass | reached `step 59`, joint-position callback, and render callback |
| core_api_time_stepping | pass | reached `cleanup and exit` |
| core_experimental_simulation_callbacks | pass | reached `Step 0`, joint-position callback, and render callback |
| simulation_app_load_stage_franka | pass | loaded `/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd` and printed `Loading Complete` |
| simulation_app_async_call | pass | populated a Franka stage asynchronously and collected USD dependencies into the isolated work dir |
| testing_core_api_articulation | fail | official test traceback: `Prim path expression ['/World/microwave'] is invalid`; process exited 0 but runner correctly marked failed due traceback |
| testing_core_api_time_stepping | fail | official test printed `[fatal] PhysX status is not updated in the rendering call`; process exited 0 but runner correctly marked failed due fatal marker |

Additional asset/robot/camera checks:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r10-assets-kpi \
  --case benchmark_nucleus_kpis_json

python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r11-assets-robots-camera \
  --case core_experimental_control_frankas_test \
  --case manipulator_ur10_pick_up_test \
  --case sensor_camera_stereoscopic_depth_test
~~~

Result:

| Case | Result | Evidence |
| --- | --- | --- |
| benchmark_nucleus_kpis_json | pass | official benchmark completed; wrote `/tmp/metrics_benchmark_nucleus_kpis.json` with RTX 5090 and memory metrics |
| manipulator_ur10_pick_up_test | pass | official UR10 pick/place test-mode smoke exited cleanly |
| sensor_camera_stereoscopic_depth_test | pass | wrote `depth_sensor_distance.png` and `distance_to_image_plane.png`, both 1920x1080 grayscale PNGs, under the isolated work dir |
| core_experimental_control_frankas_test | fail | official script imports `carb` before `SimulationApp`, producing `ModuleNotFoundError: No module named 'carb'` in pip standalone mode |

Interpretation:

- Franka asset loading, USD stage loading, async stage collection, UR10 asset loading, and stereoscopic depth output are working in the strict Isaac Sim 6.0 venv.
- Some official `source/standalone_examples/testing` files and `control_frankas.py` are not clean under pip standalone execution because they violate or stress the official standalone import/runtime assumptions.

Additional core/sensor checks:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r14-core-sensors \
  --case core_api_add_frankas_test \
  --case core_api_visual_materials_test \
  --case core_api_rigid_contact_view_test \
  --case core_api_detailed_contact_data_test \
  --case core_experimental_add_cubes_test \
  --case core_experimental_visual_materials_test \
  --case sensor_camera_basic_test \
  --case sensor_rtx_lidar_basic_test \
  --case sensor_rtx_radar_basic_test \
  --case sensor_rtx_nonvisual_materials_test

python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r15-rtx-lidar-recheck \
  --case sensor_rtx_lidar_basic_test
~~~

Result:

| Case | Result | Evidence |
| --- | --- | --- |
| core_api_add_frankas_test | pass | printed reset marker and both Franka joint-position markers |
| core_api_visual_materials_test | pass | reached `Finished simulating for 10000 steps` |
| core_api_rigid_contact_view_test | pass | printed bottom-box and ground net-force markers |
| core_api_detailed_contact_data_test | pass | printed friction-force and contact-force markers |
| core_experimental_add_cubes_test | pass | printed angular velocity and world pose |
| core_experimental_visual_materials_test | fail | official script imports `carb` before `SimulationApp`, producing `ModuleNotFoundError: No module named 'carb'` |
| sensor_camera_basic_test | pass | wrote six PNG frames: `camera.frame099.png` through `camera.frame599.png` in the isolated work dir |
| sensor_rtx_lidar_basic_test | pass after warm cache | first r14 run exited 0 but runner lacked a positive marker and took 206.530s while loading remote warehouse assets; r15 recheck passed in 17.762s with exit 0 and no failure markers |
| sensor_rtx_radar_basic_test | timeout | killed after 266.004s; log reached Simulation App startup but not a completed radar smoke |
| sensor_rtx_nonvisual_materials_test | pass | created visual/non-visual material cubes and RTX Lidar |

Interpretation:

- Core API articulation construction, visual materials, contact views, camera output, RTX Lidar, and RTX nonvisual-material setup are working in the strict Isaac Sim 6.0 venv.
- RTX Radar remains unvalidated locally; current evidence is a startup timeout, not a functional failure with a Python traceback.
- The first RTX Lidar run paid remote asset/cache warmup cost; the immediate recheck shows the official example itself can pass quickly once assets are available.

Additional core/robot/motion-generation checks:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r16-core-robot-motion \
  --case core_api_add_cubes \
  --case core_api_control_robot \
  --case core_api_data_logging \
  --case core_api_simulate_robot \
  --case core_api_cloth_test \
  --case core_experimental_control_robot_numpy_cpu \
  --case core_experimental_control_robot_torch_cpu \
  --case core_experimental_control_robot_warp_cpu \
  --case core_experimental_deformable_stress_visualization_test \
  --case core_cloner_clone_ants \
  --case wheeled_jetbot_differential_move_test

python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r17-motion-generation \
  --case motion_generation_trajectory_min_time \
  --case motion_generation_trajectory_linear \
  --case motion_generation_mobile_robot_control \
  --case motion_generation_scene_interaction
~~~

Result:

| Case | Result | Evidence |
| --- | --- | --- |
| core_api_add_cubes | pass | printed dynamic-cube pose/velocity arrays; 42.218s |
| core_api_control_robot | pass | reached Franka joint target marker Reached:; 19.280s |
| core_api_data_logging | pass | saved/loaded data logger frame with joint_positions and applied_joint_positions; 10.207s |
| core_api_simulate_robot | pass | loaded Franka asset and printed Finished simulating for 1000 steps; 19.987s |
| core_api_cloth_test | pass | printed cloth 0 average height; 8.894s |
| core_experimental_control_robot_numpy_cpu | pass | exit 0 with no failure markers; 24.995s |
| core_experimental_control_robot_torch_cpu | pass | exit 0 with no failure markers; 25.126s |
| core_experimental_control_robot_warp_cpu | pass | exit 0 with no failure markers; 36.547s |
| core_experimental_deformable_stress_visualization_test | pass | printed von Mises stress range:; 7.675s |
| core_cloner_clone_ants | pass | cloned Ant articulations and exited 0 with no failure markers; 23.116s |
| wheeled_jetbot_differential_move_test | pass | official Jetbot differential drive --test path exited 0; 8.793s |
| motion_generation_trajectory_min_time | pass | printed trajectory header, Trajectory following complete!, and Example Complete; 15.951s |
| motion_generation_trajectory_linear | pass | printed Trajectory Following Example: LinearTrajectory, completion marker, and Example Complete; 14.184s |
| motion_generation_mobile_robot_control | pass | printed differential-drive controller header and Example Complete; 16.715s |
| motion_generation_scene_interaction | pass | printed scene-interaction header and Complete workflow demonstrated successfully!; 13.675s |

Interpretation:

- The strict Isaac Sim 6.0 venv now has clean local evidence for additional core API, data logger, cloth, cloner, wheeled robot, experimental control, Warp-backed control, deformable stress, and motion-generation examples.
- Motion-generation scripts use the official --test --no-window pattern rather than --headless; forcing --headless would be rejected by some script arg parsers.
- These are still bounded smokes, not exhaustive correctness benchmarks.

### Official suite coverage index

Coverage builder:

~~~bash
python3 artifacts/isaacsim/build_official_suite_coverage.py
~~~

Generated files:

- /home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_coverage.json
- /home/robot/workspace/46-marine/artifacts/isaacsim/official_suite_coverage.md

Current clean-source coverage after r18/r19/r20:

| Scope | Passed | Failed | Uncovered |
| --- | ---: | ---: | ---: |
| all inventoried Python entrypoints | 87 | 18 | 1994 |
| Isaac Sim source/standalone_examples | 70 | 7 | 130 |
| IsaacLab scripts/source | 17 | 11 | 1864 |

Interpretation:

- This is an index of evidence, not a completion claim.
- It maps smoke summary commands back to official source paths, so future work can see which official entrypoints have clean pass/fail evidence.
- Patched IsaacLab smoke successes are tracked separately from clean beta2 results.

### Additional camera and sensor checks

Camera batch:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r18-camera-sensors \
  --case sensor_camera_add_depth_sensor_usd \
  --case sensor_camera_annotator_device \
  --case sensor_camera_opencv_fisheye \
  --case sensor_camera_opencv_pinhole \
  --case sensor_camera_pre_isp_pipeline \
  --case sensor_camera_ros_projection \
  --case sensor_camera_view
~~~

Camera result: 7/7 passed.

| Case | Result | Evidence |
| --- | --- | --- |
| sensor_camera_add_depth_sensor_usd | pass | wrote example_camera_with_depth_sensor.usd, 4458 bytes |
| sensor_camera_annotator_device | pass | rgba and pointcloud tests printed PASS with no FAIL markers |
| sensor_camera_opencv_fisheye | pass | wrote camera_opencv_fisheye.png, 550945 bytes, and camera_opencv_fisheye.usd |
| sensor_camera_opencv_pinhole | pass | wrote camera_opencv_pinhole.png, 902773 bytes, and camera_opencv_pinhole.usd |
| sensor_camera_pre_isp_pipeline | pass | wrote HDR, raw sensor, and ISP binary outputs: 2457600, 614400, and 921600 bytes |
| sensor_camera_ros_projection | pass | printed Header Frame ID and wrote camera_ros.png plus camera_ros.usd |
| sensor_camera_view | pass | printed out_dir, rgb_tiled_np.shape, and depth_tiled_np.shape |

RTX and physics sensor batch:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r19-rtx-physics-sensors \
  --case sensor_rtx_lidar_config_variants_test \
  --case sensor_rtx_lidar_gmo_inspect_test \
  --case sensor_rtx_radar_gmo_inspect_test \
  --case sensor_rtx_lidar_object_ids_test \
  --case sensor_rtx_lidar_robot_integration_test \
  --case sensor_physics_contact_api \
  --case sensor_physics_effort_api \
  --case sensor_physics_imu_api \
  --case sensor_experimental_physics_contact_api \
  --case sensor_experimental_physics_effort_api \
  --case sensor_experimental_physics_imu_api \
  --case sensor_physx_rotating_lidar_test

python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r20-gmo-recheck \
  --case sensor_rtx_lidar_gmo_inspect_test \
  --case sensor_rtx_radar_gmo_inspect_test
~~~

RTX/physics result:

| Case | Result | Evidence |
| --- | --- | --- |
| sensor_rtx_lidar_config_variants_test | pass | printed available Lidar configurations |
| sensor_rtx_lidar_gmo_inspect_test | pass after marker recheck | r19 printed FULL auxiliary fields and 153600 points but marker was too strict; r20 passed with Auxiliary level: FULL and Sample Point Cloud Data |
| sensor_rtx_radar_gmo_inspect_test | pass after marker recheck | r19 printed Radar GMO fields and sample points but marker was too strict; r20 passed with Radar GMO Data and Sample Point Cloud Data |
| sensor_rtx_lidar_object_ids_test | pass | decoded StableIdMap and printed Object ID to Prim Path Mapping |
| sensor_rtx_lidar_robot_integration_test | pass | Carter + RTX Lidar integration exited cleanly after 95.123s |
| sensor_physics_contact_api | pass | marker-stopped after physics_step contact frame |
| sensor_physics_effort_api | pass | marker-stopped after Sensor Time reading |
| sensor_physics_imu_api | pass | marker-stopped after lin_acc IMU frame |
| sensor_experimental_physics_contact_api | fail | AttributeError: ContactSensor object has no attribute name |
| sensor_experimental_physics_effort_api | pass | marker-stopped after Sensor Time reading |
| sensor_experimental_physics_imu_api | fail | TypeError: IMUSensor.__init__ got unexpected keyword argument frequency |
| sensor_physx_rotating_lidar_test | pass | official test path exited 0 with no failure markers |

Interpretation:

- Camera calibration/projection, annotator-device, pre-ISP, CameraView, RTX Lidar/Radar GMO, StableIdMap object IDs, Carter+Lidar integration, classic physics contact/effort/IMU, experimental effort, and PhysX rotating lidar are now locally validated in the strict Isaac Sim 6.0 venv.
- The earlier RTX Radar basic timeout does not mean the Radar data path is broadly broken; r20 shows Radar GMO output works in the official inspect script.
- Experimental physics contact and IMU examples expose source/API drift in the current official source + pip runtime pair.

### ROS2 bridge minimum smoke

Current host state:

~~~text
which ros2: not found
/opt/ros: no local ROS distro found
colcon: not found
~~~

Official source behavior inspected:

- `isaacsim.ros2.core` defaults `ros_distro` to `system_default`.
- On Ubuntu 24.x, the source maps system default to `jazzy`.
- If `ROS_DISTRO` is absent, the extension attempts internal-library fallback.
- The bridge extension metadata says users may either source system ROS2 libraries or source the lightweight ROS2 libraries included with Isaac Sim.

First probe:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r7-ros2 \
  --case ros2_bridge_enable_extension_internal_fallback
~~~

Result: failed. The log selected internal `jazzy`, then printed:

~~~text
Could not load .../isaacsim.ros2.core/jazzy/lib/librmw_implementation.so.
Error: libament_index_cpp.so: cannot open shared object file: No such file or directory
~~~

The same official error message gave the required env:

~~~text
ROS_DISTRO=jazzy
RMW_IMPLEMENTATION=rmw_fastrtps_cpp
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/storage/isaacsim-6.0-official/venv-isaacsim-strict/lib/python3.12/site-packages/isaacsim/exts/isaacsim.ros2.core/jazzy/lib
~~~

Rerun with per-case env:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r9-ros2-env-resolved \
  --case ros2_bridge_enable_extension_internal_fallback
~~~

Result: passed. Summary:

~~~text
Summary: /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r9-ros2-env-resolved/summary.json
case: ros2_bridge_enable_extension_internal_fallback
marker: isaacsim.ros2.bridge-5.0.0] startup
failures: no Traceback, no ROS2 Bridge startup failed
~~~

Interpretation:

- Isaac Sim 6.0 ROS2 bridge extension can start locally using the bundled internal Jazzy libraries when the official env is set.
- At this point only bridge startup was validated; clock and camera timestamp checks are recorded in the next section.
- This is not a full external ROS2 validation: no system `ros2` CLI, external ROS graph, external topics/services, or ROS workspace were validated yet.

Additional internal ROS2 bridge checks:

~~~bash
python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r12-ros2-clock-camera \
  --case ros2_bridge_clock_internal \
  --case ros2_bridge_camera_tf_delay_internal

python3 artifacts/isaacsim/run_isaacsim_standalone_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaacsim-standalone-smokes-2026-05-25-r13-ros2-camera-tf \
  --case ros2_bridge_camera_tf_delay_internal
~~~

Result:

| Case | Result | Evidence |
| --- | --- | --- |
| ros2_bridge_clock_internal | pass | printed `sim time:` and `manual stepped sim time:`; exit 0 |
| ros2_bridge_camera_tf_delay_internal | pass on recheck | official test printed 5/5 valid pairs, zero missed steps, 0.000 ms average/median/std/min/max delay, and `Status:         PASS` |

Notes:

- The first camera timestamp run exited 0 and printed PASS, but the runner marked it failed because bundled `rosidl_generator_py` emitted a nonfatal debug traceback for missing `lark`.
- The runner failure markers were narrowed for this ROS2 case, and the r13 recheck passed while still detecting fatal/error/runtime failure markers.
- The per-case env remains the official internal Jazzy path: `ROS_DISTRO=jazzy`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, and `LD_LIBRARY_PATH=.../isaacsim.ros2.core/jazzy/lib`.

### Source-build audit

Official source README requires:

~~~bash
git lfs install
git lfs pull
gcc --version
g++ --version
./build.sh
~~~

Current local status:

~~~text
IsaacSim source commit: f8c8f900ff0ae2bf8bf8e2dd922cea9ff70a99bc
_build directory: missing
git rev-parse --is-shallow-repository: true
gcc: 11.4.0
g++: 13.3.0
g++-11: 11.4.0 available
git lfs ls-files: present, but full source-build LFS pull was not intentionally run
./build.sh --help: exits 1 before help because EULA has not been interactively accepted
env ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ./build.sh --help: same exit 1
~~~

Observed source-build EULA behavior:

~~~text
Do you accept the governing terms? (yes/No):
Error: NVIDIA Software License Agreement and Product-Specific Terms for NVIDIA Omniverse must be accepted to proceed.
~~~

`tools/eula_check.sh` behavior from the official source checkout:

- It exits 0 only if `.eula_accepted` exists in the current directory or repo root.
- Otherwise it prompts and accepts only interactive `yes` or `y`.
- It creates the repo-root `.eula_accepted` file after interactive acceptance.
- It does not check `ACCEPT_EULA` or `OMNI_KIT_ACCEPT_EULA`.

Interpretation:

- Source code is available locally, but a real Isaac Sim source build has not been performed.
- Do not run `./build.sh` until compiler selection is made explicit, likely `CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11`, a deliberate `git lfs pull` is accepted because it can fetch large source assets, and the source-build EULA acceptance file is intentionally created by the user/operator.

## IsaacLab 3.x additional official tutorial/demo sweep

Runner:

~~~text
/home/robot/workspace/46-marine/artifacts/isaacsim/run_isaaclab_official_next_smokes.py
~~~

Clean beta2 source:

~~~text
/mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2
commit 9fe080c1a1c73e8f8a7a8f971f98517876a99861
~~~

Commands:

~~~bash
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaaclab-official-next-smokes-2026-05-25-r4-controllers-demos \
  --case tutorial_05_run_diff_ik \
  --case tutorial_05_run_osc \
  --case demo_sensor_contact \
  --case demo_sensor_raycaster \
  --case demo_arms \
  --case demo_quadrupeds

python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaaclab-official-next-smokes-2026-05-25-r5-demos-viz-none \
  --case demo_sensor_raycaster \
  --case demo_arms \
  --case demo_quadrupeds

python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaaclab-official-next-smokes-2026-05-25-r6-raycaster-viz-none \
  --case demo_sensor_raycaster
~~~

Clean beta2 result:

| Case | Result | Evidence |
| --- | --- | --- |
| tutorial_05_run_diff_ik | pass | reached setup marker and was marker-stopped |
| tutorial_05_run_osc | fail | `omni.physics.tensors.api` missing via `isaaclab_physx.sensors.contact_sensor` |
| demo_sensor_contact | fail | same `omni.physics.tensors.api` missing path |
| demo_sensor_raycaster | fail, then pass after `--viz none` | original run failed because demos default `visualizer=["kit"]`; rerun with `--viz none` reached setup/reset/ray-hit markers |
| demo_arms | fail, then pass after `--viz none` | original run failed on explicit `kit` visualizer; rerun reached setup/reset markers |
| demo_quadrupeds | fail, then pass after `--viz none` | original run failed on explicit `kit` visualizer; rerun reached setup/reset markers |

Patch-worktree validation for the PhysX tensor import issue:

~~~bash
python3 artifacts/isaacsim/run_isaaclab_official_next_smokes.py \
  --lab-root /mnt/storage/isaacsim-6.0-official/sources/IsaacLab-release-3.0.0-beta2-oceanscale-smoke-patches \
  --logs-dir /mnt/storage/isaacsim-6.0-official/logs/isaaclab-patched-next-smokes-2026-05-25-r4-controllers-demos \
  --case tutorial_05_run_osc \
  --case demo_sensor_contact
~~~

Patched result:

| Case | Clean beta2 result | Patch-worktree result |
| --- | --- | --- |
| tutorial_05_run_osc | failed on `omni.physics.tensors.api` | passed setup marker |
| demo_sensor_contact | failed on `omni.physics.tensors.api` | passed setup/reset/contact-force markers |

Interpretation:

- Clean IsaacLab 3.0 beta2 is usable for a larger set of official demos than initially proven, but repeated failures show a consistent PhysX tensor import drift against Isaac Sim 6.0.
- The existing OceanScale smoke patch branch now covers the same import drift in the OSC/contact-sensor paths, not only deformables and earlier sensor tutorials.
- Demo scripts that default to `visualizer=["kit"]` need explicit `--viz none` for headless smoke.

## Current remaining validation scope

Not complete yet:

- Full Isaac Sim standalone example sweep under source/standalone_examples.
- Full Isaac Lab tutorial/demo sweep under scripts/tutorials and scripts/demos.
- Full external ROS 2 bridge demos; bundled internal Jazzy bridge startup, clock, and camera TF timestamp checks have passed, but no system ROS2 CLI or external ROS graph was validated.
- Replicator / SDG demos that may need assets, output directories, or long runtime.
- Asset/Nucleus-heavy demos that depend on remote Omniverse assets or LFS blobs not yet pulled; several Franka/UR10/camera asset paths have passed, but this is not the full asset catalog.
- Isaac Sim source build from IsaacSim-develop; blocked pending explicit g++-11 setup, intentional git lfs pull, and deliberate source EULA acceptance.
- Isaac Lab optional mimic install; currently blocked by egl-probe / CMake compatibility failure during default ./isaaclab.sh -i.
