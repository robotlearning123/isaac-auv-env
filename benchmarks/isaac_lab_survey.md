# Isaac Lab Ocean/Fluid Capability Survey

**Date**: 2026-05-20
**Source**: `/home/robot/workspace/08-isaaclab/isaacsim/IsaacLab/` (Isaac Lab v2.3.0)
**Also surveyed**: `/home/robot/workspace/08-isaaclab/isaacsim/OceanSim/` (separate extension, BSD-3)

---

## 1. Structure Overview

Isaac Lab v2.3.0 is organized as a monorepo under `source/`:

| Module | Path | Purpose |
|--------|------|---------|
| `isaaclab` | `source/isaaclab/isaaclab/` | Core framework: sim, sensors, envs, managers, assets, terrains |
| `isaaclab_tasks` | `source/isaaclab_tasks/isaaclab_tasks/` | Pre-built task environments |
| `isaaclab_rl` | `source/isaaclab_rl/isaaclab_rl/` | RL training wrappers (RSL-RL, RL Games) |
| `isaaclab_mimic` | `source/isaaclab_mimic/` | Imitation learning dataset generation |
| `isaaclab_assets` | `source/isaaclab_assets/` | Pre-configured robot/sensor USD assets |

Key submodules in `isaaclab/isaaclab/`:
- `sim/` — Simulation context, spawners, schemas, material configs
- `sensors/` — Camera, IMU, ContactSensor, RayCaster, FrameTransformer
- `envs/` — ManagerBasedRLEnv, DirectRLEnv, DirectMARLEnv, MDP utilities
- `assets/` — Rigid body, articulation, deformable body asset wrappers
- `terrains/` — Height-field terrain generation
- `managers/` — Action, observation, reward, event, termination managers
- `controllers/` — Differential IK, operational space, joint position controllers

**Source files cited**: `source/isaaclab/isaaclab/sim/simulation_cfg.py`, `source/isaaclab/isaaclab/sensors/__init__.py`, `source/isaaclab/isaaclab/envs/__init__.py`

---

## 2. Physics Backends

### PhysX 5 (Primary and Only Backend)

Isaac Lab uses **PhysX 5 exclusively** as its physics backend. There is NO support for MuJoCo or Warp as alternative physics solvers.

- **Evidence**: `source/isaaclab/isaaclab/sim/simulation_cfg.py:21-393` — `PhysxCfg` class configures all solver parameters
- `source/isaaclab/isaaclab/sim/simulation_context.py:26` — `import omni.physx`
- GPU-accelerated physics via PhysX 5 (configurable device: `SimulationCfg.device`)
- CCD, enhanced determinism, articulation contact ordering, GPU collision stack

### No MuJoCo or Warp Physics

- Grep for `mujoco`, `MuJoCo` across source: **0 results** (outside of water-tight mesh comments)
- Warp is used only for **ray-casting** (sensor), not for physics simulation
- **Evidence**: `source/isaaclab/isaaclab/sensors/ray_caster/ray_caster.py:16` — `import warp as wp` for mesh raycast only

### Newton Integration

- Grep for `newton` or `Newton` across isaaclab source: **0 relevant results**
- No Newton physics engine integration exists in Isaac Lab v2.3.0

### Deformable Bodies

- PhysX supports FEM-based deformable bodies
- **Evidence**: `source/isaaclab/isaaclab/sim/schemas/schemas_cfg.py:302-338` — `DeformableBodyPropertiesCfg` with self-collision, damping, sleep thresholds
- No fluid or particle system support

---

## 3. Fluid/Water/Ocean Tasks

### Isaac Lab Itself: NONE

Comprehensive grep across the entire `source/` directory for: `fluid`, `water`, `ocean`, `underwater`, `buoyancy`, `hydro`, `wave`, `drag`, `marine`:

- **No fluid simulation tasks exist**
- **No water/ocean environments exist**
- **No buoyancy models exist**
- All "water" hits are false positives:
  - `hf_terrains_cfg.py` — terrain sub-types (not water)
  - `automate_algo_utils.py:296` — "watertight mesh" comment
  - `terminations.py:38` — "fluid in nature" (referring to episode length)
  - `physics_materials_cfg.py` — material physics (not fluid)
  - `schemas_cfg.py:363` — "approximate the effect of air drag" on deformable bodies (not hydrodynamic)

### Available Task Categories

All tasks are land/air-based:
- `direct/`: ant, anymal_c, cartpole, franka_cabinet, humanoid, quadcopter, shadow_hand, allegro_hand, factory, forge, locomotion
- `manager_based/`: classic, locomotion, manipulation, navigation, loco_manipulation, locomanipulation

**Source**: `source/isaaclab_tasks/isaaclab_tasks/direct/` and `manager_based/`

### OceanSim Extension (Separate Project)

A third-party extension **OceanSim** exists at `/home/robot/workspace/08-isaaclab/isaacsim/OceanSim/`:
- **Repo**: `https://github.com/umfieldrobotics/OceanSim.git` (BSD-3 license)
- **Not part of Isaac Lab** — separate extension loaded into Isaac Sim
- Provides underwater-specific sensors (see Section 4 below)
- Does NOT provide fluid dynamics — only rendering and sensor simulation

---

## 4. Sensors

### Isaac Lab Built-in Sensors

Source: `source/isaaclab/isaaclab/sensors/__init__.py`

| Sensor | File | Description |
|--------|------|-------------|
| Camera | `sensors/camera/` | USD-prim based, wraps `UsdGeom Camera`. RGB, depth, semantic/instance segmentation, normals |
| TiledCamera | `sensors/camera/tiled_camera.py` | Tiled-rendering API variant for higher throughput |
| ContactSensor | `sensors/contact_sensor/` | Contact force reporting on physics bodies |
| FrameTransformer | `sensors/frame_transformer/` | Coordinate frame transforms between bodies |
| IMU | `sensors/imu/` | Accelerometer + gyroscope (linear acceleration, angular velocity) |
| RayCaster | `sensors/ray_caster/` | **Warp-based** custom ray-caster, supports PinholeCamera and Lidar patterns |
| RayCasterCamera | `sensors/ray_caster/ray_caster_camera.py` | Ray-caster with camera-like output |

### Camera Output Types

`source/isaaclab/isaaclab/sensors/camera/camera_cfg.py:64`:
```python
data_types: list[str] = ["rgb"]
```
Supported types (from Omniverse Replicator): `rgb`, `depth`, `pointcloud`, `semantic_segmentation`, `instance_segmentation`, `instance_id_segmentation`, `normals`, `motion_vectors`, `distance_to_image_plane`

### LiDAR Pattern

`source/isaaclab/isaaclab/sensors/ray_caster/patterns/patterns_cfg.py:203`:
- `LidarPatternCfg` — configurable azimuth/vertical FOV, channels, for ray-cast based LiDAR
- Note: this is a Warp ray-caster, NOT the RTX LiDAR extension

### NO Underwater-Specific Sensors in Isaac Lab

No sonar, DVL, depth pressure, barometer, or underwater camera distortion models.

### OceanSim Underwater Sensors (Separate Extension)

Source: `isaacsim/OceanSim/isaacsim/oceansim/sensors/`

| Sensor | File | Lines | Description |
|--------|------|-------|-------------|
| UW_Camera | `UW_Camera.py` | 210 | Underwater camera with Warp-based attenuation/backscatter rendering |
| DVLsensor | `DVLsensor.py` | 426 | Doppler Velocity Logger — 4-beam Janus config, frequency-dependent range, noise model |
| ImagingSonarSensor | `ImagingSonarSensor.py` | 632 | Imaging sonar (Oculus M370s params), Warp kernels for sonar image generation |
| BarometerSensor | `BarometerSensor.py` | 108 | Depth pressure sensor, hydrostatic pressure model |

**Key OceanSim technical details**:
- `UWrenderer_utils.py` — Warp kernel `UW_render()` applies Beer-Lambert attenuation + backscatter: `UW_RGB = raw * exp(-depth * atten) + backscatter * (1 - exp(-depth * backscatter_coeff))`
- `ImagingSonar_kernels.py` — Full Warp GPU pipeline: cartesian-to-spherical transform, intensity computation with reflectivity/attenuation, binning/averaging
- DVL uses `isaacsim.sensors.physx._range_sensor` for beam raycasting with MultivariateNormal noise
- Barometer uses hydrostatic equation: `P = P_atm + rho * g * depth`

---

## 5. Rendering

### Omniverse RTX Renderer

Source: `source/isaaclab/isaaclab/sim/simulation_cfg.py:185-300`

Configuration via `RenderCfg` class:

| Feature | RTX Variable | Notes |
|---------|-------------|-------|
| Translucency | `/rtx/translucency/enabled` | Glass/specular transmissive surfaces |
| Reflections | `/rtx/reflections/enabled` | |
| Global Illumination | `/rtx/indirectDiffuse/enabled` | |
| DLSS / DLAA / TAA / FXAA | `/rtx/post/dlss/execMode` | |
| DLSS Frame Gen | `/rtx-transient/dlssg/enabled` | Ada Lovelace+ required |
| DL Denoiser | `/rtx-transient/dldenoiser/enabled` | |
| Direct Lighting | `/rtx/directLighting/enabled` | |
| Shadows | `/rtx/shadows/enabled` | |
| Ambient Occlusion | `/rtx/ambientOcclusion/enabled` | |
| Dome Light Strategy | dome_light_upper_lower_strategy | IBL, Environment, Uniform |

### No Built-in Caustics or Volume Rendering

- No caustic, subsurface scattering, or ocean wave rendering features in Isaac Lab config
- No Hydra-based rendering pipeline switching
- Rendering is RTX path tracing / rasterization only

### OceanSim UW Rendering

OceanSim adds underwater visual effects via Warp post-processing kernel — NOT native RTX integration. Applied after standard rendering.

---

## 6. USD Scene Support

### Loading USD Assets

Source: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py`

- `spawn_from_usd(prim_path, cfg)` — Load any USD file as a simulation asset
- `spawn_from_urdf(prim_path, cfg)` — Convert URDF to USD then spawn
- `spawn_ground_plane(prim_path, cfg)` — Load Isaac Sim grid plane USD
- USD variant selection supported via `select_usd_variants`

### Custom Environments

Any USD scene can be loaded as the simulation environment. Isaac Lab uses USD as its native scene format. Custom environments are loaded by:
1. Placing USD files in a discoverable path
2. Using `spawn_from_usd()` in the scene configuration
3. Setting up physics colliders and materials on the USD prims

**Source**: `source/isaaclab/isaaclab/sim/spawners/from_files/from_files.py:38-76`

### Asset Registry

`source/isaaclab_assets/` contains pre-configured robot USD assets:
- Legged: ANYmal, Unitree (Go1/G2/H1/H1_2), Cassie, Fourier
- Manipulators: Franka, Kinova, Kuka+Allegro, Sawyer, UR, Shadow Hand, Galbot
- Mobile: Spot, Ridgeback+Franka
- Aerial: Quadcopter
- Humanoid: various
- **No underwater robots** (no AUV, ROV, or marine vehicles)

---

## 7. RL Integration

### Supported RL Frameworks

Source: `source/isaaclab_rl/isaaclab_rl/`

| Framework | Path | Notes |
|-----------|------|-------|
| RSL-RL | `rsl_rl/` | Primary framework. PPO-based, includes distillation, symmetry, RND configs |
| RL Games | `rl_games/` | PBT (Population-Based Training) variant available |

### Environment Interfaces

Source: `source/isaaclab/isaaclab/envs/__init__.py`

| Env Class | Pattern | Use Case |
|-----------|---------|----------|
| `ManagerBasedRLEnv` | Manager-based | Standard single-agent RL (config-driven) |
| `DirectRLEnv` | Direct | Custom single-agent RL (code-driven) |
| `DirectMARLEnv` | Direct | Multi-agent RL |
| `ManagerBasedRLMimicEnv` | Manager-based | Imitation learning |

### Training Scripts

- `scripts/reinforcement_learning/` — Training launchers for RSL-RL and RL Games
- `scripts/imitation_learning/` — Imitation learning dataset generation
- `scripts/sim2sim_transfer/` — Sim-to-sim transfer workflows

### Not Supported

- No Stable-Baselines3, CleanRL, or other third-party RL libs bundled
- No domain randomization for underwater-specific parameters (visibility, current, salinity)

---

## 8. Gap Analysis for Underwater Robotics Simulation

### CRITICAL GAPS (No capability exists)

| Gap | Details | Build Effort |
|-----|---------|-------------|
| **Fluid dynamics solver** | No SPH, Eulerian, or hybrid fluid simulation. PhysX only handles rigid/articulated/deformable bodies. No water volume simulation at all. | Very High — requires external GPU fluid solver (Warp/Newton) |
| **Buoyancy model** | No Archimedes principle, no variable buoyancy, no ballast simulation. Only linear/angular damping on rigid bodies exists. | Medium — custom MDP action/observation term |
| **Hydrodynamic forces** | No drag/lift coefficients, no added mass, no current forces, no wave forces. Only generic `linear_damping` / `angular_damping` in schemas. | Medium — custom force computation in env step |
| **Underwater sensor models** | No sonar, DVL, depth sensor, USBL, or underwater camera distortion in Isaac Lab core. OceanSim provides these but is a separate extension. | Low-Medium — integrate OceanSim sensors |
| **Underwater visual rendering** | No caustics, no volumetric light scattering, no water surface, no fog/visibility attenuation in Isaac Lab. OceanSim adds post-process only. | High — requires RTX material/shader work |
| **Underwater robot assets** | No AUV, ROV, or marine vehicle USD models in asset library. | Medium — create/import URDF models |
| **Water current / wave field** | No spatially-varying current field, no wave spectrum (JONSWAP, Pierson-Moskowitz), no tidal model. | High — custom GPU field solver |
| **Newton integration** | Isaac Lab has no Newton physics engine support. OceanScale's planned Newton backend is entirely separate. | High — requires Isaac Sim extension bridge |

### MODERATE GAPS (Partial capability, needs extension)

| Gap | What Exists | What's Missing |
|-----|-------------|----------------|
| **IMU** | Built-in IMU sensor | No underwater-specific bias/noise models |
| **Depth camera** | Camera supports `depth` data type | No underwater range limitation, no scattering |
| **Contact sensing** | ContactSensor on bodies | No hydrodynamic pressure distribution |
| **USD environment loading** | Can load any USD scene | No pre-built underwater scene assets |
| **Domain randomization** | MDP event system supports randomization | No water-parameter DR (visibility, current speed, salinity) |

### AVAILABLE (No gap)

| Capability | Status |
|------------|--------|
| GPU-accelerated rigid body physics (PhysX 5) | Available |
| Articulated robot simulation | Available |
| RL training pipeline (PPO via RSL-RL) | Available |
| Multi-agent RL | Available |
| Camera (RGB, depth, segmentation, normals) | Available |
| Ray-cast LiDAR pattern (Warp-based) | Available |
| IMU (accel + gyro) | Available |
| Contact force sensor | Available |
| USD scene loading | Available |
| Terrain generation | Available (land only) |
| Headless rendering | Available |
| Sim-to-sim transfer | Available |
| Imitation learning | Available |

---

## 9. OceanSim Integration Assessment

**OceanSim** (`https://github.com/umfieldrobotics/OceanSim`, BSD-3) is the most relevant existing work for underwater simulation in the Isaac Sim ecosystem.

### What OceanSim Provides

1. **Underwater camera** with Beer-Lambert attenuation/backscatter (Warp post-process)
2. **Imaging sonar** with full GPU-accelerated pipeline (Warp kernels for intensity, binning)
3. **DVL sensor** with realistic Janus beam geometry, frequency-dependent range, noise
4. **Barometer/depth sensor** with hydrostatic pressure model

### What OceanSim Does NOT Provide

1. No fluid dynamics (no SPH, no Euler, no particle methods)
2. No buoyancy simulation
3. No hydrodynamic force computation (drag, added mass, lift)
4. No water current or wave field generation
5. No caustic or volumetric rendering (post-process only)
6. No RL training integration with Isaac Lab (uses Isaac Sim standalone API)
7. No terrain/sea-floor generation
8. No underwater robot assets

### Compatibility with Isaac Lab

- OceanSim uses `isaacsim.core.api` and `isaacsim.sensors.camera` (standalone Isaac Sim API)
- It does NOT use Isaac Lab's `ManagerBasedRLEnv` or `DirectRLEnv` interfaces
- Integrating OceanSim sensors into Isaac Lab would require wrapping them as `SensorBase` subclasses
- Warp usage is compatible with Isaac Lab's existing Warp ray-caster infrastructure

---

## 10. Recommendations for OceanScale

1. **Use Isaac Lab for RL infrastructure** — ManagerBasedRLEnv, RSL-RL, observation/reward/action managers are solid. Build underwater tasks as Isaac Lab environments.

2. **Fluid simulation: external Warp kernels** — Isaac Lab has no fluid solver. OceanScale's Warp/Newton SPH approach is correct; run it alongside PhysX and apply forces to rigid bodies via the MDP system.

3. **Integrate OceanSim sensors** — Wrap OceanSim's UW_Camera, DVL, ImagingSonar, Barometer as Isaac Lab `SensorBase` subclasses. The Warp kernels are reusable.

4. **Hydrodynamic forces as MDP terms** — Implement drag, buoyancy, added mass as custom action/reward/event terms in the Isaac Lab MDP framework. Use `linear_damping`/`angular_damping` as a rough approximation for prototyping.

5. **Underwater rendering: post-process approach** — Follow OceanSim's pattern: render with standard RTX, then apply Warp attenuation/backscatter kernel. This is faster than custom shaders.

6. **No Newton-Isaac Lab bridge needed** — Since Newton isn't integrated into Isaac Lab, run Newton/Warp fluid kernels independently and couple via force application to PhysX bodies.
