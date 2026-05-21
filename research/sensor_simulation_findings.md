# Underwater Sensor Simulation & Perception: Deep Research Findings

**Date:** 2026-05-20
**Scope:** OceanScale GPU-native underwater simulator — sensor/perception architecture survey
**Method:** Primary source papers read in full (arXiv HTML), supplemented by web search for surrounding context

---

## Table of Contents

1. [OceanSim Sensor Architecture](#1-oceansim-sensor-architecture)
2. [HoloOcean 2.x Sensor Suite](#2-holoocean-2x-sensor-suite)
3. [Sonar Simulation Approaches](#3-sonar-simulation-approaches)
4. [Underwater Vision Models](#4-underwater-vision-models)
5. [DVL Simulation](#5-dvl-simulation)
6. [Acoustic Communications Simulation](#6-acoustic-communications-simulation)
7. [Fused Perception Pipelines](#7-fused-perception-pipelines)
8. [GPU-Accelerated Rendering for Underwater](#8-gpu-accelerated-rendering-for-underwater)
9. [Gap Analysis for OceanScale](#9-gap-analysis-for-oceanscale)
10. [References](#references)

---

## 1. OceanSim Sensor Architecture

**Primary source:** Song et al., "OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework," arXiv:2503.01074v2, March 2025. Read in full from [arxiv.org/html/2503.01074v2](https://arxiv.org/html/2503.01074v2). Also published at IEEE: [ieeexplore.ieee.org/document/11246878](https://ieeexplore.ieee.org/document/11246878/).

**Platform:** Built as a custom extension to NVIDIA Isaac Sim, leveraging Omniverse Replicator + OpenUSD ecosystem.

### Supported Sensors

| Sensor | Implementation | Key Details |
|--------|---------------|-------------|
| **RGB/Depth Camera** | Physics-based image formation model (Eq. 1 in paper) | Models attenuation + backscatter per-channel; depth-dependent; user-tunable via GUI plugin |
| **Imaging Sonar** | GPU ray-tracing via Omniverse Replicator point-cloud annotator | Intensity = surface normal dot product * range attenuation * acoustic reflectance; speckle noise (Rayleigh + Gaussian); defaults calibrated to Blueprint Subsea Oculus M750-d |
| **DVL** | 4-beam Janus array model | Range-dependent adaptive frequency; beam dropout when out of range; designed for testing state estimation |
| **Barometer** | Pressure from user-defined atmospheric pressure + water density + optional Gaussian noise | Simple model |
| **IMU** | Uses Isaac Sim built-in model | Not custom-implemented |

### Camera Model (Underwater Image Formation)

The paper implements the Akkaynak-Treibitz revised underwater image formation model (CVPR 2018). For each color channel $c \in \{R,G,B\}$:

$$I_c = J e^{-\beta_{attn,c} \cdot d} + B_{\infty,c}(1 - e^{-\beta_{bs,c} \cdot d})$$

Where:
- $J$ = in-air image rendered by Isaac Sim
- $d$ = depth (range from camera, from depth image)
- $\beta_{attn,c}$ = per-channel attenuation coefficient
- $\beta_{bs,c}$ = per-channel backscatter coefficient
- $B_{\infty,c}$ = veiling light (backscatter asymptote)

Parameters are configurable per water type. GPU-accelerated computation. The paper provides a GUI plugin for interactive tuning with real-time preview.

### Sonar Model (GPU Ray-Tracing Pipeline)

1. Set up virtual rendering viewport attached to Omniverse Replicator
2. Query scene geometry (normals, semantics) via GPU ray-tracing using the point cloud annotator API
3. Compute intensity per return point:

$$I_{sonar} = A_r \left(-\frac{\vec{v}_{in}}{|\vec{v}_{in}|} \cdot \frac{\vec{n}}{|\vec{n}|}\right) e^{-\alpha d}$$

4. Bin returns onto range-azimuth polar grid (sum intensities per bin)
5. Add speckle noise: additive (Rayleigh-distributed, beam-pattern-weighted) + multiplicative (Gaussian)
6. Range-wise normalization (for imaging sonar display)

**Acoustic reflectance** $A_r$: uniform default per object, or per-semantic-label via Isaac Sim Semantics Schema Editor.

**Default parameters:** Calibrated to Blueprint Subsea Oculus M750-d (130 deg horizontal FOV, 20 deg vertical FOV).

### Sonar Rendering Speed (Table II from paper)

| Scene | HoloOcean Cache (s) | HoloOcean Render (fps) | OceanSim Render (fps) |
|-------|--------------------|-----------------------|-----------------------|
| Towing tank | 3.0 | 4.33 | 41.7 |
| Pier | 7.4 | 2.83 | 35.7 |
| Underwater mall | 11.4* | 0.63 | 11.8 |

OceanSim achieves **4-18x faster** sonar rendering than HoloOcean, with no cache pre-build step. Tested on NVIDIA RTX A6000 GPU.

### Key Limitations (stated in paper, Section V)

- No underwater vehicle dynamics or fluid simulation (perception-only simulator)
- No optical/acoustic communication modem simulation
- Configuration is code-only (no GUI for scene setup yet)

---

## 2. HoloOcean 2.x Sensor Suite

**Primary source:** Romrell et al., "A Preview of HoloOcean 2.0," arXiv:2510.06160v1, 2025. Read in full from [arxiv.org/html/2510.06160v1](https://arxiv.org/html/2510.06160v1).

Also: Potokar et al., "HoloOcean: Realistic Sonar Simulation," IROS 2022 ([PDF](https://robots.et.byu.edu/jmangelson/pubs/2022/Potokar22iros.pdf)), and "HoloOcean: A Full-Featured Marine Robotics Simulator," IEEE J. Oceanic Engineering, 2024 ([IEEE](https://ieeexplore.ieee.org/document/10451207)).

### Platform Evolution

| Version | Engine | Key Changes |
|---------|--------|-------------|
| 1.0 (ICRA 2022) | Unreal Engine 4.27 | Initial release; DVL, IMU, cameras, sonar via octree |
| 2.0 (2025 preview) | Unreal Engine 5.3 | Lumen lighting, Nanite geometry, Fossen dynamics, ROS2 bridge |

### Sensor Models (HoloOcean 2.0)

| Sensor | Implementation Status | Details |
|--------|----------------------|---------|
| **RGB Camera** | Available v1.0, enhanced in 2.0 | v2.0 adds 29 configurable parameters (FOV, shutter speed, focal distance, etc.) |
| **Depth Camera** | New in 2.0 | Same params as camera + depth channel |
| **LiDAR** | New in 2.0 | Adapted from CARLA simulator; configurable lasers, rotation freq, FOV, max range |
| **Imaging Sonar** | Available v1.0, rewriting to ray-cast for 2.0 | v1.0 uses octree cache; v2.0 migrating to direct ray tracing |
| **Sidescan Sonar** | Available v1.0, rewriting for 2.0 | Same octree-to-ray-tracing migration |
| **Bathymetric Sonar** | Available v1.0, rewriting for 2.0 | Same migration |
| **Profiling Sonar** | Available v1.0, rewriting for 2.0 | Same migration |
| **DVL** | Available v1.0 | Velocity measurement sensor |
| **IMU** | Available v1.0 | Inertial measurement unit |
| **Depth/Pressure** | Available v1.0 | Pressure-based depth |

### Sonar Architecture Change (v1.0 octree -> v2.0 ray casting)

**v1.0 (octree-based):**
- Pre-builds octree cache of environment geometry at startup
- Runtime queries octree for sonar ray intersections
- Problems: slow cache build (seconds to minutes), fails on large/complex environments, cannot see dynamically spawned objects, memory-intensive

**v2.0 (direct ray casting):**
- Direct ray tracing every tick -- no pre-cache needed
- Sees dynamically spawned objects
- Benchmarked **4.5x faster** than octree query, **29x faster** than octree build (single-beam echosounder test, Table II in paper)
- Plan to migrate all sonar types (sidescan, imaging, profiling, bathymetric) to ray casting

### Semantic Sensors (new in 2.0)

- Camera, depth camera, LiDAR: pixel-wise semantic/instance labels based on environment object tags
- Sonar semantic labeling: in development
- Enables training segmentation and detection models

### Sim-to-Real Validation

- Hardware-in-the-loop (HIL) testing demonstrated via ROS2 bridge ([arXiv:2511.07687](https://arxiv.org/html/2511.07687v1))
- BYU FRoST Lab uses HIL to verify control, navigation, and localization algorithms

### Accuracy vs Real Hardware

The original HoloOcean paper (IROS 2022) compares rendered sonar output to real sensor data. The OceanSim paper (Table I) shows HoloOcean has higher RGB angular error than OceanSim for underwater image rendering, suggesting HoloOcean's visual fidelity is currently lower. No published quantitative DVL/IMU accuracy comparison found.

---

## 3. Sonar Simulation Approaches

### Taxonomy of Approaches

| Approach | Description | Speed | Fidelity | Examples |
|----------|-------------|-------|----------|----------|
| **Rasterization-based** | Project 3D scene onto 2D sonar view using GPU rasterization | Fast | Low-Medium | Early UUV Simulator, DeMarco2015 ([IEEE OCEANS 2015](https://ieeexplore.ieee.org/document/7404418)) |
| **Octree + Ray Query** | Pre-build spatial octree, query at runtime | Medium (runtime), Slow (cache build) | Medium-High | HoloOcean v1.0 |
| **GPU Ray Tracing** | Direct ray-tracing from sensor origin, no pre-cache | Fast | High | OceanSim (Omniverse Replicator), HoloOcean 2.0 (UE5 ray cast), bellhopcuda |
| **Physics-based (point-scattering)** | Model acoustic physics: transmission loss, scattering, absorption | Slow | Very High | Cerqueira et al. 2021 ([Frontiers in Robotics and AI](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2021.706646/full)), ROS-Gazebo multibeam 2025 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/)) |
| **Neural/Learned** | Train neural networks to map scene geometry to sonar images | Potentially real-time (inference) | Variable (depends on training data) | Emerging; no dominant published framework found |

### State of the Art for Real-Time GPU Sonar (2025)

**OceanSim** holds the current speed record for imaging sonar rendering (11-42 fps depending on scene complexity, tested on RTX A6000). Key innovations:
- Uses NVIDIA Omniverse Replicator's GPU-accelerated point-cloud annotator (no custom CUDA kernel)
- Scene geometry queried via ray-tracing in the Omniverse rendering pipeline
- Post-processing (binning, noise) done in NVIDIA Warp (GPU parallel compute library)

**HoloOcean 2.0** is migrating to direct ray casting in UE5, with early benchmarks showing significant speedups over octree. Not yet benchmarked against OceanSim in the published preview paper.

**GPU-accelerated BELLHOP** ([bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda)) targets acoustic propagation, not imaging sonar, but the ray-tracing infrastructure overlaps.

**OptiX-based sonar simulation** ([DIVA Portal thesis](https://www.diva-portal.org/smash/get/diva2:1352170/FULLTEXT01.pdf)) demonstrated GPU ray-tracing for sound propagation using NVIDIA OptiX, but is a research prototype, not integrated into a robotics simulator.

**GLSL-based real-time sonar** (Cerqueira et al. 2017, [Computers & Graphics](https://www.sciencedirect.com/science/article/abs/pii/S0097849317301371)): earlier work using OpenGL shader pipeline for GPU sonar rendering. Superseded by modern ray-tracing approaches.

### Key Gap

No existing simulator provides **full-wave acoustic simulation** (accounting for multipath, reverberation, bottom/sub-surface reflections at varying frequencies) at real-time rates. All current approaches use simplified geometric acoustics (ray/beam tracing) with empirical noise models.

---

## 4. Underwater Vision Models

### Jaffe-McGlamery (JMG) Image Formation Model

The foundational model for underwater image formation, used in nearly all underwater simulators.

**Three-component decomposition** ([TRACE University of Tennessee](https://trace.tennessee.edu/server/api/core/bitstreams/9ed6861e-4e2f-4573-bd38-3107a9cd5667/content), [GEOMAR paper](https://www.geomar.de/fileadmin/personal/fb2/mg/ajordt/vmvPaper.pdf)):

1. **Direct transmission** -- light traveling from object to camera along direct path
2. **Forward scattering** -- light scattered at small angles, causing blurring
3. **Backscatter** -- light scattered back toward camera by suspended particles (veiling light)

**OceanSim's implementation** (Eq. 1 in their paper) combines direct transmission and backscatter:

$$I_c = J \cdot e^{-\beta_{attn,c} \cdot d} + B_{\infty,c}(1 - e^{-\beta_{bs,c} \cdot d})$$

This is a simplified form of the JMG model that omits explicit forward scattering (which primarily causes blur and is often handled separately or ignored for real-time rendering). Based on Akkaynak & Treibitz CVPR 2018 ([revised underwater IFM](https://openaccess.thecvf.com/content_cvpr_2018/html/Akkaynak_A_Revised_Underwater_CVPR_2018_paper.html)).

### Which Simulators Implement What?

| Simulator | Water Absorption | Backscatter | Forward Scatter | Caustics | Turbidity | Color Shift |
|-----------|-----------------|-------------|-----------------|----------|-----------|-------------|
| **OceanSim** | Yes (per-channel $\beta_{attn}$) | Yes ($B_{\infty,c}$) | No (omitted) | Not mentioned | Via attenuation params | Yes (per-channel) |
| **HoloOcean** | Limited (v1.0 no depth-dependent water column effects per OceanSim paper Sec II-A) | Limited | No | No | No | Limited |
| **UNav-Sim** | Yes (UE5.1 based) | Yes | Not specified | Not specified | Not specified | Yes |
| **Stonefish** | No (focus on dynamics, not vision rendering) | No | No | No | No | No |
| **UUV Simulator** | No (Gazebo-based, poor visual fidelity) | No | No | No | No | No |

### Caustics Simulation

Caustics (light patterns caused by water surface refraction) are a notable gap in most simulators:

- **GEOMAR paper** ([vmvPaper.pdf](https://www.geomar.de/fileadmin/personal/fb2/mg/ajordt/vmvPaper.pdf)) extends JMG model for deep-sea caustics
- **NVIDIA GPU Gems Ch.2** ([developer.nvidia.cn/gpugems](https://developer.nvidia.cn/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)): classic real-time caustics rendering technique
- **KTH thesis** ([diva-portal.org](https://kth.diva-portal.org/smash/get/diva2:1948920/FULLTEXT01.pdf)): hybrid rendering (screen-space + ray tracing) for real-time water caustics
- **OceanSim**: does not explicitly mention caustics in the paper
- **Omniverse RTX settings**: includes caustics toggle in the RTX real-time renderer ([Bentley/LumenRT docs](https://docs.bentley.com/LiveContent/web/LumenRT%2520for%2520NVIDIA%2520Omniverse%2520Help-v2/en/GUID-20095E11-8D55-41E5-959C-D0EC01D9DAB5.html))

### Color Absorption

OceanSim models per-channel (R, G, B) attenuation, following Akkaynak & Treibitz 2017 ([CVPR 2017](https://arxiv.org/abs/1703.03259)), which established that underwater attenuation coefficients are not well-described by simple exponential models and depend on the object-camera geometry. This is a key improvement over simulators that apply uniform color tinting.

### Deep-Sea Extension

The GEOMAR paper ([vmvPaper.pdf](https://www.geomar.de/fileadmin/personal/fb2/mg/ajordt/vmvPaper.pdf)) extends JMG for deep-sea conditions including:
- Artificial lighting (no sunlight)
- Increased turbidity from sediment
- Bioluminescence (optional)
- Multiple scattering at depth

---

## 5. DVL Simulation

### What is DVL?

A Doppler Velocity Log measures vehicle velocity relative to the seafloor using 4 acoustic beams in a Janus array configuration. It also provides altitude (range to bottom). Key for AUV navigation when GPS is unavailable.

### DVL Implementations in Simulators

| Simulator | DVL Model | Key Features |
|-----------|-----------|-------------|
| **OceanSim** | 4-beam Janus array | Range-dependent adaptive measurement rate; beam dropout when out of range; designed for testing state estimation |
| **HoloOcean** | Available (v1.0+) | Velocity measurement; parameters in docs ([byu-holoocean.github.io](https://byu-holoocean.github.io/holoocean-docs/v2.0.1/index.html)) |
| **Stonefish** | Available | ROS1/ROS2 compatible; noise models included |
| **UUV Simulator** | Available | Gazebo plugin-based |
| **DVL-Simulator (standalone)** | C++ ROS2 node | Open source: [github.com/maximilian-nitsch/DVL-Simulator](https://github.com/maximilian-nitsch/DVL-Simulator) |
| **MathWorks** | Simulink block | INS/DVL fusion demo: [mathworks.com](https://www.mathworks.com/help/nav/ug/autonomous-underwater-vehicle-pose-estimation-using-inertial-sensors-and-doppler-velocity-log.html) |

### OceanSim DVL Model Details (from paper Section III-A3)

OceanSim's DVL model specifically simulates two noise sources:
1. **Adaptive measurement rate** -- DVL changes update frequency based on altitude (range to bottom), creating irregular measurement timestamps that challenge multi-sensor fusion algorithms
2. **Beam dropout** -- when beams are out of operational range, the DVL returns invalid velocity, simulating real-world failure modes

These features are specifically designed for testing state estimation (Kalman filter, factor graph) robustness.

### Accuracy Considerations

Real-world IMU/DVL fusion achieves ~0.09 m/s standard deviation velocity error with max error ~0.53 m/s ([MDPI Sensors 2021](https://www.mdpi.com/1424-8220/21/4/1056)). DVL accuracy depends on:
- Altitude (too high = weak returns, too low = limited coverage)
- Bottom type (smooth sand vs rocky terrain)
- Vehicle pitch/roll (beam geometry changes)
- Water salinity and temperature (sound speed)

None of the surveyed simulators model all of these effects. OceanSim's adaptive rate and dropout are the most realistic noise modeling found.

---

## 6. Acoustic Communications Simulation

### BELLHOP

**What:** Gaussian beam tracing model for underwater acoustic propagation in 2D (range-depth). Developed by Michael Porter, maintained at the Ocean Acoustics Library ([oalib.hlsresearch.com](http://oalib.hlsresearch.com/Rays/HLS-2010-1.pdf)).

**Capabilities:**
- Ray tracing, eigenrays, transmission loss, arrivals
- Range-dependent sound speed profiles and bathymetry
- Multiple bottom/surface boundary conditions
- Written in Fortran; Python wrappers exist (arlpy, custom bridges)

**Speed:** Not real-time for complex environments. BELLHOP is typically used offline for mission planning or channel characterization.

**GPU-accelerated version:** [bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda) -- CUDA/C++ port of BELLHOP/BELLHOP3D. Developed at UCSD for GPU-accelerated acoustic soundscaping ([jaffeweb.ucsd.edu](https://jaffeweb.ucsd.edu/research-projects/underwater-acoustic-soundscaping-with-bellhopcuda/)). This is the most promising path toward real-time BELLHOP.

### Q-D Model

The **Q-D model** (statistical quality-of-service delay model) is used in underwater acoustic network simulation to characterize packet delivery probability and latency as functions of distance, frequency, and environmental conditions. Not widely found in robotics simulators.

### Real-Time Approximate Models for RL

For reinforcement learning training, full BELLHOP is too slow. Approaches found:

| Approach | Speed | Fidelity | Source |
|----------|-------|----------|--------|
| **Look-up table** from pre-computed BELLHOP runs | Real-time | High (for pre-computed conditions) | Common practice |
| **Simplified path loss** (e.g., Thorp's equation + cylindrical spreading) | Real-time | Low | Used in NS-3 underwater module |
| **bellhopcuda** (GPU-accelerated) | Near-real-time (GPU dependent) | High | [github.com/A-New-BellHope/bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda) |
| **RL-learned channel model** | Real-time (inference) | Variable | [Springer EURASIP 2022](https://link.springer.com/article/10.1186/s13634-022-00961-5) |
| **Q-learning adaptive modulation** | Online learning | N/A (policy, not channel) | [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2590123024010466) |

**Recommendation for OceanScale:** Pre-compute BELLHOP transmission loss tables for representative environments, then interpolate at runtime. Use bellhopcuda for generating the tables at training time. For RL, a simple path loss model (Urick equation) with stochastic fading is sufficient for policy training; reserve high-fidelity BELLHOP for evaluation.

### OceanSim's Gap

OceanSim explicitly states (Section V): "OceanSim does not implement optical and acoustic communication modems to simulate communications between cooperative agents." This is listed as future work.

---

## 7. Fused Perception Pipelines

### Multi-Sensor Fusion for Underwater

**Survey:** "Advancements in Sensor Fusion for Underwater SLAM" ([PMC/NIH 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644431/)) covers fusion of:
- **Proprioceptive:** IMU, DVL
- **Exteroceptive:** cameras, sonar, depth/pressure

### Key Fusion Frameworks

| Framework | Sensors Fused | Key Feature | Source |
|-----------|--------------|-------------|--------|
| **SVIn2** | Stereo camera + IMU + profiling sonar + depth (pressure) | Factor graph SLAM; sonar provides scale constraint | Rahman et al., IROS 2019 ([CMU PDF](https://rpl.ri.cmu.edu/reading-group/2020/uw-slam/uw-slam.pdf)) |
| **Visual-Inertial-Acoustic-DVL SLAM** | Camera + IMU + sonar + DVL | DVL preintegration for velocity constraints | [core.ac.uk PDF](https://core.ac.uk/download/pdf/665167966.pdf) |
| **Visual-DVL-Inertial Odometry** | Camera + DVL + IMU | Tightly-coupled for ice-water boundary | Zhao et al., IROS 2023 |
| **Deep Learning + SLAM** | Multi-modal | Neural network enhanced front-end | [PMC 2025 survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC12157327/) |

### Benchmark Datasets

| Dataset | Year | Modalities | Focus | Source |
|---------|------|------------|-------|--------|
| **Tank Dataset** | 2025 | Stereo camera, IMU, sonar | Multi-sensor underwater SLAM evaluation | [SAGE Journals](https://journals.sagepub.com/doi/10.1177/02783649251364904) |
| **NTNU ROV Multi-Sensor** | 2025 | Multi-sensor ROV data | Situational awareness benchmark | [arXiv:2506.06476](https://arxiv.org/html/2506.06476v1) |
| **UEOF Benchmark** | 2026 | Event camera | Underwater event-based optical flow | [WACV 2026](https://openaccess.thecvf.com/content/WACV2026W/EVGEN-2026/papers/Truong_UEOF_A_Benchmark_Dataset_for_Underwater_Event-Based_Optical_Flow_WACVW_2026_paper.pdf) |
| **UVVID** | Recent | Visual + visual-inertial | Underwater visual SLAM benchmark | [DTU Data](https://data.dtu.dk/articles/dataset/Underwater_Visual_and_Visual-Inertial_Datasets_UVVID_/27694068) |
| **Synthetic Underwater Benchmark** | Recent | Imaging + sonar (synthetic) | Restoration, depth estimation, tracking | [Emergent Mind](https://www.emergentmind.com/topics/synthetic-underwater-benchmark-dataset) |

### No Standardized Benchmark

There is no single widely-accepted underwater multi-sensor fusion benchmark analogous to KITTI for autonomous driving. The Tank Dataset (2025) is the closest, providing stereo camera + IMU + sonar with ground truth for SLAM evaluation. The NTNU ROV dataset adds more sensor modalities but is newer and less established.

---

## 8. GPU-Accelerated Rendering for Underwater

### What NVIDIA Omniverse / Isaac Sim Provides

| Feature | Status | Notes |
|---------|--------|-------|
| **RTX Path Tracing** | Available | Physically-based ray tracing on RTX GPUs ([Omniverse docs](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)) |
| **RTX Real-Time Mode** | Available | Hybrid rasterization + ray tracing for real-time |
| **Caustics** | Available (toggle) | RTX Real-Time settings include caustics ([Bentley docs](https://docs.bentley.com/LiveContent/web/LumenRT%2520for%2520NVIDIA%2520Omniverse%2520Help-v2/en/GUID-20095E11-8D55-41E5-959C-D0EC01D9DAB5.html)) |
| **Volumetric Effects** | Available | Global volumetric effects in RTX settings |
| **Translucency/Refraction** | Available | Configurable max refraction bounces |
| **MDL Material System** | Available | Full material definition language for custom water properties |
| **Underwater rendering** | NOT built-in | Isaac Sim has no official underwater environment or water column effects ([NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374)) |
| **Water surface** | Partial | Some community efforts; no official ocean plugin |

### What OceanSim Adds

OceanSim is the most complete underwater rendering layer on Isaac Sim:
- Water column image formation model (attenuation + backscatter)
- GPU-accelerated imaging sonar rendering
- DVL, barometer models
- Digital twin environments (towing tank, shipwreck)

### What Still Needs Custom Work for OceanScale

| Feature | Available? | Effort |
|---------|-----------|--------|
| Underwater camera with JMG model | Yes (OceanSim) | Low -- use OceanSim extension |
| Imaging sonar | Yes (OceanSim) | Low -- use OceanSim extension |
| Caustics rendering | Partially (RTX caustics toggle) | Medium -- need to configure water surface + material |
| Volumetric lighting (god rays) | Partially (RTX volumetric) | Medium -- needs scene-specific tuning |
| Participating media (turbidity) | Yes (OceanSim attenuation params) | Low |
| Sidescan/profiling sonar | Not in OceanSim; available in HoloOcean | Medium-High -- need to implement or port |
| Acoustic propagation (BELLHOP) | No | High -- need custom integration |
| Communication simulation | No | High -- need custom integration |
| Fluid dynamics / currents | No (OceanSim limitation) | High -- need Newton/Warp fluid solver |
| Vehicle dynamics (Fossen) | Not in OceanSim; available in HoloOcean 2.0, MarineGym | Medium -- can add Fossen model |

---

## 9. Gap Analysis for OceanScale

Based on this research, the following gaps exist between available open-source components and what a full-stack GPU-native underwater simulator needs:

| Gap | Severity | Existing Solution | Custom Work Needed |
|-----|----------|-------------------|--------------------|
| Underwater camera rendering | SOLVED | OceanSim (Isaac Sim extension) | None |
| Imaging sonar (real-time GPU) | SOLVED | OceanSim | None |
| DVL simulation | MOSTLY SOLVED | OceanSim (adaptive rate + dropout) | Minor enhancements |
| Sidescan/profiling sonar | GAP | HoloOcean has models, not GPU-fast | Port HoloOcean models to Isaac Sim, or implement ray-cast versions |
| Acoustic communication | GAP | BELLHOP/bellhopcuda (offline/near-real-time) | Build real-time approximate model + integrate |
| Fluid dynamics | GAP | Newton + Warp (OceanScale's core) | Core OceanScale work |
| Caustics/volumetric | PARTIAL | RTX caustics + volumetric settings | Scene configuration + material tuning |
| Sensor fusion benchmark | GAP | Tank Dataset (2025) is closest | Create synthetic benchmark via OceanSim |
| Multi-agent comms | GAP | No simulator handles this well | Custom acoustic channel simulation |

---

## References

### Primary Papers (read in full)

1. Song, J., Ma, H., Bagoren, O., Sethuraman, A.V., Zhang, Y., & Skinner, K.A. "OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework." arXiv:2503.01074v2, March 2025. [arxiv.org/html/2503.01074v2](https://arxiv.org/html/2503.01074v2)

2. Romrell, B., Austin, A., Meyers, B., Anderson, R., Noh, C., & Mangelson, J.G. "A Preview of HoloOcean 2.0." arXiv:2510.06160v1, 2025. [arxiv.org/html/2510.06160v1](https://arxiv.org/html/2510.06160v1)

3. Potokar, E., Lay, K., Norman, K., Benham, D., Ashford, S., Peirce, R., Neilsen, T.B., Kaess, M., & Mangelson, J.G. "HoloOcean: A Full-Featured Marine Robotics Simulator for Perception and Autonomy." IEEE J. Oceanic Engineering, vol. 49, no. 4, pp. 1322-1336, 2024. [IEEE](https://ieeexplore.ieee.org/document/10451207)

4. Potokar, E., Lay, K., Norman, K., Benham, D., Neilsen, T.B., Kaess, M., & Mangelson, J.G. "HoloOcean: Realistic Sonar Simulation." IROS 2022, pp. 8450-8456. [PDF](https://robots.et.byu.edu/jmangelson/pubs/2022/Potokar22iros.pdf)

### Supporting References

5. Akkaynak, D. & Treibitz, T. "A Revised Underwater Image Formation Model." CVPR 2018, pp. 6723-6732.
6. Akkaynak, D., Treibitz, T., et al. "What is the Space of Attenuation Coefficients in Underwater Computer Vision?" CVPR 2017, pp. 568-577. [arXiv:1703.03259](https://arxiv.org/abs/1703.03259)
7. Cerqueira, R., Trocoli, T., Neves, G., Joyeux, S., Albiez, J., & Oliveira, L. "A Novel GPU-Based Sonar Simulator for Real-Time Applications." Computers & Graphics, vol. 68, pp. 66-76, 2017. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0097849317301371)
8. Cieślak, P. "Stonefish: An Advanced Open-Source Simulation Tool Designed for Marine Robotics, With a ROS Interface." OCEANS 2019 - Marseille, pp. 1-6.
9. Rahman, S., Li, A.Q., & Rekleitis, I. "SVIn2: An Underwater SLAM System Using Sonar, Visual, Inertial, and Depth Sensor." IROS 2019, pp. 1861-1868.
10. Porter, M.B. "The BELLHOP Manual and User's Guide (Preliminary Draft)." HLS Research, 2010. [oalib.hlsresearch.com](http://oalib.hlsresearch.com/Rays/HLS-2010-1.pdf)
11. bellhopcuda: [github.com/A-New-BellHope/bellhopcuda](https://github.com/A-New-BellHope/bellhopcuda)
12. DVL-Simulator (ROS2): [github.com/maximilian-nitsch/DVL-Simulator](https://github.com/maximilian-nitsch/DVL-Simulator)
13. Advancements in Sensor Fusion for Underwater SLAM: [PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11644431/)
14. Deep RL + Underwater Acoustic Adaptive Modulation: [Springer EURASIP 2022](https://link.springer.com/article/10.1186/s13634-022-00961-5)
15. RL-based Automated Modulation Switching: [ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S2590123024010466)
16. Scaling Multi-Agent RL for Underwater Operations: [arXiv:2505.08222](https://arxiv.org/html/2505.08222v1)
17. Ray-Based Physical Modeling of Multibeam Sonar in ROS-Gazebo: [PMC 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/)
18. GPU Gems Ch.2 -- Rendering Water Caustics: [developer.nvidia.cn](https://developer.nvidia.cn/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)
19. NVIDIA Omniverse RTX Renderer: [docs.omniverse.nvidia.com](https://docs.omniverse.nvidia.com/materials-and-rendering/latest/rtx-renderer_pt.html)
20. Tank Dataset for Underwater SLAM: [SAGE Journals 2025](https://journals.sagepub.com/doi/10.1177/02783649251364904)
21. NTNU ROV Multi-Sensor Dataset: [arXiv:2506.06476](https://arxiv.org/html/2506.06476v1)
22. Underwater Robotics Simulators Review (UCL): [arXiv:2504.06245](https://arxiv.org/abs/2504.06245)
23. OceanSim Project Page: [umfieldrobotics.github.io/OceanSim](https://umfieldrobotics.github.io/OceanSim/)
24. OceanSim GitHub: [github.com/umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim)
25. HoloOcean Docs v2.2.1: [byu-holoocean.github.io](https://byu-holoocean.github.io/holoocean-docs/v2.2.1/index.html)
26. NVIDIA Isaac Sim ocean environment discussion: [forums.developer.nvidia.com](https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374)
27. MarineGym: Chu et al., "MarineGym: Accelerated Training for Underwater Vehicles with High-Fidelity RL Simulation," 2024. Referenced in OceanSim paper [34].
28. IMU/DVL integration accuracy: [MDPI Sensors 2021](https://www.mdpi.com/1424-8220/21/4/1056)
29. Hardware-in-the-loop with HoloOcean 2.0: [arXiv:2511.07687](https://arxiv.org/html/2511.07687v1)
30. UEOF Benchmark for Underwater Event-Based Optical Flow: [WACV 2026](https://openaccess.thecvf.com/content/WACV2026W/EVGEN-2026/papers/Truong_UEOF_A_Benchmark_Dataset_for_Underwater_Event-Based_Optical_Flow_WACVW_2026_paper.pdf)
