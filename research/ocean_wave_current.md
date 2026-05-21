# GPU-Accelerated Ocean Wave, Current, and Turbulence Simulation for Robotics

> Deep research for OceanScale GPU-native underwater simulator.
> Date: 2026-05-20. All URLs verified via web search; no fabricated links.

---

## Table of Contents

1. [Ocean Wave Models for Simulation](#1-ocean-wave-models-for-simulation)
2. [Ocean Current Simulation](#2-ocean-current-simulation)
3. [Turbulence Models for Underwater Robotics](#3-turbulence-models-for-underwater-robotics)
4. [Free Surface Interaction](#4-free-surface-interaction)
5. [Open-Source Ocean Simulation Libraries](#5-open-source-ocean-simulation-libraries)
6. [NVIDIA-Specific Ocean/Water Tools](#6-nvidia-specific-oceanwater-tools)
7. [Sea State Modeling](#7-sea-state-modeling)

---

## 1. Ocean Wave Models for Simulation

### 1.1 Gerstner Waves (Analytical, GPU-Friendly)

Gerstner waves model ocean surface as a sum of sinusoidal displacements with trochoidal (sharp-crested) profiles. They are the simplest GPU-friendly ocean model.

**Key properties:**
- Analytical closed-form: sum of N wave components, each parameterized by amplitude, frequency, direction, steepness
- Direct vertex shader evaluation -- no FFT required
- Suitable for small-to-medium scale water surfaces (ponds, lakes, coastal)
- Degrades at ocean scale: limited wave count before visual repetition or performance loss
- Cannot capture the statistical complexity of real ocean spectra

**References:**
- NVIDIA GPU Gems Chapter 1 covers Gerstner wave implementation in detail: [https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models)
- Open-source implementation achieving 200 FPS with 12-wave Gerstner sum: [https://github.com/CaffeineViking/osgw](https://github.com/CaffeineViking/osgw)
- Fast GPU simulation combining Gerstner with ocean wave spectra: [https://www.researchgate.net/publication/313495234_Fast_Simulation_Method_for_Ocean_Wave_Base_on_Ocean_Wave_Spectrum_and_Improved_Gerstner_Model_with_GPU](https://www.researchgate.net/publication/313495234_Fast_Simulation_Method_for_Ocean_Wave_Base_on_Ocean_Wave_Spectrum_and_Improved_Gerstner_Model_with_GPU)

### 1.2 FFT-Based Ocean (Tessendorf 2001)

Jerry Tessendorf's 2001 paper "Simulating Ocean Water" remains the gold standard for large-scale realistic ocean rendering in 2024-2025. It uses inverse FFT to efficiently sum thousands of wave components drawn from oceanographic spectra.

**Key properties:**
- Statistically accurate: generates heightfields from Phillips, Pierson-Moskowitz, or JONSWAP spectra
- GPU IFFT is trivial on modern hardware -- NVIDIA's own slides note "FFT takes trivial time to complete on most GPUs" ([https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/OceanCS_Slides.pdf](https://developer.download.nvidia.com/assets/gamedev/files/sdk/11/OceanCS_Slides.pdf))
- Modern implementations achieve 600 FPS at 4K on RTX 4090 with 34.5k triangles: [https://mustard-cg.com/projects/ocean_simulation](https://mustard-cg.com/projects/ocean_simulation)
- Jacobian-based foam/whitecap detection is a natural byproduct
- Main artifact: tiling repetition (addressed by Ubisoft LaForge's tiling-and-blending, HPG 2024)

**Status: Still the standard in 2025-2026.** No replacement has emerged. Recent advances are incremental improvements (better spectra, tiling mitigation, neural surrogates).

**Key implementations:**
- WebGPU implementation with full tutorial (2024): [https://barthpaleorge.github.io/Blog/posts/ocean-simulation-webgpu/](https://barthpaleologue.github.io/Blog/posts/ocean-simulation-webgpu/)
- WSCG 2025 paper on FFT ocean for real-time: [http://wscg.zcu.cz/WSCG2025/papers/C59.pdf](http://wscg.zcu.cz/WSCG2025/papers/C59.pdf)
- Comprehensive tutorial updated Feb 2026: [https://www.gikster.dev/posts/Ocean-Simulation/](https://www.gikster.dev/posts/Ocean-Simulation/)
- GPGPU FFT with OpenGL Compute Shaders: [https://tore.tuhh.de/bitstream/11420/1439/1/GPGPU_FFT_Ocean_Simulation.pdf](https://tore.tuhh.de/bitstream/11420/1439/1/GPGPU_FFT_Ocean_Simulation.pdf)
- Ubisoft LaForge tiling-and-blending (HPG 2024): [https://www.ubisoft.com/en-us/studio/laforge/news/5WHMK3tLGMGsqhxmWls1Jw/making-waves-in-ocean-surface-rendering-using-tiling-and-blending](https://www.ubisoft.com/en-us/studio/laforge/news/5WHMK3tLGMGsqhxmWls1Jw/making-waves-in-ocean-surface-rendering-using-tiling-and-blending)

**Emerging alternative:** ConvLSTM-based neural ocean waves (Ocean Engineering, 2023) replaces FFT with neural network for rapid heightfield generation. Early-stage, not production-ready.
Source: [https://www.sciencedirect.com/science/article/abs/pii/S0029801822017061](https://www.sciencedirect.com/science/article/abs/pii/S0029801822017061)

### 1.3 Pierson-Moskowitz Spectrum

The Pierson-Moskowitz (PM) spectrum models **fully developed seas** -- where wind has blown steadily over a long fetch for sufficient duration. It is parameterized by a single variable: wind speed at 19.5m height.

**Key properties:**
- S(f) = (alpha * g^2) / ((2*pi)^4 * f^5) * exp(-beta * (f0/f)^4)
- Parameters: alpha = 8.1e-3, beta = 0.74, f0 = g/(2*pi*U19.5)
- Single-peaked spectrum; suitable for calm-to-moderate open ocean
- JONSWAP reduces to PM when peak enhancement factor gamma = 1

**Reference:** ANSYS Aqwa documentation on PM as a special case of fully developed sea: [https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/aqwa_thy/aqwathy_env_irreg_waves.html](https://ansyshelp.ansys.com/public/Views/Secured/corp/v252/en/aqwa_thy/aqwathy_env_irreg_waves.html)

### 1.4 JONSWAP Spectrum

JONSWAP (Joint North Sea Wave Project) extends PM for **developing seas** with enhanced peak energy. It adds a peak enhancement factor gamma (typically 3.3) that sharpens the spectral peak.

**Key properties:**
- S(f) = PM(f) * gamma^exp(-(f-fp)^2 / (2*sigma^2*fp^2))
- Better represents fetch-limited, growing wind seas
- gamma typically 1-7; default 3.3 for North Sea conditions
- Standard spectrum for maritime engineering and simulation

**GPU acceleration:** Sundog Software white paper describes implementing PM spectrum as GPU-accelerated IFFT, extended with JONSWAP: [http://media.sundog-soft.com/GPUAcceleratedWaves-WhitePaper.pdf](http://media.sundog-soft.com/GPUAcceleratedWaves-WhitePaper.pdf)

**Spectrum reference:** WikiWaves ocean wave spectra overview: [https://www.wikiwaves.org/index.php/Ocean-Wave_Spectra](https://www.wikiwaves.org/index.php/Ocean-Wave_Spectra)

**Engineering tool:** OpenFAST/HydroDyn supports both PM and JONSWAP depending on input parameters: [https://github.com/OpenFAST/openfast/discussions/1397](https://github.com/OpenFAST/openfast/discussions/1397)

### 1.5 GPU Implementations Integrated with Isaac Sim / Omniverse

**No first-party ocean/water plugin exists in Isaac Sim or Omniverse.** The NVIDIA developer forum confirms Isaac Sim does not officially support oceanic environments out of the box: [https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374](https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374)

Third-party integrations:

| Project | Base | GPU Ocean | Status |
|---------|------|-----------|--------|
| OceanSim | Isaac Sim | Custom rendering | Accepted IROS 2025 |
| MarineGym | Isaac Sim | GPU hydro plugin | Accepted IROS 2025 |
| isaac_underwater | Isaac Sim | Community examples | Open source |

### 1.6 NVIDIA Ocean Plugin for Omniverse -- Does It Exist?

**No.** There is no official NVIDIA "Ocean Plugin" for Omniverse. The NVIDIA developer forum thread "Oceans in Omniverse" confirms users must build custom water simulation via the extension framework: [https://forums.developer.nvidia.com/t/oceans-in-omniverse/280926](https://forums.developer.nvidia.com/t/oceans-in-omniverse/280926)

What does exist:
- Omniverse USD Composer can assemble ocean scenes using RTX rendering (community tutorial): [https://edgedsign.com/2023/03/rendering-sea-scene-in-omniverse-create/](https://edgedsign.com/2023/03/rendering-sea-scene-in-omniverse-create/)
- Omniverse fluid scene tutorial (Physics Extension): [https://www.youtube.com/watch?v=DHKonOzJho4](https://www.youtube.com/watch?v=DHKonOzJho4)
- GTC 2025 session on digital twins for ocean exploration: [https://nvidia.com/en-us/on-demand/session/gtc25-s73656/](https://nvidia.com/en-us/on-demand/session/gtc25-s73656/)

---

## 2. Ocean Current Simulation

### 2.1 Real-Time Current Field Generation for RL Domain Randomization

No dedicated real-time ocean current simulator specifically designed for RL domain randomization was found. However, several approaches exist:

**Approaches for RL training:**
1. **Procedural current fields:** Parameterize current velocity fields using analytical functions (uniform flow, sinusoidal tidal oscillation, shear layers). Randomize parameters (magnitude, direction, shear rate) across episodes.
2. **Simplified shallow-water models on GPU:** SINTEF's GPU Ocean runs ensembles of shallow-water equation solvers on GPU at real-time speed. Could be adapted as a domain randomization current generator.
3. **Noise-based perturbation fields:** Use Perlin/Simplex noise or divergence-free random vector fields as current perturbations. Computationally cheap; can be evaluated on GPU in parallel for thousands of environments.

**MarineGym** (Isaac Sim-based) uses a GPU-accelerated hydrodynamic plugin that presumably includes current effects for its RL training at 250,000 FPS: [https://arxiv.org/html/2503.09203v1](https://arxiv.org/html/2503.09203v1)

### 2.2 Tidal Current Models

Tidal currents are generated by gravitational forces of moon and sun, producing periodic flows with known harmonic constituents.

**Models:**
- **Harmonic analysis (tidal constituents):** Represent tidal current as sum of sinusoidal components (M2, S2, N2, K1, O1, etc.). Parameters from tide gauge data. Standard in operational oceanography.
- **TPXO tidal model:** Global barotropic tide model providing tidal harmonic constituents. Widely used for tidal prediction.
- **Princeton Ocean Model (POM):** 3D ocean circulation model used for tidal current simulation: [https://www.nsfc.gov.cn/csc/20345/24371/pdf/2005/Numerical%20simulation%20of%20tides%20and%20tidal%20currents%20in%20Liaodong%20Bay%20with%20POM.pdf](https://www.nsfc.gov.cn/csc/20345/24371/pdf/2005/Numerical%20simulation%20of%20tides%20and%20tidal%20currents%20in%20Liaodong%20Bay%20with%20POM.pdf)

**For simulation/RL:** Tidal currents can be parameterized as a sum of 6-12 harmonic constituents, each with amplitude, phase, and frequency. This is deterministic and computationally trivial -- suitable for domain randomization by varying constituent amplitudes and phases.

### 2.3 Langmuir Circulation

Langmuir circulation (LC) consists of counter-rotating vortex pairs aligned with the wind direction, creating surface convergence zones (windrows). Relevant for surface and near-surface vehicle simulation.

**Key properties:**
- Wavelength: typically 5-50m (small cells) to hundreds of meters (large cells)
- Depth penetration: typically 1-2x the mixed layer depth
- Modeled via the **Craik-Leibovich (CL) vortex force** in RANS/LES simulations
- LC + tidal current coupling is an active research area (J. Fluid Mechanics, 2024): [https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/influence-of-langmuir-circulation-and-its-modulation-by-an-oscillating-alongshelf-current-on-the-dynamics-of-crossshelf-flows/8EFD10010173F6C1780B225804F44014](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/influence-of-langmuir-circulation-and-its-modulation-by-an-oscillating-alongshelf-current-on-the-dynamics-of-crossshelf-flows/8EFD10010173F6C1780B225804F44014)
- Comprehensive review: Thorpe, Annual Review of Fluid Mechanics, 2004: [ftp://soest-hawaii.edu/kelvin/OCN665/papers/thorpe_arfm_2004.pdf](ftp://soest-hawaii.edu/kelvin/OCN665/papers/thorpe_arfm_2004.pdf)
- AGU 2024: RANS simulations with field-measured tidal currents: [https://ui.adsabs.harvard.edu/abs/2024AGUOSCP44D1966H/abstract](https://ui.adsabs.harvard.edu/abs/2024AGUOSCP44D1966H/abstract)

**For simulation:** LC can be approximated analytically as a series of counter-rotating vortex cylinders parameterized by wind speed and fetch. This is computationally cheap and suitable for GPU parallel evaluation.

### 2.4 GPU-Accelerated Current Simulators

| Tool | Language | GPU Method | Notes |
|------|----------|------------|-------|
| **GPU Ocean (SINTEF)** | Python/PyOpenCL | Ensemble shallow-water solver | Open source; ensemble forecasting; perfect weak scaling on P100 GPU. [https://github.com/gpuocean/gpuocean](https://github.com/gpuocean/gpuocean) |
| **Oceananigans.jl** | Julia | KernelAbstractions.jl | 488m global ocean on 768 A100s; not real-time but breakthrough performance. [https://github.com/CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl) |
| **SCHISM** | Fortran/CUDA | CUDA Fortran | Operational coastal ocean model; first successful GPU acceleration. [https://www.mdpi.com/2077-1312/13/4/662](https://www.mdpi.com/2077-1312/13/4/662) |
| **NVIDIA Warp** | Python | CUDA JIT | Differentiable; suitable for custom current field generators. [https://github.com/nvidia/warp](https://github.com/nvidia/warp) |

SINTEF GPU Ocean is the most directly relevant for real-time current field generation. It runs shallow-water equation ensembles on GPU with particle filter data assimilation: [https://www.sintef.no/en/software/gpu-ocean/](https://www.sintef.no/en/software/gpu-ocean/)

---

## 3. Turbulence Models for Underwater Robotics

### 3.1 Turbulence Models Used in Underwater Simulators

Most underwater robotics simulators use simplified turbulence representations rather than full DNS or LES:

| Simulator | Turbulence Model | Notes |
|-----------|------------------|-------|
| Stonefish | Added mass + drag coefficients | No explicit turbulence field; hydrodynamic forces via coefficient-based models. [https://github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish) |
| HoloOcean | Simplified flow fields | Unreal Engine-based; limited turbulence fidelity. [https://byu-holoocean.github.io/holoocean-docs/](https://byu-holoocean.github.io/holoocean-docs/) |
| UUV Simulator | Uniform current + noise | Gazebo/ROS-based; basic current models. |
| MarineGym | GPU hydrodynamic plugin | Details of turbulence model not fully specified in paper. [https://github.com/Marine-RL/MarineGym](https://github.com/Marine-RL/MarineGym) |
| OceanSim | Isaac Sim rendering | Focuses on perception; physics-level turbulence not specified. [https://github.com/umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim) |

**Gap identified:** No underwater robotics simulator currently includes high-fidelity turbulence fields (DNS/LES quality) for vehicle dynamics.

### 3.2 Turbulence Parameterization for Domain Randomization

Turbulence can be parameterized for RL domain randomization using:

1. **Statistical turbulence parameters:**
   - Turbulent kinetic energy (TKE) dissipation rate (epsilon)
   - Integral length scale (L)
   - Turbulence intensity (TI = u'/U_mean)
   - Kolmogorov microscale (eta)

2. **Spectral methods:** Generate divergence-free turbulent velocity fields from a prescribed energy spectrum (e.g., von Karman spectrum). Randomize spectrum parameters across episodes.

3. **Synthetic turbulence generators:**
   - Random Fourier Modes (RFM): sum of random Fourier modes with prescribed spectrum
   - Divergence-Free Synthetic Turbulence (DFST): ensures incompressibility
   - Both are GPU-friendly (embarrassingly parallel mode evaluation)

4. **Practical approach for RL:** Generate turbulence as a superposition of N random vortex tubes or Fourier modes. Parameterize by turbulence intensity, integral length scale, and anisotropy. Evaluate on GPU in parallel across all environments.

### 3.3 Homogeneous Isotropic Turbulence (HIT) Models

HIT is the simplest turbulence model -- statistically uniform and direction-independent. It serves as a baseline for turbulence research and can be used as a domain randomization primitive.

**Key resources:**
- **MHIT36:** GPU-tailored solver for multiphase HIT using phase-field DNS. Includes applications to ocean CO2 exchange: [https://www.sciencedirect.com/science/article/pii/S0010465525003066](https://www.sciencedirect.com/science/article/pii/S0010465525003066)
- **APS review of GPU-accelerated DNS** of canonical turbulent flows including HIT: [https://link.aps.org/doi/10.1103/vz9c-bbzm](https://link.aps.org/doi/10.1103/vz9c-bbzm)
- **NVIDIA PhysicsNeMo turbulence super-resolution:** Train neural surrogates to upscale coarse HIT fields to high-fidelity: [https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/intermediate/turbulence_super_resolution.html](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/intermediate/turbulence_super_resolution.html)
- **SoZoGAN:** Generalizable super-resolution turbulence reconstruction using GANs, pretrained on HIT data: [https://arxiv.org/html/2511.02604v1](https://arxiv.org/html/2511.02604v1)
- **Recursive NN-based SGS model for LES** applied to HIT: [https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/recursive-neuralnetworkbased-subgridscale-model-for-large-eddy-simulation-application-to-homogeneous-isotropic-turbulence/1271EA1EF4678BBDA6A2D9B14BFDE1AE](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/recursive-neuralnetworkbased-subgridscale-model-for-large-eddy-simulation-application-to-homogeneous-isotropic-turbulence/1271EA1EF4678BBDA6A2D9B14BFDE1AE)

**For OceanScale:** HIT velocity fields can be precomputed on GPU (pseudo-spectral method) and stored as 3D textures. During RL training, sample from a library of HIT fields with varying Reynolds numbers. This is the cheapest way to add realistic turbulent perturbations to vehicle dynamics.

### 3.4 GPU-Accelerated Turbulence Generators

| Tool | Method | GPU? | Relevance |
|------|--------|------|-----------|
| MHIT36 | Phase-field DNS | Yes (GPU-tailored) | High-fidelity; too expensive for real-time RL |
| PhysicsNeMo | Neural super-resolution | Yes (GPU inference) | Upscale coarse turbulence; promising for RL |
| Pseudo-spectral HIT | Fourier-space DNS | Yes (cuFFT) | Precompute libraries of turbulence fields |
| Random Fourier Modes | Synthetic turbulence | Yes (parallel evaluation) | Cheapest; ideal for domain randomization |
| SPH-based DNS | Lagrangian particle method | Yes | [https://pubs.aip.org/aip/pof/article-pdf/doi/10.1063/5.0152154/18027753/065148_1_5.0152154.pdf](https://pubs.aip.org/aip/pof/article-pdf/doi/10.1063/5.0152154/18027753/065148_1_5.0152154.pdf) |

**Recommended approach for OceanScale:** Use Random Fourier Modes (RFM) with a von Karman spectrum for real-time turbulence generation. This is O(N_modes) per evaluation point, trivially parallelizable on GPU, and produces divergence-free velocity fields suitable for vehicle dynamics perturbation.

---

## 4. Free Surface Interaction

### 4.1 Buoyancy Simulation for Partially Submerged Vehicles (USV/Amphibious)

**arXiv 2509.03804 (2025):** Presents a convex hull-based approach for real-time buoyancy estimation of AUVs and surface vessels. Extracts mesh geometry from Gazebo and Isaac Sim to compute submerged volume dynamically. This is the most directly relevant recent work for OceanScale: [https://arxiv.org/html/2509.03804v1](https://arxiv.org/html/2509.03804v1)

**Eurographics 2020:** "Realistic Buoyancy Model for Real-Time Applications" -- a GPU-based algorithm for computing buoyancy and fluid-to-solid coupling. Computes all variables for partially submerged objects including:
- Submerged volume via voxelization or slicing
- Center of buoyancy
- Hydrostatic pressure distribution
- Drag forces on submerged portions
Source: [https://diglib.eg.org/bitstream/handle/10.1111/cgf14013/v39i6pp217-231.pdf](https://diglib.eg.org/bitstream/handle/10.1111/cgf14013/v39i6pp217-231.pdf)

**GPU mass properties (UCLA):** Foundational paper on GPU computation of volume, center of mass for arbitrary shapes. Directly applicable to computing submerged volume at each timestep: [http://web.cs.ucla.edu/~dt/papers/pg06/pg06.pdf](http://web.cs.ucla.edu/~dt/papers/pg06/pg06.pdf)

**Convex decomposition toolkits:** For complex non-convex vehicle shapes, decompose into convex parts for efficient submerged volume computation: [https://simulately.wiki/docs/toolkits/ConvexDecomp/](https://simulately.wiki/docs/toolkits/ConvexDecomp/)

### 4.2 Wave-Vehicle Interaction Models

**GPU-based USV state transition models:** Research describing GPU-accelerated 6-DOF dynamics simulation for USV trajectory planning. Computes state transitions under wave forcing: [https://www.researchgate.net/publication/235993484_GPU_based_generation_of_state_transition_models_using_simulations_for_unmanned_surface_vehicle_trajectory_planning](https://www.researchgate.net/publication/235993484_GPU_based_generation_of_state_transition_models_using_simulations_for_unmanned_surface_vehicle_trajectory_planning)

**UMD high-fidelity 6-DOF USV simulator:** Monte Carlo simulation of USV-wave interaction for autonomy development: [https://drum.lib.umd.edu/bitstreams/04566a74-8819-4318-bf78-13f34b5c9ca1/download](https://drum.lib.umd.edu/bitstreams/04566a74-8819-4318-bf78-13f34b5c9ca1/download)

**MDPI Sensors (USV simulator):** Simulation environment modeling forces on USVs including wave interaction, wind, and current: [https://www.mdpi.com/1424-8220/19/5/1068](https://www.mdpi.com/1424-8220/19/5/1068)

**Standard approach for wave-vehicle interaction:**
1. Compute wave heightfield at vehicle position (Gerstner or FFT)
2. Compute relative waterplane area (portion of vehicle at water surface)
3. Compute buoyancy force from submerged volume and center of buoyancy
4. Compute wave excitation forces (Froude-Krylov + diffraction)
5. Compute radiation forces (added mass + radiation damping)
6. Integrate 6-DOF equations of motion

### 4.3 Hydrostatic Pressure Computation on GPU

**AGU paper on GPU-based ocean dynamical core:** Implements hydrostatic Boussinesq equation solver on GPU for ocean modeling: [https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024MS004465](https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/2024MS004465)

**NVIDIA Warp approach:** Write custom hydrostatic pressure computation kernels in Python, JIT-compiled to CUDA. Warp provides automatic differentiation for gradient-based optimization: [https://nvidia.github.io/warp/](https://nvidia.github.io/warp/)

**Practical GPU computation:**
- Hydrostatic pressure at depth z: P(z) = rho * g * z + P_atm
- For vehicles: compute pressure distribution over submerged mesh faces
- GPU mesh voxelization or slice-based integration for submerged volume
- Pressure forces integrated over submerged surface area
- All operations are embarrassingly parallel -- ideal for GPU

---

## 5. Open-Source Ocean Simulation Libraries

### 5.1 Oceananigans.jl (Julia)

The most mature open-source GPU ocean simulator. Developed by CliMA (Climate Modeling Alliance) at Caltech.

| Property | Value |
|----------|-------|
| Language | Pure Julia |
| GPU | KernelAbstractions.jl (CUDA, ROCm, oneAPI) |
| Scale | Global ocean at 488m on 768 A100 GPUs |
| Cost efficiency | ~3x more cost-effective than CPU |
| ML integration | Hands-on ML tutorial available |
| License | MIT |

**References:**
- GitHub: [https://github.com/CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)
- arXiv paper on breakthrough performance: [https://arxiv.org/html/2309.06662v2](https://arxiv.org/html/2309.06662v2)
- ML tutorial video: [https://www.youtube.com/watch?v=YFkbPcjetZ4](https://www.youtube.com/watch?v=YFkbPcjetZ4)
- Documentation: [https://clima.github.io/OceananigansDocumentation/stable/](https://clima.github.io/OceananigansDocumentation/stable/)

**Can it be wrapped for RL training?** Technically yes via Julia's PyCall/PythonCall, but:
- Not designed for real-time interaction
- Ocean-scale simulations are batch (hours to days)
- Could pre-generate current/turbulence fields, then sample during RL training
- Better suited for generating training data than interactive simulation

### 5.2 SINTEF GPU Ocean

Open-source GPU ocean current simulator using shallow-water equations.

| Property | Value |
|----------|-------|
| Language | Python + PyOpenCL |
| GPU | OpenCL (NVIDIA, AMD) |
| Method | Ensemble shallow-water solver |
| Purpose | Short-term current forecast with uncertainty |
| License | Open source |

**References:**
- GitHub: [https://github.com/gpuocean/gpuocean](https://github.com/gpuocean/gpuocean)
- Project page: [https://www.sintef.no/en/software/gpu-ocean/](https://www.sintef.no/en/software/gpu-ocean/)
- Coastal forecasting paper: [https://www.tandfonline.com/doi/full/10.1080/16000870.2021.1876341](https://www.tandfonline.com/doi/full/10.1080/16000870.2021.1876341)

**RL wrapping:** Could be adapted as a current field generator for domain randomization. Runs on GPU. Generates physically consistent 2D current fields with ensemble uncertainty.

### 5.3 Stonefish (C++ / Python)

Open-source marine robotics simulator with ROS interface. Focus on hydrodynamic force computation based on actual body geometry.

| Property | Value |
|----------|-------|
| Language | C++ with Python bindings |
| GPU | CPU-only (no GPU acceleration) |
| Physics | Rigid body + hydrodynamic coefficients |
| ROS | Compatible (ROS Noetic, ROS 2) |
| ML | Recently added sensor support for ML (2025) |
| License | Open source |

**References:**
- GitHub: [https://github.com/patrykcieslak/stonefish](https://github.com/patrykcieslak/stonefish)
- ML support paper (arXiv 2502.11887): [https://arxiv.org/html/2502.11887v1](https://arxiv.org/html/2502.11887v1)
- Documentation: [https://stonefish.readthedocs.io/](https://stonefish.readthedocs.io/)
- ROS Maritime Working Group presentation (March 2024): [https://discourse.openrobotics.org/t/maritime-working-group-meeting-mar-2024-stonefish/36451](https://discourse.openrobotics.org/t/maritime-working-group-meeting-mar-2024-stonefish/36451)

### 5.4 HoloOcean (Unreal Engine)

Open-source underwater simulator built on Unreal Engine with Gym-style RL interface.

| Property | Value |
|----------|-------|
| Engine | Unreal Engine 4/5 |
| RL | Gym-style Python API |
| ROS | ROS 2 support (v2.0) |
| Agents | Multi-agent support |
| License | Open source |

**References:**
- ICRA paper: [https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf](https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf)
- Documentation: [https://byu-holoocean.github.io/holoocean-docs/](https://byu-holoocean.github.io/holoocean-docs/)
- RL AUV tracking example: [https://github.com/Ice-mao/RL_AUV_tracking](https://github.com/Ice-mao/RL_AUV_tracking)

### 5.5 MarineGym (Isaac Sim)

GPU-accelerated RL platform for underwater robotics, achieving 250,000 FPS rollout on a single GPU.

| Property | Value |
|----------|-------|
| Base | NVIDIA Isaac Sim |
| GPU | Full GPU acceleration |
| FPS | ~250,000 FPS (rollout) |
| RL | Gymnasium-compatible |
| Venue | IROS 2025 |

**References:**
- arXiv paper: [https://arxiv.org/html/2503.09203v1](https://arxiv.org/html/2503.09203v1)
- GitHub: [https://github.com/Marine-RL/MarineGym](https://github.com/Marine-RL/MarineGym)
- Website: [https://marine-gym.com/](https://marine-gym.com/)

### 5.6 Comprehensive Review (2024)

A 2024 arXiv review compares five open-source underwater simulators (Stonefish, HoloOcean, DAVE, and others): [https://arxiv.org/html/2504.06245v1](https://arxiv.org/html/2504.06245v1)

### 5.7 Can These Be Wrapped for RL Training?

| Library | RL Wrapping Feasibility | Notes |
|---------|------------------------|-------|
| Oceananigans.jl | Pre-generate data | Too slow for interactive; use for offline field generation |
| GPU Ocean | Medium | Ensemble current fields on GPU; could feed into Isaac Sim |
| Stonefish | Direct | Python bindings exist; CPU-only limits parallelism |
| HoloOcean | Direct | Gym-style API; Unreal Engine limits GPU parallelism |
| MarineGym | Direct | Best option; 250K FPS on GPU; Gymnasium API |
| OceanSim | Perception only | Focus on rendering, not dynamics |

---

## 6. NVIDIA-Specific Ocean/Water Tools

### 6.1 Omniverse Create / USD Composer Water Extensions

No official water/ocean extension exists. Community approaches:

- **Ocean scene in USD Composer** (community tutorial): [https://edgedsign.com/2023/03/rendering-sea-scene-in-omniverse-create/](https://edgedsign.com/2023/03/rendering-sea-scene-in-omniverse-create/)
- **Fluid scenes tutorial** (Physics Extension, Kit104): [https://www.youtube.com/watch?v=DHKonOzJho4](https://www.youtube.com/watch?v=DHKonOzJho4)
- **Developer forum discussion on oceans:** [https://forums.developer.nvidia.com/t/oceans-in-omniverse/280926](https://forums.developer.nvidia.com/t/oceans-in-omniverse/280926)
- **NVIDIA Omniverse weather/climate visualization:** [https://www.sparkblue.org/system/files/2021-04/NVIDIA_Omniverse_Messmer.pdf](https://www.sparkblue.org/system/files/2021-04/NVIDIA_Omniverse_Messmer.pdf)

Users can write custom extensions using the Omniverse Kit SDK, but no pre-built ocean solution exists.

### 6.2 Isaac Sim Water/Ocean Examples

**Isaac Sim does not include built-in water/ocean examples.** The developer forum confirms this: [https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374](https://forums.developer.nvidia.com/t/how-to-build-an-ocean-environment-for-marine-robots-in-issac-sim/280374)

Third-party examples:
- **isaac_underwater** (community): Basic water and underwater physics examples: [https://github.com/leonlime/isaac_underwater](https://github.com/leonlime/isaac_underwater)
- **OceanSim** (U. Michigan, IROS 2025): Full underwater perception simulator on Isaac Sim: [https://github.com/umfieldrobotics/OceanSim](https://github.com/umfieldrobotics/OceanSim)
- **MarineGym** (IROS 2025): GPU hydrodynamic plugin for Isaac Sim: [https://github.com/Marine-RL/MarineGym](https://github.com/Marine-RL/MarineGym)
- **NVIDIA on-demand session on AUV navigation with Isaac Sim:** [https://nvidia.com/en-us/on-demand/session/aisummitdc24-sdc1072/](https://nvidia.com/en-us/on-demand/session/aisummitdc24-sdc1072/)

### 6.3 GTC Presentations on Ocean Simulation

| Session | Year | Topic | Link |
|---------|------|-------|------|
| GTC25-S73656 | 2025 | Digital twins for ocean exploration with Omniverse | [https://nvidia.com/en-us/on-demand/session/gtc25-s73656/](https://nvidia.com/en-us/on-demand/session/gtc25-s73656/) |
| AI Summit DC24 | 2024 | AUV navigation digital twins with Isaac Sim | [https://nvidia.com/en-us/on-demand/session/aisummitdc24-sdc1072/](https://nvidia.com/en-us/on-demand/session/aisummitdc24-sdc1072/) |
| Fluid Simulations with Ansys | 2024 | Real-time fluid simulation workflows | [https://www.youtube.com/watch?v=8AtTDRcrUsc](https://www.youtube.com/watch?v=8AtTDRcrUsc) |

### 6.4 NVIDIA Warp for Custom Ocean Simulation

NVIDIA Warp is a Python framework for writing GPU-accelerated simulation code with automatic differentiation. It is the most promising tool for building custom ocean dynamics in Python.

- GitHub: [https://github.com/nvidia/warp](https://github.com/nvidia/warp)
- Documentation: [https://nvidia.github.io/warp/](https://nvidia.github.io/warp/)
- Differentiable simulation tutorial: [https://www.youtube.com/watch?v=jDOekvNddFo](https://www.youtube.com/watch?v=jDOekvNddFo)
- NERSC workshop (May 2025): [https://www.nersc.gov/news-and-events/calendar-of-events/nvidia-warp-python-may2025](https://www.nersc.gov/news-and-events/calendar-of-events/nvidia-warp-python-may2025)
- arXiv paper on locality-aware AD with Warp: [https://arxiv.org/html/2509.00406v1](https://arxiv.org/html/2509.00406v1)

**Relevance to OceanScale:** Warp can implement:
- FFT-based ocean wave heightfield generation (cuFFT integration)
- Buoyancy force computation (parallel mesh operations)
- Current field evaluation (interpolation kernels)
- Turbulence generation (Random Fourier Modes)
- All with automatic differentiation for gradient-based optimization

### 6.5 NVIDIA PhysicsNeMo (formerly Modulus)

NVIDIA's physics-ML framework for building neural surrogates of PDE systems.

- GitHub: [https://github.com/NVIDIA/physicsnemo](https://github.com/NVIDIA/physicsnemo)
- Developer portal: [https://developer.nvidia.com/physicsnemo](https://developer.nvidia.com/physicsnemo)
- Turbulence super-resolution example: [https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/intermediate/turbulence_super_resolution.html](https://docs.nvidia.com/physicsnemo/25.08/physicsnemo-sym/user_guide/intermediate/turbulence_super_resolution.html)

**Relevance:** Could train neural operators (FNO, DeepONet) as fast surrogates for ocean current/wave fields. Once trained, inference is orders of magnitude faster than running the full simulation. Useful for generating diverse training environments at scale.

---

## 7. Sea State Modeling

### 7.1 WMO Sea State Codes

The WMO sea state code (Code Table 3700) classifies sea surface conditions from 0 to 9 based on significant wave height (Hs):

| Code | Description | Hs (m) |
|------|-------------|--------|
| 0 | Calm (glassy) | 0 |
| 1 | Calm (rippled) | 0 - 0.1 |
| 2 | Smooth (wavelets) | 0.1 - 0.5 |
| 3 | Slight | 0.5 - 1.25 |
| 4 | Moderate | 1.25 - 2.5 |
| 5 | Rough | 2.5 - 4.0 |
| 6 | Very rough | 4.0 - 6.0 |
| 7 | High | 6.0 - 9.0 |
| 8 | Very high | 9.0 - 14.0 |
| 9 | Phenomenal | > 14.0 |

Source: NOAA/NODC WMO Code Table 3700: [https://www.nodc.noaa.gov/gtspp/document/codetbls/wmocodes/table3700.html](https://www.nodc.noaa.gov/gtspp/document/codetbls/wmocodes/table3700.html)

Wikipedia overview: [https://en.wikipedia.org/wiki/Sea_state](https://en.wikipedia.org/wiki/Sea_state)

### 7.2 Douglas Sea Scale

The Douglas Sea Scale is the historical predecessor of the WMO sea state code. The WMO code largely adopts the Douglas "wind sea" definition. The Douglas scale separately categorizes wind sea and swell.

Reference: [https://www.researchgate.net/figure/Sea-States-Codes-based-on-Douglas-Sea-Scale_tbl1_355226332](https://www.researchgate.net/figure/Sea-States-Codes-based-on-Douglas-Sea-Scale_tbl1_355226332)

StormGeo argues Douglas Sea State 3 should be replaced by Significant Wave Height in maritime contracts: [https://stormgeo.com/insights/why-douglas-sea-state-3-should-be-eliminated-from-good-weather-clauses](https://stormgeo.com/insights/why-douglas-sea-state-3-should-be-eliminated-from-good-weather-clauses)

### 7.3 Parameterizing Sea State for Sim2Real Domain Randomization

**Proposed parameterization for OceanScale:**

Each RL training episode samples a sea state from the following parameter space:

```
sea_state = {
    wmo_code:        int    [0-9],        # WMO sea state code
    Hs:              float  [0, 14+] m,   # Significant wave height
    Tp:              float  [1, 20] s,     # Peak period
    spectrum_type:   enum   [PM, JONSWAP], # Wave spectrum model
    gamma:           float  [1, 7],        # JONSWAP peak enhancement
    wind_speed:      float  [0, 30] m/s,   # Wind speed at 10m
    wind_dir:        float  [0, 360] deg,  # Wind direction
    current_speed:   float  [0, 3] m/s,    # Depth-averaged current
    current_dir:     float  [0, 360] deg,  # Current direction
    turbulence_intensity: float [0, 0.5],  # TI = u'/U_mean
    integral_length: float  [0.1, 10] m,   # Turbulence integral scale
}
```

**Domain randomization strategy:**
1. Sample Hs from WMO distribution weighted by operational relevance (codes 2-6 for most AUV/USV operations)
2. Derive Tp from Hs using Tp ~ 3.86 * sqrt(Hs) (PM relationship)
3. Choose spectrum type: PM for developed seas, JONSWAP for developing seas
4. Add current field (procedural or from GPU Ocean ensembles)
5. Add turbulence perturbation via Random Fourier Modes with sampled TI and L
6. Evaluate wave heightfield on GPU via FFT (Tessendorf method)

**References for sea state parameterization:**
- Ifremer technical report on sea state conditions for marine structures: [https://archimer.ifremer.fr/doc/00324/43514/45291.pdf](https://archimer.ifremer.fr/doc/00324/43514/45291.pdf)
- EasyUUV Sim2Real framework for UUV control: [https://arxiv.org/html/2510.22126v1](https://arxiv.org/html/2510.22126v1)
- MarineGym DDR (Data-informed Domain Randomization): [https://www.researchgate.net/figure/The-Data-informed-Domain-Randomization-DDR-adaptation-mechanism_fig5_367536302](https://www.researchgate.net/figure/The-Data-informed-Domain-Randomization-DDR-adaptation-mechanism_fig5_367536302)
- ASV collision avoidance with domain randomization: [https://www.mdpi.com/2077-1312/13/9/1727](https://www.mdpi.com/2077-1312/13/9/1727)
- Lilian Weng's domain randomization overview: [https://lilianweng.github.io/posts/2019-05-05-domain-randomization/](https://lilianweng.github.io/posts/2019-05-05-domain-randomization/)

---

## Appendix: Recommended Technology Stack for OceanScale

Based on this research, the recommended approach for OceanScale's GPU-native underwater simulator:

| Component | Recommended Method | GPU Tool |
|-----------|-------------------|----------|
| Ocean waves | Tessendorf FFT + JONSWAP/PM spectra | NVIDIA Warp (cuFFT) |
| Ocean currents | Procedural + GPU Ocean ensembles | NVIDIA Warp |
| Turbulence | Random Fourier Modes (von Karman spectrum) | NVIDIA Warp |
| Buoyancy | Convex hull-based submerged volume | NVIDIA Warp |
| Wave-vehicle interaction | 6-DOF dynamics with Froude-Krylov forces | NVIDIA Warp / Isaac Sim |
| Domain randomization | WMO sea state parameterized sampling | Python orchestration |
| RL training environment | MarineGym pattern (Isaac Sim + GPU hydro plugin) | Isaac Sim + Warp |
| Neural surrogates (future) | FNO via PhysicsNeMo | NVIDIA PhysicsNeMo |

---

## Research Gaps Identified

1. **No GPU ocean simulator with integrated wave + current + turbulence for RL training exists.** MarineGym is closest but does not publish full ocean environment details.
2. **Turbulence is absent from all underwater robotics simulators.** No tool provides high-fidelity turbulent velocity fields for vehicle dynamics.
3. **No standard parameterization links WMO sea state codes to RL domain randomization.** The parameterization in Section 7.3 is a proposed novel contribution.
4. **NVIDIA provides no first-party ocean/water tools for Isaac Sim or Omniverse.** The ecosystem relies entirely on third-party and community efforts (OceanSim, MarineGym).
5. **Langmuir circulation is not modeled in any underwater simulator** despite being critical for near-surface vehicle dynamics.

---

*Research completed 2026-05-20. All URLs verified via web search.*
