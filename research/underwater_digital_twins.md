# Underwater Digital Twins: Neural Rendering, Reconstruction & Sim2Real

> Research survey for OceanScale — underwater digital twin pipeline for sim2real.
> Date: 2026-05-20. All URLs verified via web search.

---

## Table of Contents

1. [3D Gaussian Splatting for Underwater](#1-3d-gaussian-splatting-3dgs-for-underwater)
2. [NeRF for Underwater](#2-nerf-for-underwater)
3. [NVIDIA NuRec / Reconstruction](#3-nvidia-nurec--reconstruction)
4. [NVIDIA Cosmos](#4-nvidia-cosmos)
5. [Synthetic Underwater Data Generation](#5-synthetic-underwater-data-generation)
6. [Sonar 3D Reconstruction](#6-sonar-3d-reconstruction)
7. [Underwater SLAM for Digital Twins](#7-underwater-slam-for-digital-twins)
8. [Recommended Pipeline for OceanScale](#8-recommended-pipeline-for-oceanscale)

---

## 1. 3D Gaussian Splatting (3DGS) for Underwater

### 1.1 Key Papers

Underwater 3DGS is a rapidly growing sub-field (2024-2025), with at least 8 papers specifically addressing aquatic challenges:

| Paper | Venue | Key Innovation | URL |
|-------|-------|----------------|-----|
| **Water-Adapted 3DGS** | Frontiers in Marine Science 2025 | Complexity-adaptive point distribution + depth-adaptive multi-scale radius rendering | https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1573612/full |
| **UW-GS** | WACV 2025 | Distractor-aware (marine snow, floating particles) | https://pui-nantheera.github.io/research/CIC/MyUnderwaterWorld.html |
| **SeaSplat** | IEEE 2024 | Physically grounded image formation model for water medium | Cited in https://bmva-archive.org.uk/bmvc/2025/assets/papers/Paper_367/paper.pdf |
| **MarineSTD-GS** | ACM 2025 | Spatiotemporal degradation awareness (time-varying water effects) | https://dl.acm.org/doi/10.1145/3746027.3754888 |
| **UW-3DGS** | OpenReview 2024-25 | General robust underwater adaptation | https://openreview.net/revisions?id=K7z0eZGFTh |
| **WaterSplatting** | 3DV 2025 | Fast explicit geometry reconstruction in water | https://colab.ws/articles/10.1109%2F3dv66043.2025.00094 |
| **Plenodium** | NeurIPS 2025 | Plenoptic representation + 3DGS for underwater | https://neurips.cc/virtual/2025/poster/117833 |
| **SWAGS** | arXiv 2025 | Semantic-guided augmentation for water scenes | https://arxiv.org/html/2509.00800v1 |

### 1.2 Can 3DGS Handle Water Medium?

Yes, but requires specialized modifications. Standard 3DGS assumes clear-air radiance transfer. Underwater environments introduce:
- **Light attenuation** (exponential with distance, wavelength-dependent)
- **Backscattering** (veiling luminance from suspended particles)
- **Color distortion** (red attenuates first, then green, then blue)
- **Marine snow / floating particles** (distractors that break multi-view consistency)

SeaSplat and SeaThru-NeRF both use the physically-based **SeaThru image formation model** (Akkaynak & Treibitz, 2019) to decompose the observed signal into direct transmission + backscatter. Water-adapted 3DGS extends this with adaptive point density and rendering radius.

### 1.3 Real-Time Performance

3DGS inherently supports real-time rendering (>100 FPS for typical scenes on RTX-class GPUs). Underwater variants maintain this advantage because:
- The Gaussian representation is unchanged; only the rendering equation is modified
- SeaSplat adds ~15-20% overhead for medium-aware rendering (unverified — needs benchmarking)
- WaterSplatting (3DV 2025) specifically targets fast reconstruction as a NeRF alternative

For simulation use in Isaac Sim, the pipeline would be: reconstruct underwater scene with underwater-3DGS, export to USDZ, load as environment asset.

### 1.4 Integration with Isaac Sim / Omniverse

3DGS-to-Isaac-Sim is now a mature pipeline (see Section 3). The key bridge is **USDZ export** from Gaussian Splat outputs. NVIDIA's 3DGRUT supports direct USDZ output. However, the underwater-specific 3DGS methods have NOT been integrated into NuRec yet — they are academic implementations. A custom pipeline would be needed:
1. Capture underwater imagery (ROV/AUV multi-view)
2. Reconstruct with UW-GS or SeaSplat
3. Export PLY → convert to USDZ via NVIDIA tools
4. Load into Isaac Sim with water volume overlay

---

## 2. NeRF for Underwater

### 2.1 SeaThru-NeRF

The foundational paper for underwater NeRF.

- **Title**: SeaThru-NeRF: Neural Radiance Fields in Scattering Media
- **Venue**: CVPR 2023
- **Authors**: Levy, Peleg, Pearl, Rosenbaum, Akkaynak, Treibitz, Korman
- **arXiv**: https://arxiv.org/abs/2304.07743
- **PDF**: https://openaccess.thecvf.com/content/CVPR2023/papers/Levy_SeaThru-NeRF_Neural_Radiance_Fields_in_Scattering_Media_CVPR_2023_paper.pdf
- **Code**: https://github.com/deborahLevy130/seathru_NeRF
- **Nerfstudio integration**: https://docs.nerf.studio/nerfology/methods/seathru_nerf.html

**Key contribution**: Replaces the standard NeRF volume rendering equation with the SeaThru image formation model, jointly estimating:
- Scene radiance (true color, "water-removed")
- Medium (water) parameters: backscatter, attenuation coefficients
- 3D geometry (density field)

This means SeaThru-NeRF outputs both a corrected scene (as if water were removed) AND the water properties at each point — critical for building physics-accurate underwater digital twins.

### 2.2 Other Underwater NeRF Papers

| Paper | Year | Key Focus | URL |
|-------|------|-----------|-----|
| **AquaNeRF** | 2024-25 | NeRF with distractor removal; outperforms Nerfacto by ~7.5% PSNR and SeaThru-NeRF by ~6.2% | https://arxiv.org/abs/2502.16351 |
| **ISPRS 2025 Comparative Study** | 2025 | Benchmarks NeRF vs SeaThru-NeRF vs 3DGS for underwater 3D reconstruction | https://isprs-archives.copernicus.org/articles/XLVIII-M-9-2025/1475/2025/isprs-archives-XLVIII-M-9-2025-1475-2025.pdf |
| **NeRF for the Real World Survey** | Jan 2025 | Broad survey covering scattering media, challenging environments | https://arxiv.org/html/2501.13104v1 |

### 2.3 Can NeRF Generate Training Data for Underwater RL?

Yes, with caveats. Two approaches are documented:

1. **NeRF-as-simulator**: Tube-NeRF (https://arxiv.org/html/2311.14153v2) creates a NeRF-based simulator for RL policy training from RGB images. Policies trained in NeRF simulators transfer to real robots.
2. **NeRF for sim2real transfer**: Sim2Real Transfer of Vision-guided Bipedal Motion Skills (https://ieeexplore.ieee.org/iel7/10160211/10160212/10161544.pdf) combines NeRF + end-to-end deep RL.

For underwater specifically, the pipeline would be:
1. Capture real underwater scene with multi-view camera rig
2. Reconstruct with SeaThru-NeRF → get water-removed scene + water parameters
3. Export geometry to mesh/point cloud → USD scene
4. Use reconstructed water parameters to configure Isaac Sim's water volume
5. Train RL policies in the reconstructed scene with physics-accurate water

**Limitation**: NeRF rendering is slow (~1-5 FPS) compared to 3DGS (~100+ FPS). For RL training with millions of steps, convert NeRF output to mesh/3DGS first.

Curated resource: https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics

---

## 3. NVIDIA NuRec / Reconstruction

### 3.1 What is NuRec?

**NVIDIA Omniverse NuRec** is a neural reconstruction library that brings real-world scenes into simulation. It takes multi-view images, LiDAR, or sensor data and produces photorealistic 3D scenes loadable directly into Omniverse/Isaac Sim.

- **Developer blog**: https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/
- **Smartphone workflow tutorial**: https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/
- **Dataset**: https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec
- **Available in**: Isaac Sim 5.0 (open-source on GitHub)

### 3.2 Underlying Technology

NuRec uses two key algorithms:

| Algorithm | Role | URL |
|-----------|------|-----|
| **3DGUT** (3D Gaussian with Unscented Transforms) | Handles distorted cameras (fisheye, wide-angle) and secondary rays | https://arxiv.org/html/2412.12507v2 |
| **3DGRUT** (3D Gaussian Ray Tracing) | Ray-traces volumetric Gaussian particles instead of rasterization; supports refractions | https://github.com/nv-tlabs/3dgrut |

The full pipeline: COLMAP (sparse reconstruction) → 3DGUT/3DGRUT (dense Gaussian model) → USDZ export → Omniverse/Isaac Sim.

**Critical detail** (from NVIDIA forum https://forums.developer.nvidia.com/t/questions-about-compatibility-between-3dgrut-reconstruction-data-and-nurec-format-and-understanding-the-nurec-checkpoint-structure/350317): Standard 3DGRUT outputs cannot be directly loaded by NuRec without conversion. The data structures differ.

### 3.3 Can NuRec Be Used for Underwater?

NuRec is designed for **in-air** scenes. It does NOT natively model water medium (attenuation, scattering). However:

- **Approach 1 (pre-process)**: Use SeaThru-NeRF to remove water effects from images first, then feed "water-removed" images to NuRec. This would reconstruct the clean scene geometry.
- **Approach 2 (post-process)**: Reconstruct with NuRec normally, then add water volume and underwater post-processing in Isaac Sim.
- **Approach 3 (custom)**: Extend NuRec's rendering equation with SeaThru image formation model — requires modifying the 3DGUT codebase.

### 3.4 Gaussian Splatting in Omniverse

The ecosystem has matured significantly:

- **USDZ export**: 3DGRUT now exports to USDZ for direct Omniverse loading (https://radiancefields.com/nvidia-adds-usdz-support-to-3dgrut-and-beta-for-omniverse-and-isaac-sim)
- **RTX Realtime 2.0**: Gaussian splats render with high quality in Omniverse USD Composer (https://forums.developer.nvidia.com/t/nurec-rendering-with-omnidirectionalstereo/347751)
- **Multi-GPU**: Large-scale scene reconstruction supported (GTC 2026 session: https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81757/)
- **4DGS**: Evercoast integrates dynamic/captured performance Gaussian splats into Omniverse (https://insights.evercoast.ai/p/integrating-4d-gaussian-splats-into)
- **Tutorial**: Build photorealistic Isaac Sim environments with 3DGS (https://www.youtube.com/watch?v=DYeutmDZOyI)

---

## 4. NVIDIA Cosmos

### 4.1 What is Cosmos?

**NVIDIA Cosmos** is a platform of generative World Foundation Models (WFMs) for Physical AI. It generates physics-aware synthetic video from text, image, or video inputs.

- **Official page**: https://www.nvidia.com/en-us/ai/cosmos/
- **Research paper**: https://arxiv.org/html/2501.03575v1
- **Key capabilities**:
  - Generate up to **30-second predictive video worlds**
  - Three model variants: **Predict**, **Transfer**, **Reason**
  - Cosmos Predict 2.5 is the latest version
  - Open and customizable platform

### 4.2 Ocean / Marine Scene Generation

The top-level search results do NOT show explicit underwater/ocean scene generation examples from NVIDIA. However:

- Cosmos is prompt-driven and general-purpose. Given text prompts describing underwater scenes, it can theoretically generate underwater video.
- **Cosmos-Drive-Dreams** (https://arxiv.org/abs/2506.09042) demonstrates a specialized pipeline for autonomous driving scenarios. A similar "Cosmos-Dive-Dreams" pipeline for underwater is conceivable but does not exist as a published product.
- **Integration with Omniverse**: Cosmos generates photoreal video from controlled 3D scenarios built in Omniverse (https://forums.developer.nvidia.com/t/how-to-generate-photoreal-synthetic-data-from-omniverse-environments-with-cosmos/319808). This means: build an underwater scene in Isaac Sim → Cosmos generates diverse training videos from it.

### 4.3 Synthetic Training Data Generation

- **Official guide**: https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/
- **Scaling blog**: https://blogs.nvidia.com/blog/scaling-physical-ai-omniverse/
- **AWS deployment**: https://aws.amazon.com/blogs/hpc/running-nvidia-cosmos-world-foundation-models-on-aws/
- **Spheron GPU cloud**: https://www.spheron.network/blog/deploy-nvidia-cosmos-gpu-cloud-synthetic-data/

The workflow for OceanScale would be:
1. Build underwater scene in Isaac Sim (using USD assets, water volume, lighting)
2. Use Cosmos to generate diverse camera trajectories, lighting variations, weather conditions
3. Output: massive synthetic underwater training dataset for perception models

**Gap**: No published examples of Cosmos generating underwater-specific content. Would need experimentation to validate quality.

---

## 5. Synthetic Underwater Data Generation

### 5.1 Simulator-Based Generation

**OceanSim** (https://umfieldrobotics.github.io/OceanSim/ and https://arxiv.org/html/2503.01074v1) is a GPU-accelerated underwater robot perception simulator that provides:
- Real-time imaging sonar rendering
- Fast synthetic data generation
- GPU-accelerated underwater optical effects (caustics, attenuation, scattering)
- Designed for training underwater robotics algorithms

This is the most directly relevant simulator for OceanScale's use case. (unverified: whether it integrates with Isaac Sim or runs standalone)

**NVIDIA Isaac Sim** + custom water assets can also generate synthetic underwater data, with the advantage of direct ROS 2 integration, domain randomization, and the full Omniverse rendering pipeline.

### 5.2 Domain Adaptation (Synthetic → Real)

The synthetic-to-real gap is a major research topic for underwater:

| Paper / Method | Approach | URL |
|----------------|----------|-----|
| **SubmergeStyleGAN** | GAN models underwater optical phenomena (attenuation, backscattering, absorption) using depth maps; generates paired synthetic-real training data | https://ieeexplore.ieee.org/document/10410962/ |
| **Two-step Domain Adaptation** | Does NOT require synthetic training data; two-step DA for underwater enhancement | https://www.sciencedirect.com/science/article/pii/S0031320321005045 |
| **TUDA** | Two-phase Underwater Domain Adaptation network minimizing synthetic-real gap | https://www.scribd.com/document/726166548/Domain-Adaptation-for-Underwater-Image-Enhancement |
| **Content-Style Separation** | Separates content and style features for underwater DA | https://www.researchgate.net/publication/362934030 |
| **Contrastive Learning + GAN (2025)** | Latest approach to generating realistic underwater images | https://arxiv.org/html/2505.14296v1 |
| **Marine Species Detection (2025)** | Practical application: training marine species detectors with synthetic data + DA | https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1581778/full |

### 5.3 Benchmarks for Synthetic Underwater Data Quality

| Dataset | Type | Size | Purpose | URL |
|---------|------|------|---------|-----|
| **UIEB** | Real-world | 950 images | Image enhancement benchmark (890 with reference) | https://li-chongyi.github.io/proj_benchmark.html |
| **OD-UIUA** | Real-world | 1,200 images | Object detection utility assessment | https://www.mdpi.com/2072-4292/17/11/1906 |
| **CSUID** | Synthetic | Varied | Color cast, blur, low-light, contrast combinations | https://pmc.ncbi.nlm.nih.gov/articles/PMC11327533/ |
| **SUID** | Synthetic | Scene pairs | Full-reference enhancement evaluation | Cited in https://digitalcommons.mtu.edu/cgi/viewcontent/?article=36165&context=michigantech-p |
| **UID2021** | Real-world | Multiple | No-reference quality assessment | https://arxiv.org/pdf/2204.08813 |
| **Underwater-ImageNet** | Synthetic | Train/test split | Domain adaptation benchmark | https://github.com/fordevoted/UIESS |

**Key finding from IEEE benchmark paper** (https://ieeexplore.ieee.org/iel7/6287639/8948470/09130676.pdf): Networks trained on domain-adapted synthetic data produce better results than those trained on non-adapted synthetic data. Domain adaptation is essential, not optional.

---

## 6. Sonar 3D Reconstruction

### 6.1 Sonar → 3D Scene Models

Yes, sonar data can be used to build 3D scene models. Multiple approaches exist:

| Approach | Sonar Type | Output | URL |
|----------|-----------|--------|-----|
| **Sonar + Vision Fusion** | Multi-sensor | Fused 3D reconstruction via plane-based approach | https://arxiv.org/html/2511.00392v1 |
| **Acoustic 3D Reconstruction** | Imaging sonar | 3D models of underwater infrastructure | https://elib.dlr.de/217488/1/MARESEC_2025_paper_33%2520%25284%2529.pdf |
| **Down-Looking Sonar** | Down-looking | 3D point clouds of underwater targets | https://www.sciencedirect.com/science/article/abs/pii/S0263224126005269 |
| **BlueView Imaging Sonar** | 2D imaging sonar | 3D maps from 2D sonar data | https://www.scribd.com/document/689661977/Underwater-3D-reconstruction-using-BlueView-imaging-sonar |
| **Side-Scan Sonar** | Side-scan | 3D reconstruction via normal-depth fusion | https://link.springer.com/article/10.1007/s44295-023-00013-0 |
| **Neural Bathymetry** | Bathymetric sonar | Neural network represents seafloor height from coordinates | https://www.researchgate.net/figure/Overview-pipeline-of-the-proposed-method-The-bathymetry-is-represented-by-a-neural_fig1_362315958 |

Comprehensive overview: https://www.mdpi.com/2077-1312/11/5/949

### 6.2 Bathymetric Mapping → USD Scene Pipeline

No single end-to-end tool converts bathymetric sonar data directly to OpenUSD. The pipeline would be custom-built:

1. **Sonar → Point Cloud**: Use sonar processing software (e.g., Qimera, PLOTREX) to generate georeferenced 3D point clouds
2. **Point Cloud → Mesh**: Poisson surface reconstruction or neural SDF fitting (e.g., Neural Bathymetry approach)
3. **Mesh → USD**: Use NVIDIA Omniverse's point cloud extension (https://docs.omniverse.nvidia.com/extensions/latest/ext_pointclouds.html) or mesh-to-USD converters
4. **USD → Isaac Sim**: Load as terrain/scene asset, add water volume, configure rendering

Community discussion on sonar-to-point-cloud: https://discuss.bluerobotics.com/t/sonar-out-conversion-to-3d-point-cloud/6130

### 6.3 GPU-Accelerated Sonar Processing

| Tool / Paper | Function | URL |
|-------------|----------|-----|
| **ASASIN** | GPU-accelerated time-domain backprojection for SAS image reconstruction | https://arxiv.org/pdf/2101.05888 |
| **GPU-based real-time SAS** | Hybrid CPU/GPU SAS processing on-board AUVs | https://www.semanticscholar.org/paper/GPU-based-real-time-synthetic-aperture-sonar-Baralli-Couillard/801afe883a68a4cf20f84ce2015169c6816e4d4a |
| **NVIDIA OptiX ray-tracing** | GPU ray-tracing for underwater sound propagation simulation | https://www.diva-portal.org/smash/get/diva2:1352170/FULLTEXT01.pdf |
| **OceanSim** | Real-time imaging sonar rendering on GPU | https://umfieldrobotics.github.io/OceanSim/ |
| **Kraken Robotics InSAS** | Real-time interferometric SAS with GPU, ultra-low power | https://www.krakenrobotics.com/wp-content/uploads/2025/07/Real-Time-InSAS-Processing-with-Ultra-Low-Power-Consumption.pdf |
| **GPU Monte Carlo** | GPU-accelerated Monte Carlo for underwater single-photon detection | https://cocc.xmu.edu.cn/GPU-Accelerated.pdf |

---

## 7. Underwater SLAM for Digital Twins

### 7.1 State of the Art

| Paper / System | Method | Key Feature | URL |
|----------------|--------|-------------|-----|
| **Deep Learning + Underwater SLAM survey** | Multi-sensor DL fusion | Challenges and strategies for underwater SLAM with deep learning | https://pmc.ncbi.nlm.nih.gov/articles/PMC12157327/ |
| **Visual SLAM overview** | Optical vision | Extensive overview of visual SLAM for underwater vehicles | https://www.sciencedirect.com/science/article/abs/pii/S002980182402612X |
| **Visual SLAM for raised sediments** | Adaptive visual | Classifies underwater images to adapt SLAM pipeline | https://www.mdpi.com/2077-1312/12/5/716 |
| **Low-cost visual-inertial SLAM** | Visual-inertial | Scalable for widespread ROV deployment | https://agu.confex.com/agu/OSM24/meetingapp.cgi/Paper/1490139 |
| **Pose-graph SLAM** | Pose graph | Vehicle navigation with image-based pose estimation | https://onlinelibrary.wiley.com/doi/10.1002/rob.22375 |
| **AquaNav** | Vision-based | Real-time multi-robot underwater mapping | https://rlab.cs.dartmouth.edu/publications/xanthidis2022isrr.pdf |
| **ETH Zurich survey** | Comprehensive | Quantitative comparison of visual navigation methods for UUVs | https://www.research-collection.ethz.ch/bitstreams/8dc727d8-306f-47fb-be53-eb75b9f5ba8d/download |

### 7.2 SLAM Output → OpenUSD Scene

No off-the-shelf converter exists. Custom pipeline needed:

1. **SLAM output**: Typically a pose graph + sparse/semi-dense point cloud (ORB-SLAM, COLMAP) or dense mesh (ElasticFusion, KinectFusion variants)
2. **Dense reconstruction**: Feed SLAM poses into SeaThru-NeRF or underwater-3DGS for photorealistic dense reconstruction
3. **Export to USD**: Convert mesh/point cloud to USD format
   - NVIDIA's point cloud extension for Omniverse: https://docs.omniverse.nvidia.com/extensions/latest/ext_pointclouds.html
   - 3DGRUT USDZ export for Gaussian representations
   - Mesh → USD via USD Python SDK or Omniverse Connector
4. **Scene assembly**: Add water volume, configure underwater rendering parameters, add dynamic elements (currents, particles)

### 7.3 Real-Time Mapping → Simulation Loop

This is the "holy grail" for underwater digital twins but remains largely unrealized:

1. **Live SLAM**: AUV/ROV runs visual-inertial SLAM in real-time, building a progressively denser map
2. **Incremental update**: New map sections streamed to Isaac Sim as USD scene deltas
3. **Sim-in-the-loop**: Robot policy runs in Isaac Sim against the latest map, plans next action
4. **Action → observation**: Robot executes action, new observations update the map

Key enablers:
- **Isaac Sim ROS 2 bridge**: For real-time communication between SLAM and simulation
- **OpenUSD layer-based composition**: Supports incremental scene updates without reloading
- **OceanSim**: GPU-accelerated sonar rendering for real-time sensor simulation

NVIDIA documentation on immersive software-in-the-loop testing with OpenUSD and Isaac Sim: https://docs.nvidia.com/learning/physical-ai/going-further-with-robotics/latest/digital-twin-robotics/index.html

---

## 8. Recommended Pipeline for OceanScale

Based on the research, here is the recommended architecture for building underwater digital twins:

### Phase 1: Data Capture

- **Visual**: Multi-view underwater camera rig on ROV (GoPros or industrial cameras)
- **Sonar**: Imaging sonar (BlueView or similar) for geometric ground truth in low-visibility
- **Navigation**: Visual-inertial odometry for camera pose estimation

### Phase 2: Reconstruction

```
Camera images + poses
    |
    v
SeaThru-NeRF or UW-GS  ──→  "Water-removed" 3D scene + water parameters
    |                              |
    v                              v
Export to mesh/USDZ         Water properties (attenuation, scattering)
    |                              |
    v                              v
Isaac Sim USD scene    +    Configured water volume
```

### Phase 3: Synthetic Data Generation

```
Isaac Sim underwater scene
    |
    v
Cosmos WFM  ──→  Diverse synthetic underwater training videos
    |
    v
SubmergeStyleGAN or TUDA  ──→  Domain-adapted for real-world transfer
```

### Phase 4: Sim2Real Training

```
Synthetic underwater data
    |
    v
Train perception / RL policies
    |
    v
Deploy to real AUV/ROV
    |
    v
Real data → update digital twin (SLAM loop)
```

### Key Technologies to Track

| Technology | Maturity | Relevance |
|-----------|----------|-----------|
| SeaThru-NeRF | Research (CVPR 2023, code available) | HIGH — water-aware reconstruction |
| UW-GS / WaterSplatting | Research (2025) | HIGH — fast underwater 3DGS |
| NVIDIA NuRec | Production (Isaac Sim 5.0) | MEDIUM — needs underwater extension |
| NVIDIA Cosmos | Production (Predict 2.5) | MEDIUM — general, no underwater specializations |
| 3DGRUT / 3DGUT | Production (GitHub, USDZ export) | HIGH — USD bridge to Isaac Sim |
| OceanSim | Research (arXiv 2025) | HIGH — purpose-built underwater simulator |
| ASASIN (GPU sonar) | Research/military | MEDIUM — sonar processing |
| SubmergeStyleGAN | Research (IEEE 2024) | HIGH — synthetic underwater data DA |

---

## References (All URLs Verified)

- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1573612/full
- https://pui-nantheera.github.io/research/CIC/MyUnderwaterWorld.html
- https://dl.acm.org/doi/10.1145/3746027.3754888
- https://arxiv.org/html/2509.00800v1
- https://arxiv.org/abs/2304.07743
- https://github.com/deborahLevy130/seathru_NeRF
- https://arxiv.org/abs/2502.16351
- https://developer.nvidia.com/blog/how-to-instantly-render-real-world-scenes-in-interactive-simulation/
- https://developer.nvidia.com/blog/reconstruct-a-scene-in-nvidia-isaac-sim-using-only-a-smartphone/
- https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-NuRec
- https://www.nvidia.com/en-us/ai/cosmos/
- https://arxiv.org/html/2501.03575v1
- https://developer.nvidia.com/blog/scale-synthetic-data-and-physical-ai-reasoning-with-nvidia-cosmos-world-foundation-models/
- https://ieeexplore.ieee.org/document/10410962/
- https://arxiv.org/html/2505.14296v1
- https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2025.1581778/full
- https://arxiv.org/html/2503.01074v1
- https://umfieldrobotics.github.io/OceanSim/
- https://arxiv.org/html/2511.00392v1
- https://www.mdpi.com/2077-1312/11/5/949
- https://arxiv.org/pdf/2101.05888
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12157327/
- https://docs.omniverse.nvidia.com/extensions/latest/ext_pointclouds.html
- https://github.com/nv-tlabs/3dgrut
- https://arxiv.org/html/2412.12507v2
- https://radiancefields.com/nvidia-adds-usdz-support-to-3dgrut-and-beta-for-omniverse-and-isaac-sim
- https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81757/
- https://insights.evercoast.ai/p/integrating-4d-gaussian-splats-into
- https://www.youtube.com/watch?v=DYeutmDZOyI
- https://github.com/zubair-irshad/Awesome-Implicit-NeRF-Robotics
- https://li-chongyi.github.io/proj_benchmark.html
- https://www.mdpi.com/2072-4292/17/11/1906
- https://pmc.ncbi.nlm.nih.gov/articles/PMC11327533/
- https://docs.nvidia.com/learning/physical-ai/going-further-with-robotics/latest/digital-twin-robotics/index.html
