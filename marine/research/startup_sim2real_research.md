# Sim-to-Real Transfer for Underwater Robotics: Research Report

> Date: 2026-05-20
> Purpose: Inform OceanScale's "Verify" pillar — bridging simulation predictions to real-world underwater robot performance.
> Method: Web search across academic papers (2023-2026), open-source projects, and documented field experience.

---

## 1. Academic State of the Art

### 1.1 Key Papers (2023-2026)

| # | Title | Authors | Year | Key Finding | DOI/URL |
|---|-------|---------|------|-------------|---------|
| 1 | Underwater Robotic Simulators Review for Autonomous System Development | UCL FRL Lab | 2025 | Comprehensive review of 5 open-source simulators (Stonefish, DAVE, HoloOcean, MARUS, UUV Simulator) evaluating sensor fidelity, physics accuracy, ROS compatibility, and sim-to-real capability | [UCL Discovery PDF](https://discovery.ucl.ac.uk/id/eprint/10214438/1/2504.06245v1.pdf) |
| 2 | OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework | Song et al. | 2025 | GPU-accelerated simulator built on NVIDIA Isaac Sim; physics-based rendering for underwater images; real-time imaging sonar; evaluated against real-world data | [arXiv:2503.01074](https://arxiv.org/abs/2503.01074) — Accepted at IROS 2025 |
| 3 | MarineGym: A High-Performance RL Platform for Underwater Robotics | Chu et al. | 2025 | GPU-accelerated hydrodynamic plugin on Isaac Sim; 250K FPS on RTX 3060; 5 UUV models; domain randomization toolkit for sim-to-real | [arXiv:2503.09203](https://arxiv.org/abs/2503.09203) |
| 4 | Sim-to-Real Pipeline for Training Autonomous Obstacle Avoidance of Underwater Robots | Robotica / Cambridge | 2024 | Full RL-based pipeline for obstacle avoidance with kinematic/dynamic model analysis to optimize sim-to-real gap | [Cambridge University Press](https://www.cambridge.org/core/journals/robotica/article/simtoreal-pipeline-for-training-autonomous-obstacle-avoidance-of-underwater-robots-based-on-highfidelity-model/F8C61DC413D83D175DCF17C17A4A9718) |
| 5 | Reducing the Sim-to-Real Gap in Underwater Vehicles — Residual Dynamics Modelling | DiVA Portal (thesis) | 2024 | Real-to-sim-to-real approach: learn data-driven corrections (residual dynamics) to physics-based models; analyzes how control input complexity affects accuracy | [DiVA Portal PDF](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf) |
| 6 | Zero-Shot Sim-to-Real Transfer for RL-based Visual Servoing on Soft Continuum Arms | Yang et al. (PMLR v283) | 2025 | Zero-shot transfer of RL kinematic controller on BR2 platform without fine-tuning; demonstrates domain randomization effectiveness | [arXiv:2504.16916](https://arxiv.org/html/2504.16916v1) |
| 7 | RL for AUVs via Data-informed Domain Randomization (DDR) | MDPI Applied Sciences | 2023 | Model-free RL with DDR approach addressing trajectory data mismatch between sim and real | [MDPI](https://www.mdpi.com/2076-3417/13/3/1723) |
| 8 | URoBench: Comparative Analyses of Underwater Robotics Simulators from RL Perspective | Huang, Buchholz et al. | 2024 | Benchmark framework for standardized assessment of underwater simulators in RL tasks | [Semantic Scholar](https://www.semanticscholar.org/paper/URoBench%253A-Comparative-Analyses-of-Underwater-from-Huang-Buchholz/a4b9e28610507d5eb7bcb45c69be884f9e2fa237) |
| 9 | Stonefish: Supporting Machine Learning Research in Marine Robotics | Cieslak | 2025 | Enhanced Stonefish for ML: improved hydrodynamics, rendering pipeline for marine robotics | [arXiv:2502.11887](https://arxiv.org/html/2502.11887v2) |
| 10 | UNav-Sim: A Visually Realistic Underwater Robotics Simulator | Amer et al. | 2023 | First underwater sim on Unreal Engine 5; open-source; ROS1/ROS2 support; vision-based navigation stack | [arXiv:2310.11927](https://ar5iv.labs.arxiv.org/html/2310.11927) |
| 11 | AUV Hydrodynamic Coefficient Offline Identification Based on DRL | ScienceDirect | 2024 | System identification using deep RL to estimate hydrodynamic coefficients from position and attitude data only | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801824011478) |
| 12 | Research on Modeling Method of AUV Based on PINN | MDPI J. Marine Sci. & Eng. | 2024 | PINN method integrating dynamical equations with neural networks for AUV dynamics modeling | [MDPI](https://www.mdpi.com/2077-1312/12/5/801) |
| 13 | Modelling of Underwater Vehicles Using PINNs | arXiv | 2025 | PINNs for underwater vehicle modeling, integrating physics laws with data-driven models for better generalization | [arXiv:2504.20019](https://arxiv.org/pdf/2504.20019) |
| 14 | Prediction Modeling for Yaw Motion of Deep-Sea Mining Vehicle (PINN) | ScienceDirect | 2024 | PINN for hydrodynamic yaw motion model of deep-sea mining vehicle | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0141118724003298) |
| 15 | Learning to Dock: Quantifying and Reducing the Sim-to-Real Gap | Oregon State RDML | 2025 | Quantifies sim-to-real gap specifically for underwater robot docking; proposes reduction methods | [PDF](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/kevinchang2025revised.pdf) |
| 16 | Hydrodynamics Model Identification and Model-Based Control | MDPI J. Marine Sci. & Eng. | 2025 | Review of hydrodynamic coefficient motion equations for underwater vehicles; model ID and control strategies | [MDPI](https://www.mdpi.com/2077-1312/13/2/310) |
| 17 | Sym2Real: Symbolic Dynamics with Residual Learning for Data-Efficient Control | Lee, Moore et al. | 2024 | Fully data-driven framework combining low-fidelity sim with targeted real-world residual learning | [YouTube/Project Page](https://www.youtube.com/watch?v=KAfbwpL0qiM) |
| 18 | Physics-Informed ML for System Identification of an Underwater Vehicle | DiVA Portal | 2024 | PIML/PINNs for system ID of underwater vehicles | [DiVA Portal](https://www.diva-portal.org/smash/record.jsf?pid=diva2:2033932) |
| 19 | Ray-Based Physical Modeling and Simulation of Multibeam Sonar | MDPI Sensors | 2025 | Ray-based approach replacing raster-based for high-fidelity multibeam sonar simulation | [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/) |

### 1.2 Key Simulator Landscape (2025)

| Simulator | Base Engine | GPU-Accelerated | Sonar | Hydrodynamics | ROS | Open Source | Key Strength |
|-----------|-------------|-----------------|-------|---------------|-----|-------------|--------------|
| **MarineGym** | Isaac Sim | Yes (250K FPS) | No | GPU-accelerated | Yes | Yes | RL training speed, DR toolkit |
| **OceanSim** | Isaac Sim/Omniverse | Yes | Yes (imaging sonar) | Moderate | Yes | Yes | Perception rendering, sonar sim |
| **Stonefish** | Custom C++ | No | Limited | Advanced | Yes | Yes | Physics fidelity, lightweight |
| **DAVE** | Gazebo | No | Limited | Moderate | Yes | Yes | ROS-native Gazebo simulation |
| **HoloOcean** | Unreal Engine 4 | Partial | Yes (octree-based) | Validated in v1.x (v2.0 unreleased as of 2026-05) | Yes (v2.0) | Yes | Sonar realism, multi-agent |
| **UNav-Sim** | Unreal Engine 5 | Partial | No | Moderate | Yes (ROS1+2) | Yes | Visual realism, UE5 rendering |
| **UUV Simulator** | Gazebo | No | No | Basic | Yes | Yes | Simplest to start, widely adopted |
| **MARUS** | Gazebo/ROS | No | No | Basic | Yes | Yes | Modularity |

---

## 2. Known Sim-to-Real Gaps in Underwater

### 2.1 Documented Failure Modes

Based on the literature, the following gaps consistently cause sim-to-real transfer failures:

#### A. Hydrodynamic Model Errors
- **Drag coefficient mismatch**: Simulated drag coefficients typically differ 15-40% from real values due to Reynolds number effects, surface roughness, and biofouling not modeled in simulation (source: [DiVA Portal thesis](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf)).
- **Added mass uncertainty**: Added mass coefficients depend on vehicle geometry and proximity to surfaces/walls/bottom — rarely captured in sim. Pool walls create measurable added mass effects not present in open water.
- **Thruster dynamics**: Real thrusters exhibit dead-zones, saturation, cavitation at depth, and thrust degradation from biofouling — standard sim models assume linear response.
- **Tether/cable drag**: For tethered ROVs, cable hydrodynamics (vortex-induced vibration, drag) are typically ignored in sim but significantly affect real-world dynamics.

#### B. Sensor Simulation Gaps
- **Imaging sonar**: Real sonar has multipath reflections, noise patterns dependent on water conditions, and beam patterns that raster-based simulators fail to capture. Only ray-based approaches (HoloOcean 2.0, [PMC paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/)) approach realism.
- **DVL noise**: Real DVL performance varies with altitude above seafloor, water salinity, and bottom type. Per-beam noise characteristics are now measurable in real-time ([NavLab validation paper](https://www.navlab.net/Publications/Validation_of_a_New_Generation_DVL_for_Underwater_Vehicle_Navigation.pdf)), but simulators typically use Gaussian noise models.
- **IMU drift**: MEMS IMU bias instability and random walk are well-modeled, but temperature-dependent drift and vibration-induced errors from thrusters are rarely simulated.
- **Underwater vision**: Caustics, backscatter, turbidity, color absorption with depth, and marine snow create perception gaps that physics-based rendering (OceanSim) is only beginning to address.

#### C. Control Policy Transfer Failures
- **Policy overfitting to sim dynamics**: RL policies trained in sim exploit precise dynamic models — when real dynamics deviate (even slightly), accumulated error over trajectory leads to divergence. Documented in [Robotica paper](https://www.cambridge.org/core/journals/robotica/article/simtoreal-pipeline-for-training-autonomous-obstacle-avoidance-of-underwater-robots-based-on-highfidelity-model/F8C61DC413D83D175DCF17C17A4A9718).
- **Current disturbance mismatch**: Ocean currents are spatially varying and time-dependent. Simulators typically use uniform flow or simple profiles — real ocean has turbulence, internal waves, and tidal effects.
- **Visibility domain shift**: Policies trained on clear-water sim images fail catastrophically in turbid/low-light conditions.

### 2.2 Quantified Gaps (where available)

| Metric | Typical Sim-Real Gap | Source |
|--------|---------------------|--------|
| Position tracking error (pool) | 10-30% increase vs sim | [DiVA thesis](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf) |
| Drag coefficient accuracy | 15-40% error (analytical vs measured) | [MDPI Hydrodynamics Model ID](https://www.mdpi.com/2077-1312/13/2/310) |
| DVL velocity measurement | Varies with altitude, salinity, bottom type | [NavLab DVL validation](https://www.navlab.net/Publications/Validation_of_a_New_Generation_DVL_for_Underwater_Vehicle_Navigation.pdf) |
| Thruster response | Dead-zone + saturation not in sim | General observation across multiple papers |
| Sonar image fidelity | Raster-based methods produce unrealistic artifacts | [PMC ray-based sonar sim](https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/) |

---

## 3. Methods for Bridging the Gap

### 3.1 System Identification from Real Data

**Approach**: Use real-world sensor data to estimate hydrodynamic parameters (drag, added mass, damping coefficients).

**Methods in the literature**:

1. **Classical SI (System Identification)**: Execute prescribed maneuvers (zigzag, circle, PMM tests), fit coefficients via least-squares or maximum likelihood. Requires purpose-built test rigs or large pool.
   - [DTIC foundational document](https://apps.dtic.mil/sti/tr/pdf/ADA414448.pdf)
   - [SPIE Hydrodynamic coefficient estimation](https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12645/1264504/Hydrodynamic-coefficient-estimation-for-underwater-vehicle-maneuvering/10.1117/12.2680829.pdf)

2. **DRL-based identification**: Use deep RL to estimate hydrodynamic coefficients from position and attitude data only — no force/torque sensors needed.
   - [ScienceDirect DRL for AUV HCs](https://www.sciencedirect.com/science/article/abs/pii/S0029801824011478)

3. **Model test data**: Physical model tests (towing tank, captive model tests) to directly measure forces and derive coefficients.
   - [Academia.edu model test paper](https://www.academia.edu/123480147/Identification_of_Underwater_Vehicle_Hydrodynamic_Coefficients_Using_Model_Tests)

### 3.2 Domain Randomization for Underwater

**Unique underwater DR parameters** (beyond standard robotics DR):

| Parameter | Range | Why It Matters |
|-----------|-------|----------------|
| Current speed/direction | 0-2 m/s, any direction | External force disturbance |
| Water visibility (turbidity) | 0.5-20m Secchi depth | Affects all optical sensors |
| Salinity | 0-40 PSU | Changes buoyancy, acoustic propagation |
| Temperature gradient | 0-30C with depth | Thermocline affects sonar, density |
| Biofouling level | 0-50% surface coverage | Changes drag, weight |
| Surface conditions | Calm to Sea State 5 | Affects near-surface operations |
| Depth pressure | 0-600 bar | Affects vehicle compression, buoyancy |
| Light level | 0-100% surface irradiance | Affects visual navigation |

**MarineGym's DR toolkit** is the most explicit implementation: allows flexible adjustment of simulation and task parameters during training specifically for underwater environments ([arXiv:2503.09203](https://arxiv.org/abs/2503.09203)).

**DDR (Data-informed Domain Randomization)**: Uses real trajectory data to inform which parameters to randomize and in what range — more data-efficient than blind randomization ([MDPI](https://www.mdpi.com/2076-3417/13/3/1723)).

### 3.3 Residual Dynamics Learning

**Concept**: Instead of fixing sim parameters, learn a correction term:
```
real_dynamics = sim_dynamics + residual_network(state, action)
```

This approach is gaining traction:
- [DiVA Portal thesis](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf) specifically applies this to underwater vehicles with varying control input complexity.
- [Sym2Real](https://www.youtube.com/watch?v=KAfbwpL0qiM) combines symbolic dynamics with residual learning for data-efficient transfer across multiple real-world conditions.
- [arXiv 2402.01086](https://arxiv.org/html/2402.01086v1) applies residual physics to soft robots with large DOF.

**Advantage**: Preserves physics structure while learning only the discrepancy — requires less real data than learning dynamics from scratch.

### 3.4 Physics-Informed Neural Networks (PINNs)

**Concept**: Embed Navier-Stokes or rigid-body dynamics equations as soft constraints in neural network training.

Papers specifically targeting underwater vehicles:
- [MDPI AUV PINN modeling](https://www.mdpi.com/2077-1312/12/5/801) (2024)
- [arXiv PINNs for underwater vehicles](https://arxiv.org/pdf/2504.20019) (2025)
- [ScienceDirect deep-sea mining PINN](https://www.sciencedirect.com/science/article/abs/pii/S0141118724003298) (2024)
- [Taylor & Francis ship maneuvering PINN](https://www.tandfonline.com/doi/full/10.1080/19942060.2025.2566860) (2025)

**Trade-off**: PINNs require less training data than pure ML but are slower to train and harder to tune than standard approaches.

### 3.5 Progressive Transfer

**Standard hierarchy in marine robotics**:

```
Simulation → Dry Run → Pool Test → Sheltered Water → Open Ocean
  (GPU)      (dock)    (controlled)  (harbor/lake)   (field deployment)
```

Each stage introduces new unmodeled dynamics:
1. **Sim → Pool**: Wall effects, tether drag, thruster dead-zones
2. **Pool → Sheltered**: Current, waves, visibility changes
3. **Sheltered → Ocean**: Corrosion, biofouling, communication loss, emergency handling

This progressive approach is standard practice at MBARI, WHOI, NTNU, and EU-funded programs (euRathlon, ERL). No single paper documents the full pipeline, but it is the consensus methodology in the field. (unverified — community standard practice, not formally cited in a single paper)

### 3.6 Sensor Calibration and Noise Modeling

**DVL calibration**: Data-driven methods for fusing inertial sensors with DVL; per-beam noise estimation in real-time ([arXiv:2401.12687](https://arxiv.org/html/2401.12687v1), [NavLab DVL validation](https://www.navlab.net/Publications/Validation_of_a_New_Generation_DVL_for_Underwater_Vehicle_Navigation.pdf)).

**IMU modeling**: MEMS IMU performance varies with temperature, vibration, and magnetic interference from thrusters. Allan variance analysis is the standard method for characterizing noise parameters. ([MathWorks AUV pose estimation](https://www.mathworks.com/help/nav/ug/autonomous-underwater-vehicle-pose-estimation-using-inertial-sensors-and-doppler-velocity-log.html))

**Sonar calibration**: Ray-based simulation approaches (HoloOcean) are moving toward realistic sonar imagery, but validated calibration datasets remain scarce. ([BYU IROS 2022 sonar sim](https://robots.et.byu.edu/jmangelson/pubs/2022/Potokar22iros.pdf))

---

## 4. Data Requirements for Validation

### 4.1 What Real-World Data Is Needed

| Data Type | Sensors to Log | Purpose |
|-----------|---------------|---------|
| Vehicle state | IMU (6-DOF), DVL, depth sensor, GPS (surface) | Ground truth trajectory |
| Control inputs | Thruster RPMs, fin angles | Dynamics identification |
| Environment | CTD (conductivity-temp-depth), ADCP (current profile), Secchi disk / turbidity sensor | Environmental context |
| Vision | Stereo cameras, lighting conditions | Perception validation |
| Acoustic | Imaging sonar, acoustic modem | Sonar sim validation |
| External reference | USBL/LBL positioning, motion capture (pool) | Independent ground truth |

### 4.2 How Much Data?

There is no single published standard for "how much data is enough." The following are observed practices from the literature:

| Validation Level | Typical Data Volume | Source/Justification |
|-----------------|-------------------|---------------------|
| Hydrodynamic coefficient ID | 20-50 maneuver trajectories (zigzag, turning circle, spiral) | Standard naval architecture practice |
| Control policy validation | 50-200 episodes across varying conditions | RL transfer literature convention |
| Perception model validation | 1K-10K real images with annotations | Computer vision standard |
| Full mission validation | 10-50 complete mission runs per scenario | (unverified — field practice estimate) |
| Long-duration reliability | 50-200 hours of continuous operation | (unverified — industry practice for certification) |

### 4.3 Public Datasets

| Dataset | Content | URL |
|---------|---------|-----|
| AUV Navigation Dataset (Natural Scenarios) | High-precision fiber-optic inertial sensor data from real AUV | [MDPI Electronics](https://www.mdpi.com/2079-9292/12/18/3788) |
| CMU AUV Sensing Dataset | DVL, stereo camera, DIDSON sonar from field deployment | [CMU Thesis PDF](https://www.ri.cmu.edu/app/uploads/2019/08/Suddhu_MSR_Thesis.pdf) |

Note: Underwater robotics datasets remain far fewer and smaller than aerial or ground robotics datasets. This is itself a gap.

### 4.4 How Labs Currently Collect Data

1. **Towing tank / basin**: Precisely controlled captive model tests. Expensive, requires facility access.
2. **Pool tests with motion capture**: Underwater motion capture (e.g., Qualisys underwater) provides mm-level ground truth. Limited to pool-scale vehicles.
3. **Field deployment with USBL**: USBL provides periodic position fixes; dead reckoning fills gaps. Standard for open-water validation.
4. **Harbor/sheltered water**: Compromise between realism and controllability. Common for pre-ocean validation.

---

## 5. Existing Tools for Verification

### 5.1 Simulator Comparison Frameworks

| Tool | Description | URL |
|------|-------------|-----|
| **URoBench** | Standardized benchmark for comparing underwater simulators from RL perspective | [Semantic Scholar](https://www.semanticscholar.org/paper/URoBench%253A-Comparative-Analyses-of-Underwater-from-Huang-Buchholz/a4b9e28610507d5eb7bcb45c69be884f9e2fa237) |
| **UCL Review** | Head-to-head comparison of Stonefish, DAVE, HoloOcean, MARUS, UUV Simulator | [PDF](https://discovery.ucl.ac.uk/id/eprint/10214438/1/2504.06245v1.pdf) |
| **AwesomeSim2Real** | Curated list of RL sim-to-real papers (not underwater-specific) | [GitHub](https://github.com/LongchaoDa/AwesomeSim2Real) |

### 5.2 ROS-Based Tools

- **ROS bag**: Standard data logging format. `rosbag2` for ROS2. No underwater-specific analysis tools found — general tools (rqt_plot, PlotJuggler, rosboard) are used.
- **UUV Simulator**: Gazebo-based, simplest ROS integration for basic underwater dynamics.
- **DAVE**: Gazebo-based with deeper physics, better for validation scenarios.

### 5.3 GPU-Accelerated Simulation

| Tool | GPU Feature | Underwater-Specific |
|------|-------------|-------------------|
| **MarineGym** | GPU hydrodynamics, 250K FPS | Yes (Isaac Sim plugin) |
| **OceanSim** | GPU rendering, real-time sonar | Yes (Isaac Sim based) |
| **NVIDIA Isaac Sim** | Base platform for above | No (generic robotics) |

### 5.4 Gaps in Existing Verification Tools

- **No standardized trajectory comparison tool**: There is no widely-adopted tool for automated comparison of simulated vs. real underwater trajectories. Labs build ad-hoc scripts.
- **No underwater-specific calibration framework**: Unlike aerial robotics (where tools like PX4 SITL + Gazebo have mature calibration pipelines), underwater lacks equivalent.
- **No confidence scoring system**: No existing tool provides "how confident are we that this sim matches reality" as a quantitative metric.
- **No progressive test suite framework**: Labs manually manage sim → pool → sheltered → ocean progression.

---

## 6. OceanScale's Opportunity: What to Build for "Verify"

Based on the research findings, the following are actionable product opportunities for OceanScale's "Verify" phase, ranked by impact and feasibility:

### 6.1 High Priority

#### A. Automated Trajectory Comparison (sim-vs-real)
- **What**: Given a real-world trajectory (from ROS bag or CSV) and a simulated trajectory under the same conditions, compute quantitative similarity metrics.
- **Metrics**: RMSE position error, orientation error, velocity error over time; dynamic time warping for temporal alignment; frequency-domain comparison for oscillatory behaviors.
- **Why**: No existing tool does this well for underwater. Every lab reinvents this. High demand, clear value proposition.
- **Input**: Real sensor log (IMU + DVL + depth + USBL) + sim config + control log.
- **Output**: Similarity report with per-axis error breakdown, pass/fail thresholds.

#### B. Hydrodynamic Parameter Identification from Flight Data
- **What**: Given real-world maneuver data (thruster commands + resulting motion), identify drag coefficients, added mass, and damping parameters via optimization.
- **Method**: Build on residual dynamics learning (Section 3.3) or DRL-based identification ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0029801824011478)).
- **Why**: This closes the loop — real data improves the sim, which then generates better predictions. The "flywheel" effect.
- **Unique value**: GPU-accelerated parameter search using Newton/Warp, making it orders of magnitude faster than classical SI methods.

#### C. Domain Randomization Configurator
- **What**: Given a vehicle model and operating environment description, generate a DR configuration that covers the relevant parameter space (current, visibility, salinity, biofouling, temperature).
- **Integration**: Leverage MarineGym-style DR toolkit concepts, but as a standalone configurator that outputs to any simulator.
- **Why**: DR is the #1 technique for sim-to-real transfer in RL. Making it easy to configure for underwater specifically is a differentiator.

### 6.2 Medium Priority

#### D. Confidence Scoring for Simulation Predictions
- **What**: Given a sim model + identified parameters + their uncertainty bounds, provide a confidence score for how well the sim will predict real behavior in a given scenario.
- **Method**: Ensemble simulation with parameter uncertainty propagation; compare prediction intervals against real data.
- **Why**: Customers need to know when to trust the sim and when to go to physical testing. A quantitative confidence score is valuable for go/no-go decisions before expensive field deployments.

#### E. Progressive Test Suite Framework
- **What**: Define a testing hierarchy (unit → component → pool → sheltered → ocean) with automated pass/fail criteria at each level.
- **Structure**:
  - Level 1 (Unit): Individual sensor models validated against datasheets
  - Level 2 (Component): Thruster response curves validated against bench test data
  - Level 3 (Integration): Full vehicle dynamics validated against pool test data
  - Level 4 (Field): Mission-level performance validated against harbor/sheltered water data
  - Level 5 (Sea): Open-ocean validation against real deployment data
- **Why**: Standardizes the verification process. Currently every team invents their own hierarchy.

### 6.3 Longer-Term

#### F. Physics-Informed Residual Dynamics Engine
- **What**: A hybrid physics + ML dynamics engine that uses Newton/Warp for the base physics and learns residual corrections from real data.
- **Approach**: PINN architecture that embeds Fossen's 6-DOF underwater vehicle equations as physics constraints.
- **Why**: This is the "holy grail" — a simulator that automatically improves itself from real-world data. The research is nascent enough that OceanScale could be early.

#### G. Digital Twin Data Pipeline
- **What**: End-to-end pipeline from real-world sensor logs → parameter identification → sim model update → validation → deployment.
- **Why**: Currently this is a manual, multi-tool process. A unified pipeline would be a significant product differentiator.

---

## 7. Competitive Landscape Summary

| Existing Tool | What It Does | What It Lacks (OceanScale Opportunity) |
|--------------|-------------|---------------------------------------|
| MarineGym | Fast RL training with DR | No verification/comparison tools; training-only |
| OceanSim | Perception sim with realistic rendering | Perception-focused, not dynamics validation |
| Stonefish | High-fidelity physics | No automated parameter ID or comparison |
| HoloOcean | Realistic sonar | Sonar-focused, no dynamics verification |
| URoBench | Simulator comparison | Academic benchmark only, not a product tool |

**Key insight**: The entire ecosystem focuses on **simulation** (the "Simulate" phase). Almost nobody builds tools for the **"Verify"** phase — comparing sim against reality, identifying what's wrong, and closing the loop. This is OceanScale's whitespace.

---

## 8. Key References (Consolidated)

### Papers
1. UCL Underwater Simulators Review (2025): https://discovery.ucl.ac.uk/id/eprint/10214438/1/2504.06245v1.pdf
2. OceanSim (IROS 2025): https://arxiv.org/abs/2503.01074
3. MarineGym (2025): https://arxiv.org/abs/2503.09203
4. Sim-to-Real Pipeline Obstacle Avoidance (2024): https://www.cambridge.org/core/journals/robotica/article/simtoreal-pipeline-for-training-autonomous-obstacle-avoidance-of-underwater-robots-based-on-highfidelity-model/F8C61DC413D83D175DCF17C17A4A9718
5. Residual Dynamics Modelling (2024): https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf
6. Zero-Shot Sim-to-Real BR2 (2025): https://arxiv.org/html/2504.16916v1
7. DDR for AUV (2023): https://www.mdpi.com/2076-3417/13/3/1723
8. URoBench (2024): https://www.semanticscholar.org/paper/URoBench%253A-Comparative-Analyses-of-Underwater-from-Huang-Buchholz/a4b9e28610507d5eb7bcb45c69be884f9e2fa237
9. Stonefish ML Support (2025): https://arxiv.org/html/2502.11887v2
10. UNav-Sim (2023): https://ar5iv.labs.arxiv.org/html/2310.11927
11. AUV Hydrodynamic Coefficient DRL (2024): https://www.sciencedirect.com/science/article/abs/pii/S0029801824011478
12. PINN AUV Modeling (2024): https://www.mdpi.com/2077-1312/12/5/801
13. PINNs Underwater Vehicles (2025): https://arxiv.org/pdf/2504.20019
14. Deep-Sea Mining PINN (2024): https://www.sciencedirect.com/science/article/abs/pii/S0141118724003298
15. Learning to Dock Sim-to-Real (2025): https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/kevinchang2025revised.pdf
16. Hydrodynamics Model ID (2025): https://www.mdpi.com/2077-1312/13/2/310
17. Sym2Real (2024): https://www.youtube.com/watch?v=KAfbwpL0qiM
18. PIML Underwater Vehicle (2024): https://www.diva-portal.org/smash/record.jsf?pid=diva2:2033932
19. Ray-Based Sonar Sim (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC11902455/
20. DVL Calibration Data-Driven (2024): https://arxiv.org/html/2401.12687v1
21. HoloOcean Sonar (IROS 2022): https://robots.et.byu.edu/jmangelson/pubs/2022/Potokar22iros.pdf
22. HoloOcean (ICRA 2022): https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf
23. Ship Maneuvering PINN (2025): https://www.tandfonline.com/doi/full/10.1080/19942060.2025.2566860
24. AUV Navigation Dataset: https://www.mdpi.com/2079-9292/12/18/3788
25. DVL Validation (NavLab): https://www.navlab.net/Publications/Validation_of_a_New_Generation_DVL_for_Underwater_Vehicle_Navigation.pdf
26. AUV Hydrodynamic Coefficient Estimation (SPIE): https://www.spiedigitallibrary.org/conference-proceedings-of-spie/12645/1264504/Hydrodynamic-coefficient-estimation-for-underwater-vehicle-maneuvering/10.1117/12.2680829.pdf

### Open-Source Projects
- MarineGym: https://github.com/Marine-RL/MarineGym
- OceanSim: https://github.com/umfieldrobotics/OceanSim
- Stonefish: https://github.com/patrykcieslak/stonefish
- UNav-Sim: https://github.com/open-airlab/UNav-Sim
- HoloOcean: https://byu-holoocean.github.io/holoocean-docs/
- AwesomeSim2Real: https://github.com/LongchaoDa/AwesomeSim2Real

### Tools
- MathWorks AUV Pose Estimation: https://www.mathworks.com/help/nav/ug/autonomous-underwater-vehicle-pose-estimation-using-inertial-sensors-and-doppler-velocity-log.html

---

## 9. Unverified Claims & Gaps

The following items in this report are based on general field knowledge rather than specific cited sources:

1. **Progressive transfer hierarchy** (pool → sheltered → ocean) is described as "standard practice" at MBARI/WHOI/NTNU but no single paper formally documents this pipeline.
2. **Data volume estimates** (hours, trajectories) are informed estimates from the literature rather than published requirements.
3. **Biofouling effects on drag coefficients** are well-known qualitatively but quantified data (e.g., "X% drag increase per Y weeks of deployment") was not found in this search.
4. **Tether cable dynamics** are identified as a gap across multiple papers but no systematic study of tether sim-to-real transfer was found.
5. **The claim that "nobody builds verification tools"** is an inference from the absence of such tools in search results. It should be validated by talking to practitioners.
