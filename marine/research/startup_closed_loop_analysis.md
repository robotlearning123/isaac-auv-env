# OceanScale Closed-Loop Ocean Engineering Lifecycle Analysis

**Date:** 2026-05-20
**Purpose:** Map the complete Design → Simulate → Test → Verify → Deploy → Feedback loop for underwater robotics. Identify where the loop breaks today and what OceanScale must build to close each gap.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Phase 1: Design](#2-phase-1-design)
3. [Phase 2: Simulate](#3-phase-2-simulate)
4. [Phase 3: Test](#4-phase-3-test)
5. [Phase 4: Verify (Sim-to-Real)](#5-phase-4-verify-sim-to-real)
6. [Phase 5: Deploy](#6-phase-5-deploy)
7. [Phase 6: Feedback (Closing the Loop)](#7-phase-6-feedback-closing-the-loop)
8. [Existing Closed-Loop Platforms](#8-existing-closed-loop-platforms)
9. [Gap Analysis](#9-gap-analysis)
10. [Sources](#10-sources)

---

## 1. Executive Summary

The underwater robotics industry lacks a unified closed-loop platform. Today, engineers use disconnected tools across each lifecycle phase — CAD in one application, simulation in another, testing in a third, deployment via custom middleware, and feedback through manual data analysis. No single platform connects these phases for underwater vehicles.

General-purpose PLM (Product Lifecycle Management) platforms exist — Siemens Xcelerator, PTC Windchill, Autodesk Fusion — but none are ocean-specific. The closest analogues are:

- **Siemens Xcelerator** (CAD + Sim + Test for ships, but not underwater robots)
- **NVIDIA Omniverse/Isaac Sim** (Sim + Test + Deploy for general robotics, underwater support emerging via OceanSim)
- **ROS 2 ecosystem** (Sim + Test + Deploy, but fragmented)
- **Greensea OPENSEA** (Deploy + Feedback, but proprietary)

**The fundamental gap:** There is no platform that takes an engineer from "I need an AUV rated to 500m" through parametric design, GPU-accelerated simulation, automated verification, and deployment — then closes the loop with real-world feedback.

OceanScale's opportunity is to be that platform.

---

## 2. Phase 1: Design

### 2.1 Current Tools for Underwater Robot Design

**CAD Platforms (General-Purpose, adapted for marine):**

| Tool | Role in Marine Design | Limitations |
|------|----------------------|-------------|
| **SolidWorks** (Dassault) | Industry standard for mechanical design of AUV/ROV pressure vessels, frames, housings. FEA for pressure analysis. | No marine-specific modules. Hydrodynamics require external CFD. Source: [ResearchGate — Design Analysis of AUV Using CAD](https://www.researchgate.net/publication/334479028_Design_analysis_and_modelling_of_autonomous_underwater_vehicle_AUV_using_CAD) |
| **Fusion 360** (Autodesk) | Growing adoption for startup-scale marine robot design. Cloud-based collaboration. Integrated simulation. | Limited CFD capabilities. Not purpose-built for marine. |
| **FreeCAD** | Open-source option used by academic teams. | No marine plugins. Manual workflow. |
| **Rhino 3D + Orca3D** | Hull modeling + hydrostatics in one workflow. Excellent for concept design. | Not a full mechanical CAD; no FEA for pressure vessels. |

**Marine-Specific Design Tools:**

| Tool | Function | Source |
|------|----------|--------|
| **Maxsurf** (Bentley Systems) | Hull modeling, hydrostatics, stability, resistance prediction, motion analysis. Full vessel design suite. | [Bentley — Maxsurf](https://www.bentley.com/en/products/brands/maxsurf) |
| **Orca3D** (plugin for Rhino) | Hull design, hydrostatics, stability, resistance prediction (Holtrop, Savitsky methods). Weight/cost tracking. | [Orca3D — Official](https://orca3d.com/) |
| **GHS (General HydroStatics)** | Industry gold standard for stability calculations, damage stability, regulatory compliance (IMO, USCG). Highly scriptable. | [Creative Systems — GHS](https://ghsport.com/) |
| **Wageningen B-Series tools** | Propeller/thruster selection based on Kt/Kq open-water diagrams. Online generators exist. | [B-Series Propeller Generator](https://www.wageningen-b-series-propeller.com/); [U. Michigan Reference](https://deepblue.lib.umich.edu/bitstreams/2f3da55b-78ee-478c-b5f0-c3b8d33989e/download) |
| **OpenPROP** | Open-source propeller design/optimization tool (MIT). | Academic software, limited maintenance. |

### 2.2 Current Design Workflow (Typical)

1. **Requirements definition:** "Need AUV rated to 500m, 4-hour endurance, side-scan sonar payload" — typically a manual document (Word/Excel).
2. **Hull form design:** Rhino + Orca3D or Maxsurf for hydrostatics and initial hull shape. Iterative: designers manually adjust, re-check stability, re-check resistance.
3. **Mechanical design:** SolidWorks/Fusion for pressure vessels, frames, endcaps, O-ring grooves, penetrators. FEA for crush depth verification.
4. **Thruster/propeller selection:** Wageningen B-series charts or MARIN software, cross-referenced with manufacturer catalogs (Technadyne, Innerspace, Copenhagen Subsea).
5. **Payload integration:** Manual CAD placement, cable routing, buoyancy calculations (often in Excel).
6. **Weight and balance:** Spreadsheet tracking of component weights, moments, and centers of gravity. Manual iteration to achieve neutral buoyancy.

### 2.3 What's Automated vs. Manual

| Step | Status |
|------|--------|
| Hull hydrostatics | Semi-automated (Orca3D/Maxsurf compute from geometry) |
| Resistance prediction | Semi-automated (empirical formulas in Orca3D/Maxsurf) |
| Pressure vessel FEA | Manual setup, automated solve |
| Propeller selection | Mostly manual (lookup charts + catalogs) |
| Weight/balance tracking | Manual (spreadsheets) |
| Buoyancy calculation | Manual |
| Payload integration | Manual |
| Requirements-to-design traceability | None |

### 2.4 Gap: No Parametric Design for Underwater Robots

There is no tool where an engineer inputs "500m depth rating, 4 knots cruise, side-scan sonar" and gets a parametrically generated vehicle design. Ship design has parametric tools (Maxsurf, NAPA); underwater robot design does not.

---

## 3. Phase 2: Simulate

### 3.1 Current Underwater Simulators (2024-2025 Landscape)

**Comprehensive Comparison:**

| Simulator | Engine | Hydrodynamics | Sensors | ROS | GPU-Accel | License | Source |
|-----------|--------|---------------|---------|-----|-----------|---------|--------|
| **Stonefish** | Bullet Physics | High (geometry-based drag, buoyancy, added mass) | DVL, IMU, depth, cameras, event camera, thermal | ROS 1 & 2 | No (CPU) | Open source (C++) | [stonefish.readthedocs.io](https://stonefish.readthedocs.io/); [arXiv:2502.11887](https://arxiv.org/html/2502.11887v1) |
| **HoloOcean 2.0** | Unreal Engine | High (nonlinear quadratic damping, Coriolis, control surfaces) | Imaging sonar, depth, DVL | ROS 2 | Partial (GPU rendering) | Academic | [arXiv:2510.06160](https://arxiv.org/html/2510.06160v1); [holoocean-docs](https://byu-holoocean.github.io/holoocean-docs/) |
| **OceanSim** | NVIDIA Isaac Sim | Moderate (uses PhysX) | Cameras, sonar (via custom plugins) | ROS 2 | Yes (GPU ray tracing) | Open source | [GitHub — OceanSim](https://github.com/umfieldrobotics/OceanSim); [arXiv:2503.01074](https://arxiv.org/html/2503.01074v2) |
| **UUV Simulator** | Gazebo/ODE | Basic (linear damping coefficients) | DVL, IMU, sonar (ray-trace), pressure | ROS 1 & 2 | No | Open source | [GitHub — uuv_simulator](https://github.com/uuvsimulator/uuv_simulator) |
| **MARUS** | Unreal Engine | Moderate | Sonar, cameras | ROS 2 | Yes (GPU rendering) | Open source | [MARUS — GitHub](https://github.com/marus-project/marus) |
| **DAVE** | Gazebo | Moderate | DVL, IMU, sonar, cameras | ROS | No | Open source | [GitHub — dave](https://github.com/Field-Robotics-Lab/dave) |
| **MarineGym** | Custom (GPU) | Moderate | Basic | Planned | Yes | Open source | [arXiv:2503.09203](https://arxiv.org/pdf/2503.09203) |

**Source for comparative review:** [arXiv:2504.06245v1 — Underwater Robotic Simulators Review](https://arxiv.org/html/2504.06245v1); [UCL FRL Review](https://frl-ucl.github.io/projects/underwater-robotic-simulators.html)

### 3.2 Physics Missing from Current Simulators

| Physics Phenomenon | Current Status |
|-------------------|----------------|
| **Nonlinear hydrodynamic damping** | Only HoloOcean 2.0 and Stonefish model this well |
| **Added mass / Coriolis coupling** | Partially modeled in Stonefish, HoloOcean 2.0 |
| **Vortex-induced vibration (VIV)** | Not modeled in any real-time simulator |
| **Wave-induced forces** | Basic (sine-wave surface) in most; no spectral sea state modeling |
| **Ocean currents (spatially varying)** | Uniform current in most; no CFD-coupled flow fields |
| **Thermal stratification** | Not modeled |
| **Salinity gradients** | Not modeled |
| **Underwater visibility (turbidity, scattering)** | Only OceanSim (physics-based rendering); basic fog model in others |
| **Caustics / light patterns** | Only OceanSim |
| **Biofouling drag effects** | Not modeled |
| **Cable/tether dynamics** | Basic spring models in some; no full cable hydrodynamics |
| **Acoustic propagation** | Simplified ray-tracing for sonar; no full acoustic wave equation |
| **Multi-body contact (docking)** | Limited; rigid-body only in most |

### 3.3 Sensor Simulation Status

| Sensor | Simulation Quality | Limitations |
|--------|--------------------|-------------|
| **IMU** | Good in most simulators (drift + noise models) | No temperature-dependent drift; no magnetic anomaly simulation |
| **DVL** | Adequate (velocity + noise) | No bottom-lock loss simulation; no water-track mode |
| **Depth/Pressure** | Good (simple conversion + noise) | No tidal variation simulation |
| **Side-scan sonar** | Basic ray-tracing in UUV Simulator, HoloOcean | No backscatter model; no sediment type variation |
| **Multibeam echosounder** | Basic ray-tracing | No water column effects; simplified seafloor reflectivity |
| **Forward-looking sonar** | HoloOcean has imaging sonar model | Limited resolution; no multipath |
| **Optical camera** | Good in OceanSim (physics-based underwater rendering); basic in others | Visibility/turbidity not modeled in most |
| **USBL/LBL acoustic positioning** | Not simulated in any open-source tool | Commercial tools (e.g., QINSy) exist but are separate |
| **CTD (Conductivity-Temperature-Depth)** | Not simulated | No salinity/temperature field simulation |

### 3.4 Environmental Simulation Gaps

- **Sea state modeling:** Most simulators use flat water or simple sine waves. No spectral wave models (JONSWAP, Pierson-Moskowitz).
- **Current fields:** Uniform or simple profiles. No mesoscale eddies, tidal currents, or CFD-derived flow fields.
- **Visibility:** Only OceanSim has physics-based turbidity/scattering. Others use fog approximation.
- **Bathymetry:** Flat seabed or simple imported meshes. No procedural terrain generation for realistic seafloor.

---

## 4. Phase 3: Test

### 4.1 Pre-Water Testing Methods

**Hardware-in-the-Loop (HIL):**

HIL testing for marine vehicles connects real hardware (sensors, computers, actuators) to a simulated environment. The simulator generates sensor stimuli in real-time while the actual vehicle controller runs on real hardware.

| HIL System | Target | Status | Source |
|------------|--------|--------|--------|
| **HoloOcean HIL** | Torpedo-class AUVs | Demonstrated with CougUV vehicle; SIL + HIL verified | [arXiv:2511.07687](https://arxiv.org/abs/2511.07687) |
| **Work-class ROV HIL** | ROV motion controllers | Full pipeline: design → implementation → verification | [ScienceDirect — HIL for Work-Class ROVs](https://www.sciencedirect.com/science/article/abs/pii/S0029801824029433) |
| **DP-HIL Simulator** | Dynamic Positioning systems | Practical testing of DP computer hardware/software for marine vessels | [DP Conference 2025](https://dynamic-positioning.com/wp-content/uploads/2025/12/control_johansen.pdf) |
| **TREE-C HIL** | Offshore/subsea operations | Commercial HIL for remote handling systems | [TREE-C — HIL for Marine](https://www.tree-c.nl/general/hardware-in-the-loop-simulators-for-marine-environments-explained/) |

**Software-in-the-Loop (SIL):**

SIL testing runs the actual vehicle software against a simulator without real hardware. Cheaper and faster than HIL.

- ROS 2 + Gazebo/Stonefish is the most common SIL setup
- HoloOcean 2.0 now supports ROS 2 integration for SIL testing

### 4.2 What Can ONLY Be Tested in Simulation

| Test Type | Why Simulation Only |
|-----------|-------------------|
| **Edge case scenarios** (equipment failure, extreme currents, entanglement) | Too dangerous or expensive to reproduce in water |
| **Reinforcement learning training** (millions of episodes) | Physically impossible at required scale |
| **Sensor degradation testing** (progressive sonar failure, IMU drift) | Difficult to safely induce in real hardware |
| **Multi-vehicle coordination** (swarm of 50+ AUVs) | Cost-prohibitive with real vehicles |
| **Deep-water scenarios** (>1000m) | Requires expensive ship time |
| **Collision testing** | Risk of destroying expensive hardware |
| **Parametric sweeps** (100+ design variants) | Each variant requires physical construction |

### 4.3 What MUST Be Tested in Water

| Test Type | Why Water Only |
|-----------|---------------|
| **Pressure vessel integrity** (crush test) | No simulation accurately predicts material failure under hydrostatic pressure |
| **Thruster performance curves** (bollard pull, cavitation) | CFD is approximate; real fluid behavior differs |
| **Acoustic communication range** | Sound propagation is highly environment-dependent |
| **Biofouling effects** | Biological processes cannot be simulated |
| **Electromagnetic interference** | Vehicle-specific; depends on actual wiring and grounding |
| **Launch and recovery** | Ship-based operations; human factors |
| **Connector/cable durability** | Fatigue under real bending + pressure cycling |

### 4.4 Gap: No Automated Test Suite for Underwater Vehicles

Unlike software engineering (CI/CD pipelines, pytest, etc.), there is no standard automated test suite for AUV/ROV software. Each team builds custom test scripts. There is no equivalent of `pytest` for marine robotics.

---

## 5. Phase 4: Verify (Sim-to-Real)

### 5.1 Methods for Sim-to-Real Transfer in Underwater Robotics

**Domain Randomization for Underwater:**

Underwater domain randomization varies:
- **Visual properties:** Lighting, water clarity (turbidity), color attenuation, caustics, scattering
- **Physics properties:** Drag coefficients, added mass, current speed/direction, thruster efficiency
- **Sensor properties:** Noise levels, bias, dropout rate, update frequency

Source: [MDPI — RL for AUVs via Data-informed Domain Randomization](https://www.mdpi.com/2076-3417/13/3/1723); [arXiv:2504.06245v1](https://arxiv.org/html/2504.06245v1)

**Data-informed Domain Randomization (DDR):**

A method that adjusts randomization parameter weights based on real-world data, narrowing the distribution to match observed reality. Demonstrated for AUV control policies.

Source: [MDPI Applied Sciences — DDR for AUVs](https://www.mdpi.com/2076-3417/13/3/1723)

**DrEureka (LLM-Guided Sim-to-Real):**

Uses large language models to automatically design reward functions and simulation parameters for sim-to-real transfer. Published at RSS 2024.

Source: [RSS 2024 — DrEureka](https://www.roboticsproceedings.org/rss20/p094.pdf)

### 5.2 Sim-to-Real for AUVs (Specific Results)

**Adaptive Control Parameter Transfer (IJRR 2024):**

A method combining Maximum Entropy Deep Reinforcement Learning with model-based control for sim-to-real transfer of AUV adaptive control parameters. Demonstrated successful transfer from simulation to physical AUV.

Source: [IJRR 2024 — Sim-to-Real for AUV Adaptive Control](https://journals.sagepub.com/doi/10.1177/02783649241272115)

**Underwater Docking Study (Oregon State):**

Quantified the sim-to-real gap specifically for underwater robot docking — a precision task. Found that real-world sensor noise, current disturbances, and positioning errors significantly exceed simulation predictions.

Source: [Oregon State — Learning to Dock](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/kevinchang2025revised.pdf)

### 5.3 Sim-to-Real Gap: Common Failure Modes

| Failure Mode | Description | Source |
|-------------|-------------|--------|
| **Simplified hydrodynamics** | Simulators use linear damping; real vehicles experience nonlinear drag, vortex shedding, added mass effects | [DiVA Portal — Reducing Sim-to-Real Gap](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf) |
| **Sensor model mismatch** | Simulated sonar/IMU noise doesn't match real-world distributions; multipath, scattering ignored | [arXiv:2510.20808 — Reality Gap](https://arxiv.org/html/2510.20808v1) |
| **Cable/tether forces** | Often omitted from simulation but dominant forces in tethered ROVs | Industry consensus |
| **Environmental variability** | Temperature, salinity, visibility, biofouling vary dramatically and are hard to simulate | [DiVA Portal](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf) |
| **Actuator dynamics** | Thruster response curves differ from manufacturer specs; aging and biofouling change performance | Industry consensus |
| **Structural compliance** | Simulators model rigid bodies; real vehicles flex under pressure and hydrodynamic loads | [arXiv:2510.20808](https://arxiv.org/html/2510.20808v1) |

### 5.4 Digital Twin Approaches for Marine Vehicles

| Approach | Description | Source |
|----------|-------------|--------|
| **NATO CMRE architecture** | Digital twin framework for Maritime Unmanned Systems, bridging simulation and real-world data | [NATO STO — Digital Twin Bridge](https://publications.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-MSG-207/MP-MSG-207-20.pdf) |
| **UUV multi-scale motion prediction** | Fuses real-time sensor data (speed, attitude, depth) with hydrodynamic simulation for motion prediction | [MDPI — DT for UUV Motion](https://www.mdpi.com/2077-1312/14/6/557) |
| **Underwater teleoperation DT** | VR-based digital twin for underwater robot teleoperation, providing real-time visual feedback | [Heriot-Watt — DT Below Surface](https://pure.hw.ac.uk/ws/portalfiles/portal/137253956/2402.07556v1.pdf) |
| **EU ESR10 submarine DT** | Framework where virtual model uses live data to simulate physical system behavior in real-time | [EU Research Portal](https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e517e1bd68&appId=PPGMS) |
| **Pursuit-evasion gaming DT** | Uses digital twin for adversarial strategy optimization in underwater grasping tasks | [ScienceDirect — DT Pursuit-Evasion](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003047) |

---

## 6. Phase 5: Deploy

### 6.1 Software Deployment to Underwater Robots

**Middleware Platforms:**

| Platform | Architecture | Key Feature | Source |
|----------|-------------|-------------|--------|
| **ROS 2** | DDS-based pub-sub | Industry standard for robot software. Growing adoption in AUV/ROV. Modular package architecture. | [ROS 2 — Official](https://docs.ros.org/) |
| **MOOS-IvP** | Custom pub-sub (MOOS) + behavior optimizer (IvP Helm) | MIT-developed. Excels in behavior-based autonomy via multi-objective optimization. C++. | [MIT MOOS-IvP](https://oceanai.mit.edu/moos-ivp/) |
| **DUNE (LSTS)** | Task-based architecture + IMC protocol | University of Porto. Full toolchain: DUNE (onboard) + Neptus (C2 GUI) + IMC (messages). Fleet-level coordination. | [LSTS — GitHub](https://github.com/LSTS) |
| **Greensea OPENSEA** | Open-architecture middleware (proprietary) | 20 years of development. Used on commercial ROVs/AUVs. Includes edge computing (OPENSEA Edge) and GNCC solutions (IQNS). | [Greensea IQ](https://greenseaiq.com/our-technology/) |

**Software Update Workflow (typical):**

1. Develop/test in simulation (Gazebo/Stonefish + ROS 2)
2. Deploy to vehicle computer via SSH/SCP or Docker push
3. No standard OTA (over-the-air) update mechanism for underwater vehicles
4. Updates typically done pre-mission while vehicle is on deck
5. Underwater: acoustic modems for telemetry only (bandwidth ~kbps), not software updates

### 6.2 Field Testing Workflows

1. **Pre-mission checklist:** Manual verification of all systems (battery, sensors, comms, propulsion)
2. **Launch and recovery:** Ship-based crane/A-frame or shore-based. Manual operation.
3. **Mission execution:** AUV follows pre-programmed waypoints; ROV teleoperated via tether
4. **Monitoring:** Surface vessel receives telemetry via acoustic modem (AUV) or fiber tether (ROV)
5. **Recovery:** Vehicle surfaces (AUV) or is winched up (ROV)
6. **Data download:** Physical connection (Ethernet/USB) to retrieve mission data

### 6.3 Gap: No CI/CD for Marine Robots

Unlike cloud/software engineering, there is no continuous integration/continuous deployment pipeline for underwater vehicle software. Every mission requires manual verification.

---

## 7. Phase 6: Feedback (Closing the Loop)

### 7.1 Post-Mission Analysis Tools

| Tool | Function | Source |
|------|----------|--------|
| **rosbag** (ROS) | Record and replay all ROS topics (sensor data, commands, state). De facto standard for ROS-based vehicles. | [ROS — rosbag2](https://docs.ros.org/) |
| **Neptus** (LSTS) | Mission planning + execution monitoring + post-mission analysis in one GUI. Supports AUV/ASV/UAV fleets. | [LSTS — Neptus](https://whale.fe.up.pt/neptus/info.html) |
| **MB-System** | Open-source bathymetric data processing from multibeam sonar. | [MB-System — GitHub](https://github.com/dwcaress/MB-System) |
| **DUNE logs** | Structured log format from DUNE middleware. Parseable for replay and analysis. | [LSTS — DUNE](https://github.com/LSTS/dune) |

### 7.2 How Real-World Data Flows Back into Design

**Current state (mostly manual):**

1. Engineer downloads sensor logs after mission
2. Compares sensor readings (IMU drift, sonar data, depth) to simulation predictions
3. Manually adjusts simulation parameters (drag coefficients, sensor noise models)
4. No automated parameter identification from real-world data
5. Design changes based on mission experience are communicated via documents/meetings, not through the CAD/simulation tools

**Emerging approaches:**

- **System identification from flight data:** Methods exist for aerial vehicles (PX4, ArduPilot) but not standardized for underwater
- **Digital twin synchronization:** Research-stage (see Section 5.4). No production tools.
- **Data-informed domain randomization:** Adjusts sim parameters based on real data distributions

### 7.3 The Broken Loop

The feedback loop is the **most broken** part of the lifecycle today:

| What Should Happen | What Actually Happens |
|-------------------|----------------------|
| Real-world sensor data automatically updates simulation parameters | Manual CSV/log analysis by engineers |
| Design changes are automatically validated against mission data | Design changes communicated via documents/meetings |
| Simulation fidelity improves continuously with each mission | Simulation parameters are rarely updated post-mission |
| Failure modes detected in the field automatically generate regression tests | Failure reports written manually, tests sometimes added |
| Component wear/aging data feeds back into design lifetime estimates | Component replacement based on fixed schedules, not condition monitoring |

---

## 8. Existing Closed-Loop Platforms

### 8.1 General Engineering Platforms

**Siemens Xcelerator:**
- CAD (NX) + Simulation (Simcenter, STAR-CCM+) + Test + Manufacturing + PLM
- Marine-specific: acquired FORAN (ship design software) for digital twin of ships
- Used by Arc Boat Company for electric boat design
- **Gap for underwater robots:** Focused on ships and surface vessels, not underwater vehicles. No underwater sensor simulation, no underwater autonomy.
- Source: [Siemens Marine Engineering](https://www.siemens.com/en-us/solutions/engineering-simulation/marine-engineering/); [Siemens acquires FORAN](https://www.eenewseurope.com/en/siemens-acquires-marine-digital-twin-software/)

**NVIDIA Omniverse + Isaac Sim:**
- Design collaboration (USD) + Physics simulation (PhysX) + Rendering (RTX) + Robotics (Isaac Sim) + Digital twin
- OceanSim built on top of Isaac Sim for underwater perception
- **Gap:** No marine-specific physics (hydrodynamics, buoyancy, added mass not built-in). OceanSim fills part of this but is research-stage, not production. No CAD integration. No deployment tools.
- Source: [OceanSim on Isaac Sim](https://arxiv.org/html/2503.01074v2); [NVIDIA Isaac Sim](https://docs.nvidia.com/isaacsim/)

**Ansys Twin Builder:**
- Digital twin platform for predictive maintenance
- Connects simulation models to real-time sensor data
- **Gap:** No underwater domain knowledge. No autonomy/robotics. Industrial IoT focus.
- Source: [Ansys — Twin Builder](https://www.ansys.com/products/digital-twin)

**PTC Windchill + Creo:**
- PLM + CAD + simulation management
- Strong in manufacturing lifecycle
- **Gap:** No simulation engine, no marine domain, no robotics

**Altair Cassini:**
- Unified Cloud Platform for simulation + CAD + PLM
- **Gap:** No marine/underwater domain, no robotics

### 8.2 Ocean-Specific Platforms

**Greensea OPENSEA:**
- The closest thing to a "marine robotics operating system"
- 20 years of development, deployed on commercial ROVs/AUVs worldwide
- Includes: navigation, control, perception, mission management
- Recent: OPENSEA Edge (edge computing), IQNS (GNCC solution), submerged C2 interface
- **Covers:** Deploy + partial Feedback phases
- **Does NOT cover:** Design, Simulate, Verify phases. No CAD integration. No GPU simulation.
- Source: [Greensea IQ](https://greenseaiq.com/our-technology/)

**EU Digital Twin of the Ocean (DTO):**
- Mercator Ocean International (EU-funded)
- Digital replica of the ocean itself (currents, temperature, salinity, ecosystems)
- NOT a vehicle design/simulation tool — it models the ocean environment
- Could be a data source for OceanScale's environmental simulation
- Source: [Digital Twin Ocean](https://digitaltwinocean.mercator-ocean.eu/); [EU DTO](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe/eu-missions-horizon-europe/restore-our-ocean-and-waters/european-digital-twin-ocean_en)

**ROS 2 Ecosystem:**
- Sim (Gazebo/Stonefish) + Test (SIL/HIL) + Deploy (ROS 2 runtime)
- **Covers:** Simulate + Test + Deploy phases
- **Does NOT cover:** Design (no CAD), Verify (no sim-to-real tools), Feedback (no automated parameter update)
- Fragmented: many separate packages, no unified workflow
- Source: [ROS 2](https://docs.ros.org/)

### 8.3 Summary: No One Covers the Full Loop

| Platform | Design | Simulate | Test | Verify | Deploy | Feedback |
|----------|--------|----------|------|--------|--------|----------|
| Siemens Xcelerator | Ships only | Ships only | Partial | No | No | Partial (PLM) |
| NVIDIA Omniverse | Partial (USD) | General robotics | Partial | Partial | No | No |
| Greensea OPENSEA | No | No | No | No | Yes | Partial |
| ROS 2 ecosystem | No | Partial | Partial | No | Yes | No |
| EU DTO | No | Ocean env only | No | No | No | Ocean data |
| **OceanScale (target)** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

---

## 9. Gap Analysis

### 9.1 Where the Loop Breaks (Ranked by Severity)

**Gap 1: No parametric design tool for underwater vehicles (CRITICAL)**

- Engineers cannot go from requirements ("500m AUV with side-scan sonar") to a parametric design
- Each vehicle is designed from scratch using general-purpose CAD tools
- Ship design has parametric tools (Maxsurf, NAPA); underwater robot design does not
- **What OceanScale must build:** A configuration-driven design system that generates vehicle geometry, hydrostatics, thruster sizing, and weight/balance from high-level requirements. Think "config.yaml → CAD-ready vehicle model."

**Gap 2: No high-fidelity GPU-accelerated underwater physics (CRITICAL)**

- Current simulators are CPU-based (Stonefish, UUV Simulator) or have limited physics (OceanSim uses PhysX, not a fluid solver)
- No real-time fluid-structure interaction
- No spectral wave modeling
- No spatially-varying current fields
- **What OceanScale must build:** GPU-accelerated hydrodynamics (Newton + Warp) with proper fluid-structure coupling, spectral sea states, and CFD-quality current fields at real-time rates.

**Gap 3: No sim-to-real verification pipeline (HIGH)**

- No automated method to validate simulation against real sensor data
- No automated parameter identification from mission data
- Domain randomization is manual and ad-hoc
- **What OceanScale must build:** A verification pipeline that ingests real mission data, compares against simulation, identifies parameter mismatches, and automatically calibrates simulation parameters.

**Gap 4: No automated testing framework for marine vehicles (HIGH)**

- No CI/CD equivalent for underwater robotics
- Each team builds custom test scripts
- No regression test suites
- **What OceanScale must build:** A test framework (like pytest but for marine robotics) with standard test cases for navigation, control, sensor processing, and mission logic. Supports SIL and HIL execution.

**Gap 5: No closed-loop feedback from field to simulation (HIGH)**

- Simulation parameters are rarely updated after real-world missions
- No automated sensor data replay into simulation
- No continuous model improvement
- **What OceanScale must build:** A feedback system that ingests mission logs, extracts simulation-relevant parameters, and updates the vehicle's digital twin automatically.

**Gap 6: No unified data model across phases (MEDIUM)**

- CAD uses STEP/IGES, simulation uses URDF/SDF, testing uses rosbag, deployment uses Docker/images
- No common data model that spans design → simulation → test → deployment
- **What OceanScale must build:** A unified vehicle model format that captures geometry, physics parameters, sensor configurations, and mission behavior — used across all lifecycle phases.

**Gap 7: No environmental data integration (MEDIUM)**

- Simulators use static, simplified environments
- Real ocean data (currents, temperature, salinity from DTO, Copernicus, NOAA) is not integrated
- **What OceanScale must build:** API integration with ocean data sources (EU DTO, Copernicus Marine Service) to generate simulation environments from real ocean conditions at specific locations and times.

### 9.2 OceanScale Minimum Viable Loop

To demonstrate a closed-loop platform, OceanScale should build these in order:

1. **Simulate (Phase 2)** — This is the core. GPU-accelerated underwater simulation with Newton + Warp. Already in progress.
2. **Test (Phase 3)** — SIL testing framework with standard test suites. Natural extension of simulation.
3. **Verify (Phase 4)** — Sim-to-real comparison tools. Data ingestion from mission logs. Parameter identification.
4. **Feedback (Phase 6)** — Automated parameter update from real-world data. Digital twin synchronization.
5. **Design (Phase 1)** — Parametric design tool. Largest scope; defer to post-MVP.
6. **Deploy (Phase 5)** — ROS 2 integration and deployment tooling. Leverage existing ecosystem rather than build from scratch.

### 9.3 Competitive Moat

OceanScale's moat is the **closed loop itself**. Individual phases have competitors:
- Simulate: Stonefish, HoloOcean, OceanSim
- Deploy: ROS 2, MOOS-IvP, Greensea
- Design: SolidWorks, Fusion, Orca3D

But **no one connects all phases** for underwater robotics. The data model that spans design → simulation → test → deployment → feedback creates switching costs and network effects. Each mission that feeds back into the simulation makes the platform more valuable.

---

## 10. Sources

All URLs cited in this document:

### Phase 1: Design
- [ResearchGate — Design Analysis of AUV Using CAD](https://www.researchgate.net/publication/334479028_Design_analysis_and_modelling_of_autonomous_underwater_vehicle_AUV_using_CAD)
- [Orca3D Official](https://orca3d.com/)
- [GHS — Creative Systems](https://ghsport.com/)
- [B-Series Propeller Generator](https://www.wageningen-b-series-propeller.com/)
- [U. Michigan — Wageningen B-Series Reference](https://deepblue.lib.umich.edu/bitstreams/2f3da55b-78ee-478c-b5f0-c3b8d33989e/download)
- [ResearchGate — Optimal Propeller Selection B-Series](https://www.researchgate.net/profile/Khanh-Ngo-2/publication/331518402_Optimal_Selection_of_Marine_Propellers_Based_on_Wageningen_B-Series/links/5c7de3cf299bf1268d3922a3/Optimal-Selection-of-Marine-Propellers-Based-on-Wageningen-B-Series.pdf)
- [Strathclyde — ML Propeller Design (2024)](https://strathprints.strath.ac.uk/93985/1/Tadros-etal-2024-A-unified-cross-series-marine-propeller-design-method-based-on-machine-learning.pdf)
- [MARIN — Wageningen B and C&D Series](https://www.marin.nl/api/marin/downloads/get-download/1484/0/82/block_files_mZ6qMHyLx7fVw/0kYequyDbpG8djoR8wFuCgoNXII=/R116_p26-27_%27Wageningen%2520B%2527%2520followed%2520by%2520the%2520future-ready%2520C&D-series.pdf)

### Phase 2: Simulate
- [Stonefish Documentation](https://stonefish.readthedocs.io/)
- [Stonefish GitHub](https://github.com/patrykcieslak/stonefish)
- [Stonefish ML Paper (2025)](https://arxiv.org/html/2502.11887v1)
- [HoloOcean 2.0 Paper](https://arxiv.org/html/2510.06160v1)
- [HoloOcean Docs](https://byu-holoocean.github.io/holoocean-docs/)
- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim)
- [OceanSim Paper](https://arxiv.org/html/2503.01074v2)
- [UUV Simulator GitHub](https://github.com/uuvsimulator/uuv_simulator)
- [DAVE GitHub](https://github.com/Field-Robotics-Lab/dave)
- [MarineGym Paper](https://arxiv.org/pdf/2503.09203)
- [arXiv:2504.06245 — Underwater Simulators Review](https://arxiv.org/html/2504.06245v1)
- [UCL FRL Simulator Review](https://frl-ucl.github.io/projects/underwater-robotic-simulators.html)
- [ROS Discourse — Simulator Comparison](https://discourse.openrobotics.org/t/simulating-a-ros2-biomimetic-auv-gazebo-dave-or-stonefish/52325)

### Phase 3: Test
- [HoloOcean HIL Paper](https://arxiv.org/abs/2511.07687)
- [ScienceDirect — HIL for Work-Class ROVs](https://www.sciencedirect.com/science/article/abs/pii/S0029801824029433)
- [DP-HIL Simulator](https://dynamic-positioning.com/wp-content/uploads/2025/12/control_johansen.pdf)
- [TREE-C — HIL for Marine](https://www.tree-c.nl/general/hardware-in-the-loop-simulators-for-marine-environments-explained/)

### Phase 4: Verify
- [IJRR 2024 — Sim-to-Real AUV Adaptive Control](https://journals.sagepub.com/doi/10.1177/02783649241272115)
- [Oregon State — Learning to Dock](https://research.engr.oregonstate.edu/rdml/sites/research.engr.oregonstate.edu.rdml/files/kevinchang2025revised.pdf)
- [arXiv:2510.20808 — Reality Gap in Robotics](https://arxiv.org/html/2510.20808v1)
- [DiVA Portal — Reducing Sim-to-Real Gap for Underwater Vehicles](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf)
- [MDPI — DDR for AUVs](https://www.mdpi.com/2076-3417/13/3/1723)
- [RSS 2024 — DrEureka](https://www.roboticsproceedings.org/rss20/p094.pdf)
- [NATO STO — Digital Twin Bridge](https://publications.sto.nato.int/publications/STO%20Meeting%20Proceedings/STO-MP-MSG-207/MP-MSG-207-20.pdf)
- [MDPI — DT for UUV Motion Prediction](https://www.mdpi.com/2077-1312/14/6/557)
- [Heriot-Watt — DT Below Surface](https://pure.hw.ac.uk/ws/portalfiles/portal/137253956/2402.07556v1.pdf)
- [EU Research Portal — Submarine DT](https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e517e1bd68&appId=PPGMS)
- [ScienceDirect — DT Pursuit-Evasion](https://www.sciencedirect.com/science/article/abs/pii/S1568494625003047)

### Phase 5: Deploy
- [MIT MOOS-IvP](https://oceanai.mit.edu/moos-ivp/)
- [LSTS GitHub](https://github.com/LSTS)
- [Frontiers — Interoperability Among UMVs](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2020.00091/full)
- [Greensea IQ — Technology](https://greenseaiq.com/our-technology/)
- [Greensea OPENSEA Edge](https://greenseaiq.com/news/greensea-launches-opensea-edge/)
- [Greensea IQNS](https://greenseaiq.com/bayonet-for-defense/software-systems/)
- [Open Toolkit for Underwater Field Robotics](https://www.researchgate.net/publication/398806539_An_Open_Toolkit_for_Underwater_Field_Robotics)

### Phase 6: Feedback
- [Taylor & Francis — Underwater DT Applications Review](https://www.tandfonline.com/doi/full/10.1080/27525783.2025.2605418)
- [arXiv — Digital Twin AI: Opportunities and Challenges](https://arxiv.org/html/2601.01321v1)
- [NVIDIA Docs — Isaac Sim Digital Twin](https://docs.nvidia.com/learning/physical-ai/going-further-with-robotics/latest/digital-twin-robotics/index.html)

### Phase 7: Existing Platforms
- [Siemens Marine Engineering](https://www.siemens.com/en-us/solutions/engineering-simulation/marine-engineering/)
- [Siemens — FORAN Acquisition](https://www.eenewseurope.com/en/siemens-acquires-marine-digital-twin-software/)
- [Siemens — Arc Boats](https://news.siemens.com/de-ch/siemens-xcelerator-arc-boat-company/)
- [Digital Twin Ocean — Mercator](https://digitaltwinocean.mercator-ocean.eu/)
- [EU Digital Twin Ocean](https://research-and-innovation.ec.europa.eu/funding/funding-opportunities/funding-programmes-and-open-calls/horizon-europe/eu-missions-horizon-europe/restore-our-ocean-and-waters/european-digital-twin-ocean_en)
- [Bain — Closed-Loop PLM](https://www.bain.com/insights/the-feedback-machine-the-magic-of-closed-loop-product-life-cycle-management-global-machinery-and-equipment-report-2024/)
- [HAL-Inria — Closed-Loop PLM Paper](https://inria.hal.science/hal-01764183/document)
- [PTC — What is PLM](https://www.ptc.com/en/technologies/plm)
- [Autodesk PLM](https://www.autodesk.com/industry/design-manufacturing/plm/plm-product-lifecycle-management)

### Phase 8: General
- [CAD Journal — Submarine Robot Virtual Prototype (2025)](https://www.cad-journal.net/files/vol_22/CAD_22(S7)_2025_270-284.pdf)
- [arXiv — Open Toolkit for Underwater Field Robotics](https://arxiv.org/html/2512.15597v1)
- [RoboNation — ARIEL AUV (RoboSub 2025)](https://robonation.org/app/uploads/sites/4/2025/07/RS25_TDR_University-of-Haifa-ANSFL.pdf)
- [U. Michigan Field Robotics — Research](https://fieldrobotics.engin.umich.edu/research)
- [Sciopen — Digital Twin in Marine Domain (2025)](https://www.sciopen.com/article/10.26599/OCEAN.2025.9470001)
