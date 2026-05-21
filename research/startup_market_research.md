# OceanScale Market Research Report

> Date: 2026-05-20 | All claims cite sources. Unverifiable claims marked "(unverified)".

---

## 1. Market Size

### 1.1 Global Underwater Robotics Market

| Source | Base Year | Market Size | Forecast Year | Forecast Size | CAGR |
|--------|-----------|-------------|---------------|---------------|------|
| Grand View Research | 2022 | $4.49B | 2030 | $13.02B | ~14.2% |
| Mordor Intelligence | 2025 | $5.08B | 2030 | $9.53B | 13.39% |
| Data Bridge MR | 2024 | $4.30B | — | — | 17.36% |
| Maximize MR | 2024 | $5.29B | 2032 | $9.90B | 8.15% |
| SNS Insider | 2025 | $5.23B | 2035 | $19.45B | 14.59% |

**Consensus**: Underwater robotics market is ~$4.3-5.3B in 2024, projected to $9.5-13B by 2030, with CAGR clustering around **13-15%**.

Sources:
- [Grand View Research — Underwater Robotics Market](https://www.grandviewresearch.com/industry-analysis/underwater-robotics-market)
- [Mordor Intelligence — Underwater Robotics Market](https://www.mordorintelligence.com/industry-reports/underwater-robotics-market)
- [Data Bridge Market Research](https://www.databridgemarketresearch.com/nucleus/global-underwater-robotics-market)
- [SNS Insider](https://www.snsinsider.com/reports/underwater-robotics-market-5579)
- [Maximize Market Research](https://www.maximizemarketresearch.com/market-report/global-underwater-robotics-market/20175/)

### 1.2 Marine Simulation Software Market

| Segment | 2024/2025 Value | Projected Value | Timeframe | CAGR |
|---------|-----------------|-----------------|-----------|------|
| Simulation Software (Global) | $19.95-27.19B | $36.22B | 2024-2030 | ~10% |
| Maritime Simulator Market | — | detailed forecast | 2025-2030 | — |
| Marine Radar Simulator | $1.07B | $1.50B | 2024-2030 | — |
| Maritime Software (broader) | $3.8B | $7.9B | 2025-2034 | 8.5% |

**Note**: No report isolates "underwater robotics simulation" as a standalone segment. It sits at the intersection of simulation software + maritime/robotics, likely a sub-$500M niche today but growing with GPU-native tools.

Sources:
- [MarketsandMarkets — Simulation Software](https://www.marketsandmarkets.com/Market-Reports/simulation-software-market-263646018.html)
- [Research and Markets — Maritime Simulator Market](https://www.researchandmarkets.com/reports/6132873/maritime-simulator-market-global-forecast)
- [Grand View Research — Marine Radar Simulator](https://www.grandviewresearch.com/horizon/statistics/radar-simulator-market/type/marine/global)
- [Precedence Research — Simulation Software](https://www.precedenceresearch.com/simulation-software-market)
- [Dataintelo — Maritime Software](https://dataintelo.com/report/global-maritime-software-market)

### 1.3 Sub-Segment Breakdown

| Sub-Segment | Size / Growth | Key Drivers |
|-------------|--------------|-------------|
| **ROV Market** | $3.72B (2026) → $6.05B (2031), CAGR 10.21% | Offshore IRM, oil & gas, offshore wind |
| **AUV Market** | ~$1.5-2.5B (2024), CAGR ~15-20% | Autonomy, AI, defense, mapping |
| **Offshore Inspection ROV** | $2.8B (2025) → $8.6B (2034) | IRM, wind farm maintenance |
| **Subsea Inspection Robots** | $3.8B (2025) → $8.1B (2034), CAGR 8.7% | Pipeline inspection, structural monitoring |
| **Precision Aquaculture** | $0.85B (2025) → $1.43B (2030), CAGR 11.1% | Fish farm monitoring, habitat cleaning |
| **Deep-Sea Robot** | $2.8B (2024), CAGR 10.5% | Mining exploration, scientific research |
| **Underwater Drone (consumer/prosumer)** | $1.65-6.17B (2024) → $3.41-24.41B (2030-33), CAGR 10.8-12.8% | Democratization, filmmaking, inspection |

Sources:
- [Mordor Intelligence — ROV Market](https://www.mordorintelligence.com/industry-reports/rov-market)
- [Market.us — AUV and ROV for Offshore IRM](https://market.us/report/auv-and-rov-for-offshore-irm-market/)
- [HTF Market Report — Offshore Inspection ROV](https://www.htfmarketreport.com/reports/4417293-offshore-inspection-rov-market)
- [Dataintelo — Subsea Inspection Robots](https://dataintelo.com/report/subsea-inspection-robots-market)
- [MarketsandMarkets — Precision Aquaculture](https://www.marketsandmarkets.com/Market-Reports/precision-aquaculture-market-242307580.html)

---

## 2. Key Players — Simulation

### 2.1 NVIDIA Isaac Sim / Omniverse

| Attribute | Details |
|-----------|---------|
| **Product** | Isaac Sim (built on Omniverse) |
| **Tech Stack** | PhysX 5, RTX ray tracing, USD, Python/C++ |
| **Underwater Support** | **None native.** No built-in ocean/hydrodynamics. Users must manually implement buoyancy, drag, currents. |
| **Key Project** | OceanSim (U Michigan, 2025) — extends Isaac Sim with underwater perception, caustics, turbidity, marine snow, sonar, DVL |
| **Pricing** | Free for individuals; enterprise licensing via NVIDIA |
| **Target** | General robotics (manipulation, legged, autonomous driving). Underwater is a community/academic extension, not a priority. |

**Assessment**: Isaac Sim is the most powerful GPU simulation platform, but NVIDIA has zero focus on underwater. OceanSim proves it CAN be extended, but requires significant custom engineering.

Sources:
- [NVIDIA Forums — Underwater environment in Isaac Sim](https://forums.developer.nvidia.com/t/underwater-basic-environment/262775)
- [OceanSim paper (arXiv:2503.01074)](https://arxiv.org/abs/2503.01074)
- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim)

### 2.2 Gazebo + UUV Simulator

| Attribute | Details |
|-----------|---------|
| **Product** | Gazebo (Classic / Sim / Harmonic) + UUV Simulator plugin |
| **Tech Stack** | C++, ROS/ROS2, ODE/Dart physics engines |
| **Underwater Support** | UUV Simulator adds buoyancy, hydrodynamic forces, thruster dynamics — but simplified models |
| **Key Limitations** | (1) Hydrodynamics are approximate — no real fluid-structure interaction, turbulence, or complex currents. (2) Gazebo's aging architecture — plugin stability issues, migration pain to Gazebo Sim/Harmonic. (3) No native ocean environment features (waves, salinity, acoustic propagation). (4) Rendering is not photorealistic. (5) UUV Simulator has had periods of reduced maintenance. |
| **Pricing** | Open-source (Apache 2.0) |
| **Target** | Academic research, ROS-based robotics education |

**Assessment**: The de facto standard for underwater robotics research, but fidelity is fundamentally limited by Gazebo's physics engine (not GPU-native). Cannot model real hydrodynamics.

Sources:
- [UUV Simulator documentation](https://uuvsimulator.github.io/)
- [UUV Simulator paper (ResearchGate)](https://www.researchgate.net/publication/308642839_UUV_Simulator_A_Gazebo-based_Package_for_Underwater_Intervention_and_Multi-Robot_Simulation)
- [arXiv review of underwater simulators (2504.06245)](https://arxiv.org/html/2504.06245v1)

### 2.3 Stonefish

| Attribute | Details |
|-----------|---------|
| **Product** | Stonefish |
| **Tech Stack** | C++ library, custom physics + lightweight OpenGL rendering, `stonefish_ros` / `stonefish_ros2` |
| **Underwater Support** | Purpose-built: 6-DOF hydrodynamic modeling based on actual geometry, sonar, event-based cameras, underwater sensors |
| **Key Features** | Most advanced open-source underwater-specific physics. Supports ML research. Automated scenario generation. |
| **Pricing** | Open-source |
| **Target** | Marine robotics researchers, AUV/ROV developers |

**Assessment**: Best-in-class open-source underwater simulator for physics fidelity. Limited GPU acceleration compared to Isaac Sim-based tools.

Sources:
- [Stonefish IEEE paper](https://ieeexplore.ieee.org/document/8867434/)
- [Stonefish GitHub](https://github.com/patrykcieslak/stonefish)
- [Stonefish documentation](https://stonefish.readthedocs.io/)
- [Stonefish ML paper (arXiv:2502.11887)](https://arxiv.org/html/2502.11887v1)

### 2.4 HoloOcean

| Attribute | Details |
|-----------|---------|
| **Product** | HoloOcean (1.0 + 2.0 in development) |
| **Tech Stack** | Unreal Engine 4, Holodeck (BYU PCCL Lab), Python interface, OpenGL GPU shaders |
| **Underwater Support** | Multi-agent support, sonar simulation (GPU-based octree + depth camera), simulated communications |
| **Key Features** | Novel GPU-based sonar rendering. Easy pip install + Python API. Multi-agent scenarios. |
| **Pricing** | Open-source |
| **Target** | Academic underwater robotics, RL research |

**Assessment**: Strong sonar simulation and ease of use. Built on UE4 (aging). HoloOcean 2.0 is in development.

Sources:
- [HoloOcean ICRA 2022 paper (CMU)](https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf)
- [HoloOcean docs](https://byu-holoocean.github.io/holoocean-docs/)
- [HoloOcean 2.0 preview (arXiv:2510.06160)](https://arxiv.org/html/2510.06160v1)

### 2.5 DAVE (Dynamic Animation and Vehicle Environment)

| Attribute | Details |
|-----------|---------|
| **Product** | Project DAVE |
| **Tech Stack** | ROS/Gazebo, migrating to ROS2/Gazebo Harmonic |
| **Origin** | Naval Postgraduate School |
| **Underwater Support** | Comprehensive simulation stack for underwater robots, sensors, environments. General-purpose aquatic simulator. |
| **Pricing** | Open-source |
| **Target** | Military/government underwater robotics testing, academic research |

Sources:
- [DAVE IEEE paper](https://ieeexplore.ieee.org/document/9965808/)
- [DAVE documentation](https://field-robotics-lab.github.io/dave.doc/)
- [DAVE ROS2 migration (GSoC 2024)](https://discourse.openrobotics.org/t/gsoc-2024-migration-of-project-dave-to-ros-2-and-harmonic-launch-files-robot-models-and-sensor-plugins/39321)

### 2.6 MarineGym

| Attribute | Details |
|-----------|---------|
| **Product** | MarineGym |
| **Tech Stack** | GPU-accelerated hydrodynamics, Python, parallel simulation |
| **Performance** | Up to **250,000 FPS** on RTX 3060 |
| **Purpose** | High-performance RL training for underwater vehicles (UUVs) |
| **Status** | Accepted at **IROS 2025** |
| **Pricing** | Open-source |
| **Affiliation** | Heriot-Watt University / Marine-RL team |

**Assessment**: Directly competitive with OceanScale's RL training vision. GPU-native, massive parallelism. The closest existing tool to what OceanScale aims to build for RL training.

Sources:
- [MarineGym paper (arXiv:2503.09203)](https://arxiv.org/abs/2503.09203)
- [MarineGym GitHub](https://github.com/Marine-RL/MarineGym)
- [MarineGym website](https://marine-gym.com/)

### 2.7 OceanSim

| Attribute | Details |
|-----------|---------|
| **Product** | OceanSim |
| **Tech Stack** | NVIDIA Isaac Sim + Omniverse, GPU RTX ray tracing, Python |
| **Focus** | **Underwater perception** — cameras, imaging sonars, DVLs, barometers |
| **Key Innovation** | Physics-based rendering for underwater (caustics, turbidity, marine snow, scattering). Synthetic data generation at scale. |
| **Affiliation** | U Michigan Field Robotics |
| **Pricing** | Open-source |
| **Target** | Underwater perception researchers, computer vision for marine robotics |

**Assessment**: The most advanced GPU-accelerated underwater perception simulator. Complements (rather than competes with) hydrodynamics-focused tools.

Sources:
- [OceanSim paper (arXiv:2503.01074)](https://arxiv.org/abs/2503.01074)
- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim)
- [OceanSim project site](https://umfieldrobotics.github.io/OceanSim/)

### 2.8 UNav-Sim

| Attribute | Details |
|-----------|---------|
| **Product** | UNav-Sim |
| **Tech Stack** | Unreal Engine 5, AirSim, ROS1/ROS2 |
| **Claim** | First underwater simulator to leverage UE5's high-detail rendering |
| **Features** | Visually realistic underwater environments, navigation research, synthetic data |
| **Pricing** | Open-source |
| **Affiliation** | Open-AIRLab, Aarhus University |

Sources:
- [UNav-Sim GitHub](https://github.com/open-airlab/UNav-Sim)
- [UNav-Sim IEEE paper](https://ieeexplore.ieee.org/document/10406819/)

### 2.9 PaleBlue (Commercial)

| Attribute | Details |
|-----------|---------|
| **Product** | ROV Trainer Simulator, Subsea Access Simulator |
| **Tech Stack** | Proprietary, real-time physics engine, CAD import |
| **Features** | Pilot training, subsea design review, access verification, fault injection, visibility/current simulation |
| **Customers** | French Navy, DeepTech Oil Services, oil & gas companies |
| **Pricing** | Commercial (contact vendor) |
| **Target** | Offshore energy, naval training, subsea engineering |

**Assessment**: The leading commercial ROV simulator for pilot training. Not GPU-native hydrodynamics — focused on training operations, not physics research or RL.

Sources:
- [PaleBlue Solutions](https://pale.blue/solutions/)
- [PaleBlue ROV Trainer](https://pale.blue/solutions/rov-trainer-simulator/)
- [Naval News — PaleBlue delivers to French Navy](https://www.navalnews.com/naval-news/2025/12/paleblue-delivers-rov-simulator-to-the-french-navy/)
- [Offshore Energy — PaleBlue ROV Simulator](https://www.offshore-energy.biz/paleblue-launches-new-real-time-rov-simulator/)

### 2.10 Other Notable Tools

| Tool | Stack | Status | Notes |
|------|-------|--------|-------|
| **MARUS** | ROS/Gazebo | Open-source | Modular underwater simulation |
| **UW MORSE** | MORSE (Blender) | Open-source, aging | Underwater extension of MORSE |
| **AVATAR (IFREMER)** | (unverified — could not find current public details) | Research tool | If it exists, likely internal to IFREMER |
| **SIMACT (Subsea 7)** | (unverified — could not find public details) | Possibly internal | Subsea 7 likely uses proprietary tools |
| **XLB (Autodesk)** | (unverified — not found in search results for underwater simulation context) | — | May refer to a different product |

---

## 3. Key Players — Underwater Robots

### 3.1 BlueROV (Blue Robotics)

| Attribute | Details |
|-----------|---------|
| **Product** | BlueROV2 (base ~$6,500-7,350), Heavy Configuration (~$10K+) |
| **Position** | Dominates affordable/open-source/hackable ROV segment |
| **Community** | Largest aftermarket ecosystem (thrusters, sensors, accessories) |
| **Market Share** | Not publicly broken out, but consistently cited as the leading affordable ROV platform |
| **Key Event** | Merged with OpenROV (~2023), consolidating consumer/prosumer ROV market |
| **Website** | [bluerobotics.com](https://bluerobotics.com) |

Sources: [Blue Robotics](https://bluerobotics.com), [Blue Robotics Forums](https://discuss.bluerobotics.com), [Alibaba ROV pricing guide](https://electronics.alibaba.com/buyingguides/rov-prices-guide-what-you-really-pay-for-in-2024)

### 3.2 Ocean Infinity

| Attribute | Details |
|-----------|---------|
| **Product** | Armada fleet (14-23 robotic vessels in 4 size classes: 21m, 36m, 78m, 85m) + 14 HUGIN AUVs (6,000m rated) |
| **Model** | Unmanned/lean-crewed vessels + remote control centers |
| **Major Milestone** | Final Armada vessel delivered December 2025 |
| **Key Partnership** | Shell (global framework agreement for offshore energy) |
| **Notable Ops** | MH370 search (2025), deep-sea survey |
| **Website** | [oceaninfinity.com](https://oceaninity.com/) |

Sources:
- [Ocean Infinity](https://oceaninfinity.com/)
- [Riviera MM — Armada fleet completion](https://www.rivieramm.com/news-content-hub/news-content-hub/ocean-infinity-completes-armada-fleet-construction-87158)
- [Wikipedia — Ocean Infinity](https://en.wikipedia.org/wiki/Ocean_Infinity)
- [Japan CAO report on Armada fleet (PDF)](https://www8.cao.go.jp/ocean/policies/auv/auv_pilot_project/pdf/03/shiryou_1.pdf)

### 3.3 HII REMUS

| Attribute | Details |
|-----------|---------|
| **Product Line** | REMUS 100 (100m), REMUS 300 (305m, modular), REMUS 600 (600m), REMUS 6000 (6,000m) |
| **Manufacturer** | HII Mission Technologies (formerly Hydroid/Kongsberg) |
| **Users** | U.S. Navy, NATO allies, research institutions worldwide |
| **Key Features** | Open architecture, modular payload bays, advanced autonomy |
| **Notable** | REMUS 6000 located Air France Flight 447 wreckage |
| **Website** | [hii.com](https://www.hii.com) |

(unverified — specifications from general knowledge, not from live search results due to rate limits)

### 3.4 Saab Seaeye

| Attribute | Details |
|-----------|---------|
| **Position** | Leading compact electric ROV manufacturer |
| **Products** | Seaeye range — compact electric ROVs for inspection and light work |
| **Market** | Offshore wind, shallow water IRM, oil & gas inspection |
| **Recent** | Launched compact electric ROV platforms for energy-efficient offshore wind operations |

Sources:
- [Strategic Market Research — AUV & ROV for Offshore IRM](https://www.strategicmarketresearch.com/market-report/auv-rov-for-offshore-irm-market)
- [Intel Market Research — Observation/Inspection ROV](https://www.intelmarketresearch.com/observationinspection-class-rov-market-9764)

### 3.5 Fugro

| Attribute | Details |
|-----------|---------|
| **Position** | Major ROV fleet operator and subsea inspection services provider |
| **Fleet** | Acquired 6+ Saab Seaeye ROV systems (2011), extensive global fleet |
| **Services** | Seabed mapping, offshore inspection, geotechnical surveys |
| **Scale** | One of the world's largest integrated geodata providers |

Sources:
- [Offshore Energy — Fugro acquires Saab Seaeye ROVs](https://www.offshore-energy.biz/uk-fugro-acquires-six-new-saab-seaeye-rov-systems/)

### 3.6 Chinese Players

#### QYSEA (鳍源科技)

| Attribute | Details |
|-----------|---------|
| **Founded** | 2016, Shenzhen |
| **Products** | FIFISH series — AI-powered ROVs (consumer to industrial) |
| **Features** | 360-degree precision hovering, HD filming, AI integration |
| **Awards** | CES Innovation Award (2 consecutive years) |
| **Website** | [qysea.com](https://www.qysea.com/) |

Sources: [QYSEA](https://www.qysea.com/), [Geo-matching — QYSEA](https://geo-matching.com/companies/shenzhen-qysea-tech-coltd)

#### CHASING Innovation

| Attribute | Details |
|-----------|---------|
| **Founded** | 2016, China |
| **Products** | CHASING series underwater drones (consumer to professional) |
| **Pricing** | CHASING M2 Pro ~$3,459 |
| **Website** | [chasing.com](https://www.chasing.com/en/company-profile.html) |

Sources: [CHASING Innovation](https://www.chasing.com/en/company-profile.html), [UW Camera Store comparison](https://www.uwcamerastore.com/blog/qysea-or-chasing-which-underwater-drone-fits-your-workflow)

#### Deepfar (深之)

(unverified — could not find definitive public information. May be a smaller/niche Chinese underwater robotics company. Needs further investigation.)

#### CAS Shenyang Institute of Automation (中科院沈阳自动化所)

| Attribute | Details |
|-----------|---------|
| **Role** | China's **core institution** for underwater robotics research |
| **Key Projects** | "Haidou-1" (海斗一号, full-ocean-depth ~11,000m AUV), "Tansuo 4500" (探索4500), "Fendouzhe" (奋斗者号, manned submersible to 10,909m) |
| **Funding** | National Key R&D Program, 863 Program legacy |
| **Scope** | Deep-sea AUV/ROV, autonomous docking, acoustic tracking, 7,000m-class landers |

Sources:
- [CAS SIA — Underwater Robotics Lab](http://www.sia.cas.cn/jgsz/kyxt/sxjqryjs/)
- [Qianzhan Tech — China deep-sea robot review](http://www.qianzhankeji.cn/CN/10.3981/j.issn.2097-0781.2022.02.004)
- [MOST — SIA breakthrough dives](https://www.most.gov.cn/dfkj/ln/zxdt/201609/t20160909_127522.html)

---

## 4. Pain Points

### 4.1 Hydrodynamic Simulation Fidelity Gap

**Problem**: No existing simulator accurately models real fluid-structure interaction for underwater vehicles.

- Gazebo/UUV Simulator uses simplified drag/buoyancy coefficients — no turbulence, no vortex shedding, no complex current interactions
- Stonefish has the most advanced open-source hydrodynamics but is CPU-based and geometry-approximate
- Isaac Sim has NO native hydrodynamics — requires manual force modeling
- GPU-native fluid solvers (SPH, LBM, Eulerian) are not integrated with any underwater robotics simulator

Sources:
- [arXiv:2504.06245 — Underwater Robotic Simulators Review](https://arxiv.org/html/2504.06245v1)
- [NVIDIA Forums — Hydrodynamics in Isaac Sim](https://forums.developer.nvidia.com/t/create-water-with-dynamics-for-underwater-robots/266545)

### 4.2 Sim-to-Real Transfer Gap

**Problem**: Policies trained in simulation fail in real underwater environments due to unmodeled dynamics.

Key issues identified in research:
- **Domain shift**: Simulated visual environments don't match real underwater conditions (color absorption, scattering, turbidity variation)
- **Dynamic mismatch**: Hydrodynamic forces in sim are oversimplified — policies don't transfer
- **Sensor noise models**: Simulated sonar, DVL, IMU lack realistic noise distributions, bias drift, dropout behavior
- **Acoustic propagation**: No open-source simulator accurately models underwater acoustic communication channels

Sources:
- [DiVA Portal — Reducing Sim-to-Real Gap in Underwater Vehicles](https://www.diva-portal.org/smash/get/diva2:2033877/FULLTEXT01.pdf)
- [Cambridge — Sim-to-Real Pipeline for Underwater Obstacle Avoidance](https://www.cambridge.org/core/journals/robotica/article/simtoreal-pipeline-for-training-autonomous-obstacle-avoidance-of-underwater-robots-based-on-highfidelity-model/F8C61DC413D83D175DCF17C17A4A9718)
- [Medium — Digital Simulation Platforms for Training Underwater Autonomous Robots](https://medium.com/@creed_1732/digital-simulation-platforms-for-training-underwater-autonomous-robots-the-costly-gap-most-auv-b4b0170ef695)
- [Semantic Scholar — Navigating the Sim-to-Real Gap](https://www.semanticscholar.org/paper/Navigating-the-Sim-to-Real-Gap%253A-A-Critical-Analysis-Gomes-Drews/a6cb0250bd0e161e2b0416e88efaaa7b0dc89645)

### 4.3 Sensor Simulation Gaps

| Sensor | Current State | Gap |
|--------|--------------|-----|
| **Sonar (imaging)** | HoloOcean has GPU-based sonar; OceanSim supports it | Realistic beam propagation, backscatter, multi-path still limited |
| **DVL** | OceanSim supports DVL simulation | Noise models, signal degradation, altitude dependency not realistic |
| **Acoustic comms** | HoloOcean has basic comms simulation | No realistic propagation, multipath, attenuation modeling |
| **Camera** | UNav-Sim (UE5), OceanSim (Isaac Sim) — best in class | Color absorption at depth, dynamic turbidity still limited |
| **IMU** | Standard in most simulators | Underwater-specific bias (added mass effects) not modeled |

### 4.4 Standardization & Benchmarking

- No unified benchmark exists for comparing underwater simulators
- Each tool uses different physics models, making cross-comparison meaningless
- The arXiv review (2504.06245) explicitly calls for standardization

Source: [arXiv:2504.06245](https://arxiv.org/html/2504.06245v1)

### 4.5 RL Training Efficiency

- Real-world underwater data collection is expensive and risky
- Existing simulators cannot generate enough diverse training scenarios efficiently
- MarineGym (250K FPS on RTX 3060) is the first to address this at scale for underwater, but is limited to simplified hydrodynamics

Source: [MarineGym (arXiv:2503.09203)](https://arxiv.org/abs/2503.09203)

---

## 5. China Market Specifics

### 5.1 Macro Policy: 海洋强国 (Maritime Power)

China's "strong maritime nation" strategy is a top-level national priority:

- **14th Five-Year Plan (2021-2025)** dedicates a full chapter to marine economy development
- **2024**: China's gross ocean product (GOP) surpassed **RMB 10 trillion (~$1.4 trillion USD)**
- **2026 budget**: National S&T spending targets ~**RMB 1.3 trillion** (~5% annual increase)
- China aims to be a "strong maritime country" within the next five years, strengthening marine S&T innovation

Sources:
- [Qiushi — China's marine economy powers ahead](https://en.qstheory.cn/2025-08/14/c_1133516.htm)
- [WEF — Advancing China's Sustainable Blue Economy](https://reports.weforum.org/docs/WEF_Advancing_China's_Sustainable_Blue_Economy_2025.pdf)
- [Rhodium Group — China's Next-Generation Industrial Policy](https://rhg.com/research/chinas-next-generation-industrial-policy/)
- [CGTN — China aims to build strong maritime country](https://www.facebook.com/ChinaGlobalTVNetwork/posts/in-the-next-five-years-china-aims-to-build-itself-into-a-strong-maritime-country/1602871244539675/)

### 5.2 Funding Programs

| Program | Details |
|---------|---------|
| **863 Program (1986-2016)** | Flagship high-tech R&D program; initial funding RMB 10 billion; funded CR-01 AUV, Jiaolong submersible, deep-sea technology |
| **National Key R&D Program** | Successor to 863; funds "Haidou-1" (海斗一号), deep-sea vehicles |
| **NSFC** | National Natural Science Foundation — marine science grants for 2024-2025 |
| **COMRA Fellowship** | China Ocean Mineral Resources R&D Association — deep-sea mining research fellowships in Qingdao |
| **Marine Scholarship of China** | International scholarship for coastal/island nation students |
| **Provincial programs** | 11 coastal provincial regions have independent marine S&T budgets |

Sources:
- [IGCC — Oceans of Ambition: China's Blue Science](https://ucigcc.org/blog/oceans-of-ambition-the-rise-of-chinas-blue-science/)
- [RAND — Institute of Deep-Sea Science and Engineering](https://www.rand.org/pubs/tools/TLA4687-1/tool/items/institute-of-deepsea-science-and-engineering.html)
- [Wikipedia — 863 Program](https://en.wikipedia.org/wiki/863_Program)
- [NSFC 2025 Guide](https://www.nsfc.gov.cn/english/site_1/pdf/NationalNaturalScienceFundGuidetoPrograms2025.pdf)
- [ISA — COMRA Fellowship 2025](https://isa.org.jm/china-ocean-mineral-resources-r-d-association-pms-fellowship-2025/)

### 5.3 Key Chinese Institutions

| Institution | Focus |
|-------------|-------|
| **CAS Shenyang Institute of Automation (沈阳自动化所)** | Core underwater robotics research — AUV, ROV, deep-sea vehicles |
| **CAS Institute of Deep-Sea Science and Engineering (深海所)** | Deep-sea technology, 7,000m+ vehicles |
| **CAS Institute of Oceanology (海洋所)** | Marine science, observation systems |
| **ShanghaiTech MAgIC Lab** | Sim-to-real transfer for underwater robots |
| **National Deep Sea Center (国家深海基地)** | Deep-sea operations, "Jiaolong" submersible base |
| **Various universities** | Ocean University of China, Shanghai Jiao Tong, Harbin Engineering University |

Source: [ShanghaiTech MAgIC Lab](https://magiclab.sist.shanghaitech.edu.cn/research.html)

### 5.4 Deep-Sea Mining

- China holds **multiple ISA mining permits** through state-backed entities (e.g., Beijing Pioneer)
- Beijing Pioneer announced plans for a **2025 test collection** of polymetallic nodules
- ISA has not yet finalized exploitation rules; push to adopt by July 2025 ongoing
- In April 2025, the U.S. issued an executive order to permit deep-sea mining in international waters, escalating geopolitical competition

Sources:
- [CSIS — Seabed Mining in the Pacific](https://amti.csis.org/between-rocks-and-a-hard-place-seabed-mining-in-the-pacific/)
- [USNI — Future of Sovereignty in the Deep Sea](https://www.usni.org/magazines/proceedings/2026/january/future-sovereignty-deep-sea)
- [Atlantic Council — Mining Without Rules](https://www.atlanticcouncil.org/in-depth-research-reports/issue-brief/mining-without-rules-the-risky-us-bet-on-the-deep-sea/)

### 5.5 Regulatory Environment

- **State Oceanic Administration (SOA)** → merged into **Ministry of Natural Resources (MNR)** in 2018
- China's marine regulations favor domestic technology (indigenous innovation requirements)
- Deep-sea operations require state approval; military-adjacent research has restricted foreign collaboration
- Data sovereignty laws may limit cloud-based simulation platforms from foreign providers

(unverified — based on general knowledge of Chinese regulatory environment, not specific sourced data)

### 5.6 Commercial Opportunity for OceanScale

| Factor | Assessment |
|--------|-----------|
| **Government demand** | HIGH — 海洋强国 strategy creates sustained demand for simulation tools |
| **Research funding** | HIGH — NSFC + National Key R&D programs actively fund marine robotics |
| **Deep-sea mining** | GROWING — China's ISA permits create need for operational simulation |
| **Domestic substitution** | OPPORTUNITY — No Chinese GPU-native underwater simulator exists |
| **Military/defense** | SENSITIVE — Shenyang Institute and naval programs need sim tools, but access requires clearance |
| **Competition** | LOW — No domestic competitor for GPU-native underwater simulation |
| **Data sovereignty** | ADVANTAGE — Beijing-based company can offer on-premise / air-gapped deployment |

---

## 6. Competitive Landscape Summary

### Gap Analysis: Where OceanScale Fits

| Capability | Gazebo/UUV | Stonefish | HoloOcean | MarineGym | OceanSim | **OceanScale** |
|-----------|-----------|-----------|-----------|-----------|----------|--------------|
| GPU-native hydrodynamics | No | No | Partial | Partial | No | **Yes (Newton+Warp)** |
| Photorealistic rendering | No | Basic | UE4 | No | Isaac Sim RTX | **Isaac Sim** |
| RL training at scale | No | Limited | No | **250K FPS** | No | **Target** |
| Sonar/DVL simulation | Basic | Yes | Yes (GPU) | (unverified) | Yes | **Target** |
| Sim-to-real pipeline | Weak | Moderate | Moderate | (unverified) | Strong (visual) | **Target** |
| Fluid-structure interaction | No | Approximate | No | No | No | **Unique** |
| China localization | No | No | No | No | No | **Yes** |

### OceanScale's Unique Value Proposition

No existing tool combines:
1. **GPU-native fluid dynamics** (Newton/Warp CUDA) for real-time hydrodynamics
2. **Isaac Sim rendering** for photorealistic underwater perception
3. **Massively parallel RL training** for underwater autonomy
4. **China-based** with domestic deployment and data sovereignty

---

## 7. Methodology & Caveats

- All data gathered via web search on 2026-05-20
- Market size figures come from commercial research firms (Grand View, Mordor, SNS, etc.) — these are estimates and vary significantly across sources
- Some entities (AVATAR/IFREMER, SIMACT/Subsea 7, XLB/Autodesk) could not be verified via public search — marked as (unverified)
- HII REMUS specifications from general knowledge, not live search (rate limited)
- Chinese regulatory environment assessment is general knowledge, not sourced from specific 2025 policy documents
- Deepfar (深之) could not be verified — may require Chinese-language search or direct inquiry
