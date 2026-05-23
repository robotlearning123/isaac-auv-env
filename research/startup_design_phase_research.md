# OceanScale Design Phase Research

**Date:** 2026-05-20
**Scope:** How underwater robots are designed today, gaps in tooling, and what OceanScale should build to automate/simulate the design phase of the closed-loop ocean engineering lifecycle.

---

## 1. Current Design Workflows

### 1.1 AUVs (Autonomous Underwater Vehicles)

**Software tools used:**
- **MATLAB & Simulink** — end-to-end modeling, simulation, control design, trade studies. The dominant commercial platform for interdisciplinary AUV design integration.
  - Source: [MathWorks AUV Solutions](https://www.mathworks.com/solutions/aerospace-defense/auv.html)
- **Gazebo Harmonic + ROS 2 + ArduPilot** — open-source digital twin workflow for AUV simulation and control.
  - Source: [Instagram community demo](https://www.instagram.com/p/DOb3PrhEigy/)
- **SOLIDWORKS** — mechanical/structural design of frames, pressure vessels, thruster housings.
  - Source: [GrabCAD ROV models](https://grabcad.com/library/software/solidworks/tag/rov)
- **Orca3D (Rhino 3D plugin)** — hull design, hydrostatics, stability, marine CFD, speed/power prediction.
  - Source: [Orca3D official](https://orca3d.com/)
- **OpenFOAM / ANSYS Fluent** — CFD for hull drag optimization and flow analysis.
  - Source: [ScienceDirect hull optimization](https://www.sciencedirect.com/science/article/abs/pii/S0029801824025940)

**Key design decisions:**
- Hull shape (torpedo, glider, sphere, flatfish) — dictates drag coefficient and mission profile
- Thruster type and count — tunnel thrusters vs rim-driven vs propeller
- Battery chemistry and capacity — lithium-ion vs hydrogen fuel cell; determines endurance
- Sensor suite — sonar, camera, DVL, IMU, pressure, acoustic modem
- Materials — titanium, aluminum, carbon fiber, 3D-printed composites
- Pressure rating — depth rating drives hull thickness, O-ring design, connector selection

**Timelines and iterations:**
- Academic/research AUVs: 1-3 years, 2-4 prototype iterations before water
- Industry AUVs: 3-5 years, 5+ iterations with structured testing pipeline
- ITB (Institute of Technology Bandung) went through 3+ prototypes since 2001
- Fraunhofer recommends structured tool chain: rapid prototyping -> simulation -> water testing -> mission-ready
  - Source: [Fraunhofer AUV testing tool chain](https://publica.fraunhofer.de/bitstreams/55283b5d-6393-47b8-811c-252ecc976325/download)
  - Source: [ITB underwater vehicle development](https://www.academia.edu/173897/Design_Development_and_Testing_of_Underwater_Vehicles_ITB_Experience)

**Costs:**
- Low-cost research AUV: $400-$5,000 (small scale, 3D-printed)
  - Source: [Cost-effective AUV design](https://www.researchgate.net/publication/304022131_Design_of_a_cost-effective_autonomous_underwater_vehicle)
- Medium-cost open-source AUV (MeCO): ~$10,000-$50,000
  - Source: [MeCO open-source AUV](https://arxiv.org/html/2503.10928v1)
- Deep-sea rated AUV: $500,000-$5,000,000+ (titanium hull, certified connectors, pressure-rated electronics)
  - Source: [10,000m class AUV](https://www.mdpi.com/2077-1312/12/11/2097)
- Hydrogen fuel cell AUV life-cycle cost analysis available
  - Source: [Life cycle cost of hydrogen fuel cell AUV](https://www.sciencedirect.com/science/article/pii/S0029801824006371)

### 1.2 ROVs (Remotely Operated Vehicles)

**Software tools:**
- **SOLIDWORKS** — detailed mechanical design of frames, pressure housings, thruster modules
  - Source: [ResearchGate ROV 3D model](https://www.researchgate.net/figure/3D-model-of-ROV-created-by-SOLIDWORKS-ROV-Remotely-operated-vehicle_fig1_381761380)
- **Orca3D** — hydrostatics, stability for ROV buoyancy calculations
- **MATLAB/Simulink** — 6-DOF dynamics modeling, control design
  - Source: [6-DOF ROV hydrodynamic model](https://www.researchgate.net/publication/276184856_Modelling_Design_and_Robust_Control_of_a_Remotely_Operated_Underwater_Vehicle)

**Design decisions:** Open-frame vs closed-frame, tether management, manipulator arms, tooling skids, camera/lighting placement, thruster vectoring configuration.

**Key difference from AUV:** ROVs have tether (power + data), so endurance is unlimited but tether drag and management become primary design constraints.

### 1.3 USVs (Unmanned Surface Vessels)

Less specialized tooling — often designed with standard naval architecture software:
- **Maxsurf** / **Rhino + Orca3D** for hull design
- **SOLIDWORKS/Onshape** for structural design
- Standard marine propulsion sizing tools

### 1.4 Deep-Sea Mining Robots

- **Impossible Metals** uses simulated ocean environments to test AI, cameras, robotic arms for selective mineral harvesting. Their Eureka I AUV completed proof-of-concept trial.
  - Source: [Impossible Metals Eureka](https://impossiblemetals.com/technology/robotic-collection-system/)
  - Source: [YouTube: Impossible Metals testing](https://www.youtube.com/watch?v=Cnemmtm5PqI)
- Design constraints: extreme depth rating, mineral collection mechanism, selective harvesting AI, launch and recovery systems.

### 1.5 Underwater Inspection Robots

- Focus on sensor integration (cameras, sonar, NDT sensors), path planning for coverage
- Marine cable inspection robots face accuracy, reliability, and efficiency challenges
  - Source: [Marine cable inspection robot](https://www.sciencedirect.com/science/article/pii/S2667241325000151)

---

## 2. Parametric Design Tools and Automation

### 2.1 Parametric Hull Design

**DEKC Maritime** describes parametric hull models as 3D hull shapes alterable by single numerical parameters. This is the foundation for automated design space exploration.
- Source: [DEKC: The Many Faces of Parametric Hull Design](https://www.dekc-maritime.com/news2/the-many-faces-of-parametric-hull-design)

**Comprehensive review** of parametric models, numerical codes, and surrogate modeling for hull form optimization available from ULiege.
- Source: [ULiege parametric hull optimization thesis](https://matheo.uliege.be/bitstream/2268.2/22262/4/Smit_DHANANI_Master_Thesis_Report.pdf)

### 2.2 Automated Hull Shape Optimization

Multiple approaches exist:

| Method | Source | Key Result |
|---|---|---|
| **Bayesian Optimization (BO-LCB)** | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0029801824021152) | Most sample-efficient for UUV design; first study applying BO + DNN surrogates |
| **Constrained BO** | [ACM](https://dl.acm.org/doi/10.1145/3576914.3587530) / [arXiv](https://arxiv.org/abs/2302.14732) | Constraint handling in hull design optimization; [GitHub repo](https://github.com/vardhah/ConstraintBOUUVHullDesign) |
| **AI + CFD** | [arXiv](https://arxiv.org/pdf/2302.09441) | Search for universal minimum drag underwater vehicle |
| **Automated drag platform** | [Hrcak](https://hrcak.srce.hr/file/478526) | 2D unstructured mesh + adaptive methods |
| **MDO (Multidisciplinary)** | [NASA ADS](https://ui.adsabs.harvard.edu/abs/2017dcab.conf...51C/abstract) | Energy consumption minimization via hull decomposition |
| **Bio-inspired** | [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/19942060.2021.1940287) | Humpback whale-inspired hull shapes |
| **Open-source optimization** | [GitHub: UUV-design-optimization](https://github.com/vardhah/UUV-design-optimization) | BO + deep learning surrogate code available |

### 2.3 Thruster Sizing and Design

- **Tunnel thruster geometry** adaptation for over-actuated AUV platforms
  - Source: [MDPI tunnel thruster design](https://www.mdpi.com/2077-1312/12/11/2021)
- **Parametric design of hubless rim-driven thrusters** for small marine vehicles
  - Source: [KCI rim-driven thruster](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003323299)
- **Reconfigurable thruster configuration estimation** algorithms exist
  - Source: [Springer: Thruster configuration estimation](https://link.springer.com/chapter/10.1007/978-3-642-28572-1_45)
  - Source: [MIT DSpace: Thruster config estimation](https://dspace.mit.edu/handle/1721.1/137112)

### 2.4 Battery/Endurance Calculators

- AUV endurance = onboard energy / (propulsion power + hotel load + sensor power)
- Endurance is driven by hull drag (speed^2 to speed^3 relationship), battery chemistry, and sensor power budget
  - Source: [AUV design considering energy](https://eprints.soton.ac.uk/466611/1/1230769.pdf)
- No standardized open-source endurance calculator found — each team builds their own spreadsheet or MATLAB script.

---

## 3. USD/OpenUSD for Robot Design

### 3.1 How USD Works for Robot Assembly

NVIDIA's "Anatomy of a Robot Asset Structure" documentation describes the standard approach:
- **Prims** are the fundamental building blocks (rigid bodies, joints, collision geometries, visual meshes)
- Physics properties (mass, inertia, collision) are authored as USD attributes on prims
- Joints connect rigid body prims with defined axes, limits, and drive parameters
- Robot asset hierarchy: root Xform -> body prims with RigidBody API -> joint prims with Joint API
  - Source: [NVIDIA: Anatomy of a Robot Asset Structure](https://docs.nvidia.com/learning/physical-ai/going-further-with-robotics/latest/best-practices-for-robotics-and-openusd/01-anatomy-of-robot-asset-structure.html)
  - Source: [OpenUSD Rigid Body Physics Proposal](https://openusd.org/dev/wp_rigid_body_physics.html)

### 3.2 URDF to USD Conversion

The standard pipeline is:
1. Design robot in CAD (Onshape, SOLIDWORKS, Fusion 360)
2. Export as URDF (using CAD plugins or manual authoring)
3. Import URDF into Isaac Sim using built-in URDF Importer extension
4. Importer converts URDF to USD with physics properties preserved
5. Result is a USD asset usable in Isaac Sim and Isaac Lab

**Key limitation:** URDF -> USD is well-supported. USD -> URDF is NOT supported (unidirectional).
  - Source: [Isaac Sim URDF import docs](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html)
  - Source: [Isaac Lab import new asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)
  - Source: [NVIDIA Forum: USD to URDF not supported](https://forums.developer.nvidia.com/t/cannot-convert-usd-to-urdf/359049)
  - Source: [YouTube: CAD to URDF to Isaac Sim](https://www.youtube.com/watch?v=KCHmYvYF_6c)

### 3.3 Newton's USD Support

Newton has deep USD integration via `ModelBuilder.add_usd()`:

```python
from newton import ModelBuilder
builder = ModelBuilder()
builder.add_usd(source="robot.usda")
model = builder.finalize()
```

**Custom attributes in USD for Newton:**
Newton supports custom attributes authored in USD with a declaration-first pattern:
- Declare on `PhysicsScene` prim with metadata specifying assignment (model/state/control/contact) and frequency (body/shape/joint/joint_dof/joint_coord/articulation)
- Assign on individual prims with override values
- Access in Python via `model.attr_name.numpy()` or namespaced `control.namespace_a.attr.numpy()`

Supported types: float, bool, int, float2, float3, float4, quatf -> mapped to Warp types.

This means parameters like mass, inertia, thruster positions, hydrodynamic coefficients CAN be exposed as USD custom attributes and read by Newton.
  - Source: [Newton USD parsing docs](https://github.com/newton-physics/newton/blob/main/docs/concepts/usd_parsing.md)
  - Source: [Newton custom attributes docs](https://github.com/newton-physics/newton/blob/main/docs/concepts/custom_attributes.md)
  - Source: [Newton Physics official docs](https://newton-physics.github.io/newton/stable/)
  - Source: [NVIDIA Blog: Newton + OpenUSD](https://blogs.nvidia.com/blog/newton-physics-engine-openusd/)

### 3.4 Newton Model Building API

Newton provides `ModelBuilder` for programmatic robot construction:
- `add_urdf()` — load URDF files directly (auto-parses kinematic structure, collision geometry, inertial properties)
- `add_usd()` — load USD files with custom attributes
- `add_ground_plane()` — add environment geometry
- Joint configuration: `default_joint_cfg.armature`, `target_ke`, `target_kd`, shape friction
- Kinematic utilities: `eval_fk()`, `eval_ik()`, `eval_jacobian()`, `eval_mass_matrix()`
  - Source: [Newton GitHub repo](https://github.com/newton-physics/newton)

---

## 4. Configuration-Driven Design Approaches

### 4.1 ROS 2 Robot Description (URDF/Xacro)

The ROS ecosystem uses URDF (Unified Robot Description Format) as the canonical robot definition:
- XML-based, defines links (rigid bodies) and joints (connections)
- **Xacro** extends URDF with macro substitution and parameterization — enables config-driven design
- ROS 2 Humble/Jazzy uses URDF for robot state publisher, MoveIt, Nav2
  - Source: [Nav2 URDF setup](https://docs.nav2.org/setup_guides/urdf/setup_urdf.html)
  - Source: [ROS 2 URDF tutorial](https://automaticaddison.com/create-and-visualize-a-mobile-robot-with-urdf-ros-2-jazzy/)

### 4.2 Isaac Lab Configuration System

Isaac Lab migrated from YAML-driven configs (IsaacGymEnvs) to **Python-based configuration classes** using `configclass`:
- Robot and environment definitions are Python dataclasses
- Supports both direct Python API and YAML entry points for direct environments
- Deployment configs still use YAML (`env.yaml` for environment, another for neural network)
  - Source: [Isaac Lab migration from IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
  - Source: [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
  - Source: [Isaac Sim policy deployment](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/isaac_lab_tutorials/tutorial_policy_deployment.html)

### 4.3 Modular Underwater Vehicle Platforms

- **Modularis** — open-source modular underwater robot testbed for rapid perception/control algorithm testing
  - Source: [Modularis arXiv](https://arxiv.org/html/2401.06243v1)
- **SeaDrone** — modular reconfigurable underwater robotic system
  - Source: [IEEE SeaDrone](http://ieeexplore.ieee.org/document/6964420/)
- **Dynamically reconfigurable AUV** — can change geometric configuration of actuators during operation
  - Source: [MDPI Sensors](https://www.mdpi.com/1424-8220/22/9/3379)
- **SE(3) Modular Thruster Control Algorithm** — supports arbitrary number of thrusters and configurations
  - Source: [ICRA 2010 PDF](https://fileadmin.cs.lth.se/ai/Proceedings/ICRA2010/MainConference/data/papers/1835.pdf)

### 4.4 PX_IsaacSim_URDF_Importer

A ROS 2 package for URDF/xacro to USD conversion pipeline with:
- Sensor integration (cameras, LiDARs)
- Joint configuration management
- Configuration-driven workflow
  - Source: [PX_IsaacSim_URDF_Importer GitHub](https://github.com/ProximityRobotics/PX_IsaacSim_URDF_Importer)

---

## 5. Design Optimization Methods

### 5.1 CFD-Based Hull Shape Optimization

The state of the art in 2024-2025:
- **Full turbulent continuous CFD** for simultaneous nose and tail optimization
  - Source: [ScienceDirect AUV hull optimization](https://www.sciencedirect.com/science/article/abs/pii/S0029801824025940)
- **Ducted propeller integration** with hull shape optimization
  - Source: [Academia.edu AUV shape + ducted propeller](https://www.academia.edu/23842045/Shape_optimization_of_an_autonomous_underwater_vehicle_with_a_ducted_propeller_using_computational_fluid_dynamics_analysis)

### 5.2 Multi-Objective Optimization for AUVs

- **Speed vs endurance vs depth** trade-offs formalized in MDO frameworks
  - Source: [Texas A&M biomimetic AUV MDO](https://oaktrust.library.tamu.edu/server/api/core/bitstreams/52845cff-3db3-4859-8ecb-860b71083929/content)
- **Energy consumption minimization** via hull decomposition into optimized sections
  - Source: [NASA ADS MDO of AUV](https://ui.adsabs.harvard.edu/abs/2017dcab.conf...51C/abstract)

### 5.3 Bayesian Optimization in Robot Design

**BO-LCB (Bayesian Optimization - Lower Confidence Bound)** is the most sample-efficient framework for UUV design optimization, according to a 2024 Ocean Engineering study — the first to apply BO + DNN surrogates to UUV design.
  - Source: [ScienceDirect BO for UUV](https://www.sciencedirect.com/science/article/pii/S0029801824021152)
  - Source: [Open-source code](https://github.com/vardhah/UUV-design-optimization)

Constrained BO handles engineering constraints (volume, stability, manufacturing) alongside the optimization objective.
  - Source: [ACM Constrained BO](https://dl.acm.org/doi/10.1145/3576914.3587530)
  - Source: [GitHub: ConstraintBOUUVHullDesign](https://github.com/vardhah/ConstraintBOUUVHullDesign)

### 5.4 Existing Underwater Simulators for Design Validation

| Simulator | Base | Key Feature | Source |
|---|---|---|---|
| **OceanSim** | Isaac Sim | GPU-accelerated perception simulation, underwater rendering | [arXiv](https://arxiv.org/html/2503.01074v2), [GitHub](https://github.com/umfieldrobotics/OceanSim) |
| **MarineGym** | Isaac Sim | GPU hydrodynamics, 250K FPS on RTX 3060, RL training | [arXiv](https://arxiv.org/html/2503.09203v1), [GitHub](https://github.com/Marine-RL/MarineGym), [marine-gym.com](https://marine-gym.com/) |
| **Gazebo + UUV Simulator** | Gazebo | ROS-integrated, community-standard | (established) |

---

## 6. OceanScale's "Design" Pillar — Recommendations

### 6.1 The Gap

No existing tool provides an integrated, simulation-first, config-driven design pipeline for underwater vehicles. Today's workflow is fragmented:

1. CAD (SOLIDWORKS/Onshape) for mechanical design
2. Orca3D/Rhino for hull shape and hydrostatics
3. MATLAB/Simulink for dynamics and control modeling
4. OpenFOAM/ANSYS for CFD (separate from dynamics)
5. Gazebo/custom sim for testing control policies
6. Physical prototype -> tank test -> iterate

Each tool is a silo. Data doesn't flow between them. A hull shape change in CAD requires manual re-export and re-setup in CFD, then manual re-parameterization in Simulink. This is why design takes years and costs millions.

### 6.2 What OceanScale Should Build

**Config-driven robot definition -> auto-generate simulation-ready USD + Newton model:**

```yaml
# example: vehicles/bluerov_heavy.yaml
vehicle:
  name: "BlueROV2-Heavy"
  type: rov
  hull:
    shape: cuboid          # cuboid | torpedo | sphere | custom_usd
    length: 0.457          # m
    width: 0.338           # m
    height: 0.254          # m
    material: acetal       # affects density, strength
    wall_thickness: 0.005  # m
    depth_rating: 300      # m
  thrusters:
    count: 8
    type: T200             # Blue Robotics T200
    positions:             # (x, y, z) relative to CG
      - [ 0.15,  0.15, 0.0]   # front-right horizontal
      - [ 0.15, -0.15, 0.0]   # front-left horizontal
      - [-0.15,  0.15, 0.0]   # rear-right horizontal
      - [-0.15, -0.15, 0.0]   # rear-left horizontal
      - [ 0.15,  0.0,  0.0]   # front-right vertical
      - [-0.15,  0.0,  0.0]   # rear-right vertical
      - [ 0.15,  0.0, -0.0]   # front-left vertical
      - [-0.15,  0.0, -0.0]   # rear-left vertical
    orientations: [...]        # per-thruster orientation vectors
    max_thrust: 5.1            # N per thruster
  buoyancy:
    target: neutral            # neutral | positive | negative
    offset_cg_cb: [0, 0, 0.01]  # CG-CB offset for stability
  sensors:
    - type: camera
      position: [0.20, 0, 0]
      orientation: [0, 0, 0]
      fov: 90
    - type: imu
      position: [0, 0, 0]
    - type: depth
      position: [0, 0, -0.05]
  battery:
    chemistry: liion
    capacity_Wh: 300
    voltage: 14.8
  mass: 11.5  # kg total
```

**Pipeline stages:**

1. **YAML -> USD generator**
   - Parse vehicle config YAML
   - Generate USD scene with correct prim hierarchy (PhysicsScene -> Xform -> RigidBody prims -> Joint prims)
   - Attach Newton custom attributes for hydrodynamic coefficients (drag, added mass, lift), thruster limits, sensor parameters
   - Use Newton's `add_usd()` custom attribute system to expose all parameters

2. **Auto-compute hydrodynamic properties from geometry**
   - From hull shape parameters, compute:
     - Drag coefficients (Cd) using empirical formulas (ITTC, Hoerner) or pre-computed CFD lookup tables
     - Added mass coefficients using ellipsoid approximation (Lamb, 1932)
     - Buoyancy and stability (metacentric height)
   - Write these as Newton custom attributes on the PhysicsScene prim

3. **Auto-generate Newton model**
   - Call `ModelBuilder` with generated USD
   - Add hydrodynamic force kernel (Warp kernel applying Fossen 6-DOF forces)
   - Configure thrusters as force actuators with position and orientation from config

4. **Parametric design exploration (batch over configs)**
   - Define design variables in YAML: hull length, diameter, thruster count, battery capacity
   - Generate N configurations by sweeping parameters
   - Run batched Newton simulation (GPU-parallel, 1000s of configs simultaneously)
   - Evaluate each config on: max speed, endurance, depth rating, stability margin, maneuverability
   - Use BO-LCB for intelligent exploration of design space (not brute-force grid)

5. **Design rule checking**
   - Stability: metacentric height > 0 (CG below CB for positive stability)
   - Buoyancy: net buoyancy within acceptable range
   - Power budget: propulsion + hotel load < battery capacity for target endurance
   - Thruster authority: each DOF controllable (thruster allocation matrix full rank)
   - Depth rating: wall stress < material yield / safety factor
   - Sensor coverage: field of regard covers required inspection zones

### 6.3 Technical Feasibility with Current Stack

| Component | Newton/USD Support | Status |
|---|---|---|
| Robot model from USD | `ModelBuilder.add_usd()` | Production-ready |
| Custom attributes in USD | Newton custom attr system | Production-ready |
| URDF import | `ModelBuilder.add_urdf()` | Production-ready |
| GPU-parallel simulation | Warp + MuJoCo-Warp solver | Production-ready |
| Hydrodynamic forces | Custom Warp kernel (OceanScale builds) | Build required |
| Sensor simulation | Isaac Sim RTX + custom Warp kernels | Partial (build sensors) |
| Batched design sweep | Isaac Lab env config pattern | Pattern exists, adapt |
| BO-LCB optimization | Open-source code available | Integrate |
| CFD surrogate models | Pre-computed lookup / DNN surrogate | Build required |

### 6.4 Priority Order (MVP to Full)

**Phase 1 (v0.1) — Config-to-Sim:**
- YAML vehicle config parser
- USD scene generator with Newton custom attributes
- Newton model auto-builder
- Basic hydrodynamic force kernel (Fossen drag + added mass)
- Single-vehicle simulation with sensor output

**Phase 2 (v0.2) — Design Exploration:**
- Parameter sweep over hull dimensions and thruster configurations
- Auto-compute hydro coefficients from geometry
- Power budget calculator
- Thruster allocation matrix rank check
- Stability margin calculator

**Phase 3 (v0.3) — Optimization:**
- BO-LCB integration for design space exploration
- Multi-objective Pareto front (speed vs endurance vs depth)
- Surrogate model training from simulation results
- CFD lookup table generation for common hull shapes

**Phase 4 (v0.4) — Design Validation:**
- Design rule checker with pass/fail reporting
- Automatic URDF generation from config (for ROS 2 interop)
- Design report generation (mass budget, power budget, stability analysis)
- CAD export integration (STEP/STL from parametric hull shapes)

### 6.5 Differentiation from Existing Tools

| Capability | MATLAB/Simulink | MarineGym | OceanSim | **OceanScale (proposed)** |
|---|---|---|---|---|
| Config-driven robot definition | Manual | Hardcoded models | Hardcoded models | **YAML -> USD -> Newton** |
| Auto-compute hydro coefficients | Manual | GPU-accelerated | Manual | **From geometry, auto** |
| Batched design exploration | Manual scripts | No | No | **GPU-parallel sweep** |
| Bayesian optimization | Separate toolbox | No | No | **Integrated BO-LCB** |
| Design rule checking | Manual | No | No | **Automated pass/fail** |
| USD-native scene | No | No | Yes | **Yes (Newton native)** |
| Open-source | No (commercial) | Yes | Yes | **Yes (Apache-2.0)** |

---

## Sources

### Design Workflows
- [MathWorks AUV Solutions](https://www.mathworks.com/solutions/aerospace-defense/auv.html)
- [Orca3D Marine Design](https://orca3d.com/)
- [Orca3D Marine CFD](https://rhino3dzine.com/stories/advanced-forms/revolutionizing-marine-cfd-with-orca3d/)
- [NavCad + Orca3D integration](https://ndar.com/how-navcad-and-orca3D-marine-cfd-work-together-to-improve-vessel-design/)
- [GrabCAD ROV models in SOLIDWORKS](https://grabcad.com/library/software/solidworks/tag/rov)
- [6-DOF ROV hydrodynamic model](https://www.researchgate.net/publication/276184856_Modelling_Design_and_Robust_Control_of_a_Remotely_Operated_Underwater_Vehicle)
- [MeCO open-source AUV](https://arxiv.org/html/2503.10928v1)
- [Fraunhofer AUV testing tool chain](https://publica.fraunhofer.de/bitstreams/55283b5d-6393-47b8-811c-252ecc976325/download)
- [ITB underwater vehicle development](https://www.academia.edu/173897/Design_Development_and_Testing_of_Underwater_Vehicles_ITB_Experience)
- [Cost-effective AUV design ($400)](https://www.researchgate.net/publication/304022131_Design_of_a_cost-effective_autonomous_underwater_vehicle)
- [10,000m class AUV](https://www.mdpi.com/2077-1312/12/11/2097)
- [Life cycle cost of hydrogen fuel cell AUV](https://www.sciencedirect.com/science/article/pii/S0029801824006371)
- [Impossible Metals Eureka](https://impossiblemetals.com/technology/robotic-collection-system/)
- [Marine cable inspection robot](https://www.sciencedirect.com/science/article/pii/S2667241325000151)

### Parametric Design
- [DEKC: Parametric Hull Design](https://www.dekc-maritime.com/news2/the-many-faces-of-parametric-hull-design)
- [ULiege parametric hull optimization thesis](https://matheo.uliege.be/bitstream/2268.2/22262/4/Smit_DHANANI_Master_Thesis_Report.pdf)
- [ScienceDirect: BO + DNN for UUV design](https://www.sciencedirect.com/science/article/pii/S0029801824021152)
- [ACM: Constrained BO for AUV hull](https://dl.acm.org/doi/10.1145/3576914.3587530)
- [arXiv: AI + CFD minimum drag](https://arxiv.org/pdf/2302.09441)
- [Hrcak: Automated drag optimization platform](https://hrcak.srce.hr/file/478526)
- [NASA ADS: MDO of AUV hull](https://ui.adsabs.harvard.edu/abs/2017dcab.conf...51C/abstract)
- [Taylor & Francis: Bio-inspired hull](https://www.tandfonline.com/doi/full/10.1080/19942060.2021.1940287)
- [GitHub: UUV-design-optimization](https://github.com/vardhah/UUV-design-optimization)
- [GitHub: ConstraintBOUUVHullDesign](https://github.com/vardhah/ConstraintBOUUVHullDesign)
- [MDPI: Tunnel thruster design](https://www.mdpi.com/2077-1312/12/11/2021)
- [KCI: Rim-driven thruster parametric design](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003323299)
- [Springer: Thruster configuration estimation](https://link.springer.com/chapter/10.1007/978-3-642-28572-1_45)
- [MIT DSpace: Thruster config](https://dspace.mit.edu/handle/1721.1/137112)
- [AUV design considering energy](https://eprints.soton.ac.uk/466611/1/1230769.pdf)

### USD / Isaac Sim / Newton
- [NVIDIA: Anatomy of Robot Asset Structure](https://docs.nvidia.com/learning/physical-ai/going-further-with-robotics/latest/best-practices-for-robotics-and-openusd/01-anatomy-of-robot-asset-structure.html)
- [Isaac Sim URDF import](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/robot_setup/import_urdf.html)
- [Isaac Lab import new asset](https://isaac-sim.github.io/IsaacLab/main/source/how-to/import_new_asset.html)
- [NVIDIA Forum: USD to URDF not supported](https://forums.developer.nvidia.com/t/cannot-convert-usd-to-urdf/359049)
- [NVIDIA Blog: Newton + OpenUSD](https://blogs.nvidia.com/blog/newton-physics-engine-openusd/)
- [Newton GitHub](https://github.com/newton-physics/newton)
- [Newton official docs](https://newton-physics.github.io/newton/stable/)
- [NVIDIA Developer: Newton](https://developer.nvidia.com/newton-physics)
- [Linux Journal: Newton at Linux Foundation](https://www.linuxjournal.com/content/linux-foundation-welcomes-newton-next-open-physics-engine-robotics)
- [GTC 2026: Newton intro session](https://www.nvidia.com/en-us/on-demand/session/gtc26-s81613/)
- [OpenUSD Rigid Body Physics Proposal](https://openusd.org/dev/wp_rigid_body_physics.html)
- [Isaac Sim OpenUSD tuning tutorials](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/openusd_tuning_tutorials/index.html)

### Config-Driven Design
- [Isaac Lab migration from IsaacGymEnvs](https://isaac-sim.github.io/IsaacLab/main/source/migration/migrating_from_isaacgymenvs.html)
- [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- [Isaac Sim policy deployment](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/isaac_lab_tutorials/tutorial_policy_deployment.html)
- [Nav2 URDF setup](https://docs.nav2.org/setup_guides/urdf/setup_urdf.html)
- [ROS 2 URDF tutorial](https://automaticaddison.com/create-and-visualize-a-mobile-robot-with-urdf-ros-2-jazzy/)
- [PX_IsaacSim_URDF_Importer](https://github.com/ProximityRobotics/PX_IsaacSim_URDF_Importer)
- [Modularis open-source AUV](https://arxiv.org/html/2401.06243v1)
- [ICRA 2010: SE(3) modular thruster control](https://fileadmin.cs.lth.se/ai/Proceedings/ICRA2010/MainConference/data/papers/1835.pdf)
- [MDPI: Dynamically reconfigurable AUV](https://www.mdpi.com/1424-8220/22/9/3379)

### Underwater Simulators
- [OceanSim arXiv](https://arxiv.org/html/2503.01074v2)
- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim)
- [MarineGym arXiv](https://arxiv.org/html/2503.09203v1)
- [MarineGym GitHub](https://github.com/Marine-RL/MarineGym)
- [MarineGym website](https://marine-gym.com/)
- [MarineGym earlier paper](https://arxiv.org/html/2410.14117v1)
