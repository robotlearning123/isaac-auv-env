# Disney Research Contributions to Fluid Simulation, Newton, and Physics Engines

Research date: 2026-05-20

---

## 1. Disney Kamino Solver (NOT VBD)

### What is Kamino?

Kamino is a GPU-based physics solver developed by **Disney Research** (Zurich robotics lab, led by Moritz Bacher) for massively parallel simulation of heterogeneous, highly-coupled **rigid multi-body mechanical systems**. It specifically targets systems with **kinematic loops** -- closed-chain mechanisms like four-bar linkages, parallel robots, and legged robots with parallel structures -- which are notoriously difficult for standard GPU simulators.

**Paper**: "Kamino: GPU-based Massively Parallel Simulation of Multi-Body Systems with Challenging Topologies"
- arXiv: https://arxiv.org/abs/2603.16536
- Authors: Vassilios Tsounis, Guirec Maloisel, Christian Schumacher, Ruben Grandia, Arian Serifi, Daniel Muller, Moritz Bacher
- Affiliation: Disney Research (with some authors also at NVIDIA)
- Official project page: https://disneyresearch.github.io/kamino/

### Is Kamino for fluid simulation?

**No.** Kamino is exclusively for **rigid multi-body dynamics**. From the source code (`solver_kamino.py`):

> "A physics solver for simulating constrained multi-body systems containing kinematic loops, under-/overactuation, joint-limits, hard frictional contacts and restitutive impacts."

The solver uses the **Proximal-ADMM algorithm** to solve forward dynamics formulated as a Nonlinear Complementarity Problem (NCP) over bilateral kinematic joint constraints and unilateral constraints (joint limits, contacts).

The validation in the source code explicitly rejects:
- particles
- springs
- triangle/tetrahedral/edge elements
- muscles
- equality constraints
- distance/cable/gimbal joints

These are all deformable/soft-body features. Kamino is purely rigid-body with joints.

### Performance characteristics

- GPU-accelerated via NVIDIA Warp (CUDA kernel compilation from Python)
- Supports massively parallel simulation: thousands of environments simultaneously
- Configurable PADMM convergence criteria (primal/dual/complementarity tolerances)
- Sparse and dense Jacobian modes
- Warm-starting for contact forces
- Examples demonstrate 16+ parallel worlds running simultaneously

### Can Kamino handle water/ocean?

**No.** Kamino has no fluid, particle, or continuum mechanics capability. It is a rigid-body dynamics solver only.

---

## 2. VBD (Vertex Block Descent) -- Separate from Kamino

**VBD is NOT a Disney product.** It was developed by Anka He Chen et al. at the **Utah Graphics Lab**:

- Paper: https://arxiv.org/html/2403.1036v1
- Project page: https://graphics.cs.utah.edu/research/projects/vbd/
- WarpVBD implementation: https://github.com/mmichelis/WarpVBD

VBD handles **deformable body simulation** (cloth, soft bodies, cables). It is integrated into Newton as a separate solver (`newton._src.solvers.vbd`), alongside but independent of Kamino.

Newton's solver lineup:
| Solver | Developer | Domain |
|--------|-----------|--------|
| Kamino | Disney Research | Rigid multi-body, kinematic loops |
| VBD | Utah Graphics Lab | Deformable bodies (cloth, cables) |
| Implicit MPM | NVIDIA | Granular materials, continuum mechanics |
| Featherstone | Standard algorithm | Articulated body dynamics |
| MuJoCo | DeepMind | General robotics simulation |
| XPBD | Standard algorithm | Position-based dynamics |
| Style3D | Style3D Inc. | Fashion/cloth simulation |

---

## 3. Disney's Contribution to Newton Specifically

### What Disney contributed

Disney Research contributed the **Kamino solver** as a backend to the Newton physics engine. This is their primary technical contribution.

From the Newton source code, the Kamino integration includes:
- Complete solver implementation (~40+ Python files in `newton/_src/solvers/kamino/`)
- Proximal-ADMM solver for constrained dynamics
- Forward kinematics solver for loop closure
- GPU collision detector
- USD asset pipeline with KaminoSceneAPI custom attributes
- Example assets: DR TestMech, DR Legs (Disney Research robot hardware)

### Newton co-development

Newton was announced at **GTC 2025 (March 2025)** as a collaboration between:
- **NVIDIA**: Core framework, NVIDIA Warp, OpenUSD integration
- **Google DeepMind**: Co-development for robot learning
- **Disney Research / Walt Disney Imagineering**: Kamino solver, robotics expertise (BDX Droids)

- Linux Foundation announcement: https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning
- NVIDIA blog: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/

### Is there a Disney fluid plugin for Newton/Omniverse?

**No.** Disney's contribution to Newton (Kamino) is rigid-body only. The fluid-capable solver in Newton is **Implicit MPM** (`newton._src.solvers.implicit_mpm`), which was developed by NVIDIA.

---

## 4. Disney Research Fluid Simulation History

### Disney Animation's "Splash" Engine (for Moana)

For *Moana* (2016), Walt Disney Animation Studios developed **Splash**, a proprietary fluid simulation engine:
- Based on the **APIC (Affine Particle-In-Cell)** method (related to FLIP family)
- Key developer: Alexey Stomakhin
- Reference: Jiang et al. 2015 (APIC method)
- SIGGRAPH 2017 talk: "Moana: Performing Water" -- https://disneyanimation.com/publications/moana-performing-water/
- SIGGRAPH 2017 talk: "The Ocean and Water Pipeline of Disney's Moana"
- Splash continued to be used for *Moana 2* (2024)

Sources:
- https://phys.org/news/2017-01-mathematicians-ocean-life-disney-moana.html
- https://www.sidefx.com/community/walt-disney-animation-studios-moana/
- https://www.aswf.io/blog/disneys-moana-2-sets-sail-with-help-from-open-source-tools/

### Physics-Informed Neural Correctors for Fluid Control

Disney Research Studios published work on using neural networks to maintain physical plausibility in art-directed fluid edits:
- https://studios.disneyresearch.com/2023/05/07/physicsinformed-neural-corrector-for-deformationbased-fluid-control/

### ETH Zurich Collaboration (AI Fire/Fluid)

ETH Zurich researchers (closely tied to Disney Research Zurich) developed AI-based simulation for smoke, fire, and fluids used in Pixar films:
- https://ethz.ch/en/news-and-events/eth-news/news/2023/06/how-ai-technology-from-eth-animates-the-fire-creatures-in-the-latest-pixar-cinema-movie.html

### Key point: Splash is proprietary

Splash is an internal Disney Animation tool. It is **not open-source** and is **not integrated into Newton or any public framework**. The ocean simulation pipeline for Moana combined Splash with Houdini and Maya.

---

## 5. Open-Source Simulation Tools from Disney

| Tool | Status | Domain |
|------|--------|--------|
| Kamino (via Newton) | Open-source (Apache-2.0) | Rigid multi-body dynamics |
| Splash | Proprietary | FLIP/APIC fluid simulation |
| Rig-Space Physics | Published method | Animation pipeline integration |

Disney has NOT released any open-source fluid simulation tool. Their fluid work (Splash) remains proprietary to Walt Disney Animation Studios.

---

## 6. Newton's Implicit MPM Solver (for Fluid/Granular Simulation)

The fluid-capable solver in Newton is **Implicit MPM**, located at:
- `newton/_src/solvers/implicit_mpm/`

This solver includes:
- Implicit MPM solver kernels
- Rheology solver (for non-Newtonian fluids)
- Contact solver with Coulomb friction
- Rasterized collision detection
- Grain rendering

This is an **NVIDIA-developed** solver, not a Disney contribution. It can handle granular materials and potentially fluid-like substances, but is primarily targeted at terrain/deformable ground for robot locomotion training.

References:
- NVIDIA blog on cloth + MPM: https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/

---

## 7. Summary for OceanScale

For underwater/ocean simulation purposes:

1. **Disney's Kamino solver is irrelevant to fluid simulation.** It handles rigid-body kinematic loops (robotic hands, parallel mechanisms, legged robots).

2. **Disney's Splash engine is the most relevant Disney fluid work** (APIC solver for Moana), but it is proprietary and not available in any open-source framework.

3. **Newton's Implicit MPM solver** (NVIDIA, not Disney) is the closest thing to fluid capability in Newton, but it targets granular/terrain simulation for robotics, not ocean-scale fluid dynamics.

4. **For ocean simulation**, the relevant open-source stacks remain:
   - NVIDIA Flex/PhysX (SPH + FLIP hybrid)
   - WarpImplicit MPM (within Newton)
   - External: Houdini FLIP, MantaFlow, Blender FLIP
   - Research: Splash-style APIC (proprietary Disney)

---

## Key URLs

- Kamino paper: https://arxiv.org/abs/2603.16536
- Kamino project page: https://disneyresearch.github.io/kamino/
- Newton GitHub: https://github.com/newton-physics/newton
- Newton NVIDIA blog: https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/
- Linux Foundation announcement: https://www.linuxfoundation.org/press/linux-foundation-announces-contribution-of-newton-by-disney-research-google-deepmind-and-nvidia-to-accelerate-open-robot-learning
- Newton contact-rich capabilities: https://developer.nvidia.com/blog/newton-adds-contact-rich-manipulation-and-locomotion-capabilities-for-industrial-robotics/
- VBD paper: https://arxiv.org/html/2403.1036v1
- VBD project page: https://graphics.cs.utah.edu/research/projects/vbd/
- WarpVBD: https://github.com/mmichelis/WarpVBD
- Moana performing water: https://disneyanimation.com/publications/moana-performing-water/
- Disney neural fluid corrector: https://studios.disneyresearch.com/2023/05/07/physicsinformed-neural-corrector-for-deformationbased-fluid-control/
- ASWF Moana 2 open source: https://www.aswf.io/blog/disneys-moana-2-sets-sail-with-help-from-open-source-tools/
- Newton MPM + cloth blog: https://developer.nvidia.com/blog/train-a-quadruped-locomotion-policy-and-simulate-cloth-manipulation-with-nvidia-isaac-lab-and-newton/
- Newton Newton's Classroom video: https://www.youtube.com/watch?v=Aws58jyn998
- Isaac Lab Newton integration docs: https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html
- Kamino GTC 2026 mention: https://www.linkedin.com/posts/risman-adnan-bb726b5_kamino-jensen-introduced-in-gtc-2026-tackles-activity-7442938007923650560-0iZW
- Moritz Bacher publications: https://www.baecher.info/publications/
- ETH Zurich AI fire/fluid: https://ethz.ch/en/news-and-events/eth-news/news/2023/06/how-ai-technology-from-eth-animates-the-fire-creatures-in-the-latest-pixar-cinema-movie.html
- Phys.org Moana water: https://phys.org/news/2017-01-mathematicians-ocean-life-disney-moana.html
- SideFX Moana: https://www.sidefx.com/community/walt-disney-animation-studios-moana/
