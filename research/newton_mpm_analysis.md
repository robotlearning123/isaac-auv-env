# Newton 1.2.0 MPM Analysis for Underwater Robotics

**Date**: 2026-05-20
**Newton version**: 1.2.0
**Install path**: `/home/robot/workspace/46-marine/.venv/lib/python3.12/site-packages/newton/`
**Researcher note**: All claims cite source file paths and line numbers. No fabrication.

---

## 1. MPM Example File Analysis

### 1.1 example_mpm_twoway_coupling.py

**Source**: `newton/examples/mpm/example_mpm_twoway_coupling.py`

**What it simulates**: A dozen rigid body boxes dropped onto a sand bed (2m x 2m x 0.5m). Two-way coupling: rigid bodies exert forces on sand particles, sand particles exert reaction forces back on rigid bodies via impulse collection and force application kernels.

**Key parameters**:
- FPS: 100, frame_dt: 0.01s, 4 substeps (line 91-95)
- Voxel size: 0.05m (5 cm) (line 113)
- Sand density: 2500 kg/m^3 (line 358)
- Particles per cell: 3.0 (line 357)
- Grid type: "fixed" with padding=50, max_active_cell_count=2^15 (lines 123-125)
- Strain basis: "P0", max_iterations: 50 (lines 127-129)

**Architecture pattern** (lines 188-253):
1. Rigid bodies solved via MuJoCo (`SolverMuJoCo`) with substeps
2. Sand solved via `SolverImplicitMPM` once per frame
3. Body forces from sand computed via custom `compute_body_forces` kernel
4. Previous sand forces subtracted before MPM step, re-collected after

**Can be tuned for water/ocean?**: Partially. The two-way coupling architecture is exactly what's needed for AUV interaction with fluid. The sand material parameters (density 2500, friction 0.75) would need fluid parameters instead. No viscosity parameter is set in this example.

**Isaac Lab RL integration**: The example uses `torch.jit.load` for a pretrained policy in the Anymal example (see 1.4), demonstrating that Newton can interface with PyTorch policies. No direct Isaac Lab API integration is shown.

---

### 1.2 example_mpm_viscous.py

**Source**: `newton/examples/mpm/example_mpm_viscous.py`

**What it simulates**: Viscoplastic fluid flowing through a funnel-shaped mesh collider with a narrow aperture at the bottom. Demonstrates mesh-based collision with MPM particles.

**Key parameters**:
- FPS: 240, frame_dt: 1/240s (line 119)
- Voxel size: 0.005m (5 mm) (line 138)
- Density: 1000 kg/m^3 -- **WATER DENSITY** (line 122)
- Viscosity: 50.0 Pa*s (line 123)
- Tensile yield ratio: 1.0 (line 124)
- Friction: 0.0 (line 125)
- Funnel friction: 0.0 (line 127)
- Solver: max_iterations=250, tolerance=1e-6 (lines 136-137)
- Strain basis: "P0", velocity basis: "Q1", collider basis: "S2" (lines 139-141)

**Water relevance**: This is the closest example to fluid simulation. Default density is exactly 1000 kg/m^3 (water). Viscosity of 50 Pa*s is very high (honey-like); water is ~0.001 Pa*s. The `tensile_yield_ratio=1.0` and `friction=0.0` create a purely viscous, non-frictional material -- the right direction for fluid.

**Material property setting pattern** (lines 58-60):
```python
self.model.mpm.viscosity.fill_(options.viscosity)
self.model.mpm.tensile_yield_ratio.fill_(options.tensile_yield_ratio)
self.model.mpm.friction.fill_(options.friction)
```

---

### 1.3 example_mpm_multi_material.py

**Source**: `newton/examples/mpm/example_mpm_multi_material.py`

**What it simulates**: Three material types in one simulation -- sand, snow, and mud -- interacting simultaneously. Demonstrates per-particle material property assignment.

**Key parameters**:
- FPS: 60, voxel_size: 0.05m (lines 16, 187)
- Sand: density=2500, default params (lines 117-124)
- Snow: density=300, yield_pressure=2e4, tensile_yield_ratio=0.2, friction=0.1, hardening=10.0, dilatancy=1.0 (lines 41-45)
- Mud: density=1000, yield_pressure=1e10, yield_stress=300, tensile_yield_ratio=1.0, friction=0.0, viscosity=100.0 (lines 48-53)

**Water relevance**: The mud material (density=1000, viscosity=100, friction=0.0) is another fluid-like configuration. The per-particle material assignment API (`model.mpm.<property>[indices].fill_(value)`) is exactly how you'd define water regions vs sediment regions.

**Multi-material API** (lines 39-53): Per-particle material properties are set by indexing into the model arrays with particle index arrays:
```python
self.model.mpm.yield_pressure[snow_particles].fill_(2.0e4)
self.model.mpm.viscosity[mud_particles].fill_(100.0)
```

---

### 1.4 example_mpm_anymal.py

**Source**: `newton/examples/mpm/example_mpm_anymal.py`

**What it simulates**: ANYmal C quadruped robot with a pretrained RL walking policy, coupled with MPM sand. The robot walks on sand using keyboard commands or auto-forward.

**Key parameters**:
- FPS: 50, 4 substeps (lines 77-81)
- Voxel size: 0.03m (3 cm) (line 310)
- Particles per cell: 3.0 (line 311)
- Sand density: 2500 (line 111)
- Sand bed: [-0.5, -0.5, 0.0] to [0.5, 2.5, 0.15] (lines 112-113)
- Transfer scheme: "pic" (line 128)
- Collider velocity mode: "backward" (line 137)
- Air drag: 1.0 (line 136)

**Critical for underwater robotics**: This example demonstrates the exact pattern needed for an AUV interacting with MPM fluid:
1. Robot loaded from URDF (line 57-65)
2. `COLLIDE_PARTICLES` flag filtering -- only SHANK bodies collide with sand (lines 68-71)
3. Separate graph captures for robot and sand steps (lines 196-207)
4. `setup_collider(body_mass=wp.zeros_like(...))` treats robot as kinematic (lines 155-158)
5. Pretrained policy via `torch.jit.load` (line 177)

**GPU requirement**: Explicitly checks for GPU (lines 343-345):
```python
if wp.get_device().is_cpu:
    print("Error: This example requires a GPU device.")
    sys.exit(1)
```

---

### 1.5 example_mpm_granular.py

**Source**: `newton/examples/mpm/example_mpm_granular.py`

**What it simulates**: Granular material (sand-like) falling onto various collider shapes (cube, wedge, concave). The most configurable example with all material parameters exposed as CLI arguments.

**Key parameters**:
- FPS: 60, voxel_size: 0.1m (lines 237, 274)
- Density: 1000 kg/m^3 (line 241)
- Young's modulus: 1e15 Pa (very stiff, essentially rigid) (line 245)
- Poisson ratio: 0.3 (line 246)
- Friction: 0.68 (line 247)
- Yield pressure: 1e12 Pa (line 249)
- Tensile yield ratio: 0.0 (line 250)
- Viscosity: 0.0 (line 254)
- Transfer scheme: "apic" (line 265)
- Integration scheme: "pic" or "gimp" (line 266)
- Grid types: sparse, fixed, dense (line 256)
- Collider basis: "S2" (line 269)
- Velocity basis: "Q1" (line 270)
- Strain basis: "P0" (line 268)

**Water relevance**: Low. This is granular material with zero viscosity and zero tensile yield. However, the comprehensive CLI argument interface makes it a good template for experimenting with water-like parameters.

---

### 1.6 example_mpm_snow_ball.py

**Source**: `newton/examples/mpm/example_mpm_snow_ball.py`

**What it simulates**: Snow avalanche and snow ball rolling down a heightfield terrain slope. Uses per-particle snow rheology with compression visualization.

**Key parameters**:
- FPS: 60, voxel_size: 0.1m (lines 373-374, 399)
- Density: 400 kg/m^3 (line 377)
- Young's modulus: 1.4e6 Pa (line 378)
- Poisson ratio: 0.3 (line 379)
- Friction: 0.5 (line 380)
- Damping: 0.01 (line 381)
- Yield pressure: 1.4e6 Pa (line 382)
- Tensile yield ratio: 0.2 (line 383)
- Hardening: 5.0 (line 385)
- Dilatancy: 1.0 (line 386)
- Solver sequence: ("cg", "gauss-seidel") (lines 388-394)
- Terrain: heightfield mesh, 5m x 20m, 45-degree slope (lines 52-54)

**Water relevance**: Low. Snow mechanics (hardening, dilatancy, compression tracking) are specific to compressible materials. However, the heightfield terrain interaction pattern could be adapted for underwater terrain.

---

### 1.7 example_mpm_beam_twist.py

**Source**: `newton/examples/mpm/example_mpm_beam_twist.py`

**What it simulates**: Elastic beam made of MPM particles, twisted at one end while the other end is clamped. Demonstrates elastic solid mechanics, stress visualization, and kinematic boundary particles.

**Key parameters**:
- FPS: 240, voxel_size: 0.25m (lines 268, 287)
- Density: 1000 kg/m^3 (line 270)
- Young's modulus: 5e6 Pa (line 271)
- Poisson ratio: 0.45 (line 272) -- **near-incompressible**
- Damping: 0.001 (line 273)
- Strain basis: "P1d" (line 283)
- Velocity basis: "Q1" (line 284)
- Solver: "cr" (conjugate residual) (line 278)
- Warmstart: "particles" (line 73)
- Twist: 360 degrees over 1000 frames (lines 105-106)

**Water relevance**: Moderate. The Poisson ratio of 0.45 (near-incompressible) and density of 1000 kg/m^3 are water-like. This demonstrates that the MPM solver can handle near-incompressible elasticity, which is the starting point for incompressible fluid simulation.

---

### 1.8 example_mpm_grain_rendering.py

**Source**: `newton/examples/mpm/example_mpm_grain_rendering.py`

**What it simulates**: Sand column collapse with high-resolution grain rendering. Each MPM particle is represented by multiple render grains for visual fidelity.

**Key parameters**:
- FPS: 60, voxel_size: 0.1m (lines 19, 131)
- Density: 2500 (line 98)
- Points per particle: 8 (line 132)

**Water relevance**: Low for physics. The rendering pipeline (`sample_render_grains` and `update_render_grains`) could be repurposed for rendering water surface particles.

---

## 2. Implicit MPM Solver Source Analysis

### 2.1 solver_implicit_mpm.py (2554 lines)

**Source**: `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py`

**Class**: `SolverImplicitMPM(SolverBase)` (line 614)

**Docstring** (lines 615-640):
> Implements an implicit Material Point Method (MPM) algorithm roughly following [1], extended with a GPU-friendly rheology solver supporting pressure-dependent yield (Drucker-Prager), viscosity, dilatancy, and isotropic hardening/softening. This variant is particularly well-suited for very stiff materials and the fully inelastic limit. It is less versatile than traditional explicit MPM but offers unconditional stability with respect to the time step.

Reference [1]: https://doi.org/10.1145/2897824.2925877

**Config dataclass** (lines 643-711):
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_iterations | int | 250 | Rheology solver max iterations |
| tolerance | float | 1e-4 | Rheology solver tolerance |
| solver | str/Sequence | "auto" | Solver: gs, jacobi, cg, cr, gmres |
| warmstart_mode | str | "auto" | none, auto, particles, grid, smoothed |
| collider_velocity_mode | str | "forward" | forward, backward |
| voxel_size | float | 0.1 | Grid voxel edge length |
| grid_type | str | "sparse" | sparse, dense, fixed |
| grid_padding | int | 0 | Extra empty cells around particles |
| transfer_scheme | str | "apic" | apic or pic |
| integration_scheme | str | "pic" | pic or gimp |
| critical_fraction | float | 0.0 | Fraction threshold for yield collapse |
| air_drag | float | 1.0 | Background air drag |
| collider_basis | str | "S2" | Q1, S2, pic, pic8, pic27 |
| strain_basis | str | "P0" | P0, P1d, Q1, Q1d, pic, pic8, pic27 |
| velocity_basis | str | "Q1" | Q1, B2, B3 |

**Register custom attributes** (lines 714-913): Registers per-particle material and state attributes in the "mpm" namespace:

Model attributes (per-particle material params):
- `mpm:young_modulus` (default: 1e15 Pa) -- lines 742-751
- `mpm:poisson_ratio` (default: 0.3) -- lines 752-761
- `mpm:damping` (default: 0.0) -- lines 762-771
- `mpm:hardening` (default: 0.0) -- lines 772-781
- `mpm:friction` (default: 0.5) -- lines 782-791
- `mpm:yield_pressure` (default: 1e15 Pa) -- lines 792-801
- `mpm:tensile_yield_ratio` (default: 0.0) -- lines 802-811
- `mpm:yield_stress` (default: 0.0 Pa) -- lines 812-821
- `mpm:hardening_rate` (default: 1.0) -- lines 822-831
- `mpm:softening_rate` (default: 1.0) -- lines 832-841
- `mpm:dilatancy` (default: 0.0) -- lines 842-851
- `mpm:viscosity` (default: 0.0 Pa*s) -- lines 852-861

State attributes (per-particle state variables):
- `mpm:particle_qd_grad` (mat33, default: 0) -- APIC velocity gradient
- `mpm:particle_elastic_strain` (mat33, default: identity) -- elastic deformation gradient
- `mpm:particle_Jp` (float, default: 1.0) -- plastic deformation gradient determinant
- `mpm:particle_stress` (mat33, default: 0) -- Cauchy stress tensor
- `mpm:particle_transform` (mat33, default: identity) -- overall deformation gradient for rendering

**Step pipeline** (lines 1554-1617, method `_step_impl`):
1. Rasterize colliders to discrete space
2. Compute unconstrained (ballistic) velocity + inverse mass matrix
3. Build collider rigidity operator (for dynamic bodies)
4. Build elasticity compliance matrix and RHS
5. Build plasticity system (strain matrix, yield surface parameters)
6. Load warmstart from previous step
7. Solve rheology (coupled strain/contact/velocity)
8. Save warmstart for next step
9. Update and advect particles

**Two-way coupling support** (lines 1002-1049, method `setup_collider`):
- `collider_adhesion` parameter exists (line 1008, line 1009)
- `collider_friction` per-mesh friction (line 1007)
- `collider_margins` per-mesh SDF offsets (line 1006)
- `body_mass` -- pass zeros for kinematic, real mass for dynamic (line 1012)
- `model` parameter allows reading colliders from a different model (line 1010)

**Collider impulse collection** (lines 1092-1110, method `collect_collider_impulses`):
Returns impulse values, positions, and collider IDs that map to body indices via `collider_body_index`.

### 2.2 implicit_mpm_model.py (598 lines)

**Source**: `newton/_src/solvers/implicit_mpm/implicit_mpm_model.py`

**MaterialParameters struct** (lines 247-276): Contains arrays for young_modulus, poisson_ratio, damping, friction, yield_pressure, tensile_yield_ratio, yield_stress, viscosity, hardening, hardening_rate, softening_rate, dilatancy.

**Feature flags** (lines 348-351):
```python
self.has_viscosity = bool(np.any(self.material_parameters.viscosity.numpy() > 0))
self.has_dilatancy = bool(np.any(self.material_parameters.dilatancy.numpy() > 0))
```

**Default adhesion**: 0.0 Pa (line 35). Adhesion is a collider property, not a particle property. It models attractive forces between particles and collider surfaces.

### 2.3 rheology_solver_kernels.py (1477 lines)

**Source**: `newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py`

**YieldParamVec** (lines 39-72): 6-component vector encoding yield surface:
- [0] p_max * sqrt(3/2) -- compressive yield pressure
- [1] p_min * sqrt(3/2) -- tensile yield pressure
- [2] s_max -- deviatoric yield stress
- [3] mu * p_max -- frictional shear limit
- [4] dilatancy
- [5] viscosity

**Yield surface model** (lines 92-107, `shear_yield_stress`): Piecewise-linear Drucker-Prager with tension cutoff. For a given normal stress r_N, the maximum deviatoric stress is:
- If r_N < p1: s + mu * (r_N - p_min)
- If r_N > p2: s + mu * (p_max - r_N)
- Otherwise: s + mu * p2

**Backward mode explicitly disabled** (line 36):
```python
wp.set_module_options({"enable_backward": False})
```

### 2.4 contact_solver_kernels.py

**Source**: `newton/_src/solvers/implicit_mpm/contact_solver_kernels.py`

**Backward mode explicitly disabled** (line 7):
```python
wp.set_module_options({"enable_backward": False})
```

**Coulomb friction solver** (lines 42-83, `project_on_friction_cone` and `solve_coulomb_isotropic`): Standard Coulomb friction model with adhesion support. The adhesion parameter (line 90) allows modeling attractive forces.

---

## 3. Hydroelastic Contact System

### 3.1 sdf_hydroelastic.py

**Source**: `newton/_src/geometry/sdf_hydroelastic.py`

**Pipeline** (docstring lines 10-27):
1. Broadphase: OBB intersection tests between SDF shape pairs
2. Octree refinement: hierarchical subdivision to find iso-voxels
3. Marching cubes: extract contact surface triangles
4. Contact generation: contacts at triangle centroids
5. Contact reduction: reduce contacts via HydroelasticContactReduction

**Configuration**: `ShapeConfig(is_hydroelastic=True, kh=1e9)` (line 23)

**Relevance to underwater**: The hydroelastic contact system models compliant surface contact with distributed forces over contact patches rather than point contacts. This is useful for soft-body grasping and manipulation, not directly for fluid simulation.

### 3.2 Example: example_nut_bolt_hydro.py

**Source**: `newton/examples/contacts/example_nut_bolt_hydro.py`

Demonstrates mesh collision between a nut and bolt using hydroelastic SDF contacts. Not directly relevant to fluid simulation.

### 3.3 Example: example_robot_panda_hydro.py

**Source**: `newton/examples/robot/example_robot_panda_hydro.py`

Franka Panda arm with SDF hydroelastic contacts for pick-and-place manipulation. Demonstrates manipulation with compliant contact, not fluid interaction.

---

## 4. Differentiable Simulation Examples

**Source**: `newton/examples/diffsim/`

Six examples:
1. `example_diffsim_ball.py` -- particle bouncing, gradient of position w.r.t. initial velocity
2. `example_diffsim_bear.py` -- cloth/soft body bear optimization
3. `example_diffsim_cloth.py` -- cloth simulation optimization
4. `example_diffsim_drone.py` -- drone control optimization
5. `example_diffsim_soft_body.py` -- soft body material parameter optimization
6. `example_diffsim_spring_cage.py` -- spring cage optimization

**Key pattern** (from diffsim_ball.py lines 86-87):
```python
self.model = scene.finalize(requires_grad=True)
```

**Warp Tape-based autodiff** (lines 123-128):
```python
self.tape = wp.Tape()
with self.tape:
    self.forward()
self.tape.backward(self.loss)
```

**MPM is NOT differentiable**: All three MPM kernel modules explicitly disable backward mode:
- `implicit_mpm_solver_kernels.py` line 16: `wp.set_module_options({"enable_backward": False})`
- `rheology_solver_kernels.py` line 36: `wp.set_module_options({"enable_backward": False})`
- `contact_solver_kernels.py` line 7: `wp.set_module_options({"enable_backward": False})`

No diffsim example uses MPM.

---

## 5. Key Questions Answered

### 5.1 Can Newton's MPM simulate water (not just granular/snow)?

**Partially yes, with significant caveats.**

The viscous example (`example_mpm_viscous.py`) demonstrates viscoplastic fluid with water density (1000 kg/m^3), zero friction, and configurable viscosity. The multi-material example shows mud with density=1000, viscosity=100, friction=0.0.

**Water-like parameter recipe** (synthesized from viscous + beam examples):
```python
mpm_options = SolverImplicitMPM.Config()
mpm_options.voxel_size = 0.005  # fine grid for fluid detail

model.mpm.viscosity.fill_(0.001)       # water viscosity: ~0.001 Pa*s
model.mpm.friction.fill_(0.0)           # no friction
model.mpm.tensile_yield_ratio.fill_(1.0) # tension allowed
model.mpm.yield_pressure.fill_(1e15)     # high yield = nearly incompressible
model.mpm.young_modulus.fill_(1e15)      # very stiff
model.mpm.poisson_ratio.fill_(0.499)     # near-incompressible (water)
model.mpm.dilatancy.fill_(0.0)           # no volume change on shear
model.mpm.yield_stress.fill_(0.0)        # no yield stress
model.mpm.damping.fill_(0.0)             # no extra damping
```

**Limitations**:
1. The implicit MPM solver is designed for elasto-plastic and viscoplastic materials, not purely incompressible Navier-Stokes. There is no dedicated incompressibility constraint or pressure projection step.
2. Water's viscosity (0.001 Pa*s) is many orders of magnitude lower than the examples (50-100 Pa*s). At such low viscosity, the implicit solver may not provide significant benefit over explicit methods, and the yield surface model may introduce artifacts.
3. No surface tension model exists in the MPM solver.
4. No free-surface detection or rendering is provided (only particle-based rendering).
5. The Poisson ratio can approach 0.5 (incompressible) but the solver uses a penalty-based approach, not a true mixed formulation for incompressibility.

### 5.2 What's the particle count performance on RTX 5090?

**No benchmarks are available in the codebase.** None of the examples report particle counts or FPS metrics. The examples use these approximate scales:

| Example | Voxel Size | Bed Dimensions | Est. Particles |
|---------|-----------|----------------|----------------|
| twoway_coupling | 0.05m | 2x2x0.5m | ~60,000 |
| viscous | 0.005m | 0.1x0.1x0.2m | ~4,000 |
| multi_material | 0.05m | 1x1x0.25m per layer | ~30,000 |
| anymal | 0.03m | 1x3x0.15m | ~50,000 |
| granular | 0.1m | 2x2x2m | ~8,000 |
| snow_ball | 0.1m | 5x20x0.8m | ~100,000 |
| beam_twist | 0.25m | 5x1x1m | ~2,000 |
| grain_rendering | 0.1m | 1x1x2m | ~6,000 |

Performance claims would require actual benchmarks. The solver is GPU-only (the anymal example explicitly exits on CPU, line 343-345) and uses CUDA graph capture for acceleration.

The grid allocation supports `max_active_cell_count` (default: -1 = unlimited, fixed grid uses 2^15 = 32768 cells). The sparse grid (`fem.Nanogrid`) is the default for dynamic scenarios.

**Performance estimates based on architecture** (NOT benchmarked, extrapolation only):
- The implicit solver uses iterative methods (GS, CG, CR, GMRES) with max 250 iterations
- CUDA graph capture is used for fixed grids (lines 180-186 of twoway_coupling)
- RTX 5090 has 32,768 CUDA cores and 32 GB VRAM
- Rough estimate: 100K-500K particles at interactive rates (10-60 FPS) depending on voxel size and material complexity

### 5.3 Can MPM two-way coupling work with an AUV rigid body?

**Yes, this is explicitly supported and demonstrated.**

The two-way coupling example (`example_mpm_twoway_coupling.py`) shows:
1. Separate rigid body model and sand model (lines 117-118)
2. MPM solver reads colliders from the rigid body model (line 133)
3. MuJoCo solver handles rigid body dynamics (line 136)
4. Custom kernel computes forces from sand to rigid bodies (lines 26-55)
5. Impulse collection for reaction forces (lines 220-228)

The ANYmal example (`example_mpm_anymal.py`) extends this with:
1. Full robot from URDF (line 57-65)
2. Per-body collision filtering (`COLLIDE_PARTICLES` flag, lines 68-71)
3. Kinematic collider mode (`body_mass=wp.zeros_like(...)`, lines 155-158)
4. Or dynamic collider mode (default, using real body mass)
5. Pretrained RL policy control (line 177)

**For an AUV**, the pattern would be:
1. Load AUV URDF/USD into a `ModelBuilder`
2. Create MPM particles for water
3. Call `SolverImplicitMPM.register_custom_attributes(builder)` before adding particles
4. Set water material parameters on particles
5. Call `mpm_solver.setup_collider(model=auv_model)` to register AUV as collider
6. Step: solve AUV with MuJoCo, then solve MPM, then exchange forces

**Key method**: `setup_collider()` accepts `collider_adhesion` parameter (line 1008 of solver_implicit_mpm.py) for modeling attractive fluid-body forces (simplified buoyancy/adhesion).

### 5.4 How does the implicit MPM solver compare to explicit?

**Newton 1.2.0 only ships an implicit MPM solver.** There is no explicit MPM solver in the codebase. A search for `explicit_mpm`, `ExplicitMPM`, `SolverMPM` returned zero results.

The implicit solver's advantages (from docstring, line 623-624):
> This variant is particularly well-suited for very stiff materials and the fully inelastic limit. It is less versatile than traditional explicit MPM but offers unconditional stability with respect to the time step.

The solver's pipeline (lines 1554-1617):
1. Transfer particles to grid (PIC/APIC/GIMP)
2. Assemble implicit system (compliance + strain matrices)
3. Iterative solve (GS/CG/CR/GMRES) with warmstart
4. Update strains (elastic + plastic)
5. Advect particles

This means larger time steps are stable, which is beneficial for RL training. However, the "less versatile" caveat means some MPM features (e.g., explicit time integration, certain material models) are not available.

### 5.5 Is the MPM differentiable (can we get gradients)?

**No. The MPM solver is explicitly non-differentiable.**

All three MPM kernel modules disable Warp's automatic differentiation:
- `implicit_mpm_solver_kernels.py` line 16: `wp.set_module_options({"enable_backward": False})`
- `rheology_solver_kernels.py` line 36: `wp.set_module_options({"enable_backward": False})`
- `contact_solver_kernels.py` line 7: `wp.set_module_options({"enable_backward": False})`

The diffsim examples use `SolverSemiImplicit` (diffsim_ball.py line 94), `SolverXPBD` (cloth/soft body), or `SolverVBD` (soft body) -- none use MPM.

**Workarounds for gradient-based optimization**:
1. Use finite differences (as shown in `check_grad()` in diffsim_ball.py lines 210-248)
2. Use the rigid body + MuJoCo solver for the AUV (which IS differentiable) and treat the MPM fluid as an environment perturbation
3. Train RL policies using the MPM environment as a black-box simulator (model-free RL)

### 5.6 What's the API for adding custom fluid materials?

**Custom materials are defined via per-particle attributes.**

The API (from `register_custom_attributes`, lines 714-913):

```python
# Before adding particles:
SolverImplicitMPM.register_custom_attributes(builder)

# After model.finalize():
model.mpm.viscosity.fill_(0.001)           # Pa*s
model.mpm.friction.fill_(0.0)
model.mpm.young_modulus.fill_(1e15)        # Pa (infinite = rigid/pressure-based)
model.mpm.poisson_ratio.fill_(0.499)       # near-incompressible
model.mpm.damping.fill_(0.0)              # s
model.mpm.yield_pressure.fill_(1e15)       # Pa
model.mpm.tensile_yield_ratio.fill_(1.0)   # 1.0 = equal tension/compression
model.mpm.yield_stress.fill_(0.0)          # Pa
model.mpm.hardening.fill_(0.0)
model.mpm.hardening_rate.fill_(1.0)
model.mpm.softening_rate.fill_(1.0)
model.mpm.dilatancy.fill_(0.0)
```

**Per-particle assignment** (from multi_material example):
```python
water_indices = wp.array([...], dtype=int, device=model.device)
model.mpm.viscosity[water_indices].fill_(0.001)
model.mpm.friction[water_indices].fill_(0.0)
```

**No custom constitutive model API.** The yield surface is fixed as a Drucker-Prager variant with viscosity and dilatancy extensions. You cannot define arbitrary constitutive models -- you can only tune the existing parameters.

---

## 6. Summary and Recommendations for Underwater Robotics

### What Newton MPM Can Do
1. **Viscoplastic fluid simulation** with configurable viscosity, friction, and density
2. **Two-way coupling** between MPM particles and rigid body robots
3. **GPU-accelerated** simulation with CUDA graph capture
4. **RL policy integration** via PyTorch JIT (demonstrated with ANYmal)
5. **Multi-material** simulation in a single scene

### What Newton MPM Cannot Do
1. **True incompressible fluid** -- no pressure projection or incompressibility constraint
2. **Differentiable simulation** -- backward mode explicitly disabled
3. **Surface tension** -- not modeled
4. **Free surface rendering** -- only particle-based visualization
5. **Explicit MPM** -- not available; only implicit solver
6. **Custom constitutive models** -- fixed Drucker-Prager + viscosity + dilatancy

### Recommended Approach for OceanScale
For simulating an AUV in water, the most practical approach with Newton 1.2.0 is:

1. **Use the two-way coupling pattern** from `example_mpm_twoway_coupling.py`
2. **Load AUV as a URDF** with collision shapes, following `example_mpm_anymal.py`
3. **Tune MPM parameters for viscous fluid** following `example_mpm_viscous.py` with reduced viscosity
4. **Use RL training** (model-free) rather than gradient-based optimization
5. **Start with coarse voxel size** (0.05-0.1m) for fast iteration, refine later
6. **Consider whether MPM is the right choice** -- Newton's rigid body solvers (MuJoCo, Featherstone) with simple hydrodynamic forces may be more practical for RL training than full MPM fluid simulation

### Performance Expectations
Without running benchmarks, realistic expectations for RTX 5090:
- 50K-100K particles at 5cm voxel: likely 30-60 FPS (interactive)
- 100K-500K particles at 2cm voxel: likely 5-15 FPS (offline/slow)
- Sub-millimeter voxels (as in viscous example): only small domains practical

---

## 7. File Index

| File | Lines | Content |
|------|-------|---------|
| `newton/examples/mpm/example_mpm_twoway_coupling.py` | 391 | Rigid-sand two-way coupling |
| `newton/examples/mpm/example_mpm_viscous.py` | 278 | Viscoplastic fluid in funnel |
| `newton/examples/mpm/example_mpm_multi_material.py` | 197 | Sand + snow + mud |
| `newton/examples/mpm/example_mpm_anymal.py` | 348 | ANYmal walking on sand |
| `newton/examples/mpm/example_mpm_granular.py` | 286 | Granular with colliders |
| `newton/examples/mpm/example_mpm_snow_ball.py` | 410 | Snow avalanche + ball |
| `newton/examples/mpm/example_mpm_beam_twist.py` | 332 | Elastic beam twist |
| `newton/examples/mpm/example_mpm_grain_rendering.py` | 143 | Grain rendering |
| `newton/_src/solvers/implicit_mpm/solver_implicit_mpm.py` | 2554 | Main solver |
| `newton/_src/solvers/implicit_mpm/implicit_mpm_model.py` | 598 | Model + material params |
| `newton/_src/solvers/implicit_mpm/implicit_mpm_solver_kernels.py` | ~600 | Grid/particle kernels |
| `newton/_src/solvers/implicit_mpm/rheology_solver_kernels.py` | 1477 | Yield surface + iterative solve |
| `newton/_src/solvers/implicit_mpm/contact_solver_kernels.py` | ~300 | Friction + impulse |
| `newton/_src/solvers/implicit_mpm/solve_rheology.py` | ~500 | Solver orchestration |
| `newton/_src/geometry/sdf_hydroelastic.py` | ~600 | Hydroelastic contacts |
| `newton/examples/diffsim/example_diffsim_ball.py` | 276 | Diffsim with rigid bodies |
