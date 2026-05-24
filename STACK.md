# Tech Stack — In-Depth Analysis (v0.1.0)

**Date:** 2026-05-15  
**Scope:** Layer-by-layer technical analysis of the proposed next-gen underwater simulator stack — what each component is, why it was chosen, how it works, where it fits, when it runs, alternatives considered, and known gotchas.

Companion docs: `SURVEY.md` (state-of-the-art), `ARCHITECTURE.md` (architecture).

---

## 0. How We Choose the Main Stack — Decision Framework

This is the most consequential set of decisions in the project. Picking wrong burns 6-12 months of refactoring later. Picking by "what's trending on arXiv this week" guarantees you pick wrong. This section is the explicit framework; §1-§16 are its outputs; §17 is the lock + watchlist.

### 0.1 The Five Forces Acting on Stack Choice

Every layer is pulled by five competing forces:

1. **Capability** — does it do the thing we need?
2. **Maturity** — will it still work in 12 months without us patching it?
3. **Ecosystem fit** — does it compose cleanly with our other choices?
4. **Differentiation cost** — where do we *want* to spend our innovation budget?
5. **Reversibility** — if we're wrong, how painful is migration?

These conflict. The state-of-the-art (max capability) is by definition immature. The most mature (e.g., MuJoCo CPU) underdelivers on capability. The framework is **how we trade them off layer by layer**.

### 0.2 Where We Add Value vs. Where We Buy

The first question for every layer:

> Is this layer *where our research contribution lives*?

If **yes**, we build it ourselves and depend on nothing fragile. Differentiation costs money — spend it where it matters.

If **no**, we buy mature, well-governed dependencies and don't touch them. Buying = depending on someone else's correctness, governance, and roadmap.

Applied to our stack:

| Layer | Our value? | Decision |
|---|---|---|
| Hydrodynamics on GPU | **YES** — this is the whole point | **Build** (custom Warp kernels) |
| Underwater sensor physics (sonar, AK-T, DVL) | **YES** | **Build** |
| Benchmark tasks & DR for marine | **YES** | **Build** |
| Multi-medium integration (water/air, vehicle/manipulator) | **YES** | **Build** |
| General rigid-body physics | no | **Buy** (Newton + MJWarp) |
| Scene description / asset format | no | **Buy** (OpenUSD) |
| RTX rendering | no | **Buy** (Omniverse) |
| RL algorithms (PPO, SAC, world models) | no | **Buy** (skrl/rl_games/RSL-RL/DreamerV3) |
| GPU kernel JIT + autograd | no | **Buy** (Warp) |
| Env management framework | no (compose) | **Buy** (Isaac Lab) |

**Rule of thumb**: every layer marked "buy" must be (a) widely-adopted, (b) well-governed, (c) license-compatible, (d) Linux-x86_64 + CUDA-12. If a candidate misses any one, it's not stable enough to buy — keep looking or build.

### 0.3 Maturity Tiers — What Counts as "Mature Enough"

We tier dependencies by governance + adoption + age:

- **Tier S** — Industry-consortium-governed, ≥3 years stable API, ≥1k production users.  
  Examples: OpenUSD (Pixar/LF), CUDA 12 (NVIDIA), Python 3.12, PyTorch 2.x.
- **Tier A** — Single-vendor but with consortium contributions; semver stable; multiple labs in production.  
  Examples: Newton 1.0 (LF + NVIDIA + DeepMind + Disney), Isaac Sim 6 (NVIDIA + ROS community), MuJoCo (DeepMind), Warp (NVIDIA, but only post-1.0).
- **Tier B** — Single vendor / single lab; active maintenance; semver-ish; some production use.  
  Examples: rl_games (one core dev but Isaac Lab default), skrl (one core dev, multi-paradigm), RSL-RL (ETH lab, robust).
- **Tier C** — Research project, code accompanying a paper, no API stability promise.  
  Examples: MarineGym, OceanSim, GS-Playground, EasyUUV.

**Rule:**
- **Core simulator** (physics + scene + kernel runtime): **Tier A or above**, no exceptions.
- **RL algorithm + env framework**: Tier B acceptable because env-agnostic — cheap to swap.
- **Tier C projects** are **read** for ideas and **wrapped** for compatibility; never imported as a hard dep.

This is the rule that killed GS-Playground as a dep but kept it useful for ideas.

### 0.4 Ecosystem Gravity — Bet With the Field, Not Against It

The robotics simulation field has visible centers of gravity in 2026:

- **Scene format**: USD wins. URDF/SDF/MJCF all have USD bridges. Roll-your-own JSON is over.
- **Physics**: GPU-native is the new floor. CPU-only physics (Gazebo classic, PyBullet) is a research toy, not a foundation.
- **Open governance**: LF + Apache-2.0 is the new robotics norm. NVIDIA proprietary works but adds friction.
- **Kernel programming**: Warp won. Triton stayed in LLM-land; Numba stayed CPU-leaning; raw CUDA is for hot inner loops only.
- **RL framework**: vectorized GPU envs (Isaac Lab class) won. Single-env Gym wrappers around CPU sims are a dead end at scale.

We pick the centers of gravity. This is *not* the same as "pick the newest" — USD is 10 years old, MuJoCo is 12 years old, Apache-2.0 is older than most of us. The newest tech is rarely the center of gravity; the consolidating consensus is.

**Rule**: if a candidate is *not* on a visible center of gravity, default to skepticism.

### 0.5 Reversibility & Lock-In — Choose Stickiness Carefully

Every dep has a cost-to-migrate. Roughly ranked, from cheapest to switch to most expensive:

```
RL algorithm    < Renderer < Env framework < GPU kernel lang
< Physics engine < Scene format < License
```

Implication: we can afford to experiment at the RL-algorithm layer (try rl_games, then RSL-RL, then DreamerV3 — env stays the same). We **cannot** afford to experiment at the scene-format layer (USD → SDF → URDF would re-cost the whole asset library).

**Rule**: spend caution budget on the right end of the spectrum. Lock in scene format and license *first*; treat algorithm choices as fluid.

This is also why we keep PhysX fallback in Isaac Lab — physics engine is sticky, so we hedge.

### 0.6 The Trap: Confusing Capability with Suitability

The most common stack mistake in research code is **confusing the best capability claim with the right fit**.

Three concrete traps observed in this survey:

1. **Throughput-blind selection.** GS-Playground's "10⁴ FPS at 640×480" is a *rendering* number. Underwater RL is mostly *not* rendering-bound. Choosing a renderer because of vision-task FPS when 90% of training is state-based wastes the optimization.
2. **Newest-paper bias.** A 2026-04 RSS paper *feels* like it must be more advanced than a 2024 LF project. But Newton 1.0 has 4 institutions, 1.8k commits, and consensus governance; GS-Playground is one lab's preview.
3. **Algorithmic-completeness bias.** Genesis has SPH + MPM + PBD — sounds underwater-ready! But it has no AUV models, no hydrodynamic coefficients, no Fossen validation. The solvers exist but the *domain glue* doesn't.

**Rule**: read the benchmark methodology, not the benchmark number. Read what's *not* shipping, not just what is. Read the author's own "limitations" section first.

### 0.7 The Process

For every stack-level decision:

1. **Define the layer's job** in one sentence.
2. **List 3-5 candidates** (mature first, then trending).
3. **Score each on Five Forces** (§0.1).
4. **Apply maturity tier rule** (§0.3) — eliminate any failing the tier floor.
5. **Apply ecosystem gravity check** (§0.4) — prefer the consolidation, not the new wave.
6. **Apply reversibility check** (§0.5) — spend caution where it's sticky.
7. **Document why-rejected** for each alternative — future-you will ask.
8. **Pin the choice and version** in §17 (Main Stack Lock).
9. **Anything that loses to the chosen pick goes on Watchlist** (§17.2), reviewed every 2 weeks.

For new arrivals (papers, repos, projects we discover mid-flight):

10. **Default: read for ideas, do not import.**
11. If it seems load-bearing: write a §16-style critical evaluation against the New-Arrival Policy (§17.3).
12. Adoption requires passing **all five bars** in §17.3 + explicit decision recorded.

### 0.8 What We Are Betting On (and Could Be Wrong About)

Being honest about uncertainty is part of the framework. The main stack rests on these bets:

| Bet | Confidence | What would invalidate it |
|---|---|---|
| LF + NVIDIA + DeepMind + Disney keep Newton converging | High | One major partner exits; license restructured |
| USD adoption continues across robotics | Very high | Would require Apple/Pixar/NVIDIA to all reverse |
| Warp's API stabilizes post-1.0 | Medium-high | 2.x breaking changes — already saw `warp.sim` removed |
| Isaac Lab 3.0 Newton path goes GA in 2026 | Medium | If experimental status persists, we stay on PhysX longer |
| PyTorch ≥ JAX in robotics | Medium-high | DeepMind / Brax shift could move the field; we mitigate via skrl multi-paradigm |
| RTX 5090 / Blackwell drivers remain stable | High | One bad driver = lost weeks |
| BlueROV2 Heavy as validation target stays available | Very high | BlueRobotics is profitable, ubiquitous |

We accept these bets. We **do not** bet on:
- Any single-lab preview project (GS-Playground, EasyUUV, etc.) staying maintained.
- Any specific RL algorithm being SOTA in 12 months — that's why the env is decoupled.
- Any 2026 paper's benchmark numbers reproducing — we re-benchmark on our own hardware.

### 0.9 When to Re-Open the Question

Stack lock is not forever. Trigger conditions for re-evaluation:

1. **A Tier-A dep ships a major version bump** with breaking changes (e.g., Newton 2.0, Isaac Lab 4.0).
2. **A new entrant earns Tier-A status** — happens roughly annually in this field.
3. **A core bet from §0.8 is invalidated** — partner exits, governance shift, license change.
4. **Our v0.x retro shows a layer is dominating maintenance time** — that's the layer we picked wrong.
5. **Annual review** — late January each year, audit Watchlist against Main Stack.

Anything else: maintain the lock. Read the new paper, write a Watchlist entry, move on.

---

---

## 0. Stack Bird's-Eye

```
┌──────────────────────────────────────────────────────────────────────┐
│ L7  RESEARCHER UX        Jupyter • TensorBoard • W&B • Trackio        │
│ L6  RL ALGORITHMS        skrl 2.0 • rl_games • RSL-RL • DreamerV3     │
│ L5  ENV / TASK LAYER     Isaac Lab 3.0 ManagerBasedEnv + our terms    │
│ L4  SIM CORE             OceanScale package (this project)            │
│                          ├ Hydro kernels (Warp, Fossen + SPH + waves) │
│                          ├ Sensor kernels (sonar RT, AK-T vision, DVL) │
│                          ├ Acoustic comms (BELLHOP + Q-D)             │
│                          └ Scenario/DR toolkit                        │
│ L3  PHYSICS              Newton 1.x (MuJoCo-Warp + Kamino + MPM)      │
│ L2  GPU PRIMITIVES       NVIDIA Warp 1.13 (Python→CUDA, autograd)     │
│ L1  RENDERING            Omniverse RTX (Isaac Sim 6) • NuRec GS       │
│ L0  SCENE                OpenUSD (UsdPhysics + mjcPhysics + Newton)   │
│ L−1 RUNTIME              CUDA 12 • PyTorch 2.7 • JAX 0.6 • Python 3.12│
└──────────────────────────────────────────────────────────────────────┘
```

Each layer below covers: **What / Why / How / Where / When / Alternatives / Gotchas**.

---

## 1. OpenUSD — Scene Graph (Layer 0)

### What
Universal Scene Description: hierarchical, composition-based 3D scene format from Pixar / Linux Foundation. Stores geometry, materials, transforms, physics (via `UsdPhysics`, `mjcPhysics`, `NewtonPhysics` schemas), sensors, lights, animation. ASCII (`.usda`), binary (`.usdc`), packaged (`.usdz`), or layered via `.usd`.

### Why
1. **Single source of truth** — vehicles, terrain, water, tether, payload, sensors all live in one composed graph. Newton, Isaac Sim, MuJoCo, Omniverse all consume USD natively in 2026.
2. **Composition (referencing + variants + layers)** — a benchmark scenario is a thin USD layer overriding a base AUV asset; domain randomization is a variant-set on hydrodynamic coefficients.
3. **Instancing** — 8,000 parallel AUV envs share one prim master via `UsdGeomPointInstancer` or `UsdSkel`; memory stays O(scene) not O(envs).
4. **Version control friendly** — Isaac Sim 6 split assets into `geometry (.usdc)` and `physics/materials (.usda)` so meaningful diffs are possible in git.
5. **Bidirectional with MJCF/URDF** — Newton ships `mujoco-usd-converter`; Isaac Sim 6 importers go both ways. We do not pick a side.

### How
- Asset structure 3.0 (Isaac Sim 6.0): `vehicle.usd` references `geometry.usdc` (mesh+material binary) + `physics_physx.usda` + `physics_mujoco.usda` + `physics_newton.usda`. Backends pick what they need.
- Scene composition uses **sublayers** (additive) and **references** (instancing). Currents/wave field is a sublayer; the AUV is a reference.
- Custom schemas: we will define `oceanscale:Hydrodynamics` schema (Fossen coefficients, added-mass matrix), `oceanscale:Sonar` (cone, beam count, frequency), `oceanscale:Current` (velocity field).
- Runtime read via **Fabric Scene Delegate** (zero-copy, GPU-residence) in Isaac Sim 6.

### Where
Top of the stack. Every other layer consumes (or writes to) USD.

### When
- Authoring: Omniverse USD Composer, or `pxr.Usd` Python API for procedural scenes.
- Training: USD is loaded once into Fabric; subsequent state lives in GPU buffers.
- Resets: USD is *not* re-parsed every reset — Isaac Lab clones the GPU state from a cached snapshot.

### Alternatives considered & rejected
| Alt | Why rejected |
|---|---|
| URDF | No materials, no lights, no sensors, no composition. Dead-end for visual sim-to-real. |
| MJCF | MuJoCo-only. Not the right scene graph for multi-engine. |
| SDF (Gazebo) | Gazebo Harmonic only; CPU bound; community shifting to USD. |
| Unity prefabs | Locked into Unity proprietary tooling. MARUS' weakness. |
| Roll-your-own JSON | Re-inventing 15 years of Pixar work. |

### Gotchas
- **Learning curve** — USD has 10+ concepts (Stage, Layer, Prim, Reference, Inherit, Specialize, Composition Arcs, Variant Sets, Fabric, Hydra…). Plan a 2-week onboarding for new contributors.
- **Tooling fragmentation** — `pxr` Python API ≠ `omni.usd` Kit API. We commit to `pxr` (vanilla USD ≥25.11) at the core; Isaac Sim is an *editor*, not a runtime dependency.
- **USD ≥25.11 required** for nested body support that Newton 1.0 needs.

`★ Insight ─────────────────────────────────────`
USD isn't a 3D format; it's a **versioned, composable database for scene state**. Treat layers like git branches — your benchmark scenarios are PRs against base assets, not copy-pasted scene files. This is why USD scales to 8,000 envs where URDF chokes at 8.
`─────────────────────────────────────────────────`

---

## 2. NVIDIA Warp — GPU Kernel Layer (Layer 2)

### What
A Python-to-CUDA JIT framework: write `@wp.kernel` functions in restricted Python, launch them on the GPU with `wp.launch(...)`, get differentiable autograd via `wp.Tape()`. Interops zero-copy with PyTorch (`wp.to_torch`) and JAX (`wp.to_jax`).

### Why
1. **The only place to write custom physics in Python and get autograd for free.**
2. Newton, MuJoCo-Warp, OceanSim, MarineGym successors — all converge on Warp. It's the bottom of the next-gen GPU robotics stack.
3. JIT compiles each kernel once per shape signature; subsequent launches are pure CUDA.
4. Differentiable through the tape — gradients flow through hydro forces, sensor rays, and `wp.spatial_vector` SE(3) math.
5. Zero-copy interop avoids the `torch.tensor(numpy(...))` round-trip that kills throughput.

### How
```python
import warp as wp

@wp.kernel
def fossen_drag(
    state_vel: wp.array(dtype=wp.spatial_vectorf),  # 6-DOF velocities
    drag_lin: wp.array(dtype=wp.vec3f),             # diag(D_v)
    drag_quad: wp.array(dtype=wp.vec3f),            # diag(D_|v|v)
    out_force: wp.array(dtype=wp.spatial_vectorf),  # 6-DOF wrench out
):
    i = wp.tid()
    v = state_vel[i]
    # linear part: F = -(D_lin * v + D_quad * |v| * v)  componentwise
    lin = wp.spatial_top(v)          # ω
    ang = wp.spatial_bottom(v)       # v
    f = -(drag_lin[i] * ang + drag_quad[i] * wp.vec3f(
        wp.abs(ang.x)*ang.x, wp.abs(ang.y)*ang.y, wp.abs(ang.z)*ang.z))
    out_force[i] = wp.spatial_vector(wp.vec3f(0.0), f)
```
- One env per `tid()` — trivially parallel.
- `wp.array` lives on the GPU; the tape records ops; `tape.backward(...)` propagates gradients.
- Launch: `wp.launch(fossen_drag, dim=num_envs, inputs=[...])`.

### Where
Cross-cutting. **Three places we use Warp:**
1. Inside Newton's existing solvers (we don't touch).
2. As OceanScale's hydro/sensor/comm kernels (the bulk of our code).
3. In RL post-processing (e.g., reward computation, observation normalization) for zero-copy with Newton state.

### When
Compile: on first launch (JIT). Subsequent launches: pure CUDA dispatch (≤2 µs overhead per launch on H100).

### Alternatives considered & rejected
| Alt | Why rejected |
|---|---|
| Raw CUDA C++ | 10× development cost; no autograd unless we hand-roll. |
| Numba CUDA | No autograd; slower than Warp; less robotics ecosystem. |
| CuPy | NumPy-on-GPU; not kernel-level; no autograd. |
| Triton | Lower-level (block-tile programming); great for matmul, awkward for spatial physics. |
| JAX (`jax.jit` + `pmap`) | Excellent for end-to-end JAX (PureJaxRL track) but doesn't compose with PhysX/Newton's PyTorch-tensor world. We keep JAX as a *secondary* backend via skrl. |
| `torch.compile` + custom ops | Fragile for kernel-style code; long compile times. |

### Gotchas
- Warp's Python subset is **restricted** — no `numpy`, no `if-elif` chains beyond 3 branches in some cases, no dynamic memory inside kernels. Pre-allocate everything.
- Debugging is harder than PyTorch — `wp.print` is the main tool; use small `dim` (≤16) for development.
- Compile cache lives in `~/.cache/warp/`. Wipe on Warp version bump.
- `warp.sim` is **deprecated** as of Newton 1.0 — code that uses it is on borrowed time.

`★ Insight ─────────────────────────────────────`
Warp is the bottom of the dependency graph for the entire 2026 robotics stack. Newton is "just" a curated collection of Warp kernels with a state-management layer on top. Every custom hydrodynamic force, sonar ray, or wave field we write *is* the simulator — Newton is the contact + integrator harness around our kernels.
`─────────────────────────────────────────────────`

---

## 3. Newton 1.x — Physics Engine (Layer 3)

### What
Open-source, GPU-accelerated, multi-solver physics engine built on Warp + OpenUSD. Ships at GTC 2026 (1.0 GA, 2026-03-18; v1.2.0 2026-05-12). Apache-2.0. Linux Foundation governance.

Solver lineup:
- **Rigid body:** MuJoCo-Warp (primary), Kamino (Disney's closed-loop chains), Featherstone (legacy), SemiImplicit, XPBD.
- **Deformables:** Kamino VBD (Vertex Block Descent) — cables, cloth, soft tissue.
- **Granular / soft volumetric:** MPM (Material Point Method) with two-way rigid coupling.
- **Contact:** SDF library + hydroelastic contact (Drake-derived).
- **No native fluid solver** — this is the gap we fill.

### Why
1. **Drop-in physics that's not vendor-locked** — Apache-2.0, Linux Foundation. Unlike PhysX (NVIDIA TOS).
2. **Differentiable** — Warp tape propagates through Newton solvers. Critical for system ID and policy-gradient methods that exploit dynamics gradients.
3. **USD-native** — Newton's state lives on USD prims; no asset translation step.
4. **Multi-solver in one engine** — we can use MuJoCo-Warp for the AUV body and Kamino VBD for the tether in the *same* timestep.
5. **First-class Isaac Lab integration** — Isaac Lab 3.0 Beta already has Newton as a swappable backend.
6. **Performance**: NVIDIA-claimed 252× MJX (locomotion), 475× MJX (manipulation) on RTX PRO 6000 Blackwell. Independent benchmarks not yet published — treat as "best case".

### How
- **State**: `newton.Model` (kinematic structure, masses, joint limits) + `newton.State` (q, qd, qdd) + `newton.Control` (actuator inputs). All `wp.array` under the hood.
- **Stepping**: `solver.step(model, state, control, dt)`. Pluggable solver enumeration.
- **Coupling**: solvers communicate via shared `Contacts` structure; rigid-deformable two-way via constraint forces.
- **Custom forces** (our hook): pre-step callbacks add `wp.spatial_vectorf` wrenches to `state.body_f`. This is exactly where Tier-1 Fossen drag and Tier-3 wave forcing inject.

### Where
Layer 3. Sits between OceanScale's Warp kernels (which it consumes via pre-step hooks) and Isaac Lab (which wraps it).

### When
Every simulation timestep. Default `dt = 1/240 s` (HoloOcean parity); we'll allow `dt = 1/500 s` for fast manipulator dynamics.

### Alternatives considered
| Engine | Status | Why not primary |
|---|---|---|
| **PhysX 5** | Mature, fastest, NVIDIA proprietary | License limits redistribution; we want LF-governed code. Keep as **fallback** via Isaac Lab 3.0 multi-backend. |
| **MJX / MJWarp direct** | Stable, JAX-native (MJX), JAX/Warp (MJWarp) | Loses Newton's deformables + USD-native state. Keep as **secondary** for end-to-end JAX track. |
| **PyBullet / Bullet** | Stonefish/MarineGym use Bullet history | CPU; no autograd; no GPU parallelism at our scale. |
| **DART** | Academic | Smaller ecosystem; no GPU. |
| **Genesis** | Universal solvers (SPH+MPM+PBD) | Marine not built-in (issue #682); single-vendor; smaller community than Newton. Keep as **inspiration**. |
| **Brax** | JAX-native physics | Deprecating physics; pivoting to RL library. |
| **RaiSim** | Locomotion-focused, commercial | Closed source for some features. |

### Gotchas
- Newton's Isaac Lab integration is **experimental** (May 2026): "breaking changes expected", deformables + grippers PhysX-only in `isaaclab_physx`. We commit a **dual-backend** strategy until Newton-in-IsaacLab GA.
- "475× MJX" claim is on a **specific benchmark** (manipulation, RTX PRO 6000 Blackwell). Real-world underwater RL gains will be smaller because we add custom kernels. Plan for 100-150× over Gazebo-class baselines, not 500×.
- Newton's MPM is general but **slow for AUV-scale water bodies** (millions of particles). Use MPM only for localized free-surface near manipulators.

`★ Insight ─────────────────────────────────────`
Newton's signature design choice is **solver pluggability**. MuJoCo-Warp for rigid, Kamino for closed-loop, VBD for soft, MPM for granular — and *our SPH/Fossen kernels* slot in as another pluggable solver. The architecture invites third-party physics in a way that PhysX, fundamentally, does not.
`─────────────────────────────────────────────────`

---

## 4. MuJoCo-Warp (MJWarp) — Primary Rigid Solver (inside L3)

### What
MuJoCo 3.x's solver algorithms reimplemented in Warp. Co-developed by NVIDIA + Google DeepMind. The default rigid-body backend inside Newton.

### Why
1. **MuJoCo is the most-validated 3D rigid-body simulator in robotics** (decade-long use, contact accuracy, smooth gradients).
2. MJWarp inherits MuJoCo's **built-in stateless fluid model** (inertia-based + ellipsoid drag with Kutta lift, Magnus, blunt/slender drag) — our **Tier-0** hydro is free.
3. Optimized for **throughput** over latency — exactly the RL workload.
4. Same MJCF model spec as legacy MuJoCo — assets port for free.

### How
- Compiles MJCF → Warp arrays. Step is a single CUDA dispatch per env in the batch.
- Fluid forces computed alongside contacts in the same step. Per-geom 5 tunable params (density, viscosity, blunt drag, slender drag, angular drag).
- Implicit / implicitfast integrators required for stability under fluid forces (we use `implicit`).

### Where
The dominant rigid-body workhorse inside Newton. Every AUV body is an MJWarp body unless we explicitly opt-out.

### When
Every timestep. Configured at scene load time via `mjcPhysics` schema in USD.

### Alternatives considered
- **MJX (JAX)** — equivalent capability, JAX backend. Use for end-to-end JAX (PureJaxRL/Stoix). Keep alive as MJX adapter in OceanScale.
- **Kamino** — keep for closed-loop chains (manipulator parallel linkages, drones, but not main AUV).
- **Featherstone (Newton legacy)** — slower; use only for compatibility checks.

### Gotchas
- MJWarp is **less performant than CPU MuJoCo for a single env** (10× slower). It pays off at batch ≥256.
- MJWarp scales worse than MJX for very-many-geom scenes (>1000 geoms). Our scenes are <500 geoms (vehicles + terrain + sonar reflectors) so we're fine.
- The fluid model is **stateless** — no wakes, no vortex shedding, no fluid-fluid interaction. Hence Tier-2 Warp SPH for those phenomena.

`★ Insight ─────────────────────────────────────`
The MuJoCo ellipsoid fluid model is wildly underrated for AUVs. It captures added-mass-like effective inertia, lift via Kutta/Magnus, and quadratic drag — covering 80% of an AUV's analytic Fossen model for free, on the GPU, with autograd. Most underwater papers reach for Fossen-via-PyTorch (MarineGym) before checking what MuJoCo already gives.
`─────────────────────────────────────────────────`

---

## 5. OceanScale Core — Our Custom Layer (Layer 4)

### What
The actual contribution: Warp kernels and USD schemas implementing underwater-specific physics and sensors that Newton/MJWarp lack.

### 5.1 Hydrodynamics
Four tiers, configurable per-vehicle / per-task:

| Tier | What | Cost (RTX 5090, per env, per step) | When |
|---|---|---|---|
| 0 | MJWarp built-in ellipsoid fluid | ~free | Default. Cruise, station-keep, basic tasks. |
| 1 | Custom Warp Fossen kernel (full added-mass M_A matrix + Coriolis C_A(ν) + non-diagonal damping) | ~5 µs | When MuJoCo's diagonal ellipsoid is too coarse — e.g., flat-plate AUVs, asymmetric hulls. Matches MarineGym fidelity. |
| 2 | Localized Warp SPH around manipulator/thruster (~5k particles per ROI) | ~200 µs / ROI | Manipulator FSI, thruster wake, plume tracking. |
| 3 | 2D Stable-Fluid Eulerian for free surface, Gerstner waves | ~50 µs (shared across all envs in scene) | Surface vehicles, near-surface AUVs. |

**Force injection**: Tier 1-3 forces accumulate into a `wp.spatial_vectorf` per body and inject via Newton's pre-step hook into `state.body_f`. Gradients flow back through `wp.Tape`.

### 5.2 Sensor Kernels
| Sensor | Kernel | Throughput target |
|---|---|---|
| Single-beam echo sounder | BVH ray cast, time-of-flight | <0.01 ms / vehicle |
| Multibeam (256 beams) | 256-wide ray bundle, energy aggregation | <0.5 ms / vehicle |
| Side-scan / FLS / imaging | Beam-fan ray sweep + per-bin energy + Lommel-Seeliger BRDF | <2 ms / vehicle |
| DVL (4-beam Doppler) | Per-beam range + relative velocity + dropout logic | <0.05 ms / vehicle |
| IMU | Analytic linear/angular accel + Allan-variance noise model | <0.005 ms / vehicle |
| Pressure | depth + Gaussian noise | trivial |
| Camera RGB | Omniverse RTX path-trace + Akkaynak-Treibitz (AK-T) post Warp pass | ~5 ms / view (bottleneck) |
| Modem (acoustic) | BELLHOP precomputed channel impulse + Q-D Doppler/MP, runtime lookup | <0.1 ms / pair |

### 5.3 Wave / Current Field
- Currents: time-varying `wp.array` velocity field on a 3D grid (`wp.HashGrid` for sparse memory). Sampled per body at COG + 6 control surfaces.
- Waves: Gerstner sum-of-sines kernel for visuals + analytic gradient for wave-induced acceleration on near-surface bodies. Stable-Fluid 2D for higher-fidelity surface coupling (Tier 3).

### 5.4 Domain Randomization
A `RandomizerCfg` term in Isaac Lab applies per-env perturbations at reset time:
- Inertial: mass × U(0.8, 1.2), COB-COM offset (±5 cm), added-mass matrix scaling.
- Actuator: thruster gain × U(0.7, 1.3), deadband, time constant.
- Environment: current direction (uniform on sphere), current speed N(μ, σ), wave height (Pierson-Moskowitz), water Jerlov type {I, II, III, 1C, 5C, 9C}, turbidity (β), salinity.
- Sensor: IMU bias (Markov), DVL dropout rate, sonar SNR, camera ISO.
- Faults: thruster failure (1 of N), hull damage (extra drag), tether tension spike.

### 5.5 Where this code lives
`oceanscale/` Python package:
```
oceanscale/
├── hydro/      # Tier 0-3 kernels + USD schema
├── sensors/    # sonar/dvl/imu/camera/comms kernels
├── scene/      # USD authoring helpers, scenarios
├── tasks/      # Isaac Lab env definitions
├── randomize/  # DR terms
└── viz/        # Live overlay, plotting
```

### When (call order per step)
```
1. Isaac Lab gym.step(actions)
2. action → actuator force (per-vehicle thruster map)
3. OceanScale pre-step hook:
     • sample current field at each body COG
     • compute Fossen drag/added-mass forces (Tier 1)
     • compute wave forcing on near-surface bodies (Tier 3)
     • inject all into Newton state.body_f
4. Newton.solver.step(dt) — MJWarp integrates rigid bodies,
   Kamino integrates tether, MPM integrates manipulator FSI region
5. OceanScale post-step hook:
     • compute observations: DVL, IMU, pressure, sonar (sub-sampled to sensor Hz)
     • compute reward terms
     • record contacts / collisions for termination check
6. Return obs/reward/done to RL.
```

---

## 6. Isaac Sim 6.0 + Omniverse RTX — Rendering & Tools (L1)

### What
NVIDIA's USD-native robotics development environment. Provides:
- **RTX renderer** — real-time path tracing for RGB/depth/normals/segmentation/thermal.
- **NuRec** — 3D Gaussian Splatting library, photoreal neural reconstructions from real footage.
- **Replicator** — synthetic-data generator with domain randomization.
- **URDF/MJCF/USD importers**.
- **Kit framework** — extension system for custom GUIs and pipelines.

### Why
1. RTX rendering is **state-of-the-art for visual sim-to-real** — Lumen-class lighting, caustics, subsurface scatter. Photorealism that beats Unity and competes with UE5.
2. NuRec gives us "snap a real ocean basin, train against it" workflow — unique to NVIDIA's stack.
3. Replicator's randomization saves us months of writing our own DR for visual variation.
4. Isaac Sim 6.0 introduces **multi-physics-backend** (PhysX ↔ Newton), so we're not locked.
5. Active ROS 2 Jazzy + Humble integration, hardware-accelerated H.264 publishing.

### Why not Unreal Engine 5 (HoloOcean's path)
- UE EULA changed in 2024-2025; HoloOcean had to leave PyPI. Friction.
- UE5 is a game engine; programmable robotics workflows are second-class.
- Headless training requires custom build pipeline.
- We sacrifice some visual edge (UE5 Lumen > Omniverse RTX in some scenes) for tooling.

### Why not Unity (MARUS' path)
- Closed-source rendering pipeline.
- Smaller robotics community in 2026.
- USD support exists but is afterthought, not native.

### How
- We use Isaac Sim only at **dev time** (Composer GUI, Replicator authoring). Headless training runs on **`isaacsim --enable-omni-isaac-headless-native`** or **Isaac Lab kit-less** (3.0 introduces this; no Kit needed for pure compute).
- USD scenes authored in Composer. Renderer uses Hydra → RTX backend.
- For training, we render only when the policy is vision-conditioned; for state-based RL we skip rendering entirely (10× throughput gain).

### Where
L1 (rendering). Optional at training time, mandatory at evaluation/visualization time.

### When
- Dev: always, for scene authoring.
- Training (state-based RL): **never** — skip Hydra, run pure compute on Newton.
- Training (vision-based RL): every N steps (sub-sampled to camera Hz, typically 30 Hz).
- Eval: always, for video logging.

### Gotchas
- Isaac Sim 6.0 is **Early Developer Release** (May 2026) — GA expected later in 2026. Plan a binary-rebuild step in CI when GA drops.
- License: Isaac Sim code is Apache-2.0, but the **Omniverse Kit + assets** use NVIDIA Isaac Sim Additional Software License. Free for R&D, redistribution requires NVIDIA AI Enterprise.
- Renderer requires an RTX-class GPU (Maxwell+ for compute, RTX 20-series+ for ray tracing). Our headless training can use H100/L40s/B200 (no display, RTX OK).
- Windows 10 support dropped 2025-10-14. Linux Ubuntu 22.04/24.04 is the supported path.

---

## 7. Isaac Lab 3.0 — RL Environment Framework (L5)

### What
Isaac Sim's RL training framework. `ManagerBasedEnv` with composable terms:
- **SceneCfg** — what's in the world.
- **ObservationsCfg** — what the agent sees.
- **ActionsCfg** — how the agent acts.
- **RewardsCfg** — what's rewarded.
- **TerminationsCfg** — when episodes end.
- **EventsCfg** — domain randomization, perturbations.
- **CurriculumCfg** — task progression.

3.0 Beta adds:
- **Factory-based multi-backend** — same env, PhysX or Newton.
- **Warp-native data pipelines** — `.data.*` properties return `wp.array`, not `torch.Tensor`. Convert with `wp.to_torch()` only where needed.
- **Pluggable renderers** — RTX or alternative.
- **Kit-less install** — compute-only deployments without Omniverse Kit.

### Why
1. **Doesn't reinvent gym** — `gym.Env` compatible, plugs into rl_games/skrl/RSL-RL/PureJaxRL out of the box.
2. **Composable** — terms can be mixed and inherited. Our `StationKeepingEnv` extends a base `UnderwaterEnv` that owns the hydro hooks.
3. **Multi-GPU/multi-node** — built-in NCCL with rl_games/RSL-RL/skrl.
4. **ONNX export** standard.
5. **Asymmetric actor-critic** (privileged critic state) is standard, which is essential for sim-to-real on AUVs.

### How
```python
@configclass
class StationKeepingEnvCfg(ManagerBasedRLEnvCfg):
    scene = StationKeepingSceneCfg(num_envs=8192, env_spacing=20.0)
    observations = ObservationsCfg(
        policy=ObsTerm(func=mdp.body_lin_vel_w),  # 3
        # + body ang vel, position error, depth, current estimate ...
        critic=ObsTerm(func=mdp.true_current_field),  # privileged
    )
    actions = ActionsCfg(thrust=ThrustActionCfg(asset_name="bluerov2_heavy"))
    rewards = RewardsCfg(
        position_error = RewTerm(func=mdp.l2_pos_error, weight=-1.0),
        energy = RewTerm(func=mdp.thrust_l2, weight=-0.01),
    )
    events = EventsCfg(
        reset_state = EventTerm(mode="reset", func=randomize_pose),
        randomize_current = EventTerm(mode="reset", func=randomize_current_field),
        randomize_mass = EventTerm(mode="reset", func=mdp.randomize_rigid_body_mass,
                                   params={"asset_cfg": SceneEntityCfg("bluerov2_heavy"),
                                           "mass_distribution_params": (0.8, 1.2)}),
    )
```

### Where
L5. Wraps Newton; is wrapped by rl_games/skrl/RSL-RL.

### When
Training loop entry/exit point. `env.step(actions)` is the canonical call.

### Alternatives considered
- **Roll-your-own gym wrapper around Newton**: more control, no upstream features. Reject unless Isaac Lab proves too rigid.
- **OmniIsaacGymEnvs**: deprecated; predecessor to Isaac Lab.
- **gymnasium + DM-Control**: no Newton support; doesn't help.

### Gotchas
- 3.0 Beta has **Newton in `isaaclab_newton`** subpackage; deformables, surface grippers, material randomization remain **PhysX-only** in 3.0.0. Plan: develop on PhysX, port to Newton when Newton gains those features (likely 3.1).
- Migration from 2.x to 3.0 is non-trivial; we start fresh on 3.x.
- ROS 2 Jazzy integration changed in 6.0; legacy 5.1 bridges may need rewriting.

---

## 8. RL Algorithm Backends (L6)

### 8.1 skrl 2.0 — ACTIVE DEFAULT (v0.1)
**What**: Modular RL library supporting PyTorch, JAX, and **Warp** backends. Works with Isaac Lab + MuJoCo Playground + Brax.

**Why default**: spans all three array libraries. We can switch from PyTorch (default) to JAX (for PureJaxRL-style end-to-end) without changing the env or algorithm code. Modular agents — policy + value + critic + memory are all swappable.

**Decision note**: SB3 deprecated in v0.0.3. skrl 2.0+ is the default. RSL-RL is the performance baseline for all comparisons.

**When**: most experiments. ~95% of training runs.

**Gotcha**: JAX backend may have CuDNN version conflict with PyTorch 2.7 (CuDNN 9.7 vs 9.8). Stick to one backend per env install.

### 8.2 rl_games 1.6.5 — Throughput
**What**: PPO/A2C/SAC implementations optimized for Isaac Lab. PBT (Population-Based Training), multi-node NCCL, ONNX export.

**Why**: ~10-15% faster than skrl-PyTorch for pure throughput. Battle-tested across NVIDIA benchmarks.

**When**: when we need raw FPS — early baselines, sample-efficient ablations.

### 8.3 RSL-RL v5 — Sim-to-real PPO
**What**: ETH's minimalist PPO + DAgger-style behavior cloning + symmetry augmentation + RND curiosity.

**Why**: ETH's sim-to-real story (ANYmal, Spot) is the gold standard for zero-shot transfer. The library has the *right defaults* for that workflow.

**When**: when targeting real hardware deployment. Once we have BlueROV2 Heavy validation.

### 8.4 PureJaxRL / Stoix — End-to-end JAX
**What**: Env + agent both in JAX, single `jit` over the whole rollout. 1000-4000× over PyTorch baselines on classic benchmarks.

**Why**: ablation / research speedup. When we want to run 10M-step ablations in minutes.

**When**: research experiments, not production training.

### 8.5 DreamerV3 / TD-MPC2 — World models
**What**: Latent dynamics models that predict observations forward, trained alongside actor-critic.

**Why**: sample efficiency. AUV training is expensive (250k FPS still ≪ 43M FPS Genesis Franka because of hydro). World models cut sample count 10-100×.

**When**: for the **sparse-reward, hardware-validated** tasks (docking, pipe inspection). DreamerV3's single-config-fits-all property is a big draw.

### 8.6 Stable-Baselines3 — Legacy
**What**: Popular PyTorch RL library used in v0.0.x for early prototyping.

**Status**: **Deprecated as of v0.0.3.** CPU rollout collection — slow with Isaac Lab vectorized GPU envs. Less Isaac Lab integration. Retained for backward compat only; do not start new experiments with SB3.

### Why not CleanRL
- Single-algorithm, research-focused; less Isaac Lab integration than skrl/rl_games.

---

## 9. Specific Algorithms / Models We'll Build

### 9.1 Tier-1 Fossen Warp Kernel
**What**: Full 6-DOF AUV dynamics — added-mass M_A, Coriolis C_A(ν), nonlinear damping D(ν) ν.

**Math (Fossen 2021, eq 8.71)**:
```
M_RB · ν̇ + C_RB(ν) ν + M_A · ν̇ + C_A(ν) ν + D(ν) ν + g(η) = τ
```
- M_RB inherited from MJWarp inertia.
- M_A, C_A, D — our kernel.
- g(η) — restoring force, also our kernel for non-symmetric hulls.
- τ — thruster forces from action mapping.

**Why kernel**: PyTorch tensor math (MarineGym's approach) loses ~2-3× perf vs Warp due to op-launch overhead at 8000+ envs. Warp fuses the lot.

**Differentiability**: yes — Tape captures the kernel.

### 9.2 Sonar Ray-cast Kernel
**What**: Warp BVH built from USD scene + per-beam ray launch + per-bin energy accumulation + scatter/absorption model.

**How**:
```
for each (env, beam, sub-ray):
    hit, dist, normal, material = wp.bvh.intersect(...)
    angle = dot(normal, -ray.dir)
    energy = beam_power * pow(angle, lambertian_exp)
              * exp(-absorption_coef * dist)
              * 1.0 / (dist*dist)
    bin = int(dist * speed_of_sound / bin_dt)
    atomic_add(image[env, beam, bin], energy)
```

**Tricks**:
- BVH built once per scene (static geom) + per-step refit (moving bodies).
- ATOMIC_ADD into image buffer — Warp supports.
- Speckle: per-bin multiplicative noise log-normal.

**Throughput**: ≥256 vehicles × 256 beams × 1024 bins on RTX 5090 at sensor Hz (~10 Hz).

### 9.3 Akkaynak-Treibitz Camera Model
**What**: Three-component image formation (direct + forward-scatter + backscatter), refined from the original Jaffe-McGlamery framework.

**Pipeline**:
1. Omniverse RTX renders Lambertian RGB + depth.
2. Warp post-pass applies:
   - **Direct**: I_d = I_rgb · exp(-c · z)  (Beer-Lambert, c = Jerlov-coef per channel; AK-T refines attenuation coefficients)
   - **Forward scatter**: convolve I_d with depth-dependent Gaussian PSF.
   - **Backscatter**: add B_∞ · (1 - exp(-c · z))  (ambient water glow).
   - Caustics: pre-baked or animated noise pattern modulated by depth.

**Differentiable** — used for inverse rendering ablations.

### 9.4 BELLHOP / Q-D Acoustic Channel
**What**: Underwater acoustic propagation. Bell­HOP precomputes channel impulse responses for a sound-speed profile + bathymetry; we sample at runtime.

**Caching**:
- Precompute IR for a 20×20 grid of (Tx_depth, Rx_depth) pairs at scene load.
- Q-D model overlays time-varying Doppler/multipath from instantaneous Tx-Rx geometry.
- Warp kernel does the lookup + IR convolution at run-time.

**Latency**: < 0.1 ms per (Tx, Rx) pair on H100.

**Why pre-compute**: full BELLHOP is ~10 ms (CPU). Not RL-fast.

---

## 10. Data Layout / Memory Plan

**Key principle**: everything on the GPU, no CPU round-trip.

| Buffer | Dtype | Size (8192 envs) | Lifecycle |
|---|---|---|---|
| Newton State (q, qd) | float32 | ~32 MB | persistent |
| MJWarp model (per env if random mass) | float32 | ~64 MB | reset-time |
| Sonar image buffer | float32 | 8192·256·1024·4B = 8 GB | persistent (ring of 4 frames) |
| Camera RGB | uint8 | 8192·480·640·3B = 7 GB | sub-sampled (skip 28/30 frames) |
| Current field | float32 | 256³·4B·3 = 200 MB | per-scene, persistent |
| Wave grid | float32 | 1024²·4B = 4 MB | persistent |
| Obs buffer (policy) | float32 | 8192·64·4B = 2 MB | per-step |
| Action buffer | float32 | 8192·8·4B = 256 KB | per-step |
| BELLHOP cache | float32 | 20·20·1024·4B = 1.6 MB | per-scene |

**RTX 5090 has 32 GB GDDR7** — comfortable for 8192 envs at state-based RL; capped to ~2048 envs for vision-based.

**H100 80GB** — 32k envs for state-based.

---

## 11. Dev Environment / Tooling (L−1)

### Runtime
- Ubuntu 24.04 LTS
- Python 3.12 (Isaac Lab 3.0 baseline; Isaac Sim 6.0 ROS 2 Jazzy support)
- CUDA 12.8
- PyTorch 2.7 (CuDNN 9.7)
- JAX 0.6 in separate env (CuDNN 9.8 conflict)
- Warp 1.13
- USD ≥25.11

### Package management
- **uv** (preferred; fast, Python 3.13-aware) or **mamba** fallback.
- One env per backend (PyTorch / JAX) to avoid CuDNN clash.

### Code quality
- ruff (lint + format), mypy (types), pytest, hypothesis (property tests for kernel correctness).
- pre-commit hooks.
- CI on RTX 5090 self-hosted runner (or rented H100 spot).

### Reproducibility
- Hydra for config; W&B for experiment tracking; deterministic CUDA flags toggled per experiment.

---

## 12. Risk Register (Technical)

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Newton-in-Isaac-Lab breaking-change burns a sprint | Medium | High | Pin to a Newton tag; PhysX fallback always-green. |
| MJWarp ellipsoid fluid model breaks at high-speed (>3 m/s) AUV | Low | Medium | Tier-1 Warp kernel handles the regime above 2 m/s. Validate. |
| RTX 5090 driver issues in CUDA 12.8 | Low | Medium | Pin driver 570+; test against H100 in CI. |
| BELLHOP precompute time at scene load is multi-second | High | Low | Cache to disk; reuse across resets. |
| ROS 2 Jazzy bridge regression in Isaac Sim 6.0 GA | Medium | Low | We don't depend on ROS for training, only eval. |
| Genesis releases its own marine extension and steals attention | Low | Medium | Cross-port our hydro to Genesis as a fallback. |

---

## 13. When Things Run (Lifecycle View)

```
Scene authoring time:
  ▸ USD composer opens vehicle USD + writes mjcPhysics + oceanscale:Hydrodynamics
  ▸ BVH for static geom is pre-built and cached
  ▸ BELLHOP IR grid is pre-computed and cached

Env construction (Python):
  ▸ IsaacLab.ManagerBasedRLEnv.__init__
    ▸ Newton.Model.from_usd(stage_path)
    ▸ OceanScale.register_kernels(model)
    ▸ Allocate wp.arrays for obs/reward/action

Per step (×8192 envs in parallel):
  ▸ action → torque (Warp kernel)
  ▸ pre-step hook: current sample, Fossen kernel, wave kernel → body_f
  ▸ Newton.step(dt=1/240s) — MJWarp + Kamino + MPM (per env)
  ▸ post-step: sensor kernels (sub-sampled to sensor Hz)
  ▸ obs/reward/term computed (Warp kernel)
  ▸ skrl/rl_games consumes obs → produces actions

Per reset (per env, on done):
  ▸ pose / velocity randomization (Warp kernel)
  ▸ EventsCfg randomizes mass / current / waves / sensor noise
  ▸ first action issued

Periodically (every N steps):
  ▸ Render sub-sample (vision-based RL only)
  ▸ Log to W&B / TensorBoard

Eval-time (offline):
  ▸ Same env, single-vehicle
  ▸ Full RTX path tracing + AK-T post-pass for video
  ▸ ONNX export, ROS 2 bridge for hardware validation
```

---

## 14. Where to Start (Concrete First-Sprint Tasks)

1. **Stand up environment**: Ubuntu 24.04, CUDA 12.8, Python 3.12, install Isaac Sim 6.0 Early Dev + Isaac Lab 3.0 Beta + Newton 1.2 + Warp 1.13.
2. **Smoke test**: run Isaac Lab's `Cartpole-v0` on Newton backend, then on PhysX backend. Verify ≥100k FPS / env on the lab's RTX 5090.
3. **Port MarineGym BlueROV2 Heavy**: convert its URDF + Fossen plugin to MJCF + USD; load into Newton; verify station-keeping baseline matches MarineGym's published numbers (250k FPS).
4. **First custom kernel**: write the Fossen drag Warp kernel from §5.1. Unit-test against MarineGym's PyTorch implementation on identical states (numerical match to 1e-5).
5. **First sensor kernel**: write the single-beam echo sounder ray-cast kernel; visualize against OceanSim's reference output.
6. **First Isaac Lab env**: `StationKeepingEnv` with Tier-0 fluid only; train PPO via rl_games for 10M steps; capture FPS curve.

By end of week 2, we should have **a trained station-keeping policy with verified throughput**, the foundational kernels in place, and the dev loop dialed in.

---

## 15. Cross-Cutting "Why" Summary

| Decision | Driving force |
|---|---|
| OpenUSD everywhere | Multi-engine compatibility; future-proof; LF-governed. |
| Newton as primary physics | Apache-2.0; Warp-native; MuJoCo-Warp inheritance; LF-governed. |
| Custom Warp kernels (not PyTorch) | 2-3× perf at 8k+ envs; fuses with Newton; differentiable. |
| Tier-0 first (MuJoCo ellipsoid fluid) | Free; validated; covers cruise regime; ship faster. |
| Isaac Lab 3.0 over rolling our own | ManagerBasedEnv composability; multi-GPU built-in; ONNX standard. |
| skrl as default RL backend | Spans PyTorch/JAX/Warp; doesn't lock us in. |
| Apache-2.0 license throughout | Industry adoption; matches Newton/Isaac/OceanSim. |
| Single-GPU first | Lab's RTX 5090 is the development reality. |
| Validate on BlueROV2 Heavy | Cheap, ubiquitous, MarineGym-compatible; lowest sim-to-real friction. |
| Wrap MarineGym hydro + OceanSim sensors initially | Credit instead of reinvent; both Apache-compatible. |
| Differentiable everywhere via Warp Tape | System-ID with CMA-ES *or* gradients; policy-gradient methods that exploit dynamics. |

---

## 16. GS-Playground & 3DGS Photoreal Rendering — Critical Evaluation

Added per request 2026-05-15.

### 16.1 What it actually is (verified facts)
- **Repo**: [`discoverse-dev/gs_playground`](https://github.com/discoverse-dev/gs_playground), MIT, 343 stars, **98.7% Jupyter Notebook / 1.3% Python**. The split is itself a signal — this is a paper companion, not a library.
- **Paper**: arXiv 2604.25459, **accepted to RSS 2026** (2026-04-28).
- **What it claims to be**: a "high-throughput photorealistic simulator" that pairs a parallel rigid-body physics engine (**MotrixSim**, an in-house ground-up engine) with **batched 3D Gaussian Splatting (3DGS)** rendering. Real2Sim from a single RGB image → neural reconstruction in <5 min.
- **Headline number**: ~10⁴ FPS at 640×480 over **2048 parallel scenes on a single RTX 4090**.
- **Validated tasks**: Unitree Go2 quadruped locomotion (10 min training, 1024 envs), Unitree G1 humanoid (6 hr, 2048 envs), Airbot Play block grasping, Go2 visual navigation. All deployed zero-shot or near-zero-shot on real hardware.
- **Predecessor**: [DISCOVERSE](https://github.com/discoverse-dev/DISCOVERSE) (arXiv 2507.21981), which paired MuJoCo + 3DGS.
- **Project page**: [gsplayground.github.io](https://gsplayground.github.io/).

### 16.2 Where it actually shines
1. **Rendering throughput**: outperforms Isaac Sim's ray-tracing renderer at 1280×720 batch sizes that OOM PhysX-RTX. For vision-based RL this is a real edge.
2. **Rigid-Link Gaussian Kinematics (RLGK)**: binds 3DGS clusters to rigid-body frames so visual state updates cost zero extra GPU time once bodies move. Physics-render sync is perfect at any speed. Genuinely clever.
3. **Point pruning** (PUP-3DGS / SpeedySplat heritage): ≥90% Gaussians removed with <0.05 PSNR drop. Makes 8k-scene batches memory-feasible.
4. **Real2Sim pipeline** (single RGB → neural reconstruction in 5 min, with object meshes + sub-mm poses): if it works as advertised, this is the fastest known path from real footage to RL-ready scene.
5. **End-to-end validated**: not a paper kernel — they trained policies in sim and ran them on real Go2 / G1.

### 16.3 Critical problems for **underwater** use
The endorsements above are real, but the architecture clashes with marine simulation on multiple axes:

**A. 3DGS bakes lighting into the assets — and lighting *is* the underwater physics.**  
- The team explicitly acknowledges: *"3D Gaussian Splatting struggles with handling randomized lighting and shadows. Asset generation is currently dependent on the lighting conditions of the source images."*
- Underwater rendering is dominated by **depth-dependent Beer-Lambert absorption (per RGB channel)**, **Akkaynak-Treibitz (AK-T) direct + forward-scatter + backscatter**, **caustics**, and **turbidity (Jerlov water types I → 9C)**. None of these are static appearance — they are functions of camera-to-object distance, water column properties, and lighting geometry that *change every step*.
- Baking these into Gaussians means **a single 3DGS asset is only valid at one (depth, turbidity, lighting) point**. Domain randomization across Jerlov types collapses.
- Mitigation in the gs_playground paper is "future work: algorithmic relighting." Not shipped.

**B. RLGK is rigid-only; we need fluid + soft.**  
- Their own future-work note: *"Representing deformable objects (cloth, fluids) or soft-body manipulation remains a challenge. The team plans to integrate particle-based dynamics (like PBD or MPM)."*
- Waves, tethers, manipulator-fluid interaction, plumes — all soft/fluid. RLGK has no answer today.

**C. MotrixSim is yet another physics engine — and a closed-development one.**  
- Custom ground-up engine, velocity-impulse with constraint islands. We have **no benchmark, no third-party audit, no license info, no ROS bridge, no Isaac Lab adapter, no USD interop** published.
- Adopting it means **leaving Newton + USD ecosystem** that the rest of the 2026 robotics stack (NVIDIA, DeepMind, Disney, Linux Foundation) has converged on. That's a bet against the consensus.
- One academic lab is the entire bus factor. Compare to Newton's governance.

**D. "Early preview release" — most of what matters is unreleased.**  
- From the repo README: *"The repository is currently an early public preview. It contains a minimal batch rendering benchmark and two minimal demos. The full simulator, assets, datasets, training code, and evaluation suite will be released in stages."*
- 98.7% notebooks signals the public release is demos, not a library.
- For a project we'd build on, this is unstable foundation in May 2026.

**E. NVIDIA Omniverse NuRec already ships 3DGS in Isaac Sim 6.0.**  
- Isaac Sim 6.0's "Kit 110.0 Upgrade brings support for NVIDIA Omniverse **NuRec 3D Gaussian splatting libraries with Fabric Scene Delegate integration**" (release notes).
- We get GPU-batched 3DGS rendering, USD-native, inside the rest of our chosen stack — without taking on MotrixSim or RLGK dependency.
- The strategic question is therefore: **do we need gs_playground at all if NuRec gives us 3DGS inside Isaac Sim?** Likely no, except for ideas.

**F. Throughput comparison is renderer-vs-renderer, not sim-vs-sim.**  
- The "outperforms Isaac Sim ray-tracing" benchmark holds only when rendering is the bottleneck. Underwater RL is mostly *not* rendering-bound — it's hydro + sensor-physics-bound. Skipping rendering entirely (state-based RL) gets you 10× more than any renderer choice.

**G. Underwater photography has different acquisition constraints than gs_playground's Bridge-v2 source.**  
- Bridge-v2 is mostly indoor kitchen scenes with even lighting.
- Underwater capture requires: artificial lights, color-cast correction, water-column compensation, often low-contrast / low-light. 
- The Real2Sim claim has not been demonstrated for underwater inputs.

### 16.4 What we **can** selectively adopt
1. **The Real2Sim *idea*** — but execute via Omniverse NuRec (Isaac Sim 6 native) rather than gs_playground. NuRec inherits Fabric Scene Delegate and our USD scene graph.
2. **3DGS for above-water portions** of multi-medium scenarios — harbor backdrops for USV training, where lighting is stable. Run on NuRec.
3. **Point-pruning insight**: if we go heavy on 3DGS for surface scenes, applying SpeedySplat/PUP-3DGS pruning is a near-free 10× memory win.
4. **RLGK as a pattern** — even with NuRec, the idea of "rigid frame → Gaussian cluster" with zero-overhead reposing is the right architecture. We replicate the pattern, not the code.
5. **Their published validation methodology** — Go2/G1 training time + zero-shot transfer is a useful template for our BlueROV2 validation claims.

### 16.5 What we **don't** adopt
1. **MotrixSim** — we stay on Newton.
2. **gs_playground codebase as a dependency** — too early, too narrow, too rigid-only.
3. **Single-image Real2Sim for underwater scenes** — pipeline unproven for our domain; needs underwater-specific image-enhancement preprocessing first.
4. **RLGK as the visual-state primitive** — Omniverse Fabric Scene Delegate already does GPU-resident state updates; we don't need a parallel mechanism.
5. **Bake-the-lighting strategy** — fundamentally wrong for underwater.

### 16.6 Verdict in one line
**Architecturally inspiring, materially unusable for an underwater simulator.** Read the paper, copy two ideas (rigid-bound Gaussians, point-pruning), implement them on top of NuRec inside Isaac Sim 6.0, and skip the dependency.

### 16.7 Action items (added to v0.1.x backlog)
- [ ] Spike: render a BlueROV2 thruster against a NuRec-imported test scene; measure FPS vs. RTX path-trace on RTX 5090.
- [ ] Evaluate Omniverse NuRec's 3DGS reposing API — does it match RLGK's zero-overhead claim?
- [ ] Sanity-check: capture 10 underwater clips of a pool wall with BlueROV2 GoPro, attempt Real2Sim with off-the-shelf 3DGS (e.g., gsplat, splatfacto), check whether color-cast correction is needed before splatting works.
- [ ] If 3DGS proves viable for surface scenes, write a USD schema `oceanscale:GaussianBackdrop` that wraps NuRec scene refs.

### 16.8 Sources
- [GS-Playground GitHub](https://github.com/discoverse-dev/gs_playground)
- [GS-Playground project page](https://gsplayground.github.io/)
- [arXiv 2604.25459 — GS-Playground paper](https://arxiv.org/abs/2604.25459)
- [DISCOVERSE GitHub (predecessor)](https://github.com/discoverse-dev/DISCOVERSE)
- [arXiv 2507.21981 — DISCOVERSE paper](https://arxiv.org/html/2507.21981)
- Isaac Sim 6.0 release notes (NuRec support) — see §6 references.

---

## 17. Main Stack Lock & New-Arrival Policy

Added 2026-05-15 per explicit user feedback: "we should have a main tech stack, and be careful for new tech/research/project/paper etc, coz fast update daily."

### 17.1 Main Stack (locked — change requires §17.3 review)

| Layer | Choice | Pin / Version | Why locked |
|---|---|---|---|
| Scene graph | **OpenUSD** | ≥25.11 | Industry consensus; USDA-diffable; multi-engine. |
| Rendering | **Omniverse RTX** (path trace) + **NuRec** (3DGS) | inside Isaac Sim 6.0 | Already in our stack; no new dependency. |
| Physics | **Newton 1.x** (primary) + **PhysX 5** (fallback via Isaac Lab multi-backend) | Newton ≥1.2, pin tag | LF governance; Apache-2.0; Warp-native; multi-solver. |
| Rigid solver | **MuJoCo-Warp** | Newton-bundled | Best-validated rigid + built-in fluid drag. |
| GPU kernels | **NVIDIA Warp** | 1.13.x | Differentiable; bottom of the new stack. |
| Env framework | **Isaac Lab 3.0** | v0.2 target (was in locked list) | ManagerBasedEnv composability; multi-GPU; ONNX. **Moved to watchlist** — v0.1 uses Newton standalone + skrl (per ADR-007). v0.2 integration planned. |
| RL default | **skrl 2.0** | 2.0.x | Spans PyTorch / JAX / Warp. **ACTIVE DEFAULT.** |
| RL throughput | **rl_games 1.6.5** | 1.6.x | Battle-tested PPO + NCCL multi-node. |
| RL sim-to-real | **RSL-RL v5** | v5.x | ETH conventions for zero-shot. **Performance baseline.** |
| World model (optional) | **DreamerV3** | as published | Sample-efficient for sparse rewards. |
| Hydrodynamics | **Custom OceanScale Warp kernels** (Tier 0-3) | this project | Nobody else solves it on Newton+USD. |
| Sensors | **Custom OceanScale Warp kernels** (sonar/AK-T/DVL/IMU/BELLHOP) | this project | Same — fill the gap. |
| Vehicle baseline | **BlueROV2 Heavy** | URDF/USD | Cheap, ubiquitous, MarineGym-compatible. |
| Reference hydro impl | **MarineGym** (wrap, then re-implement in Warp) | arXiv 2503.09203 | Credit; don't reinvent. |
| Reference sensor impl | **OceanSim** (study + selectively reuse) | arXiv 2503.01074 | Same. |
| Runtime | Ubuntu 24.04, CUDA 12.8, Python 3.12, PyTorch 2.7 | pinned in pyproject.toml | One canonical env. |
| License (outbound) | **Apache-2.0** (code), **CC-BY-4.0** (scenes), **MIT** (policies) | this project | Matches Newton/Isaac; industry-friendly. |

**Anything not on this list is "external to the main stack" and lives in §17.2 — Watchlist.**

### 17.2 Watchlist (track, do not depend)

Updated as new things appear. Status values:
- `monitor` — interesting, no action
- `evaluate` — write §17.x style critical evaluation
- `spike` — short prototype to confirm or deny a specific claim
- `adopt` — passes §17.3 — proceed to main stack
- `reject` — explicitly out of scope (re-evaluate annually)

| Project / Paper | First seen | Status (May 2026) | Notes |
|---|---|---|---|
| **GS-Playground** (RSS 2026, gs_playground) | 2026-05-15 | **reject** as dependency, **monitor** for ideas | See §16. 3DGS bakes lighting; MotrixSim lock-in; preview-only. Take 2 patterns, no dep. |
| **NVIDIA NuRec** (3DGS inside Isaac Sim 6) | 2026-03 (Isaac Sim 6 dev) | **spike** | Already in main stack via Isaac Sim 6. Test 3DGS reposing speed on RTX 5090 vs RTX path trace. |
| **Genesis** (universal solvers) | 2024-12 | **monitor** | No marine; SPH solvers worth studying. Watch Issue #682. |
| **JaxLrauv / JaxMARL** (multi-agent JaxRL) | 2025-05 | **monitor** | Useful pattern for MARL benchmarks in v0.4. |
| **EasyUUV** (LLM-DR pipeline) | 2025-10 | **monitor** | DR strategy reference, not a sim. |
| **HoloOcean 2.x** | 2025-10 | **monitor** | Source of validated Fossen coefficients + sonar reference. Compare numerically. |
| **Stonefish 1.6 + ICRA 2025** | 2025-05 | **monitor** | Hydro reference + ROS2 pattern. |
| **DreamerV3 / TD-MPC2** | 2025 | **evaluate** | Decide world-model adoption at v0.3 once sample efficiency matters. |
| **Isaac Lab 3.0** (env framework) | 2025 | **adopt at v0.2** | Moved from locked list. Newton backend still experimental; v0.1 uses Newton standalone + skrl. Full Isaac Lab integration deferred to v0.2. |
| **NeRD (Neural Robot Dynamics)** | 2025-2026 | **monitor** | Learned residual dynamics — possible Tier-1 enhancement. |
| **Hybrid Neural-MPM** (arXiv 2505.18926) | 2025-05 | **monitor** | Real-time CFD candidate for Tier-2. |
| **Q-D acoustic channel** (arXiv 2501.04238) | 2025-01 | **adopt** | Direct ingredient for our BELLHOP+Q-D kernel. |

### 17.3 New-Arrival Evaluation Policy (must pass ALL 5)

When a new sim / paper / repo appears, before adding it to the main stack write a §16-style evaluation and confirm:

1. **Governance / bus-factor ≥ current.** Linux-Foundation, NVIDIA, ETH, or institution-level beats single-lab. Single-lab beats single-author. Single-author preview ⇒ reject as dependency.
2. **Ecosystem fit.**
   - USD-native or has USD bridge.
   - Apache-2.0 / MIT / BSD compatible (no GPL-AGPL contamination of the sim core; OK for tooling).
   - Warp / PyTorch / JAX interop without copy.
   - Linux x86_64 Python 3.12 supported.
3. **Solves a real gap we have *today*.** Not "could be useful in 18 months." Cite the concrete v0.x.y task it unblocks.
4. **Production-readiness.**
   - Stable API (semver or pinned tag).
   - Non-preview release.
   - At least one third-party real-hardware deployment.
   - CI / tests visible.
5. **Net positive cost-benefit.** Migration effort + retraining time + new bus-factor risk + license review < the benefit. Default decision: **no**.

**Failure of any bar ⇒ Watchlist, not Main Stack.**

### 17.4 Update cadence
- Watchlist reviewed every **2 weeks** (Friday).
- Main Stack reviewed every **3 months** or when a key dep hits a major version bump.
- Annually: re-litigate `reject` entries.
- All evaluations live as numbered sections in this file. No verbal lore, no Slack-only.

### 17.5 Anti-patterns (do not)
- Do not silently `pip install` something new and refactor against it. PR-or-equivalent + §17.x evaluation first.
- Do not let a paper deadline dictate a dep change.
- Do not migrate "just to be modern" — modern is what we already locked.
- Do not accept "but their benchmark is faster" without checking the benchmark matches our workload (vision-bound vs sim-bound; latency vs throughput; per-env vs total).
- Do not adopt a project whose own README says "early preview, full release coming soon" as a core dep.

---

## 18. Open Questions (need answers before v0.1)

1. **What real AUV do we calibrate against?** BlueROV2 Heavy is the default. Confirm.
2. **Single or multi-GPU at v0.1?** Single-GPU first is faster to iterate. Confirm.
3. **JAX backend in v0.1, or defer to v0.3?** Defer recommended — focus.
4. **ROS 2 bridge from day 1 or v0.3?** v0.3, after hardware validation.
5. **Visual policy in v0.1 or state-based only?** State-based only — 10× faster iteration.
6. **Public open-source from day 1, or after v0.2?** After v0.2 — release with first reproducible benchmark.
