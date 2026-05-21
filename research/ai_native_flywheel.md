# AI-Native Flywheel for Ocean Engineering

**Date:** 2026-05-20
**Status:** Research design document
**Companion:** `DESIGN.md` (simulator architecture), `SURVEY.md` (state-of-the-art)

---

## 0. Executive Summary

The OceanScale flywheel is a closed-loop pipeline that turns the traditional ocean engineering cycle (months-per-iteration) into an AI-accelerated cycle measured in hours. Each stage -- simulation, training, verification, deployment, feedback -- is augmented by AI agents that generate, optimize, and analyze, compressing the loop from idea to tested policy to under one day.

**Target flywheel iteration:** <24 hours from hypothesis to validated policy
**Target zero-shot training time:** <1 hour on single H100/RTX 5090

---

## 1. Flywheel Architecture

```
     ┌──────────────────────────────────────────────────────────────────┐
     │                     OCEANSCALE AI FLYWHEEL                      │
     │                                                                  │
     │   ┌──────────┐    ┌──────────┐    ┌────────────┐    ┌────────┐  │
     │   │ SIMULATE │───>│  TRAIN   │───>│  VERIFY    │───>│ DEPLOY │  │
     │   │ (GPU-    │    │ (RL + WM │    │ (sim2real  │    │ (AUV   │  │
     │   │  parallel│    │  + diff  │    │  gap +     │    │  test  │  │
     │   │  envs)   │    │  physics)│    │  safety)   │    │  data) │  │
     │   └────┬─────┘    └──────────┘    └────────────┘    └───┬────┘  │
     │        ^                                                   │      │
     │        │              ┌──────────────┐                     │      │
     │        └──────────────│   FEEDBACK   │<────────────────────┘      │
     │                       │ (real data → │                            │
     │                       │  sim refine) │                            │
     │                       └──────┬───────┘                            │
     │                              ^                                    │
     │                              │                                    │
     │          ┌───────────────────┴───────────────────────┐            │
     │          │         AI ACCELERATION LAYER              │            │
     │          │  (auto-scenarios, code gen, analysis, docs)│            │
     │          └───────────────────────────────────────────┘            │
     └──────────────────────────────────────────────────────────────────┘
```

### 1.1 Stage 1: Simulate (GPU-Parallel Underwater Environments)

**What:** Run thousands of parallel underwater environments on a single GPU, each with different vehicle configs, ocean conditions, and task parameters.

**Stack:** Newton 1.x + Warp custom hydro kernels + Isaac Lab 3.0 env wrappers

**Process:**
1. AI agent generates scenario configs (vehicle type, task, environment parameters)
2. Domain randomization module applies configurable perturbations
3. GPU-parallel Newton physics engine runs all envs simultaneously
4. Warp kernels compute hydrodynamics (Fossen 6-DOF), sonar, vision in parallel
5. Observations streamed to training backend via zero-copy Warp-to-Torch bridge

**Throughput targets (from DESIGN.md Section 8):**

| Task | Single H100 | Single RTX 5090 |
|---|---|---|
| Station-keeping (tier-0) | >=1M FPS | >=600k FPS |
| Pipe-following (tier-1 + sonar) | >=150k FPS | >=80k FPS |
| Multi-AUV MARL (tier-1 + comms) | >=50k FPS | >=25k FPS |
| Manipulator FSI (tier-2) | >=10k FPS | >=5k FPS |

**Parallel environment counts by GPU (estimated):**

| GPU | VRAM | Simple envs (tier-0) | Complex envs (tier-1+sonar) | Heavy envs (tier-2+FSI) |
|---|---|---|---|---|
| RTX 5090 | 32 GB GDDR7 | 8,192-16,384 | 2,048-4,096 | 512-1,024 |
| H100 | 80 GB HBM3 | 16,384-32,768 | 4,096-8,192 | 1,024-2,048 |
| B200 | 192 GB HBM3e | 32,768-65,536 | 8,192-16,384 | 2,048-4,096 |

These estimates are extrapolated from Isaac Lab scaling data (source: [Isaac Lab arXiv paper](https://arxiv.org/html/2511.04831v1), [GitHub IsaacLab issue #2761](https://github.com/isaac-sim/IsaacLab/issues/2761)). Actual counts depend on sensor payload and physics tier.

### 1.2 Stage 2: Train (RL + World Models + Differentiable Physics)

**What:** Train policies using multiple RL backends, optionally augmented with learned world models for sample efficiency and differentiable physics for gradient-based optimization.

**Stack:** skrl 2.0 (default), rl_games (throughput), RSL-RL (sim2real PPO), DreamerV3/TD-MPC2 (world models)

**Training pipeline:**
1. Select RL algorithm based on task characteristics
2. Run massively parallel rollouts in GPU-parallel envs
3. Asymmetric actor-critic: privileged info (ground-truth currents, obstacles) for critic; sensor observations for actor
4. Domain randomization during training for robustness
5. ONNX export upon convergence

**Algorithm selection guide:**

| Task type | Primary algorithm | Fallback | Rationale |
|---|---|---|---|
| Continuous control (single AUV) | PPO (RSL-RL) | SAC | Well-understood, stable, fast |
| Multi-agent coordination | MAPPO (skrl) | QMIX | Scalable to heterogeneous fleets |
| Sparse-reward exploration | DreamerV3 | TD-MPC2 | World model enables planning |
| System identification | Differentiable physics (Warp grad) | PPO with shaped reward | Gradient-based, data-efficient |
| Manipulation contact-rich | PPO + RND | DreamerV3 | Curiosity-driven for contact |

**Training time targets:**

| Task | Env count (RTX 5090) | Steps to convergence | Wall time (target) |
|---|---|---|---|
| Station-keeping | 4,096 | ~50M steps | <15 min |
| Pipe-following | 2,048 | ~200M steps | <1 hour |
| Multi-AUV MARL | 1,024 | ~500M steps | <2 hours |
| Manipulator | 512 | ~300M steps | <3 hours |

Assumptions: 80k FPS throughput, PPO batch size 32768, standard hyperparameters.

### 1.3 Stage 3: Verify (Automated Sim2Real Gap Measurement)

**What:** Automatically quantify the gap between simulated and real-world performance, test policy robustness, and flag failure modes before real-world deployment.

**Stack:** Custom verification suite + automated DR calibration + W&B/MLflow tracking

**Verification pipeline:**
1. **Physics fidelity gates** -- compare simulation outputs against known analytical solutions (Fossen equations, acoustic ray tracing)
2. **Sim2real gap estimation** -- run policy in progressively degraded simulations:
   - Add sensor noise beyond training DR range
   - Inject unmodeled dynamics (thruster saturation, tether snag)
   - Vary environmental parameters outside DR bounds
3. **Automated stress testing** -- adversarial scenario generation:
   - Worst-case current profiles
   - Sensor failure injection
   - Multi-vehicle collision scenarios
4. **Performance regression detection** -- compare new policy against baseline on standardized benchmark suite
5. **Safety validation** -- check collision rates, depth limits, communication timeouts

**Gap metrics (from arXiv survey on reality gap, [arXiv 2510.20808](https://arxiv.org/html/2510.20808v1)):**

| Metric | Definition | Target |
|---|---|---|
| Return ratio | return_real / return_sim | >=0.7 for zero-shot |
| State divergence | KL(sim_states, real_states) | <threshold per task |
| Failure rate delta | fail_real - fail_sim | <5 percentage points |
| Control smoothness | jerk comparison | qualitatively similar |

### 1.4 Stage 4: Deploy (AUV Testing + Data Collection)

**What:** Deploy verified policies to real AUV hardware for in-water testing and data collection.

**Stack:** ONNX runtime on embedded GPU (Jetson Orin) + ROS 2 bridge + telemetry logging

**Deployment pipeline:**
1. Export policy to ONNX (already required by DESIGN.md Section 8)
2. Optimize with TensorRT for Jetson Orin NX inference (<5ms latency target)
3. Deploy to vehicle via ROS 2 action server
4. Run structured test missions (progressive difficulty)
5. Collect synchronized telemetry: sensor data, control commands, ground-truth pose (via external tracking)
6. Stream logs to cloud storage for feedback stage

**Hardware targets:**
- BlueROV2 Heavy (primary validation platform)
- Jetson Orin NX (16GB) for onboard inference
- External tracking: Qualisys / UWB for ground truth

### 1.5 Stage 5: Feedback (Real-World Data to Simulation Refinement)

**What:** Close the loop by feeding real-world data back into simulation to reduce the sim2real gap.

**Stack:** DDR (Data-Informed Domain Randomization) hooks from DESIGN.md Section 7.3 + system ID via differentiable physics

**Feedback pipeline:**
1. **Real2Sim alignment:**
   - Fit simulation parameters to match real trajectories (mass, drag coefficients, thruster gains)
   - Use differentiable Warp kernels to compute gradients of trajectory loss w.r.t. physics parameters
   - Update USD scene properties with calibrated values
2. **Residual learning:**
   - Train a residual dynamics model on (sim_prediction, real_observation) pairs
   - Inject residual corrections during next training cycle
3. **DR calibration:**
   - Adjust DR distribution bounds to match real-world variability observed in data
   - Tighten distributions where sim matches reality (faster training); widen where it does not (better robustness)
4. **Scenario discovery:**
   - AI analyzes real-world failures to identify unmodeled phenomena
   - Generates new simulation scenarios to cover discovered edge cases
5. **Continuous model update:**
   - Water conditions database (turbidity, salinity, temperature profiles from real dives)
   - Vehicle health model (battery degradation, biofouling effects on drag)

### 1.6 Loop Speed: How Fast Can We Spin?

**Bottleneck analysis and time targets:**

```
Stage              Traditional    AI-Accelerated    Speedup
─────────────────  ────────────   ──────────────    ────────
Scenario design    1-2 weeks      10 minutes        500-1000x
Simulation setup   2-3 days       30 minutes        100-150x
Policy training    1-2 weeks      30-90 minutes     100-500x
Verification       3-5 days       1-2 hours         40-60x
Real-world test    1-2 days       1-2 days          1x (hard bound)
Feedback analysis  1-2 weeks      2-4 hours         30-80x
─────────────────  ────────────   ──────────────
TOTAL              4-8 weeks      2-4 days          10-20x
```

The real-world test is the hard bound. The target of <1 day applies to fully in-simulation iterations (hypothesis to verified policy). Full loop including water testing targets 2-4 days.

---

## 2. AI Acceleration at Each Stage

### 2.1 AI for Simulation Design

**Auto-generate scenarios:** LLM agent takes natural-language task description and produces:
- USD scene graph (vehicle placement, obstacle layout, terrain)
- Environment config (wave parameters, current profiles, water properties)
- DR config (which parameters to randomize, distribution bounds)

```
User: "Test pipe-following in strong cross-current with degraded sonar"
  │
  ▼
AI Agent generates:
  ├── USD scene: pipe_50m.usd + cross_current_0.5ms.yaml
  ├── DR config: sonar_snr [5-15 dB], current_speed [0.3-0.8 m/s]
  ├── Task config: pipe_following_v2.yaml
  └── Verification spec: max_cross_track_error=0.5m
```

**Tools:**
- NVIDIA USD Code NIM microservice for USD Python code generation ([source](https://www.hpcwire.com/bigdatawire/this-just-in/nvidia-announces-generative-ai-models-and-nim-microservices-for-openusd-language-geometry-physics-and-materials/))
- LLM fine-tuned on OceanScale scenario YAML schema
- Procedural generation library for underwater terrain (Perlin noise bathymetry, random obstacle placement)

**Adaptive domain randomization (ADR):**
- Start with wide DR distributions
- After each training iteration, analyze which DR parameters correlate with policy failure
- Narrow distributions for well-handled parameters; widen for failure-correlated ones
- Method: Active Domain Randomization ([source](https://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf)) + flow-based DR ([ICML 2025](https://icml.cc/virtual/2025/poster/46239))

### 2.2 AI for Code Generation

**Warp kernel generation:**
- LLM generates Warp Python kernel from natural-language specification of physics
- Kernel compiled to CUDA via Warp JIT -- no hand-written CUDA needed
- Verification: auto-generated unit tests compare kernel output against analytical solution

```
Spec: "Fossen 6-DOF hydrodynamic forces for ellipsoidal body with
       added mass matrix, linear+quadratic drag, Coriolis, restoring"
  │
  ▼
AI generates: fossen_6dof_kernel.wp (Warp kernel)
  │
  ▼
Auto-verified against: analytical Fossen test cases (sphere, cylinder)
```

**USD scene creation:**
- Text-to-USD pipeline: describe vehicle/environment, get OpenUSD scene
- NVIDIA USD Code NIM already supports this pattern
- SimReady assets provide pre-validated physics properties

**RL environment wrappers:**
- AI generates Isaac Lab 3.0 `ManagerBasedRLEnv` configuration from task description
- Auto-generates reward functions, observation spaces, action spaces
- Template-based with LLM filling in task-specific logic

### 2.3 AI for Experiment Design

**What to test next (adaptive experimentation):**
- Bayesian optimization over hyperparameter space
- Multi-fidelity: run cheap low-env-count experiments first, graduate promising configs to full-scale
- Automatic curriculum generation: start easy (calm water, simple task), progressively add difficulty

**Experiment selection criteria:**
1. **Maximize information gain** -- which experiment reduces uncertainty about sim2real gap the most?
2. **Failure-driven** -- prioritize experiments that probe known failure modes
3. **Coverage** -- ensure DR space is uniformly explored
4. **Efficiency** -- prefer shorter experiments when discriminating between similar configs

**Implementation:**
- W&B Sweeps or Optuna for hyperparameter optimization
- Custom Bayesian optimizer over (scenario config, DR config, training config) space
- Integration with Isaac Lab's built-in hyperparameter sweep tools

### 2.4 AI for Analysis

**Automated failure analysis:**
- After each training run, AI agent:
  1. Segments trajectories into phases (approach, execute, recover)
  2. Identifies failure modes (collision, timeout, oscillation, drift)
  3. Clusters failures by root cause
  4. Suggests DR parameter adjustments to address each failure mode

**Performance regression detection:**
- Every policy checkpoint is evaluated on a fixed benchmark suite
- Statistical comparison against baseline (t-test on returns, bootstrap CIs)
- Flag regressions exceeding threshold (configurable per metric)
- Track: reward, success rate, collision rate, control smoothness, energy consumption

**Automated reporting:**
- Generate markdown report per training run: config, metrics, failure analysis, recommendations
- Dashboard: W&B or MLflow with custom OceanScale panels
- Trend analysis across flywheel iterations: is the sim2real gap shrinking?

### 2.5 AI for Documentation

**Auto-update specs:**
- When simulation parameters change (e.g., calibrated drag coefficient), AI updates:
  - USD scene documentation
  - DESIGN.md parameter tables
  - Benchmark task specifications

**Generate reports:**
- Training run reports (per-iteration)
- Verification reports (pre-deployment)
- Sim2real gap reports (post-deployment)
- All in structured markdown, committed to repo

**Policy cards:**
- Every exported ONNX policy gets a "policy card" documenting:
  - Training config, DR ranges, environment version
  - Benchmark scores (sim and real)
  - Known failure modes and operational limits
  - Recommended deployment conditions

---

## 3. Infrastructure Requirements

### 3.1 Compute

| Tier | Hardware | Use case | Estimated cost |
|---|---|---|---|
| Development | 1x RTX 5090 (32 GB) | Single-researcher training, prototyping | ~$2,000 |
| Training cluster | 4-8x H100 (80 GB) | Multi-node PPO, large-scale DR sweeps | Cloud: ~$20-40/hr |
| Production training | 4-8x B200 (192 GB) | Max parallelism, longest tasks | Cloud: ~$40-80/hr |
| Edge inference | Jetson Orin NX (16 GB) | Onboard AUV policy execution | ~$500 |
| Storage | NVMe SSD + S3 | Checkpoints, logs, telemetry | ~$100/mo |

**Cloud vs local strategy:**
- Local (RTX 5090): development, small experiments, verification
- Cloud (H100/B200): large-scale training sweeps, multi-node runs
- Hybrid: develop locally, burst to cloud for full training runs

**Recommended cloud providers:** Lambda Labs, RunPod, CoreWeave (GPU-first, cheaper than AWS/GCP for sustained training)

### 3.2 Data Pipeline

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  SIMULATION LOGS │     │  TRAINING LOGS   │     │  REAL-WORLD DATA │
│                  │     │                  │     │                  │
│ • Env states     │     │ • Rewards        │     │ • Sensor streams │
│ • Sensor outputs │     │ • Loss curves    │     │ • Control cmds   │
│ • Physics params │     │ • Hyperparams    │     │ • Ground truth   │
│ • DR configs     │     │ • Checkpoints    │     │ • Video          │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                         │
         ▼                        ▼                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     DATA LAKE (S3 / MinIO)                          │
│                                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │ Trajectory │  │  Policy    │  │   Env      │  │  Telemetry │     │
│  │  Store     │  │  Registry  │  │  Catalog   │  │  Archive   │     │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  EXPERIMENT TRACKING (W&B / MLflow)                  │
│  • Run comparison dashboards                                         │
│  • Artifact versioning (models, configs)                             │
│  • Metric alerting (regression detection)                            │
│  • Team collaboration                                                │
└──────────────────────────────────────────────────────────────────────┘
```

**Data formats:**
- Simulation logs: HDF5 (trajectory data) + JSON (configs)
- Training logs: W&B native format + JSONL (metrics stream)
- Real-world data: ROS 2 bag files (.mcap) + HDF5 (post-processed)
- Policies: ONNX (.onnx) + policy card JSON

**Data volume estimates:**
- Single training run (pipe-following, 2k envs, 200M steps): ~50-200 GB raw trajectories
- Compressed (subsampled): ~5-20 GB per run
- Real-world dive: ~10-50 GB per dive (video + sensors)
- Target: 1 TB storage per month of active flywheel operation

### 3.3 CI/CD for Simulation

**Physics accuracy tests (run on every PR):**
1. **Hydrodynamic validation:** Fossen kernel output matches analytical solution to within 1% for standard test bodies
2. **Sensor validation:** sonar ray-casting matches geometric ground truth; vision pipeline matches known underwater image formation model
3. **Throughput regression:** FPS does not drop below 90% of baseline for each benchmark task
4. **Training smoke test:** PPO converges on station-keeping within 5M steps (fast sanity check)

```yaml
# .github/workflows/sim-ci.yml (conceptual)
name: Simulation CI
on: [push, pull_request]
jobs:
  physics-validation:
    runs-on: [self-hosted, gpu]
    steps:
      - run: pytest tests/physics/ --gpu --tolerance 0.01
  sensor-validation:
    runs-on: [self-hosted, gpu]
    steps:
      - run: pytest tests/sensors/ --gpu
  training-smoke:
    runs-on: [self-hosted, gpu]
    timeout-minutes: 30
    steps:
      - run: python -m oceanscale.train --task station_keep --max-steps 5M --assert-converge
  throughput-benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - run: python -m oceanscale.bench --tasks all --assert-fps-baseline 0.9
```

### 3.4 Versioning

Everything is versioned. Every flywheel iteration is reproducible.

| Artifact | Versioning scheme | Storage |
|---|---|---|
| Simulation code | Git (semver tags) | GitHub |
| USD scenes | Git LFS | GitHub |
| Newton/Warp versions | Pip lockfile (uv.lock) | Repo |
| Training configs | Git (YAML files) | GitHub |
| Policy checkpoints | W&B Artifacts / MLflow | Cloud |
| ONNX policies | Git LFS + policy card JSON | GitHub |
| DR configs | Git (YAML files) | GitHub |
| Real-world data | S3 with date-stamped keys | Cloud |
| Experiment metadata | W&B / MLflow runs | Cloud |

**Reproducibility contract:** Given a policy card (training config hash + environment version + DR config), the entire training run can be reproduced exactly.

---

## 4. Scale Targets

### 4.1 Parallel Environments per GPU

Estimates based on Isaac Lab scaling patterns ([source](https://arxiv.org/html/2511.04831v1)) plus OceanScale-specific sensor and hydro overhead:

| GPU | CUDA Cores | VRAM | Tier-0 (hydro only) | Tier-1 (+sonar) | Tier-2 (+SPH FSI) |
|---|---|---|---|---|---|
| RTX 4090 | 16,384 | 24 GB | 4,096-8,192 | 1,024-2,048 | 256-512 |
| RTX 5090 | 21,760 | 32 GB GDDR7 | 8,192-16,384 | 2,048-4,096 | 512-1,024 |
| H100 SXM | 16,896 | 80 GB HBM3 | 16,384-32,768 | 4,096-8,192 | 1,024-2,048 |
| B200 | ~18,000+ | 192 GB HBM3e | 32,768-65,536 | 8,192-16,384 | 2,048-4,096 |

VRAM is the primary bottleneck, not compute. Sonar ray-tracing and vision passes are VRAM-heavy. Headless mode (no rendering) roughly doubles environment count.

### 4.2 Training Time Targets

**Zero-shot policy training (<1 hour target):**

| Task | Env count (RTX 5090) | Steps needed | FPS | Wall time |
|---|---|---|---|---|
| Station-keeping | 4,096 | 50M | 600k | ~1.4 min |
| Pipe-following | 2,048 | 200M | 80k | ~42 min |
| Docking (multi-stage) | 2,048 | 300M | 80k | ~63 min |
| Multi-AUV MARL | 1,024 | 500M | 25k | ~333 min |
| Manipulator (FSI) | 512 | 300M | 5k | ~600 min |

Single-GPU zero-shot in <1 hour is achievable for tasks up to pipe-following complexity. MARL and FSI tasks require multi-GPU or relaxed time targets.

**Multi-GPU scaling:**
- 4x H100 (NCCL): ~3.5x throughput over single H100 (communication overhead ~15%)
- 8x H100: ~6x throughput
- MARL on 8x H100: 500M steps in ~55 minutes (within 1-hour target)

### 4.3 Flywheel Iteration Time (<1 day for sim-only; <4 days full loop)

**Sim-only iteration (idea to verified policy):**

```
00:00  Hypothesis formulated (natural language)
00:05  AI generates scenario config + DR config + training config
00:10  Training starts (GPU-parallel envs)
01:00  Training converges (pipe-following benchmark)
01:05  Verification suite starts
02:00  Verification complete, gap metrics computed
02:10  AI generates analysis report + recommendations
02:30  Policy exported to ONNX, policy card generated
───────────────────────────────────
Total: ~2.5 hours (target: <4 hours for standard tasks)
```

**Full loop iteration (including real-world test):**

```
Day 1, 00:00-02:30  Sim-only iteration (above)
Day 1, 03:00-05:00  ONNX deployment prep, ROS 2 integration test
Day 1-2              Real-world test dive(s)
Day 2, evening       Telemetry uploaded, feedback analysis starts
Day 3, morning       AI completes feedback analysis
Day 3, afternoon     Updated configs ready for next iteration
───────────────────────────────────
Total: ~3 days (target: <4 days)
```

---

## 5. Comparison with Existing Approaches

### 5.1 NVIDIA Isaac Lab 3.0 Workflow

**What Isaac Lab provides:**
- GPU-parallel simulation framework built on Omniverse
- Manager-based environment configuration
- Built-in RL backends (skrl, rl_games, RSL-RL)
- Domain randomization toolkit
- USD-native scene management

**What Isaac Lab does NOT provide (that OceanScale adds):**
- Underwater-specific physics (hydrodynamics, sonar, acoustic comms)
- Closed-loop flywheel automation (Isaac Lab is a framework, not a pipeline)
- AI-driven scenario generation and experiment design
- Automated sim2real gap measurement
- Real-world data feedback integration

**Relationship:** OceanScale is built ON TOP of Isaac Lab (as specified in DESIGN.md). Isaac Lab is the simulation substrate; OceanScale adds the underwater domain and the AI flywheel orchestration.

Sources: [Isaac Lab paper](https://arxiv.org/html/2511.04831v1), [NVIDIA Newton blog](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)

### 5.2 Google DeepMind Robotics Stack

**Stack components:**
- **RT-2 / Gemini Robotics:** Vision-language-action models for general-purpose manipulation ([source](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/))
- **Gemini Robotics-ER:** Embodied reasoning for novel tasks ([source](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/))
- **MuJoCo MJX / MuJoCo Warp:** GPU-accelerated physics (co-developed with NVIDIA for Newton)
- **RT-X / Open X-Embodiment:** Large-scale cross-embodiment dataset

**Key differences from OceanScale:**

| Dimension | DeepMind | OceanScale |
|---|---|---|
| Domain | General manipulation, indoor | Underwater, marine robotics |
| Physics | MuJoCo (rigid body) | Newton + Warp (fluid dynamics) |
| Training paradigm | Foundation model (VLA) + fine-tuning | Task-specific RL + DR |
| Sim2real approach | Large-scale data diversity | Domain randomization + differentiable calibration |
| Scale | Google-scale compute | Single-lab to small cluster |

**What to learn from DeepMind:**
- The data flywheel pattern: real deployments -> data -> model improvement -> better deployments
- Foundation model approach for generalization across tasks
- Open X-Embodiment as a model for open underwater robotics datasets

### 5.3 Automated ML for Robotics (AutoML, NAS for RL)

**State of the art (as of 2025):**

| Technique | Application to robotics RL | Maturity |
|---|---|---|
| AutoRL (hyperparameter optimization) | PPO hyperparams, reward shaping | Production-ready (Optuna, W&B Sweeps) |
| NAS for policy networks | Lightweight architectures for edge deployment | Research stage |
| Curriculum learning automation | Progressive difficulty selection | Available (goal-conditioned RL) |
| Auto-domain-randomization | ADR (OpenAI), flow-based DR (ICML 2025) | Maturing |
| Meta-RL for fast adaptation | Quick fine-tuning from sim to specific real robot | Research stage |

**Sources:** [AutoRL overview (arXiv)](https://arxiv.org/html/2201.05000v2), [Flow-based DR (ICML 2025)](https://icml.cc/virtual/2025/poster/46239), [ADR (GitHub)](https://github.com/bay3s/auto-dr)

**OceanScale's approach:**
- Use AutoRL for hyperparameter optimization (Optuna/W&B integration)
- Use ADR for domain randomization bounds (not manual tuning)
- Defer NAS to post-v1.0 (edge deployment optimization is a later concern)
- Curriculum learning via AI-designed progressive scenarios

---

## 6. Specific Tooling

### 6.1 Simulation and Physics

| Tool | Role | Version | Notes |
|---|---|---|---|
| Newton 1.x | Physics engine core | Latest GA | MuJoCo-Warp primary solver ([GitHub](https://github.com/newton-physics/newton)) |
| NVIDIA Warp | Custom kernels (hydro, sensors) | 1.x | JIT Python to CUDA ([GitHub](https://github.com/nvidia/warp)) |
| Isaac Lab 3.0 | RL environment framework | Latest GA | Manager-based env config |
| Isaac Sim 6.0 | Rendering, USD management | Latest | Omniverse RTX renderer |
| OpenUSD | Scene description standard | 24.x+ | Single source of truth for all scenes |

### 6.2 Training

| Tool | Role | Version | Notes |
|---|---|---|---|
| skrl 2.0 | Default RL backend | Latest | PyTorch + JAX + Warp support |
| rl_games 1.6.5 | High-throughput PPO | Latest | Multi-node via NCCL |
| RSL-RL v5 | Sim2real PPO + symmetry | Latest | Proven on ANYmal locomotion |
| DreamerV3 | World model RL | Reference impl | Sample-efficient sparse rewards |
| TD-MPC2 | Alternative world model | Latest | Model-predictive control |

### 6.3 Experiment Tracking and MLOps

| Tool | Role | Notes |
|---|---|---|
| Weights & Biases | Primary experiment tracker | Rich RL visualization, team collaboration, sweep management |
| MLflow | Alternative / backup tracker | Open-source, self-hosted option |
| Optuna | Hyperparameter optimization | Bayesian optimization over training config space |
| Git + Git LFS | Code and artifact versioning | USD scenes, ONNX policies |
| S3 / MinIO | Data lake storage | Trajectories, telemetry, checkpoints |

### 6.4 Deployment

| Tool | Role | Notes |
|---|---|---|
| ONNX | Policy export format | Required for all benchmark policies |
| TensorRT | Inference optimization | ONNX to TensorRT for Jetson deployment |
| ROS 2 | Onboard communication | Action server pattern for policy execution |
| Jetson Orin NX | Edge compute | 16 GB, sufficient for ONNX policy + sensor processing |

---

## 7. Implementation Roadmap

### Phase 1: Foundation (v0.1-v0.2, weeks 1-12)

- [ ] Core simulation pipeline (Newton + Isaac Lab env)
- [ ] Single-task RL training (station-keeping)
- [ ] Basic DR toolkit
- [ ] ONNX export
- [ ] W&B integration for experiment tracking
- [ ] Manual flywheel (human drives each stage)

### Phase 2: Automation (v0.3-v0.4, weeks 12-28)

- [ ] AI scenario generation (LLM to YAML/USD pipeline)
- [ ] Automated DR calibration (ADR)
- [ ] Verification suite (physics gates, stress tests)
- [ ] Real-world data feedback pipeline (BlueROV2)
- [ ] Automated failure analysis
- [ ] CI/CD pipeline for simulation accuracy

### Phase 3: Full Flywheel (v0.5-v1.0, weeks 28-40)

- [ ] End-to-end automated flywheel loop
- [ ] Multi-task policy training
- [ ] Differentiable physics for system ID
- [ ] World model integration (DreamerV3)
- [ ] Full sim2real benchmark suite
- [ ] Policy cards and automated documentation
- [ ] Public release + paper

---

## 8. Key References

1. Newton physics engine: [GitHub](https://github.com/newton-physics/newton), [NVIDIA blog](https://developer.nvidia.com/blog/announcing-newton-an-open-source-physics-engine-for-robotics-simulation/)
2. Isaac Lab: [arXiv paper](https://arxiv.org/html/2511.04831v1), [migration guide](https://isaac-sim.github.io/IsaacLab/main/source/migration/comparing_simulation_isaacgym.html)
3. NVIDIA Warp: [GitHub](https://github.com/nvidia/warp)
4. Google DeepMind RT-2: [blog](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/), [project page](https://robotics-transformer2.github.io/)
5. Gemini Robotics: [blog](https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/)
6. AutoRL overview: [arXiv](https://arxiv.org/html/2201.05000v2)
7. Active Domain Randomization: [PMLR](https://proceedings.mlr.press/v100/mehta20a/mehta20a.pdf)
8. Flow-based DR (ICML 2025): [poster](https://icml.cc/virtual/2025/poster/46239)
9. ADR implementation: [GitHub](https://github.com/bay3s/auto-dr)
10. Reality gap survey: [arXiv](https://arxiv.org/html/2510.20808v1)
11. ContactGaussian-WM (physics-grounded world model): [arXiv](https://arxiv.org/html/2602.11021v1)
12. PIN-WM (physics-informed world model): [RSS 2025](https://roboticsconference.org/program/papers/153/)
13. NVIDIA USD Code NIM (AI-generated USD): [HPC Wire](https://www.hpcwire.com/bigdatawire/this-just-in/nvidia-announces-generative-ai-models-and-nim-microservices-for-openusd-language-geometry-physics-and-materials/)
14. ASID (autonomous sim refinement): [TWIML podcast](https://twimlai.com/podcast/twimlai/bridging-the-sim2real-gap-in-robotics)
15. GPU comparison for Isaac Lab RL: [GitHub issue #2761](https://github.com/isaac-sim/IsaacLab/issues/2761)
