# OceanScale Roadmap

Items marked *exploratory* are not yet committed and may change or be dropped.

---

## v0.1 -- Near-term

Identity-reset baseline. Ship a credible v0.1 SDK with one vehicle, one task, and honest benchmarks.

### SDK and tooling

- [x] Agent-native repo (per-tool configs)
- [x] `POSITIONING.md` locked as brand law
- [x] `ARCHITECTURE.md` technical proposal
- [x] Tag-driven Cloudflare Pages deployment pipeline
- [x] Source install path verified from the repo
- [ ] PyPI publish workflow verified end-to-end
- [ ] Newton + Warp install path verified on Colab T4
- [x] Isaac Sim 6 / Isaac Lab 3 validation lane verified locally

### BlueROV2 stabilization

- [x] Hydro Tier-1 Fossen 6-DOF Warp kernels with autograd gate tests
- [x] Von Benzon 2022 6-DOF reference port with regression tests
- [x] PPO hover policy converges (explained_variance 0.901 at 1M steps)
- [x] Headless MP4 renderer (single-panel + cinematic 4-panel)
- [x] CLI entry point: `oceanscale demo bluerov2-hover`
- [ ] Lateral drift in hover eval (~0.41m over 33s) -- v0.2 target

### Benchmarks and validation

- [x] OceanScale vs PyBullet apples-to-apples benchmark (10.49x at n=64)
- [x] NVIDIA ecosystem benchmark matrix (28 scripts)
- [ ] Sim-to-real validation with physical BlueROV2 (needs hardware partner)

### Demo and website

- [x] Landing page (Astro 6 + Tailwind v4, bilingual zh/en)
- [x] Demo section with install command, Colab badge, benchmark table
- [ ] Replace placeholder video with real render at release tag

### Shipping criteria for v0.1

1. Source install plus `uv run oceanscale demo bluerov2-hover` works on supported GPUs
2. Benchmark table on website matches reproducible `benchmarks/RESULTS.md`
3. CHANGELOG entry + version tag
4. Public PyPI release remains a separate gate until the publish workflow is verified end-to-end

---

## v0.x -- Medium-term

Expand from one vehicle and one task to a multi-vehicle, multi-environment platform.

### Vehicles

- [ ] BlueROV2 Heavy (exploratory)
- [ ] IVER3-class AUV (exploratory)
- [ ] REMUS-class AUV (exploratory)

### Environments and physics

- [ ] Tier-1 Fossen stabilization (lateral drift fix, thruster wash coupling)
- [ ] Tier-2 SPH fluid model for manipulator/thruster wakes (exploratory, gated on throughput)
- [ ] Free-surface waves (Gerstner/FFT Warp kernel)
- [ ] Time-varying current fields
- [ ] Domain randomization toolkit (mass, inertia, currents, turbidity, sensor noise)

### Sensors

- [ ] Forward-looking sonar (Warp BVH ray cast)
- [ ] Sidescan and multibeam sonar
- [ ] Jaffe-McGlamery underwater vision pass
- [ ] DVL (4-beam Doppler, bottom-lock / water-track)
- [ ] Acoustic modem (BELLHOP wrapper)

### ROS2 integration

- [x] ROS2 bridge module (oceanscale.ros2) — publish vehicle state, sensors, TF, clock
- [x] Twist command subscriber (cmd_vel → 6-DOF action)
- [x] Emergency stop subscriber
- [x] Standalone CLI entry point (scripts/ros2_bridge.py)
- [x] ROS2 launch file utilities
- [x] 54 unit tests (mocked rclpy, no system ROS2 required)
- [x] External ROS2 graph validation — 25/25 real rclpy tests on Ubuntu 24.04 + Jazzy
- [ ] Isaac Sim 6.0 ROS2 bridge integration test

### RL and benchmark tasks

- [ ] Station-keeping under current + wave
- [ ] Pipe / cable following with sonar
- [ ] Docking (visual + sonar, multi-stage)
- [ ] Multi-agent acoustic-localization MARL
- [ ] Bathymetric survey with sidescan
- [ ] ONNX export for every benchmark policy

### Throughput targets (single GPU)

| Task | Target | Status |
|------|--------|--------|
| Station-keep, tier-0 | >=600k FPS | exploratory |
| Pipe-follow, tier-1 + sonar | >=80k FPS | exploratory |
| Multi-AUV MARL, tier-1 + comms | >=25k FPS | exploratory |
| Manipulator FSI, tier-2 | >=5k FPS | exploratory |

---

## vN -- Long-term (exploratory)

Directions that depend on evidence not yet available. Promotion gates from `POSITIONING.md` apply.

### Tier-2: ocean world model

**Promotion gate:** a shipped demo shows the learned layer improving simulator fidelity or transfer, with a public benchmark against a relevant non-learned baseline.

### Tier-3: ocean foundation model

**Promotion gate:** (a) data scale evidence, (b) model trained at meaningful parameter scale, (c) distribution leverage.

Until these gates are met, these remain in research phase.

### Scope expansion

Current scope: underwater robotics (AUVs, ROVs). Expansion to broader marine robotics revisited when:

1. The same simulator primitive clearly serves underwater and surface systems
2. At least two serious conversations pull toward USVs or broader autonomy
3. The category can expand without diluting focus

---

## Cross-reference table

| Domain | Owner | Content |
|--------|-------|---------|
| Vision, category, scope, voice, lexicon | `POSITIONING.md` | Brand law |
| Technical architecture, physics tiers, sensor design | `ARCHITECTURE.md` | Engineering proposal |
| Scheduling, sequencing, shipping criteria | `ROADMAP.md` (this file) | What ships when |
| Version history, per-release changes | `CHANGELOG.md` | What shipped |
| Visual identity, color, typography | `DESIGN.md` | Visual brand law |
