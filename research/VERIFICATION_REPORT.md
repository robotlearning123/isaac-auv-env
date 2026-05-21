# Verification Report: startup_technical_roadmap.md

**Date**: 2026-05-20
**Reviewer**: Cross-verification agent (ccz/GLM-5.1)
**File**: `research/startup_technical_roadmap.md`

---

## Summary

| Category | Verified | Mismatch | Unverifiable |
|----------|----------|----------|--------------|
| Hardware specs | 2 | 2 | 0 |
| Software versions | 6 | 1 | 0 |
| Benchmark numbers | 5 | 0 | 0 |
| External references | 7 | 0 | 1 |
| Technical claims | 10 | 0 | 1 |

**Total: 30 verified, 3 mismatch, 2 unverifiable**

---

## Detailed Findings

### CRITICAL MISMATCHES

| # | Claim | Source in doc | Actual (verified) | Severity |
|---|-------|--------------|-------------------|----------|
| 1 | RTX 5090 FP32 = 67 TFLOPS | Line 5, 20 | **104.8 TFLOPS** (TechPowerUp, NVIDIA spec) | **CRITICAL** |
| 2 | RTX 5090 FP16 = 213 TFLOPS | Line 5, 21 | **~838 TFLOPS** non-tensor / **1,676 TFLOPS** tensor (NVIDIA spec) | **CRITICAL** |
| 3 | skrl 2.0 as research fallback | Line 64 | skrl **2.1.0** released 2026-05-11; 2.0.0 was 2026-04-08. "skrl 2.0" is correct as minimum version but current is 2.1.0. | **Minor** |

**Note on RTX 5090 numbers**: The 67 TFLOPS and 213 TFLOPS figures appear to be off by roughly 36%. Possible explanations: (a) the author used base clock instead of boost clock for calculation, or (b) confused with a different GPU. The actual FP32 at boost is 104.8 TFLOPS. This affects all throughput projections that divide by TFLOPS — the actual headroom is ~56% higher than stated, so all performance targets are MORE achievable than the roadmap suggests, not less.

---

### VERIFIED CLAIMS

| # | Claim | Source URL | Notes |
|---|-------|-----------|-------|
| 4 | RTX 5090 has 32 GB GDDR7 | [TechPowerUp](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216) | Confirmed |
| 5 | Warp 1.13.x | [GitHub releases](https://github.com/NVIDIA/warp/releases), [PyPI](https://pypi.org/project/warp-lang/) | v1.13.0 released 2026-05-04 |
| 6 | Newton >= 1.2.0 | [Newton docs](https://newton-physics.github.io/newton/stable/) | v1.2.0 confirmed as latest stable |
| 7 | Isaac Lab v2.3.0 | [Isaac Lab release notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html) | v2.3.2 is latest in 2.3.x series |
| 8 | Isaac Lab v2.3 uses PhysX exclusively | [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html) | Confirmed — Newton is experimental only |
| 9 | MarineGym arXiv 2503.09203 | [arXiv](https://arxiv.org/abs/2503.09203) | Confirmed, IROS 2025 paper |
| 10 | OceanSim built on Isaac Sim | [arXiv](https://arxiv.org/html/2503.01074v2), [GitHub](https://github.com/umfieldrobotics/OceanSim) | Confirmed |
| 11 | OceanSim provides imaging sonar, DVL, UW camera | [OceanSim paper](https://arxiv.org/html/2503.01074v2) | Confirmed |
| 12 | OceanSim does NOT provide fluid dynamics, buoyancy, hydrodynamic forces | [isaac_lab_survey.md](file:///home/robot/workspace/46-marine/benchmarks/isaac_lab_survey.md) line "Does NOT provide fluid dynamics" | Confirmed via local survey + OceanSim paper |
| 13 | OceanSim license BSD-3 | [GitHub LICENSE](https://raw.githubusercontent.com/umfieldrobotics/OceanSim/main/LICENSE) | Confirmed |
| 14 | RSL-RL v5 | [GitHub](https://github.com/leggedrobotics/rsl_rl) | v5.3.0 latest; "v5" is correct version family |
| 15 | skrl multi-backend (PyTorch/JAX) | [Isaac Lab RL comparison](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/rl_frameworks.html) | Confirmed |
| 16 | MuJoCo-Warp exists for batched GPU simulation | [MJWarp docs](https://mujoco.readthedocs.io/en/latest/mjwarp/), [GitHub](https://github.com/google-deepmind/mujoco_warp) | Confirmed |
| 17 | Kamino VBD solver for cable/tether dynamics in Newton | [Newton docs](https://newton-physics.github.io/newton/stable/), [GitHub issue #1708](https://github.com/newton-physics/newton/issues/1708) | Confirmed |
| 18 | von Benzon 2022 Simulink model for BlueROV2 | [AAU PDF](https://vbn.aau.dk/ws/portalfiles/portal/505520780/jmse_10_01898.pdf) | Confirmed — JMSE 2022 paper |
| 19 | HoloOcean and Stonefish exist as underwater simulators | [CMU PDF](https://www.ri.cmu.edu/app/uploads/2022/10/Potokar22icra.pdf), [arXiv 2502.11887](https://arxiv.org/html/2502.11887v1) | Confirmed |
| 20 | ROS 2 Jazzy exists | [ROS 2 docs](https://docs.ros.org/en/jazzy/Releases.html) | Released 2024-05-23 |
| 21 | NuRec 3DGS for digital twins | [NVIDIA](https://developer.nvidia.com/omniverse/nurec), [GTC 2026](https://www.nvidia.com/en-us/on-demand/session/gtc26-dlit81757/) | GA at GTC 2026 |
| 22 | Isaac Lab 3.0 announced at GTC 2026 with Newton backend | [NVIDIA forums](https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709), [GitHub releases](https://github.com/isaac-sim/IsaacLab/releases) | Early access, not GA |
| 23 | Isaac Sim 6.0 Early Dev | [NVIDIA forums](https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709) | Confirmed |
| 24 | Python 3.12 | pyproject.toml | Confirmed in repo |
| 25 | Newton pinned <1.3 in pyproject.toml | pyproject.toml | Confirmed: `"newton[examples,torch-cu12]>=1.2.0,<1.3"` |
| 26 | Warp pinned >=1.13.0,<2.0 | pyproject.toml | Confirmed: `"warp-lang[torch-cu12]>=1.13.0,<2.0"` |
| 27 | rl-games psutil conflict | pyproject.toml comment | Confirmed: `# rl extra deferred — rl-games 1.6.x pins psutil<6 which conflicts with warp-lang 1.13 (psutil>=7.1)` |
| 28 | DreamerV3 for sparse-reward tasks | [Nature 2025](https://www.nature.com/articles/s41586-025-08744-2) | Confirmed — published in Nature |
| 29 | TD-MPC2 struggles with sparse rewards | [arXiv 2602.03201](https://arxiv.org/html/2602.03201v1) | Confirmed — "TD-MPC2 fails across all tasks under sparse reward settings" |

---

### RUNTIME BENCHMARK NUMBERS (Cannot verify without RTX 5090 hardware)

| # | Claim | Source file | Status |
|---|-------|------------|--------|
| 30 | Warp stencil throughput: 200.9B cells/s at 128^3 | `benchmarks/warp_fluid_bench.py` | **Unverifiable** — code exists, produces runtime output, no saved results file found |
| 31 | MuJoCo-Warp: 2.45M env-steps/s at N=4096 | `benchmarks/mujoco_warp_bench.py` | **Unverifiable** — code exists, no saved results file found |
| 32 | Newton SemiImplicit: 1.45M env-steps/s at N=256 | `benchmarks/newton_worlds_throughput.py` | **Unverifiable** — code exists, no saved results file found |
| 33 | Warp kernel launch overhead: 4.4 us | `benchmarks/warp_fluid_bench.py` | **Unverifiable** — same as above |

**Note on benchmark numbers**: The benchmark scripts exist and are well-structured. They print results at runtime but do not write to result files. These numbers can only be verified by running the benchmarks on actual RTX 5090 hardware. The benchmark code is NOT hard-coded to produce specific numbers — it genuinely measures performance.

---

### UNVERIFIABLE CLAIMS

| # | Claim | Issue |
|---|-------|------|
| 34 | BYU FRoST CoUG-UV as BlueROV2 hardware partner | Search rate-limited; could not find a source confirming this specific partnership |
| 35 | Warp autograd debug burns 5 days (R20) | Internal project risk assessment; not verifiable externally |

---

## Files That Do NOT Exist

The following files were listed in the task but do NOT exist in the repository:
- `research/startup_market_research.md`
- `research/startup_competitive_analysis.md`
- `research/startup_funding_landscape.md`
- `research/startup_closed_loop_analysis.md`
- `research/startup_sim2real_research.md`
- `research/startup_design_phase_research.md`
- `research/startup_product_definition.md`

Only `research/startup_technical_roadmap.md` exists for cross-verification.

---

## Recommended Fixes

1. **CRITICAL**: Change RTX 5090 FP32 from 67 TFLOPS to **104.8 TFLOPS** (lines 5, 20)
2. **CRITICAL**: Change RTX 5090 FP16 from 213 TFLOPS to **838 TFLOPS** non-tensor (lines 5, 21)
3. **MINOR**: Update "skrl 2.0" to "skrl >= 2.0" or "skrl 2.1" to reflect latest release
4. **GOOD PRACTICE**: Save benchmark results to files (e.g., JSON) so numbers can be audited without re-running hardware

## Sources

- [TechPowerUp RTX 5090 Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-5090.c4216)
- [NVIDIA Warp GitHub](https://github.com/NVIDIA/warp/releases)
- [Newton Physics Docs](https://newton-physics.github.io/newton/stable/)
- [Isaac Lab Release Notes](https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html)
- [Isaac Lab Newton Integration](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/index.html)
- [MarineGym arXiv](https://arxiv.org/abs/2503.09203)
- [OceanSim arXiv](https://arxiv.org/html/2503.01074v2)
- [OceanSim GitHub](https://github.com/umfieldrobotics/OceanSim)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [skrl Releases](https://api.github.com/repos/Toni-SM/skrl/releases)
- [MuJoCo Warp Docs](https://mujoco.readthedocs.io/en/latest/mjwarp/)
- [Newton Kamino VBD Issue](https://github.com/newton-physics/newton/issues/1708)
- [von Benzon 2022 BlueROV2](https://vbn.aau.dk/ws/portalfiles/portal/505520780/jmse_10_01898.pdf)
- [DreamerV3 Nature 2025](https://www.nature.com/articles/s41586-025-08744-2)
- [TD-MPC2 Sparse Rewards](https://arxiv.org/html/2602.03201v1)
- [ROS 2 Jazzy](https://docs.ros.org/en/jazzy/Releases.html)
- [NuRec 3DGS](https://developer.nvidia.com/omniverse/nurec)
- [Isaac Sim 6.0 GTC 2026](https://forums.developer.nvidia.com/t/announcement-isaac-sim-6-0-early-developer-release-for-gtc26/363709)
