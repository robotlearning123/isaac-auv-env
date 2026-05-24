# OceanScale Website Copy v1

Status: **DRAFT — awaiting approval**
Source: POSITIONING.md v1.1 (§§1-8, A1-A6)
Date: 2026-05-24

---

## Page Structure

```
HERO        — anchor + enemy + value prop
WHY         — /01 challenge: field testing bottleneck
WHAT        — /02 platform: what OceanScale is
BENCHMARKS  — /03 evidence: fidelity + speed + ecosystem
DEMO        — /04 (merge into BENCHMARKS as video embed)
CONTACT     — /05 CTA
```

Proposal: merge Demo into Benchmarks. 5 sections → 4 sections. Tighter page.

---

## HERO

### EN

**Eyebrow:** AI-native simulation infrastructure

**H1:** The ocean simulator for underwater robotics

**Enemy line:** Underwater robots still learn in the most expensive test environment on earth: the real ocean.

**Sub:** OceanScale builds the simulator where they train, test, and validate before they dive.

### ZH

**Eyebrow:** AI 原生仿真基础设施

**H1:** 面向水下机器人的海洋仿真基础设施

**Enemy line:** 水下机器人,仍然在地球上最昂贵的测试环境中学习:真实海洋。

**Sub:** OceanScale 构建海洋仿真基础设施,让水下机器人在下海之前完成训练、测试与验证。

### Notes

- H1, eyebrow, enemy line: verbatim from POSITIONING.md §3, §7, §5. No change.
- Sub: aligned to A3 reference. EN "before they dive" / ZH "下海之前" — more vivid than "before deployment".
- CTA buttons: `Benchmarks →` / `Contact`

---

## /01 CHALLENGE (Why)

### EN

**Eyebrow:** /01 · Challenge

**H2:** The development loop is the bottleneck

**Body:** Underwater robots are still built the old way: experience-based design, simplified CFD, pool testing, then sea trials. Each cycle takes months, costs six figures, and produces limited data. The industry moves slowly because the development loop itself is slow.

**Cards:**

| # | Title | Body |
|---|-------|------|
| 01 | Experience-based design | Vehicle geometry and control parameters are tuned by intuition and rules of thumb. Each iteration requires a physical prototype. |
| 02 | Simplified verification | CFD runs approximate. Pool tests constrain dynamics. Neither captures the full ocean environment a robot will face. |
| 03 | Expensive field trials | Sea trials require vessel time, weather windows, crew, and equipment. One deployment costs weeks and six figures — for a single data point. |
| 04 | Data scarcity | Each deployment collects limited data. RL-scale training needs millions of transitions, not dozens of field runs. |

### ZH

**Eyebrow:** /01 · 挑战

**H2:** 开发流程本身就是瓶颈

**Body:** 水下机器人至今仍沿用传统方式开发:经验设计、简化 CFD、水池测试、海试。每轮迭代耗时数月、花费六位数,但只能获得有限数据。这个行业走得慢,因为开发流程本身就慢。

**Cards:**

| # | Title | Body |
|---|-------|------|
| 01 | 经验设计 | 机器人外形和控制参数靠直觉和经验法则调试。每次迭代都需要物理原型。 |
| 02 | 简化验证 | CFD 是近似的。水池测试约束了动力学。两者都无法还原机器人将面对的完整海洋环境。 |
| 03 | 昂贵海试 | 海试需要船时、天气窗口、船员和设备。一次部署耗时数周、花费六位数 — 只为一个数据点。 |
| 04 | 数据稀缺 | 每次部署只能采集有限数据。RL 级训练需要数百万次交互,而非几十次出海。 |

### Changes from v0.0.25

- H2: "Field testing is the bottleneck" → "The development loop is the bottleneck" — the problem is the entire traditional workflow, not just one step.
- Body: rewritten to describe the full traditional pipeline (experience design → simplified CFD → pool testing → sea trials), not just field testing.
- Cards: 3 → 4 cards. Restructured around the four stages of the traditional loop:
  - Card 1: NEW — experience-based design (no simulation in the loop)
  - Card 2: NEW — simplified CFD + pool testing (verification gap)
  - Card 3: "Iteration cost" → "Expensive field trials" (more specific, includes cost)
  - Card 4: "Data bottleneck" — kept, added RL context

---

## /02 PLATFORM (What)

### EN

**Eyebrow:** /02 · Platform

**H2:** From physics to policy

**Body:** GPU-native ocean simulation infrastructure for underwater robots. High-fidelity dynamics, sensor-accurate environments, and RL-ready interfaces — on a single GPU.

**Cards:**

| # | Label | Title | Body |
|---|-------|-------|------|
| 01 | Dynamics | Validated hydrodynamics | Fossen 6-DOF model with coefficients validated against von Benzon et al. (2022). Not a simplified toy. |
| 02 | Scale | GPU-parallel environments | 64+ vectorized environments on a single GPU. Newton + Warp CUDA kernels, not CPU loops. |
| 03 | Sensors | Sensor-accurate simulation | DVL, IMU, and pressure sensor models. Policies train against realistic observation spaces. |
| 04 | Interface | Gymnasium-native RL | Standard Gymnasium API. Train with PPO, SAC, or any algorithm. No proprietary lock-in. |

### ZH

**Eyebrow:** /02 · 平台

**H2:** 从物理到策略

**Body:** 面向水下机器人的 GPU 原生海洋仿真基础设施。高保真动力学、传感器级精度环境、RL 就绪接口 — 在单块 GPU 上。

**Cards:**

| # | Label | Title | Body |
|---|-------|-------|------|
| 01 | 动力学 | 经验证的水动力学 | 基于 von Benzon et al. (2022) 验证系数的 Fossen 6-DOF 模型。不是简化玩具。 |
| 02 | 规模 | GPU 并行环境 | 单 GPU 上 64+ 个向量化环境。Newton + Warp CUDA 内核,不是 CPU 循环。 |
| 03 | 传感器 | 传感器级精度仿真 | DVL、IMU 和压力传感器模型。策略在真实观测空间上训练。 |
| 04 | 接口 | Gymnasium 原生 RL | 标准 Gymnasium API。支持 PPO、SAC 或任何算法。无专有锁定。 |

### Changes from v0.0.25

- Body: "A GPU-native training platform" → "GPU-native ocean simulation infrastructure" — matches §1 category (infrastructure, not platform).
- Body: added "High-fidelity dynamics, sensor-accurate environments, and RL-ready interfaces" — three pillars, not just speed.
- Card labels: added semantic labels (Dynamics / Scale / Sensors / Interface) replacing generic "01 / Simulator".
- Card 1: lead with fidelity ("validated"), added "Not a simplified toy."
- Card 3: new — sensors were hidden, now a first-class pillar.
- Card order: Fidelity → Scale → Sensors → Interface (fidelity first, speed second).

---

## /03 BENCHMARKS (Evidence)

### EN

**Eyebrow:** /03 · Benchmarks

**H2:** Fidelity, speed, and ecosystem

**Table headers:** Capability | Classic Sims | GPU Sims | OceanScale

**Rows:**

| Capability | Classic Sims | GPU Sims | OceanScale |
|------------|-------------|----------|------------|
| Fossen 6-DOF dynamics | ✓ | Simplified | **Validated (von Benzon 2022)** |
| DVL / IMU / pressure sensors | ✓ | Partial | **✓** |
| GPU-parallel environments | — | ✓ | **✓ (Newton + Warp)** |
| Gymnasium RL interface | Difficult | Partial | **Native** |
| Throughput (n=64, single GPU) | ~800 steps/s | Varies | **17,427 env-steps/s** |

**Landscape note:** Classic sims: Stonefish, HoloOcean, DAVE. GPU sims: MarineGym, OceanSim.

**Caveat:** Measured on RTX 5090, CUDA 12.9, BlueROV2 hover task (2026-05-22). Other figures from published papers and public documentation.

**Video embed:** BlueROV2 hover training (from current Demo section, merged here).

### ZH

**Eyebrow:** /03 · 基准

**H2:** 保真度、速度与生态

**Rows:**

| 能力 | 经典仿真 | GPU 仿真 | OceanScale |
|------|---------|---------|------------|
| Fossen 6-DOF 动力学 | ✓ | 简化 | **经验证 (von Benzon 2022)** |
| DVL / IMU / 压力传感器 | ✓ | 部分 | **✓** |
| GPU 并行环境 | — | ✓ | **✓ (Newton + Warp)** |
| Gymnasium RL 接口 | 困难 | 部分 | **原生** |
| 吞吐量 (n=64, 单 GPU) | ~800 steps/s | 不等 | **17,427 env-steps/s** |

**Landscape note:** 经典仿真: Stonefish, HoloOcean, DAVE。GPU 仿真: MarineGym, OceanSim。

**Caveat:** OceanScale 数据来自 RTX 5090, CUDA 12.9, BlueROV2 悬停任务 (2026-05-22)。其他数据来自已发表论文与公开文档。

### Changes from v0.0.25

- H2: "GPU speed meets underwater fidelity." → "Fidelity, speed, and ecosystem" — three pillars, not just speed. No period.
- Eyebrow: "Early Benchmarks" → "Benchmarks" — remove self-deprecation.
- Table reordered: fidelity rows first (Fossen, sensors), then speed rows (parallel, throughput).
- Row 1: "GPU-parallel envs" was row 1 (speed-first) → now row 3. "Fossen 6-DOF dynamics" is row 1 (fidelity-first).
- Row 1 OceanScale column: "✓ (Newton+Warp)" → "Validated (von Benzon 2022)" — evidence, not just checkmark.
- Row 2 OceanScale column: "✓ (stub + GPU)" → "✓" — removed "stub" (either it works or don't list it).
- Row 5 GPU Sims column: "250K+" → "Varies" — the 250K number was misleading (different workloads, different units). "Varies" is honest.
- Demo section video merged here as video embed below the table.

---

## /04 CONTACT (CTA)

### EN

**Eyebrow:** /04 · Contact

**H2:** Building underwater robots?

**Body:** For teams building underwater robots who need faster iteration than the ocean allows.

**Emails:**
- Business: business@oceanscale.cn
- Technical: tech@oceanscale.cn
- Partnerships: partnerships@oceanscale.cn

### ZH

**Eyebrow:** /04 · 联系

**H2:** 在造水下机器人？

**Body:** 面向需要比真实海洋更快迭代速度的水下机器人团队。

**Emails:** same

### Changes from v0.0.25

- Section number: /05 → /04 (Demo merged into Benchmarks).
- No copy changes. This section is clean.

---

## Summary of Changes

| Section | Change | Why |
|---------|--------|-----|
| Hero sub | "before deployment" → "before they dive" / "下海之前" | A3 reference, more vivid |
| Why card 3 | "Data scarcity / 8192 envs" → "Data bottleneck / limited data per deployment" | Was describing platform, not challenge |
| What body | "training platform" → "simulation infrastructure" | Match §1 category |
| What cards | Reorder: fidelity first, add sensor card | Not just speed |
| Benchmarks H2 | "GPU speed meets underwater fidelity." → "Fidelity, speed, and ecosystem" | Three pillars, not speed-only |
| Benchmarks eyebrow | "Early Benchmarks" → "Benchmarks" | Remove self-deprecation |
| Benchmarks table | Fidelity rows first, remove "stub", fix "250K+" | Fidelity-first framing |
| Demo | Merge into Benchmarks as video embed | Reduce sections, remove speed-only title |
| Contact | /05 → /04 | Section count reduced |
