---
eyebrow: "Benchmarks"
title: "基准数据"
headers: ["指标", "OceanScale", "UWSim", "HoloOcean", "Gazebo"]
rows:
  - ["单环境吞吐 (RTX 5090)", "**1,326** env-steps/s", "~200", "~500", "~100"]
  - ["4096 环境吞吐", "**4.59 M** env-steps/s", "—", "—", "—"]
  - ["多环境并行扩展", "4096 verified", "—", "partial", "partial"]
  - ["GPU-native physics", "Newton", "—", "—", "—"]
  - ["RL gym 接口", "原生", "3rd-party", "3rd-party", "3rd-party"]
  - ["SPH 流体 kernel", "✓", "—", "—", "—"]
  - ["Fossen 6-DoF", "✓", "✓", "partial", "—"]
  - ["ROS 2 集成", "v0.3", "✓", "✓", "✓"]
caveat: "OceanScale 数据来自 launch-standardized benchmark（2026-05-24, RTX 5090）。其他工具数据来自公开文档与社区报告,仅作量级参考。"
---
