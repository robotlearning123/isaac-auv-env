---
eyebrow: "Benchmarks"
title: "基准数据"
headers: ["指标", "OceanScale", "UWSim", "HoloOcean", "Gazebo"]
rows:
  - ["Single-env FPS (RTX 5090)", "**11,435**", "~200", "~500", "~100"]
  - ["Multi-env throughput @ 8192 envs", "**90 M** env-steps/s", "—", "—", "—"]
  - ["Multi-env parallel scaling", "8192+", "—", "partial", "partial"]
  - ["GPU-native physics", "Newton", "—", "—", "—"]
  - ["RL gym 接口", "原生", "3rd-party", "3rd-party", "3rd-party"]
  - ["SPH 流体 kernel", "✓", "—", "—", "—"]
  - ["Fossen 6-DoF", "✓", "✓", "partial", "—"]
  - ["ROS 2 集成", "v0.3", "✓", "✓", "✓"]
caveat: "OceanScale 数据来自 W2 实测(2026-05-15, RTX 5090)。其他工具数据来自公开文档与社区报告,仅作量级参考。"
---
