---
eyebrow: "Benchmarks"
title: "Benchmark data"
headers: ["Metric", "OceanScale", "UWSim", "HoloOcean", "Gazebo"]
rows:
  - ["Single-env throughput (RTX 5090)", "**1,326** env-steps/s", "~200", "~500", "~100"]
  - ["Multi-env throughput @ 4096 envs", "**4.59 M** env-steps/s", "—", "—", "—"]
  - ["Multi-env parallel scaling", "4096 verified", "—", "partial", "partial"]
  - ["GPU-native physics", "Newton", "—", "—", "—"]
  - ["RL gym interface", "Native", "3rd-party", "3rd-party", "3rd-party"]
  - ["SPH fluid kernel", "✓", "—", "—", "—"]
  - ["Fossen 6-DoF", "✓", "✓", "partial", "—"]
  - ["ROS 2 integration", "v0.3", "✓", "✓", "✓"]
caveat: "OceanScale figures measured by the launch-standardized benchmark (2026-05-24, RTX 5090). Other tools' figures come from public documentation and community reports; use them as order-of-magnitude references."
---
