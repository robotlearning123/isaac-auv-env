export default {
  eyebrow: "Benchmarks",
  title: "Numbers, side by side",
  headers: ["Metric", "OceanScale", "UWSim", "HoloOcean", "Gazebo"],
  rows: [
    ["Single-env FPS (RTX 5090)", "**11,435**", "~200", "~500", "~100"],
    ["Multi-env throughput @ 8192 envs", "**90 M** env-steps/s", "—", "—", "—"],
    ["Multi-env parallel scaling", "8192+", "—", "partial", "partial"],
    ["GPU-native physics", "Newton", "—", "—", "—"],
    ["RL gym interface", "Native", "3rd-party", "3rd-party", "3rd-party"],
    ["SPH fluid kernel", "✓", "—", "—", "—"],
    ["Fossen 6-DoF", "✓", "✓", "partial", "—"],
    ["ROS 2 integration", "v0.3", "✓", "✓", "✓"],
  ],
  caveat: "OceanScale figures measured by W2 benchmark (2026-05-15, RTX 5090). Other tools' figures sourced from public documentation and community reports; actual values depend on scene and hardware.",
};
