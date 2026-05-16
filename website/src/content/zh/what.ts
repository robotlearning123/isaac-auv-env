export default {
  eyebrow: "Platform",
  title: "OceanScale 的能力",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · 单环境 BlueROV2 (RTX 5090)",
      title: "完整 RL 训练管线",
      body: "RTX 5090 单卡实测。环境到策略,端到端在仿真里跑通。",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen 多环境吞吐",
      body: "8192 并行环境,Fossen 6-DoF 内核。900× 目标已达成。",
    },
    {
      metric: "6-DoF",
      metricLabel: "Fossen + SPH 自研",
      title: "海洋机器人专属物理核",
      body: "刚体动力学 + SPH 流体仿真,为海洋场景定制。",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native 物理 + 渲染栈",
      body: "Anthropic + NVIDIA + Lightwheel + Apple 联合维护的 Newton 引擎。",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium 原生",
      title: "RL 接口即插即用",
      body: "rl_games / stable-baselines3 / RSL-RL 直接接入。",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "真实海洋场景库",
      body: "船 / AUV / ROV / 传感器资产 + procedural ocean 生成。",
    },
  ],
};
