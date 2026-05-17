export default {
  eyebrow: "Platform",
  title: "OceanScale 如何为海洋机器人,打造 GPU-native 仿真引擎",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · 单环境 BlueROV2 (RTX 5090)",
      title: "完整 RL 训练管线",
      body: "RTX 5090 单卡端到端。从环境到策略,全程在仿真里完成,无需出海试验。",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen 多环境吞吐",
      body: "8192 并行环境,Fossen 6-DoF 内核。单卡达到生产规模 RL 所需吞吐的 900 倍。",
    },
    {
      metric: "6-DoF",
      metricLabel: "Fossen + SPH 自研内核",
      title: "海洋机器人专属物理核",
      body: "刚体动力学 + SPH 流体,为海洋场景定制。替代通用 CFD,用专为海洋机器人调优的内核。",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native 物理 + 渲染栈",
      body: "Newton 物理引擎,由 Anthropic、NVIDIA、Lightwheel、Apple 联合维护。Isaac Sim 6 光线渲染。完整 GPU-native 栈。",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium 原生",
      title: "RL 接口即插即用",
      body: "原生 gym / gymnasium 兼容。rl_games、stable-baselines3、RSL-RL 无需适配层直接接入。",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "真实海洋场景库",
      body: "船 / AUV / ROV / 传感器资产,加上 procedural ocean 生成。海洋机器人训练的世界层。",
    },
  ],
};
