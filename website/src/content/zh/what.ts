export default {
  eyebrow: "Platform",
  title: "我们造的东西",
  capabilities: [
    {
      metric: "11,435",
      metricLabel: "FPS · 单环境 BlueROV2 (RTX 5090)",
      title: "完整 RL 训练管线",
      body: "单卡端到端实测。环境到策略,全程在仿真里完成。",
    },
    {
      metric: "90 M",
      metricLabel: "env-steps/s @ 8192 envs",
      title: "Fossen 多环境吞吐",
      body: "8192 并行环境,Fossen 6-DoF 内核。900× 目标已达成。",
    },
    {
      metric: "6-DoF",
      metricLabel: "Fossen + SPH 自研内核",
      title: "海洋机器人专属物理核",
      body: "刚体动力学 + SPH 流体,为海洋场景定制。",
    },
    {
      metric: null,
      metricLabel: "Newton + Isaac Sim 6",
      title: "GPU-native 物理 + 渲染栈",
      body: "Newton 引擎由 Anthropic + NVIDIA + Lightwheel + Apple 联合维护。Isaac Sim 6 渲染。",
    },
    {
      metric: null,
      metricLabel: "gym / gymnasium 原生",
      title: "RL 接口即插即用",
      body: "rl_games · stable-baselines3 · RSL-RL,直接接入。",
    },
    {
      metric: null,
      metricLabel: "Multi-agent · Procedural",
      title: "真实海洋场景库",
      body: "船 / AUV / ROV / 传感器资产 + procedural ocean 生成。",
    },
  ],
};
