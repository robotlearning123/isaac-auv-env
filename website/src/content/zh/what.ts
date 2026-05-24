export default {
  eyebrow: "平台",
  title: "从物理到策略",
  capabilities: [
    {
      metric: null,
      metricLabel: "01 / 仿真器",
      title: "GPU 原生海洋仿真",
      body: "Newton + Warp 物理引擎，在 GPU 上规模化运行传感器级精度的海洋环境。",
    },
    {
      metric: null,
      metricLabel: "02 / 环境",
      title: "大规模并行训练",
      body: "单 GPU 上 64+ 个并行环境，用于策略搜索。",
    },
    {
      metric: null,
      metricLabel: "03 / 动力学",
      title: "Fossen 水动力学",
      body: "基于 von Benzon et al. (2022) 验证系数的 6-DOF 动力学模型。",
    },
    {
      metric: null,
      metricLabel: "04 / 接口",
      title: "Gymnasium 原生 RL",
      body: "标准 Gymnasium 接口。支持 PPO、SAC 或任何 RL 算法。",
    },
  ],
};
