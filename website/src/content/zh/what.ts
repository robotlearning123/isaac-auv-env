export default {
  eyebrow: "Platform",
  title: "物理到策略。",
  capabilities: [
    {
      metric: null,
      metricLabel: "01 / Sim Core",
      title: "仿真内核",
      body: "Newton + Warp 物理路径，面向高吞吐训练循环。",
    },
    {
      metric: null,
      metricLabel: "02 / Vector Fabric",
      title: "环境矩阵",
      body: "8192 环境向量化，为策略搜索提供规模。",
    },
    {
      metric: null,
      metricLabel: "03 / Marine Physics",
      title: "水动力与传感器",
      body: "6-DoF、水流近似、声学与水下感知接口。",
    },
    {
      metric: null,
      metricLabel: "04 / Robotics Bridge",
      title: "机器人训练接口",
      body: "Gymnasium、OpenUSD 与 ROS 2 接口（Isaac Sim：未来集成路径）。",
    },
  ],
};
