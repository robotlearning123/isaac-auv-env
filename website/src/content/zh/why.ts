export default {
  title: "为什么海洋机器人需要专门的仿真基础设施",
  problems: [
    {
      title: "真实海上试验,锁定迭代节奏",
      body: "单日海上成本 ¥10,000–¥100,000+。腐蚀、缠绕、设备丢失。海况无法复现。迭代速度长期受自然条件设上限,而不是受工程能力设上限。",
    },
    {
      title: "经典 CFD,够不到 RL 训练规模",
      body: "高保真 CFD:1 秒物理 = 1–100 小时计算。RL 训练所需 1M+ episodes,数量级不可达。即便 GPU 加速,也距离海洋机器人 RL 所需的 real-time × 1000× 门槛仍远。",
    },
    {
      title: "现有水下仿真器,浪费现代 GPU 算力",
      body: "Gazebo / UWSim / HoloOcean 仍是单线程 CPU,百级 FPS。RTX 5090 / H100 算力闲置。多智能体并行能力接近零。",
    },
    {
      title: "Sim-to-real gap,卡死策略上桥",
      body: "流体保真度不足,仿真训练的策略在水里崩溃。缺乏可信的 sim-to-real 校准方法。传感器模型(声纳、水下相机)被过度简化,带来虚假信心。",
    },
    {
      title: "AI-native 接口缺失",
      body: "gym / gymnasium 适配缺失或低质。多智能体场景支持差。多模态海洋数据集(sonar / optical / IMU)不开放,基础模型训练受困于数据稀缺。",
    },
  ],
};
