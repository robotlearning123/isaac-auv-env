export default {
  title: "海洋机器人发展的五个枷锁",
  problems: [
    {
      title: "真实海上测试,昂贵到锁死创新节奏",
      body: "出海一天 ¥10,000–¥100,000+。腐蚀、缠绕、设备丢失。环境不可复现,进度看天气。没有高保真仿真,迭代节奏被自然界设了上限。",
    },
    {
      title: "CFD 太慢,RL 训练根本不可达",
      body: "高保真 CFD:1 秒物理 = 1–100 小时计算。 RL 训练所需的 1M+ episodes 数量级不可达。 GPU 加速的 CFD 也够不到 real-time × 1000 倍的实战门槛。",
    },
    {
      title: "现有水下仿真器,浪费现代 GPU 算力",
      body: "Gazebo / UWSim / HoloOcean 仍是单线程 CPU,百级 FPS。 RTX 5090 / H100 的算力被白白浪费。多智能体并行能力接近零。",
    },
    {
      title: "Sim-to-real gap,卡死策略上桥",
      body: "流体保真度不足,策略迁移失败。缺乏可信的 sim-to-real 校准方法。传感器仿真常被过度简化 —— 仿真里完美,水里崩溃。",
    },
    {
      title: "AI-native 接口缺失,生态没追上时代",
      body: "gym/gymnasium 适配缺失或低质。多智能体场景支持差。多模态海洋数据集 (sonar / optical / IMU) 不开放,基础模型训练受困于数据稀缺。",
    },
  ],
};
