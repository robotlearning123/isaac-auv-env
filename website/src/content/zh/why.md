---
title: "为什么需要 OceanScale"
problems:
  - title: "真实海上测试昂贵到锁死创新节奏"
    body: "出海一天 ¥10,000–¥100,000+(船时 + 人员 + 设备折旧)。设备损坏率高(海水腐蚀、压力、缠绕、丢失)。环境不可重复(洋流、能见度、生物干扰)。进度受天气与季节限制。没有高保真仿真,迭代速度被自然界设了上限。"
  - title: "CFD 太慢,RL 训练根本不可达"
    body: "高保真 CFD 计算 1 秒物理需要 1–100 小时。RL 训练所需的 1M+ episodes 在数量级上根本无解。即使 GPU 加速的 CFD 也无法达到 real-time × 1000 倍的实际门槛。"
  - title: "现有水下仿真器浪费现代 GPU 算力"
    body: "Gazebo / UWSim / HoloOcean 仍是单线程 CPU,几十到几百 FPS,完全发挥不出 RTX 5090 / H100 的真实能力。多智能体与多场景并行能力极差。"
  - title: "Sim-to-real gap 仍卡死策略上桥"
    body: "流体动力学保真度不足导致策略迁移失败。缺乏有原理的 sim-to-real 校准方法。水下传感器(sonar、camera)仿真常被过度简化 —— 模型在仿真里完美,在水里崩溃。"
  - title: "AI-native 接口缺失,生态没追上时代"
    body: "gym/gymnasium 适配缺失或质量低劣。多智能体场景支持差。多模态海洋数据集(sonar/optical/IMU)不开放,基础模型训练受困于数据稀缺。"
---
