# OceanScale 素材生产计划

Status: **DRAFT**
Date: 2026-05-24
方式: **全部 AI 生成, 无实拍**

---

## 视觉风格统一

**色调:** 深海暗色 (#060b12) + 品牌青 (#2dd4bf)
**质感:** 电影级 CG, NVIDIA GTC 发布会风格, 非卡通
**光线:** 水下散射光 + 青色辅助光 + 体积光
**机器人:** 写实工业级 ROV/AUV, 不是玩具, 有推进器/线缆/传感器细节
**环境:** 深水, 粒子悬浮, 海底地形, 洋流可视化
**仿真感:** 网格线、坐标轴、数据叠加层 — 表明这是仿真不是实拍

---

## 视频 (5 个)

### V1 — Hero 背景 [P0]

**内容:** GPU 仿真全景 — 成千上万个水下机器人在虚拟海洋中并行训练
**镜头:**
- 深水空间, 青色网格线构成的虚拟海洋
- 数千个 ROV/AUV 矩阵排列, 每个在独立执行不同任务
- 缓慢俯瞰拉远, 展示规模感
- 数据粒子/轨迹线从机器人身上辐射出来

**时长:** 10-15s 无缝循环
**分辨率:** 1920x1080
**工具:** Seedance 2.0
**Prompt 方向:** "Cinematic top-down view of thousands of underwater robots training simultaneously in a dark virtual ocean environment, teal grid lines, volumetric lighting, GPU simulation visualization, NVIDIA Omniverse style, dark background, 4K"

---

### V2 — 挑战背景 [P0]

**内容:** 传统开发流程的痛苦 — 设计→CFD→水池→海试
**镜头:**
- 工程师对着屏幕调 CAD 模型 (暗色调)
- CFD 网格在屏幕上缓慢计算
- 水池测试: 小型水池中机器人有限活动
- 海试: 船甲板, 吊装, 等待, 恶劣天气

**时长:** 10s 循环
**工具:** Seedance 2.0
**Prompt 方向:** "Dark cinematic montage: engineer working late on CAD model, CFD simulation slowly computing on screen, small pool test with underwater robot, ship deck in storm deploying ROV, waiting, frustration, dark moody lighting"

---

### V3 — 平台展示 [P1]

**内容:** OceanScale 产品运行画面
**镜头:**
- 代码编辑器中 Python 代码运行
- 3D 视窗中机器人在仿真环境内运动
- 传感器数据实时叠加 (DVL 矢量, IMU 姿态, 声纳扇形)
- 训练曲线实时上升

**时长:** 8-10s
**工具:** Seedance 2.0 + 屏幕录制合成
**Prompt 方向:** "Dark IDE with Python code, 3D viewport showing underwater robot in simulation with sensor overlays, teal accent color, data visualization, clean dark UI"

---

### V4 — 大规模训练 (基准 section) [P1]

**内容:** 并行训练的近景 — 展示每个环境内的细节
**镜头:**
- 矩阵视角: 16x16 或更大的环境网格
- 每格内一个机器人执行不同阶段的任务
- 有些在学习悬停, 有些在导航, 有些在对接
- 右下角叠加性能指标 (env-steps/s, reward)

**时长:** 8-10s
**工具:** Seedance 2.0
**Prompt 方向:** "Grid matrix view of 256 underwater simulation environments, each cell showing a robot at different training stages, dark background, teal highlights, performance metrics overlay, GPU compute visualization"

---

### V5 — 联系背景 [P2]

**内容:** 宁静的深海 + 单个机器人准备出发
**镜头:**
- 深海环境, 安静, 粒子漂浮
- 一个机器人从仿真网格中浮出, 进入真实海洋
- sim-to-real 的意象

**时长:** 8s 循环
**工具:** Seedance 2.0
**Prompt 方向:** "Single underwater robot emerging from a teal grid simulation environment into realistic dark ocean, transition from virtual to real, volumetric light from above, peaceful, cinematic"

---

## 图片 (8 个)

### I1 — Hero poster (V1 静帧) [P0]

**内容:** V1 视频的关键帧 — 成千上万机器人俯瞰
**工具:** GPT-Image-2
**Prompt:** "Top-down cinematic view of thousands of small underwater robots arranged in grid formation inside a dark virtual ocean, teal grid lines and coordinate axes visible, volumetric teal light, particles, NVIDIA Omniverse simulation style, 16:9"

---

### I2 — 物理引擎 card [P1]

**内容:** GPU 上的多体物理求解可视化
**工具:** GPT-Image-2
**Prompt:** "Dark visualization of GPU-accelerated physics: multiple rigid bodies, fluid particles, contact forces shown as vector arrows, teal color scheme on black background, scientific visualization style, clean"

---

### I3 — 规模 card [P1]

**内容:** GPU 服务器 + 并行环境矩阵
**工具:** GPT-Image-2
**Prompt:** "Dark GPU server rack with teal LED lights, holographic grid of hundreds of tiny underwater simulation environments floating in front, each showing a robot, infrastructure aesthetic"

---

### I4 — 传感器 card [P1]

**内容:** 机器人周围的传感器数据可视化
**工具:** GPT-Image-2
**Prompt:** "Underwater robot surrounded by sensor data visualization: DVL velocity vectors, IMU orientation axes, sonar fan beam, pressure depth lines, teal on dark background, technical overlay, clean"

---

### I5 — 接口 card [P1]

**内容:** 简洁的 Python 代码 + Gymnasium 界面
**工具:** GPT-Image-2 或 截图
**Prompt:** "Clean dark IDE showing 8 lines of Python code: import oceanscale, create environment, train policy, teal syntax highlighting on dark background, minimal, JetBrains Mono font"

---

### I6 — 竞争象限 [P1]

**内容:** 保真度 × 速度 × 生态 三维定位图
**工具:** SVG 手动制作 (非 AI 生成)
**说明:** OceanScale 在右上角 (高保真+高速), 品牌色高亮, 其他竞品灰色

---

### I7 — 挑战 poster (V2 静帧) [P0]

**内容:** 传统流程的视觉 — 工程师/水池/船甲板
**工具:** GPT-Image-2
**Prompt:** "Split image: left side shows engineer frustrated at CAD screen in dark office, right side shows ROV being crane-lifted into stormy ocean from ship deck, dark moody cinematic lighting, 16:9"

---

### I8 — OG Image (社交分享图) [P2]

**内容:** 品牌卡片 — logo + 标题 + 仿真背景
**工具:** GPT-Image-2
**Prompt:** "OceanScale brand card: dark background, teal sonar rings logo, text 'The ocean simulator for underwater robotics', underwater simulation grid in background, 1200x630"

---

## 生产顺序

```
Phase 1 (P0, 先上线):
  I1 Hero poster → V1 Hero 视频
  I7 挑战 poster → V2 挑战视频

Phase 2 (P1, 第二批):
  I2-I5 四张 card 图
  I6 竞争象限 SVG
  V3 平台视频
  V4 大规模训练视频

Phase 3 (P2, 补充):
  V5 联系背景
  I8 OG Image
```

## 现有素材处置

| 文件 | 决定 |
|------|------|
| `sph-demo.mp4` | **临时 Hero 背景** — 直到 V1 制作完成 |
| `auv-demo.mp4` | **临时基准视频** — 直到 V4 制作完成 |
| `hero-storm.mp4` | **临时挑战背景** — 直到 V2 制作完成 |
| `hero-ambient.mp4` | **临时联系背景** — 直到 V5 制作完成 |
| 其余旧素材 | Phase 1 完成后删除, 不混用风格 |
