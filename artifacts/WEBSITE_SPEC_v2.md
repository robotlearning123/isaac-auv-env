# OceanScale 网站规格 v2

---

## HERO (视频背景 + 品牌名)

**视频:** 大规模并行仿真全景

> OceanScale

---

## 问题 (enemy line + old vs new 对比)

ZH:
> 真实海洋是最昂贵的测试场

```
旧流程:  经验设计 → 简化CFD → 水池 → 海试 → 一个数据点
OceanScale:  仿真 → 成千上万并行训练 → 验证 → 部署
```

EN:
> The real ocean is the most expensive test bed

```
Old loop:  design → simplified CFD → pool → sea trial → one data point
OceanScale:  simulate → thousands of parallel runs → validate → deploy
```

---

## 产品 — 面向水下机器人的虚拟海洋

ZH:
> **面向水下机器人的虚拟海洋**

| 物理引擎 | 规模 | 传感器 | 接口 |
|---------|------|-------|------|
| GPU 原生多体动力学 | 成千上万个环境同时运行 | DVL、IMU、压力、声纳、相机 | Gymnasium 原生 RL |

EN:
> **A virtual ocean for underwater robots**

| Physics engine | Scale | Sensors | Interface |
|---------------|-------|---------|-----------|
| GPU-native multibody dynamics | Thousands of environments simultaneously | DVL, IMU, pressure, sonar, camera | Gymnasium-native RL |

---

## 证据

ZH:
> **为保真度和规模而生**

| 能力 | 经典仿真 | GPU 仿真 | OceanScale |
|------|---------|---------|------------|
| GPU 原生物理 | — | 部分 | **✓** |
| 水动力学 | ✓ | 简化 | **✓** |
| 传感器仿真 | ✓ | 部分 | **✓** |
| 大规模并行 | — | ✓ | **✓** |
| Gymnasium RL | 困难 | 部分 | **原生** |

EN:
> **Built for fidelity and scale**

| Capability | Classic Sims | GPU Sims | OceanScale |
|------------|-------------|----------|------------|
| GPU-native physics | — | Partial | **✓** |
| Hydrodynamics | ✓ | Simplified | **✓** |
| Sensor simulation | ✓ | Partial | **✓** |
| Massively parallel | — | ✓ | **✓** |
| Gymnasium RL | Difficult | Partial | **Native** |

---

## 联系

ZH:
> **在造水下机器人？**

EN:
> **Building underwater robots?**

Intent: AUV / ROV / 仿真集成 / 科研 / 其他
姓名 / 邮箱 / 公司 / 留言 / 提交

---

## 素材

| 素材 | 内容 |
|------|------|
| Hero 视频 | 大规模并行仿真全景 |
| 4 张 card 图 | 物理/规模/传感器/接口 |
