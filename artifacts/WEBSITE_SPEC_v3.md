# OceanScale 网站规格 v3

4 sections. Opus + GPT 讨论确认.

---

## 1. HERO

全屏视频背景。品牌名居中偏下。

**视频 [新制作]:** 大规模并行仿真全景 — 计算感和规模感, 不是机器人游泳
**文字:** OceanScale
**背景:** 视频是唯一视觉高潮

---

## 2. 问题

暗色背景 + GPT-Image-2 生成的系统对比图。

**对比图 [新制作]:** 一张基础设施级系统图
- 上方: 旧流程串行管道, 5 阶段, 海试处明显瓶颈, 最终一个输出点 (灰/暗琥珀)
- 下方: OceanScale 仿真核心扇出成千上万并行轨迹, 汇聚验证→部署 (品牌青)
- 无文字, 无标签 — HTML 叠加文案
- 风格: NVIDIA GTC / SpaceX 运控, 线条级技术图, 非 PPT

**文字叠加 (HTML):**

ZH:
> 真实海洋是最昂贵的测试场

EN:
> The real ocean is the most expensive test bed

**无 cards, 无段落。图 + 一句话。**

---

## 3. 产品

纯暗色 + 微弱网格纹理。H2 + 4 cards 带配图。

ZH:
> **面向水下机器人的虚拟海洋**

| Card | 标题 | 一句话 | 配图 [新制作] |
|------|------|-------|-------------|
| 物理引擎 | GPU 原生多体动力学 | 刚体、流体、接触、水动力在 GPU 上统一求解 | GPU 物理求解可视化 |
| 规模 | 大规模并行 | 成千上万个环境同时运行 | GPU 计算矩阵/并行执行网格 |
| 传感器 | 传感器级精度 | DVL、IMU、压力、声纳、相机 | 传感器数据叠加在机器人周围 |
| 接口 | Gymnasium 原生 | 标准 API, 支持任何 RL 算法 | 暗色 IDE 代码截图 |

EN:
> **A virtual ocean for underwater robots**

| Card | Title | One line | Image [new] |
|------|-------|---------|-------------|
| Physics | GPU-native multibody dynamics | Rigid bodies, fluids, contacts, hydrodynamics solved together on GPU | GPU physics solve visualization |
| Scale | Massively parallel | Thousands of environments running simultaneously | GPU compute matrix / parallel execution grid |
| Sensors | Sensor-accurate | DVL, IMU, pressure, sonar, camera | Sensor data overlay around robot |
| Interface | Gymnasium-native | Standard API, any RL algorithm | Dark IDE code screenshot |

**背景:** 纯暗色 #060b12 + 极淡网格线

---

## 4. 联系

纯暗色背景。H2 + intent selector + 表单。

ZH:
> **在造水下机器人？**

EN:
> **Building underwater robots?**

**Intent:** AUV / ROV / 仿真集成 / 科研 / 其他
**表单:** 姓名 / 邮箱 / 公司 / 留言
**按钮:** 提交 / Submit

**背景:** 纯暗色, 干净

---

## 素材清单 (全部新制作)

| # | 素材 | 类型 | 工具 | 用途 |
|---|------|------|------|------|
| 1 | Hero 视频 | 视频 10-15s loop | Seedance | Hero 背景 |
| 2 | 流程对比图 | 图 16:9 | GPT-Image-2 | 问题 section |
| 3 | 物理引擎 card | 图 | GPT-Image-2 | 产品 card |
| 4 | 规模 card | 图 | GPT-Image-2 | 产品 card |
| 5 | 传感器 card | 图 | GPT-Image-2 | 产品 card |
| 6 | 接口 card | 图 | GPT-Image-2 | 产品 card |

6 个素材。1 视频 + 5 图。全部新制作。

---

## 视觉节奏 (Opus + GPT 确认)

```
Hero   — 视频 (计算感, 规模感)      唯一视频
问题    — 暗色 + 系统对比图          图是主体
产品    — 纯暗 + 网格 + card 配图    技术感
联系    — 纯暗色                    干净
```

一个视觉高潮 (Hero), 然后逐步收敛到技术和行动。
