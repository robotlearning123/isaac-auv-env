# OceanScale — NVIDIA 官方流体 Feature Gap Analysis

> 8 个 agent 并行调研，全部声明已对照官方文档/源码验证。
> Warp 1.13.0 + Newton 1.2.0，RTX 5090 环境。
> 日期：2026-05-24

---

## 当前已使用的 NVIDIA API

| Feature | 文件 | API | 状态 |
|---------|------|-----|------|
| Warp kernel JIT | 全部 fluid/, hydro/ | `@wp.kernel`, `wp.launch()` | ✅ |
| HashGrid SPH | `fluid/grid.py:841` | `wp.HashGrid`, `hash_grid_query()` | ✅ |
| Newton SolverImplicitMPM | `fluid/mpm.py:87` | `newton.solvers.SolverImplicitMPM` | ✅ |
| Newton SolverSemiImplicit | `newton_env.py:118` | `newton.solvers.SolverSemiImplicit` | ✅ |
| Newton SolverMuJoCo | `newton_env.py:120` | `newton.solvers.SolverMuJoCo` | ✅ |
| Warp 原子操作 | `grid.py:265`, `tier1_kernels.py:34` | `wp.atomic_add()` | ✅ |
| Warp 数学库 | `wave.py:64-75` | `wp.cosh/sinh/cos/sin/sqrt` | ✅ |

**自己手写的物理**（未用官方 API）：
- Chorin 投影法 Eulerian CFD (`grid.py:24-228`)
- SPH 密度+力 Müller kernels (`grid.py:647-738`)
- Airy 波浪 + JONSWAP/PM 谱求和 O(N×M) (`wave.py:28-163`)
- Fossen 6-DOF 水动力 (`tier1_kernels.py:17-246`)
- 球体边界近似 (`grid.py:271 _apply_boundary_sphere`)

---

## ★★★ 直接相关 — 核心基础设施升级

### 1. wp.Volume (NanoVDB) 替代密集网格
**来源**: nvidia.github.io/warp/stable/modules/volumes.html · Warp 1.13.0 .pyi stubs 验证

**内存对比**（10 场 × float32）：

| 分辨率 | 密集数组 | wp.Volume (1% 活跃) | 倍率 |
|--------|---------|---------------------|------|
| 64³ | 10 MB | ~0.1 MB | 100x |
| 256³ | 640 MB | ~6.4 MB | 100x |
| 512³ | 5.1 GB | ~51 MB | 100x |
| 1024³ | **40 GB (超 GPU)** | ~400 MB | 100x |

**已验证 API**:
```python
# 稀疏分配（只在 ROV/表面/边界附近分配）
vol = wp.Volume.allocate_by_tiles(tile_points, voxel_size=0.1, bg_value=0.0)

# 各向异性体素（水平 >> 垂直分辨率）—— v1.13 新增
vol = wp.Volume.allocate(min, max, voxel_size=(1.0, 0.5, 1.0))

# 速度场采样
velocity = volume_sample_v(vol_id, world_pos, wp.Volume.LINEAR)  # vec3f

# 可微采样（RL 梯度回传）
scalar = volume_sample_grad_f(vol_id, uvw, mode, grad)  # grad 是输出

# 活跃体素迭代（只处理有流体的区域）
voxels = vol.get_voxels()  # vec3i array
wp.launch(kernel, dim=vol.get_voxel_count())
```

**集成难度**: 中 · **价值**: 从实验室级 → 虚拟海洋的关键跳板

---

### 2. wp.Mesh 真实几何 FSI
**来源**: Warp 1.13.0 built-ins · help(wp.Mesh) 验证

替代 `grid.py:271 _apply_boundary_sphere` 球体近似 → 真实 BlueROV2 几何。

**已验证 API**:
```python
mesh = wp.Mesh(points, indices, velocities=vel,
               groups=env_groups)  # multi-env via group-aware BVH (v1.11+)

# 最近点查询（流体 → 物体边界距离）
result = mesh_query_point_sign_normal(mesh_id, point, max_dist, epsilon)
# → result, sign, face, u, v

# 运动边界速度
v = mesh_eval_velocity(mesh_id, face, u, v)  # no-slip 边界条件

# AABB 查询（批量碰撞检测）
query = mesh_query_aabb(mesh_id, low, high)

# Multi-env: group-aware 查询
root = mesh_get_group_root(mesh_id, env_idx)
hit = mesh_query_ray(mesh_id, start, dir, max_t, root)
```

**注意**: Mesh 查询**不可微**（离散拓扑操作）。

**集成难度**: 中 · **价值**: 非对称拖曳力精确计算，sim2real 核心

---

### 3. mesh_query_ray 物理传感器
**来源**: Warp 1.13.0 built-ins 验证

当前 `sensors/dvl.py` 是解析模型 → 升级为物理光线投射。

```python
# DVL 声束 → 真实海底地形回波
hit = mesh_query_ray(seabed_mesh_id, beam_origin, beam_dir, max_range, root)
# → hit.result, hit.t (距离), hit.normal, hit.face

# 快速遮挡检测
blocked = mesh_query_ray_anyhit(mesh_id, start, dir, max_t, root)

# 声纳扇形扫描
count = mesh_query_ray_count_intersections(mesh_id, start, dir, root)
```

**BVH 加速可选**:
```python
bvh = wp.Bvh(lowers, uppers, groups=env_groups)
query = bvh_query_ray(bvh_id, start, dir, root)
```

**集成难度**: 中 · **价值**: DVL/声纳从数学模型 → 几何回波

---

### 4. 可微流体仿真 + PyTorch RL
**来源**: example_navier_stokes_perturbation.py · Warp differentiability docs · interoperability_pytorch.html

**已证明可行**:
- `example_navier_stokes_perturbation.py` 完整 backprop through FFT NS + wp.Tape + wp.optim.Adam
- Warp ↔ PyTorch 零拷贝梯度桥

```python
# PyTorch policy → Warp physics → 梯度回流
wp_state = wp.from_torch(policy_output)  # requires_grad=True
tape = wp.Tape()
with tape:
    wp.launch(physics_step, inputs=[wp_state], outputs=[wp_loss])
tape.backward(loss=wp_loss)
# policy_output.grad 已填充 → PyTorch optimizer.step()
```

**可微 vs 不可微**:
| 操作 | 可微? |
|------|-------|
| volume_sample_f / volume_sample_grad_f | ✅ |
| wp.atomic_add | ✅ |
| tile_fft / tile_ifft | ✅ (v1.12+) |
| HashGrid 查询 | ❌ |
| Mesh 查询 | ❌ |
| MarchingCubes | ❌ |

**OceanScale 做可微流体 RL 将是 Warp 生态中首例。**

**集成难度**: 高 · **价值**: 梯度策略优化 + differentiable physics 论文

---

### 5. Tile FFT 波浪加速
**来源**: Warp 1.13.0 .pyi stubs · test_tile_fft.py · example_fft_poisson_navier_stokes_2d.py

替代 `wave.py:51` 的 `for c in range(n_components)` 显式求和。

```python
@wp.kernel
def wave_fft_kernel(spectrum: wp.array2d(dtype=wp.vec2f),
                    spatial: wp.array2d(dtype=wp.vec2f)):
    data = wp.tile_load(spectrum, shape=(64, 64))
    wp.tile_ifft(data)  # 谱 → 空间
    wp.tile_store(spatial, data)
```

**要求**: Warp 编译时需 MathDx 支持。
**数据类型**: `wp.vec2f` (float32 complex) 或 `wp.vec2d`。
**限制**: FFT 沿最后一维计算，tile 大小通常 power-of-2。

**集成难度**: 中 · **价值**: 大规模波场 O(N log N) vs O(N×M)

---

### 6. CUDA Graph 捕获
**来源**: Warp runtime docs · wp.ScopedCapture 验证

```python
# 捕获整个 env.step() 为 CUDA graph
with wp.ScopedCapture() as capture:
    for i in range(sim_steps):
        wp.launch(step_kernel, inputs=[...])

# 重放 — 零 Python 开销
wp.capture_launch(capture.graph)

# 序列化部署 (v1.13)
wp.capture_save(capture.graph, "oceanscale_step", inputs={...}, outputs={...})
# → .wrp 文件，支持 C++ 回放
```

**集成难度**: 低 · **价值**: RL rollout 消除 Python 开销

---

## ★★ 有用 — 提升仿真保真度

### 7. Newton add_rod() 系缆仿真
**来源**: Newton 1.2.0 inspect.signature 验证 · 4 个 cable 示例

```python
body_ids, joint_ids = builder.add_rod(
    positions=tether_points,  # list[Vec3]
    radius=0.01,
    stretch_stiffness=1e4, stretch_damping=10.0,
    bend_stiffness=100.0, bend_damping=1.0
)
# 分支网络
builder.add_rod_graph(nodes, edges, radius, stretch_stiffness, ...)
```

**已测试**: 10 段 rod + SolverSemiImplicit → 重力下垂确认。
**示例**: cable_bundle_hysteresis（Dahl 摩擦）, cable_pile, cable_twist, cable_y_junction。

**集成难度**: 低 · **价值**: ROV 脐带缆/系泊绳

---

### 8. Newton 布料 — 水下柔性结构
**来源**: Newton 1.2.0 验证 · 8 个 cloth 示例

```python
builder.add_cloth_grid(pos, rot, vel, dim_x=20, dim_y=20,
                       cell_x=0.05, cell_y=0.05, mass=0.1,
                       tri_ke=1e3, tri_ka=1e3, tri_kd=10.0,
                       drag_coefficient=0.1, lift_coefficient=0.0)
builder.color()  # SolverVBD 必需
```

**求解器**: SolverVBD（vertex block descent）或 SolverStyle3D（投影动力学）。

**集成难度**: 中 · **价值**: 渔网、拖网、ROV 防护罩

---

### 9. Newton MPM 双向耦合 (FSI)
**来源**: example_mpm_twoway_coupling.py 验证

**官方模式**:
```python
# 1. MPM solver 计算碰撞冲量
solver.collect_collider_impulses()

# 2. 自定义 Warp kernel 转换为刚体力
@wp.kernel
def apply_forces(particle_impulses, body_f):
    wp.atomic_add(body_f, body_idx, spatial_force)

# 3. 刚体 solver 步进
```

Newton state 全部是原生 `wp.array` → GPU 零拷贝。

**集成难度**: 中 · **价值**: ROV 刚体 ↔ MPM 流体真实力交换

---

### 10. wp.MarchingCubes 海面提取
**来源**: wp.MarchingCubes API 验证

```python
mc = wp.MarchingCubes(nx=128, ny=128, nz=128)
mc.surface(sdf_field, threshold=0.0)
verts, indices = mc.verts, mc.indices  # 三角网格
```

**集成难度**: 低 · **价值**: Volume SDF → 海面三角网格（渲染/碰撞）

---

### 11. AdaptiveNanogrid 多分辨率 FEM
**来源**: warp.fem docs 验证

```python
# ROV 附近精细，远处粗糙
grid = warp.fem.adaptive_nanogrid_from_field(volume, field_fn, ...)
```

**集成难度**: 高 · **价值**: 自适应精度，计算资源最优分配

---

## ★ 有潜力 — 未来扩展

| # | Feature | API | 用途 |
|---|---------|-----|------|
| 12 | Newton SolverFeatherstone | `newton.solvers.SolverFeatherstone` | 多关节 ROV 臂/机械手 |
| 13 | Newton SolverKamino | `newton.solvers.SolverKamino` | 运动学闭链约束 |
| 14 | Newton HydroelasticSDF | `newton.geometry.HydroelasticSDF` | 平滑接触力（非流体相关，名字误导） |
| 15 | warp.fem Shallow Water | `example_shallow_water.py` | 公里级海面高度场 |
| 16 | Newton add_shape_heightfield | `builder.add_shape_heightfield()` | 海底地形碰撞 |
| 17 | Newton 8 关节类型 | ball/cable/d6/distance/fixed/free/prismatic/revolute | 复杂机构 |

---

## ❌ 已排除（不相关）

| 之前猜测 | 实际情况 | 来源 |
|----------|---------|------|
| NVIDIA Modulus 海洋模型 | **大气天气模型**（FourCastNet），不做海洋 | docs.nvidia.com/modulus |
| Omniverse Flow | **烟火专用**，不做水 | nvidia-omniverse.github.io/PhysX/flow |
| PhysX Flex 流体 | **已废弃**，流体迁移到 Newton/Warp | developer.nvidia.com/physx-sdk |
| Multi-env FEM | **UNRELEASED**，不在 1.13，GH-1407 | CHANGELOG unreleased section |
| NVIDIA WaveWorks | **可能已停产**，未在官方页面找到 | (not found) |
| float16 HashGrid | API 存在但精度太低（3.3 位有效数字，max 65504），不适合海洋坐标 | help(wp.HashGrid) |

---

## 推荐行动优先级

| 优先级 | Feature | 难度 | ROI | 理由 |
|--------|---------|------|-----|------|
| **P0** | wp.Volume (NanoVDB) | 中 | ★★★★★ | 64³ → 公里级域的唯一路径 |
| **P0** | CUDA Graph 捕获 | 低 | ★★★★ | 立竿见影的 RL 训练加速 |
| **P1** | wp.Mesh FSI | 中 | ★★★★ | sim2real 保真度核心 |
| **P1** | mesh_query_ray 传感器 | 中 | ★★★★ | DVL/声纳物理化 |
| **P1** | Newton add_rod 系缆 | 低 | ★★★ | 真实运维场景必需 |
| **P2** | Tile FFT 波浪 | 中 | ★★★ | 大规模波场性能 |
| **P2** | 可微 NS + RL | 高 | ★★★★★ | 生态首例，论文价值 |
| **P2** | MPM 双向耦合 | 中 | ★★★ | 真实流体力交换 |
| **P3** | MarchingCubes | 低 | ★★ | 海面可视化 |
| **P3** | Cloth 布料 | 中 | ★★ | 渔网/柔性结构 |
| **P3** | AdaptiveNanogrid | 高 | ★★★ | 自适应精度（依赖 P0） |

---

## NVIDIA 水下流体技术栈结论

**NVIDIA 官方没有海洋/水下模拟。OceanScale 在 Warp + Newton 之上构建的水下物理层是独特的。**

真正的 gap 不是"缺少官方海洋 API"，而是**未充分利用已有的通用基础设施**：
- 稀疏存储 (Volume) — 解锁规模
- 几何查询 (Mesh) — 解锁保真度
- 自动微分 (Tape) — 解锁可微物理
- 图捕获 (ScopedCapture) — 解锁性能
- 柔体/系缆 (rod/cloth) — 解锁场景完整性

这些都是 Warp/Newton 的**通用能力**，OceanScale 是第一个把它们组合成水下仿真平台的项目。
