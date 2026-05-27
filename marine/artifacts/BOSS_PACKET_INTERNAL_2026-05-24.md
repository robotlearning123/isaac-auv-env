# OceanScale 老板汇报一页纸

日期：2026-05-24
范围：内部汇报，不作为公开发布材料。

## 30 秒结论

OceanScale 的核心 alpha 技术能力已经有本地证据支撑：RTX 5090 上的 Newton + Warp + Torch CUDA 仿真、BlueROV2 / Fossen 6-DOF、Gymnasium vectorized env、benchmark、fidelity validation、测试和 package build 都能跑通。

但当前不建议 public launch。主要问题不是核心仿真，而是 release/version/CI/PyPI/live-site/data hygiene 没有收口。建议定位为：内部 alpha 技术验证通过，下一步做 release pipeline hardening 和设计伙伴预览准备。

## 已验证事实

| 项 | 当前证据 | 结论 |
|---|---|---|
| GPU 技术栈 | RTX 5090，Torch 2.11.0+cu128，Warp 1.13.0，Newton 1.2.0 | 可运行 |
| Python package | `oceanscale 0.1.0a0`，`uv build` + `twine check` 通过 | 本地可构建 |
| Product core | `ROVEnv`、`BatchedVecEnv`、BlueROV2、station keeping / docking / waypoint env files | 核心产品面存在 |
| Benchmark | `benchmarks/competitive/results.json` | 主 benchmark source of truth 已生成 |
| Throughput | n=64: 85,824 env-steps/s；n=4096: 4,588,922 env-steps/s | 可作为内部性能证据 |
| Fidelity | von Benzon comparison pass；position error 0.013%；attitude RMS 0.0025 deg | 可作为内部保真证据 |
| Test gate | ruff、mypy、pytest、package-data policy、package-internal version identity、launch-suite verifier、website build、package build、twine check、diff check | 本地 gate 通过 |

## 不能公开承诺

当前不要对外说：

- PyPI 已可安装。
- Public website 已经和最新 benchmark 对齐。
- sonar / physical camera / acoustic comms 已实现。
- cross-coupling damping 已启用。
- Bundled hover demo 已证明高质量 convergence。
- 项目已经 public launch-ready。
- 当前 `v0.0.x` tag train 代表 Python package 版本。

## 当前主要风险

| 风险 | 事实 | 影响 |
|---|---|---|
| Version identity 分裂 | Python package 内部 identity 已由测试锁住，但 `VERSION=0.0.2`、website package `0.0.1`、远端 tag `v0.0.x` 仍和 Python package `0.1.0-alpha` / `0.1.0a0` 不一致 | 发布语义不清，PyPI workflow 失败 |
| PyPI 不公开发布 | 当前决策是不 public to PyPI；workflow 只 build/check package artifact | 外部用户不能 pip install，这是当前策略 |
| CI runner 已按 ARC 收口 | workflow runs-on 均为 `oceanscale-arc` | 符合 runner reality |
| Live site stale | 线上仍显示旧 benchmark claim | 不能用于 public launch |
| Data boundary 未定 | `.npz` 是 supported artifact；`.pkl` tracked 但未 packaged | 需要决定删除或标记 legacy |
| Product demo 质量未定 | `oceanscale demo bluerov2-hover` 已可通过 bundled SB3 fallback 跑完；latest smoke position error 1.2721m | 不能承诺高质量 hover/convergence |

## 需要老板拍板

| Decision | 推荐 | 为什么 |
|---|---|---|
| Release identity | Python alpha 固定为 `0.1.0a0` / `0.1.0-alpha` | 与当前构建和 benchmark 一致 |
| Release lane | website 和 Python package 发布分离 | 避免 website tag 触发 PyPI mismatch |
| CI runner | 已改到 `oceanscale-arc` | 符合当前 runner reality |
| Data cleanup | 批准删除或标记 `vec_normalize.pkl` legacy | 当前未 packaged、未发现读取路径 |
| Public launch | 暂缓 | 当前用户指令和事实状态都指向 internal readiness |

## 建议下一步

1. 不做 public launch，先做 internal alpha hardening。
2. 建一个小范围 release/pipeline 修复分支。
3. 先修版本和 PyPI tag policy，再修 CI runner，再处理 data hygiene。
4. 重跑 final gate，并把 GitHub Actions / PyPI 状态补进 evidence index。
5. 只有在以上通过后，再进入 public website 和公开 README 对齐阶段。

## 一句话给外部前的内部判断

OceanScale 现在可以作为内部 alpha 和设计伙伴技术预览准备推进；不能把当前状态包装成公开发布完成。
