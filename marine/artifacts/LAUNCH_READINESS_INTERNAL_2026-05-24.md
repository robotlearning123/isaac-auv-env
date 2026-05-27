# OceanScale 内部 Launch Readiness 报告

日期：2026-05-24
范围：内部汇报，不作为公开发布材料。当前目标是确认 OceanScale 是否具备面向设计伙伴、投资人沟通、团队执行的 alpha launch 证据链。

## 结论

OceanScale 的核心仿真能力已经有真实本地证据：Newton + Warp + Torch CUDA 在 RTX 5090 上可以运行 BlueROV2 / Fossen 6-DOF 动力学、vectorized Gymnasium 环境、RL 训练入口、传感器 stub、benchmark 和 package build。当前 owner 决策是 not release、CI use ARC、not public to PyPI；内部 alpha readiness suite 可交接，public launch 另算。

一句话：技术核心和内部证据链已能展示；release/PyPI/public website 按当前决策暂缓。

## 当前已验证栈

| 层 | 当前证据 | 状态 |
|---|---|---|
| Python package | oceanscale 0.1.0a0 | 已构建、twine check 通过 |
| Physics | Newton 1.2.0 | 本地环境可用 |
| GPU kernels | Warp 1.13.0 | RTX 5090 上可用 |
| Torch | 2.11.0+cu128 | CUDA runtime 12.8 |
| GPU | NVIDIA GeForce RTX 5090 | driver 580.95.05，NVIDIA-SMI CUDA 13.0 |
| RL API | Gymnasium 1.2.3 | ROVEnv / BatchedVecEnv 可用 |
| RL libs | Stable-Baselines3 2.8.0，skrl 2.1.0 | 本地安装可用 |
| Website | Astro 6.3.3，Tailwind 4.3.0 | pnpm build 通过 |

## 当前 benchmark 证据

来源：benchmarks/competitive/results.json，命令为 uv run python benchmarks/competitive/oceanscale_standard_bench.py。

| n_envs | env-steps/s | wall time | steps |
|---:|---:|---:|---:|
| 1 | 1,326 | 0.1508 s | 200 |
| 64 | 85,824 | 0.1491 s | 200 |
| 256 | 303,206 | 0.1689 s | 200 |
| 1024 | 1,272,106 | 0.1610 s | 200 |
| 4096 | 4,588,922 | 0.1785 s | 200 |

同一次 benchmark 还验证了：

- physics terms：added mass、Coriolis/centripetal、linear/quadratic damping、restoring forces、thruster low-pass/deadband、ocean-current coupling。
- cross-coupling damping：当前 v0.1 关闭，因为系数没有独立识别。
- sensors：IMU、DVL、depth/pressure stubs 存在。
- 不存在：sonar、真实 camera model、acoustic comms。
- fidelity：von Benzon comparison 通过，position error 0.013%，attitude RMS 0.0025 deg。

## 已通过的本地 gate

上一轮完整 gate 已验证：

- ruff check。
- mypy oceanscale/。
- full pytest：373 passed，5 xfailed，无 warnings。
- website pnpm build。
- uv build。
- twine check。
- wheel full dependency install smoke。

本轮新增验证：

- website build 通过。
- benchmark JSON 可解析。
- benchmark script ruff 通过。
- git diff --check 通过。
- package build + twine check 通过。

## 已完成的 launch evidence 整理

- 修正 benchmark generator：版本号改为读取 installed package metadata，不再硬编码 0.0.2。
- 修正 feature truth：IMU / DVL / pressure stubs 为 true；sonar / camera / acoustic comms 为 false。
- 修正 physics truth：cross-coupling damping 为 false。
- 更新 benchmarks/competitive/results.json 为 2026-05-24 当前结果。
- 更新 benchmarks/RESULTS.md、benchmarks/README.md、benchmarks/COMPETITIVE_MATRIX.md，将标准化 benchmark 和历史 PyBullet CPU baseline 分开。
- 官网内容中已替换过期主 benchmark 数字：从旧的 17,427 主叙事改为 85,824；17,427 仅保留为 PyBullet CPU baseline sweep。
- 移除/修正过期 claim：11,435 FPS、90M env-steps/s、8192 envs 等。
- 新增 `artifacts/BOSS_PACKET_INTERNAL_2026-05-24.md`：老板一页纸，直接用于内部汇报开场。
- 新增 `artifacts/INTERNAL_LAUNCH_RUNBOOK_2026-05-24.md`：内部 alpha 启动 runbook，包含复现命令、demo 路径和审批边界。
- 新增 `artifacts/INTERNAL_HARDENING_BACKLOG_2026-05-24.md`：把 release identity、PyPI、CI runner、data boundary、public-deferred 拆成执行 tracks。
- 新增 `artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json`：machine-readable readiness manifest，已通过 JSON 解析校验。
- 刷新外部 pipeline 证据：Release Deploy 历史最近 v0.0.21 到 v0.0.25 全部成功；当前 release workflow 已改为 manual-only。历史 Publish to PyPI 失败保留为记录；当前 workflow 只 build/check package artifact，不发布 PyPI/TestPyPI。
- 新增 `artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md`：all-test 和 tech-stack 覆盖矩阵；当前 `uv run pytest tests/ --collect-only -q` 为 378 collected，`uv run pytest tests/ -q` 复跑结果为 373 passed，5 xfailed，无 warnings，23.11s。
- 新增 `tests/test_package_data.py`：自动锁住 package-data 只声明 supported zip/npz，不声明 pickle 或 wildcard；focused result 为 4 passed。
- 新增 `tests/test_version_identity.py`：自动锁住 pyproject normalized version、installed metadata、`oceanscale.__version__` 一致；focused result 为 2 passed。release tag / `VERSION` / website version mismatch 仍是 blocker。
- 新增 `scripts/verify_internal_launch_suite.py` 和 `tests/test_launch_suite_verifier.py`：自动校验 manifest source-of-truth、benchmark truth、not-public 边界、pipeline blocker、审批边界和 completion-audit scope；focused result 为 2 passed。
- 新增 `scripts/refresh_external_launch_state.py`、`tests/test_external_launch_state.py` 和 `artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json`：只读刷新 GitHub runs、PyPI 查询、release/PyPI deferral policy 和 workflow runner scan；focused result 为 4 passed。
- 新增 `scripts/refresh_package_artifact_state.py`、`tests/test_package_artifact_state.py` 和 `artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json`：刷新 wheel/sdist/twine/no-deps install smoke 和 data artifact 边界；focused result 为 3 passed。
- 新增 `artifacts/PACKAGE_DATA_EVIDENCE_INTERNAL_2026-05-24.md`：wheel/sdist/installed wheel data artifact 证据；当前 package artifact 包含 supported zip/npz，不包含 source-only `.pkl`。
- 新增 `artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md`：CLI、core/task env、sensor、VecEnv 和 8-step training smoke 通过；hover demo 已通过 bundled SB3 fallback 修复为 exit 0，但 latest smoke final position error 为 1.2721m，不应宣称高质量 hover convergence。

## 仍未完成的关键 blocker

1. Version identity 不一致
   - VERSION = 0.0.2
   - pyproject.toml = 0.1.0-alpha，规范化为 0.1.0a0
   - oceanscale/__init__.py = 0.1.0a0
   - website/package.json = 0.0.1
   - 最近远端 tag 是 v0.0.25

2. Public PyPI 暂缓
   - 当前决策是不 public to PyPI。
   - workflow 已改成 package build/check only。
   - 历史失败原因仍记录在 external state artifact 中。

3. CI runner 配置已按项目规则收口
   - .github/workflows 当前 runs-on 均为 oceanscale-arc。
   - external launch state artifact 记录 workflow_runner_policy=green。

4. Data boundary 未最终确认
   - oceanscale/data/vec_normalize.pkl 仍被 git tracked。
   - pyproject 只打包 data/*.zip 和 data/*.npz。
   - vec_normalize.npz 是当前 package-supported artifact。
   - 删除 tracked data 文件需要确认，当前未删。

5. 内部文档仍需统一
   - root README、oceanscale/README.md、docs/getting-started.md、docs/v0.1_brief.md 仍有部分旧叙事。
   - 用户已明确 not public；后续应优先维护内部 readiness artifact 和老板汇报，不急于改官网/公开 README。

6. Product demo quality 未最终确认
   - `oceanscale demo bluerov2-hover --device cuda` 现在可通过 bundled SB3 checkpoint 运行并 exit 0。
   - latest smoke：1000 steps，total_reward=271.56，position error 1.2721m，depth error 1.1282m。
   - 当前可说 demo command 可执行；不应说 bundled policy 已达到高质量 hover/convergence。
   - 若要公开 demo，需要决定是否生成并 bundle 更强 skrl `.pt` policy，或明确标注为 train-first / smoke demo。

## 推荐下一步

优先级 1：经 owner 批准后处理 vec_normalize.pkl。
- 推荐删除 tracked .pkl，保留 .npz 作为 supported artifact。
- 删除前再次确认没有代码路径读取 .pkl。

优先级 2：远端 CI 跑完后补充 ARC run ids。

优先级 3：如果未来切回 public/release lane，再确认 release identity、PyPI/TestPyPI 和 live website。

优先级 4：做最终 launch gate。
- ruff、mypy、pytest full、website build、uv build、twine check、wheel install smoke。
- GitHub workflow 状态。
- live website 状态。
- PyPI/TestPyPI 状态。

## 给老板的一句话版本

OceanScale 的核心 alpha 能力已经跑通，并且有 RTX 5090 benchmark、fidelity validation、测试、构建和 wheel smoke 证据支撑；现在最大风险不是仿真核心，而是 release/version/CI/PyPI/data hygiene。下一步要把发布链路和内部证据包收口，再决定是否进入设计伙伴试用阶段。
