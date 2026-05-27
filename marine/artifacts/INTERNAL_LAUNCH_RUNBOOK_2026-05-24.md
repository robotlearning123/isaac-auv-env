# OceanScale 内部 Alpha Launch Runbook

日期：2026-05-24
范围：内部启动准备、老板汇报、设计伙伴技术预览。不作为公开发布材料。

## 0. 当前原则

用户已明确：not public。

因此本 runbook 的目标是把内部证据链、复现步骤、决策项和下一步执行顺序收口。除非 owner 明确批准，不做以下动作：

- 不部署或更新 live website。
- 不修改 public marketing copy。
- 不修改 GitHub Actions / CI workflow。
- 不删除 tracked data 文件。
- 不发布 PyPI / TestPyPI。

## 1. 当前 launch 分类

| Area | Status | 内部判断 |
|---|---|---|
| Core simulation product | GREEN | ROVEnv、BatchedVecEnv、BlueROV2、Fossen 6-DOF、Newton/Warp/Torch CUDA 本地可验证 |
| Benchmark evidence | GREEN | 标准 benchmark JSON 已生成，RTX 5090 上有 n=1 到 n=4096 结果 |
| Fidelity evidence | GREEN | von Benzon comparison pass |
| Local test gate | GREEN | ruff、mypy、pytest、website build、package build、twine check 通过 |
| Data boundary | YELLOW | `.npz` 为 supported artifact；`.pkl` tracked 但未 packaged，需要 owner 决策 |
| Version identity | RED | `VERSION`、Python package、website package、git tag train 不一致 |
| PyPI pipeline | DEFERRED | 当前决策是不 public to PyPI；workflow 只 build/check package |
| CI runner policy | GREEN | 所有 workflow runs-on 均为 oceanscale-arc |
| Live public site | RED for launch | 可访问但 claim stale；当前阶段不处理，因为 not public |

一句话：内部 alpha 技术验证可用；public launch 不 ready。

## 2. Artifact suite

| Artifact | 用途 | 使用场景 |
|---|---|---|
| `artifacts/BOSS_PACKET_INTERNAL_2026-05-24.md` | 老板一页纸 | 直接用于管理层同步和决策会开场 |
| `artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json` | machine-readable readiness manifest | 给后续 agent / 自动检查读取当前状态 |
| `artifacts/INTERNAL_HARDENING_BACKLOG_2026-05-24.md` | 内部 hardening backlog | 把 release/CI/PyPI/data/public-deferred 拆成可执行 tracks |
| `artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md` | test/tech-stack coverage matrix | 把 378 个测试按 launch 风险领域映射 |
| `artifacts/PACKAGE_DATA_EVIDENCE_INTERNAL_2026-05-24.md` | package/data artifact evidence | 验证 wheel/sdist/installed wheel 的 data 边界 |
| `artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md` | product/API capability evidence | 验证 CLI、env、task、sensor、VecEnv、train/demo surface |
| `artifacts/LAUNCH_READINESS_INTERNAL_2026-05-24.md` | 老板/团队中文总报告 | 会议前发给老板，说明当前状态和风险 |
| `artifacts/LAUNCH_FINAL_GATE_INTERNAL_2026-05-24.md` | gate checklist | 判断哪些本地/外部 gate 已通过或失败 |
| `artifacts/LAUNCH_EVIDENCE_INDEX_INTERNAL_2026-05-24.md` | 证据索引 | 查 product/data/benchmark/pipeline/results 的 source of truth |
| `artifacts/RELEASE_DECISION_MATRIX_INTERNAL_2026-05-24.md` | owner 决策表 | release identity、CI、PyPI、data cleanup 的取舍 |
| `artifacts/INTERNAL_LAUNCH_RUNBOOK_2026-05-24.md` | 执行 runbook | 本文件，指导后续内部启动准备 |
| `artifacts/INTERNAL_LAUNCH_HANDOFF_2026-05-24.md` | 最终交接入口 | 汇总已完成、blocker、审批项和复跑命令 |
| `artifacts/LAUNCH_COMPLETION_AUDIT_INTERNAL_2026-05-24.md` | completion audit | 按原始目标逐项证明哪些完成、哪些仍 blocked/deferred |
| `scripts/verify_internal_launch_suite.py` | suite verifier | 可执行校验 manifest、artifact 引用、benchmark truth、审批边界和 not-public 状态 |
| `artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json` | external state artifact | 只读记录 GitHub runs、PyPI 查询和 workflow runner scan |
| `artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json` | package artifact state | 记录 wheel/sdist/twine/no-deps install smoke 和 data artifact 边界 |

## 3. Internal demo path

推荐内部 demo 只展示当前已验证能力：

1. Python package import / CLI 存在。
2. BlueROV2 / ROVEnv 基础 reset-step loop。
3. BatchedVecEnv throughput benchmark。
4. Fidelity validation result。
5. Current evidence package and final gate status。

不建议在内部 demo 中承诺：

- sonar model。
- physical camera model。
- acoustic comms。
- cross-coupling damping 已启用。
- PyPI 可安装。
- public website 数据已对齐。
- full public launch readiness。

## 4. Reproduction commands

在 repo root 执行：

```bash
uv run ruff check oceanscale tests benchmarks/competitive/oceanscale_standard_bench.py
uv run mypy oceanscale/ --show-error-codes --no-error-summary
uv run pytest tests/ -q
uv run pytest tests/test_version_identity.py tests/test_package_data.py tests/test_package_artifact_state.py tests/test_launch_suite_verifier.py tests/test_external_launch_state.py -q
uv run python scripts/verify_internal_launch_suite.py
uv run python scripts/refresh_external_launch_state.py
uv run python scripts/refresh_package_artifact_state.py
uv run mypy scripts/refresh_external_launch_state.py scripts/refresh_package_artifact_state.py scripts/verify_internal_launch_suite.py --show-error-codes --no-error-summary
uv run python benchmarks/competitive/oceanscale_standard_bench.py
uv run python -m json.tool benchmarks/competitive/results.json
uv build --out-dir /tmp/oceanscale-build-finalgate
uvx --from twine twine check /tmp/oceanscale-build-finalgate/*
git diff --check
```

Website build 需在 `website/` 下执行：

```bash
pnpm build
```

External checks：

```bash
uvx --from pip pip index versions oceanscale
gh run list --workflow "Publish to PyPI" --limit 5
gh run list --workflow "Release Deploy" --limit 5
```

注意：external checks 只用于读状态，不代表批准发布。

## 5. Boss packet talking points

建议老板汇报按这个顺序讲：

1. 核心结论：技术核心可验证，public launch pipeline 未 ready。
2. 已完成：RTX 5090 benchmark、fidelity、full local tests、package build、内部证据包。
3. 当前不公开的原因：版本体系、PyPI、CI runner、live site claim stale、data boundary。
4. 需要老板决策：是否删除或标记 legacy `.pkl`；release/PyPI/public website 仅未来切回 public lane 时再议。
5. 下一步：做一个小而明确的 internal alpha hardening PR/branch，而不是直接 public launch。

## 6. Owner decision checklist

以下事项需要明确批准后才能执行：

| Decision | 推荐选择 | 原因 |
|---|---|---|
| Python package identity | `0.1.0a0` / `0.1.0-alpha` | 与当前 pyproject、installed metadata、benchmark artifact 一致 |
| Website/Python release split | 分离 | 避免 `v0.0.x` website tag 继续触发 PyPI mismatch |
| CI runner | `oceanscale-arc` | 符合当前 ARC runner policy；`ship-arc` 是 swarm-test 专用 |
| Data cleanup | 删除 `oceanscale/data/vec_normalize.pkl` | 当前未 packaged、未发现读取路径，`.npz` 是 supported artifact |
| Public site alignment | 暂缓 | 用户已明确 not public |

## 7. Next execution sequence after approval

如果 owner 批准 release/CI/data 三类动作，建议分成小 PR 或小提交顺序执行：

1. Release identity PR
   - 明确 Python package release tag pattern。
   - 防止 website tag 触发 PyPI publish。
   - 更新内部 release notes，不先改 public marketing。

2. CI runner
   - 已替换为 `oceanscale-arc`。
   - 后续只需在远端触发后记录结果。

3. Data hygiene PR
   - 删除或标记 legacy `.pkl`。
   - 确认 package-data 只包含 supported artifacts。
   - 重跑 package build / twine check。

4. Final internal gate
   - 重跑第 4 节所有本地命令。
   - 记录 GitHub Actions 状态。
   - 记录 PyPI/TestPyPI 状态。

5. Public readiness lane
   - 仅在用户明确切换到 public launch 后执行。
   - 对齐 live site、公开 README、公开 benchmark 文案和 deploy 状态。

## 8. Completion definition

内部 alpha readiness 完成需要满足：

- Evidence index 覆盖 product、data、benchmark、pipeline、results。
- Final gate 当前命令可复现并记录结果。
- Boss packet 能清楚区分已验证事实、风险、待决策项。
- Release/PyPI/CI/data 的 owner 决策已记录。
- 所有 public-facing stale claims either 明确暂缓或已在 public lane 中修复。

Public launch readiness 另算；当前不能用内部 alpha readiness 代替 public launch。
