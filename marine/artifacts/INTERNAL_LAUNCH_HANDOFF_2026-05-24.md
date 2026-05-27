# OceanScale 内部 Launch Handoff

日期：2026-05-24
范围：内部 alpha readiness；不作为公开发布材料。

## 当前结论

OceanScale 的核心技术栈和产品内核已经具备内部 alpha 证据：RTX 5090 上的 Newton + Warp + Torch CUDA、BlueROV2/Fossen 6-DOF、Gymnasium/vectorized env、传感器 stub、训练入口、hover demo command、benchmark、package build、test gate 都已本地验证。

当前 owner 决策是 not release、CI use ARC、not public to PyPI。内部 alpha suite 可交接；public launch 仍不在当前范围内。

## 本轮已完成

| Area | 已完成结果 |
|---|---|
| Test/stack audit | 378 collected tests；full pytest 最新记录为 373 passed、5 xfailed、no warnings |
| Tech stack verification | oceanscale 0.1.0a0；Newton 1.2.0；Warp 1.13.0；Torch 2.11.0+cu128；RTX 5090 |
| Benchmark source | benchmarks/competitive/results.json 刷新为 2026-05-24 当前结果，n=4096 为 4,588,922 env-steps/s |
| Fidelity evidence | von Benzon comparison pass；position error 0.013%；attitude RMS 0.0025 deg |
| Package/data policy | wheel/sdist/installed wheel 均包含 zip/npz，不包含 source-only .pkl；tests/test_package_data.py 4 passed |
| Package-internal version guard | tests/test_version_identity.py 2 passed，锁住 pyproject normalized version、installed metadata、oceanscale.__version__ |
| Product surface | CLI/help/version、core/task env、sensor、VecEnv、8-step train smoke、hover demo command 均可执行 |
| Warning cleanup | Full pytest 不再出现 prior SB3 render_mode warning |
| Internal artifact suite | boss packet、readiness report、runbook、evidence index、manifest、decision matrix、hardening backlog、final gate、test matrix、package/product evidence 已建立 |

## 当前 source of truth

| 主题 | 文件 |
|---|---|
| 老板汇报 | artifacts/BOSS_PACKET_INTERNAL_2026-05-24.md |
| 总报告 | artifacts/LAUNCH_READINESS_INTERNAL_2026-05-24.md |
| Evidence index | artifacts/LAUNCH_EVIDENCE_INDEX_INTERNAL_2026-05-24.md |
| Final gate | artifacts/LAUNCH_FINAL_GATE_INTERNAL_2026-05-24.md |
| Machine-readable manifest | artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json |
| Test/tech-stack matrix | artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md |
| Package/data evidence | artifacts/PACKAGE_DATA_EVIDENCE_INTERNAL_2026-05-24.md |
| Product capability evidence | artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md |
| Hardening backlog | artifacts/INTERNAL_HARDENING_BACKLOG_2026-05-24.md |
| Execution runbook | artifacts/INTERNAL_LAUNCH_RUNBOOK_2026-05-24.md |
| Completion audit | artifacts/LAUNCH_COMPLETION_AUDIT_INTERNAL_2026-05-24.md |
| Launch suite verifier | scripts/verify_internal_launch_suite.py |
| External launch state | artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json |
| Package artifact state | artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json |

## 不应对外承诺

- PyPI 已可安装。
- Public website 已和最新 benchmark 对齐。
- sonar、physical camera model、acoustic comms 已实现。
- cross-coupling damping 已启用。
- Bundled hover demo 已证明高质量 convergence。
- 当前 v0.0.x tag train 代表 Python package 版本。
- 项目已经 public launch-ready。

## 剩余 blocker

| Blocker | 当前事实 | 需要什么 |
|---|---|---|
| Release identity | VERSION=0.0.2，Python package 0.1.0-alpha / 0.1.0a0，website 0.0.1，tag train v0.0.x | owner 决策 release/version policy |
| Public PyPI | 历史 Publish to PyPI 失败已保留在 external state；当前 workflow 只 build/check package artifact，不发布 PyPI/TestPyPI | 仅未来切回 public/PyPI lane 时再处理 |
| CI runner | 所有 workflow runs-on 均为 oceanscale-arc | 当前已收口 |
| Data boundary | vec_normalize.pkl tracked 但未 packaged，当前未发现读取路径 | owner 批准删除或标记 legacy |
| Public surface | live site 可访问但 claim stale；用户已明确 not public | 继续 deferred，除非用户明确切换 public lane |
| Demo quality | hover demo command exit 0，但 latest smoke position error 1.2721m | 决定是否生成并 bundle 更强 policy，或内部限定为 smoke demo |

## 收尾复跑命令

~~~bash
uv run ruff check oceanscale tests benchmarks/competitive/oceanscale_standard_bench.py
uv run mypy oceanscale/ --show-error-codes --no-error-summary
uv run pytest tests/ -q
uv run pytest tests/test_version_identity.py tests/test_package_data.py tests/test_package_artifact_state.py tests/test_cli.py tests/test_vec_env.py tests/test_launch_suite_verifier.py tests/test_external_launch_state.py -q
uv run python scripts/verify_internal_launch_suite.py
uv run python scripts/refresh_external_launch_state.py
uv run python scripts/refresh_package_artifact_state.py
uv run mypy scripts/refresh_external_launch_state.py scripts/refresh_package_artifact_state.py scripts/verify_internal_launch_suite.py --show-error-codes --no-error-summary
uv run python -m json.tool benchmarks/competitive/results.json
uv run python -m json.tool artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json
uv build --out-dir /tmp/oceanscale-final-handoff-build-20260524
uvx --from twine twine check /tmp/oceanscale-final-handoff-build-20260524/*
git diff --check
~~~

Website local build, still not public deploy:

~~~bash
cd website && pnpm build
~~~

## 建议交接顺序

1. 当前不做 release、不发布 PyPI/TestPyPI、不部署 public website。
2. 若 owner 批准，处理 vec_normalize.pkl 删除或 legacy 标记。
3. 未来只有用户明确切到 public launch 时，再处理 release identity、PyPI/TestPyPI、live website、公开 README、公开 benchmark copy。
