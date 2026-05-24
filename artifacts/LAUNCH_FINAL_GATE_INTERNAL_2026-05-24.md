# OceanScale 内部 Final Gate Checklist

日期：2026-05-24
范围：内部 launch readiness，不作为公开发布材料。

## Gate 结论

当前本地代码、构建、package-data policy、package-internal version identity gate 通过；外部 release/PyPI/live-site gate 未通过。因此状态是：核心产品本地可验证，外部 launch pipeline 未 ready。

## 本地 gate 结果

| Gate | 命令 | 结果 |
|---|---|---|
| Ruff | `uv run ruff check oceanscale tests benchmarks/competitive/oceanscale_standard_bench.py` | PASS |
| Mypy | `uv run mypy oceanscale/ --show-error-codes --no-error-summary` | PASS |
| Pytest full | `uv run pytest tests/ -q` | PASS：373 passed，5 xfailed，no warnings，23.11s |
| Package data policy | `uv run pytest tests/test_package_data.py -q` | PASS：4 passed |
| Version identity policy | `uv run pytest tests/test_version_identity.py -q` | PASS：2 passed，pyproject normalized version matches installed metadata and `oceanscale.__version__` |
| Launch suite verifier | `uv run python scripts/verify_internal_launch_suite.py` + `uv run pytest tests/test_launch_suite_verifier.py -q` | PASS：15 verifier checks pass；2 tests passed |
| External launch state refresh | `uv run python scripts/refresh_external_launch_state.py` + `uv run pytest tests/test_external_launch_state.py -q` | PASS：artifact generated；4 tests passed；release/public PyPI deferred；workflow runner policy green |
| Package artifact state refresh | `uv run python scripts/refresh_package_artifact_state.py` + `uv run pytest tests/test_package_artifact_state.py -q` | PASS：package artifact green；source data hygiene yellow；3 tests passed |
| Launch suite scripts typecheck | `uv run mypy scripts/refresh_external_launch_state.py scripts/refresh_package_artifact_state.py scripts/verify_internal_launch_suite.py --show-error-codes --no-error-summary` | PASS |
| Website build | `cd website && pnpm build` | PASS：2 pages built |
| Benchmark JSON | `uv run python -m json.tool benchmarks/competitive/results.json` | PASS |
| Package build + twine | `uv build --out-dir /tmp/oceanscale-build-finalgate` + `twine check` | PASS：wheel + sdist passed |
| Diff whitespace | `git diff --check` | PASS |
| Package data artifact | `uv build --out-dir /tmp/oceanscale-package-evidence-20260524-1229` + wheel/sdist inspection + installed wheel smoke from `/tmp` | PASS：zip/npz included，pkl excluded |
| Product/API smoke | CLI version/help + CUDA n_envs=4 env/task/sensor/VecEnv smoke + 8-step train smoke + hover demo | PASS executable；hover demo uses bundled SB3 checkpoint, quality caveat remains |
| Warning/demo fix package build | `uv build --out-dir /tmp/oceanscale-warning-fix-build-20260524` + `twine check` | PASS：wheel + sdist passed after CLI/VecEnv fixes |

## 当前 benchmark source of truth

文件：`benchmarks/competitive/results.json`

| n_envs | env-steps/s |
|---:|---:|
| 1 | 1,326 |
| 64 | 85,824 |
| 256 | 303,206 |
| 1024 | 1,272,106 |
| 4096 | 4,588,922 |

Fidelity gate：PASS，von Benzon comparison position error 0.013%，attitude RMS 0.0025 deg。

## 外部 gate 结果

| Gate | 当前证据 | 状态 |
|---|---|---|
| GitHub Release Deploy | 最近 v0.0.21 到 v0.0.25 历史运行成功；当前 workflow 已改为 manual-only，不再 tag 自动 release | DEFERRED / NOT RELEASE |
| Public PyPI publish | 原 Publish to PyPI 历史运行失败；当前 workflow 已改为 package build/check only，无 PyPI/TestPyPI publish action | DEFERRED / NOT PUBLIC |
| PyPI package availability | `pip index versions oceanscale` 返回 no matching distribution；当前策略是不 public to PyPI | DEFERRED |
| Live EN/ZH website reachability | `https://oceanscale-web.pages.dev/` 和 `/zh/` 可访问 | PASS |
| Live website claim alignment | live site 仍显示旧 `17,427` 和 `8192` claim | FAIL / STALE |
| CI runner policy | 所有 workflow `runs-on` 均为 `oceanscale-arc` | PASS |

## Data/package boundary

当前 `oceanscale/data`：

- `bluerov2_station_keep_final.zip`：tracked，packaged。
- `vec_normalize.npz`：tracked，packaged，当前 supported artifact。
- `vec_normalize.pkl`：tracked，不在 package-data 里。

代码/测试/benchmark 中当前只找到 `vec_normalize.npz` 引用，没有发现 `.pkl` 读取路径。删除 `.pkl` 仍需 owner 确认，因为项目规则要求删除数据文件前确认。

Package artifact 复核：当前 wheel 和 sdist 不包含 `vec_normalize.pkl`。从临时 venv 安装 wheel 后，installed `oceanscale.data` 也只包含 `__init__.py`、`bluerov2_station_keep_final.zip`、`vec_normalize.npz`，`has_pkl=False`。

## Release identity blocker

当前版本源不一致：

- `VERSION=0.0.2`
- `pyproject.toml version = 0.1.0-alpha`，规范化为 `0.1.0a0`
- `oceanscale/__init__.py = 0.1.0a0`
- `website/package.json = 0.0.1`
- 远端 tag train 当前是 `v0.0.x`，最新为 `v0.0.25`

历史后果：原 `Publish to PyPI` workflow 用 tag `0.0.x` 对比 pyproject `0.1.0-alpha`，因此失败。当前按 owner 决策改为不 public to PyPI，workflow 只 build/check artifact。

最新外部证据：run `26365095288` 在 `Verify tag matches package version` 步骤失败，错误为 `Tag 0.0.25 does not match pyproject.toml version 0.1.0-alpha`。

本地新增 guard：`tests/test_version_identity.py` 已确认 pyproject normalized version、installed package metadata 和 `oceanscale.__version__` 三者一致。它不解决 `VERSION`、website package version 或远端 tag train 的 release-level mismatch。

## Recommended next actions

1. 经批准后删除或标记 `oceanscale/data/vec_normalize.pkl`，保留 `.npz` 为 supported artifact。
2. 若未来切回 release/public lane，再重新决策 release identity、PyPI/TestPyPI、public website deploy。
3. 当前继续保持 not release、not public to PyPI、not public website deploy。

## 当前是否可 launch

可标记为内部 alpha readiness suite complete；不应标记为 public launch-ready。

可用于内部老板汇报和设计伙伴技术预览准备；不建议作为公开发布或 PyPI launch。原因不是核心仿真能力，而是外部发布链路和 live surface 仍未对齐。
