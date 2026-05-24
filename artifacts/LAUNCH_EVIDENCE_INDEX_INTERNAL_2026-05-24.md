# OceanScale 内部 Launch Evidence Index

日期：2026-05-24
范围：内部 readiness 索引，不作为公开发布材料。

## 用法

这个文件把 launch 所需的 data、benchmark、pipeline、product、results 证据统一索引到当前仓库文件、验证命令和状态。它不是 marketing copy；任何对外 claim 都应先回到这里找 source of truth。

配套内部材料：

- `artifacts/BOSS_PACKET_INTERNAL_2026-05-24.md`：老板一页纸。
- `artifacts/INTERNAL_LAUNCH_RUNBOOK_2026-05-24.md`：内部执行 runbook。
- `artifacts/INTERNAL_HARDENING_BACKLOG_2026-05-24.md`：release/CI/PyPI/data hardening backlog。
- `artifacts/LAUNCH_READINESS_MANIFEST_INTERNAL_2026-05-24.json`：machine-readable readiness manifest。
- `artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md`：all-test 和 tech-stack 覆盖矩阵。
- `artifacts/PACKAGE_DATA_EVIDENCE_INTERNAL_2026-05-24.md`：wheel/sdist/installed wheel data artifact 证据。
- `artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md`：CLI、env、task、sensor、VecEnv、train/demo product capability 证据。
- `artifacts/INTERNAL_LAUNCH_HANDOFF_2026-05-24.md`：最终内部交接入口，汇总已完成、blocker、审批项和复跑命令。
- `artifacts/LAUNCH_COMPLETION_AUDIT_INTERNAL_2026-05-24.md`：按原始目标逐项审计 test、tech stack、data、benchmark、pipeline、product、results 和 not-public 边界。
- `artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json`：只读刷新生成的外部 pipeline 状态，包括 GitHub runs、PyPI 查询和 workflow runner scan。
- `artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json`：刷新生成的 package artifact 状态，包括 wheel/sdist/twine/no-deps install smoke 和 data 边界。

## 1. Product / 产品能力

| 项 | Source of truth | 当前状态 | 证据强度 |
|---|---|---|---|
| Python CLI | `pyproject.toml` `[project.scripts]` -> `oceanscale.cli:main`，`oceanscale/cli.py` | 存在，wheel build 包含 entry point | 强 |
| Core env | `oceanscale/rov_env.py` `ROVEnv` | Gymnasium-compatible env 存在 | 强 |
| Vectorized env | `oceanscale/vec_env.py` `BatchedVecEnv` | 支持 batched stepping | 强 |
| BlueROV2 model | `oceanscale/vehicles/bluerov2.py` | 当前主 vehicle model | 强 |
| Tasks | `oceanscale/envs/station_keeping_env.py`，`docking_env.py`，`waypoint_env.py` | 多 task 文件存在；launch 主证据仍集中在 hover / BlueROV2 | 中 |
| Sensors | `oceanscale/sensors/{imu,dvl,pressure}.py` | IMU / DVL / pressure stubs 存在 | 强 |
| Missing sensors | benchmark feature matrix | sonar、physical camera model、acoustic comms 不存在 | 强 |
| Rendering | `oceanscale/rendering.py` | headless MP4 / matplotlib-based rendering | 中 |
| Product smoke | `artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md` | CLI help/version、core/task env、sensor、VecEnv、train smoke 通过；hover demo 通过 bundled SB3 fallback 可执行，但 quality 有 caveat | 强 |

## 2. Data / 数据与模型 artifact

| 文件 | Git tracked | Packaged | 当前用途 | 状态 |
|---|---:|---:|---|---|
| `oceanscale/data/bluerov2_station_keep_final.zip` | yes | yes | bundled checkpoint | 保留 |
| `oceanscale/data/vec_normalize.npz` | yes | yes | supported VecNormalize artifact | 保留 |
| `oceanscale/data/vec_normalize.pkl` | yes | no | 未发现当前读取路径 | 待 owner 批准后删除或标记 legacy |
| `oceanscale/data/__init__.py` | yes | package module | package marker | 保留 |

当前证据：`pyproject.toml` package-data 只包含 `data/*.zip` 和 `data/*.npz`。`rg` 只发现 `.npz` 的当前引用，没有发现 `.pkl` 读取路径。

Package artifact 证据：`uv build --out-dir /tmp/oceanscale-package-evidence-20260524-1229` 生成的 wheel/sdist 均包含 `bluerov2_station_keep_final.zip` 和 `vec_normalize.npz`，不包含 `vec_normalize.pkl`。从 `/tmp` 安装 wheel 后，`importlib.resources.files('oceanscale.data')` 也只看到 supported zip/npz，`has_pkl=False`。

Automated policy 证据：`tests/test_package_data.py` 已加入 full pytest，focused result 为 `4 passed`，覆盖 supported zip/npz 存在、package-data 声明 zip/npz、未声明 pickle 或 wildcard。

Automated artifact refresh：`scripts/refresh_package_artifact_state.py` 生成 `artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json`。当前状态为 package artifact GREEN、source data hygiene YELLOW。

## 3. Benchmark / 性能与 fidelity

| Artifact | 命令 | 当前结果 | 状态 |
|---|---|---|---|
| `benchmarks/competitive/results.json` | `uv run python benchmarks/competitive/oceanscale_standard_bench.py` | version 0.1.0a0，timestamp 2026-05-24T11:54:13-0400 | 主 benchmark source |
| n=1 throughput | same | 1,326 env-steps/s | PASS |
| n=64 throughput | same | 85,824 env-steps/s | PASS |
| n=256 throughput | same | 303,206 env-steps/s | PASS |
| n=1024 throughput | same | 1,272,106 env-steps/s | PASS |
| n=4096 throughput | same | 4,588,922 env-steps/s | PASS |
| Fidelity | same | von Benzon PASS，0.013% position error，0.0025 deg attitude RMS | PASS |
| Historical CPU baseline | `benchmarks/RESULTS.md` / `benchmarks/oceanscale_vs_bullet_results.json` | PyBullet sweep：17,427 env-steps/s at n=64，10.5x over PyBullet n=1 | 历史 baseline，不应混作主 benchmark |

## 4. Test / 本地质量 gate

| Gate | Command | Latest observed result | 状态 |
|---|---|---|---|
| Ruff | `uv run ruff check oceanscale tests benchmarks/competitive/oceanscale_standard_bench.py` | All checks passed | PASS |
| Mypy | `uv run mypy oceanscale/ --show-error-codes --no-error-summary` | exit 0 | PASS |
| Pytest full | `uv run pytest tests/ -q` | 373 passed，5 xfailed，no warnings，23.11s | PASS |
| Package data policy | `uv run pytest tests/test_package_data.py -q` | 4 passed | PASS |
| Package artifact state | `uv run python scripts/refresh_package_artifact_state.py`; `uv run pytest tests/test_package_artifact_state.py -q` | package artifact green；source data hygiene yellow；3 passed | PASS as artifact, YELLOW as source hygiene |
| Version identity policy | `uv run pytest tests/test_version_identity.py -q` | 2 passed | PASS for package-internal identity |
| Launch suite verifier | `uv run python scripts/verify_internal_launch_suite.py`; `uv run pytest tests/test_launch_suite_verifier.py -q` | 15 checks pass；2 tests passed | PASS for internal suite consistency |
| External launch state parser | `uv run pytest tests/test_external_launch_state.py -q` | 4 passed | PASS for read-only pipeline evidence parser |
| Launch suite scripts typecheck | `uv run mypy scripts/refresh_external_launch_state.py scripts/refresh_package_artifact_state.py scripts/verify_internal_launch_suite.py --show-error-codes --no-error-summary` | exit 0 | PASS |
| Website build | `cd website && pnpm build` | 2 pages built | PASS |
| Benchmark JSON | `uv run python -m json.tool benchmarks/competitive/results.json` | exit 0 | PASS |
| Package build | `uv build --out-dir /tmp/oceanscale-build-finalgate` | wheel + sdist built | PASS |
| Twine check | `twine check /tmp/oceanscale-build-finalgate/*` | wheel + sdist passed | PASS |
| Whitespace | `git diff --check` | exit 0 | PASS |

## 5. Pipeline / 发布和 CI

| Pipeline | Source | 当前外部状态 | 状态 |
|---|---|---|---|
| Release Deploy | `.github/workflows/release.yml` | manual-only；不再 tag 自动 release | DEFERRED / not release |
| Public PyPI publish | `.github/workflows/publish-pypi.yml` | 已改为 package build/check only，无 PyPI/TestPyPI publish action | DEFERRED / not public |
| PyPI availability | `uvx --from pip pip index versions oceanscale` | no matching distribution；当前策略是不 public to PyPI | DEFERRED |
| CI runner policy | `.github/workflows/*.yml` | 所有 `runs-on` 均为 `oceanscale-arc` | PASS |
| ARC reality | repo AGENTS / runner policy | 应使用 `oceanscale-arc` 或其他明确 scale set | 未实施 |
| External launch state artifact | `uv run python scripts/refresh_external_launch_state.py` | release/public PyPI deferred；workflow runner policy green | PASS |

## 6. Live surface / 线上状态

| Surface | URL / command | 当前状态 |
|---|---|---|
| EN live site | `https://oceanscale-web.pages.dev/` | 可访问，但仍显示旧 benchmark claim：17,427 和 8192 |
| ZH live site | `https://oceanscale-web.pages.dev/zh/` | 可访问，但仍显示旧 benchmark claim：17,427 和 8192 |
| Local website build | `cd website && pnpm build` | 本地 build pass |

用户已明确 `not public`，因此当前阶段不继续动 public surface。若进入公开发布阶段，live site 必须重新对齐当前 benchmark source of truth。

## 7. Version identity

| Source | Current value | 状态 |
|---|---|---|
| `VERSION` | 0.0.2 | 与 package 不一致 |
| `pyproject.toml` | 0.1.0-alpha | package source |
| installed / normalized | 0.1.0a0 | 当前 build artifact |
| `oceanscale/__init__.py` | 0.1.0a0 | 与 package normalized 一致 |
| `website/package.json` | 0.0.1 | 独立 website package version |
| Git tag train | v0.0.x，latest v0.0.25 | 与 Python package version 不一致 |

Automated guard：`tests/test_version_identity.py` 已确认 pyproject normalized version、installed metadata 和 `oceanscale.__version__` 一致。这个 guard 只覆盖 Python package 内部 identity；不覆盖 `VERSION`、website package version 或远端 release tag policy。

## 8. Current readiness classification

| Area | Status | Reason |
|---|---|---|
| Local product core | GREEN | tests/build/package/benchmark pass |
| Benchmark evidence | GREEN | standardized JSON + results docs exist |
| Data hygiene | YELLOW | `.pkl` tracked but not packaged |
| Release identity | DEFERRED | not release; package-internal identity guard is green |
| PyPI | DEFERRED | not public to PyPI; package build/check remains green |
| CI runner policy | GREEN | all workflow runners use oceanscale-arc |
| Live public site | RED for claim alignment | reachable but stale |

当前可用于内部老板汇报和设计伙伴技术预览准备；不建议标记 public launch-ready。
