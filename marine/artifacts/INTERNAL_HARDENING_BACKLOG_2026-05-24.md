# OceanScale Internal Hardening Backlog

日期：2026-05-24
范围：内部 alpha readiness；不作为公开发布材料。

## Current Evidence Refresh

本文件基于当前 worktree 和实时外部状态刷新，不依赖旧结论。

| Area | Current evidence | Status |
|---|---|---|
| Worktree | `git status --short` 显示大量本地修改和内部 artifacts 未追踪 | DIRTY / expected during readiness work |
| Version sources | `VERSION=0.0.2`；`pyproject.toml=0.1.0-alpha`；`oceanscale/__init__.py=0.1.0a0`；`website/package.json=0.0.1` | RED |
| Package-internal version guard | `tests/test_version_identity.py` confirms pyproject normalized version, installed metadata, and `oceanscale.__version__` match | GREEN for package internals |
| Workflows | 所有 workflow `runs-on` 均为 `oceanscale-arc` | GREEN |
| Release Deploy | 最近 `v0.0.21` 到 `v0.0.25` GitHub Release Deploy 全部 success | GREEN for website deploy path |
| Public PyPI | 当前 workflow 只 build/check package artifact，不发布 PyPI/TestPyPI | DEFERRED / not public |
| Latest PyPI failure | run `26365095288`：`Tag 0.0.25 does not match pyproject.toml version 0.1.0-alpha` | ROOT CAUSE CONFIRMED |
| PyPI availability | `uvx --from pip pip index versions oceanscale` 返回 no matching distribution | DEFERRED / not public |
| Data package boundary | package-data 包含 `data/*.zip` 和 `data/*.npz`；`vec_normalize.pkl` tracked 但未 packaged | YELLOW / approval required |
| Package artifact contents | wheel/sdist/installed wheel inspection confirms zip/npz included and `.pkl` excluded | GREEN for release artifact |
| Package data policy tests | `tests/test_package_data.py` guards zip/npz package-data and excludes pickle/wildcard packaging | GREEN |
| Launch suite verifier | `scripts/verify_internal_launch_suite.py` verifies manifest paths, benchmark truth, not-public status, pipeline policy, approvals, and completion audit scope | GREEN |
| External launch state artifact | `scripts/refresh_external_launch_state.py` writes `artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json` from read-only GitHub/PyPI/workflow checks | GREEN as evidence; public release/PyPI deferred |
| Package artifact state | `scripts/refresh_package_artifact_state.py` writes `artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json` from build/twine/wheel/sdist/install smoke | GREEN as package artifact; YELLOW as source data hygiene |
| Product demo surface | `oceanscale demo bluerov2-hover --device cuda` exits 0 via bundled SB3 fallback; latest smoke position error 1.2721m | YELLOW for demo quality |
| Benchmark source | `benchmarks/competitive/results.json` version `0.1.0a0`，timestamp `2026-05-24T11:54:13-0400` | GREEN |

## Track 0: No-Approval Internal Work

这些动作可继续做，因为只整理内部证据，不改变发布面：

1. 保持 evidence index、final gate、boss packet、runbook 和 manifest 同步。
2. 把每次 gate 命令、外部 run id、PyPI 查询结果记录到内部 artifact。
3. 为老板会准备一份明确的 go/no-go 表。
4. 维护 public-deferred 清单，避免把公开 launch 和内部 alpha readiness 混在一起。

Acceptance evidence：

- 内部 artifacts 有明确 source of truth、命令、状态和 blocker。
- `git diff --check -- artifacts/...` 通过。
- JSON manifest 可被 `python -m json.tool` 解析。

## Track 1: Release Identity

Problem：当前 Python package、website package、VERSION 和 git tag train 表达的是不同版本线。当前不 release、不 public to PyPI，因此这是 future-release decision，不是内部 alpha blocker。

推荐决策：

- Python package alpha identity 固定为 `0.1.0a0` / `0.1.0-alpha`。
- website release 和 Python package release 分离。
- 停止让 `v0.0.x` website tag 触发 PyPI 发布。

需要批准后修改的对象：

- `.github/workflows/publish-pypi.yml` tag trigger / guard。
- 可能需要 `VERSION` 语义说明或同步。
- 可能需要内部 release note，不先改公开 marketing。

Acceptance evidence：

- Python package release tag 与 `pyproject.toml` version 一致，或 workflow 对非 Python tag neutral skip。
- `uv run pytest tests/test_version_identity.py -q` stays green for pyproject metadata and `oceanscale.__version__` consistency。
- GitHub Publish to PyPI 不再因 website tag 失败。
- 新 tag policy 在 internal runbook 中有记录。

## Track 2: PyPI Pipeline

Problem：PyPI 上当前没有 `oceanscale` distribution；当前 owner 决策是不 public to PyPI。

当前决策：`.github/workflows/publish-pypi.yml` 只 build/check package artifact，不发布 PyPI/TestPyPI。未来如切回 public/PyPI lane，再使用 Python-package-specific tag，例如 `oceanscale-v0.1.0a0` 或 `python-v0.1.0a0`。

需要批准后修改的对象：

- `.github/workflows/publish-pypi.yml`。
- release process / tag instructions。

Acceptance evidence：

- TestPyPI publish 通过或明确跳过策略通过。
- PyPI publish 通过后，`pip index versions oceanscale` 能查到目标版本。
- `uvx --from pip pip install oceanscale==<version>` smoke 通过。

## Track 3: CI Runner Policy

Problem：repo 指令要求 ARC scale set，不允许 legacy `self-hosted`。

当前证据：

- 当前 `.github/workflows/*.yml` 均使用 `oceanscale-arc`。

推荐决策：本 repo 默认改为 `oceanscale-arc`，除非 owner 指定更精确 scale set。`ship-arc` 是 swarm-test 专用，不用于 OceanScale。

Acceptance evidence：

- workflow YAML 不再出现 `self-hosted` label 或 `[self-hosted, linux, x64]` array。
- 远端 workflow run 能在 ARC runner 上完成。
- Final gate artifact 记录 run ids 和 conclusion。

## Track 4: Data Boundary

Problem：`oceanscale/data/vec_normalize.pkl` tracked 但未 packaged；当前引用只指向 `.npz` supported artifact。

Package artifact evidence：wheel、sdist、installed wheel 均不包含 `vec_normalize.pkl`；当前风险是 source hygiene / ambiguity，不是 wheel packaging leak。

推荐决策：删除 `.pkl`，保留 `vec_normalize.npz`。

需要批准原因：repo 指令要求删除 data file 前确认。

Acceptance evidence：

- 删除或 legacy 标记策略获 owner 明确确认。
- `rg` 不存在 `.pkl` runtime read path。
- `uv run pytest tests/test_package_data.py -q` 通过，确保 supported zip/npz package-data policy 不回退。
- `uv build` + `twine check` 通过。
- wheel 内容只包含 supported data artifacts。

## Track 5: Product / Benchmark / Results Suite

当前状态：核心证据已经存在，但要继续保持 source of truth 单一。

Required invariants：

- `benchmarks/competitive/results.json` 是当前标准 benchmark source。
- 历史 PyBullet 17,427 只能作为 historical CPU baseline，不能作为主性能 claim。
- sensor truth 必须保持：IMU / DVL / depth-pressure stubs true；sonar / physical camera / acoustic comms false。
- physics truth 必须保持：cross-coupling damping false，除非后续有独立系数识别和测试。

Acceptance evidence：

- benchmark generator 和 results JSON 一致。
- benchmark docs 与 internal evidence 不冲突。
- final gate 记录 benchmark timestamp 和 machine/GPU context。

## Track 5A: Product Demo Surface

Problem：core env、task env、sensor、VecEnv、train smoke 和 hover demo command 均可执行；剩余问题是 bundled SB3 demo policy quality 不足以支撑 high-quality hover/convergence claim。

当前证据：

- CLI now falls back from missing `bluerov2_skrl_policy.pt` to packaged `bluerov2_station_keep_final.zip`。
- `uv run oceanscale demo bluerov2-hover --device cuda` exits 0 and completes 1000 steps。
- Latest smoke result: total_reward=271.56, position error 1.2721m, depth error 1.1282m。
- `oceanscale train bluerov2-hover --total 8 --n_envs 4 --device cuda` can save skrl policy/value files to `/tmp`。

Recommended decision：before any public demo claim, decide whether this SB3 fallback quality is acceptable for internal demos only or whether to generate and bundle a stronger skrl `.pt` policy。

Acceptance evidence：

- `uv run oceanscale demo bluerov2-hover --device cuda` exits 0 in a clean package/runtime context。
- Package-data evidence is refreshed if a new `.pt` artifact is bundled。
- CLI smoke test covers demo behavior。

## Track 6: Public Surface Deferred

当前用户指令是 `not public`，所以公开发布面暂缓。

Deferred until explicit approval：

- live website deploy。
- public README / docs marketing alignment。
- public benchmark copy update。
- PyPI public release announcement。

如果之后切换到 public launch，必须先重新验证 live EN/ZH site、公开 README、package install path 和 release notes。

## Recommended Order

1. 保持内部 artifacts 同步，当前 suite 可交接。
2. 如 owner 批准，处理 data boundary：删除或 legacy 标记 `vec_normalize.pkl`。
3. 等远端 CI 跑完后，把 ARC run ids 和 conclusion 补进 external state。
4. 只有用户明确切回 public/release lane，才重新处理 release identity、PyPI/TestPyPI、live website 和公开 docs。

## Current Go/No-Go

- Internal alpha technical review：GO。
- Design-partner technical preview prep：GO with caveats，需明确不承诺 PyPI/public site。
- Public launch：DEFERRED / not public。
