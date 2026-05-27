# OceanScale Internal Launch Completion Audit

日期：2026-05-24
范围：内部 alpha readiness；不作为公开发布材料。

## Audit 标准

本文件按原始目标逐项审计：review all test for our project/startup/tech stack, prepare for startup launch until we have a suite of data/benchmark/pipeline/product/results/etc。

判断规则：

- GREEN：当前 worktree 或外部状态有直接证据证明。
- YELLOW：已有可用证据，但仍有限制、owner 决策或质量 caveat。
- RED：当前证据直接显示未完成或外部 gate 失败。
- DEFERRED：用户已明确 not public，当前阶段不执行。

## Requirement-by-requirement Audit

| Requirement | Authoritative evidence inspected | Current state | Status |
|---|---|---|---|
| Review all tests | uv run pytest tests/ --collect-only -q；uv run pytest tests/ -q；artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md | 378 tests collected；373 passed，5 xfailed，no warnings，23.11s；xfails 已按 launch impact 记录 | GREEN for internal alpha |
| Verify tech stack | tests/test_environment.py；tests/test_version_compat.py；tests/test_warp_smoke.py；tests/test_newton_smoke.py；artifacts/TEST_TECH_STACK_MATRIX_INTERNAL_2026-05-24.md | oceanscale 0.1.0a0；Newton 1.2.0；Warp 1.13.0；Torch 2.11.0+cu128；RTX 5090 | GREEN local |
| Prepare benchmark evidence | benchmarks/competitive/results.json；benchmarks/RESULTS.md；benchmarks/README.md；benchmarks/COMPETITIVE_MATRIX.md | Standard benchmark source exists and is JSON-valid；n=4096 is 4,588,922 env-steps/s；fidelity pass | GREEN |
| Prepare data/package evidence | artifacts/PACKAGE_ARTIFACT_STATE_INTERNAL_2026-05-24.json；pyproject.toml package-data；tests/test_package_data.py；artifacts/PACKAGE_DATA_EVIDENCE_INTERNAL_2026-05-24.md；wheel/sdist/twine/no-deps install smoke | Supported zip/npz packaged；source-only pkl not packaged；policy test 4 passed；package artifact state green | YELLOW because tracked pkl cleanup needs owner approval |
| Prepare product evidence | artifacts/PRODUCT_CAPABILITY_EVIDENCE_INTERNAL_2026-05-24.md；tests/test_cli.py；tests/test_vec_env.py；CLI/demo/train smokes | CLI, envs, sensors, VecEnv, short train, hover demo command are executable | YELLOW because bundled hover policy quality is smoke-level only |
| Prepare pipeline evidence | artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json；workflow runner scan；release/PyPI policy scan；historical GitHub run list；PyPI index query | All workflow runners use oceanscale-arc；release deploy is manual-only；public PyPI/TestPyPI publish path is disabled；historical run state preserved | GREEN for current not-release/not-public policy |
| Prepare results/report suite | artifacts/BOSS_PACKET_INTERNAL_2026-05-24.md；LAUNCH_READINESS_INTERNAL；LAUNCH_EVIDENCE_INDEX；LAUNCH_FINAL_GATE；manifest；runbook；handoff；scripts/verify_internal_launch_suite.py | Internal report suite exists, points to source-of-truth artifacts, and has an executable consistency verifier | GREEN |
| Respect not public | User instruction；runbook/handoff/evidence index | No public deploy or PyPI publish performed in this pass; live public surface remains deferred | GREEN for instruction compliance |
| Be launch-ready | External pipeline, PyPI, CI runner, version identity, live surface, data cleanup | Internal alpha suite is ready for review under not-release/not-public constraints; public launch remains out of scope | GREEN internal / DEFERRED public |

## Current External State Rechecked

| Check | Result |
|---|---|
| Historical Publish to PyPI latest 5 | v0.0.21 through v0.0.25 all failure; public publish path now disabled |
| Latest Publish to PyPI run | 26365095288, failure |
| Release Deploy latest 5 | v0.0.21 through v0.0.25 all success |
| Latest Release Deploy run | 26365095318, success |
| PyPI availability | No matching distribution found for oceanscale |
| Workflow runner scan | all workflow runs-on entries use oceanscale-arc |
| External launch state artifact | artifacts/EXTERNAL_LAUNCH_STATE_INTERNAL_2026-05-24.json records overall_public_launch_pipeline = deferred_not_public |

## Completion Decision

The internal launch evidence suite is prepared and verified for the current owner decision: not release, CI uses ARC, not public to PyPI. Public launch remains out of scope:

1. Source data boundary for vec_normalize.pkl still needs owner approval if cleanup is desired.
2. Public surface remains deferred until the user explicitly changes the not-public instruction.
3. PyPI/TestPyPI remains disabled until the user explicitly changes the not-public-to-PyPI instruction.

## Next Meaningful Work

Without owner approval, keep the suite accurate and internally useful: update evidence after any code or external state change, keep final gate reproducible, and use the boss packet / decision matrix to get the three approvals above.

With approval, the next implementation order is:

1. Fix release identity and PyPI tag policy.
2. Fix workflow runners to ARC.
3. Resolve vec_normalize.pkl.
4. Re-run final local gate plus remote workflow/PyPI checks.
5. Only then revisit public website and public launch copy.
