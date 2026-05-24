# OceanScale 内部 Release Decision Matrix

日期：2026-05-24
范围：内部决策表，不作为公开发布材料。

## 当前需要 owner 决策的事项

当前 owner 已决策：not release、CI use ARC、not public to PyPI。CI runner 已改到 ARC，PyPI workflow 已改为 build/check only。仍需要 owner 决策的是 tracked data 文件 `vec_normalize.pkl` 是否删除或标记 legacy。

## 1. Release identity

| Option | 做法 | 优点 | 风险 | 推荐 |
|---|---|---|---|---|
| A. Python alpha 使用 `0.1.0a0` / `0.1.0-alpha` | 将 launch package identity 固定为 0.1 alpha；后续 Python publish tag 使用匹配版本 | 与当前 build artifact、`pyproject.toml`、`__version__` 一致 | 需要停止用 v0.0.x tag 触发 Python publish，或拆分 website/package release | 推荐 |
| B. 回退到 `0.0.x` package identity | 改 Python package version 到 0.0.x，使其匹配当前 website tag train | 与现有 tags 一致 | 会倒退当前 package identity；benchmark artifact 已是 0.1.0a0；更容易混淆 | 不推荐 |
| C. Website 和 Python 完全分离 | website 继续 v0.0.x；Python 用 v0.1.0a0；workflow 按路径/tag pattern 分离 | 最符合 monorepo reality | 需要改 release workflow 和 docs | 推荐与 A 组合 |

建议：采用 A + C。Python package 走 `0.1.0a0`，website tag train 不再触发 PyPI publish，或 PyPI workflow 只响应 Python package release tag。

## 2. CI runner policy

| Option | 做法 | 优点 | 风险 | 推荐 |
|---|---|---|---|---|
| A. 改到 `oceanscale-arc` | 将 general repo workflows 的 `runs-on` 改为 `oceanscale-arc` | 符合 runner reality；去掉 obsolete self-hosted | 已完成 | 已选 |
| B. 保留现状 | 不改 `self-hosted` / `ubuntu-latest` | 无变更风险 | 与 repo 指令冲突；CI 真实性弱 | 不推荐 |
| C. 分层 runner | GPU heavy jobs 用 `oceanscale-arc`，纯 summary job 也统一 ARC 或明确例外 | 更精确 | 需要确认 ARC 环境是否覆盖所有 job dependencies | 可选 |

建议：先统一到 `oceanscale-arc`，除非 owner 明确指定其他 scale set。`ship-arc` 是 swarm-test 专用，不建议本 repo 使用。

## 3. PyPI workflow

| Option | 做法 | 优点 | 风险 | 推荐 |
|---|---|---|---|---|
| A. 只在 Python package tag 触发 | 例如 `python-v*` 或 `oceanscale-v*` 触发 PyPI；website tags 不触发 | 避免 website deploy 误触发 Python publish | 需要调整 tag policy | 推荐 |
| B. 保持 `v*.*.*` 单一 tag | 所有 release 都同时尝试 website + PyPI | 简单 | 当前已经失败；monorepo 两个 version 源冲突 | 不推荐 |
| C. 在 workflow 中跳过不匹配 tag | 保持触发但 version mismatch 时 neutral/skip | 减少红灯 | 仍然浪费 run，release 语义不清 | 次选 |

当前已选：not public to PyPI。workflow 只 build/check package artifact；未来如切回 PyPI lane，再采用 A。

## 4. Data boundary

| Option | 做法 | 优点 | 风险 | 推荐 |
|---|---|---|---|---|
| A. 删除 `vec_normalize.pkl` | 因未 packaged、未发现读取路径，保留 `.npz` | 减少 pickle 风险和数据歧义 | 需要 owner 批准删除 tracked data | 推荐 |
| B. 保留但标记 legacy | 不删文件，只在内部文档说明不 supported | 零删除风险 | 长期继续混淆；pickle 仍在 repo | 次选 |
| C. 重新打包 `.pkl` | 把 pkl 加到 package-data | 兼容潜在旧路径 | 增加 pickle 安全/兼容风险；当前无证据需要 | 不推荐 |

建议：A，但必须先获得 owner 明确确认。

## 5. Live site claim alignment

| Option | 做法 | 优点 | 风险 | 推荐 |
|---|---|---|---|---|
| A. 内部阶段不动 live site | 只记录 stale blocker | 符合用户 `not public` 指令 | live surface 继续 stale，不能 public launch | 当前推荐 |
| B. 部署本地已修正内容 | 让 live site 对齐 85,824 benchmark | public surface 更真实 | 用户已说 not public；需要 release/CI 决策 | 暂缓 |
| C. 临时隐藏 benchmark | 降低 stale claim 风险 | 会改 public site | 暂缓 |

建议：当前阶段 A。若进入 public launch，再做 B 或 C。

## 6. Minimal approved change set once owner says go

当前最小剩余 change set：

1. 如果 owner 批准，删除或 legacy 标记 `oceanscale/data/vec_normalize.pkl`。
2. 等远端 CI 跑完后，记录 ARC run ids。
3. 如果未来进入 public launch，再重新处理 release identity、PyPI/TestPyPI 和 live site alignment。

## 7. Current recommendation to boss

不要把当前状态叫 public launch-ready。可以叫：内部 alpha 技术验证和证据套件完成；release/PyPI/public website 按当前决策暂缓。
