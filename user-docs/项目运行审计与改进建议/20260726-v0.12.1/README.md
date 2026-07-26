# Deep Agent v0.12.1 发布闭环补充

日期：2026-07-26

## 结论

v0.12.0 的 502 项全量离线验证和隔离实例冒烟本身没有失败。GitHub Actions run `30209606973` 的三个 Python 任务都在 Ruff 阶段停止，根因是开放下限 `ruff>=0.6.0` 在 CI 中安装了 Ruff 0.16.0，而本地发布门使用 Ruff 0.15.21。Ruff 0.16.0 扩大了默认规则集，因而一次性报出 210 个历史策略告警；pytest 与 compileall 当时根本未执行。

官方破坏性变更证据：`https://github.com/astral-sh/ruff/releases/tag/0.16.0`，该说明明确记录默认规则从 59 增加到 413。

## 错误位置与修改

| 位置 | 错误 | v0.12.1 修改 |
| --- | --- | --- |
| `pyproject.toml` | `ruff>=0.6.0` 允许 CI 静默升级 | 精确固定 `ruff==0.15.21` |
| `[tool.ruff]` | 没有显式 lint 契约 | 要求 0.15.21，选择 `E4/E7/E9/F` |
| `.github/workflows/test.yml` | 失败时不直观显示工具版本 | Ruff 检查前输出 `ruff --version` |
| 回归测试 | 缺少对发布 lint 契约的保护 | 新增配置回归，防止再次放宽 |

没有对 210 个告警执行 `ruff --fix`，因为它们是新默认规则对历史代码的新策略审查，不是本轮 Runtime 缺陷。后续如要扩大 lint 规则，应按规则分批、独立评审和测试。

## 参考项目边界

`free/claude-code` 的固定 `claude` 分支快照没有 Python、Ruff、pytest 或 CI workflow，无法照搬具体修复。其根目录提交了 `bun.lock`，说明发布自动化需要精确依赖闭包；v0.12.1 采用了同类原则，但没有复制其无许可源码。

## 验证与在线边界

- v0.12.0 Runtime 基线：502 tests passed，Ruff/format/compileall/pip/diff 与隔离实例通过。
- v0.12.1 本地聚焦：Ruff 0.15.21 check/format、9 项相关测试、compileall、版本和 diff check 通过。
- GitHub Actions：run `30210431120` 由实现提交 `cac5fd7b885420d783df1ce625387dc40f8939fe` 触发并成功；Python 3.11/3.12/3.13 均通过 Ruff 0.15.21 check、format、503 项测试和 compileall。
- 在线 DeepSeek 认证：用户 API 余额耗尽，本轮不读取真实 Key、不发起付费请求；余额恢复后只补一次有预算上限的在线验收。

## 后续改进建议

1. 将 lint 规则扩展与工具版本升级分成独立 PR，先评审规则再修代码。
2. 为 Python 发布工具增加可审查的 constraints/lock 生成流程，但不盲目锁死运行时底层库。
3. 保持重型能力默认关闭；在线认证恢复前不改变 DeepSeek-only 路由。
4. 继续执行 v0.12.0 整合报告中的 Durable Intent Journal、生命周期预算和事务式文档流水线等分期建议。
