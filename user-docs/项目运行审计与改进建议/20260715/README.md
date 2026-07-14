# Deep Agent v0.11.0 实例运行审计与改进建议

审计日期：2026-07-15

审计对象：

- 用户文档与验收入口：`/mnt/d/detail/deepseek`
- 程序源码：`~/AI-Agent`
- GitHub：`https://github.com/huanmengyishi/work`
- 发布版本：`v0.11.0`
- 发布提交：`113592ce32d7ee2920c1ac6d1d2bea092a2f7c`

本次没有读取、输出或复制真实 API Key，也没有读取用户 Memory、Session、日志、浏览器会话或私有工具附件。在线能力只核对已有脱敏验收记录，没有再次消耗 DeepSeek API 余额。

## 1. 总结结论

结论分为三个层级：

| 层级 | 结论 | 证据 |
| --- | --- | --- |
| 安装与启动 | 通过 | 全局 `agent --version` 与虚拟环境均返回 `0.11.0`；隔离环境的 `agent init` 成功 |
| 本地功能与回归 | 通过 | 当前提交重新执行 `454 passed in 18.18s`；Ruff、format check、compileall、pip check、git diff check 全部通过 |
| 小中型真实任务 | 有成功证据 | 六 Word 汇总与约 2500 字总结的已有在线语义门均通过 |
| 大型复杂闭环 | 尚不能判定稳定 | 两次大型 TypeScript 候选验收均以 Session 未完成结束；最终发布提交没有可归属的修复后在线大型成功记录 |
| 当前用户 Key/账户可用性 | 本次未验证 | 按安全约束未读取真实配置，也未运行会消耗余额的 `doctor --online` |

因此，v0.11.0 可以作为可运行的开发/试用实例，用于离线工具、交互操作和已有成功范围内的小中型任务；暂不建议把大型仓库的全自动“分析 -> 决策 -> 修改 -> 验证”闭环当成无人值守生产能力。

## 2. 已完成的现场验证

### 2.1 本地源码与 GitHub

- 本地 `main`、抓取后的 `origin/main` 和远端 `refs/heads/main` 均为 `113592ce32d7ee2920c1ac6d1d2bea092a2f7c`。
- 注解 tag `v0.11.0` 的 tag object 为 `67be913e2498eb9cf5c5496811181c7b8ee6e4db`，解引用后指向同一发布提交。
- GitHub Actions run `29372482649` 状态为 `completed/success`。
- 工作流覆盖 Python 3.11、3.12、3.13，执行 Ruff、全量 pytest 和 compileall。
- GitHub 当前没有正式 Release 对象，只有 Git tag；仓库 API 也没有识别到许可证。

### 2.2 当前实例

- 全局入口：`~/.local/bin/agent -> ~/AI-Agent/launcher/agent`。
- `agent --version`：`deep-agent 0.11.0`。
- 当前虚拟环境：Python 3.14.4。
- 隔离临时 HOME/XDG 下执行 `agent doctor` 正常，能够列出 Git、Python、Docker、Pandoc、LibreOffice、Node 等能力状态。
- 隔离临时项目执行 `agent init` 成功，生成 `.project-agent` 的项目配置、索引和上下文文件。
- 真实 PTY 中启动 REPL 成功；空 Enter 有明确反馈；`Ctrl+C` 返回可恢复提示；`/exit` 退出码为 0。

### 2.3 回归与静态检查

```text
pytest:                 454 passed in 18.18s
ruff check:             passed
ruff format --check:    90 files already formatted
compileall:             passed
pip check:              no broken requirements
git diff --check:       passed
Git worktree:           clean before文档交付
```

### 2.4 已有在线验收

| 案例 | 状态 | 关键指标 |
| --- | --- | --- |
| 六 Word 汇总 | 通过 | 9 次主循环请求，19/19 工具成功，约 57.8 秒 |
| 约 2500 字总结 | 通过 | 3 次主循环请求，2/2 工具成功，约 32.8 秒 |
| 大型 TypeScript，`final-prepublish-single-rerun` | 失败 | 36 次逻辑请求，1,477,342 tokens，仅 `session_completed` 未通过 |
| 大型 TypeScript，`final-20260715T025700` | 失败 | 38 次逻辑请求，1,573,936 tokens，Session、八项优点、Bug/限制和验证报告均未完成 |

第二条大型记录生成于 2026-07-15 03:04，而发布提交创建于 05:54、最终修复提交创建于 06:16，因此不能直接归因于最终 `113592c`。但记录没有保存 `git_head`、dirty 状态或代码包哈希，也就无法证明它对应哪一个候选版本。这本身是验收可追溯性缺陷。最终版本仍缺少一次有成本上限、可明确归属的修复后大型在线成功记录。

## 3. 已确认需要修改的问题

### P0-1：缺少 Key 时先触发约 79.3 MiB 向量模型下载

复现：在全新隔离 HOME/XDG、未配置 `DEEPSEEK_API_KEY` 的环境运行一个普通任务。程序先下载 Chroma 默认 ONNX 模型，约 9 秒后才报告缺少 Key 并退出 1。

影响：

- 缺少 Key 的失败路径产生非必要网络流量、等待和磁盘写入。
- `memory.vector_enabled` 默认是 `true`，与项目“重型能力默认关闭”的约束不一致。
- CI 测试配置普遍把 Vector 关闭，因此没有覆盖默认配置与已安装 Chroma 的组合。

建议修改位置：

- `agent/config.py`：把新安装的 `memory.vector_enabled` 默认值改为 `false`；迁移只补默认值，不覆盖现有用户选择。
- `agent/cli.py::run_once`：在创建项目、Memory 和向量组件前做无泄密的 Key 存在性预检。
- `agent/memory.py::MemoryStore`、`agent/vector.py::OptionalChromaStore`：把 Chroma collection/embedding function 改成真正按首次向量操作延迟创建。
- `tests/test_cli.py`、新增 Vector 启动测试：模拟 Chroma 已安装但无 Key，断言不访问网络、不创建模型缓存，并在短时间内退出 1。

临时规避：首次运行前先执行 `agent doctor`，并在不需要语义 Memory 时设置：

```yaml
memory:
  vector_enabled: false
```

### P0-2：任务语义失败时 CLI 仍可能返回退出码 0

当前 `agent/cli.py::run_once` 只在抛出异常时返回 1；只要 `runtime.run()` 返回字符串，最后固定返回 0。Runtime 即使把 Session 标为 `failed` 并返回“任务尚未完成/请 Resume”，CLI 仍会被 shell、CI 或调度器视为成功。

已有两条大型失败记录的 `process_returncode` 都是 0，证实这不是纯理论问题。

建议修改位置：

- `agent/cli.py::run_once`：运行后加载 `runtime.last_session_id` 对应状态。
- 建议退出码：`0=completed`、`2=incomplete/resumable`、`1=不可恢复错误`、`130=用户中断`。
- `tests/test_cli.py`：增加 completed、failed/resumable、exception 和 KeyboardInterrupt 四类退出码测试。
- 文档和 Queue/Daemon 调用方统一使用同一状态映射，避免各自猜测答案文本。

临时规避：自动化调用不能只看进程退出码，还要检查 Session 状态或最终输出中的明确未完成标记。

### P0-3：大型任务缺少总请求/总 Token/成本硬预算

当前有工具轮次上限，但压缩、纠正、续写和 final synthesis 使用独立模型请求计数。大型失败记录分别消耗 36/38 次逻辑请求和约 147.7 万/157.4 万 tokens，仍未形成完成答案。

建议修改位置：

- `agent/config.py`：增加默认保守的 `max_model_requests`、`max_total_tokens`、`max_wall_seconds`；成本金额只能在价格表明确且版本化时作为可选估算。
- `agent/state.py`：持久化预算、已用量和终止原因，Resume 继承剩余预算或要求用户明确追加。
- `agent/runtime.py`：每次模型请求前统一检查预算；预计超限时直接进入有证据的最终总结，不再继续探索。
- `agent/convergence.py`：为 inspect、decision、implement、verify 分配阶段预算，不能让检查阶段耗尽实现和验证额度。
- `scripts/reliability_case_runner.py`：验收必须设置预算并把“预算内完成”作为语义门。

这属于新能力，建议作为 v0.12.0，而不是在 0.11.x 中临时堆叠未经在线验证的策略。

### P0-4：大型任务的计划迁移仍过度依赖模型自报

失败记录已经执行静态检查，却仍因 `implement`、`verify` 未完成触发 hard limit；模型在“没有充分 Bug 证据”和“必须进入 implement”之间反复摇摆，后期又在读取受限时尝试缺少稳定 old_text 的 `file_diff`。

建议修改位置：

- `agent/task_plan.py`、`agent/convergence.py`、`agent/runtime.py`：用真实工具证据自动完成通用 verify，而不是只依赖 `agent.update_step`。
- 增加结构化 finding-decision：`bug_found / no_reproducible_bug / blocked`，携带最小证据和目标文件。
- `no_reproducible_bug` 在满足检查与一次验证后应自动合法跳过 conditional implement，直接进入最终报告。
- 一旦进入保留实现窗口，禁止在没有 read hash/old_text 证据时尝试 `file_diff`，并尽早输出“未发现充分证据”的诚实结论。

## 4. 文档、发布和工程一致性问题

### P1-1：工作日志仍停留在发布候选态

当前工作日志写着提交、tag、推送和远端核验“暂待”，但 GitHub 已完成：

- `main`：`113592ce32d7ee2920c1ac6d1d2bea092a2f7c`
- `v0.11.0`：解引用后指向同一提交
- Actions：run `29372482649` 成功

需要更新当前目录与 `~/AI-Agent/user-docs` 的 Markdown/Word 工作日志，并区分“v0.11.0 发布提交”与后续审计文档提交。

### P1-2：验收摘要遗漏了更新的一次失败记录

使用说明和工作日志把 36 请求、1,477,342 tokens 写成“最新大型记录”，但当前目录还有后生成的 38 请求、1,573,936 tokens 失败记录。后者虽然早于最终发布提交，仍应进入候选历史，不能静默遗漏。

### P1-3：验收产物缺少版本归属信息

`metrics.json` 没有记录：

- Agent 版本与源码 `git_head`
- worktree dirty 状态或源码包 SHA-256
- Python/依赖锁版本
- 脱敏配置摘要与配置 schema
- Prompt/fixture/runner 的哈希

建议在 `scripts/reliability_case_runner.py` 生成不可变 `run-manifest.json`。没有这些字段的旧记录只能叫“候选运行记录”，不能作为某个最终 commit 的发布证明。

### P1-4：本地 editable 安装元数据陈旧

当前 `.venv` 能导入 `regex` 和 `wcwidth`，所有测试也通过；但 `deep_agent-0.11.0.dist-info/METADATA` 只声明 PyYAML，因为它早于当前 `pyproject.toml`。所以本机 `pip check` 不能证明新增核心依赖声明已同步到安装元数据。

建议本机执行：

```bash
cd ~/AI-Agent
.venv/bin/pip install -e .
.venv/bin/pip check
```

发布 CI 还应增加“构建 wheel -> 新虚拟环境安装 wheel -> `agent --version`/最小 CLI 冒烟”，避免 editable/PYTHONPATH 掩盖打包问题。

### P1-5：Launcher 的安装提示会安装全部重型可选依赖

`launcher/agent` 在缺少虚拟环境时提示 `pip install -r requirements.txt`，而该文件包含 Browser、Vector、Document、Semantic 等全部依赖；主 README 则推荐核心 `pip install -e .`。两条路径不一致，也容易造成大体积安装。

建议把 Launcher 提示改为核心 editable 安装，并把可选 extra 明确列出。

### P1-6：缺 Key 错误仍建议把 Key 写入 `model.yaml`

`agent/deepseek.py` 的错误信息建议配置 `model.api_key`，但用户文档要求使用权限为 0600 的 `secrets.env`，生成的 `model.yaml` 通常是 0644。即使配置目录通常为 0700，这个提示仍与安全文档冲突。

建议错误信息只指向 `~/.config/deep-agent/secrets.env`，并逐步弃用明文 `model.api_key`；兼容期只读取、不再推荐。

## 5. 后续改进优先级

### v0.11.1：低风险可靠性修复

1. Key 预检前置，Vector 默认关闭并真正延迟加载。
2. CLI 退出码反映 Session 终态。
3. 修正 Key 配置提示和 Launcher 安装提示。
4. 补齐对应自动化测试、使用说明、工作日志和发布说明。

### v0.12.0：大型任务预算与确定性阶段门

1. 总请求、总 Token、总时长预算。
2. 结构化 finding-decision 与 no-bug 完成路径。
3. 通用 verify 的工具证据自动完成。
4. 有预算上限的一次大型在线验收；需用户在运行前明确同意 API 成本。

### 后续工程化

1. 网络层补充 `Retry-After`、jitter、流式 heartbeat 和 chunk watchdog。
2. JS/TS runner 读取 npm/bun/pnpm/yarn 的实际 package-manager 元数据。
3. CI 增加 Python 3.14、wheel 安装、可选 extra 冒烟和 secret/dependency 扫描。
4. 创建 GitHub Release，附发布说明和文档校验值；补充许可证、作者和项目 URL 等包元数据。
5. 逐步拆分接近 3000 行的 `agent/runtime.py` 和大型 convergence 模块，降低变更回归面。

## 6. 建议验收门

修复不能只以单元测试通过为结束，至少满足：

1. 全新 HOME、无 Key、Vector extra 已安装时，普通任务在 2 秒内退出 1，且不访问网络、不创建 ONNX 缓存。
2. Runtime 返回 failed/resumable Session 时，直接 CLI 退出码非 0；completed 时为 0。
3. Resume 前后总预算单调、可审计，不因新 turn 自动清零。
4. conditional no-bug 案例在预算内完成 verify 和最终报告，不产生写操作。
5. 全量 pytest、Ruff、format、compileall、pip check 和真实 PTY 通过。
6. GitHub Actions 对发布 commit 全绿，远端 main/tag 指向同一提交。
7. 最后才进行一次有明确 token/请求/时长上限的大型在线验收；失败后先复盘，不立即重复消耗余额。

详细命令见同目录的 `复验清单.md`。
