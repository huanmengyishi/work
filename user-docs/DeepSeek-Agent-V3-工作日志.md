# DeepSeek Agent V3 工作日志（0.12.2）

日期：2026-07-27

状态：v0.12.1 已完整推送且 Actions 全绿，但 GitHub 根目录仍残留两份 0.10.0 Word，导致用户在仓库首页看到旧工作日志。v0.12.2 已统一根目录与 `user-docs/` 的当前文档入口，并加入防回归测试。实现提交 `c8d73790331776c98ce9cdb8d2141bb8dc0a6eee` 已推送，Actions run `30241601163` 在 Python 3.11/3.12/3.13 上均 504 passed。本补丁不改 Runtime/schema，不读取 Key，不调用 DeepSeek API。

## 1. 本轮目标

整合并处置四份材料：

- `我来针对这六大发展方向.docx`
- `0.12改进.docx`
- `DeepSeek Agent V3 后续开发规划总体目标DeepSeek Agent V3 v0.docx`
- `基于对代码库的全面审查.docx`

优先解决会导致实例启动失败、卡死、高额消耗、误报完成或破坏数据的问题；不为“全部照单实现”引入未经验证的备用 Provider、并行写 Agent、自动 A/B 或大规模 Runtime 重写。

## 2. 基线与参考边界

- 开始版本：v0.11.0
- 开始 HEAD：`7b575fa21da772379122b904b77ce25a79192267`
- 开始基线：454 tests passed
- 功能版本：v0.12.0
- 发布闭环补丁：v0.12.1
- 文档布局补丁：v0.12.2
- 最终 AgentState schema：7
- 最终核心接口契约：3

参考项目：`https://gitee.com/free/claude-code/tree/claude/`，固定 commit `b17913e26fd4278ad5cd4b32ed3bde86bf1444e9`。

参考仓库自述为泄露的 Anthropic 专有源码快照且无可复制许可证。本轮只参考可观察行为：Runtime 持有任务状态、完成前执行校验钩子、会话级有界输出、工具调用/结果配对、显式并发安全和有限恢复。该快照没有 Python、Ruff 或 CI workflow，v0.12.1 只借鉴其 `bun.lock` 的精确依赖原则，没有声称照搬不存在的 Ruff 配置，也没有复制其源码。

## 3. 主要根因

1. API Key 预检晚于 Project/Memory/Vector 构造，缺 Key 也可能触发重型 Vector 初始化。
2. CLI 依据“是否抛异常”而非 Session 终态返回，未完成任务可能错误退出 0。
3. 工具轮次有上限，但上下文压缩、纠正、续写和 final synthesis 没有统一请求/Token/时间门。
4. 计划缺少语义和验证元数据，模型可预填终态，进度容易自报。
5. Word 完成门只看工具成功与路径，空 parse data 可冒充验证；Session 摘要没有按 delete/undo 最终态回放。
6. Memory 类型、容量和置信度反馈缺少严格边界；Vector 默认和加载时机过重。
7. 性能改进没有安全历史，而自动调参方案又会造成隐私、成本和不可复现风险。
8. `fcntl`、shell 参数拼接、Health 构造和依赖下限存在工程一致性问题。
9. 发布开发依赖只有 `ruff>=0.6.0` 下限，而 lint 选择依赖工具默认值；Ruff 0.16.0 将默认规则从 59 扩大到 413 后，CI 立即报出 210 个历史策略告警。
10. v0.12.1 最新 Word 已在 `user-docs/`，但源码根目录仍跟踪两份与归档副本字节相同的 0.10.0 Word，导致 GitHub 首页暴露旧版。

## 4. 实现

### 4.1 启动、状态与预算

- 缺 Key 在 Project、Memory、Vector 前失败，提示私有 `secrets.env`。
- Vector 新默认关闭；旧配置显式开启时也延迟到首次真实操作。
- CLI 退出码：完成 0、可 Resume 未完成 2、Runtime/配置异常 1。
- 新增每 turn 模型请求、Token、耗时预算；所有阶段网络前预留并 checkpoint。
- 预算耗尽保存明确失败原因、计划、工具证据和 Resume 指令。
- 协议纠正和异常 finish 恢复次数可配置；HTTP 重试仍只由 `DeepSeekClient` 负责。

### 4.2 Task Graph 与可信进度

- PlanStep 新增 `parent_id`、`step_type`、`estimated_tool_rounds`、`artifact_ids`、`validation_rules`、`progress_weight`。
- schema 6 升级到 7，接口契约 2 升级到 3。
- Convergence 优先使用步骤语义，同时兼容旧固定 ID。
- 新计划拒绝 completed/failed/skipped 预填，依赖/父节点缺失、自环和循环均失败。
- Progress 只按持久化的已满足步骤加权；running 不到 100%，不猜 ETA。

### 4.3 Artifact、Word 与 Session

- 新增 JSON、YAML、Python 和 DOCX 有界校验。
- DOCX 校验 ZIP/OPC 必需成员、路径、重复 member、加密/符号链接、CRC、XML、压缩比、解压总量和非空正文。
- `document.parse` 生成无正文 receipt，只含相对路径、格式、大小、SHA-256、完整性和检查结果。
- Runtime 不直接读取任意文件，只消费 ToolResult receipt；空/截断/不完整 receipt fail-closed。
- Session 回放 apply/delete/undo 最终血缘，已删除或撤销创建的产物不再 verified。
- 旧 Session 无 receipt 的 Word parse 需 Resume 后重新 parse 一次。

### 4.4 Memory 与观察性优化

- 新写入严格使用 MemoryKind；旧未知类型兼容读取。
- 容量维护限制条数、负载、扫描和报告数量，默认 dry-run；Correction/Decision 受保护。
- 显式 feedback ID 幂等更新置信度，配置正负幅度和上下限；普通使用不自动加分。
- Vector 采用线程安全一次性延迟加载。
- 性能历史只保存安全标量，按 run_id 幂等、容量有界；不保存 Prompt/正文/参数/凭据，不自动调参。

### 4.5 工程一致性

- 配置、项目、Daemon、Queue 和 Session 使用统一 POSIX/Windows 文件锁层。
- `shell=True` 参数改用 `shlex.join()`，特殊字符保持单参数且不误执行。
- ToolManager 支持 CapabilityHealth 注入；CLI 通过方法切换 yolo/super-yolo。
- Chroma/Playwright 依赖下限统一；launcher 默认只建议核心 editable 安装。
- Memory、反馈和性能历史默认值进入 add-only 配置迁移。

### 4.6 v0.12.1 CI 根因修复

- `pyproject.toml` 精确固定 `ruff==0.15.21`，并用 `required-version` 拒绝静默漂移。
- 显式选择原有 `E4/E7/E9/F` lint 契约，不依赖 Ruff 版本默认值。
- Actions 在检查前显示 `ruff --version`；新增回归测试防止版本约束和规则契约被意外放宽。
- 不对 210 个新告警执行盲目 `--fix`；规则扩展留作后续独立、可评审任务。

### 4.7 v0.12.2 GitHub 文档入口修复

- 删除源码根目录重复的 0.10.0 Word；其字节相同副本仍保留在老版归档。
- 在 GitHub 根目录和 `user-docs/` 同时发布当前版本 Word，README 增加最新 Markdown/Word 直达链接。
- 新增发布布局回归：根目录只能有当前版本的两份 Word，`user-docs/` 必须存在对应文件。

## 5. 测试与实例审计

最终离线结果：

```text
502 tests collected and passed
Ruff check passed
Ruff format --check agent tests scripts passed
compileall agent scripts passed
pip check: no broken requirements
git diff --check passed
v0.12.1 聚焦回归：9 passed
GitHub Actions Python 3.11/3.12/3.13：各 503 passed
v0.12.2 发布布局/Word 聚焦回归：10 passed
GitHub Actions v0.12.2 Python 3.11/3.12/3.13：各 504 passed
```

聚焦验证包含：

- Key/Vector 启动、CLI 0/1/2；
- Budget/Resilience、Progress/Artifact；
- Memory 类型、容量、保护和反馈；
- 性能历史隐私、幂等和容量；
- Runtime/Convergence/Event/Session/Resume；
- DOCX receipt、空回执、delete/undo 血缘；
- Windows/POSIX 锁、无 fcntl CLI import、shell 元字符、Health DI 和依赖一致；
- 真实 PTY 的 Enter、空输入、处理中状态、Ctrl+C 和 Unicode 宽度。

隔离 XDG 实例冒烟通过：

```text
deep-agent 0.12.0
agent --help -> 0
agent init -> 成功创建隔离项目
缺 Key 普通任务 -> 1，且不创建 .project-agent
launcher/agent --version -> deep-agent 0.12.0
```

v0.12.0 的隔离实例记录保持有效；v0.12.1/0.12.2 未修改 Runtime，并已聚焦确认 `deep-agent 0.12.2`。本轮没有读取真实 Key，没有运行 `doctor --online`，也没有发起付费 DeepSeek 请求。原因是用户 API 余额耗尽；余额恢复后只补一次有请求、Token 和时间上限的代表性在线验收。

## 6. 审查结论与未实现项

原判断不成立：Resume 在 v0.11.0 已保留计划、节点、工具证据和逐批次 checkpoint，因此没有按错误前提重写。

不采纳：Runtime 重复网络重试、备用 Provider、默认 LLM 反思/抽取、自动 A/B/自动调参、未隔离的并行写 Agent、直接分块部分写最终文件、复制无许可证参考源码。

延期：跨 Resume Session 生命周期预算、章节级事务式文档流水线、动态 replan、Durable Intent Journal、外部副作用 exactly-once、Memory import/export、原生 Windows 实机和新的大型在线任务成功记录。

## 7. 版本、提交与发布

- 版本：`0.12.2`
- AgentState schema：`7`
- 核心接口契约：`3`
- v0.12.0 实现与离线验证：`c1fd3fa00a2457f65165dabb279e0df423868d0b`
- v0.12.0 文档与旧 tag：`146c59efdfe6eed2c6d1f6f9cf7019fb0fc5018c`；不强制移动已公开标签
- v0.12.1 实现提交：`cac5fd7b885420d783df1ce625387dc40f8939fe`
- GitHub 推送：成功；Actions run `30210431120` 成功，Python 3.11/3.12/3.13 的 Ruff、format、503 项测试和 compileall 全部通过
- v0.12.1 最终提交/tag：`eb111ad13357a45622c19d38d75efda8c5cfb0dc`，Actions run `30211093526` 成功
- v0.12.2 实现与文档布局提交：`c8d73790331776c98ce9cdb8d2141bb8dc0a6eee`
- v0.12.2 初次 GitHub 推送：成功；Actions run `30241601163` 成功，Python 3.11/3.12/3.13 的 Ruff、format、504 项测试和 compileall 全部通过
- v0.12.2 最终发布元数据提交：由本日志所在的 `v0.12.2` tag 标识；提交无法在自身内容中递归写入自己的哈希
- `v0.12.2` tag 与最终 `main`：本日志提交后创建、推送并由外部命令核验

## 8. 文档归档

四份输入材料已无损归档到：

```text
/mnt/d/detail/deepseek/历史资料/改进建议/20260726-v0.12.0/
```

综合审计、错误位置、处置矩阵、实例结论和后续建议位于：

```text
/mnt/d/detail/deepseek/项目运行审计与改进建议/20260726-v0.12.0/
/mnt/d/detail/deepseek/项目运行审计与改进建议/20260726-v0.12.1/
/mnt/d/detail/deepseek/项目运行审计与改进建议/20260727-v0.12.2/
```

旧版 Word 在新版 Word 生成并复验后移入 `老版使用说明/` 和 `老版工作日志/`。
