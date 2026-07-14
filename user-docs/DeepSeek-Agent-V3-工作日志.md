# DeepSeek Agent V3 工作日志（0.11.0）

日期：2026-07-15

状态：源码与离线回归已完成；Word/文本在线案例通过；最新大型 TypeScript 在线案例仍为失败记录，修复后只完成确定性回归，尚未再次消耗 API 余额重跑；提交、tag、推送和远端核验等待发布闭环。

## 1. 目标

接手 v0.10.0 之后未完成的可靠 Agent Loop 工作，以固定参考项目的真实源码为对照，不做“错一点补一点”的零散修改，统一处理以下端到端链路：

1. Agent Loop 和工具轮次计数。
2. 轮次终止、finish reason、最终答案合成和 DSML 防御。
3. 单结果、同轮和历史工具结果预算，以及私有附件读取。
4. micro-compaction、主动语义压缩、确定性 fallback 和熔断。
5. 显式安全并发、串行屏障、中断后的工具调用/结果配对。
6. 计划完成判定、conditional mutation、artifact/Word 证据门。
7. 测试命令选择、网络重试、Resume 和终端输出。
8. 整理用户目录、测试与验收、历史建议、旧检查点和版本化文档。

约束保持不变：DeepSeek 是唯一推理 Provider；架构维持 `CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission`；所有工具继续经过 `ToolRequest -> PermissionManager -> ToolResult`；不读取、输出、复制或提交真实 API Key 和私有运行数据。

## 2. 固定参考与审计边界

参考项目固定为：

```text
https://gitee.com/free/claude-code/tree/claude/
commit b17913e26fd4278ad5cd4b32ed3bde86bf1444e9
```

审计使用该 commit 的完整校验副本，而不是根据网页片段猜测实现。差异矩阵覆盖 Agent Loop、终止、工具结果预算、micro/auto compaction、熔断、工具编排、计划判定、测试命令、网络重试、Resume 和终端宽度。

从参考项目确认并借鉴的方向：

- 工具结果使用会话范围的本地持久化和模型可见预览；私有权限、路径与配额检查是本地加强。
- 并发必须由能力显式声明安全，写操作保持顺序屏障。
- 中断后仍要保留一一对应的工具调用/结果协议。
- 上下文压缩与恢复必须有界，并防止重复压缩死循环。
- 终端按 grapheme cluster 和显示单元格宽度处理。

以下内容是 Deep Agent 为 DeepSeek 协议和现有 Python Runtime 推导、实现并验证的本地设计，不能归因于参考项目：

- soft/hard 工具轮次和预留实现/验证窗口。
- finish-reason 防御、DSML 协议正文拒绝和独立 final synthesis。
- conditional-mutation、artifact/Word 完成门与 hard phase 验证附件例外。
- DeepSeek 语义摘要失败后的 AgentState/evidence 确定性投影。

当前没有宣称完全复刻参考项目。Resume 消息链、snip/compact 边界和 replacement-decision 重放仍不同；JS/TS runner 尚未完整消费 bun/pnpm/yarn package-manager 元数据；网络重试尚无 `Retry-After`、jitter、heartbeat 和显式 chunk watchdog；通用 `verify` 完成状态仍主要由模型更新，Word/artifact 才有更强工具证据门。

## 3. 根因组

### 3.1 Agent Loop 与终止语义混在一个固定轮次里

旧 Runtime 把每次模型请求都当作轮次，固定上限同时承担工具执行、上下文恢复和最终答复。压缩或截断续写会提前吃掉预算；模型没有工具调用时直接返回或在上限处使用通用失败消息，无法区分“已完成”“仍需计划迁移”“工具关闭后需要总结”。未知或过滤型 finish reason 携带的工具调用也缺少统一零执行规则。

### 3.2 工具输出只有局部截断，没有端到端请求预算

单工具虽然有输出限制，但同轮多工具、跨轮历史、Tool Schema、输出预留和安全缓冲未统一估算。大型结果要么重复占据消息历史，要么只剩过短预览；压缩失败缺少持久熔断，容易在 Resume 后重复浪费请求。

### 3.3 工具编排缺少显式并发契约和中断配对

仅凭“看起来像读取”并行会误把需要确认或带隐含副作用的能力放进线程池。中断发生在一组工具执行中间时，可能留下 assistant tool call 没有对应 result，造成下一次请求协议损坏或 Resume 重放不确定。

### 3.4 大任务探索占满轮次，计划完成判定过松或过硬

large/deep 任务会反复读取同一目标，在进入实现和验证前耗尽预算。另一方面，基线测试本身有错误时，旧 verify 语义会诱导模型为“全绿”硬凑修改；无真实缺陷证据的条件性修改又没有合法跳过 `implement` 的路径。artifact 和 Word 任务还需要比普通模型自报更强的受管写入及复验门。

### 3.5 Read/Edit、Resume 和终端存在恢复摩擦

旧 `read_file` 的显示行号与源码之间只用空格，模型可能把显示 padding 当作缩进；无变化 diff 会在后续搜索阶段才失败。Resume 清空近期错误后才构造恢复上下文，并继承当轮读取/停滞额度，导致恢复后仍不能行动。终端按 code point 裁剪会拆开旗帜、肤色、Keycap、VS16、ZWJ 和 CJK 宽字符。

### 3.6 最新大型 TypeScript 在线验收暴露两项路由根因

最新一次大型 TypeScript 在线验收已经通过其他语义门，但 `session_completed` 未通过。复盘确认不是再增加工具轮次能解决，而是路由在长 Prompt 中出现两项相互影响的误判：

1. artifact 检测把“不要输出凭据”“忽略生成文件”等否定要求跨分句传播到后面的普通动作或 artifact 词，形成错误 artifact 完成门。
2. “若没有独立可复现缺陷，则跳过 implement，但仍完成 verify”这类较长条件句没有被识别为 conditional mutation，Runtime 因而不允许合法跳过无证据修改。

模型在错误完成门和不可跳过实现之间继续消耗工具轮次，最终只剩 Session 未完成。

## 4. 统一实现

### 4.1 可靠 Agent Loop

- 工具轮次只在同一 assistant 响应的全部调用都有按原顺序配对的 `ToolResult` 且完成 checkpoint 后增加。
- 4/8/16/24 作为 simple/standard/large/deep 的软目标，`runtime.max_tool_rounds_hard_limit` 默认 32 作为工具执行硬上限。
- 上下文压缩、length continuation、纠正响应和 final synthesis 使用独立模型请求计数。
- `stop`、`tool_calls`、空/缺失 finish reason 可继续正常语义；`content_filter` 和未知 finish reason 的工具调用全部丢弃、零执行，并仅允许一次有界纠正。
- 截断工具 JSON 不执行；主循环和 final synthesis 都拒绝结构化工具调用或已知 DeepSeek DSML 工具协议正文作为用户答案。
- 工具循环关闭后使用独立、无工具的 final synthesis；未满足计划、artifact 或验证门时返回包含精确 Session ID 的 Resume 指令。
- soft target 只有在 Task Graph、真实非计划工具、single-validation 和 artifact 等统一执行证据全部满足时才关闭工具；缺失项一次聚合提示后继续到 hard limit。
- 主循环遇到“结构化 tool calls + DSML 正文”会丢弃整批调用并零执行。

### 4.2 工具结果、附件和上下文预算

- 增加单结果 12,000 字符、同轮 48,000、历史 96,000、输出预留 24,000 以及 context safety buffer 的分层预算。
- 超过持久化阈值的完整有界结果写入当前 Session 的 `.project-agent/tool-results/`，返回带 SHA-256、大小和 request ID 的预览。
- 单附件最大 8 MiB；每 Session 最大 512 个、总计 256 MiB；写入和读取检查路径、符号链接、write-once request ID、数量和总大小。
- AgentState 只保存预览和附件元数据；超 8 MiB、上游截断或持久化失败时明确只保留有界正文，不宣称完整正文可用。
- 历史压缩先 micro-compact 旧的大结果，再降为 metadata，必要时将最旧完整 call/result 组折叠为有界证据，确保协议配对。
- 完整请求估算包括 System/消息、Tool Schema、输出预留和安全缓冲。主动语义摘要使用同一 DeepSeek 模型但关闭工具和 Thinking。
- 首次 tool-history 压缩异常应用确定性 hard fallback；连续三次语义压缩失败打开持久熔断，Resume 后继续生效，成功压缩才重置；熔断打开时 provider-overflow semantic stage 也保持零模型压缩调用。
- hard phase 普通探索附件保持关闭；仅当前 Session 的测试、诊断、文档复验、staged diff 或受限验证 shell 附件可读取两次，每次最多 12,000 字符。

### 4.3 工具编排和中断

- 仅连续、active、`concurrency_safe=True`、权限恰好为 read 且无需确认的调用允许并发，默认最多四个读取工具。
- 任意 mutation、需要确认、未显式声明安全或能力不可用的调用都是串行屏障。
- 结果按模型调用顺序返回，而不是按线程完成顺序。
- `Ctrl+C` 或其他 `BaseException` 时保留已完成结果，为未开始/未解决调用生成失败结果；完整配对并 checkpoint 后重新抛出原中断。
- 子进程工具限制 head/tail 输出，超时或取消时终止完整进程组。

### 4.4 收敛、计划和证据门

- large/deep 追踪连续只读轮次、低收益轮次、重复目标和剩余预算；先关闭宽泛发现，再关闭目标读取和 Shell/Python 探索别名。
- 保留 plan update、implement 和 verify 能力，并为实现证据及验证附件提供严格次数上限。
- 只有 conditional-mutation 计划的 `implement` 可在没有独立缺陷证据时 skipped；scope、inspection 和 verify 不能跳过。
- verify 改为“执行适当验证并如实报告”，不强制研究快照或既有错误基线全绿。
- requested artifact 必须有 managed-write 证据；Word 还必须经过 render、apply、重新打开解析，并禁止凭空添加用户没有要求的生成日期。
- TaskRoute schema 2 保存有界文件/目录 hints；目录精确绑定 `make_dir`，文件与 Word 按 active apply/delete/undo 和 `preview_id` lineage 判定，后续删除不再冒充完成。
- `run_tests framework=auto` 先检查语言 manifest，再考虑通用 `tests/`；npm 只运行存在的 allowlist script。

### 4.5 最新路由修复

- artifact 证据按换行、标点和转折词形成的有界分句判断，不再让否定词跨分句泄漏。
- 明确过滤 credential-output ban 和 ignored generated-file 片段，同时保留独立的正向报告/Word 生成请求。
- conditional mutation 同时识别短模板和“条件证据子句 + 后续 no-change/skip implement 子句”的长距离组合。
- unconditional bug fix 仍不可跳过，避免为了通过验收而把所有修复请求放宽。
- 显式“只运行一次验证”请求新增 `single-validation` 路由；一次实际验证后移除等价 schema，并拒绝 shell/LSP/test 替代调用。
- `not_executed` 不消耗唯一验证机会；同批按真实 ToolResult 串行判定，识别 wrapper 并在执行前拒绝 compound 多验证命令和 hard-phase 写入参数变体。

### 4.6 Read/Edit、Resume、终端和依赖

- `read_file` 改为六列右对齐行号、明确的 `→` 边界和源码正文；工具 schema 明示箭头前不是源码。
- `file_diff` 在搜索前拒绝 `old_text == new_text`。
- Resume 从 `execution_context.recent_error` 恢复上轮错误；新 turn 重置读取/停滞门和验证附件次数，同时保留已见目标和语义压缩熔断。
- Resume 保持原目标和模型/任务模式的单调升级，但明确不提供外部副作用 exactly-once。
- Console 使用 `regex \X` 与 `wcwidth` 处理 grapheme cluster 和终端显示宽度。
- 新增 `regex`、`wcwidth` 为核心依赖；全新 Python 3.14 虚拟环境已验证可安装 `.[dev]`。

## 5. 测试与验证

### 5.1 离线全量结果

```text
454 tests collected；全量 pytest 通过
Ruff check passed
Ruff format --check passed
compileall passed
git diff --check passed
pip check：无依赖冲突
全新 Python 3.14 虚拟环境安装 `.[dev,browser,semantic]`、pip check 和同一 454 项全量通过
18 项 CLI/Console 通过；真实 PTY 覆盖 40 列 Enter/空输入/help、Ctrl+C、Thinking/Resume，以及 25 列复杂 grapheme 进度行
```

覆盖范围包括：

- 工具轮次与模型请求分离、soft/hard 上限、独立 final synthesis。
- finish reason、length continuation、DSML 拒绝和零执行异常响应。
- 单结果/同轮/历史预算、附件配额、hash、路径和符号链接防护。
- micro-compaction、主动压缩、失败 fallback、持久熔断和 Resume。
- 显式并发安全、串行屏障、中断配对和有序结果。
- conditional mutation、artifact/Word 完成门、hard phase 验证附件。
- `行号→源码`、no-op diff、测试命令选择和网络重试现状。
- schema 6、Session、ContextPackage、Event/Permission 边界和隐私。
- Unicode grapheme、CJK 宽度、旗帜、肤色、Keycap、VS16 与 ZWJ。

### 5.2 最新 focused 回归

针对 artifact 跨分句误判和 conditional-mutation 漏判的 focused tests 全部通过。新增确定性端到端场景证明：

1. Prompt 中的 credential-output ban 或 ignored generated files 不会形成伪 artifact 要求。
2. 独立、明确的报告/Word 输出请求仍会形成 artifact 完成门。
3. 无独立可复现缺陷时可以只跳过 conditional `implement`，但必须执行并如实报告 verify。
4. 修复后的模拟 Agent 在 4 次 main-loop 请求和 1 次 final synthesis 内完成，DSML 正文不会被当成成功答案。
5. 第一次静态检查失败后，等价 shell 与 LSP 验证均在执行前被拒绝，实际验证命令仍只有一次。

该结果是确定性离线回归，不是新的 DeepSeek 在线验收。

## 6. 在线验收与余额控制

已通过：

- 六 Word 文档汇总案例通过在线语义门。
- 约 2500 字文本总结案例通过在线语义门。

尚未通过：

- 最新大型 TypeScript 在线验收记录为 36 次逻辑请求、31 个工具轮次、1,477,342 tokens。
- 除 `session_completed` 外的验收门均通过；最终结果仍是 Session 未完成，因此该案例整体必须记为失败。
- 根因是 artifact 跨分句误判与 conditional-mutation 漏判，不是简单增加工具轮次。
- 修复后已完成 4 main + 1 final synthesis 确定性回归；为避免继续消耗用户 DeepSeek API 余额，没有再次在线重跑。

结论边界：离线全量和 focused 回归可以声明通过，Word/文本在线案例可以声明通过；大型 TypeScript 在线案例不能声明通过，也不能把确定性模拟写成在线成功。

## 7. 审查发现与剩余风险

1. final synthesis、soft/hard turn、DSML 防御和完成门都是本地设计，不能写成参考项目原生行为。
2. Resume 已恢复本轮活性并保留必要状态，但还不等价参考项目的消息链、snip/compact 边界和 replacement-decision 重放。
3. JS/TS managed test runner 对 npm script 有 allowlist，但尚未完整读取 bun/pnpm/yarn package-manager 元数据。
4. 网络重试仍是同 Key 有限指数退避与 `401/403/429` Key 切换，没有 `Retry-After`、jitter、heartbeat 或显式 chunk watchdog。
5. 通用 verify 状态仍主要由模型调用计划工具更新；Word/artifact 才有更强的 managed-write/re-open 工具证据门。
6. Event Bus 是进程内同步总线；Resume 也不是 Durable Intent Journal。中断中的外部副作用可能需要人工核验。
7. 主循环、主动语义摘要、overflow 恢复和 final synthesis 都可能产生 API 请求。余额受限时必须先离线验证，再只运行一次有界在线验收；失败后先分析，不能立刻重复请求。

## 8. 目录与文档整理

用户协作根目录只保留当前 `AGENTS.md`、`README.md`、当前工作日志和分类目录。材料按以下路径归档：

```text
/mnt/d/detail/deepseek/测试与验收/
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.9.0/
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.9.1/
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.10.0/
/mnt/d/detail/deepseek/历史资料/
/mnt/d/detail/deepseek/老版使用说明/
/mnt/d/detail/deepseek/老版工作日志/
```

发布仓库中的脱敏副本位于：

```text
~/AI-Agent/user-docs/测试与验收/实用案例/
~/AI-Agent/user-docs/历史资料/
~/AI-Agent/user-docs/老版使用说明/
~/AI-Agent/user-docs/老版工作日志/
```

只同步脱敏文档、案例和验收摘要；不提交 PTY capture、Session、Memory、日志、浏览器状态、缓存、完整私有附件或 `.project-agent`。

## 9. 版本与发布状态

```text
目标版本：v0.11.0
AgentState schema：6
工作分支：feat/v0.11.0-reliable-agent-loop
基线：95038c9（v0.10.0）
最终提交哈希：暂待发布闭环
Git tag：暂待创建 v0.11.0
GitHub main 推送：暂待发布闭环
远端 main/tag 指向核验：暂待发布闭环
GitHub Actions：暂待推送后核验
```

本日志不提前填写一个会因“更新提交哈希”而再次变化的自引用提交。最终发布回复必须给出实际版本、最终 commit、tag、远端核验和 Actions 结果。

文档发布阶段没有调用 DeepSeek 模型，也没有读取或输出真实 API Key。两份 0.11.0 Word 已由版本化 Markdown 完全离线生成；ZIP 完整性、Pandoc 关键章节、标题/作者和固定 `2026-07-15T00:00:00Z` 核心属性均通过验证，当前目录与源码镜像的 SHA-256 逐对一致。旧 Word 继续保存在 `老版使用说明/` 和 `老版工作日志/`。

## 10. 回滚和下一步

回滚目标为 v0.10.0：

```bash
cd ~/AI-Agent
git switch --detach v0.10.0
.venv/bin/pip install -e .
```

0.11.0 配置迁移只增加缺失默认值，不覆盖用户配置。不得为了回滚删除 Memory、`.project-agent`、Session 或私有附件。v0.10.0 会把 schema 6 Session 拒绝为未来 schema，不能直接 Resume；应先在 v0.11.0 完成该 Session，或保留原数据并在 v0.10.0 新建 Session。

下一步按顺序执行：

1. 完成最终只读代码、循环和发布审计，处理任何 blocker。
2. 再核对候选提交不含 Key、PTY、metrics、Session、Memory、缓存或 `.project-agent`。
3. 提交 v0.11.0，fast-forward `main`，创建 `v0.11.0` tag，推送 main/tag。
4. 核验远端 main/tag 指向同一预期 commit，并检查 GitHub Actions。
5. 不再为本次发布在线重跑大型案例；只有用户以后明确同意新的 API 成本时，才用全新 Workspace/Session 做一次修复后验收。
6. 后续版本再评估 package-manager 元数据、`Retry-After`/jitter/watchdog、通用 verify 证据和 Durable Intent Journal，不在 0.11.0 临时追加未验证功能。
