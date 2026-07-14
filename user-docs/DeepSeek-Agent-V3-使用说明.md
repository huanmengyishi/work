# DeepSeek Agent V3 使用说明（0.11.0）

更新时间：2026-07-15

## 1. 当前版本与项目边界

Deep Agent 是安装在 WSL Ubuntu 中、以项目目录为 Workspace 的 DeepSeek CLI Agent。当前版本为 `0.11.0`，AgentState schema 为 `6`。程序仍保持以下单向边界：

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ContextBuilder -> ContextPackage -> PromptBuilder
ToolRequest -> PermissionManager -> ToolResult
```

DeepSeek 是唯一推理 Provider。0.11.0 没有加入其他模型、第二套 Runtime 或绕过 Tool Manager 的命令执行入口。

目录职责：

```text
~/AI-Agent/                         程序源码、测试、发布说明
~/.config/deep-agent/              用户配置与私有 API Key
~/.local/share/deep-agent/         SQLite、Vector、日志、指标、健康与 Daemon 状态
<项目>/.project-agent/             项目上下文、索引、Session、快照、缓存、工具结果附件
/mnt/d/detail/deepseek/             用户文档、验收材料与协作入口，不是程序源码目录
```

不得把 `~/.config/deep-agent` 中的真实 Key，或 `~/.local/share/deep-agent`、项目 `.project-agent` 中的 Memory、Session、日志、浏览器会话、缓存和附件复制到 Git、文档或聊天记录。

## 2. 安装、升级与启动

从源码创建或更新虚拟环境：

```bash
cd ~/AI-Agent
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
.venv/bin/agent --version
```

`pip install -e .` 会安装核心运行依赖 PyYAML、`regex` 和 `wcwidth`。后两项用于按 Unicode grapheme cluster 和终端单元格宽度裁剪进度行，避免拆开 CJK、组合字符、旗帜、肤色、Keycap、VS16 和 ZWJ emoji。Browser、Vector、语义索引和文档处理的额外依赖仍按需安装。

在要处理的项目目录启动，而不是固定在程序目录启动：

```bash
cd /任意/项目目录
agent init
agent
agent "分析当前项目并给出验证结论"
```

首次初始化会在当前项目根目录创建 `.project-agent/`。若自然语言任务恰好以管理命令名开头，可用 `--` 取消命令分派：

```bash
agent -- doctor this code path
```

## 3. 私有 DEEPSEEK_API_KEY

推荐且程序会自动读取的私有文件：

```text
~/.config/deep-agent/secrets.env
```

编辑并收紧权限：

```bash
nano ~/.config/deep-agent/secrets.env
chmod 600 ~/.config/deep-agent/secrets.env
```

文件中填写一个 Key：

```bash
DEEPSEEK_API_KEY=replace_with_your_valid_key
```

也可用英文逗号或中文逗号配置 Key 池：

```bash
DEEPSEEK_API_KEY=key_1,key_2,key_3
```

程序会去除空格、空值和重复项。HTTP `401`、`403`、`429` 会切换下一个 Key；网络错误、超时、`408` 和临时 `5xx` 会对同一 Key 做有限指数退避。流式响应已经输出部分内容后不会自动重放，以免重复工具副作用，而是保存 Session 供 Resume。

验证配置：

```bash
agent doctor
agent doctor --online
```

`agent doctor --online` 只报告 Key 数量、状态码和可用性，不应输出 Key 内容。本次文档整理没有读取、打印或复制任何真实 Key。

## 4. 主要命令

```bash
agent --help
agent --version
agent doctor
agent doctor --online
agent init
agent "实现并验证这个功能"
agent
agent sessions
agent resume
agent resume --session SESSION_ID "继续原任务并完成验证"
agent context show
agent context refresh
agent context index
agent tools
agent tools --all
agent mcp status
agent mcp tools
agent mcp config
agent projects
agent memory search "query"
agent memory add Knowledge "title" "content" --global-memory
agent memory list --kind Correction
agent memory stats
agent memory maintain
agent memory maintain --apply
agent queue "任务一" "任务二"
agent queue resume --id QUEUE_ID
agent parallel "任务1" "任务2" "任务3" "任务4" "任务5" "任务6" "任务7" "任务8"
agent health
agent daemon start
agent daemon status
agent daemon stop
```

交互命令：

```text
/new
/resume [session-id]
/sessions
/status
/undo
/yolo on|off
/super-yolo on|off
/help
/clear
/exit
```

普通任务输入后按一次 `Enter` 提交。提交后 UI 会立即显示正在处理、Thinking 用时、执行模式、模型轮次、计划步骤和工具状态。空输入会明确提示，不会静默执行。`Ctrl+C` 返回可恢复的交互状态；之后用 `/resume` 或 `agent resume --session ...` 继续。

`agent resume` 不带补充 Prompt 时进入最新 Session 的 REPL；`--session` 可指定 ID 或唯一前缀。交互历史以 `0600` 保存于 `~/.local/share/deep-agent/cache/repl_history`。

## 5. 0.11.0 的可靠 Agent Loop

执行模式由本地规则选择，不额外消耗一次 API 请求：

```text
simple    简单事实问题；thinking 关闭；4 个工具轮次软目标
standard  普通分析/编码；high thinking；默认 8 个工具轮次软目标
large     整仓库/长文档；分块计划；16 个工具轮次软目标
deep      审计/重构/根因分析；max thinking；24 个工具轮次软目标
```

0.11.0 区分“模型请求”和“工具轮次”。一条 assistant 响应中的所有工具调用都获得按原顺序配对的 `ToolResult` 并完成 checkpoint 后，才增加一个工具轮次。上下文压缩、`finish_reason=length` 的有界续写、纠正响应和最终合成都有独立计数，不消耗工具轮次。

4/8/16/24 是软目标，默认硬上限为 32。只有统一执行证据门已经满足时，soft target 才会关闭工具；计划、真实非计划工具证据、单次验证证据或指定 artifact 仍缺失时，会给出一次聚合提示并继续使用剩余 hard-limit 预算。硬上限仍会停止继续执行工具。大任务会在反复读取耗尽实现和验证预算前依次关闭宽泛探索、目标读取以及 Shell/Python 探索别名，并保留实现、验证和计划迁移能力。

Runtime 接受 `stop`、`tool_calls` 以及缺失/空 finish reason。`content_filter` 或未知 finish reason 会丢弃伴随的全部工具调用并零执行，最多进行一次纠正；截断的工具调用 JSON 也不会执行。工具循环关闭后进入独立、无工具的 final synthesis；结构化工具调用或已知 DeepSeek DSML 工具协议正文不会被误当成成功答案。主循环若同时出现结构化调用与 DSML 正文，会丢弃整批调用并保持零执行。

这套 soft/hard 工具轮次、finish-reason 防御、DSML 拒绝和独立 final synthesis 是 Deep Agent 的本地设计，不是参考项目原生机制。

## 6. 工具结果预算、压缩和私有附件

每次正常请求依次经过：

```text
单个结构化 ToolResult 硬限制
  -> 同一 assistant 轮次聚合限制
  -> 完整工具历史 micro/metadata compaction
  -> Tool Schema、输出预留和安全缓冲在内的完整请求估算
  -> 必要时由同一 DeepSeek 模型生成无工具上下文摘要
  -> 多次失败后的确定性紧急投影
  -> 发起模型请求
```

工具调用与结果保持连续配对。每个执行批次在提交 worker 前冻结 State、Session ID、run ID 和 EventBus 归属；中断后仍在退出的只读 worker 即使遇到 ToolManager rebind，也只能写回原 Session。AgentState 只保存有界预览、哈希和附件元数据，不保存无限制原始正文。序列化结果超过 12,000 bytes 时，在安全检查和配额允许的情况下，有界结果会写到当前 Session 私有的 `.project-agent/tool-results/`；单附件上限 8 MiB，每 Session 最多 512 个、总计 256 MiB。写入检查 request ID、SHA-256、路径、符号链接、数量和总大小；`tool_result_read` 每次只返回有界片段。附件持久化失败不会把已经成功的工具副作用篡改为失败，而是保留真实 success、受限 head/tail、哈希和 `attachment_persistence_error`，避免模型重复执行副作用。

micro-compaction 先缩减旧的大结果，再降为 metadata，必要时把最旧的完整 call/result 组折叠为确定性证据，不能留下半对协议消息。语义压缩连续三次失败会打开持久熔断；Resume 后仍保持熔断，直到一次成功压缩重置。熔断打开时，服务端 overflow 的第二阶段也不会再次调用模型压缩，而是直接使用确定性 fallback。关闭主动压缩不会取消硬请求限制、输出预留、附件配额或紧急投影。

hard phase 不会重新开放普通探索附件。只有 `implement` 或 `verify` 活跃时，才允许从当前 Session 中由测试、诊断、文档复验、staged diff 或受限验证 shell 产生的验证附件读取最多两次、每次最多 12,000 字符。

## 7. 工具编排、计划和验证语义

只有连续工具调用同时满足“能力可用、显式 `concurrency_safe=True`、权限恰好为 read、无需确认”时才可并发；所有写操作和不满足条件的调用都是串行屏障。即使 `Ctrl+C` 或其他 `BaseException` 中断，已完成结果会保留，未开始或未解决调用会得到合成失败结果，完整配对后 checkpoint，再重新抛出原中断；迟到结果仍使用批次开始时冻结的归属。

大任务计划通常为 `scope -> inspect-chunks -> implement/synthesize -> verify`。只有 conditional-mutation 计划中的 `implement` 可以在无独立缺陷证据时标为 skipped；`scope`、检查和 `verify` 仍必须完成。已完成的必需 Task Graph 还必须至少有一次非计划、非 Runtime-denied 的真实工具执行，不能只靠 `agent_update_step` 自报完成。验证的完成含义是执行适当命令并如实报告结果，不要求本来就有错误的研究快照被强行修到全绿。

请求生成的 artifact 必须有受管写入证据；TaskRoute schema 2 最多保存 32 个清洗后的文件/目录 hint。明确目录名必须匹配成功 `make_dir`，明确文件名必须匹配成功 `file_apply`；只有类型提示时才按 `.docx` 或 `.pdf` 扩展名匹配，不能用无关写入冒充产物。Runtime 按路径重放 apply/delete/undo，创建后删除的文件不再算完成，撤销删除后才可恢复；Word 的 render、`file_apply`、重新打开解析和日期检查还绑定到同一 `preview_id` 与目标文件。artifact 意图按有界分句判断，因此“不要输出凭据”“忽略生成文件”等禁令不会跨分句误触发 artifact 要求；独立、明确的报告生成请求仍会被执行。

`read_file` 输出为“右对齐六列行号 + `→` + 原始源码”，例如：

```text
     1→export const value = 1
```

箭头和箭头前的显示行号都不是源码。`file_diff` 在搜索前拒绝 `old_text == new_text`，避免无意义修改。

`run_tests framework=auto` 按 Python markers、`package.json`、Cargo、Go、Gradle、Maven 的顺序选择框架。npm 只运行实际存在的 allowlist script，优先 `test`、`typecheck`、`check`、`lint`、`build`。当前尚未完整读取 bun/pnpm/yarn 的 package-manager 元数据，相关项目应在任务中明确测试命令并核对实际执行结果。

任务若明确要求“只运行一次验证”，TaskRouter 会记录 `single-validation`。即使任务不需要 Task Graph，结束前也必须存在一次真实验证尝试；普通读取、Runtime-denied 和 handler 前 `not_executed` 调用不算。第一次实际测试、诊断或受识别的验证 shell 命令无论成功、缺少依赖还是遇到既有基线错误，都算这一次；之后等价的 test/LSP/shell 尝试会被 Runtime 拒绝。`uv run`、`npm --prefix`、`timeout`、`python -I -m pytest` 等包装形式会被识别；一条 shell 中含多个验证命令时会在执行前拒绝，避免验证探针级联消耗轮次和 API 余额。hard phase 还会拒绝带值的 write/apply/update/bless/accept/w/install-types 修复参数。

## 8. 关键配置文件和默认值

首次启动会在 `~/.config/deep-agent/` 创建并只补充缺失默认值，不覆盖已有用户值：

```text
config.yaml       Runtime、Model、Context、工具总开关、Memory、Daemon、权限和 Event
model.yaml        DeepSeek 模型覆盖
tools.yaml        Capability 元数据、权限和确认要求
memory.yaml       Memory 索引行为
mcp.yaml          MCP server 配置
secrets.env       私有 API Key，权限 0600
```

### 8.1 Model、Runtime、收敛和 Context

```yaml
model:
  provider: deepseek
  base_url: https://api.deepseek.com
  chat_path: /chat/completions
  model: deepseek-v4-pro
  context_window_tokens: 65536
  temperature: 0.2
  max_tokens: 4096
  api_key_env: DEEPSEEK_API_KEY
  reasoning_effort: null
  thinking: null
  timeout_seconds: 300
  network_retries: 2
  retry_base_seconds: 1.0
  routing:
    enabled: true
    tier: auto             # auto | fast | standard | deep
    fast_model: null
    standard_model: null
    deep_model: null

runtime:
  task_mode: auto          # auto | simple | standard | large | deep
  adaptive_thinking: true
  max_tool_rounds: 8
  max_tool_rounds_hard_limit: 32
  large_project_source_files: 500
  large_project_files: 2000
  progress_interval_seconds: 10
  show_thinking: true
  show_reasoning_content: true
  max_reasoning_display_chars: 4000
  max_user_request_chars: 250000
  auto_summarize: true
  write_lessons: true
  checkpoint_each_tool: true
  queue_stop_on_failure: true
  parallel_min_tasks: 8
  parallel_max_workers: 4
  capability_failure_threshold: 3
  convergence:
    enabled: true
    max_consecutive_exploration_rounds: 6
    reserved_tool_rounds: 4
    max_tool_calls_per_round: 16
    max_parallel_read_tools: 4
    max_length_continuations: 2
    max_implementation_evidence_reads: 2
    max_validation_attachment_reads: 2
    single_tool_result_chars: 12000
    same_round_tool_result_chars: 48000
    aggregate_tool_result_chars: 96000
    output_reserve_chars: 24000
    compacted_tool_result_chars: 1200
    keep_recent_tool_results: 4
    compaction_failure_limit: 3
    auto_compaction_enabled: true
    auto_compaction_max_tokens: 2048
    context_safety_buffer_tokens: 8192

context:
  max_files: 5000
  max_index_file_bytes: 1000000
  max_symbol_files: 500
  max_prompt_chars: 32000
  max_context_file_chars: 8000
  max_user_request_chars: 32000
  package_limits:
    simple: 12000
    standard: 32000
    large: 48000
    deep: 64000
  max_package_chars_hard_limit: 96000
  max_task_context_chars: 8000
  max_session_context_chars: 6000
  max_memory_context_chars: 8000
  max_capability_context_chars: 8000
  max_recovery_context_chars: 6000
  semantic_index_enabled: false
  semantic_languages:
    - python
    - javascript
    - typescript
    - tsx
    - java
    - go
    - rust
```

三个档位都是 DeepSeek 能力策略，不是多个 Provider。`fast_model`、`standard_model`、`deep_model` 默认为 `null`，会回落到 `model.model`；只填写已经由当前 DeepSeek API 确认可用的模型名。`provider` 不是 `deepseek` 时程序会拒绝启动。

`runtime.convergence.enabled: false` 会同时关闭收敛 nudge、同轮/历史 micro-compaction 和主动摘要，但单结果限制、私有附件配额、完整请求预算和紧急投影仍然生效。`auto_compaction_enabled: false` 只关闭主动 DeepSeek 摘要，适合余额紧张时减少额外请求。

### 8.2 工具、附件、LSP 和 HTTP

```yaml
tools:
  shell:
    enabled: true
    timeout_seconds: 120
  python:
    enabled: true
    timeout_seconds: 120
  git:
    enabled: true
    timeout_seconds: 120
  document:
    enabled: true
    timeout_seconds: 180
    max_input_bytes: 25000000
  ocr:
    enabled: true
    timeout_seconds: 180
  docker:
    enabled: true
    timeout_seconds: 180
  browser:
    enabled: true
    timeout_seconds: 180
    max_download_bytes: 100000000
  file:
    enabled: true
    max_file_bytes: 2000000
  template:
    enabled: true
    timeout_seconds: 300
    max_input_bytes: 67108864
  tool_result:
    enabled: true
    max_attachment_bytes: 8388608
    persist_threshold_bytes: 12000
    preview_chars: 12000
    max_read_chars: 32000
    max_attachments_per_session: 512
    max_session_bytes: 268435456
  http:
    enabled: false
    timeout_seconds: 30
    max_response_bytes: 1048576
    allowed_domains: []
  lsp:
    enabled: true
    timeout_seconds: 60
    max_diagnostics: 200
    auto_after_file_apply: true
```

HTTP 默认关闭。启用时必须配置域名 allowlist，只允许受限 GET/POST JSON，并继续经过 Capability Registry、Permission Manager、超时和 1 MiB 响应上限。

`lsp_diagnostics` 支持 `.py`、`.js`、`.jsx`、`.ts`、`.tsx`。Python 使用 Pyright，JavaScript/TypeScript 使用 `tsc --noEmit`；各引擎可独立降级。`file_apply` 成功与随后发现的诊断错误会分开表达。

### 8.3 权限、Event、Memory 和 Daemon

```yaml
permissions:
  enforce: true
  restrict_cwd_to_project: true
  deny_capabilities: []
  auto_approve_capabilities:
    - file.apply
    - file.undo
  yolo: false
  super_yolo: false

events:
  jsonl_log: true
  metrics_enabled: true

memory:
  retrieval_limit: 8
  vector_enabled: true
  smart_reflection: false
  dedupe_similarity: 0.94
  default_confidence: 0.7
  expiry_days: 365
  protect_kinds:
    - Correction
    - Decision

daemon:
  enabled: false
  poll_interval_seconds: 10
  memory_maintenance_seconds: 3600
  queue_enabled: false
  queue_timeout_seconds: 3600
```

安全模式默认要求确认；`--auto-approve` 只自动同意配置的快照型能力；`--yolo` 跳过确认但保留路径、危险命令、Docker 和 sudo 硬策略；`--super-yolo` 还会绕过 Permission Manager 硬限制，必须只用于明确授权的主机操作。Linux 自身权限和密码检查仍然有效。

Event Bus 是进程内同步总线，不是跨进程 Broker 或 exactly-once 系统。Session 和实际进入 ContextPackage 的 Memory usage 是 required；自动 Memory/Reflection、Capability Health、Audit、Metrics 和 UI progress 是 best-effort。Audit 只允许有界元数据，不记录 Prompt、reasoning、消息、AgentState、工具参数值、stdout/stderr、正文或凭据。

`agent memory maintain` 仅预览；加 `--apply` 才会归并高相似度记录并清理满足条件的过期记录。Correction 和 Decision 默认受保护。

Daemon 默认关闭。开启后按项目维护增量索引和 Memory；Queue 仍需显式设置 `daemon.queue_enabled: true`。

### 8.4 MCP

`~/.config/deep-agent/mcp.yaml` 的关键开关：

```yaml
mcp:
  enabled: false
  startup_timeout_seconds: 15
  call_timeout_seconds: 120
  resource_timeout_seconds: 60
  max_servers: 10
  max_tools: 80
  servers: []
```

MCP 支持 stdio、Streamable HTTP、SSE、tools/list、tools/call 和可选 resources/read。远端能力仍经过 Registry 和 Permission Manager；URL 凭据、重定向和未经明确配置的环境变量会被拒绝。不要把 DeepSeek Key 默认透传给 MCP server。

## 9. 文件修改、诊断与回滚快照

受管源码修改流程：

```text
file_diff 生成 unified diff 预览
  -> file_apply 校验原文件 SHA-256
  -> 创建 Session 快照
  -> 原子写入
  -> 校验结果
  -> 对支持的 Python/JS/TS 文件自动诊断
```

撤销最近一次受管修改：

```text
/undo
```

也可由模型调用 `file_undo`。如果快照后文件又被人工修改，Agent 会拒绝覆盖新内容。Git 项目只记录分支、HEAD 和状态，不会擅自 stash、切换或丢弃用户改动。

## 10. Resume 的能力与边界

schema 6 会保存 Session 身份、原目标、计划、阶段化模型计数、收敛状态、已见目标、压缩熔断和有界工具证据。Resume 会增加 turn，恢复上一轮错误，并重置本轮读取/停滞额度，使中断后的任务可以继续；原任务模式和模型档位只能保持或升级，不会因“继续”降级。

Resume 不是 Durable Intent Journal，也不是外部副作用的 exactly-once 重放。工具在外部系统已生效但回包前中断时，仍需人工核验。它也尚未等价复刻参考项目的消息链、snip/compact 边界和 replacement-decision 重放。

## 11. 参考项目与实现边界

固定参考为：

```text
项目：https://gitee.com/free/claude-code/tree/claude/
commit：b17913e26fd4278ad5cd4b32ed3bde86bf1444e9
```

本轮使用该固定 commit 的完整校验副本进行端到端差异审计，覆盖 Agent Loop、轮次终止、工具结果预算、micro/auto compaction 与熔断、工具编排、计划完成判定、测试命令选择、网络重试、Resume 和终端输出。参考影响包括：会话范围的本地工具结果落盘与模型可见预览、显式并发安全、工具调用/结果配对、grapheme 显示宽度以及有界压缩/恢复思路。权限、路径、符号链接、数量、总字节和哈希检查属于 Deep Agent 本地的私有附件设计，不能归因于参考项目。

必须区分以下本地设计：

- DeepSeek-only Python Runtime 和现有架构边界。
- soft/hard 工具轮次与预留实现/验证窗口。
- finish-reason 防御、DSML 正文拒绝和独立 final synthesis。
- conditional-mutation、artifact/Word 完成门和验证附件例外。
- 语义压缩失败后的 AgentState/evidence 确定性投影。

当前没有声称与参考项目完全等价。已知差异还包括：Resume 重放边界不同；JS/TS managed test runner 未完整读取 bun/pnpm/yarn 元数据；网络重试尚无参考项目的 `Retry-After`、jitter、heartbeat 和显式 chunk watchdog；通用 `verify` 状态仍主要由模型更新，只有 Word/artifact 等流程有更强工具证据门。

## 12. 验证方式与当前验收状态

不调用 DeepSeek 的离线验证命令：

```bash
cd ~/AI-Agent
.venv/bin/python -m pytest
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent tests scripts
git diff --check
.venv/bin/pip check
```

0.11.0 当前离线结果：

```text
454 tests collected；全量 pytest 通过
Agent Loop、终止、预算、附件、压缩、并发、中断、Resume、PTY、Unicode 回归通过
指定 artifact 路径、聚合执行证据门、soft→hard 延长、不可变工具归属和真实附件 fallback 回归通过
针对 artifact 跨分句误判、conditional-mutation 漏判和 single-validation wrapper 的 focused 回归通过
修复后确定性端到端：4 次 main-loop 请求 + 1 次 final synthesis 通过
Ruff check、Ruff format check、compileall、git diff --check、pip check 通过
18 项 CLI/Console 测试通过；真实 PTY 覆盖 40 列 Enter/空输入/help、Ctrl+C、Thinking，以及 25 列含旗帜、肤色、Keycap、ZWJ/组合字符的进度行
全新 Python 3.14 环境安装 `.[dev,browser,semantic]`，并通过同一 454 项全量回归与 pip check；仅安装 `.[dev]` 时，可选 Browser/Semantic 集成测试需要对应 extra
```

在线验收必须单独看待：

- Word 汇总案例已通过在线语义门。
- 约 2500 字文本总结案例已通过在线语义门。
- 已归档的 `final-prepublish-single-rerun` 大型 TypeScript 验收未通过：runner 记录 36 次逻辑请求、31 个工具轮次、1,477,342 tokens；仅 `session_completed` 门失败。
- 当前验收目录另有一条生成于发布提交之前的 `final-20260715T025700` 失败记录：38 次逻辑请求、32 个工具轮次、1,573,936 tokens。它没有保存 `git_head` 或源码包哈希，不能可靠归属到最终发布提交，也不能静默当作不存在。
- artifact 跨分句误判与 conditional-mutation 漏判修复后已通过确定性 4 main + 1 final synthesis 回归；最终提交 `113592c` 没有可明确归属的修复后大型在线成功记录。

因此，当前可以声明离线全量与 focused 回归通过，也可以声明 Word/文本在线案例通过；不能声明 0.11.0 大型 TypeScript 在线验收已全通过。

2026-07-15 的实例运行审计、已确认问题、修改位置和分版本建议位于：

```text
/mnt/d/detail/deepseek/项目运行审计与改进建议/20260715/
~/AI-Agent/user-docs/项目运行审计与改进建议/20260715/
```

分类后的离线实用案例位于：

```text
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.9.0/
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.9.1/
/mnt/d/detail/deepseek/测试与验收/实用案例/v0.10.0/
~/AI-Agent/user-docs/测试与验收/实用案例/
```

订单案例故意保留两个基线失败，用于观察定位和修复流程，不代表 Deep Agent 主测试失败。无需 API Key 的示例：

```bash
cd ~/AI-Agent
PYTHONPATH=. .venv/bin/python user-docs/测试与验收/实用案例/v0.9.1/interface-routing-demo.py
PYTHONPATH=. .venv/bin/python user-docs/测试与验收/实用案例/v0.10.0/event-runtime-demo.py
```

## 13. 风险、节省余额和回滚

- 主 Agent Loop、主动语义压缩、context overflow 恢复和 final synthesis 都可能产生 DeepSeek 请求。余额紧张时先完整运行离线测试，只在离线通过且确有必要时进行一次有界在线验收；可关闭 `runtime.convergence.auto_compaction_enabled` 减少主动摘要请求。
- 最新大型 TypeScript 案例修复后尚未再次在线验收，不能把确定性回归替代在线结论。
- 若安装了 Chroma 且保留当前默认 `memory.vector_enabled: true`，全新环境即使缺少 API Key，也可能在报告缺 Key 前下载约 79.3 MiB 的默认 ONNX 模型；不需要语义 Memory 时先关闭该开关。
- 当前直接任务 CLI 只在异常时返回非零；Session 以 failed/resumable 结束但 Runtime 正常返回文本时，进程仍可能返回 0。自动化必须同时检查 Session 终态或明确的未完成标记。
- 切换提交或更新 `pyproject.toml` 后应重新执行 `.venv/bin/pip install -e .`，避免 editable 安装元数据滞后于源码依赖声明。
- DeepSeek 网络重试有次数和退避上限，但目前没有 `Retry-After`、jitter、heartbeat 或显式 chunk watchdog。
- `show_reasoning_content` 会显示模型返回的 reasoning，可能较长；可关闭该开关。
- 私有附件有大小和总量上限，完整结果不一定全部进入模型上下文。
- `--super-yolo` 会绕过 Permission Manager；Shell/Python 仍是宿主机进程，应只对可信任务使用。
- Event Bus 和 Resume 都不提供外部副作用的 exactly-once 保证。

回滚到 v0.10.0：

```bash
cd ~/AI-Agent
git switch --detach v0.10.0
.venv/bin/pip install -e .
```

返回当前主线：

```bash
cd ~/AI-Agent
git switch main
.venv/bin/pip install -e .
```

0.11.0 的配置迁移只增加缺失默认值，不覆盖已有配置。不要为了回滚删除 Memory、`.project-agent`、Session 或私有工具附件。v0.10.0 不能直接 Resume schema 6 Session，会把它拒绝为未来 schema；应先在 v0.11.0 完成该 Session，或保留数据并在 v0.10.0 新建 Session。
