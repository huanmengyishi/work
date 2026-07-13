# DeepSeek Agent V3 使用说明（0.10.0）

更新时间：2026-07-14

## 1. 系统定位

Deep Agent 是安装在 WSL Ubuntu 中的项目型 CLI Agent。Agent 是工具，当前项目目录才是 Workspace。

```bash
cd /任意/项目目录
agent
```

首次运行会在项目根目录创建 `.project-agent/`，保存项目上下文、索引、会话、快照和缓存。程序、配置、全局数据与项目数据相互分离：

```text
~/AI-Agent/                         程序、测试、发布说明
~/.config/deep-agent/              配置和 API Key
~/.local/share/deep-agent/         SQLite、Chroma、日志、指标、健康与 Daemon 状态
<项目>/.project-agent/             项目上下文、索引、Session、快照
```

## 2. DEEPSEEK_API_KEY 设置位置

推荐且程序会自动读取的位置：

```text
~/.config/deep-agent/secrets.env
```

编辑方法：

```bash
nano ~/.config/deep-agent/secrets.env
```

单个 Key：

```bash
DEEPSEEK_API_KEY=sk-your-key
```

多个 Key 可以使用英文逗号或中文逗号分隔：

```bash
DEEPSEEK_API_KEY=key_1,key_2,key_3
```

程序会自动去除空格、空值和重复 Key，并在 `401`、`403`、`429` 时切换下一个 Key。不要把 Key 放进项目目录、Git、README 或聊天记录。

设置后执行：

```bash
chmod 600 ~/.config/deep-agent/secrets.env
agent doctor --online
```

本次未读取或执行真实 Key 在线验证。用户可运行 `agent doctor --online`，程序只显示数量和状态，不输出 Key 内容。

## 3. 常用启动方式

```bash
cd /mnt/d/detail/deepseek
agent init
agent
agent "分析当前项目并给出修改建议"
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
/clear
/exit
```

普通任务输入完成后按一次 `Enter` 即提交。终端随后会显示“正在处理请求，请稍候...”，这表示 Agent 已开始调用 DeepSeek 和工具，不再等待继续输入。空回车不会执行任务，会提示正确用法。运行中需要返回交互界面时按 `Ctrl+C`，随后可 `/resume` 继续会话。

0.9.0 提交后会持续显示 `Thinking` 已用秒数、任务模式、DeepSeek 推理档位、模型轮次、Task Graph 当前步骤和工具状态。DeepSeek thinking 模式返回 `reasoning_content` 时会边生成边显示，不再长时间无输出。若流式响应在已经输出部分内容后断开，Agent 不会自动重复请求，以免重复工具调用；它会保存可恢复 Session 并给出 Session ID。

安全模式是默认模式。`--yolo` 自动同意普通工具调用，但仍受路径、危险命令和 sudo 策略保护。`--super-yolo` 绕过 Permission Manager 的硬限制，可允许 sudo、外部路径、特权 Docker 和破坏性命令；操作系统自身的密码和权限检查仍然有效。

```bash
agent --yolo
agent --super-yolo
```

永久开关位于 `~/.config/deep-agent/config.yaml`：

```yaml
permissions:
  yolo: false
  super_yolo: false
```

## 4. 0.5.0、0.6.0、0.7.0 演进

### 0.5.0：安全执行与外部连接基线

发布原因：先建立可信修改闭环，再扩展外部能力。

主要能力：`file_diff -> file_apply -> file_undo`、Git/文件快照、MCP stdio/HTTP/SSE、MCP Resources、浏览器持久会话与下载、受限 HTTP、Queue、8 任务阈值的 Git worktree 并行、Memory 管理和纠错学习。

后续方向：从“工具齐全”转向“统一决策状态”。

### 0.6.0：决策智能

发布原因：让 Resume、Queue、Parallel 共用同一 Task Graph 和 AgentState，而不是各自维护状态。

主要能力：依赖感知 Planner、Workspace Memory、Reflection、Execution Context、Capability Health。

后续方向：提高修改代码后的即时反馈和大型项目长期维护能力。

### 0.7.0：深度代码理解与长期维护

发布原因：代码修改后需要立即诊断，Memory 需要去重与生命周期，Session 和项目索引需要长期运行而不无限膨胀。

主要能力：

- Python 使用 Pyright，JavaScript/TypeScript 使用 `tsc --noEmit`，每个诊断引擎可独立降级。
- `file_apply` 成功后自动诊断，写入成功与代码诊断错误分开表达。
- Tree-sitter 旁路索引增加模块摘要、导出符号和内部 import 关系。
- Memory 增加可信度、使用次数、最后使用时间、过期时间和归并关系。
- `/resume` 压缩历史工具消息，保留上次结果、AgentState、Execution Context、当前上下文和 Memory。
- 可选 Daemon 负责增量索引和 Memory 整理，默认关闭。
- SQLite WAL、FTS 回填、Queue 跨进程锁和更严格的日志脱敏。

后续方向：V1.0 只作为规划，不在当前版本实施。目标是冻结 Runtime 接口、让 Context Builder 成为唯一上下文入口、完成 Event Bus 副作用闭环，并保持所有新能力经过 Capability Registry 与 Permission Manager。

### 0.7.1：交互输入可靠性补丁

发布原因：WSL/GNU Readline 彩色 Prompt 中的 ANSI 控制字符未声明为不可见字符，在窄终端或自动换行时会导致光标位置计算错误，表现为按 Enter 后看起来仍在输入。

主要修复：

- Readline 模式下正确标记 ANSI 颜色控制字符，不参与光标和换行长度计算。
- 输入前刷新输出，确保 Prompt 已显示后才开始读取。
- 空回车显示明确提示，不再静默重新等待。
- 任务按 Enter 提交后立刻显示“正在处理请求”，避免误以为仍在输入状态。
- 运行中按 `Ctrl+C` 返回交互界面，可用 `/resume` 继续已检查点保存的会话。
- 项目根目录识别仅接受有效 Git 元数据，不会再被空 `.git` 占位目录错误劫持。

### 0.8.0：自适应深度执行与可见 Thinking

发布原因：简单问题不应承担复杂任务的成本；大规模文本/代码和困难问题又不能塞进一次无界推理，否则容易超时、静默等待或遗漏范围。

执行模式：

```text
simple    简单事实问答，关闭 thinking，最多 4 轮
standard  普通分析/编码，高强度 thinking，默认 8 轮
large     整仓库/长文档，分块扫描 + Task Graph，最多 16 轮
deep      审计/重构/根因分析，max thinking，最多 24 轮
```

模式由本地规则选择，不额外消耗一次 API 请求。`large/deep` 会自动建立 `scope -> inspect-chunks -> implement/synthesize -> verify` 的依赖计划；读取文件、Prompt、工具输出、轮次和可见推理都有上限。短句“继续”恢复时会保留原来的 deep 策略和计划。

配置：

```yaml
runtime:
  task_mode: auto       # auto / simple / standard / large / deep
  adaptive_thinking: true
  max_tool_rounds: 8
  max_tool_rounds_hard_limit: 32
  large_project_source_files: 500
  large_project_files: 2000
  progress_interval_seconds: 10
  show_thinking: true
  show_reasoning_content: true

model:
  timeout_seconds: 300
  network_retries: 2
  retry_base_seconds: 1.0
```

网络超时、`408` 和临时 `5xx` 会对同一个 Key 做有限指数退避；`401/403/429` 仍切换下一个 Key。

并发和输入边界也已加固：首次 Project 初始化使用项目锁，同一 Session 不能被两个进程同时 Resume，Context/Workspace 缓存原子写入；Daemon 精确校验 argv 与进程 starttime，Queue 超时清理整个进程组；Shell 输出、HTTP 请求体/Header、文档输入、浏览器下载和 MCP 分页都有硬上限。

注意：普通/YOLO 的 Docker 会拒绝 host root/socket/device/namespace。Shell/Python 仍是宿主机进程，程序能限制请求里的工作目录并拦截已知危险模式，但命令文本内部的绝对路径最终由 Linux 权限而非项目级 OS 沙箱隔离。处理不可信任务时使用默认安全模式。

### 0.9.0：统一上下文、任务路由与 DeepSeek 模型路由

0.9.0 将模型前的决策拆成三个本地、确定性的步骤：

```text
用户请求
  -> Task Router：任务类型、规模、风险、simple/standard/large/deep
  -> Model Router：DeepSeek fast/standard/deep 档位
  -> Context Builder：统一生成受限 ContextPackage
  -> Prompt Renderer -> Agent Runtime -> Capability -> Permission
```

Task Route 和 Model Route 会写入 Session。`/resume` 的模式和精确模型只能保持或升级，“继续”等短输入不会把深度任务降为简单任务。模型不能自行选择模型，也没有引入其他 Provider。

ContextPackage 统一装配 Task、Execution、Session、项目说明、README/配置、Workspace、Semantic、Memory 和 Capability 摘要。默认字符预算按模式为 12000/32000/48000/64000，硬上限 96000；标题和分隔符也计入。只有实际进入 Package 的 Memory 才增加使用次数。失败恢复 Memory 使用每轮总计 6000 字符的有界 delta，不再无限追加。

模型路由配置位于 `~/.config/deep-agent/config.yaml` 或 `model.yaml`：

```yaml
model:
  provider: deepseek
  model: deepseek-v4-pro
  routing:
    enabled: true
    tier: auto              # auto / fast / standard / deep
    fast_model: null
    standard_model: null
    deep_model: null

runtime:
  max_user_request_chars: 250000

context:
  max_user_request_chars: 32000
  package_limits:
    simple: 12000
    standard: 32000
    large: 48000
    deep: 64000
  max_package_chars_hard_limit: 96000
  max_recovery_context_chars: 6000
```

三个档位默认都安全回落到 `model.model`。程序不会猜测或内置未经当前 API 验证的“快速模型”名称；只有确认某个 DeepSeek 模型可用后，才应填写对应 `*_model`。`provider` 不是 `deepseek` 时会直接拒绝启动。ContextPackage 预算包含有界的用户请求，但不包含固定 System Prompt、单独发送的 Tool Schema 和后续 ToolResult；这些输入仍由各自上限控制。超过 `runtime.max_user_request_chars` 的粘贴输入会被拒绝，并提示先保存为项目文件，再由 large/deep 模式完整分块读取，避免静默丢失大段正文。

### 0.9.1：核心接口稳定化

0.9.1 不做大规模 Runtime 重构，而是为 v1.0 冻结现有边界：

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ContextBuilder -> ContextPackage -> PromptBuilder
```

关键变化：

- ContextBuilder 是进入模型前唯一的上下文选择入口。
- PromptBuilder 只接受一个 `ContextPackage`，不再接收 State、Snapshot、Memory 或能力摘要等分散参数，也不读取文件。
- AgentState 增加 schema 常量、`validate()` 和冻结身份字段；旧 Session 可兼容恢复，未来未知 schema 会明确拒绝。
- TaskRouter 是唯一分类器；计划模板只消费 TaskRoute，不再重复扫描 Prompt。旧 `TaskStrategySelector` 仅作 deprecated 兼容层。
- Model Router 增加可解释成本级别：简单低风险为 `low`，普通及大型只读任务为 `balanced`，deep/高风险/架构重构/重复失败为 `high`。这只是本地资源选择提示，不是实际账单统计。
- Event Bus 增加稳定事件字段、`run_id`、订阅取消和订阅者异常隔离。当前仍是进程内同步总线，不提供持久化重放或跨进程保证。

DeepSeek 仍是唯一 Provider。没有加入 OpenAI、Anthropic 或其他模型；三个档位仍只使用用户配置并确认可用的 DeepSeek 模型名。

外部扩展若还调用旧 `PromptBuilder.append_resume()`，或向 `build_initial()` 传入分散参数，需要改为先构建 ContextPackage。程序仓库内部调用已全部迁移。

### 0.10.0：Event Bus 整体迁移

0.10.0 在保持上述接口与 DeepSeek 唯一 Provider 不变的前提下，将 Runtime 自动副作用统一注册到 `RuntimeEventPipelines`：

```text
EventBus
  -> required：Session checkpoint/finalize、Context Memory usage
  -> best-effort：自动 Memory/Reflection、Capability Health
  -> best-effort：Audit、Metrics、UI Thinking/Progress
```

关键语义：

- required 事件必须有精确 owner；缺失或写入失败会 fail-closed。
- Session owner 是否已提交通过命名 delivery 结果区分；owner 未写入时不会从不确定状态继续 finalize，只有 owner 已成功而其他 required observer 故障时才可安全落 failed terminal。
- Session 必须先 finalize，之后才发布 `task.finished/task.failed`，避免从未持久化终态生成 Memory 或指标。
- 实际进入 ContextPackage 的 Memory 使用 SQLite `usage_id` 原子去重；成功后才更新 AgentState，Resume 的新 turn 可再次强化一次。
- ToolManager 不再直写 Capability Health；权限检查和 handler 完成后才发布 `tool.finished`。Health 写失败不会改变已得到的 ToolResult。
- Audit 只记录有界元数据，不记录 Prompt、reasoning、messages、AgentState、工具参数值、stdout/stderr、正文或凭据。
- Metrics 只记录公开 task/model/tool 事件的计数、总工具耗时和失败数；64 KiB 以上旧指标文件不会解析。
- Thinking 片段经 `ui.progress.updated` 实时交给 ConsoleUI，但 Audit 丢弃内容、Metrics 不统计，UI 故障不会中断任务。

配置位于 `~/.config/deep-agent/config.yaml`，本次只补默认值，不覆盖已有配置：

```yaml
events:
  jsonl_log: true
  metrics_enabled: true
```

运行数据位置：

```text
~/.local/share/deep-agent/logs/       元数据审计 JSONL
~/.local/share/deep-agent/metrics/    每项目聚合指标 JSON
~/.local/share/deep-agent/capability-health/
```

Event Bus 仍是进程内同步总线，不提供跨进程 Broker、重放或进程崩溃后的 exactly-once。需要可靠重放的工具副作用应在后续实现独立 Durable Intent Journal，而不能让 Event 绕过 Permission Manager。完整开发边界见 `docs/architecture-v0.10.0.md`。

## 5. 安全文件修改与回滚

Agent 的源码修改流程：

```text
file_diff 生成 unified diff 预览
    -> file_apply 校验原文件哈希
    -> 创建 Session 快照
    -> 原子写入
    -> 校验写入结果
    -> 对 Python/JS/TS 自动运行诊断
```

回滚：

```text
/undo
```

或模型调用 `file_undo`。如果文件在快照后又被人工修改，Agent 会拒绝覆盖更新内容。

## 6. LSP Diagnostics

手动诊断能力名为 `lsp_diagnostics`，支持 `.py`、`.js`、`.jsx`、`.ts`、`.tsx`。

依赖安装位置：

```text
~/.local/share/deep-agent/node-tools
```

当前工具：Pyright 1.1.411、TypeScript 7.0.2、typescript-language-server 5.3.0。

诊断返回文件、行、列、严重级别、错误代码和消息。扫描会跳过 `.git`、`.project-agent`、虚拟环境、`node_modules`、`dist`、`build` 等目录。

配置：

```yaml
tools:
  lsp:
    enabled: true
    timeout_seconds: 60
    max_diagnostics: 200
    auto_after_file_apply: true
```

## 7. Semantic Context

默认关闭。在 `~/.config/deep-agent/config.yaml` 中启用：

```yaml
context:
  semantic_index_enabled: true
  semantic_languages:
    - python
    - javascript
    - typescript
    - java
    - go
    - rust
```

生成文件：

```text
.project-agent/index.semantic.json
```

它不会替换轻量 `index.json`。Prompt 只加载受限摘要，完整索引留在文件中；0.9.0 还会把 Semantic 与其他来源一起纳入 ContextPackage 总预算，避免长 README 将其静默挤掉。

## 8. Memory 生命周期

查看和维护：

```bash
agent memory list
agent memory search "docker proxy"
agent memory stats
agent memory edit 123
agent memory delete 123
agent memory maintain
agent memory maintain --apply
```

`maintain` 默认只预览。`--apply` 才执行：

- 合并高相似度的 Correction、Lesson、Reflection。
- 合并标签、使用次数和可信度，并保留 `merged_into` 追踪关系。
- 删除已过期、低可信度且非保护类型的 Memory。
- Correction 和 Decision 默认不会自动过期。

配置：

```yaml
memory:
  dedupe_similarity: 0.94
  default_confidence: 0.7
  expiry_days: 365
  protect_kinds:
    - Correction
    - Decision
  smart_reflection: false
```

## 9. Daemon

Daemon 默认关闭，只在需要后台增量维护时启动：

```bash
cd 项目目录
agent daemon start
agent daemon status
agent daemon stop
```

功能：轮询文件变化、刷新 `index.json`、Workspace Memory 和可选语义索引，定期执行 Memory 生命周期维护。PID、锁、状态和日志位于：

```text
~/.local/share/deep-agent/daemon/<ProjectID>/
```

配置：

```yaml
daemon:
  enabled: false
  poll_interval_seconds: 10
  memory_maintenance_seconds: 3600
  queue_enabled: false
```

`queue_enabled` 默认关闭。开启后 Daemon 才会寻找 `pending` Queue，并使用安全的 `--auto-approve` 模式执行。Queue 自带跨进程锁，防止前台与后台重复运行。

单个后台 Queue 的绝对超时由 `daemon.queue_timeout_seconds` 控制，默认 3600 秒，避免任务数过多时无限等待。

## 10. MCP、HTTP、Browser 与 OCR

MCP 配置：

```text
~/.config/deep-agent/mcp.yaml
```

MCP 默认关闭，支持 stdio、Streamable HTTP、SSE、tools/list、tools/call 和可选 resources/read。远端工具仍通过 Capability Registry 和 Permission Manager。

受限 HTTP 默认关闭，启用时必须配置域名白名单、30 秒超时和 1 MiB 响应限制。

浏览器持久 Session：

```text
.project-agent/browser-sessions/<session_name>/
```

下载：

```text
.project-agent/downloads/<session_name>/
```

OCR/文档统一调用 `document.parse()`，优先复用已有 `~/.local/bin/ai-parser`，并可降级到 pdftotext、Tesseract、ImageMagick。模型最终只处理 Markdown。

## 11. Queue 与 Parallel

```bash
agent queue "任务一" "任务二"
agent queue list
agent queue show --id QUEUE_ID
agent queue resume --id QUEUE_ID
```

Parallel 仅在至少 8 个明确独立任务时启用，要求干净 Git 工作树。每个任务在临时 worktree 中运行，补丁逐个 `git apply --check` 后应用，失败或冲突不会直接污染主工作区。

## 12. 健康检查与故障定位

```bash
agent --version
agent doctor
agent doctor --online
agent health
agent health --reset lsp.diagnostics
agent health --reset
```

能力状态：Available、Unavailable、Need Config、Disabled、Broken。Unavailable/Broken 能力不会放入模型 Tool Schema 和 Prompt 能力摘要。

当前 0.10.0 本地验收：

```text
173 tests passed（含真实 PTY Thinking 流式显示）
Ruff check passed
Ruff format check passed
compileall passed
ContextPackage 总预算、完整边界截断、私有 Memory 不落盘 passed
Task/Model 路由、失败升级、Resume 单调保持 passed
DeepSeek streaming/tool-call assembly passed
网络重试与部分流禁止重放 passed
路径、符号链接、Queue 并发、Docker 参数与进程组超时回归 passed
Interface Contract、Event、AgentState、ContextPackage、路由与 Permission 顺序 passed
Session required owner/顺序/失败恢复、Memory usage 幂等与 Resume passed
Capability Health best-effort、Audit/Metrics 隐私与畸形数据 passed
GitHub Actions：Python 3.11 / 3.12 / 3.13；发布后核验运行号
Actions：checkout@v5、setup-python@v6，使用当前 Node.js runtime，无 Node.js 20 弃用警告
```

## 13. 参考工程取舍

参考了 `https://gitee.com/free/claude-code`。采纳：有界上下文、资源路径/大小/数量审计、能力降级可观测、可恢复状态、显式 Tool Loop。

未采纳：第二套 Java/Spring Runtime、多模型供应商抽象、全屏 TUI、JAR 插件体系、任意 Hook、遥测、无限制 Skills/Agents 加载。这些内容会破坏当前“DeepSeek 唯一模型”和 `CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission` 边界。

## 14. 新功能扩展规则

以后新增工具时：

1. 在 `agent/tools/` 创建单一入口。
2. 接受 `ToolRequest`，返回 `ToolResult`。
3. 在 Capability Registry 注册参数、权限、超时、输入输出格式和可用性。
4. 通过 Permission Manager，不允许 Runtime 或模型直接 `subprocess` 绕过工具层。
5. 为成功、失败、超时、路径越界和缺少依赖添加测试。
6. 副作用通过 Event Bus 记录，敏感值必须脱敏。
7. 重型功能默认关闭，配置迁移只增加新默认值，不覆盖用户值。

## 15. 可直接运行的实用案例

当前目录新增：

```text
实用案例-v0.9.0/
实用案例-v0.9.1/
实用案例-v0.10.0/
```

其中 `order-summary-demo/` 是一个带真实 CSV、业务规则、故意保留缺陷和回归测试的小型订单汇总项目。先阅读 `实用案例-v0.9.0/README.md`，再依次体验 simple 解释、standard Bug 修复、large 全项目分析、deep 财务/安全审计与 `/resume`。案例基线的 2 个测试应当失败，这是用于观察 Agent 定位和修复过程的预期状态，不属于 Deep Agent 主项目测试失败。

快速开始：

```bash
cd /mnt/d/detail/deepseek/实用案例-v0.9.0/order-summary-demo
python3 -m unittest discover -s tests -v
agent
```

0.9.1 的 `interface-routing-demo.py` 不需要 API Key，可直接观察
TaskRouter、cost-aware ModelRouter、TaskRoute-only 计划工厂和 Event Bus：

```bash
cd ~/AI-Agent
PYTHONPATH=. .venv/bin/python user-docs/实用案例-v0.9.1/interface-routing-demo.py
```

0.10.0 的 `event-runtime-demo.py` 同样不需要 API Key，可观察 required owner、best-effort 故障隔离、Memory usage 幂等、安全 Audit 和聚合 Metrics：

```bash
cd ~/AI-Agent
PYTHONPATH=. .venv/bin/python user-docs/实用案例-v0.10.0/event-runtime-demo.py
```

## 16. 风险与回滚

- `show_thinking` 展示的是 DeepSeek API 返回的 reasoning 内容，可能较长；可在配置中关闭。
- 已输出部分 reasoning 后的流断线不会自动重放，需要 `/resume`，这是避免重复工具副作用的安全设计。
- `--super-yolo` 仍会绕过 Permission Manager；普通模式和 `--yolo` 已增加 Docker host/root/socket/device 防护。

回滚到上一版不会删除配置、Memory 或项目数据：

```bash
cd ~/AI-Agent
git switch --detach v0.9.1
.venv/bin/pip install -e .
```

恢复最新版执行 `git switch main`。0.10.0 仅新增默认值、指标文件和 SQLite 幂等表，不覆盖配置，也不删除 Session、Memory 或项目数据；0.9.1 会安全忽略这些新增数据。
