# DeepSeek Agent V3 0.5.0 使用说明

本文是当前 Deep Agent 的主使用入口。程序安装在 `~/AI-Agent`，当前目录用于
查看说明、保存工作日志和提出后续需求。Agent 是工具，执行 `agent` 时所在的项目
目录才是 Workspace。

## 一、当前能力

当前版本：`0.5.0`。

- 任意项目目录启动、项目注册、移动后 UUID 保持。
- DeepSeek Key 池、中文逗号/英文逗号解析和失败轮换。
- Runtime、AgentState、Prompt Builder、Context Builder、Session Resume。
- SQLite FTS + Chroma Memory、自动 Summary/Lesson/Bug/Decision。
- 用户纠错写入 Correction、工具失败后的本地经验自愈检索。
- Memory 列出、编辑、删除和统计。
- 安全文件预览、SHA-256 冲突保护、快照应用与撤销。
- Shell、Python、Git、Docker、OCR、文档、Playwright 和安全模板。
- MCP stdio、Streamable HTTP、legacy SSE、Tools 和 Resources。
- 受限 `http_request`，域名白名单、30 秒和 1 MiB 上限。
- 可选 Tree-sitter 语义旁路索引。
- 持久串行任务队列和满足门槛时的 Git worktree 并行执行。
- Safe、Auto approve、YOLO、SUPER YOLO 四级批准模式。

未实现：MCP Prompts/Subscriptions、Web UI、通用 Multi-Agent、自动语义重构图。

## 二、目录与数据

```text
~/AI-Agent/                         程序、测试、维护文档
~/.config/deep-agent/              配置和 secrets.env
~/.local/share/deep-agent/         SQLite、Chroma、日志、备份、临时 worktree
<项目>/.project-agent/             项目上下文、索引、会话和运行状态
```

项目私有目录：

```text
.project-agent/
  context.md                       人工维护的长期项目事实
  architecture.md                  架构说明
  todo.md                          项目 TODO
  index.json                       轻量文件和符号索引
  index.semantic.json              可选 Tree-sitter 语义索引
  sessions/                        会话 JSON/Markdown
  snapshots/                       文件修改快照
  browser-sessions/                Playwright 登录状态
  downloads/                       浏览器下载
  queues/                          持久任务队列
  parallel/                        并行 patch 和报告
```

这些运行数据已加入 `.project-agent/.gitignore`。项目位于 `/mnt/d` 时 DrvFS 可能
显示权限 `777`；高敏感浏览器身份建议放到 `~/Projects` 或启用 DrvFS metadata。

## 三、DeepSeek API Key

API Key 推荐放在：

```text
~/.config/deep-agent/secrets.env
```

```bash
nano ~/.config/deep-agent/secrets.env
```

单 Key 或 Key 池：

```bash
DEEPSEEK_API_KEY=key_1,key_2,key_3
```

支持英文逗号 `,` 和中文逗号 `，`，自动去空格、空项和重复值。HTTP `401`、
`403`、`429` 时尝试下一个 Key，日志不显示 Key 内容。

```bash
chmod 600 ~/.config/deep-agent/secrets.env
agent doctor --online
```

不要把 Key 放进源码、项目、`model.yaml`、README、Git 或 MCP 配置示例。

## 四、基础使用

```bash
agent --version
agent doctor
agent doctor --online
cd /path/to/project
agent init
agent "分析当前项目"
agent
```

交互命令：

```text
/new
/resume [session-id]
/sessions
/status
/undo [snapshot-id]
/yolo on|off
/super-yolo on|off
/help
/clear
/exit
```

会话：

```bash
agent sessions
agent resume "继续最近任务"
agent resume --session SESSION_ID "继续测试"
```

上下文：

```bash
agent context show
agent context refresh
agent context index
```

## 五、纠错学习与 Memory

当用户明确否定 Agent 的事实或行为时，System Prompt 要求模型修正后调用
`memory_add` 写入 `Correction`，并携带 `correction:<topic>` 与项目名标签。
程序侧会拒绝缺少 `correction:*` 标签的 Correction，且自动补项目名标签。

例子：

```text
用户：不对，这个服务端口是 8080，不是 8000。
Agent：修正回答，并记录 Correction：correction:port。
```

工具返回失败后，Runtime 从当前项目和全局 Memory 中搜索与错误关键字相关的
Correction/Lesson，以 `Failure Recovery Memory` 加入下一轮 Prompt。只做本地
SQLite 检索，不增加 DeepSeek API 调用；同一条经验每次执行最多补注一次。

管理命令：

```bash
agent memory search "端口错误"
agent memory list --limit 50
agent memory list --kind Correction
agent memory list --tag correction:port
agent memory stats
agent memory edit 123
agent memory edit 123 --content "修正后的内容" --tag correction:port --tag 项目名
agent memory delete 123
```

不带参数的 `edit` 使用 `$EDITOR`，内容是临时 YAML；也可使用 flags 非交互修改。
编辑和删除同步 SQLite FTS 与 Chroma。

手工新增 Correction：

```bash
agent memory add Correction "服务端口" "该服务使用 8080" \
  --tag correction:port
```

## 六、安全文件修改与权限

源码修改协议：

```text
file_diff -> file_apply -> file_undo
```

`file_diff` 只生成预览；`file_apply` 校验原文件 SHA-256、保存 Session 快照并原子
写入；`file_undo` 在文件仍匹配 Agent 应用版本时回退，避免覆盖后续人工修改。

| 模式 | 启动方式 | 行为 |
|---|---|---|
| Safe | `agent` | 高风险工具逐次确认，权限策略生效。 |
| Auto approve | `agent --auto-approve` | 默认只自动批准快照支持的 apply/undo。 |
| YOLO | `agent --yolo` | 跳过确认，仍拦截 sudo、项目外 cwd、特权 Docker。 |
| SUPER YOLO | `agent --super-yolo` | 跳过确认并绕过 Permission Manager 硬策略。 |

持久开关位于 `~/.config/deep-agent/config.yaml`：

```yaml
permissions:
  yolo: false
  super_yolo: false
```

SUPER YOLO 允许发起 `sudo`，但不保存、不猜测、不绕过 sudo 密码；操作系统认证
仍然生效。默认保持关闭，只对明确任务临时开启。

## 七、MCP

配置：

```text
~/.config/deep-agent/mcp.yaml
```

状态：

```bash
agent mcp status
agent mcp tools
agent mcp config
```

MCP 整体和每个服务器默认关闭，默认最多 10 个服务器、80 个能力。支持：

- `stdio`：本地子进程。
- `streamable_http`：HTTP POST JSON-RPC、Session ID、JSON/SSE 响应。
- `sse`：legacy GET event stream + POST endpoint。
- Tools：`tools/list`、`tools/call`。
- Resources：显式 `resources_enabled: true` 后注册 `resources/read` 能力。

Streamable HTTP 示例：

```yaml
mcp:
  enabled: true
  servers:
    - name: knowledge
      enabled: true
      transport: streamable_http
      url: https://mcp.example.com/mcp
      headers: {}
      tool_allowlist:
        - search_*
      resources_enabled: true
      resource_uri_allowlist:
        - docs://public/*
```

legacy SSE 将 `transport` 改为 `sse`，`url` 指向事件流。HTTP 传输拒绝嵌入 URL
凭据和自动重定向。stdio MCP 不继承 `DEEPSEEK_API_KEY`，除非服务器配置明确写入
`env_passthrough`。`mcp.yaml` 权限为 `600`。

## 八、受限 HTTP 工具

默认关闭。在 `~/.config/deep-agent/config.yaml` 中设置唯一开关和域名白名单：

```yaml
tools:
  http:
    enabled: true
    timeout_seconds: 30
    max_response_bytes: 1048576
    allowed_domains:
      - api.example.com
```

工具 `http_request` 只支持 GET 和 POST JSON，超时最多 30 秒，响应最多 1 MiB。
子域名匹配允许项；拒绝 URL 内凭据、Authorization/Cookie/API-Key header 和自动
重定向。该工具与 MCP 互补，适合没有 MCP Server 的临时 API。

## 九、可选语义索引

已安装 `tree-sitter-language-pack`，默认关闭。启用：

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

然后：

```bash
agent context refresh
```

结果写入 `.project-agent/index.semantic.json`，包含类、函数/方法层级、行号和 import
来源；摘要会加入项目 Prompt。它不替换 `index.json`，缺少 grammar 或解析失败时
不阻断 Agent。

## 十、任务队列

直接执行多个串行任务：

```bash
agent queue "任务一" "任务二" "任务三"
```

显式形式与管理：

```bash
agent queue run "任务一" "任务二"
agent queue list
agent queue show --id QUEUE_ID
agent queue resume --id QUEUE_ID
agent queue --continue-on-error "任务一" "任务二"
```

每个任务创建独立 Session，队列 JSON 位于 `.project-agent/queues/`。默认失败即暂停；
Ctrl+C 会将当前项标为 paused，resume 跳过已完成项并从未完成项继续。

## 十一、Git Worktree 并行

只有至少 8 个明确独立任务时允许：

```bash
agent parallel --workers 4 \
  "任务1" "任务2" "任务3" "任务4" \
  "任务5" "任务6" "任务7" "任务8"
```

前置条件：当前项目是 Git 仓库，除 `.project-agent` 外工作区干净。每项创建独立
临时分支和 worktree，默认子 Agent 使用 `--auto-approve`。如果顶层使用 `--yolo`
或 `--super-yolo`，风险级别会传给子 Agent。

每项从基线提交生成 binary patch，主工作区逐项运行 `git apply --check` 后应用。
冲突不会强制覆盖；patch 和 `report.json` 保存在 `.project-agent/parallel/<run-id>/`。
worktree 与临时分支最终清理。此功能不会自动判断任务是否真的独立，用户必须确保
任务修改范围不重叠。

## 十二、完整工作流

```text
cd 项目 -> agent
  -> 加载 XDG 配置和 Key 池
  -> 识别/初始化项目并更新 Project Registry
  -> 构建轻量索引和可选语义索引
  -> SQLite FTS + Chroma 检索 Memory
  -> 创建 AgentState 和 Session 检查点
  -> Prompt Builder 组装 Context/Memory/Tools/User
  -> DeepSeek 推理并产生 ToolRequest
  -> Capability Registry + Permission Manager
  -> 本地/MCP/HTTP/浏览器工具执行
  -> 失败时检索 Correction/Lesson 自愈上下文
  -> 保存 Session、Summary 和经验
  -> 更新 SQLite、Chroma 和 Markdown Memory
```

## 十三、维护与验证

```bash
cd ~/AI-Agent
.venv/bin/python -m pytest -q
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent scripts
```

2026-07-13 最终结果：

```text
版本: 0.5.0
pytest: 40 passed
Ruff lint/format: passed
compileall: passed
agent doctor --online: 5/5 keys ready
MCP stdio / Streamable HTTP / legacy SSE / Resources: passed
HTTP allowlist / size / timeout / redirect policy: passed
Memory CRUD / Correction / failure recovery: passed
Tree-sitter semantic sidecar: passed
Queue resume / Git worktree threshold and patch merge: passed
```

详细实施取舍见 `DeepSeek-Agent-V3-工作日志.md`。源码维护说明见
`~/AI-Agent/docs/implementation.md`。

## 十四、提出后续需求

```text
目标：增加什么能力？
触发：哪个 CLI、工具或事件触发？
输入：读取哪些文件、服务和参数？
输出：修改、显示或记住什么？
权限：哪些动作默认关闭、需确认或允许 YOLO？
恢复：失败、中断、冲突如何恢复？
记忆：项目经验还是全局经验？
示例：给出一个真实使用案例。
```
