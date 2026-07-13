# DeepSeek Agent V3 使用说明（0.7.0）

更新时间：2026-07-13

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
~/.local/share/deep-agent/         SQLite、Chroma、日志、备份、Daemon 状态
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

当前验收结果：5/5 Key 可用。程序只显示数量和状态，不输出 Key 内容。

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

它不会替换轻量 `index.json`。Prompt 只加载受限摘要，完整索引留在文件中，最终 Context 长度仍受 `context.max_prompt_chars` 限制。

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

当前 0.7.0 验收：

```text
55 tests passed
Ruff check passed
Ruff format check passed
compileall passed
DeepSeek online check: 5/5 keys ready
Docker hello-world passed
Daemon run-once cleanup passed
All enabled capabilities healthy
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
