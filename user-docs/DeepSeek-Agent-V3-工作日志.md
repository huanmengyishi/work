# DeepSeek Agent V3 0.5.0 工作日志

## 一、任务信息

```text
实施日期：2026-07-13
程序目录：~/AI-Agent
说明目录：/mnt/d/detail/deepseek
升级前版本：0.3.2
升级后版本：0.5.0
模型提供方：DeepSeek
运行环境：WSL2 Ubuntu
```

目标是按“记忆闭环 -> 外部连接与语义理解 -> 编排与隔离执行”的顺序实现
`0.3.3`、`0.4.0`、`0.5.0`，同时保持原 Runtime、权限、快照、Session 和 Memory
数据向后兼容。

## 二、升级前保护

程序目录没有 Git 仓库，因此修改前创建源码备份：

```text
~/.local/share/deep-agent/backup/AI-Agent-0.3.2-before-roadmap-20260713.tar.gz
```

SHA-256：

```text
a6e012b5d76e39f324d2bd2e119ece9b90c828f3ea0280802a73d8e35223db58
```

备份排除 `.venv`、缓存和 egg-info，不包含用户项目、Key 或正式 Memory 数据。

## 三、路线取舍

| 建议 | 结论 | 实际处理 |
|---|---|---|
| Prompt 指令驱动纠错记忆 | 采纳并加固 | Prompt 要求调用 `memory_add`，程序侧再校验 Correction 标签并补项目名。 |
| Memory list/delete/edit/stats | 采纳 | SQLite、FTS、Chroma 同步；edit 支持 `$EDITOR` YAML 和 flags。 |
| Tool 失败自动检索经验 | 采纳 | 本地搜索 Correction/Lesson，下一轮补注，不增加 API 调用。 |
| MCP Streamable HTTP/SSE | 采纳 | 使用标准库实现，保持 stdio 与 Capability 管道不变。 |
| MCP Resources | 调整后采纳 | 每个服务器必须显式 `resources_enabled: true`，URI 可独立白名单。 |
| 受限 HTTP | 采纳并加固 | 单一开关、域名白名单、30s、1 MiB、拒绝敏感 header 和重定向。 |
| Tree-sitter 语义索引 | 采纳 | 默认关闭，旁路 `index.semantic.json`，摘要加入 Prompt。 |
| >=8 项时并行 | 调整后采纳 | 作为显式 CLI，不由模型自动猜测任务独立性。 |
| Git worktree 隔离 | 采纳 | clean Git 前置、每任务 worktree、patch check、冲突不强制应用。 |
| 持久任务队列 | 采纳 | 原子 JSON、独立 Session、中断暂停、恢复跳过完成项。 |

没有实现通用 Multi-Agent。当前并行只是多个隔离的 Deep Agent 子进程，不引入
角色协商、共享黑板或跨 Agent 状态合并，避免过早扩大复杂度。

## 四、0.3.3 记忆进化闭环

### 1. Correction 协议

System Prompt 增加明确规则：用户在当前对话中明确否定事实、路径、端口、API 或
行为后，Agent 应先给出修正，再调用 `memory_add` 写入 `Correction`。要求标签：

```text
correction:<topic>
当前项目名
```

仅依赖 Prompt 不够可靠，因此 ToolManager 和 CLI 都增加程序侧约束：缺少
`correction:*` 会拒绝；项目名缺失时自动补齐。内容不得保存凭据或短期偏好。

### 2. Memory CRUD 与统计

MemoryStore 新增：

```text
get_memory
list_memories
update_memory
delete_memory
stats
search_recovery
```

CLI 新增：

```text
agent memory list
agent memory delete <id>
agent memory edit <id>
agent memory stats
```

`list` 支持 kind/tag/global/limit；`edit` 无参数时调用 `$EDITOR` 编辑临时 YAML，
也支持 flags。SQLite FTS 由触发器同步，Chroma 使用同一 ID upsert/delete。

### 3. 工具失败自愈

ToolResult 失败后，Runtime 从错误文本提取最多 8 个有效关键字，只搜索当前项目和
全局范围中的 `Correction`、`Lesson`。命中的经验以独立 system 消息：

```text
Failure Recovery Memory
```

加入下一轮。初始 Prompt 已出现的经验仍可在失败时重新用诊断语义强调，但同一条
经验在一次执行内最多补注一次。没有额外 DeepSeek 调用。

## 五、0.4.0 MCP、HTTP 与语义索引

### 1. MCP Streamable HTTP

新增 HTTP JSON-RPC 客户端：

- POST initialize/tools/list/tools/call/resources/read。
- 读取并回传 `Mcp-Session-Id`。
- 接受 `application/json` 和 `text/event-stream` 响应。
- Session 关闭时尝试 DELETE。
- 4 MiB MCP 响应保护。
- 拒绝 URL 内嵌凭据与自动重定向。
- loopback 地址绕过 WSL Clash，避免本地 MCP 被代理截获。

### 2. legacy SSE

实现 GET event stream，解析服务器 `endpoint` 事件，再通过该 endpoint POST
JSON-RPC。启动完成后将底层 socket 改为无空闲读取超时，避免正常静默连接在
15 秒握手超时后断开。关闭时捕获标准库流关闭竞态并 join 读取线程。

### 3. Resources

每个启用 MCP 服务器可注册一个动态能力：

```text
mcp.<server>.resources.read
```

服务器需要显式 `resources_enabled: true`，避免旧 stdio 配置突然增加工具。URI
使用 `resource_uri_allowlist` 过滤，仍经过 Permission Manager、ToolResult、事件
和状态记录。`tools_enabled: false` 可创建只提供 Resources 的连接。

### 4. 受限 HTTP

新增 `HttpTool` 和 `http_request` Capability。只支持 GET/POST JSON，允许域名由
`config.yaml` 配置，子域名可匹配。上限：

```text
timeout <= 30 秒
response <= 1 MiB
```

拒绝 URL 凭据、Authorization、Cookie、Proxy-Authorization、API-Key header 和
所有自动重定向。默认关闭，且最终收敛为唯一开关 `tools.http.enabled`，移除早期
实现中重复的 `allow_http`/capability enabled 激活要求。

### 5. Tree-sitter 语义旁路索引

安装：

```text
tree-sitter-language-pack 1.12.5
```

支持 Python、JavaScript、TypeScript、Java、Go、Rust。默认关闭；启用后生成：

```text
.project-agent/index.semantic.json
```

内容包括结构项、父子层级、函数/方法、类、行号、签名和 import 来源。摘要进入
Runtime Project Context，但不替换现有 `index.json`。缺少 grammar 或单文件解析
失败只记录状态，不阻断 Agent。

## 六、0.5.0 队列与隔离并行

### 1. 持久任务队列

新增 `TaskQueueManager`，队列位于 `.project-agent/queues/`。每项记录：

```text
id / prompt / status / session_id / result / error / updated_at
```

队列每次状态变化原子写入 JSON。默认任务失败即暂停；`--continue-on-error` 可继续。
Ctrl+C 将当前任务标为 paused。恢复时已完成项跳过，failed/pending 项重新执行。
加载旧 JSON 时按字段白名单构建对象，未来增加字段不会破坏旧版本读取。

用户要求的直接语法已支持：

```bash
agent queue "任务一" "任务二"
```

### 2. Git worktree 并行

新增 `ParallelWorktreeRunner`，只有任务数达到配置门槛（默认 8）才执行。前置：

- Git 仓库。
- 除 `.project-agent` 外主工作区干净。
- 用户明确保证任务彼此独立。

流程：

```text
读取基线 HEAD
  -> 每项建立临时 branch/worktree
  -> ThreadPool 并发启动 agent 子进程
  -> 从基线 HEAD 生成 binary patch
  -> 主工作区 git apply --check
  -> 无冲突则逐项 apply
  -> 保存 patch/report.json
  -> 清理 worktree 和临时分支
```

patch 从基线提交提取，因此子 Agent 即使自行 commit 也不会丢失改动。创建部分
worktree 后失败会回收已创建资源。`.project-agent` 明确排除，不会把 Session、
UUID、浏览器身份或快照带回主分支。

默认子 Agent 只使用 `--auto-approve`。顶层显式 YOLO/SUPER YOLO 时才向子进程
传递相同级别。

## 七、实施中发现并解决的问题

### 1. WSL Clash 截获 loopback

本地 HTTP 测试最初返回 `HTTP 502`，原因是 urllib 读取了全局代理。本地
`127.0.0.1/localhost` 请求现在显式使用空 ProxyHandler，远端继续走现有代理。

### 2. Resources 破坏旧 MCP 工具计数

初始实现默认给每个 stdio Server 增加 Resource 能力，使旧测试从 2 工具变为 3。
调整为 `resources_enabled: true` 显式启用，保持旧配置行为不变。

### 3. `.project-agent` 造成 Git dirty 误判

ProjectManager 会创建运行数据，Git status 因此不为空。并行 clean 检查和 patch
提取现在都排除 `.project-agent`，但不会忽略用户其他未提交文件。

### 4. `.gitignore` 并发重复追加

并行 smoke 时 `agent init` 与 Memory 命令同时解析项目，早期 add-only 追加产生
重复条目。迁移改为去重后的原子临时文件替换，现有重复项已清理。

### 5. SSE 关闭线程 warning

关闭长连接时 Python 3.14 HTTP 流内部出现 `NoneType.close` 竞态。读取线程现在捕获
关闭异常，`close()` 短暂 join，最终测试无 warning。

### 6. 配置重复写入

HTTP 早期有三个激活层。默认配置每次补入后迁移删除，会造成无意义重写。最终
默认模板和迁移均统一为 `config.yaml` 的单一开关，重复加载哈希稳定。

## 八、安全与兼容性

- 所有新重量能力默认关闭：MCP、HTTP、semantic index。
- MCP Server 仍逐个 opt-in，Resources 再次显式 opt-in。
- MCP HTTP header 只存在权限 `600` 的 `mcp.yaml` 和进程内存，不进入事件 payload。
- `http_request` 不接受敏感认证 header，避免模型把凭据写进 Session/tool arguments。
- 队列、并行报告和 semantic index 已加入 `.project-agent/.gitignore`。
- ProjectManager 对旧项目执行 add-only/去重迁移，不覆盖 context、UUID 或用户配置。
- 当前项目在 `/mnt/d`，DrvFS 权限显示 `777` 是挂载语义，不是 chmod 失败逻辑。

## 九、测试与验收

自动化测试使用临时目录、本地 HTTP Server、临时 Git 仓库和假的 agent 可执行文件，
不写正式 Memory，不调用真实 DeepSeek。

最终结果：

```text
pytest: 40 passed
Ruff lint: passed
Ruff format: passed
compileall: passed
agent doctor: passed
agent doctor --online: 5/5 keys ready
tools.yaml repeated-load hash: stable
```

新增覆盖：

```text
Memory CRUD/stats/Chroma delete
Correction 标签校验和项目标签
Tool failure -> Recovery Memory
HTTP allowlist/size/redirect/loopback proxy
MCP Streamable HTTP/Session ID/Tools/Resources
MCP legacy SSE 初始化和资源读取
semantic index sidecar + Prompt summary
queue pause/resume and completed-task skip
parallel >=8 threshold/dirty guard/worktree patch merge
```

真实 CLI smoke：

```text
agent version: 0.5.0
Memory add/list/edit/stats/delete: passed
parallel < 8 rejection: passed
project private-path migration and deduplication: passed
MCP config/secrets permissions: 600
```

## 十、文件与数据位置

```text
程序：~/AI-Agent
维护说明：~/AI-Agent/docs/implementation.md
测试：~/AI-Agent/tests
配置：~/.config/deep-agent
DeepSeek Key：~/.config/deep-agent/secrets.env
MCP：~/.config/deep-agent/mcp.yaml
Memory DB：~/.local/share/deep-agent/sqlite/memory.db
Vector：~/.local/share/deep-agent/vector
并行临时 worktree：~/.local/share/deep-agent/worktrees
源码备份：~/.local/share/deep-agent/backup/AI-Agent-0.3.2-before-roadmap-20260713.tar.gz
使用说明：/mnt/d/detail/deepseek/README.md
工作日志：/mnt/d/detail/deepseek/DeepSeek-Agent-V3-工作日志.md
```

## 十一、后续建议

1. MCP Prompts、资源列表和变更订阅，继续使用现有 transport/Capability 边界。
2. Memory 审计 UI：合并重复 Correction、可信度和过期策略。
3. 语义索引增量更新、调用关系和引用关系，但仍保持可选旁路。
4. 队列的重试策略、计划时间和资源配额。
5. 并行任务的文件范围声明与锁，减少 patch 冲突。
6. Web UI 复用 AgentState、Session、Queue、Event Bus，不另建执行核心。
7. 只有明确证明角色分工收益时再评估 Multi-Agent。

新增功能应继续保持：CLI 只解析，Runtime 只编排，工具统一走 Capability 和
Permission，状态可序列化，任务终止副作用走 Event Bus，重量能力默认关闭。
