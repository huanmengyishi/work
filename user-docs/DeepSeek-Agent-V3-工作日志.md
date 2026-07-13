# DeepSeek Agent V3 工作日志（0.10.0）

日期：2026-07-14

## 目标

在 v0.9.1 接口稳定化已经完整发布后，整体迁移 Runtime 自动副作用到 Event Bus：Session、Context Memory usage、自动 Memory/Reflection、Capability Health、Audit、Metrics 和可见 Thinking/Progress，同时保持 `CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission`、`ToolRequest -> PermissionManager -> ToolResult` 与 DeepSeek 唯一 Provider 不变。

## 根因

1. v0.9.1 的 Event Bus 只有稳定通知模型，Runtime 仍直调 Session checkpoint/finalize、Memory usage，并内联注册 Audit/Memory Pipeline。
2. ToolManager 在 ToolResult 后直写 Capability Health，使自动副作用分散在执行路径。
3. 普通 `publish()` 只能隔离异常，无法表达“Session 必须成功”与“Audit 失败可以忽略”的差异。
4. Session 内部事件需要 live AgentState/messages，但 wildcard JSONL 若通用字符串化会泄露请求、模型消息、工具正文或凭据。
5. Memory usage 与 terminal Memory 虽有不同幂等需求，缺少统一 run/evidence 规则。
6. Thinking 已可见，但仍由 Runtime 直接调用 UI handler，未纳入统一自动副作用注册。

## 实现

### Event delivery

- 新增 `EventDelivery`、`EventDispatch`、`EventDispatchError`。
- `subscribe()` 增加 required/name 元数据并拒绝同事件重名；`dispatch_required()` 要求至少一个精确 required owner，best-effort/wildcard observer 都不算 owner。
- required handler 失败时 fail-closed；best-effort 失败只进入 dispatch 结果，后续订阅者仍执行。
- 嵌套 publish 不覆盖外层最终错误信息。

### RuntimeEventPipelines

- 新增 `agent/event_pipelines.py`，作为 Session、Memory usage、Memory/Reflection、Health、Audit、Metrics 和 UI progress 的唯一自动注册点。
- Runtime 删除 `JsonlEventLogger` 与 `MemoryPipeline` 的内联注册。
- 查询仍直接调用；Event 不充当 Context/Session/Memory RPC。

### Session 顺序与恢复

- checkpoint/finalize 改发 required `session.*.requested`。
- 执行顺序固定为 Session finalize 成功后再发布 `task.finished/task.failed`。
- checkpoint 若由 Session owner 自身失败会立即停止，不从不确定状态继续 finalize；若命名 Session owner delivery 已成功、只是其他 required observer 失败，则可安全写 failed terminal 供 Resume。finalize 自身失败不会递归重试，也不会发布虚假 terminal Event。
- 增加缺 owner、写盘失败、顺序、终态不重复和 AST 证明旧直调删除测试。

### Memory usage 与自动学习

- 实际进入 ContextPackage 的 Memory 才发布 required `memory.usage.recorded`。
- 新增 SQLite `memory_usage_events`；同一 usage_id 在一个事务中更新 use_count 并写幂等凭证。
- replay evidence 不同会拒绝；完全相同则 no-op。Runtime 成功后才写 `state.loaded_memories`。
- Resume 的新 turn 使用新 run_id，可在再次实际包含时强化一次。
- terminal Memory/Reflection 保留 `pipeline_runs.run_id` 幂等。

### Capability Health 与权限边界

- 删除 ToolManager 内 `health.record()` 直调。
- `tool.finished` 只携带 bounded metadata：capability、request ID、参数/结果字段数量、成功、耗时、health failure 与脱敏错误摘要。
- 不携带参数值、path、stdout/stderr、result body 或 data keys。
- 只有基础设施故障 marker 降低健康；业务校验失败不降级。
- Health subscriber 是 best-effort，失败不改变已经得到的 ToolResult；Permission 仍先于 handler/Event。

### Audit、Metrics 与 Thinking

- Audit 使用按事件/字段 allow-list 的元数据投影，不字符串化未知对象。
- Prompt、reasoning、messages、AgentState、工具参数/输出、凭据全部禁止落 JSONL。
- Audit 目录/文件使用 0700/0600，拒绝符号链接并尽可能 no-follow 打开。
- Metrics 只保存允许的 task/model/tool 事件计数、工具总耗时和失败数；对畸形/非字典 JSON、bool、负数和超大整数严格拒绝或饱和。
- Metrics 旧文件读取硬限 64 KiB，写入使用私有临时文件、fsync 和原子替换，拒绝符号链接。
- Metrics、Capability Health、Daemon 与 Parallel worktree 使用 project ID 的稳定安全 storage key；正常历史 ID 保持原路径，损坏或恶意元数据中的路径分隔符不能逃逸全局数据目录。
- Thinking/Progress 改为 best-effort `ui.progress.updated`；ConsoleUI 仍实时显示 reasoning，但 Audit payload 为空、Metrics 不统计。真实 PTY 验证 deep 任务可见 Thinking 片段和最终答案。

### 配置、文档和案例

- 版本升级为 0.10.0。
- 配置只新增 `events.metrics_enabled: true`，不覆盖用户已有值。
- 新增 `docs/architecture-v0.10.0.md`、`docs/releases/v0.10.0.md`。
- 更新程序 README、实现说明、中文使用说明。
- 新增离线 `实用案例-v0.10.0/event-runtime-demo.py`，展示 required/best-effort、幂等、Audit 和 Metrics。
- GitHub Actions 继续使用 `actions/checkout@v5` 与 `actions/setup-python@v6`，避免 Node.js 20 弃用警告。

## 测试与审查

```text
173 tests passed（含真实 PTY Thinking）
Interface Contract / Runtime / Permission 顺序 passed
Session required owner、故障注入、顺序、Resume passed
Memory usage 重复事件、证据冲突、Resume passed
Capability Health best-effort 与 payload 隐私 passed
Audit/Metrics 脱敏、恶意对象、畸形/超大文件、权限/符号链接 passed
Ruff check passed
agent/tests Ruff format check passed
compileall passed
git diff --check passed
```

发布验证只对正式源码、测试、脚本和 v0.10.0 案例执行 format check；旧版带故意缺陷的业务案例保持历史基线，不作为本版本格式化目标。

## 版本与发布

目标版本：`v0.10.0`。最终提交哈希、tag、GitHub Actions 运行号和远端核验结果在发布闭环完成后以检查点与最终回复为准，避免文档自引用导致无休止追加提交。

本次未读取、输出、复制或提交真实 API Key，也未执行在线 DeepSeek 请求。

## 未实施与下一步

- 未加入非 DeepSeek Provider。
- 未加入多 Agent、Worker Runtime、GitHub/Jira/Notion 生态。
- 未实现跨进程 Event Broker、重放或 exactly-once 分布式保证。
- 下一可靠性版本优先设计 Durable Intent Journal：在有副作用工具执行前记录 intent/hash，执行后记录 result/commit，解决进程崩溃后的可靠 Resume；不能直接用 Event Bus 替代事务日志。
