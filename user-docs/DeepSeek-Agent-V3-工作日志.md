# DeepSeek Agent V3 工作日志（0.9.1）

日期：2026-07-14

## 目标

在不进行大规模 Runtime 重构的前提下，为 v1.0 核心接口冻结准备：让 Context Builder 成为唯一上下文入口，冻结 `CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission` 契约，最小化稳定 Event Bus，整理 AgentState schema，增加 DeepSeek-only cost-aware 路由，并解决 N3-006 的双分类风险。

## 根因

1. PromptBuilder 虽已有 Package 正式入口，但仍保留 v0.8 的 State/Snapshot/Memory 分散参数与 `append_resume()`，外部调用仍可绕过 ContextPackage。
2. Runtime 仍实例化 `TaskStrategySelector` 生成计划，使 deprecated facade 留在正式路径；`task_strategy.py` 的分类、路由和计划职责不够清楚。
3. AgentState schema 只有版本数字，没有正式字段顺序、冻结身份和统一 `validate()`，损坏或未来状态可能在更深处才失败。
4. 现有 EventBus 能发布和订阅，但没有稳定 schema、run 关联、反序列化、取消订阅和嵌套发布错误语义，后续迁移缺少可依赖边界。
5. Model Router 已有 fast/standard/deep 能力档位，但没有单独持久化成本级别和充分可解释的本地选择原因。

## 实现

### Context 与 Prompt

- PromptBuilder 公开入口冻结为 `build_initial(package: ContextPackage)` 和 `build_resume(package: ContextPackage)`。
- 删除 `append_resume()`、`_runtime_context()` 及所有分散参数兼容入口。
- Prompt 模块不导入 ContextBuilder、ContextSnapshot 或 AgentState，也没有文件读取路径。
- Runtime 的 initial/Resume 测试记录 ContextBuilder 产出的 Package，并验证传给 PromptBuilder 的是同一对象。

### Interface Contract

- 新增 `agent/contracts.py`，定义核心链、Context 链、Event schema 和 AgentState schema/字段常量。
- 新增 Interface Contract Test，冻结 Runtime 入口签名、Prompt 单 Package 签名、ContextPackage 字段、ToolRequest/ToolResult 字段、ModelRoute 字段、AgentState 字段和 Permission 前置执行顺序。
- 契约测试证明模型工具调用必须先经过 PermissionManager，再进入注册 handler。

### Task、计划与 Model Router

- TaskRouter 成为唯一分类器，统一生成 type/scale/risk/mode/score/reasons/failure/mutation 证据。
- 新增 `TaskPlanFactory`，只接受 TaskRoute，不再分析 Prompt 或维护第二套评分规则。
- Runtime 删除 TaskStrategySelector 的 import、实例化和调用。
- `task_strategy.py` 仅保留 deprecated DTO/facade；兼容调用全部委托 TaskRouter/ModelRouter/TaskPlanFactory，并产生 DeprecationWarning。
- ModelRoute 新增 `cost_class`：low、balanced、high；旧 Session 缺字段时按 tier 推导。
- 本地策略：simple+low -> fast/low；普通与大型只读 -> standard/balanced；deep、高风险、架构/重构或重复失败 -> deep/high。
- 显式模型 tier 和显式 task mode 继续优先；DeepSeek 是唯一 Provider，其他 Provider fail-closed。

### AgentState

- 增加 `SCHEMA_VERSION`、`SERIALIZED_FIELDS`、`FROZEN_FIELDS`、`validate()` 和冻结基线。
- 冻结 session_id、project、working_directory、created_at；请求、路线、计划进度、输出和 updated_at 可变。
- 校验 Session 身份、ISO 时间、状态、plan ID/依赖/环、current/completed 派生字段、Memory/Tool 集合、Task/Model route、DeepSeek Provider、cost class 和 Context manifest 边界。
- schema v1 仍兼容加载并只修复历史派生 plan 字段；未来未知 schema 明确拒绝。

### 最小 Event Bus

- Event schema v1 冻结：schema_version、id、name、timestamp、project_id、session_id、run_id、payload。
- 增加 `to_dict()`、`from_dict()`、`effective_run_id` 和时间戳验证。
- `subscribe()` 返回取消函数；增加幂等 `unsubscribe()`；`publish()` 可接收事件名或既有 Event。
- 一个订阅者失败不会阻止后续订阅者；嵌套 publish 不覆盖外层最终错误。
- Runtime、ToolManager、MemoryPipeline 补充 run_id 关联；JSONL 使用相同事件结构并继续脱敏。
- 本次没有迁移 Session、Memory 等全部副作用，也没有实现 durable broker、重放、跨进程顺序或事务交付。

### 文档与案例

- 新增 `docs/architecture-v0.9.1.md`，详细说明所有权、Event API、禁止事项和未来扩展清单。
- 架构文档进一步给出 Event 发布者的命名/关联/脱敏规则、可取消订阅者模板、同步执行风险、幂等键和分步迁移顺序；明确不能把 `publish()` 成功当成持久化保证。
- 架构文档补充 AgentState 扩展约定：新字段必须有旧 Session 默认值与迁移测试，不允许就地改义旧字段，`validate()` 必须保持确定性且无副作用。
- 新增 `docs/releases/v0.9.1.md`、更新源码 README、实现说明和中文使用说明。
- 新增 `实用案例-v0.9.1/interface-routing-demo.py`，可离线演示四种路由、计划和 Event 错误隔离。
- GitHub Actions 保持 `actions/checkout@v5` 与 `actions/setup-python@v6`，避免 Node.js 20 弃用警告。

## 测试与审查

```text
130 tests passed（完整测试含真实 PTY）
Interface Contract tests passed
Event/AgentState contracts passed
ContextPackage/Prompt/Runtime/Router/Permission regression passed
Ruff check passed
Ruff format check passed
compileall passed
git diff --check passed
离线接口实用案例 passed
```

最终审查额外修正：

1. TaskPlanFactory 最初若只按 task_type 判断，会把“架构并实现”误当只读计划；TaskRouter 现持久化 mutation-request，计划工厂仍只消费 TaskRoute。
2. Event 新增 run_id 后，Runtime/Tool/MemoryPipeline 统一传递，旧 payload run_id 仍由 effective_run_id 兼容。
3. Event payload 保持防御性浅复制而非 MappingProxy，避免破坏现有 `dict` 序列化/消费者契约；文档明确 handler 不应并发修改。
4. AgentState 冻结字段由内部基线在每次 validate/状态变更时核验，避免只提供手工比较方法而没有实际约束。

## 版本与发布

版本升级为 `v0.9.1`。提交哈希、tag、GitHub Actions 运行号和远端核验结果在发布闭环末端填写，并以最终回复为准，避免文档自引用导致无限追加提交。

本次没有读取或输出真实 API Key，也没有运行在线 DeepSeek 请求。

## 审查问题处理

- N3-001：当前工作区 `.project-agent/context.md` 版本更新为 0.9.1。
- N3-002：GitHub Actions TODO 标记已完成。
- N3-003：后续改进建议更新为 0.9.1，并将 Actions 移到已完成。
- N3-004/N3-005：当前 Markdown 工作日志只保留 0.9.1；0.9.0、0.8.0、0.7.1 历史日志归档到老版工作日志。
- N3-006：TaskRouter 唯一分类，TaskPlanFactory 单一计划职责，TaskStrategy deprecated。

## 下一步：Event Bus 整体迁移

用户在 v0.9.1 实施过程中明确要求：必须完成本版本全部工作、tag、推送和 Actions 核验之后，再开始 Event Bus 整体迁移。

下一版本将另建检查点，不混入 v0.9.1。优先建立副作用清单、事件命名与幂等策略，再逐步迁移 Session、Memory、Audit、Metrics 等。任何工具执行仍必须遵循 `ToolRequest -> PermissionManager -> ToolResult`；Event 只能承接记录和副作用编排，不能成为绕过权限的执行通道。
