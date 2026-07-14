# Deep Agent v0.9.1 接口稳定化实用案例

本案例不是另一个 Agent，也不调用真实 API。它用一个可直接运行的
Python 脚本演示 v0.9.1 最重要的三个开发接口：

1. `TaskRouter -> ModelRouter -> TaskPlanFactory`：同一个 DeepSeek Runtime
   如何按任务难度选择执行方式和成本级别。
2. `EventBus`：如何订阅、发布、关联 `run_id`、隔离失败订阅者并取消订阅。
3. `AgentState.validate()`：如何在保存或 Resume 前验证冻结身份和状态。

## 运行

```bash
cd ~/AI-Agent
PYTHONPATH=. .venv/bin/python user-docs/测试与验收/实用案例/v0.9.1/interface-routing-demo.py
```

预期看到四类请求分别路由到 low、balanced 或 high，并打印 starter
plan；随后 Event Bus 中一个故意失败的订阅者不会阻止审计订阅者收到
事件，最后 AgentState 验证通过。

## 观察重点

- 简单问题不启用高成本 thinking。
- 大型只读审查使用分块计划，但不自动升到最高成本档。
- 高风险重构和重复失败升到 deep/high。
- `TaskPlanFactory` 只接收 `TaskRoute`，没有第二套 Prompt 分类。
- 事件字段可以保存为稳定字典；订阅者错误出现在 `last_errors`。
- DeepSeek 仍是唯一 Provider。

## 在真实项目中使用

用户日常无需直接调用这些 Python 接口，只需运行：

```bash
cd /你的项目
agent "解释这个函数"
agent "检查整个仓库并总结，不修改文件"
agent "全面修复高风险权限问题并运行测试"
```

开发扩展时再参考此脚本和 `docs/architecture-v0.9.1.md`。不要让插件
直接读取 Prompt 文件、直接执行系统命令，或绕过 PermissionManager。
