# Deep Agent v0.9.0 实用案例：订单汇总器

这个案例用于真实体验 Deep Agent 的任务路由、可见 Thinking、统一 ContextPackage、工具调用、测试验证和 Session Resume。目录内是一个可运行的小型 Python 项目，不包含 API Key。

## 1. 案例目标

程序读取 `data/orders.csv`，按商品汇总有效订单的数量和金额，并将结果输出为 JSON。目前故意保留了两个真实缺陷：

1. `cancelled` 订单没有被排除。
2. 金额使用二进制浮点数累计，可能出现财务精度问题。

测试已经描述正确行为，因此适合让 Agent 定位根因、修改代码并验证。

## 2. 启动

先确保私有 Key 位于：

```text
~/.config/deep-agent/secrets.env
```

然后进入示例项目：

```bash
cd /mnt/d/detail/deepseek/实用案例-v0.9.0/order-summary-demo
agent init
agent
```

普通输入按一次 `Enter` 提交。终端会显示任务模式、DeepSeek 档位、Thinking 已用时间、模型轮次和工具状态。

## 3. 推荐体验顺序

### A. simple：快速解释

输入：

```text
什么是 Decimal，它为什么适合金额计算？只解释，不修改文件。
```

预期：本地 Task Router 选择 simple/低风险，默认使用 fast 档位；若没有配置 fast 专用模型，仍安全回落到 `model.model`，但关闭 Thinking。

### B. standard：定位并修复明确 Bug

输入：

```text
运行当前测试，定位失败根因，修复 cancelled 订单仍被计入和金额精度问题。先查看相关文件，只修改必要代码，最后重新运行测试并说明证据。
```

预期流程：

```text
Task Router -> bug_fix / medium / medium / standard
Model Router -> DeepSeek standard / high thinking
Context Builder -> 受限 ContextPackage
read/search/test -> file_diff -> file_apply -> test
```

Agent 应优先读取 `AGENTS.md`、`README.md`、`order_summary.py` 和测试，通过受控文件流程修改。正确结果应使测试全部通过。

### C. large：处理多文件和数据范围

输入：

```text
分析整个示例项目和所有订单数据文件，分块检查数据质量、代码结构、测试覆盖和使用文档。不要修改文件，给出按优先级排序的完整改进报告。
```

预期：选择 large，建立 `scope -> inspect-chunks -> synthesize -> verify` Task Graph，分块读取而不是一次把所有内容塞进 Prompt。

### D. deep：高风险审计与完整变更

输入：

```text
全面审计整个订单汇总项目，重点检查财务精度、输入校验、路径安全、异常处理和回归测试。先给出依赖计划，再修复所有能够复现的高风险问题，完成后运行全部测试并总结剩余风险。
```

预期：选择 deep/高风险，使用 max Thinking 和最多 24 轮受控执行。终端会流式显示 DeepSeek API 返回的 `reasoning_content`（可在配置中关闭）。

### E. Resume：中断后继续

执行 large/deep 任务时按 `Ctrl+C` 返回交互界面，然后输入：

```text
/resume
```

或：

```text
/resume SESSION_ID
```

再输入：

```text
继续，从保存的计划和验证步骤恢复，不要重复已经完成的修改。
```

v0.9.0 会保留或升级原 Task Route 和精确 DeepSeek Model Route，不会因为“继续”很短而降级。

## 4. 人工验证

在 Agent 修改前运行：

```bash
python3 -m unittest discover -s tests -v
```

应看到失败。修复后再次运行，应全部通过。

运行程序：

```bash
python3 order_summary.py data/orders.csv
```

正确结果中：

- `keyboard` 数量为 3，金额为 `59.97`。
- `mouse` 数量为 1，金额为 `15.50`。
- `cancelled` 的 99 个 mouse 不得计入。

## 5. 回滚 Agent 修改

交互界面输入：

```text
/undo
```

或删除 `.project-agent/` 后重新复制本案例目录。`.project-agent/` 是运行数据，不应提交到 Git。

## 6. 观察 v0.9.0 路由状态

任务执行后可输入：

```text
/status
```

Session JSON 位于：

```text
.project-agent/sessions/<session-id>.json
```

其中可看到 `task_route`、`model_route` 和 `context_manifest`。这些状态只用于理解运行过程；不要手工写入真实凭据。
