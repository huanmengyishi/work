# v0.10.0 Event Runtime 离线案例

这个案例不需要 API Key，不访问网络，也不读取用户配置。它在临时目录中演示：

- required Session owner 成功与失败；
- best-effort subscriber 失败不回滚 required 结果；
- Memory usage 事件重复投递只计一次；
- Audit 不保存 Prompt/reasoning/messages/工具正文；
- Metrics 只保存事件计数、总耗时和失败数。

运行：

```bash
cd ~/AI-Agent
PYTHONPATH=. .venv/bin/python user-docs/实用案例-v0.10.0/event-runtime-demo.py
```

预期最后显示 `Event Runtime demo passed.`。脚本会打印一份安全 Audit 记录和聚合 Metrics；其中不会出现脚本内的 private marker。

开发新 subscriber 时可复制脚本中的模式：

1. 数据必须持久化成功才能继续：精确订阅、`required=True`、发布端使用 `dispatch_required()`。
2. Audit/Metrics/UI/Health 等观察性功能：保持 best-effort，失败不得改变主业务结果。
3. 重复可能发生的数据库副作用：为 run/evidence 建立稳定 idempotency key，并在同一事务写结果和凭证。
4. 不要把 live State、messages、Prompt、reasoning、工具参数或输出交给通用 logger。
