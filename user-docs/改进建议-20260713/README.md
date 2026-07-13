# Deep Agent v0.9.1 后续改进建议

日期：2026-07-13

本目录只放当前没有直接实施、需要单独设计或用户决策的改进方向。v0.9.1 已完成接口稳定化；用户已要求在其发布闭环之后另版实施 Event Bus 整体迁移。

## 已完成

1. GitHub Actions 已恢复并采用 Python 3.11/3.12/3.13 矩阵。
2. Actions 已升级为 `checkout@v5` 与 `setup-python@v6`，不再使用 Node.js 20 action runtime。
3. ContextPackage、Task/Model Router、Interface Contract、最小 Event schema 和 AgentState validate 已在 v0.9.1 落地。

## P1：下一补丁优先

1. Parallel worker 双层超时：为单 worker 和整个 run 分别设上限；超时后终止进程组，默认保存 patch/report，是否保留 worktree 由配置决定。
2. 工具调用 Journal：在有副作用工具执行前持久化 intent，执行后持久化 result；Resume 按 request_id 对账，解决 SIGKILL 落在“工具成功、checkpoint 未写”窗口时的重复执行风险。
3. Session revision/CAS：本轮已用 per-session flock 拒绝并发 Resume；后续若要允许协作 turn，需要显式 revision、冲突提示和合并协议。
4. MCP/外部输入累计字节：本轮已有页数、工具数量、重复 cursor、HTTP/文档/下载上限；后续增加所有协议响应的累计序列化字节预算。
5. Event Bus 整体迁移：必须在 v0.9.1 完整发布之后另建版本；先列出 Session、Memory、Audit、Metrics 等副作用，再逐项迁移、双写验证和去除旧路径。

## P2：规模化能力

1. Context 分层摘要：文件清单 -> 模块摘要 -> 相关代码块，按任务图步骤动态换入，减少超大仓库每轮重复 Context。
2. 大文档 Map-Reduce：对 PDF/Word/Markdown 分块生成带来源位置的局部摘要，再做交叉核对和总摘要。
3. 计划预算：为每一步记录预计/实际模型轮次、工具时间、Context 字符数；超过预算先压缩或拆分，而非继续堆轮次。
4. 结果证据索引：最终答复中的每个关键结论关联文件、行号、测试或 ToolResult，便于复核。

## P3：工程化

1. 发布构建：生成 wheel/sdist、校验版本/tag一致性，并附 SHA-256。
2. Coverage 基线：从核心 Runtime/Permission/Session/Queue 设覆盖率门槛，不追求机械 100%。
3. 统一错误类型：逐步引入 Config/Model/Tool/Session/Queue 异常层次，CLI 只负责映射退出码和用户提示。

## 需要用户决定

1. LICENSE：MIT、Apache-2.0 或保留全权利。法律授权不能由 Agent 自动替用户选择。
2. 是否默认展示完整 DeepSeek reasoning。当前按用户要求展示；若更重视隐私，可默认只显示阶段摘要并提供 `/thinking full`。
3. Parallel 最小任务数是否从 8 改为可配置的 4。当前 8 是已有架构策略，不属于缺陷。
