# DeepSeek Agent V3 工作日志（0.13.0）

日期：2026-07-29

状态：0.13.0 候选代码、完整离线门、隔离安装和唯一在线代表案例均已通过，正在
完成提交、tag、GitHub Actions 和远端文件树核验。

## 1. 目标

- 任务成功后至多使用一次额外 DeepSeek 调用精炼本次经验。
- `memory.smart_reflection` 默认开启，但当前 turn 工具调用少于 5 次时不反思。
- 落实《DeepSeek Agent V3 综合改进方案》及三份辅助审查中的兼容建议。
- 拆分过长模块，建立热/温/冷数据边界，补齐长期恢复和可选自适应能力。
- 将指定 Claude Code 快照的可观察优秀设计整理为本地 clean-room 参考包。
- GitHub 只保留应用、测试/CI、当前发布说明和最新版简明用户文档；详细开发
  数据和历史报告只保存在本地。

## 2. 主要根因

1. 原智能反思默认关闭，且模板式 Memory 没有调用模型理解任务经验。
2. 反思 client 与 Runtime 预算分离，可能重复调用或在微小任务上浪费 API。
3. Runtime、Convergence 和 Tool Manager 职责过密，状态和工具记录缺少统一硬门。
4. Resume 会加载过多历史，Memory 查询缺少有界温缓存和严格 transfer 边界。
5. 方案中的退避、策略计划、Intent Journal、Artifact 生命周期和可选实验能力尚未
   完整接入现有单 Runtime 路径。
6. GitHub 当前树积累了历史开发报告、旧版 Word 和测试样例，公开入口过重且易误导。

## 3. 实现

### Memory 精炼

- 成功完成后，使用注入的同一 DeepSeek client 发起至多一次无工具、受预算逻辑请求；
  底层仍服从有界网络重试与 Key 轮转。
- 当前 turn 少于 5 次受管工具调用时硬跳过；配置不能把门槛降到 5 以下。
- 新安装或缺失配置键时默认 `smart_reflection: true`；升级时已有显式 `false` 保持
  不变。失败任务、预算不足、异常 finish、非法 JSON、工具注入和重复尝试均安全
  跳过，不改变已完成结果。
- 输入、输出和标签均做凭据脱敏；模型返回的 `kind`、`confidence` 和 `tags` 真实
  写入 Memory，而不是被本地模板覆盖。

### 架构与工具

- Runtime 成为组合根，主循环 `_execute` 缩为 46 行；Setup、Response、Termination、
  Tool Batch、Validation、Compaction 和 Synthesis 独立。
- Context Window、History Compaction、History Evidence 与 Convergence 分层。
- ToolExecutor 统一执行权限前后生命周期；Capability 声明数据驱动并按需、线程安全
  初始化。
- 核心接口契约升至 5，补齐 Runtime、Tools、Session、Context 和 Prompt Protocol。
- DeepSeek Key 池轮转加锁，但网络 I/O 不持锁。

### 状态、Session 与 Memory

- AgentState schema 升至 8；首次可保留 200 个真实工具 receipt，溢出后为 1 个
  裁剪标记加最多 199 个真实 receipt，同时限制总状态、单记录、键、集合和嵌套深度。
- Session Resume 流式加载最近 16 个完整轮次；被保留的完整 JSONL 有
  count/bytes/SHA-256 校验，并由最多 4 个、总计 128 MiB 的冷世代管理。
- Memory 增加 128 项 TTL/LRU 查询缓存、严格版本化 JSON import/export、项目与全局
  Knowledge 隔离，以及写入/反馈/维护后的缓存一致性。
- History Snip 在模型压缩前做零模型、完整轮次安全裁剪。

### 长期运行与可选策略

- Durable Intent Journal 使用有界哈希链、并发锁和 projection bridge。
- ArtifactRegistry 只接受受管生命周期证据，保存产物状态和跨步骤血缘。
- Capability 失败接入轮次级渐进退避和熔断；成功后复位，不新增网络重试层。
- TaskPlanFactory 按文档、缺陷、功能、变更和审查等类型生成验证型计划。
- Adaptive Convergence、Strategy Adjuster 和确定性 ExperimentRunner 已实现但默认关闭，
  只读取有界安全标量，仍保持 DeepSeek-only。

### 文档与本地参考

- Claude Code `claude` 分支固定到 commit
  `b17913e26fd4278ad5cd4b32ed3bde86bf1444e9`、tree
  `68ef0259999328e531cc23c81fc80e81cbdabecb`。
- 只归纳公开可观察的抽象设计，不复制无许可证源码、Prompt、Schema、协议、测试或
  品牌素材；完整参考包和 Word 只留本地，不上传 GitHub。
- GitHub 候选树已移除历史发布说明、详细审计、旧版用户文档和案例副本；本地清理前
  bundle 与 tar 快照均已校验。

## 4. 测试与实例纪律

最终本地结果：

```text
Memory 精炼专项：19 项通过
Session/State 专项：25 项通过
Convergence/History 相关：175 项通过
Runtime 拆分相关扩大回归：约 250 项通过
最终主线聚焦批次：109 项通过
完整 pytest：679 passed in 21.15s
真实 PTY 专项：29 passed；CLI 专项：15 passed
Ruff 0.15.21、format（142 files）、compileall、pip check、git diff --check：通过
全新临时 venv 严格离线安装、--version、--help、agent init、launcher：通过
隔离无 Key 普通任务：退出 1，未创建 .project-agent
实际两份 Word：三处字节一致，CRC/OPC、正文、core.version、版本和日期元数据通过
python -m agent --version：deep-agent 0.13.0
```

真实实例严格遵循：全部离线门全绿后，每轮发布最多选择一个与本次改动直接相关、
成本有界的案例。任一在线错误立即停止，先修复并重跑全部离线/CI 门，不连续调用
消耗 DeepSeek API。本轮只运行一个 Memory 精炼代表案例：无工具、无重试、输出上限
768 tokens；`1` 个 HTTP 尝试，4.3 秒，finish reason 为 `stop`，结构化 Lesson 被
接受，confidence 合法，未运行第二个案例。只记录脱敏状态，不记录 Key 或原始响应。

隔离安装最终使用 `--no-index --no-deps --no-build-isolation` 全绿。此前一次离线缓存
缺依赖后曾访问 PyPI 安装公开包，但没有调用 DeepSeek；最终合规隔离流程已在新的
临时目录完整重跑。

## 5. 审查结论

- Artifact Word 完成门在旧版已经接入受管 receipt，本轮是在现有边界上加强状态血缘，
  没有让 Runtime 直接读取任意文件。
- 不采用备用 Provider、绕过 Tool Manager 的 Hook/Plugin、自动远程安装、默认并行写
  Agent 或复制来源实现；这些做法违反当前模型、权限或许可证边界。
- Plugin/Marketplace、IDE/Remote/Voice/Cron 和自动 Worktree 等只进入本地未来储备，
  不为“全部实现”而添加缺少用例与隔离的代码。
- 配置迁移只增加缺失默认值，不覆盖用户已有配置；重型与高风险能力保持默认关闭。

## 6. 版本与发布

- 版本：`0.13.0`
- AgentState schema：`8`
- 核心接口契约：`5`
- 起始提交：`9b90f98d75552f09330e94904c8abce929e19856`
- 实现提交：待最终提交后记录
- 最终 tag：`v0.13.0`（待创建）
- GitHub 推送：待完成
- GitHub Actions：待完成
- 远端 main/tag 与文件树核验：待完成

提交哈希不能递归写入包含其自身的提交；最终发布提交由 `v0.13.0` tag 标识，最终
回复会给出 main/tag 的精确哈希与 Actions 结果。

## 7. 本地资料与下一步

详细逐项矩阵、测试命令、失败根因、清理快照和 clean-room 参考包位于：

```text
/mnt/d/detail/deepseek/本地开发资料/20260727-综合改进/
/mnt/d/detail/deepseek/本地开发资料/参考设计/Claude-Code-claude-b17913e/
```

四份综合方案原件已无损归档到：

```text
/mnt/d/detail/deepseek/历史资料/改进建议/20260728-v0.13.0/
```

后续只在有明确用例、默认关闭、权限隔离、故障注入和预算证明时启用未来储备能力。
