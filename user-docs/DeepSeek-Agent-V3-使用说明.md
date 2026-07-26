# DeepSeek Agent V3 使用说明（0.12.0）

更新时间：2026-07-26

## 1. 版本与目录边界

当前版本为 `0.12.0`，AgentState schema 为 `7`，核心接口契约为 `3`。架构保持：

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ContextBuilder -> ContextPackage -> PromptBuilder
ToolRequest -> PermissionManager -> ToolResult
```

DeepSeek 是唯一推理 Provider。0.12.0 没有加入备用模型、第二套 Runtime、自动 A/B 或绕过 Tool Manager 的命令入口。

目录职责：

```text
~/AI-Agent/                         程序源码、测试和发布说明
~/.config/deep-agent/              用户配置与私有 API Key
~/.local/share/deep-agent/         Memory、Vector、日志、指标和运行状态
<项目>/.project-agent/             项目上下文、Session、快照和私有工具附件
/mnt/d/detail/deepseek/             用户文档、需求归档和审计交付
```

不得把真实 Key、Memory、Session、日志、浏览器会话、缓存或 `.project-agent` 私有数据提交到 Git 或复制到文档。

## 2. 安装、升级与启动

```bash
cd ~/AI-Agent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/agent --version
```

可选重型能力按需安装，默认不要求：

```bash
.venv/bin/python -m pip install -e '.[browser,semantic,document,vector]'
```

在需要处理的项目目录启动：

```bash
cd /path/to/project
agent init
agent
agent "分析当前项目并给出验证结论"
```

## 3. 私有 DEEPSEEK_API_KEY

推荐位置：

```text
~/.config/deep-agent/secrets.env
```

```bash
chmod 600 ~/.config/deep-agent/secrets.env
```

文件中填写：

```text
DEEPSEEK_API_KEY=replace_with_your_valid_key
```

支持英文或中文逗号分隔的 Key 池。不要把 Key 写进源码、项目配置、测试输出或聊天记录。

0.12.0 在 Project、Memory 和 Vector 初始化前检查 Key。缺 Key 时普通任务退出 `1`，不会先创建项目状态或加载 Chroma。当前用户 API 余额已耗尽，本次发布未调用 `doctor --online` 或真实 DeepSeek 任务；余额恢复后再补一次有预算上限的在线验收。

## 4. 常用命令

```bash
agent --help
agent --version
agent doctor
agent doctor --online
agent init
agent "实现并验证这个功能"
agent sessions
agent resume
agent resume --session SESSION_ID "继续原任务"
agent context show
agent context refresh
agent tools
agent memory search "query"
agent memory list --kind Correction
agent memory stats
agent memory maintain
agent memory maintain --apply
agent health
```

退出码：

```text
0  Session 已完成
1  配置、启动或 Runtime 异常
2  Session 未完成但已保存，可 Resume
```

交互命令包括 `/new`、`/resume`、`/sessions`、`/status`、`/undo`、`/yolo on|off`、`/super-yolo on|off`、`/help` 和 `/exit`。

普通任务按一次 `Enter` 提交。提交后立即显示处理中状态。空输入有明确反馈；`Ctrl+C` 返回可恢复状态。ANSI Prompt 的不可见控制字符由 Readline 正确包裹，CJK 和组合 emoji 按终端显示宽度裁剪。

## 5. 0.12.0 可靠性改进

每个 turn 默认有三个跨阶段预算：

```yaml
runtime:
  budget:
    enabled: true
    max_model_requests_per_turn: 64
    max_total_tokens_per_turn: 1000000
    max_elapsed_seconds_per_turn: 3600
  resilience:
    max_corrective_rounds: 2
    max_abnormal_finish_recoveries: 1
```

预算覆盖主循环、上下文压缩、截断续写、overflow 恢复和最终合成。每次网络请求前先预留并 checkpoint；预算耗尽后保存失败原因和 Resume 指令，不继续请求或执行新工具。它是每 turn 预算，不是跨所有 Resume 的 Session 生命周期总预算。

PlanStep 新增 `step_type`、`parent_id`、预计工具轮次、Artifact ID、验证规则和进度权重。新计划不能预填 completed/failed/skipped，依赖和父节点循环会被拒绝。UI 进度只计算已满足步骤；运行中不会猜 ETA 或显示 100%。

## 6. Artifact 与 Word 验证

0.12.0 提供有界、确定性的 JSON、YAML、Python 和 DOCX 校验。DOCX 检查：

- ZIP/OPC 必需成员、路径和重复 member；
- CRC、XML、加密或符号链接 member；
- 压缩比、解压总量和 ZIP Bomb；
- `word/document.xml` 中存在非空正文。

`document.parse` 生成不含正文的受管验证 receipt，记录相对路径、格式、大小、SHA-256、完整性和检查结果。Runtime 只消费 ToolResult 元数据，不绕过 Tool Manager 读取文件。

Session 摘要回放 `file_apply`、delete 和 undo 的最终状态。已删除或撤销创建的产物不会继续显示 verified。旧 Session 中没有 receipt 的 Word parse 记录会安全失效；Resume 后重新执行一次 `document_parse` 即可。

## 7. Memory、Vector 与性能历史

新 Memory 写入必须使用受支持的 `MemoryKind`，旧数据库未知类型仍可读取。容量维护默认 dry-run，只有 `memory maintain --apply` 才执行淘汰；Correction 和 Decision 受保护。显式 feedback ID 保证置信度更新幂等，普通检索不会自动把“使用过”当成“正确”。

主要默认值：

```yaml
memory:
  vector_enabled: false
  max_items: 5000
  max_storage_mb: 100
  capacity_scan_limit: 5000
  capacity_report_limit: 100
  protect_kinds: [Correction, Decision]
  confidence:
    use_bonus: 0.02
    contradiction_penalty: 0.15
    lower_bound: 0.1
    upper_bound: 0.95

events:
  performance_history_enabled: true
  performance_history_max_records: 200
```

即使旧配置显式打开 Vector，Chroma 也只在第一次真实向量操作时加载。性能历史只保存有界安全标量和 `run_id`，不保存 Prompt、正文、工具参数或凭据，也不会自动修改路由和配置。

配置迁移只补充缺失默认值，不覆盖已有用户值。

## 8. 验证结果

2026-07-26 的离线结果：

```text
502 tests passed
Ruff check passed
Ruff format check passed（agent/tests/scripts）
compileall passed
pip check passed
git diff --check passed
```

隔离 XDG 实例冒烟通过：`deep-agent 0.12.0`、`--help`、`agent init`、缺 Key 退出 `1` 且不创建 `.project-agent`、launcher 版本一致。全量测试包含真实 PTY 回归。

未运行新的在线 DeepSeek 请求。历史 v0.11.0 六 Word 和短文本在线案例仍有成功记录，但大型 TypeScript 候选失败，不能把 0.12.0 离线回归写成大型在线成功。

## 9. 风险与回滚

仍未实现：跨 Resume 生命周期预算、Durable Intent Journal、外部副作用 exactly-once、章节级事务式文档流水线、动态 replan、并行写步骤/子 Agent、自动 A/B/自动调参、Memory import/export 和备用 Provider。

回滚前先完成或导出 v0.12.0 Session。v0.11.0 不能加载 schema 7 Session：

```bash
cd ~/AI-Agent
git switch --detach v0.11.0
.venv/bin/python -m pip install -e .
```

不要删除 Memory、Session、Vector、日志或项目 `.project-agent` 数据。返回最新版：`git switch main`。

## 10. 审计与参考

本轮整合报告位于：

```text
/mnt/d/detail/deepseek/项目运行审计与改进建议/20260726-v0.12.0/
```

四份原始需求已无损归档到：

```text
/mnt/d/detail/deepseek/历史资料/改进建议/20260726-v0.12.0/
```

参考项目固定为 `https://gitee.com/free/claude-code/tree/claude/` commit `b17913e26fd4278ad5cd4b32ed3bde86bf1444e9`。其 README 自述为泄露的 Anthropic 专有源码快照且没有可复制许可证，因此本项目只核对运行时持有任务状态、工具结果配对、有界输出和有限恢复等行为边界，不复制源码。
