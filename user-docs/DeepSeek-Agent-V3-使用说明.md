# DeepSeek Agent V3 使用说明（0.13.0）

更新日期：2026-07-29

## 1. 版本与边界

当前版本为 `0.13.0`，AgentState schema 为 `8`，核心接口契约为 `5`。

DeepSeek 是唯一推理模型。所有系统动作继续经过：

```text
CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission
ToolRequest -> PermissionManager -> ToolResult
```

0.13.0 没有加入备用模型，也没有让 Runtime、Prompt 或模型绕过 Tool Manager。
Vector、自适应策略和实验框架仍默认关闭。

目录职责：

```text
~/AI-Agent/                         程序、测试和当前公开文档
~/.config/deep-agent/              用户配置和私有 API Key
~/.local/share/deep-agent/         Memory、Vector、日志和运行数据
<项目>/.project-agent/             项目上下文、Session、快照和附件
/mnt/d/detail/deepseek/             本地用户文档、历史资料和开发证据
```

不得提交或复制真实 Key、Memory、Session、日志、浏览器会话、缓存和项目私有数据。

## 2. 安装与启动

```bash
cd ~/AI-Agent
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
.venv/bin/agent --version
```

可选重型能力按需安装：

```bash
.venv/bin/python -m pip install -e '.[browser,semantic,document,vector]'
```

在目标项目中启动：

```bash
cd /path/to/project
agent init
agent
agent "分析当前项目并验证改动"
```

普通输入按一次 `Enter` 提交。提交后会显示处理状态；空输入有提示；`Ctrl+C`
返回可恢复的交互状态。

## 3. 配置 DeepSeek Key

推荐位置：

```text
~/.config/deep-agent/secrets.env
```

```bash
chmod 600 ~/.config/deep-agent/secrets.env
```

文件内容：

```text
DEEPSEEK_API_KEY=replace_with_your_valid_key
```

支持英文或中文逗号分隔的 Key 池。不要把 Key 写进源码、项目配置、测试输出或
聊天记录。先运行不联网的 `agent doctor`；`agent doctor --online` 会逐个校验已配置
Key，瞬时错误还可能触发重试，因此可能产生多次真实 DeepSeek 请求，只在确实需要
在线校验时使用。

## 4. 常用命令

```bash
agent --help
agent --version
agent doctor
agent init
agent "实现并验证这个功能"
agent sessions
agent resume
agent resume --session SESSION_ID "继续原任务"
agent context show
agent context refresh
agent tools --all
agent health
agent memory search "query"
agent memory list --kind Correction
agent memory stats
agent memory maintain
agent memory maintain --apply
agent memory export backup.json --scope project
agent memory import backup.json --target-scope preserve --conflict skip
```

Memory `maintain` 和 `cleanup` 默认只预览，只有加 `--apply` 才修改数据。导入导出
有格式、记录数、路径、字符和字节上限，并拒绝符号链接等不安全输入。导出文件含
真实 Memory 内容，不得提交 Git；使用 `--force` 或 `--conflict replace` 前先另做备份。

交互命令包括 `/new`、`/resume`、`/sessions`、`/status`、`/undo`、
`/yolo on|off`、`/super-yolo on|off`、`/help` 和 `/exit`。

退出码：

```text
0  Session 已完成
1  配置、启动或 Runtime 异常
2  Session 未完成但已保存，可以 Resume
```

## 5. Memory 智能精炼

0.13.0 对新安装或缺失配置键使用以下默认值：

```yaml
memory:
  smart_reflection: true
  smart_reflection_min_tool_calls: 5
  smart_reflection_max_input_chars: 12000
  smart_reflection_max_output_tokens: 768
  smart_reflection_max_output_chars: 5000
```

触发条件同时满足时，成功任务会额外发起一次 DeepSeek 逻辑请求，总结“这次学到的
经验”：

- 当前 turn 已成功完成任务；
- 当前 turn 至少执行 5 次受管工具调用；
- 统一模型请求、Token 和时间预算仍有余额；
- 本 turn 尚未尝试过精炼。

硬门槛不能通过配置降到 5 以下。精炼请求不带工具，每 turn 最多一次逻辑请求；
底层仍服从有界网络重试和 Key 轮转。异常、超时、预算不足、非法 JSON 或非法工具
协议都会安全跳过，不改变已经完成的任务。模型返回的 `kind`、`confidence` 和
脱敏后的 `tags` 会真实落库。升级时已有显式 `false` 不会被迁移覆盖。若不需要
额外请求，可设置：

```yaml
memory:
  smart_reflection: false
```

## 6. 长任务与状态管理

- Runtime 主循环、请求恢复、完成门、工具批处理和终态处理已拆分，但仍是同一
  Runtime 和同一权限链。
- AgentState 最多保留 200 条热工具记录，并限制总大小、单记录、字典键、集合和
  嵌套深度。
- Resume 只把最近 16 个完整模型轮次加载到内存；被保留的完整 JSONL 按带哈希
  校验的冷世代管理，最多 4 个冷世代、总计 128 MiB。
- History Snip 在模型压缩前做零模型、完整轮次安全裁剪，默认开启。
- Durable Intent Journal 与 ArtifactRegistry 保存有界恢复意图和受管产物血缘。
- Memory 查询使用 128 项 TTL/LRU 温缓存；写入、反馈、维护和 transfer 会正确
  失效缓存。

可选策略默认关闭：

```yaml
optimizer:
  adaptive_convergence_enabled: false
  strategy_adjustment_enabled: false
  experiments:
    enabled: false

runtime:
  convergence:
    history_snip_enabled: true
```

配置迁移只补缺失默认值，不覆盖用户已有选择。

## 7. 验证

本地复验命令：

```bash
cd ~/AI-Agent
.venv/bin/ruff check agent tests scripts
.venv/bin/ruff format --check agent tests scripts
.venv/bin/python -m compileall -q agent tests scripts
.venv/bin/python -m pytest
.venv/bin/python -m pip check
git diff --check
```

测试包含真实 PTY 的 Enter、处理中状态、空输入、Unicode 宽度和可恢复 `Ctrl+C`。
发布测试还会核对 GitHub 根目录仅有当前两份 Word、`user-docs/` 仅有当前四份文档，
并重新打开 Word 校验 OPC/CRC、正文、版本和日期元数据。

真实 DeepSeek 实例遵循单案例纪律：全部离线门通过后，每轮发布最多运行一个与
改动直接相关、预算有界的案例；出现任一错误立即停止，不连续重试消耗 API。

## 8. 风险与回滚

- 开启智能精炼后，符合门槛的成功任务最多多一次模型请求。
- schema 8 Session 不保证能被旧版读取；回滚前先完成或保留当前 Session。
- Adaptive、Strategy Adjuster 和 ExperimentRunner 是可选能力，不建议未验证即开启。
- `--super-yolo` 会放宽硬权限策略，应视为高风险模式。

回滚到上一版：

```bash
cd ~/AI-Agent
git switch --detach v0.12.2
.venv/bin/python -m pip install -e .
```

不要删除 Memory、Session、Vector、日志或 `.project-agent` 数据。返回最新版时执行
`git switch main` 并重新安装 editable 包。

## 9. 文档范围

GitHub 只发布实际应用、测试/CI、当前发布说明和最新版简明使用说明/工作日志。
逐项落实矩阵、测试明细、错误排查、参考设计和历史修改建议仅保存在本地：

```text
/mnt/d/detail/deepseek/本地开发资料/
/mnt/d/detail/deepseek/历史资料/
```
