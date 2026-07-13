# DeepSeek Agent V3 工作日志（0.8.0）

日期：2026-07-13

## 0.8.0 自适应深度执行工作

### 目标

逐条复核 `analysis-20260713`，修复全部能够复现的问题；同时解决困难问题思考超时、长时间无可见输出，以及大规模文本/代码不能分块处理的问题。参考 `https://gitee.com/free/claude-code` 的 Task、Thinking、Compact、Resume 思路，但保持 DeepSeek 唯一模型和现有权限架构。

### 根因

1. 所有请求共用同一个 8 轮策略，简单问题浪费、复杂问题预算不足。
2. 复杂任务只依赖 Prompt 要求模型自行规划，首次长推理仍可能超时。
3. 非流式请求在首字节前没有持续进度，用户误以为程序卡死。
4. 网络瞬断与临时 5xx 零重试；流式部分输出若盲目重放可能重复工具副作用。
5. 旧审查报告包含过时或误报项，同时遗漏私有 Memory、符号链接、Queue 路径穿越、Docker 参数绕过和进程组孤儿等真实问题。

### 实现

- 新建 `agent/task_strategy.py`，本地选择 simple/standard/large/deep。
- large/deep 自动建立有依赖、有完成标准、有重试次数的 starter Task Graph。
- DeepSeek thinking 按任务选择 disabled/high/max，SSE 流式重组 reasoning/content/tool calls。
- Console 显示 elapsed Thinking、模式、轮次、当前步骤、工具状态和 reasoning delta。
- 网络超时、408、500/502/503/504 做有限指数退避；认证/限流切 Key。
- 部分流中断禁止自动重放，Session 标记 interrupted/resumable 并报告准确 ID。
- Resume 继承更强的原策略和计划，不被短续写降级。
- Planner 修复 ID 清理冲突，超过 50 步时显式警告。
- Daemon Queue 加绝对超时；Shell 超时终止整个进程组。
- 修复 worktree Git branch、TSX 默认语义索引、损坏 symbol 的 None 渲染。
- Memory 候选倒排索引降低去重比较量，同时保留前缀不同的相似记录召回。
- JSONL 内容正则脱敏，日志目录/文件使用 700/600。
- `.project-agent/memory/` 加入 ignore 和私有路径；拒绝 Agent 目录/源码符号链接逃逸。
- Queue ID 先校验、锁内重载 canonical 状态、唯一临时文件、按 mtime 选最新。
- 普通/YOLO 模式拒绝 Docker root、socket、device 和 host namespace 访问。
- 新增 GitHub Actions Python 3.11/3.12/3.13 矩阵，自动执行 Ruff、pytest 和 compileall。
- 首次 Project 初始化增加项目锁和原子 YAML；Context/Workspace 缓存改用唯一临时文件。
- 同一 Session 的并发 Resume 增加 per-session flock，拒绝重复 turn。
- Daemon 使用精确 argv + `/proc` starttime 身份校验；Queue 超时终止整个进程组。
- Shell 输出、HTTP 请求体/Header、文档输入、浏览器下载和 MCP 分页增加硬上限。

### 审查结论

已确认并修复：C-002、C-003、C-004、C-005、C-007、C-008、S-002、T-006，以及审查未列出的 TSX、Memory 私有边界、symlink、Queue traversal/stale load、Docker 变体、子进程超时问题。

不处理的旧结论：`docs/releases/` 实际已存在；React 检测和 secrets.env 引号内 `=` 已修；当前目录本来就是文档入口，不应单独初始化为第二套源码仓库；parallel 的 8 任务阈值是明确架构策略；代码/用户文档中英分工不是运行缺陷；LICENSE 需要用户选择法律许可，不能替用户假定。

### 测试

```text
86 tests passed
Ruff check passed
Ruff format check passed
compileall passed
真实 PTY：空 Enter、/help、/exit passed
DeepSeek SSE reasoning/tool call assembly passed
超时 retry、partial stream no replay passed
symlink/traversal/Queue stale state/Docker escape/process group passed
Memory 1200 条候选性能回归 passed
```

### 版本与发布

版本升级为 `v0.8.0`（新能力使用 minor 版本）。源码 README、实现说明、`docs/releases/v0.8.0.md`、用户 README、工作日志和 Word 文档同步更新。

核心发布提交：`1770b39`；后续仅追加发布核验文档。GitHub 远端 `main` 与注释标签 `v0.8.0` 已在发布闭环末端通过 `git ls-remote` 核验为同一提交；最终对象哈希以 GitHub 标签和本次最终回复为准，避免文档提交自引用造成无限追加提交。

推送时发现当前 GitHub OAuth 凭据只有 `repo`、没有 `workflow` scope，GitHub 明确拒绝新增 `.github/workflows/test.yml`。为不阻塞源码、安全修复、测试、文档和标签发布，本轮将 CI 工作流草案留作后续改进，未在远端声称 CI 已启用；本地 Ruff、pytest、format、compileall 全部通过。

### 下一步

- 为 Parallel worker 增加按任务/全局超时与保留失败 worktree 策略。
- 为工具执行增加 durable intent/result journal，缩小 SIGKILL 后恢复重复副作用窗口。
- 为 Session 增加 revision/CAS，支持未来可控的并发 turn 合并策略。
- 若用户选择许可证，再增加 LICENSE；许可证属于法律授权，不自动假设。

## 一、工作目标

在 `v0.7.0` 后修复 WSL/GNU Readline 交互输入提交问题，并补充跨 AI 协作规范。V1.0 仅写入后续规划，不在本次实施。

## 0.7.1 补丁工作

### 问题表现

交互模式中，用户输入任务后按 `Enter` 在部分 WSL 终端中会看起来没有提交，光标或换行位置错位，容易误认为 Agent 仍在等待继续输入。

### 根因

`ConsoleUI` 使用 GNU Readline 的 `input()` 读取命令，但彩色 Prompt 包含裸 ANSI 转义序列。Readline 会把这些不可见字节误算为可见字符，造成光标位置和自动换行计算错误。问题不是 DeepSeek 请求未发出，而是输入界面的可见状态不可靠。

### 修复

- Readline 活跃时用 `\001` 与 `\002` 包裹 ANSI 开始/结束序列。
- 调用 `input()` 前刷新 stdout。
- 空 Enter 输出具体提示。
- 自然语言任务提交后输出处理状态。
- `Ctrl+C` 不再让交互程序直接退出，保留可恢复 Session 并回到 Prompt。

### 验证

- 新增两项 ConsoleUI 回归测试。
- 在真实 PTY 中验证：空 Enter 显示提示，`/help + Enter` 立即执行，`/exit` 正常退出。
- 完整测试、Ruff、格式和编译检查通过。

### 附加根目录健壮性修复

在隔离终端验证中发现，Project Manager 以前只要看见祖先目录中存在 `.git` 就将其认定为 Git 根。空 `.git` 占位目录会让项目错误落到父目录。现已要求 `.git/HEAD` 存在，或 `.git` 文件以 `gitdir:` 指向 worktree 元数据，并新增对应回归测试。

### 发布

版本升级为 `v0.7.1`，发布到现有 GitHub 仓库并创建标签。当前目录的使用说明、工作日志、Word 文档和 `AGENTS.md` 同步更新。

正式发布提交：`94b6bab`（`fix: Deep Agent v0.7.1 interactive input`）。GitHub `main` 已推送，`v0.7.1` 注释标签指向该提交。

用户关闭 Word 后，已将旧版 `0.7.0` 使用说明归档到 `老版使用说明/`，工作日志归档到 `老版工作日志/`。新版 `0.7.1` Word 文档保留在当前目录，作为唯一的当前版本说明。

## 二、版本推送理由与优化方向

### v0.5.0

推送理由：形成可稳定使用的安全基线，先解决文件预览、快照回滚、外部 MCP、浏览器登录态、下载、Queue 和 worktree 并行等实际闭环。

对应提交：`4958b59`。

优化方向：不继续堆叠工具，转向统一的 Planner、状态和项目事实。

### v0.6.0

推送理由：把复杂任务、Resume、Queue 和 Parallel 统一到 AgentState 与依赖感知 Task Graph，并增加 Workspace Memory、Reflection、Execution Context 和 Capability Health。

对应提交：`9fd2595`。

优化方向：增加修改后即时诊断、语义关系、Memory 生命周期和后台增量维护。

### v0.7.0

推送理由：使 Agent 能在写入代码后立即发现 Python/JS/TS 问题，长期控制 Prompt 与 Memory 增长，并允许后台低频维护索引。

优化方向：V1.0 冻结核心 Runtime 接口，统一 Context Builder 和 Event Bus 副作用边界，持续通过 Capability Registry 扩展能力。

## 三、0.7.0 实现过程

### 1. LSP Diagnostics

- 新建 `agent/tools/lsp.py`。
- Pyright 解析 JSON Diagnostics。
- TypeScript 使用 `tsc --noEmit`，解析文件、行、列、错误码和消息。
- Pyright 与 TypeScript 独立判断可用性，不要求同时安装。
- 扫描跳过依赖、构建、Git、Agent 私有目录。
- `file_apply` 后自动诊断，但诊断错误不把已成功的原子写入标成失败。

### 2. Semantic Context

- `index.semantic.json` 增加 module summary、exports、relationships。
- Python 相对 import 和 JS/TS 相对 import 映射到项目内部文件。
- 外部依赖不生成虚假的内部边。
- Prompt 只加载受限摘要，最终强制执行 `max_prompt_chars`。

### 3. Memory 生命周期

- SQLite 增加 `confidence`、`use_count`、`last_used_at`、`expires_at`、`merged_into`。
- 普通检索命中后更新使用次数和最后使用时间。
- 非保护 Memory 自动设置默认过期时间。
- `agent memory maintain` 预览去重和过期清理。
- `agent memory maintain --apply` 执行维护。
- 合并使用“首选记录相似度”分组，避免相似链导致端点低于阈值仍被归并。
- merged 记录从 list、stats、recent、FTS、LIKE、Vector fallback 全部过滤。
- 删除主记录时一并清理归并子记录和向量数据。

### 4. Compact Resume

- 参考 Claude Code 的有界上下文思想。
- Resume 不再无限保留历史 raw Tool transcript。
- 保留 System Prompt、上一轮最终结果摘要、AgentState、Execution Context、当前项目 Context、Memory 和能力摘要。

### 5. Daemon

- 新建 `agent/daemon.py`。
- `agent daemon start/run/status/stop`。
- 标准库轮询，无额外 watchdog 依赖。
- 每项目独立 PID、文件锁、stop 文件、状态 JSON 和日志。
- PID 操作前检查 `/proc/<pid>/cmdline`，避免 PID 复用误杀无关进程。
- 默认只维护 Context 和 Memory，Queue 后台执行默认关闭。

### 6. 并发和存储

- SQLite 使用 WAL、30 秒 busy timeout。
- 旧数据库 FTS 数量不一致时执行 rebuild，避免迁移前 Memory 无法全文检索。
- Queue 增加跨进程文件锁，防止多个终端或 Daemon 重复执行。
- 默认配置迁移增加跨进程锁和唯一临时文件，修复多个 Agent 同时启动时 `config.yaml.tmp` 被抢占的问题。

## 四、完整代码审查发现与修复

1. Memory 合并后可能被 FTS/limit/recent/stats 重新返回：已统一 SQL 条件与 Vector fallback。
2. Memory 相似链可能过度合并：改为 keeper-relative 分组。
3. `recent()` SQL 的 AND/OR 优先级错误：增加括号。
4. 过期时间字符串直接比较不够稳健：改为 timezone-aware datetime。
5. LSP 误要求 Pyright 和 tsc 同时存在：改为独立降级。
6. `file_apply` 写入成功但诊断报错时整体失败：改为成功结果附带诊断。
7. Context 添加语义摘要后可能超过总上限：最终渲染再次截断。
8. Resume 历史消息无限增长：改为 compact checkpoint。
9. Runtime 资源释放只依赖进程退出：增加显式 `runtime.close()`，并在 REPL/Queue/一次性命令退出时调用。
10. Daemon 每轮恢复 active Queue 会干扰前台任务：删除该行为。
11. Daemon PID 可能复用：增加命令行和项目根路径身份检查。
12. Daemon 和 Memory 输出未完全使用 AppConfig.data_dir：统一配置注入。
13. Queue 缺少跨进程互斥：增加 `fcntl.flock`。
14. Parallel worker 异常会丢失任务编号：保留 future 对应的 index 和 prompt。
15. Broken 能力仍出现在 Prompt 摘要和 loaded_tools：统一按 Health 过滤。
16. Task Graph current_step 可能指向依赖未满足步骤：只选择 ready pending step。
17. Workspace Memory 假设 package.json dependencies 一定是对象：增加类型检查。
18. Workspace Memory 读取异常可中断 Context：增加 OSError 容错。
19. Event 日志脱敏覆盖不足：增加 cookie、password、secret、apikey 等字段。
20. 多 Agent 同时配置迁移发生唯一临时文件竞争：加入 `.config.lock` 和 UUID 临时文件。

## 五、参考工程分析

参考工程：`https://gitee.com/free/claude-code`。

采纳：

- Context 必须有边界，不能依赖模型窗口无限增长。
- 资源加载必须限制路径、数量、大小和格式。
- 工具不可用、需配置、已禁用、故障必须可观察。
- Session/worktree 状态必须支持恢复。
- Tool Loop 必须显式，不能让框架隐藏权限边界。

不采纳：

- Java/Spring 第二 Runtime。
- 多模型 Provider 抽象，当前坚持 DeepSeek 唯一模型。
- 全屏 TUI、JAR 插件、任意 Hook、遥测。
- 无限制 Skills/Agents 自动加载。
- 任何绕过 Capability/Permission 的调用路径。

## 六、测试与运行验收

最终结果：

```text
55 tests passed
Ruff check passed
Ruff format check passed
compileall passed
12 个 agent health 并发启动通过
DeepSeek API Key 5/5 在线通过
Docker daemon 与代理正常
docker run --rm hello-world 通过
Capability Health：所有已启用能力 Available
Daemon run --once：完成索引维护并清理 PID
```

## 七、关键文件

```text
~/AI-Agent/agent/daemon.py
~/AI-Agent/agent/tools/lsp.py
~/AI-Agent/agent/memory.py
~/AI-Agent/agent/context.py
~/AI-Agent/agent/prompt.py
~/AI-Agent/agent/task_queue.py
~/AI-Agent/tests/test_v07_intelligence.py
~/AI-Agent/docs/releases/v0.7.0.md
```

## 八、后续 V1.0 方向

V1.0 暂不实施，只保留以下方向：

1. 冻结 `CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission` 核心接口。
2. Prompt Builder 不直接读取文件，所有上下文统一由 Context Builder 提供。
3. Runtime 不直接产生模块副作用，统一发布 Event，由订阅者维护 Memory、Session、Reflection、Workspace 和统计。
4. GitHub、Jira、Notion、数据库、更多 MCP 和开发工具继续通过 Capability Registry 接入。
5. 保持 DeepSeek 唯一模型，不引入多 Agent 调度中心。
