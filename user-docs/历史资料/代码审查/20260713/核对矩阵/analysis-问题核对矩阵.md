# analysis-20260713 问题核对矩阵

日期：2026-07-13

## 已确认并修复

| 编号 | 结论 | v0.8.0 处理 |
|---|---|---|
| C-002 | 损坏/外部索引可渲染 `None` | 过滤缺 path/kind/name 的 symbol |
| C-003 | 网络瞬断零重试 | timeout/408/5xx 同 Key 指数退避；401/403/429 切 Key |
| C-004 | 计划超过 50 步静默截断 | RuntimeWarning 明示原数和截断数 |
| C-005 | 清理后的步骤 ID 冲突 | 自动追加稳定数字后缀 |
| C-007 | Daemon Queue 超时无上限 | `daemon.queue_timeout_seconds` 默认 3600 |
| C-008 | Memory 去重全局 O(n²) | kind/scope 分区 + 字符 shingle 候选，精确相似度最终确认并保留近似标识符召回 |
| S-002 | 日志只按字段名脱敏 | 增加 Bearer、sk-*、API_KEY/TOKEN 内容正则；700/600 权限 |
| T-006 | 无 PTY 自动测试 | 新增真实 PTY Enter、空输入、/help、/exit |
| D-001 | `步骤8bug.docx` 为空 | 归档到中间步骤，保留为历史输入证据并在本矩阵说明 |
| D-002/3/4 | 项目 Agent 模板缺事实 | 本轮填写 architecture/context/todo（运行私有，不推送） |
| D-005 | 步骤文档堆积 | 移入 `中间步骤/` |
| R-004 | 无 CI | 已生成 Python 3.11/3.12/3.13 工作流并本地核验，但当前 OAuth 缺 `workflow` scope，GitHub 拒绝推送；保留为明确的发布后事项 |

## 已由旧版本修复或报告过时，不重复修改

| 编号 | 证据 |
|---|---|
| R-002 / DOC-002 | 远端 main 已有 `docs/releases/v0.5.0` 至 `v0.7.1`；本轮新增 v0.8.0 |
| C-012 | `workspace_memory.py` 已解析 dependencies/devDependencies 对象精确检测 React 等框架 |
| S-001 | `load_secrets_file` 已使用 `partition('=') + shlex.split`，支持引号内 `=` 和 export 空格 |
| C-001 | 短 detached hash 的切片不会产生空值；本轮另补了 worktree `.git` 文件支持 |
| T-001~T-005 | daemon/reflection/workspace/planner 等已在 v06/v07 测试覆盖，报告按“文件名”误判 |
| DOC-001 | 58 是展开后的 test case 数，不是测试文件数；v0.8.0 明确写 86 cases |
| D-007 | 当前目录按 AGENTS.md 是文档入口，不应初始化为第二个源码仓库；源码仓库在 `~/AI-Agent` |

## 不作为缺陷处理

| 项目 | 原因 |
|---|---|
| C-006 parallel 最少 8 任务 | 是文档明确的阈值策略；是否调低需产品选择 |
| S-003 messages 全脱敏 | 安全默认；完整 prompt 日志会增加凭据和隐私风险 |
| 代码/文档中英混合 | 系统 Prompt/源码英文、用户文档中文是合理分工，不构成运行错误 |
| R-003 GitHub description/topics | 已补充英文项目简介与 `deepseek/ai-agent/cli/python/wsl/developer-tools` topics |
| R-001 LICENSE | 确实缺少，但属于法律授权选择；本轮不替用户默认授予 MIT/Apache 权利 |

## 旧报告未发现、v0.8.0 新增修复

- 自适应 simple/standard/large/deep 任务策略和 large/deep starter Task Graph。
- DeepSeek reasoning SSE 可见输出、工具调用 delta 重组、部分流禁止重放。
- `.project-agent/memory` 私有边界与 gitignore。
- Agent 目录/源码文件符号链接逃逸。
- Queue ID 路径穿越、锁前陈旧快照导致重复执行。
- Docker `--mount src=/`、socket、device、host namespace 变体。
- Shell 超时遗留子进程。
- TSX 被默认 semantic_languages 静默跳过。
- Resume 短提示把 deep 策略降级。
- 首次 Project 初始化竞态与 Context/Workspace 固定临时文件竞态。
- 同一 Session 并发 Resume 的最后写入者覆盖和重复副作用风险。
- Daemon PID 子串误识别、PID 复用和 Queue 超时遗留孙进程。
- HTTP 请求体/Header、文档、下载、MCP 分页和 Shell 输出无统一硬上限。
