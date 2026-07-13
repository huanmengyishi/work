# Deep Agent 协作规范

本文件是进入此目录工作的 Claude、Codex、Cursor、Gemini、DeepSeek 或其他 AI 的强制协作约定。开始修改前先阅读本文件和当前 `README.md`。

## 1. 项目边界

- 当前目录 `/mnt/d/detail/deepseek` 是用户文档、需求和协作入口，不是 Deep Agent 程序源码目录。
- 程序源码位于 `~/AI-Agent`。
- 用户配置和 API Key 位于 `~/.config/deep-agent`；不得读取、输出、提交或复制真实 Key。
- 运行数据位于 `~/.local/share/deep-agent`；不得提交 Memory、Vector、日志、浏览器会话、缓存或项目 `.project-agent` 私有数据。
- GitHub 发布仓库为 `https://github.com/huanmengyishi/work`。每次完成实质性工作后必须推送到该仓库。

## 2. 每次工作的强制闭环

1. 阅读现有代码、`README.md`、`DeepSeek-Agent-V3-工作日志.md` 与本文件，确认当前版本和未完成事项。
2. 先定位问题和影响范围，再修改代码；不得用临时绕过替代根因修复。
3. 为每个缺陷或新能力增加/更新自动化测试，并运行与变更相称的验证。
4. 更新当前目录的 `README.md`，说明用户如何使用、配置、验证或回滚该功能。
5. 更新当前目录的 `DeepSeek-Agent-V3-工作日志.md`，记录目标、根因、改动、测试、版本、提交和后续方向。
6. 对用户可见的重大变更，重新生成版本化 Word 使用说明和工作日志；将旧版 Word 放入 `老版使用说明/` 或 `老版工作日志/`，不要让根目录堆积旧文件。
7. 更新程序内 `README.md`、`docs/releases/` 和版本号；修复类变更使用 patch 版本号，新能力使用 minor 版本号。
8. 将源码、测试、发布说明和最新版用户文档同步到 GitHub 仓库；提交信息应说明版本和目的，创建对应 Git tag，推送 `main` 和 tag。
9. 推送后验证远端 `main` 与 tag 指向预期提交，并在最终回复中给出版本、提交、验证结果和文档位置。

## 3. 技术约束

- 保持架构边界：`CLI -> Runtime -> AgentState -> Prompt -> Capability -> Permission`。
- DeepSeek 是唯一推理模型；不得无授权加入多模型 Provider 或替代模型。
- 工具必须通过 `ToolRequest -> PermissionManager -> ToolResult`；Runtime、Prompt 或模型不得直接绕过 Tool Manager 执行系统命令。
- 任何写文件行为优先遵循 `file_diff -> file_apply -> file_undo` 的快照流程。
- 重型能力默认关闭，配置迁移只能增加默认值，不能覆盖用户配置。
- 所有外部输入都要限制路径、数量、大小、超时和输出长度；日志必须脱敏 API Key、Cookie、密码、Token、Secret。
- 不要使用 `git reset --hard`、强制覆盖用户改动、删除用户 Memory 或项目数据，除非用户明确要求。

## 4. 文档要求

使用说明至少包含：版本、安装/启动、`DEEPSEEK_API_KEY` 位置、主要命令、配置开关、验证方式、风险或回滚方式。

工作日志至少包含：日期、目标、根因、实现、测试、审查发现、版本、提交哈希、GitHub 推送结果、下一步方向。

文档必须与实际代码一致。不要在文档、提交、日志或测试输出中写入真实凭据。

## 5. 交互输入规范

- 普通任务输入后按一次 `Enter` 提交。
- Prompt 使用 ANSI 颜色时，Readline 模式必须将不可见控制字符用 `\001`/`\002` 包裹。
- 提交后 UI 必须显示正在处理状态；空输入必须有明确反馈；`Ctrl+C` 应返回可恢复的交互状态。
- 任何交互 UI 修改必须通过真实 PTY 或等效终端测试，不能只依赖普通单元测试。
