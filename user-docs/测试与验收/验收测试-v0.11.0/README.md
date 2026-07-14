# Deep Agent v0.11.0 测试与验收摘要

日期：2026-07-15

## 边界

本目录只保存可发布的脱敏摘要。原始 PTY、Session、Workspace、Memory、附件、缓存、日志、完整工具参数和模型推理不得同步到源码仓库。

参考项目固定为 `free/claude-code` 的 `claude` 分支提交 `b17913e26fd4278ad5cd4b32ed3bde86bf1444e9`。参考原生机制与 Deep Agent 本地设计分别记录，不能把 soft/hard tool turns、final synthesis 或 DSML 防御写成参考项目原生功能。

## 离线验证

当前冻结候选收集并通过 454 项 pytest；全新 Python 3.14 环境安装 `.[dev,browser,semantic]`、`pip check` 和同一全量也通过。新增的确定性回归直接覆盖在线失败的原始关键句，并覆盖：

- 禁止读取或输出凭据不能被误判为要求生成文件；
- 明确创建报告文件时，即使同时禁止输出密钥，仍必须保留 artifact 门；
- “没有充分证据则跳过 implement、不要修改”必须识别为 conditional mutation；
- 4 次 main-loop 工具轮后进入 1 次 tool-free final synthesis；
- 无可证实缺陷时 Session 可以在完成 verify 后成功结束；
- 不产生 `file_apply`、managed-write 错误或 Resume 提示。
- 第一次静态检查失败后，等价 shell 与 LSP 验证会在执行前被拒绝，实际验证仍只有一次。
- `not_executed` 不消耗唯一验证机会，wrapper 可识别，compound 多验证 shell 在执行前拒绝；
- TaskRoute schema 2 的文件/目录 hints、精确 `make_dir`、active apply/delete/undo 与 Word preview lineage；
- structured tool calls 与 DSML 正文同现时整批零执行，以及 open compaction circuit 的 overflow 路径零语义压缩调用；
- 18 项 CLI/Console，其中真实 25 列 PTY 不拆分旗帜、肤色、Keycap、ZWJ 或组合字符。

最终静态检查、PTY、全新安装和 GitHub Actions 结果以发布工作日志为准。

## 在线验收

### 六 Word 汇总

结果：通过。

- 9 次逻辑模型请求；
- 19 次工具调用；
- 完成受管 render、apply 和重新打开验证；
- artifact、内容和 Session 语义门全部通过。

### 约 2500 字总结

结果：通过。

- 3 次逻辑模型请求；
- 2 次只读工具调用；
- 无文件写入；
- 内容、证据和 Session 语义门全部通过。

### 大型 TypeScript 研究快照

最后一次修复前在线运行结果：未整体通过，不能写成成功。

- 36 次逻辑模型请求，其中 31 个完成工具结果批次；
- 55 次工具调用；
- API 报告总计 1,477,342 tokens；
- 无 context-compaction 模型请求，无 HTTP 传输重试；
- 最终中文报告包含至少 8 项带源码路径的优点、Bug/限制结论、修改文件、验证结果和剩余风险；
- 内容相关检查全部通过，唯一失败项是 `session_completed`。

确定性根因是 TaskRouter 把跨分句的“文件……不得读取或输出凭据”误判为 artifact 请求，同时漏判较长的 conditional-mutation 表述。模型因此不能合法跳过无证据的 implement，最终完整报告又被虚假的 managed-write 门拒绝。

当前代码已改为同一分句内识别 artifact，并逐候选过滤否定动作；conditional mutation 组合“无充分证据”和“skip/no-change”两个有界信号。修复后的原句已通过上述确定性端到端回归。为避免再次消耗大量 DeepSeek API 余额，没有继续在线重跑，因此不得宣称“大型案例修复后在线通过”。

## 发布规则

源码仓库只同步本摘要，不同步 `runs/`、`metrics.json`、PTY、原始最终回答、Session ID、Workspace 路径或二进制验收产物。后续只有在用户明确同意 API 成本后，才可使用全新 Workspace 和全新 Session 做一次修复后大型验收；不得 Resume 旧失败 Session 冒充通过。
