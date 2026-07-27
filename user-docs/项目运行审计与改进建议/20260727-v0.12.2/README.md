# Deep Agent v0.12.2 GitHub 文档发布核验

日期：2026-07-27

## 核验结论

v0.12.1 的源码、最新 Markdown、Word、审计材料、tag 和 Actions 确已推送。用户看到 0.10.0 不是 GitHub 丢失了 0.12.1，而是仓库根目录遗留两份 0.10.0 Word；最新文件位于 `user-docs/`，但 GitHub 首页更显眼地展示了根目录旧文件。

## 修复

- 确认根目录 0.10.0 Word 与老版归档中的文件 SHA-256 相同，删除根目录重复副本不会丢失历史。
- GitHub 根目录仅保留当前版本的使用说明和工作日志 Word。
- `user-docs/` 保留当前 Markdown/Word，老版 Word 继续放在老版目录。
- 源码 README 提供最新中文文档直接链接。
- 自动测试检查根目录只有当前版本两份 Word，并且 `user-docs/` 存在同版文件。

## 边界

本补丁不改 Runtime、AgentState schema 7、核心契约 3、配置或用户数据。用户 API 余额尚未恢复，不读取 Key，不执行在线 DeepSeek 认证。

## 后续建议

1. 版本发布测试继续检查根目录和 `user-docs/` 一致性。
2. 老版文档只能进入归档目录，不再在 GitHub 根目录保留重复副本。
3. 在线认证仍等余额恢复后，按请求/Token/时间上限只补一次。
