# v0.12.0 四份输入材料归档

归档日期：2026-07-26

本目录保存本轮 v0.12.0 开发前由用户提供的四份原始 Word 材料。文件仅从协作根目录移动到历史资料目录，文件名和二进制内容未改动；它们描述的是基于 v0.11.0 的问题判断与建议，不代表当前工作树或最终 v0.12.0 的发布状态。

## 文件与完整性

| 文件 | SHA-256 |
| --- | --- |
| `我来针对这六大发展方向.docx` | `4179a1cefe147e9269a8fa92afee069441a26a1db69b0dd5935014aebfabcc4c` |
| `0.12改进.docx` | `fec1536c4db3bafb68eb1d2b46773f31514aac06705b8e5eece22907a3ebef8c` |
| `DeepSeek Agent V3 后续开发规划总体目标DeepSeek Agent V3 v0.docx` | `a16ba37262bfd11e37f2265ff09d43d233f5b3eeb6e71af87b4669d5b089de73` |
| `基于对代码库的全面审查.docx` | `9f40f9820ffbcb301b75970d0567e962d554aa66366be9dad5cc7fe0286d9b81` |

可用以下命令复核 ZIP 与哈希完整性：

```bash
cd /mnt/d/detail/deepseek/历史资料/改进建议/20260726-v0.12.0
unzip -t '我来针对这六大发展方向.docx'
unzip -t '0.12改进.docx'
unzip -t 'DeepSeek Agent V3 后续开发规划总体目标DeepSeek Agent V3 v0.docx'
unzip -t '基于对代码库的全面审查.docx'
sha256sum *.docx
```

## 使用边界

- 四份材料中的行号来自当时的 v0.11.0 快照；代码变化后应按符号和测试定位，不能机械套用旧行号。
- 材料把 Gitee `free/claude-code` 的 `claude` 分支称为“官方源码”。参考仓库自身说明该内容是 Anthropic 专有源码泄露快照，且没有授予可复制许可证。本项目只对照可观察的行为边界和公开接口思想，不直接复制代码。
- “备用 Provider”与本项目 DeepSeek-only 约束冲突，不实施。
- Runtime 层重复网络重试、未经验证的并行子 Agent、自动 A/B 调参、分块部分写入等建议有副作用或架构风险，不能因材料中给出伪代码就直接照搬。
- 四份材料的逐项处置结论见 [`四份材料逐项处置矩阵.md`](../../../项目运行审计与改进建议/20260726-v0.12.0/四份材料逐项处置矩阵.md)。
