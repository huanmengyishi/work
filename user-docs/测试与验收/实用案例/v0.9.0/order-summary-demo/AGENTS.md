# 示例项目规则

- 只修改本目录内文件。
- 金额必须使用 `decimal.Decimal`，不得用二进制浮点数累计。
- 只汇总 `status=paid` 的订单。
- 非法数量、金额、状态或缺失字段必须给出包含行号的明确错误。
- 修改后必须运行 `python3 -m unittest discover -s tests -v`。
- 不要读取或写入任何 API Key、Cookie、Token 或项目外文件。
