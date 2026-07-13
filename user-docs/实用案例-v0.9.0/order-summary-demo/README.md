# Order Summary Demo

读取 CSV 订单，按商品输出有效订单的数量和金额汇总。

```bash
python3 order_summary.py data/orders.csv
python3 -m unittest discover -s tests -v
```

CSV 字段：`order_id,product,quantity,unit_price,status`。

业务规则：

- 只统计 `paid` 订单。
- 数量必须是正整数。
- 单价必须是非负十进制金额。
- 金额以两位小数输出。

当前版本故意含有测试可复现的缺陷，供 Deep Agent v0.9.0 演示定位、修复和验证。
