from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def summarize_orders(path: Path) -> dict[str, dict[str, object]]:
    totals: dict[str, dict[str, object]] = defaultdict(lambda: {"quantity": 0, "amount": 0.0})
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            product = row["product"].strip()
            quantity = int(row["quantity"])
            unit_price = float(row["unit_price"])
            # Intentional demo bug: cancelled orders are not filtered.
            totals[product]["quantity"] += quantity
            totals[product]["amount"] += quantity * unit_price

    return {
        product: {
            "quantity": values["quantity"],
            "amount": f"{values['amount']:.2f}",
        }
        for product, values in sorted(totals.items())
    }


def main(argv: list[str] | None = None) -> int:
    args = list(argv or sys.argv[1:])
    if len(args) != 1:
        print("usage: python3 order_summary.py <orders.csv>", file=sys.stderr)
        return 2
    result = summarize_orders(Path(args[0]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
