from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from order_summary import summarize_orders


class OrderSummaryTests(unittest.TestCase):
    def test_paid_orders_are_summarized_with_decimal_precision(self) -> None:
        result = summarize_orders(Path(__file__).parents[1] / "data" / "orders.csv")

        self.assertEqual(
            result,
            {
                "keyboard": {"quantity": 3, "amount": "59.97"},
                "mouse": {"quantity": 1, "amount": "15.50"},
            },
        )

    def test_invalid_quantity_reports_source_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "orders.csv"
            path.write_text(
                "order_id,product,quantity,unit_price,status\n"
                "1001,keyboard,not-a-number,19.99,paid\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"line 2.*quantity"):
                summarize_orders(path)


if __name__ == "__main__":
    unittest.main()
