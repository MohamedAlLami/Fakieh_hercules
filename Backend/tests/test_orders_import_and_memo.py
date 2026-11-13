"""Orders from the planning Excel sheet, and the Dispatch and Receive Memo page."""

import unittest

from services import dispatch_memo as memo
from services import weighbridge_orders as wo


def row(**over):
    base = {"row": 2, "farm_code": 471, "farm_name": "Tarba (2)", "quantity_exp": 25,
            "feed_code": 102020006, "production_num": 564025}
    base.update(over)
    return base


class ImportPlanTests(unittest.TestCase):
    def test_a_sheet_row_becomes_an_order(self):
        plan = wo.plan_import([row()])
        self.assertEqual([], plan["errors"])
        self.assertEqual(
            {"row": 2, "order_no": 564025, "product_code": "102020006", "quantity_kg": 25000.0,
             "farm_code": "471", "farm_name": "Tarba (2)", "client_name": "Tarba (2) - 471"},
            plan["items"][0],
        )

    def test_quantity_is_in_tonnes_and_defaults_to_25000_kg(self):
        self.assertEqual(22000.0, wo.import_quantity_kg(22))
        self.assertEqual(5000.0, wo.import_quantity_kg("5"))
        self.assertEqual(25000.0, wo.import_quantity_kg(None))
        self.assertEqual(25000.0, wo.import_quantity_kg(" "))
        self.assertEqual(24500.0, wo.import_quantity_kg(24500))  # already kilograms

    def test_excel_numbers_arrive_as_floats_or_text(self):
        plan = wo.plan_import([row(farm_code=471.0, feed_code="102020006", production_num="564025 ")])
        self.assertEqual(("471", "102020006", 564025),
                         tuple(plan["items"][0][k] for k in ("farm_code", "product_code", "order_no")))

    def test_bad_rows_are_reported_with_their_sheet_row_and_the_rest_still_import(self):
        plan = wo.plan_import([
            row(),
            row(row=3, production_num="abc"),
            row(row=4, production_num=564026, feed_code=None),
            row(row=5, production_num=564027, farm_code=""),
            row(row=6, production_num=564028, quantity_exp=-1),
            row(row=7, production_num=564025),
            row(row=8, production_num=564029),
        ])
        self.assertEqual([564025, 564029], [i["order_no"] for i in plan["items"]])
        self.assertEqual([3, 4, 5, 6, 7], [e["row"] for e in plan["errors"]])
        self.assertIn("already on row 2", plan["errors"][-1]["reason"])

    def test_empty_lines_are_ignored(self):
        blank = {"row": 3, "farm_code": None, "farm_name": "", "quantity_exp": None, "feed_code": None,
                 "production_num": None}
        plan = wo.plan_import([row(), blank])
        self.assertEqual((1, 0), (len(plan["items"]), len(plan["errors"])))

    def test_not_a_sheet_is_refused(self):
        with self.assertRaises(wo.OrderError):
            wo.plan_import(None)


TRIP = {
    "id": 7, "ticket": "TE-BE722FC0", "truck_plate": "DEMO-1002", "truck_company": "Demo Haulage",
    "material_code": "100", "material_name": "Maize", "order_client_name": "Tarba (2) - 471",
    "first_weight_kg": 15240.0, "first_ts": "2026-11-13T17:10:00+00:00",
    "second_weight_kg": 41760.0, "second_ts": "2026-11-13T17:11:00+00:00", "net_kg": 26520.0,
    "driver_name": "Demo Driver", "bags": 520, "exit_operator": "Ahmed", "memo_notes": "Seal <4471> & co",
}


class MemoTests(unittest.TestCase):
    def test_values_use_plant_time_and_plain_formats(self):
        v = memo.memo_values(TRIP)
        self.assertEqual("13/11/2026", v["date"])
        self.assertEqual(("08:10 PM", "08:11 PM"), (v["arrivalTime"], v["departureTime"]))  # Riyadh = UTC+3
        self.assertEqual(("15,240 kg", "41,760 kg", "26,520 kg"), (v["entryWeight"], v["exitWeight"], v["netWeight"]))
        self.assertEqual(("520", "Maize - 100", "Demo Haulage"), (v["bags"], v["product"], v["carrier"]))

    def test_missing_details_stay_empty_for_the_pen(self):
        v = memo.memo_values({"ticket": "TE-1"})
        self.assertEqual({"TE-1", ""}, set(v.values()))

    def test_page_carries_the_values_and_escapes_typed_text(self):
        page = memo.build_memo_html(TRIP)
        self.assertIn('data-memo="operationNo">TE-BE722FC0<', page)
        self.assertIn('data-memo="clientName">Tarba (2) - 471<', page)
        self.assertIn("Seal &lt;4471&gt; &amp; co", page)
        self.assertNotIn("<4471>", page)
        self.assertIn("QMS : F/TRN/04", page)

    def test_file_name_is_safe(self):
        self.assertEqual("Dispatch-Memo-TE-1.pdf", memo.memo_file_name({"ticket": 'TE-1"/..\\'}))

    def test_no_browser_gives_a_clear_error_not_a_crash(self):
        original = memo.browser_candidates
        memo.browser_candidates = lambda: []
        try:
            with self.assertRaises(memo.MemoPdfError):
                memo.html_to_pdf("<html></html>")
        finally:
            memo.browser_candidates = original


if __name__ == "__main__":
    unittest.main()
