"""A pallet run that crosses the start of the report period (midnight) only counts what it made
inside the period: Product KG = last reading - reading at the period start."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from services import pallet_report_service as prs

TZ = timezone(timedelta(hours=3))
MIDNIGHT = datetime(2026, 11, 18, 0, 0, tzinfo=TZ)
LATER = MIDNIGHT + timedelta(days=2)


def minute(i, minute_offset, qty, line="P4"):
    at = MIDNIGHT + timedelta(minutes=minute_offset)
    return SimpleNamespace(id=i, line=line, source1=605, source2=None, destination1=841, destination2=0,
                           quantity=qty, running=True, full_run=None, selection=4,
                           material_code=None, material_name=None, recorded_at=at, created_at=at)


def order_10pm_to_1230am():
    """Order from 22:00 to 00:30, counting 100 kg a minute from 0. Counter is 12,000 at 23:59."""
    rows = [minute(i, -120 + i, 100.0 * i) for i in range(151)]  # 22:00 .. 00:30
    day1 = [r for r in rows if r.recorded_at < MIDNIGHT]
    day2 = [r for r in rows if r.recorded_at >= MIDNIGHT]
    return rows, day1, day2


class PeriodSplitTests(unittest.TestCase):
    def test_part_after_midnight_counts_only_what_was_made_after_midnight(self):
        rows, day1, day2 = order_10pm_to_1230am()
        run = prs.build_history_sessions(day2, now=LATER, before_period={"P4": day1[-1]})[0]
        self.assertEqual(11900.0, run["qty_at_period_start"])
        self.assertTrue(run["continued_from_before_period"])
        self.assertEqual(15000.0 - 11900.0, run["product_kg"])
        self.assertEqual(11900.0, run["start_qty"])

    def test_part_before_midnight_starts_from_zero_and_is_unchanged(self):
        rows, day1, day2 = order_10pm_to_1230am()
        run = prs.build_history_sessions(day1, now=LATER, before_period={})[0]
        self.assertEqual(11900.0, run["product_kg"])
        self.assertFalse(run["continued_from_before_period"])

    def test_the_two_days_add_up_to_the_whole_order(self):
        rows, day1, day2 = order_10pm_to_1230am()
        whole = prs.build_history_sessions(rows, now=LATER)[0]["product_kg"]
        part1 = prs.build_history_sessions(day1, now=LATER)[0]["product_kg"]
        part2 = prs.build_history_sessions(day2, now=LATER, before_period={"P4": day1[-1]})[0]["product_kg"]
        self.assertEqual(whole, part1 + part2)

    def test_a_run_that_started_after_a_stop_is_not_split(self):
        before = minute(1, -30, 9000.0)           # an earlier run, ended 30 minutes before
        new_run = [minute(2 + i, i, 50.0 * i) for i in range(20)]
        run = prs.build_history_sessions(new_run, now=LATER, before_period={"P4": before})[0]
        self.assertFalse(run["continued_from_before_period"])
        self.assertEqual(950.0, run["product_kg"])

    def test_only_the_first_run_of_the_line_in_the_period_is_split(self):
        rows, day1, day2 = order_10pm_to_1230am()
        second = [minute(500 + i, 200 + i, 10.0 * i) for i in range(10)]  # a new run later that day
        runs = prs.build_history_sessions(day2 + second, now=LATER, before_period={"P4": day1[-1]})
        by_start = {r["actual_start_time"]: r for r in runs}
        late = by_start[second[0].recorded_at.isoformat()]
        self.assertEqual(90.0, late["product_kg"])
        self.assertFalse(late["continued_from_before_period"])

    def test_counter_reset_inside_the_period_falls_back_to_counting_from_zero(self):
        before = minute(1, -1, 30000.0)
        reset = [minute(2 + i, i, 100.0 * i) for i in range(10)]  # counter restarted at 0 just after midnight
        run = prs.build_history_sessions(reset, now=LATER, before_period={"P4": before})[0]
        self.assertFalse(run["continued_from_before_period"])
        self.assertEqual(900.0, run["product_kg"])

    def test_other_lines_are_not_affected(self):
        rows, day1, day2 = order_10pm_to_1230am()
        p1 = [minute(900 + i, i, 70.0 * i, line="P1") for i in range(5)]
        runs = prs.build_history_sessions(day2 + p1, now=LATER, before_period={"P4": day1[-1]})
        p1_run = next(r for r in runs if r["line"] == "P1")
        self.assertEqual(280.0, p1_run["product_kg"])
        self.assertFalse(p1_run["continued_from_before_period"])

    def test_without_a_period_start_nothing_changes(self):
        rows, day1, day2 = order_10pm_to_1230am()
        run = prs.build_history_sessions(day2, now=LATER)[0]
        self.assertEqual(15000.0, run["product_kg"])
        self.assertFalse(run["continued_from_before_period"])


if __name__ == "__main__":
    unittest.main()
