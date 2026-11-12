"""Recipe and product name of a pallet run come from the silos it runs between."""

import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from services import pallet_material as pm
from services import pallet_report_service as prs

BASE = datetime(2026, 11, 12, 8, 0, tzinfo=timezone.utc)
SILOS = {
    601: ("", ""), 602: ("12355", "BR-GR"), 603: ("13335", "BR-ST"), 604: ("12355", "BR-GR"),
    806: ("12355", "BR-GR"), 843: ("13335", "BR-ST"), 801: ("13335", "BR-ST"), 901: ("999", "Out Of Service"),
}


class PickTests(unittest.TestCase):
    def test_source_silo_wins(self):
        self.assertEqual(("12355", "BR-GR"), pm.pick_material([602], [843], SILOS))

    def test_empty_source_falls_back_to_the_destination_bin(self):
        self.assertEqual(("13335", "BR-ST"), pm.pick_material([601, None], [843, 801], SILOS))

    def test_two_sources_with_different_materials_show_both(self):
        self.assertEqual(("13335 + 12355", "BR-ST + BR-GR"), pm.pick_material([603, 604], [806], SILOS))

    def test_two_sources_with_the_same_material_show_it_once(self):
        self.assertEqual(("12355", "BR-GR"), pm.pick_material([602, 604], [], SILOS))

    def test_placeholders_and_unknown_silos_are_not_materials(self):
        self.assertEqual((None, None), pm.pick_material([901, 777, 0], [601], SILOS))

    def test_a_code_without_a_name_still_shows_the_code(self):
        code, _name = pm.pick_material([5], [], {5: ("987654", "")})
        self.assertEqual("987654", code)


def snap(i, minute, code=None, name=None, line="P1"):
    at = BASE + timedelta(minutes=minute)
    return SimpleNamespace(id=i, line=line, source1=602, source2=None, destination1=806, destination2=0,
                           quantity=100.0 * i, running=True, full_run=None, selection=None,
                           material_code=code, material_name=name, recorded_at=at, created_at=at)


class SessionTests(unittest.TestCase):
    def test_run_shows_the_latest_material_its_minutes_recorded(self):
        rows = [snap(1, 0, "12355", "BR-GR"), snap(2, 1, "12355", "BR-GR"), snap(3, 2, None, None)]
        run = prs.build_history_sessions(rows, now=BASE + timedelta(hours=1))[0]
        self.assertEqual(("12355", "BR-GR"), (run["material"], run["production_name"]))

    def test_run_stored_before_this_existed_has_no_material(self):
        old = [SimpleNamespace(**{k: v for k, v in vars(snap(1, 0)).items() if not k.startswith("material")})]
        run = prs.build_history_sessions(old, now=BASE + timedelta(hours=1))[0]
        self.assertIsNone(run["material"])
        self.assertIsNone(run["production_name"])

    def test_rows_get_the_material_of_their_silos(self):
        rows = [{"source1": 601, "source2": None, "destination1": 843, "destination2": 801}]
        original = pm.current_silo_materials
        pm.current_silo_materials = lambda numbers, session=None: SILOS
        try:
            pm.annotate_snapshot_rows(rows)
        finally:
            pm.current_silo_materials = original
        self.assertEqual(("13335", "BR-ST"), (rows[0]["material_code"], rows[0]["material_name"]))


if __name__ == "__main__":
    unittest.main()
