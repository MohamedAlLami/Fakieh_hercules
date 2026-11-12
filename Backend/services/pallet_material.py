"""Recipe and product name of a pallet run, taken from the silos it runs between.

The pallet block in the PLC (DB7) carries only silo numbers. The Storage page already knows
which material every silo holds (table silo_status, kept in step with the PLC), so a run's
material is the material of its source silo; when the source silo carries no material, the
material of its destination bin.

Nothing here may break a PLC collection cycle: every database read is wrapped and a failure
simply leaves the run without a material.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

# Placeholders the plant uses for "nothing in this silo".
NOT_A_MATERIAL = {"", "0", "00", "000", "1", "999"}

SiloMaterials = Dict[int, Tuple[str, str]]  # silo_no -> (code, name)


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _silos(values: Iterable[Any]) -> List[int]:
    out: List[int] = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in out:
            out.append(number)
    return out


def _name_for(code: str, name: str) -> str:
    if name:
        return name
    try:
        from constants.material_codes import resolve_material_name

        return _clean(resolve_material_name(code))
    except Exception:
        return ""


def pick_material(
    sources: Iterable[Any], destinations: Iterable[Any], silo_materials: Mapping[int, Tuple[str, str]]
) -> Tuple[Optional[str], Optional[str]]:
    """(code, name) for a run: the source silo's material, else the destination bin's.

    Two silos holding different materials are both shown, joined with " + ".
    """
    for group in (sources, destinations):
        found: List[Tuple[str, str]] = []
        for silo in _silos(group):
            code, name = silo_materials.get(silo, ("", ""))
            code, name = _clean(code), _clean(name)
            if code in NOT_A_MATERIAL:
                continue
            if all(code != seen for seen, _ in found):
                found.append((code, _name_for(code, name)))
        if found:
            names = [name for _, name in found if name]
            return " + ".join(code for code, _ in found), (" + ".join(names) or None)
    return None, None


def _rows_to_map(rows: Iterable[Any]) -> SiloMaterials:
    return {int(row[0]): (_clean(row[1]), _clean(row[2])) for row in rows}


def current_silo_materials(silo_numbers: Iterable[Any], session: Any = None) -> SiloMaterials:
    """What Storage shows right now for these silos. Empty on any failure."""
    numbers = _silos(silo_numbers)
    if not numbers:
        return {}
    try:
        from sqlalchemy import text

        if session is None:
            from models import db

            session = db.session
        with session.begin_nested():
            rows = session.execute(
                text(
                    "SELECT silo_no, material_code, material_name FROM public.silo_status "
                    "WHERE silo_no = ANY(:nos)"
                ),
                {"nos": numbers},
            ).fetchall()
        return _rows_to_map(rows)
    except Exception as exc:
        logger.warning("Could not read silo materials: %s", exc)
        return {}


def silo_materials_at(silo_numbers: Iterable[Any], at: datetime, session: Any = None) -> SiloMaterials:
    """What these silos held at a past moment, from the silo change history. Empty when unknown."""
    numbers = _silos(silo_numbers)
    if not numbers or at is None:
        return {}
    try:
        from sqlalchemy import text

        if session is None:
            from models import db

            session = db.session
        with session.begin_nested():
            rows = session.execute(
                text(
                    "SELECT DISTINCT ON (silo_no) silo_no, material_code, material_name "
                    "FROM public.silo_status_history "
                    "WHERE silo_no = ANY(:nos) AND changed_at <= :at "
                    "ORDER BY silo_no, changed_at DESC"
                ),
                {"nos": numbers, "at": at},
            ).fetchall()
        return _rows_to_map(rows)
    except Exception as exc:
        logger.warning("Could not read silo material history: %s", exc)
        return {}


def route_silos(route: Mapping[str, Any]) -> Tuple[List[int], List[int]]:
    return (
        _silos([route.get("source1"), route.get("source2")]),
        _silos([route.get("destination1"), route.get("destination2")]),
    )


def material_for_route(route: Mapping[str, Any], session: Any = None) -> Tuple[Optional[str], Optional[str]]:
    """Material of a route as Storage shows it now."""
    sources, destinations = route_silos(route)
    return pick_material(sources, destinations, current_silo_materials(sources + destinations, session))


def annotate_snapshot_rows(rows: List[Dict[str, Any]], session: Any = None) -> List[Dict[str, Any]]:
    """Stamp each minute snapshot with the material its silos hold at that minute."""
    wanted: List[int] = []
    for row in rows:
        sources, destinations = route_silos(row)
        wanted += sources + destinations
    materials = current_silo_materials(wanted, session)
    for row in rows:
        sources, destinations = route_silos(row)
        row["material_code"], row["material_name"] = pick_material(sources, destinations, materials)
    return rows


def fill_missing_session_materials(sessions: List[Dict[str, Any]], session: Any = None) -> List[Dict[str, Any]]:
    """Fill runs that were stored without a material.

    A run that is still open takes what Storage shows now. A finished run takes what the silo
    change history says the silos held when the run started; with no history it stays empty
    rather than showing today's material on yesterday's run.
    """
    for item in sessions:
        if item.get("material"):
            continue
        sources, destinations = route_silos(item)
        if item.get("status") == "RUNNING":
            materials = current_silo_materials(sources + destinations, session)
        else:
            try:
                started = datetime.fromisoformat(item["actual_start_time"])
            except Exception:
                continue
            materials = silo_materials_at(sources + destinations, started, session)
        item["material"], item["production_name"] = pick_material(sources, destinations, materials)
    return sessions
