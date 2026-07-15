"""Serve the single Dose-Quality model.

One model, one prediction, three answers derived from its 5-class
probabilities (risk / severity / direction) — guaranteed mutually consistent.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import joblib
import pandas as pd

from . import dose_quality, features
from .train_dose import META_PATH, MODEL_PATH


def is_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)


@lru_cache(maxsize=1)
def _model():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _meta() -> dict:
    with open(META_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_info() -> dict:
    meta = dict(_meta())
    meta["ready"] = True
    return meta


def _band(risk: float) -> str:
    if risk >= 0.66:
        return "High"
    if risk >= 0.33:
        return "Medium"
    return "Low"


def _material_name(code: str) -> str:
    return next(
        (m["material_name"] for m in _meta()["options"]["materials"] if str(m["material_code"]) == code),
        code,
    )


def _derive(probs: dict) -> dict:
    """Collapse the 5-class probability dict into risk / severity / direction.

    Severity is conditional on the risk decision so the two never contradict:
    if the dose is flagged (Medium/High risk), severity is watch or severe —
    never 'on target'. (Taking a naive argmax over all 5 buckets could return
    'on_target' as the single most likely bucket even when the aggregate
    flagged probability is higher, which reads as a contradiction.)
    """
    p_on = probs.get("on_target", 0.0)
    risk = 1.0 - p_on
    band = _band(risk)
    flagged = band in ("High", "Medium")

    p_watch = probs.get("over_watch", 0.0) + probs.get("under_watch", 0.0)
    p_severe = probs.get("over_severe", 0.0) + probs.get("under_severe", 0.0)
    if flagged:
        severity = "severe" if p_severe >= p_watch else "watch"
        sev_conf = (max(p_watch, p_severe) / max(p_watch + p_severe, 1e-9))
    else:
        severity = "on_target"
        sev_conf = p_on

    p_over = probs.get("over_watch", 0.0) + probs.get("over_severe", 0.0)
    p_under = probs.get("under_watch", 0.0) + probs.get("under_severe", 0.0)
    denom = p_over + p_under
    direction = "over" if p_over >= p_under else "under"
    dir_conf = (max(p_over, p_under) / denom) if denom > 1e-9 else 0.0

    top_class = max(probs, key=probs.get)
    return {
        "risk": round(risk, 3),
        "risk_pct": round(risk * 100, 1),
        "band": band,
        "flagged": flagged,
        "prediction": "Likely out-of-tolerance" if flagged else "Likely on-target",
        "severity": severity,
        "severity_label": {"on_target": "On target", "watch": "Watch — moderate deviation",
                           "severe": "Severe deviation"}[severity],
        "severity_confidence_pct": round(sev_conf * 100, 1),
        "direction": direction,
        "direction_label": "Likely to run high" if direction == "over" else "Likely to run low",
        "direction_confidence_pct": round(dir_conf * 100, 1),
        "top_class": top_class,
        "top_class_label": dose_quality.DOSE_LABELS.get(top_class, top_class),
        "class_probabilities": {c: round(probs.get(c, 0.0) * 100, 1) for c in dose_quality.DOSE_CLASSES},
    }


def _probs_for(frame_row: dict) -> dict:
    model = _model()
    frame = pd.DataFrame([frame_row])[features.FEATURE_ORDER]
    proba = model.predict_proba(frame)[0]
    classes = list(model.classes_)
    return {c: float(p) for c, p in zip(classes, proba)}


def predict(material_code: str, product_name: str | None, setpoint: float,
            quantity=None, category=None) -> dict:
    material_code = str(material_code)
    feat = features.row_to_features({"setpoint": setpoint, "material_code": material_code})
    probs = _probs_for(feat)
    out = _derive(probs)
    out["inputs"] = {
        "material_code": material_code,
        "material_name": _material_name(material_code),
        "product_name": product_name,
        "setpoint": setpoint,
    }
    return out


def predict_batch(rows: list[dict]) -> list[dict | None]:
    """Vectorized: one predict_proba call for many rows (used by the live feed)."""
    if not rows:
        return []
    valid_idx, feats = [], []
    for i, r in enumerate(rows):
        material_code = str(r.get("material_code") or "")
        setpoint = r.get("setpoint")
        if not material_code or not setpoint:
            continue
        feats.append(features.row_to_features({"setpoint": setpoint, "material_code": material_code}))
        valid_idx.append(i)

    results: list[dict | None] = [None] * len(rows)
    if not feats:
        return results
    model = _model()
    frame = pd.DataFrame(feats)[features.FEATURE_ORDER]
    proba_matrix = model.predict_proba(frame)
    classes = list(model.classes_)
    for idx, proba in zip(valid_idx, proba_matrix):
        results[idx] = _derive({c: float(p) for c, p in zip(classes, proba)})
    return results
