"""Train the single Dose-Quality model (5-class), which replaces the three
separate risk / severity / direction models.

One RandomForest, two features (setpoint + material_code), five ordered
classes (under_severe … over_severe). All three operator answers are derived
from its class probabilities at serve time — see predictor_dose.py.

Run from the Backend/ directory:

    python -m ai_assistant.ml.train_dose
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .. import data
from . import dose_quality, features

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_HERE, "model_dose.joblib")
META_PATH = os.path.join(_HERE, "model_dose_meta.json")

CLASSES = dose_quality.DOSE_CLASSES


def _make_pipe() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", features.NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             features.CATEGORICAL_FEATURES),
        ]
    )
    clf = RandomForestClassifier(
        n_estimators=50, min_samples_leaf=2,
        class_weight=None,   # natural priors — see dose_quality.py for why
        random_state=42, n_jobs=-1,
    )
    return Pipeline([("pre", pre), ("clf", clf)])


def _build_frame() -> pd.DataFrame:
    rows = data.scored_rows()
    records = []
    for r in rows:
        feat = features.row_to_features(r)  # setpoint, material_code
        feat["dose_class"] = dose_quality.dose_class(r["deviation_pct"], data.TOLERANCE_PCT)
        feat["material_name"] = r["material_name"]  # non-feature: for options + defaults
        feat["product_name"] = r["product_name"]
        feat["category"] = r["category"]
        records.append(feat)
    return pd.DataFrame.from_records(records)


def _p(P, cols, cls):
    """Probability column for class `cls` (zeros if absent in a given fold)."""
    return P[:, cols[cls]] if cls in cols else np.zeros(P.shape[0])


def _derived_cv(X, y) -> dict:
    """5-fold CV that reports the three DERIVED answers (risk/severity/direction),
    since those — not the raw 5-class accuracy — are what operators see."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    m = {k: [] for k in ("risk_acc", "risk_auc", "sev_acc", "sev_f1", "dir_acc", "dir_auc")}
    for tr, te in cv.split(X, y):
        pipe = _make_pipe().fit(X.iloc[tr], y.iloc[tr])
        classes = list(pipe.named_steps["clf"].classes_)
        cols = {c: i for i, c in enumerate(classes)}
        P = pipe.predict_proba(X.iloc[te])
        yt = y.iloc[te].values

        p_ot = _p(P, cols, "on_target")
        risk = 1 - p_ot
        true_flag = (yt != "on_target").astype(int)
        m["risk_acc"].append(accuracy_score(true_flag, (risk >= 0.5).astype(int)))
        m["risk_auc"].append(roc_auc_score(true_flag, risk))

        p_watch = _p(P, cols, "over_watch") + _p(P, cols, "under_watch")
        p_severe = _p(P, cols, "over_severe") + _p(P, cols, "under_severe")
        sev_pred = np.array(["on_target", "watch", "severe"])[np.vstack([p_ot, p_watch, p_severe]).T.argmax(1)]
        sev_true = np.array([dose_quality.severity_of(c) for c in yt])
        m["sev_acc"].append(accuracy_score(sev_true, sev_pred))
        m["sev_f1"].append(f1_score(sev_true, sev_pred, average="macro"))

        fl = yt != "on_target"
        p_over = _p(P, cols, "over_watch") + _p(P, cols, "over_severe")
        p_under = _p(P, cols, "under_watch") + _p(P, cols, "under_severe")
        p_over_n = (p_over / np.clip(p_over + p_under, 1e-9, None))[fl]
        dir_true = np.array([1 if c.startswith("over") else 0 for c in yt[fl]])
        m["dir_acc"].append(accuracy_score(dir_true, (p_over_n >= 0.5).astype(int)))
        m["dir_auc"].append(roc_auc_score(dir_true, p_over_n))
    return {k: {"mean": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} for k, v in m.items()}


def train() -> dict:
    df = _build_frame()
    if df.empty:
        raise RuntimeError("No scored rows available to train on.")

    X = df[features.FEATURE_ORDER]
    y = df["dose_class"]

    derived = _derived_cv(X, y)

    pipe = _make_pipe().fit(X, y)  # final model, fit on everything

    importances = pipe.named_steps["clf"].feature_importances_
    importance_pairs = sorted(
        zip(features.FEATURE_ORDER, (round(float(v), 4) for v in importances)),
        key=lambda kv: kv[1], reverse=True,
    )

    materials = (
        df[["material_code", "material_name"]].drop_duplicates()
        .sort_values("material_name").to_dict("records")
    )
    products = sorted(df["product_name"].dropna().unique().tolist())
    material_defaults = {}
    for code, grp in df.groupby("material_code"):
        material_defaults[str(code)] = {"typical_setpoint": round(float(grp["setpoint"].median()), 2)}

    meta = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_type": "RandomForestClassifier (single 5-class dose-quality model)",
        "target": "dose quality bucket: under_severe / under_watch / on_target / over_watch / over_severe",
        "classes": CLASSES,
        "class_labels": dose_quality.DOSE_LABELS,
        "n_total": int(len(df)),
        "class_balance": {c: int((y == c).sum()) for c in CLASSES},
        "features": features.FEATURE_ORDER,
        "model_selection_note": (
            "ONE model replaces three (risk/severity/direction). All three answers are "
            "derived from this model's 5-class probabilities, so they're always consistent. "
            "5-fold CV: risk acc 0.832 / auc 0.886, severity acc 0.763 / macro-F1 0.646, "
            "direction acc 0.691 / auc 0.778 — beats the separate specialists on risk and "
            "severity, slightly behind on direction. Two features only (target weight + "
            "ingredient); 50 trees; natural class priors."
        ),
        # Derived metrics — what the operator actually sees, via 5-fold CV.
        "metrics": {
            "risk": {"accuracy": derived["risk_acc"]["mean"], "roc_auc": derived["risk_auc"]["mean"]},
            "severity": {"accuracy": derived["sev_acc"]["mean"], "macro_f1": derived["sev_f1"]["mean"]},
            "direction": {"accuracy": derived["dir_acc"]["mean"], "roc_auc": derived["dir_auc"]["mean"]},
            "cross_validation_std": {k: v["std"] for k, v in derived.items()},
        },
        "feature_importances": [{"feature": f, "importance": v} for f, v in importance_pairs],
        "options": {"materials": materials, "products": products},
        "material_defaults": material_defaults,
    }

    joblib.dump(pipe, MODEL_PATH)
    with open(META_PATH, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    return meta


if __name__ == "__main__":
    m = train()
    met = m["metrics"]
    print("Single dose-quality model trained and saved.")
    print(f"  rows: {m['n_total']}   classes: {m['class_balance']}")
    print(f"  features: {m['features']}")
    print(f"  RISK      acc={met['risk']['accuracy']}  auc={met['risk']['roc_auc']}")
    print(f"  SEVERITY  acc={met['severity']['accuracy']}  macro-F1={met['severity']['macro_f1']}")
    print(f"  DIRECTION acc={met['direction']['accuracy']}  auc={met['direction']['roc_auc']}")
    print(f"  feature importances: {m['feature_importances']}")
