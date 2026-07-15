"""Feature engineering shared by training and prediction.

Deliberately minimal. An empirical bake-off (5-fold cross-validation, see
scratchpad/simplify_*.py history) showed that just TWO inputs — the target
weight (``setpoint``) and which ingredient it is (``material_code``) — match
the previous seven-feature model on all three tasks (risk, severity,
direction), within CV noise:

    task        7 features            2 features (this)
    risk        acc .815 / auc .885   acc .815 / auc .887
    severity    acc .730 / f1  .642   acc .724 / f1  .635
    direction   acc .705 / auc .796   acc .714 / auc .800

The dropped features were either mathematically redundant for a tree model
(``log_setpoint`` and ``is_micro`` are monotonic transforms of ``setpoint``,
and decision trees are invariant to monotonic transforms of a single input)
or carried no additional signal on top of setpoint + material (``quantity``,
``product_name``, ``category``). Fewer inputs → a simpler, faster,
easier-to-explain model with the same accuracy.

Keeping this in one place guarantees the model is trained and queried with the
exact same feature layout.
"""

from __future__ import annotations

NUMERIC_FEATURES = ["setpoint"]
CATEGORICAL_FEATURES = ["material_code"]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def row_to_features(row: dict) -> dict:
    """Turn a normalized data row (or a prediction request) into model features.

    Extra keys on the input (product_name, quantity, category, …) are simply
    ignored, so existing callers that still pass them keep working unchanged.
    """
    return {
        "setpoint": float(row.get("setpoint") or 0.0),
        "material_code": str(row.get("material_code") or ""),
    }
