"""The single Dose-Quality model's label scheme + derivations.

One model, five ordered buckets along the signed-deviation spectrum:

    under_severe < under_watch < on_target < over_watch < over_severe

From ONE prediction (the model's class probabilities) we read off all three
things an operator cares about, and they can never contradict each other
(unlike three separate models):

  * RISK      = P(not on_target)              — will this dose miss tolerance?
  * SEVERITY  = on_target / watch / severe    — how far off (magnitude)?
  * DIRECTION = over / under                  — which way?

This replaced three separate models (risk / severity / direction). A 5-fold CV
bake-off showed the single model actually beats the specialists on risk
(0.832 vs 0.812 acc) and severity (0.763 vs 0.720), giving up only a little on
direction (0.691 vs 0.726) — a clean win for simplicity and explainability.
Uses natural class priors (no class_weight balancing): balancing over-weighted
the rare 'severe under-dose' bucket and dragged the on-target/flagged boundary,
hurting risk accuracy.
"""

from __future__ import annotations

from .severity import WATCH_MAX_PCT  # single source of truth for the watch/severe line

# Ordered low -> high so a UI can render them as a left-to-right spectrum.
DOSE_CLASSES = ["under_severe", "under_watch", "on_target", "over_watch", "over_severe"]

DOSE_LABELS = {
    "under_severe": "Severe under-dose",
    "under_watch": "Slight under-dose",
    "on_target": "On target",
    "over_watch": "Slight over-dose",
    "over_severe": "Severe over-dose",
}


def dose_class(deviation_pct: float | None, tolerance_pct: float) -> str:
    """Bucket a signed deviation_pct into one of the 5 dose-quality classes."""
    if deviation_pct is None:
        return "unknown"
    a = abs(deviation_pct)
    if a <= tolerance_pct:
        return "on_target"
    magnitude = "severe" if a > WATCH_MAX_PCT else "watch"
    direction = "over" if deviation_pct > 0 else "under"
    return f"{direction}_{magnitude}"


# --- Derivations: collapse the 5-class probabilities into each sub-answer -----
def severity_of(cls: str) -> str:
    """on_target / watch / severe (magnitude, direction discarded)."""
    if cls == "on_target":
        return "on_target"
    return "severe" if cls.endswith("severe") else "watch"


def direction_of(cls: str) -> str | None:
    """over / under, or None for on-target (no direction)."""
    if cls == "on_target":
        return None
    return "over" if cls.startswith("over") else "under"
