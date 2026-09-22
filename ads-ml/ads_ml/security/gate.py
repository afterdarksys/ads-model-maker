"""Fail a new security model when macro F1 falls too far from a baseline."""

from __future__ import annotations


def regression_gate(
    current: dict,
    baseline: dict,
    max_drop: float = 0.02,
) -> dict:
    """Compare macro F1. Missing numbers fail closed."""
    if max_drop < 0:
        raise ValueError("max_drop cannot be negative")
    if "macro_f1" not in current or "macro_f1" not in baseline:
        return {
            "pass": False,
            "reason": "macro_f1 is required on the current report and the baseline",
        }

    current_score = float(current["macro_f1"])
    baseline_score = float(baseline["macro_f1"])
    drop = baseline_score - current_score
    passed = drop <= max_drop
    return {
        "pass": passed,
        "macro_f1": current_score,
        "baseline_macro_f1": baseline_score,
        "drop": drop,
        "max_drop": max_drop,
        "reason": "ok" if passed else "macro_f1 dropped further than the allowed tolerance",
    }
