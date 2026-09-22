"""Examples the model is least sure about, for a person to label next."""

from __future__ import annotations


def review_queue(examples: list[dict], limit: int = 20) -> list[dict]:
    """Sort by the gap between the top two class probabilities.

    Each example needs ``text`` and ``probabilities`` mapping label to
    probability. The returned rows do not copy anything except text, the
    suggested label, and the margin.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")

    ranked = []
    for example in examples:
        probabilities = example.get("probabilities") or {}
        if len(probabilities) < 2:
            raise ValueError("each example needs probabilities for at least two labels")
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]
        ranked.append({
            "text": example.get("text", ""),
            "suggested_label": top_label,
            "margin": float(top_score) - float(second_score),
            "confidence": float(top_score),
        })
    ranked.sort(key=lambda row: row["margin"])
    return ranked[:limit]
