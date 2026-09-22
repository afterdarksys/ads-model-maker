"""Per-class metrics, a threshold sweep, and a calibration table.

Security labels are uneven. Accuracy hides a model that never predicts
the rare class. These numbers are what a model maker checks before serving.
"""

from __future__ import annotations


def classification_report(
    labels_true: list[str],
    labels_pred: list[str],
    label_names: list[str],
) -> dict:
    if len(labels_true) != len(labels_pred):
        raise ValueError("labels_true and labels_pred must be the same length")

    per_class = []
    f1_scores = []
    correct = 0
    for name in label_names:
        tp = sum(1 for truth, pred in zip(labels_true, labels_pred) if truth == name and pred == name)
        fp = sum(1 for truth, pred in zip(labels_true, labels_pred) if truth != name and pred == name)
        fn = sum(1 for truth, pred in zip(labels_true, labels_pred) if truth == name and pred != name)
        support = sum(1 for truth in labels_true if truth == name)
        correct += tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        f1_scores.append(f1)
        per_class.append({
            "label": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        })

    total = len(labels_true)
    return {
        "accuracy": (correct / total) if total else 0.0,
        "macro_f1": (sum(f1_scores) / len(f1_scores)) if f1_scores else 0.0,
        "per_class": per_class,
        "examples": total,
    }


def threshold_sweep(
    positive_flags: list[bool],
    positive_scores: list[float],
    thresholds: list[float] | None = None,
) -> list[dict]:
    """Precision and recall for one positive class as the cutoff moves.

    ``positive_flags`` is true when the example really is the positive class.
    ``positive_scores`` is the model's probability for that class.
    """
    if len(positive_flags) != len(positive_scores):
        raise ValueError("flags and scores must be the same length")
    if thresholds is None:
        thresholds = [index / 10 for index in range(1, 10)]

    rows = []
    for cutoff in thresholds:
        tp = fp = fn = 0
        for flag, score in zip(positive_flags, positive_scores):
            predicted = score >= cutoff
            if predicted and flag:
                tp += 1
            elif predicted and not flag:
                fp += 1
            elif flag and not predicted:
                fn += 1
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        rows.append({
            "threshold": cutoff,
            "precision": precision,
            "recall": recall,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
        })
    return rows


def calibration_buckets(
    confidences: list[float],
    correct: list[bool],
    bins: int = 5,
) -> list[dict]:
    """Group predictions by confidence and report how often they were right."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct must be the same length")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    buckets = [
        {"low": index / bins, "high": (index + 1) / bins, "count": 0, "correct": 0}
        for index in range(bins)
    ]
    for confidence, was_correct in zip(confidences, correct):
        if confidence < 0 or confidence > 1:
            raise ValueError("confidence must be between 0 and 1")
        index = min(bins - 1, int(confidence * bins))
        buckets[index]["count"] += 1
        if was_correct:
            buckets[index]["correct"] += 1

    for bucket in buckets:
        count = bucket["count"]
        bucket["accuracy"] = (bucket["correct"] / count) if count else 0.0
    return buckets
