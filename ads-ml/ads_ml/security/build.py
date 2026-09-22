"""Train a portable security classifier and write the review packet.

The packet is the model file the agent loads, a model card, per-class
metrics, a threshold sweep, a calibration table, an active-learning queue,
and a regression gate against an optional baseline.
"""

from __future__ import annotations

import json
from pathlib import Path

from ads_ml.report import model_footprint
from ads_ml.security.active import review_queue
from ads_ml.security.card import write_model_card
from ads_ml.security.gate import regression_gate
from ads_ml.security.linear import LinearTextClassifier
from ads_ml.security.metrics import calibration_buckets, classification_report, threshold_sweep
from ads_ml.security.prepare import INTENDED_USE, OUT_OF_SCOPE, PreparedCorpus, prepare_security_jsonl


def _split(records: list[dict], labels: list[str]) -> tuple[list[dict], list[dict]]:
    """Keep at least one example of every present class in the training side."""
    grouped: dict[str, list[dict]] = {label: [] for label in labels}
    for record in records:
        grouped[record["label"]].append(record)

    train: list[dict] = []
    evaluation: list[dict] = []
    for label in labels:
        group = grouped[label]
        if not group:
            continue
        if len(group) == 1:
            train.extend(group)
            continue
        cut = max(1, int(len(group) * 0.8))
        if cut >= len(group):
            cut = len(group) - 1
        train.extend(group[:cut])
        evaluation.extend(group[cut:])
    if not evaluation:
        evaluation = list(train)
    return train, evaluation


def build_security_model(
    jsonl_path: Path | str,
    task: str,
    output_dir: Path | str,
    *,
    baseline_metrics: dict | None = None,
    max_f1_drop: float = 0.02,
    dim: int = 1024,
    epochs: int = 8,
) -> dict:
    prepared = prepare_security_jsonl(jsonl_path, task)
    if len(prepared.records) < 2:
        raise ValueError("need at least two usable examples after preparation")

    present = {record["label"] for record in prepared.records}
    if len(present) < 2:
        raise ValueError("need at least two classes after preparation")

    train_rows, eval_rows = _split(prepared.records, prepared.labels)
    classifier = LinearTextClassifier(
        prepared.labels,
        dim=dim,
        task=task,
        epochs=epochs,
    )
    classifier.fit(
        [row["text"] for row in train_rows],
        [row["label"] for row in train_rows],
    )

    predictions = []
    confidences = []
    correct_flags = []
    queue_examples = []
    positive_name = prepared.labels[-1]
    positive_flags = []
    positive_scores = []
    for row in eval_rows:
        predicted = classifier.predict(row["text"])
        predictions.append(predicted["label"])
        confidences.append(predicted["confidence"])
        was_correct = predicted["label"] == row["label"]
        correct_flags.append(was_correct)
        queue_examples.append({
            "text": row["text"],
            "probabilities": predicted["probabilities"],
        })
        positive_flags.append(row["label"] == positive_name)
        positive_scores.append(predicted["probabilities"][positive_name])

    metrics = classification_report(
        [row["label"] for row in eval_rows],
        predictions,
        prepared.labels,
    )
    sweep = threshold_sweep(positive_flags, positive_scores)
    calibration = calibration_buckets(confidences, correct_flags)
    queue = review_queue(queue_examples, limit=min(20, len(queue_examples)))
    gate = (
        regression_gate(metrics, baseline_metrics, max_drop=max_f1_drop)
        if baseline_metrics is not None
        else {"pass": True, "reason": "no baseline supplied"}
    )

    parameters = dim * len(prepared.labels) + len(prepared.labels)
    footprint = model_footprint(
        num_params=parameters,
        data_chars=sum(len(row["text"]) for row in prepared.records),
        data_docs=len(prepared.records),
        hidden_dim=dim,
        depth=1,
        seq_len=64,
        batch_size=1,
    )

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.ads.json"
    classifier.save(model_path)
    card_text = write_model_card(
        destination / "MODEL_CARD.md",
        task=task,
        labels=prepared.labels,
        metrics=metrics,
        data_summary=prepared.summary(),
        redaction_counts=prepared.redaction_counts,
        footprint=footprint,
        intended_use=INTENDED_USE[task],
        out_of_scope=OUT_OF_SCOPE,
    )

    report = {
        "task": task,
        "model_path": str(model_path),
        "metrics": metrics,
        "threshold_sweep": sweep,
        "calibration": calibration,
        "review_queue": queue,
        "gate": gate,
        "data": prepared.summary(),
        "footprint": footprint,
        "eval_examples": len(eval_rows),
        "train_examples": len(train_rows),
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["model_card"] = card_text
    return report


def prepared_summary(prepared: PreparedCorpus) -> dict:
    return prepared.summary()
