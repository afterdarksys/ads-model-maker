"""Prepare a security JSONL file for training.

Each line is an object with ``text`` and ``label``. The label has to be one
of the task's classes. Preparation always redacts secrets, rewrites defanged
indicators, and drops near-duplicate text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ads_ml.security.dedup import drop_near_duplicates
from ads_ml.security.ioc import normalize_defanged
from ads_ml.security.redact import redact_text

TASKS: dict[str, list[str]] = {
    "phishing_email": ["benign", "phishing"],
    "alert_severity": ["info", "low", "medium", "high", "critical"],
    "log_category": ["auth", "network", "endpoint", "identity", "application"],
    "secret_exposure": ["clean", "secret"],
}

INTENDED_USE = {
    "phishing_email": "Rank inbound email text as phishing or benign for an analyst queue.",
    "alert_severity": "Suggest a severity for a written security alert so a queue can be sorted.",
    "log_category": "Group a short log line into an operational category.",
    "secret_exposure": "Flag text that still looks like it contains a credential after redaction.",
}

OUT_OF_SCOPE = (
    "Do not use the score as an allow or block decision by itself. "
    "The model does not detonate files, scan networks, or prove that a message is safe."
)


@dataclass
class PreparedCorpus:
    task: str
    labels: list[str]
    records: list[dict]
    redaction_counts: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    rejected: int = 0

    def summary(self) -> dict:
        return {
            "task": self.task,
            "labels": list(self.labels),
            "kept": len(self.records),
            "duplicates_removed": self.duplicates_removed,
            "rejected": self.rejected,
            "redaction_counts": dict(self.redaction_counts),
        }


def prepare_security_jsonl(
    path: Path | str,
    task: str,
    *,
    dedup_threshold: int = 3,
) -> PreparedCorpus:
    if task not in TASKS:
        raise ValueError(f"unknown security task {task!r}; choose one of {sorted(TASKS)}")
    labels = TASKS[task]
    allowed = set(labels)

    raw_records = []
    redaction_counts: dict[str, int] = {}
    rejected = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not JSON") from exc
            text = payload.get("text") if isinstance(payload, dict) else None
            label = payload.get("label") if isinstance(payload, dict) else None
            if not isinstance(text, str) or not text.strip() or label not in allowed:
                rejected += 1
                continue
            redacted = redact_text(text)
            for name, count in redacted.counts.items():
                redaction_counts[name] = redaction_counts.get(name, 0) + count
            normalized = normalize_defanged(redacted.text)
            raw_records.append({"text": normalized.strip(), "label": label})

    kept, dropped = drop_near_duplicates(raw_records, threshold=dedup_threshold)
    return PreparedCorpus(
        task=task,
        labels=labels,
        records=kept,
        redaction_counts=redaction_counts,
        duplicates_removed=dropped,
        rejected=rejected,
    )
