"""JSON description of a training job.

The file is the contract the UI and the CLI both write. Unknown keys are
rejected so a typo does not silently drop a setting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

MODEL_TYPES = ("classifier", "embedder", "generator")
MODEL_SIZES = ("micro", "tiny", "small", "base", "medium", "large")

_FIELDS = {
    "model_type",
    "model_size",
    "epochs",
    "batch_size",
    "learning_rate",
    "labels",
    "data_path",
    "output_dir",
    "max_seq_len",
    "text_field",
    "label_field",
}


@dataclass
class JobConfig:
    model_type: str
    model_size: str = "small"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    labels: list[str] = field(default_factory=list)
    data_path: str = ""
    output_dir: str = "./output"
    max_seq_len: int = 512
    text_field: str = "text"
    label_field: str = "label"

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "model_size": self.model_size,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "labels": list(self.labels),
            "data_path": self.data_path,
            "output_dir": self.output_dir,
            "max_seq_len": self.max_seq_len,
            "text_field": self.text_field,
            "label_field": self.label_field,
        }


def load_job_config(path: Path | str) -> JobConfig:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("job config must be a JSON object")

    unknown = sorted(set(payload) - _FIELDS)
    if unknown:
        raise ValueError(f"unknown job config keys: {unknown}")

    model_type = payload.get("model_type")
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")

    model_size = payload.get("model_size", "small")
    if model_size not in MODEL_SIZES:
        raise ValueError(f"model_size must be one of {MODEL_SIZES}")

    epochs = int(payload.get("epochs", 3))
    batch_size = int(payload.get("batch_size", 16))
    max_seq_len = int(payload.get("max_seq_len", 512))
    if epochs < 1 or batch_size < 1 or max_seq_len < 1:
        raise ValueError("epochs, batch_size, and max_seq_len must be at least 1")

    labels = payload.get("labels", [])
    if not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("labels must be a list of non-empty strings")
    if model_type == "classifier" and len(labels) < 2:
        raise ValueError("a classifier job needs at least two labels")

    learning_rate = float(payload.get("learning_rate", 2e-4))
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    data_path = str(payload.get("data_path", ""))
    output_dir = str(payload.get("output_dir", "./output"))
    if not data_path:
        raise ValueError("data_path is required")

    return JobConfig(
        model_type=model_type,
        model_size=model_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        labels=labels,
        data_path=data_path,
        output_dir=output_dir,
        max_seq_len=max_seq_len,
        text_field=str(payload.get("text_field", "text")),
        label_field=str(payload.get("label_field", "label")),
    )
