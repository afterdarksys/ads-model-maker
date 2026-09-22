"""Hashed character-ngram linear classifier.

The on-disk format is ``ads-linear-v1``. The Go agent scores the same file.
Both sides must keep this feature function:

- lowercase the text
- wrap it in ``<`` and ``>``
- take every contiguous n-gram (default 3)
- SHA-256 the n-gram bytes
- index = first 4 bytes, little-endian, modulo dim
- sign = +1 when byte 5 is even, otherwise -1
- add the sign into that bucket
- score[class] = bias[class] + sum(weight[class][index] * bucket)
- probabilities are a stable softmax

This is a text classifier for labeled security notes. It is not a malware
scanner and it does not execute the text it scores.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

FORMAT = "ads-linear-v1"


def ngrams(text: str, size: int) -> list[str]:
    if size < 1:
        raise ValueError("ngram size must be at least 1")
    padded = "<" + text.lower() + ">"
    if len(padded) <= size:
        return [padded]
    return [padded[index:index + size] for index in range(len(padded) - size + 1)]


def feature_buckets(text: str, size: int, dim: int) -> dict[int, float]:
    if dim < 1:
        raise ValueError("dim must be at least 1")
    buckets: dict[int, float] = {}
    for gram in ngrams(text, size):
        digest = hashlib.sha256(gram.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        buckets[index] = buckets.get(index, 0.0) + sign
    return buckets


def stable_softmax(scores: list[float]) -> list[float]:
    if not scores:
        raise ValueError("scores are empty")
    peak = max(scores)
    shifted = [math.exp(score - peak) for score in scores]
    total = sum(shifted)
    if total == 0:
        raise ValueError("softmax total is zero")
    return [value / total for value in shifted]


class LinearTextClassifier:
    """L2-regularized softmax regression over hashed character n-grams."""

    def __init__(
        self,
        labels: list[str],
        *,
        dim: int = 1024,
        ngram: int = 3,
        learning_rate: float = 0.2,
        epochs: int = 8,
        l2: float = 0.01,
        task: str = "",
    ):
        if len(labels) < 2:
            raise ValueError("at least two labels are required")
        if len(set(labels)) != len(labels):
            raise ValueError("labels must be unique")
        self.labels = list(labels)
        self.dim = dim
        self.ngram = ngram
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.task = task
        self.bias = [0.0 for _ in labels]
        self.weights = [[0.0 for _ in range(dim)] for _ in labels]

    def _scores(self, text: str) -> list[float]:
        buckets = feature_buckets(text, self.ngram, self.dim)
        scores = []
        for class_index, bias in enumerate(self.bias):
            score = bias
            row = self.weights[class_index]
            for index, value in buckets.items():
                score += row[index] * value
            scores.append(score)
        return scores

    def predict(self, text: str) -> dict:
        if not text.strip():
            raise ValueError("text is empty")
        scores = self._scores(text)
        probabilities = stable_softmax(scores)
        best = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "label": self.labels[best],
            "confidence": probabilities[best],
            "probabilities": {
                label: probabilities[index] for index, label in enumerate(self.labels)
            },
        }

    def fit(self, texts: list[str], label_names: list[str]) -> None:
        if len(texts) != len(label_names):
            raise ValueError("texts and labels must be the same length")
        if not texts:
            raise ValueError("training data is empty")
        name_to_index = {label: index for index, label in enumerate(self.labels)}
        unknown = sorted({label for label in label_names if label not in name_to_index})
        if unknown:
            raise ValueError(f"unknown labels: {unknown}")

        order = list(range(len(texts)))
        rng = random.Random(0)
        for _ in range(self.epochs):
            rng.shuffle(order)
            for row in order:
                target = name_to_index[label_names[row]]
                buckets = feature_buckets(texts[row], self.ngram, self.dim)
                probabilities = stable_softmax(self._scores_from_buckets(buckets))
                for class_index in range(len(self.labels)):
                    gradient = probabilities[class_index] - (1.0 if class_index == target else 0.0)
                    weight_row = self.weights[class_index]
                    for index, value in buckets.items():
                        weight_row[index] -= self.learning_rate * (
                            gradient * value + self.l2 * weight_row[index]
                        )
                    self.bias[class_index] -= self.learning_rate * gradient

    def _scores_from_buckets(self, buckets: dict[int, float]) -> list[float]:
        scores = []
        for class_index, bias in enumerate(self.bias):
            score = bias
            row = self.weights[class_index]
            for index, value in buckets.items():
                score += row[index] * value
            scores.append(score)
        return scores

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "task": self.task,
            "labels": list(self.labels),
            "ngram": self.ngram,
            "dim": self.dim,
            "bias": list(self.bias),
            "weights": [list(row) for row in self.weights],
        }

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "LinearTextClassifier":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != FORMAT:
            raise ValueError("not an ads-linear-v1 model")
        labels = payload["labels"]
        model = cls(
            labels,
            dim=int(payload["dim"]),
            ngram=int(payload["ngram"]),
            task=str(payload.get("task", "")),
        )
        bias = payload["bias"]
        weights = payload["weights"]
        if len(bias) != len(labels) or len(weights) != len(labels):
            raise ValueError("bias and weights must have one row per label")
        for row in weights:
            if len(row) != model.dim:
                raise ValueError("weight row length does not match dim")
        model.bias = [float(value) for value in bias]
        model.weights = [[float(value) for value in row] for row in weights]
        return model
