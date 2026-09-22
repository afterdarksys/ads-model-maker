"""Local corpus: add, update, remove, and query training records.

Records live in one JSONL file. The content hash is the full SHA-256 of
the text, so the same paragraph is not stored twice unless dedup is off.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CorpusRecord:
    id: str
    text: str
    content_hash: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "content_hash": self.content_hash,
            "metadata": self.metadata,
        }


class Corpus:
    """Small on-disk corpus. Rewrites the JSONL file on each mutation."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.records: dict[str, CorpusRecord] = {}
        if self.path.exists():
            self._load()

    def add(
        self,
        text: str,
        record_id: str | None = None,
        metadata: dict | None = None,
        dedup: bool = True,
    ) -> CorpusRecord:
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("text is empty")
        content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if dedup:
            for existing in self.records.values():
                if existing.content_hash == content_hash:
                    return existing

        record_id = record_id or content_hash[:16]
        if record_id in self.records:
            raise ValueError(f"record id already exists: {record_id}")

        record = CorpusRecord(
            id=record_id,
            text=cleaned,
            content_hash=content_hash,
            metadata=dict(metadata or {}),
        )
        self.records[record_id] = record
        self._save()
        return record

    def update(
        self,
        record_id: str,
        text: str | None = None,
        metadata: dict | None = None,
    ) -> CorpusRecord:
        current = self.records.get(record_id)
        if current is None:
            raise KeyError(f"record not found: {record_id}")
        if text is not None:
            cleaned = text.strip()
            if not cleaned:
                raise ValueError("text is empty")
            current.text = cleaned
            current.content_hash = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        if metadata is not None:
            current.metadata = dict(metadata)
        self._save()
        return current

    def remove(self, record_id: str) -> bool:
        if record_id not in self.records:
            return False
        del self.records[record_id]
        self._save()
        return True

    def query(
        self,
        text: str | None = None,
        metadata: dict | None = None,
        limit: int = 50,
    ) -> list[CorpusRecord]:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        needle = text.casefold() if text else None
        matches = []
        for record in self.records.values():
            if needle is not None and needle not in record.text.casefold():
                continue
            if metadata is not None and any(
                record.metadata.get(key) != value for key, value in metadata.items()
            ):
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def stats(self) -> dict:
        chars = sum(len(record.text) for record in self.records.values())
        return {
            "documents": len(self.records),
            "characters": chars,
            "bytes_at_rest": self.path.stat().st_size if self.path.exists() else 0,
        }

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                record_id = payload.get("id")
                text = payload.get("text")
                if not isinstance(record_id, str) or not isinstance(text, str):
                    raise ValueError(f"{self.path}:{line_number} is missing id or text")
                self.records[record_id] = CorpusRecord(
                    id=record_id,
                    text=text,
                    content_hash=payload.get("content_hash")
                    or hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    metadata=payload.get("metadata") or {},
                )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for record in self.records.values():
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        temporary.replace(self.path)
