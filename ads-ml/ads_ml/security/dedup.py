"""Drop near-duplicate alerts before they dominate a training set.

A ticket pasted five times with a different id teaches the model the
ticket, not the class. Similarity is a 64-bit simhash over word tokens.
"""

from __future__ import annotations

import hashlib
import re


def simhash(text: str, bits: int = 64) -> int:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not tokens:
        return 0
    vector = [0] * bits
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        hashed = int.from_bytes(digest[:8], "little")
        for bit in range(bits):
            if hashed & (1 << bit):
                vector[bit] += 1
            else:
                vector[bit] -= 1
    value = 0
    for bit, weight in enumerate(vector):
        if weight > 0:
            value |= 1 << bit
    return value


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def drop_near_duplicates(records: list[dict], threshold: int = 3) -> tuple[list[dict], int]:
    """Keep the first record in each near-duplicate group.

    ``threshold`` is the maximum Hamming distance treated as the same alert.
    """
    if threshold < 0:
        raise ValueError("threshold cannot be negative")
    kept: list[dict] = []
    seen: list[int] = []
    dropped = 0
    for record in records:
        digest = simhash(str(record.get("text", "")))
        if any(hamming(digest, previous) <= threshold for previous in seen):
            dropped += 1
            continue
        kept.append(record)
        seen.append(digest)
    return kept, dropped
