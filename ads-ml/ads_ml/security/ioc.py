"""Undo common defanging so a security model sees the indicator shape.

Threat reports write hxxp and [.] so a reader does not click the link.
A classifier trained on the defanged form misses the same text once it is
written normally. This only rewrites those spellings. It does not resolve
DNS, fetch URLs, or contact the host.
"""

from __future__ import annotations

import re

_REPLACEMENTS = (
    (re.compile(r"(?i)hxxps"), "https"),
    (re.compile(r"(?i)hxxp"), "http"),
    (re.compile(r"(?i)\[\.\]|\(\.\)|\[dot\]"), "."),
    (re.compile(r"(?i)\[@\]|\(at\)"), "@"),
)


def normalize_defanged(text: str) -> str:
    cleaned = text
    for pattern, replacement in _REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)
    return cleaned
