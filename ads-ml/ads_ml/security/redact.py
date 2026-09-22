"""Remove secrets from text before it becomes training data.

Threats: a security corpus often contains pasted tokens, cloud keys, and
private key blocks. Those must not be learned as features or written into
a model card. This does not detect every secret, and it does not encrypt
the original file. The caller still has to keep the source file private.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Patterns name the class of secret. The matched text is counted and
# replaced. It is never returned.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "pem_private_key",
        re.compile(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
            r"[\s\S]+?"
            r"-----END (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
        ),
    ),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]+=*")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password|passwd)\b\s*[:=]\s*\S+"
        ),
    ),
]


@dataclass
class RedactionResult:
    text: str
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def removed(self) -> int:
        return sum(self.counts.values())


def redact_text(text: str) -> RedactionResult:
    """Replace recognized secrets with a fixed marker."""
    counts: dict[str, int] = {}
    cleaned = text
    for name, pattern in _PATTERNS:
        cleaned, found = pattern.subn("[REDACTED]", cleaned)
        if found:
            counts[name] = found
    return RedactionResult(text=cleaned, counts=counts)
