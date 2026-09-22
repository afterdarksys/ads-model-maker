"""
Password strength classification dataset and labeling pipeline.

This module provides tools for:
1. Loading password datasets from OCI Object Storage or local files
2. Labeling passwords by crackability tier using heuristics
3. Creating PyTorch datasets for training password strength classifiers

Labeling Strategy:
- Passwords are labeled based on estimated crackability, not just complexity rules
- A password like "Summer2024!" meets all complexity requirements but is WEAK
  because it follows common patterns (Season + Year + Symbol)
- We use multiple signals: dictionary words, patterns, character substitutions,
  keyboard sequences, and known weak password structures
"""

import re
import math
from enum import Enum
from dataclasses import dataclass
from typing import Iterator, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset


class PasswordStrengthTier(str, Enum):
    """Password strength classification tiers."""

    WEAK = "weak"  # Crackable in seconds to minutes
    MEDIUM = "medium"  # Crackable in hours to days
    STRONG = "strong"  # Would take significant compute


# Common password patterns that indicate weakness
COMMON_PATTERNS = [
    # Season + Year patterns
    r"(?i)(spring|summer|fall|winter|autumn)(20\d{2}|19\d{2})",
    # Month + Year
    r"(?i)(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(20\d{2}|19\d{2})",
    # Word + Numbers + Symbol
    r"(?i)^[a-z]+\d{2,4}[!@#$%^&*]+$",
    # Capitalized word + numbers
    r"^[A-Z][a-z]+\d+$",
    # Password + variations
    r"(?i)p[a@]ss?w[o0]rd",
    # Qwerty variations
    r"(?i)qwerty|asdf|zxcv",
    # 123456 variations
    r"12345|54321|11111|00000",
    # Names + birth year pattern
    r"(?i)^[a-z]{3,10}(19[5-9]\d|20[0-2]\d)$",
    # Love patterns
    r"(?i)^i?love[a-z]*\d*$",
    # Welcome patterns
    r"(?i)welcome\d*",
    # Admin patterns
    r"(?i)admin\d*",
    # Letmein patterns
    r"(?i)letmein\d*",
    # Repeated characters
    r"(.)\1{3,}",
    # Sequential letters
    r"abcd|bcde|cdef|defg|efgh|fghi|ghij|hijk|ijkl|jklm|klmn|lmno|mnop|nopq|opqr|pqrs|qrst|rstu|stuv|tuvw|uvwx|vwxy|wxyz",
    # Sequential numbers
    r"0123|1234|2345|3456|4567|5678|6789|9876|8765|7654|6543|5432|4321|3210",
    # Keyboard walks
    r"(?i)qazwsx|wsxedc|edcrfv|rfvtgb|tgbyhn|yhnujm|!qaz|@wsx|#edc",
    # Common substitutions that are well-known
    r"(?i)[a@][s$][s$]",  # a$$, @ss
    r"(?i)l[e3]et|1337|h[a4]ck|h[a@]x",
]

# Common weak base words (will be checked with variations)
WEAK_BASE_WORDS = {
    "password", "admin", "root", "user", "login", "welcome", "master",
    "dragon", "monkey", "shadow", "sunshine", "princess", "football",
    "baseball", "soccer", "hockey", "basketball", "qwerty", "letmein",
    "trustno", "access", "flower", "hello", "charlie", "donald", "batman",
    "superman", "harley", "robert", "daniel", "jennifer", "michelle",
    "jordan", "hunter", "ranger", "buster", "thomas", "tigger", "soccer",
    "killer", "george", "andrew", "jessica", "ashley", "nicole", "joshua",
    "amanda", "samantha", "whatever", "secret", "starwars", "computer",
    "internet", "pokemon", "matrix", "cheese", "pepper", "diamond",
    "ginger", "cookie", "summer", "winter", "spring", "autumn", "monday",
    "friday", "orange", "banana", "purple", "yellow", "silver", "golden",
}

# Leet speak substitutions
LEET_MAP = {
    "a": ["4", "@"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7"],
    "l": ["1"],
    "b": ["8"],
    "g": ["9"],
}


def _normalize_leet(password: str) -> str:
    """Convert leet speak back to normal letters."""
    result = password.lower()
    for letter, substitutes in LEET_MAP.items():
        for sub in substitutes:
            result = result.replace(sub, letter)
    return result


def _calculate_entropy(password: str) -> float:
    """Calculate Shannon entropy of password."""
    if not password:
        return 0.0

    # Count character frequencies
    freq = {}
    for char in password:
        freq[char] = freq.get(char, 0) + 1

    # Calculate entropy
    entropy = 0.0
    length = len(password)
    for count in freq.values():
        prob = count / length
        entropy -= prob * math.log2(prob)

    return entropy * length


def _get_charset_size(password: str) -> int:
    """Estimate character set size used in password."""
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_symbol = any(not c.isalnum() for c in password)

    size = 0
    if has_lower:
        size += 26
    if has_upper:
        size += 26
    if has_digit:
        size += 10
    if has_symbol:
        size += 32  # Common symbols

    return max(size, 10)


def _has_dictionary_word(password: str, min_length: int = 4) -> bool:
    """Check if password contains common dictionary words."""
    normalized = _normalize_leet(password)

    # Check for weak base words
    for word in WEAK_BASE_WORDS:
        if word in normalized:
            return True

    return False


def _matches_common_pattern(password: str) -> bool:
    """Check if password matches known weak patterns."""
    for pattern in COMMON_PATTERNS:
        if re.search(pattern, password):
            return True
    return False


def _has_keyboard_pattern(password: str) -> bool:
    """Check for keyboard walk patterns."""
    keyboard_rows = [
        "qwertyuiop",
        "asdfghjkl",
        "zxcvbnm",
        "1234567890",
    ]

    lower = password.lower()

    for row in keyboard_rows:
        for i in range(len(row) - 3):
            if row[i : i + 4] in lower or row[i : i + 4][::-1] in lower:
                return True

    return False


@dataclass
class PasswordFeatures:
    """Extracted features from a password for classification."""

    length: int
    entropy: float
    charset_size: int
    has_upper: bool
    has_lower: bool
    has_digit: bool
    has_symbol: bool
    has_dictionary_word: bool
    matches_pattern: bool
    has_keyboard_walk: bool
    repeated_chars: int
    sequential_chars: int


class PasswordLabeler:
    """
    Labels passwords by estimated crackability.

    Uses a combination of:
    - Entropy calculation
    - Pattern matching for known weak structures
    - Dictionary word detection (with leet speak normalization)
    - Keyboard walk detection
    - Length and character class analysis
    """

    def __init__(
        self,
        weak_threshold: float = 35.0,
        strong_threshold: float = 60.0,
    ):
        """
        Initialize labeler with entropy thresholds.

        Args:
            weak_threshold: Entropy below this is WEAK
            strong_threshold: Entropy above this is STRONG
        """
        self.weak_threshold = weak_threshold
        self.strong_threshold = strong_threshold

    def extract_features(self, password: str) -> PasswordFeatures:
        """Extract features from a password."""
        return PasswordFeatures(
            length=len(password),
            entropy=_calculate_entropy(password),
            charset_size=_get_charset_size(password),
            has_upper=any(c.isupper() for c in password),
            has_lower=any(c.islower() for c in password),
            has_digit=any(c.isdigit() for c in password),
            has_symbol=any(not c.isalnum() for c in password),
            has_dictionary_word=_has_dictionary_word(password),
            matches_pattern=_matches_common_pattern(password),
            has_keyboard_walk=_has_keyboard_pattern(password),
            repeated_chars=self._count_repeated(password),
            sequential_chars=self._count_sequential(password),
        )

    def _count_repeated(self, password: str) -> int:
        """Count repeated character sequences."""
        count = 0
        prev = ""
        streak = 0
        for c in password:
            if c == prev:
                streak += 1
            else:
                if streak >= 3:
                    count += streak
                streak = 1
                prev = c
        if streak >= 3:
            count += streak
        return count

    def _count_sequential(self, password: str) -> int:
        """Count sequential character runs."""
        count = 0
        prev_ord = 0
        streak = 0
        for c in password.lower():
            curr_ord = ord(c)
            if curr_ord == prev_ord + 1 or curr_ord == prev_ord - 1:
                streak += 1
            else:
                if streak >= 3:
                    count += streak
                streak = 1
            prev_ord = curr_ord
        if streak >= 3:
            count += streak
        return count

    def calculate_score(self, password: str) -> float:
        """
        Calculate a crackability score (0-100).

        Lower scores = easier to crack = weaker password
        """
        features = self.extract_features(password)

        # Start with base entropy score
        score = min(features.entropy * 1.5, 50.0)

        # Length bonus (capped)
        if features.length >= 16:
            score += 15
        elif features.length >= 12:
            score += 10
        elif features.length >= 10:
            score += 5

        # Character class bonuses
        class_count = sum(
            [features.has_upper, features.has_lower, features.has_digit, features.has_symbol]
        )
        score += class_count * 5

        # Penalties for weaknesses
        if features.has_dictionary_word:
            score -= 25
        if features.matches_pattern:
            score -= 30
        if features.has_keyboard_walk:
            score -= 20
        if features.repeated_chars > 0:
            score -= features.repeated_chars * 3
        if features.sequential_chars > 0:
            score -= features.sequential_chars * 2

        # Short passwords are always weak regardless of other factors
        if features.length < 8:
            score = min(score, 20)

        return max(0.0, min(100.0, score))

    def label(self, password: str) -> PasswordStrengthTier:
        """
        Label a password by strength tier.

        Args:
            password: The password to classify

        Returns:
            PasswordStrengthTier enum value
        """
        score = self.calculate_score(password)

        if score < self.weak_threshold:
            return PasswordStrengthTier.WEAK
        elif score >= self.strong_threshold:
            return PasswordStrengthTier.STRONG
        else:
            return PasswordStrengthTier.MEDIUM

    def label_batch(self, passwords: list[str]) -> list[PasswordStrengthTier]:
        """Label a batch of passwords."""
        return [self.label(p) for p in passwords]


class PasswordDataset(Dataset):
    """
    PyTorch dataset for password classification.

    Supports loading from:
    - OCI Object Storage
    - Local files
    - In-memory lists
    """

    def __init__(
        self,
        passwords: list[str],
        labels: list[PasswordStrengthTier] | None = None,
        labeler: PasswordLabeler | None = None,
        max_length: int = 64,
    ):
        """
        Initialize dataset.

        Args:
            passwords: List of passwords
            labels: Pre-computed labels (optional)
            labeler: Labeler to use if labels not provided
            max_length: Maximum password length for encoding
        """
        self.passwords = passwords
        self.max_length = max_length
        self.label_names = [t.value for t in PasswordStrengthTier]
        self.label_to_id = {t.value: i for i, t in enumerate(PasswordStrengthTier)}

        if labels is not None:
            self.labels = labels
        elif labeler is not None:
            self.labels = labeler.label_batch(passwords)
        else:
            # Default labeler
            labeler = PasswordLabeler()
            self.labels = labeler.label_batch(passwords)

    def __len__(self) -> int:
        return len(self.passwords)

    def __getitem__(self, idx: int) -> dict:
        password = self.passwords[idx]
        label = self.labels[idx]

        # Character-level encoding (works well for passwords)
        input_ids = self._encode(password)

        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones(self.max_length, dtype=torch.long),
            "labels": torch.tensor(self.label_to_id[label.value], dtype=torch.long),
        }

    def _encode(self, password: str) -> torch.Tensor:
        """
        Encode password as character IDs.

        Uses a simple but effective encoding:
        - 0: padding
        - 1-26: lowercase a-z
        - 27-52: uppercase A-Z
        - 53-62: digits 0-9
        - 63+: symbols (mapped by ord)
        """
        ids = []
        for c in password[: self.max_length]:
            if c.islower():
                ids.append(ord(c) - ord("a") + 1)
            elif c.isupper():
                ids.append(ord(c) - ord("A") + 27)
            elif c.isdigit():
                ids.append(ord(c) - ord("0") + 53)
            else:
                # Symbols: use modulo to keep in reasonable range
                ids.append(63 + (ord(c) % 32))

        # Pad to max_length
        ids.extend([0] * (self.max_length - len(ids)))

        return torch.tensor(ids, dtype=torch.long)

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        labeler: PasswordLabeler | None = None,
        max_passwords: int | None = None,
        **kwargs,
    ) -> "PasswordDataset":
        """
        Load dataset from a text file (one password per line).

        Args:
            file_path: Path to password file
            labeler: Optional labeler
            max_passwords: Limit number of passwords loaded
        """
        passwords = []
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if max_passwords and i >= max_passwords:
                    break
                password = line.strip()
                if password:
                    passwords.append(password)

        return cls(passwords, labeler=labeler, **kwargs)

    @classmethod
    def from_oci(
        cls,
        bucket: str,
        prefix: str = "",
        labeler: PasswordLabeler | None = None,
        max_passwords: int | None = None,
        **kwargs,
    ) -> "PasswordDataset":
        """
        Load dataset from OCI Object Storage.

        Args:
            bucket: OCI bucket name
            prefix: Object prefix filter
            labeler: Optional labeler
            max_passwords: Limit number of passwords loaded
        """
        from ads_ml.storage import OCIStorage

        storage = OCIStorage(bucket=bucket)
        passwords = []

        for obj_name in storage.list_objects(prefix):
            if max_passwords and len(passwords) >= max_passwords:
                break

            for line in storage.stream_lines(obj_name):
                if max_passwords and len(passwords) >= max_passwords:
                    break
                password = line.strip()
                if password:
                    passwords.append(password)

        return cls(passwords, labeler=labeler, **kwargs)


class StreamingPasswordDataset(IterableDataset):
    """
    Streaming dataset for very large password files.

    Doesn't load all passwords into memory - streams from storage.
    """

    def __init__(
        self,
        source: Callable[[], Iterator[str]],
        labeler: PasswordLabeler | None = None,
        max_length: int = 64,
    ):
        """
        Initialize streaming dataset.

        Args:
            source: Callable that returns an iterator of passwords
            labeler: Password labeler
            max_length: Maximum password length
        """
        self.source = source
        self.labeler = labeler or PasswordLabeler()
        self.max_length = max_length
        self.label_names = [t.value for t in PasswordStrengthTier]
        self.label_to_id = {t.value: i for i, t in enumerate(PasswordStrengthTier)}

    def __iter__(self):
        for password in self.source():
            password = password.strip()
            if not password:
                continue

            label = self.labeler.label(password)
            input_ids = self._encode(password)

            yield {
                "input_ids": input_ids,
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "labels": torch.tensor(self.label_to_id[label.value], dtype=torch.long),
            }

    def _encode(self, password: str) -> torch.Tensor:
        """Encode password as character IDs."""
        ids = []
        for c in password[: self.max_length]:
            if c.islower():
                ids.append(ord(c) - ord("a") + 1)
            elif c.isupper():
                ids.append(ord(c) - ord("A") + 27)
            elif c.isdigit():
                ids.append(ord(c) - ord("0") + 53)
            else:
                ids.append(63 + (ord(c) % 32))

        ids.extend([0] * (self.max_length - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    @classmethod
    def from_oci(
        cls,
        bucket: str,
        prefix: str = "",
        **kwargs,
    ) -> "StreamingPasswordDataset":
        """
        Create streaming dataset from OCI Object Storage.

        Args:
            bucket: OCI bucket name
            prefix: Object prefix filter
        """
        from ads_ml.storage import OCIStorage

        def source():
            storage = OCIStorage(bucket=bucket)
            for obj_name in storage.list_objects(prefix):
                yield from storage.stream_lines(obj_name)

        return cls(source, **kwargs)
