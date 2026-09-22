"""
Text chunking strategies for model training.

Chunks are the fundamental unit of training data.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class Chunk:
    """A chunk of text for training."""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict

    @property
    def char_count(self) -> int:
        return len(self.text)


class Chunker(ABC):
    """Base class for text chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        """Split text into chunks."""
        pass


class SemanticChunker(Chunker):
    """
    Chunk text based on semantic boundaries.

    Tries to keep paragraphs together, respects headings,
    and maintains context with overlap.
    """

    def __init__(
        self,
        max_chars: int = 1500,
        min_chars: int = 200,
        overlap_chars: int = 100,
        respect_sentences: bool = True,
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_chars = overlap_chars
        self.respect_sentences = respect_sentences

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        metadata = metadata or {}

        # Split into paragraphs, then break any paragraph that is
        # itself larger than max_chars. Otherwise one long block
        # becomes a single unbounded chunk.
        paragraphs = []
        for para in self._split_paragraphs(text):
            paragraphs.extend(self._split_long_paragraph(para))

        chunks = []
        current_text = ""
        current_start = 0
        char_pos = 0

        for para in paragraphs:
            para_len = len(para)

            # If adding this paragraph exceeds max, finalize current chunk
            if current_text and len(current_text) + para_len + 2 > self.max_chars:
                chunk = self._create_chunk(
                    current_text.strip(),
                    len(chunks),
                    current_start,
                    char_pos,
                    metadata,
                )
                chunks.append(chunk)

                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_text)
                current_text = overlap_text + "\n\n" if overlap_text else ""
                current_start = max(0, char_pos - len(overlap_text)) if overlap_text else char_pos

            current_text += para + "\n\n"
            char_pos += para_len + 2

        # Don't forget the last chunk
        if current_text.strip():
            chunk = self._create_chunk(
                current_text.strip(),
                len(chunks),
                current_start,
                char_pos,
                metadata,
            )
            chunks.append(chunk)

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, para: str) -> list[str]:
        """Split a paragraph that exceeds max_chars, preferring sentence boundaries."""
        if len(para) <= self.max_chars:
            return [para]

        pieces = []
        start = 0
        while start < len(para):
            end = min(start + self.max_chars, len(para))
            if end < len(para) and self.respect_sentences:
                window = para[start:end]
                cut = window.rfind('. ')
                if cut >= self.min_chars:
                    end = start + cut + 1
            piece = para[start:end].strip()
            if piece:
                pieces.append(piece)
            if end <= start:
                end = start + 1
            start = end
        return pieces

    def _get_overlap(self, text: str) -> str:
        """Get overlap text from end of current chunk."""
        if self.overlap_chars <= 0:
            return ""
        if len(text) <= self.overlap_chars:
            return text

        overlap = text[-self.overlap_chars:]

        if self.respect_sentences:
            # Try to start at a sentence boundary
            sentence_start = overlap.rfind('. ')
            if sentence_start > 0:
                return overlap[sentence_start + 2:]

        return overlap

    def _create_chunk(
        self,
        text: str,
        index: int,
        start: int,
        end: int,
        metadata: dict,
    ) -> Chunk:
        return Chunk(
            text=text,
            index=index,
            start_char=start,
            end_char=end,
            metadata={**metadata, 'chunker': 'semantic'},
        )


class TokenChunker(Chunker):
    """
    Chunk text based on token count.

    Uses a tokenizer to ensure chunks fit within model context.
    """

    def __init__(
        self,
        tokenizer,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        metadata = metadata or {}
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")

        # Tokenize full text
        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)

        if total_tokens <= self.max_tokens:
            return [Chunk(
                text=text,
                index=0,
                start_char=0,
                end_char=len(text),
                metadata={**metadata, 'chunker': 'token', 'token_count': total_tokens},
            )]

        # Overlap must be strictly smaller than the window. Otherwise
        # start_token never advances and the loop does not terminate.
        overlap = self.overlap_tokens
        if overlap < 0:
            overlap = 0
        if overlap >= self.max_tokens:
            overlap = self.max_tokens - 1

        chunks = []
        start_token = 0

        while start_token < total_tokens:
            end_token = min(start_token + self.max_tokens, total_tokens)

            # Get chunk tokens and decode
            chunk_tokens = tokens[start_token:end_token]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Approximate character positions
            # (not exact due to tokenization)
            char_ratio = len(text) / total_tokens
            start_char = int(start_token * char_ratio)
            end_char = int(end_token * char_ratio)

            chunks.append(Chunk(
                text=chunk_text,
                index=len(chunks),
                start_char=start_char,
                end_char=end_char,
                metadata={
                    **metadata,
                    'chunker': 'token',
                    'token_count': len(chunk_tokens),
                },
            ))

            if end_token >= total_tokens:
                break

            next_start = end_token - overlap
            if next_start <= start_token:
                next_start = start_token + 1
            start_token = next_start

        return chunks


class SentenceChunker(Chunker):
    """
    Chunk text by sentences.

    Groups sentences until max size is reached.
    Good for Q&A and retrieval tasks.
    """

    def __init__(
        self,
        max_chars: int = 1000,
        min_chars: int = 100,
        overlap_sentences: int = 1,
    ):
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        metadata = metadata or {}

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current_sentences = []
        current_start = 0
        char_pos = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # Check if we need to start a new chunk
            current_len = sum(len(s) + 1 for s in current_sentences)
            if current_sentences and current_len + sentence_len > self.max_chars:
                chunk_text = ' '.join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=char_pos,
                    metadata={**metadata, 'chunker': 'sentence'},
                ))

                # Keep overlap sentences
                if self.overlap_sentences > 0:
                    current_sentences = current_sentences[-self.overlap_sentences:]
                    current_start = char_pos - sum(len(s) + 1 for s in current_sentences)
                else:
                    current_sentences = []
                    current_start = char_pos

            current_sentences.append(sentence)
            char_pos += sentence_len + 1

        # Last chunk. A short tail is merged into the previous chunk so
        # unique trailing sentences are not discarded.
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if chunks and len(chunk_text) < self.min_chars:
                previous = chunks[-1]
                previous.text = f"{previous.text} {chunk_text}".strip()
                previous.end_char = char_pos
            else:
                chunks.append(Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=current_start,
                    end_char=char_pos,
                    metadata={**metadata, 'chunker': 'sentence'},
                ))

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        # Could be improved with nltk or spacy
        sentence_endings = re.compile(r'(?<=[.!?])\s+')
        sentences = sentence_endings.split(text)
        return [s.strip() for s in sentences if s.strip()]


class FixedChunker(Chunker):
    """Split text into fixed character windows with optional overlap."""

    def __init__(self, max_chars: int = 1000, overlap_chars: int = 0):
        if max_chars < 1:
            raise ValueError("max_chars must be at least 1")
        if overlap_chars < 0:
            raise ValueError("overlap_chars cannot be negative")
        if overlap_chars >= max_chars:
            overlap_chars = max_chars - 1
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, text: str, metadata: Optional[dict] = None) -> list[Chunk]:
        metadata = metadata or {}
        if not text:
            return []

        step = self.max_chars - self.overlap_chars
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.max_chars, len(text))
            chunks.append(Chunk(
                text=text[start:end],
                index=len(chunks),
                start_char=start,
                end_char=end,
                metadata={**metadata, 'chunker': 'fixed'},
            ))
            if end >= len(text):
                break
            start += step
        return chunks


def auto_chunk(
    text: str,
    strategy: str = 'semantic',
    **kwargs,
) -> list[Chunk]:
    """
    Automatically chunk text with sensible defaults.

    Args:
        text: Text to chunk
        strategy: 'semantic', 'sentence', or 'fixed'
        **kwargs: Strategy-specific parameters

    Returns:
        List of chunks
    """
    if strategy == 'semantic':
        chunker = SemanticChunker(**kwargs)
    elif strategy == 'sentence':
        chunker = SentenceChunker(**kwargs)
    elif strategy == 'fixed':
        chunker = FixedChunker(**kwargs)
    else:
        raise ValueError(
            f"unknown chunk strategy {strategy!r}; choose semantic, sentence, or fixed"
        )

    return chunker.chunk(text)
