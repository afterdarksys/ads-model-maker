"""Turn a file into extracted text and training chunks."""

from dataclasses import dataclass
from pathlib import Path

from ads_ml.ingest.chunker import Chunk, auto_chunk
from ads_ml.ingest.documents import ExtractedDocument, extract_document


@dataclass
class IngestResult:
    """One ingested file."""

    document: ExtractedDocument
    chunks: list[Chunk]

    @property
    def text(self) -> str:
        return self.document.full_text


class IngestPipeline:
    """Extract a supported file and chunk it with one strategy."""

    def __init__(self, strategy: str = "semantic", **chunk_kwargs):
        self.strategy = strategy
        self.chunk_kwargs = chunk_kwargs

    def ingest_file(self, file_path: Path | str) -> IngestResult:
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"file not found: {path}")
        document = extract_document(path)
        chunks = auto_chunk(
            document.full_text,
            strategy=self.strategy,
            **self.chunk_kwargs,
        )
        return IngestResult(document=document, chunks=chunks)
