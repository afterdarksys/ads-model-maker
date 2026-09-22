# Data Ingestion Pipeline

## Overview

The data ingestion system handles multiple input formats and normalizes them into a training-ready format. This is the most critical component for user experience—if we can't ingest their data easily, they won't use the platform.

## Supported Formats

### Tier 1: Core Formats (MVP)

| Format | Library | Notes |
|--------|---------|-------|
| PDF | `pypdf2`, `pdfplumber`, `unstructured` | Text + layout extraction |
| DOCX | `python-docx`, `unstructured` | Full formatting support |
| TXT/MD | Native | Direct text processing |
| CSV | `pandas` | Structured data |
| JSON/JSONL | Native | Structured data |
| PNG/JPEG | `Pillow`, `opencv` | Image + OCR |

### Tier 2: Extended Formats (Post-MVP)

| Format | Library | Notes |
|--------|---------|-------|
| DOC (legacy) | `antiword`, `textract` | Binary format |
| RTF | `striprtf` | Rich text |
| HTML | `beautifulsoup4`, `trafilatura` | Web content |
| XML | `lxml` | Structured markup |
| XLSX/XLS | `openpyxl`, `xlrd` | Spreadsheets |
| PPT/PPTX | `python-pptx` | Presentations |
| EML/MSG | `email`, `extract-msg` | Email |
| MBOX | `mailbox` | Email archives |

### Tier 3: Specialized Formats

| Format | Library | Notes |
|--------|---------|-------|
| DICOM | `pydicom` | Medical imaging |
| Parquet | `pyarrow` | Columnar data |
| Arrow | `pyarrow` | Columnar data |
| SQLite | `sqlite3` | Database export |
| TIFF | `Pillow` | Multi-page images |

## Ingestion Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────┘

     ┌──────────────┐
     │   Raw Data   │
     │   (Upload)   │
     └──────┬───────┘
            │
            ▼
┌──────────────────────┐
│   Format Detection   │  ← Magic bytes + extension
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Format Handler     │  ← Plugin-based architecture
│  ┌────────────────┐  │
│  │ PDF Handler    │  │
│  │ DOCX Handler   │  │
│  │ Image Handler  │  │
│  │ CSV Handler    │  │
│  │ ...            │  │
│  └────────────────┘  │
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Text Extraction    │
│  ┌────────────────┐  │
│  │ OCR (images)   │  │
│  │ Layout parse   │  │
│  │ Table extract  │  │
│  │ Entity detect  │  │
│  └────────────────┘  │
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Normalization      │
│  ┌────────────────┐  │
│  │ Unicode norm   │  │
│  │ Whitespace     │  │
│  │ Encoding fix   │  │
│  │ Language det   │  │
│  └────────────────┘  │
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Chunking           │
│  ┌────────────────┐  │
│  │ Semantic split │  │
│  │ Token-aware    │  │
│  │ Overlap config │  │
│  └────────────────┘  │
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Embedding          │  ← Optional, for semantic search
│  ┌────────────────┐  │
│  │ Vector embed   │  │
│  │ Index build    │  │
│  └────────────────┘  │
└──────────┬───────────┘
            │
            ▼
┌──────────────────────┐
│   Storage            │
│  ┌────────────────┐  │
│  │ Processed data │  │
│  │ Metadata       │  │
│  │ Statistics     │  │
│  └────────────────┘  │
└──────────────────────┘
```

## Format Handlers

### PDF Handler

```python
class PDFHandler(BaseHandler):
    """
    PDF processing with multiple extraction strategies.
    Falls back through methods until successful extraction.
    """

    STRATEGIES = [
        'pdfplumber',      # Best for tables and layout
        'pypdf2',          # Fast text extraction
        'unstructured',    # AI-powered extraction
        'ocr_fallback',    # For scanned PDFs
    ]

    def extract(self, file_path: Path) -> ExtractedDocument:
        # Detect if scanned (image-based) or text-based
        is_scanned = self._detect_scanned(file_path)

        if is_scanned:
            return self._ocr_extract(file_path)

        for strategy in self.STRATEGIES:
            try:
                return getattr(self, f'_extract_{strategy}')(file_path)
            except ExtractionError:
                continue

        raise ExtractionError(f"All strategies failed for {file_path}")

    def _extract_pdfplumber(self, file_path: Path) -> ExtractedDocument:
        """Extract with layout preservation"""
        import pdfplumber

        pages = []
        tables = []

        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                page_tables = page.extract_tables()

                pages.append(PageContent(
                    text=text,
                    page_number=page.page_number,
                    bbox=page.bbox,
                ))
                tables.extend(page_tables)

        return ExtractedDocument(
            pages=pages,
            tables=tables,
            metadata=self._extract_metadata(file_path),
        )
```

### Image Handler with OCR

```python
class ImageHandler(BaseHandler):
    """
    Image processing with OCR capabilities.
    Supports various image formats and OCR engines.
    """

    OCR_ENGINES = ['tesseract', 'easyocr', 'paddleocr']

    def extract(self, file_path: Path) -> ExtractedDocument:
        image = self._load_image(file_path)

        # Preprocess for better OCR
        processed = self._preprocess(image)

        # Run OCR
        text, confidence = self._run_ocr(processed)

        # Extract additional features if needed
        layout = self._detect_layout(image)

        return ExtractedDocument(
            pages=[PageContent(
                text=text,
                confidence=confidence,
                layout=layout,
            )],
            metadata={
                'dimensions': image.size,
                'format': image.format,
                'ocr_engine': self.ocr_engine,
            }
        )

    def _preprocess(self, image: Image) -> Image:
        """Preprocess image for better OCR results"""
        # Convert to grayscale
        gray = image.convert('L')

        # Denoise
        denoised = self._denoise(gray)

        # Deskew
        deskewed = self._deskew(denoised)

        # Binarize
        binary = self._binarize(deskewed)

        return binary

    def _run_ocr(self, image: Image) -> tuple[str, float]:
        """Run OCR with fallback engines"""
        import pytesseract

        # Try Tesseract first (fastest)
        try:
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT
            )
            text = ' '.join(data['text'])
            confidence = sum(data['conf']) / len(data['conf'])
            return text, confidence
        except Exception:
            pass

        # Fallback to EasyOCR (more accurate for some cases)
        import easyocr
        reader = easyocr.Reader(['en'])
        results = reader.readtext(np.array(image))
        text = ' '.join([r[1] for r in results])
        confidence = sum([r[2] for r in results]) / len(results)
        return text, confidence
```

### Document Handler (DOCX)

```python
class DocxHandler(BaseHandler):
    """
    DOCX processing with full formatting extraction.
    """

    def extract(self, file_path: Path) -> ExtractedDocument:
        from docx import Document

        doc = Document(file_path)

        sections = []
        current_section = None

        for para in doc.paragraphs:
            # Detect headers
            if para.style.name.startswith('Heading'):
                if current_section:
                    sections.append(current_section)
                current_section = Section(
                    title=para.text,
                    level=int(para.style.name[-1]),
                    content=[],
                )
            else:
                if current_section is None:
                    current_section = Section(title='', level=0, content=[])
                current_section.content.append(para.text)

        if current_section:
            sections.append(current_section)

        # Extract tables
        tables = []
        for table in doc.tables:
            table_data = []
            for row in table.rows:
                row_data = [cell.text for cell in row.cells]
                table_data.append(row_data)
            tables.append(table_data)

        return ExtractedDocument(
            sections=sections,
            tables=tables,
            metadata=self._extract_core_properties(doc),
        )
```

## Text Chunking Strategies

### Semantic Chunking

```python
class SemanticChunker:
    """
    Chunk text based on semantic boundaries.
    Preserves context better than fixed-size chunking.
    """

    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap: int = 50,
    ):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap = overlap
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    def chunk(self, text: str) -> list[Chunk]:
        # Split by paragraphs first
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para_tokens = len(self.tokenizer.encode(para))

            if current_size + para_tokens > self.max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk))
                    # Keep overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = [overlap_text] if overlap_text else []
                    current_size = len(self.tokenizer.encode(overlap_text)) if overlap_text else 0

            current_chunk.append(para)
            current_size += para_tokens

        if current_chunk:
            chunks.append(self._create_chunk(current_chunk))

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs, preserving structure"""
        # Split on double newlines
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]
```

### Token-Aware Chunking

```python
class TokenAwareChunker:
    """
    Chunk text with awareness of token boundaries.
    Ensures chunks don't exceed model context limits.
    """

    def __init__(
        self,
        model_name: str,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, text: str) -> list[Chunk]:
        tokens = self.tokenizer.encode(text)
        chunks = []

        start = 0
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))

            # Try to find a good break point (sentence end)
            if end < len(tokens):
                end = self._find_break_point(tokens, start, end)

            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(Chunk(
                text=chunk_text,
                token_count=len(chunk_tokens),
                start_idx=start,
                end_idx=end,
            ))

            start = end - self.overlap_tokens

        return chunks
```

## Data Normalization

```python
class TextNormalizer:
    """
    Normalize text for consistent processing.
    """

    def normalize(self, text: str) -> str:
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)

        # Fix encoding issues
        text = self._fix_encoding(text)

        # Normalize whitespace
        text = self._normalize_whitespace(text)

        # Remove control characters
        text = self._remove_control_chars(text)

        return text

    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues"""
        # Fix mojibake
        try:
            text = text.encode('latin-1').decode('utf-8')
        except (UnicodeDecodeError, UnicodeEncodeError):
            pass

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving structure"""
        # Replace multiple spaces with single space
        text = re.sub(r'[ \t]+', ' ', text)
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # Remove trailing whitespace from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        return text
```

## Database Ingestion

### PostgreSQL Connector

```python
class PostgresConnector(DatabaseConnector):
    """
    Ingest data from PostgreSQL databases.
    """

    def connect(self, config: DBConfig) -> None:
        import psycopg2
        self.conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            database=config.database,
            user=config.user,
            password=config.password,
        )

    def extract_tables(self, tables: list[str] = None) -> Iterator[TableData]:
        cursor = self.conn.cursor()

        if tables is None:
            # Get all tables
            cursor.execute("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """)
            tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            # Get schema
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = %s
            """, (table,))
            schema = cursor.fetchall()

            # Get data
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()

            yield TableData(
                name=table,
                schema=schema,
                rows=rows,
            )
```

## Go Agent Filesystem Collection

The Go agent handles local filesystem scanning:

```go
// collector/filesystem.go
package collector

type FileCollector struct {
    config    *Config
    scanner   *Scanner
    uploader  *Uploader
}

func (c *FileCollector) Collect(paths []string, filters FileFilters) (*CollectionResult, error) {
    var files []FileInfo

    for _, path := range paths {
        err := filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
            if err != nil {
                return err
            }

            if d.IsDir() {
                return nil
            }

            // Apply filters
            if !c.matchesFilters(p, filters) {
                return nil
            }

            info, err := c.scanner.ScanFile(p)
            if err != nil {
                log.Printf("Warning: failed to scan %s: %v", p, err)
                return nil
            }

            files = append(files, info)
            return nil
        })

        if err != nil {
            return nil, fmt.Errorf("failed to walk %s: %w", path, err)
        }
    }

    return &CollectionResult{
        Files:     files,
        TotalSize: c.calculateTotalSize(files),
        FileCount: len(files),
    }, nil
}
```

## Processing Pipeline

```python
class IngestionPipeline:
    """
    Orchestrate the full ingestion pipeline.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.handlers = self._init_handlers()
        self.chunker = SemanticChunker(
            max_chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        self.normalizer = TextNormalizer()

    async def process(self, source: DataSource) -> ProcessedDataset:
        """Process a data source into training-ready format"""

        documents = []

        async for file_path in source.iter_files():
            # Detect format
            format_type = self._detect_format(file_path)

            # Get handler
            handler = self.handlers.get(format_type)
            if not handler:
                logger.warning(f"No handler for {format_type}, skipping {file_path}")
                continue

            # Extract content
            try:
                extracted = await handler.extract(file_path)
            except ExtractionError as e:
                logger.error(f"Failed to extract {file_path}: {e}")
                continue

            # Normalize text
            for page in extracted.pages:
                page.text = self.normalizer.normalize(page.text)

            # Chunk content
            chunks = self.chunker.chunk(extracted.full_text)

            documents.append(ProcessedDocument(
                source_path=file_path,
                chunks=chunks,
                metadata=extracted.metadata,
            ))

        return ProcessedDataset(
            documents=documents,
            total_chunks=sum(len(d.chunks) for d in documents),
            config=self.config,
        )
```

## Quality Metrics

Track ingestion quality:

```python
@dataclass
class IngestionMetrics:
    total_files: int
    successful_files: int
    failed_files: int
    total_chunks: int
    total_tokens: int
    avg_chunk_size: float
    language_distribution: dict[str, int]
    format_distribution: dict[str, int]
    ocr_confidence_avg: float
    processing_time_seconds: float
```

## Error Handling

```python
class IngestionError(Exception):
    """Base exception for ingestion errors"""
    pass

class FormatNotSupportedError(IngestionError):
    """File format not supported"""
    pass

class ExtractionError(IngestionError):
    """Failed to extract content"""
    pass

class OCRError(IngestionError):
    """OCR processing failed"""
    pass

class ChunkingError(IngestionError):
    """Text chunking failed"""
    pass
```

## Configuration

```json
{
  "ingestion": {
    "supported_formats": ["pdf", "docx", "txt", "md", "csv", "json", "png", "jpg"],
    "max_file_size_mb": 100,
    "chunking": {
      "strategy": "semantic",
      "max_chunk_tokens": 512,
      "overlap_tokens": 50,
      "min_chunk_tokens": 100
    },
    "ocr": {
      "enabled": true,
      "engine": "tesseract",
      "languages": ["eng"],
      "confidence_threshold": 0.6
    },
    "normalization": {
      "unicode_form": "NFKC",
      "lowercase": false,
      "remove_extra_whitespace": true
    },
    "parallel_workers": 4
  }
}
```
