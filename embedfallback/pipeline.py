"""The public, user-facing API surface: `IngestionPipeline`.

    from embedfallback import IngestionPipeline, providers

    pipeline = IngestionPipeline(
        providers=[providers.Google(api_key="..."), providers.OpenAI(api_key="...")],
        chunk_strategy="semantic",
    )
    result = pipeline.ingest("my_document.pdf")
    for chunk in result.chunks:
        my_own_vector_db.add(chunk.vector, chunk.text, chunk.metadata)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .alignment import AlignmentEngine
from .chunking import ChunkStrategy, get_chunker
from .orchestrator import IngestionOrchestrator, IngestionResult
from .providers.base import EmbeddingProvider
from .router import ProviderRouter


@dataclass
class IngestedChunk:
    vector: list[float]
    text: str
    provider: str
    alignment_confidence: float
    metadata: dict


class PipelineResult:
    """Thin, user-friendly wrapper around the orchestrator's raw
    IngestionResult, exposing a flat `.chunks` list."""

    def __init__(self, raw: IngestionResult):
        self._raw = raw
        self.doc_id = raw.doc_id
        self.canonical_provider = raw.canonical_provider
        self.canonical_model = raw.canonical_model
        self.resume_warnings = raw.resume_warnings
        self.provider_switches = raw.provider_switches
        self.elapsed_seconds = raw.elapsed_seconds

        self.chunks: list[IngestedChunk] = [
            IngestedChunk(
                vector=vector,
                text=text,
                provider=meta.provider,
                alignment_confidence=meta.alignment_confidence,
                metadata={
                    "chunk_id": meta.chunk_id,
                    "provider": meta.provider,
                    "model": meta.model,
                    "aligned_to": meta.aligned_to,
                    "alignment_confidence": meta.alignment_confidence,
                    "canonical_provider": raw.canonical_provider,
                },
            )
            for vector, text, meta in zip(raw.vectors, raw.chunk_texts, raw.metadata)
        ]

    def __repr__(self) -> str:
        return (
            f"PipelineResult(doc_id={self.doc_id!r}, chunks={len(self.chunks)}, "
            f"canonical_provider={self.canonical_provider!r}, "
            f"provider_switches={self.provider_switches})"
        )


def _load_document_text(source: str) -> tuple[str, str]:
    """Loads a document from a path (txt/pdf/md/etc) or treats the input as
    raw text if it isn't an existing file path. Returns (doc_id, text).
    """
    path = Path(source)
    if not path.exists():
        # Treat as raw text; doc_id is a hash-free short id derived from
        # content for stability across calls with the same text.
        import hashlib
        doc_id = "text-" + hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
        return doc_id, source

    doc_id = path.stem
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            import pypdf
        except ImportError as e:
            raise ImportError(
                "Reading .pdf files requires the 'pypdf' package. "
                "Install it with: pip install pypdf"
            ) from e
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return doc_id, text

    if suffix in (".html", ".htm"):
        try:
            from bs4 import BeautifulSoup
        except ImportError as e:
            raise ImportError(
                "Reading .html files requires the 'beautifulsoup4' package. "
                "Install it with: pip install beautifulsoup4"
            ) from e
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return doc_id, soup.get_text(separator="\n")

    # Plain text / markdown / anything else readable as text.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return doc_id, f.read()


class IngestionPipeline:
    def __init__(
        self,
        providers: list[EmbeddingProvider],
        chunk_strategy: str | ChunkStrategy = "recursive",
        anchor_corpus_extra: list[str] | None = None,
        chunk_kwargs: dict | None = None,
        checkpoint_dir: str | Path | None = None,
        concurrency: int = 5,
    ):
        self.providers = providers
        if len(providers) < 2:
            import warnings
            warnings.warn(
                f"IngestionPipeline configured with only {len(providers)} provider(s). "
                "If this provider hits a rate limit or quota exhaustion mid-document, "
                "ingestion will raise NoProviderAvailableError with no fallback option. "
                "Configure at least 2-3 providers for real resilience against rate limits.",
                UserWarning,
                stacklevel=2,
            )
        chunker = (
            chunk_strategy
            if isinstance(chunk_strategy, ChunkStrategy)
            else get_chunker(chunk_strategy, **(chunk_kwargs or {}))
        )
        router = ProviderRouter(providers)
        aligner = AlignmentEngine()

        orchestrator_kwargs = {"concurrency": concurrency}
        if checkpoint_dir is not None:
            orchestrator_kwargs["checkpoint_dir"] = checkpoint_dir

        self.orchestrator = IngestionOrchestrator(
            router=router,
            chunker=chunker,
            aligner=aligner,
            anchor_corpus_extra=anchor_corpus_extra,
            **orchestrator_kwargs,
        )

    def ingest(self, source: str, doc_id: str | None = None) -> PipelineResult:
        """`source` may be a file path (.txt, .md, .pdf, .html) or raw text.
        `doc_id` overrides the auto-derived document id (needed if you want
        deterministic resume behavior across raw-text calls).
        """
        auto_doc_id, text = _load_document_text(source)
        final_doc_id = doc_id or auto_doc_id
        raw = self.orchestrator.ingest_document(final_doc_id, text)
        return PipelineResult(raw)
