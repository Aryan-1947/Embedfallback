"""Wires chunking -> routing -> embedding -> alignment -> output into one
resumable pipeline, with resume-time checkpoint validation.

Chunks within a document are embedded in concurrent batches (default 5 at
a time) rather than strictly one-at-a-time, to reduce total ingestion time
for large documents. Correctness (fallback, alignment, checkpointing,
resumability) is unchanged -- concurrency only affects throughput.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .alignment import AlignmentEngine, get_anchor_corpus
from .chunking import Chunk, ChunkStrategy
from .providers.base import EmbeddingProvider, RateLimitExceeded
from .router import NoProviderAvailableError, ProviderRouter

logger = logging.getLogger("embedfallback.orchestrator")

DEFAULT_CHECKPOINT_DIR = Path.home() / ".embedfallback" / "checkpoints"
DEFAULT_CONCURRENCY = 5


@dataclass
class ChunkMetadata:
    chunk_id: str
    provider: str
    model: str
    aligned_to: str | None
    alignment_confidence: float


@dataclass
class IngestionResult:
    doc_id: str
    vectors: list[list[float]]
    chunk_texts: list[str]
    metadata: list[ChunkMetadata]
    canonical_provider: str
    canonical_model: str
    resume_warnings: list[str] = field(default_factory=list)
    provider_switches: int = 0
    elapsed_seconds: float = 0.0


class ResumeValidationError(Exception):
    """Raised when a checkpoint can't be trusted as-is (e.g. its canonical
    provider is no longer configured). Fails loudly rather than silently
    re-picking a provider and quietly mixing vector spaces.
    """


class IngestionOrchestrator:
    def __init__(
        self,
        router: ProviderRouter,
        chunker: ChunkStrategy,
        aligner: AlignmentEngine | None = None,
        anchor_corpus_extra: list[str] | None = None,
        checkpoint_dir: Path | str = DEFAULT_CHECKPOINT_DIR,
        max_provider_attempts_per_batch: int | None = None,
        concurrency: int = DEFAULT_CONCURRENCY,
    ):
        self.router = router
        self.chunker = chunker
        self.aligner = aligner or AlignmentEngine()
        self.corpus_texts, self.corpus_version = get_anchor_corpus(anchor_corpus_extra)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_provider_attempts_per_batch = (
            max_provider_attempts_per_batch or len(router.providers)
        )
        # How many chunks to embed concurrently. 1 reproduces the old
        # strictly-sequential behavior exactly.
        self.concurrency = max(1, concurrency)
        self._router_lock = threading.Lock()

    # -- checkpointing -----------------------------------------------------

    def _checkpoint_path(self, doc_id: str) -> Path:
        return self.checkpoint_dir / f"{doc_id}.ingestion_state.json"

    def _load_checkpoint(self, doc_id: str) -> dict | None:
        path = self._checkpoint_path(doc_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_checkpoint(self, doc_id: str, state: dict) -> None:
        path = self._checkpoint_path(doc_id)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        tmp.replace(path)

    def _validate_checkpoint(self, checkpoint: dict, resume_warnings: list[str]) -> bool:
        canonical_name = checkpoint["canonical_provider"]
        canonical_model = checkpoint["canonical_model"]

        provider = self.router._by_name.get(canonical_name)
        if provider is None:
            raise ResumeValidationError(
                f"Checkpoint's canonical provider '{canonical_name}' is not in the "
                f"current provider configuration. Cannot safely resume -- either "
                f"re-add that provider or start a fresh ingestion (delete "
                f"{self._checkpoint_path(checkpoint['doc_id']).name})."
            )

        if provider.model != canonical_model:
            msg = (
                f"Checkpoint's canonical model '{canonical_model}' for provider "
                f"'{canonical_name}' does not match the currently configured "
                f"model '{provider.model}'. Treating remaining chunks as a NEW "
                f"canonical-provider situation; alignment will be re-derived "
                f"rather than reusing any cached transform for the old model."
            )
            logger.warning(msg)
            resume_warnings.append(msg)
            return False

        if checkpoint.get("corpus_version") != self.corpus_version:
            msg = (
                f"Checkpoint's anchor corpus_version '{checkpoint.get('corpus_version')}' "
                f"differs from the currently configured corpus_version "
                f"'{self.corpus_version}' (base corpus or anchor_corpus_extra changed). "
                f"Alignment transforms will be recomputed against the new corpus."
            )
            logger.warning(msg)
            resume_warnings.append(msg)

        return True

    # -- per-chunk embedding (runs inside worker threads) -------------------

    def _embed_one_chunk(
        self, chunk: Chunk, canonical_holder: list,
    ) -> tuple[list[float], str, ChunkMetadata, bool]:
        """Embeds a single chunk, trying providers with fallback. Thread-safe:
        all reads/writes of shared router state and the canonical-provider
        holder happen under self._router_lock. `canonical_holder` is a
        2-element list [provider_name, model] acting as a mutable box shared
        across concurrently-running chunks in the same batch (and across
        batches), so whichever chunk's embed call completes first legitimately
        establishes the canonical provider for the whole document -- exactly
        matching the original sequential "first successful embed wins" rule.
        """
        excluded: set[str] = set()
        attempts = 0

        while True:
            attempts += 1
            if attempts > self.max_provider_attempts_per_batch:
                raise NoProviderAvailableError(dict(self.router.cooldowns))

            with self._router_lock:
                provider = self.router.get_available_provider(exclude=excluded)
                is_canonical_pinning_moment = canonical_holder[0] is None

            try:
                result = provider.embed([chunk.text])
            except RateLimitExceeded as e:
                with self._router_lock:
                    self.router.mark_rate_limited(
                        provider.name, e.kind, e.retry_after_seconds, reason=e.message
                    )
                excluded.add(provider.name)
                continue

            with self._router_lock:
                if is_canonical_pinning_moment and canonical_holder[0] is None:
                    canonical_holder[0] = provider.name
                    canonical_holder[1] = provider.model
                canonical_name = canonical_holder[0]

            vector = result.vectors[0]
            if provider.name != canonical_name:
                canonical_provider_obj = self.router._by_name[canonical_name]
                aligned_vectors, confidence = self.aligner.align(
                    [vector], provider, canonical_provider_obj,
                    self.corpus_version, self.corpus_texts,
                )
                vector = aligned_vectors[0]
                meta = ChunkMetadata(
                    chunk_id=chunk.id, provider=provider.name, model=provider.model,
                    aligned_to=canonical_name, alignment_confidence=confidence,
                )
                switched = True
            else:
                meta = ChunkMetadata(
                    chunk_id=chunk.id, provider=provider.name, model=provider.model,
                    aligned_to=None, alignment_confidence=1.0,
                )
                switched = False

            return vector, chunk.text, meta, switched

    # -- main entrypoint -----------------------------------------------------

    def ingest_document(self, doc_id: str, text: str) -> IngestionResult:
        start_time = time.time()
        resume_warnings: list[str] = []
        checkpoint = self._load_checkpoint(doc_id)

        all_chunks: list[Chunk] = self.chunker.split(text, doc_id)
        completed_ids: set[str] = set()
        canonical_provider_name: str | None = None
        canonical_model: str | None = None

        if checkpoint is not None:
            checkpoint["doc_id"] = doc_id
            trusted = self._validate_checkpoint(checkpoint, resume_warnings)
            completed_ids = set(checkpoint.get("completed_chunk_ids", []))
            if trusted:
                canonical_provider_name = checkpoint["canonical_provider"]
                canonical_model = checkpoint["canonical_model"]
            else:
                canonical_provider_name = None
                canonical_model = None

        vectors: list[list[float]] = []
        chunk_texts: list[str] = []
        metadata: list[ChunkMetadata] = []
        provider_switches = 0

        if checkpoint is not None and completed_ids:
            saved_chunk_data = checkpoint.get("chunk_data", {})
            for chunk in all_chunks:
                if chunk.id in completed_ids and chunk.id in saved_chunk_data:
                    saved = saved_chunk_data[chunk.id]
                    vectors.append(saved["vector"])
                    chunk_texts.append(saved["text"])
                    metadata.append(ChunkMetadata(
                        chunk_id=chunk.id,
                        provider=saved["provider"],
                        model=saved["model"],
                        aligned_to=saved.get("aligned_to"),
                        alignment_confidence=saved["alignment_confidence"],
                    ))

        remaining = [c for c in all_chunks if c.id not in completed_ids]

        canonical_holder = [canonical_provider_name, canonical_model]

        i = 0
        while i < len(remaining):
            batch = remaining[i:i + self.concurrency]

            batch_results: dict[str, tuple] = {}
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                future_to_chunk = {
                    executor.submit(self._embed_one_chunk, chunk, canonical_holder): chunk
                    for chunk in batch
                }
                for future in as_completed(future_to_chunk):
                    chunk = future_to_chunk[future]
                    batch_results[chunk.id] = future.result()

            # Preserve original chunk order within the batch, regardless of
            # which thread happened to finish first.
            for chunk in batch:
                vector, chunk_text, meta, switched = batch_results[chunk.id]
                vectors.append(vector)
                chunk_texts.append(chunk_text)
                metadata.append(meta)
                completed_ids.add(chunk.id)
                if switched:
                    provider_switches += 1

            canonical_provider_name, canonical_model = canonical_holder[0], canonical_holder[1]

            self._save_checkpoint(doc_id, {
                "canonical_provider": canonical_provider_name,
                "canonical_model": canonical_model,
                "corpus_version": self.corpus_version,
                "completed_chunk_ids": sorted(completed_ids),
                "chunk_data": {
                    m.chunk_id: {
                        "vector": v,
                        "text": t,
                        "provider": m.provider,
                        "model": m.model,
                        "aligned_to": m.aligned_to,
                        "alignment_confidence": m.alignment_confidence,
                    }
                    for v, t, m in zip(vectors, chunk_texts, metadata)
                },
            })

            i += self.concurrency

        elapsed = time.time() - start_time
        return IngestionResult(
            doc_id=doc_id,
            vectors=vectors,
            chunk_texts=chunk_texts,
            metadata=metadata,
            canonical_provider=canonical_provider_name or "",
            canonical_model=canonical_model or "",
            resume_warnings=resume_warnings,
            provider_switches=provider_switches,
            elapsed_seconds=elapsed,
        )