"""Chunking strategies. Kept intentionally simple/dependency-light -- if
your project already has its own chunking logic, wire it in by implementing
the same `ChunkStrategy` interface and passing an instance to
`IngestionPipeline(chunk_strategy=...)`.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    start_char: int
    end_char: int


class ChunkStrategy(ABC):
    @abstractmethod
    def split(self, text: str, doc_id: str) -> list[Chunk]:
        ...


class FixedSizeChunker(ChunkStrategy):
    """Splits into fixed-size character windows with optional overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text: str, doc_id: str) -> list[Chunk]:
        chunks = []
        step = self.chunk_size - self.overlap
        i = 0
        idx = 0
        n = len(text)
        while i < n:
            end = min(i + self.chunk_size, n)
            chunks.append(Chunk(id=f"{doc_id}:{idx}", text=text[i:end], start_char=i, end_char=end))
            idx += 1
            if end == n:
                break
            i += step
        return chunks


class RecursiveCharacterChunker(ChunkStrategy):
    """Tries to split on progressively finer separators (paragraphs, then
    sentences, then words) so chunks break on natural boundaries rather than
    mid-word, while still respecting a max chunk size.
    """

    _SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 1000, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        if not separators:
            return [text]
        sep, rest = separators[0], separators[1:]
        if len(text) <= self.chunk_size:
            return [text]
        pieces = text.split(sep)
        if len(pieces) == 1:
            return self._split_text(text, rest)

        merged: list[str] = []
        current = ""
        for piece in pieces:
            candidate = current + (sep if current else "") + piece
            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                if len(piece) > self.chunk_size:
                    merged.extend(self._split_text(piece, rest))
                    current = ""
                else:
                    current = piece
        if current:
            merged.append(current)
        return merged

    def split(self, text: str, doc_id: str) -> list[Chunk]:
        pieces = self._split_text(text, self._SEPARATORS)
        chunks = []
        cursor = 0
        for idx, piece in enumerate(pieces):
            start = text.find(piece, cursor)
            if start == -1:
                start = cursor
            end = start + len(piece)
            chunks.append(Chunk(id=f"{doc_id}:{idx}", text=piece, start_char=start, end_char=end))
            cursor = end
        return chunks


class SemanticChunker(ChunkStrategy):
    """Splits on sentence boundaries, then greedily groups adjacent
    sentences using embedding-similarity breakpoints. This is the strategy
    most likely to produce 100+ small chunks per document (short sentences,
    fine granularity), which is precisely the case that exhausts free-tier
    rate limits fastest -- the whole reason this library's fallback/alignment
    machinery exists.

    Note: computing similarity breakpoints requires an embedding call itself.
    To avoid a circular dependency on the provider layer, this
    implementation accepts an already-embedded-sentence similarity callback
    lazily, and falls back to fixed-length sentence grouping if none is
    provided (still valid chunks, just without the similarity-based
    breakpoint refinement).
    """

    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(
        self,
        target_chunk_size: int = 500,
        similarity_fn=None,  # optional: Callable[[str, str], float]
        similarity_threshold: float = 0.5,
    ):
        self.target_chunk_size = target_chunk_size
        self.similarity_fn = similarity_fn
        self.similarity_threshold = similarity_threshold

    def split(self, text: str, doc_id: str) -> list[Chunk]:
        sentences = [s for s in self._SENTENCE_SPLIT_RE.split(text) if s.strip()]
        chunks = []
        idx = 0
        cursor = 0
        current = ""
        current_start = 0

        for sentence in sentences:
            start = text.find(sentence, cursor)
            if start == -1:
                start = cursor
            end = start + len(sentence)
            cursor = end

            should_break = False
            if current and len(current) + len(sentence) > self.target_chunk_size:
                should_break = True
            elif current and self.similarity_fn is not None:
                sim = self.similarity_fn(current, sentence)
                if sim < self.similarity_threshold:
                    should_break = True

            if should_break:
                chunks.append(
                    Chunk(id=f"{doc_id}:{idx}", text=current, start_char=current_start, end_char=start)
                )
                idx += 1
                current = sentence
                current_start = start
            else:
                current = (current + " " + sentence) if current else sentence
                if not current_start and not chunks:
                    current_start = start

        if current:
            chunks.append(
                Chunk(id=f"{doc_id}:{idx}", text=current, start_char=current_start, end_char=cursor)
            )
        return chunks


def get_chunker(strategy: str, **kwargs) -> ChunkStrategy:
    strategy = strategy.lower()
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    if strategy == "recursive":
        return RecursiveCharacterChunker(**kwargs)
    if strategy == "semantic":
        return SemanticChunker(**kwargs)
    raise ValueError(f"Unknown chunk strategy: {strategy!r}. Use 'fixed', 'recursive', or 'semantic'.")
