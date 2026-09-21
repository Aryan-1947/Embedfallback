import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from embedfallback.alignment import AlignmentEngine
from embedfallback.chunking import FixedSizeChunker
from embedfallback.orchestrator import IngestionOrchestrator, ResumeValidationError
from embedfallback.providers.base import (
    EmbeddingProvider, EmbeddingResult, RateLimitExceeded, RateLimitKind,
)
from embedfallback.router import ProviderRouter


class DeterministicFakeProvider(EmbeddingProvider):
    """Embeds text deterministically as a function of its hash, so the same
    text always gets the same vector -- lets us test alignment without real
    API calls. Can be told to always raise RateLimitExceeded to simulate a
    provider that's dead.
    """

    def __init__(self, name, dimension=16, model="fake-v1", always_rate_limit=False, kind=RateLimitKind.SHORT_COOLDOWN):
        self.name = name
        self.model = model
        self.dimension = dimension
        self.rpm_limit = 100
        self.always_rate_limit = always_rate_limit
        self.kind = kind

    def embed(self, texts):
        if self.always_rate_limit:
            raise RateLimitExceeded(self.name, kind=self.kind, retry_after_seconds=9999)
        rng_vectors = []
        for t in texts:
            seed = abs(hash((self.name, t))) % (2**32)
            rng = np.random.default_rng(seed)
            rng_vectors.append(rng.normal(size=self.dimension).tolist())
        return EmbeddingResult(vectors=rng_vectors, provider=self.name, model=self.model, dimension=self.dimension)


@pytest.fixture
def tmp_checkpoint_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_full_ingestion_with_forced_fallback_and_alignment(tmp_checkpoint_dir):
    primary = DeterministicFakeProvider("primary", always_rate_limit=True)
    backup = DeterministicFakeProvider("backup", always_rate_limit=False)

    router = ProviderRouter([primary, backup])
    chunker = FixedSizeChunker(chunk_size=50, overlap=0)
    orchestrator = IngestionOrchestrator(
        router=router, chunker=chunker, aligner=AlignmentEngine(),
        checkpoint_dir=tmp_checkpoint_dir,
    )

    text = "Sentence one here. " * 20
    result = orchestrator.ingest_document("doc1", text)

    assert result.canonical_provider == "backup"  # primary always fails, so backup becomes canonical
    assert len(result.vectors) == len(result.metadata)
    assert all(m.provider == "backup" for m in result.metadata)
    # No actual fallback occurred here since primary never succeeds even once;
    # canonical pinning happens on first *successful* embed.
    assert result.provider_switches == 0


def test_resume_validation_fails_loudly_when_canonical_provider_removed(tmp_checkpoint_dir):
    provider_a = DeterministicFakeProvider("provider_a")
    router = ProviderRouter([provider_a])
    chunker = FixedSizeChunker(chunk_size=50, overlap=0)
    orchestrator = IngestionOrchestrator(
        router=router, chunker=chunker, checkpoint_dir=tmp_checkpoint_dir,
    )

    text = "Some content. " * 20
    orchestrator.ingest_document("doc2", text)

    # Simulate resuming with a DIFFERENT provider config that no longer
    # includes provider_a.
    provider_b = DeterministicFakeProvider("provider_b")
    router2 = ProviderRouter([provider_b])
    orchestrator2 = IngestionOrchestrator(
        router=router2, chunker=chunker, checkpoint_dir=tmp_checkpoint_dir,
    )

    with pytest.raises(ResumeValidationError):
        orchestrator2.ingest_document("doc2", text)


def test_resume_with_changed_model_string_warns_and_rederives(tmp_checkpoint_dir):
    provider_v1 = DeterministicFakeProvider("provider_a", model="fake-v1")
    router = ProviderRouter([provider_v1])
    chunker = FixedSizeChunker(chunk_size=1000, overlap=0)  # 1 chunk only, forces incomplete-looking state
    orchestrator = IngestionOrchestrator(router=router, chunker=chunker, checkpoint_dir=tmp_checkpoint_dir)

    text = "short doc"
    orchestrator.ingest_document("doc3", text)

    # Manually rewrite the checkpoint to simulate a partially-completed doc
    # under an old model string.
    checkpoint_path = Path(tmp_checkpoint_dir) / "doc3.ingestion_state.json"
    import json
    state = json.loads(checkpoint_path.read_text())
    state["completed_chunk_ids"] = []  # pretend nothing finished yet
    checkpoint_path.write_text(json.dumps(state))

    provider_v2 = DeterministicFakeProvider("provider_a", model="fake-v2")
    router2 = ProviderRouter([provider_v2])
    orchestrator2 = IngestionOrchestrator(router=router2, chunker=chunker, checkpoint_dir=tmp_checkpoint_dir)

    result = orchestrator2.ingest_document("doc3", text)
    assert any("model" in w.lower() for w in result.resume_warnings)
    assert result.canonical_model == "fake-v2"
