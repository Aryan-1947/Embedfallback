"""Hugging Face Inference API embedding provider adapter.

Calls HF's hosted feature-extraction endpoint over HTTP -- no local model
download, no local compute. Two error modes matter here:
  - 429: standard rate limiting.
  - 503 with an `estimated_time` field: the model is "cold" and being
    loaded onto HF's servers. This isn't really a rate limit, but it's
    functionally the same "come back later" signal, so we treat it as a
    SHORT_COOLDOWN using the estimated_time as retry_after when present.
"""

from __future__ import annotations

import requests

from .base import EmbeddingProvider, EmbeddingResult, RateLimitExceeded, RateLimitKind

_API_URL_TMPL = "https://router.huggingface.co/hf-inference/models/{model}/pipeline/feature-extraction"


class HuggingFaceInferenceProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        rpm_limit: int | None = None,
        timeout: float = 30.0,
    ):
        self.name = "huggingface"
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.rpm_limit = rpm_limit
        self.timeout = timeout

    def max_batch_size(self) -> int:
        return 50

    def embed(self, texts: list[str]) -> EmbeddingResult:
        url = _API_URL_TMPL.format(model=self.model)
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"inputs": texts, "options": {"wait_for_model": False}},
            timeout=self.timeout,
        )

        if resp.status_code == 429:
            raise self._classify_429(resp)

        if resp.status_code == 503:
            raise self._classify_503(resp)

        resp.raise_for_status()
        data = resp.json()
        # feature-extraction returns per-token vectors for some models;
        # mean-pool over tokens if we got a 3D list, otherwise assume it's
        # already sentence-level.
        vectors = []
        for item in data:
            if isinstance(item[0], list):
                # token-level: mean pool
                n_tokens = len(item)
                dim = len(item[0])
                pooled = [sum(tok[d] for tok in item) / n_tokens for d in range(dim)]
                vectors.append(pooled)
            else:
                vectors.append(item)

        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=self.model,
            dimension=len(vectors[0]) if vectors else self.dimension,
        )

    def _classify_429(self, resp) -> RateLimitExceeded:
        retry_after = resp.headers.get("Retry-After")
        retry_after = float(retry_after) if retry_after else None
        return RateLimitExceeded(
            self.name,
            kind=RateLimitKind.SHORT_COOLDOWN if retry_after else RateLimitKind.UNKNOWN,
            retry_after_seconds=retry_after,
            message="Hugging Face Inference API rate limit (429)",
        )

    def _classify_503(self, resp) -> RateLimitExceeded:
        try:
            body = resp.json()
            estimated_time = body.get("estimated_time")
        except ValueError:
            estimated_time = None
        return RateLimitExceeded(
            self.name,
            kind=RateLimitKind.SHORT_COOLDOWN,
            retry_after_seconds=float(estimated_time) if estimated_time else 20.0,
            message=f"Hugging Face model is loading (503): estimated_time={estimated_time}",
        )