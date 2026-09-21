"""Voyage AI embedding provider adapter.

Voyage returns 429s for both RPM (requests/minute) and TPM (tokens/minute)
overruns -- their docs don't expose a separate "daily quota" concept the
way Google does, so any 429 here is treated as a short cooldown. Free-trial
accounts without a payment method are capped at 3 RPM / 10K TPM, which is
much stricter than a funded account -- expect frequent short waits on a
fresh free key.
"""

from __future__ import annotations

import requests

from .base import EmbeddingProvider, EmbeddingResult, RateLimitExceeded, RateLimitKind

_API_URL = "https://api.voyageai.com/v1/embeddings"


class VoyageEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "voyage-3.5",
        dimension: int = 1024,
        rpm_limit: int | None = 3,  # conservative default matching free-trial RPM
        input_type: str | None = "document",
        timeout: float = 30.0,
    ):
        self.name = "voyage"
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.rpm_limit = rpm_limit
        self.input_type = input_type
        self.timeout = timeout

    def max_batch_size(self) -> int:
        return 128  # matches Voyage's documented recommended batch size

    def embed(self, texts: list[str]) -> EmbeddingResult:
        payload = {"input": texts, "model": self.model}
        if self.input_type:
            payload["input_type"] = self.input_type

        resp = requests.post(
            _API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )

        if resp.status_code == 429:
            raise self._classify_429(resp)

        resp.raise_for_status()
        data = resp.json()
        ordered = sorted(data["data"], key=lambda e: e["index"])
        vectors = [e["embedding"] for e in ordered]
        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=self.model,
            dimension=len(vectors[0]) if vectors else self.dimension,
        )

    def _classify_429(self, resp: "requests.Response") -> RateLimitExceeded:
        retry_after = resp.headers.get("Retry-After")
        retry_after = float(retry_after) if retry_after else None
        try:
            body = resp.json()
            message = (body.get("detail") or body.get("error") or "").lower()
        except ValueError:
            message = ""
        # Voyage's docs describe only RPM/TPM (per-minute) limits, no
        # separate daily/monthly cap concept -- treat all 429s as short
        # cooldowns, defaulting to 60s if no explicit Retry-After is given.
        return RateLimitExceeded(
            self.name,
            kind=RateLimitKind.SHORT_COOLDOWN,
            retry_after_seconds=retry_after,
            message=f"Voyage rate limit (429): {message or 'no detail provided'}",
        )