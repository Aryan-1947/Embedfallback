"""Cohere embedding provider adapter.

Cohere's v1 API returns 429s for rate limiting and generally includes a
`Retry-After` (or `retry-after`) header. Trial-key monthly call caps surface
as a 429 too, with the message mentioning "trial key" / "monthly limit" --
that's a QUOTA_EXHAUSTED situation (no point retrying within the hour), so
we check the message body the same way as the other adapters rather than
trusting the header blindly.
"""

from __future__ import annotations

import requests

from .base import EmbeddingProvider, EmbeddingResult, RateLimitExceeded, RateLimitKind

_API_URL = "https://api.cohere.com/v1/embed"

_QUOTA_HINTS = ("trial key", "monthly limit", "monthly quota")


class CohereEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "embed-english-v3.0",
        dimension: int = 1024,
        rpm_limit: int | None = 100,
        input_type: str = "search_document",
        timeout: float = 30.0,
    ):
        self.name = "cohere"
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.rpm_limit = rpm_limit
        self.input_type = input_type
        self.timeout = timeout

    def max_batch_size(self) -> int:
        return 96  # Cohere's documented max texts per /embed call

    def embed(self, texts: list[str]) -> EmbeddingResult:
        resp = requests.post(
            _API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "texts": texts,
                "input_type": self.input_type,
            },
            timeout=self.timeout,
        )

        if resp.status_code == 429:
            raise self._classify_429(resp)

        resp.raise_for_status()
        data = resp.json()
        vectors = data["embeddings"]
        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=self.model,
            dimension=self.dimension,
        )

    def _classify_429(self, resp: "requests.Response") -> RateLimitExceeded:
        retry_after = self._parse_retry_after_header(resp)

        try:
            body = resp.json()
            message = (body.get("message") or "").lower()
        except ValueError:
            message = ""

        if any(hint in message for hint in _QUOTA_HINTS):
            return RateLimitExceeded(
                self.name,
                kind=RateLimitKind.QUOTA_EXHAUSTED,
                retry_after_seconds=retry_after,
                message=f"Cohere quota limit: {message}",
            )

        if retry_after is not None:
            return RateLimitExceeded(
                self.name,
                kind=RateLimitKind.SHORT_COOLDOWN,
                retry_after_seconds=retry_after,
                message=f"Cohere rate limit: {message}",
            )

        return RateLimitExceeded(
            self.name,
            kind=RateLimitKind.UNKNOWN,
            retry_after_seconds=None,
            message=f"Cohere 429 with no Retry-After header: {message}",
        )

    @staticmethod
    def _parse_retry_after_header(resp: "requests.Response") -> float | None:
        val = resp.headers.get("Retry-After") or resp.headers.get("retry-after")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None
