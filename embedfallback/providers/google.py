"""Google (Gemini) embedding provider adapter.

Google's embedding endpoint returns 429s for both per-minute RPM overruns
and daily/free-tier quota exhaustion. The HTTP status code alone can't tell
them apart -- the distinguishing signal lives in the error body, under
`error.status` ("RESOURCE_EXHAUSTED") and, more usefully, in
`error.details[].violations[].quotaId` (or the free-text `error.message`),
which typically names something like "GenerateRequestsPerMinutePerProject"
(a per-minute limit) vs. "GenerateRequestsPerDayPerProject" or
"FreeTier...PerDay" (a daily cap).

This adapter tries to read that structured detail first, and falls back to
substring matching on the message if the response body doesn't include the
structured `details` field (older API versions, proxies, etc), and also
parses a "please retry in Ns" pattern from the message body when no
Retry-After header is present.
"""

from __future__ import annotations

import json
import re

import requests

from .base import EmbeddingProvider, EmbeddingResult, RateLimitExceeded, RateLimitKind

_API_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:embedContent"
)
_BATCH_API_URL_TMPL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents"
)

# Substrings observed in Google's quota-violation identifiers/messages that
# indicate a *daily/longer* cap rather than a per-minute one.
# NOTE: "free_tier"/"freetier" is intentionally NOT included here -- that
# substring describes the *access level* (free vs paid), not the *reset
# period*. Google's embed_content_free_tier_requests metric is actually a
# per-minute limit despite the name.
_DAILY_QUOTA_HINTS = ("perday", "per_day", "daily")
_MINUTE_QUOTA_HINTS = ("perminute", "per_minute", "rpm")


class GoogleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-embedding-001",
        dimension: int = 3072,
        rpm_limit: int | None = 100,
        timeout: float = 60.0,
    ):
        self.name = "google"
        self.api_key = api_key
        self.model = model
        self.dimension = dimension
        self.rpm_limit = rpm_limit
        self.timeout = timeout

    def max_batch_size(self) -> int:
        return 100  # Google's batchEmbedContents cap as of current API docs

    def embed(self, texts: list[str]) -> EmbeddingResult:
        url = _BATCH_API_URL_TMPL.format(model=self.model)
        requests_payload = {
            "requests": [
                {
                    "model": f"models/{self.model}",
                    "content": {"parts": [{"text": t}]},
                }
                for t in texts
            ]
        }
        resp = requests.post(
            url,
            params={"key": self.api_key},
            json=requests_payload,
            timeout=self.timeout,
        )

        if resp.status_code == 429:
            raise self._classify_429(resp)

        resp.raise_for_status()
        data = resp.json()
        vectors = [e["values"] for e in data.get("embeddings", [])]
        return EmbeddingResult(
            vectors=vectors,
            provider=self.name,
            model=self.model,
            dimension=self.dimension,
        )

    def _classify_429(self, resp: "requests.Response") -> RateLimitExceeded:
        retry_after = self._parse_retry_after_header(resp)
        if retry_after is None:
            retry_after = self._parse_retry_after_from_body(resp)

        try:
            body = resp.json()
        except (json.JSONDecodeError, ValueError):
            return RateLimitExceeded(
                self.name, kind=RateLimitKind.UNKNOWN, retry_after_seconds=retry_after,
                message="Google 429 with unparseable body; cannot classify.",
            )

        error = body.get("error", {})
        message = (error.get("message") or "").lower()

        quota_ids = []
        for detail in error.get("details", []):
            for violation in detail.get("violations", []) or []:
                qid = violation.get("quotaId", "")
                if qid:
                    quota_ids.append(qid.lower())

        haystack = " ".join(quota_ids) + " " + message

        if any(hint in haystack for hint in _DAILY_QUOTA_HINTS):
            return RateLimitExceeded(
                self.name,
                kind=RateLimitKind.QUOTA_EXHAUSTED,
                retry_after_seconds=retry_after,
                message=f"Google daily/quota limit hit: {message or quota_ids}",
            )
        if any(hint in haystack for hint in _MINUTE_QUOTA_HINTS):
            return RateLimitExceeded(
                self.name,
                kind=RateLimitKind.SHORT_COOLDOWN,
                retry_after_seconds=retry_after,
                message=f"Google per-minute limit hit: {message or quota_ids}",
            )

        # No recognized daily/minute hint in the identifier or message. If
        # we DO have a concrete, short retry_after value, trust that over
        # guessing -- a real numeric wait time is stronger evidence of a
        # short cooldown than the absence of a keyword match.
        if retry_after is not None and retry_after <= 300:
            return RateLimitExceeded(
                self.name,
                kind=RateLimitKind.SHORT_COOLDOWN,
                retry_after_seconds=retry_after,
                message=f"Google 429 with explicit short retry_after: {message}",
            )

        return RateLimitExceeded(
            self.name,
            kind=RateLimitKind.UNKNOWN,
            retry_after_seconds=retry_after,
            message=f"Google 429 without a recognized quota identifier: {message}",
        )

    @staticmethod
    def _parse_retry_after_header(resp: "requests.Response") -> float | None:
        val = resp.headers.get("Retry-After")
        if val is None:
            return None
        try:
            return float(val)
        except ValueError:
            return None

    @staticmethod
    def _parse_retry_after_from_body(resp: "requests.Response") -> float | None:
        """Google sometimes puts the wait time in the message body instead
        of a Retry-After header, e.g. 'please retry in 47.77152523s.' --
        extract that number when the header isn't present.
        """
        try:
            text = resp.text
        except Exception:
            return None
        match = re.search(r"retry in (\d+(?:\.\d+)?)s", text)
        if match:
            return float(match.group(1))
        return None