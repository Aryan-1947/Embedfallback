"""Common interface every embedding provider adapter must implement.

The goal of this layer is to normalize wildly different provider SDKs and
error formats into one shape (`EmbeddingResult`) and one error type
(`RateLimitExceeded`) so the rest of the pipeline never has to know which
provider it's talking to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class RateLimitKind(Enum):
    """Why a provider rejected a request. This distinction is the whole
    point of RateLimitExceeded -- a per-minute limit and a daily quota
    look identical at the HTTP layer (usually a 429) but call for very
    different cooldown behavior.
    """

    SHORT_COOLDOWN = "short_cooldown"    # e.g. per-minute RPM limit -- retry soon
    QUOTA_EXHAUSTED = "quota_exhausted"  # e.g. daily/monthly cap -- retry much later
    UNKNOWN = "unknown"                  # provider gave no reliable signal -- guess conservatively


class RateLimitExceeded(Exception):
    """Raised by any EmbeddingProvider.embed() when the provider rejects
    the request for rate/quota reasons. Adapters are responsible for
    classifying `kind` from the provider's actual error payload rather
    than assuming every 429 is the same thing.
    """

    def __init__(
        self,
        provider: str,
        kind: RateLimitKind = RateLimitKind.UNKNOWN,
        retry_after_seconds: float | None = None,
        message: str | None = None,
    ):
        self.provider = provider
        self.kind = kind
        self.retry_after_seconds = retry_after_seconds  # None = provider gave no number
        self.message = message or f"{provider} rate limit exceeded ({kind.value})"
        super().__init__(self.message)


@dataclass
class EmbeddingResult:
    vectors: list[list[float]]
    provider: str
    model: str
    dimension: int
    # Per-vector token/char counts are occasionally useful for cost/rate
    # accounting upstream; optional so adapters that don't have it don't
    # need to fabricate a value.
    input_count: int = field(default=0)

    def __post_init__(self):
        if self.input_count == 0:
            self.input_count = len(self.vectors)


class EmbeddingProvider(ABC):
    """Every concrete adapter (Google, OpenAI, Cohere, Local, ...) implements
    this interface. Instances are expected to be cheap to construct and to
    lazy-load any heavy resources (e.g. local model weights) on first use.
    """

    name: str
    model: str
    dimension: int
    rpm_limit: int | None  # None = unlimited (e.g. local models)

    @abstractmethod
    def embed(self, texts: list[str]) -> EmbeddingResult:
        """Embed a batch of texts.

        Must raise RateLimitExceeded on provider-specific rate limit errors,
        classifying `kind` from the provider's actual error payload/status
        rather than assuming -- e.g. distinguish Google's per-minute 429 from
        a daily-quota 429 using the error reason/message body, not just the
        HTTP status code.

        Any other provider error (auth, malformed input, network) should be
        allowed to propagate as-is; this layer only normalizes rate limits.
        """
        raise NotImplementedError

    def max_batch_size(self) -> int:
        """Override if the provider enforces a hard per-request batch cap.
        Default is a conservative value that works for most APIs.
        """
        return 100

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}(model={self.model!r}, rpm_limit={self.rpm_limit!r})"
