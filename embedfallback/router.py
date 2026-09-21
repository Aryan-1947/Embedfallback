"""Decides which provider handles the next batch of chunks, tracking
cooldowns of different durations per the RateLimitKind classification.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .providers.base import EmbeddingProvider, RateLimitKind

logger = logging.getLogger("embedfallback.router")

# Default cooldown durations, in seconds, used when a provider's error
# didn't include a usable retry_after value.
_SHORT_COOLDOWN_DEFAULT = 60.0
_QUOTA_EXHAUSTED_DEFAULT = 6 * 60 * 60.0  # 6 hours; see also `next_utc_midnight`
_UNKNOWN_DEFAULT = 5 * 60.0


def _next_utc_midnight_seconds_away() -> float:
    """Seconds from now until the next UTC midnight. Used as an alternative
    QUOTA_EXHAUSTED default when the caller prefers "reset at midnight"
    semantics over a flat duration (many providers' daily quotas do reset
    at UTC midnight).
    """
    import datetime

    now = datetime.datetime.now(datetime.timezone.utc)
    tomorrow = (now + datetime.timedelta(days=1)).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    return (tomorrow - now).total_seconds()


@dataclass
class CooldownState:
    available_at: float          # unix timestamp
    kind: RateLimitKind
    reason: str = ""


class NoProviderAvailableError(Exception):
    """Raised when every configured provider is currently in cooldown."""

    def __init__(self, cooldowns: dict[str, CooldownState]):
        self.cooldowns = cooldowns
        soonest = min(cooldowns.values(), key=lambda c: c.available_at) if cooldowns else None
        wait = max(0.0, soonest.available_at - time.time()) if soonest else 0.0
        super().__init__(
            f"All {len(cooldowns)} provider(s) are in cooldown. "
            f"Soonest available in ~{wait:.0f}s."
        )


class ProviderRouter:
    def __init__(
        self,
        providers: list[EmbeddingProvider],
        quota_exhausted_default_seconds: float | None = None,
        use_utc_midnight_for_quota: bool = True,
    ):
        if not providers:
            raise ValueError("ProviderRouter requires at least one provider")
        self.providers = providers  # ordered by priority
        self._by_name = {p.name: p for p in providers}
        self.cooldowns: dict[str, CooldownState] = {}
        self._quota_default = quota_exhausted_default_seconds
        self._use_utc_midnight = use_utc_midnight_for_quota

    def get_available_provider(
        self, exclude: set[str] | None = None
    ) -> EmbeddingProvider:
        """Returns the highest-priority provider not currently in cooldown
        (and not in `exclude`, e.g. a provider that just failed for a
        non-rate-limit reason within this same batch attempt).

        Raises NoProviderAvailableError if every provider is unavailable.
        """
        exclude = exclude or set()
        now = time.time()
        for provider in self.providers:
            if provider.name in exclude:
                continue
            state = self.cooldowns.get(provider.name)
            if state is None or state.available_at <= now:
                return provider
        raise NoProviderAvailableError(dict(self.cooldowns))

    def mark_rate_limited(
        self,
        provider_name: str,
        kind: RateLimitKind,
        retry_after: float | None = None,
        reason: str = "",
    ) -> CooldownState:
        """Records a cooldown for `provider_name`. Duration is based on the
        classification, not a flat default:
          - retry_after given (any kind): use it directly.
          - SHORT_COOLDOWN, no retry_after: default 60s.
          - QUOTA_EXHAUSTED, no retry_after: default to next UTC midnight
            (or a configurable flat duration, e.g. 6h) -- NOT 60s.
          - UNKNOWN: conservative middle ground (5 min), with a warning
            logged since the guess may be wrong in either direction.
        """
        now = time.time()

        if retry_after is not None:
            duration = max(0.0, float(retry_after))
        elif kind == RateLimitKind.SHORT_COOLDOWN:
            duration = _SHORT_COOLDOWN_DEFAULT
        elif kind == RateLimitKind.QUOTA_EXHAUSTED:
            if self._quota_default is not None:
                duration = self._quota_default
            elif self._use_utc_midnight:
                duration = _next_utc_midnight_seconds_away()
            else:
                duration = _QUOTA_EXHAUSTED_DEFAULT
        else:  # UNKNOWN
            duration = _UNKNOWN_DEFAULT
            logger.warning(
                "Provider '%s' returned an unrecognized rate-limit signal "
                "(kind=UNKNOWN). Applying a conservative %.0fs cooldown, "
                "but this guess may be too long or too short. Reason: %s",
                provider_name, duration, reason,
            )

        state = CooldownState(available_at=now + duration, kind=kind, reason=reason)
        self.cooldowns[provider_name] = state
        logger.info(
            "Provider '%s' cooling down for %.0fs (kind=%s)%s",
            provider_name, duration, kind.value,
            f": {reason}" if reason else "",
        )
        return state

    def clear_cooldown(self, provider_name: str) -> None:
        """Manually clear a provider's cooldown (e.g. user override, or a
        successful health-check probe)."""
        self.cooldowns.pop(provider_name, None)

    def status(self) -> dict[str, dict]:
        """Human-readable snapshot of every provider's current availability,
        useful for the CLI / evaluation harness output."""
        now = time.time()
        out = {}
        for provider in self.providers:
            state = self.cooldowns.get(provider.name)
            if state is None or state.available_at <= now:
                out[provider.name] = {"available": True}
            else:
                out[provider.name] = {
                    "available": False,
                    "seconds_remaining": round(state.available_at - now, 1),
                    "kind": state.kind.value,
                    "reason": state.reason,
                }
        return out
