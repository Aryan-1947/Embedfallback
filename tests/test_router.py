import time

import pytest

from embedfallback.providers.base import EmbeddingProvider, EmbeddingResult, RateLimitKind
from embedfallback.router import NoProviderAvailableError, ProviderRouter


class FakeProvider(EmbeddingProvider):
    def __init__(self, name, model="fake-model", rpm_limit=100):
        self.name = name
        self.model = model
        self.dimension = 8
        self.rpm_limit = rpm_limit

    def embed(self, texts):
        return EmbeddingResult(
            vectors=[[0.0] * self.dimension for _ in texts],
            provider=self.name, model=self.model, dimension=self.dimension,
        )


def test_get_available_provider_returns_highest_priority():
    router = ProviderRouter([FakeProvider("a"), FakeProvider("b")])
    assert router.get_available_provider().name == "a"


def test_short_cooldown_uses_retry_after_when_given():
    router = ProviderRouter([FakeProvider("a"), FakeProvider("b")])
    state = router.mark_rate_limited("a", RateLimitKind.SHORT_COOLDOWN, retry_after=5)
    assert 4.9 <= (state.available_at - time.time()) <= 5.1
    assert router.get_available_provider().name == "b"


def test_short_cooldown_default_is_60s_without_retry_after():
    router = ProviderRouter([FakeProvider("a")])
    state = router.mark_rate_limited("a", RateLimitKind.SHORT_COOLDOWN, retry_after=None)
    assert 59 <= (state.available_at - time.time()) <= 61


def test_quota_exhausted_default_is_not_60s():
    router = ProviderRouter([FakeProvider("a")], quota_exhausted_default_seconds=6 * 3600)
    state = router.mark_rate_limited("a", RateLimitKind.QUOTA_EXHAUSTED, retry_after=None)
    remaining = state.available_at - time.time()
    assert remaining > 3600  # must be hours, not the 60s short-cooldown default


def test_unknown_kind_uses_conservative_middle_ground():
    router = ProviderRouter([FakeProvider("a")])
    state = router.mark_rate_limited("a", RateLimitKind.UNKNOWN, retry_after=None)
    remaining = state.available_at - time.time()
    assert 60 < remaining < 3600  # between the short and quota defaults


def test_no_provider_available_raises_with_soonest_wait():
    router = ProviderRouter([FakeProvider("a")])
    router.mark_rate_limited("a", RateLimitKind.SHORT_COOLDOWN, retry_after=30)
    with pytest.raises(NoProviderAvailableError):
        router.get_available_provider()


def test_exclude_set_skips_providers_within_same_attempt():
    router = ProviderRouter([FakeProvider("a"), FakeProvider("b")])
    assert router.get_available_provider(exclude={"a"}).name == "b"


def test_clear_cooldown_restores_availability():
    router = ProviderRouter([FakeProvider("a")])
    router.mark_rate_limited("a", RateLimitKind.SHORT_COOLDOWN, retry_after=999)
    router.clear_cooldown("a")
    assert router.get_available_provider().name == "a"
