"""
The shared state backend: sliding-window rate limiting, TTL values, and the
bounds that stop a flood of distinct keys exhausting memory.
"""
from __future__ import annotations

import time

import pytest

from api import state


@pytest.fixture(autouse=True)
def clean_backend(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    state.reset_backend()
    yield
    state.reset_backend()


class TestSlidingWindow:
    def test_counts_within_window(self):
        b = state.MemoryBackend()
        assert [b.hit("k", 60) for _ in range(3)] == [1, 2, 3]

    def test_old_events_fall_out(self):
        b = state.MemoryBackend()
        b.hit("k", 0.05)
        b.hit("k", 0.05)
        time.sleep(0.08)
        assert b.hit("k", 0.05) == 1  # both earlier hits aged out

    def test_count_does_not_record(self):
        b = state.MemoryBackend()
        b.hit("k", 60)
        assert b.count("k", 60) == 1
        assert b.count("k", 60) == 1  # still 1 — counting is not a hit

    def test_keys_are_independent(self):
        b = state.MemoryBackend()
        b.hit("a", 60)
        b.hit("a", 60)
        assert b.hit("b", 60) == 1

    def test_clear(self):
        b = state.MemoryBackend()
        b.hit("k", 60)
        b.clear("k")
        assert b.count("k", 60) == 0


class TestValues:
    def test_set_get_delete(self):
        b = state.MemoryBackend()
        b.set("k", {"user": 7}, 60)
        assert b.get("k") == {"user": 7}
        b.delete("k")
        assert b.get("k") is None

    def test_expiry(self):
        b = state.MemoryBackend()
        b.set("k", "v", 0.05)
        time.sleep(0.08)
        assert b.get("k") is None

    def test_missing_key_is_none(self):
        assert state.MemoryBackend().get("nope") is None


class TestBounds:
    def test_event_keys_are_bounded(self, monkeypatch):
        """A spoofed-IP flood must not grow the map without limit."""
        b = state.MemoryBackend()
        monkeypatch.setattr(b, "MAX_KEYS", 100)
        for i in range(500):
            b.hit(f"ip-{i}", 60)
        assert len(b._events) <= 100

    def test_flooding_does_not_wipe_all_counters(self, monkeypatch):
        """
        Pruning drops the COLDEST half rather than clearing everything, so an
        attacker can't reset an active limiter by flooding new keys.
        """
        b = state.MemoryBackend()
        monkeypatch.setattr(b, "MAX_KEYS", 50)
        for _ in range(5):
            b.hit("victim", 60)
        for i in range(200):
            b.hit(f"flood-{i}", 60)
        assert len(b._events) > 0

    def test_one_hot_key_has_bounded_event_history(self, monkeypatch):
        b = state.MemoryBackend()
        monkeypatch.setattr(b, "MAX_EVENTS_PER_KEY", 100)
        for _ in range(10_000):
            b.hit("one-hot-ip", 300)
        assert len(b._events["one-hot-ip"]) == 100
        assert b.count("one-hot-ip", 300) == 100


class TestHelpers:
    def test_rate_limit_triggers_past_the_limit(self):
        for _ in range(3):
            assert not state.rate_limit_exceeded("t", 3, 60)
        assert state.rate_limit_exceeded("t", 3, 60)  # the 4th is over

    def test_failure_tracking_and_clear(self):
        assert state.record_failure("acct", 60) == 1
        assert state.record_failure("acct", 60) == 2
        assert state.failure_count("acct", 60) == 2
        state.clear_failures("acct")
        assert state.failure_count("acct", 60) == 0


class TestBackendSelection:
    def test_defaults_to_memory(self):
        assert isinstance(state.get_backend(), state.MemoryBackend)
        assert state.using_shared_store() is False

    def test_unreachable_redis_degrades_instead_of_crashing(self, monkeypatch):
        """
        A Redis blip must not take the whole app down; it degrades to in-memory
        and logs loudly. Failing closed here would be worse than the downgrade.
        """
        monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")  # nothing listening
        state.reset_backend()
        assert isinstance(state.get_backend(), state.MemoryBackend)
