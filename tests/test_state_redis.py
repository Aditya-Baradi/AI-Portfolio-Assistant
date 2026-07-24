"""Integration checks for the real Redis state backend.

The normal test job skips these. CI's Redis job sets REDIS_TEST_URL and runs
this file against an actual Redis service so serialization, transactions,
expiry, and concurrent hit accounting are all exercised end to end.
"""
from __future__ import annotations

import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from api.state import RedisBackend


REDIS_TEST_URL = os.getenv("REDIS_TEST_URL", "").strip()
pytestmark = pytest.mark.skipif(not REDIS_TEST_URL, reason="REDIS_TEST_URL is not set")


@pytest.fixture
def backend():
    instance = RedisBackend(REDIS_TEST_URL)
    yield instance
    instance.close()


def _key(label: str) -> str:
    return f"evergreen:test:{label}:{uuid.uuid4().hex}"


def test_round_trip_and_delete(backend):
    key = _key("value")
    backend.set(key, {"user": 7, "verified": True}, 30)
    assert backend.get(key) == {"user": 7, "verified": True}
    backend.delete(key)
    assert backend.get(key) is None


def test_concurrent_hits_are_not_lost(backend):
    key = _key("hits")
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            counts = list(pool.map(lambda _index: backend.hit(key, 60), range(100)))
        assert sorted(counts) == list(range(1, 101))
        assert backend.count(key, 60) == 100
    finally:
        backend.clear(key)


def test_one_hot_key_is_capped(backend, monkeypatch):
    key = _key("bounded")
    monkeypatch.setattr(backend, "MAX_EVENTS_PER_KEY", 100)
    try:
        for _ in range(500):
            backend.hit(key, 300)
        assert backend.count(key, 300) == 100
    finally:
        backend.clear(key)


def test_rejects_non_positive_expiry(backend):
    with pytest.raises(ValueError, match="positive"):
        backend.hit(_key("window"), 0)
    with pytest.raises(ValueError, match="positive"):
        backend.set(_key("ttl"), "value", 0)
