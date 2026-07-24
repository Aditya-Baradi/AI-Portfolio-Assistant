"""
Shared, short-lived state: rate-limit windows, login lockouts, pending 2FA.

Why this module exists
----------------------
All of this state used to be plain dicts in `backend.py`, which is what pinned
the app to a single uvicorn worker: with >1 process, rate limiting fragments,
lockouts stop being global, and an in-progress 2FA login lands on a worker that
has never heard of it.

Why not slowapi / fastapi-limiter
---------------------------------
Both are route decorators whose limits are fixed at import time, and neither
gives you a general key/value store. We need three different things backed by
the SAME shared store (sliding-window counters, lockout counters, and pending
2FA handles), so a small backend interface is a better fit than bolting a
route-limiter library onto one of the three.

Backends
--------
memory  (default) - process-local. Correct for a single worker; the app warns
                    at startup if it is used in production.
redis             - set REDIS_URL. Sliding windows use a ZSET manipulated in a
                    single pipeline, so counting is atomic across workers.

The interface is deliberately tiny: sliding-window counting, and get/set/delete
with a TTL.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid

logger = logging.getLogger("evergreen.state")


class StateBackend:
    """Interface for the shared short-lived state store."""

    # Well above the largest configured application limit (120/minute), but
    # finite so one hot key cannot consume unbounded memory during its window.
    MAX_EVENTS_PER_KEY = 1_000

    def hit(self, key: str, window_s: float) -> int:
        """Record an event against `key` and return how many fall inside the window."""
        raise NotImplementedError

    def count(self, key: str, window_s: float) -> int:
        """Events inside the window WITHOUT recording a new one."""
        raise NotImplementedError

    def clear(self, key: str) -> None:
        raise NotImplementedError

    def get(self, key: str):
        raise NotImplementedError

    def set(self, key: str, value, ttl_s: float) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def healthy(self) -> bool:
        """Whether the backing store is currently reachable."""
        return True

    def close(self) -> None:
        """Release backend resources."""


class MemoryBackend(StateBackend):
    """
    Process-local backend. Thread-safe, and bounded so a flood of distinct keys
    (e.g. spoofed IPs) can't grow it without limit.
    """

    MAX_KEYS = 50_000

    def __init__(self) -> None:
        self._events: dict[str, list[float]] = {}
        self._values: dict[str, tuple[object, float]] = {}
        self._lock = threading.Lock()

    def _prune_locked(self, now: float) -> None:
        for k, (_v, exp) in list(self._values.items()):
            if exp < now:
                self._values.pop(k, None)
        if len(self._events) > self.MAX_KEYS:
            # Drop the coldest half rather than clearing everything, so an
            # attacker can't wipe everyone's counters by flooding new keys.
            ordered = sorted(self._events.items(), key=lambda kv: max(kv[1], default=0.0))
            for k, _ in ordered[: len(ordered) // 2]:
                self._events.pop(k, None)
        if len(self._values) > self.MAX_KEYS:
            ordered_v = sorted(self._values.items(), key=lambda kv: kv[1][1])
            for k, _ in ordered_v[: len(ordered_v) // 2]:
                self._values.pop(k, None)

    def hit(self, key: str, window_s: float) -> int:
        now = time.time()
        with self._lock:
            recent = [t for t in self._events.get(key, []) if now - t < window_s]
            recent.append(now)
            if len(recent) > self.MAX_EVENTS_PER_KEY:
                recent = recent[-self.MAX_EVENTS_PER_KEY:]
            self._events[key] = recent
            self._prune_locked(now)
            return len(recent)

    def count(self, key: str, window_s: float) -> int:
        now = time.time()
        with self._lock:
            recent = [t for t in self._events.get(key, []) if now - t < window_s]
            if recent:
                self._events[key] = recent
            else:
                self._events.pop(key, None)
            return len(recent)

    def clear(self, key: str) -> None:
        with self._lock:
            self._events.pop(key, None)

    def get(self, key: str):
        now = time.time()
        with self._lock:
            entry = self._values.get(key)
            if not entry:
                return None
            value, expires = entry
            if expires < now:
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: str, value, ttl_s: float) -> None:
        now = time.time()
        with self._lock:
            self._values[key] = (value, now + ttl_s)
            self._prune_locked(now)

    def delete(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)


class RedisBackend(StateBackend):
    """
    Redis-backed sliding window using a sorted set per key.

    Each event is a member scored by its timestamp. One pipeline does:
    drop everything older than the window, add the new event, count, re-arm the
    TTL — so concurrent workers can't race the count.
    """

    def __init__(self, url: str) -> None:
        import redis  # imported lazily so redis is an optional dependency

        timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "2.0"))
        if not 0.1 <= timeout <= 30:
            raise ValueError("REDIS_SOCKET_TIMEOUT must be between 0.1 and 30 seconds")
        self._r = redis.Redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=timeout,
            socket_timeout=timeout,
            health_check_interval=30,
            client_name="evergreen-state",
        )
        self._r.ping()  # fail fast at startup rather than on first request

    def hit(self, key: str, window_s: float) -> int:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        now = time.time()
        # A timestamp+PID can collide when concurrent requests land in the same
        # microsecond. A random suffix makes every recorded hit a unique ZSET
        # member, including across threads and workers.
        member = f"{time.time_ns()}:{os.getpid()}:{uuid.uuid4().hex}"
        pipe = self._r.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, now - window_s)
        pipe.zadd(key, {member: now})
        # Keep the newest N members. With <=N members the negative stop rank is
        # outside the set and Redis removes nothing.
        pipe.zremrangebyrank(key, 0, -(self.MAX_EVENTS_PER_KEY + 1))
        pipe.zcard(key)
        pipe.expire(key, max(1, math.ceil(window_s)))
        return int(pipe.execute()[3])

    def count(self, key: str, window_s: float) -> int:
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        now = time.time()
        pipe = self._r.pipeline(transaction=True)
        pipe.zremrangebyscore(key, 0, now - window_s)
        pipe.zcard(key)
        return int(pipe.execute()[1])

    def clear(self, key: str) -> None:
        self._r.delete(key)

    def get(self, key: str):
        raw = self._r.get(f"v:{key}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return raw

    def set(self, key: str, value, ttl_s: float) -> None:
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        self._r.setex(f"v:{key}", max(1, math.ceil(ttl_s)), json.dumps(value))

    def delete(self, key: str) -> None:
        self._r.delete(f"v:{key}")

    def healthy(self) -> bool:
        try:
            return bool(self._r.ping())
        except Exception:
            return False

    def close(self) -> None:
        self._r.close()


_backend: StateBackend | None = None
_backend_lock = threading.Lock()


def get_backend() -> StateBackend:
    """The configured backend (Redis when REDIS_URL is set, else in-memory)."""
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is not None:
                return _backend
            url = os.getenv("REDIS_URL", "").strip()
            if url:
                try:
                    _backend = RedisBackend(url)
                    logger.info("Shared state backend: redis")
                except Exception as e:
                    # Production's startup audit treats this downgrade as fatal.
                    # Development stays available but gets an explicit error.
                    logger.error(
                        "REDIS_URL is set but unusable (%s); using in-memory state. "
                        "Rate limits and lockouts are per-process until this is fixed.",
                        e,
                    )
                    _backend = MemoryBackend()
            else:
                _backend = MemoryBackend()
    return _backend


def reset_backend() -> None:
    """Drop the cached backend (tests, and after config changes)."""
    global _backend
    with _backend_lock:
        previous, _backend = _backend, None
    if previous is not None:
        previous.close()


def using_shared_store() -> bool:
    backend = get_backend()
    return isinstance(backend, RedisBackend) and backend.healthy()


# ---------------------------------------------------------------------------
# Rate limiting + lockout helpers built on the backend
# ---------------------------------------------------------------------------

def rate_limit_exceeded(key: str, limit: int, window_s: float) -> bool:
    """Record a hit against `key`; True when it has now exceeded `limit`."""
    return get_backend().hit(key, window_s) > limit


def record_failure(key: str, window_s: float) -> int:
    """Count a failed attempt (e.g. bad password) and return the running total."""
    return get_backend().hit(key, window_s)


def failure_count(key: str, window_s: float) -> int:
    return get_backend().count(key, window_s)


def clear_failures(key: str) -> None:
    """Called after a successful login so a user isn't punished for old typos."""
    get_backend().clear(key)
