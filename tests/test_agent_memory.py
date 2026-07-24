"""
Agent conversation memory.

The bug this guards against: the old implementation kept a module-global dict
of ChatMessageHistory objects that was never evicted, so it grew by one entry
per (user, chat) for the life of the process. History now comes from SQLite
with a bounded LRU in front of it.
"""
from __future__ import annotations

import pytest

from api import langchain_agent as agent


@pytest.fixture(autouse=True)
def clean_cache():
    agent._history_cache.clear()
    yield
    agent._history_cache.clear()


@pytest.fixture
def db(tmp_path, monkeypatch):
    import api.db as db_mod
    from api import crypto

    monkeypatch.setattr(db_mod, "DB_PATH", tmp_path / "mem.db")
    monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
    crypto.reset_cache()
    db_mod.init_db()
    yield db_mod
    crypto.reset_cache()


class TestCacheIsBounded:
    def test_cache_does_not_grow_without_limit(self, db, monkeypatch):
        monkeypatch.setattr(agent, "MAX_CACHED_SESSIONS", 10)
        for i in range(200):
            agent._get_history(f"u1.c{i}")
        assert agent.cached_session_count() <= 10

    def test_eviction_is_least_recently_used(self, db, monkeypatch):
        monkeypatch.setattr(agent, "MAX_CACHED_SESSIONS", 3)
        for sid in ("u1.ca", "u1.cb", "u1.cc"):
            agent._get_history(sid)

        agent._get_history("u1.ca")   # touch the oldest, making it newest
        agent._get_history("u1.cd")   # forces one eviction

        assert "u1.ca" in agent._history_cache   # kept: recently used
        assert "u1.cb" not in agent._history_cache  # evicted: coldest


class TestHistoryComesFromTheDatabase:
    def test_reads_persisted_messages(self, db):
        uid = db.register_user("mem@example.com", "password123")
        db.touch_chat(uid, "chat1", first_message="hello")
        db.add_message("chat1", "user", "what is my volatility?")
        db.add_message("chat1", "assistant", "About 18% a year.")

        history = agent._get_history(f"u{uid}.cchat1")
        assert [m.content for m in history] == [
            "what is my volatility?", "About 18% a year."]

    def test_survives_a_cold_cache(self, db):
        """A restart must not lose history — SQLite is the durable record."""
        uid = db.register_user("cold@example.com", "password123")
        db.touch_chat(uid, "c9", first_message="hi")
        db.add_message("c9", "user", "remember this")

        agent._history_cache.clear()  # simulate a fresh process
        assert agent._get_history(f"u{uid}.cc9")[0].content == "remember this"

    def test_history_is_truncated_to_the_window(self, db):
        uid = db.register_user("long@example.com", "password123")
        db.touch_chat(uid, "clong", first_message="start")
        for i in range(50):
            db.add_message("clong", "user", f"msg {i}")

        assert len(agent._get_history(f"u{uid}.cclong")) <= agent.HISTORY_TURNS

    def test_unparseable_session_id_yields_empty_history(self, db):
        assert agent._get_history("not-a-session-id") == []


class TestForgetting:
    def test_forget_session(self, db):
        agent._get_history("u1.cx")
        agent.forget_session("u1.cx")
        assert "u1.cx" not in agent._history_cache

    def test_forget_user_sessions_removes_only_that_user(self, db):
        for sid in ("u1.ca", "u1.cb", "u2.ca"):
            agent._get_history(sid)

        removed = agent.forget_user_sessions(1)
        assert removed == 2
        assert "u2.ca" in agent._history_cache
        assert not any(k.startswith("u1.") for k in agent._history_cache)

    def test_forget_user_does_not_match_a_prefix_collision(self, db):
        """User 1 and user 11 are different people."""
        for sid in ("u1.ca", "u11.ca"):
            agent._get_history(sid)
        agent.forget_user_sessions(1)
        assert "u11.ca" in agent._history_cache


class TestSessionIdSanitising:
    @pytest.mark.parametrize("raw,expected_absent", [
        ("../../etc/passwd", "/"),
        ("..\\..\\windows", "\\"),
        ("a/b/c", "/"),
    ])
    def test_path_separators_are_stripped(self, raw, expected_absent):
        assert expected_absent not in agent._sanitize_session_id(raw)

    def test_empty_falls_back(self):
        assert agent._sanitize_session_id("") == "anonymous"
        assert agent._sanitize_session_id(None) == "anonymous"

    def test_normal_session_id_is_preserved(self):
        assert agent._sanitize_session_id("u12.cabc-123") == "u12.cabc-123"

    def test_unsafe_ids_cannot_collapse_to_the_same_cache_key(self):
        assert agent._sanitize_session_id("u1.ca/b") != agent._sanitize_session_id("u1.cab")


class TestRemember:
    def test_turn_is_appended_to_a_cached_session(self, db):
        uid = db.register_user("app@example.com", "password123")
        db.touch_chat(uid, "c1", first_message="hi")
        sid = f"u{uid}.cc1"
        agent._get_history(sid)

        agent._remember(sid, "question", "answer")
        contents = [m.content for m in agent._history_cache[sid]]
        assert contents[-2:] == ["question", "answer"]

    def test_cached_history_stays_bounded_across_many_turns(self, db):
        uid = db.register_user("chatty@example.com", "password123")
        db.touch_chat(uid, "c1", first_message="hi")
        sid = f"u{uid}.cc1"
        agent._get_history(sid)
        for i in range(100):
            agent._remember(sid, f"q{i}", f"a{i}")
        assert len(agent._history_cache[sid]) <= agent.HISTORY_TURNS
