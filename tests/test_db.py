"""Tests for the SQL auth/portfolio/chat layer (offline, temp database)."""
from concurrent.futures import ThreadPoolExecutor

import pytest

import api.db as db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


class TestAuth:
    def test_database_permissions_are_owner_only_on_posix(self, monkeypatch):
        calls = []
        monkeypatch.setattr(db, "_IS_POSIX", True)
        monkeypatch.setattr(db.os, "chmod", lambda path, mode: calls.append((path, mode)))
        with db._connect():
            pass
        assert calls
        assert all(mode == 0o600 for _path, mode in calls)

    def test_register_and_login(self):
        uid = db.register_user("Alice@Example.com", "hunter2secret")
        assert db.verify_login("alice@example.com", "hunter2secret") == uid  # case-insensitive email

    def test_password_is_hashed_not_plaintext(self):
        db.register_user("bob@example.com", "supersecret123")
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
        conn.close()
        assert "supersecret123" not in stored
        assert stored.startswith("$2")  # bcrypt format

    def test_wrong_password_rejected(self):
        db.register_user("carol@example.com", "correcthorse1")
        with pytest.raises(db.AuthError):
            db.verify_login("carol@example.com", "wrongpassword")

    def test_unknown_email_same_error_as_bad_password(self):
        db.register_user("dave@example.com", "password123")
        with pytest.raises(db.AuthError) as e1:
            db.verify_login("nobody@example.com", "password123")
        with pytest.raises(db.AuthError) as e2:
            db.verify_login("dave@example.com", "nope-nope-nope")
        assert str(e1.value) == str(e2.value)  # doesn't reveal which was wrong

    def test_duplicate_email_rejected(self):
        db.register_user("eve@example.com", "password123")
        with pytest.raises(db.AuthError):
            db.register_user("EVE@example.com", "password456")

    def test_short_password_rejected(self):
        with pytest.raises(db.AuthError):
            db.register_user("frank@example.com", "short")

    def test_token_roundtrip_and_revoke(self):
        uid = db.register_user("gina@example.com", "password123")
        token = db.issue_token(uid)
        user = db.user_for_token(token)
        assert user["id"] == uid and user["email"] == "gina@example.com"
        db.revoke_token(token)
        assert db.user_for_token(token) is None

    def test_bad_token_rejected(self):
        assert db.user_for_token("not-a-real-token") is None
        assert db.user_for_token("") is None

    def test_email_is_canonicalized_and_validated(self):
        uid = db.register_user("  Test@BÜCHER.de ", "password123")
        assert db.get_user_identity(uid)["email"] == "test@xn--bcher-kva.de"
        assert db.verify_login("TEST@bücher.de", "password123") == uid
        for bad in ("missing-at", "a@localhost", ".a@example.com", "a..b@example.com"):
            with pytest.raises(db.AuthError):
                db.register_user(bad, "password123")

    def test_policy_acceptance_is_versioned_and_exportable(self):
        uid = db.register_user(
            "policy@example.com",
            "password123",
            accepted_policy_version="v1",
        )
        assert db.has_policy_acceptance(uid, "v1")
        db.record_policy_acceptance(uid, "v2")
        assert [r["policy_version"] for r in db.list_policy_acceptances(uid)] == [
            "v1",
            "v2",
        ]

    def test_rotation_is_idempotent_during_concurrent_grace(self):
        uid = db.register_user("rotate-db@example.com", "password123")
        original = db.issue_token(uid)

        with ThreadPoolExecutor(max_workers=6) as pool:
            successors = list(pool.map(lambda _i: db.rotate_token(original), range(12)))

        assert None not in successors
        assert len(set(successors)) == 1
        successor = successors[0]
        assert db.user_for_token(original) is None
        assert db.user_for_token(successor)["id"] == uid
        with db._connect() as conn:
            rows = conn.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE user_id = ?",
                (uid,),
            ).fetchone()[0]
        assert rows <= db.MAX_SESSIONS_PER_USER + 1

        # Revoking the successor revokes its short-lived predecessor family,
        # so the old token cannot resurrect the session during grace.
        db.revoke_token(successor)
        assert db.rotate_token(original) is None


class TestPortfolio:
    def test_upsert_and_get(self):
        uid = db.register_user("hank@example.com", "password123")
        assert db.get_portfolio(uid) is None
        db.save_portfolio(uid, {"holdings": [{"ticker": "AAPL"}]})
        assert db.get_portfolio(uid)["holdings"][0]["ticker"] == "AAPL"
        # Re-import replaces (update path)
        db.save_portfolio(uid, {"holdings": [{"ticker": "MSFT"}]})
        assert db.get_portfolio(uid)["holdings"][0]["ticker"] == "MSFT"

    def test_portfolios_are_per_user(self):
        u1 = db.register_user("u1@example.com", "password123")
        u2 = db.register_user("u2@example.com", "password123")
        db.save_portfolio(u1, {"holdings": [{"ticker": "NVDA"}]})
        assert db.get_portfolio(u2) is None


class TestChats:
    def test_chat_created_with_title_from_first_message(self):
        uid = db.register_user("iris@example.com", "password123")
        db.touch_chat(uid, "chat-1", "What is my portfolio   worth today?")
        chats = db.list_chats(uid)
        assert len(chats) == 1
        assert chats[0]["title"] == "What is my portfolio worth today?"

    def test_messages_roundtrip(self):
        uid = db.register_user("jack@example.com", "password123")
        db.touch_chat(uid, "chat-1", "hello")
        db.add_message("chat-1", "user", "hello")
        db.add_message("chat-1", "assistant", "hi there")
        msgs = db.get_messages("chat-1")
        assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_pruned_to_last_10(self):
        uid = db.register_user("kate@example.com", "password123")
        for i in range(13):
            db.touch_chat(uid, f"chat-{i}", f"message {i}")
        chats = db.list_chats(uid)
        assert len(chats) == db.MAX_CHATS_PER_USER == 10
        ids = {c["id"] for c in chats}
        assert "chat-12" in ids and "chat-0" not in ids  # newest kept, oldest pruned

    def test_chat_ownership(self):
        u1 = db.register_user("l1@example.com", "password123")
        u2 = db.register_user("l2@example.com", "password123")
        db.touch_chat(u1, "chat-x", "mine")
        assert db.chat_belongs_to_user(u1, "chat-x")
        assert not db.chat_belongs_to_user(u2, "chat-x")

    def test_cross_user_chat_reads_and_writes_are_rejected(self):
        owner = db.register_user("chat-owner@example.com", "password123")
        other = db.register_user("chat-other@example.com", "password123")
        db.touch_chat(owner, "shared-id", "private title")
        db.add_message(owner, "shared-id", "user", "private message")

        with pytest.raises(db.ChatOwnershipError):
            db.touch_chat(other, "shared-id", "attacker title")
        with pytest.raises(db.ChatOwnershipError):
            db.add_message(other, "shared-id", "user", "attacker message")
        with pytest.raises(db.ChatOwnershipError):
            db.get_messages_for_user(other, "shared-id")
        with pytest.raises(db.ChatOwnershipError):
            db.begin_chat_turn(other, "shared-id", "attacker turn", 50)

        assert db.get_messages_for_user(owner, "shared-id")[0]["content"] == "private message"
        assert db.count_user_messages_today(other) == 0

    def test_quota_reservation_is_atomic_and_charged_to_actor(self):
        uid = db.register_user("quota@example.com", "password123")
        assert db.begin_chat_turn(uid, "quota-chat", "one", 2) == 1
        assert db.begin_chat_turn(uid, "quota-chat", "two", 2) == 2
        with pytest.raises(db.ChatQuotaError):
            db.begin_chat_turn(uid, "quota-chat", "three", 2)
        assert db.count_user_messages_today(uid) == 2

    def test_chat_content_is_encrypted_at_rest(self):
        from api import crypto

        uid = db.register_user("opaque-chat@example.com", "password123")
        db.touch_chat(uid, "opaque", "my secret title")
        db.add_message(uid, "opaque", "user", "my secret message")
        with db._connect() as conn:
            title = conn.execute(
                "SELECT title FROM chats WHERE id = 'opaque'"
            ).fetchone()[0]
            content = conn.execute(
                "SELECT content FROM messages WHERE chat_id = 'opaque'"
            ).fetchone()[0]
        assert crypto.is_encrypted(title) and "secret title" not in title
        assert crypto.is_encrypted(content) and "secret message" not in content
        assert db.list_chats(uid)[0]["title"] == "my secret title"
        assert db.get_messages_for_user(uid, "opaque")[0]["content"] == "my secret message"

    def test_chat_retention_prunes_complete_turns(self, monkeypatch):
        monkeypatch.setattr(db, "MAX_CHAT_TURNS", 3)
        uid = db.register_user("retention@example.com", "password123")
        db.touch_chat(uid, "retained", "turn zero")
        for i in range(5):
            db.append_chat_turn(uid, "retained", f"user {i}", f"assistant {i}")
        messages = db.get_messages_for_user(uid, "retained")
        assert [(m["role"], m["content"]) for m in messages] == [
            ("user", "user 2"),
            ("assistant", "assistant 2"),
            ("user", "user 3"),
            ("assistant", "assistant 3"),
            ("user", "user 4"),
            ("assistant", "assistant 4"),
        ]

    def test_scrub_migration_is_durably_marked_and_removes_plaintext(self):
        from pathlib import Path

        uid = db.register_user("scrub@example.com", "password123")
        marker = b"FORENSIC-PLAINTEXT-MARKER-9d6109"
        with db._connect() as conn:
            conn.execute(
                "DELETE FROM maintenance_migrations WHERE name = ?",
                (db.ENCRYPTION_SCRUB_MIGRATION,),
            )
            conn.execute(
                """INSERT INTO profiles (user_id, data, updated_at)
                   VALUES (?, ?, ?)""",
                (uid, marker.decode("ascii"), db._now()),
            )

        assert db.encrypt_existing_rows() == 1
        with db._connect() as conn:
            assert conn.execute(
                "SELECT 1 FROM maintenance_migrations WHERE name = ?",
                (db.ENCRYPTION_SCRUB_MIGRATION,),
            ).fetchone()

        paths = [Path(db.DB_PATH), Path(str(db.DB_PATH) + "-wal")]
        raw = b"".join(path.read_bytes() for path in paths if path.exists())
        assert marker not in raw


class TestEmailTokenConcurrency:
    def test_reset_token_can_only_be_consumed_once_concurrently(self):
        uid = db.register_user("reset-race@example.com", "password123")
        raw = db.create_email_token(uid, "reset", ttl_minutes=30)
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(lambda _i: db.consume_email_token(raw, "reset"), range(4))
            )
        assert results.count(uid) == 1
        assert results.count(None) == 3


class TestAlertCheckConcurrency:
    def test_only_one_worker_claims_due_recomputation(self):
        uid = db.register_user("alert-race@example.com", "password123")
        with ThreadPoolExecutor(max_workers=8) as pool:
            claims = list(
                pool.map(lambda _i: db.claim_alert_check(uid), range(16))
            )

        assert claims.count(True) == 1
        assert claims.count(False) == 15
