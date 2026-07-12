"""Tests for the SQL auth/portfolio/chat layer (offline, temp database)."""
import pytest

import api.db as db


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


class TestAuth:
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
