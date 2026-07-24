"""Email verification + password-reset flow (offline: email sends are mocked)."""
import pytest

import api.db as db
from api.routers.legal import POLICY_VERSION


def registration(email: str, password: str = "password123") -> dict:
    return {
        "email": email,
        "password": password,
        "accept_terms": True,
        "policy_version": POLICY_VERSION,
    }


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    db.init_db()


# --- DB layer: one-time email tokens ----------------------------------------

class TestEmailTokens:
    def test_create_returns_raw_but_stores_hash(self):
        uid = db.register_user("a@example.com", "password123")
        raw = db.create_email_token(uid, "verify", ttl_minutes=60)
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        stored = conn.execute("SELECT token_hash FROM email_tokens").fetchone()[0]
        conn.close()
        assert raw and raw != stored          # raw token is not stored in the clear
        assert stored == db._hash_token(raw)  # only its SHA-256 hash is

    def test_consume_is_single_use(self):
        uid = db.register_user("b@example.com", "password123")
        raw = db.create_email_token(uid, "verify", ttl_minutes=60)
        assert db.consume_email_token(raw, "verify") == uid
        assert db.consume_email_token(raw, "verify") is None  # burned

    def test_wrong_purpose_rejected(self):
        uid = db.register_user("c@example.com", "password123")
        raw = db.create_email_token(uid, "reset", ttl_minutes=60)
        assert db.consume_email_token(raw, "verify") is None

    def test_expired_token_rejected(self):
        uid = db.register_user("d@example.com", "password123")
        raw = db.create_email_token(uid, "reset", ttl_minutes=-1)  # already expired
        assert db.consume_email_token(raw, "reset") is None

    def test_new_token_invalidates_previous(self):
        uid = db.register_user("e@example.com", "password123")
        old = db.create_email_token(uid, "reset", ttl_minutes=60)
        new = db.create_email_token(uid, "reset", ttl_minutes=60)
        assert db.consume_email_token(old, "reset") is None  # superseded
        assert db.consume_email_token(new, "reset") == uid


class TestVerificationAndReset:
    def test_verify_flag(self):
        uid = db.register_user("f@example.com", "password123")
        assert db.is_email_verified(uid) is False
        db.set_email_verified(uid)
        assert db.is_email_verified(uid) is True

    def test_set_password_changes_hash_and_revokes_sessions(self):
        uid = db.register_user("g@example.com", "password123")
        tok = db.issue_token(uid)
        assert db.user_for_token(tok) is not None
        db.set_password(uid, "brandnewpass1")
        assert db.user_for_token(tok) is None                      # sessions revoked
        assert db.verify_login("g@example.com", "brandnewpass1") == uid

    def test_set_password_enforces_length(self):
        uid = db.register_user("h@example.com", "password123")
        with pytest.raises(db.AuthError):
            db.set_password(uid, "short")

    def test_user_id_by_email(self):
        uid = db.register_user("i@example.com", "password123")
        assert db.user_id_by_email("I@Example.com") == uid  # case-insensitive
        assert db.user_id_by_email("missing@example.com") is None


# --- endpoint layer: forgot / reset / verify (email mocked) -----------------

@pytest.fixture
def client(monkeypatch):
    from starlette.testclient import TestClient
    import api.email_send as es
    from api import backend

    from api import state

    sent = {}
    monkeypatch.setattr(es, "send_verification_email", lambda to, raw: sent.__setitem__("verify", (to, raw)))
    monkeypatch.setattr(es, "send_password_reset_email", lambda to, raw: sent.__setitem__("reset", (to, raw)))
    # Rate-limit and lockout counters live in the shared state backend, which is
    # process-wide. Every TestClient request looks like the same IP, so start
    # each test from a clean backend or one test's attempts 429 the next.
    state.reset_backend()
    c = TestClient(backend.app)
    c.sent = sent
    return c


class TestForgotResetEndpoints:
    def test_forgot_does_not_enumerate(self, client):
        client.post("/auth/register", json=registration("known@example.com"))
        unknown = client.post("/auth/forgot", json={"email": "ghost@example.com"})
        known = client.post("/auth/forgot", json={"email": "known@example.com"})
        assert unknown.status_code == known.status_code == 200
        assert unknown.json() == known.json()  # identical response, can't tell them apart

    def test_full_reset_flow(self, client):
        client.post("/auth/register", json=registration("user@example.com"))
        client.post("/auth/forgot", json={"email": "user@example.com"})
        raw = client.sent["reset"][1]
        r = client.post("/auth/reset", json={"token": raw, "password": "newpassword456"})
        assert r.status_code == 200
        # single-use
        assert client.post("/auth/reset", json={"token": raw, "password": "another789"}).status_code == 400
        # new password works
        assert client.post("/auth/login",
                           json={"email": "user@example.com", "password": "newpassword456"}).status_code == 200

    def test_reset_marks_email_verified(self, client):
        reg = client.post("/auth/register", json=registration("v@example.com")).json()
        h = {"Authorization": "Bearer " + reg["token"]}
        # a reset link is not yet consumed; verify flag still false
        assert client.get("/me", headers=h).json()["email_verified"] is False
        client.post("/auth/forgot", json={"email": "v@example.com"})
        client.post("/auth/reset", json={"token": client.sent["reset"][1], "password": "newpassword456"})
        # log back in (reset revoked the old session) and confirm verified
        tok = client.post("/auth/login",
                          json={"email": "v@example.com", "password": "newpassword456"}).json()["token"]
        assert client.get("/me", headers={"Authorization": "Bearer " + tok}).json()["email_verified"] is True

    def test_verify_endpoint(self, client):
        client.post("/auth/register", json=registration("w@example.com"))
        raw = client.sent["verify"][1]
        r = client.get(f"/verify?token={raw}", follow_redirects=False)
        assert r.status_code == 303 and "verified=1" in r.headers["location"]
        bad = client.get(f"/verify?token={raw}", follow_redirects=False)  # reused
        assert "verified=0" in bad.headers["location"]

    def test_post_verify_consumes_fragment_token_as_json(self, client):
        client.post("/auth/register", json=registration("post-verify@example.com"))
        raw = client.sent["verify"][1]
        r = client.post("/auth/verify", json={"token": raw})
        assert r.status_code == 200
        assert r.json() == {"ok": True, "email_verified": True}
        assert client.post("/auth/verify", json={"token": raw}).status_code == 400
