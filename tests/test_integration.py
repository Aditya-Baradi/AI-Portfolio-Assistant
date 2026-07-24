"""
End-to-end tests through the real ASGI app.

These cover the things unit tests can't: that the routers are actually wired
up, that the email-verification gate really blocks the endpoints it should,
that lockout engages, and that security headers are present on every response.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from starlette.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    import api.db as db
    from api import crypto, state

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "itest.db")
    monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
    monkeypatch.delenv("BACKUP_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)
    crypto.reset_cache()
    state.reset_backend()
    db.init_db()

    # Never send real mail from a test.
    import api.email_send as es
    sent: dict = {}
    monkeypatch.setattr(es, "send_verification_email",
                        lambda to, raw: sent.__setitem__("verify", (to, raw)))
    monkeypatch.setattr(es, "send_password_reset_email",
                        lambda to, raw: sent.__setitem__("reset", (to, raw)))

    from api import backend
    with TestClient(backend.app) as c:
        c.sent = sent
        c.db = db
        yield c

    state.reset_backend()
    crypto.reset_cache()


def register(client, email="user@example.com", password="password123"):
    from api.routers.legal import POLICY_VERSION

    r = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": password,
            "accept_terms": True,
            "policy_version": POLICY_VERSION,
        },
    )
    assert r.status_code == 200, r.text
    return r.json()["token"]


def auth(token):
    return {"Authorization": f"Bearer {token}"}


def verify(client, email):
    """Consume the verification token the registration email would have carried."""
    _to, raw = client.sent["verify"]
    r = client.get(f"/verify?token={raw}", follow_redirects=False)
    assert r.status_code == 303
    return r


# ---------------------------------------------------------------------------
# Email verification gate  (problem #5)
# ---------------------------------------------------------------------------

class TestEmailVerificationGate:
    #: Endpoints that must refuse an unverified account.
    GATED = [
        ("get", "/chats"),
        ("get", "/watchlist"),
        ("get", "/alerts"),
        ("get", "/profile"),
        ("get", "/portfolio/projection"),
        ("get", "/portfolio/diversification"),
        ("get", "/stocks/projections"),
    ]

    @pytest.mark.parametrize("method,path", GATED)
    def test_unverified_is_blocked(self, client, method, path):
        token = register(client)
        r = getattr(client, method)(path, headers=auth(token))
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "email_unverified"

    def test_chat_is_blocked_before_verification(self, client):
        token = register(client)
        r = client.post("/chat", json={"message": "hi", "chat_id": "c1"},
                        headers=auth(token))
        assert r.status_code == 403
        assert r.json()["detail"]["code"] == "email_unverified"

    def test_upload_is_blocked_before_verification(self, client):
        token = register(client)
        r = client.post("/upload", headers=auth(token),
                        files={"file": ("p.csv", b"ticker,shares\nAAPL,1\n", "text/csv")})
        assert r.status_code == 403

    def test_access_granted_after_verification(self, client):
        token = register(client)
        verify(client, "user@example.com")
        r = client.get("/watchlist", headers=auth(token))
        assert r.status_code == 200
        assert r.json()["tickers"] == []

    def test_me_works_unverified_and_reports_status(self, client):
        """/me must stay reachable — the UI needs it to show the resend prompt."""
        token = register(client)
        r = client.get("/me", headers=auth(token))
        assert r.status_code == 200
        assert r.json()["email_verified"] is False

    def test_password_reset_marks_verified(self, client):
        register(client, "reset@example.com")
        client.post("/auth/forgot", json={"email": "reset@example.com"})
        _to, raw = client.sent["reset"]
        r = client.post("/auth/reset", json={"token": raw, "password": "newpassword456"})
        assert r.status_code == 200

        login = client.post("/auth/login",
                            json={"email": "reset@example.com", "password": "newpassword456"})
        assert login.json()["email_verified"] is True


# ---------------------------------------------------------------------------
# Account lockout  (problem #8)
# ---------------------------------------------------------------------------

class TestAccountLockout:
    def test_repeated_failures_lock_the_account(self, client):
        from api.deps import LOCKOUT_THRESHOLD

        register(client, "target@example.com")
        codes = []
        for _ in range(LOCKOUT_THRESHOLD + 2):
            r = client.post("/auth/login",
                            json={"email": "target@example.com", "password": "wrong"})
            codes.append(r.status_code)
        assert 429 in codes, "account should lock after repeated failures"

    def test_lockout_is_per_account_not_global(self, client, monkeypatch):
        """Locking one account must not lock a different one."""
        from api import deps

        monkeypatch.setattr(deps, "AUTH_RATE_LIMIT", 10_000)  # isolate from the IP limit
        register(client, "victim@example.com")
        register(client, "bystander@example.com", "password123")

        for _ in range(deps.LOCKOUT_THRESHOLD + 1):
            client.post("/auth/login", json={"email": "victim@example.com",
                                             "password": "wrong"})

        r = client.post("/auth/login", json={"email": "bystander@example.com",
                                             "password": "password123"})
        assert r.status_code == 200

    def test_successful_login_clears_failures(self, client, monkeypatch):
        from api import deps

        monkeypatch.setattr(deps, "AUTH_RATE_LIMIT", 10_000)
        register(client, "clears@example.com", "password123")
        for _ in range(3):
            client.post("/auth/login", json={"email": "clears@example.com",
                                             "password": "wrong"})
        assert client.post("/auth/login", json={"email": "clears@example.com",
                                                "password": "password123"}).status_code == 200
        from api.deps import LOCKOUT_WINDOW, _lock_key
        from api import state
        assert state.failure_count(_lock_key("clears@example.com"), LOCKOUT_WINDOW) == 0


# ---------------------------------------------------------------------------
# Session tokens  (problem #12)
# ---------------------------------------------------------------------------

class TestSessions:
    def test_browser_can_authenticate_with_hardened_cookie_only(self, client):
        token = register(client, "cookie@example.com")
        assert token  # Bearer compatibility remains available to API clients.
        cookie = client.cookies.get("evergreen_session")
        assert cookie
        response = client.get("/me")
        assert response.status_code == 200
        assert response.json()["email"] == "cookie@example.com"

    def test_registration_cookie_has_csrf_and_script_access_defenses(self, client):
        from api.routers.legal import POLICY_VERSION

        response = client.post(
            "/auth/register",
            json={
                "email": "flags@example.com",
                "password": "password123",
                "accept_terms": True,
                "policy_version": POLICY_VERSION,
            },
        )
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "path=/" in cookie

    def test_bad_token_is_rejected(self, client):
        assert client.get("/me", headers=auth("not-a-real-token")).status_code == 401

    def test_logout_revokes(self, client):
        token = register(client)
        client.post("/auth/logout", headers=auth(token))
        assert client.get("/me", headers=auth(token)).status_code == 401

    def test_idle_session_expires(self, client, monkeypatch):
        import api.db as db

        token = register(client)
        assert client.get("/me", headers=auth(token)).status_code == 200

        # Fast-forward past the idle window by ageing the stored timestamps.
        with db._connect() as conn:
            conn.execute("UPDATE auth_tokens SET last_used_at = ?",
                         ("2000-01-01T00:00:00.000000",))
        assert client.get("/me", headers=auth(token)).status_code == 401

    def test_concurrent_sessions_are_capped(self, client):
        import api.db as db

        uid = db.register_user("many@example.com", "password123")
        for _ in range(db.MAX_SESSIONS_PER_USER + 5):
            db.issue_token(uid)
        with db._connect() as conn:
            n = conn.execute("SELECT COUNT(*) FROM auth_tokens WHERE user_id = ?",
                             (uid,)).fetchone()[0]
        assert n <= db.MAX_SESSIONS_PER_USER

    def test_change_password_invalidates_old_sessions(self, client):
        token = register(client, "rotate@example.com", "password123")
        r = client.post("/auth/change-password",
                        json={"current_password": "password123",
                              "new_password": "brandnewpass1"},
                        headers=auth(token))
        assert r.status_code == 200
        assert client.get("/me", headers=auth(token)).status_code == 401
        assert client.get("/me", headers=auth(r.json()["token"])).status_code == 200

    def test_cookie_password_change_does_not_reflect_replacement_token(self, client):
        register(client, "cookie-change@example.com", "password123")
        r = client.post(
            "/auth/change-password",
            json={
                "current_password": "password123",
                "new_password": "brandnewpass1",
            },
        )
        assert r.status_code == 200
        assert r.json() == {"ok": True}
        assert client.cookies.get("evergreen_session")
        assert client.get("/me").status_code == 200


# ---------------------------------------------------------------------------
# Security headers and legal surface
# ---------------------------------------------------------------------------

class TestSecurityHeaders:
    def test_headers_present(self, client):
        r = client.get("/")
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "no-referrer"
        assert "X-Request-ID" in r.headers

    def test_csp_has_a_nonce_and_no_unsafe_inline_script(self, client):
        csp = client.get("/").headers["Content-Security-Policy"]
        assert "nonce-" in csp
        script_src = [p for p in csp.split(";") if p.strip().startswith("script-src")][0]
        assert "unsafe-inline" not in script_src
        assert "frame-ancestors 'none'" in csp
        assert "object-src 'none'" in csp

    def test_nonce_differs_per_request(self, client):
        a = client.get("/").headers["Content-Security-Policy"]
        b = client.get("/").headers["Content-Security-Policy"]
        assert a != b, "a reused nonce defeats the entire point of using one"

    def test_page_nonce_matches_the_header(self, client):
        r = client.get("/")
        csp = r.headers["Content-Security-Policy"]
        nonce = csp.split("nonce-")[1].split("'")[0]
        assert f'nonce="{nonce}"' in r.text
        assert "__CSP_NONCE__" not in r.text  # placeholder fully substituted

    def test_sensitive_and_nonce_responses_are_not_cached(self, client):
        token = register(client, "cache-headers@example.com")
        for response in (client.get("/"), client.get("/me", headers=auth(token))):
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["Pragma"] == "no-cache"
        assert client.get("/terms").headers["Cache-Control"] == "no-cache"

    def test_cross_origin_isolation_headers_are_present(self, client):
        headers = client.get("/").headers
        assert headers["Cross-Origin-Opener-Policy"] == "same-origin"
        assert headers["Cross-Origin-Resource-Policy"] == "same-origin"
        assert headers["X-Permitted-Cross-Domain-Policies"] == "none"

    def test_request_id_is_validated_before_reflection(self, client):
        valid = "support-case_123.abc"
        assert client.get("/", headers={"X-Request-ID": valid}).headers[
            "X-Request-ID"
        ] == valid

        hostile = "../../bad request id"
        reflected = client.get("/", headers={"X-Request-ID": hostile}).headers[
            "X-Request-ID"
        ]
        assert reflected != hostile
        assert all(ch.isalnum() or ch in "._-" for ch in reflected)

    def test_oversized_json_body_is_rejected_before_parsing(self, client):
        from api.deps import MAX_UPLOAD_BYTES

        response = client.post(
            "/auth/login",
            content=b"x" * (MAX_UPLOAD_BYTES + 1),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["detail"] == "Request body is too large."


class TestLegalEndpoints:
    def test_disclosures(self, client):
        d = client.get("/legal/disclosures").json()
        assert "not_advice" in d and d["model_limitations"]

    def test_terms_and_privacy_render(self, client):
        for path in ("/terms", "/privacy"):
            r = client.get(path)
            assert r.status_code == 200
            assert "<h" in r.text

    def test_healthz_reports_readiness(self, client):
        body = client.get("/healthz").json()
        assert body["status"] == "ok"
        assert body["checks"]["backup_encryption_ready"] is False
        assert "market_data_contract_attested" in body["checks"]
        assert "market_data_licensed" not in body["checks"]
        # The dev fixture uses yfinance + in-memory state, so it must NOT
        # claim to be production-ready.
        assert body["checks"]["production_ready"] is False


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_bad_ticker_rejected(self, client):
        token = register(client)
        verify(client, "user@example.com")
        r = client.post("/watchlist/../../etc/passwd", headers=auth(token))
        assert r.status_code in (404, 422)

    def test_oversized_chat_message_rejected(self, client):
        token = register(client)
        verify(client, "user@example.com")
        r = client.post("/chat", json={"message": "x" * 10_000, "chat_id": "c1"},
                        headers=auth(token))
        assert r.status_code == 422

    def test_registration_requires_a_real_password(self, client):
        from api.routers.legal import POLICY_VERSION

        r = client.post(
            "/auth/register",
            json={
                "email": "a@b.com",
                "password": "short",
                "accept_terms": True,
                "policy_version": POLICY_VERSION,
            },
        )
        assert r.status_code == 400

    def test_forgot_does_not_enumerate(self, client):
        register(client, "known@example.com")
        a = client.post("/auth/forgot", json={"email": "ghost@example.com"})
        b = client.post("/auth/forgot", json={"email": "known@example.com"})
        assert a.status_code == b.status_code == 200
        assert a.json() == b.json()


# ---------------------------------------------------------------------------
# Hardened browser sessions, consent, chat ownership and two-factor auth
# ---------------------------------------------------------------------------

class TestBrowserSessionsAndConsent:
    def test_registration_requires_current_explicit_consent(self, client):
        from api.routers.legal import POLICY_VERSION

        missing = client.post(
            "/auth/register",
            json={"email": "terms@example.com", "password": "password123"},
        )
        assert missing.status_code == 400
        assert missing.json()["detail"] == {
            "code": "policy_acceptance_required",
            "version": POLICY_VERSION,
            "message": "You must accept the current Terms and Privacy Policy.",
        }

    def test_verified_legacy_user_must_reconsent_but_me_still_works(self, client):
        from api.routers.legal import POLICY_VERSION

        uid = client.db.register_user(
            "legacy-policy@example.com",
            "password123",
            accepted_policy_version="older-version",
        )
        client.db.set_email_verified(uid)
        token = client.db.issue_token(uid)
        assert client.get("/me", headers=auth(token)).status_code == 200
        gated = client.get("/chats", headers=auth(token))
        assert gated.status_code == 403
        assert gated.json()["detail"]["code"] == "policy_acceptance_required"
        assert gated.json()["detail"]["version"] == POLICY_VERSION

    def test_cookie_is_httponly_strict_and_authenticates(self, client):
        from api.deps import SESSION_COOKIE_NAME

        register(client, "cookie@example.com")
        cookie = client.cookies.get(SESSION_COOKIE_NAME)
        assert cookie
        assert client.get("/me").status_code == 200

    def test_cookie_refresh_never_exposes_raw_token(self, client):
        from api.deps import SESSION_COOKIE_NAME

        old = register(client, "cookie-refresh@example.com")
        refreshed = client.post("/auth/refresh")
        assert refreshed.status_code == 200
        assert refreshed.json() == {"ok": True}
        assert "X-Session-Token" not in refreshed.headers
        assert "token" not in refreshed.text
        new_cookie = client.cookies.get(SESSION_COOKIE_NAME)
        assert new_cookie and new_cookie != old
        assert client.get("/me", headers=auth(old)).status_code == 401
        assert client.get("/me").status_code == 200

    def test_explicit_bearer_refresh_returns_idempotent_successor(self, client):
        token = register(client, "bearer-refresh@example.com")
        first = client.post("/auth/refresh", headers=auth(token))
        second = client.post("/auth/refresh", headers=auth(token))
        assert first.status_code == second.status_code == 200
        assert first.json()["token"] == second.json()["token"]
        assert first.headers["X-Session-Token"] == first.json()["token"]
        assert client.get("/me", headers=auth(token)).status_code == 401
        assert client.get("/me", headers=auth(first.json()["token"])).status_code == 200

    def test_production_cookie_sets_secure_flag(self, client, monkeypatch):
        from api.routers.legal import POLICY_VERSION

        monkeypatch.setenv("EVERGREEN_ENV", "production")
        response = client.post(
            "/auth/register",
            json={
                "email": "secure-cookie@example.com",
                "password": "password123",
                "accept_terms": True,
                "policy_version": POLICY_VERSION,
            },
        )
        cookie = response.headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=strict" in cookie
        assert "secure" in cookie


class TestOwnedChatTransactions:
    def _verified(self, client, email):
        token = register(client, email)
        verify(client, email)
        return token

    def test_cold_history_does_not_include_current_message_twice(self, client, monkeypatch):
        from api.routers import chat as chat_router

        token = self._verified(client, "cold-chat@example.com")
        observed = []

        def fake_agent(message, session_id, *, user_id, chat_id):
            observed.append(client.db.get_messages_for_user(user_id, chat_id))
            return f"answer to {message}"

        monkeypatch.setattr(chat_router, "run_portfolio_agent", fake_agent)
        first = client.post(
            "/chat",
            json={"message": "first question", "chat_id": "cold-chat"},
            headers=auth(token),
        )
        second = client.post(
            "/chat",
            json={"message": "second question", "chat_id": "cold-chat"},
            headers=auth(token),
        )
        assert first.status_code == second.status_code == 200
        assert observed[0] == []
        assert [row["content"] for row in observed[1]] == [
            "first question",
            "answer to first question",
        ]
        stored = client.db.get_messages_for_user(
            client.db.user_id_by_email("cold-chat@example.com"),
            "cold-chat",
        )
        assert [row["content"] for row in stored] == [
            "first question",
            "answer to first question",
            "second question",
            "answer to second question",
        ]

    def test_foreign_chat_id_is_rejected_before_quota_or_model_call(self, client, monkeypatch):
        from api.routers import chat as chat_router

        owner_token = self._verified(client, "owner-route@example.com")
        calls = []
        monkeypatch.setattr(
            chat_router,
            "run_portfolio_agent",
            lambda message, session_id, **context: calls.append(context) or "private answer",
        )
        assert client.post(
            "/chat",
            json={"message": "owner message", "chat_id": "claimed-id"},
            headers=auth(owner_token),
        ).status_code == 200

        attacker_token = self._verified(client, "attacker-route@example.com")
        foreign_write = client.post(
            "/chat",
            json={"message": "overwrite it", "chat_id": "claimed-id"},
            headers=auth(attacker_token),
        )
        foreign_read = client.get(
            "/chats/claimed-id/messages",
            headers=auth(attacker_token),
        )
        assert foreign_write.status_code == foreign_read.status_code == 404
        assert len(calls) == 1
        attacker_id = client.db.user_id_by_email("attacker-route@example.com")
        assert client.db.count_user_messages_today(attacker_id) == 0

    def test_chat_id_canonical_alphabet_prevents_cache_collisions(self, client, monkeypatch):
        from api.routers import chat as chat_router

        token = self._verified(client, "chat-id@example.com")
        monkeypatch.setattr(chat_router, "run_portfolio_agent", lambda *a, **k: "unused")
        for unsafe in ("a/b", "a b", "a.b", "../other"):
            response = client.post(
                "/chat",
                json={"message": "hello", "chat_id": unsafe},
                headers=auth(token),
            )
            assert response.status_code == 422

    def test_model_failure_is_reported_as_a_generic_upstream_error(
        self, client, monkeypatch
    ):
        from api.routers import chat as chat_router

        token = self._verified(client, "model-failure@example.com")

        def fail(*_args, **_kwargs):
            raise RuntimeError("provider body with secret internals")

        monkeypatch.setattr(chat_router, "run_portfolio_agent", fail)
        response = client.post(
            "/chat",
            json={"message": "hello", "chat_id": "failed-turn"},
            headers=auth(token),
        )
        assert response.status_code == 502
        assert response.json()["detail"] == (
            "The assistant is temporarily unavailable. Please try again later."
        )
        assert "provider body" not in response.text


class TestTwoFactorHardening:
    def _enable(self, client, email="twofa@example.com", password="password123"):
        import pyotp

        token = register(client, email, password)
        setup = client.post(
            "/auth/2fa/setup",
            json={"password": password},
            headers=auth(token),
        )
        assert setup.status_code == 200, setup.text
        body = setup.json()
        assert body["qr"].startswith("data:image/png;base64,")

        uid = client.db.user_id_by_email(email)
        pending = __import__("api.state", fromlist=["get_backend"]).get_backend().get(
            f"2fa:setup:{uid}"
        )
        assert body["secret"] not in json.dumps(pending)
        assert pending["secret_ciphertext"].startswith("enc:v1:")

        enabled = client.post(
            "/auth/2fa/enable",
            json={"code": pyotp.TOTP(body["secret"]).now()},
            headers=auth(token),
        )
        assert enabled.status_code == 200, enabled.text
        return token, body["secret"]

    def test_setup_requires_password_and_replacement_requires_current_totp(
        self,
        client,
    ):
        import pyotp

        token, current_secret = self._enable(client)
        missing_current = client.post(
            "/auth/2fa/setup",
            json={"password": "password123"},
            headers=auth(token),
        )
        assert missing_current.status_code == 401
        replacement = client.post(
            "/auth/2fa/setup",
            json={
                "password": "password123",
                "current_code": pyotp.TOTP(current_secret).now(),
            },
            headers=auth(token),
        )
        assert replacement.status_code == 200
        new_secret = replacement.json()["secret"]
        enabled = client.post(
            "/auth/2fa/enable",
            json={"code": pyotp.TOTP(new_secret).now()},
            headers=auth(token),
        )
        assert enabled.status_code == 200
        uid = client.db.user_id_by_email("twofa@example.com")
        assert client.db.get_totp_secret(uid) == new_secret

    def test_pending_login_attempt_cap_contributes_to_account_lockout(
        self,
        client,
        monkeypatch,
    ):
        import pyotp
        from api import deps

        monkeypatch.setattr(deps, "AUTH_RATE_LIMIT", 10_000)
        _token, secret = self._enable(client, "cap-2fa@example.com")
        wrong = "000000" if pyotp.TOTP(secret).now() != "000000" else "111111"

        pending = client.post(
            "/auth/login",
            json={"email": "cap-2fa@example.com", "password": "password123"},
        ).json()["pending"]
        statuses = [
            client.post(
                "/auth/2fa/verify",
                json={"pending": pending, "code": wrong},
            ).status_code
            for _ in range(5)
        ]
        assert statuses[:4] == [401] * 4
        assert statuses[4] == 429
        assert client.post(
            "/auth/2fa/verify",
            json={"pending": pending, "code": pyotp.TOTP(secret).now()},
        ).status_code == 401

        # Failures are account-wide, not reset by obtaining a fresh challenge.
        pending2 = client.post(
            "/auth/login",
            json={"email": "cap-2fa@example.com", "password": "password123"},
        ).json()["pending"]
        more = [
            client.post(
                "/auth/2fa/verify",
                json={"pending": pending2, "code": wrong},
            ).status_code
            for _ in range(3)
        ]
        assert more[-1] == 429
        assert client.post(
            "/auth/login",
            json={"email": "cap-2fa@example.com", "password": "password123"},
        ).status_code == 429

    def test_password_reset_revokes_totp_sessions_and_pending_login(self, client):
        import pyotp

        old_token, secret = self._enable(client, "recover-2fa@example.com")
        pending = client.post(
            "/auth/login",
            json={"email": "recover-2fa@example.com", "password": "password123"},
        ).json()["pending"]
        client.post("/auth/forgot", json={"email": "recover-2fa@example.com"})
        reset_token = client.sent["reset"][1]
        assert client.post(
            "/auth/reset",
            json={"token": reset_token, "password": "replacement123"},
        ).status_code == 200

        uid = client.db.user_id_by_email("recover-2fa@example.com")
        assert client.db.get_totp_secret(uid) is None
        assert client.get("/me", headers=auth(old_token)).status_code == 401
        assert client.post(
            "/auth/2fa/verify",
            json={"pending": pending, "code": pyotp.TOTP(secret).now()},
        ).status_code == 401
        login = client.post(
            "/auth/login",
            json={"email": "recover-2fa@example.com", "password": "replacement123"},
        )
        assert login.status_code == 200
        assert login.json().get("requires_2fa") is not True

    def test_pending_login_success_is_single_claim_under_concurrency(
        self,
        client,
        monkeypatch,
    ):
        import pyotp

        _token, secret = self._enable(client, "claim-2fa@example.com")
        pending = client.post(
            "/auth/login",
            json={"email": "claim-2fa@example.com", "password": "password123"},
        ).json()["pending"]
        uid = client.db.user_id_by_email("claim-2fa@example.com")
        with client.db._connect() as conn:
            before = conn.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE user_id = ? AND rotated_at IS NULL",
                (uid,),
            ).fetchone()[0]

        entered = threading.Event()
        release = threading.Event()

        def slow_verify(_self, _code, valid_window=0):
            entered.set()
            assert release.wait(5)
            return True

        monkeypatch.setattr(pyotp.TOTP, "verify", slow_verify)
        payload = {"pending": pending, "code": pyotp.TOTP(secret).now()}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(client.post, "/auth/2fa/verify", json=payload)
            assert entered.wait(5)
            second = client.post("/auth/2fa/verify", json=payload)
            release.set()
            first = first_future.result(timeout=5)
        assert sorted((first.status_code, second.status_code)) == [200, 409]
        with client.db._connect() as conn:
            after = conn.execute(
                "SELECT COUNT(*) FROM auth_tokens WHERE user_id = ? AND rotated_at IS NULL",
                (uid,),
            ).fetchone()[0]
        assert after == before + 1

    def test_setup_enable_is_single_claim_under_concurrency(
        self,
        client,
        monkeypatch,
    ):
        import pyotp

        token = register(client, "claim-setup@example.com")
        setup = client.post(
            "/auth/2fa/setup",
            json={"password": "password123"},
            headers=auth(token),
        ).json()
        uid = client.db.user_id_by_email("claim-setup@example.com")
        version_before = client.db.get_auth_version(uid)
        entered = threading.Event()
        release = threading.Event()

        def slow_verify(_self, _code, valid_window=0):
            entered.set()
            assert release.wait(5)
            return True

        monkeypatch.setattr(pyotp.TOTP, "verify", slow_verify)
        payload = {"code": pyotp.TOTP(setup["secret"]).now()}
        with ThreadPoolExecutor(max_workers=2) as pool:
            first_future = pool.submit(
                client.post,
                "/auth/2fa/enable",
                json=payload,
                headers=auth(token),
            )
            assert entered.wait(5)
            second = client.post(
                "/auth/2fa/enable",
                json=payload,
                headers=auth(token),
            )
            release.set()
            first = first_future.result(timeout=5)
        assert sorted((first.status_code, second.status_code)) == [200, 409]
        assert client.db.get_auth_version(uid) == version_before + 1
