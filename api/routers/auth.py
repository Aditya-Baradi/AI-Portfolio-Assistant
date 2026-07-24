"""
Registration, sign-in, two-factor, email verification and password reset.

Authentication supports both Bearer tokens (for existing/API clients) and an
HttpOnly SameSite cookie (for the browser UI). Session rotation is explicit so
parallel requests cannot unexpectedly invalidate one another.
"""
from __future__ import annotations

import base64
import io
import logging
import secrets as _secrets
import time

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Cookie,
    Header,
    HTTPException,
    Request,
    Response,
)
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from api import db, state
from api.deps import (
    LOCKOUT_THRESHOLD,
    SESSION_COOKIE_NAME,
    assert_not_locked,
    check_auth_rate,
    clear_login_failures,
    client_ip,
    current_user,
    record_login_failure,
    session_token,
)

logger = logging.getLogger("evergreen.auth")
router = APIRouter(tags=["auth"])

PENDING_2FA_TTL = 300.0
PENDING_2FA_MAX_ATTEMPTS = 5
TWOFA_SETUP_MAX_ATTEMPTS = 5
TWOFA_CLAIM_WINDOW = 15.0


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=256)
    name: str | None = Field(default=None, max_length=60)
    accept_terms: bool = False
    policy_version: str | None = Field(default=None, max_length=64)


class TwoFAVerify(BaseModel):
    pending: str = Field(min_length=16, max_length=128)
    code: str = Field(min_length=6, max_length=16)


class TwoFASetupBody(BaseModel):
    password: str = Field(min_length=1, max_length=256)
    current_code: str | None = Field(default=None, max_length=16)


class TwoFACode(BaseModel):
    code: str = Field(min_length=6, max_length=16)


class PasswordBody(BaseModel):
    password: str = Field(min_length=1, max_length=256)


class ForgotBody(BaseModel):
    email: str = Field(min_length=3, max_length=254)


class ResetBody(BaseModel):
    token: str = Field(min_length=16, max_length=512)
    password: str = Field(min_length=1, max_length=256)


class TokenBody(BaseModel):
    token: str = Field(min_length=16, max_length=512)


def _cookie_secure() -> bool:
    from api.crypto import is_production

    return is_production()


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        max_age=db.TOKEN_ABSOLUTE_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=_cookie_secure(),
        samesite="strict",
        path="/",
    )
    response.headers["Cache-Control"] = "no-store"


def _clear_session_cookie(response: Response) -> None:
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=_cookie_secure(),
        httponly=True,
        samesite="strict",
    )
    response.headers["Cache-Control"] = "no-store"


def _lock_email(value: str) -> str:
    try:
        return db.normalize_email(value)
    except db.AuthError:
        # Invalid addresses still get a stable, bounded lockout bucket.
        return (value or "").strip().lower()[:254] or "<invalid-email>"


def _raise_failed_signin(email: str, detail: str) -> None:
    if record_login_failure(email) >= LOCKOUT_THRESHOLD:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many failed sign-in attempts for this account. "
                "Try again in 15 minutes, or reset your password."
            ),
        )
    raise HTTPException(status_code=401, detail=detail)


def _send_verification(user_id: int, email: str) -> bool:
    """Issue and send a verification token without exposing delivery errors."""
    from api import email_send

    try:
        raw = db.create_email_token(user_id, "verify", ttl_minutes=60 * 24)
        email_send.send_verification_email(email, raw)
        return True
    except Exception:
        logger.exception("Verification email failed for user %s", user_id)
        return False


def _deliver_password_reset(email_input: str, ip: str) -> None:
    """Background delivery keeps provider latency outside the HTTP response."""
    from api import email_send

    user_id = db.user_id_by_email(email_input)
    if not user_id:
        return
    identity = db.get_user_identity(user_id) or {}
    email = str(identity.get("email") or "")
    try:
        raw = db.create_email_token(user_id, "reset", ttl_minutes=30)
        email_send.send_password_reset_email(email, raw)
        db.log_event(user_id, "password_reset_requested", ip)
    except Exception:
        logger.exception("Password-reset email failed for user %s", user_id)


@router.post("/auth/register")
def register(creds: Credentials, request: Request, response: Response):
    check_auth_rate(request)
    from api.routers.legal import POLICY_VERSION

    if not creds.accept_terms or creds.policy_version != POLICY_VERSION:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "policy_acceptance_required",
                "version": POLICY_VERSION,
                "message": "You must accept the current Terms and Privacy Policy.",
            },
        )
    try:
        user_id = db.register_user(
            creds.email,
            creds.password,
            creds.name,
            accepted_policy_version=POLICY_VERSION,
        )
    except db.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    identity = db.get_user_identity(user_id) or {}
    email = str(identity.get("email") or "")
    db.log_event(user_id, "register", client_ip(request))
    sent = _send_verification(user_id, email)
    token = db.issue_token(user_id)
    _set_session_cookie(response, token)
    return {
        "token": token,
        "email": email,
        "name": db.get_user_name(user_id),
        "email_verified": False,
        "verification_sent": sent,
        "policy_version": POLICY_VERSION,
    }


@router.post("/auth/login")
def login(creds: Credentials, request: Request, response: Response):
    check_auth_rate(request)
    email = _lock_email(creds.email)
    assert_not_locked(email)

    try:
        user_id = db.verify_login(creds.email, creds.password)
    except db.AuthError as exc:
        _raise_failed_signin(email, str(exc))

    identity = db.get_user_identity(user_id) or {}
    email = str(identity.get("email") or email)
    if db.get_totp_secret(user_id):
        pending = _secrets.token_urlsafe(24)
        state.get_backend().set(
            f"2fa:pending:{pending}",
            {
                "user_id": user_id,
                "auth_version": int(identity.get("auth_version", -1)),
                "expires_at": time.time() + PENDING_2FA_TTL,
            },
            PENDING_2FA_TTL,
        )
        # A browser signing into a different 2FA account must not silently keep
        # using the previous account's cookie during the challenge.
        _clear_session_cookie(response)
        return {"requires_2fa": True, "pending": pending, "email": email}

    clear_login_failures(email)
    db.log_event(user_id, "login", client_ip(request))
    token = db.issue_token(user_id)
    _set_session_cookie(response, token)
    return {
        "token": token,
        "email": email,
        "name": db.get_user_name(user_id),
        "email_verified": db.is_email_verified(user_id),
    }


@router.post("/auth/2fa/verify")
def twofa_verify(body: TwoFAVerify, request: Request, response: Response):
    check_auth_rate(request)
    import pyotp

    backend = state.get_backend()
    key = f"2fa:pending:{body.pending}"
    attempt_key = f"2fa:pending-attempt:{body.pending}"
    claim_key = f"2fa:pending-claim:{body.pending}"
    pending = backend.get(key)
    if not isinstance(pending, dict):
        backend.delete(key)
        raise HTTPException(status_code=401, detail="Login expired. Start again.")

    try:
        user_id = int(pending["user_id"])
        expected_version = int(pending["auth_version"])
        expires_at = float(pending["expires_at"])
    except (KeyError, TypeError, ValueError):
        backend.delete(key)
        raise HTTPException(status_code=401, detail="Login expired. Start again.") from None

    identity = db.get_user_identity(user_id)
    if (
        expires_at <= time.time()
        or not identity
        or int(identity["auth_version"]) != expected_version
    ):
        backend.delete(key)
        state.clear_failures(attempt_key)
        raise HTTPException(status_code=401, detail="Login expired. Start again.")

    email = str(identity["email"])
    assert_not_locked(email)
    if state.failure_count(attempt_key, PENDING_2FA_TTL) >= PENDING_2FA_MAX_ATTEMPTS:
        backend.delete(key)
        raise HTTPException(status_code=429, detail="Too many code attempts. Start again.")
    if state.rate_limit_exceeded(claim_key, 1, TWOFA_CLAIM_WINDOW):
        raise HTTPException(status_code=409, detail="This code attempt is already in progress.")

    secret = db.get_totp_secret(user_id)
    if not secret or not pyotp.TOTP(secret).verify(body.code.strip(), valid_window=1):
        db.log_event(user_id, "failed_2fa", client_ip(request))
        attempts = state.record_failure(attempt_key, PENDING_2FA_TTL)
        account_failures = record_login_failure(email)
        if (
            attempts >= PENDING_2FA_MAX_ATTEMPTS
            or account_failures >= LOCKOUT_THRESHOLD
        ):
            backend.delete(key)
            raise HTTPException(
                status_code=429,
                detail="Too many code attempts. Start sign-in again later.",
            )
        state.clear_failures(claim_key)
        raise HTTPException(status_code=401, detail="Wrong code. Try again.")

    # Burn the handle before minting a session. The atomic claim above prevents
    # another worker that already read it from crossing this boundary.
    backend.delete(key)
    state.clear_failures(attempt_key)
    clear_login_failures(email)
    db.log_event(user_id, "login", client_ip(request))
    token = db.issue_token(user_id)
    state.clear_failures(claim_key)
    _set_session_cookie(response, token)
    return {
        "token": token,
        "email": email,
        "name": identity.get("name"),
        "email_verified": db.is_email_verified(user_id),
    }


@router.post("/auth/2fa/setup")
def twofa_setup(
    body: TwoFASetupBody,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """Start 2FA setup/replacement after re-authenticating the account owner."""
    user = current_user(authorization, session_cookie, response)
    check_auth_rate(request)
    assert_not_locked(user["email"])

    existing = db.get_totp_secret(user["id"])
    password_ok = db.verify_password(user["id"], body.password)
    current_code_ok = True
    if existing:
        import pyotp

        current_code_ok = bool(
            body.current_code
            and pyotp.TOTP(existing).verify(body.current_code.strip(), valid_window=1)
        )
    if not password_ok or not current_code_ok:
        _raise_failed_signin(user["email"], "Password or current authenticator code is wrong.")

    clear_login_failures(user["email"])
    import pyotp
    import qrcode

    secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user["email"],
        issuer_name="Evergreen",
    )
    image = qrcode.make(uri)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    qr_data_uri = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

    from api import crypto

    state.clear_failures(f"2fa:setup-attempt:{user['id']}")
    state.get_backend().set(
        f"2fa:setup:{user['id']}",
        {
            # Redis may use AOF/RDB persistence, so pending secrets receive the
            # same envelope encryption as enabled TOTP secrets in SQLite.
            "secret_ciphertext": crypto.encrypt(secret),
            "replacement": bool(existing),
            "auth_version": db.get_auth_version(user["id"]),
            "expires_at": time.time() + PENDING_2FA_TTL,
        },
        PENDING_2FA_TTL,
    )
    response.headers["Cache-Control"] = "no-store"
    return {"secret": secret, "uri": uri, "qr": qr_data_uri}


@router.post("/auth/2fa/enable")
def twofa_enable(
    body: TwoFACode,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    user = current_user(authorization, session_cookie, response)
    check_auth_rate(request)
    import pyotp

    backend = state.get_backend()
    key = f"2fa:setup:{user['id']}"
    attempt_key = f"2fa:setup-attempt:{user['id']}"
    claim_key = f"2fa:setup-claim:{user['id']}"
    setup = backend.get(key)
    if not isinstance(setup, dict):
        backend.delete(key)
        raise HTTPException(status_code=400, detail="Start setup again.")
    try:
        from api import crypto

        secret = crypto.decrypt(str(setup["secret_ciphertext"]))
        if not secret:
            raise ValueError("missing secret")
        expected_version = int(setup["auth_version"])
        expires_at = float(setup["expires_at"])
    except (KeyError, TypeError, ValueError):
        backend.delete(key)
        raise HTTPException(status_code=400, detail="Start setup again.") from None
    if expires_at <= time.time() or db.get_auth_version(user["id"]) != expected_version:
        backend.delete(key)
        state.clear_failures(attempt_key)
        raise HTTPException(status_code=400, detail="Setup expired. Start again.")
    if state.failure_count(attempt_key, PENDING_2FA_TTL) >= TWOFA_SETUP_MAX_ATTEMPTS:
        backend.delete(key)
        raise HTTPException(status_code=429, detail="Too many code attempts. Start again.")
    if state.rate_limit_exceeded(claim_key, 1, TWOFA_CLAIM_WINDOW):
        raise HTTPException(status_code=409, detail="This code attempt is already in progress.")
    if not pyotp.TOTP(secret).verify(body.code.strip(), valid_window=1):
        attempts = state.record_failure(attempt_key, PENDING_2FA_TTL)
        if attempts >= TWOFA_SETUP_MAX_ATTEMPTS:
            backend.delete(key)
            raise HTTPException(status_code=429, detail="Too many code attempts. Start again.")
        state.clear_failures(claim_key)
        raise HTTPException(status_code=401, detail="Wrong code. Scan the QR and try again.")

    replacement = bool(setup.get("replacement"))
    # Consume setup before committing the enabled secret. A crash may force the
    # user to restart setup, but cannot leave a reusable success handle.
    backend.delete(key)
    db.set_totp_secret(user["id"], secret)
    state.clear_failures(attempt_key)
    db.revoke_other_tokens(user["id"], user["_auth_token"])
    db.log_event(
        user["id"],
        "2fa_replaced" if replacement else "2fa_enabled",
        client_ip(request),
    )
    state.clear_failures(claim_key)
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


@router.post("/auth/2fa/disable")
def twofa_disable(
    body: PasswordBody,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    user = current_user(authorization, session_cookie, response)
    check_auth_rate(request)
    assert_not_locked(user["email"])
    if not db.verify_password(user["id"], body.password):
        _raise_failed_signin(user["email"], "Wrong password.")
    clear_login_failures(user["email"])
    db.set_totp_secret(user["id"], None)
    state.get_backend().delete(f"2fa:setup:{user['id']}")
    state.clear_failures(f"2fa:setup-attempt:{user['id']}")
    db.revoke_other_tokens(user["id"], user["_auth_token"])
    db.log_event(user["id"], "2fa_disabled", client_ip(request))
    response.headers["Cache-Control"] = "no-store"
    return {"ok": True}


@router.post("/auth/refresh")
def refresh_session(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    check_auth_rate(request)
    old = session_token(authorization, session_cookie)
    replacement = db.rotate_token(old)
    if not replacement:
        # Do not clear the browser cookie here: another tab may have completed
        # rotation between this request being sent and this response arriving.
        raise HTTPException(status_code=401, detail="Session expired. Sign in again.")
    if bool(getattr(request.state, "auth_from_cookie", False)):
        # Never reflect an HttpOnly credential into script-readable JSON or a
        # response header. The replacement remains cookie-only.
        _set_session_cookie(response, replacement)
        return {"ok": True}

    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Session-Token"] = replacement
    return {"token": replacement}


@router.post("/auth/logout")
def logout(
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    raw = session_token(authorization, session_cookie)
    if raw:
        db.revoke_token(raw)
    _clear_session_cookie(response)
    return {"ok": True}


# --- email verification + password reset ------------------------------------

@router.post("/auth/verify")
def verify_email_json(body: TokenBody, request: Request):
    """Consume a token received from the URL fragment and return JSON."""
    check_auth_rate(request)
    user_id = db.consume_email_token(body.token, "verify")
    if not user_id:
        raise HTTPException(status_code=400, detail="This verification link is invalid or expired.")
    db.set_email_verified(user_id)
    return {"ok": True, "email_verified": True}


@router.get("/verify")
def verify_email(token: str):
    """Legacy email-link endpoint retained for old links and bookmarks."""
    user_id = db.consume_email_token(token, "verify")
    if not user_id:
        return RedirectResponse(url="/?verified=0", status_code=303)
    db.set_email_verified(user_id)
    return RedirectResponse(url="/?verified=1", status_code=303)


@router.post("/auth/verify/resend")
def resend_verification(
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    user = current_user(authorization, session_cookie, response)
    check_auth_rate(request)
    if db.is_email_verified(user["id"]):
        return {"ok": True, "already_verified": True}
    if not _send_verification(user["id"], user["email"]):
        raise HTTPException(
            status_code=502,
            detail="Couldn't send the email right now. Try again shortly.",
        )
    return {"ok": True}


@router.post("/auth/forgot")
def forgot_password(body: ForgotBody, request: Request, background: BackgroundTasks):
    """
    Always return the same response whether the canonical address exists.
    """
    check_auth_rate(request)
    # Account lookup and provider I/O both happen after the response begins, so
    # known/unknown addresses share the same request-time boundary.
    background.add_task(_deliver_password_reset, body.email, client_ip(request))
    return {
        "ok": True,
        "message": "If an account exists for that email, a reset link is on its way.",
    }


@router.post("/auth/reset")
def reset_password(body: ResetBody, request: Request, response: Response):
    check_auth_rate(request)
    # Validate before consuming the single-use token so a correct link is not
    # burned merely because the replacement password is malformed.
    try:
        db.validate_password(body.password, label="New password")
    except db.AuthError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None

    user_id = db.consume_email_token(body.token, "reset")
    if not user_id:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired.")
    db.set_password(user_id, body.password)
    db.set_email_verified(user_id)
    db.log_event(user_id, "password_changed", client_ip(request))

    identity = db.get_user_identity(user_id)
    if identity:
        clear_login_failures(str(identity["email"]))
    state.get_backend().delete(f"2fa:setup:{user_id}")
    state.clear_failures(f"2fa:setup-attempt:{user_id}")
    _clear_session_cookie(response)
    return {"ok": True}
