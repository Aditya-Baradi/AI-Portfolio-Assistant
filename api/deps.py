"""
Shared request-handling pieces used by every router: authentication,
client-IP resolution, rate limiting, account lockout, and the small
validators that several endpoints need.

Auth model
----------
`current_user` resolves a Bearer token to a user. `verified_user` additionally
requires a confirmed email address — apply it to anything that stores real
financial data or spends money, so an unverified signup can look around but
cannot use the app as an anonymous compute/mail resource.

Rate limiting and lockout live in `api.state`, which is backed by Redis when
REDIS_URL is set and by process memory otherwise. That is what allows the app
to run with more than one worker.
"""
from __future__ import annotations

import hashlib
import logging

from fastapi import HTTPException, Request, Response

from api import db, state
from api.observability import bind_user

logger = logging.getLogger("evergreen.deps")

# --- limits ------------------------------------------------------------------

MAX_UPLOAD_BYTES = 2_000_000
MAX_HOLDINGS = 500
CHAT_DAILY_LIMIT = 50

# HttpOnly session cookie used by browser clients. Bearer tokens remain
# supported for API and existing clients.
SESSION_COOKIE_NAME = "evergreen_session"

#: Per-IP limit on authentication attempts.
AUTH_RATE_LIMIT = 10
AUTH_RATE_WINDOW = 300.0

#: Per-ACCOUNT lockout. IP limits alone don't stop a distributed password
#: spray against one inbox, so failures are also counted per email address.
LOCKOUT_THRESHOLD = 8
LOCKOUT_WINDOW = 900.0

#: Generic read limit, applied to the expensive analytics endpoints.
API_RATE_LIMIT = 120
API_RATE_WINDOW = 60.0


def client_ip(request: Request) -> str:
    """
    Peer address as resolved by the ASGI server.

    Never parse X-Forwarded-For in application code: Uvicorn rewrites
    ``request.client`` only for its explicitly trusted proxy allow-list.
    Direct clients therefore cannot choose their rate-limit/audit identity by
    supplying a forwarding header.
    """
    return request.client.host if request.client else "unknown"


# --- rate limiting -----------------------------------------------------------

def check_auth_rate(request: Request) -> None:
    """Per-IP sliding-window limit on auth endpoints. Raises 429 when exceeded."""
    ip = client_ip(request)
    if state.rate_limit_exceeded(f"auth:ip:{ip}", AUTH_RATE_LIMIT, AUTH_RATE_WINDOW):
        logger.warning("Auth rate limit hit", extra={"ip": ip})
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait a few minutes and try again.",
        )


def check_api_rate(request: Request, bucket: str = "api") -> None:
    """Coarse per-IP limit for the expensive analytics endpoints."""
    ip = client_ip(request)
    if state.rate_limit_exceeded(f"{bucket}:ip:{ip}", API_RATE_LIMIT, API_RATE_WINDOW):
        raise HTTPException(status_code=429, detail="Too many requests. Slow down a moment.")


# --- per-account lockout -----------------------------------------------------

def _lock_key(email: str) -> str:
    canonical = (email or "").strip().lower().encode("utf-8")
    # Redis operators need a stable per-account bucket, not the account's
    # plaintext address in key listings/snapshots.
    return f"lock:email-sha256:{hashlib.sha256(canonical).hexdigest()}"


def assert_not_locked(email: str) -> None:
    """Raise 429 when an account has too many recent failed sign-ins."""
    if state.failure_count(_lock_key(email), LOCKOUT_WINDOW) >= LOCKOUT_THRESHOLD:
        raise HTTPException(
            status_code=429,
            detail=("Too many failed sign-in attempts for this account. "
                    "Try again in 15 minutes, or reset your password."),
        )


def record_login_failure(email: str) -> int:
    n = state.record_failure(_lock_key(email), LOCKOUT_WINDOW)
    if n >= LOCKOUT_THRESHOLD:
        logger.warning("Account locked after repeated failures",
                       extra={"email_hash": hash(str(email).lower()) & 0xFFFFFF})
    return n


def clear_login_failures(email: str) -> None:
    state.clear_failures(_lock_key(email))


# --- authentication ----------------------------------------------------------

def session_token(
    authorization: str | None,
    session_cookie: str | None = None,
) -> str:
    """
    Resolve a raw session credential with explicit Authorization precedence.

    A cookie is considered only when the Authorization header is absent. This
    avoids ambiguous credentials and preserves existing Bearer behavior.
    """
    if authorization is not None:
        if authorization.lower().startswith("bearer "):
            return authorization[7:].strip()
        return ""
    return (session_cookie or "").strip()


def _token_from_header(
    authorization: str | None,
    session_cookie: str | None = None,
) -> str:
    """Backwards-compatible alias for callers that imported the old helper."""
    return session_token(authorization, session_cookie)


def current_user(
    authorization: str | None,
    session_cookie: str | None = None,
    response: Response | None = None,
) -> dict:
    """Resolve a Bearer token, or an HttpOnly cookie when no header is sent."""
    raw_token = session_token(authorization, session_cookie)
    user = db.user_for_token(raw_token)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    bind_user(user["id"])
    # Rotation is explicit (/auth/refresh) so concurrent browser/API requests
    # are not invalidated underneath one another. This signal lets clients
    # refresh at a safe boundary without breaking legacy Bearer clients.
    if user.get("stale") and response is not None:
        response.headers["X-Session-Refresh"] = "required"
    user["_auth_token"] = raw_token
    user["_auth_source"] = "bearer" if authorization is not None else "cookie"
    return user


def verified_user(
    authorization: str | None,
    session_cookie: str | None = None,
    response: Response | None = None,
) -> dict:
    """
    Like `current_user`, but requires a confirmed email address.

    Used to gate everything that stores financial data, spends money on model
    calls, or sends mail. Returns 403 with a machine-readable code so the UI
    can show a "resend verification" prompt rather than a dead end.
    """
    user = current_user(authorization, session_cookie, response)
    if not db.is_email_verified(user["id"]):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "email_unverified",
                "message": ("Please confirm your email address to use this feature. "
                            "Check your inbox for the verification link."),
            },
        )
    # Lazy import avoids a deps <-> router import cycle.
    from api.routers.legal import POLICY_VERSION

    if not db.has_policy_acceptance(user["id"], POLICY_VERSION):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "policy_acceptance_required",
                "version": POLICY_VERSION,
                "message": "Please accept the current Terms and Privacy Policy.",
            },
        )
    return user


# --- small shared validators -------------------------------------------------

def valid_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t or len(t) > 10 or not all(c.isalnum() or c in ".-" for c in t):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol.")
    return t


def agent_session_id(user_id: int, chat_id: str) -> str:
    """Chat memory is per (user, chat); the portfolio is shared per user."""
    return f"u{user_id}.c{chat_id}"


def weights_from_pf(pf) -> dict:
    """{ticker: dollar value} from a stored portfolio, dropping unusable rows."""
    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        if tkr:
            try:
                key = str(tkr).upper()
                weights[key] = weights.get(key, 0.0) + float(val)
            except (TypeError, ValueError):
                continue
    return {t: v for t, v in weights.items() if v > 0}


def live_prices(tickers: list) -> dict:
    """Latest market price per ticker (cached download); empty dict on failure."""
    from datetime import datetime, timedelta

    from api.observability import report
    from api.portfolio_core import _download_adj_close_matrix

    try:
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(tickers, start, end).ffill()
        last = prices.iloc[-1]
        return {t: float(last[t]) for t in prices.columns if last[t] == last[t]}
    except Exception as e:
        report(logger, "Live price lookup failed", e, tickers=str(tickers)[:200])
        return {}
