"""
Evergreen application assembly.

This module is deliberately thin: middleware, startup checks, and router
registration. The endpoints themselves live in `api/routers/`, one module per
slice of the API, so no single file owns the whole surface any more.

Startup performs a readiness audit (see `_startup_audit`). In production it
refuses to boot on a misconfiguration that would be unsafe to serve — for
example, a data adapter without an operator attestation of the required
contractual rights, or encrypted fields with no master key. In development it
prints the same findings as warnings and carries on.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

# Configuration must be loaded before importing ``api.db``: that module
# resolves EVERGREEN_DATA_DIR while it is imported. Previously .env happened
# to load later as a side effect of the chat module, so local operators who set
# a custom data directory silently wrote app.db somewhere else.
load_dotenv()

from api import db
from api.observability import (
    bind_request,
    bind_user,
    configure_logging,
    configure_sentry,
    new_request_id,
)
from api.routers import account, alerts, auth, chat, legal, plan, portfolio, stocks

configure_logging()
configure_sentry()
logger = logging.getLogger("evergreen")


class RequestTooLarge(Exception):
    """Internal signal raised when a chunked request crosses the body cap."""


class RequestBodyLimitMiddleware:
    """
    Enforce the body limit on declared *and* streamed request bytes.

    Checking only ``Content-Length`` leaves chunked requests unbounded and lets
    an unauthenticated client make the framework buffer an arbitrarily large
    JSON body before Pydantic validation runs.
    """

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = int(max_bytes)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        raw_lengths = [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == b"content-length"
        ]
        if raw_lengths:
            try:
                if len(raw_lengths) != 1:
                    raise ValueError
                declared = int(raw_lengths[0])
                if declared < 0:
                    raise ValueError
            except (TypeError, ValueError):
                response = JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
                await response(scope, receive, send)
                return
            if declared > self.max_bytes:
                response = JSONResponse(
                    status_code=413,
                    content={"detail": "Request body is too large."},
                )
                await response(scope, receive, send)
                return

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise RequestTooLarge
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestTooLarge:
            if response_started:
                raise
            response = JSONResponse(
                status_code=413,
                content={"detail": "Request body is too large."},
            )
            await response(scope, receive, send)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Startup housekeeping and the configuration readiness audit."""
    from api.crypto import is_production

    # Do this FIRST: if the key is wrong, nothing else about this process is
    # trustworthy and we must not serve empty portfolios to real users.
    _verify_encryption_key()

    removed = db.purge_expired_tokens()
    if removed:
        logger.info("Purged %d expired session token(s).", removed)

    problems = _startup_audit()
    if not problems:
        logger.info("Startup audit passed; instance is configured for public use.")
    elif is_production():
        for p in problems:
            logger.error("STARTUP BLOCKER: %s", p)
        raise RuntimeError(
            "Refusing to start in production with an unsafe configuration:\n  - "
            + "\n  - ".join(problems)
        )
    else:
        logger.warning("Startup audit found %d issue(s) — fine for local development, "
                       "but this instance is NOT ready for public use:", len(problems))
        for p in problems:
            logger.warning("  - %s", p)

    yield


app = FastAPI(
    title="Evergreen",
    description="Educational portfolio analytics. Not investment advice.",
    version="1.0.0",
    lifespan=lifespan,
)
db.init_db()

from api.deps import MAX_UPLOAD_BYTES

app.add_middleware(RequestBodyLimitMiddleware, max_bytes=MAX_UPLOAD_BYTES)

# The backend serves the frontend itself, so only same-host origins are needed.
# Set APP_ORIGINS (comma-separated) to your real domain(s) when deploying.
_DEFAULT_ORIGINS = [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:8010", "http://localhost:8010",  # verification port
]
ALLOWED_ORIGINS = [
    o.strip().rstrip("/")
    for o in os.getenv("APP_ORIGINS", "").split(",")
    if o.strip()
] \
    or _DEFAULT_ORIGINS


def _public_url_problem(value: str, label: str, *, origin_only: bool) -> str | None:
    """Validate an operator-supplied public URL without making a network call."""
    if not value:
        return f"{label} is not set."
    if value == "*" or any(ch.isspace() for ch in value):
        return f"{label} must be an explicit HTTPS URL, never a wildcard."
    try:
        parsed = urlsplit(value)
        # Accessing port performs its own range/integer validation.
        _ = parsed.port
    except ValueError:
        return f"{label} is not a valid URL."
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or (origin_only and parsed.path not in {"", "/"})
    ):
        kind = "origin" if origin_only else "base URL"
        return (
            f"{label} must be an explicit HTTPS {kind} without credentials, "
            "a query string, or a fragment."
        )
    return None


def _web_configuration_problems() -> list[str]:
    """Unsafe public URL/CORS settings that production must refuse."""
    problems: list[str] = []
    configured_origins = [
        origin.strip().rstrip("/")
        for origin in os.getenv("APP_ORIGINS", "").split(",")
        if origin.strip()
    ]
    if not configured_origins:
        problems.append("APP_ORIGINS is not set; CORS is allowing localhost origins only.")
    else:
        for index, origin in enumerate(configured_origins, start=1):
            problem = _public_url_problem(
                origin,
                f"APP_ORIGINS entry {index}",
                origin_only=True,
            )
            if problem:
                problems.append(problem)

    app_base_url = os.getenv("APP_BASE_URL", "").strip().rstrip("/")
    problem = _public_url_problem(app_base_url, "APP_BASE_URL", origin_only=True)
    if problem:
        problems.append(
            problem
            if app_base_url
            else (
                "APP_BASE_URL is not set; verification and password-reset links "
                "cannot target the public deployment."
            )
        )
    return problems

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """
    Per-request CSP nonce, standard hardening headers, and request correlation.

    A fresh nonce lets the single-file UI keep its inline <script> blocks
    WITHOUT 'unsafe-inline', so injected script can't execute even if something
    slips past output-escaping. index_page() stamps this same nonce into the
    page's <script> tags. Inline styles stay allowed (they can't exfiltrate
    anything, and there are hundreds of style="" attributes).
    """
    import secrets

    supplied_request_id = request.headers.get("x-request-id", "")
    request_id = (
        supplied_request_id
        if 1 <= len(supplied_request_id) <= 64
        and all(ch.isalnum() or ch in "._-" for ch in supplied_request_id)
        else new_request_id()
    )
    bind_request(request_id)
    # Context variables can outlive a handler on reused worker threads. Reset
    # the identity before authentication binds this request's user so an
    # anonymous request can never inherit a previous request's log attribution.
    bind_user(None)

    nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = nonce

    # The UI authenticates with an HttpOnly cookie. Most existing routers still
    # accept an Authorization header, so translate the cookie inside the trusted
    # ASGI boundary instead of exposing the token to JavaScript. Explicit bearer
    # credentials always win for API clients.
    from api.deps import SESSION_COOKIE_NAME

    cookie_token = request.cookies.get(SESSION_COOKIE_NAME, "")
    has_bearer = bool(request.headers.get("authorization"))
    # Routers sometimes need to preserve explicit-Bearer compatibility without
    # reflecting a cookie credential (or its rotated successor) into
    # JavaScript-readable response data.  Record the pre-injection source on
    # trusted request state before adding the internal Authorization header.
    request.state.auth_from_cookie = bool(cookie_token and not has_bearer)
    csrf_rejected = False
    if cookie_token and not has_bearer:
        if request.method.upper() not in {"GET", "HEAD", "OPTIONS"}:
            origin = request.headers.get("origin", "").rstrip("/")
            fetch_site = request.headers.get("sec-fetch-site", "").lower()
            own_origin = str(request.base_url).rstrip("/")
            permitted_origins = {own_origin, *ALLOWED_ORIGINS}
            csrf_rejected = bool(
                fetch_site == "cross-site"
                or (origin and origin not in permitted_origins)
            )
        if not csrf_rejected:
            request.scope["headers"].append(
                (b"authorization", f"Bearer {cookie_token}".encode("ascii"))
            )

    if csrf_rejected:
        resp = JSONResponse(
            status_code=403,
            content={"detail": "Cross-site authenticated request rejected."},
        )
    else:
        resp = await call_next(request)

    resp.headers["X-Request-ID"] = request_id
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    resp.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    resp.headers["Cross-Origin-Resource-Policy"] = "same-origin"
    resp.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    if request.url.path in {
        "/terms",
        "/privacy",
        "/legal/disclosures",
        "/legal/version",
    }:
        # Policies are public, but revalidate so a policy-version bump or
        # operator-field correction cannot be hidden behind a stale proxy.
        resp.headers["Cache-Control"] = "no-cache"
    else:
        # The index contains a per-response CSP nonce; caching it would turn
        # that nonce into a reusable value. Health and authenticated responses
        # also must never be served stale.
        resp.headers["Cache-Control"] = "no-store"
        resp.headers["Pragma"] = "no-cache"
    resp.headers["Content-Security-Policy"] = (
        f"default-src 'self'; script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "form-action 'self'; object-src 'none'"
    )
    # HSTS only matters over TLS, and setting it on plain http during local
    # development would pin the browser to https://localhost.
    if request.url.scheme == "https":
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


# Serve the frontend from the backend itself (http://127.0.0.1:8000).
# Don't use VS Code Live Server for this app: it reloads the page whenever a
# workspace file changes, and the backend writes files (app.db, chat_memory/,
# cache/) on every request — causing an endless reload/flicker loop.
INDEX_FILE = Path(__file__).resolve().parent / "index.html"


@app.get("/", include_in_schema=False)
def index_page(request: Request):
    # Inject the request's CSP nonce into the page's <script> tags so they match
    # the Content-Security-Policy header set by the security_headers middleware.
    nonce = getattr(request.state, "csp_nonce", "")
    html = INDEX_FILE.read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
    return HTMLResponse(html)


for _router in (auth.router, account.router, portfolio.router, chat.router,
                stocks.router, alerts.router, plan.router, legal.router):
    app.include_router(_router)


# ---------------------------------------------------------------------------
# Startup readiness audit
# ---------------------------------------------------------------------------

def _startup_audit() -> list[str]:
    """Configuration problems that make this instance unfit to serve the public."""
    from api import crypto, market_data, state
    from api.observability import sentry_enabled
    from scripts.backup_db import backup_encryption_ready

    problems: list[str] = []
    problems.extend(market_data.check_production_ready())
    problems.extend(legal.legal_configuration_problems())
    problems.extend(_web_configuration_problems())

    if not os.getenv("EVERGREEN_MASTER_KEY", "").strip():
        problems.append(
            "EVERGREEN_MASTER_KEY is not set, so data at rest is protected by a "
            "development key file next to the database. Set it from a secret manager."
        )
    if not backup_encryption_ready():
        problems.append(
            "BACKUP_ENCRYPTION_KEY is missing, invalid, or reuses "
            "EVERGREEN_MASTER_KEY; production backups require a distinct "
            "authenticated-encryption key."
        )
    if not state.using_shared_store():
        problems.append(
            "REDIS_URL is not set, so rate limits, lockouts and pending 2FA live in "
            "process memory. This instance MUST run a single worker; do not scale out."
        )
    if not os.getenv("RESEND_API_KEY", "").strip():
        problems.append(
            "RESEND_API_KEY is not set, so verification and password-reset email "
            "cannot be delivered. Email verification gates the whole app.")
    if not os.getenv("EMAIL_FROM", "").strip():
        problems.append(
            "EMAIL_FROM is not set to a verified production sender identity."
        )
    if not os.getenv("OPENAI_API_KEY", "").strip():
        problems.append(
            "OPENAI_API_KEY is not set, so the authenticated chat feature cannot run."
        )
    if not sentry_enabled():
        problems.append(
            "Sentry error monitoring did not initialize; set a valid SENTRY_DSN "
            "and verify outbound monitoring connectivity."
        )

    # Touching the cipher here surfaces a bad/missing key at boot rather than on
    # the first user request.
    try:
        crypto.encrypt("startup-probe")
    except crypto.CryptoError as e:
        problems.append(f"At-rest encryption is not usable: {e}")

    return problems


def _verify_encryption_key() -> None:
    """
    Refuse to serve if the master key can't decrypt what's already stored.

    This is fatal in EVERY environment, not just production. Booting with the
    wrong key makes every portfolio read return None, which the UI renders as
    "no portfolio imported yet" — indistinguishable from data loss, and it
    invites the user to import over the top of data that was perfectly fine.
    Far better to stop and say so.
    """
    from api import crypto

    try:
        crypto.verify_key()
    except crypto.CryptoError as e:
        logger.error("ENCRYPTION KEY MISMATCH: %s", e)
        raise RuntimeError(
            f"Refusing to start: {e}\n\n"
            "The database contains data encrypted with a different key. Set "
            "EVERGREEN_MASTER_KEY to the key this database was created with. "
            "If you rotated it, include the previous key as a second "
            "comma-separated value: EVERGREEN_MASTER_KEY=<new>,<old>\n"
            "Starting anyway would show every user an empty portfolio."
        ) from e
