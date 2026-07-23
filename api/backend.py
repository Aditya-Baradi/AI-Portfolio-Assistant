# backend.py (inside api/)

import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import db
from api.langchain_agent import run_portfolio_agent
from api.portfolio_core import parse_portfolio_file, holdings_info

app = FastAPI()
db.init_db()

logger = logging.getLogger("evergreen")

# The backend serves the frontend itself, so only same-host origins are
# needed. Add your real domain here when deploying.
ALLOWED_ORIGINS = [
    "http://127.0.0.1:8000", "http://localhost:8000",
    "http://127.0.0.1:8010", "http://localhost:8010",  # verification port
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request, call_next):
    # A fresh per-request nonce lets the single-file UI keep its inline <script>
    # blocks WITHOUT 'unsafe-inline', so injected script can't execute even if
    # something slips past output-escaping. index_page() stamps this same nonce
    # into the page's <script> tags. Inline styles stay allowed (they can't
    # exfiltrate anything, and there are hundreds of style="" attributes).
    import secrets

    nonce = secrets.token_urlsafe(16)
    request.state.csp_nonce = nonce
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Content-Security-Policy"] = (
        f"default-src 'self'; script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:"
    )
    return resp

# Serve generated charts (equity curves etc.) for the chat UI.
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

# Serve the frontend from the backend itself (http://127.0.0.1:8000).
# Don't use VS Code Live Server for this app: it reloads the page whenever a
# workspace file changes, and the backend writes files (app.db, chat_memory/,
# cache/) on every request — causing an endless reload/flicker loop.
INDEX_FILE = Path(__file__).resolve().parent / "index.html"


@app.get("/")
def index_page(request: Request):
    # Inject the request's CSP nonce into the page's <script> tags so they match
    # the Content-Security-Policy header set by the security_headers middleware.
    nonce = getattr(request.state, "csp_nonce", "")
    html = INDEX_FILE.read_text(encoding="utf-8").replace("__CSP_NONCE__", nonce)
    return HTMLResponse(html)

MAX_UPLOAD_BYTES = 2_000_000
MAX_HOLDINGS = 500
CHAT_DAILY_LIMIT = 50


def _current_user(authorization: str | None) -> dict:
    """Resolve 'Authorization: Bearer <token>' to a user or raise 401."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user = db.user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


def _agent_session_id(user_id: int, chat_id: str) -> str:
    """Chat memory is per (user, chat); the portfolio is shared per user."""
    return f"u{user_id}.c{chat_id}"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    email: str
    password: str
    name: str | None = None  # only used when creating an account


# Brute-force protection: sliding-window limiter per client IP. In-memory is
# fine for a single-process app; swap for Redis if this ever runs on >1 worker.
AUTH_RATE_LIMIT = 10          # attempts allowed...
AUTH_RATE_WINDOW = 300.0      # ...per this many seconds, per IP
_AUTH_ATTEMPTS: dict = {}


def _check_auth_rate(request: Request):
    import time

    ip = _client_ip(request)
    now = time.time()
    recent = [t for t in _AUTH_ATTEMPTS.get(ip, []) if now - t < AUTH_RATE_WINDOW]
    if len(recent) >= AUTH_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait a few minutes and try again.",
        )
    recent.append(now)
    _AUTH_ATTEMPTS[ip] = recent
    if len(_AUTH_ATTEMPTS) > 10_000:  # don't let the map grow unbounded
        _AUTH_ATTEMPTS.clear()


def _client_ip(request: Request) -> str:
    """
    Real client IP, correct behind our reverse proxy.

    In production the app sits behind Caddy on an internal-only port (compose
    uses `expose`, not `ports`), so nothing but Caddy can reach it. Caddy sets
    X-Forwarded-For and APPENDS the true connecting IP as the last entry, so we
    take the RIGHTMOST value: a client can prepend a spoofed IP, but cannot
    forge the entry Caddy adds. With no proxy (local dev) there is no such
    header and we fall back to the socket peer.
    """
    xff = request.headers.get("x-forwarded-for")
    if xff:
        parts = [p.strip() for p in xff.split(",") if p.strip()]
        if parts:
            return parts[-1]
    return request.client.host if request.client else "unknown"


@app.post("/auth/register")
def register(creds: Credentials, request: Request):
    _check_auth_rate(request)
    try:
        user_id = db.register_user(creds.email, creds.password, creds.name)
    except db.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    db.log_event(user_id, "register", _client_ip(request))
    _send_verification(user_id, creds.email.strip().lower())
    token = db.issue_token(user_id)
    return {"token": token, "email": creds.email.strip().lower(),
            "name": db.get_user_name(user_id)}


def _send_verification(user_id: int, email: str) -> bool:
    """Issue a verification token and email it. Best-effort: never blocks the
    caller if email delivery fails (the user can request a resend later)."""
    from api import email_send

    try:
        raw = db.create_email_token(user_id, "verify", ttl_minutes=60 * 24)
        email_send.send_verification_email(email, raw)
        return True
    except Exception:
        # Log server-side so a config/network/provider failure is diagnosable
        # (the caller only sees a generic message). A common cause is the server
        # process starting before RESEND_API_KEY was set in .env — restart it.
        logger.exception("Verification email failed for user %s", user_id)
        return False


# Two-factor state that only lives for a couple of minutes: pending login
# handles (password passed, waiting for the code) and not-yet-confirmed
# secrets during setup. In-memory is fine for a single-process app.
_PENDING_2FA: dict = {}
_PENDING_2FA_SETUP: dict = {}
PENDING_2FA_TTL = 300.0


def _prune_pending():
    import time

    now = time.time()
    for d in (_PENDING_2FA, _PENDING_2FA_SETUP):
        for k in [k for k, (_, ts) in d.items() if now - ts > PENDING_2FA_TTL]:
            d.pop(k, None)


@app.post("/auth/login")
def login(creds: Credentials, request: Request):
    _check_auth_rate(request)
    try:
        user_id = db.verify_login(creds.email, creds.password)
    except db.AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))

    if db.get_totp_secret(user_id):
        import secrets as _secrets
        import time

        _prune_pending()
        pending = _secrets.token_urlsafe(24)
        _PENDING_2FA[pending] = (user_id, time.time())
        return {"requires_2fa": True, "pending": pending, "email": creds.email.strip().lower()}

    db.log_event(user_id, "login", _client_ip(request))
    token = db.issue_token(user_id)
    return {"token": token, "email": creds.email.strip().lower(),
            "name": db.get_user_name(user_id)}


class TwoFAVerify(BaseModel):
    pending: str
    code: str


@app.post("/auth/2fa/verify")
def twofa_verify(body: TwoFAVerify, request: Request):
    _check_auth_rate(request)
    import time

    import pyotp

    _prune_pending()
    entry = _PENDING_2FA.get(body.pending)
    if not entry or time.time() - entry[1] > PENDING_2FA_TTL:
        raise HTTPException(status_code=401, detail="Login expired. Start again.")
    user_id = entry[0]
    secret = db.get_totp_secret(user_id)
    if not secret or not pyotp.TOTP(secret).verify(body.code.strip(), valid_window=1):
        db.log_event(user_id, "failed_2fa", _client_ip(request))
        raise HTTPException(status_code=401, detail="Wrong code. Try again.")
    _PENDING_2FA.pop(body.pending, None)
    db.log_event(user_id, "login", _client_ip(request))
    token = db.issue_token(user_id)
    with db._connect() as conn:
        row = conn.execute("SELECT email, name FROM users WHERE id = ?", (user_id,)).fetchone()
    return {"token": token, "email": row["email"] if row else "",
            "name": row["name"] if row else None}


@app.post("/auth/2fa/setup")
def twofa_setup(authorization: str | None = Header(default=None)):
    """Start enabling 2FA: returns the secret, otpauth URI, and a QR (SVG data URI)."""
    user = _current_user(authorization)
    import base64
    import io
    import time

    import pyotp
    import qrcode
    import qrcode.image.svg

    secret = pyotp.random_base32()
    uri = pyotp.totp.TOTP(secret).provisioning_uri(
        name=user["email"], issuer_name="Evergreen"
    )
    img = qrcode.make(uri, image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    qr_data_uri = "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()

    _prune_pending()
    _PENDING_2FA_SETUP[user["id"]] = (secret, time.time())
    return {"secret": secret, "uri": uri, "qr": qr_data_uri}


class TwoFACode(BaseModel):
    code: str


@app.post("/auth/2fa/enable")
def twofa_enable(body: TwoFACode, request: Request, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    import pyotp

    _prune_pending()
    entry = _PENDING_2FA_SETUP.get(user["id"])
    if not entry:
        raise HTTPException(status_code=400, detail="Start setup first.")
    secret = entry[0]
    if not pyotp.TOTP(secret).verify(body.code.strip(), valid_window=1):
        raise HTTPException(status_code=401, detail="Wrong code. Scan the QR and try again.")
    db.set_totp_secret(user["id"], secret)
    _PENDING_2FA_SETUP.pop(user["id"], None)
    db.log_event(user["id"], "2fa_enabled", _client_ip(request))
    return {"ok": True}


class PasswordBody(BaseModel):
    password: str


@app.post("/auth/2fa/disable")
def twofa_disable(body: PasswordBody, request: Request, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if not db.verify_password(user["id"], body.password):
        raise HTTPException(status_code=401, detail="Wrong password.")
    db.set_totp_secret(user["id"], None)
    db.log_event(user["id"], "2fa_disabled", _client_ip(request))
    return {"ok": True}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        db.revoke_token(authorization[7:].strip())
    return {"ok": True}


@app.get("/me")
def me(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {
        "email": user["email"],
        "name": user.get("name"),
        "has_portfolio": db.get_portfolio(user["id"]) is not None,
        "twofa_enabled": db.get_totp_secret(user["id"]) is not None,
        "email_verified": db.is_email_verified(user["id"]),
    }


# --- email verification + password reset ------------------------------------

@app.get("/verify")
def verify_email(token: str):
    """Land here from the verification email link, then bounce to the app."""
    user_id = db.consume_email_token(token, "verify")
    if not user_id:
        return RedirectResponse(url="/?verified=0", status_code=303)
    db.set_email_verified(user_id)
    return RedirectResponse(url="/?verified=1", status_code=303)


@app.post("/auth/verify/resend")
def resend_verification(request: Request, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    _check_auth_rate(request)
    if db.is_email_verified(user["id"]):
        return {"ok": True, "already_verified": True}
    if not _send_verification(user["id"], user["email"]):
        raise HTTPException(status_code=502, detail="Couldn't send the email right now. Try again shortly.")
    return {"ok": True}


class ForgotBody(BaseModel):
    email: str


@app.post("/auth/forgot")
def forgot_password(body: ForgotBody, request: Request):
    """
    Start a password reset. Always returns 200 with the same message whether or
    not the email exists, so this can't be used to enumerate accounts.
    """
    _check_auth_rate(request)
    from api import email_send

    email = (body.email or "").strip().lower()
    user_id = db.user_id_by_email(email)
    if user_id:
        try:
            raw = db.create_email_token(user_id, "reset", ttl_minutes=30)
            email_send.send_password_reset_email(email, raw)
            db.log_event(user_id, "password_reset_requested", _client_ip(request))
        except Exception:
            # Log server-side (never revealed to the caller — no enumeration).
            logger.exception("Password-reset email failed for user %s", user_id)
    return {"ok": True, "message": "If an account exists for that email, a reset link is on its way."}


class ResetBody(BaseModel):
    token: str
    password: str


@app.post("/auth/reset")
def reset_password(body: ResetBody, request: Request):
    _check_auth_rate(request)
    user_id = db.consume_email_token(body.token, "reset")
    if not user_id:
        raise HTTPException(status_code=400, detail="This reset link is invalid or has expired.")
    try:
        db.set_password(user_id, body.password)
    except db.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # A password reset proves control of the inbox, so treat the email as verified.
    db.set_email_verified(user_id)
    db.log_event(user_id, "password_changed", _client_ip(request))
    return {"ok": True}


# ---------------------------------------------------------------------------
# Account management
# ---------------------------------------------------------------------------

class NameBody(BaseModel):
    name: str


@app.post("/account/name")
def account_name(body: NameBody, authorization: str | None = Header(default=None)):
    """Set or change the display name shown in the app."""
    user = _current_user(authorization)
    name = body.name.strip()[:60]
    if not name:
        raise HTTPException(status_code=400, detail="Please enter a name.")
    db.set_user_name(user["id"], name)
    return {"ok": True, "name": name}


class ChangePassword(BaseModel):
    current_password: str
    new_password: str


@app.post("/auth/change-password")
def change_password(body: ChangePassword, request: Request,
                    authorization: str | None = Header(default=None)):
    """Change the password and sign out every session (a new token is returned)."""
    user = _current_user(authorization)
    try:
        db.change_password(user["id"], body.current_password, body.new_password)
    except db.AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    db.log_event(user["id"], "password_changed", _client_ip(request))
    token = db.issue_token(user["id"])
    return {"ok": True, "token": token}


@app.get("/account/export")
def account_export(authorization: str | None = Header(default=None)):
    """Everything the app stores about this user, as one JSON download."""
    from fastapi.responses import JSONResponse

    user = _current_user(authorization)
    chats = db.list_chats(user["id"])
    for c in chats:
        c["messages"] = db.get_messages(c["id"])
    data = {
        "email": user["email"],
        "name": user.get("name"),
        "profile": db.get_profile(user["id"]),
        "portfolio": db.get_portfolio(user["id"]),
        "watchlist": db.list_watchlist(user["id"]),
        "chats": chats,
        "recent_activity": db.recent_events(user["id"], 100),
    }
    return JSONResponse(
        content=data,
        headers={"Content-Disposition": 'attachment; filename="portfolio-assistant-export.json"'},
    )


@app.post("/account/delete")
def account_delete(body: PasswordBody, request: Request,
                   authorization: str | None = Header(default=None)):
    """Delete the account and everything owned by it. Requires the password."""
    user = _current_user(authorization)
    if not db.verify_password(user["id"], body.password):
        raise HTTPException(status_code=401, detail="Wrong password.")
    # remove per-chat memory files for this user (named u{id}.c{chat}...)
    try:
        from api.portfolio_core import MEMORY_DIR

        for p in Path(MEMORY_DIR).glob(f"u{user['id']}.*"):
            p.unlink(missing_ok=True)
    except Exception:
        pass
    db.delete_user(user["id"])
    return {"ok": True}


@app.get("/account/activity")
def account_activity(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"events": db.recent_events(user["id"], 20)}


# ---------------------------------------------------------------------------
# Watchlist & alerts
# ---------------------------------------------------------------------------

@app.get("/watchlist")
def get_watchlist(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"tickers": db.list_watchlist(user["id"])}


@app.post("/watchlist/{ticker}")
def watch(ticker: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    db.add_watch(user["id"], _valid_ticker(ticker))
    return {"ok": True}


@app.delete("/watchlist/{ticker}")
def unwatch(ticker: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    db.remove_watch(user["id"], _valid_ticker(ticker))
    return {"ok": True}


@app.get("/alerts")
def get_alerts(authorization: str | None = Header(default=None)):
    """
    The user's alerts. Recomputed lazily at most every 6 hours: a big weekly
    portfolio drop, and holdings whose news has turned clearly negative.
    """
    user = _current_user(authorization)
    if db.alerts_due(user["id"]):
        try:
            _compute_alerts(user["id"])
        except Exception:
            pass  # alerts are best-effort; never break the page
        db.mark_alerts_checked(user["id"])
    alerts = db.list_alerts(user["id"])
    return {"alerts": alerts, "unseen": sum(1 for a in alerts if not a["seen"])}


@app.post("/alerts/seen")
def alerts_seen(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    db.mark_alerts_seen(user["id"])
    return {"ok": True}


def _compute_alerts(user_id: int) -> None:
    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix
    from api.sentiment import cached_ticker_sentiment

    pf = db.get_portfolio(user_id)
    if not pf:
        return
    weights = _weights_from_pf(pf)
    if not weights:
        return

    # 1) Portfolio dropped >5% over the last week?
    try:
        import numpy as np

        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(list(weights.keys()), start, end).ffill()
        usable = [t for t in weights if t in prices.columns]
        if usable and len(prices) >= 6:
            w = np.array([weights[t] for t in usable])
            w = w / w.sum()
            norm = prices[usable] / prices[usable].iloc[0]
            curve = norm.dot(w)
            week_chg = float(curve.iloc[-1] / curve.iloc[-6] - 1.0)
            if week_chg < -0.05:
                db.add_alert(user_id,
                             f"Your portfolio is down {abs(week_chg) * 100:.1f}% over the past week.")
    except Exception:
        pass

    # 2) Any holding whose news turned clearly negative?
    for t in list(weights.keys())[:25]:
        try:
            s = cached_ticker_sentiment(t)
            if s["n_headlines"] >= 3 and s["avg_score"] <= -0.15:
                db.add_alert(user_id, f"News about {t} has turned negative "
                                      f"(score {s['avg_score']:+.2f}).", ticker=t)
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    chat_id: str


@app.post("/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    """
    Declared sync so FastAPI runs it in a worker thread — a slow agent call
    (e.g. FinRL optimization) doesn't block the event loop for other users.
    """
    user = _current_user(authorization)
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message.")
    if not req.chat_id:
        raise HTTPException(status_code=422, detail="Missing chat_id.")

    # Every message costs real OpenAI money; cap per user per day.
    if db.count_user_messages_today(user["id"]) >= CHAT_DAILY_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"You've hit today's limit of {CHAT_DAILY_LIMIT} messages. It resets at midnight UTC.",
        )

    sid = _agent_session_id(user["id"], req.chat_id)

    # The agent's tools resolve the portfolio straight from the DB using the
    # user id encoded in this session id (see get_session_portfolio), so there's
    # nothing to copy into process memory here.
    db.touch_chat(user["id"], req.chat_id, first_message=req.message)
    db.add_message(req.chat_id, "user", req.message)

    answer = run_portfolio_agent(req.message, session_id=sid)

    db.add_message(req.chat_id, "assistant", answer)
    return {"answer": answer}


@app.get("/chats")
def chats(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"chats": db.list_chats(user["id"])}


@app.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if not db.chat_belongs_to_user(user["id"], chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"messages": db.get_messages(chat_id)}


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """
    Import (or re-import to update) the signed-in user's portfolio from a
    CSV/JSON file. Stored in SQL so it persists across sessions and restarts.
    """
    user = _current_user(authorization)

    limit_mb = MAX_UPLOAD_BYTES // 1_000_000
    # Reject an oversized upload up front via Content-Length, before reading it
    # into memory (a client can omit/understate this, so we also cap the read).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {limit_mb} MB.")

    # Read at most the limit + 1 byte; if we get that extra byte, it's too big.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {limit_mb} MB.")

    from api.portfolio_core import extract_raw_rows, looks_like_transactions, replay_transactions

    # Brokerage ACTIVITY exports (a list of buys/sells/dividends) need to be
    # replayed, not read as holdings. They're the one file type that contains
    # what the user actually paid, so use them to recover the true cost basis.
    rows = extract_raw_rows(file.filename, content)
    if rows and looks_like_transactions(rows):
        return _import_transactions(user, rows)

    try:
        parsed = parse_portfolio_file(file.filename, content)
    except ValueError as e:
        return {"message": f"Error parsing portfolio file: {e}"}
    except Exception:
        return {"message": "Uploaded file is not valid CSV/JSON portfolio."}

    holdings = parsed.get("holdings", []) if isinstance(parsed, dict) else parsed
    n = len(holdings) if isinstance(holdings, list) else 0
    if n > MAX_HOLDINGS:
        return {"message": f"Too many holdings ({n}). Limit is {MAX_HOLDINGS}."}

    had_one = db.get_portfolio(user["id"]) is not None
    db.save_portfolio(user["id"], parsed)
    db.log_event(user["id"], "portfolio_import", _client_ip(request))

    verb = "Updated" if had_one else "Imported"
    return {"message": f"{verb} your portfolio: {n} holdings."}


def _live_prices(tickers: list) -> dict:
    """Latest market price per ticker (cached download); empty dict on failure."""
    try:
        from datetime import datetime, timedelta

        from api.portfolio_core import _download_adj_close_matrix

        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(tickers, start, end).ffill()
        last = prices.iloc[-1]
        return {t: float(last[t]) for t in prices.columns if last[t] == last[t]}
    except Exception:
        return {}


def _import_transactions(user: dict, rows: list) -> dict:
    """
    Merge a transaction history into the user's portfolio.

    With an existing portfolio: trades set the real average cost (and first
    buy date) on matching holdings, and open positions the portfolio doesn't
    have yet are added. Without one: the replayed positions become the
    portfolio. Shares of existing holdings are never changed, since the
    snapshot import is the authority on what the user holds now.
    """
    from api.portfolio_core import replay_transactions

    result = replay_transactions(rows)
    positions = result["positions"]
    notes = []

    pf = db.get_portfolio(user["id"])
    if pf is None:
        if not positions:
            return {"message": (
                "That looks like a transaction history, but no open positions could be "
                "rebuilt from it (only dividends/sells or rows without prices). "
                "Import a holdings file first, then re-upload this one to set your cost basis."
            )}
        prices = _live_prices(list(positions.keys()))
        holdings = []
        for t, p in positions.items():
            price = prices.get(t)
            value = round(p["shares"] * price, 2) if price else round(p["shares"] * p["avg_cost"], 2)
            h = {"ticker": t, "shares": p["shares"], "purchase_price": p["avg_cost"],
                 "current_dollars": value}
            if price:
                h["close"] = round(price, 4)
            if p["first_buy"]:
                h["purchase_date"] = p["first_buy"]
            holdings.append(h)
        db.save_portfolio(user["id"], {"holdings": holdings})
        notes.append(f"Rebuilt {len(holdings)} positions from {result['n_trades']} trades.")
    else:
        holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
        by_ticker = {}
        for h in holdings if isinstance(holdings, list) else []:
            if isinstance(h, dict):
                tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
                if tkr:
                    by_ticker.setdefault(str(tkr).strip().upper(), []).append(h)

        updated, added = [], []
        for t, p in positions.items():
            if t in by_ticker:
                for h in by_ticker[t]:
                    h["purchase_price"] = p["avg_cost"]
                    if p["first_buy"]:
                        h["purchase_date"] = p["first_buy"]
                updated.append(t)
            else:
                price = _live_prices([t]).get(t)
                value = round(p["shares"] * price, 2) if price else round(p["shares"] * p["avg_cost"], 2)
                h = {"ticker": t, "shares": p["shares"], "purchase_price": p["avg_cost"],
                     "current_dollars": value}
                if p["first_buy"]:
                    h["purchase_date"] = p["first_buy"]
                holdings.append(h)
                added.append(t)

        if not updated and not added:
            return {"message": (
                "That transaction file had no trades with usable prices, so nothing changed. "
                f"({result['n_cash_events']} dividend/cash rows were found; those don't affect cost basis.)"
            )}
        db.save_portfolio(user["id"], pf if isinstance(pf, dict) else {"holdings": holdings})
        if updated:
            notes.append(f"Set real cost basis for {', '.join(sorted(updated))}.")
        if added:
            notes.append(f"Added {', '.join(sorted(added))}.")

    if result["closed_tickers"]:
        notes.append(f"Skipped closed positions: {', '.join(result['closed_tickers'])}.")
    if result["skipped_tickers"]:
        notes.append(f"Some rows for {', '.join(result['skipped_tickers'])} had no price and were ignored.")
    if result["n_cash_events"]:
        notes.append(f"{result['n_cash_events']} dividend/cash rows ignored.")

    return {"message": "Transaction history imported. " + " ".join(notes)}


class HoldingUpdate(BaseModel):
    ticker: str
    shares: float
    avg_cost: float | None = None


def _valid_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t or len(t) > 10 or not all(c.isalnum() or c in ".-" for c in t):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol.")
    return t


@app.post("/portfolio/holding")
def upsert_holding(update: HoldingUpdate, authorization: str | None = Header(default=None)):
    """
    Add a stock to the portfolio or change an existing position's share count
    (and optionally its average cost). New tickers are validated against
    yfinance so typos don't pollute the portfolio.
    """
    user = _current_user(authorization)
    ticker = _valid_ticker(update.ticker)
    if update.shares <= 0:
        raise HTTPException(status_code=422, detail="Shares must be a positive number.")
    if update.avg_cost is not None and update.avg_cost <= 0:
        raise HTTPException(status_code=422, detail="Average cost must be positive.")

    pf = db.get_portfolio(user["id"]) or {"holdings": []}
    if not isinstance(pf, dict):
        pf = {"holdings": pf}
    holdings = pf.get("holdings", [])
    existing = [h for h in holdings if isinstance(h, dict)
                and str(h.get("ticker") or h.get("symbol") or h.get("tic") or "").strip().upper() == ticker]

    price = _live_prices([ticker]).get(ticker)
    if not price and not existing:
        raise HTTPException(status_code=422, detail=f"No price data found for '{ticker}'. Is the symbol right?")

    row = {"ticker": ticker, "shares": float(update.shares)}
    if price:
        row["close"] = round(price, 4)
        row["current_dollars"] = round(update.shares * price, 2)
    if update.avg_cost is not None:
        row["purchase_price"] = float(update.avg_cost)
    elif existing:
        # keep the cost/date the user already had on this position
        old = existing[0]
        if old.get("purchase_price"):
            row["purchase_price"] = old["purchase_price"]
        if old.get("purchase_date"):
            row["purchase_date"] = old["purchase_date"]
        if "current_dollars" not in row and old.get("close"):
            row["close"] = old["close"]
            row["current_dollars"] = round(update.shares * float(old["close"]), 2)
    if "current_dollars" not in row:
        basis = row.get("purchase_price")
        row["current_dollars"] = round(update.shares * basis, 2) if basis else 0.0

    pf["holdings"] = [h for h in holdings if h not in existing] + [row]
    if len(pf["holdings"]) > MAX_HOLDINGS:
        raise HTTPException(status_code=422, detail=f"Portfolio is at the {MAX_HOLDINGS}-holding limit.")
    db.save_portfolio(user["id"], pf)
    return {"ok": True, "ticker": ticker, "action": "updated" if existing else "added"}


@app.delete("/portfolio/holding/{ticker}")
def delete_holding(ticker: str, authorization: str | None = Header(default=None)):
    """Remove a stock from the portfolio entirely."""
    user = _current_user(authorization)
    ticker = _valid_ticker(ticker)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    if not isinstance(pf, dict):
        pf = {"holdings": pf}
    holdings = pf.get("holdings", [])
    kept = [h for h in holdings if not (isinstance(h, dict)
            and str(h.get("ticker") or h.get("symbol") or h.get("tic") or "").strip().upper() == ticker)]
    if len(kept) == len(holdings):
        raise HTTPException(status_code=404, detail=f"No holding '{ticker}' in your portfolio.")
    pf["holdings"] = kept
    db.save_portfolio(user["id"], pf)
    return {"ok": True, "ticker": ticker, "action": "removed"}


def _weights_from_pf(pf) -> dict:
    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        if tkr:
            try:
                weights[str(tkr).upper()] = weights.get(str(tkr).upper(), 0.0) + float(val)
            except Exception:
                continue
    return {t: v for t, v in weights.items() if v > 0}


@app.get("/portfolio/sentiment")
def portfolio_sentiment(authorization: str | None = Header(default=None)):
    """
    Per-holding news sentiment for the signed-in user: each stock gets the
    average sentiment score of its recent financial-news headlines plus a
    Buy / Hold / Sell signal derived from that score. Values shown are live
    (shares x current yfinance price) when price data is available.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    from api.sentiment import cached_ticker_sentiment, sentiment_backend, signal_from_score

    info = holdings_info(pf)
    if not info:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    # Live value per holding; fall back to the imported snapshot value.
    values = {t: meta["stored_value"] for t, meta in info.items()}
    shares_now = {t: meta["shares"] for t, meta in info.items()}
    try:
        import pandas as pd

        from api.data_cache import cached_splits, split_factor
        from api.portfolio_core import _download_adj_close_matrix

        end = datetime.today().strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(list(info.keys()), "2020-01-01", end)
        last = prices.ffill().iloc[-1]
        splits = cached_splits(list(info.keys()))
        for t, meta in info.items():
            shares = meta["shares"]
            if meta["purchase_date"] is not None and shares:
                shares *= split_factor(splits.get(t, []), meta["purchase_date"].strftime("%Y-%m-%d"))
            shares_now[t] = shares
            cur = float(last[t]) if t in last.index and not pd.isna(last[t]) else None
            if shares and cur:
                values[t] = shares * cur
    except Exception:
        pass

    # Trailing-12-month dividend income per holding (best-effort).
    incomes = {}
    try:
        from api.data_cache import cached_dividends_ttm

        divs = cached_dividends_ttm(list(info.keys()))
        for t in info:
            if shares_now.get(t) and divs.get(t):
                incomes[t] = round(shares_now[t] * divs[t], 2)
    except Exception:
        pass

    # One news fetch per ticker is network-bound: fan out, and lean on the
    # 15-minute sentiment cache so repeat loads are instant.
    def one(t):
        try:
            return cached_ticker_sentiment(t)
        except Exception:
            return {"ticker": t, "avg_score": 0.0, "label": "No data", "n_headlines": 0}

    tickers = list(info.keys())
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, tickers))

    rows = [
        {
            "ticker": t,
            "value": round(values.get(t) or 0.0, 2),
            "income": incomes.get(t, 0.0),
            "score": res["avg_score"],
            "label": res["label"],
            "n_headlines": res["n_headlines"],
            "signal": signal_from_score(res["avg_score"], res["n_headlines"]),
        }
        for t, res in zip(tickers, results)
    ]
    rows.sort(key=lambda r: -(r["value"] or 0))
    return {
        "as_of": datetime.today().strftime("%Y-%m-%d"),
        "backend": sentiment_backend(),
        "total_value": round(sum(r["value"] or 0 for r in rows), 2),
        "total_income": round(sum(r["income"] for r in rows), 2),
        "holdings": rows,
    }


# ---------------------------------------------------------------------------
# Investor profile + plan (recommendations, retirement outlook)
# ---------------------------------------------------------------------------

class ProfileUpdate(BaseModel):
    years_to_retirement: int
    risk_tolerance: int
    max_volatility_pct: float
    goal: str = "balanced"
    monthly_contribution: float = 0.0
    goal_amount: float = 0.0


@app.get("/profile")
def get_profile(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return db.get_profile(user["id"]) or {}


@app.post("/profile")
def set_profile(update: ProfileUpdate, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if not (1 <= update.years_to_retirement <= 60):
        raise HTTPException(status_code=422, detail="Years to retirement must be between 1 and 60.")
    if not (1 <= update.risk_tolerance <= 10):
        raise HTTPException(status_code=422, detail="Risk tolerance must be between 1 and 10.")
    if not (5 <= update.max_volatility_pct <= 80):
        raise HTTPException(status_code=422, detail="Max volatility must be between 5% and 80%.")
    if update.goal not in ("growth", "balanced", "income"):
        raise HTTPException(status_code=422, detail="Goal must be growth, balanced, or income.")
    if update.monthly_contribution < 0:
        raise HTTPException(status_code=422, detail="Monthly contribution can't be negative.")
    if update.goal_amount < 0:
        raise HTTPException(status_code=422, detail="Goal amount can't be negative.")
    profile = update.model_dump()
    db.save_profile(user["id"], profile)
    return {"ok": True, **profile}


def _portfolio_mu_sigma(pf):
    """(mu, sigma, value0) estimated from 3y of the portfolio's history, or None."""
    try:
        from datetime import datetime, timedelta

        import numpy as np

        from api.portfolio_core import _download_adj_close_matrix, _perf_metrics, RF_ANNUAL_DEFAULT

        weights = _weights_from_pf(pf)
        if not weights:
            return None
        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(list(weights.keys()), start, end)
        usable = [t for t in weights if t in prices.columns]
        if not usable:
            return None
        w = np.array([weights[t] for t in usable])
        w = w / w.sum()
        rets = prices[usable].pct_change().dropna().dot(w)
        m = _perf_metrics(rets, RF_ANNUAL_DEFAULT)
        return m["CAGR"], m["volatility"], sum(weights.values())
    except Exception:
        return None


@app.get("/plan/recommendations")
def plan_recommendations(authorization: str | None = Header(default=None)):
    """Stocks that fit the saved profile, plus a check on the current portfolio's risk."""
    user = _current_user(authorization)
    profile = db.get_profile(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Save your plan settings first.")

    from api.recommend import recommend_stocks

    pf = db.get_portfolio(user["id"])
    owned = set(_weights_from_pf(pf).keys()) if pf else set()
    try:
        recs = recommend_stocks(profile, owned=owned)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not screen stocks: {e}")

    check = None
    if pf:
        est = _portfolio_mu_sigma(pf)
        if est:
            _mu, sigma, _v0 = est
            check = {
                "portfolio_volatility": round(sigma, 4),
                "max_volatility_pct": profile["max_volatility_pct"],
                "too_risky": sigma * 100 > profile["max_volatility_pct"],
            }
    return {"recommendations": recs, "portfolio_check": check}


@app.get("/plan/retirement")
def plan_retirement(authorization: str | None = Header(default=None)):
    """
    Monte Carlo outlook to retirement driven by the PLAN's risk settings, so
    moving the risk slider visibly changes the outcome.

    The risk score sets the volatility to simulate (clamped by the user's max),
    and expected return follows the classic risk/reward trade-off: more swings,
    more expected growth (3% base + 0.40 per unit of volatility, capped at
    15%/yr). The imported portfolio only provides the starting value.
    """
    user = _current_user(authorization)
    profile = db.get_profile(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Save your plan settings first.")

    from api.recommend import monthly_needed_for, retirement_paths, target_volatility

    sigma = min(target_volatility(profile["risk_tolerance"]),
                float(profile["max_volatility_pct"]) / 100.0)
    mu = min(0.03 + 0.40 * sigma, 0.15)

    value0 = 0.0
    pf = db.get_portfolio(user["id"])
    if pf:
        value0 = sum(_weights_from_pf(pf).values())

    goal = float(profile.get("goal_amount") or 0.0)
    monthly = profile.get("monthly_contribution", 0.0)
    out = retirement_paths(mu, sigma, value0, monthly,
                           profile["years_to_retirement"], goal=goal or None)
    out["based_on"] = "your plan"

    # If the goal looks shaky, say what monthly amount would make it likely.
    if goal > 0 and out.get("p_goal", 1.0) < 0.6:
        needed = monthly_needed_for(goal, mu, sigma, value0, profile["years_to_retirement"])
        if needed is not None and needed > monthly:
            out["monthly_needed"] = needed
    return out


@app.get("/portfolio/whatif")
def portfolio_whatif(authorization: str | None = Header(default=None)):
    """
    "What if I had held the suggested mix instead?" comparison.

    The suggested mix comes from the conservative optimizer (min-vol + cap,
    blended toward equal weight) fitted on data from 3 years ago to 1 year
    ago, then BOTH portfolios are compared over the most recent year the
    optimizer never saw — an out-of-sample test, not a curve fit.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    weights = _weights_from_pf(pf)
    if len(weights) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 holdings to compare allocations.")

    from datetime import datetime, timedelta

    from api.portfolio_core import backtest_vs_benchmark
    from api.predict_agent import run_conservative_optimization

    today = datetime.today()
    test_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    try:
        opt_w, _prices, method = run_conservative_optimization(
            list(weights.keys()), train_start, test_start
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not optimize: {e}")

    try:
        cur = backtest_vs_benchmark(weights, test_start, end, return_curves=True)
        opt = backtest_vs_benchmark(opt_w, test_start, end, return_curves=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backtest failed: {e}")

    # Align the three curves on shared dates.
    cur_map = dict(zip(cur["curves"]["dates"], cur["curves"]["portfolio"]))
    opt_map = dict(zip(opt["curves"]["dates"], opt["curves"]["portfolio"]))
    ben_map = dict(zip(cur["curves"]["dates"], cur["curves"]["benchmark"]))
    dates = [d for d in cur["curves"]["dates"] if d in opt_map]

    return {
        "start": test_start,
        "end": end,
        "method": method,
        "dates": dates,
        "curves": {
            "current": [cur_map[d] for d in dates],
            "optimized": [opt_map[d] for d in dates],
            "benchmark": [ben_map[d] for d in dates],
        },
        "returns": {
            "current": cur["portfolio"]["total_return"],
            "optimized": opt["portfolio"]["total_return"],
            "benchmark": cur["benchmark_metrics"]["total_return"],
        },
        "optimized_weights": dict(sorted(
            ((t, round(w, 4)) for t, w in opt_w.items()), key=lambda kv: -kv[1]
        )),
    }


@app.get("/portfolio/rebalance")
def portfolio_rebalance(authorization: str | None = Header(default=None)):
    """
    Dollar trades that move the current portfolio to the conservative
    optimizer's target mix (same target the What-if tab compares against).
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    weights = _weights_from_pf(pf)
    if len(weights) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 holdings.")

    from datetime import datetime, timedelta

    from api.predict_agent import run_conservative_optimization
    from api.recommend import rebalance_trades

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
    try:
        target, _prices, _method = run_conservative_optimization(list(weights.keys()), start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not optimize: {e}")

    total = sum(weights.values())
    trades = rebalance_trades(weights, target, total)
    return {"total_value": round(total, 2), "trades": trades}


@app.get("/portfolio/diversification")
def portfolio_diversification(authorization: str | None = Header(default=None)):
    """How the money is spread across stocks and sectors, with concentration warnings."""
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    weights = _weights_from_pf(pf)
    if not weights:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    from api.data_cache import cached_sectors

    total = sum(weights.values())
    stock_w = sorted(((t, v / total) for t, v in weights.items()), key=lambda kv: -kv[1])

    try:
        sectors_raw = cached_sectors(list(weights.keys()))
    except Exception:
        sectors_raw = {}
    sector_w: dict = {}
    for t, v in weights.items():
        s = sectors_raw.get(t) or "Unknown"
        sector_w[s] = sector_w.get(s, 0.0) + v / total
    sector_list = sorted(sector_w.items(), key=lambda kv: -kv[1])

    warnings_out = []
    if stock_w and stock_w[0][1] > 0.25:
        warnings_out.append(
            f"{stock_w[0][0]} is {stock_w[0][1] * 100:.0f}% of your money. If that one "
            f"stock has a bad year, your whole portfolio does too."
        )
    if sector_list and sector_list[0][0] != "Unknown" and sector_list[0][1] > 0.40:
        warnings_out.append(
            f"{sector_list[0][1] * 100:.0f}% of your money is in {sector_list[0][0]}. "
            f"Sectors often fall together."
        )

    return {
        "stocks": [{"ticker": t, "weight": round(w, 4)} for t, w in stock_w],
        "sectors": [{"sector": s, "weight": round(w, 4)} for s, w in sector_list],
        "warnings": warnings_out,
    }


@app.get("/stocks/news/{ticker}")
def stock_news(ticker: str, authorization: str | None = Header(default=None)):
    """
    Recent scored headlines for one ticker (the drill-down under a holding).
    Served from the same cached fetch that produced the row's score, so the
    headlines always match the number shown.
    """
    _current_user(authorization)
    ticker = _valid_ticker(ticker)

    from api.sentiment import cached_ticker_sentiment

    try:
        res = cached_ticker_sentiment(ticker)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch news: {e}")
    return {
        "ticker": ticker,
        "headlines": [
            {"title": h["title"], "publisher": h["publisher"], "url": h["url"],
             "score": h["score"], "label": h["label"]}
            for h in res.get("headlines", [])
        ],
    }


@app.get("/portfolio/history")
def portfolio_history(authorization: str | None = Header(default=None)):
    """
    What actually happened: the portfolio's value since the earliest purchase,
    against putting the same dollars into the S&P 500 (SPY) on the same dates.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    from datetime import datetime

    import pandas as pd

    from api.portfolio_core import _download_adj_close_matrix

    info = holdings_info(pf)
    dated = {t: m for t, m in info.items() if m["purchase_date"] is not None and m["shares"]}
    if not dated:
        raise HTTPException(status_code=422, detail="No purchase dates in the portfolio.")

    start = min(m["purchase_date"] for m in dated.values()).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix(list(dated.keys()) + ["SPY"], start, end).ffill()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")
    if getattr(prices.index, "tz", None) is not None:
        prices.index = prices.index.tz_localize(None)
    if "SPY" not in prices.columns or prices.empty:
        raise HTTPException(status_code=502, detail="Benchmark data unavailable.")

    # Adjusted closes bake in splits, so current shares x adj close is the
    # position's value at any time after purchase.
    port = pd.Series(0.0, index=prices.index)
    spy = pd.Series(0.0, index=prices.index)
    spy_px = prices["SPY"]
    for t, m in dated.items():
        if t not in prices.columns:
            continue
        col = prices[t]
        buy = m["purchase_date"]
        after = col.index >= buy
        if not after.any():
            continue
        first_idx = col.index[after][0]
        px0 = float(col.loc[first_idx])
        if px0 <= 0 or pd.isna(px0):
            continue
        shares = float(m["shares"])
        port.loc[after] += shares * col[after]
        # Same dollars into SPY on the same day.
        dollars = shares * px0
        spy0 = float(spy_px.loc[first_idx])
        if spy0 > 0:
            spy.loc[after] += dollars * (spy_px[after] / spy0)

    mask = port > 0
    port, spy = port[mask], spy[mask]
    if len(port) < 2:
        raise HTTPException(status_code=422, detail="Not enough history to chart.")

    step = max(1, len(port) // 300)  # keep the payload small
    idx = list(range(0, len(port), step))
    if idx[-1] != len(port) - 1:
        idx.append(len(port) - 1)
    return {
        "dates": [port.index[i].strftime("%Y-%m-%d") for i in idx],
        "portfolio": [round(float(port.iloc[i]), 2) for i in idx],
        "benchmark": [round(float(spy.iloc[i]), 2) for i in idx],
        "since": start,
        "portfolio_now": round(float(port.iloc[-1]), 2),
        "benchmark_now": round(float(spy.iloc[-1]), 2),
    }


def _stock_projection(col, ticker: str) -> dict:
    """12-month GBM projection for one ticker from ~3 years of daily closes."""
    import math

    from api.portfolio_core import _perf_metrics, RF_ANNUAL_DEFAULT

    col = col.dropna()
    if len(col) < 60:
        raise ValueError("Not enough price history to estimate a range.")
    m = _perf_metrics(col.pct_change().dropna(), RF_ANNUAL_DEFAULT)
    price0 = float(col.iloc[-1])
    mu, sigma = m["CAGR"], m["volatility"]

    months = list(range(0, 13))
    median, optimistic, pessimistic = [], [], []
    for mo in months:
        t = mo / 12.0
        med = price0 * (1.0 + mu) ** t
        band = sigma * math.sqrt(t)
        median.append(round(med, 2))
        optimistic.append(round(med * math.exp(band), 2))
        pessimistic.append(round(med * math.exp(-band), 2))

    return {
        "ticker": ticker,
        "price": round(price0, 2),
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "stats": {"est_annual_return": round(mu, 4), "volatility": round(sigma, 4)},
    }


@app.get("/stocks/projections")
def stocks_projections(authorization: str | None = Header(default=None)):
    """
    Per-stock 12-month projections: watched tickers first, then every holding
    (biggest first). Works with only a watchlist, only a portfolio, or both.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    weights = _weights_from_pf(pf) if pf else {}
    watched = db.list_watchlist(user["id"])
    if not weights and not watched:
        raise HTTPException(status_code=404, detail="No portfolio or watchlist yet.")

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    tickers = list(dict.fromkeys(watched + sorted(weights, key=lambda k: -weights[k])))
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix(tickers, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")

    out = []
    for t in tickers:
        if t in prices.columns:
            try:
                proj = _stock_projection(prices[t], t)
                proj["watched"] = t in watched
                out.append(proj)
            except Exception:
                continue  # not enough history for this one; skip it
    return {"as_of": end, "stocks": out, "watchlist": watched}


@app.get("/stocks/projection/{ticker}")
def stock_projection(ticker: str, authorization: str | None = Header(default=None)):
    """12-month price projection for ANY ticker (the Stocks-page search)."""
    _current_user(authorization)
    ticker = _valid_ticker(ticker)

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix([ticker], start, end)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker}'. Is the symbol right?")
    if ticker not in prices.columns:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker}'. Is the symbol right?")
    try:
        return _stock_projection(prices[ticker], ticker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/portfolio/projection")
def portfolio_projection(authorization: str | None = Header(default=None)):
    """
    Estimated 12-month projection for the signed-in user's portfolio.

    Uses 3 years of (cached) price history to estimate the portfolio's
    annualized return and volatility, then projects the current value forward
    with a median path and +/-1 sigma bands (lognormal/GBM approximation).
    Educational estimate, not a forecast guarantee.
    """
    user = _current_user(authorization)

    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    import math
    from datetime import datetime, timedelta

    import numpy as np

    from api.portfolio_core import _download_adj_close_matrix, _perf_metrics, RF_ANNUAL_DEFAULT

    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        if tkr:
            try:
                weights[str(tkr).upper()] = weights.get(str(tkr).upper(), 0.0) + float(val)
            except Exception:
                continue
    weights = {t: v for t, v in weights.items() if v > 0}
    if not weights:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    value0 = sum(weights.values())
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    try:
        prices = _download_adj_close_matrix(list(weights.keys()), start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")

    usable = [t for t in weights if t in prices.columns]
    if not usable:
        raise HTTPException(status_code=422, detail="No price history for the portfolio tickers.")
    w = np.array([weights[t] for t in usable])
    w = w / w.sum()
    rets = prices[usable].pct_change().dropna().dot(w)
    m = _perf_metrics(rets, RF_ANNUAL_DEFAULT)

    # Anchor the projection at the LIVE portfolio value (shares x current
    # yfinance price, split-adjusted), not the snapshot stored in the file.
    try:
        import pandas as pd

        from api.data_cache import cached_splits, split_factor

        info = holdings_info(pf)
        splits = cached_splits(list(info.keys()))
        last = prices.ffill().iloc[-1]
        live_total = 0.0
        for t, meta in info.items():
            shares = meta["shares"]
            if meta["purchase_date"] is not None and shares:
                shares *= split_factor(splits.get(t, []), meta["purchase_date"].strftime("%Y-%m-%d"))
            cur = float(last[t]) if t in last.index and not pd.isna(last[t]) else None
            live_total += shares * cur if (shares and cur) else meta["stored_value"]
        if live_total > 0:
            value0 = live_total
    except Exception:
        pass  # fall back to the stored snapshot value

    mu, sigma = m["CAGR"], m["volatility"]
    months = list(range(0, 13))
    median, optimistic, pessimistic = [], [], []
    for mo in months:
        t = mo / 12.0
        med = value0 * (1.0 + mu) ** t
        band = sigma * math.sqrt(t)
        median.append(round(med, 2))
        optimistic.append(round(med * math.exp(band), 2))
        pessimistic.append(round(med * math.exp(-band), 2))

    return {
        "current_value": round(value0, 2),
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "stats": {
            "est_annual_return": round(mu, 4),
            "volatility": round(sigma, 4),
            "sharpe": round(m["Sharpe"], 2),
            "max_drawdown": round(m["max_drawdown"], 4),
            "history_years": 3,
            "n_holdings": len(usable),
        },
    }
