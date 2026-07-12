"""
SQLite storage for accounts, sessions, portfolios, and chat history.

Security model:
- Passwords are hashed with bcrypt (per-password salt, cost factor 12).
  Plaintext passwords are never stored or logged.
- Auth tokens are 256-bit random values (secrets.token_urlsafe) with a
  7-day expiry, checked on every authenticated request.
- All SQL uses parameterized queries.

Every function opens its own short-lived connection, so the module is safe
to call from FastAPI's threadpool without sharing connections across threads.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import bcrypt

DB_PATH = Path("app.db")

TOKEN_TTL_DAYS = 7
MAX_CHATS_PER_USER = 10
MIN_PASSWORD_LEN = 8

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    name          TEXT,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolios (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    data       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS profiles (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    data       TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chats (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title      TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id    TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind       TEXT NOT NULL,
    ip         TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS watchlist (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker  TEXT NOT NULL,
    PRIMARY KEY (user_id, ticker)
);
CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    ticker     TEXT,
    message    TEXT NOT NULL,
    seen       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alert_checks (
    user_id    INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    checked_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chats_user    ON chats(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
CREATE INDEX IF NOT EXISTS idx_events_user   ON events(user_id, id DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_user   ON alerts(user_id, id DESC);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # Migrations for databases created before these columns existed.
        try:
            conn.execute("ALTER TABLE users ADD COLUMN totp_secret TEXT")
        except sqlite3.OperationalError:
            pass  # column already there
        try:
            conn.execute("ALTER TABLE users ADD COLUMN name TEXT")
        except sqlite3.OperationalError:
            pass  # column already there
        # Tokens are now stored hashed (64 hex chars); revoke any legacy
        # plaintext tokens so they can't be used straight from a stolen DB.
        conn.execute("DELETE FROM auth_tokens WHERE length(token) <> 64")


def _now() -> str:
    # Microsecond precision: chat pruning orders by this, and several chats
    # can be created within the same second.
    return datetime.utcnow().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Users & auth
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Raised for registration/login failures with a user-safe message."""


def register_user(email: str, password: str, name: str | None = None) -> int:
    email = (email or "").strip().lower()
    if "@" not in email or len(email) < 5:
        raise AuthError("Please enter a valid email address.")
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    name = (name or "").strip()[:60] or None

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                (email, pw_hash.decode("ascii"), name, _now()),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise AuthError("An account with that email already exists.")


def get_user_name(user_id: int) -> str | None:
    with _connect() as conn:
        row = conn.execute("SELECT name FROM users WHERE id = ?", (user_id,)).fetchone()
        return row["name"] if row else None


def set_user_name(user_id: int, name: str | None) -> None:
    name = (name or "").strip()[:60] or None
    with _connect() as conn:
        conn.execute("UPDATE users SET name = ? WHERE id = ?", (name, user_id))


def verify_login(email: str, password: str) -> int:
    """Return the user id on success; raise AuthError (generic message) otherwise."""
    email = (email or "").strip().lower()
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()

    # bcrypt.checkpw is constant-time; run it even for unknown emails so the
    # timing doesn't reveal whether the account exists.
    stored = row["password_hash"].encode("ascii") if row else bcrypt.gensalt()
    ok = False
    try:
        ok = bcrypt.checkpw((password or "").encode("utf-8"), stored)
    except ValueError:
        ok = False
    if not row or not ok:
        raise AuthError("Invalid email or password.")
    return int(row["id"])


def _hash_token(token: str) -> str:
    """Tokens are stored as SHA-256 so a copied app.db can't impersonate users."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(days=TOKEN_TTL_DAYS)).isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (_hash_token(token), user_id, expires),
        )
    return token


def user_for_token(token: str) -> dict | None:
    """Return {'id', 'email', 'name'} for a valid token, else None. Expired tokens are purged."""
    if not token:
        return None
    hashed = _hash_token(token)
    with _connect() as conn:
        row = conn.execute(
            """SELECT u.id, u.email, u.name, t.expires_at FROM auth_tokens t
               JOIN users u ON u.id = t.user_id WHERE t.token = ?""",
            (hashed,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] < _now():
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (hashed,))
            return None
        return {"id": int(row["id"]), "email": row["email"], "name": row["name"]}


def revoke_token(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (_hash_token(token),))


def revoke_all_tokens(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))


def change_password(user_id: int, current: str, new: str) -> None:
    """Verify the current password, set the new one, sign out everywhere."""
    if len(new or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"New password must be at least {MIN_PASSWORD_LEN} characters.")
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row or not bcrypt.checkpw((current or "").encode("utf-8"),
                                     row["password_hash"].encode("ascii")):
        raise AuthError("Current password is incorrect.")
    new_hash = bcrypt.hashpw(new.encode("utf-8"), bcrypt.gensalt(rounds=12))
    with _connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (new_hash.decode("ascii"), user_id))
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))


def verify_password(user_id: int, password: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row:
        return False
    try:
        return bcrypt.checkpw((password or "").encode("utf-8"),
                              row["password_hash"].encode("ascii"))
    except ValueError:
        return False


def delete_user(user_id: int) -> None:
    """Remove the account and everything owned by it (FKs cascade)."""
    with _connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))


# --- two-factor auth (TOTP) -------------------------------------------------

def set_totp_secret(user_id: int, secret: str | None) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET totp_secret = ? WHERE id = ?", (secret, user_id))


def get_totp_secret(user_id: int) -> str | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT totp_secret FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return row["totp_secret"] if row else None


# --- audit log ---------------------------------------------------------------

def log_event(user_id: int, kind: str, ip: str | None = None) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO events (user_id, kind, ip, created_at) VALUES (?, ?, ?, ?)",
            (user_id, kind, ip, _now()),
        )
        # keep the log bounded per user
        conn.execute(
            """DELETE FROM events WHERE user_id = ? AND id NOT IN (
                   SELECT id FROM events WHERE user_id = ? ORDER BY id DESC LIMIT 100)""",
            (user_id, user_id),
        )


def recent_events(user_id: int, limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kind, ip, created_at FROM events WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# --- watchlist ---------------------------------------------------------------

def add_watch(user_id: int, ticker: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (user_id, ticker) VALUES (?, ?)",
            (user_id, ticker.upper()),
        )


def remove_watch(user_id: int, ticker: str) -> None:
    with _connect() as conn:
        conn.execute(
            "DELETE FROM watchlist WHERE user_id = ? AND ticker = ?",
            (user_id, ticker.upper()),
        )


def list_watchlist(user_id: int) -> list[str]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ticker FROM watchlist WHERE user_id = ? ORDER BY ticker", (user_id,)
        ).fetchall()
    return [r["ticker"] for r in rows]


# --- alerts ------------------------------------------------------------------

def alerts_due(user_id: int, hours: float = 6.0) -> bool:
    """True when this user's alerts haven't been recomputed recently."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT checked_at FROM alert_checks WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return True
    cutoff = (datetime.utcnow() - timedelta(hours=hours)).isoformat(timespec="microseconds")
    return row["checked_at"] < cutoff


def mark_alerts_checked(user_id: int) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO alert_checks (user_id, checked_at) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE SET checked_at = excluded.checked_at""",
            (user_id, _now()),
        )


def add_alert(user_id: int, message: str, ticker: str | None = None) -> None:
    with _connect() as conn:
        # Don't stack duplicates of an alert the user hasn't seen yet.
        dup = conn.execute(
            "SELECT 1 FROM alerts WHERE user_id = ? AND message = ? AND seen = 0",
            (user_id, message),
        ).fetchone()
        if dup:
            return
        conn.execute(
            "INSERT INTO alerts (user_id, ticker, message, created_at) VALUES (?, ?, ?, ?)",
            (user_id, ticker, message, _now()),
        )
        conn.execute(
            """DELETE FROM alerts WHERE user_id = ? AND id NOT IN (
                   SELECT id FROM alerts WHERE user_id = ? ORDER BY id DESC LIMIT 50)""",
            (user_id, user_id),
        )


def list_alerts(user_id: int, limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, ticker, message, seen, created_at FROM alerts
               WHERE user_id = ? ORDER BY id DESC LIMIT ?""",
            (user_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def mark_alerts_seen(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("UPDATE alerts SET seen = 1 WHERE user_id = ?", (user_id,))


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

def save_portfolio(user_id: int, parsed: dict) -> None:
    """Insert or replace the user's portfolio (re-importing a file updates it)."""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO portfolios (user_id, data, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET data = excluded.data,
                                                  updated_at = excluded.updated_at""",
            (user_id, json.dumps(parsed, ensure_ascii=False), _now()),
        )


def get_portfolio(user_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM portfolios WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["data"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Investor profiles (retirement horizon, risk tolerance, goals)
# ---------------------------------------------------------------------------

def save_profile(user_id: int, profile: dict) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO profiles (user_id, data, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET data = excluded.data,
                                                  updated_at = excluded.updated_at""",
            (user_id, json.dumps(profile, ensure_ascii=False), _now()),
        )


def get_profile(user_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row["data"])
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Chats & messages
# ---------------------------------------------------------------------------

def touch_chat(user_id: int, chat_id: str, first_message: str) -> None:
    """
    Create the chat on first use (title = trimmed first message) or bump its
    updated_at. Keeps only the newest MAX_CHATS_PER_USER chats per user.
    """
    chat_id = chat_id or str(uuid.uuid4())
    title = " ".join((first_message or "New chat").split())[:46] or "New chat"
    now = _now()
    with _connect() as conn:
        conn.execute(
            """INSERT INTO chats (id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at""",
            (chat_id, user_id, title, now, now),
        )
        conn.execute(
            """DELETE FROM chats WHERE user_id = ? AND id NOT IN (
                   SELECT id FROM chats WHERE user_id = ?
                   ORDER BY updated_at DESC LIMIT ?)""",
            (user_id, user_id, MAX_CHATS_PER_USER),
        )


def chat_belongs_to_user(user_id: int, chat_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()
    return row is not None


def add_message(chat_id: str, role: str, content: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (chat_id, role, content, _now()),
        )


def list_chats(user_id: int) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, title, updated_at FROM chats
               WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?""",
            (user_id, MAX_CHATS_PER_USER),
        ).fetchall()
    return [dict(r) for r in rows]


def count_user_messages_today(user_id: int) -> int:
    """User-authored chat messages since UTC midnight (for the daily cost cap)."""
    day_start = datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM messages m
               JOIN chats c ON c.id = m.chat_id
               WHERE c.user_id = ? AND m.role = 'user' AND m.created_at >= ?""",
            (user_id, day_start),
        ).fetchone()
    return int(row["n"])


def get_messages(chat_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]
