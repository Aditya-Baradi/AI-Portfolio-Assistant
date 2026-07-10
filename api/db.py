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
CREATE INDEX IF NOT EXISTS idx_chats_user    ON chats(user_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, id);
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


def _now() -> str:
    # Microsecond precision: chat pruning orders by this, and several chats
    # can be created within the same second.
    return datetime.utcnow().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Users & auth
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Raised for registration/login failures with a user-safe message."""


def register_user(email: str, password: str) -> int:
    email = (email or "").strip().lower()
    if "@" not in email or len(email) < 5:
        raise AuthError("Please enter a valid email address.")
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")

    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
                (email, pw_hash.decode("ascii"), _now()),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        raise AuthError("An account with that email already exists.")


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


def issue_token(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(days=TOKEN_TTL_DAYS)).isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires),
        )
    return token


def user_for_token(token: str) -> dict | None:
    """Return {'id', 'email'} for a valid token, else None. Expired tokens are purged."""
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            """SELECT u.id, u.email, t.expires_at FROM auth_tokens t
               JOIN users u ON u.id = t.user_id WHERE t.token = ?""",
            (token,),
        ).fetchone()
        if not row:
            return None
        if row["expires_at"] < _now():
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))
            return None
        return {"id": int(row["id"]), "email": row["email"]}


def revoke_token(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE token = ?", (token,))


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


def get_messages(chat_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    return [dict(r) for r in rows]
