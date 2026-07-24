"""
SQLite storage for accounts, sessions, portfolios, and chat history.

Security model:
- Passwords are hashed with bcrypt (per-password salt, cost factor 12).
  Plaintext passwords are never stored or logged.
- Auth tokens are 256-bit random values (secrets.token_urlsafe), stored
  SHA-256-hashed, with a SLIDING expiry: a token is valid for
  TOKEN_IDLE_DAYS of inactivity and TOKEN_ABSOLUTE_DAYS in total, whichever
  comes first. Active sessions are re-issued (rotated) periodically, so a
  token captured today does not stay valid for a week.
- Sensitive columns are encrypted at rest with an envelope scheme (see
  api/crypto.py): TOTP secrets, portfolio/profile JSON, and chat titles and
  message content. A stolen app.db therefore does not expose working 2FA
  secrets, holdings, or conversation text.
- All SQL uses parameterized queries.

Every function opens its own short-lived connection, so the module is safe
to call from FastAPI's threadpool without sharing connections across threads.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import sqlite3
import unicodedata
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import bcrypt

logger = logging.getLogger("evergreen.db")
_IS_POSIX = os.name == "posix"
_permission_warning_emitted = False


def data_dir() -> Path:
    """
    Directory for everything the app writes at runtime (database, caches,
    charts, backups).

    Configurable via EVERGREEN_DATA_DIR so a container can mount a volume for
    persistent data WITHOUT mounting over /app and shadowing the application
    code — which would silently pin the deployment to whatever code first
    populated the volume.
    """
    path = Path(os.getenv("EVERGREEN_DATA_DIR", ".")).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


DB_PATH = data_dir() / "app.db"

# Session lifetime. Idle expiry keeps abandoned sessions short-lived; the
# absolute cap bounds how long a single stolen token can ever be replayed,
# even if the attacker keeps it warm.
TOKEN_IDLE_DAYS = 2
TOKEN_ABSOLUTE_DAYS = 7
#: Re-issue a token once it is older than this, so long-lived sessions rotate.
TOKEN_ROTATE_AFTER_HOURS = 12
# A rotated predecessor cannot authenticate, but may be refreshed again for a
# few seconds so simultaneous tabs do not turn a harmless race into logout.
TOKEN_ROTATION_GRACE_SECONDS = 10
#: Cap on concurrent sessions per user; the oldest are evicted beyond this.
MAX_SESSIONS_PER_USER = 10

# Backwards-compatible alias (older code/tests referenced this name).
TOKEN_TTL_DAYS = TOKEN_ABSOLUTE_DAYS

MAX_CHATS_PER_USER = 10
# Retain the newest complete user/assistant turns in each chat. Pruning uses
# user-message boundaries so it never cuts the pair just written by the route.
MAX_CHAT_TURNS = 500
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_BYTES = 72

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    email         TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    name          TEXT,
    totp_secret   TEXT,
    email_verified INTEGER NOT NULL DEFAULT 0,
    auth_version  INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS auth_tokens (
    token      TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    expires_at TEXT NOT NULL,
    issued_at TEXT,
    last_used_at TEXT,
    rotated_at TEXT,
    family_id TEXT,
    rotation_successor TEXT
);
CREATE TABLE IF NOT EXISTS crypto_keys (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    wrapped_dek TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS email_tokens (
    token_hash TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    purpose    TEXT NOT NULL CHECK (purpose IN ('verify', 'reset')),
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL
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
    user_id    INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_usage (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    day     TEXT NOT NULL,
    used    INTEGER NOT NULL CHECK (used >= 0),
    PRIMARY KEY (user_id, day)
);
CREATE TABLE IF NOT EXISTS policy_acceptances (
    user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    policy_version TEXT NOT NULL,
    accepted_at    TEXT NOT NULL,
    PRIMARY KEY (user_id, policy_version)
);
CREATE TABLE IF NOT EXISTS maintenance_migrations (
    name         TEXT PRIMARY KEY,
    completed_at TEXT NOT NULL
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
    # Overwrite deleted/updated payload cells instead of leaving recoverable
    # plaintext in freelist pages.
    conn.execute("PRAGMA secure_delete=ON")
    _harden_db_permissions()
    return conn


def _harden_db_permissions() -> None:
    """Best-effort owner-only permissions for SQLite files on POSIX."""
    global _permission_warning_emitted

    if not _IS_POSIX:
        return
    try:
        for path in (
            Path(DB_PATH),
            Path(str(DB_PATH) + "-wal"),
            Path(str(DB_PATH) + "-shm"),
        ):
            if path.exists():
                os.chmod(path, 0o600)
    except OSError as exc:
        if not _permission_warning_emitted:
            logger.warning("Could not restrict database file permissions: %s", exc)
            _permission_warning_emitted = True


def _ensure_column(
    conn: sqlite3.Connection,
    table: str,
    column: str,
    definition: str,
) -> None:
    """Apply one additive migration without hiding unrelated SQL failures."""
    columns = {
        str(row["name"])
        for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)
        # Migrations for databases created before these columns existed.
        _ensure_column(conn, "users", "totp_secret", "TEXT")
        _ensure_column(conn, "users", "name", "TEXT")
        _ensure_column(
            conn,
            "users",
            "email_verified",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            "users",
            "auth_version",
            "INTEGER NOT NULL DEFAULT 0",
        )
        _ensure_column(
            conn,
            "messages",
            "user_id",
            "INTEGER REFERENCES users(id) ON DELETE CASCADE",
        )
        # Attribute legacy messages to the owner of their chat. New writes
        # always provide the authenticated actor explicitly.
        conn.execute(
            """UPDATE messages
               SET user_id = (SELECT user_id FROM chats WHERE chats.id = messages.chat_id)
               WHERE user_id IS NULL"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id, created_at)"
        )
        # Sliding-expiry and rotation tracking.
        _ensure_column(conn, "auth_tokens", "issued_at", "TEXT")
        _ensure_column(conn, "auth_tokens", "last_used_at", "TEXT")
        _ensure_column(conn, "auth_tokens", "rotated_at", "TEXT")
        _ensure_column(conn, "auth_tokens", "family_id", "TEXT")
        _ensure_column(conn, "auth_tokens", "rotation_successor", "TEXT")
        # A pre-migration token becomes its own session family.
        conn.execute(
            "UPDATE auth_tokens SET family_id = token WHERE family_id IS NULL"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_tokens_family "
            "ON auth_tokens(user_id, family_id)"
        )
        # Tokens are now stored hashed (64 hex chars); revoke any legacy
        # plaintext tokens so they can't be used straight from a stolen DB.
        conn.execute("DELETE FROM auth_tokens WHERE length(token) <> 64")
        # Backfill timestamps on sessions issued before the sliding window.
        conn.execute(
            "UPDATE auth_tokens SET issued_at = ? WHERE issued_at IS NULL", (_now(),))
        conn.execute(
            "UPDATE auth_tokens SET last_used_at = ? WHERE last_used_at IS NULL", (_now(),))

    encrypt_existing_rows()


# ---------------------------------------------------------------------------
# Encryption key storage (see api/crypto.py for the envelope scheme)
# ---------------------------------------------------------------------------

def get_wrapped_dek() -> str | None:
    """The data key, encrypted under the master key. None before first use."""
    with _connect() as conn:
        row = conn.execute("SELECT wrapped_dek FROM crypto_keys WHERE id = 1").fetchone()
    return row["wrapped_dek"] if row else None


def set_wrapped_dek(wrapped: str) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO crypto_keys (id, wrapped_dek, created_at) VALUES (1, ?, ?)
               ON CONFLICT(id) DO UPDATE SET wrapped_dek = excluded.wrapped_dek""",
            (wrapped, _now()),
        )


def sample_encrypted_value() -> str | None:
    """
    Any one stored ciphertext, for startup key verification. None when the
    database holds nothing encrypted yet.
    """
    with _connect() as conn:
        for sql in (
            "SELECT data AS v FROM portfolios WHERE data LIKE 'enc:v1:%' LIMIT 1",
            "SELECT data AS v FROM profiles   WHERE data LIKE 'enc:v1:%' LIMIT 1",
            "SELECT totp_secret AS v FROM users WHERE totp_secret LIKE 'enc:v1:%' LIMIT 1",
            "SELECT title AS v FROM chats WHERE title LIKE 'enc:v1:%' LIMIT 1",
            "SELECT content AS v FROM messages WHERE content LIKE 'enc:v1:%' LIMIT 1",
            """SELECT rotation_successor AS v FROM auth_tokens
               WHERE rotation_successor LIKE 'enc:v1:%' LIMIT 1""",
        ):
            row = conn.execute(sql).fetchone()
            if row and row["v"]:
                return row["v"]
    return None


class EncryptionMigrationError(RuntimeError):
    """A plaintext-at-rest migration could not complete safely."""


ENCRYPTION_SCRUB_MIGRATION = "encrypted-storage-scrub-v1"


def encrypt_existing_rows() -> int:
    """
    Migrate any rows still holding plaintext secrets into ciphertext.

    Idempotent and safe to run on every startup: `crypto.is_encrypted` skips
    anything already converted, so a database that has been through this once
    costs a single cheap scan. Returns the number of rows rewritten.
    """
    from api import crypto

    changed = 0
    scrub_needed = False
    try:
        # Initialise/verify the envelope key before taking SQLite's write lock:
        # first use may need its own short DB transaction to create the DEK.
        crypto.encrypt("migration-key-preflight")
        with _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            scrub_needed = conn.execute(
                "SELECT 1 FROM maintenance_migrations WHERE name = ?",
                (ENCRYPTION_SCRUB_MIGRATION,),
            ).fetchone() is None
            rows = conn.execute(
                "SELECT id, totp_secret FROM users WHERE totp_secret IS NOT NULL"
            ).fetchall()
            for row in rows:
                if not crypto.is_encrypted(row["totp_secret"]):
                    conn.execute(
                        "UPDATE users SET totp_secret = ? WHERE id = ?",
                        (crypto.encrypt(row["totp_secret"]), row["id"]),
                    )
                    changed += 1

            for table in ("portfolios", "profiles"):
                for row in conn.execute(f"SELECT user_id, data FROM {table}").fetchall():
                    if not crypto.is_encrypted(row["data"]):
                        conn.execute(
                            f"UPDATE {table} SET data = ? WHERE user_id = ?",
                            (crypto.encrypt(row["data"]), row["user_id"]),
                        )
                        changed += 1

            for row in conn.execute("SELECT id, title FROM chats").fetchall():
                if not crypto.is_encrypted(row["title"]):
                    conn.execute(
                        "UPDATE chats SET title = ? WHERE id = ?",
                        (crypto.encrypt(row["title"]), row["id"]),
                    )
                    changed += 1

            for row in conn.execute("SELECT id, content FROM messages").fetchall():
                if not crypto.is_encrypted(row["content"]):
                    conn.execute(
                        "UPDATE messages SET content = ? WHERE id = ?",
                        (crypto.encrypt(row["content"]), row["id"]),
                    )
                    changed += 1

            for row in conn.execute(
                """SELECT token, rotation_successor FROM auth_tokens
                   WHERE rotation_successor IS NOT NULL"""
            ).fetchall():
                if not crypto.is_encrypted(row["rotation_successor"]):
                    conn.execute(
                        "UPDATE auth_tokens SET rotation_successor = ? WHERE token = ?",
                        (crypto.encrypt(row["rotation_successor"]), row["token"]),
                    )
                    changed += 1

        if changed or scrub_needed:
            # The UPDATE transaction is encrypted now, but old plaintext may
            # still exist in WAL frames or free pages. Checkpoint/truncate,
            # rebuild the database, then truncate any WAL frames VACUUM wrote.
            # This must run outside the migration transaction.
            with _connect() as maintenance:
                checkpoint = maintenance.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if checkpoint and int(checkpoint[0]) != 0:
                    raise sqlite3.OperationalError(
                        "could not exclusively checkpoint the encryption migration"
                    )
                maintenance.execute("VACUUM")
                checkpoint = maintenance.execute(
                    "PRAGMA wal_checkpoint(TRUNCATE)"
                ).fetchone()
                if checkpoint and int(checkpoint[0]) != 0:
                    raise sqlite3.OperationalError(
                        "could not truncate WAL after encryption migration"
                    )
            # A durable marker makes the forensic scrub a one-time migration.
            # It is written only after both maintenance operations succeed.
            with _connect() as conn:
                conn.execute(
                    """INSERT INTO maintenance_migrations (name, completed_at)
                       VALUES (?, ?)
                       ON CONFLICT(name) DO UPDATE
                       SET completed_at = excluded.completed_at""",
                    (ENCRYPTION_SCRUB_MIGRATION, _now()),
                )
    except (
        sqlite3.Error,
        crypto.CryptoError,
        OSError,
        ValueError,
        TypeError,
        UnicodeError,
    ) as exc:
        raise EncryptionMigrationError(
            "At-rest encryption migration failed; refusing to serve mixed plaintext."
        ) from exc

    if changed:
        logger.info("Encrypted %d pre-existing plaintext row(s) at rest.", changed)
    return changed


def _now() -> str:
    # Microsecond precision: chat pruning orders by this, and several chats
    # can be created within the same second.
    return datetime.utcnow().isoformat(timespec="microseconds")


# ---------------------------------------------------------------------------
# Users & auth
# ---------------------------------------------------------------------------

class AuthError(Exception):
    """Raised for registration/login failures with a user-safe message."""


_EMAIL_LOCAL_RE = re.compile(r"^[a-z0-9.!#$%&'*+/=?^_`{|}~-]+$")


def normalize_email(email: str) -> str:
    """Return the canonical address used for lookup, or raise a safe error."""
    value = unicodedata.normalize("NFKC", str(email or "")).strip()
    if (
        not value
        or len(value) > 254
        or value.count("@") != 1
        or any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in value)
    ):
        raise AuthError("Please enter a valid email address.")
    local, domain_raw = value.rsplit("@", 1)
    local = local.lower()
    if (
        not local
        or len(local.encode("utf-8")) > 64
        or not _EMAIL_LOCAL_RE.fullmatch(local)
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
    ):
        raise AuthError("Please enter a valid email address.")
    try:
        domain = domain_raw.rstrip(".").encode("idna").decode("ascii").lower()
    except (UnicodeError, ValueError):
        raise AuthError("Please enter a valid email address.") from None
    labels = domain.split(".")
    if (
        len(labels) < 2
        or len(labels[-1]) < 2
        or any(
            not label
            or len(label) > 63
            or label.startswith("-")
            or label.endswith("-")
            or not all(ch.isalnum() or ch == "-" for ch in label)
            for label in labels
        )
    ):
        raise AuthError("Please enter a valid email address.")
    canonical = f"{local}@{domain}"
    if len(canonical.encode("utf-8")) > 254:
        raise AuthError("Please enter a valid email address.")
    return canonical


def _validate_password(password: str, *, label: str = "Password") -> bytes:
    raw = (password or "").encode("utf-8")
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AuthError(f"{label} must be at least {MIN_PASSWORD_LEN} characters.")
    if len(raw) > MAX_PASSWORD_BYTES:
        raise AuthError(f"{label} must be at most {MAX_PASSWORD_BYTES} UTF-8 bytes.")
    return raw


def validate_password(password: str, *, label: str = "Password") -> None:
    """Public validation hook for flows that must validate before burning a token."""
    _validate_password(password, label=label)


def register_user(
    email: str,
    password: str,
    name: str | None = None,
    *,
    accepted_policy_version: str | None = None,
) -> int:
    email = normalize_email(email)
    password_raw = _validate_password(password)
    policy_version = (accepted_policy_version or "").strip()[:64] or None
    if accepted_policy_version is not None and not policy_version:
        raise AuthError("A valid policy version is required.")
    name = (name or "").strip()[:60] or None

    pw_hash = bcrypt.hashpw(password_raw, bcrypt.gensalt(rounds=12))
    try:
        with _connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, name, created_at) VALUES (?, ?, ?, ?)",
                (email, pw_hash.decode("ascii"), name, _now()),
            )
            user_id = int(cur.lastrowid)
            if policy_version:
                conn.execute(
                    """INSERT INTO policy_acceptances
                       (user_id, policy_version, accepted_at) VALUES (?, ?, ?)""",
                    (user_id, policy_version, _now()),
                )
            return user_id
    except sqlite3.IntegrityError:
        # Deliberately does not confirm that the address is already registered.
        raise AuthError(
            "Unable to create this account. Try signing in or resetting your password."
        ) from None


def record_policy_acceptance(user_id: int, policy_version: str) -> None:
    version = (policy_version or "").strip()[:64]
    if not version:
        raise ValueError("policy_version is required")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO policy_acceptances (user_id, policy_version, accepted_at)
               VALUES (?, ?, ?)
               ON CONFLICT(user_id, policy_version)
               DO UPDATE SET accepted_at = excluded.accepted_at""",
            (user_id, version, _now()),
        )


def has_policy_acceptance(user_id: int, policy_version: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            """SELECT 1 FROM policy_acceptances
               WHERE user_id = ? AND policy_version = ?""",
            (user_id, (policy_version or "").strip()),
        ).fetchone()
    return row is not None


def list_policy_acceptances(user_id: int) -> list[dict]:
    """Versioned legal consents for account export/audit."""
    with _connect() as conn:
        rows = conn.execute(
            """SELECT policy_version, accepted_at FROM policy_acceptances
               WHERE user_id = ? ORDER BY accepted_at""",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


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
    try:
        email = normalize_email(email)
    except AuthError:
        email = ""
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
    """
    Mint a new session token.

    `expires_at` is the ABSOLUTE deadline; idle expiry is enforced separately
    against `last_used_at` so an unused session dies well before this.
    """
    token = secrets.token_urlsafe(32)
    family_id = secrets.token_urlsafe(18)
    now = datetime.utcnow()
    expires = (now + timedelta(days=TOKEN_ABSOLUTE_DAYS)).isoformat(timespec="seconds")
    stamp = now.isoformat(timespec="microseconds")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO auth_tokens
               (token, user_id, expires_at, issued_at, last_used_at, family_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (_hash_token(token), user_id, expires, stamp, stamp, family_id),
        )
        # Bound concurrent sessions so a user whose password leaked doesn't
        # accumulate an unbounded pile of valid tokens.
        conn.execute(
            """DELETE FROM auth_tokens
               WHERE user_id = ? AND rotated_at IS NULL AND token NOT IN (
                   SELECT token FROM auth_tokens WHERE user_id = ?
                   AND rotated_at IS NULL ORDER BY issued_at DESC LIMIT ?)""",
            (user_id, user_id, MAX_SESSIONS_PER_USER),
        )
    return token


def user_for_token(token: str) -> dict | None:
    """
    Resolve a token to {'id', 'email', 'name', 'stale'}, or None.

    Enforces both expiries and slides the idle window forward on each use.
    'stale' is True once the session is older than TOKEN_ROTATE_AFTER_HOURS,
    which tells the caller to hand the client a freshly minted token.
    Expired tokens are deleted as they are encountered.
    """
    if not token:
        return None
    hashed = _hash_token(token)
    now = datetime.utcnow()
    now_s = now.isoformat(timespec="microseconds")

    with _connect() as conn:
        row = conn.execute(
            """SELECT u.id, u.email, u.name, t.expires_at, t.issued_at,
                      t.last_used_at, t.rotated_at
               FROM auth_tokens t JOIN users u ON u.id = t.user_id WHERE t.token = ?""",
            (hashed,),
        ).fetchone()
        if not row:
            return None

        # A predecessor retained for refresh-race grace is never valid for
        # ordinary authentication.
        if row["rotated_at"]:
            grace_cutoff = (
                now - timedelta(seconds=TOKEN_ROTATION_GRACE_SECONDS)
            ).isoformat(timespec="microseconds")
            if row["rotated_at"] < grace_cutoff:
                conn.execute("DELETE FROM auth_tokens WHERE token = ?", (hashed,))
            return None

        # Absolute deadline.
        if row["expires_at"] < now_s:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (hashed,))
            return None

        # Idle deadline.
        last_used = row["last_used_at"] or row["issued_at"] or now_s
        idle_cutoff = (now - timedelta(days=TOKEN_IDLE_DAYS)).isoformat(timespec="microseconds")
        if last_used < idle_cutoff:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (hashed,))
            return None

        conn.execute("UPDATE auth_tokens SET last_used_at = ? WHERE token = ?", (now_s, hashed))

        issued = row["issued_at"] or now_s
        rotate_cutoff = (now - timedelta(hours=TOKEN_ROTATE_AFTER_HOURS)).isoformat(
            timespec="microseconds")
        return {
            "id": int(row["id"]),
            "email": row["email"],
            "name": row["name"],
            "stale": issued < rotate_cutoff,
        }


def rotate_token(token: str) -> str | None:
    """
    Atomically replace one valid token while preserving its absolute deadline.

    The old token becomes unusable for authentication in the same transaction
    that creates the new one. Its hash remains briefly refreshable and stores
    the successor encrypted, so concurrent/replayed refreshes return the SAME
    token instead of amplifying one predecessor into many week-long sessions.
    """
    if not token:
        return None
    from api import crypto

    old_hash = _hash_token(token)
    now = datetime.utcnow()
    now_s = now.isoformat(timespec="microseconds")
    idle_cutoff = (now - timedelta(days=TOKEN_IDLE_DAYS)).isoformat(timespec="microseconds")
    grace_cutoff = (
        now - timedelta(seconds=TOKEN_ROTATION_GRACE_SECONDS)
    ).isoformat(timespec="microseconds")
    replacement = secrets.token_urlsafe(32)
    replacement_hash = _hash_token(replacement)
    replacement_ciphertext = crypto.encrypt(replacement)

    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """SELECT user_id, expires_at, issued_at, last_used_at, rotated_at,
                      family_id, rotation_successor
               FROM auth_tokens WHERE token = ?""",
            (old_hash,),
        ).fetchone()
        if not row:
            return None
        last_used = row["last_used_at"] or row["issued_at"] or now_s
        if row["expires_at"] < now_s or last_used < idle_cutoff:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (old_hash,))
            return None
        if row["rotated_at"] and row["rotated_at"] < grace_cutoff:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (old_hash,))
            return None
        if row["rotated_at"]:
            successor_ciphertext = row["rotation_successor"]
            if not successor_ciphertext:
                return None
            successor = crypto.decrypt(successor_ciphertext)
            # If the successor itself was rotated almost immediately, follow
            # the short chain and return the newest active member.
            for _ in range(MAX_SESSIONS_PER_USER + 1):
                successor_row = conn.execute(
                    """SELECT rotated_at, rotation_successor FROM auth_tokens
                       WHERE token = ?""",
                    (_hash_token(successor),),
                ).fetchone()
                if not successor_row:
                    return None
                if not successor_row["rotated_at"]:
                    return successor
                if not successor_row["rotation_successor"]:
                    return None
                successor = crypto.decrypt(successor_row["rotation_successor"])
            return None

        family_id = str(row["family_id"] or old_hash)
        conn.execute(
            """UPDATE auth_tokens
               SET rotated_at = ?, family_id = ?, rotation_successor = ?
               WHERE token = ?""",
            (now_s, family_id, replacement_ciphertext, old_hash),
        )
        conn.execute(
            """INSERT INTO auth_tokens
               (token, user_id, expires_at, issued_at, last_used_at,
                rotated_at, family_id, rotation_successor)
               VALUES (?, ?, ?, ?, ?, NULL, ?, NULL)""",
            (
                replacement_hash,
                int(row["user_id"]),
                row["expires_at"],
                now_s,
                now_s,
                family_id,
            ),
        )
        # Bound active sessions. Rotated predecessor rows are separately
        # bounded by the grace window and cannot authenticate.
        conn.execute(
            """DELETE FROM auth_tokens
               WHERE user_id = ? AND rotated_at IS NULL AND token NOT IN (
                   SELECT token FROM auth_tokens WHERE user_id = ?
                   AND rotated_at IS NULL ORDER BY issued_at DESC LIMIT ?)""",
            (int(row["user_id"]), int(row["user_id"]), MAX_SESSIONS_PER_USER),
        )
        conn.execute(
            "DELETE FROM auth_tokens WHERE rotated_at IS NOT NULL AND rotated_at < ?",
            (grace_cutoff,),
        )
    return replacement


def purge_expired_tokens() -> int:
    """Delete every session past its absolute or idle deadline. Returns the count."""
    now = datetime.utcnow()
    now_s = now.isoformat(timespec="microseconds")
    idle_cutoff = (now - timedelta(days=TOKEN_IDLE_DAYS)).isoformat(timespec="microseconds")
    with _connect() as conn:
        cur = conn.execute(
            """DELETE FROM auth_tokens
               WHERE expires_at < ?
                  OR COALESCE(last_used_at, issued_at, ?) < ?""",
            (now_s, now_s, idle_cutoff),
        )
        removed = int(cur.rowcount or 0)
        grace_cutoff = (
            now - timedelta(seconds=TOKEN_ROTATION_GRACE_SECONDS)
        ).isoformat(timespec="microseconds")
        rotated = conn.execute(
            "DELETE FROM auth_tokens WHERE rotated_at IS NOT NULL AND rotated_at < ?",
            (grace_cutoff,),
        )
        return removed + int(rotated.rowcount or 0)


def revoke_token(token: str) -> None:
    hashed = _hash_token(token)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT user_id, family_id FROM auth_tokens WHERE token = ?",
            (hashed,),
        ).fetchone()
        if row and row["family_id"]:
            conn.execute(
                "DELETE FROM auth_tokens WHERE user_id = ? AND family_id = ?",
                (int(row["user_id"]), row["family_id"]),
            )
        else:
            conn.execute("DELETE FROM auth_tokens WHERE token = ?", (hashed,))


def revoke_all_tokens(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))


def revoke_other_tokens(user_id: int, keep_token: str) -> None:
    """Revoke every session except the authenticated one making this change."""
    keep_hash = _hash_token(keep_token)
    with _connect() as conn:
        conn.execute(
            "DELETE FROM auth_tokens WHERE user_id = ? AND token <> ?",
            (user_id, keep_hash),
        )


def change_password(user_id: int, current: str, new: str) -> None:
    """Verify the current password, set the new one, sign out everywhere."""
    new_raw = _validate_password(new, label="New password")
    with _connect() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    current_ok = False
    if row:
        try:
            current_ok = bcrypt.checkpw(
                (current or "").encode("utf-8"),
                row["password_hash"].encode("ascii"),
            )
        except ValueError:
            current_ok = False
    if not current_ok:
        raise AuthError("Current password is incorrect.")
    new_hash = bcrypt.hashpw(new_raw, bcrypt.gensalt(rounds=12))
    with _connect() as conn:
        conn.execute(
            """UPDATE users
               SET password_hash = ?, auth_version = auth_version + 1
               WHERE id = ?""",
            (new_hash.decode("ascii"), user_id),
        )
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))


def set_password(user_id: int, new: str) -> None:
    """
    Set a new password WITHOUT requiring the old one (used by the reset flow,
    which authenticates via a single-use email token instead). Signs out every
    session so a stolen/old token can't be reused after a reset. Inbox control
    is the recovery factor, so reset also clears TOTP and invalidates pending
    login handles through auth_version.
    """
    new_raw = _validate_password(new, label="New password")
    new_hash = bcrypt.hashpw(new_raw, bcrypt.gensalt(rounds=12))
    with _connect() as conn:
        conn.execute(
            """UPDATE users
               SET password_hash = ?, totp_secret = NULL,
                   auth_version = auth_version + 1
               WHERE id = ?""",
            (new_hash.decode("ascii"), user_id),
        )
        conn.execute("DELETE FROM auth_tokens WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM email_tokens WHERE user_id = ?", (user_id,))


def user_id_by_email(email: str) -> int | None:
    """User id for an email, or None. Used by the reset flow (no enumeration)."""
    try:
        email = normalize_email(email)
    except AuthError:
        return None
    with _connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
    return int(row["id"]) if row else None


def get_user_identity(user_id: int) -> dict | None:
    """Small auth-only projection used by pending-2FA validation."""
    with _connect() as conn:
        row = conn.execute(
            "SELECT email, name, auth_version FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return dict(row) if row else None


def get_auth_version(user_id: int) -> int:
    identity = get_user_identity(user_id)
    return int(identity["auth_version"]) if identity else -1


def is_email_verified(user_id: int) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT email_verified FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    return bool(row["email_verified"]) if row else False


def set_email_verified(user_id: int) -> None:
    with _connect() as conn:
        conn.execute("UPDATE users SET email_verified = 1 WHERE id = ?", (user_id,))


# --- one-time email tokens (verification + password reset) -------------------

def create_email_token(user_id: int, purpose: str, ttl_minutes: int) -> str:
    """
    Issue a single-use token for 'verify' or 'reset', returning the RAW token
    (only the SHA-256 hash is stored). Any previous outstanding token of the
    same purpose for this user is dropped, so only the newest link works.
    """
    if purpose not in ("verify", "reset"):
        raise ValueError("purpose must be 'verify' or 'reset'")
    raw = secrets.token_urlsafe(32)
    expires = (datetime.utcnow() + timedelta(minutes=ttl_minutes)).isoformat(timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "DELETE FROM email_tokens WHERE user_id = ? AND purpose = ?", (user_id, purpose)
        )
        conn.execute(
            """INSERT INTO email_tokens (token_hash, user_id, purpose, expires_at, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (_hash_token(raw), user_id, purpose, expires, _now()),
        )
    return raw


def consume_email_token(raw_token: str, purpose: str) -> int | None:
    """
    Validate and BURN a token: returns its user_id if it exists, matches the
    purpose, and hasn't expired; deletes it either way (single use). Expired
    rows for the purpose are purged opportunistically.
    """
    if not raw_token:
        return None
    hashed = _hash_token(raw_token)
    with _connect() as conn:
        # Serialize lookup + burn. Without an eager write transaction, two
        # workers can both SELECT the row before either DELETE begins.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT user_id, expires_at FROM email_tokens WHERE token_hash = ? AND purpose = ?",
            (hashed, purpose),
        ).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM email_tokens WHERE token_hash = ?", (hashed,))
        conn.execute(
            "DELETE FROM email_tokens WHERE purpose = ? AND expires_at < ?", (purpose, _now())
        )
        if row["expires_at"] < _now():
            return None
        return int(row["user_id"])


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
    """Store the TOTP secret encrypted; None disables 2FA."""
    from api import crypto

    with _connect() as conn:
        conn.execute(
            """UPDATE users
               SET totp_secret = ?, auth_version = auth_version + 1
               WHERE id = ?""",
            (crypto.encrypt(secret), user_id),
        )


def get_totp_secret(user_id: int) -> str | None:
    from api import crypto

    with _connect() as conn:
        row = conn.execute(
            "SELECT totp_secret FROM users WHERE id = ?", (user_id,)
        ).fetchone()
    if not row or not row["totp_secret"]:
        return None
    return crypto.decrypt(row["totp_secret"])


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


def claim_alert_check(user_id: int, hours: float = 6.0) -> bool:
    """
    Atomically reserve this account's bounded alert recomputation window.

    A read followed by ``mark_alerts_checked`` lets simultaneous requests in
    different workers all observe "due" and fan out expensive provider calls.
    Taking SQLite's write lock before comparing/upserting makes exactly one
    caller the winner. ``checked_at`` records the attempt, not a promise that
    every upstream provider succeeded; transient failures therefore cool down
    instead of becoming a request-triggered provider storm.
    """
    now = datetime.utcnow()
    now_s = now.isoformat(timespec="microseconds")
    cutoff = (now - timedelta(hours=max(float(hours), 0.0))).isoformat(
        timespec="microseconds"
    )
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT checked_at FROM alert_checks WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row and row["checked_at"] >= cutoff:
            return False
        conn.execute(
            """INSERT INTO alert_checks (user_id, checked_at) VALUES (?, ?)
               ON CONFLICT(user_id) DO UPDATE
               SET checked_at = excluded.checked_at""",
            (user_id, now_s),
        )
        return True


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
    """Insert or replace the user's portfolio (re-importing a file updates it).

    Holdings are the most sensitive data the app stores, so the JSON blob is
    encrypted at rest.
    """
    from api import crypto

    payload = crypto.encrypt(json.dumps(parsed, ensure_ascii=False))
    with _connect() as conn:
        conn.execute(
            """INSERT INTO portfolios (user_id, data, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET data = excluded.data,
                                                  updated_at = excluded.updated_at""",
            (user_id, payload, _now()),
        )


def get_portfolio(user_id: int) -> dict | None:
    from api import crypto

    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM portfolios WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(crypto.decrypt(row["data"]))
    except Exception:
        logger.exception("Could not read stored portfolio for user %s", user_id)
        return None


# ---------------------------------------------------------------------------
# Investor profiles (retirement horizon, risk tolerance, goals)
# ---------------------------------------------------------------------------

def save_profile(user_id: int, profile: dict) -> None:
    """Investor profile (goals, contributions, risk) — encrypted at rest."""
    from api import crypto

    payload = crypto.encrypt(json.dumps(profile, ensure_ascii=False))
    with _connect() as conn:
        conn.execute(
            """INSERT INTO profiles (user_id, data, updated_at) VALUES (?, ?, ?)
               ON CONFLICT(user_id) DO UPDATE SET data = excluded.data,
                                                  updated_at = excluded.updated_at""",
            (user_id, payload, _now()),
        )


def get_profile(user_id: int) -> dict | None:
    from api import crypto

    with _connect() as conn:
        row = conn.execute(
            "SELECT data FROM profiles WHERE user_id = ?", (user_id,)
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(crypto.decrypt(row["data"]))
    except Exception:
        logger.exception("Could not read stored profile for user %s", user_id)
        return None


# ---------------------------------------------------------------------------
# Chats & messages
# ---------------------------------------------------------------------------


class ChatOwnershipError(Exception):
    """Raised when a chat id belongs to a different authenticated user."""


class ChatQuotaError(Exception):
    """Raised when a user's daily model-call allocation is exhausted."""


def _chat_title(first_message: str) -> str:
    return " ".join((first_message or "New chat").split())[:46] or "New chat"


def _touch_chat_in_conn(
    conn: sqlite3.Connection,
    user_id: int,
    chat_id: str,
    encrypted_title: str,
) -> None:
    """
    Create or update a chat after checking ownership in the same transaction.

    A global chat id must never be "claimed" by a second account. In
    particular, an ON CONFLICT update without an owner predicate would let an
    attacker mutate another user's chat timestamp and then write into it.
    """
    row = conn.execute(
        "SELECT user_id FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    now = _now()
    if row:
        if int(row["user_id"]) != int(user_id):
            raise ChatOwnershipError("Chat not found.")
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, chat_id, user_id),
        )
    else:
        conn.execute(
            """INSERT INTO chats (id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chat_id, user_id, encrypted_title, now, now),
        )

    conn.execute(
        """DELETE FROM chats WHERE user_id = ? AND id NOT IN (
               SELECT id FROM chats WHERE user_id = ?
               ORDER BY updated_at DESC LIMIT ?)""",
        (user_id, user_id, MAX_CHATS_PER_USER),
    )


def touch_chat(user_id: int, chat_id: str, first_message: str) -> None:
    """
    Create the chat on first use (title = trimmed first message) or bump its
    updated_at. Keeps only the newest MAX_CHATS_PER_USER chats per user.
    """
    from api import crypto

    chat_id = chat_id or str(uuid.uuid4())
    title = crypto.encrypt(_chat_title(first_message))
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _touch_chat_in_conn(conn, user_id, chat_id, title)


def chat_belongs_to_user(user_id: int, chat_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM chats WHERE id = ? AND user_id = ?", (chat_id, user_id)
        ).fetchone()
    return row is not None


def begin_chat_turn(
    user_id: int,
    chat_id: str,
    first_message: str,
    daily_limit: int,
) -> int:
    """
    Atomically establish ownership and reserve one unit of daily model quota.

    The reservation happens before the external model call. This closes the
    concurrent-request race where several requests could all observe
    ``used < limit`` and overspend. A failed model call still consumes the
    reservation because it may already have incurred provider cost.
    """
    from api import crypto

    encrypted_title = crypto.encrypt(_chat_title(first_message))
    day = datetime.utcnow().strftime("%Y-%m-%d")
    day_start = f"{day}T00:00:00"
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _touch_chat_in_conn(conn, user_id, chat_id, encrypted_title)
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM messages
               WHERE user_id = ? AND role = 'user' AND created_at >= ?""",
            (user_id, day_start),
        ).fetchone()
        actual = int(row["n"])
        usage_row = conn.execute(
            "SELECT used FROM chat_usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()
        reserved = int(usage_row["used"]) if usage_row else 0
        used = max(actual, reserved)
        if used >= int(daily_limit):
            raise ChatQuotaError("Daily chat limit reached.")
        used += 1
        conn.execute(
            """INSERT INTO chat_usage (user_id, day, used) VALUES (?, ?, ?)
               ON CONFLICT(user_id, day) DO UPDATE SET used = excluded.used""",
            (user_id, day, used),
        )
        return used


def _assert_chat_owner(
    conn: sqlite3.Connection,
    user_id: int,
    chat_id: str,
) -> None:
    row = conn.execute(
        "SELECT user_id FROM chats WHERE id = ?",
        (chat_id,),
    ).fetchone()
    if not row or int(row["user_id"]) != int(user_id):
        raise ChatOwnershipError("Chat not found.")


def _prune_chat_messages(conn: sqlite3.Connection, chat_id: str) -> None:
    """Drop turns older than the newest MAX_CHAT_TURNS at a user boundary."""
    cutoff = conn.execute(
        """SELECT id FROM messages
           WHERE chat_id = ? AND role = 'user'
           ORDER BY id DESC LIMIT 1 OFFSET ?""",
        (chat_id, MAX_CHAT_TURNS - 1),
    ).fetchone()
    if cutoff:
        conn.execute(
            "DELETE FROM messages WHERE chat_id = ? AND id < ?",
            (chat_id, int(cutoff["id"])),
        )


def add_message(*args) -> None:
    """
    Append one message with an ownership check in the same transaction.

    Preferred form: ``add_message(user_id, chat_id, role, content)``. The
    three-argument form is retained for trusted internal migration/tests and
    infers the owner from the chat row transactionally; request handlers must
    never use that compatibility form.
    """
    from api import crypto

    if len(args) == 4:
        user_id, chat_id, role, content = args
        expected_user_id: int | None = int(user_id)
    elif len(args) == 3:
        chat_id, role, content = args
        expected_user_id = None
    else:
        raise TypeError("add_message expects (user_id, chat_id, role, content)")

    encrypted = crypto.encrypt(content)
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT user_id FROM chats WHERE id = ?",
            (chat_id,),
        ).fetchone()
        if not row:
            raise ChatOwnershipError("Chat not found.")
        owner_id = int(row["user_id"])
        if expected_user_id is not None and owner_id != expected_user_id:
            raise ChatOwnershipError("Chat not found.")
        conn.execute(
            """INSERT INTO messages (chat_id, user_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (chat_id, owner_id, role, encrypted, _now()),
        )
        _prune_chat_messages(conn, chat_id)


def append_chat_turn(
    user_id: int,
    chat_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Persist a complete user/assistant exchange as one owned transaction."""
    from api import crypto

    encrypted_user = crypto.encrypt(user_message)
    encrypted_assistant = crypto.encrypt(assistant_message)
    now = _now()
    with _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        _assert_chat_owner(conn, user_id, chat_id)
        conn.executemany(
            """INSERT INTO messages
               (chat_id, user_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (
                (chat_id, user_id, "user", encrypted_user, now),
                (chat_id, user_id, "assistant", encrypted_assistant, now),
            ),
        )
        conn.execute(
            "UPDATE chats SET updated_at = ? WHERE id = ? AND user_id = ?",
            (now, chat_id, user_id),
        )
        _prune_chat_messages(conn, chat_id)


def list_chats(user_id: int) -> list[dict]:
    from api import crypto

    with _connect() as conn:
        rows = conn.execute(
            """SELECT id, title, updated_at FROM chats
               WHERE user_id = ? ORDER BY updated_at DESC LIMIT ?""",
            (user_id, MAX_CHATS_PER_USER),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["title"] = crypto.decrypt(item["title"])
        result.append(item)
    return result


def count_user_messages_today(user_id: int) -> int:
    """Committed or reserved user model calls since UTC midnight."""
    day = datetime.utcnow().strftime("%Y-%m-%d")
    day_start = f"{day}T00:00:00"
    with _connect() as conn:
        row = conn.execute(
            """SELECT COUNT(*) AS n FROM messages
               WHERE user_id = ? AND role = 'user' AND created_at >= ?""",
            (user_id, day_start),
        ).fetchone()
        usage = conn.execute(
            "SELECT used FROM chat_usage WHERE user_id = ? AND day = ?",
            (user_id, day),
        ).fetchone()
    return max(int(row["n"]), int(usage["used"]) if usage else 0)


def get_messages(chat_id: str) -> list[dict]:
    """
    Trusted internal read for account export/backwards compatibility.

    Request routes and agent memory must use ``get_messages_for_user``.
    """
    from api import crypto

    with _connect() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE chat_id = ? ORDER BY id",
            (chat_id,),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["content"] = crypto.decrypt(item["content"])
        result.append(item)
    return result


def get_messages_for_user(user_id: int, chat_id: str) -> list[dict]:
    """Read one chat after checking ownership in the same DB transaction."""
    from api import crypto

    with _connect() as conn:
        conn.execute("BEGIN")
        _assert_chat_owner(conn, user_id, chat_id)
        rows = conn.execute(
            """SELECT role, content, created_at FROM messages
               WHERE chat_id = ? AND user_id = ? ORDER BY id""",
            (chat_id, user_id),
        ).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["content"] = crypto.decrypt(item["content"])
        result.append(item)
    return result
