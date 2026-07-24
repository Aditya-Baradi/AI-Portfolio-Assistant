"""
Application-level encryption for sensitive data at rest.

Design: **envelope encryption** (the standard pattern, and the reason this
module exists rather than a single module-level Fernet key).

    master key (KEK)  -- from the environment, never stored in the DB
        wraps
    data key (DEK)    -- random, generated once, stored WRAPPED in the DB
        encrypts
    field ciphertext  -- TOTP secrets, portfolio JSON, profile JSON

Why envelope rather than encrypting fields directly with the env key:
rotating the master key only re-wraps ONE row (the DEK), instead of
re-encrypting every portfolio in the database. Key rotation goes from an
O(rows) migration to an O(1) update.

Why field-level rather than SQLCipher: SQLCipher encrypts the whole file but
requires building a patched SQLite (a third-party fork) and is painful to
install on Windows. Field-level encryption keeps the DB a stock SQLite file
that any tool can open, while the columns that actually matter stay opaque.

Configuration
-------------
EVERGREEN_MASTER_KEY   base64 urlsafe 32-byte key (generate with
                       `python -m api.crypto keygen`). Comma-separate to
                       rotate: "newkey,oldkey" — the first encrypts, all of
                       them can decrypt.
EVERGREEN_ENV          set to "production" to make a missing master key a
                       hard startup failure instead of a dev fallback.

In development, if no master key is set, one is generated and persisted to
`.evergreen_key` (gitignored) so the app just runs. That file is NOT suitable
for production and the app says so, loudly, at startup.
"""
from __future__ import annotations

import os
import logging
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger("evergreen.crypto")

# Marker prefix on every ciphertext this module produces. Values WITHOUT it are
# treated as legacy plaintext and passed through by decrypt(), so an existing
# database keeps working and re-encrypts itself on the next write.
_PREFIX = "enc:v1:"

#: Explicit override for the development key file. Normally None, meaning
#: "derive it from the data directory" — see `dev_key_file()`. Tests set this.
DEV_KEY_FILE: Path | None = None


def dev_key_file() -> Path:
    """
    Location of the development-only master key.

    Defaults to sitting beside the database (EVERGREEN_DATA_DIR) so a mounted
    data volume keeps the two together: a key without its database, or a
    database without its key, is useless.
    """
    if DEV_KEY_FILE is not None:
        return Path(DEV_KEY_FILE)
    from api.db import data_dir

    return data_dir() / ".evergreen_key"

_dek_fernet: Fernet | None = None  # cached data-key cipher


class CryptoError(Exception):
    """Raised when encryption is misconfigured in a way we must not paper over."""


def generate_key() -> str:
    """A fresh urlsafe-base64 32-byte key, suitable for EVERGREEN_MASTER_KEY."""
    return Fernet.generate_key().decode("ascii")


def is_production() -> bool:
    return os.getenv("EVERGREEN_ENV", "").strip().lower() in ("production", "prod")


def _master_keys() -> list[str]:
    """
    Master keys, newest first. The first encrypts; all of them can decrypt,
    which is what makes zero-downtime key rotation possible.
    """
    raw = os.getenv("EVERGREEN_MASTER_KEY", "").strip()
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if keys:
        return keys

    if is_production():
        raise CryptoError(
            "EVERGREEN_MASTER_KEY is not set. Refusing to start in production "
            "with unencrypted secrets. Generate one with: python -m api.crypto keygen"
        )

    # Development fallback: persist a key locally so restarts can still read
    # what previous runs wrote. Never acceptable in production.
    path = dev_key_file()
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return [key]
    key = generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(key, encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass  # best-effort; Windows ACLs don't map cleanly
    logger.warning(
        "No EVERGREEN_MASTER_KEY set - generated a DEVELOPMENT key at %s. "
        "Set EVERGREEN_MASTER_KEY from a secret manager before deploying.",
        path,
    )
    return [key]


def _kek() -> MultiFernet:
    """The key-encryption key(s), as a MultiFernet so old keys still decrypt."""
    keys = _master_keys()
    try:
        return MultiFernet([Fernet(k.encode("ascii")) for k in keys])
    except (ValueError, TypeError) as e:
        raise CryptoError(
            f"EVERGREEN_MASTER_KEY is not a valid Fernet key: {e}. "
            "Generate one with: python -m api.crypto keygen"
        ) from e


def _load_or_create_dek() -> Fernet:
    """
    Fetch the wrapped data key from the DB (creating it on first run) and
    unwrap it with the master key.
    """
    from api import db

    wrapped = db.get_wrapped_dek()
    kek = _kek()

    if wrapped is None:
        dek = Fernet.generate_key()
        db.set_wrapped_dek(kek.encrypt(dek).decode("ascii"))
        logger.info("Generated a new data encryption key (wrapped with the master key).")
        return Fernet(dek)

    try:
        dek = kek.decrypt(wrapped.encode("ascii"))
    except InvalidToken as e:
        raise CryptoError(
            "Could not unwrap the data key with EVERGREEN_MASTER_KEY. The key is "
            "wrong or was rotated without keeping the previous value. Include the "
            "old key as a second comma-separated entry to recover."
        ) from e

    # Opportunistic re-wrap: if the DEK was wrapped with an older master key,
    # MultiFernet.rotate re-wraps it under the newest one. O(1), no data touched.
    try:
        rewrapped = kek.rotate(wrapped.encode("ascii")).decode("ascii")
        if rewrapped != wrapped:
            db.set_wrapped_dek(rewrapped)
            logger.info("Re-wrapped the data key under the current master key.")
    except Exception:
        pass  # rotation is an optimisation, never a failure path

    return Fernet(dek)


def _cipher() -> Fernet:
    global _dek_fernet
    if _dek_fernet is None:
        _dek_fernet = _load_or_create_dek()
    return _dek_fernet


def reset_cache() -> None:
    """Drop the cached data key. Used by tests and after key rotation."""
    global _dek_fernet
    _dek_fernet = None


def encrypt(plaintext: str | None) -> str | None:
    """Encrypt a string for storage. None and '' pass through unchanged."""
    if plaintext is None or plaintext == "":
        return plaintext
    token = _cipher().encrypt(str(plaintext).encode("utf-8")).decode("ascii")
    return _PREFIX + token


def decrypt(value: str | None) -> str | None:
    """
    Decrypt a stored string.

    Values without the ciphertext marker are legacy plaintext written before
    encryption was introduced; they are returned as-is so the app keeps working
    against an existing database. Callers re-encrypt them on the next write.
    """
    if value is None or value == "":
        return value
    if not str(value).startswith(_PREFIX):
        return value  # legacy plaintext
    token = str(value)[len(_PREFIX):]
    try:
        return _cipher().decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken:
        logger.error("Failed to decrypt a stored value - wrong or rotated-away key.")
        raise CryptoError("Stored data could not be decrypted with the configured key.")


def is_encrypted(value: str | None) -> bool:
    return bool(value) and str(value).startswith(_PREFIX)


def verify_key() -> None:
    """
    Confirm the configured master key can unwrap the database's data key.

    Called at startup. Without this, booting against an existing database with
    the wrong key is silent: every decrypt fails, the read paths log and return
    None, and the UI simply shows "no portfolio imported yet" — which looks
    exactly like data loss and invites the user to re-import over the top of
    perfectly good ciphertext.

    Raises CryptoError when the key cannot unwrap existing data. A fresh
    database (no wrapped key yet) is fine.
    """
    from api import db

    if db.get_wrapped_dek() is None:
        return  # nothing encrypted yet; the first write creates the data key

    reset_cache()
    _cipher()  # raises CryptoError if the master key is wrong

    # Prove the data key actually decrypts what is stored, not just that it
    # unwrapped — a re-wrapped-but-mismatched key would otherwise pass.
    sample = db.sample_encrypted_value()
    if sample is not None:
        decrypt(sample)  # raises CryptoError on mismatch


if __name__ == "__main__":  # pragma: no cover - operator utility
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "keygen":
        print(generate_key())
    else:
        print("usage: python -m api.crypto keygen")
