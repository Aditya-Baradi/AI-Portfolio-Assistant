#!/usr/bin/env python
"""
Consistent backups of the Evergreen SQLite database.

Why not just copy app.db
------------------------
The app runs SQLite in WAL mode. A plain file copy of `app.db` while the server
is running can miss committed transactions still living in `app.db-wal`, or
capture a torn page mid-checkpoint. The result is a backup that looks fine and
restores corrupt.

This uses SQLite's own online backup API (`Connection.backup`), which takes a
transactionally consistent snapshot of a live database without stopping the
app.

What is NOT in the backup
-------------------------
The master encryption key. Portfolios, investor profiles and TOTP secrets are
encrypted at rest with a key derived from EVERGREEN_MASTER_KEY, which lives in
your secret manager, not the database.

The complete backup is additionally protected with authenticated encryption
from BACKUP_ENCRYPTION_KEY, a separate key that must never equal
EVERGREEN_MASTER_KEY. Production backup creation refuses to run without it.
Keep both keys in your secret manager and disaster-recovery escrow. Losing
either one makes the restored financial data unusable.
Back the keys up separately, and test that you can restore with them—an
untested backup is a hope, not a plan.

Usage
-----
    python scripts/backup_db.py                       # -> $EVERGREEN_DATA_DIR/backups
    python scripts/backup_db.py --out /mnt/backups    # custom directory
    python scripts/backup_db.py --keep 14             # retention (default 7)
    python scripts/backup_db.py --verify-only FILE    # integrity-check a backup
    python scripts/backup_db.py --restore FILE --to PATH
    python scripts/backup_db.py --generate-key        # new backup-encryption key

Scheduling
----------
cron (daily at 03:15, keeping 30 days):

    15 3 * * *  cd /app && /usr/local/bin/python scripts/backup_db.py --keep 30 \
                  >> /var/log/evergreen-backup.log 2>&1

Docker backup: run it in the app container so it sees the same volume, e.g.
`docker compose exec -T app python scripts/backup_db.py --keep 30`.

Restore only while the app is stopped:

    docker compose stop app
    docker compose run --rm app python scripts/backup_db.py \
        --restore /app/data/backups/app-...db.gz.enc --to /app/data/app.db --force
    docker compose start app

Copy backups to encrypted off-host/object storage and alert on missed jobs. A
backup on the same disk as the database protects against a bad migration, not
host loss or ransomware. Rehearse a restore at least quarterly.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import gzip
import os
import shutil
import sqlite3
import struct
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_KEEP = 7
_ENCRYPTED_MAGIC = b"EVERGREEN-BACKUP\x01"
_ENCRYPTION_CHUNK_BYTES = 1024 * 1024


class BackupEncryptionError(RuntimeError):
    """Backup key or authenticated-encryption failure."""


def _data_dir() -> Path:
    configured = os.getenv("EVERGREEN_DATA_DIR", "").strip()
    return Path(configured) if configured else Path(".")


def default_db() -> Path:
    return _data_dir() / "app.db"


def default_out() -> Path:
    return _data_dir() / "backups"


def _log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}] {msg}")


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _decode_backup_key(value: str) -> bytes:
    try:
        decoded = base64.b64decode(value.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError, binascii.Error) as exc:
        raise BackupEncryptionError(
            "BACKUP_ENCRYPTION_KEY must be a URL-safe base64-encoded 32-byte key"
        ) from exc
    if len(decoded) != 32:
        raise BackupEncryptionError(
            "BACKUP_ENCRYPTION_KEY must decode to exactly 32 bytes"
        )
    return decoded


def _backup_keys(*, required: bool) -> list[bytes]:
    raw = os.getenv("BACKUP_ENCRYPTION_KEY", "").strip()
    if not raw:
        if required:
            raise BackupEncryptionError(
                "BACKUP_ENCRYPTION_KEY is required for production backups"
            )
        return []
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    master_tokens = {
        token.strip()
        for token in os.getenv("EVERGREEN_MASTER_KEY", "").split(",")
        if token.strip()
    }
    if any(token in master_tokens for token in tokens):
        raise BackupEncryptionError(
            "BACKUP_ENCRYPTION_KEY must be distinct from EVERGREEN_MASTER_KEY"
        )
    return [_decode_backup_key(token) for token in tokens]


def backup_encryption_ready() -> bool:
    """Boolean readiness signal; never exposes or logs key material."""
    try:
        return bool(_backup_keys(required=True))
    except BackupEncryptionError:
        return False


def _read_exact(stream, size: int) -> bytes:
    value = stream.read(size)
    if len(value) != size:
        raise BackupEncryptionError("Encrypted backup is truncated")
    return value


def _encrypt_file(source: Path, target: Path, key: bytes) -> None:
    """Stream a file into a chunked AES-GCM container."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    cipher = AESGCM(key)
    nonce_prefix = os.urandom(8)
    counter = 0
    with source.open("rb") as plaintext, target.open("wb") as encrypted:
        encrypted.write(_ENCRYPTED_MAGIC)
        encrypted.write(nonce_prefix)
        while True:
            chunk = plaintext.read(_ENCRYPTION_CHUNK_BYTES)
            if not chunk:
                break
            if counter >= 2**32 - 1:
                raise BackupEncryptionError("Backup is too large for the encryption format")
            counter_bytes = counter.to_bytes(4, "big")
            sealed = cipher.encrypt(
                nonce_prefix + counter_bytes,
                chunk,
                _ENCRYPTED_MAGIC + counter_bytes,
            )
            encrypted.write(struct.pack(">I", len(chunk)))
            encrypted.write(sealed)
            counter += 1

        # A mandatory authenticated final record detects clean truncation at a
        # chunk boundary as well as arbitrary byte corruption.
        counter_bytes = counter.to_bytes(4, "big")
        final_tag = cipher.encrypt(
            nonce_prefix + counter_bytes,
            b"",
            _ENCRYPTED_MAGIC + counter_bytes + b":final",
        )
        encrypted.write(struct.pack(">I", 0))
        encrypted.write(final_tag)


def _decrypt_file(source: Path, target: Path, key: bytes) -> None:
    """Authenticate and stream-decrypt one encrypted backup."""
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    cipher = AESGCM(key)
    try:
        with source.open("rb") as encrypted, target.open("wb") as plaintext:
            if _read_exact(encrypted, len(_ENCRYPTED_MAGIC)) != _ENCRYPTED_MAGIC:
                raise BackupEncryptionError("Not an Evergreen encrypted backup")
            nonce_prefix = _read_exact(encrypted, 8)
            counter = 0
            while True:
                length = struct.unpack(">I", _read_exact(encrypted, 4))[0]
                if length > _ENCRYPTION_CHUNK_BYTES:
                    raise BackupEncryptionError("Encrypted backup has an invalid chunk size")
                counter_bytes = counter.to_bytes(4, "big")
                sealed = _read_exact(encrypted, length + 16)
                if length == 0:
                    cipher.decrypt(
                        nonce_prefix + counter_bytes,
                        sealed,
                        _ENCRYPTED_MAGIC + counter_bytes + b":final",
                    )
                    if encrypted.read(1):
                        raise BackupEncryptionError(
                            "Encrypted backup has trailing unauthenticated data"
                        )
                    break
                chunk = cipher.decrypt(
                    nonce_prefix + counter_bytes,
                    sealed,
                    _ENCRYPTED_MAGIC + counter_bytes,
                )
                plaintext.write(chunk)
                counter += 1
    except InvalidTag as exc:
        raise BackupEncryptionError(
            "Encrypted backup authentication failed (wrong key or corrupt file)"
        ) from exc


def _decrypt_with_available_key(source: Path, target: Path) -> None:
    keys = _backup_keys(required=True)
    last_error: BackupEncryptionError | None = None
    for key in keys:
        try:
            _decrypt_file(source, target, key)
            return
        except BackupEncryptionError as exc:
            last_error = exc
            target.unlink(missing_ok=True)
    raise last_error or BackupEncryptionError("No backup decryption key is available")


def _looks_gzipped(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(2) == b"\x1f\x8b"


@contextlib.contextmanager
def _materialized(path: Path):
    """Yield SQLite, transparently decrypting and expanding a backup."""
    temporaries: list[Path] = []
    try:
        current = path
        if path.suffix.lower() == ".enc":
            fd, name = tempfile.mkstemp(prefix="evergreen-decrypt-", suffix=".payload")
            os.close(fd)
            decrypted = Path(name)
            temporaries.append(decrypted)
            _decrypt_with_available_key(path, decrypted)
            current = decrypted

        if current.suffix.lower() == ".gz" or _looks_gzipped(current):
            fd, name = tempfile.mkstemp(prefix="evergreen-verify-", suffix=".db")
            os.close(fd)
            expanded = Path(name)
            temporaries.append(expanded)
            with gzip.open(current, "rb") as compressed, expanded.open("wb") as output:
                shutil.copyfileobj(compressed, output)
            current = expanded
        yield current
    finally:
        for temporary in temporaries:
            temporary.unlink(missing_ok=True)


def integrity_ok(path: Path) -> bool:
    """Run SQLite's own integrity check against a .db or .db.gz backup."""
    try:
        with _materialized(path) as materialized:
            conn = sqlite3.connect(_readonly_uri(materialized), uri=True)
            try:
                result = conn.execute("PRAGMA integrity_check").fetchone()
            finally:
                # sqlite3.Connection's context manager commits/rolls back but
                # does not close. Explicit close is required before Windows can
                # unlink/replace the staging file.
                conn.close()
        return bool(result) and result[0] == "ok"
    except (
        BackupEncryptionError,
        OSError,
        EOFError,
        gzip.BadGzipFile,
        sqlite3.Error,
    ) as e:
        _log(f"ERROR: integrity check could not run on {path}: {e}")
        return False


def _copy_sqlite(source: Path, target: Path) -> None:
    """Create a transactionally consistent SQLite copy at target."""
    src = sqlite3.connect(_readonly_uri(source), uri=True)
    try:
        dst = sqlite3.connect(target)
        try:
            src.backup(dst)
        finally:
            dst.close()
    finally:
        src.close()


def _secure_mode(path: Path) -> None:
    """Backups contain sensitive financial data even though fields are encrypted."""
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ACLs do not map cleanly to POSIX modes.
        pass


def backup(db_path: Path, out_dir: Path, compress: bool = True) -> Path:
    """Take an atomically published, verified snapshot of a live database."""
    if not db_path.is_file():
        raise FileNotFoundError(f"No database at {db_path}")

    production = os.getenv("EVERGREEN_ENV", "development").strip().lower() == "production"
    keys = _backup_keys(required=production)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base_name = f"app-{stamp}.db"
    sqlite_staging = out_dir / f".{base_name}.{uuid.uuid4().hex}.tmp"
    payload_staging: Path | None = None
    encrypted_staging: Path | None = None

    try:
        _copy_sqlite(db_path, sqlite_staging)
        _secure_mode(sqlite_staging)
        if not integrity_ok(sqlite_staging):
            raise RuntimeError("Backup failed its integrity check.")

        size = sqlite_staging.stat().st_size
        if compress:
            base_name += ".gz"
            payload_staging = out_dir / f".{base_name}.{uuid.uuid4().hex}.tmp"
            with sqlite_staging.open("rb") as source, gzip.open(
                payload_staging, "wb", compresslevel=6
            ) as output:
                shutil.copyfileobj(source, output)
            _secure_mode(payload_staging)
        else:
            payload_staging = sqlite_staging

        if keys:
            base_name += ".enc"
            encrypted_staging = out_dir / f".{base_name}.{uuid.uuid4().hex}.tmp"
            _encrypt_file(payload_staging, encrypted_staging, keys[0])
            _secure_mode(encrypted_staging)
            target = out_dir / base_name
            os.replace(encrypted_staging, target)
        else:
            target = out_dir / base_name
            os.replace(payload_staging, target)
            _log(
                "WARNING: wrote an unencrypted development backup because "
                "BACKUP_ENCRYPTION_KEY is not set"
            )

        sqlite_staging.unlink(missing_ok=True)
        if not integrity_ok(target):
            target.unlink(missing_ok=True)
            raise RuntimeError("Published backup failed decrypt/decompress verification.")
        _log(
            f"Wrote {target} "
            f"({size:,} database bytes -> {target.stat().st_size:,} stored)"
        )
        return target
    finally:
        sqlite_staging.unlink(missing_ok=True)
        if payload_staging is not None:
            payload_staging.unlink(missing_ok=True)
        if encrypted_staging is not None:
            encrypted_staging.unlink(missing_ok=True)


def restore(source: Path, destination: Path, force: bool = False) -> Path:
    """Verify and atomically restore a backup while the application is stopped."""
    if not source.is_file():
        raise FileNotFoundError(f"No backup at {source}")
    sidecars = [Path(f"{destination}-wal"), Path(f"{destination}-shm")]
    if (destination.exists() or any(path.exists() for path in sidecars)) and not force:
        raise FileExistsError(
            f"{destination} already exists (or has WAL sidecars); stop the app and "
            "pass --force to replace it"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.restore-{uuid.uuid4().hex}.tmp"
    try:
        with _materialized(source) as materialized:
            if not integrity_ok(materialized):
                raise RuntimeError("Backup is corrupt; restore aborted.")
            _copy_sqlite(materialized, staging)
        _secure_mode(staging)
        if not integrity_ok(staging):
            raise RuntimeError("Restored staging database failed integrity check.")
        if force:
            for sidecar in sidecars:
                sidecar.unlink(missing_ok=True)
        os.replace(staging, destination)
        if not integrity_ok(destination):
            raise RuntimeError("Restored database failed its final integrity check.")
        _log(f"Restored {source} -> {destination}")
        return destination
    finally:
        staging.unlink(missing_ok=True)


def prune(out_dir: Path, keep: int) -> int:
    """Delete all but the `keep` most recent backups. Returns how many went."""
    if keep < 1:
        raise ValueError("--keep must be at least 1")
    backups = sorted(
        [p for p in out_dir.glob("app-*.db*") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    removed = 0
    for old in backups[keep:]:
        try:
            old.unlink()
            removed += 1
            _log(f"Pruned {old.name}")
        except OSError as e:
            _log(f"WARNING: could not prune {old.name}: {e}")
    return removed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db",
        type=Path,
        default=default_db(),
        help="database to back up (default: EVERGREEN_DATA_DIR/app.db)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=default_out(),
        help="output directory (default: EVERGREEN_DATA_DIR/backups)",
    )
    parser.add_argument("--keep", type=int, default=DEFAULT_KEEP,
                        help=f"how many backups to retain (default {DEFAULT_KEEP})")
    parser.add_argument("--no-compress", action="store_true", help="skip gzip")
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--verify-only", type=Path, metavar="FILE",
                           help="integrity-check an existing backup and exit")
    operation.add_argument(
        "--restore",
        type=Path,
        metavar="FILE",
        help="verified backup to restore (.db, .db.gz, or encrypted .enc)",
    )
    operation.add_argument("--generate-key", action="store_true",
                           help="print a new BACKUP_ENCRYPTION_KEY and exit")
    parser.add_argument("--to", type=Path, metavar="PATH",
                        help="restore destination (default: EVERGREEN_DATA_DIR/app.db)")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing restore destination and WAL sidecars")
    args = parser.parse_args(argv)

    if args.generate_key:
        from cryptography.fernet import Fernet

        print(Fernet.generate_key().decode("ascii"))
        return 0

    if args.verify_only:
        ok = integrity_ok(args.verify_only)
        _log(f"{args.verify_only}: {'ok' if ok else 'CORRUPT'}")
        return 0 if ok else 1

    try:
        if args.restore:
            restore(args.restore, args.to or default_db(), force=args.force)
            _log(
                "Restore complete. Start the app with the SAME "
                "EVERGREEN_MASTER_KEY used by this database."
            )
            return 0
        if args.to or args.force:
            parser.error("--to and --force are valid only with --restore")
        backup(args.db, args.out, compress=not args.no_compress)
        prune(args.out, args.keep)
    except Exception as e:
        _log(f"ERROR: {e}")
        return 1

    _log("Backup complete. Remember: the encryption key is NOT in this file — "
         "back up EVERGREEN_MASTER_KEY separately or the data is unrecoverable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
