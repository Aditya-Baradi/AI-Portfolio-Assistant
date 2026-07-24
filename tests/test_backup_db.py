"""Backup/restore disaster-recovery checks."""
from __future__ import annotations

import sqlite3

import pytest
from cryptography.fernet import Fernet

from scripts import backup_db


@pytest.fixture(autouse=True)
def backup_key(monkeypatch):
    monkeypatch.setenv("EVERGREEN_ENV", "development")
    monkeypatch.delenv("EVERGREEN_MASTER_KEY", raising=False)
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))


def _create_database(path, value="original"):
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE records (value TEXT NOT NULL)")
        connection.execute("INSERT INTO records VALUES (?)", (value,))
        connection.commit()
    finally:
        connection.close()


def _read_value(path):
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT value FROM records").fetchone()[0]
    finally:
        connection.close()


def test_compressed_backup_verifies_and_restores(tmp_path):
    source = tmp_path / "app.db"
    _create_database(source)

    archived = backup_db.backup(source, tmp_path / "backups")
    assert archived.suffix == ".enc"
    raw = archived.read_bytes()
    assert b"SQLite format 3" not in raw
    assert b"original" not in raw
    assert backup_db.integrity_ok(archived)

    restored = tmp_path / "restored.db"
    backup_db.restore(archived, restored)
    assert _read_value(restored) == "original"
    assert backup_db.integrity_ok(restored)


def test_restore_refuses_overwrite_without_force(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "destination.db"
    _create_database(source, "from-backup")
    _create_database(destination, "existing")
    archived = backup_db.backup(source, tmp_path / "backups")

    with pytest.raises(FileExistsError, match="--force"):
        backup_db.restore(archived, destination)
    assert _read_value(destination) == "existing"

    backup_db.restore(archived, destination, force=True)
    assert _read_value(destination) == "from-backup"


def test_tampered_encrypted_backup_is_rejected(tmp_path):
    source = tmp_path / "source.db"
    _create_database(source)
    archived = backup_db.backup(source, tmp_path / "backups")
    tampered = bytearray(archived.read_bytes())
    tampered[len(tampered) // 2] ^= 1
    archived.write_bytes(tampered)
    assert backup_db.integrity_ok(archived) is False


def test_production_backup_requires_distinct_encryption_key(tmp_path, monkeypatch):
    source = tmp_path / "source.db"
    _create_database(source)
    monkeypatch.setenv("EVERGREEN_ENV", "production")
    monkeypatch.delenv("BACKUP_ENCRYPTION_KEY")
    with pytest.raises(backup_db.BackupEncryptionError, match="required"):
        backup_db.backup(source, tmp_path / "backups")

    shared_key = Fernet.generate_key().decode("ascii")
    monkeypatch.setenv("EVERGREEN_MASTER_KEY", shared_key)
    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", shared_key)
    with pytest.raises(backup_db.BackupEncryptionError, match="distinct"):
        backup_db.backup(source, tmp_path / "backups")
    assert backup_db.backup_encryption_ready() is False

    monkeypatch.setenv("BACKUP_ENCRYPTION_KEY", Fernet.generate_key().decode("ascii"))
    assert backup_db.backup_encryption_ready() is True


def test_container_defaults_follow_data_dir(monkeypatch):
    monkeypatch.setenv("EVERGREEN_DATA_DIR", "/app/data")
    assert backup_db.default_db().as_posix() == "/app/data/app.db"
    assert backup_db.default_out().as_posix() == "/app/data/backups"
