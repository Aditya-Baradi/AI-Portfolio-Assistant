"""
Encryption at rest: the envelope scheme, key rotation, and the guarantee that
sensitive columns are unreadable in a stolen database file.
"""
from __future__ import annotations

import json
import sqlite3

import pytest

from api import crypto


@pytest.fixture(autouse=True)
def fresh_db(tmp_path, monkeypatch):
    """Isolated database + a known master key for every test."""
    import api.db as db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
    crypto.reset_cache()
    db.init_db()
    yield db
    crypto.reset_cache()


class TestRoundTrip:
    def test_encrypt_decrypt(self):
        assert crypto.decrypt(crypto.encrypt("hello")) == "hello"

    def test_ciphertext_is_not_plaintext(self):
        token = crypto.encrypt("SUPERSECRET")
        assert "SUPERSECRET" not in token
        assert crypto.is_encrypted(token)

    def test_none_and_empty_pass_through(self):
        assert crypto.encrypt(None) is None
        assert crypto.encrypt("") == ""
        assert crypto.decrypt(None) is None
        assert crypto.decrypt("") == ""

    def test_nondeterministic(self):
        """Fernet includes a random IV, so equal inputs must differ as ciphertext."""
        assert crypto.encrypt("same") != crypto.encrypt("same")

    def test_unicode_survives(self):
        value = "Ünïcodé — 日本語 — 🎉"
        assert crypto.decrypt(crypto.encrypt(value)) == value


class TestLegacyPlaintext:
    def test_unmarked_values_pass_through(self):
        """A database written before encryption existed must still be readable."""
        assert crypto.decrypt("JBSWY3DPEHPK3PXP") == "JBSWY3DPEHPK3PXP"
        assert not crypto.is_encrypted("JBSWY3DPEHPK3PXP")


class TestKeyRotation:
    def test_old_key_still_decrypts_after_rotation(self, monkeypatch):
        """
        Envelope encryption means rotating the master key re-wraps the data key
        only — previously encrypted rows stay readable without a data migration.
        """
        original = crypto.encrypt("holdings")

        old_key = crypto._master_keys()[0]
        new_key = crypto.generate_key()
        monkeypatch.setenv("EVERGREEN_MASTER_KEY", f"{new_key},{old_key}")
        crypto.reset_cache()

        assert crypto.decrypt(original) == "holdings"
        # And new writes work under the rotated key too.
        assert crypto.decrypt(crypto.encrypt("fresh")) == "fresh"

    def test_wrong_key_cannot_decrypt(self, monkeypatch):
        original = crypto.encrypt("holdings")
        monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
        crypto.reset_cache()
        with pytest.raises(crypto.CryptoError):
            crypto.decrypt(original)


class TestProductionGuard:
    def test_missing_key_in_production_is_fatal(self, monkeypatch):
        monkeypatch.delenv("EVERGREEN_MASTER_KEY", raising=False)
        monkeypatch.setenv("EVERGREEN_ENV", "production")
        monkeypatch.setattr(crypto, "DEV_KEY_FILE", crypto.Path("nonexistent-key-file"))
        crypto.reset_cache()
        with pytest.raises(crypto.CryptoError, match="EVERGREEN_MASTER_KEY"):
            crypto.encrypt("anything")

    def test_invalid_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv("EVERGREEN_MASTER_KEY", "not-a-valid-fernet-key")
        crypto.reset_cache()
        with pytest.raises(crypto.CryptoError):
            crypto.encrypt("anything")


class TestDatabaseIsOpaque:
    """The point of all this: a stolen app.db must not yield secrets."""

    def test_portfolio_is_not_readable_from_the_file(self, fresh_db):
        db = fresh_db
        uid = db.register_user("vault@example.com", "password123")
        db.save_portfolio(uid, {"holdings": [{"ticker": "NVDA", "shares": 100}]})

        raw = sqlite3.connect(db.DB_PATH).execute(
            "SELECT data FROM portfolios WHERE user_id = ?", (uid,)).fetchone()[0]

        assert "NVDA" not in raw
        assert crypto.is_encrypted(raw)
        # ...but the application still reads it correctly.
        assert db.get_portfolio(uid)["holdings"][0]["ticker"] == "NVDA"

    def test_totp_secret_is_not_readable_from_the_file(self, fresh_db):
        db = fresh_db
        uid = db.register_user("2fa@example.com", "password123")
        db.set_totp_secret(uid, "JBSWY3DPEHPK3PXP")

        raw = sqlite3.connect(db.DB_PATH).execute(
            "SELECT totp_secret FROM users WHERE id = ?", (uid,)).fetchone()[0]

        assert "JBSWY3DPEHPK3PXP" not in raw
        assert db.get_totp_secret(uid) == "JBSWY3DPEHPK3PXP"

    def test_profile_is_not_readable_from_the_file(self, fresh_db):
        db = fresh_db
        uid = db.register_user("plan@example.com", "password123")
        db.save_profile(uid, {"goal_amount": 1234567, "risk_tolerance": 7})

        raw = sqlite3.connect(db.DB_PATH).execute(
            "SELECT data FROM profiles WHERE user_id = ?", (uid,)).fetchone()[0]

        assert "1234567" not in raw
        assert db.get_profile(uid)["goal_amount"] == 1234567


class TestMigration:
    def test_existing_plaintext_rows_get_encrypted(self, fresh_db):
        db = fresh_db
        uid = db.register_user("legacy@example.com", "password123")

        # Simulate rows written by a pre-encryption version of the app.
        plain = json.dumps({"holdings": [{"ticker": "AAPL", "shares": 5}]})
        with sqlite3.connect(db.DB_PATH) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO portfolios (user_id, data, updated_at) VALUES (?, ?, ?)",
                (uid, plain, "2024-01-01T00:00:00"))
            conn.execute("UPDATE users SET totp_secret = ? WHERE id = ?",
                         ("PLAINTEXTSECRET", uid))

        # Readable before the migration (transparent passthrough)...
        assert db.get_portfolio(uid)["holdings"][0]["ticker"] == "AAPL"
        assert db.get_totp_secret(uid) == "PLAINTEXTSECRET"

        changed = db.encrypt_existing_rows()
        assert changed >= 2

        with sqlite3.connect(db.DB_PATH) as conn:
            pf_raw = conn.execute("SELECT data FROM portfolios WHERE user_id = ?",
                                  (uid,)).fetchone()[0]
            totp_raw = conn.execute("SELECT totp_secret FROM users WHERE id = ?",
                                    (uid,)).fetchone()[0]
        assert crypto.is_encrypted(pf_raw) and "AAPL" not in pf_raw
        assert crypto.is_encrypted(totp_raw) and "PLAINTEXT" not in totp_raw

        # ...and still readable after.
        assert db.get_portfolio(uid)["holdings"][0]["ticker"] == "AAPL"
        assert db.get_totp_secret(uid) == "PLAINTEXTSECRET"

    def test_migration_is_idempotent(self, fresh_db):
        db = fresh_db
        uid = db.register_user("idem@example.com", "password123")
        db.save_portfolio(uid, {"holdings": []})
        db.encrypt_existing_rows()
        assert db.encrypt_existing_rows() == 0  # nothing left to convert


class TestStartupKeyVerification:
    """
    Regression guard for a real incident: the app was started against an
    existing database with a throwaway key. Every decrypt failed, the read
    paths logged and returned None, and the UI showed "no portfolio imported
    yet" — indistinguishable from data loss. Booting must fail loudly instead.
    """

    def test_fresh_database_passes(self, fresh_db):
        crypto.verify_key()  # nothing encrypted yet; must not raise

    def test_correct_key_passes(self, fresh_db):
        db = fresh_db
        uid = db.register_user("ok@example.com", "password123")
        db.save_portfolio(uid, {"holdings": [{"ticker": "AAPL"}]})
        crypto.verify_key()

    def test_wrong_key_is_detected(self, fresh_db, monkeypatch):
        db = fresh_db
        uid = db.register_user("bad@example.com", "password123")
        db.save_portfolio(uid, {"holdings": [{"ticker": "AAPL"}]})

        monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
        crypto.reset_cache()
        with pytest.raises(crypto.CryptoError):
            crypto.verify_key()

    def test_database_initialization_fails_closed_on_wrong_key(
        self,
        fresh_db,
        monkeypatch,
    ):
        db = fresh_db
        uid = db.register_user("init-key@example.com", "password123")
        db.save_profile(uid, {"private": "value"})
        monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
        crypto.reset_cache()
        with pytest.raises(db.EncryptionMigrationError, match="refusing to serve"):
            db.init_db()

    def test_rotated_key_with_old_value_passes(self, fresh_db, monkeypatch):
        db = fresh_db
        uid = db.register_user("rot@example.com", "password123")
        db.save_portfolio(uid, {"holdings": [{"ticker": "AAPL"}]})

        old = crypto._master_keys()[0]
        monkeypatch.setenv("EVERGREEN_MASTER_KEY", f"{crypto.generate_key()},{old}")
        crypto.reset_cache()
        crypto.verify_key()  # old key retained, so this must succeed

    def test_backend_refuses_to_start_on_mismatch(self, fresh_db, monkeypatch):
        db = fresh_db
        uid = db.register_user("boot@example.com", "password123")
        db.save_portfolio(uid, {"holdings": [{"ticker": "AAPL"}]})

        # Import while the configured key is still valid. db.init_db now also
        # fails closed during import on an unwrap error; this test specifically
        # exercises the backend's explicit startup verification boundary.
        from api import backend

        monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
        crypto.reset_cache()

        with pytest.raises(RuntimeError, match="Refusing to start"):
            backend._verify_encryption_key()

    def test_sample_encrypted_value_finds_ciphertext(self, fresh_db):
        db = fresh_db
        assert db.sample_encrypted_value() is None
        uid = db.register_user("s@example.com", "password123")
        db.save_portfolio(uid, {"holdings": []})
        assert crypto.is_encrypted(db.sample_encrypted_value())
