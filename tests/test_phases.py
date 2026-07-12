"""Tests for the security + feature phases (offline)."""
import pytest

from api.recommend import monthly_needed_for, rebalance_trades, retirement_paths


@pytest.fixture()
def fresh_db(tmp_path, monkeypatch):
    import api.db as db

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_db()
    return db


class TestHashedTokens:
    def test_token_not_stored_in_plaintext(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        token = db.issue_token(uid)
        with db._connect() as conn:
            stored = conn.execute("SELECT token FROM auth_tokens").fetchone()["token"]
        assert stored != token
        assert len(stored) == 64  # sha256 hex
        assert db.user_for_token(token)["id"] == uid
        assert db.user_for_token(stored) is None  # the stored hash itself is useless

    def test_revoke_all(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        t1, t2 = db.issue_token(uid), db.issue_token(uid)
        db.revoke_all_tokens(uid)
        assert db.user_for_token(t1) is None and db.user_for_token(t2) is None

    def test_legacy_plaintext_tokens_purged_on_init(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        with db._connect() as conn:
            conn.execute(
                "INSERT INTO auth_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
                ("legacy-plaintext-token", uid, "2099-01-01T00:00:00"),
            )
        db.init_db()  # migration runs
        with db._connect() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM auth_tokens").fetchone()["n"]
        assert n == 0


class TestDisplayName:
    def test_register_with_name(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1", "  Aditya  ")
        assert db.get_user_name(uid) == "Aditya"  # trimmed
        token = db.issue_token(uid)
        assert db.user_for_token(token)["name"] == "Aditya"

    def test_register_without_name(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        assert db.get_user_name(uid) is None

    def test_set_and_clear_name(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        db.set_user_name(uid, "Casey")
        assert db.get_user_name(uid) == "Casey"
        db.set_user_name(uid, "   ")
        assert db.get_user_name(uid) is None  # blank clears it

    def test_name_capped_at_60_chars(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1", "x" * 200)
        assert len(db.get_user_name(uid)) == 60


class TestPasswordChange:
    def test_change_and_old_sessions_die(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        old_token = db.issue_token(uid)
        db.change_password(uid, "longpassword1", "evenlongerpassword2")
        assert db.user_for_token(old_token) is None
        assert db.verify_login("a@x.com", "evenlongerpassword2") == uid

    def test_wrong_current_rejected(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        with pytest.raises(db.AuthError):
            db.change_password(uid, "wrong", "evenlongerpassword2")

    def test_verify_password(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        assert db.verify_password(uid, "longpassword1") is True
        assert db.verify_password(uid, "nope") is False


class TestAccountLifecycle:
    def test_delete_user_cascades(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        db.issue_token(uid)
        db.save_portfolio(uid, {"holdings": []})
        db.add_watch(uid, "AAPL")
        db.log_event(uid, "login")
        db.delete_user(uid)
        with db._connect() as conn:
            for table in ("users", "auth_tokens", "portfolios", "watchlist", "events"):
                n = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                assert n == 0, table

    def test_events_capped_at_100(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        for _ in range(120):
            db.log_event(uid, "login")
        with db._connect() as conn:
            n = conn.execute("SELECT COUNT(*) AS n FROM events").fetchone()["n"]
        assert n == 100

    def test_totp_secret_roundtrip(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        assert db.get_totp_secret(uid) is None
        db.set_totp_secret(uid, "BASE32SECRET")
        assert db.get_totp_secret(uid) == "BASE32SECRET"
        db.set_totp_secret(uid, None)
        assert db.get_totp_secret(uid) is None


class TestWatchlistAndAlerts:
    def test_watchlist_roundtrip(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        db.add_watch(uid, "amd")
        db.add_watch(uid, "AMD")  # idempotent
        assert db.list_watchlist(uid) == ["AMD"]
        db.remove_watch(uid, "AMD")
        assert db.list_watchlist(uid) == []

    def test_alerts_dedupe_unseen(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        db.add_alert(uid, "News about NVDA has turned negative.", ticker="NVDA")
        db.add_alert(uid, "News about NVDA has turned negative.", ticker="NVDA")
        assert len(db.list_alerts(uid)) == 1
        db.mark_alerts_seen(uid)
        db.add_alert(uid, "News about NVDA has turned negative.", ticker="NVDA")
        assert len(db.list_alerts(uid)) == 2  # seen ones don't block new alerts

    def test_alerts_due_schedule(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        assert db.alerts_due(uid) is True
        db.mark_alerts_checked(uid)
        assert db.alerts_due(uid) is False
        assert db.alerts_due(uid, hours=0) is True

    def test_daily_message_count(self, fresh_db):
        db = fresh_db
        uid = db.register_user("a@x.com", "longpassword1")
        db.touch_chat(uid, "c1", "hello")
        db.add_message("c1", "user", "hello")
        db.add_message("c1", "assistant", "hi")
        db.add_message("c1", "user", "again")
        assert db.count_user_messages_today(uid) == 2


class TestGoalMath:
    def test_p_goal_reported(self):
        out = retirement_paths(0.07, 0.15, 100_000, 500, years=20, goal=200_000)
        assert 0.0 <= out["p_goal"] <= 1.0
        assert out["goal"] == 200_000

    def test_sure_thing_and_impossible(self):
        sure = retirement_paths(0.0, 0.0, 1000, 100, years=5, goal=2000)
        assert sure["p_goal"] == 1.0  # 7000 total, zero risk
        impossible = retirement_paths(0.0, 0.0, 1000, 0, years=5, goal=999_999)
        assert impossible["p_goal"] == 0.0

    def test_monthly_needed_zero_when_already_likely(self):
        assert monthly_needed_for(1000, 0.0, 0.0, 100_000, years=5) == 0.0

    def test_monthly_needed_solves(self):
        # zero growth, zero vol: need exactly (goal - value0) / months
        needed = monthly_needed_for(120_000, 0.0, 0.0, 0, years=10)
        assert needed == pytest.approx(1000, abs=25)

    def test_monthly_needed_none_when_hopeless(self):
        assert monthly_needed_for(10**12, 0.0, 0.0, 0, years=1) is None


class TestRebalanceTrades:
    def test_sells_first_and_amounts(self):
        trades = rebalance_trades(
            {"NVDA": 800, "KO": 200}, {"NVDA": 0.5, "KO": 0.5}, total_value=1000
        )
        assert trades[0] == {"ticker": "NVDA", "action": "sell", "dollars": 300.0}
        assert trades[1] == {"ticker": "KO", "action": "buy", "dollars": 300.0}

    def test_new_ticker_is_a_buy(self):
        trades = rebalance_trades({"NVDA": 1000}, {"NVDA": 0.7, "KO": 0.3}, 1000)
        assert {"ticker": "KO", "action": "buy", "dollars": 300.0} in trades

    def test_tiny_trades_dropped(self):
        trades = rebalance_trades({"A": 501, "B": 499}, {"A": 0.5, "B": 0.5}, 1000)
        assert trades == []

    def test_empty_inputs(self):
        assert rebalance_trades({}, {"A": 1.0}, 1000) == []
        assert rebalance_trades({"A": 100}, {"A": 1.0}, 0) == []
