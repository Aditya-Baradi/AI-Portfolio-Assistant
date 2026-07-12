"""Tests for transaction-history import: detection, replay, row extraction (offline)."""
import json

import pytest

from api.portfolio_core import (
    extract_raw_rows,
    looks_like_transactions,
    replay_transactions,
)


SNAPSHOT = [
    {"date": "2026-05-31", "ticker": "NVDA", "close": 211.14, "volume": 0.71, "current_dollars": 149.91},
    {"date": "2026-05-31", "ticker": "XOM", "close": 145.26, "volume": 0.41, "current_dollars": 59.98},
]

ACTIVITY = [
    {"date": "2026-02-13", "ticker": "TSLA", "close": 416.25, "volume": 0.05, "current_dollars": 20.81},
    {"date": "2026-06-30", "ticker": "PANW", "close": 340.61, "volume": 0.069, "current_dollars": -23.5},
    {"date": "2026-06-08", "ticker": "NVDA", "close": None, "volume": None, "current_dollars": 0.14},
]


class TestDetection:
    def test_snapshot_is_not_transactions(self):
        assert looks_like_transactions(SNAPSHOT) is False

    def test_negative_amounts_flag_transactions(self):
        assert looks_like_transactions(ACTIVITY) is True

    def test_repeated_ticker_across_dates_flags_transactions(self):
        rows = [
            {"date": "2026-01-02", "ticker": "AAPL", "close": 100, "volume": 1, "current_dollars": 100},
            {"date": "2026-03-02", "ticker": "AAPL", "close": 110, "volume": 1, "current_dollars": 110},
        ]
        assert looks_like_transactions(rows) is True

    def test_empty_and_junk(self):
        assert looks_like_transactions([]) is False
        assert looks_like_transactions(None) is False
        assert looks_like_transactions(["junk", 3]) is False


class TestReplay:
    def test_buys_accumulate_weighted_avg_cost(self):
        rows = [
            {"date": "2026-01-02", "ticker": "AAPL", "close": 100.0, "volume": 1.0, "current_dollars": 100.0},
            {"date": "2026-02-02", "ticker": "AAPL", "close": 200.0, "volume": 1.0, "current_dollars": 200.0},
        ]
        p = replay_transactions(rows)["positions"]["AAPL"]
        assert p["shares"] == pytest.approx(2.0)
        assert p["avg_cost"] == pytest.approx(150.0)
        assert p["first_buy"] == "2026-01-02"

    def test_partial_sell_keeps_avg_cost(self):
        rows = [
            {"date": "2026-01-02", "ticker": "AAPL", "close": 100.0, "volume": 2.0, "current_dollars": 200.0},
            {"date": "2026-02-02", "ticker": "AAPL", "close": 300.0, "volume": 1.0, "current_dollars": -300.0},
        ]
        p = replay_transactions(rows)["positions"]["AAPL"]
        assert p["shares"] == pytest.approx(1.0)
        assert p["avg_cost"] == pytest.approx(100.0)  # selling doesn't change cost basis

    def test_full_sell_closes_position(self):
        rows = [
            {"date": "2026-01-02", "ticker": "GS", "close": 100.0, "volume": 1.0, "current_dollars": 100.0},
            {"date": "2026-02-02", "ticker": "GS", "close": 120.0, "volume": 1.0, "current_dollars": -120.0},
        ]
        out = replay_transactions(rows)
        assert "GS" not in out["positions"]
        assert "GS" in out["closed_tickers"]

    def test_dividends_counted_but_ignored(self):
        rows = [{"date": "2026-01-02", "ticker": "NVDA", "close": None, "volume": None, "current_dollars": 0.14}]
        out = replay_transactions(rows)
        assert out["positions"] == {}
        assert out["n_cash_events"] == 1

    def test_price_derived_from_amount_when_close_missing(self):
        rows = [{"date": "2026-01-02", "ticker": "XOM", "close": None, "volume": 2.0, "current_dollars": 100.0}]
        p = replay_transactions(rows)["positions"]["XOM"]
        assert p["avg_cost"] == pytest.approx(50.0)

    def test_shares_only_rows_are_skipped(self):
        rows = [{"date": "2026-01-02", "ticker": "CRWD", "close": None, "volume": 0.29, "current_dollars": None}]
        out = replay_transactions(rows)
        assert out["skipped_tickers"] == ["CRWD"]
        assert out["positions"] == {}

    def test_rows_sorted_by_date_before_replay(self):
        # The sell appears first in the file but happens after the buy.
        rows = [
            {"date": "2026-02-02", "ticker": "AAPL", "close": 300.0, "volume": 2.0, "current_dollars": -600.0},
            {"date": "2026-01-02", "ticker": "AAPL", "close": 100.0, "volume": 2.0, "current_dollars": 200.0},
        ]
        assert "AAPL" not in replay_transactions(rows)["positions"]


class TestExtractRows:
    def test_json_list(self):
        rows = extract_raw_rows("a.json", json.dumps(ACTIVITY).encode())
        assert len(rows) == 3

    def test_json_nested_transactions_key(self):
        raw = json.dumps({"transactions": ACTIVITY}).encode()
        assert len(extract_raw_rows("a.json", raw)) == 3

    def test_csv_rows_with_nan_become_none(self):
        raw = b"ticker,date,close,volume,current_dollars\nAAPL,2026-01-02,100,1,100\nNVDA,2026-01-03,,,0.14\n"
        rows = extract_raw_rows("a.csv", raw)
        assert rows[0]["ticker"] == "AAPL"
        assert rows[1]["close"] is None

    def test_garbage_returns_none(self):
        assert extract_raw_rows("a.json", b"{not json") is None
        assert extract_raw_rows("a.txt", b"hello") is None
