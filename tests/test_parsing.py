"""Tests for portfolio file parsing and weight derivation (offline)."""
import json

import pytest

from api.portfolio_core import (
    parse_portfolio_file,
    weights_from_holdings_json,
    save_session_portfolio,
    get_session_portfolio,
    SESSION_PORTFOLIOS,
)


CSV_OK = b"ticker,shares,price\nAAPL,10,150\nMSFT,5,300\n"
JSON_OK = json.dumps([
    {"ticker": "AAPL", "shares": 10, "close": 150.0},
    {"ticker": "MSFT", "shares": 5, "close": 300.0},
]).encode()


class TestParsePortfolioFile:
    def test_csv_happy_path(self):
        out = parse_portfolio_file("holdings.csv", CSV_OK)
        h = out["holdings"]
        assert len(h) == 2
        assert h[0]["ticker"] == "AAPL"
        assert h[0]["current_dollars"] == pytest.approx(1500.0)

    def test_json_happy_path(self):
        out = parse_portfolio_file("holdings.json", JSON_OK)
        h = out["holdings"]
        assert len(h) == 2
        assert h[1]["current_dollars"] == pytest.approx(1500.0)

    def test_json_value_only_infers_shares(self):
        raw = json.dumps([{"ticker": "NVDA", "current_dollars": 500.0, "close": 250.0}]).encode()
        out = parse_portfolio_file("x.json", raw)
        assert out["holdings"][0]["shares"] == pytest.approx(2.0)

    def test_csv_missing_columns_raises(self):
        with pytest.raises(ValueError):
            parse_portfolio_file("bad.csv", b"name,amount\nfoo,1\n")

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError):
            parse_portfolio_file("holdings.xlsx", b"whatever")

    def test_json_dict_with_holdings_key(self):
        raw = json.dumps({"holdings": [{"ticker": "AMD", "shares": 1, "price": 100}]}).encode()
        out = parse_portfolio_file("x.json", raw)
        assert out["holdings"][0]["ticker"] == "AMD"

    def test_json_preserves_purchase_date(self):
        raw = json.dumps([
            {"ticker": "NVDA", "shares": 1, "close": 200.0, "date": "2025-11-12"},
            {"ticker": "AMD", "shares": 1, "close": 100.0},
        ]).encode()
        out = parse_portfolio_file("x.json", raw)
        assert out["holdings"][0]["purchase_date"] == "2025-11-12"
        assert "purchase_date" not in out["holdings"][1]

    def test_csv_preserves_purchase_date(self):
        raw = b"ticker,shares,price,date\nAAPL,10,150,2024-06-01\n"
        out = parse_portfolio_file("x.csv", raw)
        assert out["holdings"][0]["purchase_date"] == "2024-06-01"


class TestWeightsFromHoldings:
    def test_weights_sum_to_one(self):
        holdings = json.dumps([
            {"ticker": "AAPL", "current_dollars": 750.0},
            {"ticker": "MSFT", "current_dollars": 250.0},
        ])
        w = weights_from_holdings_json(holdings)
        assert w["AAPL"] == pytest.approx(0.75)
        assert sum(w.values()) == pytest.approx(1.0)

    def test_zero_total_raises(self):
        holdings = json.dumps([{"ticker": "AAPL", "current_dollars": 0.0}])
        with pytest.raises(ValueError):
            weights_from_holdings_json(holdings)


class TestCostBasis:
    def test_json_explicit_avg_cost_parsed(self):
        raw = json.dumps([
            {"ticker": "NVDA", "shares": 0.5, "close": 202.49, "date": "2025-11-12", "avg_cost": 122.35},
        ]).encode()
        out = parse_portfolio_file("x.json", raw)
        assert out["holdings"][0]["purchase_price"] == pytest.approx(122.35)

    def test_csv_explicit_avg_cost_parsed(self):
        raw = b"ticker,shares,price,avg_cost\nAAPL,10,150,97.5\n"
        out = parse_portfolio_file("x.csv", raw)
        assert out["holdings"][0]["purchase_price"] == pytest.approx(97.5)

    def test_holdings_info_prefers_cost_over_snapshot(self):
        from api.portfolio_core import holdings_info
        pf = {"holdings": [
            {"ticker": "NVDA", "shares": 0.5, "close": 202.49, "purchase_price": 122.35,
             "purchase_date": "2025-11-12", "current_dollars": 101.2},
            {"ticker": "XOM", "shares": 0.4, "close": 114.3, "current_dollars": 45.7},
        ]}
        info = holdings_info(pf)
        assert info["NVDA"]["purchase_price"] == pytest.approx(122.35)  # explicit cost wins
        assert info["NVDA"]["basis_is_cost"] is True
        assert info["XOM"]["purchase_price"] == pytest.approx(114.3)    # snapshot fallback
        assert info["XOM"]["basis_is_cost"] is False


class TestSplitFactor:
    def test_splits_after_purchase_compound(self):
        from api.data_cache import split_factor
        events = [["2020-08-31", 4.0], ["2024-06-10", 10.0]]
        assert split_factor(events, "2019-01-01") == 40.0   # both splits after buy
        assert split_factor(events, "2022-01-01") == 10.0   # only the second
        assert split_factor(events, "2025-01-01") == 1.0    # none since buy
        assert split_factor([], "2020-01-01") == 1.0
        assert split_factor(None, "2020-01-01") == 1.0


class TestSessionPersistence:
    def test_roundtrip_via_disk(self, tmp_path, monkeypatch):
        import api.portfolio_core as pc

        monkeypatch.setattr(pc, "MEMORY_DIR", tmp_path)
        pf = {"holdings": [{"ticker": "AAPL", "shares": 1, "current_dollars": 150.0}]}
        save_session_portfolio("tester@x.com", pf)

        # Simulate a restart: wipe the in-memory map, disk copy must survive.
        SESSION_PORTFOLIOS.clear()
        loaded = get_session_portfolio("tester@x.com")
        assert loaded == pf

    def test_missing_session_returns_none(self, tmp_path, monkeypatch):
        import api.portfolio_core as pc

        monkeypatch.setattr(pc, "MEMORY_DIR", tmp_path)
        SESSION_PORTFOLIOS.clear()
        assert get_session_portfolio("nobody") is None
