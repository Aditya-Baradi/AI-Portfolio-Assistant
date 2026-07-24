"""
The market-data provider abstraction: selection, the production licence guard,
and the shape contract every provider must satisfy.
"""
from __future__ import annotations

import pandas as pd
import pytest

from api import market_data


@pytest.fixture(autouse=True)
def clean_provider(monkeypatch):
    for var in (
        "MARKET_DATA_PROVIDER",
        "TIINGO_API_KEY",
        "POLYGON_API_KEY",
        "EVERGREEN_ENV",
        market_data.REDISTRIBUTION_ACK_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    market_data.reset_provider()
    yield
    market_data.reset_provider()


class TestSelection:
    def test_defaults_to_yfinance(self):
        assert market_data.provider_name() == "yfinance"
        assert isinstance(market_data.get_provider(), market_data.YFinanceProvider)

    def test_explicit_selection(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "stooq")
        market_data.reset_provider()
        assert isinstance(market_data.get_provider(), market_data.StooqProvider)

    def test_unknown_provider_is_an_error(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "bloomberg-terminal")
        market_data.reset_provider()
        with pytest.raises(market_data.MarketDataError, match="Unknown"):
            market_data.get_provider()

    def test_licensed_provider_requires_its_key(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "tiingo")
        market_data.reset_provider()
        with pytest.raises(market_data.MarketDataError, match="TIINGO_API_KEY"):
            market_data.get_provider()

    def test_licensed_provider_constructs_with_a_key(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
        monkeypatch.setenv("POLYGON_API_KEY", "test-key")
        market_data.reset_provider()
        p = market_data.get_provider()
        assert isinstance(p, market_data.PolygonProvider)
        assert p.production_capable is True


class TestProductionGuard:
    """
    This project has no public-display/redistribution contract for the
    development adapters, so production must require a supported commercial
    adapter plus the operator's explicit entitlement attestation.
    """

    def test_yfinance_is_flagged_as_not_production_capable(self):
        problems = market_data.check_production_ready()
        assert problems and "not a production-capable" in problems[0]

    def test_stooq_is_also_flagged(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "stooq")
        market_data.reset_provider()
        assert market_data.check_production_ready()

    @pytest.mark.parametrize("name,key", [("tiingo", "TIINGO_API_KEY"),
                                          ("polygon", "POLYGON_API_KEY")])
    def test_supported_provider_with_entitlement_attestation_passes(
        self, monkeypatch, name, key
    ):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", name)
        monkeypatch.setenv(key, "test-key")
        monkeypatch.setenv(market_data.REDISTRIBUTION_ACK_ENV, "true")
        market_data.reset_provider()
        assert market_data.check_production_ready() == []

    def test_commercial_provider_still_requires_entitlement_ack(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "tiingo")
        monkeypatch.setenv("TIINGO_API_KEY", "test-key")
        market_data.reset_provider()
        problems = market_data.check_production_ready()
        assert any(market_data.REDISTRIBUTION_ACK_ENV in problem for problem in problems)

    def test_commercial_provider_missing_key_fails_readiness(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "polygon")
        monkeypatch.setenv(market_data.REDISTRIBUTION_ACK_ENV, "true")
        market_data.reset_provider()
        assert any("POLYGON_API_KEY" in p for p in market_data.check_production_ready())

    def test_backend_startup_audit_reports_the_problem(self, monkeypatch):
        from api import backend

        problems = backend._startup_audit()
        assert any("not a production-capable" in p for p in problems)


class TestShapeContract:
    """
    Everything downstream expects yf.download's layout: a MultiIndex whose
    level 0 contains "Close". A provider that breaks this silently starves
    every metric in the app.
    """

    def test_as_close_frame_layout(self):
        idx = pd.bdate_range("2024-01-01", periods=5)
        frame = market_data._as_close_frame({
            "AAPL": pd.Series(range(5), index=idx, dtype=float),
            "MSFT": pd.Series(range(5, 10), index=idx, dtype=float),
        })
        assert isinstance(frame.columns, pd.MultiIndex)
        assert "Close" in set(frame.columns.get_level_values(0))
        assert set(frame["Close"].columns) == {"AAPL", "MSFT"}

    def test_empty_input_gives_empty_frame(self):
        assert market_data._as_close_frame({}).empty

    def test_base_provider_degrades_gracefully(self):
        """Providers without a fundamentals entitlement must return neutrals."""
        base = market_data.PriceProvider()
        assert base.splits("AAPL") == []
        assert base.dividends_ttm("AAPL") == 0.0
        assert base.sector("AAPL") == "Unknown"
        assert base.news("AAPL") == []

    def test_fallback_is_stooq(self):
        assert isinstance(market_data.get_fallback(), market_data.StooqProvider)

    def test_production_has_no_unentitled_fallback(self, monkeypatch):
        monkeypatch.setenv("EVERGREEN_ENV", "production")
        assert market_data.get_fallback() is None

    def test_stooq_has_no_fallback_of_its_own(self, monkeypatch):
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "stooq")
        market_data.reset_provider()
        assert market_data.get_fallback() is None

    def test_total_return_series_accounts_for_dividend(self):
        idx = pd.to_datetime(["2024-01-02", "2024-01-03"])
        close = pd.Series([100.0, 98.0], index=idx)
        adjusted = market_data._total_return_close(
            close, {pd.Timestamp("2024-01-03"): 2.0}
        )
        assert adjusted.iloc[0] == pytest.approx(98.0)
        assert adjusted.iloc[1] == pytest.approx(98.0)


class TestCacheKeying:
    def test_provider_is_part_of_the_cache_key(self, monkeypatch):
        """
        One provider's prices must never be served as another's — otherwise
        switching providers silently reuses stale, differently-adjusted data.
        """
        from api import data_cache

        key_a = data_cache._price_key(["AAPL"], "2024-01-01", "2024-02-01", {})
        monkeypatch.setenv("MARKET_DATA_PROVIDER", "stooq")
        market_data.reset_provider()
        key_b = data_cache._price_key(["AAPL"], "2024-01-01", "2024-02-01", {})
        assert key_a != key_b

    def test_numeric_cache_roundtrip_without_pickle(self, tmp_path):
        from api import data_cache

        idx = pd.bdate_range("2024-01-01", periods=2)
        frame = market_data._as_close_frame(
            {"AAPL": pd.Series([100.0, 101.0], index=idx)},
            source_provider="tiingo",
            price_semantics=market_data.TOTAL_RETURN_SEMANTICS,
        )
        path = tmp_path / "prices.npz"
        data_cache._write_price_cache(path, frame)
        loaded = data_cache._read_price_cache(path)
        pd.testing.assert_frame_equal(loaded, frame, check_freq=False)
        assert loaded.attrs == frame.attrs

    def test_corrupt_numeric_cache_is_rejected(self, tmp_path):
        from api import data_cache

        path = tmp_path / "bad.npz"
        path.write_bytes(b"not-a-numpy-cache")
        with pytest.raises(Exception):
            data_cache._read_price_cache(path)
