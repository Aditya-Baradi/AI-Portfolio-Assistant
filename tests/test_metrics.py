"""Offline tests for the backtest engine, using synthetic prices with known answers."""
import numpy as np
import pandas as pd
import pytest

from api.portfolio_core import _perf_metrics, backtest_vs_benchmark, compute_portfolio_metrics
from tests.conftest import make_prices


class TestPerfMetrics:
    def test_constant_return_cagr(self):
        r = 0.001
        rets = pd.Series([r] * 252)
        m = _perf_metrics(rets, rf_annual=0.0)
        assert m["CAGR"] == pytest.approx((1 + r) ** 252 - 1, rel=1e-9)
        assert m["volatility"] == pytest.approx(0.0, abs=1e-12)
        assert m["Sharpe"] == 0.0  # zero vol -> defined as 0
        assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-12)
        assert m["n_days"] == 252

    def test_known_drawdown(self):
        # +100% then -50%: peak 2.0, trough back to 1.0 -> max drawdown -50%
        rets = pd.Series([1.0, -0.5])
        m = _perf_metrics(rets)
        assert m["max_drawdown"] == pytest.approx(-0.5)
        assert m["total_return"] == pytest.approx(0.0, abs=1e-12)

    def test_rf_reduces_sharpe(self):
        rets = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 252))
        s0 = _perf_metrics(rets, rf_annual=0.0)["Sharpe"]
        s4 = _perf_metrics(rets, rf_annual=0.04)["Sharpe"]
        assert s4 < s0

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            _perf_metrics(pd.Series(dtype=float))


class TestBacktestVsBenchmark:
    def test_identical_portfolio_matches_benchmark(self, patch_prices, rng):
        rets = rng.normal(0.0005, 0.01, 252)
        prices = make_prices({"AAA": rets, "SPY": rets})
        patch_prices(prices)

        bt = backtest_vs_benchmark({"AAA": 1.0}, "2024-01-01", "2025-01-01", cost_bps=0.0)
        c = bt["comparison"]
        assert c["excess_total_return"] == pytest.approx(0.0, abs=1e-6)
        assert c["beta"] == pytest.approx(1.0, abs=1e-6)
        assert c["alpha_annual"] == pytest.approx(0.0, abs=1e-6)
        assert not c["outperformed"]

    def test_identical_with_costs_still_equal(self, patch_prices, rng):
        # Both sides pay the initial buy; a single-asset portfolio never drifts,
        # so no rebalance turnover accrues and the comparison stays a wash.
        rets = rng.normal(0.0005, 0.01, 252)
        prices = make_prices({"AAA": rets, "SPY": rets})
        patch_prices(prices)

        bt = backtest_vs_benchmark({"AAA": 1.0}, "2024-01-01", "2025-01-01", cost_bps=10.0)
        assert bt["comparison"]["excess_total_return"] == pytest.approx(0.0, abs=1e-6)

    def test_double_beta(self, patch_prices, rng):
        bench = rng.normal(0.0004, 0.01, 252)
        prices = make_prices({"LEV": 2 * bench, "SPY": bench})
        patch_prices(prices)

        bt = backtest_vs_benchmark({"LEV": 1.0}, "2024-01-01", "2025-01-01", cost_bps=0.0)
        assert bt["comparison"]["beta"] == pytest.approx(2.0, abs=0.01)

    def test_costs_reduce_portfolio_return(self, patch_prices, rng):
        a, b = rng.normal(0.001, 0.02, 252), rng.normal(0.0, 0.02, 252)
        prices = make_prices({"AAA": a, "BBB": b, "SPY": rng.normal(0.0005, 0.01, 252)})
        patch_prices(prices)

        w = {"AAA": 0.5, "BBB": 0.5}
        gross = backtest_vs_benchmark(w, "2024-01-01", "2025-01-01", cost_bps=0.0)
        net = backtest_vs_benchmark(w, "2024-01-01", "2025-01-01", cost_bps=25.0)
        assert net["portfolio"]["total_return"] < gross["portfolio"]["total_return"]

    def test_weights_renormalized_over_available(self, patch_prices, rng):
        prices = make_prices({"AAA": 0.001, "SPY": 0.0005})
        patch_prices(prices)

        # 'GONE' has no price column -> dropped, AAA renormalized to 1.0
        bt = backtest_vs_benchmark({"AAA": 0.5, "GONE": 0.5}, "2024-01-01", "2025-01-01")
        assert bt["tickers_dropped"] == ["GONE"]
        assert bt["weights_used"]["AAA"] == pytest.approx(1.0)

    def test_curves_returned_and_consistent(self, patch_prices, rng):
        rets = rng.normal(0.0005, 0.01, 100)
        prices = make_prices({"AAA": rets, "SPY": rets}, n_days=100)
        patch_prices(prices)

        bt = backtest_vs_benchmark({"AAA": 1.0}, "2024-01-01", "2025-01-01",
                                   cost_bps=0.0, return_curves=True)
        curves = bt["curves"]
        assert len(curves["dates"]) == len(curves["portfolio"]) == len(curves["benchmark"])
        # final curve value equals 1 + total_return
        assert curves["portfolio"][-1] == pytest.approx(1 + bt["portfolio"]["total_return"], rel=1e-4)

    def test_empty_weights_raise(self):
        with pytest.raises(ValueError):
            backtest_vs_benchmark({}, "2024-01-01", "2025-01-01")


class TestComputePortfolioMetrics:
    def test_equal_weight_default_and_benchmark_block(self, patch_prices, rng):
        prices = make_prices({
            "AAA": rng.normal(0.001, 0.01, 252),
            "BBB": rng.normal(0.0005, 0.01, 252),
            "SPY": rng.normal(0.0004, 0.008, 252),
        })
        patch_prices(prices)

        out = compute_portfolio_metrics(["AAA", "BBB"], "2024-01-01", "2025-01-01")
        assert sorted(out["tickers_used"]) == ["AAA", "BBB"]
        assert out["benchmark"] == "SPY"
        assert "vs_benchmark" in out and "benchmark_metrics" in out
        assert isinstance(out["vs_benchmark"]["outperformed"], bool)
        assert -1.0 <= out["max_drawdown"] <= 0.0

    def test_long_only_vol_bounded_by_max_asset_vol(self, patch_prices, rng):
        a = rng.normal(0.0, 0.02, 252)
        b = rng.normal(0.0, 0.01, 252)
        prices = make_prices({"AAA": a, "BBB": b, "SPY": rng.normal(0.0, 0.01, 252)})
        patch_prices(prices)

        import json
        out = compute_portfolio_metrics(
            ["AAA", "BBB"], "2024-01-01", "2025-01-01",
            weights_json=json.dumps({"AAA": 0.5, "BBB": 0.5}),
        )
        max_single = max(np.std(a), np.std(b)) * np.sqrt(252)
        assert out["volatility"] <= max_single + 1e-9
