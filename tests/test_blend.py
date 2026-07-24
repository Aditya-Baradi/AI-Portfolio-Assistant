"""Tests for allocation blending and sector aggregation (pure functions, offline)."""
import pytest

from api.predict_agent import _blend_allocations, _validated_allocation
from api.portfolio_core import sector_weights_from_weights, SECTOR_ETF_MAP


class TestBlendAllocations:
    def test_blend_sums_to_one(self):
        rl = {"A": 0.5, "B": 0.5}
        ms = {"A": 0.9, "B": 0.1}
        out = _blend_allocations(rl, ms, 0.5)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-6)

    def test_blend_midpoint(self):
        rl = {"A": 1.0, "B": 0.0}
        ms = {"A": 0.0, "B": 1.0}
        out = _blend_allocations(rl, ms, 0.5)
        assert out["A"] == pytest.approx(0.5)
        assert out["B"] == pytest.approx(0.5)

    def test_extremes_recover_inputs(self):
        rl = {"A": 0.7, "B": 0.3}
        ms = {"A": 0.2, "B": 0.8}
        assert _blend_allocations(rl, ms, 1.0)["A"] == pytest.approx(0.7, abs=1e-6)
        assert _blend_allocations(rl, ms, 0.0)["A"] == pytest.approx(0.2, abs=1e-6)

    def test_missing_keys_treated_as_zero(self):
        rl = {"A": 1.0}
        ms = {"B": 1.0}
        out = _blend_allocations(rl, ms, 0.5)
        assert out["A"] == pytest.approx(0.5)
        assert out["B"] == pytest.approx(0.5)

    def test_all_weights_non_negative(self):
        out = _blend_allocations({"A": 0.6, "B": 0.4}, {"A": 0.1, "B": 0.9}, 0.3)
        assert all(v >= 0 for v in out.values())

    def test_post_validation_rejects_cap_violation(self):
        with pytest.raises(ValueError, match="max_weight"):
            _validated_allocation(
                {"A": 0.9, "B": 0.1}, ["A", "B"], max_weight=0.6
            )

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1])
    def test_post_validation_rejects_invalid_weight(self, bad):
        with pytest.raises(ValueError):
            _validated_allocation({"A": bad, "B": 1.0}, ["A", "B"])


class TestSectorAggregation:
    def test_sector_weights_sum_to_one(self):
        weights = {"AAPL": 0.5, "MSFT": 0.3, "XOM": 0.2}
        smap = {"AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy"}
        sec = sector_weights_from_weights(weights, smap)
        assert sum(sec.values()) == pytest.approx(1.0)
        assert sec["Technology"] == pytest.approx(0.8)
        assert sec["Energy"] == pytest.approx(0.2)

    def test_unknown_sector_bucket(self):
        sec = sector_weights_from_weights({"ZZZ": 1.0}, {})
        assert sec == {"Unknown": 1.0}

    def test_sector_etf_map_covers_both_naming_conventions(self):
        # yfinance uses e.g. 'Technology'; GICS uses 'Information Technology'.
        for pair in [("Technology", "Information Technology"),
                     ("Financial Services", "Financials"),
                     ("Consumer Cyclical", "Consumer Discretionary")]:
            a, b = pair
            assert SECTOR_ETF_MAP.get(a) == SECTOR_ETF_MAP.get(b) is not None
