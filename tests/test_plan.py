"""Tests for the plan feature: scoring, retirement Monte Carlo, profile storage (offline)."""
import pytest

from api.recommend import (
    DEFENSIVE_SECTORS,
    UNIVERSE,
    retirement_paths,
    score_candidates,
    target_volatility,
)


METRICS = {
    # low-vol steady names
    "KO":   {"CAGR": 0.08, "volatility": 0.13, "Sharpe": 0.65},
    "JNJ":  {"CAGR": 0.07, "volatility": 0.14, "Sharpe": 0.55},
    "DUK":  {"CAGR": 0.06, "volatility": 0.15, "Sharpe": 0.45},
    # mid-vol
    "MSFT": {"CAGR": 0.18, "volatility": 0.24, "Sharpe": 0.85},
    "JPM":  {"CAGR": 0.15, "volatility": 0.25, "Sharpe": 0.70},
    # high-vol high-return
    "NVDA": {"CAGR": 0.55, "volatility": 0.48, "Sharpe": 1.10},
    "TSLA": {"CAGR": 0.20, "volatility": 0.60, "Sharpe": 0.40},
    # loser
    "NKE":  {"CAGR": -0.12, "volatility": 0.30, "Sharpe": -0.40},
}


def profile(risk=5, max_vol=30, goal="balanced", years=30, monthly=0):
    return {
        "years_to_retirement": years,
        "risk_tolerance": risk,
        "max_volatility_pct": max_vol,
        "goal": goal,
        "monthly_contribution": monthly,
    }


class TestTargetVolatility:
    def test_scale(self):
        assert target_volatility(1) == pytest.approx(0.125)
        assert target_volatility(10) == pytest.approx(0.35)
        assert target_volatility(5) == pytest.approx(0.225)

    def test_clamped(self):
        assert target_volatility(0) == target_volatility(1)
        assert target_volatility(99) == target_volatility(10)


class TestScoring:
    def test_max_volatility_is_a_hard_filter(self):
        recs = score_candidates(METRICS, profile(max_vol=20))
        tickers = {r["ticker"] for r in recs}
        assert "NVDA" not in tickers and "TSLA" not in tickers  # 48%/60% > 20%
        assert "KO" in tickers

    def test_cautious_profile_prefers_steady_names(self):
        recs = score_candidates(METRICS, profile(risk=1, max_vol=60))
        top3 = {r["ticker"] for r in recs[:3]}
        assert top3 & {"KO", "JNJ", "DUK"}   # low-vol names near the top
        assert recs[0]["ticker"] != "TSLA"

    def test_aggressive_growth_profile_ranks_nvda_first(self):
        recs = score_candidates(METRICS, profile(risk=10, max_vol=60, goal="growth"))
        assert recs[0]["ticker"] == "NVDA"   # highest CAGR near the 35% target

    def test_income_goal_boosts_defensive_sectors(self):
        assert UNIVERSE["KO"] in DEFENSIVE_SECTORS
        bal = score_candidates(METRICS, profile(risk=3, max_vol=60, goal="balanced"))
        inc = score_candidates(METRICS, profile(risk=3, max_vol=60, goal="income"))
        rank = lambda recs, t: [r["ticker"] for r in recs].index(t)
        assert rank(inc, "KO") <= rank(bal, "KO")

    def test_owned_flag(self):
        recs = score_candidates(METRICS, profile(max_vol=60), owned={"msft"})
        by = {r["ticker"]: r for r in recs}
        assert by["MSFT"]["owned"] is True
        assert by["KO"]["owned"] is False

    def test_every_pick_has_a_why(self):
        for r in score_candidates(METRICS, profile(max_vol=60)):
            assert r["why"].endswith(".")
            assert "%" in r["why"]

    def test_impossible_limits_return_empty(self):
        assert score_candidates(METRICS, profile(max_vol=5)) == []


class TestRetirementPaths:
    def test_shape_and_monotonic_band(self):
        out = retirement_paths(0.07, 0.15, 10_000, 500, years=20)
        assert len(out["years"]) == 21
        assert len(out["median"]) == 21
        last = -1
        assert out["pessimistic"][last] <= out["median"][last] <= out["optimistic"][last]

    def test_contributions_counted(self):
        out = retirement_paths(0.07, 0.15, 1_000, 100, years=10)
        assert out["total_contributed"] == pytest.approx(1_000 + 100 * 12 * 10)

    def test_zero_growth_zero_vol_is_deterministic(self):
        out = retirement_paths(0.0, 0.0, 1_000, 100, years=5)
        assert out["median"][-1] == pytest.approx(1_000 + 100 * 12 * 5)
        assert out["median"][-1] == out["optimistic"][-1] == out["pessimistic"][-1]

    def test_seeded_and_reproducible(self):
        a = retirement_paths(0.07, 0.15, 10_000, 0, years=15)
        b = retirement_paths(0.07, 0.15, 10_000, 0, years=15)
        assert a["median"] == b["median"]

    def test_years_clamped(self):
        assert len(retirement_paths(0.05, 0.1, 100, 0, years=999)["years"]) == 61


class TestProfileStorage:
    def test_roundtrip(self, tmp_path, monkeypatch):
        import api.db as db

        monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
        db.init_db()
        uid = db.register_user("plan@x.com", "longpassword1")
        assert db.get_profile(uid) is None
        db.save_profile(uid, profile(risk=7, max_vol=35, goal="growth"))
        p = db.get_profile(uid)
        assert p["risk_tolerance"] == 7
        db.save_profile(uid, profile(risk=2))
        assert db.get_profile(uid)["risk_tolerance"] == 2  # upsert replaces
