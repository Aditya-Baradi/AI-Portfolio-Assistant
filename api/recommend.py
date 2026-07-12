"""
Profile-driven stock recommendations and retirement projections.

The user's saved profile (years to retirement, risk tolerance 1-10, max
volatility, goal, monthly contribution) drives two things:

1. recommend_stocks(): screens a fixed universe of large, liquid US names.
   Anything more volatile than the user's stated maximum is excluded, then
   candidates are scored by how well their volatility matches the target
   implied by the risk score and by the goal (growth favours realized
   returns, income favours defensive sectors + steadiness, balanced favours
   risk-adjusted return). Scoring is deliberately transparent - every pick
   comes with a plain-English "why".

2. retirement_paths(): Monte Carlo of portfolio value until retirement with
   monthly contributions, reporting the median and the 16th/84th percentile
   band (roughly +/-1 sigma).
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

try:
    from .portfolio_core import _download_adj_close_matrix, _perf_metrics, RF_ANNUAL_DEFAULT
except ImportError:  # running outside the package
    from api.portfolio_core import _download_adj_close_matrix, _perf_metrics, RF_ANNUAL_DEFAULT


# Large, liquid, well-known US names across sectors. Static sector labels
# avoid ~50 extra network lookups; these don't change in practice.
UNIVERSE = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology",
    "ORCL": "Technology", "AMD": "Technology", "CSCO": "Technology",
    # Communication
    "GOOGL": "Communication", "META": "Communication", "DIS": "Communication",
    "NFLX": "Communication", "VZ": "Communication", "T": "Communication",
    # Consumer
    "AMZN": "Consumer", "TSLA": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "COST": "Consumer", "WMT": "Consumer",
    # Staples
    "PG": "Staples", "KO": "Staples", "PEP": "Staples", "CL": "Staples",
    # Health care
    "JNJ": "Health care", "UNH": "Health care", "LLY": "Health care",
    "PFE": "Health care", "ABBV": "Health care", "MRK": "Health care",
    # Financials
    "JPM": "Financials", "BAC": "Financials", "V": "Financials",
    "MA": "Financials", "GS": "Financials", "AXP": "Financials",
    # Energy / industrials
    "XOM": "Energy", "CVX": "Energy", "CAT": "Industrials",
    "HON": "Industrials", "UPS": "Industrials", "GE": "Industrials",
    # Utilities / real estate
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AMT": "Real estate",
}

# Sectors that historically hold up better in downturns; the income goal
# leans toward these.
DEFENSIVE_SECTORS = {"Staples", "Utilities", "Health care", "Real estate"}


def target_volatility(risk_tolerance: int) -> float:
    """Map risk 1-10 to a target annualized volatility (12.5% .. 35%)."""
    r = max(1, min(int(risk_tolerance), 10))
    return 0.10 + 0.025 * r


def score_candidates(metrics: dict, profile: dict, owned=None, top_n: int = 12) -> list[dict]:
    """
    Rank candidate stocks for a profile.

    `metrics` maps ticker -> {"CAGR": float, "volatility": float, "Sharpe": float}.
    Pure function (no network) so it can be tested offline.
    """
    owned = {t.upper() for t in (owned or [])}
    max_vol = float(profile.get("max_volatility_pct", 30)) / 100.0
    goal = profile.get("goal", "balanced")
    tgt = target_volatility(profile.get("risk_tolerance", 5))

    candidates = []
    for t, m in metrics.items():
        vol = m["volatility"]
        if vol <= 0 or vol > max_vol:
            continue  # louder than the user said they can stomach

        # 0..1: how close this stock's volatility sits to the risk target.
        vol_fit = max(0.0, 1.0 - abs(vol - tgt) / tgt)

        sector = UNIVERSE.get(t, "Unknown")
        if goal == "growth":
            perf = m["CAGR"]
        elif goal == "income":
            # real dividend yield when we have it, defensive-sector proxy always
            perf = (m["Sharpe"] * 0.5
                    + 4.0 * m.get("div_yield", 0.0)
                    + (0.4 if sector in DEFENSIVE_SECTORS else 0.0))
        else:  # balanced
            perf = m["Sharpe"]

        candidates.append({
            "ticker": t,
            "sector": sector,
            "volatility": round(vol, 4),
            "est_annual_return": round(m["CAGR"], 4),
            "sharpe": round(m["Sharpe"], 2),
            "div_yield": round(m.get("div_yield", 0.0), 4),
            "owned": t in owned,
            "_perf": perf,
            "_vol_fit": vol_fit,
        })

    if not candidates:
        return []

    # Rank-normalize perf so goals with different units mix fairly with fit.
    candidates.sort(key=lambda c: c["_perf"])
    n = len(candidates)
    for i, c in enumerate(candidates):
        perf_rank = i / (n - 1) if n > 1 else 1.0
        c["score"] = round(0.6 * perf_rank + 0.4 * c["_vol_fit"], 4)

    candidates.sort(key=lambda c: -c["score"])
    out = []
    for c in candidates[:top_n]:
        c.pop("_perf")
        c.pop("_vol_fit")
        c["why"] = _why(c, tgt, goal)
        out.append(c)
    return out


def _why(c: dict, tgt: float, goal: str) -> str:
    vol_pct = c["volatility"] * 100
    ret_pct = c["est_annual_return"] * 100
    steadiness = (
        "much steadier than" if c["volatility"] < tgt * 0.7
        else "a close match for" if abs(c["volatility"] - tgt) / tgt < 0.3
        else "near the top of"
    )
    bits = [f"Swings about {vol_pct:.0f}% a year, {steadiness} your comfort level"]
    if ret_pct >= 0:
        bits.append(f"returned roughly {ret_pct:.0f}%/yr over the last 3 years")
    else:
        bits.append(f"lost roughly {abs(ret_pct):.0f}%/yr over the last 3 years")
    if goal == "income" and c.get("div_yield", 0) >= 0.02:
        bits.append(f"pays about {c['div_yield'] * 100:.1f}% a year in dividends")
    elif goal == "income" and c["sector"] in DEFENSIVE_SECTORS:
        bits.append(f"{c['sector'].lower()} stocks tend to hold up in downturns")
    return "; ".join(bits) + "."


def universe_metrics(lookback_years: int = 3) -> dict:
    """3-year realized CAGR/volatility/Sharpe (+ dividend yield) for the universe."""
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")
    prices = _download_adj_close_matrix(list(UNIVERSE.keys()), start, end)

    try:
        from .data_cache import cached_dividends_ttm
    except ImportError:
        from api.data_cache import cached_dividends_ttm
    try:
        divs = cached_dividends_ttm(list(prices.columns))
    except Exception:
        divs = {}

    metrics = {}
    for t in prices.columns:
        col = prices[t].dropna()
        if len(col) < 200:
            continue
        try:
            m = _perf_metrics(col.pct_change().dropna(), RF_ANNUAL_DEFAULT)
        except Exception:
            continue
        price = float(col.iloc[-1])
        metrics[t] = {
            "CAGR": m["CAGR"], "volatility": m["volatility"], "Sharpe": m["Sharpe"],
            "div_yield": (divs.get(t, 0.0) / price) if price > 0 else 0.0,
        }
    return metrics


def recommend_stocks(profile: dict, owned=None, top_n: int = 12) -> list[dict]:
    return score_candidates(universe_metrics(), profile, owned=owned, top_n=top_n)


def retirement_paths(mu: float, sigma: float, value0: float, monthly: float,
                     years: int, sims: int = 1000, seed: int = 7,
                     goal: float | None = None) -> dict:
    """
    Monte Carlo the road to retirement.

    Each simulated year draws a return from Normal(mu, sigma); contributions
    are added through the year (approximated as half-year growth on the
    year's deposits). Returns the median path, the 16th/84th percentiles,
    and (when `goal` is set) the fraction of simulations that end above it.
    """
    years = max(1, min(int(years), 60))
    rng = np.random.default_rng(seed)
    annual_add = float(monthly) * 12.0

    values = np.full(sims, float(value0))
    med, hi, lo = [round(float(value0), 2)], [round(float(value0), 2)], [round(float(value0), 2)]
    for _ in range(years):
        r = rng.normal(mu, sigma, sims)
        r = np.maximum(r, -0.95)  # a year can't lose more than everything
        values = values * (1.0 + r) + annual_add * (1.0 + r / 2.0)
        med.append(round(float(np.percentile(values, 50)), 2))
        hi.append(round(float(np.percentile(values, 84)), 2))
        lo.append(round(float(np.percentile(values, 16)), 2))

    total_in = float(value0) + annual_add * years
    out = {
        "years": list(range(0, years + 1)),
        "median": med,
        "optimistic": hi,
        "pessimistic": lo,
        "total_contributed": round(total_in, 2),
        "assumptions": {"mu": round(mu, 4), "sigma": round(sigma, 4),
                        "monthly": float(monthly), "sims": sims},
    }
    if goal and goal > 0:
        out["goal"] = float(goal)
        out["p_goal"] = round(float((values >= goal).mean()), 3)
    return out


def monthly_needed_for(goal: float, mu: float, sigma: float, value0: float,
                       years: int, target_p: float = 0.6, cap: float = 50_000.0) -> float | None:
    """
    Smallest monthly contribution that reaches `goal` in at least `target_p`
    of simulations. Bisection over the (monotonic) contribution axis; None
    when even `cap` per month isn't enough.
    """
    def p_at(monthly):
        return retirement_paths(mu, sigma, value0, monthly, years, goal=goal)["p_goal"]

    if p_at(0) >= target_p:
        return 0.0
    if p_at(cap) < target_p:
        return None
    lo_m, hi_m = 0.0, cap
    for _ in range(12):
        mid = (lo_m + hi_m) / 2
        if p_at(mid) >= target_p:
            hi_m = mid
        else:
            lo_m = mid
    return round(hi_m, 0)


def rebalance_trades(current_weights: dict, target_weights: dict,
                     total_value: float, min_trade: float = 25.0) -> list[dict]:
    """
    Dollar trades to move a portfolio from its current mix to a target mix.
    Pure function: weights in, sorted sell-first trade list out. Tiny trades
    below `min_trade` are dropped (not worth the costs).
    """
    cur_total = sum(v for v in current_weights.values() if v > 0)
    if cur_total <= 0 or total_value <= 0:
        return []
    cur = {t.upper(): v / cur_total for t, v in current_weights.items() if v > 0}
    tgt_total = sum(v for v in target_weights.values() if v > 0)
    tgt = {t.upper(): v / tgt_total for t, v in target_weights.items() if v > 0} if tgt_total > 0 else {}

    trades = []
    for t in sorted(set(cur) | set(tgt)):
        delta = (tgt.get(t, 0.0) - cur.get(t, 0.0)) * total_value
        if abs(delta) < min_trade:
            continue
        trades.append({
            "ticker": t,
            "action": "buy" if delta > 0 else "sell",
            "dollars": round(abs(delta), 2),
        })
    trades.sort(key=lambda x: (x["action"] != "sell", -x["dollars"]))  # sells first, biggest first
    return trades
