"""
Walk-forward backtest: the agent's recommended STOCKS and SECTORS vs the S&P 500.

Flow
----
1. Load the user's portfolio (portfolio.json) -> tickers + current $ weights.
2. Optimize on a TRAIN window using the same max-Sharpe engine the agent falls
   back to (FinRL -> PyPortfolioOpt), producing the "agent" allocation.
3. Backtest that allocation OUT-OF-SAMPLE (TEST window) against SPY:
     - stock level  : the recommended ticker weights
     - sector level : the recommended sector tilt, proxied by SPDR sector ETFs
4. For reference, also backtest the current holdings and an equal-weight basket.

Run:  venv/Scripts/python.exe -m api.backtest_vs_sp500
"""
import json
import warnings

import numpy as np
import pandas as pd

from api.predict_agent import (
    load_tickers_from_portfolio_json,
    run_pypfopt_max_sharpe,
    run_conservative_optimization,
)
from api.portfolio_core import (
    get_sector_map,
    backtest_vs_benchmark,
    backtest_sectors_vs_benchmark,
    sector_weights_from_weights,
    _download_adj_close_matrix,
    _perf_metrics,
    RF_ANNUAL_DEFAULT,
    COST_BPS_DEFAULT,
)

warnings.filterwarnings("ignore")

PORTFOLIO_JSON = "portfolio.json"
BENCHMARK = "SPY"

# Optimize on the train window, then evaluate on the (out-of-sample) test window.
TRAIN_START, TRAIN_END = "2020-01-01", "2025-01-01"
TEST_START, TEST_END = "2025-01-01", "2026-07-09"


def pct(x):
    return f"{x * 100:+.2f}%"


def show_backtest(title, bt):
    p, b, c = bt["portfolio"], bt["benchmark_metrics"], bt["comparison"]
    verdict = "OUTPERFORMED" if c["outperformed"] else "underperformed"
    print(f"\n=== {title} ===")
    print(f"  window: {bt['start']} -> {bt['end']}   ({p['n_days']} trading days)")
    print(f"  {'metric':<16}{'portfolio':>14}{'S&P 500 (SPY)':>16}")
    print(f"  {'total return':<16}{pct(p['total_return']):>14}{pct(b['total_return']):>16}")
    print(f"  {'CAGR':<16}{pct(p['CAGR']):>14}{pct(b['CAGR']):>16}")
    print(f"  {'volatility':<16}{pct(p['volatility']):>14}{pct(b['volatility']):>16}")
    print(f"  {'Sharpe':<16}{p['Sharpe']:>14.2f}{b['Sharpe']:>16.2f}")
    print(f"  {'max drawdown':<16}{pct(p['max_drawdown']):>14}{pct(b['max_drawdown']):>16}")
    print(f"  --> vs S&P 500: {verdict} by {pct(c['excess_total_return'])} total return")
    print(f"      excess CAGR {pct(c['excess_CAGR'])} | alpha {pct(c['alpha_annual'])} | "
          f"beta {c['beta']:.2f} | info-ratio {c['information_ratio']:.2f}")


def run_walk_forward(tickers, wf_start, wf_end, benchmark=BENCHMARK,
                     train_years=3, cost_bps=COST_BPS_DEFAULT,
                     rebalance_freq="6MS", optimizer=None):
    """
    Rolling walk-forward backtest: at each rebalance date, re-optimize on the
    trailing `train_years`, then hold those weights until the next rebalance.
    Every period is strictly out-of-sample. Rebalance costs are charged on
    the turnover between consecutive allocations.

    Default strategy (validated): conservative optimization (min-vol + weight
    cap + 1/N shrinkage), semi-annual rebalancing — replaces the original
    quarterly max-Sharpe, which underperformed SPY out-of-sample.

    Returns (stitched_portfolio_returns, benchmark_returns, periods_info).
    """
    if optimizer is None:
        optimizer = run_conservative_optimization
    q_starts = pd.date_range(wf_start, wf_end, freq=rebalance_freq)
    if len(q_starts) < 2:
        raise ValueError("Walk-forward window too short for the rebalance frequency.")

    all_port, all_bench = [], []
    prev_w = None
    quarters = []

    for i, qs in enumerate(q_starts):
        q_end = q_starts[i + 1] if i + 1 < len(q_starts) else pd.to_datetime(wf_end)
        if q_end <= qs:
            continue
        train_start = (qs - pd.DateOffset(years=train_years)).strftime("%Y-%m-%d")
        train_end = qs.strftime("%Y-%m-%d")

        try:
            alloc, _, _ = optimizer(tickers, train_start, train_end)
        except Exception as e:
            warnings.warn(f"Optimization failed for {train_end} ({e}); skipping period.")
            continue
        alloc = {t: w for t, w in alloc.items() if w > 0}

        prices = _download_adj_close_matrix(
            list(alloc.keys()) + [benchmark], train_end, q_end.strftime("%Y-%m-%d")
        )
        rets = prices.pct_change().dropna()
        usable = [t for t in alloc if t in rets.columns]
        if not usable or benchmark not in rets.columns:
            continue
        w = np.array([alloc[t] for t in usable])
        w = w / w.sum()
        port_q = rets[usable].dot(w)
        bench_q = rets[benchmark]

        # Rebalance cost at the quarter boundary (full buy on the first one).
        w_map = {t: x for t, x in zip(usable, w)}
        if len(port_q) > 0 and cost_bps > 0:
            turnover = (
                sum(abs(w_map.get(t, 0.0) - prev_w.get(t, 0.0)) for t in set(w_map) | set(prev_w))
                if prev_w is not None else 1.0
            )
            port_q.iloc[0] -= turnover * cost_bps / 10_000.0
            if prev_w is None:
                bench_q = bench_q.copy()
                bench_q.iloc[0] -= cost_bps / 10_000.0
        prev_w = w_map

        all_port.append(port_q)
        all_bench.append(bench_q)
        top = sorted(w_map.items(), key=lambda kv: -kv[1])[:3]
        quarters.append((train_end, ", ".join(f"{t} {x*100:.0f}%" for t, x in top)))

    if not all_port:
        raise ValueError("Walk-forward produced no usable quarters.")
    return pd.concat(all_port), pd.concat(all_bench), quarters


def show_walk_forward(port_rets, bench_rets, quarters, title="WALK-FORWARD", eq_rets=None):
    p = _perf_metrics(port_rets, RF_ANNUAL_DEFAULT)
    b = _perf_metrics(bench_rets, RF_ANNUAL_DEFAULT)
    e = _perf_metrics(eq_rets, RF_ANNUAL_DEFAULT) if eq_rets is not None else None
    verdict = "OUTPERFORMED" if p["total_return"] > b["total_return"] else "underperformed"
    print(f"\n=== {title} (all out-of-sample) ===")
    print(f"  {len(quarters)} rebalances, {p['n_days']} trading days, costs {COST_BPS_DEFAULT} bps, rf {RF_ANNUAL_DEFAULT:.0%}")
    hdr = f"  {'metric':<16}{'strategy':>14}{'S&P 500 (SPY)':>16}"
    if e:
        hdr += f"{'equal-weight':>16}"
    print(hdr)
    for key, fmt in (("total_return", pct), ("CAGR", pct), ("Sharpe", None), ("max_drawdown", pct)):
        label = key.replace("_", " ")
        row = f"  {label:<16}"
        for m in ([p, b, e] if e else [p, b]):
            row += (f"{fmt(m[key]):>14}" if fmt else f"{m[key]:>14.2f}") if m is p else \
                   (f"{fmt(m[key]):>16}" if fmt else f"{m[key]:>16.2f}")
        print(row)
    print(f"  --> vs S&P 500: {verdict} by {pct(p['total_return'] - b['total_return'])} total return")
    if e:
        vs_eq = "ahead of" if p["total_return"] > e["total_return"] else "behind"
        print(f"  --> vs equal-weight: {vs_eq} 1/N by {pct(p['total_return'] - e['total_return'])}")
    print("  re-optimizations (top holdings):")
    for d, tops in quarters:
        print(f"    {d}: {tops}")


def main():
    tickers, mv_map = load_tickers_from_portfolio_json(PORTFOLIO_JSON, include_etfs=False)
    print(f"Loaded {len(tickers)} tickers from {PORTFOLIO_JSON}:")
    print("  " + ", ".join(tickers))

    # --- 1. Agent optimizes on the TRAIN window -----------------------------
    print(f"\nOptimizing (max-Sharpe) on train window {TRAIN_START}..{TRAIN_END} ...")
    alloc, _, method = run_pypfopt_max_sharpe(tickers, TRAIN_START, TRAIN_END)
    alloc = {t: w for t, w in alloc.items() if w > 0}
    print(f"  method: {method}")
    print("  recommended allocation (top holdings):")
    for t, w in sorted(alloc.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {t:<6} {pct(w)}")

    sector_map = get_sector_map(list(alloc.keys()))
    sec_w = sector_weights_from_weights(alloc, sector_map)
    print("  recommended sector tilt:")
    for s, w in sorted(sec_w.items(), key=lambda kv: -kv[1]):
        print(f"    {s:<26} {pct(w)}")

    # --- 2. OUT-OF-SAMPLE backtests vs S&P 500 ------------------------------
    print(f"\n{'#' * 60}\n# OUT-OF-SAMPLE TEST: {TEST_START} -> {TEST_END}\n{'#' * 60}")

    show_backtest("AGENT STOCKS  vs  S&P 500",
                  backtest_vs_benchmark(alloc, TEST_START, TEST_END, benchmark=BENCHMARK))

    show_backtest("AGENT SECTOR TILT (SPDR ETFs)  vs  S&P 500",
                  backtest_sectors_vs_benchmark(alloc, TEST_START, TEST_END,
                                                benchmark=BENCHMARK, sector_map=sector_map))

    # Current holdings (dollar-weighted) for reference.
    cur_w = {t: v for t, v in (mv_map or {}).items() if v and v > 0}
    if cur_w:
        show_backtest("CURRENT HOLDINGS  vs  S&P 500",
                      backtest_vs_benchmark(cur_w, TEST_START, TEST_END, benchmark=BENCHMARK))

    # Equal weight over the same tickers, for reference.
    eq_w = {t: 1.0 / len(tickers) for t in tickers}
    show_backtest("EQUAL-WEIGHT BASKET  vs  S&P 500",
                  backtest_vs_benchmark(eq_w, TEST_START, TEST_END, benchmark=BENCHMARK))

    # The most honest test: walk-forward, every period out-of-sample.
    # Extended window for more rebalance periods; 1/N run through the same
    # harness (same dates, same cost model) as the baseline to beat.
    WF_START = "2022-01-01"
    try:
        def _equal_weight_optimizer(tks, s, e):
            return {t: 1.0 / len(tks) for t in tks}, None, "1/N"

        port_rets, bench_rets, periods = run_walk_forward(tickers, WF_START, TEST_END)
        eq_rets, _, _ = run_walk_forward(tickers, WF_START, TEST_END,
                                         optimizer=_equal_weight_optimizer)
        show_walk_forward(
            port_rets, bench_rets, periods,
            title="WALK-FORWARD: min-vol capped + 1/N shrink, semi-annual",
            eq_rets=eq_rets,
        )
    except Exception as e:
        print(f"\nWalk-forward failed: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
