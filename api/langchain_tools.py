import json
from typing import List

import numpy as np
import pandas as pd
from langchain.tools import tool

from pathlib import Path


from .portfolio_core import (
    get_session_portfolio,
    load_price_history,
    recommend_portfolio,
    load_price_on_date,
    compute_metrics_from_holdings,
    backtest_vs_benchmark,
    backtest_sectors_vs_benchmark,
)
from .sentiment import (
    analyze_ticker_sentiment,
    analyze_portfolio_sentiment,
    apply_sentiment_tilt,
)


import threading

# Cap concurrent FinRL optimizations. Each run holds a worker thread for up to
# ~180s; without this a burst of chat requests could exhaust the server's
# threadpool and stall the whole app. Excess requests are told to retry instead
# of piling up.
_FINRL_SLOTS = threading.BoundedSemaphore(2)


def _session_weights(session_id: str) -> dict:
    """Ticker -> dollar value from the session's uploaded holdings ({} if none)."""
    pf = get_session_portfolio(session_id)
    if not pf:
        return {}
    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        if not tkr:
            continue
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        try:
            weights[str(tkr).upper()] = weights.get(str(tkr).upper(), 0.0) + float(val)
        except Exception:
            continue
    return {t: v for t, v in weights.items() if v > 0}

@tool
def lc_compute_metrics_from_portfolio(holdings_json: str, start: str, end: str) -> str:
    """
    Compute portfolio metrics (CAGR, volatility, Sharpe, max drawdown)
    directly from a holdings JSON file like the one the user uploaded.
    """
    try:
        result = compute_metrics_from_holdings(holdings_json, start, end)
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("lc_price_on_date_tool")
def lc_price_on_date_tool(ticker: str, date: str) -> str:
    """
    Return the historical price of a single stock ticker on or near a given date.

    Input:
      - ticker: stock symbol (e.g. "NVDA")
      - date: string in YYYY-MM-DD format

    Output (on success):
      JSON: {"ticker": "...", "requested_date": "...", "price": 123.45}
    Output (on failure):
      JSON: {"error": "..."}
    """
    try:
        price = load_price_on_date(ticker, date)
        result = {
            "ticker": ticker.upper(),
            "requested_date": date,
            "price": round(float(price), 4),
        }
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("lc_load_price_history")
def lc_load_price_history(
    tickers: List[str],
    start: str,
    end: str,
    max_points: int = 120,
) -> str:
    """
    Load daily historical prices for the given tickers between start and end dates.

    Inputs:
      - tickers: list of stock symbols (e.g. ["NVDA"])
      - start: start date in YYYY-MM-DD format
      - end: end date in YYYY-MM-DD format
      - max_points: maximum number of rows to return (downsampled if necessary)

    Output (on success):
      JSON-encoded DataFrame with orient="split" and at most max_points rows.
    Output (on failure):
      JSON: {"error": "..."}
    """
    try:
        prices = load_price_history(tickers, start, end)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        if prices.empty:
            return json.dumps(
                {
                    "error": (
                        f"No price data returned for tickers {tickers} "
                        f"between {start} and {end}."
                    )
                }
            )

        prices = prices.sort_index()

        if max_points and max_points > 0 and len(prices) > max_points:
            idx = np.linspace(0, len(prices) - 1, max_points).round().astype(int)
            prices = prices.iloc[idx]

        text = prices.to_json(date_format="iso", orient="split")
        if len(text) > 8000:
            text = text[:8000]
        return text
    except Exception as e:
        return json.dumps({"error": f"Failed to load price history: {e}"})



@tool
def lc_recommend_portfolio(
    tickers: List[str],
    start: str,
    end: str,
    constraints_json: str,
) -> str:
    """
    Recommend portfolio weights using your RL or optimization logic.

    Inputs:
      - tickers: list of stock symbols
      - start: start date in YYYY-MM-DD format
      - end: end date in YYYY-MM-DD format
      - constraints_json: JSON-encoded dict of constraint parameters

    Output (on success):
      JSON mapping ticker -> recommended weight.
    Output (on failure):
      JSON: {"error": "..."}
    """
    try:
        prices = load_price_history(tickers, start, end)
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
        if prices.empty:
            return json.dumps(
                {
                    "error": (
                        f"No price data returned for tickers {tickers} "
                        f"between {start} and {end}."
                    )
                }
            )
    except Exception as e:
        return json.dumps({"error": f"Failed to load price history: {e}"})

    try:
        constraints = json.loads(constraints_json) if constraints_json else {}
        if not isinstance(constraints, dict):
            return json.dumps(
                {"error": "constraints_json must be a JSON object of constraints."}
            )
    except Exception as e:
        return json.dumps({"error": f"Invalid constraints_json: {e}"})

    try:
        weights = recommend_portfolio(prices, constraints)
        return json.dumps(weights)
    except Exception as e:
        return json.dumps({"error": f"Failed to recommend portfolio: {e}"})


@tool
def lc_get_portfolio_holdings(session_id: str) -> str:
    """
    Return the uploaded holdings for this session as pretty-printed JSON for debugging.
    """
    pf = get_session_portfolio(session_id)
    if not pf:
        return "No portfolio uploaded for this session."
    return json.dumps(pf, indent=2)


@tool
def load_user_portfolio(session_id: str) -> str:
    """
    Load the user's portfolio holdings for this session as JSON.

    Input:
      - session_id: identifier for this chat session

    Output (on success):
      JSON list of holding objects.
    Output (on failure):
      JSON: {"error": "..."}
    """
    pf = get_session_portfolio(session_id)
    if not pf:
        return json.dumps(
            {"error": f"No portfolio found for session '{session_id}'. Please upload a portfolio file first."}
        )

    if isinstance(pf, dict):
        holdings = pf.get("holdings", [])
    else:
        holdings = pf

    if not isinstance(holdings, list) or not holdings:
        return json.dumps({"error": "Portfolio is present but empty or in an unexpected format."})

    return json.dumps(holdings)


@tool("compute_total_value")
def compute_total_value(holdings_json: str) -> str:
    """
    Deterministically compute the total current dollar value of a portfolio.

    Accepts JSON in either of these shapes:
      - [{"ticker": "...", "shares": 0.1, "price": 256.12, ...}, ...]
      - {"holdings": [ ... ], ... }

    Each holding may have:
      * current_dollars / total_value / value
      * or (shares * price/close/last_price/adj_close)

    Output:
      JSON: {"total_value": 596.62}
    """
    try:
        obj = json.loads(holdings_json)
    except Exception as e:
        return json.dumps({"error": f"Error parsing holdings_json: {e}"})

    if isinstance(obj, list):
        holdings = obj
    elif isinstance(obj, dict):
        holdings = None
        for key in ("holdings", "portfolio", "positions", "data"):
            if isinstance(obj.get(key), list):
                holdings = obj[key]
                break
        if holdings is None:
            return json.dumps({"total_value": 0.0})
    else:
        return json.dumps({"total_value": 0.0})

    total = 0.0

    for h in holdings:
        if not isinstance(h, dict):
            continue

        val = (
            h.get("current_dollars")
            or h.get("total_value")
            or h.get("value")
        )

        if val is None:
            shares = (
                h.get("shares")
                or h.get("quantity")
                or h.get("qty")
                or h.get("volume")
            )
            price = (
                h.get("price")
                or h.get("close")
                or h.get("last_price")
                or h.get("adj_close")
            )
            if shares is not None and price is not None:
                try:
                    val = float(shares) * float(price)
                except Exception:
                    val = None

        if val is not None:
            try:
                total += float(val)
            except Exception:
                pass

    return json.dumps({"total_value": round(total, 2)})

@tool("portfolio_holdings_count")
def portfolio_holdings_count(session_id: str) -> str:
    """
    Return the exact number of holdings rows stored for a session_id.
    Output: JSON {"count": int}
    """
    pf = get_session_portfolio(session_id)
    if not pf:
        return json.dumps({"error": f"No portfolio found for session '{session_id}'."})

    if isinstance(pf, dict):
        holdings = pf.get("holdings", [])
    else:
        holdings = pf

    return json.dumps({"count": len(holdings)})

@tool("lc_ticker_sentiment")
def lc_ticker_sentiment(ticker: str, limit: int = 8) -> str:
    """
    Analyze recent financial-news sentiment for a single stock ticker.

    Pulls recent headlines (Yahoo Finance and its news providers) and scores
    them as Bullish / Neutral / Bearish.

    Input:
      - ticker: stock symbol (e.g. "NVDA")
      - limit: max number of headlines to analyze (default 8)

    Output (JSON):
      {ticker, avg_score (-1..1), label, counts, n_headlines, headlines:[{title, score, label, publisher, url}]}
    """
    try:
        result = analyze_ticker_sentiment(ticker, limit=int(limit))
        # Trim headline payload so the model isn't flooded with raw text.
        result["headlines"] = [
            {k: h[k] for k in ("title", "score", "label", "publisher") if k in h}
            for h in result.get("headlines", [])[:limit]
        ]
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("lc_portfolio_sentiment")
def lc_portfolio_sentiment(session_id: str, limit_per_ticker: int = 5) -> str:
    """
    Analyze news sentiment across the user's uploaded portfolio.

    Loads the holdings for this session, scores recent news for each holding,
    and returns a dollar-weighted overall sentiment plus the most bullish and
    most bearish names.

    Input:
      - session_id: the active chat session identifier
      - limit_per_ticker: headlines to score per holding (default 5)

    Output (JSON):
      {portfolio_sentiment_score, portfolio_label, most_bullish, most_bearish, per_ticker}
    """
    weights = _session_weights(session_id)
    if not weights:
        return json.dumps({"error": f"No portfolio with usable holdings found for session '{session_id}'. Upload a portfolio first."})

    try:
        result = analyze_portfolio_sentiment(weights, limit_per_ticker=int(limit_per_ticker))
        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool("lc_backtest_portfolio")
def lc_backtest_portfolio(session_id: str, start: str = "", end: str = "") -> str:
    """
    Backtest the user's uploaded portfolio against the S&P 500 (SPY), at both
    the stock level and the sector level, and render an equity-curve chart.

    Includes realistic assumptions: transaction costs and a risk-free rate in
    Sharpe ratios.

    Input:
      - session_id: the active chat session identifier
      - start, end: optional YYYY-MM-DD dates (default: the last 12 months)

    Output (JSON): {stocks: {...}, sectors: {...}, chart_url}
      Each block has portfolio metrics, benchmark metrics, and a comparison
      (excess return, alpha, beta, information ratio, outperformed).
      chart_url is a path like /charts/x.png — include it verbatim in your answer.
    """
    from datetime import datetime, timedelta

    weights = _session_weights(session_id)
    if not weights:
        return json.dumps({"error": f"No portfolio with usable holdings found for session '{session_id}'. Upload a portfolio first."})

    if not end:
        end = datetime.today().strftime("%Y-%m-%d")
    if not start:
        start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    try:
        stocks = backtest_vs_benchmark(weights, start, end, return_curves=True)
    except Exception as e:
        return json.dumps({"error": f"Stock-level backtest failed: {e}"})

    chart_url = None
    curves = stocks.pop("curves", None)
    if curves:
        try:
            from .charting import equity_curve_png

            chart_url = equity_curve_png(
                curves["dates"], curves["portfolio"], curves["benchmark"]
            )
        except Exception:
            chart_url = None  # chart is a bonus; the numbers still stand

    try:
        sectors = backtest_sectors_vs_benchmark(weights, start, end)
        # Curves/headline detail aren't needed twice; keep the sector block lean.
        sectors = {
            "sector_weights": sectors.get("sector_weights"),
            "sector_etf_weights": sectors.get("sector_etf_weights"),
            "portfolio": sectors.get("portfolio"),
            "comparison": sectors.get("comparison"),
        }
    except Exception as e:
        sectors = {"error": f"Sector-level backtest failed: {e}"}

    return json.dumps({"stocks": stocks, "sectors": sectors, "chart_url": chart_url})


@tool("lc_sentiment_tilt")
def lc_sentiment_tilt(weights_json: str, strength: float = 0.2) -> str:
    """
    Adjust a {ticker: weight} allocation using current news sentiment.

    Each weight is scaled by (1 + strength * sentiment_score) and renormalized,
    so bullish names are overweighted and bearish names underweighted. Use after
    an optimization when the user wants news factored into the recommendation.

    Input:
      - weights_json: JSON object mapping ticker -> weight
      - strength: max relative adjustment (0..1, default 0.2 = +/-20%)

    Output (JSON): {weights, sentiment (per-ticker scores), strength}
    """
    try:
        weights = json.loads(weights_json)
        if not isinstance(weights, dict) or not weights:
            return json.dumps({"error": "weights_json must be a non-empty JSON object of ticker -> weight."})
    except Exception as e:
        return json.dumps({"error": f"Invalid weights_json: {e}"})

    try:
        return json.dumps(apply_sentiment_tilt(weights, strength=float(strength)))
    except Exception as e:
        return json.dumps({"error": str(e)})


def _user_id_from_session(session_id: str):
    """Chat session ids look like 'u{user_id}.c{chat_id}'. Pull the user_id out."""
    sid = str(session_id or "")
    if sid.startswith("u") and ".c" in sid:
        try:
            return int(sid[1:].split(".c", 1)[0])
        except (ValueError, IndexError):
            return None
    return None


@tool("lc_get_profile")
def lc_get_profile(session_id: str) -> str:
    """
    Return the user's saved investor profile (from the Plan tab): years to
    retirement, risk tolerance (1-10), max acceptable volatility, goal, and
    monthly contribution. Also returns a "target_volatility" DECIMAL fraction
    (e.g. 0.18) derived from risk tolerance and capped at the user's max — pass
    that straight into finrl_optimize_portfolio.

    Call this before recommending/optimizing a portfolio so the recommendation
    matches the user's risk tolerance. If it returns {"has_profile": false},
    the user hasn't saved a profile yet — ASK them for their risk tolerance and
    acceptable volatility (or send them to the Plan tab) before optimizing.

    Input:
      - session_id: the active chat session identifier
    """
    from api import db

    user_id = _user_id_from_session(session_id)
    if user_id is None:
        return json.dumps({"has_profile": False, "error": "Could not identify the user for this session."})

    profile = db.get_profile(user_id)
    if not profile:
        return json.dumps({"has_profile": False,
                           "note": "No saved profile. Ask the user for risk tolerance (1-10) and max volatility, or point them to the Plan tab."})

    # Map risk tolerance (1-10) to a target annualized volatility, then cap it
    # at the user's stated maximum. Mirrors api.recommend.target_volatility.
    try:
        r = max(1, min(int(profile.get("risk_tolerance", 5)), 10))
    except (TypeError, ValueError):
        r = 5
    target_vol = 0.10 + 0.025 * r
    try:
        max_vol = float(profile.get("max_volatility_pct", 30)) / 100.0
        if max_vol > 0:
            target_vol = min(target_vol, max_vol)
    except (TypeError, ValueError):
        pass

    return json.dumps({
        "has_profile": True,
        "years_to_retirement": profile.get("years_to_retirement"),
        "risk_tolerance": profile.get("risk_tolerance"),
        "max_volatility_pct": profile.get("max_volatility_pct"),
        "goal": profile.get("goal"),
        "monthly_contribution": profile.get("monthly_contribution"),
        "target_volatility": round(target_vol, 4),
    })


@tool("finrl_optimize_portfolio")
def finrl_optimize_portfolio(session_id: str, target_volatility: float = 0.0) -> str:
    """
    Run FinRL-based portfolio optimization on the user's uploaded portfolio,
    optionally targeting the user's risk tolerance.

    Input:
      - session_id: the active chat session identifier
      - target_volatility: desired annualized volatility as a DECIMAL fraction
        (e.g. 0.18 for 18%). When > 0, the allocation maximizes expected return
        at that volatility level ("max-Sharpe matched to the user's risk").
        Pass 0 (or omit) to use the default conservative allocation.
        Get this from lc_get_profile (use its "target_volatility" field), or
        ask the user for their risk tolerance / acceptable volatility first.

    Output:
      - JSON containing optimized portfolio weights, the target volatility used,
        sector breakdown, dollar targets, and trade plan
    """

    pf = get_session_portfolio(session_id)
    if not pf:
        return json.dumps({"error": f"No portfolio found for session '{session_id}'. Upload a portfolio first."})

    if isinstance(pf, dict):
        holdings = pf.get("holdings", [])
    else:
        holdings = pf

    if not isinstance(holdings, list) or not holdings:
        return json.dumps({"error": "Portfolio is present but empty or in an unexpected format."})

    mem_dir = Path("chat_memory")
    mem_dir.mkdir(exist_ok=True)

    json_path = mem_dir / f"{session_id}.finrl_input.json"
    try:
        json_path.write_text(json.dumps(holdings, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        return json.dumps({"error": f"Failed to write portfolio file: {e}"})

    # Shed load rather than stall: if the concurrency slots are full, ask the
    # user to retry instead of holding a thread waiting.
    if not _FINRL_SLOTS.acquire(blocking=False):
        return json.dumps({"error": "The optimizer is busy with other requests right now. Please try again in a minute."})

    try:
        # Import here to avoid import-time issues
        from api.predict_agent import optimize_portfolio_with_finrl
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

        tgt_vol = float(target_volatility) if target_volatility and target_volatility > 0 else None

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            optimize_portfolio_with_finrl,
            json_path=str(json_path),
            include_etfs=False,
            lookback_years=5,
            finrl_timesteps=5000,
            portfolio_value=None,
            min_trade_dollars=5.0,
            fractional_ok=True,
            target_volatility=tgt_vol,
        )

        try:
            result = future.result(timeout=180)
            return json.dumps(result, ensure_ascii=False)
        except FutureTimeoutError:
            # Don't leave the user empty-handed: fall back to the fast
            # optimizer over the same tickers, honoring the target volatility.
            try:
                from datetime import datetime, timedelta
                from api.predict_agent import (
                    run_pypfopt_max_sharpe,
                    run_max_sharpe_target_vol,
                    load_tickers_from_portfolio_json,
                )

                tickers, _ = load_tickers_from_portfolio_json(str(json_path), include_etfs=False)
                end = datetime.today().strftime("%Y-%m-%d")
                start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
                if tgt_vol:
                    alloc, _, method = run_max_sharpe_target_vol(tickers, start, end, tgt_vol)
                else:
                    alloc, _, method = run_pypfopt_max_sharpe(tickers, start, end)
                return json.dumps({
                    "method": f"{method} (RL timed out; fast fallback)",
                    "target_volatility": tgt_vol,
                    "final_allocation_weights": alloc,
                    "note": "The RL optimization exceeded 180s; these weights come from the fast optimizer instead.",
                })
            except Exception as e2:
                return json.dumps({"error": f"Optimization timed out and the fallback also failed: {e2}"})
        finally:
            executor.shutdown(wait=False)

    except ImportError as e:
        return json.dumps({"error": f"Failed to import optimization module: {e}"})
    except Exception as e:
        return json.dumps({"error": f"Optimization failed: {str(e)}"})
    finally:
        _FINRL_SLOTS.release()
