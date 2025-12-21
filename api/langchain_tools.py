import json
from typing import List

import numpy as np
import pandas as pd
from langchain.tools import tool

from .portfolio_core import (
    SESSION_PORTFOLIOS,
    load_price_history,
    recommend_portfolio,
    load_price_on_date,
    compute_metrics_from_holdings,
)

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
    pf = SESSION_PORTFOLIOS.get(session_id)
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
    pf = SESSION_PORTFOLIOS.get(session_id)
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
