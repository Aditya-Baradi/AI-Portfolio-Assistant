# langchain_tools.py

import json
from typing import Dict, List

import pandas as pd
from langchain.tools import tool

from portfolio_core import (
    load_price_history,
    compute_portfolio_metrics,
    recommend_portfolio,
)


@tool
def lc_load_price_history(tickers: List[str], start: str, end: str) -> str:
    """
    Load daily adjusted close prices for the provided tickers between start and end
    dates (YYYY-MM-DD). Returns a JSON string representation of the price DataFrame.
    """
    prices = load_price_history(tickers, start, end)
    # JSON string; easier for the LLM to pass around
    return prices.to_json()


@tool
def lc_compute_portfolio_metrics(
    prices_json: str,
    weights_json: str,
    benchmark: str = "SPY",
) -> str:
    """
    Compute portfolio metrics (CAGR, Sharpe, annual volatility, max drawdown).

    - prices_json: JSON-encoded DataFrame from lc_load_price_history
    - weights_json: JSON-encoded dict {ticker: weight}
    """
    prices = pd.read_json(prices_json)
    weights: Dict[str, float] = json.loads(weights_json)

    metrics = compute_portfolio_metrics(prices, weights, benchmark)
    return json.dumps(metrics)


@tool
def lc_recommend_portfolio(prices_json: str, constraints_json: str) -> str:
    """
    Recommend portfolio weights using your RL/optimization logic.

    - prices_json: JSON-encoded DataFrame of prices
    - constraints_json: JSON-encoded dict of constraint parameters:
        e.g. {"max_weight": 0.2, "min_weight": 0.01, "cash_ticker": "BIL"}
    """
    prices = pd.read_json(prices_json)
    constraints = json.loads(constraints_json)

    weights = recommend_portfolio(prices, constraints)
    return json.dumps(weights)
