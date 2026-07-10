# mcp_portfolio_server.py
import json
from typing import List

from mcp.server.fastmcp import FastMCP

from api.portfolio_core import (
    load_price_history,
    compute_portfolio_metrics,
    recommend_portfolio,
    backtest_vs_benchmark,
)

server = FastMCP("portfolio-mcp")


@server.tool()
async def mcp_load_price_history(
    tickers: List[str],
    start: str,
    end: str,
) -> str:
    """
    Load daily adjusted close prices for given tickers between start and end.
    """
    prices = load_price_history(tickers, start, end)
    return json.dumps(prices.to_dict(orient="list"), default=str)


@server.tool()
async def mcp_compute_portfolio_metrics(
    tickers: List[str],
    start: str,
    end: str,
    weights_json: str = "",
    benchmark: str = "SPY",
) -> str:
    """
    Compute CAGR, Sharpe ratio, volatility, max drawdown, and a benchmark
    comparison (vs SPY by default).

    weights_json: optional {ticker: weight} mapping; equal-weight if empty.
    """
    metrics = compute_portfolio_metrics(
        tickers=tickers,
        start=start,
        end=end,
        weights_json=weights_json or None,
        benchmark=benchmark,
    )
    return json.dumps(metrics)


@server.tool()
async def mcp_backtest_vs_benchmark(
    weights_json: str,
    start: str,
    end: str,
    benchmark: str = "SPY",
) -> str:
    """
    Backtest a {ticker: weight} portfolio against a benchmark index.
    Returns portfolio/benchmark metrics plus alpha, beta, and excess return.
    """
    weights = json.loads(weights_json)
    result = backtest_vs_benchmark(weights, start, end, benchmark=benchmark)
    return json.dumps(result)


@server.tool()
async def mcp_recommend_portfolio(
    tickers: List[str],
    start: str,
    end: str,
    constraints_json: str = "{}",
) -> str:
    """
    Recommend portfolio weights given tickers, a date range, and constraints.
    """
    prices = load_price_history(tickers, start, end)
    constraints = json.loads(constraints_json) if constraints_json else {}
    weights = recommend_portfolio(prices, constraints)
    return json.dumps(weights)


if __name__ == "__main__":
    server.run()
