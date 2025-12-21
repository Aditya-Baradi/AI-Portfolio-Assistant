# mcp_portfolio_server.py
import json
from typing import Dict, List

from mcp.server.fastmcp import FastMCP, Context, ToolResult
from portfolio_core import (
    load_price_history,
    compute_portfolio_metrics,
    recommend_portfolio,
)

server = FastMCP("portfolio-mcp")

@server.tool()
async def mcp_load_price_history(
    context: Context,
    tickers: List[str],
    start: str,
    end: str,
) -> ToolResult:
    """
    Load daily adjusted close prices for given tickers between start and end.
    """
    prices = load_price_history(tickers, start, end)
    # return as JSON-serializable
    json_data = prices.to_dict(orient="list")
    return ToolResult(content=json.dumps(json_data))


@server.tool()
async def mcp_compute_portfolio_metrics(
    context: Context,
    prices_json: str,
    weights_json: str,
    benchmark: str = "SPY",
) -> ToolResult:
    """
    Compute CAGR, Sharpe ratio, volatility, and max drawdown.
    prices_json: JSON from mcp_load_price_history
    weights_json: mapping {ticker: weight}
    """
    import pandas as pd

    prices_dict = json.loads(prices_json)
    prices = pd.DataFrame(prices_dict)
    weights: Dict[str, float] = json.loads(weights_json)

    metrics = compute_portfolio_metrics(prices, weights, benchmark)
    return ToolResult(content=json.dumps(metrics))


@server.tool()
async def mcp_recommend_portfolio(
    context: Context,
    prices_json: str,
    constraints_json: str,
) -> ToolResult:
    """
    Recommend portfolio weights given prices and constraints.
    """
    import pandas as pd

    prices_dict = json.loads(prices_json)
    prices = pd.DataFrame(prices_dict)
    constraints = json.loads(constraints_json)

    weights = recommend_portfolio(prices, constraints)
    return ToolResult(content=json.dumps(weights))


if __name__ == "__main__":
    # This will run a stdio server or HTTP depending on how you configure it
    server.run()
