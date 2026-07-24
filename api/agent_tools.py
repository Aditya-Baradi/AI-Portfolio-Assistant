"""Server-bound tools for the educational chat assistant.

The model is deliberately never given a user id, chat id, session id, file
path, or database key.  Those values live in :class:`AgentContext` and are
applied by the server-side dispatcher after the model chooses a tool.

This module intentionally excludes allocation optimizers, trade generators,
and sentiment-based portfolio tilts.  Those features turn an educational
analytics assistant into a personalized recommendation engine and should not
be exposed without a separate product/compliance decision.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd

from api import db

logger = logging.getLogger("evergreen.agent.tools")

MAX_TOOL_OUTPUT_CHARS = 20_000
MAX_HISTORY_DAYS = 365 * 10
MAX_SUMMARY_HOLDINGS = 50
MAX_PORTFOLIO_NEWS_TICKERS = 20
_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,9}$")


@dataclass(frozen=True)
class AgentContext:
    """Authenticated request context that is never serialized for the model."""

    user_id: int
    chat_id: str


def _json(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str, allow_nan=False)
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text
    return json.dumps(
        {
            "error": "Tool output was too large to send to the model.",
            "truncated": True,
        }
    )


def _error(message: str) -> str:
    return _json({"error": message})


def _portfolio(context: AgentContext) -> dict[str, Any] | list[Any] | None:
    return db.get_portfolio(context.user_id)


def _holdings(context: AgentContext) -> list[dict[str, Any]]:
    portfolio = _portfolio(context)
    if not portfolio:
        return []
    if isinstance(portfolio, dict):
        rows = portfolio.get("holdings", [])
    else:
        rows = portfolio
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _ticker(row: dict[str, Any]) -> str:
    return str(row.get("ticker") or row.get("symbol") or row.get("tic") or "").upper()


def _live_valuation(context: AgentContext) -> dict[str, Any]:
    from api.portfolio_core import live_portfolio_valuation

    portfolio = _portfolio(context)
    if not portfolio:
        return {
            "values": {},
            "prices": {},
            "shares": {},
            "total_value": 0.0,
            "as_of": None,
            "fallback_tickers": [],
        }
    return live_portfolio_valuation(portfolio)


def _portfolio_weights(context: AgentContext) -> dict[str, float]:
    valuation = _live_valuation(context)
    fallback = valuation.get("fallback_tickers") or []
    if fallback:
        raise ValueError(
            "Current market data is unavailable for: "
            + ", ".join(str(ticker) for ticker in fallback)
            + ". The analysis was not run on stale import values."
        )
    values = {
        str(ticker): float(value)
        for ticker, value in (valuation.get("values") or {}).items()
        if float(value) > 0
    }
    total = sum(values.values())
    return {ticker: value / total for ticker, value in values.items()} if total > 0 else {}


def _date_range(start: str, end: str) -> tuple[str, str]:
    try:
        start_date = date.fromisoformat(str(start))
        end_date = date.fromisoformat(str(end))
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD.") from exc
    if start_date >= end_date:
        raise ValueError("The start date must be before the end date.")
    if (end_date - start_date).days > MAX_HISTORY_DAYS:
        raise ValueError("The requested range may not exceed ten years.")
    if end_date > datetime.now(UTC).date() + timedelta(days=1):
        raise ValueError("The end date may not be in the future.")
    return start_date.isoformat(), end_date.isoformat()


def _safe_ticker(raw: Any) -> str:
    ticker = str(raw or "").strip().upper()
    if not _TICKER_RE.fullmatch(ticker):
        raise ValueError("Ticker must be 1-10 letters, numbers, dots, or hyphens.")
    return ticker


def _portfolio_summary(context: AgentContext, _args: dict[str, Any]) -> str:
    holdings = _holdings(context)
    if not holdings:
        return _error("No imported portfolio was found for this account.")

    valuation = _live_valuation(context)
    values = valuation.get("values") or {}
    prices = valuation.get("prices") or {}
    shares = valuation.get("shares") or {}
    rows = []
    metadata = {_ticker(holding): holding for holding in holdings}
    for ticker, value in values.items():
        holding = metadata.get(ticker, {})
        row: dict[str, Any] = {
            "ticker": ticker,
            "estimated_value": round(float(value), 2),
            "shares": shares.get(ticker),
            "latest_adjusted_price": prices.get(ticker),
        }
        for key in ("asset_type", "sector"):
            if holding.get(key) is not None:
                row[key] = holding[key]
        rows.append(row)
    rows.sort(key=lambda row: float(row["estimated_value"]), reverse=True)
    visible_rows = rows[:MAX_SUMMARY_HOLDINGS]
    visible_value = sum(float(row["estimated_value"]) for row in visible_rows)
    total_value = float(valuation.get("total_value") or 0.0)

    return _json(
        {
            "holding_count": len(rows),
            "holdings_returned": len(visible_rows),
            "estimated_portfolio_value": round(total_value, 2),
            "returned_value_coverage": round(visible_value / total_value, 4)
            if total_value > 0
            else 0.0,
            "as_of": valuation.get("as_of"),
            "source_provider": valuation.get("source_provider"),
            "price_semantics": valuation.get("price_semantics"),
            "fallback_tickers": valuation.get("fallback_tickers") or [],
            "holdings": visible_rows,
            "valuation_note": (
                "Values are analytics estimates from the latest configured "
                "market-data refresh, not a brokerage statement or executable "
                "quote. Any fallback_tickers use stored import values and must be "
                "identified explicitly."
            ),
        }
    )


def _investor_profile(context: AgentContext, _args: dict[str, Any]) -> str:
    profile = db.get_profile(context.user_id)
    if not profile:
        return _json({"has_profile": False})
    return _json({"has_profile": True, "profile": profile})


def _portfolio_metrics(context: AgentContext, args: dict[str, Any]) -> str:
    from api.portfolio_core import compute_metrics_from_holdings

    holdings = _holdings(context)
    if not holdings:
        return _error("No imported portfolio was found for this account.")
    valuation = _live_valuation(context)
    fallback = valuation.get("fallback_tickers") or []
    if fallback:
        return _error(
            "Current market data is unavailable for "
            + ", ".join(str(ticker) for ticker in fallback)
            + "; metrics were not run on stale import values."
        )
    live_values = valuation.get("values") or {}
    holdings = [
        {**holding, "current_dollars": live_values.get(_ticker(holding), 0.0)}
        for holding in holdings
    ]
    start, end = _date_range(args.get("start", ""), args.get("end", ""))
    result = compute_metrics_from_holdings(json.dumps(holdings), start, end)
    if isinstance(result, dict):
        result.setdefault(
            "limitations",
            "Historical, hypothetical analytics; not a prediction or recommendation.",
        )
    return _json(result)


def _benchmark_comparison(context: AgentContext, args: dict[str, Any]) -> str:
    from api.portfolio_core import backtest_vs_benchmark

    weights = _portfolio_weights(context)
    if not weights:
        return _error("No portfolio with usable positive values was found.")
    start, end = _date_range(args.get("start", ""), args.get("end", ""))
    result = backtest_vs_benchmark(weights, start, end, return_curves=False)
    if isinstance(result, dict):
        result["limitations"] = (
            "Hypothetical historical comparison using model assumptions and "
            "estimated data. It is not actual account performance."
        )
    return _json(result)


def _price_on_date(_context: AgentContext, args: dict[str, Any]) -> str:
    from api.portfolio_core import load_price_on_date

    ticker = _safe_ticker(args.get("ticker"))
    requested = date.fromisoformat(str(args.get("date", ""))).isoformat()
    price = float(load_price_on_date(ticker, requested))
    return _json(
        {
            "ticker": ticker,
            "requested_date": requested,
            "adjusted_price": round(price, 4),
            "price_semantics": (
                "Historical adjusted series supplied by the configured provider; "
                "not an executable quote."
            ),
        }
    )


def _price_history_summary(_context: AgentContext, args: dict[str, Any]) -> str:
    from api.portfolio_core import load_price_history

    raw_tickers = args.get("tickers")
    if not isinstance(raw_tickers, list) or not raw_tickers:
        raise ValueError("tickers must be a non-empty list.")
    if len(raw_tickers) > 10:
        raise ValueError("At most ten tickers may be summarized at once.")
    tickers = list(dict.fromkeys(_safe_ticker(t) for t in raw_tickers))
    start, end = _date_range(args.get("start", ""), args.get("end", ""))
    frame = load_price_history(tickers, start, end)
    if isinstance(frame, pd.Series):
        frame = frame.to_frame(name=tickers[0])
    if frame is None or frame.empty:
        return _error("The configured market-data provider returned no data.")

    summaries: dict[str, Any] = {}
    for ticker in tickers:
        if ticker not in frame.columns:
            continue
        series = pd.to_numeric(frame[ticker], errors="coerce").dropna()
        if series.empty:
            continue
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        summaries[ticker] = {
            "first_date": series.index[0].strftime("%Y-%m-%d"),
            "last_date": series.index[-1].strftime("%Y-%m-%d"),
            "first_adjusted_price": round(first, 4),
            "last_adjusted_price": round(last, 4),
            "period_return_pct": round((last / first - 1.0) * 100.0, 2)
            if first > 0
            else None,
            "observations": int(series.size),
        }
    return _json(
        {
            "series": summaries,
            "price_semantics": (
                "Adjusted historical series from the configured provider. "
                "Provider methodology may include corporate-action adjustments."
            ),
        }
    )


def _ticker_news_tone(_context: AgentContext, args: dict[str, Any]) -> str:
    from api.sentiment import analyze_ticker_sentiment

    ticker = _safe_ticker(args.get("ticker"))
    limit = max(3, min(int(args.get("limit", 8)), 12))
    result = analyze_ticker_sentiment(ticker, limit=limit)
    if isinstance(result, dict):
        result["headlines"] = [
            {
                key: headline[key]
                for key in ("title", "score", "label", "publisher")
                if key in headline
            }
            for headline in result.get("headlines", [])[:limit]
            if isinstance(headline, dict)
        ]
        result["interpretation"] = (
            "This describes headline language only. It is not a security rating "
            "and has no established predictive relationship with returns."
        )
    return _json(result)


def _portfolio_news_tone(context: AgentContext, args: dict[str, Any]) -> str:
    from api.sentiment import analyze_portfolio_sentiment

    weights = _portfolio_weights(context)
    if not weights:
        return _error("No portfolio with usable positive values was found.")
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    selected = dict(ordered[:MAX_PORTFOLIO_NEWS_TICKERS])
    limit = max(3, min(int(args.get("limit_per_ticker", 5)), 10))
    result = analyze_portfolio_sentiment(selected, limit_per_ticker=limit)
    if isinstance(result, dict):
        result["coverage"] = {
            "holdings_analyzed": len(selected),
            "holdings_total": len(weights),
            "portfolio_weight_analyzed": round(sum(selected.values()), 4),
            "selection": "largest current positions first",
        }
        result["interpretation"] = (
            "This is a value-weighted description of recent headline language, "
            "not a recommendation or forecast."
        )
    return _json(result)


_HANDLERS: dict[str, Callable[[AgentContext, dict[str, Any]], str]] = {
    "get_portfolio_summary": _portfolio_summary,
    "get_investor_profile": _investor_profile,
    "get_portfolio_metrics": _portfolio_metrics,
    "compare_portfolio_to_benchmark": _benchmark_comparison,
    "get_price_on_date": _price_on_date,
    "summarize_price_history": _price_history_summary,
    "get_ticker_news_tone": _ticker_news_tone,
    "get_portfolio_news_tone": _portfolio_news_tone,
}


def _function(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _function(
        "get_portfolio_summary",
        "Return the authenticated user's imported holdings and estimated values.",
        {},
    ),
    _function(
        "get_investor_profile",
        "Return the authenticated user's saved planning assumptions.",
        {},
    ),
    _function(
        "get_portfolio_metrics",
        "Calculate historical portfolio return and risk metrics over a date range.",
        {
            "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
        },
        ["start", "end"],
    ),
    _function(
        "compare_portfolio_to_benchmark",
        "Compare the current portfolio mix with the S&P 500 over a historical range.",
        {
            "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
        },
        ["start", "end"],
    ),
    _function(
        "get_price_on_date",
        "Return a historical adjusted price for one ticker.",
        {
            "ticker": {"type": "string", "description": "Stock or ETF symbol."},
            "date": {"type": "string", "description": "Date, YYYY-MM-DD."},
        },
        ["ticker", "date"],
    ),
    _function(
        "summarize_price_history",
        "Summarize adjusted-price changes; does not return a large raw time series.",
        {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 10,
            },
            "start": {"type": "string", "description": "Start date, YYYY-MM-DD."},
            "end": {"type": "string", "description": "End date, YYYY-MM-DD."},
        },
        ["tickers", "start", "end"],
    ),
    _function(
        "get_ticker_news_tone",
        "Describe the tone of recent headlines for one ticker; not a rating.",
        {
            "ticker": {"type": "string"},
            "limit": {"type": "integer", "minimum": 3, "maximum": 12},
        },
        ["ticker", "limit"],
    ),
    _function(
        "get_portfolio_news_tone",
        "Describe recent headline tone across the authenticated user's holdings.",
        {
            "limit_per_ticker": {"type": "integer", "minimum": 3, "maximum": 10},
        },
        ["limit_per_ticker"],
    ),
]


def dispatch_tool(context: AgentContext, name: str, arguments: dict[str, Any]) -> str:
    """Execute one allow-listed tool with server-owned authentication context."""

    handler = _HANDLERS.get(name)
    if handler is None:
        return _error("Unknown or unavailable tool.")
    if not isinstance(arguments, dict):
        return _error("Tool arguments must be a JSON object.")
    try:
        return handler(context, arguments)
    except (TypeError, ValueError) as exc:
        return _error(str(exc))
    except Exception:
        logger.exception(
            "Agent tool failed",
            extra={"tool_name": name, "user_id": context.user_id},
        )
        return _error("The requested analysis could not be completed.")
