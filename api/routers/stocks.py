"""Stocks page: watchlist, per-ticker projections, and headline drill-down."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from api import db
from api.deps import check_api_rate, valid_ticker, verified_user
from api.observability import report
from api.portfolio_core import holdings_info, live_portfolio_valuation

logger = logging.getLogger("evergreen.stocks")
router = APIRouter(tags=["stocks"])

MAX_WATCHLIST_TICKERS = 50
MAX_SCENARIO_TICKERS = 25

PROJECTION_DISCLAIMER = (
    "Mechanical extrapolation of the last 3 years of total-return history under "
    "a geometric-Brownian-motion model. Values after month zero are total-return "
    "equivalents with dividends notionally reinvested, not quoted share-price "
    "targets. The model assumes stable independent returns, omits fat tails and "
    "regime shifts, and is not a forecast or recommendation."
)


@router.get("/watchlist")
def get_watchlist(authorization: str | None = Header(default=None)):
    user = verified_user(authorization)
    tickers = db.list_watchlist(user["id"])
    return {
        "tickers": tickers[:MAX_WATCHLIST_TICKERS],
        "omitted_count": max(0, len(tickers) - MAX_WATCHLIST_TICKERS),
        "limit": MAX_WATCHLIST_TICKERS,
    }


@router.post("/watchlist/{ticker}")
def watch(ticker: str, authorization: str | None = Header(default=None)):
    user = verified_user(authorization)
    ticker = valid_ticker(ticker)
    current = db.list_watchlist(user["id"])
    if ticker not in current and len(current) >= MAX_WATCHLIST_TICKERS:
        raise HTTPException(
            status_code=422,
            detail=f"Watchlist limit is {MAX_WATCHLIST_TICKERS} tickers.",
        )
    db.add_watch(user["id"], ticker)
    return {"ok": True}


@router.delete("/watchlist/{ticker}")
def unwatch(ticker: str, authorization: str | None = Header(default=None)):
    user = verified_user(authorization)
    db.remove_watch(user["id"], valid_ticker(ticker))
    return {"ok": True}


@router.get("/stocks/news/{ticker}")
def stock_news(ticker: str, request: Request,
               authorization: str | None = Header(default=None)):
    """
    Recent scored headlines for one ticker (the drill-down under a holding).
    Served from the same cached fetch that produced the row's score, so the
    headlines always match the number shown.
    """
    user = verified_user(authorization)
    check_api_rate(request, "news")
    ticker = valid_ticker(ticker)

    from api.sentiment import cached_ticker_sentiment

    try:
        res = cached_ticker_sentiment(ticker)
    except Exception as e:
        report(logger, "Stock news lookup failed", e, user_id=user["id"], ticker=ticker)
        raise HTTPException(
            status_code=502,
            detail="Recent headline data is temporarily unavailable.",
        )
    return {
        "ticker": ticker,
        "headlines": [
            {"title": h["title"], "publisher": h["publisher"], "url": h["url"],
             "score": h["score"], "label": h["label"]}
            for h in res.get("headlines", [])
        ],
        "disclaimer": ("Sentiment scores describe the tone of the headline text only. "
                       "They are not analyst ratings and not a recommendation."),
    }


def _stock_projection(col, ticker: str) -> dict:
    """12-month GBM scenario from ~3 years of total-return-adjusted closes."""
    import math

    from api.portfolio_core import RF_ANNUAL_DEFAULT, _perf_metrics

    col = col.dropna()
    if len(col) < 60:
        raise ValueError("Not enough price history to estimate a range.")
    m = _perf_metrics(col.pct_change().dropna(), RF_ANNUAL_DEFAULT)
    price0 = float(col.iloc[-1])
    mu, sigma = m["CAGR"], m["volatility"]

    months = list(range(0, 13))
    median, optimistic, pessimistic = [], [], []
    for mo in months:
        t = mo / 12.0
        med = price0 * (1.0 + mu) ** t
        band = sigma * math.sqrt(t)
        median.append(round(med, 2))
        optimistic.append(round(med * math.exp(band), 2))
        pessimistic.append(round(med * math.exp(-band), 2))

    return {
        "ticker": ticker,
        "price": round(price0, 2),
        "price_label": "latest total-return-series value at month zero",
        "as_of": col.index[-1].strftime("%Y-%m-%d"),
        "series_semantics": "total-return-equivalent value with dividends reinvested",
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "median_label": "mechanical central path, not a price target",
        "band_label": "one-model-standard-deviation range, not confidence bounds",
        "stats": {"est_annual_return": round(mu, 4), "volatility": round(sigma, 4)},
    }


@router.get("/stocks/projections")
def stocks_projections(request: Request, authorization: str | None = Header(default=None)):
    """
    Per-stock 12-month projections: watched tickers first, then every holding
    (biggest first). Works with only a watchlist, only a portfolio, or both.
    """
    user = verified_user(authorization)
    check_api_rate(request, "projections")
    pf = db.get_portfolio(user["id"])
    info = holdings_info(pf) if pf else {}
    # Stored values are used only to choose a bounded processing order. Actual
    # displayed valuations are recomputed from the same fetched price frame.
    weights = {
        ticker: max(float(meta["stored_value"]), 0.0)
        for ticker, meta in info.items()
    }
    watched = db.list_watchlist(user["id"])
    if not weights and not watched:
        raise HTTPException(status_code=404, detail="No portfolio or watchlist yet.")

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    requested_tickers = list(
        dict.fromkeys(watched + sorted(weights, key=lambda k: -weights[k]))
    )
    tickers = requested_tickers[:MAX_SCENARIO_TICKERS]
    omitted = requested_tickers[MAX_SCENARIO_TICKERS:]
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix(tickers, start, end)
    except Exception as e:
        report(logger, "Stock projections price lookup failed", e, user_id=user["id"])
        raise HTTPException(
            status_code=502,
            detail="Market data is temporarily unavailable for stock scenarios.",
        )
    selected = set(tickers)
    raw_holdings = pf.get("holdings", []) if isinstance(pf, dict) else (pf or [])
    selected_pf = {
        "holdings": [
            row for row in raw_holdings
            if isinstance(row, dict)
            and str(row.get("ticker") or row.get("symbol") or row.get("tic") or "").upper()
            in selected
        ]
    }
    valuation = live_portfolio_valuation(selected_pf, price_frame=prices)

    out = []
    for t in tickers:
        if t in prices.columns:
            try:
                proj = _stock_projection(prices[t], t)
                proj["watched"] = t in watched
                out.append(proj)
            except Exception:
                continue  # not enough history for this one; skip it
    as_of = prices.index.max().strftime("%Y-%m-%d") if not prices.empty else None
    return {
        "as_of": as_of,
        "source_provider": prices.attrs.get("source_provider"),
        "price_semantics": prices.attrs.get("price_semantics"),
        "valuation_fallback_tickers": valuation["fallback_tickers"],
        "stocks": out,
        "watchlist": watched[:MAX_WATCHLIST_TICKERS],
        "coverage": {
            "requested_count": len(requested_tickers),
            "processed_count": len(tickers),
            "omitted_count": len(omitted),
            "omitted_tickers": omitted[:100],
            "batch_limit": MAX_SCENARIO_TICKERS,
        },
        "disclaimer": PROJECTION_DISCLAIMER,
    }


@router.get("/stocks/projection/{ticker}")
def stock_projection(ticker: str, request: Request,
                     authorization: str | None = Header(default=None)):
    """12-month price projection for ANY ticker (the Stocks-page search)."""
    user = verified_user(authorization)
    check_api_rate(request, "projection1")
    ticker = valid_ticker(ticker)

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix([ticker], start, end)
    except Exception as e:
        report(logger, "Single-stock scenario price lookup failed", e,
               user_id=user["id"], ticker=ticker)
        raise HTTPException(status_code=404,
                            detail=f"No usable market data was found for '{ticker}'.")
    if ticker not in prices.columns:
        raise HTTPException(status_code=404,
                            detail=f"No price data found for '{ticker}'. Is the symbol right?")
    try:
        out = _stock_projection(prices[ticker], ticker)
    except ValueError as e:
        report(logger, "Single-stock scenario rejected", e,
               user_id=user["id"], ticker=ticker)
        raise HTTPException(
            status_code=422,
            detail="There is not enough usable history to build this scenario.",
        )
    out["disclaimer"] = PROJECTION_DISCLAIMER
    out["source_provider"] = prices.attrs.get("source_provider")
    out["price_semantics"] = prices.attrs.get("price_semantics")
    return out
