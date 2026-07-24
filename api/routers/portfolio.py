"""
Portfolio import and analytics: upload, holdings CRUD, sentiment, diversification,
what-if, rebalance, history and the 12-month projection.

Everything here requires a VERIFIED account: these endpoints store real
financial data and drive expensive market-data fetches.
"""
from __future__ import annotations

import logging
import math

from fastapi import APIRouter, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from api import db
from api.deps import (
    MAX_HOLDINGS,
    MAX_UPLOAD_BYTES,
    check_api_rate,
    client_ip,
    live_prices,
    valid_ticker,
    verified_user,
)
from api.observability import report
from api.portfolio_core import (
    buy_and_hold_returns,
    holdings_info,
    live_portfolio_valuation,
    parse_portfolio_file,
)

logger = logging.getLogger("evergreen.portfolio")
router = APIRouter(tags=["portfolio"])
MAX_PUBLIC_ANALYTIC_TICKERS = 25
MAX_SHARES_PER_HOLDING = 1_000_000_000_000.0
MAX_PRICE_OR_VALUE = 1_000_000_000_000_000.0


class HoldingUpdate(BaseModel):
    ticker: str
    shares: float = Field(gt=0, le=MAX_SHARES_PER_HOLDING, allow_inf_nan=False)
    avg_cost: float | None = Field(
        default=None, gt=0, le=MAX_PRICE_OR_VALUE, allow_inf_nan=False
    )


def _enforce_analytic_ticker_bound(info: dict) -> None:
    if len(info) > MAX_PUBLIC_ANALYTIC_TICKERS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Public analytics support at most "
                f"{MAX_PUBLIC_ANALYTIC_TICKERS} holdings per request."
            ),
        )


@router.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """
    Import (or re-import to update) the signed-in user's portfolio from a
    CSV/JSON file. Stored in SQL so it persists across sessions and restarts.
    """
    user = verified_user(authorization)

    limit_mb = MAX_UPLOAD_BYTES // 1_000_000
    # Reject an oversized upload up front via Content-Length, before reading it
    # into memory (a client can omit/understate this, so we also cap the read).
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {limit_mb} MB.")

    # Read at most the limit + 1 byte; if we get that extra byte, it's too big.
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Limit is {limit_mb} MB.")

    from api.portfolio_core import extract_raw_rows, looks_like_transactions

    # Brokerage ACTIVITY exports (a list of buys/sells/dividends) need to be
    # replayed, not read as holdings. They're the one file type that contains
    # what the user actually paid, so use them to recover the true cost basis.
    rows = extract_raw_rows(file.filename, content)
    if rows and looks_like_transactions(rows):
        return _import_transactions(user, rows)

    try:
        parsed = parse_portfolio_file(file.filename, content)
    except ValueError as e:
        return {"message": f"Error parsing portfolio file: {e}"}
    except Exception:
        return {"message": "Uploaded file is not valid CSV/JSON portfolio."}

    holdings = parsed.get("holdings", []) if isinstance(parsed, dict) else parsed
    n = len(holdings) if isinstance(holdings, list) else 0
    if n > MAX_HOLDINGS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many holdings ({n}). Limit is {MAX_HOLDINGS}.",
        )
    for holding in holdings if isinstance(holdings, list) else []:
        try:
            shares = float(holding.get("shares") or 0.0)
            value = float(
                holding.get("current_dollars")
                or holding.get("total_value")
                or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            raise HTTPException(status_code=422, detail="A holding contains invalid numbers.")
        if (
            not math.isfinite(shares)
            or shares <= 0
            or shares > MAX_SHARES_PER_HOLDING
            or not math.isfinite(value)
            or value < 0
            or value > MAX_PRICE_OR_VALUE
        ):
            raise HTTPException(
                status_code=422,
                detail="Holding shares and values must be finite non-negative numbers.",
            )

    had_one = db.get_portfolio(user["id"]) is not None
    db.save_portfolio(user["id"], parsed)
    db.log_event(user["id"], "portfolio_import", client_ip(request))
    logger.info("Portfolio imported", extra={"user_id": user["id"], "holdings": n})

    verb = "Updated" if had_one else "Imported"
    return {"message": f"{verb} your portfolio: {n} holdings."}


def _import_transactions(user: dict, rows: list) -> dict:
    """
    Merge a transaction history into the user's portfolio.

    With an existing portfolio: trades set the real average cost (and first
    buy date) on matching holdings, and open positions the portfolio doesn't
    have yet are added. Without one: the replayed positions become the
    portfolio. Shares of existing holdings are never changed, since the
    snapshot import is the authority on what the user holds now.
    """
    from api.portfolio_core import replay_transactions

    result = replay_transactions(rows)
    positions = result["positions"]
    notes = []

    pf = db.get_portfolio(user["id"])
    if pf is None:
        if not positions:
            return {"message": (
                "That looks like a transaction history, but no open positions could be "
                "rebuilt from it (only dividends/sells or rows without prices). "
                "Import a holdings file first, then re-upload this one to set your cost basis."
            )}
        prices = live_prices(list(positions.keys()))
        holdings = []
        for t, p in positions.items():
            price = prices.get(t)
            value = round(p["shares"] * price, 2) if price else round(p["shares"] * p["avg_cost"], 2)
            h = {"ticker": t, "shares": p["shares"], "purchase_price": p["avg_cost"],
                 "current_dollars": value}
            if price:
                h["close"] = round(price, 4)
            if p["first_buy"]:
                h["purchase_date"] = p["first_buy"]
            holdings.append(h)
        if len(holdings) > MAX_HOLDINGS:
            raise HTTPException(
                status_code=422,
                detail=f"Transaction history exceeds the {MAX_HOLDINGS}-holding limit.",
            )
        db.save_portfolio(user["id"], {"holdings": holdings})
        notes.append(f"Rebuilt {len(holdings)} positions from {result['n_trades']} trades.")
    else:
        holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
        by_ticker = {}
        for h in holdings if isinstance(holdings, list) else []:
            if isinstance(h, dict):
                tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
                if tkr:
                    by_ticker.setdefault(str(tkr).strip().upper(), []).append(h)

        updated, added = [], []
        for t, p in positions.items():
            if t in by_ticker:
                for h in by_ticker[t]:
                    h["purchase_price"] = p["avg_cost"]
                    if p["first_buy"]:
                        h["purchase_date"] = p["first_buy"]
                updated.append(t)
            else:
                price = live_prices([t]).get(t)
                value = round(p["shares"] * price, 2) if price else round(p["shares"] * p["avg_cost"], 2)
                h = {"ticker": t, "shares": p["shares"], "purchase_price": p["avg_cost"],
                     "current_dollars": value}
                if p["first_buy"]:
                    h["purchase_date"] = p["first_buy"]
                holdings.append(h)
                added.append(t)

        if not updated and not added:
            return {"message": (
                "That transaction file had no trades with usable prices, so nothing changed. "
                f"({result['n_cash_events']} dividend/cash rows were found; those don't affect cost basis.)"
            )}
        if len(holdings) > MAX_HOLDINGS:
            raise HTTPException(
                status_code=422,
                detail=f"Merged portfolio exceeds the {MAX_HOLDINGS}-holding limit.",
            )
        db.save_portfolio(user["id"], pf if isinstance(pf, dict) else {"holdings": holdings})
        if updated:
            notes.append(f"Set real cost basis for {', '.join(sorted(updated))}.")
        if added:
            notes.append(f"Added {', '.join(sorted(added))}.")

    if result["closed_tickers"]:
        notes.append(f"Skipped closed positions: {', '.join(result['closed_tickers'])}.")
    if result["skipped_tickers"]:
        notes.append(f"Some rows for {', '.join(result['skipped_tickers'])} had no price and were ignored.")
    if result["n_cash_events"]:
        notes.append(f"{result['n_cash_events']} dividend/cash rows ignored.")

    return {"message": "Transaction history imported. " + " ".join(notes)}


@router.post("/portfolio/holding")
def upsert_holding(update: HoldingUpdate, authorization: str | None = Header(default=None)):
    """
    Add a stock to the portfolio or change an existing position's share count
    (and optionally its average cost). New tickers are validated against the
    market data provider so typos don't pollute the portfolio.
    """
    user = verified_user(authorization)
    ticker = valid_ticker(update.ticker)
    if (
        not math.isfinite(update.shares)
        or update.shares <= 0
        or update.shares > MAX_SHARES_PER_HOLDING
    ):
        raise HTTPException(status_code=422, detail="Shares must be a positive number.")
    if (
        update.avg_cost is not None
        and (
            not math.isfinite(update.avg_cost)
            or update.avg_cost <= 0
            or update.avg_cost > MAX_PRICE_OR_VALUE
        )
    ):
        raise HTTPException(status_code=422, detail="Average cost must be positive.")

    pf = db.get_portfolio(user["id"]) or {"holdings": []}
    if not isinstance(pf, dict):
        pf = {"holdings": pf}
    holdings = pf.get("holdings", [])
    existing = [h for h in holdings if isinstance(h, dict)
                and str(h.get("ticker") or h.get("symbol") or h.get("tic") or "").strip().upper() == ticker]

    price = live_prices([ticker]).get(ticker)
    if not price and not existing:
        raise HTTPException(status_code=422,
                            detail=f"No price data found for '{ticker}'. Is the symbol right?")

    row = {"ticker": ticker, "shares": float(update.shares)}
    if price:
        row["close"] = round(price, 4)
        row["current_dollars"] = round(update.shares * price, 2)
    if update.avg_cost is not None:
        row["purchase_price"] = float(update.avg_cost)
    elif existing:
        # keep the cost/date the user already had on this position
        old = existing[0]
        if old.get("purchase_price"):
            row["purchase_price"] = old["purchase_price"]
        if old.get("purchase_date"):
            row["purchase_date"] = old["purchase_date"]
        if "current_dollars" not in row and old.get("close"):
            row["close"] = old["close"]
            row["current_dollars"] = round(update.shares * float(old["close"]), 2)
    if "current_dollars" not in row:
        basis = row.get("purchase_price")
        row["current_dollars"] = round(update.shares * basis, 2) if basis else 0.0

    pf["holdings"] = [h for h in holdings if h not in existing] + [row]
    if len(pf["holdings"]) > MAX_HOLDINGS:
        raise HTTPException(status_code=422,
                            detail=f"Portfolio is at the {MAX_HOLDINGS}-holding limit.")
    db.save_portfolio(user["id"], pf)
    return {"ok": True, "ticker": ticker, "action": "updated" if existing else "added"}


@router.delete("/portfolio/holding/{ticker}")
def delete_holding(ticker: str, authorization: str | None = Header(default=None)):
    """Remove a stock from the portfolio entirely."""
    user = verified_user(authorization)
    ticker = valid_ticker(ticker)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    if not isinstance(pf, dict):
        pf = {"holdings": pf}
    holdings = pf.get("holdings", [])
    kept = [h for h in holdings if not (isinstance(h, dict)
            and str(h.get("ticker") or h.get("symbol") or h.get("tic") or "").strip().upper() == ticker)]
    if len(kept) == len(holdings):
        raise HTTPException(status_code=404, detail=f"No holding '{ticker}' in your portfolio.")
    pf["holdings"] = kept
    db.save_portfolio(user["id"], pf)
    return {"ok": True, "ticker": ticker, "action": "removed"}


@router.get("/portfolio/sentiment")
def portfolio_sentiment(request: Request, authorization: str | None = Header(default=None)):
    """
    Per-holding news TONE for the signed-in user: each stock gets the average
    sentiment score of its recent financial-news headlines plus a descriptive
    tone band. This is a summary of press coverage, NOT a trade recommendation.
    Values shown are live (shares x current price) when price data is available.
    """
    user = verified_user(authorization)
    check_api_rate(request, "sentiment")
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    from api.sentiment import cached_ticker_sentiment, sentiment_backend, signal_from_score

    info = holdings_info(pf)
    if not info:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")
    _enforce_analytic_ticker_bound(info)

    # One canonical valuation path. Imported shares are current shares; applying
    # split factors again would double-adjust normal brokerage snapshots.
    valuation = live_portfolio_valuation(pf)
    values = valuation["values"]
    shares_now = valuation["shares"]

    # Trailing-12-month dividend income per holding (best-effort).
    incomes = {}
    try:
        from api.data_cache import cached_dividends_ttm

        divs = cached_dividends_ttm(list(info.keys()))
        for t in info:
            if shares_now.get(t) and divs.get(t):
                incomes[t] = round(shares_now[t] * divs[t], 2)
    except Exception as e:
        report(logger, "Dividend income lookup failed", e, user_id=user["id"])

    # One news fetch per ticker is network-bound: fan out, and lean on the
    # 15-minute sentiment cache so repeat loads are instant.
    def one(t):
        try:
            return cached_ticker_sentiment(t)
        except Exception:
            return {"ticker": t, "avg_score": 0.0, "label": "No data", "n_headlines": 0}

    tickers = list(info.keys())
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(one, tickers))

    rows = [
        {
            "ticker": t,
            "value": round(values.get(t) or 0.0, 2),
            "income": incomes.get(t, 0.0),
            "score": res["avg_score"],
            "label": res["label"],
            "n_headlines": res["n_headlines"],
            "tone": signal_from_score(res["avg_score"], res["n_headlines"]),
        }
        for t, res in zip(tickers, results)
    ]
    rows.sort(key=lambda r: -(r["value"] or 0))
    return {
        "as_of": valuation["as_of"] or datetime.today().strftime("%Y-%m-%d"),
        "valuation_fallback_tickers": valuation["fallback_tickers"],
        "backend": sentiment_backend(),
        "total_value": round(sum(r["value"] or 0 for r in rows), 2),
        "total_income": round(sum(r["income"] for r in rows), 2),
        "holdings": rows,
        "disclaimer": ("News tone summarises recent press coverage. It is not a "
                       "forecast, a rating, or a recommendation to trade."),
    }


@router.get("/portfolio/whatif")
def portfolio_whatif(request: Request, authorization: str | None = Header(default=None)):
    """
    Retrospective comparison against a neutral equal-weight basket.

    Both the imported mix and a predefined equal-weight basket of the same
    securities are compared over the most recent year. Both are modeled as
    buy-once-and-hold baskets, with no periodic rebalancing.
    """
    user = verified_user(authorization)
    check_api_rate(request, "whatif")
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    info = holdings_info(pf)
    _enforce_analytic_ticker_bound(info)
    valuation = live_portfolio_valuation(pf)
    if valuation["fallback_tickers"]:
        raise HTTPException(
            status_code=502,
            detail=(
                "Live valuation is unavailable for: "
                + ", ".join(valuation["fallback_tickers"])
                + ". The comparison was not run with stale import-time values."
            ),
        )
    weights = {t: v for t, v in valuation["values"].items() if v > 0}
    if len(weights) < 2:
        raise HTTPException(status_code=422,
                            detail="Need at least 2 holdings to compare allocations.")

    from datetime import datetime, timedelta

    from api.portfolio_core import backtest_vs_benchmark

    today = datetime.today()
    test_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    # Neutral, predefined comparator: this is not a personalized target.
    comparator_weights = {ticker: 1.0 / len(weights) for ticker in weights}
    method = "predefined equal-weight comparator"

    try:
        cur = backtest_vs_benchmark(
            weights, test_start, end, return_curves=True, rebalance="none"
        )
        opt = backtest_vs_benchmark(
            comparator_weights,
            test_start,
            end,
            return_curves=True,
            rebalance="none",
        )
    except Exception as e:
        report(logger, "Portfolio comparator failed", e, user_id=user["id"])
        raise HTTPException(
            status_code=502,
            detail="The retrospective comparison is temporarily unavailable.",
        )
    dropped = sorted(set(cur["tickers_dropped"]) | set(opt["tickers_dropped"]))
    if dropped:
        report(
            logger,
            "Portfolio comparator rejected incomplete history",
            ValueError("Incomplete ticker coverage"),
            user_id=user["id"],
            tickers=",".join(dropped),
        )
        raise HTTPException(
            status_code=502,
            detail="Complete price history is unavailable for every holding.",
        )

    # Align the three curves on shared dates.
    cur_map = dict(zip(cur["curves"]["dates"], cur["curves"]["portfolio"]))
    opt_map = dict(zip(opt["curves"]["dates"], opt["curves"]["portfolio"]))
    ben_map = dict(zip(cur["curves"]["dates"], cur["curves"]["benchmark"]))
    dates = [d for d in cur["curves"]["dates"] if d in opt_map]

    return {
        "start": test_start,
        "end": end,
        "method": method,
        "holding_model": "buy once and hold; no rebalancing",
        "tickers_dropped": [],
        "dates": dates,
        "curves": {
            "current": [cur_map[d] for d in dates],
            "comparator": [opt_map[d] for d in dates],
            "benchmark": [ben_map[d] for d in dates],
        },
        "returns": {
            "current": cur["portfolio"]["total_return"],
            "comparator": opt["portfolio"]["total_return"],
            "benchmark": cur["benchmark_metrics"]["total_return"],
        },
        "valuation_as_of": valuation["as_of"],
        "disclaimer": (
            "Retrospective comparison with a predefined equal-weight basket of "
            "the same securities. It is not a target allocation, does not account "
            "for taxes or personal circumstances, and does not predict future results."
        ),
    }


@router.get("/portfolio/rebalance")
def portfolio_rebalance(request: Request, authorization: str | None = Header(default=None)):
    """
    Personalized buy/sell lists are intentionally unavailable.

    The application may show neutral historical model comparisons, but it does
    not turn those models into security-specific instructions for a user.
    """
    verified_user(authorization)
    check_api_rate(request, "rebalance")
    raise HTTPException(
        status_code=410,
        detail=(
            "Personalized buy/sell trade lists are disabled. "
            "Use the neutral model-comparison view for education only."
        ),
    )


@router.get("/portfolio/diversification")
def portfolio_diversification(authorization: str | None = Header(default=None)):
    """How the money is spread across stocks and sectors, with concentration warnings."""
    user = verified_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    info = holdings_info(pf)
    _enforce_analytic_ticker_bound(info)
    valuation = live_portfolio_valuation(pf)
    weights = {ticker: value for ticker, value in valuation["values"].items() if value > 0}
    if not weights:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    from api.data_cache import cached_sectors

    total = sum(weights.values())
    stock_w = sorted(((t, v / total) for t, v in weights.items()), key=lambda kv: -kv[1])

    try:
        sectors_raw = cached_sectors(list(weights.keys()))
    except Exception as e:
        report(logger, "Sector lookup failed", e, user_id=user["id"])
        sectors_raw = {}
    sector_w: dict = {}
    for t, v in weights.items():
        s = sectors_raw.get(t) or "Unknown"
        sector_w[s] = sector_w.get(s, 0.0) + v / total
    sector_list = sorted(sector_w.items(), key=lambda kv: -kv[1])

    warnings_out = []
    if stock_w and stock_w[0][1] > 0.25:
        warnings_out.append(
            f"{stock_w[0][0]} is {stock_w[0][1] * 100:.0f}% of your money. If that one "
            f"stock has a bad year, your whole portfolio does too."
        )
    if sector_list and sector_list[0][0] != "Unknown" and sector_list[0][1] > 0.40:
        warnings_out.append(
            f"{sector_list[0][1] * 100:.0f}% of your money is in {sector_list[0][0]}. "
            f"Sectors often fall together."
        )

    return {
        "valuation_as_of": valuation["as_of"],
        "valuation_fallback_tickers": valuation["fallback_tickers"],
        "stocks": [{"ticker": t, "weight": round(w, 4)} for t, w in stock_w],
        "sectors": [{"sector": s, "weight": round(w, 4)} for s, w in sector_list],
        "warnings": warnings_out,
    }


@router.get("/portfolio/history")
def portfolio_history(request: Request, authorization: str | None = Header(default=None)):
    """
    Historical performance is unavailable until a cashflow-aware transaction
    ledger exists. A holdings snapshot plus one purchase date per ticker cannot
    reconstruct deposits, withdrawals, partial trades, taxes, or dividends.
    """
    verified_user(authorization)
    check_api_rate(request, "history")
    raise HTTPException(
        status_code=410,
        detail=(
            "Accurate portfolio history requires a complete cashflow-aware "
            "transaction ledger. A current holdings snapshot is not enough, "
            "so this endpoint is disabled rather than showing a misleading chart."
        ),
    )

@router.get("/portfolio/projection")
def portfolio_projection(request: Request, authorization: str | None = Header(default=None)):
    """
    Estimated 12-month projection for the signed-in user's portfolio.

    Uses 3 years of (cached) price history to estimate the portfolio's
    annualized return and volatility, then projects the current value forward
    with a median path and +/-1 sigma bands (lognormal/GBM approximation).

    This is an extrapolation of past behaviour under a model that assumes
    returns are independent and identically distributed. Real markets are
    neither. It is an illustration of range, not a forecast.
    """
    user = verified_user(authorization)
    check_api_rate(request, "projection")

    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    import math
    from datetime import datetime, timedelta

    from api.portfolio_core import (
        RF_ANNUAL_DEFAULT,
        _download_adj_close_matrix,
        _perf_metrics,
    )

    info = holdings_info(pf)
    if not info:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")
    _enforce_analytic_ticker_bound(info)

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    try:
        prices = _download_adj_close_matrix(list(info.keys()), start, end)
    except Exception as e:
        report(logger, "Portfolio scenario price lookup failed", e, user_id=user["id"])
        raise HTTPException(
            status_code=502,
            detail="Market data is temporarily unavailable for this scenario.",
        )

    valuation = live_portfolio_valuation(pf, price_frame=prices)
    if valuation["fallback_tickers"]:
        raise HTTPException(
            status_code=502,
            detail=(
                "Current total-return prices are unavailable for: "
                + ", ".join(valuation["fallback_tickers"])
                + ". The projection was not anchored to stale imported values."
            ),
        )
    weights = {ticker: value for ticker, value in valuation["values"].items() if value > 0}
    missing = [ticker for ticker in weights if ticker not in prices.columns]
    if missing:
        raise HTTPException(
            status_code=422,
            detail="No price history for: " + ", ".join(sorted(missing)),
        )
    try:
        # Model the imported basket as held between endpoints. A daily dot
        # product would silently assume free daily rebalancing.
        rets = buy_and_hold_returns(prices[list(weights)], weights)
        m = _perf_metrics(rets, RF_ANNUAL_DEFAULT)
    except ValueError as e:
        report(logger, "Portfolio scenario rejected", e, user_id=user["id"])
        raise HTTPException(
            status_code=422,
            detail="There is not enough complete price history for this scenario.",
        )
    value0 = valuation["total_value"]

    mu, sigma = m["CAGR"], m["volatility"]
    months = list(range(0, 13))
    median, optimistic, pessimistic = [], [], []
    for mo in months:
        t = mo / 12.0
        med = value0 * (1.0 + mu) ** t
        band = sigma * math.sqrt(t)
        median.append(round(med, 2))
        optimistic.append(round(med * math.exp(band), 2))
        pessimistic.append(round(med * math.exp(-band), 2))

    return {
        "current_value": round(value0, 2),
        "valuation_as_of": valuation["as_of"],
        "source_provider": valuation["source_provider"],
        "price_semantics": valuation["price_semantics"],
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "median_label": "mechanical central path, not an expected price",
        "band_label": "one-model-standard-deviation range, not confidence bounds",
        "stats": {
            "est_annual_return": round(mu, 4),
            "volatility": round(sigma, 4),
            "sharpe": round(m["Sharpe"], 2),
            "max_drawdown": round(m["max_drawdown"], 4),
            "history_years": 3,
            "n_holdings": len(weights),
        },
        "method": "buy-and-hold historical inputs with geometric-brownian-motion extrapolation",
        "disclaimer": (
            "A mechanical scenario based on the last 3 years of total-return "
            "history, not a forecast or recommendation. It assumes stable, "
            "independent returns; ignores taxes, fees, cashflows, regime changes "
            "and fat tails; and the band is not a best/worst case or confidence interval."
        ),
    }
