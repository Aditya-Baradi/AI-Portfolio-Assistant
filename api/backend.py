# backend.py (inside api/)

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from api import db
from api.langchain_agent import run_portfolio_agent
from api.portfolio_core import parse_portfolio_file, holdings_info, SESSION_PORTFOLIOS

app = FastAPI()
db.init_db()

# Allow your frontend (index.html) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated charts (equity curves etc.) for the chat UI.
CHARTS_DIR = Path("charts")
CHARTS_DIR.mkdir(exist_ok=True)
app.mount("/charts", StaticFiles(directory=str(CHARTS_DIR)), name="charts")

# Serve the frontend from the backend itself (http://127.0.0.1:8000).
# Don't use VS Code Live Server for this app: it reloads the page whenever a
# workspace file changes, and the backend writes files (app.db, chat_memory/,
# cache/) on every request — causing an endless reload/flicker loop.
INDEX_FILE = Path(__file__).resolve().parent / "index.html"


@app.get("/")
def index_page():
    return FileResponse(INDEX_FILE)

MAX_UPLOAD_BYTES = 2_000_000
MAX_HOLDINGS = 500


def _current_user(authorization: str | None) -> dict:
    """Resolve 'Authorization: Bearer <token>' to a user or raise 401."""
    token = ""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    user = db.user_for_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not signed in.")
    return user


def _agent_session_id(user_id: int, chat_id: str) -> str:
    """Chat memory is per (user, chat); the portfolio is shared per user."""
    return f"u{user_id}.c{chat_id}"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

class Credentials(BaseModel):
    email: str
    password: str


# Brute-force protection: sliding-window limiter per client IP. In-memory is
# fine for a single-process app; swap for Redis if this ever runs on >1 worker.
AUTH_RATE_LIMIT = 10          # attempts allowed...
AUTH_RATE_WINDOW = 300.0      # ...per this many seconds, per IP
_AUTH_ATTEMPTS: dict = {}


def _check_auth_rate(request: Request):
    import time

    ip = request.client.host if request.client else "unknown"
    now = time.time()
    recent = [t for t in _AUTH_ATTEMPTS.get(ip, []) if now - t < AUTH_RATE_WINDOW]
    if len(recent) >= AUTH_RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Please wait a few minutes and try again.",
        )
    recent.append(now)
    _AUTH_ATTEMPTS[ip] = recent
    if len(_AUTH_ATTEMPTS) > 10_000:  # don't let the map grow unbounded
        _AUTH_ATTEMPTS.clear()


@app.post("/auth/register")
def register(creds: Credentials, request: Request):
    _check_auth_rate(request)
    try:
        user_id = db.register_user(creds.email, creds.password)
    except db.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = db.issue_token(user_id)
    return {"token": token, "email": creds.email.strip().lower()}


@app.post("/auth/login")
def login(creds: Credentials, request: Request):
    _check_auth_rate(request)
    try:
        user_id = db.verify_login(creds.email, creds.password)
    except db.AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))
    token = db.issue_token(user_id)
    return {"token": token, "email": creds.email.strip().lower()}


@app.post("/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if authorization and authorization.lower().startswith("bearer "):
        db.revoke_token(authorization[7:].strip())
    return {"ok": True}


@app.get("/me")
def me(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"email": user["email"], "has_portfolio": db.get_portfolio(user["id"]) is not None}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str
    chat_id: str


@app.post("/chat")
def chat(req: ChatRequest, authorization: str | None = Header(default=None)):
    """
    Declared sync so FastAPI runs it in a worker thread — a slow agent call
    (e.g. FinRL optimization) doesn't block the event loop for other users.
    """
    user = _current_user(authorization)
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message.")
    if not req.chat_id:
        raise HTTPException(status_code=422, detail="Missing chat_id.")

    sid = _agent_session_id(user["id"], req.chat_id)

    # Make the user's stored portfolio visible to the agent's tools under
    # this chat's session id.
    pf = db.get_portfolio(user["id"])
    if pf:
        SESSION_PORTFOLIOS[sid] = pf

    db.touch_chat(user["id"], req.chat_id, first_message=req.message)
    db.add_message(req.chat_id, "user", req.message)

    answer = run_portfolio_agent(req.message, session_id=sid)

    db.add_message(req.chat_id, "assistant", answer)
    return {"answer": answer}


@app.get("/chats")
def chats(authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    return {"chats": db.list_chats(user["id"])}


@app.get("/chats/{chat_id}/messages")
def chat_messages(chat_id: str, authorization: str | None = Header(default=None)):
    user = _current_user(authorization)
    if not db.chat_belongs_to_user(user["id"], chat_id):
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"messages": db.get_messages(chat_id)}


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """
    Import (or re-import to update) the signed-in user's portfolio from a
    CSV/JSON file. Stored in SQL so it persists across sessions and restarts.
    """
    user = _current_user(authorization)
    content = await file.read()

    if len(content) > MAX_UPLOAD_BYTES:
        return {"message": f"File too large ({len(content)} bytes). Limit is {MAX_UPLOAD_BYTES // 1_000_000} MB."}

    from api.portfolio_core import extract_raw_rows, looks_like_transactions, replay_transactions

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
        return {"message": f"Too many holdings ({n}). Limit is {MAX_HOLDINGS}."}

    had_one = db.get_portfolio(user["id"]) is not None
    db.save_portfolio(user["id"], parsed)

    verb = "Updated" if had_one else "Imported"
    return {"message": f"{verb} your portfolio: {n} holdings."}


def _live_prices(tickers: list) -> dict:
    """Latest market price per ticker (cached download); empty dict on failure."""
    try:
        from datetime import datetime, timedelta

        from api.portfolio_core import _download_adj_close_matrix

        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(tickers, start, end).ffill()
        last = prices.iloc[-1]
        return {t: float(last[t]) for t in prices.columns if last[t] == last[t]}
    except Exception:
        return {}


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
        prices = _live_prices(list(positions.keys()))
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
                price = _live_prices([t]).get(t)
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


class HoldingUpdate(BaseModel):
    ticker: str
    shares: float
    avg_cost: float | None = None


def _valid_ticker(t: str) -> str:
    t = (t or "").strip().upper()
    if not t or len(t) > 10 or not all(c.isalnum() or c in ".-" for c in t):
        raise HTTPException(status_code=422, detail="Invalid ticker symbol.")
    return t


@app.post("/portfolio/holding")
def upsert_holding(update: HoldingUpdate, authorization: str | None = Header(default=None)):
    """
    Add a stock to the portfolio or change an existing position's share count
    (and optionally its average cost). New tickers are validated against
    yfinance so typos don't pollute the portfolio.
    """
    user = _current_user(authorization)
    ticker = _valid_ticker(update.ticker)
    if update.shares <= 0:
        raise HTTPException(status_code=422, detail="Shares must be a positive number.")
    if update.avg_cost is not None and update.avg_cost <= 0:
        raise HTTPException(status_code=422, detail="Average cost must be positive.")

    pf = db.get_portfolio(user["id"]) or {"holdings": []}
    if not isinstance(pf, dict):
        pf = {"holdings": pf}
    holdings = pf.get("holdings", [])
    existing = [h for h in holdings if isinstance(h, dict)
                and str(h.get("ticker") or h.get("symbol") or h.get("tic") or "").strip().upper() == ticker]

    price = _live_prices([ticker]).get(ticker)
    if not price and not existing:
        raise HTTPException(status_code=422, detail=f"No price data found for '{ticker}'. Is the symbol right?")

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
        raise HTTPException(status_code=422, detail=f"Portfolio is at the {MAX_HOLDINGS}-holding limit.")
    db.save_portfolio(user["id"], pf)
    return {"ok": True, "ticker": ticker, "action": "updated" if existing else "added"}


@app.delete("/portfolio/holding/{ticker}")
def delete_holding(ticker: str, authorization: str | None = Header(default=None)):
    """Remove a stock from the portfolio entirely."""
    user = _current_user(authorization)
    ticker = _valid_ticker(ticker)
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


def _weights_from_pf(pf) -> dict:
    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        if tkr:
            try:
                weights[str(tkr).upper()] = weights.get(str(tkr).upper(), 0.0) + float(val)
            except Exception:
                continue
    return {t: v for t, v in weights.items() if v > 0}


@app.get("/portfolio/sentiment")
def portfolio_sentiment(authorization: str | None = Header(default=None)):
    """
    Per-holding news sentiment for the signed-in user: each stock gets the
    average sentiment score of its recent financial-news headlines plus a
    Buy / Hold / Sell signal derived from that score. Values shown are live
    (shares x current yfinance price) when price data is available.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    from api.sentiment import cached_ticker_sentiment, sentiment_backend, signal_from_score

    info = holdings_info(pf)
    if not info:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    # Live value per holding; fall back to the imported snapshot value.
    values = {t: meta["stored_value"] for t, meta in info.items()}
    try:
        import pandas as pd

        from api.data_cache import cached_splits, split_factor
        from api.portfolio_core import _download_adj_close_matrix

        end = datetime.today().strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(list(info.keys()), "2020-01-01", end)
        last = prices.ffill().iloc[-1]
        splits = cached_splits(list(info.keys()))
        for t, meta in info.items():
            shares = meta["shares"]
            if meta["purchase_date"] is not None and shares:
                shares *= split_factor(splits.get(t, []), meta["purchase_date"].strftime("%Y-%m-%d"))
            cur = float(last[t]) if t in last.index and not pd.isna(last[t]) else None
            if shares and cur:
                values[t] = shares * cur
    except Exception:
        pass

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
            "score": res["avg_score"],
            "label": res["label"],
            "n_headlines": res["n_headlines"],
            "signal": signal_from_score(res["avg_score"], res["n_headlines"]),
        }
        for t, res in zip(tickers, results)
    ]
    rows.sort(key=lambda r: -(r["value"] or 0))
    return {
        "as_of": datetime.today().strftime("%Y-%m-%d"),
        "backend": sentiment_backend(),
        "total_value": round(sum(r["value"] or 0 for r in rows), 2),
        "holdings": rows,
    }


@app.get("/portfolio/whatif")
def portfolio_whatif(authorization: str | None = Header(default=None)):
    """
    "What if I had held the suggested mix instead?" comparison.

    The suggested mix comes from the conservative optimizer (min-vol + cap,
    blended toward equal weight) fitted on data from 3 years ago to 1 year
    ago, then BOTH portfolios are compared over the most recent year the
    optimizer never saw — an out-of-sample test, not a curve fit.
    """
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    weights = _weights_from_pf(pf)
    if len(weights) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 holdings to compare allocations.")

    from datetime import datetime, timedelta

    from api.portfolio_core import backtest_vs_benchmark
    from api.predict_agent import run_conservative_optimization

    today = datetime.today()
    test_start = (today - timedelta(days=365)).strftime("%Y-%m-%d")
    train_start = (today - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    end = today.strftime("%Y-%m-%d")

    try:
        opt_w, _prices, method = run_conservative_optimization(
            list(weights.keys()), train_start, test_start
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not optimize: {e}")

    try:
        cur = backtest_vs_benchmark(weights, test_start, end, return_curves=True)
        opt = backtest_vs_benchmark(opt_w, test_start, end, return_curves=True)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backtest failed: {e}")

    # Align the three curves on shared dates.
    cur_map = dict(zip(cur["curves"]["dates"], cur["curves"]["portfolio"]))
    opt_map = dict(zip(opt["curves"]["dates"], opt["curves"]["portfolio"]))
    ben_map = dict(zip(cur["curves"]["dates"], cur["curves"]["benchmark"]))
    dates = [d for d in cur["curves"]["dates"] if d in opt_map]

    return {
        "start": test_start,
        "end": end,
        "method": method,
        "dates": dates,
        "curves": {
            "current": [cur_map[d] for d in dates],
            "optimized": [opt_map[d] for d in dates],
            "benchmark": [ben_map[d] for d in dates],
        },
        "returns": {
            "current": cur["portfolio"]["total_return"],
            "optimized": opt["portfolio"]["total_return"],
            "benchmark": cur["benchmark_metrics"]["total_return"],
        },
        "optimized_weights": dict(sorted(
            ((t, round(w, 4)) for t, w in opt_w.items()), key=lambda kv: -kv[1]
        )),
    }


def _stock_projection(col, ticker: str) -> dict:
    """12-month GBM projection for one ticker from ~3 years of daily closes."""
    import math

    from api.portfolio_core import _perf_metrics, RF_ANNUAL_DEFAULT

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
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "stats": {"est_annual_return": round(mu, 4), "volatility": round(sigma, 4)},
    }


@app.get("/stocks/projections")
def stocks_projections(authorization: str | None = Header(default=None)):
    """Per-stock 12-month price projections for every holding, biggest first."""
    user = _current_user(authorization)
    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")
    weights = _weights_from_pf(pf)
    if not weights:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix(list(weights.keys()), start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")

    out = []
    for t in sorted(weights, key=lambda k: -weights[k]):
        if t in prices.columns:
            try:
                out.append(_stock_projection(prices[t], t))
            except Exception:
                continue  # not enough history for this one; skip it
    return {"as_of": end, "stocks": out}


@app.get("/stocks/projection/{ticker}")
def stock_projection(ticker: str, authorization: str | None = Header(default=None)):
    """12-month price projection for ANY ticker (the Stocks-page search)."""
    _current_user(authorization)
    ticker = _valid_ticker(ticker)

    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix

    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")
    try:
        prices = _download_adj_close_matrix([ticker], start, end)
    except Exception:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker}'. Is the symbol right?")
    if ticker not in prices.columns:
        raise HTTPException(status_code=404, detail=f"No price data found for '{ticker}'. Is the symbol right?")
    try:
        return _stock_projection(prices[ticker], ticker)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/portfolio/projection")
def portfolio_projection(authorization: str | None = Header(default=None)):
    """
    Estimated 12-month projection for the signed-in user's portfolio.

    Uses 3 years of (cached) price history to estimate the portfolio's
    annualized return and volatility, then projects the current value forward
    with a median path and +/-1 sigma bands (lognormal/GBM approximation).
    Educational estimate, not a forecast guarantee.
    """
    user = _current_user(authorization)

    pf = db.get_portfolio(user["id"])
    if not pf:
        raise HTTPException(status_code=404, detail="No portfolio imported yet.")

    import math
    from datetime import datetime, timedelta

    import numpy as np

    from api.portfolio_core import _download_adj_close_matrix, _perf_metrics, RF_ANNUAL_DEFAULT

    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    weights: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        val = h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0
        if tkr:
            try:
                weights[str(tkr).upper()] = weights.get(str(tkr).upper(), 0.0) + float(val)
            except Exception:
                continue
    weights = {t: v for t, v in weights.items() if v > 0}
    if not weights:
        raise HTTPException(status_code=422, detail="Portfolio has no usable holdings.")

    value0 = sum(weights.values())
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365 * 3)).strftime("%Y-%m-%d")

    try:
        prices = _download_adj_close_matrix(list(weights.keys()), start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Price data unavailable: {e}")

    usable = [t for t in weights if t in prices.columns]
    if not usable:
        raise HTTPException(status_code=422, detail="No price history for the portfolio tickers.")
    w = np.array([weights[t] for t in usable])
    w = w / w.sum()
    rets = prices[usable].pct_change().dropna().dot(w)
    m = _perf_metrics(rets, RF_ANNUAL_DEFAULT)

    # Anchor the projection at the LIVE portfolio value (shares x current
    # yfinance price, split-adjusted), not the snapshot stored in the file.
    try:
        import pandas as pd

        from api.data_cache import cached_splits, split_factor

        info = holdings_info(pf)
        splits = cached_splits(list(info.keys()))
        last = prices.ffill().iloc[-1]
        live_total = 0.0
        for t, meta in info.items():
            shares = meta["shares"]
            if meta["purchase_date"] is not None and shares:
                shares *= split_factor(splits.get(t, []), meta["purchase_date"].strftime("%Y-%m-%d"))
            cur = float(last[t]) if t in last.index and not pd.isna(last[t]) else None
            live_total += shares * cur if (shares and cur) else meta["stored_value"]
        if live_total > 0:
            value0 = live_total
    except Exception:
        pass  # fall back to the stored snapshot value

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
        "months": months,
        "median": median,
        "optimistic": optimistic,
        "pessimistic": pessimistic,
        "stats": {
            "est_annual_return": round(mu, 4),
            "volatility": round(sigma, 4),
            "sharpe": round(m["Sharpe"], 2),
            "max_drawdown": round(m["max_drawdown"], 4),
            "history_years": 3,
            "n_holdings": len(usable),
        },
    }
