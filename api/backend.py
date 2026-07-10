# backend.py (inside api/)

from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Header, HTTPException
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


@app.post("/auth/register")
def register(creds: Credentials):
    try:
        user_id = db.register_user(creds.email, creds.password)
    except db.AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = db.issue_token(user_id)
    return {"token": token, "email": creds.email.strip().lower()}


@app.post("/auth/login")
def login(creds: Credentials):
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
