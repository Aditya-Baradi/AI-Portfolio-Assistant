# portfolio_core.py
from typing import Dict, List
import io
import json
from datetime import datetime, timedelta
import ast
import numpy as np
import pandas as pd

try:
    from .data_cache import cached_download, cached_sectors
except ImportError:  # running outside the package
    from api.data_cache import cached_download, cached_sectors

# Legacy process-local helper for offline callers/tests. Production portfolio
# state lives only in the encrypted database.
SESSION_PORTFOLIOS: Dict[str, Dict] = {}


def save_session_portfolio(session_id: str, parsed: Dict) -> None:
    """
    Store a portfolio only in the process-local legacy map.

    Plaintext ``chat_memory/*.portfolio.json`` persistence was intentionally
    removed; authenticated application state belongs in the encrypted DB.
    """
    SESSION_PORTFOLIOS[session_id] = parsed


def get_session_portfolio(session_id: str):
    """
    Fetch a portfolio saved by the legacy process-local helper.

    Authenticated application state is intentionally not inferred from a
    session identifier and is never read from plaintext chat-memory files.
    Production callers must use the encrypted database directly.
    """
    return SESSION_PORTFOLIOS.get(session_id)


def compute_total_value(holdings_json):
    try:
        if isinstance(holdings_json, str):
            try:
                data = json.loads(holdings_json)
            except json.JSONDecodeError:
                data = ast.literal_eval(holdings_json)
        else:
            data = holdings_json
    except Exception as e:
        return {"error": f"Error parsing holdings_json data: {e}"}

    try:
        df = pd.DataFrame(data)
        if "current_dollars" in df.columns:
            total = float(df["current_dollars"].sum())
        elif "total_value" in df.columns:
            total = float(df["total_value"].sum())
        else:
            return {"error": "No 'current_dollars' or 'total_value' column found in holdings."}
    except Exception as e:
        return {"error": f"Error computing total value: {e}"}

    return {"total_value": total}

def load_price_history(tickers, start: str, end: str) -> pd.DataFrame:
    """
    Download daily price history for one or more tickers between start and end (YYYY-MM-DD).
    Uses Adj Close when available.
    """
    try:
        data = cached_download(
            tickers,
            start=start,
            end=end,
            progress=False,
        )
    except Exception as e:
        raise RuntimeError(f"yfinance failed for {tickers}: {e}") from e

    if data is None or data.empty:
        raise RuntimeError(f"No price data returned for {tickers} between {start} and {end}")

    # Select just the close prices, leaving columns = tickers. Modern yfinance
    # defaults to auto_adjust=True, which returns a (field, ticker) MultiIndex
    # with an already-adjusted "Close" and no "Adj Close"; older layouts expose
    # "Adj Close". Handle both, plus the flat single-ticker case.
    if isinstance(data.columns, pd.MultiIndex):
        fields = set(data.columns.get_level_values(0))
        field = "Adj Close" if "Adj Close" in fields else ("Close" if "Close" in fields else None)
        if field is None:
            raise RuntimeError("Downloaded data has no Close/Adj Close prices.")
        data = data[field]
    else:
        field = "Adj Close" if "Adj Close" in data.columns else ("Close" if "Close" in data.columns else None)
        if field is not None:
            data = data[field]
            if isinstance(data, pd.Series):
                tkr = tickers[0] if isinstance(tickers, (list, tuple)) else tickers
                data = data.to_frame(name=str(tkr).upper())

    return data


def _normalize_holding_price_and_value(
    holding: Dict,
    *,
    price: float | None,
    value: float | None,
) -> Dict:
    """
    Given a partially filled holding dict plus optional raw price/value,
    fill in:
      - last_price
      - close
      - current_dollars
    when possible.
    """
    if price is not None:
        price = float(price)
        holding.setdefault("last_price", price)
        holding.setdefault("close", price)

    if value is not None:
        holding.setdefault("current_dollars", float(value))

    if (
        "current_dollars" not in holding
        and holding.get("shares") is not None
        and holding.get("close") is not None
    ):
        holding["current_dollars"] = round(
            float(holding["shares"]) * float(holding["close"]), 2
        )

    return holding

def _parse_holdings(holdings_json):
    if isinstance(holdings_json, str):
        try:
            data = json.loads(holdings_json)
        except json.JSONDecodeError:
            data = ast.literal_eval(holdings_json)
    else:
        data = holdings_json
    return pd.DataFrame(data)


def parse_portfolio_file(filename: str, content: bytes) -> Dict:
    """
    Parse an uploaded portfolio file into a normalized structure:

    {
      "holdings": [
        {"ticker": "AMD", "shares": 0.14452, "last_price": 256.12, "current_dollars": 37.01},
        ...
      ]
    }

    Supports:
    - CSV with columns like Ticker/Symbol and Shares/Quantity/Qty/Volume
    - JSON in several shapes, including lists or dicts with a holdings/portfolio key
    """
    name = filename.lower()

    # ---------- CSV ----------
    if name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError("CSV file is empty.")

        cols = {c.lower(): c for c in df.columns}
        ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("tic")
        shares_col = (
            cols.get("shares")
            or cols.get("quantity")
            or cols.get("qty")
            or cols.get("volume")
        )

        if not ticker_col or not shares_col:
            raise ValueError(
                "CSV must contain 'ticker'/'symbol'/'tic' and "
                "'shares'/'quantity'/'qty'/'volume' columns."
            )

        holdings: List[Dict] = []
        for _, row in df.iterrows():
            raw_ticker = str(row[ticker_col]).strip()
            if not raw_ticker:
                continue
            ticker = raw_ticker.upper()
            shares = float(row[shares_col])

            holding: Dict = {"ticker": ticker, "shares": shares}

            # Candidate price / value columns
            price = None
            for key in ("close", "price", "last_price", "adj_close"):
                if key in cols:
                    price = float(row[cols[key]])
                    break

            value = None
            for key in ("current_dollars", "total_value", "value"):
                if key in cols:
                    value = float(row[cols[key]])
                    break

            holding = _normalize_holding_price_and_value(
                holding, price=price, value=value
            )
            holding["total_value"] = holding.get("current_dollars", holding.get("total_value"))

            for key in ("purchase_date", "buy_date", "date", "acquired"):
                if key in cols and str(row[cols[key]]).strip():
                    holding["purchase_date"] = str(row[cols[key]]).strip()
                    break

            # Explicit cost basis (what the user actually paid per share).
            for key in ("avg_cost", "purchase_price", "cost_basis", "avg_price", "cost_per_share"):
                if key in cols:
                    try:
                        holding["purchase_price"] = float(row[cols[key]])
                        break
                    except Exception:
                        pass
            holdings.append(holding)

        if not holdings:
            raise ValueError("No valid rows found in CSV.")
        return {"holdings": holdings}
    

    # ---------- JSON ----------
    if name.endswith(".json"):
        raw = json.loads(content.decode("utf-8"))

        if isinstance(raw, list):
            holdings_raw = raw
        elif isinstance(raw, dict):
            holdings_raw = None
            for key in ("holdings", "portfolio", "positions", "data"):
                if isinstance(raw.get(key), list):
                    holdings_raw = raw[key]
                    break
            if holdings_raw is None:
                raise ValueError("Could not find holdings list inside JSON.")
        else:
            raise ValueError("Invalid JSON structure: expected list or dict.")

        holdings: List[Dict] = []
        for row in holdings_raw:
            if not isinstance(row, dict):
                continue

            raw_ticker = row.get("ticker") or row.get("symbol") or row.get("tic")
            if not raw_ticker:
                continue
            ticker = str(raw_ticker).strip().upper()

            shares_val = (
                row.get("shares")
                or row.get("quantity")
                or row.get("qty")
                or row.get("volume")
            )

            # Try to infer shares if only value+price are given
            if shares_val is None:
                raw_value = (
                    row.get("current_dollars")
                    or row.get("total_value")
                    or row.get("value")
                )
                raw_price = (
                    row.get("close")
                    or row.get("last_price")
                    or row.get("price")
                    or row.get("adj_close")
                )
                if raw_value is not None and raw_price is not None:
                    try:
                        shares_val = float(raw_value) / float(raw_price)
                    except Exception:
                        shares_val = None

            if shares_val is None:
                continue

            shares = float(shares_val)
            holding: Dict = {"ticker": ticker, "shares": shares}

            price = (
                row.get("close")
                or row.get("last_price")
                or row.get("price")
                or row.get("adj_close")
            )
            value = (
                row.get("current_dollars")
                or row.get("total_value")
                or row.get("value")
            )

            holding = _normalize_holding_price_and_value(
                holding, price=price, value=value
            )
            holding["total_value"] = holding.get("current_dollars", holding.get("total_value"))

            raw_date = (
                row.get("purchase_date") or row.get("buy_date")
                or row.get("date") or row.get("acquired")
            )
            if raw_date:
                holding["purchase_date"] = str(raw_date).strip()

            # Explicit cost basis (what the user actually paid per share).
            raw_cost = (
                row.get("avg_cost") or row.get("purchase_price") or row.get("cost_basis")
                or row.get("avg_price") or row.get("cost_per_share")
            )
            if raw_cost is not None:
                try:
                    holding["purchase_price"] = float(raw_cost)
                except Exception:
                    pass
            holdings.append(holding)

        if not holdings:
            raise ValueError("No valid holdings found in JSON.")
        return {"holdings": holdings}

    raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")


# ---------------------------------------------------------------------------
# Transaction-history imports (brokerage activity exports, not snapshots)
# ---------------------------------------------------------------------------

def extract_raw_rows(filename: str, content: bytes):
    """Raw row dicts from a CSV/JSON upload, before any interpretation."""
    name = filename.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            # object dtype so NaN cells really become None (floats can't hold None)
            return df.astype(object).where(pd.notna(df), None).to_dict("records")
        if name.endswith(".json"):
            raw = json.loads(content.decode("utf-8"))
            if isinstance(raw, dict):
                for key in ("holdings", "portfolio", "positions", "data",
                            "transactions", "activity", "history"):
                    if isinstance(raw.get(key), list):
                        raw = raw[key]
                        break
            if isinstance(raw, list):
                return [r for r in raw if isinstance(r, dict)]
    except Exception:
        return None
    return None


def _txn_fields(row: Dict):
    """Normalize one raw row into (ticker, date, shares, price, amount)."""
    lowered = {str(k).strip().lower(): v for k, v in row.items()}

    def pick(*keys):
        for k in keys:
            v = lowered.get(k)
            if v is not None and str(v).strip() != "":
                return v
        return None

    def num(v):
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    tkr = pick("ticker", "symbol", "tic")
    ticker = str(tkr).strip().upper() if tkr else None
    date = pick("date", "trade_date", "transaction_date", "purchase_date")
    date = str(date).strip()[:10] if date else None
    shares = num(pick("volume", "shares", "quantity", "qty"))
    price = num(pick("close", "price", "last_price"))
    amount = num(pick("current_dollars", "amount", "value", "total_value"))
    return ticker, date, shares, price, amount


def looks_like_transactions(rows) -> bool:
    """
    Heuristic check for activity/transaction exports (vs. holdings snapshots).

    Snapshots list each ticker once, on a single date, with positive values.
    Activity exports have negative amounts (sells) or the same ticker showing
    up across several dates.
    """
    if not rows or len(rows) < 2:
        return False
    tickers, dates, negatives = [], set(), 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        t, d, _shares, _price, amount = _txn_fields(r)
        if not t:
            continue
        tickers.append(t)
        if d:
            dates.add(d)
        if amount is not None and amount < 0:
            negatives += 1
    if not tickers:
        return False
    if negatives:
        return True
    return len(dates) > 1 and len(tickers) > len(set(tickers))


def replay_transactions(rows) -> Dict:
    """
    Replay trades in date order to reconstruct positions and true average cost.

    Row classification:
      - shares + price            -> trade (a negative dollar amount = sell)
      - shares + amount, no price -> trade at |amount| / shares
      - amount only               -> cash event (dividend/interest); ignored
      - shares only               -> unusable (no price); reported as skipped
    Sells reduce the position using the average-cost method.
    """
    events = []
    cash_events = 0
    skipped = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        ticker, date, shares, price, amount = _txn_fields(r)
        if not ticker:
            continue
        if shares and not price and amount:
            price = abs(amount) / shares
        if shares and price and shares > 0 and price > 0:
            side = "sell" if (amount is not None and amount < 0) else "buy"
            events.append((date or "", ticker, side, float(shares), float(price)))
        elif amount is not None:
            cash_events += 1
        else:
            skipped.append(ticker)

    events.sort(key=lambda e: e[0])

    pos: Dict[str, Dict] = {}
    for date, ticker, side, shares, price in events:
        p = pos.setdefault(ticker, {"shares": 0.0, "cost": 0.0, "first_buy": None})
        if side == "buy":
            p["shares"] += shares
            p["cost"] += shares * price
            if p["first_buy"] is None:
                p["first_buy"] = date or None
        else:
            if shares >= p["shares"] - 1e-9:
                p["shares"], p["cost"] = 0.0, 0.0  # position fully closed
            else:
                p["cost"] *= 1.0 - shares / p["shares"]
                p["shares"] -= shares

    positions = {}
    for t, p in pos.items():
        if p["shares"] > 1e-9 and p["cost"] > 0:
            positions[t] = {
                "shares": round(p["shares"], 6),
                "avg_cost": round(p["cost"] / p["shares"], 4),
                "first_buy": p["first_buy"],
            }

    return {
        "positions": positions,
        "n_trades": len(events),
        "n_cash_events": cash_events,
        "skipped_tickers": sorted(set(skipped)),
        "closed_tickers": sorted(t for t, p in pos.items()
                                 if p["shares"] <= 1e-9 and t not in positions),
    }


# ---------------------------------------------------------------------------
# Backtesting / benchmark comparison engine
# ---------------------------------------------------------------------------

ANN = 252.0  # trading days per year

# Annual risk-free rate used in Sharpe ratios (approx. 3-month T-bill).
RF_ANNUAL_DEFAULT = 0.04

# One-way trading cost in basis points per unit of turnover (commission+slippage).
COST_BPS_DEFAULT = 5.0

# GICS sector -> representative SPDR sector ETF (used for sector-level backtests)
SECTOR_ETF_MAP = {
    "Information Technology": "XLK",
    "Technology": "XLK",
    "Financials": "XLF",
    "Financial Services": "XLF",
    "Health Care": "XLV",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Cyclical": "XLY",
    "Consumer Staples": "XLP",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Materials": "XLB",
    "Basic Materials": "XLB",
    "Industrials": "XLI",
}


def _download_adj_close_matrix(tickers, start, end) -> pd.DataFrame:
    """
    Download a clean 2D DataFrame of (adjusted) close prices.

    Rows = trading days, columns = tickers. Robust to the yfinance MultiIndex
    layout and to the single-ticker case. Uses auto_adjust=True so 'Close' is
    already split/dividend adjusted (newer yfinance drops 'Adj Close').
    """
    tickers = list(dict.fromkeys(tickers))  # de-dup, preserve order
    raw = cached_download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if raw is None or raw.empty:
        raise RuntimeError(f"No price data returned for {tickers} between {start} and {end}.")
    raw_attrs = dict(getattr(raw, "attrs", {}) or {})
    semantics = raw_attrs.get("price_semantics")
    if semantics:
        try:
            from api.market_data import TOTAL_RETURN_SEMANTICS
        except ImportError:
            from .market_data import TOTAL_RETURN_SEMANTICS
        if semantics != TOTAL_RETURN_SEMANTICS:
            raise RuntimeError(
                f"Provider returned price semantics {semantics!r}; "
                f"financial analytics require {TOTAL_RETURN_SEMANTICS!r}."
            )

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        field = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else None)
        if field is None:
            raise ValueError("Downloaded data has no Close/Adj Close prices.")
        data = raw[field]
    else:
        # Single ticker -> flat columns; pick the price field and name it.
        field = "Close" if "Close" in raw.columns else ("Adj Close" if "Adj Close" in raw.columns else None)
        if field is None:
            raise ValueError("Downloaded data has no Close/Adj Close prices.")
        data = raw[[field]].copy()
        data.columns = [tickers[0]]

    if isinstance(data, pd.Series):
        data = data.to_frame()

    data = data.dropna(axis=1, how="all").sort_index()
    data.attrs.update(raw_attrs)

    # Repair partial bulk downloads. yfinance occasionally returns full history
    # for some tickers in a batch and almost nothing for the rest. Re-fetch the
    # starved (or entirely missing) tickers one at a time — a single-ticker
    # download has its own cache key and usually comes back clean — then splice
    # them in. Only a clear coverage disparity triggers this, so a batch of
    # uniformly short (legitimately young) histories won't cause needless
    # refetches.
    counts = data.notna().sum()
    best = int(counts.max()) if len(counts) else 0
    if best > 0:
        missing = [t for t in tickers if t not in data.columns]
        starved = [t for t in data.columns if int(counts[t]) < 0.5 * best]
        need = list(dict.fromkeys(missing + starved))

        repaired = {}
        for t in need:
            try:
                one = cached_download(t, start=start, end=end, progress=False, auto_adjust=True)
                if one is None or one.empty:
                    continue
                if one.attrs.get("price_semantics") not in (None, semantics):
                    continue
                if isinstance(one.columns, pd.MultiIndex):
                    lvl0 = set(one.columns.get_level_values(0))
                    f = "Close" if "Close" in lvl0 else ("Adj Close" if "Adj Close" in lvl0 else None)
                    if f is None:
                        continue
                    s = one[f]
                    if getattr(s, "ndim", 1) > 1:
                        s = s.iloc[:, 0]
                else:
                    f = "Close" if "Close" in one.columns else ("Adj Close" if "Adj Close" in one.columns else None)
                    if f is None:
                        continue
                    s = one[f]
                s = s.dropna()
                if len(s) > int(counts.get(t, 0)):
                    repaired[t] = s
            except Exception:
                continue  # this one just stays sparse/absent

        if repaired:
            new_index = data.index
            for s in repaired.values():
                new_index = new_index.union(s.index)
            data = data.reindex(new_index)
            for t, s in repaired.items():
                data[t] = s.reindex(new_index)
            data = data.dropna(axis=1, how="all").sort_index()
            data.attrs.update(raw_attrs)

    missing_after_repair = [ticker for ticker in tickers if ticker not in data.columns]
    data.attrs["requested_tickers"] = tickers
    data.attrs["present_tickers"] = [
        ticker for ticker in tickers if ticker in data.columns
    ]
    data.attrs["missing_tickers"] = missing_after_repair
    data.attrs["complete_coverage"] = not missing_after_repair
    return data


def _perf_metrics(returns: pd.Series, rf_annual: float = 0.0) -> dict:
    """
    Standard performance stats for a daily-return series.

    Sharpe uses the annualized excess return over rf divided by annualized vol.
    """
    returns = returns.dropna()
    n = len(returns)
    if n == 0:
        raise ValueError("Return series is empty; cannot compute metrics.")

    growth = float((1.0 + returns).prod())
    total_return = growth - 1.0
    cagr = growth ** (ANN / n) - 1.0
    vol = float(returns.std() * np.sqrt(ANN))
    # Guard against float noise: a constant series has vol ~1e-19, not 0.
    sharpe = (cagr - rf_annual) / vol if vol > 1e-12 else 0.0

    # Include the initial unit of capital.  Without it, a loss on the first
    # observed day becomes the curve's first "peak" and is incorrectly reported
    # as zero drawdown (e.g. [-50%, +10%] used to produce 0%).
    curve = np.concatenate(([1.0], (1.0 + returns).cumprod().to_numpy(dtype=float)))
    peak = np.maximum.accumulate(curve)
    max_dd = float(np.min(curve / peak - 1.0))

    return {
        "total_return": float(total_return),
        "CAGR": float(cagr),
        "volatility": vol,
        "Sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "n_days": int(n),
    }


def backtest_vs_benchmark(
    weights_dict: dict,
    start: str,
    end: str,
    benchmark: str = "SPY",
    rf_annual: float = RF_ANNUAL_DEFAULT,
    cost_bps: float = COST_BPS_DEFAULT,
    return_curves: bool = False,
    rebalance: str = "daily",
) -> dict:
    """
    Backtest a weighted portfolio against a benchmark (default SPY = S&P 500).

    `rebalance="daily"` resets to target weights each day. `rebalance="none"`
    buys once and lets holdings drift for the full window.
    Trading costs are charged at `cost_bps` (one-way, per unit of turnover):
    a full initial purchase for both portfolio and benchmark, plus the daily
    drift-correction turnover for the portfolio. Sharpe uses `rf_annual`.

    Returns portfolio metrics, benchmark metrics, and comparison stats
    (excess return, beta, annualized alpha, information ratio, and whether
    the portfolio out/under-performed the benchmark). With return_curves=True
    also returns cumulative growth curves for charting.
    """
    if not weights_dict:
        raise ValueError("weights_dict is empty.")
    if rebalance not in {"daily", "none"}:
        raise ValueError("rebalance must be 'daily' or 'none'.")
    try:
        all_weights = np.array([float(value) for value in weights_dict.values()])
    except (TypeError, ValueError) as exc:
        raise ValueError("Weights must be numeric.") from exc
    if not np.isfinite(all_weights).all() or (all_weights < 0).any():
        raise ValueError("Weights must be finite and non-negative.")
    if not np.isfinite(float(cost_bps)) or float(cost_bps) < 0:
        raise ValueError("cost_bps must be finite and non-negative.")

    tickers = list(weights_dict.keys())
    prices = _download_adj_close_matrix(tickers + [benchmark], start, end)

    if benchmark not in prices.columns:
        raise ValueError(f"Benchmark '{benchmark}' returned no price data for {start}..{end}.")

    common = [t for t in tickers if t in prices.columns]
    if not common:
        raise ValueError(
            f"No overlap between weights {tickers} and available price columns {list(prices.columns)}."
        )
    dropped = [t for t in tickers if t not in common]

    # Renormalize weights over the tickers we actually have prices for.
    w = np.array([float(weights_dict[t]) for t in common], dtype=float)
    if w.sum() <= 0:
        raise ValueError("Sum of usable weights is non-positive.")
    w = w / w.sum()

    rets = prices.pct_change().dropna()
    asset_rets = rets[common]
    if rebalance == "daily":
        port_rets = asset_rets.dot(w)
    else:
        positions = (1.0 + asset_rets).cumprod().mul(w, axis=1)
        curve = positions.sum(axis=1)
        port_rets = curve.pct_change()
        port_rets.iloc[0] = float(curve.iloc[0] - 1.0)
    bench_rets = rets[benchmark]

    # Align on shared dates.
    idx = port_rets.index.intersection(bench_rets.index)
    asset_rets = asset_rets.loc[idx]
    port_rets = port_rets.loc[idx]
    bench_rets = bench_rets.loc[idx]

    if cost_bps and cost_bps > 0:
        rate = cost_bps / 10_000.0
        if rebalance == "daily":
            # One-way dollars traded are half the L1 difference between the
            # drifted and target weights.
            drift = (asset_rets + 1.0).mul(w, axis=1)
            drift = drift.div(drift.sum(axis=1), axis=0)
            turnover = 0.5 * (drift - w).abs().sum(axis=1)
            port_rets = (1.0 + port_rets) * (1.0 - turnover * rate) - 1.0
        # Both sides pay the initial full purchase (100% turnover).
        port_rets.iloc[0] = (1.0 + port_rets.iloc[0]) * (1.0 - rate) - 1.0
        bench_rets = bench_rets.copy()
        bench_rets.iloc[0] = (1.0 + bench_rets.iloc[0]) * (1.0 - rate) - 1.0

    port_m = _perf_metrics(port_rets, rf_annual)
    bench_m = _perf_metrics(bench_rets, rf_annual)

    # Beta / alpha (CAPM) and information ratio vs the benchmark.
    cov = np.cov(port_rets.values, bench_rets.values)
    beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else 0.0
    alpha_annual = float(port_m["CAGR"] - (rf_annual + beta * (bench_m["CAGR"] - rf_annual)))

    excess = port_rets - bench_rets
    te = float(excess.std() * np.sqrt(ANN))  # tracking error
    info_ratio = float((excess.mean() * ANN) / te) if te != 0 else 0.0

    result = {
        "start": start,
        "end": end,
        "benchmark": benchmark,
        "rf_annual": rf_annual,
        "cost_bps": cost_bps,
        "rebalance": rebalance,
        "tickers_used": common,
        "tickers_dropped": dropped,
        "weights_used": {t: round(float(x), 6) for t, x in zip(common, w)},
        "portfolio": port_m,
        "benchmark_metrics": bench_m,
        "comparison": {
            "excess_total_return": round(port_m["total_return"] - bench_m["total_return"], 6),
            "excess_CAGR": round(port_m["CAGR"] - bench_m["CAGR"], 6),
            "alpha_annual": round(alpha_annual, 6),
            "beta": round(beta, 4),
            "tracking_error": round(te, 6),
            "information_ratio": round(info_ratio, 4),
            "sharpe_diff": round(port_m["Sharpe"] - bench_m["Sharpe"], 4),
            "outperformed": bool(port_m["total_return"] > bench_m["total_return"]),
        },
    }

    if return_curves:
        port_curve = (1.0 + port_rets).cumprod()
        bench_curve = (1.0 + bench_rets).cumprod()
        result["curves"] = {
            "dates": [d.strftime("%Y-%m-%d") for d in port_curve.index],
            "portfolio": [round(float(v), 6) for v in port_curve.values],
            "benchmark": [round(float(v), 6) for v in bench_curve.values],
        }

    return result


def holdings_info(pf) -> dict:
    """
    Per-ticker holding facts from an imported portfolio: shares, purchase date,
    cost basis, and the file's stored value (fallback only).

    Cost basis preference: an explicit avg_cost/purchase_price/cost_basis field
    (what the user actually paid) wins over the snapshot close/last price.
    """
    holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
    info: dict = {}
    for h in holdings if isinstance(holdings, list) else []:
        if not isinstance(h, dict):
            continue
        tkr = h.get("ticker") or h.get("symbol") or h.get("tic")
        if not tkr:
            continue
        t = str(tkr).upper()
        entry = info.setdefault(
            t, {"shares": 0.0, "stored_value": 0.0, "purchase_date": None,
                "purchase_price": None, "basis_is_cost": False}
        )
        try:
            entry["shares"] += float(h.get("shares") or 0.0)
        except Exception:
            pass
        try:
            entry["stored_value"] += float(h.get("current_dollars") or h.get("total_value") or h.get("value") or 0.0)
        except Exception:
            pass
        if entry["purchase_date"] is None and h.get("purchase_date"):
            try:
                entry["purchase_date"] = pd.Timestamp(str(h["purchase_date"])).tz_localize(None)
            except Exception:
                pass

        explicit = (
            h.get("purchase_price") or h.get("avg_cost")
            or h.get("cost_basis") or h.get("avg_price") or h.get("cost_per_share")
        )
        if explicit is not None and not entry["basis_is_cost"]:
            try:
                entry["purchase_price"] = float(explicit)
                entry["basis_is_cost"] = True
            except Exception:
                pass
        if entry["purchase_price"] is None:
            p = h.get("close") or h.get("last_price") or h.get("price")
            try:
                entry["purchase_price"] = float(p) if p else None
            except Exception:
                pass

    return {t: m for t, m in info.items() if m["shares"] > 0 or m["stored_value"] > 0}


def live_portfolio_valuation(
    pf,
    *,
    price_frame: pd.DataFrame | None = None,
    lookback_days: int = 14,
    max_stale_market_days: int = 3,
) -> dict:
    """
    Canonical current valuation for an imported holdings snapshot.

    Imported ``shares`` are current share counts.  They must never be multiplied
    by splits again merely because a purchase date is present.  Adjusted close
    history already handles corporate actions historically; its latest point is
    anchored to the latest quoted close.

    The return value makes every fallback explicit so callers can choose whether
    stored import-time values are acceptable for their use case.
    """
    info = holdings_info(pf)
    tickers = list(info)
    if not tickers:
        return {
            "values": {},
            "prices": {},
            "shares": {},
            "total_value": 0.0,
            "as_of": None,
            "ticker_as_of": {},
            "fallback_tickers": [],
            "source_provider": None,
            "price_semantics": None,
        }

    if price_frame is None:
        end_dt = datetime.today()
        start_dt = end_dt - timedelta(days=max(7, int(lookback_days)))
        try:
            price_frame = _download_adj_close_matrix(
                tickers,
                start_dt.strftime("%Y-%m-%d"),
                end_dt.strftime("%Y-%m-%d"),
            )
        except Exception:
            price_frame = pd.DataFrame()

    attrs = dict(getattr(price_frame, "attrs", {}) or {})
    global_latest = None
    if price_frame is not None and not price_frame.empty:
        try:
            global_latest = pd.Timestamp(price_frame.index.max()).tz_localize(None)
        except Exception:
            global_latest = None

    prices: dict[str, float] = {}
    values: dict[str, float] = {}
    shares_out: dict[str, float] = {}
    ticker_as_of: dict[str, str | None] = {}
    accepted_dates: list[pd.Timestamp] = []
    fallback: list[str] = []
    for ticker, meta in info.items():
        shares = float(meta.get("shares") or 0.0)
        shares_out[ticker] = shares
        price = None
        observed_at = None
        if (
            price_frame is not None
            and not price_frame.empty
            and ticker in price_frame.columns
        ):
            observed = price_frame[ticker].dropna()
            if not observed.empty:
                observed_at = pd.Timestamp(observed.index[-1]).tz_localize(None)
                candidate = float(observed.iloc[-1])
                market_days_old = (
                    int(np.busday_count(
                        np.datetime64(observed_at.date()),
                        np.datetime64(global_latest.date()),
                    ))
                    if global_latest is not None and observed_at < global_latest
                    else 0
                )
                if candidate > 0 and market_days_old <= max(0, int(max_stale_market_days)):
                    price = candidate
        ticker_as_of[ticker] = (
            observed_at.strftime("%Y-%m-%d") if observed_at is not None else None
        )
        if shares > 0 and price is not None:
            prices[ticker] = price
            values[ticker] = shares * price
            accepted_dates.append(observed_at)
        else:
            stored = float(meta.get("stored_value") or 0.0)
            values[ticker] = max(stored, 0.0)
            fallback.append(ticker)

    # A combined valuation cannot truthfully be newer than its oldest accepted
    # component price.
    as_of = min(accepted_dates).strftime("%Y-%m-%d") if accepted_dates else None

    return {
        "values": values,
        "prices": prices,
        "shares": shares_out,
        "total_value": float(sum(values.values())),
        "as_of": as_of,
        "ticker_as_of": ticker_as_of,
        "fallback_tickers": sorted(fallback),
        "source_provider": attrs.get("source_provider"),
        "price_semantics": attrs.get("price_semantics"),
    }


def buy_and_hold_returns(prices: pd.DataFrame, weights: dict) -> pd.Series:
    """
    Returns of a hypothetical buy-and-hold basket with no intra-period
    rebalancing.  `weights` are normalized at the first common price date.
    """
    usable = [t for t, value in weights.items() if value > 0 and t in prices.columns]
    if not usable:
        raise ValueError("No positive weights have usable price history.")
    aligned = prices[usable].ffill().dropna(how="any")
    if len(aligned) < 2:
        raise ValueError("Not enough common price history.")
    raw_w = np.array([float(weights[t]) for t in usable], dtype=float)
    if not np.isfinite(raw_w).all() or raw_w.sum() <= 0:
        raise ValueError("Weights must be finite and positive.")
    w = raw_w / raw_w.sum()
    normalized = aligned.div(aligned.iloc[0])
    curve = normalized.mul(w, axis=1).sum(axis=1)
    return curve.pct_change().dropna()


def compute_portfolio_metrics(tickers, start, end, weights_json=None, benchmark="SPY"):
    """
    Backtest a portfolio and compare it against a benchmark (S&P 500 via SPY).

    If weights_json is None, an equal-weight portfolio over `tickers` is used.
    Returns per-portfolio metrics plus a benchmark comparison block.
    """
    if weights_json is None:
        if not tickers:
            raise ValueError("tickers list is empty and no weights_json was provided")
        w_equal = 1.0 / len(tickers)
        weights_dict = {t: w_equal for t in tickers}
    else:
        weights_dict = json.loads(weights_json)

    bt = backtest_vs_benchmark(weights_dict, start, end, benchmark=benchmark)

    # Keep the original flat keys for backward compatibility, and attach the
    # full benchmark comparison so callers can show "vs S&P 500".
    port = bt["portfolio"]
    return {
        "tickers_used": bt["tickers_used"],
        "CAGR": port["CAGR"],
        "volatility": port["volatility"],
        "Sharpe": port["Sharpe"],
        "max_drawdown": port["max_drawdown"],
        "total_return": port["total_return"],
        "benchmark": benchmark,
        "benchmark_metrics": bt["benchmark_metrics"],
        "vs_benchmark": bt["comparison"],
    }


def get_sector_map(tickers) -> dict:
    """Map each ticker to its (yfinance-reported) GICS sector, 'Unknown' on failure."""
    return cached_sectors(tickers)


def sector_weights_from_weights(weights: dict, sector_map: dict) -> dict:
    """Aggregate ticker weights into sector weights (summing to 1 over known sectors)."""
    agg = {}
    for t, w in weights.items():
        sec = sector_map.get(t, "Unknown")
        agg[sec] = agg.get(sec, 0.0) + float(w)
    total = sum(agg.values())
    if total > 0:
        agg = {s: v / total for s, v in agg.items()}
    return agg


def backtest_sectors_vs_benchmark(
    weights_dict: dict,
    start: str,
    end: str,
    benchmark: str = "SPY",
    sector_map: dict | None = None,
    rf_annual: float = 0.0,
) -> dict:
    """
    Backtest the portfolio's SECTOR TILT against the S&P 500.

    Each GICS sector is proxied by its SPDR sector ETF (XLK, XLF, ...). The
    agent's ticker weights are collapsed into sector weights, mapped onto the
    sector ETFs, and that sector-weighted basket is backtested vs the benchmark.
    This isolates whether the agent's sector allocation (not stock picking)
    beat the index.
    """
    if not weights_dict:
        raise ValueError("weights_dict is empty.")

    tickers = list(weights_dict.keys())
    if sector_map is None:
        sector_map = get_sector_map(tickers)

    sec_w = sector_weights_from_weights(weights_dict, sector_map)

    # Map sectors -> ETFs, combining any sectors that share an ETF.
    etf_weights: dict = {}
    unmapped = {}
    for sector, w in sec_w.items():
        etf = SECTOR_ETF_MAP.get(sector)
        if etf is None:
            unmapped[sector] = round(w, 6)
            continue
        etf_weights[etf] = etf_weights.get(etf, 0.0) + w

    if not etf_weights:
        raise ValueError(f"No sectors could be mapped to sector ETFs (sectors seen: {list(sec_w)}).")

    # Renormalize over the mapped ETFs so the basket sums to 1.
    tot = sum(etf_weights.values())
    etf_weights = {e: w / tot for e, w in etf_weights.items()}

    bt = backtest_vs_benchmark(etf_weights, start, end, benchmark=benchmark, rf_annual=rf_annual)
    bt["sector_weights"] = {s: round(v, 6) for s, v in sorted(sec_w.items(), key=lambda kv: -kv[1])}
    bt["sector_etf_weights"] = {e: round(v, 6) for e, v in sorted(etf_weights.items(), key=lambda kv: -kv[1])}
    bt["unmapped_sectors"] = unmapped
    return bt


def weights_from_holdings_json(holdings_json: str) -> dict:
    df = _parse_holdings(holdings_json)

    if "ticker" in df.columns:
        sym_col = "ticker"
    elif "tic" in df.columns:
        sym_col = "tic"
    else:
        raise ValueError("Holdings JSON must contain 'ticker' or 'tic' column.")

    if "current_dollars" in df.columns:
        val_col = "current_dollars"
    elif "total_value" in df.columns:
        val_col = "total_value"
    else:
        raise ValueError("Holdings JSON must contain 'current_dollars' or 'total_value' column.")

    total = df[val_col].sum()
    if total <= 0:
        raise ValueError("Total portfolio value must be positive.")

    weights = (df.groupby(sym_col)[val_col].sum() / total).to_dict()
    return weights

def get_portfolio_date(holdings_json: str) -> str:
    data = json.loads(holdings_json)
    return data[0]["date"]

def compute_metrics_from_holdings(holdings_json: str, start: str, end: str, benchmark: str = "SPY"):
    weights = weights_from_holdings_json(holdings_json)
    tickers = list(weights.keys())
    weights_json = json.dumps(weights)
    return compute_portfolio_metrics(
        tickers=tickers,
        start=start,
        end=end,
        weights_json=weights_json,
        benchmark=benchmark,
    )


def recommend_portfolio(
    prices: pd.DataFrame,
    constraints: Dict,
) -> Dict[str, float]:
    """
    Stub for your existing RL/optimization logic – here you’d plug in FinRL,
    RL model, or optimization routine.
    Return a dict of {ticker: weight}.
    """
    cols = list(prices.columns)
    equal_weight = 1.0 / len(cols)
    return {t: equal_weight for t in cols}


def load_price_on_date(ticker: str, date: str) -> float:
    """
    Get the adjusted close price for `ticker` around a specific calendar date.

    - Accepts dates in either YYYY-MM-DD or MM/DD/YYYY format.
    - If the exact date is a weekend/holiday, uses the next trading day.
    - If there is still no data after that, falls back to the nearest trading day
      within a +/- 30 day window.
    """
    # Parse date flexibly
    try:
        dt = datetime.fromisoformat(date).date()
    except ValueError:
        try:
            month, day, year = map(int, date.replace("-", "/").split("/"))
            dt = datetime(year, month, day).date()
        except Exception as e:
            raise ValueError(
                f"Could not parse date '{date}'. Use YYYY-MM-DD or MM/DD/YYYY."
            ) from e

    start = dt - timedelta(days=5)
    end = dt + timedelta(days=30)

    data = cached_download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        progress=False,
    )

    if data is None or data.empty:
        raise RuntimeError(
            f"No price data for {ticker} between {start} and {end} "
            f"(ticker may be invalid or data source unavailable)."
        )

    if "Adj Close" in data.columns:
        series = data["Adj Close"]
    else:
        series = data["Close"]

    if series.empty:
        raise RuntimeError(
            f"No price series available for {ticker} between {start} and {end}."
        )

    # Prefer first trading day ON or AFTER dt…
    after = series[series.index.date >= dt]
    if not after.empty:
        return float(after.iloc[0])

    # …otherwise last trading day BEFORE dt
    before = series[series.index.date <= dt]
    if not before.empty:
        return float(before.iloc[-1])

    raise RuntimeError(
        f"No trading days near {date} for {ticker} within +/-30 days."
    )
