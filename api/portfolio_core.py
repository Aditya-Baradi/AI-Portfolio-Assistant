# portfolio_core.py
from typing import Dict, List
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
import ast
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .data_cache import cached_download, cached_sectors
except ImportError:  # running outside the package
    from api.data_cache import cached_download, cached_sectors

# In-memory session -> portfolio map populated by /upload
SESSION_PORTFOLIOS: Dict[str, Dict] = {}

MEMORY_DIR = Path("chat_memory")


def _sanitize_session_id(session_id: str) -> str:
    session_id = (session_id or "anonymous").strip()
    safe = "".join(ch for ch in session_id if ch.isalnum() or ch in ("-", "_", "@", "."))
    return safe or "anonymous"


def save_session_portfolio(session_id: str, parsed: Dict) -> None:
    """Store an uploaded portfolio in memory AND on disk so it survives restarts."""
    SESSION_PORTFOLIOS[session_id] = parsed
    try:
        MEMORY_DIR.mkdir(exist_ok=True)
        path = MEMORY_DIR / f"{_sanitize_session_id(session_id)}.portfolio.json"
        path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass  # persistence is best-effort; the in-memory copy still works


def get_session_portfolio(session_id: str):
    """Fetch a session's portfolio from memory, falling back to disk."""
    pf = SESSION_PORTFOLIOS.get(session_id)
    if pf:
        return pf
    path = MEMORY_DIR / f"{_sanitize_session_id(session_id)}.portfolio.json"
    if path.exists():
        try:
            pf = json.loads(path.read_text(encoding="utf-8"))
            SESSION_PORTFOLIOS[session_id] = pf
            return pf
        except Exception:
            return None
    return None


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

    if "Adj Close" in data.columns:
        data = data["Adj Close"]

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

    curve = (1.0 + returns).cumprod()
    peak = curve.cummax()
    max_dd = float((curve / peak - 1.0).min())

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
) -> dict:
    """
    Backtest a weighted portfolio against a benchmark (default SPY = S&P 500).

    Model: the portfolio is rebalanced daily back to the target weights.
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
    port_rets = asset_rets.dot(w)
    bench_rets = rets[benchmark]

    # Align on shared dates.
    idx = port_rets.index.intersection(bench_rets.index)
    asset_rets = asset_rets.loc[idx]
    port_rets = port_rets.loc[idx]
    bench_rets = bench_rets.loc[idx]

    if cost_bps and cost_bps > 0:
        rate = cost_bps / 10_000.0
        # Daily turnover needed to rebalance drifted weights back to target.
        drift = (asset_rets + 1.0).mul(w, axis=1)
        drift = drift.div(drift.sum(axis=1), axis=0)
        turnover = (drift - w).abs().sum(axis=1)
        port_rets = port_rets - turnover * rate
        # Both sides pay the initial full purchase (100% turnover).
        port_rets.iloc[0] -= rate
        bench_rets = bench_rets.copy()
        bench_rets.iloc[0] -= rate

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