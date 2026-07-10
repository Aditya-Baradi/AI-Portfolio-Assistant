import os
import json
import warnings
from datetime import datetime, timedelta
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

import numpy as np
import pandas as pd
import yfinance as yf

try:
    from .data_cache import cached_download, cached_sectors
except ImportError:  # running as a plain script
    from api.data_cache import cached_download, cached_sectors



import warnings
from importlib import import_module

FINRL_AVAILABLE = False

try:
    # Data processor
    DataProcessor = import_module("finrl.meta.data_processor").DataProcessor


    try:
        DRLAgent = import_module("finrl.agents.stablebaselines3.models").DRLAgent
    except ModuleNotFoundError:
        DRLAgent = import_module("finrl.agents.stable_baselines3.models").DRLAgent  


    StockPortfolioEnv = None
    for mod in (
        "finrl.meta.env_portfolio_allocation.env_portfolio",  # FinRL >=0.3.6 layout
        "finrl.meta.env_portfolio.env_portfolio",
        "finrl.env.env_portfolio",
    ):
        try:
            StockPortfolioEnv = getattr(import_module(mod), "StockPortfolioEnv")
            break
        except (ModuleNotFoundError, ImportError):
            pass

    if StockPortfolioEnv is None:
        raise ModuleNotFoundError("StockPortfolioEnv not found in known locations.")

    data_split = import_module("finrl.meta.preprocessor.preprocessors").data_split

    FINRL_AVAILABLE = True

except Exception as e:
    warnings.warn(f"FinRL not available ({e}); will use PyPortfolioOpt fallback.")



GICS_SECTORS = {
    "Energy", "Materials", "Industrials", "Consumer Discretionary",
    "Consumer Staples", "Health Care", "Financials",
    "Information Technology", "Communication Services",
    "Utilities", "Real Estate"
}


SECTOR_NORMALIZE = {
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Technology": "Information Technology",
    "Financial Services": "Financials",
}

ETF_TICKERS = {
    "SPY","VOO","QQQ","VTI","IVV","DIA","IWM",
    "XLK","XLF","XLV","XLY","XLP","XLE","XLU","XLRE","XLC","XLB","XLI"
}

THRESH = 0.5




def load_tickers_from_portfolio_json(json_path, include_etfs=False):
    """Read your portfolio.json and return a de-duplicated list of tickers and a market value map."""
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)

    tickers, weights = [], {}
    for item in rows:
        ticker = (
            item.get("Sym/Cusip")
            or item.get("Symbol")
            or item.get("ticker")
            or item.get("Name")
        )
        if not ticker:
            continue
        t = str(ticker).strip().upper()
        if (not include_etfs) and t in ETF_TICKERS:
            continue
        tickers.append(t)

        mv = item.get("Mkt Value") or item.get("MktValue") or item.get("current_dollars")
        if mv is not None:
            mv_str = str(mv).replace("$", "").replace(",", "").strip()
            try:
                weights[t] = float(mv_str)
            except ValueError:
                pass

    tickers = list(dict.fromkeys(tickers))  
    return tickers, weights  


def get_sector_map(tickers):
    """Map each ticker to a normalized GICS sector (disk-cached)."""
    raw_map = cached_sectors(tickers)
    return {t: SECTOR_NORMALIZE.get(raw, raw) for t, raw in raw_map.items()}


def sector_breakdown_from_weights(weights, sector_map):
    """
    Return sector % weights from ticker weights (sum to 100%).
    'weights' can be dollar weights (current holdings) or final allocation weights.
    """
    total = sum(weights.values())
    sector_w = defaultdict(float)
    for t, w in weights.items():
        sector = sector_map.get(t, "Unknown")
        sector_w[sector] += w
    if total > 0:
        for s in list(sector_w.keys()):
            sector_w[s] = round(100.0 * sector_w[s] / total, 2)
    return dict(sector_w)


def _download_price_matrix(tickers, start_date, end_date):
    """
    Robustly fetch a 2D DataFrame of adjusted close prices (rows = dates, cols = tickers).
    Tries auto_adjust=True first (so 'Close' is adjusted), then falls back to Adj Close.
    """
    df = cached_download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("No price data returned by yfinance.")


    if isinstance(df.columns, pd.MultiIndex):
        lvl0 = df.columns.get_level_values(0)
        if "Close" in set(lvl0):
            prices = df["Close"]
        else:

            df2 = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if isinstance(df2.columns, pd.MultiIndex) and "Adj Close" in set(df2.columns.get_level_values(0)):
                prices = df2["Adj Close"]
            else:
                prices = df2["Close"]
    else:
        
        if "Close" in df.columns:
            prices = df[["Close"]].copy()
            colname = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
            prices.columns = [colname]
        else:
            df2 = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
            if "Adj Close" in df2.columns:
                prices = df2[["Adj Close"]].copy()
                colname = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
                prices.columns = [colname]
            else:
                prices = df2[["Close"]].copy()
                colname = tickers[0] if isinstance(tickers, (list, tuple)) else str(tickers)
                prices.columns = [colname]

    prices = prices.dropna(axis=0, how="all").dropna(axis=1, how="any")
    if prices.empty:
        raise ValueError("All price columns are empty after cleaning; check tickers or date range.")
    return prices




def _build_finrl_dataframe(tickers, start_date, end_date, techs):
    """
    Build a clean, rectangular FinRL-style long dataframe (date, tic, OHLCV +
    technical indicators) directly from yfinance + stockstats.

    FinRL 0.3.7's bundled YahooFinanceProcessor is broken against modern
    yfinance (it silently drops all but the first ticker), so we do the
    download/indicator work ourselves and keep only dates present for every
    surviving ticker so the env sees perfectly rectangular data.
    """
    from stockstats import StockDataFrame

    raw = cached_download(
        tickers, start=start_date, end=end_date,
        auto_adjust=True, progress=False, group_by="ticker",
    )
    if raw is None or raw.empty:
        raise ValueError("No price data returned by yfinance for the FinRL dataframe.")

    frames = []
    for t in tickers:
        try:
            sub = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except KeyError:
            continue
        sub = sub.rename(columns=str.lower)
        needed = ["open", "high", "low", "close", "volume"]
        if not set(needed).issubset(sub.columns):
            continue
        sub = sub[needed].dropna(how="any")
        if sub.empty:
            continue
        sub["tic"] = t
        sub["date"] = pd.to_datetime(sub.index).strftime("%Y-%m-%d")
        frames.append(sub.reset_index(drop=True))

    if len(frames) < 2:
        raise ValueError(f"Need >=2 tickers with usable data; got {len(frames)}.")

    df = pd.concat(frames, ignore_index=True)

    # Keep only dates where every surviving ticker is present (rectangular).
    n_tics = df["tic"].nunique()
    counts = df.groupby("date")["tic"].nunique()
    full_dates = set(counts[counts == n_tics].index)
    df = df[df["date"].isin(full_dates)].copy()
    if df.empty:
        raise ValueError("No common trading dates across tickers after alignment.")

    # Technical indicators per ticker via stockstats.
    out = []
    for t, d in df.groupby("tic"):
        d = d.sort_values("date").reset_index(drop=True)
        sdf = StockDataFrame.retype(d.copy())
        for ind in techs:
            d[ind] = pd.Series(sdf[ind].values).fillna(0.0).values
        out.append(d)
    df = pd.concat(out, ignore_index=True)
    df = df.sort_values(["date", "tic"], ignore_index=True)
    return df


def run_finrl_portfolio_optimization(tickers, start_date, end_date, timesteps=50_000):
    """
    Train a PPO agent in FinRL's StockPortfolioEnv and return final action weights (allocation) and logs.
    """
    if not FINRL_AVAILABLE:
        raise ImportError("FinRL not installed/available")

    # FinRL's StockPortfolioEnv writes reward plots to ./results on the final
    # step; ensure the directory exists and use a headless matplotlib backend.
    try:
        import matplotlib
        matplotlib.use("Agg")
    except Exception:
        pass
    os.makedirs("results", exist_ok=True)

    techs = ["macd", "rsi_30", "cci_30", "dx_30"]
    data = _build_finrl_dataframe(tickers, start_date, end_date, techs)

    # Only the tickers that actually survived data cleaning participate.
    tickers = sorted(data["tic"].unique())

    # The portfolio-allocation env consumes a per-day covariance matrix
    # ('cov_list') plus the technical indicators. Build a rolling covariance
    # feature the way the canonical FinRL portfolio tutorial does.
    lookback = 60
    unique_dates = data["date"].unique()
    if len(unique_dates) <= lookback + 5:
        raise ValueError(
            f"Not enough trading days ({len(unique_dates)}) for lookback={lookback}; widen the date range."
        )

    data.index = data["date"].factorize()[0]
    cov_list, return_list = [], []
    for i in range(lookback, len(unique_dates)):
        window = data.loc[i - lookback : i, :]
        price = window.pivot_table(index="date", columns="tic", values="close")
        rets = price.pct_change().dropna()
        return_list.append(rets)
        cov_list.append(rets.cov().values)

    df_cov = pd.DataFrame(
        {"date": unique_dates[lookback:], "cov_list": cov_list, "return_list": return_list}
    )
    data = data.merge(df_cov, on="date")
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data["date"].factorize()[0]

    # Train / trade split (data_split re-factorizes the day index correctly).
    split_date = (
        pd.to_datetime(start_date)
        + pd.Timedelta(days=int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 0.8))
    ).strftime("%Y-%m-%d")
    train = data_split(data, unique_dates[lookback], split_date)
    trade = data_split(data, split_date, unique_dates[-1])
    if train.empty or trade.empty:
        raise ValueError("Train/test split produced empty sets; adjust dates.")

    stock_dim = len(tickers)
    # For the portfolio env the state is the (stock_dim x stock_dim) covariance
    # matrix stacked with the indicators, so state_space == stock_dim.
    state_space = stock_dim

    env_kwargs = dict(
        hmax=100,
        initial_amount=1_000_000,
        transaction_cost_pct=0.001,
        state_space=state_space,
        stock_dim=stock_dim,
        tech_indicator_list=techs,
        action_space=stock_dim,
        reward_scaling=1e-4,
    )

    env_train = StockPortfolioEnv(df=train, **env_kwargs)
    env_trade = StockPortfolioEnv(df=trade, **env_kwargs)

    agent = DRLAgent(env=env_train)
    model = agent.get_model("ppo", policy="MlpPolicy")
    trained = agent.train_model(model=model, tb_log_name="ppo", total_timesteps=timesteps)

    # For the portfolio env, DRL_prediction returns (daily_return_df, actions_df),
    # where each actions row is the softmaxed weight vector over tickers.
    df_daily_return, df_actions = agent.DRL_prediction(model=trained, environment=env_trade)

    tics = list(df_actions.columns)
    last = df_actions.iloc[-1]
    if set(tics) >= set(tickers):
        allocation = {t: float(last[t]) for t in tickers}
    else:
        allocation = {tickers[i]: float(last.iloc[i]) for i in range(len(tickers))}

    s = sum(abs(w) for w in allocation.values())
    if s > 0:
        allocation = {t: round(abs(w) / s, 6) for t, w in allocation.items()}

    return allocation, df_daily_return, df_actions

from math import floor

def compute_portfolio_value(mv_map: dict | None, default_if_missing: float = 10_000.0) -> float:
    """
    Sum current market values from your JSON (mv_map). If none available, use a sensible default.
    """
    if mv_map and isinstance(mv_map, dict):
        total = sum(v for v in mv_map.values() if isinstance(v, (int, float)))
        if total > 0:
            return float(total)
    
    return float(default_if_missing)


def compute_dollar_targets(allocation: dict[str, float], portfolio_value: float) -> dict[str, float]:
    """
    Convert allocation weights (sum≈1) to dollar targets for a given portfolio value.
    """
    return {t: round(w * portfolio_value, 2) for t, w in allocation.items()}


def get_last_prices(tickers: list[str]) -> dict[str, float]:
    """
    Get the most recent adjusted close for each ticker (robust to single/multi ticker shapes).
    """
    df = yf.download(tickers, period="5d", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return {}
    
    try:
        if isinstance(df.columns, pd.MultiIndex):
            
            prices = df["Close"].ffill().iloc[-1]
            return {str(k): float(v) for k, v in prices.items()}
        else:
            
            price = float(df["Close"].ffill().iloc[-1])
            name = tickers[0] if tickers else "TICKER"
            return {str(name): price}
    except Exception:
       
        last = df.ffill().iloc[-1]
        if isinstance(last, pd.Series):
            return {str(k): float(v) for k, v in last.items() if isinstance(v, (int, float, np.floating))}
        return {}


def build_trade_plan(
    current_values: dict[str, float],
    dollar_targets: dict[str, float],
    price_map: dict[str, float],
    min_trade_dollars: float = 5.0,
    fractional_ok: bool = True,
    share_precision: int = 3,
) -> dict[str, dict]:
    """
    Create a per-ticker BUY/SELL/HOLD plan comparing current $ vs target $.
    Returns: {ticker: {current, target, delta, action, est_price, est_shares}}
    - min_trade_dollars: ignore tiny adjustments
    - fractional_ok: if True, shares can be fractional; else we floor to whole shares
    """
    
    all_tickers = sorted(set(current_values.keys()) | set(dollar_targets.keys()))
    plan = {}

    for t in all_tickers:
        cur = float(current_values.get(t, 0.0))
        tgt = float(dollar_targets.get(t, 0.0))
        delta = round(tgt - cur, 2)

        
        action = "HOLD"
        if delta > min_trade_dollars:
            action = "BUY"
        elif delta < -min_trade_dollars:
            action = "SELL"

        price = float(price_map.get(t, 0.0)) if price_map else 0.0

        if price > 0 and action != "HOLD":
            raw_shares = delta / price
            if fractional_ok:
                est_shares = round(raw_shares, share_precision)
            else:
                
                if raw_shares > 0:
                    est_shares = float(floor(raw_shares))
                else:
                    est_shares = float(-floor(abs(raw_shares)))
        else:
            est_shares = 0.0

        plan[t] = {
            "current": round(cur, 2),
            "target": round(tgt, 2),
            "delta": delta,
            "action": action,
            "est_price": round(price, 4) if price else 0.0,
            "est_shares": est_shares,
        }

    return plan



def run_pypfopt_max_sharpe(tickers, start_date, end_date):
    """
    Try PyPortfolioOpt max-Sharpe; if unavailable/fails, fall back to:
      1) SciPy SLSQP (sum-to-1, long-only)
      2) NumPy closed-form (Σ^{-1} μ) projected to long-only, normalized
    Returns: (weights_dict, prices_df, method_str)
    """
    prices = _download_price_matrix(tickers, start_date, end_date)
    prices = prices.dropna(axis=0, how="all").dropna(axis=1, how="any")
    if prices.empty:
        raise ValueError("No price data for the given tickers/dates.")

    
    try:
        
        expected_returns = import_module("pypfopt.expected_returns")
        risk_models = import_module("pypfopt.risk_models")
        EfficientFrontier = import_module("pypfopt.efficient_frontier").EfficientFrontier

        mu = expected_returns.mean_historical_return(prices)       
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()  
        ef = EfficientFrontier(mu, S)
        _ = ef.max_sharpe()
        cleaned = ef.clean_weights()

        total = sum(cleaned.values())
        if total <= 0:
            raise ValueError("PyPortfolioOpt returned zero weights.")

        cleaned = {t: round(w / total, 6) for t, w in cleaned.items() if w > 0}
        return cleaned, prices, "pypfopt"

    except Exception as e1:
        warnings.warn(f"PyPortfolioOpt unavailable or failed ({e1}); trying SciPy SLSQP fallback.")

   
    try:
        import scipy.optimize as opt

        rets = prices.pct_change().dropna()
        mu = (rets.mean() * 252.0).values
        Sigma = (rets.cov() * 252.0).values
        cols = prices.columns.tolist()
        n = len(cols)

        def neg_sharpe(w):
            ret = w @ mu
            vol = np.sqrt(w @ Sigma @ w)
            return -ret / (vol + 1e-12)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
        bounds = [(0.0, 1.0)] * n
        w0 = np.ones(n) / n

        res = opt.minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 500})
        w = res.x if res.success else w0

        cleaned = {cols[i]: float(round(w[i], 6)) for i in range(n)}
        return cleaned, prices, "scipy"

    except Exception as e2:
        warnings.warn(f"SciPy fallback failed ({e2}); using NumPy closed-form fallback.")

    
    rets = prices.pct_change().dropna()
    mu = (rets.mean() * 252.0).values
    Sigma = (rets.cov() * 252.0).values
    cols = prices.columns.tolist()

    try:
        w = np.linalg.pinv(Sigma).dot(mu)
    except Exception:
        w = np.ones(len(cols))

    
    w = np.maximum(w, 0.0)
    s = w.sum()
    if s == 0:
        w = np.ones_like(w) / len(cols)
    else:
        w = w / s

    cleaned = {cols[i]: round(float(w[i]), 6) for i in range(len(cols))}
    return cleaned, prices, "numpy"



def _blend_allocations(rl: dict, ms: dict, rl_weight: float) -> dict:
    """
    Convex-combine two allocation dicts: rl_weight*RL + (1-rl_weight)*max-Sharpe.

    Keys present in only one side are treated as 0 in the other. The result is
    renormalized to sum to 1. This keeps the RL-learned tilt while anchoring the
    per-stock weights to the (fast, differentiated) mean-variance solution.
    """
    keys = set(rl) | set(ms)
    combined = {k: rl_weight * float(rl.get(k, 0.0)) + (1.0 - rl_weight) * float(ms.get(k, 0.0)) for k in keys}
    s = sum(combined.values())
    if s <= 0:
        return {k: round(v, 6) for k, v in combined.items()}
    return {k: round(v / s, 6) for k, v in combined.items()}


def run_min_vol_capped(tickers, start_date, end_date, max_weight: float = 0.15):
    """
    Minimum-volatility optimization with a per-name weight cap.

    Rationale (validated out-of-sample in api/backtest_vs_sp500.py's walk-forward):
    expected returns estimated from history are mostly noise, and max-Sharpe
    loads up on that noise. Min-vol uses only the covariance matrix (the
    estimable input) and the weight cap acts as regularization, preventing
    30-45% single-name bets.

    Returns (weights_dict, prices_df, method_str).
    """
    prices = _download_price_matrix(tickers, start_date, end_date)
    prices = prices.dropna(axis=0, how="all").dropna(axis=1, how="any")
    if prices.empty:
        raise ValueError("No price data for the given tickers/dates.")
    n = len(prices.columns)
    # A cap below 1/n is infeasible; relax it if the universe is small.
    max_weight = max(float(max_weight), 1.05 / n)

    try:
        risk_models = import_module("pypfopt.risk_models")
        EfficientFrontier = import_module("pypfopt.efficient_frontier").EfficientFrontier

        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        ef = EfficientFrontier(None, S, weight_bounds=(0.0, max_weight))
        ef.min_volatility()
        cleaned = {t: w for t, w in ef.clean_weights().items() if w > 0}
        total = sum(cleaned.values())
        if total <= 0:
            raise ValueError("min_volatility returned zero weights.")
        return {t: round(w / total, 6) for t, w in cleaned.items()}, prices, "minvol-capped"

    except Exception as e1:
        warnings.warn(f"PyPortfolioOpt min-vol failed ({e1}); using SciPy fallback.")

    import scipy.optimize as opt

    rets = prices.pct_change().dropna()
    Sigma = (rets.cov() * 252.0).values
    cols = prices.columns.tolist()

    def variance(w):
        return w @ Sigma @ w

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, max_weight)] * n
    w0 = np.ones(n) / n
    res = opt.minimize(variance, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 500})
    w = res.x if res.success else w0
    return {cols[i]: round(float(w[i]), 6) for i in range(n)}, prices, "minvol-scipy"


def run_conservative_optimization(tickers, start_date, end_date,
                                  max_weight: float = 0.15,
                                  eq_blend: float = 0.5):
    """
    The evidence-backed allocation used for recommendations and walk-forward:
    min-vol with a weight cap, then blended eq_blend toward equal weight
    (shrinkage toward 1/N, which is very hard to beat out-of-sample).
    """
    opt_w, prices, method = run_min_vol_capped(tickers, start_date, end_date, max_weight)
    cols = list(prices.columns)
    eq_w = {t: 1.0 / len(cols) for t in cols}
    blended = _blend_allocations(opt_w, eq_w, 1.0 - eq_blend)
    return blended, prices, f"{method}+eq(1/N={eq_blend:.0%})"


def optimize_portfolio_with_finrl(json_path:str,
                                  include_etfs=False,
                                  lookback_years=5,
                                  finrl_timesteps=50_000,
                                  portfolio_value: float | None = None,
                                  min_trade_dollars: float = 5.0,
                                  fractional_ok: bool = True,
                                  blend_rl_weight: float = 0.5):
    #
    tickers, mv_map = load_tickers_from_portfolio_json(json_path, include_etfs=include_etfs)
    if not tickers:
        raise ValueError("No tickers found in portfolio.json")

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    sector_map = get_sector_map(tickers)

    # Always compute the fast anchor first (cheap). Conservative (min-vol +
    # weight cap + 1/N shrinkage) — it beat max-Sharpe decisively in the
    # out-of-sample walk-forward test.
    ms_alloc = None
    try:
        ms_alloc, _, anchor_method = run_conservative_optimization(tickers, start_date, end_date)
    except Exception as e:
        warnings.warn(f"Conservative anchor failed ({e}); trying max-Sharpe.")
        try:
            ms_alloc, _, anchor_method = run_pypfopt_max_sharpe(tickers, start_date, end_date)
        except Exception as e2:
            warnings.warn(f"max-Sharpe anchor failed ({e2}).")

    allocation, method = None, None
    if FINRL_AVAILABLE:
        try:
            rl_alloc, df_account_value, df_actions = run_finrl_portfolio_optimization(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                timesteps=finrl_timesteps,
            )
            if ms_alloc:
                # Blend the RL tilt with the conservative anchor so individual
                # stocks differentiate reliably (not just near equal-weight).
                allocation = _blend_allocations(rl_alloc, ms_alloc, blend_rl_weight)
                method = f"finrl+{anchor_method}(rl={blend_rl_weight:.2f})"
            else:
                allocation, method = rl_alloc, "finrl"
        except Exception as e:
            warnings.warn(f"FinRL failed ({e}); falling back to the anchor optimizer.")

    if allocation is None:
        if ms_alloc:
            allocation, method = ms_alloc, anchor_method
        else:
            allocation, _, method = run_pypfopt_max_sharpe(tickers, start_date, end_date)

    
    sector_alloc_pct = sector_breakdown_from_weights(allocation, sector_map)
    current_sector_pct = sector_breakdown_from_weights(mv_map or {}, sector_map) if mv_map else {}

    missing_recommended = sorted([s for s in GICS_SECTORS if sector_alloc_pct.get(s, 0) == 0])
    underweighted_recommended = sorted([s for s, pct in sector_alloc_pct.items() if 0 < pct <= THRESH])
    missing_current = sorted(
        [s for s in GICS_SECTORS if current_sector_pct.get(s, 0) == 0]) if current_sector_pct else []

   
    pv_used = portfolio_value if isinstance(portfolio_value, (int, float)) else compute_portfolio_value(mv_map)

    dollar_targets = compute_dollar_targets(allocation, pv_used)

   
    current_dollars = {t: float(mv_map.get(t, 0.0)) for t in allocation.keys()} if mv_map else {t: 0.0 for t in
                                                                                                allocation.keys()}

    
    last_prices = get_last_prices(list(allocation.keys()))

    trade_plan = build_trade_plan(
        current_values=current_dollars,
        dollar_targets=dollar_targets,
        price_map=last_prices,
        min_trade_dollars=min_trade_dollars,
        fractional_ok=fractional_ok,
        share_precision=3,
    )

    result = {
        "method": method,
        "tickers": tickers,
        "final_allocation_weights": allocation,  
        "sector_allocation_percent": sector_alloc_pct,  
        "current_sector_allocation_percent": current_sector_pct,  
        "missing_gics_sectors_recommended": missing_recommended,
        "underweighted_sectors_recommended": underweighted_recommended,
        "missing_gics_sectors_current": missing_current,

       
        "portfolio_value_used": round(pv_used, 2),
        "dollar_targets": dollar_targets,  
        "current_dollars": {k: round(v, 2) for k, v in current_dollars.items()},
        "last_prices": last_prices,  
        "trade_plan": trade_plan,  
    }
    return result


def format_optimization_report(result: dict, top_n: int = 10, min_trade_dollars: float = 25.0, max_trades: int = 12) -> str:
    method = result.get("method", "unknown")
    pv = result.get("portfolio_value_used")

    alloc = result.get("final_allocation_weights", {}) or {}
    top_alloc = sorted(alloc.items(), key=lambda kv: kv[1], reverse=True)[:top_n]

    sec = result.get("sector_allocation_percent", {}) or {}
    sec_sorted = sorted(sec.items(), key=lambda kv: kv[1], reverse=True)

    plan = result.get("trade_plan", {}) or {}
    moves = []
    for t, p in plan.items():
        action = p.get("action")
        delta = float(p.get("delta", 0.0))
        if action in ("BUY", "SELL") and abs(delta) >= float(min_trade_dollars):
            moves.append((t, action, delta, p.get("est_shares", 0.0), p.get("est_price", 0.0)))
    moves = sorted(moves, key=lambda x: abs(x[2]), reverse=True)[:max_trades]

    lines = []
    lines.append(f"Optimization method: {method}")
    if isinstance(pv, (int, float)):
        lines.append(f"Portfolio value used: ${pv:,.2f}")
    lines.append("")
    lines.append(f"Final allocation (top {len(top_alloc)}):")
    for t, w in top_alloc:
        lines.append(f"- {t}: {w*100:.2f}%")
    lines.append("")
    lines.append("Sector allocation:")
    for s, pct in sec_sorted:
        lines.append(f"- {s}: {pct:.2f}%")
    lines.append("")
    lines.append(f"Trade plan (>|${min_trade_dollars:.0f}|, top {len(moves)}):")
    if not moves:
        lines.append("- No trades above threshold.")
    else:
        for t, action, delta, shares, price in moves:
            lines.append(f"- {action} {t}: ${delta:,.2f} (~{shares} shares @ ${price})")

    return "\n".join(lines)



if __name__ == "__main__":

    json_path = "C:/Users/adity/AI_Financial_Advisor/AI-Portfolio-Assistant/portfolio.json"
    out = optimize_portfolio_with_finrl(
        json_path=json_path,
        include_etfs=False,
        lookback_years=5,
        finrl_timesteps=25_000,
        portfolio_value=None,
        min_trade_dollars=5.0,
        fractional_ok=True
    )
    print(json.dumps(out, indent=2))
