import os
import json
import warnings
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import yfinance as yf



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
        "finrl.meta.env_portfolio.env_portfolio",  
        "finrl.env.env_portfolio",                 
    ):
        try:
            StockPortfolioEnv = getattr(import_module(mod), "StockPortfolioEnv")
            break
        except ModuleNotFoundError:
            pass

    if StockPortfolioEnv is None:
        raise ModuleNotFoundError("StockPortfolioEnv not found in known locations.")

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
    """Map each ticker to a normalized GICS sector."""
    sector_map = {}
    for t in tickers:
        try:
            yft = yf.Ticker(t)
            try:
                info = yft.get_info()
            except Exception:
                info = yft.info
            raw = info.get("sector") or "Unknown"
        except Exception:
            raw = "Unknown"
        sector_map[t] = SECTOR_NORMALIZE.get(raw, raw)
    return sector_map


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
    df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
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




def run_finrl_portfolio_optimization(tickers, start_date, end_date, timesteps=50_000):
    """
    Train a PPO agent in FinRL's StockPortfolioEnv and return final action weights (allocation) and logs.
    """
    if not FINRL_AVAILABLE:
        raise ImportError("FinRL not installed/available")

    
    dp = DataProcessor(data_source="yahoofinance",
                       start_date=start_date, end_date=end_date, time_interval="1D")
    dp.download_data(ticker_list=tickers)
    dp.clean_data()
    techs = ["macd", "rsi_30", "cci_30", "dx_30"]  
    dp.add_technical_indicator(tech_indicator_list=techs)

    data = dp.dataframe  
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("No data returned by DataProcessor")

   
    split_date = (pd.to_datetime(start_date) + pd.Timedelta(days=int((pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 0.8))).strftime("%Y-%m-%d")
    train = data[data.date < split_date]
    trade = data[data.date >= split_date]
    if train.empty or trade.empty:
        raise ValueError("Train/test split produced empty sets; adjust dates.")

    stock_dim = len(tickers)
    state_space = 1 + 2 * stock_dim + len(techs) * stock_dim

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
    trained = agent.train_model(model=model, total_timesteps=timesteps)

    
    df_account_value, df_actions = agent.DRL_prediction(model=trained, environment=env_trade)

    
    if list(df_actions.columns) == tickers:
        allocation = {t: float(df_actions.iloc[-1][t]) for t in tickers}
    else:
        allocation = {tickers[i]: float(df_actions.iloc[-1, i]) for i in range(len(tickers))}

    
    s = sum(abs(w) for w in allocation.values())
    if s > 0:
        allocation = {t: round(abs(w) / s, 6) for t, w in allocation.items()}

    return allocation, df_account_value, df_actions

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



def optimize_portfolio_with_finrl(json_path:str,
                                  include_etfs=False,
                                  lookback_years=5,
                                  finrl_timesteps=50_000,
                                  portfolio_value: float | None = None,
                                  min_trade_dollars: float = 5.0,
                                  fractional_ok: bool = True):
    #
    tickers, mv_map = load_tickers_from_portfolio_json(json_path, include_etfs=include_etfs)
    if not tickers:
        raise ValueError("No tickers found in portfolio.json")

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=365 * lookback_years)).strftime("%Y-%m-%d")

    sector_map = get_sector_map(tickers)

    
    if FINRL_AVAILABLE:
        try:
            allocation, df_account_value, df_actions = run_finrl_portfolio_optimization(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                timesteps=finrl_timesteps
            )
            method = "finrl"
        except Exception as e:
            warnings.warn(f"FinRL failed ({e}); falling back to max-Sharpe.")
            allocation, _, method = run_pypfopt_max_sharpe(tickers, start_date, end_date)
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
