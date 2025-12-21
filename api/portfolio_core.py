# portfolio_core.py
from typing import Dict, List
import pandas as pd
import yfinance as yf
import numpy as np

def load_price_history(
    tickers: List[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Return daily adjusted open and close prices for the given tickers."""
    data = yf.download(tickers, start=start, end=end, progress=False)
    return data[["Adj Close", "Open"]].dropna(how="all")


def compute_portfolio_metrics(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: str = "SPY",
) -> Dict[str, float]:
    """Compute CAGR, volatility, Sharpe, max drawdown for the portfolio."""
    # align weights with columns
    w = np.array([weights[t] for t in prices.columns if t in weights])
    w = w / w.sum()

    rets = prices.pct_change().dropna()
    port_rets = (rets * w).sum(axis=1)

    # basic metrics (daily → annual assuming 252 trading days)
    mean = port_rets.mean()
    std = port_rets.std()
    sharpe = (mean * 252) / (std * np.sqrt(252)) if std > 0 else 0.0

    cum = (1 + port_rets).cumprod()
    peak = cum.cummax()
    dd = (cum / peak) - 1
    max_dd = dd.min()

    years = (prices.index[-1] - prices.index[0]).days / 365.25
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0.0

    return {
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "volatility_annual": float(std * np.sqrt(252)),
        "max_drawdown": float(max_dd),
    }


def recommend_portfolio(
    prices: pd.DataFrame,
    constraints: Dict,
) -> Dict[str, float]:
    """
    Stub for your existing predict_agent logic – here you’d plug in FinRL,
    RL model, or optimization routine.
    Return a dict of {ticker: weight}.
    """
    # TODO: hook up your actual RL / optimization logic.
    cols = list(prices.columns)
    equal_weight = 1.0 / len(cols)
    return {t: equal_weight for t in cols}
