"""Shared fixtures: synthetic price data so tests never touch the network."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_prices(columns_returns: dict, n_days: int = 252, start="2024-01-02", seed=7):
    """
    Build a price DataFrame from per-column daily-return generators.

    columns_returns maps column name -> either a constant daily return (float)
    or a 1D array of daily returns of length n_days.
    """
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days + 1)
    data = {}
    for col, r in columns_returns.items():
        if np.isscalar(r):
            rets = np.full(n_days, float(r))
        else:
            rets = np.asarray(r, dtype=float)
        prices = 100.0 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
        data[col] = prices
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def patch_prices(monkeypatch):
    """Patch portfolio_core's price download with a fixed synthetic frame."""
    import api.portfolio_core as pc

    def _patch(prices_df):
        monkeypatch.setattr(pc, "_download_adj_close_matrix", lambda tickers, s, e: prices_df.copy())

    return _patch
