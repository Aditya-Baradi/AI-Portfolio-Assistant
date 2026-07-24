"""Pure regression tests for hold-between-rebalance mechanics."""

import pandas as pd
import pytest

from api.backtest_vs_sp500 import _hold_period_returns


def test_hold_period_returns_drift_weights_without_daily_reset():
    returns = pd.DataFrame(
        {"AAA": [1.0, -0.5], "BBB": [0.0, 0.0]},
        index=pd.bdate_range("2026-01-02", periods=2),
    )
    period, ending = _hold_period_returns(
        returns, {"AAA": 0.5, "BBB": 0.5}
    )
    assert period.iloc[0] == pytest.approx(0.5)
    assert period.iloc[1] == pytest.approx(-1 / 3)
    assert ending == pytest.approx({"AAA": 0.5, "BBB": 0.5})
