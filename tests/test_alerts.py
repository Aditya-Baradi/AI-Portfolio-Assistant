"""Alert notices stay bounded and do not overstate what snapshot data proves."""

from __future__ import annotations

import pandas as pd


def test_alert_copy_labels_historical_proxy_and_current_tone(monkeypatch):
    from api.routers import alerts
    import api.portfolio_core as portfolio_core
    import api.sentiment as sentiment

    monkeypatch.setattr(
        alerts.db,
        "get_portfolio",
        lambda _user_id: {
            "holdings": [{"ticker": "TEST", "current_dollars": 100.0}]
        },
    )
    prices = pd.DataFrame(
        {"TEST": [100.0, 100.0, 99.0, 98.0, 96.0, 90.0]},
        index=pd.date_range("2026-01-02", periods=6, freq="B"),
    )
    monkeypatch.setattr(
        portfolio_core,
        "_download_adj_close_matrix",
        lambda *_args, **_kwargs: prices,
    )
    monkeypatch.setattr(
        sentiment,
        "cached_ticker_sentiment",
        lambda _ticker: {
            "n_headlines": 3,
            "avg_score": -0.25,
        },
    )
    recorded: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        alerts.db,
        "add_alert",
        lambda _user_id, message, ticker=None: recorded.append((message, ticker)),
    )

    alerts._compute_alerts(7)

    messages = [message for message, _ticker in recorded]
    assert any("imported holding-mix historical proxy" in message for message in messages)
    assert any("not actual account performance" in message for message in messages)
    assert any("headline language" in message for message in messages)
    assert all("your portfolio is down" not in message.lower() for message in messages)
    assert all("turned negative" not in message.lower() for message in messages)
