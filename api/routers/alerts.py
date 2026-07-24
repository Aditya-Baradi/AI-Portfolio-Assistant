"""Bounded, explicitly-labeled portfolio and headline heuristics."""
from __future__ import annotations

import logging

from fastapi import APIRouter, Header, Request

from api import db
from api.deps import check_api_rate, verified_user, weights_from_pf
from api.observability import report

logger = logging.getLogger("evergreen.alerts")
router = APIRouter(tags=["alerts"])


@router.get("/alerts")
def get_alerts(
    request: Request,
    authorization: str | None = Header(default=None),
):
    """
    The user's heuristic notices. A database claim permits at most one lazy
    recomputation per account every six hours, even across workers.
    """
    check_api_rate(request, "alerts")
    user = verified_user(authorization)
    if db.claim_alert_check(user["id"]):
        try:
            _compute_alerts(user["id"])
        except Exception as e:
            report(logger, "Alert computation failed", e, user_id=user["id"])
    alerts = db.list_alerts(user["id"])
    return {"alerts": alerts, "unseen": sum(1 for a in alerts if not a["seen"])}


@router.post("/alerts/seen")
def alerts_seen(
    request: Request,
    authorization: str | None = Header(default=None),
):
    check_api_rate(request, "alerts-seen")
    user = verified_user(authorization)
    db.mark_alerts_seen(user["id"])
    return {"ok": True}


def _compute_alerts(user_id: int) -> None:
    from datetime import datetime, timedelta

    from api.portfolio_core import _download_adj_close_matrix
    from api.sentiment import cached_ticker_sentiment

    pf = db.get_portfolio(user_id)
    if not pf:
        return
    weights = weights_from_pf(pf)
    if not weights:
        return

    # 1) Did an imported holding-mix proxy drop >5% over five sessions?
    try:
        import numpy as np

        end = datetime.today().strftime("%Y-%m-%d")
        start = (datetime.today() - timedelta(days=14)).strftime("%Y-%m-%d")
        prices = _download_adj_close_matrix(list(weights.keys()), start, end).ffill()
        usable = [t for t in weights if t in prices.columns]
        if usable and len(prices) >= 6:
            w = np.array([weights[t] for t in usable])
            w = w / w.sum()
            norm = prices[usable] / prices[usable].iloc[0]
            curve = norm.dot(w)
            week_chg = float(curve.iloc[-1] / curve.iloc[-6] - 1.0)
            if week_chg < -0.05:
                db.add_alert(
                    user_id,
                    "An imported holding-mix historical proxy moved down "
                    f"{abs(week_chg) * 100:.1f}% across the last five trading "
                    "sessions. This is not actual account performance.",
                )
    except Exception as e:
        report(logger, "Weekly drop alert failed", e, user_id=user_id)

    # 2) Which bounded subset currently has sufficiently negative headline
    # language? This does not claim the tone changed from a previous state.
    for t in list(weights.keys())[:20]:
        try:
            s = cached_ticker_sentiment(t)
            if s["n_headlines"] >= 3 and s["avg_score"] <= -0.15:
                db.add_alert(
                    user_id,
                    f"Recent headline language for {t} scored negative "
                    f"({s['avg_score']:+.2f}) in this heuristic sample.",
                    ticker=t,
                )
        except Exception as e:
            report(
                logger,
                "Headline-tone alert failed",
                e,
                user_id=user_id,
                ticker=t,
            )
            continue
