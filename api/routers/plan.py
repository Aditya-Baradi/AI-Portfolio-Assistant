"""
Investor profile and a deliberately limited retirement-planning scenario.

A note on the retirement model, because this is the least empirical thing the
app does and it used to be presented as though it were analysis:

The simulation does NOT use the user's actual holdings to estimate expected
return. It uses a stylised risk/reward line — expected return rises with the
volatility the user says they can tolerate — and only takes the starting value
from the portfolio. That is a planning heuristic for exploring "what if I save
more / take more risk", not a projection of this particular portfolio.

Every response therefore carries `assumptions`, stating the numbers driving it
in plain language, and the UI renders them next to the chart. If you change the
model, change the text with it.
"""
from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from api import db
from api.deps import check_api_rate, verified_user
from api.portfolio_core import live_portfolio_valuation

logger = logging.getLogger("evergreen.plan")
router = APIRouter(tags=["plan"])

# Stylised risk/reward line used by the retirement simulation.
BASE_NOMINAL_RETURN = 0.03   # nominal heuristic; inflation is not modeled
RETURN_PER_VOL = 0.40        # extra expected return per unit of volatility taken
MAX_ASSUMED_RETURN = 0.15    # hard cap; above this the model is fantasy
MAX_RETIREMENT_VALUATION_TICKERS = 25


class ProfileUpdate(BaseModel):
    years_to_retirement: int = Field(ge=1, le=60)
    risk_tolerance: int = Field(ge=1, le=10)
    max_volatility_pct: float = Field(ge=5, le=80, allow_inf_nan=False)
    goal: str = "balanced"
    monthly_contribution: float = Field(
        default=0.0, ge=0, le=10_000_000, allow_inf_nan=False
    )
    goal_amount: float = Field(
        default=0.0, ge=0, le=1_000_000_000_000_000, allow_inf_nan=False
    )


@router.get("/profile")
def get_profile(authorization: str | None = Header(default=None)):
    user = verified_user(authorization)
    return db.get_profile(user["id"]) or {}


@router.post("/profile")
def set_profile(update: ProfileUpdate, authorization: str | None = Header(default=None)):
    user = verified_user(authorization)
    if not all(math.isfinite(value) for value in (
        update.max_volatility_pct,
        update.monthly_contribution,
        update.goal_amount,
    )):
        raise HTTPException(status_code=422, detail="Plan numbers must be finite.")
    if not (1 <= update.years_to_retirement <= 60):
        raise HTTPException(status_code=422,
                            detail="Years to retirement must be between 1 and 60.")
    if not (1 <= update.risk_tolerance <= 10):
        raise HTTPException(status_code=422, detail="Risk tolerance must be between 1 and 10.")
    if not (5 <= update.max_volatility_pct <= 80):
        raise HTTPException(status_code=422, detail="Max volatility must be between 5% and 80%.")
    if update.goal not in ("growth", "balanced", "income"):
        raise HTTPException(status_code=422, detail="Goal must be growth, balanced, or income.")
    if update.monthly_contribution < 0:
        raise HTTPException(status_code=422, detail="Monthly contribution can't be negative.")
    if update.goal_amount < 0:
        raise HTTPException(status_code=422, detail="Goal amount can't be negative.")
    profile = update.model_dump()
    db.save_profile(user["id"], profile)
    return {"ok": True, **profile}


@router.get("/plan/recommendations")
def plan_recommendations(request: Request, authorization: str | None = Header(default=None)):
    """Disabled personalized security-screen surface."""
    verified_user(authorization)
    check_api_rate(request, "recs")
    raise HTTPException(
        status_code=410,
        detail=(
            "Personalized ranked security screens are disabled. "
            "The application provides neutral portfolio descriptions and "
            "retrospective model comparisons only."
        ),
    )

@router.get("/plan/retirement")
def plan_retirement(request: Request, authorization: str | None = Header(default=None)):
    """
    Monte Carlo outlook to retirement driven by the PLAN's risk settings.

    IMPORTANT: expected return here is derived from the risk level the user
    chose, not from their holdings. See the module docstring. The response
    spells this out in `assumptions` so the UI can show it beside the chart.
    """
    user = verified_user(authorization)
    check_api_rate(request, "retirement")
    profile = db.get_profile(user["id"])
    if not profile:
        raise HTTPException(status_code=404, detail="Save your plan settings first.")

    from api.recommend import retirement_paths, target_volatility

    sigma = min(target_volatility(profile["risk_tolerance"]),
                float(profile["max_volatility_pct"]) / 100.0)
    mu = min(BASE_NOMINAL_RETURN + RETURN_PER_VOL * sigma, MAX_ASSUMED_RETURN)

    value0 = 0.0
    valuation_as_of = None
    pf = db.get_portfolio(user["id"])
    if pf:
        holdings = pf.get("holdings", []) if isinstance(pf, dict) else pf
        unique_tickers = {
            str(row.get("ticker") or row.get("symbol") or row.get("tic")).upper()
            for row in holdings if isinstance(row, dict)
            and (row.get("ticker") or row.get("symbol") or row.get("tic"))
        }
        if len(unique_tickers) > MAX_RETIREMENT_VALUATION_TICKERS:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Retirement valuation supports at most "
                    f"{MAX_RETIREMENT_VALUATION_TICKERS} holdings."
                ),
            )
        valuation = live_portfolio_valuation(pf)
        if valuation["fallback_tickers"]:
            raise HTTPException(
                status_code=502,
                detail=(
                    "Current prices are unavailable for: "
                    + ", ".join(valuation["fallback_tickers"])
                    + ". The scenario was not anchored to stale imported values."
                ),
            )
        value0 = valuation["total_value"]
        valuation_as_of = valuation["as_of"]

    goal = float(profile.get("goal_amount") or 0.0)
    monthly = profile.get("monthly_contribution", 0.0)
    out = retirement_paths(mu, sigma, value0, monthly,
                           profile["years_to_retirement"], goal=goal or None)
    out["based_on"] = "your plan settings"
    out["valuation_as_of"] = valuation_as_of

    # Be explicit about what drove the numbers. This is the difference between
    # a planning tool and a projection dressed up as one.
    #
    # EXTEND the assumptions block that retirement_paths already returns
    # (mu / sigma / monthly / sims) rather than replacing it — the UI reads
    # those keys directly.
    out.setdefault("assumptions", {}).update({
        "starting_value": round(value0, 2),
        "years": profile["years_to_retirement"],
        "return_model": (
            f"Nominal assumed return = {BASE_NOMINAL_RETURN:.0%} + "
            f"{RETURN_PER_VOL:.2f} x volatility, "
            f"capped at {MAX_ASSUMED_RETURN:.0%}."
        ),
        "uses_your_holdings": False,
        "currency_basis": "nominal dollars; inflation is not modeled",
        "taxes_and_fees": "not modeled",
    })
    out["disclaimer"] = (
        "This is a nominal-dollar planning scenario for the risk level selected, "
        "not a forecast of actual holdings and not financial advice. The goal "
        "percentage is only the share of simulated paths meeting the goal under "
        "the stated assumptions; it is not a real-world probability. Inflation, "
        "taxes, fees, changing contributions and market-regime shifts are omitted."
    )
    return out
