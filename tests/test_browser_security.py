"""Static browser trust-boundary regression tests."""
from __future__ import annotations

from pathlib import Path

INDEX = Path(__file__).resolve().parents[1] / "api" / "index.html"


def _page() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_authentication_is_not_persisted_in_web_storage():
    page = _page()
    assert "authToken" not in page
    assert 'localStorage.setItem("auth' not in page
    assert 'localStorage.getItem("auth' not in page
    assert "HttpOnly cookie" in page


def test_boot_restores_cookie_session_through_me():
    page = _page()
    assert 'api("/me"' in page
    assert 'credentials: "same-origin"' in page


def test_personalized_trade_and_ranked_pick_ui_is_not_public():
    page = _page()
    assert 'api("/portfolio/rebalance"' not in page
    assert 'api("/plan/recommendations"' not in page
    assert 'api("/portfolio/history"' not in page
    assert "Stocks that fit your plan" not in page
    assert "How to get there" not in page
    assert '"Bullish"' not in page
    assert '"Bearish"' not in page


def test_retirement_language_is_simulation_language():
    page = _page()
    assert "Median simulated outcome" in page
    assert "Modeled paths reaching" in page
    assert "not the probability that you will reach the goal" in page
    assert ">Most likely<" not in page
    assert "Chance of ${money(ret.goal)}" not in page


def test_cookie_session_rotation_never_enters_browser_storage():
    page = _page()
    assert '"/auth/refresh"' in page
    assert 'credentials: "same-origin"' in page
    assert 'const res = await api("/me"' in page
    assert 'fetch(API_BASE + "/me"' not in page
    assert "sessionAgeCheckTimer = setInterval" in page
    assert "X-Session-Token" not in page
    assert "sessionStorage.setItem(\"auth" not in page


def test_two_factor_setup_requires_reauthentication():
    page = _page()
    assert "Confirm your password before" in page
    assert "creates a new authenticator secret" in page
    assert 'password: document.getElementById("tfSetupPw").value' in page


def test_projection_is_labeled_as_an_illustration_not_a_forecast():
    page = _page()
    assert "12-month model illustration" in page
    assert "Historical CAGR input" in page
    assert "Illustrative total-return series paths" in page
    assert "Total-return-equivalent model range" in page
    assert "Estimated price ranges" not in page
    assert "likely range" not in page.lower()


def test_neutral_comparator_is_not_presented_as_an_optimized_target():
    page = _page()
    assert "Equal-weight comparison" in page
    assert ".optimized" not in page
    assert "recommended allocation or trade plan" in page
