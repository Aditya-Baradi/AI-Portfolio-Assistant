"""Trust-boundary regression tests for model tools."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from api import crypto, db
from api.agent_tools import TOOL_SCHEMAS, AgentContext, dispatch_tool
from api.langchain_agent import (
    _enforce_financial_boundary,
    _get_history,
    _load_history,
    run_portfolio_agent,
)


@pytest.fixture(autouse=True)
def clear_agent_cache():
    from api import langchain_agent

    langchain_agent._history_cache.clear()
    yield
    langchain_agent._history_cache.clear()


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "agent-security.db")
    monkeypatch.setenv("EVERGREEN_MASTER_KEY", crypto.generate_key())
    crypto.reset_cache()
    db.init_db()
    yield
    crypto.reset_cache()


def test_model_tool_schemas_never_expose_identity_fields():
    serialized = json.dumps(TOOL_SCHEMAS).lower()
    for forbidden in (
        "user_id",
        "userid",
        "session_id",
        "sessionid",
        "chat_id",
        "chatid",
        "database",
        "file_path",
        "token",
    ):
        assert forbidden not in serialized


def test_dispatcher_binds_portfolio_to_server_context(isolated_db, monkeypatch):
    from api import agent_tools

    first = db.register_user("first-agent@example.com", "password123")
    second = db.register_user("second-agent@example.com", "password123")
    db.save_portfolio(
        first,
        {"holdings": [{"ticker": "FIRST", "shares": 1, "current_dollars": 10}]},
    )
    db.save_portfolio(
        second,
        {"holdings": [{"ticker": "SECOND", "shares": 1, "current_dollars": 20}]},
    )
    monkeypatch.setattr(
        agent_tools,
        "_live_valuation",
        lambda context: {
            "values": {"FIRST": 10.0} if context.user_id == first else {"SECOND": 20.0},
            "prices": {"FIRST": 10.0} if context.user_id == first else {"SECOND": 20.0},
            "shares": {"FIRST": 1.0} if context.user_id == first else {"SECOND": 1.0},
            "total_value": 10.0 if context.user_id == first else 20.0,
            "as_of": "2026-07-23",
            "fallback_tickers": [],
            "source_provider": "test",
            "price_semantics": "test adjusted price",
        },
    )

    result = json.loads(
        dispatch_tool(
            AgentContext(user_id=first, chat_id="owned"),
            "get_portfolio_summary",
            {},
        )
    )
    assert [row["ticker"] for row in result["holdings"]] == ["FIRST"]
    assert "SECOND" not in json.dumps(result)


def test_history_loader_rejects_a_foreign_chat(isolated_db):
    owner = db.register_user("owner-agent@example.com", "password123")
    attacker = db.register_user("attacker-agent@example.com", "password123")
    db.touch_chat(owner, "private-chat", first_message="private")
    db.add_message("private-chat", "user", "my private history")

    assert _load_history(
        f"u{attacker}.cprivate-chat",
        user_id=attacker,
        chat_id="private-chat",
    ) == []


def test_history_cache_key_is_derived_from_authenticated_context(isolated_db):
    owner = db.register_user("cache-owner@example.com", "password123")
    attacker = db.register_user("cache-attacker@example.com", "password123")
    db.touch_chat(owner, "private-cache-chat", first_message="private")
    db.add_message("private-cache-chat", "user", "owner-only history")

    owner_history = _get_history(
        "caller-controlled-label",
        user_id=owner,
        chat_id="private-cache-chat",
    )
    assert owner_history
    assert _get_history(
        "caller-controlled-label",
        user_id=attacker,
        chat_id="private-cache-chat",
    ) == []


def test_agent_refuses_to_infer_auth_from_session_text():
    with pytest.raises(ValueError, match="Authenticated user"):
        run_portfolio_agent("Show my portfolio", session_id="u1.cguessed")


@pytest.mark.parametrize(
    "unsafe",
    [
        "You should buy AAPL today.",
        "SELL 12 shares of TSLA.",
        "Allocate 40% into NVDA.",
        "My recommendation is to hold MSFT.",
    ],
)
def test_model_trade_instructions_are_blocked_deterministically(unsafe):
    answer, blocked = _enforce_financial_boundary(unsafe)
    assert blocked is True
    assert "can’t provide" in answer
    assert unsafe not in answer


def test_descriptive_risk_analysis_passes_boundary_filter():
    text = "AAPL represented 28% of the historical basket, which increased concentration risk."
    assert _enforce_financial_boundary(text) == (text, False)


def test_model_call_is_nonstored_bounded_and_serial(isolated_db, monkeypatch):
    from api import langchain_agent

    uid = db.register_user("model-contract@example.com", "password123")
    db.touch_chat(uid, "contract-chat", first_message="hello")
    captured = []

    class FakeCompletions:
        def create(self, **kwargs):
            captured.append(kwargs)
            message = SimpleNamespace(tool_calls=[], content="Historical summary only.")
            return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    fake_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    monkeypatch.setattr(langchain_agent, "_get_client", lambda: fake_client)

    answer = run_portfolio_agent(
        "Summarize historical risk.",
        user_id=uid,
        chat_id="contract-chat",
    )

    assert answer == "Historical summary only."
    assert len(captured) == 1
    request = captured[0]
    assert request["store"] is False
    assert request["parallel_tool_calls"] is False
    assert request["max_completion_tokens"] <= 2_000
    assert request["messages"][0]["role"] == "system"
    assert request["tools"] == TOOL_SCHEMAS
