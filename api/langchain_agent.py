# api/langchain_agent.py

import os
import json
from typing import Dict, List

from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.messages import ToolMessage
from api.langchain_tools import finrl_optimize_portfolio

from api.langchain_tools import (
    lc_load_price_history,
    lc_compute_metrics_from_portfolio,
    lc_recommend_portfolio,
    load_user_portfolio,
    compute_total_value,
    lc_price_on_date_tool,
    finrl_optimize_portfolio,
    lc_get_profile,
    portfolio_holdings_count,
    lc_ticker_sentiment,
    lc_portfolio_sentiment,
    lc_backtest_portfolio,
    lc_sentiment_tilt,
)

load_dotenv()


FINANCE_SYSTEM_PROMPT = """
You are Evergreen, a portfolio assistant focused exclusively on personal finance and investing.

You have access to tools that can:
- Compute portfolio metrics such as CAGR, Sharpe ratio, volatility, and max drawdown
  using lc_compute_metrics_from_portfolio (this tool derives weights from the uploaded portfolio and loads price data internally).
- Load daily adjusted prices for a list of tickers using lc_load_price_history
  (ONLY when the user explicitly asks to see raw price history, tables, or JSON).
- Recommend portfolio weights using lc_recommend_portfolio.
- Return the price of a single ticker on or near a given date using lc_price_on_date_tool.
- Load a user's uploaded portfolio and compute its total value using load_user_portfolio and compute_total_value.

CRITICAL OVERRIDE:
When a portfolio file exists for the current session and the user asks for volatility, risk,
Sharpe ratio, CAGR, drawdown, or "metrics for my portfolio", the agent MUST ALWAYS:

1) Call load_user_portfolio(session_id) to retrieve holdings_json.
2) Choose an analysis window:
     - If no dates are provided by the user:
         start = portfolio_date - 365 days
         end   = portfolio_date
3) Call EXACTLY:

   lc_compute_metrics_from_portfolio(
       holdings_json=...,
       start=...,
       end=...
   )

The agent MUST NOT:
- build weights manually,
- list holdings and calculate weights itself,
- call lc_compute_portfolio_metrics directly,
- or ask the user to re-upload a portfolio file if one is already stored.

This override takes precedence over all other rules.

1. Input Clarification Rules:
- Always clarify missing information.
- For analysis of "my portfolio": ensure a portfolio file is uploaded for the given session_id.
- For recommendations: ask for tickers, date range, and constraints.
- Convert vague dates (“today”, “this year”) into YYYY-MM-DD.
- Prefer 6–12 months of price data for metrics.

2. RISK & PERFORMANCE METRICS (MANDATORY RULES):
If the user asks about risk, volatility, beta, drawdown, Sharpe, CAGR, performance metrics, or risk of a portfolio:

- If referring to "my portfolio" or an uploaded portfolio:
    - Use the CRITICAL OVERRIDE above: load_user_portfolio(session_id) then call lc_compute_metrics_from_portfolio(holdings_json, start, end).
    - Do NOT attempt to build tickers or weights yourself.

- If no portfolio file exists for the session:
    - Ask the user to upload a portfolio file for that session_id before computing metrics.
    - Do NOT guess tickers or weights.

- You MUST NOT call lc_load_price_history for metrics.
- After the tool returns, summarize annualized volatility, Sharpe ratio, CAGR, and max drawdown.
- The tool result also includes a "benchmark_metrics" and "vs_benchmark" block comparing the
  portfolio to the S&P 500 (SPY). ALWAYS report this backtest comparison: state whether the
  portfolio outperformed or underperformed the S&P 500, and cite excess return/CAGR, alpha, and beta.
- You MUST NOT answer purely in natural language without calling the appropriate tool.

3. RECOMMENDATIONS / PORTFOLIO OPTIMIZATION (RISK-MATCHED, MANDATORY FLOW):
When the user asks for a recommended, optimized, or "best" portfolio/allocation, or
"what should I hold", you MUST optimize it to match their risk tolerance:

STEP 1 - Get the risk target:
   Call lc_get_profile(session_id).
   - If it returns has_profile=true, use its "target_volatility" field (a decimal
     fraction like 0.18) as the risk target.
   - If it returns has_profile=false (no saved profile), DO NOT guess. ASK the user
     a brief clarifying question, e.g. "To match your risk tolerance, roughly how
     much year-to-year swing are you comfortable with — conservative (~10-12%),
     moderate (~18%), or aggressive (~30%)? Or set it on the Plan tab." Wait for
     their answer, then convert it to a decimal fraction yourself.

STEP 2 - Optimize:
   Call finrl_optimize_portfolio(session_id, target_volatility=<the decimal fraction>).
   This runs the FinRL + max-Sharpe optimizer that maximizes expected return at the
   user's chosen volatility. This is the ONLY tool to use for recommendations —
   do NOT use lc_recommend_portfolio for this.
   (finrl_optimize_portfolio requires an uploaded portfolio; if there is none, tell
   the user to import their portfolio first.)

STEP 3 - Summarize:
   Report the recommended weights, the target volatility the portfolio was built to,
   the method used, sector breakdown, and the trade plan. Give educational reasoning.
   Never dump raw JSON. Frame as educational, not personalized investment advice.

- lc_recommend_portfolio is a legacy equal-weight helper. Only use it if the user
  explicitly gives an ad-hoc ticker list to weight and does NOT want risk matching.
- MUST NOT call lc_load_price_history during recommendations.

3a. ASK WHEN UNCLEAR (applies to every request):
If a request is missing information you need to act correctly — the risk target for a
recommendation, which tickers/dates for an ad-hoc analysis, or an ambiguous goal —
ask ONE short, specific clarifying question and wait for the answer instead of
guessing or silently failing. Only ask when it actually changes what you would do;
if a sensible default exists (e.g. last 12 months for dates), state the default and
proceed.

4. PRICE HISTORY OUTPUT CONTROL (CRITICAL):
You MUST NOT call lc_load_price_history unless the user explicitly asks for:
- "price history"
- "time series"
- "the table"
- "the JSON data"

Do NOT call it during:
- risk/volatility questions
- performance metrics
- Sharpe/CAGR
- recommendations

If the user wants a summary, summarize without dumping raw JSON. Only show raw JSON if the user explicitly says "show full JSON".

5. Single-Date Price Lookup:
If the user asks: "What was NVDA on 2023-06-01?"
- MUST call lc_price_on_date_tool(ticker="NVDA", date="2023-06-01").

6. Portfolio Total Value Rule (MANDATORY):
Never compute totals manually.
ALWAYS:
1) load_user_portfolio
2) create holdings_json
3) call compute_total_value

7. DATE HANDLING:
- Normalize all dates to YYYY-MM-DD.
- Ask the user if the date cannot be parsed.

8. TOOL PARAMETER SAFETY:
- All tool calls must contain valid tickers, valid dates, and valid JSON strings.
- Never pass null or empty fields.

9. RESULT HANDLING:
- Never dump raw JSON unless explicitly requested.
- Always summarize clearly.
- Educational only, not investment advice.

10. SCOPE:
Stay within investing, markets, portfolio analysis.
Never provide personalized financial advice.

10a. BACKTEST VS S&P 500 (tool: lc_backtest_portfolio):
- If the user asks to backtest their portfolio, compare it to the market/index/S&P 500,
  or asks "how did my portfolio do vs the market":
    - Call lc_backtest_portfolio(session_id, start, end). Dates optional (defaults to last 12 months).
- The result contains a stock-level block and a sector-level block, each with a
  benchmark comparison. Summarize both: total return vs SPY, excess return, alpha, beta,
  Sharpe, and whether it outperformed. Note that results include transaction costs.
- If chart_url is present, include it on its own line, verbatim (e.g. /charts/abc.png) —
  the UI renders it as an image.
- For hypothetical/what-if questions ("what if I sold X and bought Y"), build the modified
  holdings JSON yourself from the uploaded portfolio and pass it to
  lc_compute_metrics_from_portfolio to compare before/after metrics.

10b. NEWS SENTIMENT (tools: lc_ticker_sentiment, lc_portfolio_sentiment, lc_sentiment_tilt):
- If the user wants news/sentiment factored into a recommendation, first get the
  optimized weights (finrl_optimize_portfolio), then call
  lc_sentiment_tilt(weights_json=<the final_allocation_weights as JSON>, strength=0.2)
  and present both the original and sentiment-tilted weights.
- If the user asks about news, sentiment, "what's the market saying", headlines, or how
  news might affect a stock or their portfolio:
    - For a single ticker: call lc_ticker_sentiment(ticker, limit).
    - For "my portfolio" sentiment: call lc_portfolio_sentiment(session_id).
- Summarize the Bullish/Neutral/Bearish label and the average score, and cite a few of the
  most bullish/bearish headlines. Make clear this is news-derived sentiment, not a price target.
- You MAY combine sentiment with metrics/optimization to give a fuller, better-informed picture,
  but always frame it as educational, not personalized investment advice.

11. Portfolio File Rule (MANDATORY, REINFORCEMENT):
If the user has uploaded a portfolio file for the current session:

1. You MUST ignore any tickers or weights mentioned in conversation that conflict with the uploaded file.
2. You MUST obtain the normalized portfolio via:
     load_user_portfolio(session_id)
   This returns holdings_json.

3. You MUST derive weights automatically from that file by calling:

   lc_compute_metrics_from_portfolio(
       holdings_json=...,
       start=...,
       end=...
   )

4. You MUST NOT call lc_compute_portfolio_metrics directly when a portfolio file exists.
   You MUST NOT build weight dictionaries yourself.

5. The tickers used in the metric calculation MUST exactly equal the "tic" values
   in the user's uploaded portfolio JSON.

6. For any request involving volatility, risk, Sharpe ratio, CAGR, performance,
   or "metrics for my portfolio", ALWAYS rely on the uploaded portfolio and
   lc_compute_metrics_from_portfolio as described above.

Never guess or hallucinate tickers or weights.
"""




tools = [
    lc_load_price_history,
    lc_compute_metrics_from_portfolio,
    lc_recommend_portfolio,
    load_user_portfolio,
    compute_total_value,
    lc_price_on_date_tool,
    finrl_optimize_portfolio,
    lc_get_profile,
    portfolio_holdings_count,
    lc_ticker_sentiment,
    lc_portfolio_sentiment,
    lc_backtest_portfolio,
    lc_sentiment_tilt,
]


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FINANCE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Session id: {session_id}\n\n"
            "{input}"
        ),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# The LLM client is built lazily so importing this module doesn't require an
# OPENAI_API_KEY — only actually running a chat does. This keeps the app and the
# whole test suite importable in environments without secrets (e.g. CI).
_agent_executor = None


def _get_agent_executor():
    global _agent_executor
    if _agent_executor is None:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        agent = create_tool_calling_agent(llm, tools, prompt)
        _agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    return _agent_executor

# ---------- Persistent running memory (file-per-user) ----------

MEMORY_DIR = "chat_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

_session_store: Dict[str, ChatMessageHistory] = {}

# Age out stale on-disk chat memory so the directory can't grow without bound.
# Chat history also lives in the SQL database, so pruning an idle session's file
# only resets the agent's in-memory scratchpad for that (long-abandoned) chat.
CHAT_MEMORY_TTL_DAYS = 30
_last_prune = 0.0


def _prune_memory_dir() -> None:
    import time

    global _last_prune
    now = time.time()
    if now - _last_prune < 3600:  # at most once an hour per process
        return
    _last_prune = now
    cutoff = now - CHAT_MEMORY_TTL_DAYS * 86400
    try:
        for p in Path(MEMORY_DIR).glob("*.json"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass
    except Exception:
        pass


def _sanitize_session_id(session_id: str) -> str:
    session_id = (session_id or "anonymous").strip()
    if not session_id:
        session_id = "anonymous"
    safe = []
    for ch in session_id:
        if ch.isalnum() or ch in ("-", "_", "@", "."):
            safe.append(ch)
    return "".join(safe) or "anonymous"


def _memory_path(session_id: str) -> str:
    sid = _sanitize_session_id(session_id)
    return os.path.join(MEMORY_DIR, f"{sid}.json")


def _serialize_messages(messages):
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            continue
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, SystemMessage):
            role = "system"
        else:
            continue
        out.append({"role": role, "content": m.content})
    return out



def _deserialize_messages(data: List[dict]) -> List[BaseMessage]:
    """
    Convert JSON dicts -> LangChain message objects.
    """
    out: List[BaseMessage] = []
    for item in data:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
    return out


def _load_persistent_history(session_id: str) -> List[BaseMessage]:
    """
    Load existing history for this user/session from disk,
    or return an empty list if no file yet.
    """
    path = _memory_path(session_id)
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return _deserialize_messages(data)
    except Exception:
        return []

    return []


def _save_persistent_history(session_id: str, history: ChatMessageHistory) -> None:
    _prune_memory_dir()
    path = _memory_path(session_id)
    try:
        safe_messages = [m for m in history.messages if not isinstance(m, ToolMessage)]
        serializable = _serialize_messages(safe_messages)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
    except Exception:
        pass



def _get_session_history(session_id: str) -> ChatMessageHistory:
    """
    Return (or create) a ChatMessageHistory for this session_id.

    - First time a session_id is seen after server start:
      * Load its JSON file from disk (if any)
      * Seed the ChatMessageHistory with those messages
    - After that, reuse the in-memory history (no repeated file reads)
    """
    if session_id not in _session_store:
        history = ChatMessageHistory()
        loaded_msgs = _load_persistent_history(session_id)
        for msg in loaded_msgs:
            history.add_message(msg)
        _session_store[session_id] = history
    return _session_store[session_id]



def run_portfolio_agent(message: str, session_id: str = "default") -> str:
    session_id = _sanitize_session_id(session_id)
    history = _get_session_history(session_id)

    non_tool_history = [m for m in history.messages if not isinstance(m, ToolMessage)]
    recent = non_tool_history[-8:]

    result = _get_agent_executor().invoke(
        {
            "input": message,
            "session_id": session_id,
            "chat_history": recent,
        }
    )

    history.add_user_message(message)
    history.add_ai_message(result["output"])
    _save_persistent_history(session_id, history)

    return result["output"]



def load_portfolio_memory(session_id: str):
    """
    Legacy helper to load a portfolio JSON from disk if needed.
    (Currently portfolios are kept in SESSION_PORTFOLIOS instead.)
    """
    path = Path("chat_memory") / f"{session_id}.portfolio.json"
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None
