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

from api.langchain_tools import (
    lc_load_price_history,
    lc_compute_metrics_from_portfolio,
    lc_recommend_portfolio,
    load_user_portfolio,
    compute_total_value,
    lc_price_on_date_tool,   
)

load_dotenv()


FINANCE_SYSTEM_PROMPT = """
You are an AI Portfolio Assistant focused exclusively on personal finance and investing.

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
- You MUST NOT answer purely in natural language without calling the appropriate tool.

3. RECOMMENDATIONS (UPDATED TOOL SIGNATURE):
When the user asks for an optimized or recommended portfolio:
- Collect tickers, start, end, constraints_json.
- MUST call ONLY:

  lc_recommend_portfolio(
      tickers=[...],
      start=...,
      end=...,
      constraints_json=...
  )

- MUST NOT pass prices_json.
- MUST NOT call lc_load_price_history during recommendations.
- After the tool returns, summarize recommended weights and provide educational reasoning.

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




llm = ChatOpenAI(
    model="gpt-4o-mini",  
    temperature=0,
)

tools = [
    lc_load_price_history,
    lc_compute_metrics_from_portfolio,
    lc_recommend_portfolio,
    load_user_portfolio,
    compute_total_value,
    lc_price_on_date_tool,
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

# Build a tool-calling agent
agent = create_tool_calling_agent(llm, tools, prompt)

# Wrap it in an executor so we can just call `.invoke(...)`
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
)

# ---------- Persistent running memory (file-per-user) ----------

MEMORY_DIR = "chat_memory"
os.makedirs(MEMORY_DIR, exist_ok=True)

_session_store: Dict[str, ChatMessageHistory] = {}


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

    result = agent_executor.invoke(
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
