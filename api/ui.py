from __future__ import annotations
from dotenv import load_dotenv

load_dotenv()

import os
import json
import time
from typing import Iterable, List, Dict, Optional

import streamlit as st


FINANCE_SYSTEM_PROMPT = """
You are AI Portfolio Assistant focused exclusively on personal finance and investing.

Allowed scope:
- Markets and instruments: stocks, bonds, ETFs, indexes, mutual funds, options (high level), commodities, FX.
- Portfolio construction: asset allocation, diversification, CAPM, factor tilts, rebalancing, efficient frontier, Black-Litterman (high level).
- Risk & performance: Sharpe, Sortino, beta/alpha, volatility, drawdown, VaR/CVaR, CAGR, turnover, fees.
- Backtesting & evaluation: walk-forward, train/test splits, benchmark comparisons (e.g., S&P 500).

Out of scope:
- Non-financial topics; medical/legal/tax advice beyond high-level education; trading signals or guarantees; sensitive/unsafe requests.

Tone & format:
- Be concise, educational, and cautious.
- When advice sounds personalized, add: “This is educational information, not investment advice.”
- Prefer short bullets + an action checklist when appropriate.
"""



MEMORY_DIR = os.getenv("CHAT_MEMORY_DIR", "chat_memory")
os.makedirs(MEMORY_DIR, exist_ok=True)
MAX_HISTORY = 200  


def sanitize_user_id(user_id: str) -> str:
    """Keep only safe chars for filenames; fallback to 'anonymous'."""
    user_id = user_id.strip()
    if not user_id:
        return "anonymous"
    safe = []
    for ch in user_id:
        if ch.isalnum() or ch in ("-", "_", "@", "."):
            safe.append(ch)
    return "".join(safe) or "anonymous"


def memory_path_for_user(user_id: str) -> str:
    user_id = sanitize_user_id(user_id)
    return os.path.join(MEMORY_DIR, f"{user_id}.json")


def load_persistent_history(user_id: str) -> List[Dict[str, str]]:
    """Load chat history for a specific user from disk, or return default greeting."""
    path = memory_path_for_user(user_id)
    if not os.path.exists(path):
        return [
            {
                "role": "assistant",
                "content": "Hello! I’m your AI financial assistant. How can I help you today?",
            }
        ]
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and all(
            isinstance(m, dict) and "role" in m and "content" in m for m in data
        ):
            return data
    except Exception:
        pass

    return [
        {
            "role": "assistant",
            "content": "Hello! I’m your AI financial assistant. How can I help you today?",
        }
    ]


def save_persistent_history(user_id: str, history: List[Dict[str, str]]) -> None:
    """Save chat history for a specific user to disk."""
    path = memory_path_for_user(user_id)
    try:
        trimmed = history[-MAX_HISTORY:]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(trimmed, f, ensure_ascii=False, indent=2)
    except Exception:
        
        pass




class ChatEngine:
    name = "echo-engine"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        last = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return f"(echo) {last}"

    def stream_complete(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        text = self.complete(messages)
        for token in text.split():
            yield token + " "
            time.sleep(0.01)


class OpenAIEngine(ChatEngine):
    """Uses OpenAI 1.x/2.x client only."""

    def __init__(self, model: Optional[str] = None):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.name = f"openai:{self.model}"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
        )
        return resp.choices[0].message.content or ""

    def stream_complete(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content




def get_engine() -> ChatEngine:
    key = os.getenv("OPENAI_API_KEY")
    print("DEBUG: has OPENAI_API_KEY?", bool(key))

    if not key:
        return ChatEngine()

    eng = OpenAIEngine()
    print("DEBUG: Using engine:", eng.name)
    return eng




ENGINE: ChatEngine = get_engine()
gpt_agent = None  




def generate_stream(user_message: str) -> Iterable[str]:
    """
    Yield response chunks (strings) given a new user message.
    Currently always uses ENGINE.stream_complete (no gpt_agent).
    """
    messages: List[Dict[str, str]] = [{"role": "system", "content": FINANCE_SYSTEM_PROMPT}]

    for m in st.session_state.chat_history:
        messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        for piece in ENGINE.stream_complete(messages):
            yield piece
    except Exception as e:
        yield f"[engine_error: {e}]"




st.set_page_config(
    page_title="AI Portfolio Assistant",
    page_icon="💹",
    layout="centered",
)


with st.sidebar:
    st.markdown("### AI Portfolio Assistant")

    default_user_id = st.session_state.get("user_id", "anonymous")
    user_id_input = st.text_input(
        "User ID",
        value=default_user_id,
        help="Each ID gets its own chat memory (e.g., email, username, or alias).",
    )
    user_id = sanitize_user_id(user_id_input)
    st.session_state.user_id = user_id

    st.caption(
        f"Engine: **`{ENGINE.name}`**\n\n"
        f"Memory file: `chat_memory/{user_id}.json`"
    )

    if "chat_history" not in st.session_state or st.session_state.get(
        "_history_user_id"
    ) != user_id:
        st.session_state.chat_history = load_persistent_history(user_id)
        st.session_state._history_user_id = user_id

    if st.button("🧹 Clear conversation"):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hello! I’m your AI financial assistant. How can I help you today?",
            }
        ]
        st.session_state._history_user_id = user_id
        save_persistent_history(user_id, st.session_state.chat_history)
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Upload a file")
    uploaded_file = st.file_uploader(
        "Upload portfolio / data file",
        type=["txt", "md", "json", "csv"],
        label_visibility="collapsed",
    )
    send_file = st.button("Send uploaded file to assistant")


st.markdown(
    """
    <style>
    .big-title { font-size: 1.4rem; font-weight: 600; margin-bottom: 0.25rem; }
    .muted { color: #9ca3af; font-size: 0.9rem; }
    </style>
    <div class="big-title">AI Portfolio Assistant</div>
    <div class="muted">
      Ask about allocation, risk, backtests, or your holdings.
      This is educational information, not investment advice.
    </div>
    <hr style="opacity:0.2">
    """,
    unsafe_allow_html=True,
)


for msg in st.session_state.chat_history:
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.markdown(msg["content"])


if uploaded_file is not None and send_file:
    try:
        content = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        content = ""

    user_msg = f"Uploaded file: **{uploaded_file.name}**"
    st.session_state.chat_history.append({"role": "user", "content": user_msg})
    save_persistent_history(user_id, st.session_state.chat_history)

    with st.chat_message("user"):
        st.markdown(user_msg)

    file_prompt = f"File name: {uploaded_file.name}\n\n{content}"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for delta in generate_stream(file_prompt):
            full_response += delta
            placeholder.markdown(full_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
    save_persistent_history(user_id, st.session_state.chat_history)
    st.experimental_rerun()


user_input = st.chat_input(f"[{user_id}] Message AI Portfolio Assistant…")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    save_persistent_history(user_id, st.session_state.chat_history)

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        for delta in generate_stream(user_input):
            full_response += delta
            placeholder.markdown(full_response)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": full_response}
    )
    save_persistent_history(user_id, st.session_state.chat_history)
