import os
import sys
import re
import time
import json
import sqlite3
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------
# Optional OpenAI dependency
# ---------------------------
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:
    OPENAI_AVAILABLE = False
    OpenAI = None


# ---------------------------
# SQLite persistence layer
# ---------------------------
class ChatDB:
    def __init__(self, path: str = "chatbot_memory.db"):
        self.path = path
        self._ensure_schema()

    def _ensure_schema(self):
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,       -- 'user' | 'assistant' | 'system'
                    content TEXT NOT NULL
                )
                """
            )
            con.commit()
        finally:
            con.close()

    def save_message(self, session_id: str, role: str, content: str, ts: Optional[float] = None):
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            cur.execute(
                "INSERT INTO messages (ts, session_id, role, content) VALUES (?, ?, ?, ?)",
                (ts or time.time(), session_id, role, content),
            )
            con.commit()
        finally:
            con.close()

    def load_history(self, session_id: str, limit: int = 20) -> List[Tuple[str, str]]:
        """
        Returns a list of (role, content) for the latest `limit` messages in this session.
        """
        con = sqlite3.connect(self.path)
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT role, content FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            )
            rows = cur.fetchall()
            rows.reverse()
            return rows
        finally:
            con.close()


# ---------------------------
# Prompt templates
# ---------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant for a college STEM student. "
    "Be practical and step-by-step when needed. If the question is unclear, "
    "ask one targeted clarifying question. If the query seems out of scope, say so politely."
)

CHAT_PROMPT_TEMPLATE = """\
[System]
{system}

[Conversation]
{history}

[User]
{user_input}

[Assistant]
"""

def render_history(turns: List[Tuple[str, str]]) -> str:
    """Turn [(role, content), ...] into a readable block for the prompt."""
    lines = []
    for role, text in turns:
        role_cap = role.capitalize()
        lines.append(f"{role_cap}: {text}")
    return "\n".join(lines)


# ---------------------------
# Fallback detection
# ---------------------------
FALLBACK_PATTERNS = [
    r"\bI(?:\s*do\s*not|\s*don't)\s*know\b",
    r"\bI(?:\s*am|\s*'m)\s*not\s*sure\b",
    r"\bno\s+idea\b",
    r"\bnot\s+enough\s+information\b",
]

def needs_fallback(answer: str) -> bool:
    if not answer or len(answer.strip()) < 3:
        return True
    for pat in FALLBACK_PATTERNS:
        if re.search(pat, answer, flags=re.IGNORECASE):
            return True
    return False


# ---------------------------
# Simple rule-based demo model (no API required)
# ---------------------------
def rule_based_response(user_input: str) -> str:
    text = user_input.strip().lower()
    if any(k in text for k in ["hello", "hi", "hey"]):
        return "Hey! How can I help with your coursework or projects today?"
    if "derivative" in text and "x^2" in text.replace(" ", ""):
        return "The derivative of x^2 is 2x."
    if "sharpe" in text:
        return "Sharpe ratio = (portfolio return − risk-free rate) / portfolio volatility."
    if "sql" in text and "join" in text:
        return "Common SQL joins: INNER, LEFT, RIGHT, FULL; INNER returns rows with matching keys on both tables."
    if "help" in text:
        return "Sure — tell me the problem and any constraints, and I'll break it down step-by-step."
    # Default generic
    return "I'm here! Ask me about math, programming, or your AI Portfolio Assistant. If it's something else, give me a hint."


# ---------------------------
# MCQ test bank
# ---------------------------
MCQ_QUESTIONS = [
    {
        "id": 1,
        "question": "Which SQL join returns only rows with matching keys in both tables?",
        "choices": ["A) LEFT JOIN", "B) RIGHT JOIN", "C) INNER JOIN", "D) FULL OUTER JOIN"],
        "answer": "C",
        "explanation": "INNER JOIN keeps only matches on both sides."
    },
    {
        "id": 2,
        "question": "The derivative of x^2 is:",
        "choices": ["A) x", "B) 2x", "C) x^3", "D) 2"],
        "answer": "B",
        "explanation": "d/dx (x^2) = 2x."
    },
    {
        "id": 3,
        "question": "Sharpe ratio is defined as:",
        "choices": [
            "A) Portfolio return / volatility",
            "B) (Portfolio return − risk-free rate) / volatility",
            "C) Risk-free rate / portfolio return",
            "D) Alpha / beta"
        ],
        "answer": "B",
        "explanation": "Sharpe uses excess return divided by volatility."
    },
    {
        "id": 4,
        "question": "In Big-O, binary search runs in:",
        "choices": ["A) O(n)", "B) O(n log n)", "C) O(log n)", "D) O(1)"],
        "answer": "C",
        "explanation": "Binary search halves the search space each step => logarithmic."
    },
    {
        "id": 5,
        "question": "Which HTTP status means 'Not Found'?",
        "choices": ["A) 200", "B) 301", "C) 400", "D) 404"],
        "answer": "D",
        "explanation": "404 Not Found."
    }
]


# ---------------------------
# Chatbot core
# ---------------------------
@dataclass
class Chatbot:
    session_id: str = "default"
    db_path: str = "chatbot_memory.db"
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    model: str = "gpt-4o-mini"  # name is ignored in rule-based mode
    max_history_turns: int = 12
    _db: ChatDB = field(init=False)
    _client: Optional[object] = field(init=False, default=None)
    _use_llm: bool = field(init=False, default=False)

    def __post_init__(self):
        self._db = ChatDB(self.db_path)
        api_key = os.getenv("OPENAI_API_KEY")
        self._use_llm = bool(api_key) and OPENAI_AVAILABLE
        if self._use_llm:
            try:
                self._client = OpenAI(api_key=api_key)
            except Exception:
                self._use_llm = False
                self._client = None

    # --- Memory helpers ---
    def _get_history(self) -> List[Tuple[str, str]]:
        return self._db.load_history(self.session_id, limit=self.max_history_turns)

    def _append(self, role: str, content: str):
        self._db.save_message(self.session_id, role, content)

    # --- Prompting ---
    def _build_prompt(self, user_input: str) -> str:
        history = self._get_history()
        history_text = render_history(history)
        return CHAT_PROMPT_TEMPLATE.format(
            system=self.system_prompt, history=history_text, user_input=user_input
        )

    # --- Generation ---
    def _generate(self, user_input: str) -> str:
        if not self._use_llm:
            return rule_based_response(user_input)

        # LLM path
        prompt = self._build_prompt(user_input)
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"{render_history(self._get_history())}\n\nUser: {user_input}"},
                ],
                temperature=0.5,
                max_tokens=400,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text
        except Exception as e:
            return f"(LLM error: {e})"

    # --- Public API ---
    def ask(self, user_input: str) -> str:
        self._append("user", user_input)
        answer = self._generate(user_input)

        if needs_fallback(answer):
            answer = (
                "I might not have enough context to answer that well. "
                "Could you add a bit more detail (topic, goal, constraints)? "
                "Examples: 'SQL join for X vs Y', 'Sharpe ratio in my backtest', or 'Debugging C segmentation fault'."
            )

        self._append("assistant", answer)
        return answer

    def run_console(self):
        print("Chatbot Core Prototype")
        print("----------------------")
        print(f"Session: {self.session_id} | DB: {self.db_path}")
        print("Type 'exit' to quit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if user_input.lower() in {"exit", "quit"}:
                print("Bye!")
                break
            reply = self.ask(user_input)
            print(f"Bot: {reply}\n")


# ---------------------------
# Test runner
# ---------------------------
def run_mcq_test() -> int:
    print("Running Multiple-Choice Test\n-----------------------------")
    score = 0
    for q in MCQ_QUESTIONS:
        print(f"Q{q['id']}. {q['question']}")
        for choice in q["choices"]:
            print("   ", choice)
        ans = input("Your answer (A/B/C/D): ").strip().upper()
        correct = ans == q["answer"]
        if correct:
            score += 1
            print("✅ Correct!\n")
        else:
            print(f"❌ Incorrect. Correct answer: {q['answer']} — {q['explanation']}\n")
    print(f"Final score: {score}/{len(MCQ_QUESTIONS)}")
    return score


# ---------------------------
# CLI
# ---------------------------
def parse_args(argv: List[str]):
    p = argparse.ArgumentParser(description="Chatbot Core Prototype")
    p.add_argument("--db", type=str, default="chatbot_memory.db", help="Path to SQLite DB file")
    p.add_argument("--session", type=str, default="default", help="Session ID to thread conversations")
    p.add_argument("--test", action="store_true", help="Run the MCQ test suite")
    p.add_argument("--model", type=str, default="gpt-4o-mini", help="OpenAI chat model name (if API available)")
    return p.parse_args(argv)


def main(argv: List[str] = None):
    args = parse_args(argv or sys.argv[1:])
    bot = Chatbot(session_id=args.session, db_path=args.db, model=args.model)

    if args.test:
        _ = run_mcq_test()
        return 0

    bot.run_console()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
