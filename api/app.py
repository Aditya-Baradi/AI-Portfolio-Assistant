from __future__ import annotations

import os
import json
import time
from typing import Iterable, List, Dict, Optional
from importlib import import_module

from flask import Flask, request, jsonify, Response, send_from_directory
from flask_cors import CORS


try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass


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


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
app = Flask(__name__, static_folder=None)
CORS(app)

def sse(data: dict) -> str:
    """Format one Server-Sent Events message line."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


class ChatEngine:
    name = "echo-engine"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"(echo) {last}"

    def stream_complete(self, messages: List[Dict[str, str]]) -> Iterable[str]:
        text = self.complete(messages)
        for token in text.split():
            yield token + " "
            time.sleep(0.01)

class OpenAIEngine(ChatEngine):
    """Uses OpenAI if OPENAI_API_KEY is set."""
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
            if delta and getattr(delta, "content", None):
                yield delta.content

def get_engine() -> ChatEngine:
    if os.getenv("OPENAI_API_KEY"):
        try:
            return OpenAIEngine()
        except Exception:
            
            return ChatEngine()
    return ChatEngine()

ENGINE: ChatEngine = get_engine()


def try_import(name: str):
    try:
        return import_module(name)
    except Exception:
        return None

gpt_agent = try_import("gpt_agent")  


@app.get("/")
def serve_front_page():
    """Serve your front-end file from repo root."""
    return send_from_directory(ROOT_DIR, "front_page.html")

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok", "engine": ENGINE.name})

@app.post("/v1/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    raw = data.get("messages") or []
    stream = bool(data.get("stream", False))

    
    messages: List[Dict[str, str]] = [{"role": "system", "content": FINANCE_SYSTEM_PROMPT}]

    
    for m in raw:
        if isinstance(m, dict) and "role" in m and "content" in m:
            messages.append({"role": str(m["role"]), "content": str(m["content"])})
        else:
            messages.append({"role": "user", "content": str(m)})

    
    if gpt_agent:
        for cand in ("generate_response", "chat", "respond", "main"):
            if hasattr(gpt_agent, cand):
                fn = getattr(gpt_agent, cand)
                try:
                    if stream:
                        
                        result = fn(messages)
                        if hasattr(result, "__iter__") and not isinstance(result, str):
                            def gen_iter():
                                try:
                                    for piece in result:
                                        yield sse({"delta": str(piece)})
                                    yield sse({"event": "done"})
                                except Exception as e:
                                    yield sse({"error": str(e)})
                            return Response(gen_iter(), mimetype="text/event-stream")
                        
                        def gen_one():
                            try:
                                yield sse({"delta": str(result)})
                                yield sse({"event": "done"})
                            except Exception as e:
                                yield sse({"error": str(e)})
                        return Response(gen_one(), mimetype="text/event-stream")
                    else:
                        out = fn(messages)
                        if isinstance(out, str):
                            return jsonify({"reply": out, "model": f"gpt_agent.{cand}"})
                        return jsonify({"reply": json.dumps(out), "model": f"gpt_agent.{cand}"})
                except Exception:
                    
                    pass

    
    if stream:
        def gen():
            try:
                for piece in ENGINE.stream_complete(messages):
                    yield sse({"delta": piece})
                yield sse({"event": "done"})
            except Exception as e:
                yield sse({"error": str(e)})
        return Response(gen(), mimetype="text/event-stream")
    else:
        try:
            reply = ENGINE.complete(messages)
            return jsonify({"reply": reply, "model": ENGINE.name})
        except Exception as e:
            return jsonify({"error": "engine_error", "message": str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=True)
