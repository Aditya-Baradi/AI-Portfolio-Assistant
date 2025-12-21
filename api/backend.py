# backend.py (inside api/)

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json

from api.langchain_agent import run_portfolio_agent
from api.portfolio_core import parse_portfolio_file, SESSION_PORTFOLIOS

app = FastAPI()

# Allow your frontend (index.html) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str  # user / session identifier


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.

    The LangChain agent handles all memory (per session_id) on the backend.
    """
    answer = run_portfolio_agent(req.message, session_id=req.session_id)
    return ChatResponse(answer=answer)


# Optional: where we might store other per-session artifacts
MEMORY_DIR = Path("chat_memory")
MEMORY_DIR.mkdir(exist_ok=True)


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload a portfolio file (CSV or JSON), normalize it, and stash it
    in SESSION_PORTFOLIOS so LangChain tools can access it.
    """
    content = await file.read()

    try:
        parsed = parse_portfolio_file(file.filename, content)
    except ValueError as e:
        return {"message": f"Error parsing portfolio file: {e}"}
    except Exception:
        return {"message": "Uploaded file is not valid CSV/JSON portfolio."}

    # Store the parsed portfolio in the in-memory session map
    SESSION_PORTFOLIOS[session_id] = parsed

    # Try to count number of holdings
    if isinstance(parsed, dict):
        holdings = parsed.get("holdings", [])
        n = len(holdings)
    elif isinstance(parsed, list):
        n = len(parsed)
    else:
        n = 0

    print(f"Stored {n} holdings for session {session_id!r}")
    return {
        "message": f"Uploaded portfolio for session '{session_id}' with {n} holdings."
    }
