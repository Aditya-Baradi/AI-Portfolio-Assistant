"""
The chat advisor endpoints.

Chat requires a VERIFIED account: every message spends real model credit, so an
unverified signup must not be able to use it as free inference.
"""
from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Cookie, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field

from api import db
from api.deps import (
    CHAT_DAILY_LIMIT,
    SESSION_COOKIE_NAME,
    agent_session_id,
    check_api_rate,
    verified_user,
)
from api.langchain_agent import run_portfolio_agent
from api.observability import report

logger = logging.getLogger("evergreen.chat")
router = APIRouter(tags=["chat"])

MAX_MESSAGE_CHARS = 4000
CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_CHARS)
    chat_id: str = Field(max_length=64)


@router.post("/chat")
def chat(
    req: ChatRequest,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    """
    Declared sync so FastAPI runs it in a worker thread — a slow agent call
    or market-data call doesn't block the event loop for other users.
    """
    check_api_rate(request, "chat")
    user = verified_user(authorization, session_cookie, response)
    if not req.message.strip():
        raise HTTPException(status_code=422, detail="Empty message.")
    chat_id = req.chat_id.strip()
    if not CHAT_ID_RE.fullmatch(chat_id):
        raise HTTPException(status_code=422, detail="Invalid chat_id.")

    # Ownership check, chat creation and quota reservation are one write
    # transaction, so cross-account ids and concurrent cap bypasses fail before
    # an external model call can incur cost.
    try:
        used = db.begin_chat_turn(
            user["id"],
            chat_id,
            req.message,
            CHAT_DAILY_LIMIT,
        )
    except db.ChatOwnershipError:
        raise HTTPException(status_code=404, detail="Chat not found.") from None
    except db.ChatQuotaError:
        raise HTTPException(
            status_code=429,
            detail=(f"You've hit today's limit of {CHAT_DAILY_LIMIT} messages. "
                    f"It resets at midnight UTC."),
        ) from None

    sid = agent_session_id(user["id"], chat_id)

    # The current user message is intentionally persisted only after the agent
    # completes. A cold in-process cache therefore loads prior turns from the
    # DB and cannot include the same current message twice.
    try:
        answer = run_portfolio_agent(
            req.message,
            session_id=sid,
            user_id=user["id"],
            chat_id=chat_id,
        )
    except Exception as e:
        report(
            logger,
            "Chat model turn failed",
            e,
            user_id=user["id"],
            chat_id=chat_id,
        )
        raise HTTPException(
            status_code=502,
            detail="The assistant is temporarily unavailable. Please try again later.",
        ) from None
    try:
        db.append_chat_turn(user["id"], chat_id, req.message, answer)
    except db.ChatOwnershipError:
        # The chat was deleted or ownership state changed while the external
        # call ran. Never recreate it implicitly after that boundary.
        raise HTTPException(status_code=404, detail="Chat not found.") from None
    logger.info("Chat turn served",
                extra={"user_id": user["id"], "messages_today": used})
    return {"answer": answer, "messages_remaining": max(0, CHAT_DAILY_LIMIT - used)}


@router.get("/chats")
def chats(
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    user = verified_user(authorization, session_cookie, response)
    return {"chats": db.list_chats(user["id"])}


@router.get("/chats/{chat_id}/messages")
def chat_messages(
    chat_id: str,
    response: Response,
    authorization: str | None = Header(default=None),
    session_cookie: str | None = Cookie(default=None, alias=SESSION_COOKIE_NAME),
):
    user = verified_user(authorization, session_cookie, response)
    try:
        messages = db.get_messages_for_user(user["id"], chat_id)
    except db.ChatOwnershipError:
        raise HTTPException(status_code=404, detail="Chat not found.")
    return {"messages": messages}
