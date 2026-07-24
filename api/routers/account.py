"""Account self-service: profile name, password change, data export, deletion."""
from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api import db
from api.deps import (
    LOCKOUT_THRESHOLD,
    assert_not_locked,
    check_auth_rate,
    clear_login_failures,
    client_ip,
    current_user,
    record_login_failure,
)
from api.observability import report

logger = logging.getLogger("evergreen.account")
router = APIRouter(tags=["account"])


class NameBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class ChangePassword(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=1, max_length=256)


class PasswordBody(BaseModel):
    password: str = Field(min_length=1, max_length=256)


def _record_sensitive_reauth_failure(email: str) -> None:
    if record_login_failure(email) >= LOCKOUT_THRESHOLD:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many failed verification attempts for this account. "
                "Try again later or reset your password."
            ),
        )


@router.get("/me")
def me(response: Response, authorization: str | None = Header(default=None)):
    user = current_user(authorization, response=response)
    from api.routers.legal import POLICY_VERSION

    return {
        "email": user["email"],
        "name": user.get("name"),
        "has_portfolio": db.get_portfolio(user["id"]) is not None,
        "twofa_enabled": db.get_totp_secret(user["id"]) is not None,
        "email_verified": db.is_email_verified(user["id"]),
        "policy_version": POLICY_VERSION,
        "policy_accepted": db.has_policy_acceptance(user["id"], POLICY_VERSION),
    }


@router.post("/account/name")
def account_name(body: NameBody, authorization: str | None = Header(default=None)):
    """Set or change the display name shown in the app."""
    user = current_user(authorization)
    name = body.name.strip()[:60]
    if not name:
        raise HTTPException(status_code=400, detail="Please enter a name.")
    db.set_user_name(user["id"], name)
    return {"ok": True, "name": name}


@router.post("/auth/change-password")
def change_password(
    body: ChangePassword,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
):
    """Change the password, revoke every old session, and start a new one."""
    user = current_user(authorization)
    check_auth_rate(request)
    assert_not_locked(user["email"])
    try:
        db.change_password(user["id"], body.current_password, body.new_password)
    except db.AuthError as e:
        if str(e).startswith("Current password"):
            _record_sensitive_reauth_failure(user["email"])
            raise HTTPException(status_code=401, detail="Current password is incorrect.")
        raise HTTPException(status_code=400, detail=str(e))
    clear_login_failures(user["email"])
    db.log_event(user["id"], "password_changed", client_ip(request))
    token = db.issue_token(user["id"])
    # Browser clients deliberately cannot read or persist session tokens. Give
    # them the replacement through the hardened cookie while retaining the JSON
    # value for existing explicit-Bearer API clients.
    from api.routers.auth import _set_session_cookie

    _set_session_cookie(response, token)
    payload: dict[str, object] = {"ok": True}
    if not getattr(request.state, "auth_from_cookie", False):
        payload["token"] = token
    return payload


@router.get("/account/export")
def account_export(authorization: str | None = Header(default=None)):
    """Everything the app stores about this user, as one JSON download."""
    user = current_user(authorization)
    chats = db.list_chats(user["id"])
    for c in chats:
        owned_loader = getattr(db, "get_messages_for_user", None)
        c["messages"] = (
            owned_loader(user["id"], c["id"])
            if owned_loader is not None
            else db.get_messages(c["id"])
        )
    data = {
        "email": user["email"],
        "name": user.get("name"),
        "email_verified": db.is_email_verified(user["id"]),
        "profile": db.get_profile(user["id"]),
        "portfolio": db.get_portfolio(user["id"]),
        "watchlist": db.list_watchlist(user["id"]),
        "alerts": db.list_alerts(user["id"], limit=50),
        "chats": chats,
        "recent_activity": db.recent_events(user["id"], 100),
    }
    acceptance_loader = getattr(db, "list_policy_acceptances", None)
    if acceptance_loader is not None:
        data["policy_acceptances"] = acceptance_loader(user["id"])
    return JSONResponse(
        content=data,
        headers={"Content-Disposition":
                 'attachment; filename="portfolio-assistant-export.json"'},
    )


@router.post("/account/delete")
def account_delete(
    body: PasswordBody,
    request: Request,
    response: Response,
    authorization: str | None = Header(default=None),
):
    """Delete the account and everything owned by it. Requires the password."""
    user = current_user(authorization)
    check_auth_rate(request)
    assert_not_locked(user["email"])
    if not db.verify_password(user["id"], body.password):
        _record_sensitive_reauth_failure(user["email"])
        raise HTTPException(status_code=401, detail="Wrong password.")
    clear_login_failures(user["email"])

    # Remove per-chat memory files for this user (named u{id}.c{chat}...).
    try:
        from api.langchain_agent import MEMORY_DIR, forget_user_sessions

        forget_user_sessions(user["id"])
        for p in Path(MEMORY_DIR).glob(f"u{user['id']}.*"):
            p.unlink(missing_ok=True)
    except (ImportError, OSError) as e:
        report(logger, "Could not fully clear chat memory on account delete", e,
               user_id=user["id"])

    db.delete_user(user["id"])
    from api.routers.auth import _clear_session_cookie

    _clear_session_cookie(response)
    logger.info("Account deleted", extra={"user_id": user["id"]})
    return {"ok": True}


@router.get("/account/activity")
def account_activity(authorization: str | None = Header(default=None)):
    user = current_user(authorization)
    return {"events": db.recent_events(user["id"], 20)}
