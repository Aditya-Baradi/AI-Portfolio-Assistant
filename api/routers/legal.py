"""
Legal and transparency endpoints.

Serves the Terms, Privacy Policy and risk disclosures as both machine-readable
JSON (so the UI can render and version-check them) and rendered HTML pages the
user can link to.

`/healthz` lives here too: it is the one endpoint that reports whether the
deployment is actually configured for public use, which is a compliance
question as much as an operational one.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("evergreen.legal")
router = APIRouter(tags=["legal"])

DOCS_ROOT = Path(__file__).resolve().parents[2]

#: Bump when the Terms or Privacy Policy change materially. The UI compares
#: this against what the user last accepted and re-prompts on a change.
POLICY_VERSION = "2026-07-23"

_LEGAL_FIELDS = {
    "OPERATOR_NAME": "LEGAL_OPERATOR_NAME",
    "LEGAL_CONTACT_EMAIL": "LEGAL_CONTACT_EMAIL",
    "LEGAL_JURISDICTION": "LEGAL_JURISDICTION",
    "HOSTING_REGION": "HOSTING_REGION",
    "LOG_RETENTION_DAYS": "LOG_RETENTION_DAYS",
    "BACKUP_RETENTION_DAYS": "BACKUP_RETENTION_DAYS",
}

DISCLOSURES = {
    "version": POLICY_VERSION,
    "not_advice": (
        "Evergreen is an educational tool. Nothing it produces is investment, "
        "financial, legal or tax advice, and no output should be relied on to make "
        "an investment decision. Evergreen is not a registered investment adviser, "
        "broker-dealer, or financial planner, and no adviser-client relationship is "
        "created by using it."
    ),
    "no_recommendations": (
        "Historical analytics, retrospective model comparisons, simulations, and "
        "news-tone readings are educational illustrations. They are not "
        "recommendations to buy, sell or hold any security."
    ),
    "past_performance": (
        "Backtests and projections use historical data. Past performance does not "
        "predict future results. Backtested results are hypothetical, benefit from "
        "hindsight, and do not represent trading in a real account."
    ),
    "model_limitations": [
        (
            "Projections assume returns are independent and normally distributed. "
            "Real markets have fat tails, momentum and regime changes."
        ),
        (
            "The retirement outlook models the risk level you select, not the "
            "holdings you actually own."
        ),
        (
            "News sentiment is a lexicon score over a small number of headlines. "
            "It has no established predictive relationship with future returns."
        ),
        (
            "Model comparisons are sensitive to the estimation window and can "
            "change materially with different dates or assumptions."
        ),
    ],
    "data": (
        "Market data is supplied by third parties and may be delayed, incomplete or "
        "wrong. Evergreen does not warrant its accuracy."
    ),
    "ai": (
        "The chat assistant is a large language model. It can be confidently wrong, "
        "misread your portfolio, and misstate figures. Verify anything that matters."
    ),
}


class PolicyAcceptance(BaseModel):
    version: str
    accept_terms: bool


def legal_configuration_problems() -> list[str]:
    """Return policy configuration that must be completed before production."""
    problems = []
    for label, env_name in _LEGAL_FIELDS.items():
        value = os.getenv(env_name, "").strip()
        if not value:
            problems.append(f"{env_name} is required to render the public policies.")
            continue
        if label.endswith("_DAYS"):
            try:
                if int(value) < 1:
                    raise ValueError
            except ValueError:
                problems.append(f"{env_name} must be a positive whole number.")
    contact = os.getenv("LEGAL_CONTACT_EMAIL", "").strip()
    if contact and ("@" not in contact or len(contact) > 254):
        problems.append("LEGAL_CONTACT_EMAIL must be a valid operator contact address.")
    if os.getenv("LEGAL_REVIEW_CONFIRMED", "").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        problems.append(
            "LEGAL_REVIEW_CONFIRMED is not set. Qualified counsel must review the "
            "Terms, Privacy Policy, product behavior, and applicable investment-"
            "adviser/data-protection obligations before public launch."
        )
    return problems


def _policy_substitutions() -> dict[str, str]:
    reviewed = os.getenv("LEGAL_REVIEW_CONFIRMED", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    values = {
        key: os.getenv(env_name, "").strip() or f"NOT CONFIGURED ({env_name})"
        for key, env_name in _LEGAL_FIELDS.items()
    }
    values["LEGAL_REVIEW_NOTICE"] = (
        ""
        if reviewed
        else (
            "> **Draft deployment notice:** the operator has not confirmed legal "
            "review. Production startup is blocked until the policy fields are "
            "completed and qualified counsel has reviewed the service."
        )
    )
    return values


def _read_doc(name: str) -> str:
    path = DOCS_ROOT / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{name} is not available.")
    text = path.read_text(encoding="utf-8")
    for key, value in _policy_substitutions().items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _render(title: str, markdown_text: str) -> str:
    """
    Minimal server-side markdown -> HTML for the policy documents.

    Deliberately tiny and escaping-first: these files are trusted repo content,
    but rendering them through an escape pass keeps the habit consistent with
    the rest of the app and costs nothing.
    """
    import html
    import re

    out, in_list = [], False
    for raw in html.escape(markdown_text).split("\n"):
        line = raw.rstrip()
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        bullet = re.match(r"^\s*[-*]\s+(.*)$", line)

        if bullet and not in_list:
            out.append("<ul>")
            in_list = True
        elif in_list and not bullet:
            out.append("</ul>")
            in_list = False

        if heading:
            level = len(heading.group(1)) + 1
            out.append(f"<h{level}>{heading.group(2)}</h{level}>")
        elif bullet:
            out.append(f"<li>{bullet.group(1)}</li>")
        elif line.strip():
            out.append(f"<p>{line}</p>")
    if in_list:
        out.append("</ul>")

    body = "\n".join(out)
    body = body.replace("**", "")  # strip leftover emphasis markers
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — Evergreen</title>
<style>
  body {{ font: 16px/1.6 system-ui, sans-serif; max-width: 46rem; margin: 0 auto;
         padding: 2rem 1.25rem 4rem; color: #1a1a1a; background: #fff; }}
  h1, h2, h3 {{ line-height: 1.25; margin-top: 2rem; }}
  h1 {{ margin-top: 0; }}
  a {{ color: #0a6; }}
  code {{ background: #f4f4f5; padding: .1em .35em; border-radius: 3px; }}
  .back {{ display: inline-block; margin-bottom: 2rem; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #14161a; color: #e6e6e6; }}
    code {{ background: #23262d; }}
  }}
</style></head>
<body><a class="back" href="/">&larr; Back to Evergreen</a>{body}</body></html>"""


@router.get("/legal/disclosures")
def disclosures():
    """Machine-readable risk disclosures, rendered by the UI."""
    return DISCLOSURES


@router.get("/legal/version")
def legal_version():
    return {"version": POLICY_VERSION}


@router.post("/legal/accept")
def accept_policy(
    body: PolicyAcceptance,
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Record affirmative acceptance of exactly the policy currently served."""
    from api import db
    from api.deps import client_ip, current_user

    user = current_user(authorization)
    if not body.accept_terms:
        raise HTTPException(status_code=400, detail="Affirmative acceptance is required.")
    if body.version != POLICY_VERSION:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "policy_version_changed",
                "version": POLICY_VERSION,
                "message": "The policy changed. Reload and review the current version.",
            },
        )
    db.record_policy_acceptance(user["id"], POLICY_VERSION)
    db.log_event(user["id"], "policy_accepted", client_ip(request))
    return {"ok": True, "version": POLICY_VERSION}


@router.get("/terms", response_class=HTMLResponse)
def terms_page():
    return HTMLResponse(_render("Terms of Service", _read_doc("TERMS.md")))


@router.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return HTMLResponse(_render("Privacy Policy", _read_doc("PRIVACY.md")))


@router.get("/healthz")
def healthz():
    """
    Liveness plus a deployment-readiness summary.

    Deliberately does NOT expose secrets or internal versions — just enough for
    an operator (or the startup banner) to see whether this instance is safe to
    put in front of the public.
    """
    import os

    from api import db, market_data, state
    from api.observability import sentry_enabled
    from scripts.backup_db import backup_encryption_ready

    key_source = "environment" if os.getenv("EVERGREEN_MASTER_KEY", "").strip() \
        else "development-file"
    database_ok = False
    try:
        with db._connect() as conn:
            database_ok = conn.execute("SELECT 1").fetchone()[0] == 1
    except sqlite3.Error:
        logger.exception("Health check could not query SQLite")

    storage_path = db.data_dir()
    storage_writable = storage_path.exists() and os.access(storage_path, os.W_OK)
    state_healthy = state.get_backend().healthy()
    provider_problems = market_data.check_production_ready()
    policy_problems = legal_configuration_problems()
    from api.backend import _public_url_problem

    app_base_url = os.getenv("APP_BASE_URL", "").strip().rstrip("/")
    configured_origins = [
        origin.strip().rstrip("/")
        for origin in os.getenv("APP_ORIGINS", "").split(",")
        if origin.strip()
    ]
    public_url_ready = (
        _public_url_problem(app_base_url, "APP_BASE_URL", origin_only=True) is None
    )
    cors_ready = bool(configured_origins) and all(
        _public_url_problem(origin, "APP_ORIGINS", origin_only=True) is None
        for origin in configured_origins
    )

    checks = {
        "market_data_provider": market_data.provider_name(),
        # Code can verify adapter configuration and an operator attestation; it
        # cannot inspect the operator's executed contract or prove entitlement.
        "market_data_contract_attested": not provider_problems,
        "shared_state": "redis" if state.using_shared_store() else "memory",
        "state_backend_healthy": state_healthy,
        "database": "ok" if database_ok else "unavailable",
        "storage_writable": storage_writable,
        "encryption_key_source": key_source,
        "backup_encryption_ready": backup_encryption_ready(),
        "email_configured": bool(os.getenv("RESEND_API_KEY", "").strip()),
        "chat_configured": bool(os.getenv("OPENAI_API_KEY", "").strip()),
        "error_monitoring_configured": sentry_enabled(),
        "public_url_configured": public_url_ready,
        "cors_configured": cors_ready,
        "legal_policy_configured": not policy_problems,
        "environment": os.getenv("EVERGREEN_ENV", "development"),
    }
    checks["production_ready"] = bool(
        checks["market_data_contract_attested"]
        and key_source == "environment"
        and checks["backup_encryption_ready"]
        and checks["shared_state"] == "redis"
        and state_healthy
        and database_ok
        and storage_writable
        and checks["email_configured"]
        and checks["chat_configured"]
        and checks["error_monitoring_configured"]
        and checks["public_url_configured"]
        and checks["cors_configured"]
        and checks["legal_policy_configured"]
    )
    liveness_ok = database_ok and storage_writable and state_healthy
    payload = {
        "status": "ok" if liveness_ok else "degraded",
        "checks": checks,
        "policy_version": POLICY_VERSION,
    }
    return JSONResponse(payload, status_code=200 if liveness_ok else 503)
