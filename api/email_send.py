"""
Transactional email via Resend (https://resend.com).

Only two messages are sent: email verification and password reset. The API key
is a restricted, send-only Resend key kept in .env (RESEND_API_KEY).

Configuration (all optional except the key):
  RESEND_API_KEY  - required to actually send; if unset, send_email raises.
  EMAIL_FROM      - sender identity. Defaults to Resend's shared onboarding
                    sender, which works with no domain setup but (on the free
                    tier) only delivers to your own Resend account address.
                    Set this to "Evergreen <no-reply@yourdomain.com>" once you
                    verify your domain in Resend.
  APP_BASE_URL    - public base URL used to build links in emails
                    (e.g. https://app.yourdomain.com). Defaults to localhost.
"""
from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()

RESEND_API_URL = "https://api.resend.com/emails"


class EmailError(Exception):
    """Raised when an email could not be sent."""


def _cfg() -> tuple[str, str, str]:
    """Read config at call time so .env changes don't require a code reload."""
    key = os.getenv("RESEND_API_KEY", "").strip()
    sender = os.getenv("EMAIL_FROM", "Evergreen <onboarding@resend.dev>").strip()
    base = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000").strip().rstrip("/")
    return key, sender, base


def is_configured() -> bool:
    return bool(_cfg()[0])


def send_email(to: str, subject: str, html: str) -> None:
    key, sender, _ = _cfg()
    if not key:
        raise EmailError("Email is not configured (RESEND_API_KEY is missing).")
    try:
        resp = httpx.post(
            RESEND_API_URL,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={"from": sender, "to": [to], "subject": subject, "html": html},
            timeout=20.0,
        )
    except httpx.HTTPError as e:
        raise EmailError(f"Could not reach the email service: {e}") from e
    if resp.status_code >= 400:
        raise EmailError(f"Email service error {resp.status_code}: {resp.text[:200]}")


def _shell(title: str, body_html: str, button_label: str, link: str) -> str:
    """Minimal, inline-styled email body (email clients ignore <style>/JS)."""
    return f"""\
<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:480px;margin:0 auto;padding:24px;color:#1a2b22">
  <h2 style="margin:0 0 12px">{title}</h2>
  <p style="margin:0 0 20px;line-height:1.5;color:#3a4a42">{body_html}</p>
  <a href="{link}" style="display:inline-block;background:#1f6f4e;color:#fff;text-decoration:none;
     padding:11px 20px;border-radius:8px;font-weight:600">{button_label}</a>
  <p style="margin:20px 0 0;font-size:12px;color:#8a978f;line-height:1.5">
    If the button doesn't work, copy this link into your browser:<br>
    <span style="word-break:break-all">{link}</span>
  </p>
</div>"""


def send_verification_email(to: str, raw_token: str) -> None:
    _, _, base = _cfg()
    link = f"{base}/verify?token={raw_token}"
    html = _shell(
        "Confirm your email",
        "Welcome to Evergreen. Confirm this address to finish setting up your account.",
        "Verify email",
        link,
    )
    send_email(to, "Verify your Evergreen email", html)


def send_password_reset_email(to: str, raw_token: str) -> None:
    _, _, base = _cfg()
    link = f"{base}/?reset={raw_token}"
    html = _shell(
        "Reset your password",
        "We received a request to reset your Evergreen password. This link expires in 30 minutes. "
        "If you didn't ask for this, you can safely ignore this email.",
        "Choose a new password",
        link,
    )
    send_email(to, "Reset your Evergreen password", html)
