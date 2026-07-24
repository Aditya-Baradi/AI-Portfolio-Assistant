"""One-time email secrets must not leak through HTTP request URLs."""
from __future__ import annotations

from api import email_send


def test_email_links_put_secrets_in_fragments(monkeypatch):
    bodies: list[str] = []
    monkeypatch.setenv("APP_BASE_URL", "https://app.example.test")
    monkeypatch.setattr(
        email_send,
        "send_email",
        lambda _to, _subject, html: bodies.append(html),
    )

    email_send.send_verification_email("user@example.test", "verify-secret")
    email_send.send_password_reset_email("user@example.test", "reset-secret")

    assert "#verify=verify-secret" in bodies[0]
    assert "#reset=reset-secret" in bodies[1]
    assert "?token=" not in bodies[0]
    assert "?reset=" not in bodies[1]
