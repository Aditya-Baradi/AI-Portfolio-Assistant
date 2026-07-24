"""Production policy configuration and rendering checks."""
from __future__ import annotations

from api.routers import legal

LEGAL_ENV = {
    "LEGAL_OPERATOR_NAME": "Example Operator LLC",
    "LEGAL_CONTACT_EMAIL": "privacy@example.test",
    "LEGAL_JURISDICTION": "Example Jurisdiction",
    "HOSTING_REGION": "Example Region",
    "LOG_RETENTION_DAYS": "30",
    "BACKUP_RETENTION_DAYS": "30",
    "LEGAL_REVIEW_CONFIRMED": "true",
}


def test_missing_legal_configuration_blocks_readiness(monkeypatch):
    for name in LEGAL_ENV:
        monkeypatch.delenv(name, raising=False)
    problems = legal.legal_configuration_problems()
    assert problems
    assert any("LEGAL_REVIEW_CONFIRMED" in item for item in problems)


def test_complete_reviewed_configuration_passes(monkeypatch):
    for name, value in LEGAL_ENV.items():
        monkeypatch.setenv(name, value)
    assert legal.legal_configuration_problems() == []


def test_policy_templates_are_fully_resolved(monkeypatch):
    for name, value in LEGAL_ENV.items():
        monkeypatch.setenv(name, value)
    for filename in ("TERMS.md", "PRIVACY.md"):
        rendered = legal._read_doc(filename)
        assert "{{" not in rendered
        assert "NOT CONFIGURED" not in rendered
        assert "Example Operator LLC" in rendered or filename == "PRIVACY.md"


def test_public_web_configuration_requires_explicit_https_origins(monkeypatch):
    from api import backend

    monkeypatch.setenv("APP_BASE_URL", "https://app.example.test")
    monkeypatch.setenv("APP_ORIGINS", "https://app.example.test")
    assert backend._web_configuration_problems() == []

    unsafe_cases = (
        ("*", "https://app.example.test"),
        ("http://app.example.test", "https://app.example.test"),
        ("https://user:pass@app.example.test", "https://app.example.test"),
        ("https://app.example.test/path", "https://app.example.test"),
        ("https://app.example.test", "javascript:alert(1)"),
        ("https://app.example.test", "https://app.example.test/#token"),
    )
    for origins, base_url in unsafe_cases:
        monkeypatch.setenv("APP_ORIGINS", origins)
        monkeypatch.setenv("APP_BASE_URL", base_url)
        assert backend._web_configuration_problems()
