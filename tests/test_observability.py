"""Structured logging and outbound telemetry privacy guarantees."""
from __future__ import annotations

import json
import logging

from api import observability


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


def test_report_preserves_exception_and_redacts_secrets():
    logger = logging.getLogger("test.observability.report")
    logger.propagate = False
    capture = _Capture()
    logger.handlers = [capture]
    logger.setLevel(logging.INFO)

    error = ValueError("provider failed")
    observability.report(
        logger,
        "degraded",
        error,
        api_key="never-log-this",
        rows=12,
        message="reserved-name-is-prefixed",
    )

    record = capture.records[0]
    assert record.exc_info == (ValueError, error, None)
    assert record.api_key == "[REDACTED]"
    assert record.rows == 12
    assert record.context_message == "reserved-name-is-prefixed"


def test_json_formatter_includes_correlation_without_secrets():
    observability.bind_request("request-123")
    observability.bind_user(42)
    record = logging.LogRecord(
        "evergreen.test",
        logging.WARNING,
        __file__,
        1,
        "fallback active",
        (),
        None,
    )
    record.degraded = True
    body = json.loads(observability.JsonFormatter().format(record))
    assert body["request_id"] == "request-123"
    assert body["user_id"] == "42"
    assert body["degraded"] is True


def test_json_formatter_scrubs_secrets_from_message_and_traceback():
    secret_key = "polygon-secret-value"
    bearer = "bearer-secret-value"
    cookie = "session-cookie-value"
    try:
        raise RuntimeError(
            "provider failed at "
            f"https://example.invalid/data?apiKey={secret_key} "
            f"Authorization: Bearer {bearer} Cookie={cookie}"
        )
    except RuntimeError as error:
        record = logging.LogRecord(
            "evergreen.test",
            logging.ERROR,
            __file__,
            1,
            "request failed: %s",
            (error,),
            (type(error), error, error.__traceback__),
        )

    rendered = observability.JsonFormatter().format(record)
    assert secret_key not in rendered
    assert bearer not in rendered
    assert cookie not in rendered
    assert "[REDACTED]" in rendered


def test_scrubber_handles_quoted_json_headers():
    raw = (
        '{"Authorization":"Bearer json-bearer-secret",'
        '"Cookie":"session=json-cookie-secret",'
        '"Set-Cookie":"session=json-set-cookie-secret",'
        '"apiKey":"json-query-secret"}'
    )
    scrubbed = observability._scrub_text(raw)
    for secret in (
        "json-bearer-secret",
        "json-cookie-secret",
        "json-set-cookie-secret",
        "json-query-secret",
    ):
        assert secret not in scrubbed
    assert scrubbed.count("[REDACTED]") >= 4


def test_sentry_event_scrubber_drops_payloads_and_credentials():
    event = {
        "request": {
            "data": {"portfolio": ["AAPL"]},
            "cookies": {"session": "secret"},
            "headers": {
                "Authorization": "Bearer secret",
                "Content-Type": "application/json",
            },
        },
        "user": {"id": "7", "email": "person@example.com", "ip_address": "127.0.0.1"},
        "extra": {"access_token": "secret", "provider": "tiingo"},
        "exception": {
            "values": [
                {
                    "type": "RuntimeError",
                    "value": "GET https://x.invalid/?apiKey=sentry-secret",
                }
            ]
        },
    }
    scrubbed = observability._scrub_sentry_event(event, None)
    assert "data" not in scrubbed["request"]
    assert "cookies" not in scrubbed["request"]
    assert scrubbed["request"]["headers"]["Authorization"] == "[REDACTED]"
    assert scrubbed["request"]["headers"]["Content-Type"] == "application/json"
    assert scrubbed["user"] == {"id": "7"}
    assert scrubbed["extra"]["access_token"] == "[REDACTED]"
    assert "sentry-secret" not in scrubbed["exception"]["values"][0]["value"]


def test_sample_rate_validation(monkeypatch):
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "1.1")
    try:
        observability._sample_rate("SENTRY_TRACES_SAMPLE_RATE")
    except ValueError as exc:
        assert "0 to 1" in str(exc)
    else:
        raise AssertionError("out-of-range sample rate was accepted")
