"""
Structured logging, request correlation, and optional error monitoring.

The app previously used bare `logging` plus a lot of `except: pass`. That keeps
the UI alive when a data source hiccups, which is the right product call, but it
also means failures are invisible. This module makes the silent paths audible
without making them fatal:

- JSON log lines (one object per line) so a log shipper can actually index them.
- A request id on every request, echoed in the `X-Request-ID` response header
  and attached to every log record emitted while handling it.
- `report()` - the function the best-effort `except` blocks call instead of
  swallowing an exception outright. It logs with full context and forwards to
  Sentry when configured, but never raises.

Sentry configuration is optional in development: the SDK ships with the
production dependency set, and setting SENTRY_DSN enables it. The production
startup audit requires initialization to succeed so a malformed DSN cannot
silently disable error monitoring.

Environment
-----------
LOG_LEVEL         default INFO
LOG_FORMAT        "json" (default) or "text" for readable local development
SENTRY_DSN        enables error monitoring when set
EVERGREEN_ENV     environment tag attached to logs and Sentry events
"""
from __future__ import annotations

import contextvars
import json
import logging
import os
import re
import sys
import time
import uuid

#: Set per request by the middleware; read by the log formatter.
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
user_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("user_id", default="-")

_sentry = None
_configured = False
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "session",
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|password|"
    r"client[_-]?secret|authorization|cookie|set-cookie|x-api-key)"
    r"(\s*[\"']?\s*[:=]\s*[\"']?)([^&\s,;\"']+)"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+")
_AUTH_HEADER = re.compile(
    r"(?i)\b(authorization|proxy-authorization)\b[\"']?\s*[:=]\s*[\"']?"
    r"(?:Bearer|Basic)?\s*[^\s,;}\"']+"
)
_COOKIE_HEADER = re.compile(
    r"(?i)\b(set-cookie|cookie)\b[\"']?\s*[:=]\s*[\"']?[^\r\n,}\"']+"
)
_URL_CREDENTIAL = re.compile(r"(://[^:/@\s]+:)([^@/\s]+)(@)")
_OPENAI_KEY = re.compile(r"\bsk-[A-Za-z0-9_-]{10,}")

# Fields the stdlib puts on every LogRecord; anything else was passed as
# `extra=` by the caller and belongs in the structured output.
_STD_ATTRS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with request/user correlation baked in."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) +
                  f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": _scrub_text(record.getMessage()),
            "request_id": request_id_var.get(),
            "env": os.getenv("EVERGREEN_ENV", "development"),
        }
        uid = user_id_var.get()
        if uid and uid != "-":
            payload["user_id"] = uid
        if record.exc_info:
            payload["exception"] = _scrub_text(self.formatException(record.exc_info))
        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                value = _scrub_value(value)
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        return json.dumps(payload, ensure_ascii=False)


class RedactingTextFormatter(logging.Formatter):
    """Human-readable formatter with the same credential scrubbing as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        return _scrub_text(super().format(record))


def _scrub_text(value: str) -> str:
    """Redact common credential forms from messages, URLs, and tracebacks."""
    text = str(value)
    text = _AUTH_HEADER.sub(
        lambda match: f"{match.group(1)}: {_REDACTED}",
        text,
    )
    text = _COOKIE_HEADER.sub(
        lambda match: f"{match.group(1)}: {_REDACTED}",
        text,
    )
    text = _BEARER_TOKEN.sub(f"Bearer {_REDACTED}", text)
    text = _SECRET_ASSIGNMENT.sub(
        lambda match: f"{match.group(1)}{match.group(2)}{_REDACTED}",
        text,
    )
    text = _URL_CREDENTIAL.sub(
        lambda match: f"{match.group(1)}{_REDACTED}{match.group(3)}",
        text,
    )
    return _OPENAI_KEY.sub(_REDACTED, text)


def _scrub_value(value):
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        clean = {}
        for key, item in value.items():
            if any(part in str(key).lower() for part in _SENSITIVE_KEY_PARTS):
                clean[key] = _REDACTED
            else:
                clean[key] = _scrub_value(item)
        return clean
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_value(item) for item in value)
    return value


def _safe_context(context: dict) -> dict:
    """Make structured context safe for LogRecord and redact obvious secrets."""
    safe = {}
    for raw_key, value in context.items():
        key = str(raw_key)
        lowered = key.lower()
        if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            value = _REDACTED
        else:
            value = _scrub_value(value)
        # logging rejects attempts to overwrite built-in LogRecord attributes.
        if key in _STD_ATTRS:
            key = f"context_{key}"
        safe[key] = value
    return safe


def _scrub_sentry_event(event, _hint):
    """Remove financial/request payloads even if an SDK default changes."""
    request = event.get("request")
    if isinstance(request, dict):
        request.pop("data", None)
        request.pop("cookies", None)
        headers = request.get("headers")
        if isinstance(headers, dict):
            request["headers"] = {
                key: (_REDACTED if any(
                    part in str(key).lower() for part in _SENSITIVE_KEY_PARTS
                ) else value)
                for key, value in headers.items()
            }
    user = event.get("user")
    if isinstance(user, dict):
        event["user"] = {"id": user["id"]} if "id" in user else {}
    extra = event.get("extra")
    if isinstance(extra, dict):
        event["extra"] = _safe_context(extra)
    return _scrub_value(event)


def _sample_rate(name: str) -> float:
    raw = os.getenv(name, "0.0").strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number from 0 to 1") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number from 0 to 1")
    return value


def configure_logging() -> None:
    """Install the root handler. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, None)
    if not isinstance(level, int):
        level = logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    if os.getenv("LOG_FORMAT", "json").lower() == "text":
        handler.setFormatter(RedactingTextFormatter(
            "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"))
    else:
        handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    if level_name not in logging.getLevelNamesMapping():
        root.warning("Invalid LOG_LEVEL=%r; using INFO.", level_name)

    # uvicorn installs its own handlers; route them through ours so every line
    # is the same shape.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True

    # These are chatty at INFO and drown out everything that matters.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)


def configure_sentry() -> bool:
    """Wire up Sentry when SENTRY_DSN is set. Returns True when enabled."""
    global _sentry
    if _sentry is not None:
        return True
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration

        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("EVERGREEN_ENV", "development"),
            release=os.getenv("EVERGREEN_RELEASE") or None,
            traces_sample_rate=_sample_rate("SENTRY_TRACES_SAMPLE_RATE"),
            # Financial app: never ship request bodies or headers to a third party.
            send_default_pii=False,
            max_request_body_size="never",
            before_send=_scrub_sentry_event,
            integrations=[LoggingIntegration(level=logging.INFO,
                                             event_level=logging.ERROR)],
        )
        _sentry = sentry_sdk
        logging.getLogger("evergreen").info("Sentry error monitoring enabled.")
        return True
    except ImportError:
        logging.getLogger("evergreen").warning(
            "SENTRY_DSN is set but sentry-sdk is not installed; monitoring is off.")
        return False
    except Exception as e:
        logging.getLogger("evergreen").warning("Could not initialise Sentry: %s", e)
        return False


def sentry_enabled() -> bool:
    """Whether Sentry initialized successfully (never exposes the DSN)."""
    return _sentry is not None


def report(
    logger: logging.Logger,
    log_message: str,
    exc: BaseException | None = None,
    **context,
):
    """
    Record a non-fatal failure from a best-effort code path.

    Use instead of a bare `except Exception: pass` so degraded behaviour is
    observable. This never raises - a failure in error reporting must not take
    down the request it is reporting on.
    """
    try:
        safe_context = _safe_context(context)
        exc_info = (type(exc), exc, exc.__traceback__) if exc is not None else None
        logger.warning(
            log_message,
            exc_info=exc_info,
            extra={"degraded": True, **safe_context},
        )
        if _sentry is not None and exc is not None:
            with _sentry.push_scope() as scope:
                scope.set_tag("degraded", True)
                for k, v in safe_context.items():
                    scope.set_extra(k, v)
                _sentry.capture_exception(exc)
    except Exception:
        pass


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def bind_request(request_id: str) -> None:
    request_id_var.set(request_id)


def bind_user(user_id) -> None:
    user_id_var.set(str(user_id) if user_id is not None else "-")
