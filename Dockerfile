FROM python:3.11.15-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --upgrade pip setuptools wheel \
    && /opt/venv/bin/python -m pip install -r requirements.txt


FROM python:3.11.15-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOG_FORMAT=json \
    EVERGREEN_ENV=production \
    EVERGREEN_DATA_DIR=/app/data \
    FORWARDED_ALLOW_IPS=127.0.0.1 \
    XDG_CACHE_HOME=/app/data/.cache

WORKDIR /app

# The compiler, Git client, pip cache, and build tooling never enter the runtime
# image. requirements-rl.txt is a separate research environment by design.
COPY --from=builder /opt/venv /opt/venv

COPY api/ api/
COPY scripts/ scripts/
# Served by the /terms and /privacy routes.
COPY TERMS.md PRIVACY.md LICENSE NOTICE ./

# Run as an unprivileged user. Runtime data (app.db, cache/, backups/)
# goes in /app/data — a subdirectory, so a mounted volume persists data WITHOUT
# shadowing the application code in /app.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Docker records health here; the deployment platform must alert/restart on an
# unhealthy state (plain Compose does not restart a merely unhealthy container).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD ["python", "/app/scripts/healthcheck.py"]

# Single worker is intentional for SQLite write coordination. Redis makes
# short-lived security state shared, but horizontal scaling still requires a
# managed database before adding replicas.
#
# Note: with EVERGREEN_ENV=production the startup audit REFUSES TO BOOT until
# EVERGREEN_MASTER_KEY and an entitled MARKET_DATA_PROVIDER are configured.
CMD ["uvicorn", "api.backend:app", "--host", "0.0.0.0", "--port", "8000"]
