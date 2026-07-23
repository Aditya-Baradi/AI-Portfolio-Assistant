FROM python:3.11-slim

WORKDIR /app

# System deps some scientific packages need at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Patch the build tooling first (clears setuptools/pip advisories).
RUN pip install --no-cache-dir --upgrade pip setuptools

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/

# Run as an unprivileged user. The app writes app.db, cache/, charts/, and
# chat_memory/ into /app (its working dir), so that must be owned by the user.
RUN useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1

EXPOSE 8000
# Single worker (uvicorn default) is REQUIRED: some auth state and the FinRL
# concurrency cap live in process memory. Do not add --workers >1 without moving
# that state to Redis first — see SECURITY.md ("Deployment constraint").
CMD ["uvicorn", "api.backend:app", "--host", "0.0.0.0", "--port", "8000"]
