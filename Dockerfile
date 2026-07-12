FROM python:3.11-slim

WORKDIR /app

# System deps some scientific packages need at build time.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ api/

# app.db, cache/, charts/, chat_memory/ live here; mount a volume to keep them.
VOLUME ["/app/data"]
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "api.backend:app", "--host", "0.0.0.0", "--port", "8000"]
