# DataPilot: one container serving the API and the built React app from the same origin.
#
#   docker build -t datapilot .
#   docker run -p 8000:8000 -e GEMINI_API_KEY=... datapilot      ->  http://localhost:8000
#
# The API key is never baked into the image: pass it at runtime (or as a host "secret").

# ---- stage 1: build the frontend ---------------------------------------------------------
FROM node:24-slim AS web
WORKDIR /web
# Dependencies first: this layer is cached until package*.json change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
# No VITE_API_URL: a production build calls the API on its own origin (/datasets, /runs...).
RUN npm run build

# ---- stage 2: the API, which also serves the built frontend ------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# Pinned dependencies first, for layer caching and reproducible builds.
COPY backend/requirements.lock.txt ./
RUN pip install -r requirements.lock.txt

COPY backend/app ./app
COPY --from=web /web/dist ./static

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 datapilot
USER datapilot

ENV PORT=8000 \
    DATAPILOT_STATIC_DIR=/app/static \
    DATAPILOT_TRACE_DIR=/tmp/datapilot-traces \
    DATAPILOT_CORS_ORIGINS=""
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

# --proxy-headers: behind a host's reverse proxy, take the client IP from X-Forwarded-For so
# per-IP rate limits apply to real visitors, not to the proxy. Hosts set $PORT.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
