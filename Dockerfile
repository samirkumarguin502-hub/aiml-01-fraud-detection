# Backend API image (CPU only).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt requirements-api.txt ./
RUN pip install -r requirements-api.txt
# Optional primary model dependency. Build with:  docker compose build --build-arg INSTALL_XGBOOST=true
ARG INSTALL_XGBOOST=false
RUN if [ "$INSTALL_XGBOOST" = "true" ]; then pip install "xgboost>=2.0"; fi

COPY backend ./backend
COPY ml ./ml
COPY monitoring ./monitoring
COPY streaming ./streaming

# Non-root user; writable state lives only under /data (SQLite) - models are mounted read-only at /models.
RUN useradd --system --uid 10001 --no-create-home app && mkdir -p /data /models && chown app /data
USER 10001

ENV APP_ENV=production \
    DATABASE_URL=sqlite:////data/fraud_platform.db \
    ARTIFACT_ROOT=/models \
    STREAM_OUTCOMES_DIR=/data/streaming_outcomes

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=4s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/health', timeout=3)"

# Authentication is NOT implemented: bind publicly only behind something that adds it.
CMD ["python", "-m", "uvicorn", "backend.app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
