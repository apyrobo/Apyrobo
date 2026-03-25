# =============================================================================
# APYROBO — Production API Container
# Multi-stage build: slim Python 3.10 runtime for the REST gateway
#
# Build:  docker build -f docker/Dockerfile -t apyrobo:latest .
# Run:    docker run -p 8080:8080 -e APYROBO_API_KEY=changeme apyrobo:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder: install all dependencies into a virtual environment
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only the files needed to install the package
COPY pyproject.toml .
COPY apyrobo/ ./apyrobo/

# Create venv and install
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[api]" || \
    pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Stage 2 — runtime: minimal image with just the venv and source
# ---------------------------------------------------------------------------
FROM python:3.10-slim AS runtime

LABEL maintainer="APYROBO Contributors"
LABEL description="APYROBO REST API gateway — production image"
LABEL org.opencontainers.image.source="https://github.com/apyrobo/apyrobo"

# Create non-root user
RUN groupadd -r apyrobo && useradd -r -g apyrobo -d /app -s /sbin/nologin apyrobo

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
# Copy source (needed for editable install)
COPY --from=builder /build/apyrobo ./apyrobo
COPY --from=builder /build/pyproject.toml ./pyproject.toml

# Create directories for persistent data; set ownership
RUN mkdir -p /app/data/audit /app/data/skills && \
    chown -R apyrobo:apyrobo /app

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APYROBO_LOG_LEVEL=INFO \
    APYROBO_AUDIT_DB=/app/data/audit/audit.db

USER apyrobo

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

CMD ["uvicorn", "apyrobo.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
