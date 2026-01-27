# MEWS - Market Early Warning System
# Multi-stage Dockerfile for lightweight CPU-only deployment
#
# Usage:
#   docker build -t mews:latest .
#   docker run --rm mews:latest --mock --verbose
#   docker run --rm -e MEWS_DATE=2024-01-15 mews:latest
#
# Environment Variables:
#   MEWS_MODE: "mock" (default) or "live"
#   MEWS_DATE: Target date in YYYY-MM-DD format (default: today)

# =============================================================================
# Stage 1: Builder - Install dependencies and build wheel
# =============================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only files needed for dependency resolution
COPY pyproject.toml ./

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (cacheable layer)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy source code
COPY core_specs/ ./core_specs/
COPY data_ingestion/ ./data_ingestion/
COPY feature_services/ ./feature_services/
COPY risk_engine/ ./risk_engine/
COPY pipeline/ ./pipeline/
COPY visualization/ ./visualization/

# Install the package
RUN pip install --no-cache-dir .

# =============================================================================
# Stage 2: Runtime - Minimal image with only runtime dependencies
# =============================================================================
FROM python:3.11-slim AS runtime

# Labels
LABEL org.opencontainers.image.title="MEWS"
LABEL org.opencontainers.image.description="Market Early Warning System"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.source="https://github.com/your-org/mews"

# Create non-root user for security
RUN groupadd --gid 1000 mews && \
    useradd --uid 1000 --gid mews --shell /bin/bash --create-home mews

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Note: core_specs YAML files are packaged in the wheel, no separate copy needed

# Environment configuration
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MEWS_MODE=mock \
    MEWS_DATE=""

# Switch to non-root user
USER mews

# Health check (optional, validates Python and package import)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pipeline.daily_run.run; print('healthy')" || exit 1

# Default entrypoint: daily pipeline
ENTRYPOINT ["python", "-m", "pipeline.daily_run.run"]

# Default arguments (can be overridden)
CMD ["--mock", "--verbose"]
