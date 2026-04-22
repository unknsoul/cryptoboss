# CryptoBoss v11.0-FINAL Production Dockerfile
# Multi-stage build for minimal production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.11-slim as production

LABEL maintainer="CryptoBoss Team"
LABEL version="11.0-FINAL"
LABEL description="Production-Grade Autonomous Trading Engine"

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash cryptoboss
RUN chown -R cryptoboss:cryptoboss /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/cryptoboss/.local
ENV PATH=/home/cryptoboss/.local/bin:$PATH

# Copy application code
COPY --chown=cryptoboss:cryptoboss src/ ./src/
COPY --chown=cryptoboss:cryptoboss configs/ ./configs/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models \
    && chown -R cryptoboss:cryptoboss /app/data /app/logs /app/models

# Switch to non-root user
USER cryptoboss

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_LEVEL=INFO
ENV ENABLE_DAILY_LOSS_LIMITS=true
ENV EXECUTION_MODE=paper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v11/health || exit 1

# Expose API port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.api.routes:app", "--host", "0.0.0.0", "--port", "8000"]
