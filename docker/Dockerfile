# =============================================================================
# iNIM Dockerfile — Intel NIM-Equivalent Inference Container
#
# Multi-stage build:
#   Stage 1 (builder): Install Python dependencies for model conversion
#   Stage 2 (runtime): Production container with OVMS + nginx + orchestrator
#
# Base image: openvino/model_server:latest-gpu
# Fallback compatible with UHD iGPUs as it includes full Intel compute-runtime
# =============================================================================

# ---------------------------------------------------------------------------
# Build Arguments
# ---------------------------------------------------------------------------
ARG OVMS_VERSION=latest-gpu
ARG PYTHON_VERSION=3.11

# =============================================================================
# Stage 1: Builder — Install Python conversion tools
# =============================================================================
FROM python:${PYTHON_VERSION}-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for model conversion
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install \
    -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# =============================================================================
# Stage 2: Runtime — Production container
# =============================================================================
FROM openvino/model_server:${OVMS_VERSION}

# Switch to root for package installation
USER root

# ---------------------------------------------------------------------------
# System packages: nginx, supervisord, diagnostic tools
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    nginx \
    supervisor \
    python3 \
    python3-pip \
    python3-venv \
    clinfo \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Copy Python packages from builder stage
# ---------------------------------------------------------------------------
COPY --from=builder /install /usr/local

# ---------------------------------------------------------------------------
# Create Python virtual environment
# ---------------------------------------------------------------------------
RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Ensure additional Python packages for the orchestrator
# ---------------------------------------------------------------------------
RUN pip install --no-cache-dir \
    "pyyaml>=6.0" \
    "requests>=2.31" \
    "huggingface_hub>=0.24"


# ---------------------------------------------------------------------------
# Create iNIM directory structure
# ---------------------------------------------------------------------------
RUN mkdir -p \
    /opt/inim/src \
    /opt/inim/scripts \
    /opt/inim/etc/manifests \
    /opt/inim/.cache/models \
    /var/log/inim

# ---------------------------------------------------------------------------
# Copy iNIM source code
# ---------------------------------------------------------------------------
COPY src/inim /opt/inim/src/inim/

# ---------------------------------------------------------------------------
# Copy configuration files
# ---------------------------------------------------------------------------
COPY configs/nginx/inim.conf /etc/nginx/conf.d/inim.conf
COPY configs/supervisord/supervisord.conf /etc/supervisor/conf.d/inim.conf
COPY configs/manifests/ /opt/inim/etc/manifests/

# Remove default nginx site
RUN rm -f /etc/nginx/sites-enabled/default \
    && rm -f /etc/nginx/conf.d/default.conf

# Ensure nginx includes our config
#RUN echo "include /etc/nginx/conf.d/*.conf;" >> /etc/nginx/nginx.conf || true

# ---------------------------------------------------------------------------
# Copy scripts
# ---------------------------------------------------------------------------
COPY scripts/entrypoint.sh /opt/inim/scripts/entrypoint.sh
COPY scripts/start_ovms.sh /opt/inim/scripts/start_ovms.sh
COPY scripts/supervisor_exit_handler.py /opt/inim/scripts/supervisor_exit_handler.py

RUN chmod +x /opt/inim/scripts/*.sh

# ---------------------------------------------------------------------------
# Set Python path
# ---------------------------------------------------------------------------
ENV PYTHONPATH="/opt/inim/src"
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Default environment variables
# ---------------------------------------------------------------------------
ENV INIM_SERVER_PORT=8000
ENV INIM_CACHE_PATH=/opt/inim/.cache
ENV INIM_LOG_LEVEL=info
ENV INIM_READY_TIMEOUT=300
ENV INIM_MANIFEST_PATH=/opt/inim/etc/manifests
ENV OVMS_LOG_LEVEL=INFO

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health/live || exit 1

# ---------------------------------------------------------------------------
# Expose port
# ---------------------------------------------------------------------------
EXPOSE 8000

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
ENTRYPOINT ["/opt/inim/scripts/entrypoint.sh"]
