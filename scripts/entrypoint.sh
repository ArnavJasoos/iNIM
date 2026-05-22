#!/bin/bash
# =============================================================================
# iNIM Container Entrypoint
#
# Sets up the runtime environment, configures nginx (TLS, CORS),
# and launches supervisord as PID 1.
# =============================================================================

set -euo pipefail

echo "════════════════════════════════════════════════"
echo " iNIM — Intel NIM-Equivalent Inference Container"
echo " Starting up..."
echo "════════════════════════════════════════════════"

# ---------------------------------------------------------------------------
# Default environment variables
# ---------------------------------------------------------------------------
export INIM_SERVER_PORT="${INIM_SERVER_PORT:-8000}"
export INIM_CACHE_PATH="${INIM_CACHE_PATH:-/opt/inim/.cache}"
export INIM_LOG_LEVEL="${INIM_LOG_LEVEL:-info}"
export INIM_READY_TIMEOUT="${INIM_READY_TIMEOUT:-300}"
export INIM_ALLOWED_ORIGINS="${INIM_ALLOWED_ORIGINS:-*}"
export INIM_OFFLINE="${INIM_OFFLINE:-0}"
export OVMS_LOG_LEVEL="${OVMS_LOG_LEVEL:-INFO}"

# ---------------------------------------------------------------------------
# Prompt user for model name if not set
# ---------------------------------------------------------------------------
if [ -z "${INIM_MODEL:-}" ]; then
    echo ""
    echo "WARNING: INIM_MODEL is not set."
    echo "Please provide a model identifier."
    echo ""
    echo "Examples:"
    echo "  - HuggingFace repo:  OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
    echo "  - HuggingFace repo:  meta-llama/Llama-3.2-3B-Instruct"
    echo "  - Local path:        /models/my-model-dir"
    echo ""

    # In interactive mode, prompt for model
    if [ -t 0 ]; then
        read -r -p "Enter model name from HuggingFace (or local path): " INIM_MODEL
        export INIM_MODEL
    else
        echo "ERROR: INIM_MODEL must be set when running non-interactively."
        echo "  docker run -e INIM_MODEL=<model> ..."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Prompt for HuggingFace token if not set (for gated models)
# ---------------------------------------------------------------------------
if [ -z "${HF_TOKEN:-}" ]; then
    # Check if cached token exists
    CACHED_TOKEN_FILE="$HOME/.cache/huggingface/token"
    if [ -f "$CACHED_TOKEN_FILE" ]; then
        export HF_TOKEN=$(cat "$CACHED_TOKEN_FILE")
        echo "INFO: Using cached HuggingFace token from $CACHED_TOKEN_FILE"
    elif [ -t 0 ]; then
        echo ""
        echo "NOTE: HF_TOKEN is not set. Some models (e.g., Llama) require authentication."
        read -r -p "Enter HuggingFace token (press Enter to skip): " HF_TOKEN
        if [ -n "$HF_TOKEN" ]; then
            export HF_TOKEN
        fi
    fi
fi

echo "Configuration:"
echo "  Model:        ${INIM_MODEL}"
echo "  Server Port:  ${INIM_SERVER_PORT}"
echo "  Cache Path:   ${INIM_CACHE_PATH}"
echo "  Log Level:    ${INIM_LOG_LEVEL}"
echo "  Offline Mode: ${INIM_OFFLINE}"
echo "  HF Token:     ${HF_TOKEN:+[SET]}"
echo ""

# ---------------------------------------------------------------------------
# Configure nginx: port
# ---------------------------------------------------------------------------
NGINX_CONF="/etc/nginx/conf.d/inim.conf"

# Update listen port
sed -i "s/listen 8000;/listen ${INIM_SERVER_PORT};/" "$NGINX_CONF"

# ---------------------------------------------------------------------------
# Configure nginx: TLS (if cert and key are provided)
# ---------------------------------------------------------------------------
if [ -n "${INIM_TLS_CERT_PATH:-}" ] && [ -n "${INIM_TLS_KEY_PATH:-}" ]; then
    echo "INFO: TLS enabled — configuring HTTPS"

    # Replace listen directive with SSL
    sed -i "s/listen ${INIM_SERVER_PORT};/listen ${INIM_SERVER_PORT} ssl;/" "$NGINX_CONF"

    # Add SSL certificate directives after the listen line
    sed -i "/listen ${INIM_SERVER_PORT} ssl;/a\\
    ssl_certificate ${INIM_TLS_CERT_PATH};\\
    ssl_certificate_key ${INIM_TLS_KEY_PATH};\\
    ssl_protocols TLSv1.2 TLSv1.3;\\
    ssl_prefer_server_ciphers on;" "$NGINX_CONF"
fi

# ---------------------------------------------------------------------------
# Configure nginx: CORS
# ---------------------------------------------------------------------------
if [ "$INIM_ALLOWED_ORIGINS" != "*" ]; then
    # Add CORS headers for specific origins
    CORS_CONFIG="
    # CORS Configuration
    add_header 'Access-Control-Allow-Origin' '${INIM_ALLOWED_ORIGINS}' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization, X-Request-Id' always;
    "
else
    CORS_CONFIG="
    # CORS Configuration (open — restrict in production)
    add_header 'Access-Control-Allow-Origin' '*' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'Content-Type, Authorization, X-Request-Id' always;
    "
fi

# Insert CORS config after the client_max_body_size line
python3 -c "
import sys
conf_path = sys.argv[1]
cors_config = sys.argv[2]
with open(conf_path, 'r') as f:
    content = f.read()
target = 'client_max_body_size 100m;'
if target in content:
    content = content.replace(target, target + '\n' + cors_config)
with open(conf_path, 'w') as f:
    f.write(content)
" "$NGINX_CONF" "$CORS_CONFIG"

# ---------------------------------------------------------------------------
# Ensure cache directories exist
# ---------------------------------------------------------------------------
mkdir -p "${INIM_CACHE_PATH}/models"

# ---------------------------------------------------------------------------
# GPU device check
# ---------------------------------------------------------------------------
if [ -d "/dev/dri" ]; then
    echo "INFO: /dev/dri found — GPU device access available"
    ls -la /dev/dri/ 2>/dev/null || true
else
    echo "WARNING: /dev/dri not found!"
    echo "  Ensure --device /dev/dri is passed to docker run"
    echo "  Without GPU access, inference will fall back to CPU"
fi

# ---------------------------------------------------------------------------
# Run clinfo for GPU diagnostics (non-fatal)
# ---------------------------------------------------------------------------
if command -v clinfo &> /dev/null; then
    echo ""
    echo "GPU Info (via clinfo):"
    clinfo --list 2>/dev/null || echo "  clinfo failed (non-fatal)"
    echo ""
fi

# ---------------------------------------------------------------------------
# Set up OVMS start script sentinel watcher
# This background loop watches for the orchestrator's sentinel file
# and starts OVMS via supervisorctl when it appears.
# ---------------------------------------------------------------------------
(
    SENTINEL="/tmp/inim_ready_to_start"
    echo "Sentinel watcher: waiting for $SENTINEL..."

    while [ ! -f "$SENTINEL" ]; do
        sleep 1
    done

    echo "Sentinel found! Starting OVMS via supervisorctl..."
    supervisorctl start ovms
) &

# ---------------------------------------------------------------------------
# Trap signals for graceful shutdown
# ---------------------------------------------------------------------------
trap_handler() {
    echo "iNIM: Received signal, shutting down gracefully..."
    # Stop OVMS first (drain in-flight requests)
    supervisorctl stop ovms 2>/dev/null || true
    sleep 2
    # Then stop nginx
    supervisorctl stop nginx 2>/dev/null || true
    # Let supervisord exit
    kill -TERM "$(cat /var/run/supervisord.pid 2>/dev/null)" 2>/dev/null || true
    exit 0
}

trap trap_handler SIGTERM SIGINT

# ---------------------------------------------------------------------------
# Launch supervisord as PID 1
# ---------------------------------------------------------------------------
echo "Starting supervisord..."
exec /usr/bin/supervisord -c /etc/supervisor/conf.d/inim.conf
