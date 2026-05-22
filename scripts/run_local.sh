#!/bin/bash
# =============================================================================
# iNIM Quick-Start Script
#
# Builds the Docker image and runs iNIM with Intel GPU pass-through.
# Usage: ./scripts/run_local.sh [MODEL_NAME]
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
IMAGE_NAME="inim"
IMAGE_TAG="latest"
CONTAINER_NAME="inim-server"
SERVER_PORT="${INIM_SERVER_PORT:-8000}"

# Get script directory (project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ---------------------------------------------------------------------------
# Ask user for model if not provided
# ---------------------------------------------------------------------------
MODEL="${1:-${INIM_MODEL:-}}"
if [ -z "$MODEL" ]; then
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║          iNIM — Quick Start                      ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    echo "Available pre-configured models:"
    echo "  1) OpenVINO/Llama-3.2-3B-Instruct-int4-ov  (recommended, ~2GB)"
    echo "  2) OpenVINO/Phi-3.5-mini-instruct-int4-ov   (~2GB)"
    echo "  3) OpenVINO/Qwen3-8B-int4-ov                (~5GB)"
    echo "  4) OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov (~4GB)"
    echo ""
    echo "Or enter any HuggingFace model ID (will be auto-converted):"
    echo ""
    read -r -p "Enter model name: " MODEL

    if [ -z "$MODEL" ]; then
        echo "No model specified. Using default: OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
        MODEL="OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
    fi
fi

# ---------------------------------------------------------------------------
# Ask for HF token if not set
# ---------------------------------------------------------------------------
HF_TOKEN_ARG=""
if [ -z "${HF_TOKEN:-}" ]; then
    # Check cached token
    CACHED_TOKEN="$HOME/.cache/huggingface/token"
    if [ -f "$CACHED_TOKEN" ]; then
        export HF_TOKEN=$(cat "$CACHED_TOKEN")
        echo "Using cached HuggingFace token"
    else
        echo ""
        read -r -p "Enter HuggingFace token (press Enter to skip): " HF_TOKEN_INPUT
        if [ -n "$HF_TOKEN_INPUT" ]; then
            export HF_TOKEN="$HF_TOKEN_INPUT"
        fi
    fi
fi

if [ -n "${HF_TOKEN:-}" ]; then
    HF_TOKEN_ARG="-e HF_TOKEN=${HF_TOKEN}"
fi

echo ""
echo "════════════════════════════════════════════════"
echo " Building iNIM Docker image..."
echo "════════════════════════════════════════════════"

cd "$PROJECT_ROOT"
docker build --no-cache\
    -f docker/Dockerfile \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" \
    .

echo ""
echo "════════════════════════════════════════════════"
echo " Starting iNIM container..."
echo "════════════════════════════════════════════════"
echo "  Model:  ${MODEL}"
echo "  Port:   ${SERVER_PORT}"
echo ""

# Stop existing container if running
docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true

# Run with Intel GPU pass-through
docker run \
    --name "${CONTAINER_NAME}" \
    --device /dev/dri \
    -v /dev/dri:/dev/dri \
    --group-add 110 \
    -p "${SERVER_PORT}:8000" \
    -e "INIM_MODEL=${MODEL}" \
    ${HF_TOKEN_ARG} \
    -e "INIM_LOG_LEVEL=${INIM_LOG_LEVEL:-info}" \
    -v "inim-cache:/opt/inim/.cache" \
    --rm \
    -it \
    "${IMAGE_NAME}:${IMAGE_TAG}"

echo ""
echo "════════════════════════════════════════════════"
echo " iNIM is running!"
echo ""
echo " Test with:"
echo "   curl http://localhost:${SERVER_PORT}/v1/health/live"
echo "   curl http://localhost:${SERVER_PORT}/v1/health/ready"
echo ""
echo " Chat:"
echo "   curl http://localhost:${SERVER_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"
echo ""
echo " Streaming:"
echo "   curl http://localhost:${SERVER_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}],\"stream\":true}'"
echo "════════════════════════════════════════════════"
