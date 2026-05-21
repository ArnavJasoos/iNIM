# iNIM — Intel NIM-Equivalent Inference Container

An OpenVINO-based LLM serving stack that mirrors NVIDIA NIM's design philosophy but targets **Intel GPU hardware** (UHD integrated, Xe integrated, Arc discrete) on Ubuntu 22.04 LTS.

iNIM wraps [OpenVINO Model Server (OVMS)](https://github.com/openvinotoolkit/model_server) with profile selection logic, universal model conversion, lifecycle supervision, proxy routing, and operational tooling — the same scaffolding NIM provides on top of vLLM.

## ✨ Key Features

- **OpenAI-Compatible API** — Drop-in replacement for NIM. Works with any OpenAI client.
- **Universal Model Support** — Accepts OpenVINO IR, safetensors, PyTorch weights, and GGUF models from any source. Auto-converts to IR at first run.
- **Intel GPU Optimization** — Continuous batching, paged attention, INT4/INT8 dynamic quantization via OpenVINO on Intel XMX hardware.
- **Profile-Based Configuration** — Automatic GPU detection and optimal profile selection across all Intel GPU tiers.
- **Production-Ready** — nginx reverse proxy, health probes, Prometheus metrics, structured logging, TLS, CORS, graceful shutdown.
- **Cache-First Architecture** — One-time model conversion; subsequent starts skip conversion entirely.

## 🏗️ Architecture

```
Container PID 1: supervisord
├── nginx (port 8000, external)
│   ├── /v1/health/live → 200 immediately
│   ├── /v1/health/ready → proxy to OVMS
│   └── /v1/* → proxy to OVMS /v3/*
├── OVMS (port 8001, loopback only)
│   ├── OpenAI-compatible inference API
│   ├── Continuous batching + paged attention
│   └── Prometheus /metrics
└── iNIM Orchestrator (exits after setup)
    ├── GPU detection (OpenVINO Core)
    ├── Profile selection (manifest YAML)
    ├── Model download + conversion
    └── OVMS config generation
```

## 📋 Prerequisites

### Host Machine
- **OS**: Ubuntu 22.04 LTS (HWE kernel 6.5+ recommended)
- **GPU**: Intel UHD (Gen11+), Xe iGPU (Core Ultra), or Arc discrete GPU
- **Docker**: 20.10+
- **Intel GPU Drivers**: Level Zero, OpenCL, IGC

### Quick Host Setup
```bash
# Run the automated host setup script
sudo bash scripts/setup_host.sh

# Reboot if a new kernel was installed
sudo reboot
```

## 🚀 Quick Start

### Option 1: Interactive (Recommended)
```bash
# Build and run — you'll be prompted for model name and HF token
./scripts/run_local.sh
```

### Option 2: Docker Run
```bash
# Build the image
docker build -f docker/Dockerfile -t inim:latest .

# Run with Intel GPU pass-through
docker run \
  --device /dev/dri \
  -v /dev/dri:/dev/dri \
  --group-add render \
  -p 8000:8000 \
  -e INIM_MODEL=OpenVINO/Llama-3.2-3B-Instruct-int4-ov \
  -v inim-cache:/opt/inim/.cache \
  inim:latest
```

### Option 3: With Custom Model
```bash
# Any HuggingFace model — auto-detects format and converts
docker run \
  --device /dev/dri \
  -v /dev/dri:/dev/dri \
  --group-add render \
  -p 8000:8000 \
  -e INIM_MODEL=meta-llama/Llama-3.2-3B-Instruct \
  -e HF_TOKEN=your_token_here \
  -v inim-cache:/opt/inim/.cache \
  inim:latest
```

## 🧪 Testing the API

```bash
# Health check
curl http://localhost:8000/v1/health/live
curl http://localhost:8000/v1/health/ready

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct",
    "messages": [{"role": "user", "content": "Hello! What is iNIM?"}],
    "max_tokens": 256
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.2-3b-instruct",
    "messages": [{"role": "user", "content": "Tell me a short story"}],
    "stream": true
  }'

# Prometheus metrics
curl http://localhost:8000/v1/metrics
```

## 📦 Supported Model Formats

| Format | Source | Conversion | Notes |
|--------|--------|-----------|-------|
| OpenVINO IR (`.xml` + `.bin`) | Pre-converted repos, previous iNIM runs | None (pass-through) | Fastest startup |
| Safetensors (`.safetensors`) | Any HuggingFace repo | `optimum-cli export openvino` | Primary path |
| PyTorch (`.bin` shards) | Any HuggingFace repo | `optimum-cli export openvino` | Same as safetensors |
| GGUF (`.gguf`) | llama.cpp ecosystem | OV GenAI native reader or dequantize+convert | Broad support |

**Format is auto-detected** — just point `INIM_MODEL` at any source and iNIM handles the rest.

## 🎮 GPU Compatibility

| GPU Tier | Example SKUs | Default Profile | Max Model (INT4) |
|----------|-------------|----------------|-----------------|
| UHD (iGPU, Gen11) | UHD 770, UHD 730 | `int4-igpu-st` | ~1B params |
| Xe iGPU (Core Ultra) | MTL, LNL, ARL | `int4-igpu-cb` | 3-7B params |
| Arc A380 (6GB) | Arc A380 | `int4-arc-cb` | 7B params |
| Arc B580 (12GB) | Arc B580 | `int4-arc-hq-cb` | 13B params |
| Arc A770 (16GB) | Arc A770 | `int4-arc-hq-cb` | 13-30B params |

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `INIM_MODEL` | *(prompted)* | HuggingFace repo ID or local path |
| `INIM_MODEL_PROFILE` | *(auto)* | Override profile selection |
| `INIM_SERVER_PORT` | `8000` | External nginx port |
| `INIM_CACHE_PATH` | `/opt/inim/.cache` | Model cache directory |
| `INIM_LOG_LEVEL` | `info` | Logging verbosity |
| `INIM_OFFLINE` | `0` | Set to `1` for air-gap mode |
| `INIM_READY_TIMEOUT` | `300` | Seconds to wait for OVMS ready |
| `HF_TOKEN` | *(prompted)* | HuggingFace auth token |
| `INIM_TLS_CERT_PATH` | — | TLS certificate path |
| `INIM_TLS_KEY_PATH` | — | TLS private key path |
| `INIM_ALLOWED_ORIGINS` | `*` | CORS allowed origins |

### Pre-Built Model Manifests

iNIM ships with manifests for popular models:
- `llama-3.2-3b-instruct` — Meta Llama 3.2 3B
- `phi-3.5-mini-instruct` — Microsoft Phi 3.5 Mini
- `qwen3-8b` — Alibaba Qwen3 8B
- `mistral-7b-instruct-v0.3` — Mistral 7B Instruct

For any other model, iNIM generates a dynamic manifest automatically.

## 🔧 Development

```bash
# Clone and install in development mode
cd iNIM
pip install -e ".[dev]"

# Run unit tests (no GPU required)
pytest tests/ -v

# Lint
ruff check src/

# Type check
mypy src/inim/
```

## 📁 Project Structure

```
iNIM/
├── docker/
│   ├── Dockerfile              # Multi-stage production build
│   └── .dockerignore
├── configs/
│   ├── nginx/inim.conf         # Reverse proxy configuration
│   ├── supervisord/            # Process supervision
│   └── manifests/              # Model profile manifests
├── src/inim/                   # Python orchestrator
│   ├── main.py                 # Entry point
│   ├── gpu_detector.py         # Intel GPU detection
│   ├── profile_selector.py     # Profile selection algorithm
│   ├── format_detector.py      # Model format auto-detection
│   ├── model_resolver.py       # Download + conversion pipeline
│   ├── converter/              # Format conversion (IR/safetensors/GGUF)
│   ├── ovms_configurator.py    # OVMS config generation
│   ├── cache_manager.py        # SHA-256 verified caching
│   └── health_monitor.py       # OVMS readiness polling
├── scripts/
│   ├── entrypoint.sh           # Container entrypoint
│   ├── setup_host.sh           # Host driver setup
│   └── run_local.sh            # Quick-start script
├── tests/                      # Unit test suite
├── requirements.txt
├── pyproject.toml
└── intel_nim_architecture_report.md
```

## 🔒 Security

- OVMS listens only on `127.0.0.1:8001` (loopback) — never exposed externally
- nginx is the sole external-facing process
- All undefined routes return `404` (secure-by-default)
- Model integrity verified via SHA-256 checksums
- TLS support via nginx
- Safetensors preferred over PyTorch pickle for weight loading

## 📊 Observability

- **Liveness**: `GET /v1/health/live` — immediate 200, no backend dependency
- **Readiness**: `GET /v1/health/ready` — true when model is loaded
- **Metrics**: `GET /v1/metrics` — Prometheus format (request latency, throughput, etc.)
- **Logging**: JSON Lines to stdout, configurable level, request correlation headers

## 🛠️ Troubleshooting

### No Intel GPU detected
```bash
# Check GPU device nodes
ls -la /dev/dri/

# Ensure render group membership
groups $USER

# Install Intel GPU drivers
sudo bash scripts/setup_host.sh
```

### Model conversion fails
```bash
# Check available disk space (conversion needs 2-3x model size)
df -h

# Try a pre-converted IR model (no conversion needed)
docker run ... -e INIM_MODEL=OpenVINO/Llama-3.2-3B-Instruct-int4-ov ...
```

### OVMS timeout
```bash
# Increase timeout for large models
docker run ... -e INIM_READY_TIMEOUT=600 ...

# Check OVMS logs
docker logs <container_id>
```

---

*Architecture reference: [intel_nim_architecture_report.md](intel_nim_architecture_report.md)*
*Target: Ubuntu 22.04 LTS | OpenVINO 2025.4 | Intel Arc/Xe/UHD GPUs*
