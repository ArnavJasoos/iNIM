# Intel NIM-Equivalent Inference Container Architecture
## An OpenVINO-Based LLM Serving Stack for Intel GPUs (UHD / Xe / Arc)
### Target Platform: Ubuntu 22.04 LTS

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [NVIDIA NIM Container Deep-Dive Analysis](#2-nvidia-nim-container-deep-dive-analysis)
3. [Component Mapping: NIM → Intel OpenVINO Stack](#3-component-mapping-nim--intel-openvino-stack)
4. [Intel GPU Driver and Compute Foundation](#4-intel-gpu-driver-and-compute-foundation)
5. [Proposed Architecture: IntelNIM (iNIM)](#5-proposed-architecture-intelnimnim)
6. [Layer-by-Layer Component Design](#6-layer-by-layer-component-design)
7. [Model Profile System](#7-model-profile-system)
8. [Container Startup Lifecycle](#8-container-startup-lifecycle)
9. [API Surface and Proxy Layer](#9-api-surface-and-proxy-layer)
10. [Model Acquisition and Caching](#10-model-acquisition-and-caching)
11. [Observability Stack](#11-observability-stack)
12. [Security Considerations](#12-security-considerations)
13. [GPU Tier Considerations](#13-gpu-tier-considerations)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Dependency Manifest](#15-dependency-manifest)

---

## 1. Executive Summary

This document specifies the architecture of **iNIM** (Intel NIM-equivalent), a production-grade LLM inference container that mirrors NVIDIA NIM's design philosophy but targets Intel GPU hardware (UHD integrated graphics, Xe integrated graphics, and Arc discrete GPUs) on Ubuntu 22.04 LTS.

NVIDIA NIM is fundamentally an **enterprise orchestration wrapper** around vLLM, packaged with:
- A model manifest and profile selection system
- An nginx reverse proxy
- Model download and caching logic
- Prometheus metrics exposure
- Health probes and lifecycle supervision

iNIM replicates every one of these layers, substituting CUDA/NVIDIA-specific components with their Intel/open-source equivalents. The primary inference backend replaces vLLM-on-CUDA with **OpenVINO Model Server (OVMS)**, which is a production-hardened C++ inference server from Intel that natively implements continuous batching, paged attention, and an OpenAI-compatible API on Intel hardware. The model optimization toolchain replaces TensorRT-LLM and NVIDIA's quantization suite with **optimum-intel + NNCF** for converting and quantizing models to OpenVINO IR format.

A core design requirement is **universal model source support**. iNIM is not restricted to the pre-converted IR models published by Intel's `OpenVINO/` organization on HuggingFace. The orchestrator implements a multi-stage model acquisition and conversion pipeline that accepts models in any of the following source formats and automatically converts them to OpenVINO IR at container startup before OVMS is launched:

- **OpenVINO IR** (`.xml` + `.bin`) — used directly, no conversion needed
- **Safetensors** (HuggingFace native format) — converted via `optimum-cli export openvino`
- **PyTorch weights** (`.bin` shards, `pytorch_model.bin`) — converted via `optimum-cli export openvino`
- **GGUF** (llama.cpp format, `.gguf`) — handled via OpenVINO GenAI's native GGUF reader (preview, for supported architectures) or via an offline dequantize-then-convert pipeline for broader architecture coverage

Converted models are written to the cache directory so subsequent container starts skip conversion entirely. The source format is detected automatically from the local path or remote repository structure, requiring no user configuration beyond pointing iNIM at the model.

Critically, no component is invented where a mature open-source equivalent already exists. OVMS itself is already the closest functional analog to NIM that Intel ships — iNIM's value is in wrapping OVMS with the profile selection logic, universal model conversion, lifecycle supervision, proxy routing, and operational tooling that NIM provides on top of vLLM.

---

## 2. NVIDIA NIM Container Deep-Dive Analysis

### 2.1 What NIM Actually Is

NIM LLM is documented by NVIDIA as "an enterprise orchestration layer for vLLM." It packages vLLM into a production-ready container and adds operational scaffolding. The container runs exactly two processes:

1. **vLLM inference backend** — loads models, runs GPU inference via CUDA, exposes an OpenAI-compatible HTTP API on port 8001 (loopback only, never directly exposed).
2. **nginx proxy** — listens externally (port 8000 by default), provides immediate liveness responses, model-aware readiness probes, request routing, TLS termination, and CORS.

A supervisor process (typically via a shell entrypoint or a simple supervisor daemon) monitors both. If either exits unexpectedly, the entire container shuts down so an orchestrator can reschedule.

### 2.2 The Model Profile System

Every NIM LLM container ships with a **model manifest** (`/opt/nim/etc/default/model_manifest.yaml`). This manifest is a catalog of pre-validated configurations called **profiles**. Each profile encodes:

- Backend engine (`vllm`, `tensorrt_llm`, or `sglang`)
- Precision (`bf16`, `fp8`, `mxfp4`, `nvfp4`)
- Tensor parallelism degree (`tp1`, `tp2`, `tp4`, `tp8`)
- Pipeline parallelism stage count (`pp1`, `pp2`)
- Feature flags (e.g. `-feat_lora` for LoRA adapter support)
- Minimum VRAM requirement per GPU

Profile ID format: `vllm-bf16-tp1-pp1` (human-readable) or a 64-character SHA hash (exact/pinned).

**Profile selection algorithm at startup:**
1. Detect installed GPU hardware (type, VRAM, count)
2. Filter manifest to compatible profiles (VRAM fit)
3. Apply priority order: custom fine-tuned > optimized hardware-specific > generic
4. Within tier: prefer lower precision (FP8 > BF16 > FP16), higher TP
5. Override with `NIM_MODEL_PROFILE` env var if set

**Selected profile determines:**
- Which model weight files to download
- How vLLM is launched (precision flags, TP degree, KV cache sizing)

### 2.3 Container Startup Sequence

```
1. Start nginx proxy → begins listening on NIM_SERVER_PORT (8000)
   → /v1/health/live returns 200 immediately (no backend dependency)
   → /v1/health/ready returns 503 (backend not yet ready)

2. GPU detection → profile selection from manifest

3. Model download (if not cached at NIM_CACHE_PATH)
   → Checks NGC, HuggingFace, or local disk

4. Launch vLLM on port 8001 (loopback only)
   → Config merged from: profile defaults → env vars → CLI passthrough args

5. Poll vLLM /health until 200 OK
   → /v1/health/ready begins returning 200 OK
   → Orchestrator begins routing traffic

6. Supervisor loop: if vLLM or nginx dies → SIGTERM container
   On SIGTERM: graceful vLLM stop first, then nginx shutdown
```

### 2.4 API Surface

NIM exposes the following endpoint categories via the nginx proxy:

| Route | Category | Routed to |
|-------|-----------|-----------|
| `/v1/chat/completions` | Inference | vLLM backend |
| `/v1/completions` | Inference | vLLM backend |
| `/v1/embeddings` | Inference | vLLM backend |
| `/v1/messages` | Inference (Anthropic-compat) | vLLM backend |
| `/v1/models` | Management | vLLM backend |
| `/v1/metrics` | Observability | vLLM backend (Prometheus) |
| `/v1/health/live` | Health | nginx (served directly) |
| `/v1/health/ready` | Health | nginx (proxies to vLLM /health) |
| All other paths | — | 404 Not Found |

### 2.5 Observability

NIM exposes three observability surfaces:
- **Health probes**: `/v1/health/live` (container alive) and `/v1/health/ready` (model loaded)
- **Prometheus metrics**: `/v1/metrics` — request latency, throughput, GPU utilization
- **Structured logging**: JSON Lines format, configurable log level, distributed tracing headers (`X-Request-Id`, `Traceparent`)

### 2.6 What NIM Does NOT Provide (Design Boundaries)

- NIM does not perform model conversion or quantization at runtime; it consumes pre-optimized weights
- NIM does not manage multi-container orchestration (that is Kubernetes/Helm's job)
- NIM does not store models; it downloads to a cache volume mounted at runtime

---

## 3. Component Mapping: NIM → Intel OpenVINO Stack

The following table maps every NIM component to its iNIM equivalent. The decision column explains whether the component is **replaced**, **kept as-is**, or **wrapped**:

| NIM Component | Role | iNIM Equivalent | Decision |
|---|---|---|---|
| vLLM (CUDA) | Inference backend | OpenVINO Model Server (OVMS) | **Replace** — OVMS is C++ native, supports continuous batching + paged attention on Intel GPU |
| nginx | Reverse proxy, health, TLS | **nginx** | **Keep** — nginx is hardware-agnostic; no NVIDIA dependency |
| NVIDIA NGC model registry | Model distribution | HuggingFace Hub (any repo/org) + local paths | **Replace** — iNIM accepts IR, safetensors, PyTorch weights, and GGUF from any source |
| TensorRT-LLM | Optimized inference kernels | OpenVINO Runtime + GenAI | **Replace** — OpenVINO Runtime handles kernel dispatch for Intel GPU via Level Zero |
| NCCL (multi-GPU comms) | Tensor parallel comms | oneCCL (Intel) | **Replace** — oneCCL is Intel's collective communications library for multi-GPU |
| CUDA compute stack | GPU compute API | Level Zero + OpenCL | **Replace** — Intel's native compute API stack |
| NVIDIA compute-runtime | GPU driver | Intel compute-runtime (NEO) | **Replace** — open source, same role |
| `NIM_MODEL_PROFILE` / manifest | Profile system | iNIM manifest + profile selector | **Implement** — must be written; no off-the-shelf equivalent exists |
| Container supervisor | Process supervision | supervisord | **Use** — mature, lightweight, platform-agnostic |
| Prometheus metrics | Observability | OVMS built-in `/metrics` + nginx-prometheus-exporter | **Keep/Adapt** — OVMS natively exposes Prometheus metrics |
| optimum-cli / NNCF | Model quantization and export | **optimum-intel + NNCF** (+ OV GenAI GGUF reader) | **Keep + Extend** — Intel's toolchain; iNIM adds format-detection logic on top |
| NGC container base image | Base OS/CUDA layers | Ubuntu 22.04 + Intel GPU PPA | **Replace** — Intel's compute-runtime packages from Intel Graphics PPA |

---

## 4. Intel GPU Driver and Compute Foundation

Before any ML inference stack can function on Intel hardware, the correct kernel and userspace driver layers must be in place. This is the foundation layer of iNIM.

### 4.1 Kernel Requirements

- **i915 kernel module**: Intel's DRM/KMS driver for iGPU (UHD / Xe). Present in mainline kernels.
- **xe kernel module**: Newer DRM driver for Xe2 and later architectures (Battlemage, Lunar Lake). Mainline since 6.8+.
- **For Arc discrete GPUs**: kernel 6.2+ recommended; 6.5+ preferred for best Arc Alchemist/Battlemage support.
- **For Ubuntu 22.04**: The HWE (Hardware Enablement) kernel tracks 6.5+ and is the recommended path. Stock 5.15 kernel has limited Arc support.

GPU firmware loading must be enabled:
```
options i915 enable_guc=3
```
This enables GuC (Graphics microController) and HuC (HEVC microController) firmware, which improves GPU scheduling and improves AI workload responsiveness.

### 4.2 Level Zero and OpenCL Stack

OpenVINO's GPU plugin communicates with Intel hardware via two compute APIs:

**Level Zero** (primary, preferred)
- Intel's low-level compute API, analogous to CUDA in the NVIDIA stack
- Required for OpenVINO GPU plugin
- Packages: `intel-level-zero-gpu`, `level-zero`

**OpenCL 3.0** (secondary, fallback)
- Packages: `intel-opencl-icd`, `ocl-icd-libopencl1`

Both are provided by **Intel NEO** — the Intel Graphics Compute Runtime — which is open source (MIT license) at `github.com/intel/compute-runtime`.

Installation on Ubuntu 22.04:
```bash
# Add Intel Graphics repository
curl -fsSL https://repositories.intel.com/graphics/intel-graphics.key | \
  gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/graphics/ubuntu jammy flex" \
  > /etc/apt/sources.list.d/intel-gpu-jammy.list

apt-get update
apt-get install -y ocl-icd-libopencl1 intel-opencl-icd intel-level-zero-gpu level-zero

# Add user to render group for GPU access without root
usermod -a -G render $CONTAINER_USER
```

### 4.3 Intel Graphics Compiler (IGC)

The Intel Graphics Compiler (`intel-igc-core`, `intel-igc-opencl`) is the LLVM-based JIT compiler that translates OpenCL/SPIR-V kernels to GPU ISA. OpenVINO's GPU plugin relies on this at runtime to JIT-compile IR model subgraphs for the target GPU microarchitecture (Xe-LP, Xe-HPG, Xe2, etc.).

**Device detection for profile selection:**

```python
import subprocess
import json

def detect_intel_gpu():
    """Detect Intel GPU hardware details for profile matching."""
    result = subprocess.run(
        ["clinfo", "--json"], capture_output=True, text=True
    )
    # Parse platform/device info
    # Alternative: use level-zero Python bindings or parse /sys/class/drm/
    ...
```

Alternatively, OpenVINO itself provides device enumeration:
```python
from openvino.runtime import Core
ie = Core()
available_devices = ie.available_devices  # ['CPU', 'GPU', 'GPU.0', 'GPU.1']
gpu_props = ie.get_property('GPU', 'FULL_DEVICE_NAME')  # 'Intel(R) Arc(TM) A770 Graphics'
gpu_mem = ie.get_property('GPU', 'GPU_DEVICE_TOTAL_MEM_SIZE')  # bytes
```

This is the correct mechanism for profile matching within iNIM — querying OpenVINO's device abstraction layer rather than parsing raw GPU driver output.

---

## 5. Proposed Architecture: iNIM (Intel NIM)

### 5.1 Architectural Overview

iNIM is a Docker container that runs three cooperating processes under supervisord:

1. **nginx** — External-facing reverse proxy (port 8000). Serves liveness immediately, routes readiness and inference requests to OVMS. Handles TLS, CORS.
2. **OVMS (OpenVINO Model Server)** — Core inference backend (port 8001, loopback). C++ inference server implementing continuous batching and paged attention via OpenVINO GenAI. Exposes OpenAI-compatible API.
3. **iNIM orchestrator** (Python) — Startup coordinator. Performs GPU detection, profile selection, model download/verification, OVMS launch with correct args, and readiness signaling.

All three are supervised by **supervisord** with fail-fast behavior: any unexpected exit causes the container to exit non-zero, signaling the orchestrator to reschedule.

### 5.2 Process Relationships

```
Container PID 1: supervisord
├── nginx (port 8000, external)
│   ├── /v1/health/live → 200 immediately (no backend)
│   ├── /v1/health/ready → proxy_pass → OVMS :8001/v3/health/ready
│   └── /v3/* → proxy_pass → OVMS :8001 (inference)
├── OVMS (port 8001, loopback only)
│   ├── /v3/chat/completions  (OpenAI-compat)
│   ├── /v3/completions
│   ├── /v3/embeddings
│   ├── /metrics              (Prometheus)
│   └── /v3/health/ready
└── iNIM orchestrator (exits after OVMS is ready)
    ├── GPU detection (OpenVINO Core.available_devices)
    ├── Profile selection (reads inim_manifest.yaml)
    ├── Model download (HuggingFace Hub or local cache)
    ├── Writes OVMS config (config.json + graph.pbtxt)
    └── Signals supervisord → start OVMS process
```

### 5.3 Port Architecture

| Port | Process | Access | Purpose |
|------|---------|--------|---------|
| 8000 | nginx | External (published) | Client-facing API + health |
| 8001 | OVMS REST | Loopback (127.0.0.1 only) | Inference backend |
| 8002 | OVMS gRPC | Loopback (127.0.0.1 only) | gRPC inference (internal use) |
| 9090 | nginx metrics stub | Loopback | Prometheus scrape point for nginx metrics |

This mirrors NIM's two-port design: external port (nginx) + internal backend (OVMS).

---

## 6. Layer-by-Layer Component Design

### 6.1 Layer 0: Host Driver Layer (Host OS)

This layer runs on the host machine, not inside the container. The container consumes it via device pass-through.

**Components:**
- Linux kernel with i915/xe DRM module and GuC/HuC firmware
- Intel NEO compute-runtime (`intel-opencl-icd`, `intel-level-zero-gpu`, `level-zero`)
- Intel Graphics Compiler (IGC) — `intel-igc-core`, `intel-igc-opencl`
- `/dev/dri/renderD*` — device nodes passed into container via Docker's `--device`

**Docker device pass-through:**
```bash
docker run \
  --device /dev/dri \
  -v /dev/dri:/dev/dri \
  --group-add render \
  ...
```

This is the Intel equivalent of NVIDIA's `--gpus all` / NVIDIA Container Toolkit.

### 6.2 Layer 1: Base Container OS

Ubuntu 22.04 LTS (Jammy) as the base image, with the Intel Graphics PPA packages installed. OpenVINO recommends Ubuntu 22.04 or 24.04 for Arc GPU deployments.

Base packages beyond standard Ubuntu:
- Intel compute-runtime packages (Level Zero + OpenCL)
- `intel-igc-core`, `intel-igc-opencl`
- `clinfo` (GPU enumeration tool, useful for diagnostics)
- `libva-drm2`, `libdrm-intel1` (for Intel media driver, optional but useful for iGPU)

### 6.3 Layer 2: OpenVINO Runtime

OpenVINO is Intel's inference optimization and deployment framework. It is to Intel hardware what CUDA+cuDNN is to NVIDIA — the foundational compute layer that ML frameworks dispatch to.

**Installation method (for Ubuntu 22.04):**
```bash
pip install openvino==2025.4.1
```
or via the Intel APT repository for the full toolkit including command-line tools.

**What OpenVINO provides:**

- **IR (Intermediate Representation) format**: The canonical serialized model format for OpenVINO, analogous to a TensorRT engine plan. Consists of `.xml` (graph) + `.bin` (weights).
- **GPU plugin**: Dispatches compute to Intel GPU via Level Zero / OpenCL. Handles kernel JIT compilation via IGC.
- **CPU plugin**: Fallback to CPU inference with AVX-512/AMX acceleration.
- **NPU plugin**: Dispatch to Neural Processing Unit (on applicable Intel SoCs).
- **Dynamic quantization**: INT4/INT8 activation quantization applied on-the-fly at inference time for GPU XMX-equipped hardware (Arc Alchemist, Battlemage, Lunar Lake, Arrow Lake).
- **KV-cache quantization**: KV cache stored in INT8/FP16/BF16 to reduce VRAM pressure.
- **Prefix caching**: Reuse of cached KV states for repeated prompt prefixes.

**OpenVINO GenAI library** (`openvino-genai`): Higher-level library built on top of OpenVINO Runtime providing:
- `LLMPipeline` class — handles tokenization, KV cache management, sampling
- Continuous batching scheduler
- Speculative decoding support

### 6.4 Layer 3: Universal Model Conversion Pipeline (First-Run)

This layer handles transforming any supported source model format into OpenVINO IR format before OVMS is launched. Unlike NIM — which requires pre-built engine artifacts and does not convert models at runtime — iNIM takes a more flexible approach: conversion runs at **first container start** when cached IR is not present, and the result is written to the cache so all subsequent starts skip it entirely. For users who pre-build IR offline, iNIM detects this and skips the layer completely.

iNIM is not restricted to models published by Intel's `OpenVINO/` organization on HuggingFace. Any model source in a supported format is acceptable — HuggingFace repos from any publisher, local directories, or files transferred out-of-band.

#### 6.4.1 Supported Source Formats and Detection Logic

The orchestrator detects the source format automatically by inspecting the local directory or remote HuggingFace repository before deciding which conversion path to execute. No user configuration is needed beyond pointing iNIM at the model source.

```
Format detection priority (applied in order):
1. Directory contains openvino_model.xml  →  already OpenVINO IR, skip conversion
2. Directory contains *.gguf              →  GGUF path (see §6.4.3)
3. Directory contains *.safetensors
   or pytorch_model*.bin + config.json   →  safetensors / PyTorch path (see §6.4.2)
4. Model source is a remote HF repo      →  probe file listing, then apply rules 1–3
```

**Detection implementation:**
```python
def detect_source_format(model_path: str) -> SourceFormat:
    import os
    from huggingface_hub import list_repo_files

    if os.path.isdir(model_path):
        files = os.listdir(model_path)
        if any(f.endswith(".xml") for f in files):
            return SourceFormat.OPENVINO_IR
        if any(f.endswith(".gguf") for f in files):
            return SourceFormat.GGUF
        if any(f.endswith(".safetensors") or f.startswith("pytorch_model") for f in files):
            return SourceFormat.PYTORCH_SAFETENSORS
        raise UnknownFormatError(f"Cannot determine format of model at {model_path}")

    # Remote HF repo — check file listing (works for any org, not just OpenVINO/)
    repo_files = list(list_repo_files(model_path, token=os.environ.get("HF_TOKEN")))
    if any(f.endswith(".xml") for f in repo_files):
        return SourceFormat.OPENVINO_IR
    if any(f.endswith(".gguf") for f in repo_files):
        return SourceFormat.GGUF
    if any(f.endswith(".safetensors") or "pytorch_model" in f for f in repo_files):
        return SourceFormat.PYTORCH_SAFETENSORS
    raise UnknownFormatError(f"Cannot determine format of remote repo {model_path}")
```

#### 6.4.2 Path A — Safetensors / PyTorch Weights → OpenVINO IR

This is the primary conversion path for models in their original HuggingFace training format, from any repository or local directory. It uses `optimum-cli`, Intel's official model conversion tool.

**Tool:** `optimum-intel + NNCF`
```bash
pip install "optimum-intel[openvino]" nncf
```

**Conversion command:**
```bash
optimum-cli export openvino \
  --model <hf_repo_id_or_local_directory> \
  --weight-format int4 \
  --group-size 128 \
  --ratio 1.0 \
  <output_ir_directory>
```

The `--model` argument accepts both a HuggingFace repo ID (`meta-llama/Llama-3.2-3B-Instruct`) and a **local directory path** containing safetensors or PyTorch weight files. iNIM can therefore convert models that were transferred out-of-band (rsync, USB, corporate model registry) with no HuggingFace Hub connectivity required.

**Quantization options (maps to iNIM precision profiles):**

| Profile Precision | `--weight-format` | `--group-size` | Notes |
|---|---|---|---|
| `fp16` | `fp16` | — | Highest accuracy; highest VRAM |
| `int8` | `int8` | — | Balanced; good for Arc 8GB+ |
| `int4` | `int4` | `128` | Default; best perf/quality on Arc with XMX |
| `int4-sym` | `int4` | `-1` (channel-wise) | Older Intel GPU; Core Ultra Series 1 |

**Output produced (required by OVMS):**
```
<output_ir_directory>/
├── openvino_model.xml          # Graph definition
├── openvino_model.bin          # Weight tensors (quantized)
├── openvino_tokenizer.xml      # Tokenizer IR (required by OVMS)
├── openvino_tokenizer.bin
├── openvino_detokenizer.xml
├── openvino_detokenizer.bin
├── config.json
└── tokenizer_config.json
```

`optimum-cli` produces all tokenizer IR files in the same pass as the model IR, satisfying OVMS's requirement for tokenizer IR alongside the model.

#### 6.4.3 Path B — GGUF → OpenVINO IR

GGUF is the model format of the llama.cpp ecosystem and is broadly distributed across HuggingFace (from publishers such as TheBloke, bartowski, lmstudio-community, and many others). iNIM handles GGUF via two sub-paths.

**Sub-path B1 — Native GGUF Reader (OpenVINO GenAI 2025.2+, preview)**

OpenVINO GenAI 2025.2 introduced a native GGUF reader that directly loads `.gguf` files and creates OpenVINO compute graphs on-the-fly — no intermediate PyTorch step, no Optimum dependency. Supported quantization types are Q4_0, Q4_K_M, Q8_0, and FP16. Validated architectures include Llama-3.1/3.2, Qwen2/2.5/3, and SmolLM.

```python
from openvino_genai import LLMPipeline

# Direct GGUF load — generates OV graph on-the-fly
pipe = LLMPipeline("/path/to/model.gguf", "GPU")

# Serialize the generated IR to cache for faster future starts
pipe.save_pretrained("/opt/inim/.cache/models/model-from-gguf/")
```

The orchestrator calls `save_pretrained()` after the first GGUF load, writing the converted IR to the cache directory. All subsequent container starts detect the cached IR and skip GGUF loading entirely.

**Important limitation:** This path is in preview as of OpenVINO 2025.4. Unsupported architectures, unusual GGUF quantization types (Q5_K_M, Q6_K, IQ3_M), and GPU-specific INT8 group-size compatibility issues can cause it to fail. The orchestrator catches these failures and falls through to Sub-path B2.

**Sub-path B2 — Offline GGUF Dequantization + optimum-cli (universal fallback)**

The universal fallback for any GGUF file regardless of architecture or quantization type. It dequantizes the GGUF back to FP16 and then runs `optimum-cli` to compress to the target profile precision. Slower than B1 but works on all architectures.

```bash
# Step 1: Dequantize GGUF → FP16 HuggingFace-compatible directory
python gguf_to_hf.py \
  --input /path/to/model.gguf \
  --output /tmp/model_fp16/ \
  --tokenizer <hf_repo_or_local_tokenizer_path>

# Step 2: Convert FP16 → OpenVINO IR with INT4 compression
optimum-cli export openvino \
  --model /tmp/model_fp16/ \
  --weight-format int4 \
  --group-size 128 \
  --output <cache_ir_directory>
```

**Tokenizer handling for GGUF:** GGUF files embed vocabulary but not the separate tokenizer IR that OVMS requires. The orchestrator resolves this by either pulling the tokenizer from the matching HuggingFace repo (via `INIM_TOKENIZER_REPO` if set, or inferred from the GGUF file's metadata) or allowing `optimum-cli` to produce tokenizer IR from the dequantized output in Sub-path B2.

**GGUF file selection from multi-file repos:**

Many HuggingFace repos publish multiple GGUF quantization variants. The orchestrator selects the best file for the target profile precision:

```python
GGUF_QUANTIZATION_PREFERENCE = {
    "int4":     ["Q4_K_M", "Q4_0", "Q5_K_M", "Q4_K_S"],
    "int8":     ["Q8_0"],
    "fp16":     ["F16", "BF16"],
    "int4-sym": ["Q4_0", "Q4_K_M"],
}

def select_gguf_file(repo_files: list[str], precision: str) -> str:
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
    for quant in GGUF_QUANTIZATION_PREFERENCE.get(precision, ["Q4_K_M"]):
        for f in gguf_files:
            if quant.lower() in f.lower():
                return f
    return gguf_files[0] if gguf_files else None
```

#### 6.4.4 Path C — Already OpenVINO IR (Pass-Through)

If the source directory already contains `openvino_model.xml` — whether from the `OpenVINO/` HuggingFace org, another publisher's pre-converted repo, a previous iNIM run, or an externally prepared IR directory — the orchestrator skips all conversion steps and passes the directory directly to OVMS. No tools are invoked; the cache manifest is written and startup proceeds immediately.

Pre-converted models from Intel's `OpenVINO/` org remain the fastest path (no conversion cost, Intel-validated accuracy), but they are treated as one option among many rather than a requirement:
- `OpenVINO/Llama-3.2-3B-Instruct-int4-ov`
- `OpenVINO/Qwen3-8B-int4-ov`
- `OpenVINO/Phi-3.5-mini-instruct-int4-ov`
- `OpenVINO/Mistral-7B-Instruct-v0.3-int4-ov`

#### 6.4.5 Conversion Decision Flowchart

```
Model source provided (INIM_MODEL or manifest source_model)
│
├── Check cache: IR already at INIM_CACHE_PATH for this model+profile?
│   └── Yes → verify checksums → OK: skip conversion, proceed to OVMS launch
│
└── No cached IR → detect source format
    │
    ├── Already IR (.xml present locally or in remote repo)
    │   └── Download/copy to cache → launch OVMS
    │
    ├── Safetensors or PyTorch weights
    │   └── Path A: optimum-cli export openvino
    │         → write IR to cache → launch OVMS
    │
    └── GGUF (.gguf present locally or in remote repo)
        ├── Download selected .gguf file to staging area
        ├── Probe architecture against supported list
        │   ├── Supported (Llama-3.x, Qwen2/2.5/3, SmolLM)
        │   │   └── Sub-path B1: OV GenAI native GGUF reader
        │   │       ├── Success → save_pretrained() to cache → launch OVMS
        │   │       └── Failure (unsupported quant, GPU group-size issue)
        │   │           └── Fall through to Sub-path B2
        │   └── Unsupported architecture OR B1 failure
        │       └── Sub-path B2: gguf_to_hf → optimum-cli
        │             → write IR to cache → launch OVMS
```

#### 6.4.6 Conversion Time Expectations

Conversion is a one-time cost. Subsequent starts read from cache and skip it entirely. The container remains live (`/v1/health/live` → 200) throughout; `/v1/health/ready` returns 503 until OVMS starts after conversion completes.

| Source Format | Model Size | Path | Approx. Time (modern CPU) |
|---|---|---|---|
| Already IR | any | C (pass-through) | Seconds |
| Safetensors / PyTorch | 3B params | A (optimum-cli INT4) | 3–8 min |
| Safetensors / PyTorch | 7B params | A (optimum-cli INT4) | 8–20 min |
| GGUF (native reader) | 3B params | B1 (OV GenAI) | 1–3 min |
| GGUF (dequantize + convert) | 7B params | B2 (gguf_to_hf + optimum-cli) | 15–30 min |

### 6.5 Layer 4: OpenVINO Model Server (OVMS)

OVMS is the core inference serving layer. It is a C++ server maintained by Intel as part of the OpenVINO toolkit ecosystem. This is the component that most directly replaces vLLM in the NIM stack.

**Key architectural parallels with vLLM:**

| vLLM (in NIM) | OVMS equivalent |
|---|---|
| PagedAttention scheduler | Continuous batching + paged attention (built-in, C++) |
| OpenAI-compatible HTTP API | OpenAI-compatible API at `/v3/chat/completions` |
| LM sampling (greedy, top-p, temp) | Same sampling algorithms via OpenVINO GenAI |
| KV cache management | Managed internally, configurable via `plugin_config` |
| Multi-model serving | Supported via `models_config_list` |
| Prometheus `/metrics` | Native Prometheus metrics endpoint |
| gRPC inference API | Native gRPC support (KServe/TF Serving protocol) |
| Streaming (SSE) | Streaming supported via chunked HTTP response |

**OVMS deployment for LLM (the graph.pbtxt model):**

OVMS deploys LLMs via a **MediaPipe graph** with an **LLM Calculator** node. The graph defines the serving pipeline declaratively:

```
# graph.pbtxt (OVMS LLM pipeline definition)
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node: {
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {
    tag_index: 'LOOPBACK:0'
    back_edge: true
  }
  node_options: {
    [type.googleapis.com/mediapipe.LLMCalculatorOptions] {
      models_path: "/models/llama-3.2-3b-int4-ov"
      max_num_seqs: 256
      cache_size: 4
      device: "GPU"
    }
  }
}
```

The `config.json` then points OVMS to this graph:
```json
{
  "model_config_list": [],
  "mediapipe_config_list": [
    {
      "name": "llama-3.2-3b-instruct",
      "base_path": "/models/llama-3.2-3b-int4-ov"
    }
  ]
}
```

**OVMS vs custom server decision:**

OVMS is mature, battle-tested on Intel hardware, and implements continuous batching and paged attention natively. Writing a custom inference server would duplicate this work without benefit. The iNIM orchestration layer is built **around** OVMS, not instead of it — OVMS stays, everything else is scaffolding.

### 6.6 Layer 5: nginx Reverse Proxy

nginx is used identically to its role in NVIDIA NIM. It is not replaced because it has no NVIDIA dependency.

**nginx configuration for iNIM:**

```nginx
upstream ovms_backend {
    server 127.0.0.1:8001;
    keepalive 32;
}

server {
    listen 8000;

    # Liveness: served directly by nginx, no backend dependency
    location = /v1/health/live {
        return 200 '{"status":"ok"}';
        add_header Content-Type application/json;
    }

    # Readiness: proxy to OVMS health endpoint
    location = /v1/health/ready {
        proxy_pass http://ovms_backend/v3/health/ready;
        proxy_read_timeout 5s;
    }

    # Inference endpoints: map NIM-style /v1/* to OVMS /v3/*
    location /v1/chat/completions {
        proxy_pass http://ovms_backend/v3/chat/completions;
        proxy_buffering off;      # Required for SSE streaming
        proxy_read_timeout 300s;
    }

    location /v1/completions {
        proxy_pass http://ovms_backend/v3/completions;
        proxy_buffering off;
        proxy_read_timeout 300s;
    }

    location /v1/embeddings {
        proxy_pass http://ovms_backend/v3/embeddings;
    }

    location /v1/models {
        proxy_pass http://ovms_backend/v3/models;
    }

    # Prometheus metrics
    location /v1/metrics {
        proxy_pass http://ovms_backend/metrics;
    }

    # Reject all other paths
    location / {
        return 404;
    }
}
```

**Note on endpoint versioning:** OVMS uses `/v3/` for its OpenAI-compatible routes (matching OpenAI's versioned paths). nginx transparently maps the NIM-standard `/v1/` prefix to OVMS's `/v3/` prefix. Client code written for NIM requires no changes.

### 6.7 Layer 6: iNIM Orchestrator (Python)

This is the only fully custom component in iNIM — there is no off-the-shelf equivalent. It serves the same purpose as NIM's internal startup logic, which NVIDIA has not open-sourced.

**Responsibilities:**

1. **GPU Detection**: Query OpenVINO Runtime for available Intel GPU devices, enumerate VRAM, device name, and architecture tier.
2. **Profile Selection**: Load `inim_manifest.yaml`, filter profiles by GPU compatibility, select optimal profile.
3. **Model Source Resolution**: Determine whether the model source is a local path or a remote HuggingFace repo (any org, any publisher). Check the local cache first.
4. **Format Detection**: Inspect the source (local directory or remote repo file listing) and classify it as OpenVINO IR, safetensors/PyTorch, or GGUF.
5. **Model Conversion** (if IR not cached): Execute the appropriate conversion path — pass-through for IR, `optimum-cli` for safetensors/PyTorch, native GGUF reader or offline dequantize pipeline for GGUF. Write resulting IR to `INIM_CACHE_PATH`.
6. **Model Verification**: Validate IR file integrity (SHA-256 checksums) and write `.inim_cache_manifest.json`.
7. **OVMS Configuration**: Write `config.json` and `graph.pbtxt` for OVMS based on selected profile and cached IR path.
8. **Signal Readiness**: Write a sentinel file or call supervisord's XML-RPC API to start the OVMS process after configuration is complete.
9. **Exit Cleanly**: The orchestrator exits 0 after successful OVMS launch; supervisord does not restart it.

---

## 7. Model Profile System

The iNIM model profile system directly mirrors NIM's manifest/profile design, adapted for Intel hardware axes.

### 7.1 Profile Naming Convention

```
ovms-<precision>-<device_tier>-<pipeline_type>[-feat_lora]
```

Where:
- `<precision>`: `fp16`, `int8`, `int4`, `int4-sym`
- `<device_tier>`: `igpu` (UHD/Xe integrated), `arc` (Arc discrete), `arc-hq` (Arc with XMX accelerators like A770/B580)
- `<pipeline_type>`: `cb` (continuous batching), `st` (stateful, for NPU or low-concurrency)
- Optional: `-feat_lora` for LoRA adapter support

**Examples:**
- `ovms-int4-arc-hq-cb` — INT4, Arc discrete GPU with XMX, continuous batching
- `ovms-int8-igpu-cb` — INT8, integrated Xe GPU, continuous batching
- `ovms-fp16-arc-cb` — FP16, Arc discrete, continuous batching (high VRAM only)
- `ovms-int4-igpu-st` — INT4, integrated GPU, stateful pipeline (low-memory iGPU)

### 7.2 Manifest File Structure

```yaml
# /opt/inim/etc/inim_manifest.yaml
version: "1.0"
model_name: "meta-llama/Llama-3.2-3B-Instruct"
model_id: "llama-3.2-3b-instruct"

profiles:
  - id: "a3f9b72c1d45e68f"
    description: "ovms-int4-arc-hq-cb"
    backend: "ovms"
    precision: "int4"
    pipeline_type: "continuous_batching"
    device_filter:
      min_vram_mb: 8192
      arch_tiers: ["arc_alchemist", "arc_battlemage"]
      xmx_required: false
    ovms_config:
      max_num_seqs: 256
      cache_size: 4           # KV cache pages in GB
      dynamic_quantization: true
      kv_cache_precision: "int8"
    source_model:
      hf_repo: "OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
      format: "openvino_ir"
    memory_estimate_mb: 7800
    runnable: true

  - id: "b7e2a14f8c9d03e5"
    description: "ovms-int4-igpu-cb"
    backend: "ovms"
    precision: "int4"
    pipeline_type: "continuous_batching"
    device_filter:
      min_vram_mb: 4096
      arch_tiers: ["xe_integrated", "uhd_integrated"]
      xmx_required: false
    ovms_config:
      max_num_seqs: 64
      cache_size: 2
      dynamic_quantization: false
      kv_cache_precision: "fp16"
    source_model:
      hf_repo: "OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
      format: "openvino_ir"
    memory_estimate_mb: 3900
    runnable: true

  - id: "c8d1f25a9e0b47c3"
    description: "ovms-fp16-arc-hq-cb"
    backend: "ovms"
    precision: "fp16"
    pipeline_type: "continuous_batching"
    device_filter:
      min_vram_mb: 16384
      arch_tiers: ["arc_alchemist", "arc_battlemage"]
    ovms_config:
      max_num_seqs: 512
      cache_size: 8
      dynamic_quantization: false
      kv_cache_precision: "fp16"
    source_model:
      hf_repo: "meta-llama/Llama-3.2-3B-Instruct"
      format: "pytorch"           # Will be converted at startup
    memory_estimate_mb: 15500
    runnable: false               # Requires >= 16GB Arc GPU
```

### 7.3 Profile Selection Algorithm

```python
def select_profile(manifest: dict, detected_gpu: GPUInfo) -> dict:
    """
    Mirrors NIM's profile selection priority:
    1. Filter by VRAM fit and arch tier
    2. Prefer higher performance profiles (arc-hq > arc > igpu)
    3. Prefer lower precision (int4 > int8 > fp16)
    4. Prefer continuous_batching over stateful
    5. Apply INIM_MODEL_PROFILE env var override
    """
    override = os.environ.get("INIM_MODEL_PROFILE")
    if override:
        return get_profile_by_id_or_description(manifest, override)

    candidates = [
        p for p in manifest["profiles"]
        if detected_gpu.vram_mb >= p["device_filter"]["min_vram_mb"]
        and detected_gpu.arch_tier in p["device_filter"]["arch_tiers"]
        and p["runnable"]
    ]

    if not candidates:
        raise NoCompatibleProfileError(
            f"GPU {detected_gpu.name} ({detected_gpu.vram_mb}MB) "
            f"is not compatible with any profile in this manifest."
        )

    # Sort: continuous_batching > stateful, int4 < int8 < fp16, arc-hq > arc > igpu
    return sorted(candidates, key=profile_score)[0]
```

---

## 8. Container Startup Lifecycle

The iNIM startup sequence mirrors NIM's step-by-step:

```
1. supervisord starts (PID 1)
   ├── Immediately starts nginx
   │   → /v1/health/live: 200 OK (from nginx directly)
   │   → /v1/health/ready: 503 (OVMS not started yet)
   └── Immediately starts iNIM orchestrator

2. iNIM orchestrator runs:
   a. Enumerate GPUs via OpenVINO Core.available_devices
   b. Read /opt/inim/etc/inim_manifest.yaml
   c. Select compatible profile (or use INIM_MODEL_PROFILE override)
   d. Print profile selection summary to stdout (mirrors NIM's output)
   e. Check INIM_CACHE_PATH for cached model IR files
      → If cached and checksums match: skip download
      → If not cached: pull from HuggingFace Hub
        → If source is "openvino_ir": huggingface-cli download <hf_repo>
        → If source is "pytorch": run optimum-cli export openvino ...
   f. Write OVMS config.json and graph.pbtxt for selected profile
   g. Write /tmp/inim_ready_to_start sentinel file
   h. Exit 0

3. supervisord detects sentinel → starts OVMS process
   OVMS args constructed from profile ovms_config:
   --rest_port 8001
   --config_path /tmp/ovms_config.json
   --target_device GPU
   --pipeline_type LM (for continuous batching profiles)

4. iNIM orchestrator (or supervisord event listener) polls OVMS:
   GET http://127.0.0.1:8001/v3/health/ready
   → Wait up to INIM_READY_TIMEOUT seconds (default 300)

5. OVMS signals ready:
   → nginx /v1/health/ready begins returning 200 OK
   → Orchestrator container exits 0 (supervisord ignores this)

6. Traffic flows:
   Client → nginx :8000 → OVMS :8001 → OpenVINO GPU plugin → Intel GPU

On SIGTERM (docker stop):
   supervisord → SIGTERM OVMS (graceful drain of in-flight requests)
   supervisord → SIGTERM nginx
   Container exits 0
```

---

## 9. API Surface and Proxy Layer

### 9.1 Client-Facing Endpoints

iNIM presents an OpenAI-compatible API identical to NIM:

| iNIM Route | OVMS Backend Route | Notes |
|---|---|---|
| `POST /v1/chat/completions` | `POST /v3/chat/completions` | Full streaming (SSE) supported |
| `POST /v1/completions` | `POST /v3/completions` | |
| `POST /v1/embeddings` | `POST /v3/embeddings` | Requires embedding model profile |
| `GET /v1/models` | `GET /v3/models` | Lists loaded model(s) |
| `GET /v1/health/live` | Served by nginx directly | Immediate, no backend dependency |
| `GET /v1/health/ready` | `GET /v3/health/ready` | True when OVMS model is loaded |
| `GET /v1/metrics` | `GET /metrics` | Prometheus format |

All other paths return `404 Not Found` — same secure-by-default posture as NIM.

### 9.2 Model Name Handling

OVMS exposes models by the `name` field in `config.json`. iNIM sets this name from:
1. `INIM_SERVED_MODEL_NAME` environment variable (if set)
2. The `model_id` field in `inim_manifest.yaml` (default)

This ensures `GET /v1/models` returns a stable, predictable model name.

### 9.3 Streaming Support

OVMS supports server-sent events (SSE) streaming for `chat/completions` with `"stream": true`. nginx is configured with `proxy_buffering off` on inference routes to pass SSE chunks through immediately. This matches NIM's streaming behavior.

---

## 10. Model Acquisition and Caching

### 10.1 Cache Structure

```
INIM_CACHE_PATH (default: /opt/inim/.cache)
└── models/
    └── OpenVINO--Llama-3.2-3B-Instruct-int4-ov/
        ├── openvino_model.xml
        ├── openvino_model.bin
        ├── openvino_tokenizer.xml
        ├── openvino_tokenizer.bin
        ├── openvino_detokenizer.xml
        ├── openvino_detokenizer.bin
        ├── config.json
        ├── tokenizer_config.json
        └── .inim_cache_manifest.json   # checksums + profile ID that produced this
```

### 10.2 Download Flow

```
Is model cached at INIM_CACHE_PATH?
├── Yes → verify checksums
│   ├── OK → skip download
│   └── Mismatch → re-download
└── No → select download strategy
    ├── Profile source.format == "openvino_ir"
    │   └── huggingface-cli download OpenVINO/<model_id>-ov \
    │         --local-dir $INIM_CACHE_PATH/models/<model_id>
    └── Profile source.format == "pytorch"
        ├── Download pytorch weights: huggingface-cli download <hf_repo>
        └── Convert: optimum-cli export openvino \
              --model <hf_repo> \
              --weight-format <profile.precision> \
              --output $INIM_CACHE_PATH/models/<model_id>
```

### 10.3 Offline / Air-Gap Deployment

For air-gap deployments, the model IR directory can be pre-mounted at `INIM_CACHE_PATH`. The orchestrator detects cached models and skips download entirely. This mirrors NIM's air-gap deployment support.

Environment variables:
- `INIM_CACHE_PATH` — base cache directory (mount a persistent volume here)
- `HF_ENDPOINT` — override HuggingFace endpoint (for enterprise mirrors)
- `HF_TOKEN` — authentication token for gated models
- `INIM_OFFLINE=1` — disable all network access, fail if model not cached

---

## 11. Observability Stack

### 11.1 Health Probes

| Endpoint | Served By | Returns 200 When |
|---|---|---|
| `/v1/health/live` | nginx (direct) | Container is running |
| `/v1/health/ready` | nginx → OVMS | Model is loaded and serving |

### 11.2 Prometheus Metrics

OVMS exposes native Prometheus metrics at `/metrics`. iNIM routes this through nginx at `/v1/metrics`. Key metrics exposed by OVMS:

- `ovms_requests_success_total` — successful inference requests
- `ovms_requests_fail_total` — failed requests
- `ovms_request_time_us` — request latency histogram (microseconds)
- `ovms_inference_time_us` — pure inference time histogram
- `ovms_current_requests` — in-flight requests gauge
- `ovms_model_version_status` — model serving status per version

**Additional Intel GPU metrics** can be obtained by integrating **Intel's oneAPI GPU metrics library** or **xpu-smi** (Intel's analog to nvidia-smi) as a sidecar metric exporter exposing:
- GPU utilization (%)
- GPU VRAM used/total
- GPU power draw (W)
- GPU clock frequency

### 11.3 Structured Logging

OVMS produces structured logs. iNIM configures:
- `OVMS_LOG_LEVEL` → maps to `INIM_LOG_LEVEL` (debug/info/warn/error)
- Distributed tracing: nginx forwards `X-Request-Id` and `Traceparent` headers to OVMS
- Log format: JSON Lines to stdout (Docker log driver compatible)

---

## 12. Security Considerations

### 12.1 Attack Surface Reduction

- OVMS listens only on `127.0.0.1:8001` (loopback) — not accessible from outside the container
- nginx is the only external-facing process
- All non-configured paths return `404` (same as NIM's secure-by-default)
- Model weights loaded from safetensors format where available (preferred over PyTorch pickle)

### 12.2 TLS Termination

nginx handles TLS. Configure via environment variables:
- `INIM_TLS_CERT_PATH`, `INIM_TLS_KEY_PATH` — paths to cert/key files
- When set, nginx enables HTTPS on port 8000

### 12.3 CORS

nginx's CORS configuration mirrors NIM:
- `INIM_ALLOWED_ORIGINS` — comma-separated list of allowed origins
- Default: `*` (open) — should be restricted in production

### 12.4 Model Integrity

- SHA-256 checksums for downloaded IR files stored in `.inim_cache_manifest.json`
- Verified on every startup before OVMS is launched
- Mismatch triggers re-download (or fail-fast in `INIM_OFFLINE=1` mode)

---

## 13. GPU Tier Considerations

### 13.1 Intel UHD Graphics (Gen9/Gen10/Gen11 — Integrated)

- Shared system memory (no dedicated VRAM)
- Effective "GPU memory" is a portion of RAM (typically 512MB–2GB reserved, up to ~16GB shared)
- Level Zero support: yes (Gen11+)
- XMX (matrix extensions): **no** — dynamic quantization benefit limited
- Recommended precision profile: `int4-sym` (symmetric, no group-size, Core Ultra Series 1 constraint)
- Recommended pipeline: `stateful` (low concurrency; continuous batching has overhead for low-concurrency iGPU scenarios)
- Max practical model size: ~3B parameters at INT4 on machines with 16GB+ RAM

### 13.2 Intel Xe Graphics (Core Ultra Series 1/2 — Integrated)

- Dedicated shared memory architecture (LPDDR5, 8–64GB shared)
- XMX support: **yes** (Meteor Lake / Lunar Lake / Arrow Lake with `--device CPU` fallback)
- Dynamic quantization: effective for INT4 models with XMX
- Continuous batching: supported (OVMS 2025.1+)
- Recommended precision: `int4` (group-size 128)
- Recommended pipeline: `continuous_batching`
- Max practical model size: 7B parameters at INT4 on 16GB+ shared memory

### 13.3 Intel Arc Discrete GPU (Alchemist A-series, Battlemage B-series)

- Dedicated GDDR6 VRAM (4GB–24GB depending on SKU)
- XMX support: **yes** (all Arc discrete SKUs)
- Dynamic quantization with INT4: full benefit on Alchemist and Battlemage
- Continuous batching: fully supported and recommended
- Recommended precision: `int4` (default), `int8` (higher quality), `fp16` (16GB+ VRAM only)
- Max practical model size: 70B at INT4 on 24GB Arc (theoretical; validated up to 13B on B580)

### 13.4 Profile Compatibility Matrix

| GPU Tier | min VRAM | Default Profile | Max Model (practical) |
|---|---|---|---|
| UHD (iGPU, Gen11) | 2GB | `int4-igpu-st` | 1B params |
| Xe iGPU (MTL/LNL) | 8GB shared | `int4-igpu-cb` | 3–7B params |
| Arc A380 (6GB) | 6GB | `int4-arc-hq-cb` | 7B params |
| Arc A770 (16GB) | 16GB | `int4-arc-hq-cb` | 13–30B params |
| Arc B580 (12GB) | 12GB | `int4-arc-hq-cb` | 13B params |
| Arc B770 (16GB) | 16GB | `int4-arc-hq-cb` | 30B params |

---

## 14. Implementation Roadmap

### Phase 1: Foundation (Driver + OVMS Baseline)
- Verify Intel GPU driver stack (Level Zero, IGC) on target Ubuntu 22.04 machine
- Pull OVMS `openvino/model_server:2025.4.1-gpu` and validate model serving manually
- Test a single model (e.g. `OpenVINO/Phi-3.5-mini-instruct-int4-ov`) via `curl`
- Validate OpenAI-compat API responses match expected format

### Phase 2: Manifest and Orchestrator
- Implement GPU detection module (OpenVINO Core enumeration)
- Define `inim_manifest.yaml` schema and write manifests for first 2–3 target models
- Implement profile selection algorithm in Python
- Implement model download logic (HF Hub, with fallback to optimum-cli conversion)
- Write OVMS `config.json` / `graph.pbtxt` generator from selected profile

### Phase 3: Container Assembly
- Write Dockerfile: Ubuntu 22.04 base → Intel GPU PPA → OpenVINO → OVMS → iNIM orchestrator
- Integrate nginx with routing config
- Configure supervisord for process supervision
- Implement fail-fast behavior (any process exit → container exit)
- Add SIGTERM handler for graceful shutdown

### Phase 4: Operational Layer
- Implement health probe logic (nginx /v1/health/live, OVMS-backed /v1/health/ready)
- Wire Prometheus metrics through nginx
- Implement structured JSON logging with correlation header forwarding
- Add model integrity checking (SHA-256)

### Phase 5: Hardening and Testing
- Validate all three GPU tiers (UHD, Xe iGPU, Arc discrete) if available
- Run OpenAI client compatibility tests
- Load test with concurrent requests (validate continuous batching behavior)
- Air-gap deployment test
- TLS configuration test

---

## 15. Dependency Manifest

### Host (Ubuntu 22.04)

| Package | Source | Purpose |
|---|---|---|
| `intel-opencl-icd` | Intel Graphics PPA | OpenCL ICD for Intel GPU |
| `intel-level-zero-gpu` | Intel Graphics PPA | Level Zero GPU driver |
| `level-zero` | Intel Graphics PPA | Level Zero loader |
| `intel-igc-core` | Intel Graphics PPA | Intel Graphics Compiler core |
| `intel-igc-opencl` | Intel Graphics PPA | IGC OpenCL frontend |
| Linux 6.5+ HWE kernel | Ubuntu HWE | i915/xe DRM module with Arc support |

### Container (Python/PyPI)

| Package | Version | Purpose |
|---|---|---|
| `openvino` | 2025.4.x | OpenVINO Runtime Python API |
| `openvino-genai` | 2025.4.x | GenAI pipeline library |
| `optimum-intel[openvino]` | ≥1.22 | HuggingFace model conversion to OV IR |
| `nncf` | ≥2.14 | Neural Network Compression Framework |
| `huggingface_hub` | ≥0.24 | Model download from HF Hub |
| `pyyaml` | ≥6.0 | Manifest parsing |
| `requests` | ≥2.31 | HTTP health polling |

### Container (System)

| Package | Source | Purpose |
|---|---|---|
| `nginx` | Ubuntu repo | Reverse proxy |
| `supervisor` | Ubuntu repo | Process supervision (supervisord) |
| `clinfo` | Ubuntu repo | GPU enumeration diagnostics |
| `python3.11` | Ubuntu/deadsnakes | Orchestrator runtime |

### Container Images (Base Options)

| Image | Purpose |
|---|---|
| `openvino/model_server:2025.4.1-gpu` | Start from Intel's official OVMS GPU image (recommended — avoids manual driver install in container) |
| `ubuntu:22.04` | Build from scratch — requires manual Intel PPA driver install |

**Recommended**: Start from `openvino/model_server:2025.4.1-gpu` as the base image. This image already contains the correct version of OpenVINO, OVMS binary, and Intel GPU runtime libraries. iNIM adds nginx, supervisord, and the orchestrator layer on top.

---

## Appendix: Environment Variable Reference

| Variable | Default | Description |
|---|---|---|
| `INIM_SERVER_PORT` | `8000` | External nginx port |
| `INIM_HEALTH_PORT` | — | Optional dedicated health probe port |
| `INIM_CACHE_PATH` | `/opt/inim/.cache` | Model cache directory |
| `INIM_MODEL_PROFILE` | — | Override profile selection |
| `INIM_SERVED_MODEL_NAME` | From manifest | Model name in `/v1/models` response |
| `INIM_LOG_LEVEL` | `info` | Logging verbosity |
| `INIM_OFFLINE` | `0` | Set to `1` for air-gap mode |
| `INIM_READY_TIMEOUT` | `300` | Seconds to wait for OVMS ready |
| `INIM_TLS_CERT_PATH` | — | Path to TLS certificate |
| `INIM_TLS_KEY_PATH` | — | Path to TLS private key |
| `INIM_ALLOWED_ORIGINS` | `*` | CORS allowed origins |
| `HF_TOKEN` | — | HuggingFace authentication token |
| `HF_ENDPOINT` | `https://huggingface.co` | HuggingFace endpoint override |
| `OVMS_LOG_LEVEL` | `INFO` | OVMS internal log level |

---

*Architecture document version 1.0 — May 2026*
*Target: Ubuntu 22.04 LTS | OpenVINO 2025.4 | OVMS 2025.4 | Intel Arc/Xe/UHD GPUs*
