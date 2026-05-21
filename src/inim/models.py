"""
Data models for iNIM orchestrator.

Defines all data classes, enums, and type structures used across the
orchestrator components — GPU info, model profiles, source formats,
OVMS configuration, and cache manifests.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SourceFormat(enum.Enum):
    """Model source format detected from local path or remote repository."""
    OPENVINO_IR = "openvino_ir"
    PYTORCH_SAFETENSORS = "pytorch_safetensors"
    GGUF = "gguf"
    UNKNOWN = "unknown"


class ArchTier(enum.Enum):
    """
    Intel GPU architecture tier for profile matching.

    Ordered from lowest capability to highest — used in profile scoring.
    """
    UHD_INTEGRATED = "uhd_integrated"          # Gen9/Gen10/Gen11 iGPU
    XE_INTEGRATED = "xe_integrated"            # Core Ultra Series 1/2 (MTL/LNL/ARL)
    ARC_ALCHEMIST = "arc_alchemist"            # Arc A-series (A310, A380, A580, A750, A770)
    ARC_BATTLEMAGE = "arc_battlemage"           # Arc B-series (B580, B570, B770)

    @classmethod
    def from_device_name(cls, device_name: str) -> "ArchTier":
        """
        Classify an Intel GPU into an architecture tier based on the
        device name string returned by OpenVINO Core.

        Examples:
            "Intel(R) Arc(TM) A770 Graphics" → ARC_ALCHEMIST
            "Intel(R) Arc(TM) B580 Graphics" → ARC_BATTLEMAGE
            "Intel(R) UHD Graphics 770"      → UHD_INTEGRATED
            "Intel(R) Graphics (MTL)"        → XE_INTEGRATED
        """
        name_lower = device_name.lower()

        # Arc discrete — Battlemage (B-series)
        if "arc" in name_lower and any(
            f"b{sku}" in name_lower for sku in ("580", "570", "770")
        ):
            return cls.ARC_BATTLEMAGE

        # Arc discrete — Alchemist (A-series)
        if "arc" in name_lower:
            return cls.ARC_ALCHEMIST

        # Xe integrated (Meteor Lake, Lunar Lake, Arrow Lake identifiers)
        xe_markers = ["mtl", "lnl", "arl", "meteor", "lunar", "arrow",
                      "core ultra", "xe"]
        if any(marker in name_lower for marker in xe_markers):
            return cls.XE_INTEGRATED

        # Fallback: UHD integrated
        return cls.UHD_INTEGRATED


class PipelineType(enum.Enum):
    """OVMS pipeline type for LLM serving."""
    CONTINUOUS_BATCHING = "continuous_batching"
    STATEFUL = "stateful"


class Precision(enum.Enum):
    """Model weight precision levels, ordered from lowest to highest quality."""
    INT4_SYM = "int4-sym"
    INT4 = "int4"
    INT8 = "int8"
    FP16 = "fp16"

    @property
    def optimum_weight_format(self) -> str:
        """Return the --weight-format flag value for optimum-cli."""
        mapping = {
            "int4-sym": "int4",
            "int4": "int4",
            "int8": "int8",
            "fp16": "fp16",
        }
        return mapping[self.value]

    @property
    def optimum_group_size(self) -> Optional[int]:
        """Return the --group-size flag value for optimum-cli, or None."""
        mapping = {
            "int4-sym": -1,     # channel-wise (symmetric)
            "int4": 128,        # group-size 128
            "int8": None,
            "fp16": None,
        }
        return mapping[self.value]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GPUInfo:
    """Information about a detected Intel GPU device."""
    name: str                          # e.g. "Intel(R) Arc(TM) A770 Graphics"
    device_id: str                     # OpenVINO device ID, e.g. "GPU.0"
    vram_mb: int                       # Total GPU memory in MB
    arch_tier: ArchTier                # Classified architecture tier
    xmx_supported: bool = False        # Has XMX (matrix extensions)

    def __post_init__(self):
        # XMX is available on all Arc discrete and Xe integrated (MTL+)
        if self.arch_tier in (
            ArchTier.ARC_ALCHEMIST,
            ArchTier.ARC_BATTLEMAGE,
            ArchTier.XE_INTEGRATED,
        ):
            self.xmx_supported = True


@dataclass
class DeviceFilter:
    """Profile device compatibility filter."""
    min_vram_mb: int
    arch_tiers: list[str]
    xmx_required: bool = False


@dataclass
class SourceModel:
    """Model source specification within a profile."""
    hf_repo: str                       # HuggingFace repo ID or local path
    format: str                        # "openvino_ir", "pytorch", "gguf"


@dataclass
class OVMSConfig:
    """OVMS runtime configuration derived from a profile."""
    max_num_seqs: int = 256
    cache_size: int = 4                # KV cache pages in GB
    dynamic_quantization: bool = True
    kv_cache_precision: str = "int8"


@dataclass
class Profile:
    """
    A single model serving profile from inim_manifest.yaml.

    Profiles encode the full configuration for a specific GPU tier
    and precision combination.
    """
    id: str
    description: str
    backend: str                       # Always "ovms" for now
    precision: str                     # "fp16", "int8", "int4", "int4-sym"
    pipeline_type: str                 # "continuous_batching" or "stateful"
    device_filter: DeviceFilter
    ovms_config: OVMSConfig
    source_model: SourceModel
    memory_estimate_mb: int
    runnable: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profile":
        """Construct a Profile from a manifest YAML dictionary."""
        return cls(
            id=data["id"],
            description=data["description"],
            backend=data.get("backend", "ovms"),
            precision=data["precision"],
            pipeline_type=data["pipeline_type"],
            device_filter=DeviceFilter(
                min_vram_mb=data["device_filter"]["min_vram_mb"],
                arch_tiers=data["device_filter"]["arch_tiers"],
                xmx_required=data["device_filter"].get("xmx_required", False),
            ),
            ovms_config=OVMSConfig(
                max_num_seqs=data["ovms_config"].get("max_num_seqs", 256),
                cache_size=data["ovms_config"].get("cache_size", 4),
                dynamic_quantization=data["ovms_config"].get("dynamic_quantization", True),
                kv_cache_precision=data["ovms_config"].get("kv_cache_precision", "int8"),
            ),
            source_model=SourceModel(
                hf_repo=data["source_model"]["hf_repo"],
                format=data["source_model"]["format"],
            ),
            memory_estimate_mb=data.get("memory_estimate_mb", 0),
            runnable=data.get("runnable", True),
        )


@dataclass
class Manifest:
    """Parsed iNIM model manifest."""
    version: str
    model_name: str                    # Full model name (e.g. "meta-llama/Llama-3.2-3B-Instruct")
    model_id: str                      # Short ID (e.g. "llama-3.2-3b-instruct")
    profiles: list[Profile] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Manifest":
        """Construct a Manifest from parsed YAML."""
        return cls(
            version=data.get("version", "1.0"),
            model_name=data["model_name"],
            model_id=data["model_id"],
            profiles=[Profile.from_dict(p) for p in data.get("profiles", [])],
        )


@dataclass
class CacheManifest:
    """
    Cache manifest stored as .inim_cache_manifest.json alongside cached IR files.

    Records the profile and checksums that produced this cached model,
    enabling validation on subsequent starts.
    """
    model_name: str
    profile_id: str
    precision: str
    source_format: str
    source_repo: str
    conversion_timestamp: str          # ISO 8601
    files: dict[str, str] = field(default_factory=dict)  # filename → SHA-256 hex digest


# ---------------------------------------------------------------------------
# GGUF quantization preference mapping (for selecting best GGUF variant)
# ---------------------------------------------------------------------------

GGUF_QUANTIZATION_PREFERENCE: dict[str, list[str]] = {
    "int4":     ["Q4_K_M", "Q4_0", "Q5_K_M", "Q4_K_S"],
    "int8":     ["Q8_0"],
    "fp16":     ["F16", "BF16"],
    "int4-sym": ["Q4_0", "Q4_K_M"],
}


def select_gguf_file(repo_files: list[str], precision: str) -> Optional[str]:
    """
    Select the best GGUF file from a repository for a given precision target.

    Searches for GGUF files matching the preferred quantization types for the
    target precision, falling back to the first GGUF file found.

    Args:
        repo_files: List of filenames in the repository.
        precision: Target precision string (e.g. "int4", "int8", "fp16").

    Returns:
        Selected GGUF filename, or None if no GGUF files exist.
    """
    gguf_files = [f for f in repo_files if f.endswith(".gguf")]
    if not gguf_files:
        return None

    for quant in GGUF_QUANTIZATION_PREFERENCE.get(precision, ["Q4_K_M"]):
        for f in gguf_files:
            if quant.lower() in f.lower():
                return f

    # Fallback: return first GGUF file
    return gguf_files[0]
