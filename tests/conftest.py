"""
Shared test fixtures for iNIM test suite.

Provides mock GPU info, sample manifests, temporary directories,
and other test utilities used across all test modules.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest

from inim.models import ArchTier, GPUInfo, Manifest, Profile


# ---------------------------------------------------------------------------
# GPU fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def arc_a770_gpu() -> GPUInfo:
    """Intel Arc A770 (16GB discrete GPU)."""
    return GPUInfo(
        name="Intel(R) Arc(TM) A770 Graphics",
        device_id="GPU.0",
        vram_mb=16384,
        arch_tier=ArchTier.ARC_ALCHEMIST,
    )


@pytest.fixture
def arc_b580_gpu() -> GPUInfo:
    """Intel Arc B580 (12GB discrete GPU)."""
    return GPUInfo(
        name="Intel(R) Arc(TM) B580 Graphics",
        device_id="GPU.0",
        vram_mb=12288,
        arch_tier=ArchTier.ARC_BATTLEMAGE,
    )


@pytest.fixture
def arc_a380_gpu() -> GPUInfo:
    """Intel Arc A380 (6GB discrete GPU)."""
    return GPUInfo(
        name="Intel(R) Arc(TM) A380 Graphics",
        device_id="GPU.0",
        vram_mb=6144,
        arch_tier=ArchTier.ARC_ALCHEMIST,
    )


@pytest.fixture
def xe_igpu() -> GPUInfo:
    """Intel Xe iGPU (Core Ultra, ~8GB shared)."""
    return GPUInfo(
        name="Intel(R) Graphics (MTL)",
        device_id="GPU.0",
        vram_mb=8192,
        arch_tier=ArchTier.XE_INTEGRATED,
    )


@pytest.fixture
def uhd_igpu() -> GPUInfo:
    """Intel UHD Graphics (2GB shared)."""
    return GPUInfo(
        name="Intel(R) UHD Graphics 770",
        device_id="GPU.0",
        vram_mb=2048,
        arch_tier=ArchTier.UHD_INTEGRATED,
    )


# ---------------------------------------------------------------------------
# Manifest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_manifest_data() -> dict:
    """Raw manifest data for Llama 3.2 3B."""
    return {
        "version": "1.0",
        "model_name": "meta-llama/Llama-3.2-3B-Instruct",
        "model_id": "llama-3.2-3b-instruct",
        "profiles": [
            {
                "id": "prof-int4-arc-hq",
                "description": "ovms-int4-arc-hq-cb",
                "backend": "ovms",
                "precision": "int4",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 8192,
                    "arch_tiers": ["arc_alchemist", "arc_battlemage"],
                },
                "ovms_config": {
                    "max_num_seqs": 256,
                    "cache_size": 4,
                    "dynamic_quantization": True,
                    "kv_cache_precision": "int8",
                },
                "source_model": {
                    "hf_repo": "OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
                    "format": "openvino_ir",
                },
                "memory_estimate_mb": 7800,
                "runnable": True,
            },
            {
                "id": "prof-int4-igpu",
                "description": "ovms-int4-igpu-cb",
                "backend": "ovms",
                "precision": "int4",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 4096,
                    "arch_tiers": ["xe_integrated"],
                },
                "ovms_config": {
                    "max_num_seqs": 64,
                    "cache_size": 2,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": "OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
                    "format": "openvino_ir",
                },
                "memory_estimate_mb": 3900,
                "runnable": True,
            },
            {
                "id": "prof-int4-uhd",
                "description": "ovms-int4-igpu-st",
                "backend": "ovms",
                "precision": "int4-sym",
                "pipeline_type": "stateful",
                "device_filter": {
                    "min_vram_mb": 2048,
                    "arch_tiers": ["uhd_integrated"],
                },
                "ovms_config": {
                    "max_num_seqs": 16,
                    "cache_size": 1,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": "OpenVINO/Llama-3.2-3B-Instruct-int4-ov",
                    "format": "openvino_ir",
                },
                "memory_estimate_mb": 2500,
                "runnable": True,
            },
            {
                "id": "prof-fp16-arc",
                "description": "ovms-fp16-arc-hq-cb",
                "backend": "ovms",
                "precision": "fp16",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 16384,
                    "arch_tiers": ["arc_alchemist", "arc_battlemage"],
                },
                "ovms_config": {
                    "max_num_seqs": 512,
                    "cache_size": 8,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": "meta-llama/Llama-3.2-3B-Instruct",
                    "format": "pytorch",
                },
                "memory_estimate_mb": 15500,
                "runnable": True,
            },
            {
                "id": "prof-not-runnable",
                "description": "ovms-int8-arc-experimental",
                "backend": "ovms",
                "precision": "int8",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 8192,
                    "arch_tiers": ["arc_alchemist"],
                },
                "ovms_config": {
                    "max_num_seqs": 128,
                    "cache_size": 4,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": "test/model",
                    "format": "pytorch",
                },
                "memory_estimate_mb": 10000,
                "runnable": False,
            },
        ],
    }


@pytest.fixture
def sample_manifest(sample_manifest_data) -> Manifest:
    """Parsed Manifest object."""
    return Manifest.from_dict(sample_manifest_data)


# ---------------------------------------------------------------------------
# Temporary directory fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory, cleaned up after test."""
    with tempfile.TemporaryDirectory(prefix="inim_test_") as d:
        yield d


@pytest.fixture
def temp_model_dir(temp_dir) -> str:
    """Create a temporary directory with mock OpenVINO IR files."""
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir)

    # Create mock IR files
    for filename in [
        "openvino_model.xml",
        "openvino_model.bin",
        "openvino_tokenizer.xml",
        "openvino_tokenizer.bin",
        "openvino_detokenizer.xml",
        "openvino_detokenizer.bin",
        "config.json",
        "tokenizer_config.json",
    ]:
        filepath = os.path.join(model_dir, filename)
        Path(filepath).write_text(f"mock content for {filename}")

    return model_dir


@pytest.fixture
def temp_safetensors_dir(temp_dir) -> str:
    """Create a temp dir with mock safetensors files."""
    model_dir = os.path.join(temp_dir, "safetensors_model")
    os.makedirs(model_dir)

    for filename in [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ]:
        Path(os.path.join(model_dir, filename)).write_text(f"mock {filename}")

    return model_dir


@pytest.fixture
def temp_gguf_dir(temp_dir) -> str:
    """Create a temp dir with mock GGUF files."""
    model_dir = os.path.join(temp_dir, "gguf_model")
    os.makedirs(model_dir)

    for filename in [
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]:
        Path(os.path.join(model_dir, filename)).write_text(f"mock {filename}")

    return model_dir


@pytest.fixture
def temp_manifest_file(temp_dir, sample_manifest_data) -> str:
    """Create a temporary manifest YAML file."""
    import yaml

    manifest_path = os.path.join(temp_dir, "test_manifest.yaml")
    with open(manifest_path, "w") as f:
        yaml.dump(sample_manifest_data, f)

    return manifest_path
