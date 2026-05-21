#!/usr/bin/env python3
"""
iNIM Orchestrator — Main entry point.

This is the startup coordinator for the iNIM container. It runs once
at container start, performing:
  1. GPU detection
  2. Profile selection from manifest
  3. Model acquisition (cache check → download → convert)
  4. OVMS configuration generation
  5. Sentinel file write to trigger OVMS start
  6. OVMS health polling until ready
  7. Clean exit

The orchestrator exits 0 after successful OVMS startup; supervisord
does not restart it. If any step fails, the orchestrator exits non-zero
and supervisord shuts down the entire container.
"""

from __future__ import annotations

import os
import sys
import signal
import time
from pathlib import Path

from inim.config import INIMConfig
from inim.exceptions import INIMError
from inim.logging_config import PhaseTimer, get_logger, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Startup banner
# ---------------------------------------------------------------------------

BANNER = r"""
  _ _   _ ___ __  __
 (_) \ | |_ _|  \/  |
 | |\  | || || |\/| |
 | | |\  || || |  | |
 |_|_| \_|___|_|  |_|

 Intel NIM-Equivalent Inference Container
 OpenVINO Model Server on Intel GPU
 v{version}
"""


def main() -> int:
    """
    Main orchestrator entry point.

    Returns:
        Exit code: 0 on success, 1 on failure.
    """
    from inim import __version__

    # --- Step 0: Load config and initialize logging ---
    config = INIMConfig.from_env()
    setup_logging(level=config.log_level, component="orchestrator")

    # Print banner
    print(BANNER.format(version=__version__), flush=True)

    # Validate configuration
    issues = config.validate()
    if issues:
        for issue in issues:
            logger.error(f"Configuration error: {issue}")

        # Only fail on critical issues (no model specified)
        critical = [i for i in issues if "INIM_MODEL" in i]
        if critical:
            return 1

    # Ensure cache directories exist
    config.ensure_directories()

    try:
        # --- Step 1: GPU Detection ---
        with PhaseTimer(logger, "gpu_detection"):
            from inim.gpu_detector import detect_primary_gpu
            gpu = detect_primary_gpu()

        logger.info(
            f"Primary GPU: {gpu.name} | "
            f"VRAM: {gpu.vram_mb}MB | "
            f"Tier: {gpu.arch_tier.value} | "
            f"XMX: {gpu.xmx_supported}",
            extra={"component": "orchestrator", "gpu": gpu.name},
        )

        # --- Step 2: Load manifest and select profile ---
        with PhaseTimer(logger, "profile_selection"):
            from inim.profile_selector import (
                find_manifest_for_model,
                load_manifest,
                select_profile,
            )

            manifest_path = _resolve_manifest(config)
            manifest = load_manifest(manifest_path)
            profile = select_profile(
                manifest=manifest,
                detected_gpu=gpu,
                profile_override=config.model_profile,
            )

        logger.info(
            f"Selected profile: {profile.description} | "
            f"Precision: {profile.precision} | "
            f"Pipeline: {profile.pipeline_type} | "
            f"Max seqs: {profile.ovms_config.max_num_seqs} | "
            f"KV cache: {profile.ovms_config.cache_size}GB",
            extra={
                "component": "orchestrator",
                "profile": profile.description,
            },
        )

        # --- Step 3: Resolve model (cache → download → convert) ---
        model_source = config.model or manifest.model_name
        cache_dir = config.get_model_cache_dir(
            profile.source_model.hf_repo.replace("/", "--")
            if profile.source_model
            else manifest.model_id
        )

        with PhaseTimer(logger, "model_resolution", model=model_source):
            from inim.model_resolver import resolve_model
            model_dir = resolve_model(
                model_source=model_source,
                cache_dir=cache_dir,
                profile=profile,
                hf_token=config.hf_token,
                offline=config.offline,
            )

        logger.info(
            f"Model ready at: {model_dir}",
            extra={"component": "orchestrator"},
        )

        # --- Step 4: Generate OVMS configuration ---
        model_name = (
            config.served_model_name
            or manifest.model_id
        )

        with PhaseTimer(logger, "ovms_config_generation"):
            from inim.ovms_configurator import generate_ovms_config
            config_path, graph_path = generate_ovms_config(
                model_dir=model_dir,
                model_name=model_name,
                profile=profile,
                config_output_path=config.ovms_config_path,
                device="GPU",
            )

        logger.info(
            f"OVMS config: {config_path} | Graph: {graph_path}",
            extra={"component": "orchestrator"},
        )

        # --- Step 5: Signal readiness for OVMS start ---
        _write_sentinel(config.sentinel_path)
        logger.info(
            f"Sentinel written: {config.sentinel_path} — OVMS will start",
            extra={"component": "orchestrator"},
        )

        # --- Step 6: Wait for OVMS to become ready ---
        with PhaseTimer(logger, "ovms_startup_wait"):
            from inim.health_monitor import wait_for_ovms_ready
            wait_for_ovms_ready(
                host="127.0.0.1",
                port=config.ovms_port,
                timeout=config.ready_timeout,
            )

        # --- Done! ---
        logger.info(
            "═══════════════════════════════════════════════════",
            extra={"component": "orchestrator"},
        )
        logger.info(
            f"iNIM is ready! Serving model '{model_name}' on port "
            f"{config.server_port}",
            extra={"component": "orchestrator", "model": model_name},
        )
        logger.info(
            f"  → POST http://localhost:{config.server_port}"
            f"/v1/chat/completions",
            extra={"component": "orchestrator"},
        )
        logger.info(
            f"  → GET  http://localhost:{config.server_port}"
            f"/v1/health/ready",
            extra={"component": "orchestrator"},
        )
        logger.info(
            "═══════════════════════════════════════════════════",
            extra={"component": "orchestrator"},
        )

        return 0

    except INIMError as e:
        logger.error(
            f"iNIM startup failed: {e}",
            extra={"component": "orchestrator"},
        )
        if e.details:
            logger.error(
                f"Details: {e.details}",
                extra={"component": "orchestrator"},
            )
        return 1

    except KeyboardInterrupt:
        logger.info(
            "Interrupted by user",
            extra={"component": "orchestrator"},
        )
        return 130

    except Exception as e:
        logger.error(
            f"Unexpected error during startup: {e}",
            extra={"component": "orchestrator"},
            exc_info=True,
        )
        return 1


def _resolve_manifest(config: INIMConfig) -> str:
    """
    Resolve the manifest file path for the current model.

    Search order:
    1. INIM_MANIFEST_PATH if it points to a specific file
    2. Search manifests directory for a file matching INIM_MODEL
    3. Generate a dynamic manifest for direct model specification
    """
    # If manifest_path points to a specific YAML file, use it
    if config.manifest_path and os.path.isfile(config.manifest_path):
        return config.manifest_path

    # Search manifests directory
    if config.manifest_path and os.path.isdir(config.manifest_path):
        from inim.profile_selector import find_manifest_for_model

        if config.model:
            found = find_manifest_for_model(config.manifest_path, config.model)
            if found:
                return found

    # If no pre-built manifest found, generate a dynamic one
    if config.model:
        return _generate_dynamic_manifest(config)

    raise INIMError(
        "No manifest found and no model specified. "
        "Set INIM_MODEL to a HuggingFace repo ID or local path, "
        "or set INIM_MANIFEST_PATH to a manifest YAML file."
    )


def _generate_dynamic_manifest(config: INIMConfig) -> str:
    """
    Generate a dynamic manifest for a model not in the pre-built manifests.

    This enables iNIM to work with any model, not just those with
    pre-defined manifests. The dynamic manifest creates profiles
    for common GPU tiers with reasonable defaults.
    """
    import yaml

    model_source = config.model
    model_id = model_source.replace("/", "-").lower()

    manifest = {
        "version": "1.0",
        "model_name": model_source,
        "model_id": model_id,
        "profiles": [
            # Arc discrete GPU — INT4 (best default for most models)
            {
                "id": f"dynamic-int4-arc-cb-{model_id[:8]}",
                "description": "ovms-int4-arc-hq-cb",
                "backend": "ovms",
                "precision": "int4",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 4096,
                    "arch_tiers": ["arc_alchemist", "arc_battlemage"],
                },
                "ovms_config": {
                    "max_num_seqs": 256,
                    "cache_size": 4,
                    "dynamic_quantization": True,
                    "kv_cache_precision": "int8",
                },
                "source_model": {
                    "hf_repo": model_source,
                    "format": "auto",
                },
                "memory_estimate_mb": 4000,
                "runnable": True,
            },
            # Arc discrete GPU — INT8
            {
                "id": f"dynamic-int8-arc-cb-{model_id[:8]}",
                "description": "ovms-int8-arc-cb",
                "backend": "ovms",
                "precision": "int8",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 8192,
                    "arch_tiers": ["arc_alchemist", "arc_battlemage"],
                },
                "ovms_config": {
                    "max_num_seqs": 128,
                    "cache_size": 4,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": model_source,
                    "format": "auto",
                },
                "memory_estimate_mb": 8000,
                "runnable": True,
            },
            # Integrated GPU — INT4
            {
                "id": f"dynamic-int4-igpu-cb-{model_id[:8]}",
                "description": "ovms-int4-igpu-cb",
                "backend": "ovms",
                "precision": "int4",
                "pipeline_type": "continuous_batching",
                "device_filter": {
                    "min_vram_mb": 2048,
                    "arch_tiers": [
                        "xe_integrated",
                        "uhd_integrated",
                    ],
                },
                "ovms_config": {
                    "max_num_seqs": 64,
                    "cache_size": 2,
                    "dynamic_quantization": False,
                    "kv_cache_precision": "fp16",
                },
                "source_model": {
                    "hf_repo": model_source,
                    "format": "auto",
                },
                "memory_estimate_mb": 2000,
                "runnable": True,
            },
        ],
    }

    # Write dynamic manifest to tmp
    manifest_path = "/tmp/inim_dynamic_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    logger.info(
        f"Generated dynamic manifest for {model_source}: {manifest_path}",
        extra={"component": "orchestrator"},
    )

    return manifest_path


def _write_sentinel(sentinel_path: str) -> None:
    """
    Write the sentinel file to signal OVMS should start.

    supervisord watches for this file to trigger the OVMS process.
    """
    os.makedirs(os.path.dirname(sentinel_path) or "/tmp", exist_ok=True)
    Path(sentinel_path).write_text(
        f"ready\ntimestamp={time.time()}\n"
    )


if __name__ == "__main__":
    sys.exit(main())
