"""
Model acquisition and resolution pipeline for iNIM orchestrator.

Coordinates the full model acquisition flow:
  1. Check local cache for valid IR files
  2. If not cached: determine source format
  3. Download and/or convert to OpenVINO IR
  4. Write cache manifest for future runs

Supports any model source: HuggingFace repos (any org), local directories,
and local files. Respects offline mode (INIM_OFFLINE=1).
"""

from __future__ import annotations

import os
from typing import Optional

from inim.cache_manager import check_cache, write_cache_manifest
from inim.converter.base import get_converter
from inim.exceptions import ModelConversionError, ModelDownloadError
from inim.format_detector import detect_source_format
from inim.logging_config import PhaseTimer, get_logger
from inim.models import Profile, SourceFormat

logger = get_logger(__name__)


def resolve_model(
    model_source: str,
    cache_dir: str,
    profile: Profile,
    hf_token: Optional[str] = None,
    offline: bool = False,
) -> str:
    """
    Resolve a model source to a local directory of OpenVINO IR files.

    This is the main entry point for the model acquisition pipeline.
    It handles the full flow from cache check through download/conversion.

    Args:
        model_source: HuggingFace repo ID, local directory, or local file path.
        cache_dir: Target cache directory for this model+profile combination.
        profile: Selected profile (determines precision for conversion).
        hf_token: Optional HuggingFace token for gated/private models.
        offline: If True, fail instead of downloading (INIM_OFFLINE=1).

    Returns:
        Path to directory containing ready-to-serve OpenVINO IR files.

    Raises:
        ModelDownloadError: If download fails or model not available offline.
        ModelConversionError: If format conversion fails.
    """
    logger.info(
        f"Resolving model: source={model_source}, cache={cache_dir}, "
        f"precision={profile.precision}",
        extra={"component": "model_resolver", "model": model_source},
    )

    # --- Step 1: Check cache ---
    with PhaseTimer(logger, "cache_check", model=model_source):
        cached_path = check_cache(cache_dir, profile.id)

    if cached_path:
        logger.info(
            f"Using cached model: {cached_path}",
            extra={"component": "model_resolver", "model": model_source},
        )
        return cached_path

    # --- Step 2: Offline check ---
    if offline:
        raise ModelDownloadError(
            f"Model not found in cache and INIM_OFFLINE=1 is set. "
            f"Cannot download {model_source}.",
            details=(
                f"Cache directory: {cache_dir}. "
                "Pre-populate the cache or unset INIM_OFFLINE."
            ),
        )

    # --- Step 3: Determine the effective source ---
    # If the profile specifies a source model, use that instead of the raw input
    effective_source = _determine_effective_source(model_source, profile)

    logger.info(
        f"Effective model source: {effective_source}",
        extra={"component": "model_resolver"},
    )

    # --- Step 4: Detect source format ---
    with PhaseTimer(logger, "format_detection", model=effective_source):
        source_format = detect_source_format(effective_source, hf_token)

    logger.info(
        f"Detected source format: {source_format.value}",
        extra={
            "component": "model_resolver",
            "model": effective_source,
        },
    )

    # --- Step 5: Convert to OpenVINO IR ---
    os.makedirs(cache_dir, exist_ok=True)

    with PhaseTimer(logger, "model_conversion", model=effective_source):
        converter = get_converter(source_format)
        logger.info(
            f"Using converter: {converter.__class__.__name__}",
            extra={"component": "model_resolver"},
        )

        ir_dir = converter.convert(
            source_path=effective_source,
            target_dir=cache_dir,
            profile=profile,
            hf_token=hf_token,
        )

    # --- Step 6: Write cache manifest ---
    with PhaseTimer(logger, "cache_manifest_write"):
        write_cache_manifest(
            cache_dir=ir_dir,
            model_name=model_source,
            profile_id=profile.id,
            precision=profile.precision,
            source_format=source_format.value,
            source_repo=effective_source,
        )

    logger.info(
        f"Model resolved successfully: {ir_dir}",
        extra={"component": "model_resolver", "model": model_source},
    )

    return ir_dir


def _determine_effective_source(
    model_source: str,
    profile: Profile,
) -> str:
    """
    Determine the effective model source to use.

    Priority:
    1. If model_source is a local path that exists, use it directly
    2. If the profile has a source_model.hf_repo, use that
    3. Fall back to model_source as an HF repo ID

    The profile's source_model can point to a different repo than
    the user's INIM_MODEL — for example, a profile might specify
    an Intel-optimized IR repo while the user specified the base model name.
    """
    # Local path takes highest priority
    if os.path.exists(model_source):
        return model_source

    # Profile-specified source
    if profile.source_model and profile.source_model.hf_repo:
        profile_source = profile.source_model.hf_repo

        # If profile source is different from user source, log the mapping
        if profile_source != model_source:
            logger.info(
                f"Profile maps model '{model_source}' → "
                f"source repo '{profile_source}'",
                extra={"component": "model_resolver"},
            )

        return profile_source

    # Default: use model_source as-is (assumed HF repo ID)
    return model_source
