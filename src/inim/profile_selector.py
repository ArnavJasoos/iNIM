"""
Profile selection algorithm for iNIM orchestrator.

Mirrors NIM's profile selection priority:
  1. Filter by VRAM fit and architecture tier
  2. Prefer higher performance profiles (arc-hq > arc > igpu)
  3. Prefer lower precision (int4 > int8 > fp16) for better perf/memory
  4. Prefer continuous_batching over stateful
  5. Apply INIM_MODEL_PROFILE env var override

The manifest is a YAML file containing a catalog of pre-validated
configurations (profiles) for a given model.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from inim.exceptions import ManifestError, NoCompatibleProfileError
from inim.logging_config import get_logger
from inim.models import ArchTier, GPUInfo, Manifest, Profile

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights for profile ranking
# ---------------------------------------------------------------------------

# Architecture tier priority: higher is better
_TIER_SCORE = {
    "arc_battlemage": 40,
    "arc_alchemist": 30,
    "xe_integrated": 20,
    "uhd_integrated": 10,
}

# Precision priority: lower precision = better performance
_PRECISION_SCORE = {
    "int4-sym": 40,
    "int4": 30,
    "int8": 20,
    "fp16": 10,
}

# Pipeline type priority
_PIPELINE_SCORE = {
    "continuous_batching": 20,
    "stateful": 10,
}


def load_manifest(manifest_path: str) -> Manifest:
    """
    Load and parse an iNIM model manifest YAML file.

    Args:
        manifest_path: Path to the manifest YAML file.

    Returns:
        Parsed Manifest object.

    Raises:
        ManifestError: If the file cannot be read or parsed.
    """
    logger.info(
        f"Loading manifest: {manifest_path}",
        extra={"component": "profile_selector"},
    )

    try:
        with open(manifest_path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        raise ManifestError(f"Manifest file not found: {manifest_path}")
    except yaml.YAMLError as e:
        raise ManifestError(
            f"Invalid YAML in manifest {manifest_path}: {e}"
        )

    if not data:
        raise ManifestError(f"Manifest file is empty: {manifest_path}")

    required_fields = ["model_name", "model_id", "profiles"]
    for field in required_fields:
        if field not in data:
            raise ManifestError(
                f"Manifest missing required field '{field}': {manifest_path}"
            )

    try:
        manifest = Manifest.from_dict(data)
    except (KeyError, TypeError, ValueError) as e:
        raise ManifestError(
            f"Error parsing manifest structure: {e}",
            details=f"File: {manifest_path}",
        )

    logger.info(
        f"Loaded manifest: {manifest.model_name} with "
        f"{len(manifest.profiles)} profiles",
        extra={"component": "profile_selector", "model": manifest.model_name},
    )

    return manifest


def find_manifest_for_model(
    manifests_dir: str,
    model_name: str,
) -> Optional[str]:
    """
    Find a manifest file matching a model name in the manifests directory.

    Searches for YAML files whose model_name or model_id matches the
    given model name. Returns the path to the first matching manifest.

    Args:
        manifests_dir: Directory containing manifest YAML files.
        model_name: Model name or ID to search for.

    Returns:
        Path to matching manifest file, or None if not found.
    """
    if not os.path.isdir(manifests_dir):
        return None

    for filename in sorted(os.listdir(manifests_dir)):
        if not filename.endswith((".yaml", ".yml")):
            continue

        filepath = os.path.join(manifests_dir, filename)
        try:
            with open(filepath, "r") as f:
                data = yaml.safe_load(f)
            if not data:
                continue

            if (
                data.get("model_name") == model_name
                or data.get("model_id") == model_name
                or model_name.lower() in data.get("model_name", "").lower()
            ):
                return filepath

        except (yaml.YAMLError, OSError):
            continue

    return None


def select_profile(
    manifest: Manifest,
    detected_gpu: GPUInfo,
    profile_override: Optional[str] = None,
) -> Profile:
    """
    Select the optimal profile for the detected GPU.

    Implements the NIM-equivalent profile selection algorithm:
      1. Check for INIM_MODEL_PROFILE override
      2. Filter profiles by VRAM and architecture tier compatibility
      3. Score and rank compatible profiles
      4. Return the highest-scoring profile

    Args:
        manifest: Parsed model manifest.
        detected_gpu: Information about the detected GPU.
        profile_override: Optional profile ID or description override
                         (from INIM_MODEL_PROFILE env var).

    Returns:
        Selected Profile object.

    Raises:
        NoCompatibleProfileError: If no profile matches the GPU.
    """
    # --- Override: exact profile selection ---
    if profile_override:
        logger.info(
            f"Profile override specified: {profile_override}",
            extra={"component": "profile_selector"},
        )
        profile = _get_profile_by_id_or_description(manifest, profile_override)
        if profile:
            logger.info(
                f"Using overridden profile: {profile.description} ({profile.id})",
                extra={
                    "component": "profile_selector",
                    "profile": profile.description,
                },
            )
            return profile
        else:
            logger.warning(
                f"Override profile '{profile_override}' not found in manifest. "
                "Falling back to automatic selection.",
                extra={"component": "profile_selector"},
            )

    # --- Filter: compatible profiles ---
    candidates = _filter_compatible_profiles(manifest, detected_gpu)

    if not candidates:
        raise NoCompatibleProfileError(
            f"GPU {detected_gpu.name} ({detected_gpu.vram_mb}MB VRAM, "
            f"tier: {detected_gpu.arch_tier.value}) is not compatible with "
            f"any profile in manifest for {manifest.model_name}.",
            details=(
                f"Available profiles: "
                f"{[p.description for p in manifest.profiles]}. "
                f"GPU arch tier: {detected_gpu.arch_tier.value}, "
                f"VRAM: {detected_gpu.vram_mb}MB."
            ),
        )

    # --- Score and rank ---
    scored = sorted(candidates, key=_profile_score, reverse=True)

    selected = scored[0]
    logger.info(
        f"Selected profile: {selected.description} ({selected.id}) — "
        f"precision: {selected.precision}, "
        f"pipeline: {selected.pipeline_type}, "
        f"est. memory: {selected.memory_estimate_mb}MB",
        extra={
            "component": "profile_selector",
            "profile": selected.description,
            "model": manifest.model_name,
        },
    )

    # Log alternatives if any
    if len(scored) > 1:
        alternatives = [p.description for p in scored[1:]]
        logger.info(
            f"Alternative compatible profiles: {alternatives}",
            extra={"component": "profile_selector"},
        )

    return selected


def _filter_compatible_profiles(
    manifest: Manifest,
    gpu: GPUInfo,
) -> list[Profile]:
    """
    Filter manifest profiles to those compatible with the detected GPU.

    A profile is compatible if:
      - The GPU's VRAM meets the profile's minimum
      - The GPU's arch tier is in the profile's allowed tiers
      - The profile is marked as runnable
      - XMX requirements are met (if specified)
    """
    candidates = []

    for profile in manifest.profiles:
        # Must be runnable
        if not profile.runnable:
            logger.debug(
                f"Skipping profile {profile.description}: not runnable",
                extra={"component": "profile_selector"},
            )
            continue

        # VRAM check
        if gpu.vram_mb < profile.device_filter.min_vram_mb:
            logger.debug(
                f"Skipping profile {profile.description}: "
                f"requires {profile.device_filter.min_vram_mb}MB VRAM, "
                f"GPU has {gpu.vram_mb}MB",
                extra={"component": "profile_selector"},
            )
            continue

        # Architecture tier check
        if gpu.arch_tier.value not in profile.device_filter.arch_tiers:
            logger.debug(
                f"Skipping profile {profile.description}: "
                f"requires {profile.device_filter.arch_tiers}, "
                f"GPU is {gpu.arch_tier.value}",
                extra={"component": "profile_selector"},
            )
            continue

        # XMX requirement check
        if profile.device_filter.xmx_required and not gpu.xmx_supported:
            logger.debug(
                f"Skipping profile {profile.description}: requires XMX",
                extra={"component": "profile_selector"},
            )
            continue

        candidates.append(profile)

    return candidates


def _profile_score(profile: Profile) -> int:
    """
    Compute a score for ranking compatible profiles.

    Higher score = preferred profile.

    Scoring dimensions:
      - Precision: int4 preferred over int8 over fp16 (perf/memory)
      - Pipeline: continuous_batching preferred over stateful
      - Memory efficiency: lower memory_estimate = better fit
    """
    score = 0
    score += _PRECISION_SCORE.get(profile.precision, 0)
    score += _PIPELINE_SCORE.get(profile.pipeline_type, 0)

    # Bonus for dynamic quantization support
    if profile.ovms_config.dynamic_quantization:
        score += 5

    return score


def _get_profile_by_id_or_description(
    manifest: Manifest,
    identifier: str,
) -> Optional[Profile]:
    """
    Look up a profile by its ID (hash) or description string.

    Supports partial matching on description for user convenience.
    """
    for profile in manifest.profiles:
        if profile.id == identifier:
            return profile
        if profile.description == identifier:
            return profile
        # Partial match on description
        if identifier.lower() in profile.description.lower():
            return profile

    return None
