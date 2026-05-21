"""
Cache management for iNIM orchestrator.

Handles model cache verification, SHA-256 checksum computation,
and .inim_cache_manifest.json read/write. The cache ensures that
model conversion is a one-time cost — subsequent container starts
read from cache and skip conversion entirely.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Optional

from inim.exceptions import CacheIntegrityError
from inim.logging_config import get_logger
from inim.models import CacheManifest

logger = get_logger(__name__)

# Cache manifest filename (hidden file alongside IR files)
CACHE_MANIFEST_FILENAME = ".inim_cache_manifest.json"


def check_cache(cache_dir: str, profile_id: str) -> Optional[str]:
    """
    Check if a valid cached model exists for the given profile.

    Verifies:
    1. The cache directory exists and contains a manifest
    2. The manifest's profile ID matches (same precision/config)
    3. All listed files exist with matching SHA-256 checksums

    Args:
        cache_dir: Path to the model's cache directory.
        profile_id: ID of the currently selected profile.

    Returns:
        Path to the cached model directory if valid, None otherwise.
    """
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_FILENAME)

    if not os.path.exists(manifest_path):
        logger.info(
            f"No cache manifest found at {manifest_path}",
            extra={"component": "cache_manager"},
        )
        return None

    # Load manifest
    try:
        manifest = load_cache_manifest(manifest_path)
    except Exception as e:
        logger.warning(
            f"Failed to load cache manifest: {e}",
            extra={"component": "cache_manager"},
        )
        return None

    # Check profile match
    if manifest.profile_id != profile_id:
        logger.info(
            f"Cache manifest profile mismatch: "
            f"cached={manifest.profile_id}, current={profile_id}",
            extra={"component": "cache_manager"},
        )
        return None

    # Verify file checksums
    try:
        verify_checksums(cache_dir, manifest)
    except CacheIntegrityError as e:
        logger.warning(
            f"Cache integrity check failed: {e}",
            extra={"component": "cache_manager"},
        )
        return None

    logger.info(
        f"Valid cache found: {cache_dir} "
        f"(profile: {manifest.profile_id}, "
        f"converted: {manifest.conversion_timestamp})",
        extra={"component": "cache_manager"},
    )
    return cache_dir


def verify_checksums(cache_dir: str, manifest: CacheManifest) -> None:
    """
    Verify SHA-256 checksums of all files in the cache manifest.

    Args:
        cache_dir: Directory containing cached IR files.
        manifest: Cache manifest with expected checksums.

    Raises:
        CacheIntegrityError: If any file is missing or has wrong checksum.
    """
    for filename, expected_hash in manifest.files.items():
        filepath = os.path.join(cache_dir, filename)

        if not os.path.exists(filepath):
            raise CacheIntegrityError(
                f"Cached file missing: {filepath}",
                details=f"Expected by manifest: {filename}",
            )

        actual_hash = compute_sha256(filepath)
        if actual_hash != expected_hash:
            raise CacheIntegrityError(
                f"Checksum mismatch for {filename}: "
                f"expected={expected_hash[:16]}..., "
                f"actual={actual_hash[:16]}...",
                details=f"File may be corrupted: {filepath}",
            )

    logger.debug(
        f"All {len(manifest.files)} file checksums verified",
        extra={"component": "cache_manager"},
    )


def write_cache_manifest(
    cache_dir: str,
    model_name: str,
    profile_id: str,
    precision: str,
    source_format: str,
    source_repo: str,
) -> CacheManifest:
    """
    Compute checksums for all IR files and write a cache manifest.

    Called after successful model conversion or download to record
    the cache state for future verification.

    Args:
        cache_dir: Directory containing the IR files.
        model_name: Full model name.
        profile_id: Profile ID that produced this cache.
        precision: Precision used (e.g., "int4", "fp16").
        source_format: Source format (e.g., "openvino_ir", "pytorch_safetensors").
        source_repo: Source repo or path.

    Returns:
        Written CacheManifest object.
    """
    # Compute checksums for all relevant files
    file_checksums: dict[str, str] = {}

    for filename in sorted(os.listdir(cache_dir)):
        filepath = os.path.join(cache_dir, filename)

        # Skip directories, hidden files (except our manifest), and very large files
        if os.path.isdir(filepath):
            continue
        if filename == CACHE_MANIFEST_FILENAME:
            continue

        # Only checksum model-related files
        relevant_extensions = (
            ".xml", ".bin", ".json", ".model", ".txt", ".safetensors",
        )
        if not any(filename.endswith(ext) for ext in relevant_extensions):
            continue

        file_checksums[filename] = compute_sha256(filepath)

    manifest = CacheManifest(
        model_name=model_name,
        profile_id=profile_id,
        precision=precision,
        source_format=source_format,
        source_repo=source_repo,
        conversion_timestamp=datetime.now(timezone.utc).isoformat(),
        files=file_checksums,
    )

    # Write manifest
    manifest_path = os.path.join(cache_dir, CACHE_MANIFEST_FILENAME)
    with open(manifest_path, "w") as f:
        json.dump(
            {
                "model_name": manifest.model_name,
                "profile_id": manifest.profile_id,
                "precision": manifest.precision,
                "source_format": manifest.source_format,
                "source_repo": manifest.source_repo,
                "conversion_timestamp": manifest.conversion_timestamp,
                "files": manifest.files,
            },
            f,
            indent=2,
        )

    logger.info(
        f"Cache manifest written: {manifest_path} "
        f"({len(file_checksums)} files checksummed)",
        extra={"component": "cache_manager"},
    )

    return manifest


def load_cache_manifest(manifest_path: str) -> CacheManifest:
    """
    Load a cache manifest from a JSON file.

    Args:
        manifest_path: Path to .inim_cache_manifest.json.

    Returns:
        Parsed CacheManifest object.
    """
    with open(manifest_path, "r") as f:
        data = json.load(f)

    return CacheManifest(
        model_name=data["model_name"],
        profile_id=data["profile_id"],
        precision=data["precision"],
        source_format=data["source_format"],
        source_repo=data["source_repo"],
        conversion_timestamp=data["conversion_timestamp"],
        files=data.get("files", {}),
    )


def compute_sha256(filepath: str, chunk_size: int = 8192) -> str:
    """
    Compute SHA-256 hex digest of a file.

    Uses chunked reading for memory efficiency with large model files.

    Args:
        filepath: Path to file.
        chunk_size: Read buffer size in bytes.

    Returns:
        SHA-256 hex digest string.
    """
    sha256 = hashlib.sha256()

    with open(filepath, "rb") as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()


def get_cache_size_mb(cache_dir: str) -> float:
    """
    Calculate total size of a cache directory in MB.

    Args:
        cache_dir: Path to cache directory.

    Returns:
        Total size in MB.
    """
    total_bytes = 0
    for dirpath, _, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_bytes += os.path.getsize(filepath)
            except OSError:
                pass
    return total_bytes / (1024 * 1024)


def clear_cache(cache_dir: str) -> None:
    """
    Remove all files in a cache directory.

    Args:
        cache_dir: Path to cache directory to clear.
    """
    import shutil

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        logger.info(
            f"Cache cleared: {cache_dir}",
            extra={"component": "cache_manager"},
        )
