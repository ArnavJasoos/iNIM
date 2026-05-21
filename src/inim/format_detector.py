"""
Model source format detection for iNIM orchestrator.

Automatically detects whether a model source (local directory or remote
HuggingFace repository) contains OpenVINO IR, safetensors/PyTorch weights,
or GGUF files. This determines which conversion path to execute.

Detection priority (applied in order):
  1. Directory contains openvino_model.xml  →  OpenVINO IR (skip conversion)
  2. Directory contains *.gguf              →  GGUF path
  3. Directory contains *.safetensors or pytorch_model*.bin + config.json
                                            →  safetensors / PyTorch path
"""

from __future__ import annotations

import os
from typing import Optional

from inim.exceptions import UnknownFormatError
from inim.logging_config import get_logger
from inim.models import SourceFormat

logger = get_logger(__name__)


def detect_source_format(
    model_path: str,
    hf_token: Optional[str] = None,
) -> SourceFormat:
    """
    Detect the source format of a model at a local path or remote HF repo.

    For local directories, directly inspects filenames.
    For remote HuggingFace repos, uses the Hub API to list repository files.

    Args:
        model_path: Local directory path or HuggingFace repo ID.
        hf_token: Optional HuggingFace token for private/gated repos.

    Returns:
        Detected SourceFormat enum value.

    Raises:
        UnknownFormatError: If the format cannot be determined.
    """
    if os.path.isdir(model_path):
        return _detect_local_format(model_path)

    # Check if it's a local file (single GGUF file path)
    if os.path.isfile(model_path):
        return _detect_single_file_format(model_path)

    # Assume it's a remote HuggingFace repo ID
    return _detect_remote_format(model_path, hf_token)


def _detect_local_format(directory: str) -> SourceFormat:
    """
    Detect model format from a local directory.

    Walks the directory (non-recursively) and checks for format-specific
    file patterns in priority order.
    """
    try:
        files = os.listdir(directory)
    except OSError as e:
        raise UnknownFormatError(
            f"Cannot list directory {directory}: {e}"
        )

    return _classify_file_list(files, directory)


def _detect_single_file_format(file_path: str) -> SourceFormat:
    """Detect format from a single file path."""
    filename = os.path.basename(file_path).lower()

    if filename.endswith(".gguf"):
        logger.info(
            f"Single GGUF file detected: {file_path}",
            extra={"component": "format_detector"},
        )
        return SourceFormat.GGUF

    if filename.endswith(".xml"):
        logger.info(
            f"OpenVINO IR XML detected: {file_path}",
            extra={"component": "format_detector"},
        )
        return SourceFormat.OPENVINO_IR

    if filename.endswith(".safetensors"):
        logger.info(
            f"Safetensors file detected: {file_path}",
            extra={"component": "format_detector"},
        )
        return SourceFormat.PYTORCH_SAFETENSORS

    raise UnknownFormatError(
        f"Cannot determine format of single file: {file_path}",
        details=f"Recognized extensions: .gguf, .xml, .safetensors",
    )


def _detect_remote_format(
    repo_id: str,
    hf_token: Optional[str] = None,
) -> SourceFormat:
    """
    Detect model format from a remote HuggingFace repository.

    Uses huggingface_hub.list_repo_files() to inspect the repo's file
    listing without downloading anything. Works with any HF org/publisher,
    not just Intel's OpenVINO/ org.
    """
    try:
        from huggingface_hub import list_repo_files
    except ImportError:
        raise UnknownFormatError(
            "huggingface_hub package not installed. Cannot detect remote repo format.",
            details="Install with: pip install huggingface_hub",
        )

    logger.info(
        f"Probing remote HuggingFace repo: {repo_id}",
        extra={"component": "format_detector"},
    )

    try:
        repo_files = list(list_repo_files(repo_id, token=hf_token))
    except Exception as e:
        raise UnknownFormatError(
            f"Cannot list files in HuggingFace repo '{repo_id}': {e}",
            details=(
                "Check that the repo exists and HF_TOKEN is set for gated models."
            ),
        )

    logger.debug(
        f"Remote repo {repo_id} contains {len(repo_files)} files",
        extra={"component": "format_detector"},
    )

    return _classify_file_list(repo_files, repo_id)


def _classify_file_list(files: list[str], source: str) -> SourceFormat:
    """
    Classify a list of filenames into a SourceFormat.

    Priority:
      1. OpenVINO IR (.xml files matching openvino_model pattern)
      2. GGUF (.gguf files)
      3. Safetensors / PyTorch (.safetensors or pytorch_model*.bin)
    """
    # Normalize to basenames for nested paths (from HF repo listings)
    basenames = [os.path.basename(f) for f in files]
    lower_files = [f.lower() for f in files]

    # Priority 1: OpenVINO IR
    if any(f.endswith(".xml") for f in basenames):
        # Verify it's an OpenVINO model XML (not just any XML)
        has_ov_xml = any(
            "openvino" in f.lower() and f.endswith(".xml")
            for f in basenames
        )
        # Also accept if there's a matching .bin alongside the .xml
        has_xml_bin_pair = (
            any(f.endswith(".xml") for f in basenames)
            and any(f.endswith(".bin") for f in basenames)
        )
        if has_ov_xml or has_xml_bin_pair:
            logger.info(
                f"Format detected: OpenVINO IR (source: {source})",
                extra={"component": "format_detector"},
            )
            return SourceFormat.OPENVINO_IR

    # Priority 2: GGUF
    if any(f.endswith(".gguf") for f in lower_files):
        logger.info(
            f"Format detected: GGUF (source: {source})",
            extra={"component": "format_detector"},
        )
        return SourceFormat.GGUF

    # Priority 3: Safetensors or PyTorch weights
    has_safetensors = any(f.endswith(".safetensors") for f in lower_files)
    has_pytorch = any(
        "pytorch_model" in f and f.endswith(".bin")
        for f in lower_files
    )
    has_config = any(f == "config.json" for f in basenames)

    if has_safetensors or (has_pytorch and has_config):
        logger.info(
            f"Format detected: safetensors/PyTorch (source: {source})",
            extra={"component": "format_detector"},
        )
        return SourceFormat.PYTORCH_SAFETENSORS

    # Could not determine format
    raise UnknownFormatError(
        f"Cannot determine model format from source: {source}",
        details=(
            f"Inspected {len(files)} files. No recognized format patterns found. "
            f"Expected: openvino_model.xml (IR), *.gguf (GGUF), "
            f"*.safetensors or pytorch_model*.bin + config.json (PyTorch)."
        ),
    )


def get_gguf_files(files: list[str]) -> list[str]:
    """Return all GGUF files from a file list."""
    return [f for f in files if f.lower().endswith(".gguf")]


def get_ir_files(files: list[str]) -> list[str]:
    """Return all OpenVINO IR files (.xml + .bin) from a file list."""
    return [
        f for f in files
        if f.endswith(".xml") or f.endswith(".bin")
    ]
