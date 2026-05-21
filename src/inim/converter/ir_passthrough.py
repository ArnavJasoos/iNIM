"""
Path C: OpenVINO IR pass-through converter.

When the model source already contains OpenVINO IR files, no conversion
is needed. This converter validates the IR completeness and copies (or
symlinks) the files to the cache directory.

This is the fastest path — used for pre-converted models from Intel's
OpenVINO/ HuggingFace org or from previous iNIM conversion runs.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from inim.converter.base import ModelConverter, validate_ir_output
from inim.exceptions import ModelConversionError
from inim.logging_config import get_logger
from inim.models import Profile, SourceFormat

logger = get_logger(__name__)


class IRPassthroughConverter(ModelConverter):
    """
    Pass-through converter for models already in OpenVINO IR format.

    Validates that all required IR files are present and copies them
    to the target cache directory. No conversion tools are invoked.
    """

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.OPENVINO_IR

    def convert(
        self,
        source_path: str,
        target_dir: str,
        profile: Profile,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Copy OpenVINO IR files from source to target directory.

        If source_path is a HuggingFace repo ID, downloads it first.
        If source_path is a local directory, copies or symlinks files.

        Args:
            source_path: Local directory or HF repo ID containing IR files.
            target_dir: Target cache directory.
            profile: Selected profile (used for logging context only).
            hf_token: Optional HF token for downloading from private repos.

        Returns:
            Path to the directory containing IR files (= target_dir).
        """
        logger.info(
            f"IR pass-through: source={source_path}, target={target_dir}",
            extra={"component": "converter.ir_passthrough"},
        )

        # If source is a remote HF repo, download it
        if not os.path.isdir(source_path):
            source_path = self._download_ir_repo(source_path, target_dir, hf_token)

        # If source == target, we're already in place
        if os.path.abspath(source_path) == os.path.abspath(target_dir):
            logger.info(
                "Source and target are the same directory, skipping copy",
                extra={"component": "converter.ir_passthrough"},
            )
        else:
            self._copy_ir_files(source_path, target_dir)

        # Validate output
        is_valid, missing = validate_ir_output(target_dir)
        if not is_valid:
            logger.warning(
                f"IR validation: missing files: {missing}. "
                "Model may still work if tokenizer files are embedded.",
                extra={"component": "converter.ir_passthrough"},
            )

        # Check at minimum that the model XML + bin are present
        model_xml = os.path.join(target_dir, "openvino_model.xml")
        model_bin = os.path.join(target_dir, "openvino_model.bin")
        if not os.path.exists(model_xml) or not os.path.exists(model_bin):
            raise ModelConversionError(
                f"IR pass-through failed: openvino_model.xml or .bin "
                f"not found in {target_dir}",
                details=f"Source was: {source_path}",
            )

        logger.info(
            f"IR pass-through complete: {target_dir}",
            extra={"component": "converter.ir_passthrough"},
        )
        return target_dir

    def validate_output(self, output_dir: str) -> bool:
        is_valid, _ = validate_ir_output(output_dir)
        return is_valid

    def _download_ir_repo(
        self,
        repo_id: str,
        target_dir: str,
        hf_token: Optional[str],
    ) -> str:
        """Download an OpenVINO IR model from HuggingFace Hub."""
        logger.info(
            f"Downloading IR model from HuggingFace: {repo_id}",
            extra={"component": "converter.ir_passthrough"},
        )

        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise ModelConversionError(
                "huggingface_hub not installed. Cannot download remote IR models.",
                details="Install with: pip install huggingface_hub",
            )

        try:
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                token=hf_token,
                # Only download model files, not readmes/licenses
                ignore_patterns=[
                    "*.md",
                    "*.txt",
                    ".gitattributes",
                    "LICENSE*",
                    "README*",
                ],
            )
            return downloaded_path
        except Exception as e:
            raise ModelConversionError(
                f"Failed to download IR model from {repo_id}: {e}",
                details="Check network connectivity and HF_TOKEN for gated models.",
            )

    def _copy_ir_files(self, source_dir: str, target_dir: str) -> None:
        """Copy all IR-related files from source to target directory."""
        os.makedirs(target_dir, exist_ok=True)

        # Copy all files (not subdirectories) from source to target
        copied = 0
        for filename in os.listdir(source_dir):
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename)

            if os.path.isfile(src):
                shutil.copy2(src, dst)
                copied += 1

        logger.info(
            f"Copied {copied} files from {source_dir} to {target_dir}",
            extra={"component": "converter.ir_passthrough"},
        )
