"""
Path B: GGUF → OpenVINO IR converter.

Handles GGUF model files via two sub-paths:
  B1: Native OpenVINO GenAI GGUF reader (fast, limited architecture support)
  B2: Offline dequantization to FP16 → optimum-cli export (universal fallback)

The converter attempts B1 first and falls back to B2 on failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Optional

from inim.converter.base import ModelConverter, validate_ir_output
from inim.exceptions import ModelConversionError
from inim.logging_config import get_logger
from inim.models import (
    GGUF_QUANTIZATION_PREFERENCE,
    Precision,
    Profile,
    SourceFormat,
    select_gguf_file,
)

logger = get_logger(__name__)

# Architectures known to be supported by OV GenAI native GGUF reader
_NATIVE_GGUF_ARCHITECTURES = [
    "llama", "llama2", "llama3",
    "qwen", "qwen2", "qwen2.5", "qwen3",
    "smollm",
    "phi", "phi3",
    "mistral",
    "gemma", "gemma2",
]

# GGUF quantization types supported by native reader
_NATIVE_GGUF_QUANT_TYPES = [
    "q4_0", "q4_k_m", "q8_0", "f16", "bf16",
]


class GGUFConverter(ModelConverter):
    """
    Converts GGUF models to OpenVINO IR.

    Sub-path B1 (native):
      Uses openvino_genai.LLMPipeline to directly load .gguf files and
      serialize to IR via save_pretrained(). Fast but limited to supported
      architectures and quantization types.

    Sub-path B2 (fallback):
      Dequantizes GGUF to FP16 HuggingFace format via gguf_to_hf.py,
      then converts to IR via optimum-cli. Universal but slower.
    """

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.GGUF

    def convert(
        self,
        source_path: str,
        target_dir: str,
        profile: Profile,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Convert a GGUF model to OpenVINO IR.

        Attempts native GGUF reader (B1) first, falls back to
        dequantize-then-convert (B2) on failure.

        Args:
            source_path: Path to .gguf file or directory containing .gguf files,
                        or HF repo ID containing .gguf files.
            target_dir: Directory to write converted IR files.
            profile: Selected profile for precision targeting.
            hf_token: Optional HF token for downloading from private repos.

        Returns:
            Path to converted IR directory.
        """
        logger.info(
            f"GGUF conversion: source={source_path}, precision={profile.precision}",
            extra={"component": "converter.gguf"},
        )

        # Resolve source to a local .gguf file path
        gguf_path = self._resolve_gguf_file(source_path, profile, hf_token)

        logger.info(
            f"Resolved GGUF file: {gguf_path}",
            extra={"component": "converter.gguf"},
        )

        # Attempt Sub-path B1: native OV GenAI GGUF reader
        try:
            result = self._convert_native(gguf_path, target_dir)
            if result:
                logger.info(
                    "Sub-path B1 (native GGUF reader) succeeded",
                    extra={"component": "converter.gguf"},
                )
                return result
        except Exception as e:
            logger.warning(
                f"Sub-path B1 (native GGUF reader) failed: {e}. "
                "Falling back to Sub-path B2 (dequantize + optimum-cli).",
                extra={"component": "converter.gguf"},
            )

        # Fallback: Sub-path B2: dequantize → optimum-cli
        return self._convert_via_dequantize(
            gguf_path, target_dir, profile, hf_token
        )

    def validate_output(self, output_dir: str) -> bool:
        is_valid, _ = validate_ir_output(output_dir)
        return is_valid

    # -------------------------------------------------------------------
    # Sub-path B1: Native GGUF reader
    # -------------------------------------------------------------------

    def _convert_native(self, gguf_path: str, target_dir: str) -> Optional[str]:
        """
        Attempt native GGUF → IR conversion via OpenVINO GenAI.

        Uses LLMPipeline to load the GGUF file directly on CPU, then
        serializes the resulting IR to the cache directory.

        Returns target_dir on success, None if native reader is unavailable.
        """
        try:
            from openvino_genai import LLMPipeline  # type: ignore[import-untyped]
        except ImportError:
            logger.info(
                "openvino_genai not installed, skipping native GGUF path",
                extra={"component": "converter.gguf"},
            )
            return None

        logger.info(
            f"Attempting native GGUF load: {gguf_path}",
            extra={"component": "converter.gguf"},
        )

        # Load on CPU for conversion (avoids GPU memory issues during conversion)
        pipe = LLMPipeline(gguf_path, "CPU")

        # Serialize to IR
        os.makedirs(target_dir, exist_ok=True)
        pipe.save_pretrained(target_dir)

        # Validate output
        is_valid, missing = validate_ir_output(target_dir)
        if not is_valid:
            logger.warning(
                f"Native GGUF conversion produced incomplete output: {missing}",
                extra={"component": "converter.gguf"},
            )
            # Check if at minimum model XML exists
            if not os.path.exists(os.path.join(target_dir, "openvino_model.xml")):
                return None

        return target_dir

    # -------------------------------------------------------------------
    # Sub-path B2: Dequantize + optimum-cli
    # -------------------------------------------------------------------

    def _convert_via_dequantize(
        self,
        gguf_path: str,
        target_dir: str,
        profile: Profile,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Universal fallback: dequantize GGUF → FP16 → optimum-cli → IR.

        Step 1: Use gguf_to_hf.py to dequantize GGUF to FP16 safetensors
        Step 2: Use optimum-cli to convert FP16 to OpenVINO IR

        This works for all architectures and quantization types but is
        slower than the native reader.
        """
        logger.info(
            "Sub-path B2: dequantize GGUF → FP16 → optimum-cli → IR",
            extra={"component": "converter.gguf"},
        )

        # Create temp directory for intermediate FP16 weights
        # Use a directory within the cache to avoid cross-device issues
        cache_base = os.path.dirname(target_dir)
        staging_dir = tempfile.mkdtemp(prefix="inim_gguf_staging_", dir=cache_base)

        try:
            # Step 1: Dequantize GGUF → FP16 HuggingFace directory
            self._dequantize_gguf(gguf_path, staging_dir, hf_token)

            # Step 2: Convert FP16 → OpenVINO IR via optimum-cli
            self._convert_fp16_to_ir(staging_dir, target_dir, profile)

        finally:
            # Clean up staging directory
            if os.path.exists(staging_dir):
                logger.debug(
                    f"Cleaning up staging directory: {staging_dir}",
                    extra={"component": "converter.gguf"},
                )
                shutil.rmtree(staging_dir, ignore_errors=True)

        # Validate final output
        is_valid, missing = validate_ir_output(target_dir)
        if not is_valid:
            model_xml = os.path.join(target_dir, "openvino_model.xml")
            if not os.path.exists(model_xml):
                raise ModelConversionError(
                    f"GGUF conversion (B2) completed but output is invalid. "
                    f"Missing: {missing}",
                    details=f"GGUF source: {gguf_path}",
                )
            logger.warning(
                f"GGUF conversion complete but some files missing: {missing}",
                extra={"component": "converter.gguf"},
            )

        logger.info(
            f"GGUF conversion (B2) complete: {target_dir}",
            extra={"component": "converter.gguf"},
        )
        return target_dir

    def _dequantize_gguf(
        self,
        gguf_path: str,
        output_dir: str,
        hf_token: Optional[str],
    ) -> None:
        """
        Dequantize a GGUF file to FP16 HuggingFace-compatible format.

        Uses the gguf_to_hf module to extract tensors from the GGUF file
        and write them as safetensors with a config.json and tokenizer files.
        """
        logger.info(
            f"Dequantizing GGUF to FP16: {gguf_path} → {output_dir}",
            extra={"component": "converter.gguf"},
        )

        # Use our bundled gguf_to_hf module
        from inim.converter.gguf_to_hf import convert_gguf_to_hf

        try:
            convert_gguf_to_hf(
                gguf_path=gguf_path,
                output_dir=output_dir,
                hf_token=hf_token,
            )
        except Exception as e:
            raise ModelConversionError(
                f"GGUF dequantization failed: {e}",
                details=f"Source: {gguf_path}",
            )

        # Verify the output has at least config.json and some weight files
        if not os.path.exists(os.path.join(output_dir, "config.json")):
            raise ModelConversionError(
                "GGUF dequantization produced no config.json",
                details=f"Check {output_dir} for output files.",
            )

    def _convert_fp16_to_ir(
        self,
        fp16_dir: str,
        target_dir: str,
        profile: Profile,
    ) -> None:
        """
        Convert FP16 HuggingFace model to OpenVINO IR via optimum-cli.

        Delegates to the same optimum-cli command used by SafetensorsConverter.
        """
        logger.info(
            f"Converting FP16 → OpenVINO IR: {fp16_dir} → {target_dir}",
            extra={"component": "converter.gguf"},
        )

        try:
            precision = Precision(profile.precision)
        except ValueError:
            precision = Precision.INT4

        cmd = [
            sys.executable, "-m", "optimum.exporters.openvino",
            "--model", fp16_dir,
            "--task", "text-generation-with-past",
            "--weight-format", precision.optimum_weight_format,
        ]

        group_size = precision.optimum_group_size
        if group_size is not None:
            cmd.extend(["--group-size", str(group_size)])

        if precision in (Precision.INT4, Precision.INT4_SYM):
            cmd.extend(["--ratio", "1.0"])

        cmd.append(target_dir)

        logger.info(
            f"Running: {' '.join(cmd)}",
            extra={"component": "converter.gguf"},
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,
            )
        except subprocess.TimeoutExpired:
            raise ModelConversionError(
                "optimum-cli conversion timed out after 2 hours",
            )
        except FileNotFoundError:
            raise ModelConversionError(
                "optimum-cli not found. Install with: "
                "pip install 'optimum-intel[openvino]' nncf --extra-index-url https://download.pytorch.org/whl/cpu",
            )

        if result.returncode != 0:
            raise ModelConversionError(
                f"optimum-cli export failed (exit code {result.returncode})",
                details=result.stderr[-2000:] if result.stderr else None,
            )

    # -------------------------------------------------------------------
    # GGUF file resolution
    # -------------------------------------------------------------------

    def _resolve_gguf_file(
        self,
        source_path: str,
        profile: Profile,
        hf_token: Optional[str],
    ) -> str:
        """
        Resolve the source to a local .gguf file path.

        Handles:
          - Direct .gguf file path
          - Local directory containing .gguf files (selects best for precision)
          - Remote HF repo containing .gguf files (downloads selected file)
        """
        # Direct file path
        if os.path.isfile(source_path) and source_path.lower().endswith(".gguf"):
            return source_path

        # Local directory with GGUF files
        if os.path.isdir(source_path):
            files = os.listdir(source_path)
            selected = select_gguf_file(files, profile.precision)
            if selected:
                return os.path.join(source_path, selected)
            raise ModelConversionError(
                f"No .gguf files found in {source_path}",
            )

        # Remote HF repo — download the best GGUF file
        return self._download_gguf_from_hf(source_path, profile, hf_token)

    def _download_gguf_from_hf(
        self,
        repo_id: str,
        profile: Profile,
        hf_token: Optional[str],
    ) -> str:
        """Download the best GGUF file from a HuggingFace repo."""
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            raise ModelConversionError(
                "huggingface_hub not installed. Cannot download GGUF from HF.",
            )

        logger.info(
            f"Listing GGUF files in HF repo: {repo_id}",
            extra={"component": "converter.gguf"},
        )

        try:
            repo_files = list(list_repo_files(repo_id, token=hf_token))
        except Exception as e:
            raise ModelConversionError(
                f"Cannot list files in HF repo {repo_id}: {e}",
            )

        selected = select_gguf_file(repo_files, profile.precision)
        if not selected:
            raise ModelConversionError(
                f"No GGUF files found in HF repo {repo_id}",
            )

        logger.info(
            f"Selected GGUF file: {selected} (precision target: {profile.precision})",
            extra={"component": "converter.gguf"},
        )

        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=selected,
                token=hf_token,
            )
            return local_path
        except Exception as e:
            raise ModelConversionError(
                f"Failed to download {selected} from {repo_id}: {e}",
            )
