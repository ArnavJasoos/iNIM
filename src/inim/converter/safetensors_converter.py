"""
Path A: Safetensors / PyTorch weights → OpenVINO IR converter.

Uses optimum-cli to convert models from HuggingFace native format
(safetensors or PyTorch weights) to OpenVINO IR with the precision
specified by the selected profile.

This is the primary conversion path for models in their original
training format, from any HuggingFace repo or local directory.
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Optional

from inim.converter.base import ModelConverter, validate_ir_output
from inim.exceptions import ModelConversionError
from inim.logging_config import get_logger
from inim.models import Precision, Profile, SourceFormat

logger = get_logger(__name__)


class SafetensorsConverter(ModelConverter):
    """
    Converts safetensors/PyTorch models to OpenVINO IR via optimum-cli.

    Supports all precision levels:
      - fp16: highest accuracy, highest VRAM
      - int8: balanced quality/performance
      - int4: best perf on Arc with XMX (group-size 128)
      - int4-sym: symmetric quantization for older Intel GPUs
    """

    @property
    def source_format(self) -> SourceFormat:
        return SourceFormat.PYTORCH_SAFETENSORS

    def convert(
        self,
        source_path: str,
        target_dir: str,
        profile: Profile,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Convert a safetensors/PyTorch model to OpenVINO IR.

        Shells out to `optimum-cli export openvino` with precision flags
        derived from the profile. The --model argument accepts both
        HuggingFace repo IDs and local directory paths.

        Args:
            source_path: HF repo ID or local directory with model weights.
            target_dir: Directory to write converted IR files.
            profile: Profile determining precision and quantization params.
            hf_token: Optional HF token for gated model access.

        Returns:
            Path to converted IR directory.

        Raises:
            ModelConversionError: If optimum-cli fails.
        """
        logger.info(
            f"Converting safetensors/PyTorch → OpenVINO IR: "
            f"source={source_path}, precision={profile.precision}",
            extra={"component": "converter.safetensors"},
        )

        # Ensure optimum-intel is available
        self._check_optimum_installed()

        # Build optimum-cli command
        cmd = self._build_command(source_path, target_dir, profile)

        # Set up environment with HF token if needed
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token

        logger.info(
            f"Running: {' '.join(cmd)}",
            extra={"component": "converter.safetensors"},
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=env,
                timeout=7200,  # 2 hour timeout for large models
            )
        except subprocess.TimeoutExpired:
            raise ModelConversionError(
                f"Model conversion timed out after 2 hours for {source_path}",
                details="Consider converting offline and mounting the IR cache.",
            )
        except FileNotFoundError:
            raise ModelConversionError(
                "optimum-cli not found. Install with: "
                "pip install 'optimum-intel[openvino]' nncf --extra-index-url https://download.pytorch.org/whl/cpu",
            )

        if result.returncode != 0:
            # Log both stdout and stderr for debugging
            logger.error(
                f"optimum-cli failed (exit code {result.returncode})",
                extra={"component": "converter.safetensors"},
            )
            if result.stdout:
                logger.error(
                    f"stdout: {result.stdout[-2000:]}",
                    extra={"component": "converter.safetensors"},
                )
            if result.stderr:
                logger.error(
                    f"stderr: {result.stderr[-2000:]}",
                    extra={"component": "converter.safetensors"},
                )

            raise ModelConversionError(
                f"optimum-cli export failed for {source_path} "
                f"(exit code {result.returncode})",
                details=result.stderr[-2000:] if result.stderr else None,
            )

        # Log success output
        if result.stdout:
            # Show last few lines of output
            last_lines = result.stdout.strip().split("\n")[-5:]
            for line in last_lines:
                logger.info(
                    f"optimum-cli: {line}",
                    extra={"component": "converter.safetensors"},
                )

        # Validate output
        is_valid, missing = validate_ir_output(target_dir)
        if not is_valid:
            # Some models may not produce all tokenizer files
            # Check if at least the model IR is present
            model_xml = os.path.join(target_dir, "openvino_model.xml")
            if not os.path.exists(model_xml):
                raise ModelConversionError(
                    f"Conversion completed but openvino_model.xml not found "
                    f"in {target_dir}",
                    details=f"Missing files: {missing}",
                )
            logger.warning(
                f"Conversion complete but some files missing: {missing}",
                extra={"component": "converter.safetensors"},
            )

        logger.info(
            f"Conversion complete: {target_dir}",
            extra={"component": "converter.safetensors"},
        )
        return target_dir

    def validate_output(self, output_dir: str) -> bool:
        is_valid, _ = validate_ir_output(output_dir)
        return is_valid

    def _build_command(
        self,
        source_path: str,
        target_dir: str,
        profile: Profile,
    ) -> list[str]:
        """
        Build the optimum-cli export command with precision flags.

        Maps profile precision to optimum-cli arguments:
          - --weight-format: int4, int8, fp16
          - --group-size: 128 (for int4), -1 (for int4-sym), omitted otherwise
          - --ratio: 1.0 (compress all layers)
        """
        cmd = [
            sys.executable, "-m", "optimum.exporters.openvino",
            "--model", source_path,
            "--task", "text-generation-with-past",
        ]

        # Precision-specific flags
        try:
            precision = Precision(profile.precision)
        except ValueError:
            # Default to int4 if precision string not recognized
            precision = Precision.INT4

        cmd.extend(["--weight-format", precision.optimum_weight_format])

        group_size = precision.optimum_group_size
        if group_size is not None:
            cmd.extend(["--group-size", str(group_size)])

        # Apply ratio 1.0 for full quantization coverage
        if precision in (Precision.INT4, Precision.INT4_SYM):
            cmd.extend(["--ratio", "1.0"])

        # Output directory
        cmd.append(target_dir)

        return cmd

    def _check_optimum_installed(self) -> None:
        """Verify that optimum-intel with OpenVINO support is installed."""
        try:
            import optimum.exporters.openvino  # noqa: F401
        except ImportError:
            raise ModelConversionError(
                "optimum-intel[openvino] is not installed. "
                "Required for safetensors/PyTorch → OpenVINO IR conversion.",
                details=(
                    "Install with: pip install 'optimum-intel[openvino]' nncf --extra-index-url https://download.pytorch.org/whl/cpu\n"
                    "Or use a pre-converted OpenVINO IR model to skip conversion."
                ),
            )
