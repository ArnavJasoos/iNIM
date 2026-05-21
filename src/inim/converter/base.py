"""
Abstract base class for model converters.

All conversion paths (IR pass-through, safetensors→IR, GGUF→IR)
implement this interface, enabling the orchestrator to treat
all conversion paths uniformly.
"""

from __future__ import annotations

import abc
from typing import Optional

from inim.models import Profile, SourceFormat


class ModelConverter(abc.ABC):
    """
    Abstract base for model format converters.

    Each converter handles one source format and produces OpenVINO IR
    files in the target directory.
    """

    @property
    @abc.abstractmethod
    def source_format(self) -> SourceFormat:
        """The source format this converter handles."""
        ...

    @abc.abstractmethod
    def convert(
        self,
        source_path: str,
        target_dir: str,
        profile: Profile,
        hf_token: Optional[str] = None,
    ) -> str:
        """
        Convert a model from source format to OpenVINO IR.

        Args:
            source_path: Path to source model (local dir, file, or HF repo ID).
            target_dir: Directory to write the converted IR files.
            profile: Selected profile (determines precision, quantization params).
            hf_token: Optional HuggingFace token for gated model access.

        Returns:
            Path to the directory containing the converted IR files.

        Raises:
            ModelConversionError: If conversion fails.
        """
        ...

    @abc.abstractmethod
    def validate_output(self, output_dir: str) -> bool:
        """
        Validate that the conversion output is complete.

        Checks for required files: openvino_model.xml, openvino_model.bin,
        tokenizer files, config.json, etc.

        Args:
            output_dir: Directory containing conversion output.

        Returns:
            True if output is valid and complete.
        """
        ...


# ---------------------------------------------------------------------------
# Required output files for OVMS LLM serving
# ---------------------------------------------------------------------------

REQUIRED_IR_FILES = [
    "openvino_model.xml",
    "openvino_model.bin",
]

REQUIRED_TOKENIZER_FILES = [
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
    "openvino_detokenizer.xml",
    "openvino_detokenizer.bin",
]

REQUIRED_CONFIG_FILES = [
    "tokenizer_config.json",
]

# Files that should be present but are not strictly required
OPTIONAL_FILES = [
    "config.json",
    "special_tokens_map.json",
    "generation_config.json",
]


def validate_ir_output(output_dir: str) -> tuple[bool, list[str]]:
    """
    Validate that an output directory contains all required IR files.

    Returns:
        Tuple of (is_valid, list_of_missing_files).
    """
    import os

    missing = []

    for filename in REQUIRED_IR_FILES:
        if not os.path.exists(os.path.join(output_dir, filename)):
            missing.append(filename)

    for filename in REQUIRED_TOKENIZER_FILES:
        if not os.path.exists(os.path.join(output_dir, filename)):
            missing.append(filename)

    for filename in REQUIRED_CONFIG_FILES:
        if not os.path.exists(os.path.join(output_dir, filename)):
            missing.append(filename)

    return len(missing) == 0, missing


def get_converter(source_format: SourceFormat) -> ModelConverter:
    """
    Factory function to get the appropriate converter for a source format.

    Args:
        source_format: Detected model source format.

    Returns:
        ModelConverter instance for the given format.

    Raises:
        ValueError: If no converter exists for the format.
    """
    from inim.converter.ir_passthrough import IRPassthroughConverter
    from inim.converter.safetensors_converter import SafetensorsConverter
    from inim.converter.gguf_converter import GGUFConverter

    converters = {
        SourceFormat.OPENVINO_IR: IRPassthroughConverter,
        SourceFormat.PYTORCH_SAFETENSORS: SafetensorsConverter,
        SourceFormat.GGUF: GGUFConverter,
    }

    converter_class = converters.get(source_format)
    if converter_class is None:
        raise ValueError(
            f"No converter available for format: {source_format.value}"
        )

    return converter_class()
