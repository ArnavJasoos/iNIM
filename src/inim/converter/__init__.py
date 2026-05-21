"""
Model conversion pipeline for iNIM.

Provides converter selection based on detected source format,
routing to the appropriate conversion path:
  - Path A: safetensors/PyTorch → OpenVINO IR (via optimum-cli)
  - Path B: GGUF → OpenVINO IR (B1: native reader, B2: dequantize fallback)
  - Path C: Already OpenVINO IR (pass-through)
"""

from inim.converter.base import ModelConverter, get_converter
from inim.converter.ir_passthrough import IRPassthroughConverter
from inim.converter.safetensors_converter import SafetensorsConverter
from inim.converter.gguf_converter import GGUFConverter

__all__ = [
    "ModelConverter",
    "get_converter",
    "IRPassthroughConverter",
    "SafetensorsConverter",
    "GGUFConverter",
]
