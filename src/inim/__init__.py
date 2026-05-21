"""
iNIM — Intel NIM-Equivalent Inference Container Orchestrator

A production-grade LLM inference container that mirrors NVIDIA NIM's design
philosophy but targets Intel GPU hardware (UHD / Xe / Arc) using OpenVINO
Model Server (OVMS) as the inference backend.
"""

__version__ = "1.0.0"
__project__ = "iNIM"

from inim.models import (
    ArchTier,
    GPUInfo,
    SourceFormat,
    Profile,
    OVMSConfig,
    CacheManifest,
)
from inim.exceptions import (
    INIMError,
    GPUDetectionError,
    NoCompatibleProfileError,
    ModelDownloadError,
    ModelConversionError,
    UnknownFormatError,
    CacheIntegrityError,
    OVMSStartupError,
)

__all__ = [
    "__version__",
    "__project__",
    # Models
    "ArchTier",
    "GPUInfo",
    "SourceFormat",
    "Profile",
    "OVMSConfig",
    "CacheManifest",
    # Exceptions
    "INIMError",
    "GPUDetectionError",
    "NoCompatibleProfileError",
    "ModelDownloadError",
    "ModelConversionError",
    "UnknownFormatError",
    "CacheIntegrityError",
    "OVMSStartupError",
]
