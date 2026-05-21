"""
Custom exception hierarchy for iNIM orchestrator.

All iNIM-specific exceptions inherit from INIMError, enabling
callers to catch all orchestrator failures with a single base class
while still handling specific failure modes individually.
"""


class INIMError(Exception):
    """Base exception for all iNIM orchestrator errors."""

    def __init__(self, message: str, details: str | None = None):
        self.details = details
        super().__init__(message)


class GPUDetectionError(INIMError):
    """
    Raised when no Intel GPU is detected or the GPU cannot be enumerated.

    Common causes:
    - Missing Intel GPU drivers (Level Zero / OpenCL)
    - /dev/dri not passed into the container
    - No Intel GPU present in the system
    """
    pass


class NoCompatibleProfileError(INIMError):
    """
    Raised when the detected GPU does not match any profile in the manifest.

    This typically means the GPU has insufficient VRAM or is an unsupported
    architecture tier for the target model.
    """
    pass


class ModelDownloadError(INIMError):
    """
    Raised when model download from HuggingFace Hub or other source fails.

    Common causes:
    - Network connectivity issues
    - Invalid or expired HF_TOKEN for gated models
    - Repository does not exist
    - INIM_OFFLINE=1 set but model not in cache
    """
    pass


class ModelConversionError(INIMError):
    """
    Raised when model format conversion fails.

    Common causes:
    - optimum-cli export failure (unsupported architecture, OOM)
    - GGUF native reader failure (unsupported quantization type)
    - GGUF dequantization failure
    - Insufficient disk space for conversion artifacts
    """
    pass


class UnknownFormatError(INIMError):
    """
    Raised when the model source format cannot be determined.

    The format detector could not find recognized file patterns
    (e.g., .xml for IR, .gguf for GGUF, .safetensors for HF format)
    in the model directory or remote repository.
    """
    pass


class CacheIntegrityError(INIMError):
    """
    Raised when cached model files fail checksum verification.

    The .inim_cache_manifest.json records SHA-256 checksums for all IR files.
    A mismatch triggers re-download unless INIM_OFFLINE=1 is set, in which
    case this exception is raised as a hard failure.
    """
    pass


class OVMSStartupError(INIMError):
    """
    Raised when OVMS fails to become ready within the configured timeout.

    The orchestrator polls OVMS at http://127.0.0.1:8001/v3/health/ready
    for up to INIM_READY_TIMEOUT seconds. If OVMS never returns 200 OK,
    this exception is raised.
    """
    pass


class ManifestError(INIMError):
    """
    Raised when the model manifest file is invalid or cannot be parsed.

    Common causes:
    - Malformed YAML syntax
    - Missing required fields (model_name, model_id, profiles)
    - Invalid profile structure
    """
    pass
