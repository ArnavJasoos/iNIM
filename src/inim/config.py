"""
Environment variable configuration for iNIM orchestrator.

Centralizes all environment variable reads with defaults, validation,
and path resolution. Maps directly to the environment variable reference
in the iNIM architecture document (Appendix).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class INIMConfig:
    """
    Resolved iNIM configuration from environment variables.

    All environment variables are read once at startup and validated.
    Paths are resolved to absolute paths.
    """

    # --- Server ---
    server_port: int = 8000
    health_port: Optional[int] = None

    # --- Model ---
    model: str = ""                            # HF repo ID, local path, or manifest ref
    model_profile: Optional[str] = None        # Override profile selection
    served_model_name: Optional[str] = None    # Model name in /v1/models response
    manifest_path: str = ""                    # Path to manifest YAML

    # --- Paths ---
    cache_path: str = "/opt/inim/.cache"
    models_dir: str = ""                       # Computed: cache_path/models

    # --- Operational ---
    log_level: str = "info"
    offline: bool = False
    ready_timeout: int = 300                   # Seconds to wait for OVMS ready

    # --- TLS ---
    tls_cert_path: Optional[str] = None
    tls_key_path: Optional[str] = None

    # --- CORS ---
    allowed_origins: str = "*"

    # --- HuggingFace ---
    hf_token: Optional[str] = None
    hf_endpoint: str = "https://huggingface.co"

    # --- OVMS ---
    ovms_log_level: str = "INFO"
    ovms_port: int = 8001
    ovms_grpc_port: int = 8002

    # --- Internal ---
    config_dir: str = "/opt/inim/etc"
    sentinel_path: str = "/tmp/inim_ready_to_start"
    ovms_config_path: str = "/tmp/ovms_config.json"

    @classmethod
    def from_env(cls) -> "INIMConfig":
        """
        Load configuration from environment variables.

        Environment variable names follow the pattern INIM_<NAME> for
        iNIM-specific settings, HF_<NAME> for HuggingFace settings,
        and OVMS_<NAME> for OVMS pass-through settings.
        """
        config = cls()

        # --- Server ---
        config.server_port = int(os.environ.get("INIM_SERVER_PORT", "8000"))
        health_port = os.environ.get("INIM_HEALTH_PORT")
        config.health_port = int(health_port) if health_port else None

        # --- Model ---
        config.model = os.environ.get("INIM_MODEL", "")
        config.model_profile = os.environ.get("INIM_MODEL_PROFILE")
        config.served_model_name = os.environ.get("INIM_SERVED_MODEL_NAME")
        config.manifest_path = os.environ.get(
            "INIM_MANIFEST_PATH",
            os.path.join(config.config_dir, "manifests"),
        )

        # --- Paths ---
        config.cache_path = os.environ.get("INIM_CACHE_PATH", "/opt/inim/.cache")
        config.models_dir = os.path.join(config.cache_path, "models")

        # --- Operational ---
        config.log_level = os.environ.get("INIM_LOG_LEVEL", "info").lower()
        config.offline = os.environ.get("INIM_OFFLINE", "0") == "1"
        config.ready_timeout = int(os.environ.get("INIM_READY_TIMEOUT", "300"))

        # --- TLS ---
        config.tls_cert_path = os.environ.get("INIM_TLS_CERT_PATH")
        config.tls_key_path = os.environ.get("INIM_TLS_KEY_PATH")

        # --- CORS ---
        config.allowed_origins = os.environ.get("INIM_ALLOWED_ORIGINS", "*")

        # --- HuggingFace ---
        config.hf_token = _resolve_hf_token()
        config.hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")

        # --- OVMS ---
        config.ovms_log_level = os.environ.get("OVMS_LOG_LEVEL", "INFO")
        config.ovms_port = int(os.environ.get("INIM_OVMS_PORT", "8001"))
        config.ovms_grpc_port = int(os.environ.get("INIM_OVMS_GRPC_PORT", "8002"))

        return config

    @property
    def tls_enabled(self) -> bool:
        """Check if TLS is configured."""
        return bool(self.tls_cert_path and self.tls_key_path)

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        os.makedirs(self.cache_path, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

    def get_model_cache_dir(self, model_id: str) -> str:
        """
        Get the cache directory for a specific model.

        The cache key is derived from the model identifier, with
        slashes replaced by double-dashes to create a flat directory structure.

        Example:
            "OpenVINO/Llama-3.2-3B-Instruct-int4-ov"
            → "/opt/inim/.cache/models/OpenVINO--Llama-3.2-3B-Instruct-int4-ov"
        """
        safe_name = model_id.replace("/", "--")
        return os.path.join(self.models_dir, safe_name)

    def validate(self) -> list[str]:
        """
        Validate configuration and return a list of issues.

        Returns an empty list if configuration is valid.
        """
        issues = []

        if not self.model and not self.manifest_path:
            issues.append(
                "Either INIM_MODEL or a manifest file must be specified. "
                "Set INIM_MODEL to a HuggingFace repo ID or local path."
            )

        if self.tls_cert_path and not self.tls_key_path:
            issues.append("INIM_TLS_CERT_PATH is set but INIM_TLS_KEY_PATH is missing.")

        if self.tls_key_path and not self.tls_cert_path:
            issues.append("INIM_TLS_KEY_PATH is set but INIM_TLS_CERT_PATH is missing.")

        if self.ready_timeout < 30:
            issues.append(
                f"INIM_READY_TIMEOUT={self.ready_timeout} is too low. "
                "Model loading typically requires at least 30 seconds."
            )

        if self.log_level not in ("debug", "info", "warn", "warning", "error"):
            issues.append(f"INIM_LOG_LEVEL='{self.log_level}' is not a valid log level.")

        return issues


def _resolve_hf_token() -> Optional[str]:
    """
    Resolve the HuggingFace authentication token.

    Checks in order:
    1. HF_TOKEN environment variable
    2. HUGGING_FACE_HUB_TOKEN environment variable (legacy)
    3. ~/.cache/huggingface/token file (huggingface-cli login)
    """
    # Priority 1: explicit env var
    token = os.environ.get("HF_TOKEN")
    if token:
        return token

    # Priority 2: legacy env var
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token

    # Priority 3: cached token file from `huggingface-cli login`
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        try:
            return token_path.read_text().strip()
        except OSError:
            pass

    return None
