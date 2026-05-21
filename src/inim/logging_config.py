"""
Structured JSON logging configuration for iNIM orchestrator.

Produces JSON Lines output for Docker log driver compatibility,
with support for request correlation via X-Request-Id headers
and configurable log levels via INIM_LOG_LEVEL.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects (JSON Lines format).

    Output format:
    {"timestamp": "...", "level": "INFO", "logger": "inim.main", "message": "...", ...}

    This is compatible with Docker's default json-file log driver and
    enables structured log aggregation in Kubernetes/ELK/Splunk pipelines.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields (request_id, traceparent, etc.)
        for key in ("request_id", "traceparent", "component", "model",
                     "profile", "gpu", "duration_ms", "phase"):
            value = getattr(record, key, None)
            if value is not None:
                log_entry[key] = value

        return json.dumps(log_entry, default=str)


def setup_logging(level: str = "info", component: str = "orchestrator") -> logging.Logger:
    """
    Configure structured JSON logging for the iNIM orchestrator.

    Args:
        level: Log level string (debug, info, warn, error).
        component: Component identifier added to every log line.

    Returns:
        Configured root logger for iNIM.
    """
    # Map level strings
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warn": logging.WARNING,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }
    log_level = level_map.get(level.lower(), logging.INFO)

    # Create iNIM root logger
    logger = logging.getLogger("inim")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates on re-init
    logger.handlers.clear()

    # JSON handler to stdout
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

    # Prevent propagation to root logger (avoids duplicate output)
    logger.propagate = False

    # Log initial message
    logger.info(
        "iNIM logging initialized",
        extra={"component": component, "level_configured": level},
    )

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a child logger under the iNIM namespace.

    Usage:
        from inim.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("GPU detected", extra={"gpu": "Arc A770"})

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        Logger instance under the 'inim' namespace.
    """
    # Ensure name is under inim namespace
    if not name.startswith("inim"):
        name = f"inim.{name}"
    return logging.getLogger(name)


class PhaseTimer:
    """
    Context manager for timing orchestrator phases with structured logging.

    Usage:
        with PhaseTimer(logger, "model_download"):
            download_model(...)
        # Logs: {"phase": "model_download", "duration_ms": 1234, ...}
    """

    def __init__(self, logger: logging.Logger, phase: str, **extra: Any):
        self.logger = logger
        self.phase = phase
        self.extra = extra
        self._start: float = 0.0

    def __enter__(self) -> "PhaseTimer":
        self._start = time.monotonic()
        self.logger.info(
            f"Starting phase: {self.phase}",
            extra={"phase": self.phase, "component": "orchestrator", **self.extra},
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        duration_ms = int((time.monotonic() - self._start) * 1000)
        if exc_type is not None:
            self.logger.error(
                f"Phase {self.phase} failed after {duration_ms}ms: {exc_val}",
                extra={
                    "phase": self.phase,
                    "duration_ms": duration_ms,
                    "component": "orchestrator",
                    **self.extra,
                },
                exc_info=True,
            )
        else:
            self.logger.info(
                f"Phase {self.phase} completed in {duration_ms}ms",
                extra={
                    "phase": self.phase,
                    "duration_ms": duration_ms,
                    "component": "orchestrator",
                    **self.extra,
                },
            )
