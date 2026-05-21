"""
OVMS health monitoring for iNIM orchestrator.

Polls the OVMS health endpoint until the model is loaded and ready
to serve requests, with exponential backoff and configurable timeout.
"""

from __future__ import annotations

import time
from typing import Optional

import requests

from inim.exceptions import OVMSStartupError
from inim.logging_config import get_logger

logger = get_logger(__name__)

# Default polling parameters
DEFAULT_INITIAL_INTERVAL = 2.0    # seconds
DEFAULT_MAX_INTERVAL = 15.0       # seconds
DEFAULT_BACKOFF_FACTOR = 1.5


def wait_for_ovms_ready(
    host: str = "127.0.0.1",
    port: int = 8001,
    timeout: int = 300,
    initial_interval: float = DEFAULT_INITIAL_INTERVAL,
    max_interval: float = DEFAULT_MAX_INTERVAL,
) -> None:
    """
    Poll OVMS health endpoint until ready or timeout.

    Polls GET http://{host}:{port}/v3/health/ready with exponential
    backoff. Logs progress at each attempt.

    Args:
        host: OVMS host (default: loopback).
        port: OVMS REST port.
        timeout: Maximum seconds to wait.
        initial_interval: Initial polling interval in seconds.
        max_interval: Maximum polling interval in seconds.

    Raises:
        OVMSStartupError: If OVMS doesn't become ready within timeout.
    """
    health_url = f"http://{host}:{port}/v3/health/ready"
    start_time = time.monotonic()
    interval = initial_interval
    attempt = 0

    logger.info(
        f"Waiting for OVMS to become ready at {health_url} "
        f"(timeout: {timeout}s)",
        extra={"component": "health_monitor"},
    )

    while True:
        attempt += 1
        elapsed = time.monotonic() - start_time

        if elapsed >= timeout:
            raise OVMSStartupError(
                f"OVMS did not become ready within {timeout} seconds "
                f"after {attempt} attempts.",
                details=(
                    f"Health endpoint: {health_url}. "
                    "Check OVMS logs for model loading errors. "
                    "Common causes: insufficient GPU memory, "
                    "corrupted model files, unsupported model architecture."
                ),
            )

        try:
            response = requests.get(health_url, timeout=5)

            if response.status_code == 200:
                logger.info(
                    f"OVMS is ready! (attempt {attempt}, "
                    f"elapsed: {elapsed:.1f}s)",
                    extra={"component": "health_monitor"},
                )
                return

            logger.debug(
                f"OVMS not ready (attempt {attempt}, "
                f"status: {response.status_code}, "
                f"elapsed: {elapsed:.1f}s)",
                extra={"component": "health_monitor"},
            )

        except requests.ConnectionError:
            logger.debug(
                f"OVMS not reachable (attempt {attempt}, "
                f"elapsed: {elapsed:.1f}s) — "
                "server may still be starting",
                extra={"component": "health_monitor"},
            )
        except requests.Timeout:
            logger.debug(
                f"OVMS health check timed out (attempt {attempt})",
                extra={"component": "health_monitor"},
            )
        except Exception as e:
            logger.debug(
                f"OVMS health check error (attempt {attempt}): {e}",
                extra={"component": "health_monitor"},
            )

        # Log progress every 10 attempts
        if attempt % 10 == 0:
            remaining = timeout - elapsed
            logger.info(
                f"Still waiting for OVMS... "
                f"(attempt {attempt}, elapsed: {elapsed:.0f}s, "
                f"remaining: {remaining:.0f}s)",
                extra={"component": "health_monitor"},
            )

        # Sleep with exponential backoff
        time.sleep(interval)
        interval = min(interval * DEFAULT_BACKOFF_FACTOR, max_interval)


def check_ovms_health(
    host: str = "127.0.0.1",
    port: int = 8001,
) -> dict:
    """
    Check OVMS health status (non-blocking).

    Returns a dictionary with health status information.

    Args:
        host: OVMS host.
        port: OVMS REST port.

    Returns:
        Dict with 'status' ('ready', 'not_ready', 'unreachable')
        and optional 'details'.
    """
    health_url = f"http://{host}:{port}/v3/health/ready"

    try:
        response = requests.get(health_url, timeout=5)

        if response.status_code == 200:
            return {"status": "ready", "details": "Model loaded and serving"}
        else:
            return {
                "status": "not_ready",
                "details": f"HTTP {response.status_code}: {response.text[:200]}",
            }

    except requests.ConnectionError:
        return {"status": "unreachable", "details": "Connection refused"}
    except requests.Timeout:
        return {"status": "unreachable", "details": "Connection timed out"}
    except Exception as e:
        return {"status": "unreachable", "details": str(e)}
