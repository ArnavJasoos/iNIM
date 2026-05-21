"""
Intel GPU detection module for iNIM orchestrator.

Detects Intel GPU hardware using OpenVINO Runtime's device enumeration API,
classifies devices into architecture tiers (UHD/Xe/Arc), and reports VRAM.
This is the Intel equivalent of NVIDIA's GPU detection in NIM.

Fallback detection via /sys/class/drm/ is provided for environments where
OpenVINO is not installed (e.g., host setup scripts).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from typing import Optional

from inim.exceptions import GPUDetectionError
from inim.logging_config import get_logger
from inim.models import ArchTier, GPUInfo

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Known device name → VRAM fallback mappings
# When OpenVINO can't report VRAM (e.g., shared memory iGPU), we use
# conservative estimates based on the SKU.
# ---------------------------------------------------------------------------
_KNOWN_VRAM_FALLBACKS: dict[str, int] = {
    # Arc Alchemist (discrete)
    "a770": 16384,
    "a750": 8192,
    "a580": 8192,
    "a380": 6144,
    "a310": 4096,
    # Arc Battlemage (discrete)
    "b770": 16384,
    "b580": 12288,
    "b570": 10240,
    # Integrated GPUs — shared memory, use conservative estimates
    # Actual available depends on system RAM and BIOS allocation
}

# Shared-memory iGPU default VRAM estimate (MB)
# For iGPUs, actual GPU-accessible memory depends on system RAM.
# We estimate ~25% of system RAM, capped at 8GB, minimum 2GB.
_IGPU_MIN_VRAM_MB = 2048
_IGPU_MAX_VRAM_MB = 8192


def detect_gpus() -> list[GPUInfo]:
    """
    Detect all Intel GPU devices available for inference.

    Uses OpenVINO Runtime Core to enumerate GPU devices and extract
    device name and memory properties. Falls back to /sys/class/drm/
    parsing on Linux if OpenVINO is unavailable.

    Returns:
        List of GPUInfo objects for all detected Intel GPUs.

    Raises:
        GPUDetectionError: If no Intel GPU is found.
    """
    gpus: list[GPUInfo] = []

    try:
        gpus = _detect_via_openvino()
    except Exception as e:
        logger.warning(
            f"OpenVINO GPU detection failed: {e}. Trying fallback methods.",
            extra={"component": "gpu_detector"},
        )
        try:
            gpus = _detect_via_sysfs()
        except Exception as e2:
            logger.warning(
                f"sysfs GPU detection also failed: {e2}",
                extra={"component": "gpu_detector"},
            )

    if not gpus:
        raise GPUDetectionError(
            "No Intel GPU detected. Ensure Intel GPU drivers are installed "
            "and /dev/dri is passed into the container (--device /dev/dri).",
            details=(
                "Checked: OpenVINO Core.available_devices, /sys/class/drm/. "
                "Required packages: intel-level-zero-gpu, level-zero, "
                "intel-opencl-icd, intel-igc-core."
            ),
        )

    for gpu in gpus:
        logger.info(
            f"Detected GPU: {gpu.name} ({gpu.device_id}) — "
            f"{gpu.vram_mb}MB VRAM — tier: {gpu.arch_tier.value} — "
            f"XMX: {gpu.xmx_supported}",
            extra={"component": "gpu_detector", "gpu": gpu.name},
        )

    return gpus


def detect_primary_gpu() -> GPUInfo:
    """
    Detect and return the primary (best) Intel GPU.

    If multiple GPUs are present, selects the one with highest
    architecture tier and VRAM — preferring discrete over integrated.

    Returns:
        GPUInfo for the primary GPU.

    Raises:
        GPUDetectionError: If no Intel GPU is found.
    """
    gpus = detect_gpus()

    # Sort: discrete > integrated, higher tier first, more VRAM first
    tier_priority = {
        ArchTier.ARC_BATTLEMAGE: 4,
        ArchTier.ARC_ALCHEMIST: 3,
        ArchTier.XE_INTEGRATED: 2,
        ArchTier.UHD_INTEGRATED: 1,
    }

    primary = max(
        gpus,
        key=lambda g: (tier_priority.get(g.arch_tier, 0), g.vram_mb),
    )

    logger.info(
        f"Selected primary GPU: {primary.name} ({primary.device_id})",
        extra={"component": "gpu_detector", "gpu": primary.name},
    )

    return primary


# ---------------------------------------------------------------------------
# Detection backends
# ---------------------------------------------------------------------------

def _detect_via_openvino() -> list[GPUInfo]:
    """
    Detect Intel GPUs via OpenVINO Runtime Core.

    This is the preferred detection method — it queries the exact same
    device abstraction that OVMS will use for inference, ensuring
    consistency between detection and runtime.
    """
    from openvino.runtime import Core  # type: ignore[import-untyped]

    core = Core()
    available = core.available_devices
    logger.debug(
        f"OpenVINO available devices: {available}",
        extra={"component": "gpu_detector"},
    )

    gpus: list[GPUInfo] = []

    for device_id in available:
        # Only process GPU devices (GPU, GPU.0, GPU.1, etc.)
        if not device_id.startswith("GPU"):
            continue

        try:
            device_name = core.get_property(device_id, "FULL_DEVICE_NAME")
        except Exception:
            device_name = f"Intel GPU ({device_id})"

        # Get VRAM — OpenVINO reports this in bytes for discrete GPUs
        vram_mb = _get_gpu_vram_ov(core, device_id, device_name)

        arch_tier = ArchTier.from_device_name(device_name)

        gpus.append(GPUInfo(
            name=device_name,
            device_id=device_id,
            vram_mb=vram_mb,
            arch_tier=arch_tier,
        ))

    return gpus


def _get_gpu_vram_ov(core, device_id: str, device_name: str) -> int:
    """
    Get GPU VRAM in MB via OpenVINO Core properties.

    For discrete GPUs, OpenVINO reports GPU_DEVICE_TOTAL_MEM_SIZE in bytes.
    For integrated GPUs with shared memory, this property may not be
    available or may report 0, in which case we fall back to known
    SKU values or system memory estimation.
    """
    try:
        vram_bytes = core.get_property(device_id, "GPU_DEVICE_TOTAL_MEM_SIZE")
        if vram_bytes and int(vram_bytes) > 0:
            return int(vram_bytes) // (1024 * 1024)
    except Exception:
        pass

    # Fallback: check known SKU table
    name_lower = device_name.lower()
    for sku, vram in _KNOWN_VRAM_FALLBACKS.items():
        if sku in name_lower:
            return vram

    # Fallback for iGPUs: estimate from system RAM
    return _estimate_igpu_vram()


def _estimate_igpu_vram() -> int:
    """
    Estimate available GPU memory for integrated GPUs based on system RAM.

    iGPUs share system memory. We conservatively estimate 25% of total
    system RAM is available for GPU use, clamped between 2GB and 8GB.
    """
    try:
        if os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        # MemTotal is in kB
                        total_kb = int(line.split()[1])
                        total_mb = total_kb // 1024
                        estimated = total_mb // 4  # 25% of system RAM
                        return max(_IGPU_MIN_VRAM_MB, min(estimated, _IGPU_MAX_VRAM_MB))
    except Exception:
        pass

    # Absolute fallback
    return _IGPU_MIN_VRAM_MB


def _detect_via_sysfs() -> list[GPUInfo]:
    """
    Fallback GPU detection via Linux sysfs.

    Parses /sys/class/drm/card*/device/ to find Intel GPU devices.
    Less precise than OpenVINO detection but works without the
    OpenVINO Python package installed.
    """
    gpus: list[GPUInfo] = []
    drm_path = Path("/sys/class/drm")

    if not drm_path.exists():
        return gpus

    for card_dir in sorted(drm_path.iterdir()):
        if not card_dir.name.startswith("card") or card_dir.name.count("-") > 0:
            continue

        vendor_path = card_dir / "device" / "vendor"
        if not vendor_path.exists():
            continue

        try:
            vendor_id = vendor_path.read_text().strip()
        except OSError:
            continue

        # Intel vendor ID is 0x8086
        if vendor_id != "0x8086":
            continue

        # Try to get device name from driver
        device_name = _get_device_name_sysfs(card_dir)
        device_id = f"GPU.{len(gpus)}"
        vram_mb = _estimate_igpu_vram()

        # Check for dedicated VRAM (discrete GPU)
        vram_total_path = card_dir / "device" / "resource"
        if vram_total_path.exists():
            discrete_vram = _parse_pci_resource_vram(vram_total_path)
            if discrete_vram > 0:
                vram_mb = discrete_vram

        arch_tier = ArchTier.from_device_name(device_name)

        gpus.append(GPUInfo(
            name=device_name,
            device_id=device_id,
            vram_mb=vram_mb,
            arch_tier=arch_tier,
        ))

    return gpus


def _get_device_name_sysfs(card_dir: Path) -> str:
    """Get GPU device name from sysfs, falling back to lspci."""
    # Try reading device ID for lspci lookup
    device_path = card_dir / "device" / "device"
    if device_path.exists():
        try:
            device_id = device_path.read_text().strip()
            # Try lspci for a human-readable name
            result = subprocess.run(
                ["lspci", "-d", f"8086:{device_id[2:]}", "-nn"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                # Extract device description from lspci output
                parts = result.stdout.strip().split(": ", 1)
                if len(parts) > 1:
                    return parts[1].split(" [")[0]
        except Exception:
            pass

    return "Intel GPU (unknown model)"


def _parse_pci_resource_vram(resource_path: Path) -> int:
    """
    Parse PCI resource file to estimate discrete GPU VRAM.

    The PCI resource file contains BAR (Base Address Register) entries.
    BAR 2 is typically the VRAM aperture for discrete GPUs.
    Returns VRAM in MB, or 0 if not determinable.
    """
    try:
        lines = resource_path.read_text().strip().split("\n")
        if len(lines) >= 3:
            # BAR 2 (index 2) — VRAM aperture
            parts = lines[2].split()
            if len(parts) >= 2:
                start = int(parts[0], 16)
                end = int(parts[1], 16)
                if end > start:
                    size_mb = (end - start + 1) // (1024 * 1024)
                    if size_mb >= 1024:  # Sanity check: at least 1GB for discrete
                        return size_mb
    except Exception:
        pass
    return 0
