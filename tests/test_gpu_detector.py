"""Tests for iNIM GPU detector module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from inim.exceptions import GPUDetectionError
from inim.models import ArchTier, GPUInfo


class TestArchTierClassification:
    """Test ArchTier.from_device_name classification."""

    def test_arc_a770(self):
        assert ArchTier.from_device_name(
            "Intel(R) Arc(TM) A770 Graphics"
        ) == ArchTier.ARC_ALCHEMIST

    def test_arc_a380(self):
        assert ArchTier.from_device_name(
            "Intel(R) Arc(TM) A380 Graphics"
        ) == ArchTier.ARC_ALCHEMIST

    def test_arc_b580(self):
        assert ArchTier.from_device_name(
            "Intel(R) Arc(TM) B580 Graphics"
        ) == ArchTier.ARC_BATTLEMAGE

    def test_arc_b770(self):
        assert ArchTier.from_device_name(
            "Intel(R) Arc(TM) B770 Graphics"
        ) == ArchTier.ARC_BATTLEMAGE

    def test_xe_mtl(self):
        assert ArchTier.from_device_name(
            "Intel(R) Graphics (MTL)"
        ) == ArchTier.XE_INTEGRATED

    def test_xe_core_ultra(self):
        assert ArchTier.from_device_name(
            "Intel(R) Core Ultra Xe Graphics"
        ) == ArchTier.XE_INTEGRATED

    def test_uhd_770(self):
        assert ArchTier.from_device_name(
            "Intel(R) UHD Graphics 770"
        ) == ArchTier.UHD_INTEGRATED

    def test_uhd_630(self):
        assert ArchTier.from_device_name(
            "Intel(R) UHD Graphics 630"
        ) == ArchTier.UHD_INTEGRATED

    def test_unknown_defaults_to_uhd(self):
        assert ArchTier.from_device_name(
            "Intel(R) Graphics"
        ) == ArchTier.UHD_INTEGRATED


class TestGPUInfo:
    """Test GPUInfo dataclass behavior."""

    def test_xmx_supported_arc(self):
        gpu = GPUInfo(
            name="Arc A770", device_id="GPU.0",
            vram_mb=16384, arch_tier=ArchTier.ARC_ALCHEMIST,
        )
        assert gpu.xmx_supported is True

    def test_xmx_supported_xe(self):
        gpu = GPUInfo(
            name="Xe iGPU", device_id="GPU.0",
            vram_mb=8192, arch_tier=ArchTier.XE_INTEGRATED,
        )
        assert gpu.xmx_supported is True

    def test_xmx_not_supported_uhd(self):
        gpu = GPUInfo(
            name="UHD 770", device_id="GPU.0",
            vram_mb=2048, arch_tier=ArchTier.UHD_INTEGRATED,
        )
        assert gpu.xmx_supported is False


class TestGPUDetection:
    """Test GPU detection functions."""

    @patch("inim.gpu_detector._detect_via_openvino")
    def test_detect_gpus_returns_list(self, mock_ov):
        from inim.gpu_detector import detect_gpus

        mock_ov.return_value = [
            GPUInfo(
                name="Intel(R) Arc(TM) A770 Graphics",
                device_id="GPU.0",
                vram_mb=16384,
                arch_tier=ArchTier.ARC_ALCHEMIST,
            ),
        ]

        gpus = detect_gpus()
        assert len(gpus) == 1
        assert gpus[0].arch_tier == ArchTier.ARC_ALCHEMIST

    @patch("inim.gpu_detector._detect_via_openvino")
    @patch("inim.gpu_detector._detect_via_sysfs")
    def test_detect_gpus_no_gpu_raises(self, mock_sysfs, mock_ov):
        from inim.gpu_detector import detect_gpus

        mock_ov.return_value = []
        mock_sysfs.return_value = []

        with pytest.raises(GPUDetectionError):
            detect_gpus()

    @patch("inim.gpu_detector._detect_via_openvino")
    def test_detect_primary_selects_best(self, mock_ov):
        from inim.gpu_detector import detect_primary_gpu

        mock_ov.return_value = [
            GPUInfo(
                name="Intel(R) UHD Graphics 770",
                device_id="GPU.0",
                vram_mb=2048,
                arch_tier=ArchTier.UHD_INTEGRATED,
            ),
            GPUInfo(
                name="Intel(R) Arc(TM) A770 Graphics",
                device_id="GPU.1",
                vram_mb=16384,
                arch_tier=ArchTier.ARC_ALCHEMIST,
            ),
        ]

        primary = detect_primary_gpu()
        assert primary.device_id == "GPU.1"
        assert primary.arch_tier == ArchTier.ARC_ALCHEMIST
