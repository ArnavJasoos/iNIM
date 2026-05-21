"""Tests for iNIM profile selector module."""

from __future__ import annotations

import pytest

from inim.exceptions import NoCompatibleProfileError
from inim.models import ArchTier, GPUInfo, Manifest
from inim.profile_selector import load_manifest, select_profile


class TestProfileSelection:
    """Test profile selection algorithm."""

    def test_arc_a770_selects_int4_arc_hq(self, sample_manifest, arc_a770_gpu):
        """Arc A770 should select int4-arc-hq-cb (best perf)."""
        profile = select_profile(sample_manifest, arc_a770_gpu)
        assert profile.description == "ovms-int4-arc-hq-cb"
        assert profile.precision == "int4"

    def test_arc_b580_selects_int4_arc_hq(self, sample_manifest, arc_b580_gpu):
        """Arc B580 should also match int4-arc-hq-cb."""
        profile = select_profile(sample_manifest, arc_b580_gpu)
        assert profile.description == "ovms-int4-arc-hq-cb"

    def test_arc_a380_selects_int4_arc(self, sample_manifest, arc_a380_gpu):
        """Arc A380 (6GB) should select int4-arc-cb profile (lower VRAM)."""
        # A380 has 6GB, so it can't use the 8GB+ profile
        # but should match the 4GB+ arc profile
        profile = select_profile(sample_manifest, arc_a380_gpu)
        assert "arc" in profile.description
        assert profile.precision == "int4"

    def test_xe_igpu_selects_igpu_cb(self, sample_manifest, xe_igpu):
        """Xe iGPU should select int4-igpu-cb."""
        profile = select_profile(sample_manifest, xe_igpu)
        assert profile.description == "ovms-int4-igpu-cb"
        assert profile.pipeline_type == "continuous_batching"

    def test_uhd_igpu_selects_stateful(self, sample_manifest, uhd_igpu):
        """UHD iGPU should select int4-igpu-st (stateful, low memory)."""
        profile = select_profile(sample_manifest, uhd_igpu)
        assert profile.description == "ovms-int4-igpu-st"
        assert profile.pipeline_type == "stateful"

    def test_override_by_id(self, sample_manifest, arc_a770_gpu):
        """Profile override by exact ID should work."""
        profile = select_profile(
            sample_manifest, arc_a770_gpu,
            profile_override="prof-fp16-arc",
        )
        assert profile.id == "prof-fp16-arc"
        assert profile.precision == "fp16"

    def test_override_by_description(self, sample_manifest, arc_a770_gpu):
        """Profile override by description should work."""
        profile = select_profile(
            sample_manifest, arc_a770_gpu,
            profile_override="ovms-fp16-arc-hq-cb",
        )
        assert profile.precision == "fp16"

    def test_override_partial_match(self, sample_manifest, arc_a770_gpu):
        """Profile override by partial description should work."""
        profile = select_profile(
            sample_manifest, arc_a770_gpu,
            profile_override="fp16",
        )
        assert profile.precision == "fp16"

    def test_non_runnable_profiles_excluded(self, sample_manifest, arc_a770_gpu):
        """Non-runnable profiles should not be selected."""
        profile = select_profile(sample_manifest, arc_a770_gpu)
        assert profile.runnable is True

    def test_no_compatible_profile_raises(self, sample_manifest):
        """GPU with insufficient VRAM for all profiles should raise."""
        tiny_gpu = GPUInfo(
            name="Intel(R) UHD Graphics 630",
            device_id="GPU.0",
            vram_mb=512,  # Too small for any profile
            arch_tier=ArchTier.UHD_INTEGRATED,
        )
        with pytest.raises(NoCompatibleProfileError):
            select_profile(sample_manifest, tiny_gpu)

    def test_prefers_continuous_batching(self, sample_manifest):
        """When multiple profiles match, prefer continuous_batching."""
        # Xe iGPU with enough VRAM should get CB over stateful
        gpu = GPUInfo(
            name="Intel(R) Graphics (MTL)",
            device_id="GPU.0",
            vram_mb=8192,
            arch_tier=ArchTier.XE_INTEGRATED,
        )
        profile = select_profile(sample_manifest, gpu)
        assert profile.pipeline_type == "continuous_batching"


class TestManifestLoading:
    """Test manifest file loading and parsing."""

    def test_load_manifest_from_file(self, temp_manifest_file):
        """Should load and parse a valid manifest YAML file."""
        manifest = load_manifest(temp_manifest_file)
        assert manifest.model_name == "meta-llama/Llama-3.2-3B-Instruct"
        assert manifest.model_id == "llama-3.2-3b-instruct"
        assert len(manifest.profiles) == 5

    def test_load_manifest_missing_file(self):
        """Should raise ManifestError for missing file."""
        from inim.exceptions import ManifestError
        with pytest.raises(ManifestError):
            load_manifest("/nonexistent/path.yaml")

    def test_profile_from_dict(self, sample_manifest_data):
        """Profile.from_dict should correctly parse all fields."""
        from inim.models import Profile

        profile_data = sample_manifest_data["profiles"][0]
        profile = Profile.from_dict(profile_data)

        assert profile.id == "prof-int4-arc-hq"
        assert profile.precision == "int4"
        assert profile.ovms_config.max_num_seqs == 256
        assert profile.ovms_config.cache_size == 4
        assert profile.device_filter.min_vram_mb == 8192
        assert "arc_alchemist" in profile.device_filter.arch_tiers
