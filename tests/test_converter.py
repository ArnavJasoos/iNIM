"""Tests for iNIM converter modules."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from inim.converter.base import get_converter, validate_ir_output
from inim.converter.ir_passthrough import IRPassthroughConverter
from inim.models import Profile, SourceFormat


class TestConverterFactory:
    """Test converter factory function."""

    def test_get_ir_converter(self):
        converter = get_converter(SourceFormat.OPENVINO_IR)
        assert isinstance(converter, IRPassthroughConverter)

    def test_get_safetensors_converter(self):
        from inim.converter.safetensors_converter import SafetensorsConverter
        converter = get_converter(SourceFormat.PYTORCH_SAFETENSORS)
        assert isinstance(converter, SafetensorsConverter)

    def test_get_gguf_converter(self):
        from inim.converter.gguf_converter import GGUFConverter
        converter = get_converter(SourceFormat.GGUF)
        assert isinstance(converter, GGUFConverter)

    def test_unknown_format_raises(self):
        with pytest.raises(ValueError):
            get_converter(SourceFormat.UNKNOWN)


class TestIRValidation:
    """Test IR output validation."""

    def test_valid_ir_output(self, temp_model_dir):
        """Complete IR directory should validate."""
        is_valid, missing = validate_ir_output(temp_model_dir)
        assert is_valid is True
        assert missing == []

    def test_missing_model_xml(self, temp_dir):
        """Missing openvino_model.xml should fail validation."""
        model_dir = os.path.join(temp_dir, "incomplete")
        os.makedirs(model_dir)
        Path(os.path.join(model_dir, "openvino_model.bin")).write_text("mock")

        is_valid, missing = validate_ir_output(model_dir)
        assert is_valid is False
        assert "openvino_model.xml" in missing


class TestIRPassthrough:
    """Test IR pass-through converter."""

    def test_passthrough_local_dir(self, temp_model_dir, temp_dir, sample_manifest):
        """Should copy IR files from source to target."""
        target_dir = os.path.join(temp_dir, "target")
        profile = sample_manifest.profiles[0]

        converter = IRPassthroughConverter()
        result = converter.convert(temp_model_dir, target_dir, profile)

        assert os.path.exists(os.path.join(result, "openvino_model.xml"))
        assert os.path.exists(os.path.join(result, "openvino_model.bin"))

    def test_passthrough_same_dir(self, temp_model_dir, sample_manifest):
        """When source==target, should not error."""
        profile = sample_manifest.profiles[0]

        converter = IRPassthroughConverter()
        result = converter.convert(temp_model_dir, temp_model_dir, profile)
        assert result == temp_model_dir

    def test_validate_output(self, temp_model_dir):
        """Should validate a complete IR directory."""
        converter = IRPassthroughConverter()
        assert converter.validate_output(temp_model_dir) is True


class TestGGUFFileSelection:
    """Test GGUF file selection from multi-variant repos."""

    def test_select_int4_prefers_q4_k_m(self):
        from inim.models import select_gguf_file

        files = [
            "model-Q4_0.gguf",
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        selected = select_gguf_file(files, "int4")
        assert selected == "model-Q4_K_M.gguf"

    def test_select_int8_prefers_q8_0(self):
        from inim.models import select_gguf_file

        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        selected = select_gguf_file(files, "int8")
        assert selected == "model-Q8_0.gguf"

    def test_select_fp16_prefers_f16(self):
        from inim.models import select_gguf_file

        files = [
            "model-Q4_K_M.gguf",
            "model-Q8_0.gguf",
            "model-F16.gguf",
        ]
        selected = select_gguf_file(files, "fp16")
        assert selected == "model-F16.gguf"

    def test_fallback_to_first_gguf(self):
        from inim.models import select_gguf_file

        files = [
            "model-IQ3_M.gguf",
        ]
        selected = select_gguf_file(files, "int4")
        assert selected == "model-IQ3_M.gguf"

    def test_no_gguf_returns_none(self):
        from inim.models import select_gguf_file

        files = ["model.safetensors", "config.json"]
        selected = select_gguf_file(files, "int4")
        assert selected is None
