"""Tests for iNIM format detector module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from inim.exceptions import UnknownFormatError
from inim.format_detector import detect_source_format
from inim.models import SourceFormat


class TestLocalFormatDetection:
    """Test format detection for local directories."""

    def test_detect_openvino_ir(self, temp_model_dir):
        """Directory with .xml files should be detected as IR."""
        fmt = detect_source_format(temp_model_dir)
        assert fmt == SourceFormat.OPENVINO_IR

    def test_detect_safetensors(self, temp_safetensors_dir):
        """Directory with .safetensors should be detected."""
        fmt = detect_source_format(temp_safetensors_dir)
        assert fmt == SourceFormat.PYTORCH_SAFETENSORS

    def test_detect_gguf(self, temp_gguf_dir):
        """Directory with .gguf files should be detected."""
        fmt = detect_source_format(temp_gguf_dir)
        assert fmt == SourceFormat.GGUF

    def test_detect_pytorch_bin(self, temp_dir):
        """Directory with pytorch_model.bin + config.json."""
        model_dir = os.path.join(temp_dir, "pytorch")
        os.makedirs(model_dir)
        Path(os.path.join(model_dir, "pytorch_model.bin")).write_text("mock")
        Path(os.path.join(model_dir, "config.json")).write_text("mock")

        fmt = detect_source_format(model_dir)
        assert fmt == SourceFormat.PYTORCH_SAFETENSORS

    def test_ir_takes_priority_over_safetensors(self, temp_dir):
        """IR should be detected even if safetensors also present."""
        model_dir = os.path.join(temp_dir, "mixed")
        os.makedirs(model_dir)
        Path(os.path.join(model_dir, "openvino_model.xml")).write_text("mock")
        Path(os.path.join(model_dir, "openvino_model.bin")).write_text("mock")
        Path(os.path.join(model_dir, "model.safetensors")).write_text("mock")

        fmt = detect_source_format(model_dir)
        assert fmt == SourceFormat.OPENVINO_IR

    def test_gguf_takes_priority_over_safetensors(self, temp_dir):
        """GGUF should be detected over safetensors."""
        model_dir = os.path.join(temp_dir, "gguf_mixed")
        os.makedirs(model_dir)
        Path(os.path.join(model_dir, "model.gguf")).write_text("mock")
        Path(os.path.join(model_dir, "model.safetensors")).write_text("mock")

        fmt = detect_source_format(model_dir)
        assert fmt == SourceFormat.GGUF

    def test_empty_dir_raises(self, temp_dir):
        """Empty directory should raise UnknownFormatError."""
        empty_dir = os.path.join(temp_dir, "empty")
        os.makedirs(empty_dir)

        with pytest.raises(UnknownFormatError):
            detect_source_format(empty_dir)

    def test_nonexistent_dir_not_treated_as_hf_repo(self, temp_dir):
        """Non-existent path that's not an HF repo should raise."""
        # This would try to query HF Hub and fail
        with pytest.raises((UnknownFormatError, Exception)):
            detect_source_format("/nonexistent/path/not-a-repo-id")


class TestSingleFileDetection:
    """Test format detection for single files."""

    def test_single_gguf_file(self, temp_dir):
        """Single .gguf file path should be detected."""
        gguf_file = os.path.join(temp_dir, "model.gguf")
        Path(gguf_file).write_text("mock")

        fmt = detect_source_format(gguf_file)
        assert fmt == SourceFormat.GGUF

    def test_single_safetensors_file(self, temp_dir):
        """Single .safetensors file path should be detected."""
        st_file = os.path.join(temp_dir, "model.safetensors")
        Path(st_file).write_text("mock")

        fmt = detect_source_format(st_file)
        assert fmt == SourceFormat.PYTORCH_SAFETENSORS

    def test_unknown_single_file_raises(self, temp_dir):
        """Single file with unknown extension should raise."""
        unknown_file = os.path.join(temp_dir, "model.unknown")
        Path(unknown_file).write_text("mock")

        with pytest.raises(UnknownFormatError):
            detect_source_format(unknown_file)
