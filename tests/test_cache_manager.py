"""Tests for iNIM cache manager module."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from inim.cache_manager import (
    check_cache,
    compute_sha256,
    get_cache_size_mb,
    load_cache_manifest,
    write_cache_manifest,
)
from inim.exceptions import CacheIntegrityError


class TestCacheManifest:
    """Test cache manifest read/write."""

    def test_write_and_read_manifest(self, temp_model_dir):
        """Should write and read back a cache manifest."""
        manifest = write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test/model",
            profile_id="test-profile-123",
            precision="int4",
            source_format="openvino_ir",
            source_repo="test/model",
        )

        assert manifest.model_name == "test/model"
        assert manifest.profile_id == "test-profile-123"
        assert len(manifest.files) > 0

        # Read it back
        manifest_path = os.path.join(
            temp_model_dir, ".inim_cache_manifest.json"
        )
        loaded = load_cache_manifest(manifest_path)

        assert loaded.model_name == manifest.model_name
        assert loaded.profile_id == manifest.profile_id
        assert loaded.files == manifest.files

    def test_checksums_are_deterministic(self, temp_model_dir):
        """Same files should produce same checksums."""
        m1 = write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test", profile_id="p1",
            precision="int4", source_format="ir",
            source_repo="test",
        )
        m2 = write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test", profile_id="p1",
            precision="int4", source_format="ir",
            source_repo="test",
        )

        assert m1.files == m2.files


class TestCacheCheck:
    """Test cache validation."""

    def test_valid_cache_returns_path(self, temp_model_dir):
        """Valid cache with matching profile should return path."""
        write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test", profile_id="profile-abc",
            precision="int4", source_format="ir",
            source_repo="test",
        )

        result = check_cache(temp_model_dir, "profile-abc")
        assert result == temp_model_dir

    def test_profile_mismatch_returns_none(self, temp_model_dir):
        """Cache with different profile ID should return None."""
        write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test", profile_id="profile-abc",
            precision="int4", source_format="ir",
            source_repo="test",
        )

        result = check_cache(temp_model_dir, "different-profile")
        assert result is None

    def test_missing_manifest_returns_none(self, temp_dir):
        """Directory without manifest should return None."""
        result = check_cache(temp_dir, "any-profile")
        assert result is None

    def test_corrupted_file_returns_none(self, temp_model_dir):
        """Modified file should fail checksum and return None."""
        write_cache_manifest(
            cache_dir=temp_model_dir,
            model_name="test", profile_id="profile-abc",
            precision="int4", source_format="ir",
            source_repo="test",
        )

        # Corrupt a file
        xml_path = os.path.join(temp_model_dir, "openvino_model.xml")
        Path(xml_path).write_text("corrupted content")

        result = check_cache(temp_model_dir, "profile-abc")
        assert result is None


class TestSHA256:
    """Test SHA-256 checksum computation."""

    def test_compute_sha256(self, temp_dir):
        """Should compute correct SHA-256 for a file."""
        filepath = os.path.join(temp_dir, "test.txt")
        Path(filepath).write_text("hello world")

        digest = compute_sha256(filepath)
        assert len(digest) == 64  # SHA-256 hex digest length
        assert digest.isalnum()

    def test_sha256_deterministic(self, temp_dir):
        """Same content should produce same hash."""
        filepath = os.path.join(temp_dir, "test.txt")
        Path(filepath).write_text("deterministic content")

        d1 = compute_sha256(filepath)
        d2 = compute_sha256(filepath)
        assert d1 == d2


class TestCacheSize:
    """Test cache size calculation."""

    def test_get_cache_size(self, temp_model_dir):
        """Should return non-zero size for a directory with files."""
        size_mb = get_cache_size_mb(temp_model_dir)
        assert size_mb > 0

    def test_empty_dir_size(self, temp_dir):
        """Empty directory should have ~0 size."""
        empty = os.path.join(temp_dir, "empty")
        os.makedirs(empty)
        size_mb = get_cache_size_mb(empty)
        assert size_mb == 0
