"""Tests for iNIM OVMS configurator module."""

from __future__ import annotations

import json
import os

import pytest

from inim.ovms_configurator import (
    generate_ovms_command,
    generate_ovms_config,
)


class TestOVMSConfigGeneration:
    """Test OVMS config.json and graph.pbtxt generation."""

    def test_generates_config_json(self, temp_model_dir, temp_dir, sample_manifest):
        """Should generate a valid config.json."""
        config_path = os.path.join(temp_dir, "config.json")
        profile = sample_manifest.profiles[0]

        generate_ovms_config(
            model_dir=temp_model_dir,
            model_name="test-model",
            profile=profile,
            config_output_path=config_path,
        )

        assert os.path.exists(config_path)

        with open(config_path) as f:
            config = json.load(f)

        assert "mediapipe_config_list" in config
        assert len(config["mediapipe_config_list"]) == 1
        assert config["mediapipe_config_list"][0]["name"] == "test-model"

    def test_generates_graph_pbtxt(self, temp_model_dir, temp_dir, sample_manifest):
        """Should generate a graph.pbtxt with correct parameters."""
        config_path = os.path.join(temp_dir, "config.json")
        profile = sample_manifest.profiles[0]

        generate_ovms_config(
            model_dir=temp_model_dir,
            model_name="test-model",
            profile=profile,
            config_output_path=config_path,
        )

        graph_path = os.path.join(temp_model_dir, "graph.pbtxt")
        assert os.path.exists(graph_path)

        with open(graph_path) as f:
            content = f.read()

        assert "HttpLLMCalculator" in content
        assert "max_num_seqs: 256" in content
        assert "cache_size: 4" in content
        assert 'device: "GPU"' in content

    def test_custom_device(self, temp_model_dir, temp_dir, sample_manifest):
        """Should use the specified device in graph.pbtxt."""
        config_path = os.path.join(temp_dir, "config.json")
        profile = sample_manifest.profiles[0]

        generate_ovms_config(
            model_dir=temp_model_dir,
            model_name="test-model",
            profile=profile,
            config_output_path=config_path,
            device="CPU",
        )

        graph_path = os.path.join(temp_model_dir, "graph.pbtxt")
        with open(graph_path) as f:
            content = f.read()

        assert 'device: "CPU"' in content

    def test_dynamic_quantization_options(self, temp_model_dir, temp_dir, sample_manifest):
        """Dynamic quantization should add KV_CACHE_PRECISION option."""
        config_path = os.path.join(temp_dir, "config.json")
        profile = sample_manifest.profiles[0]  # Has dynamic_quantization=True

        generate_ovms_config(
            model_dir=temp_model_dir,
            model_name="test-model",
            profile=profile,
            config_output_path=config_path,
        )

        graph_path = os.path.join(temp_model_dir, "graph.pbtxt")
        with open(graph_path) as f:
            content = f.read()

        assert "KV_CACHE_PRECISION" in content


class TestOVMSCommand:
    """Test OVMS launch command generation."""

    def test_default_command(self):
        """Should generate correct default OVMS command."""
        cmd = generate_ovms_command("/tmp/config.json")

        assert cmd[0] == "/ovms/bin/ovms"
        assert "--config_path" in cmd
        assert "/tmp/config.json" in cmd
        assert "--rest_port" in cmd
        assert "8001" in cmd
        assert "--rest_bind_address" in cmd
        assert "127.0.0.1" in cmd

    def test_custom_ports(self):
        """Should use custom ports."""
        cmd = generate_ovms_command(
            "/tmp/config.json",
            rest_port=9001,
            grpc_port=9002,
        )
        assert "9001" in cmd
        assert "9002" in cmd
