"""
OVMS configuration generator for iNIM orchestrator.

Generates the config.json and graph.pbtxt files required by OpenVINO
Model Server to serve an LLM with continuous batching. These files
are written at runtime based on the selected profile and cached model path.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from inim.logging_config import get_logger
from inim.models import Profile

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# graph.pbtxt template for OVMS LLM pipeline
# Uses MediaPipe graph with HttpLLMCalculator node
# ---------------------------------------------------------------------------

GRAPH_PBTXT_TEMPLATE = """\
input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"
node: {{
  name: "LLMExecutor"
  calculator: "HttpLLMCalculator"
  input_stream: "LOOPBACK:loopback"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "LLM_NODE_RESOURCES:llm"
  output_stream: "LOOPBACK:loopback"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  input_stream_info: {{
    tag_index: 'LOOPBACK:0'
    back_edge: true
  }}
  node_options: {{
    [type.googleapis.com/mediapipe.LLMCalculatorOptions] {{
      models_path: "{models_path}"
      max_num_seqs: {max_num_seqs}
      cache_size: {cache_size}
      device: "{device}"
{extra_options}    }}
  }}
}}
"""


def generate_ovms_config(
    model_dir: str,
    model_name: str,
    profile: Profile,
    config_output_path: str,
    graph_output_dir: Optional[str] = None,
    device: str = "GPU",
) -> tuple[str, str]:
    """
    Generate OVMS config.json and graph.pbtxt for the selected profile.

    Args:
        model_dir: Path to the directory containing OpenVINO IR files.
        model_name: Model serving name (used in /v1/models response).
        profile: Selected profile with OVMS configuration parameters.
        config_output_path: Path to write config.json.
        graph_output_dir: Directory for graph.pbtxt. If None, uses model_dir.
        device: Target device for inference ("GPU", "CPU", "GPU.0", etc.).

    Returns:
        Tuple of (config.json path, graph.pbtxt path).
    """
    if graph_output_dir is None:
        graph_output_dir = model_dir

    # --- Generate graph.pbtxt ---
    graph_path = os.path.join(graph_output_dir, "graph.pbtxt")
    _write_graph_pbtxt(
        output_path=graph_path,
        model_dir=model_dir,
        profile=profile,
        device=device,
    )

    # --- Generate config.json ---
    _write_config_json(
        output_path=config_output_path,
        model_name=model_name,
        model_dir=graph_output_dir,
    )

    logger.info(
        f"OVMS configuration generated: "
        f"config={config_output_path}, graph={graph_path}",
        extra={
            "component": "ovms_configurator",
            "model": model_name,
            "profile": profile.description,
        },
    )

    return config_output_path, graph_path


def _write_graph_pbtxt(
    output_path: str,
    model_dir: str,
    profile: Profile,
    device: str,
) -> None:
    """
    Write the MediaPipe graph.pbtxt for OVMS LLM pipeline.

    The graph defines how OVMS processes LLM requests with continuous
    batching, paged attention, and streaming support.
    """
    ovms_config = profile.ovms_config

    # Build extra options based on profile
    extra_lines = []

    if ovms_config.dynamic_quantization:
        extra_lines.append(
            '      plugin_config: "KV_CACHE_PRECISION={}"'.format(
                ovms_config.kv_cache_precision.upper()
            )
        )

    if profile.pipeline_type == "continuous_batching":
        extra_lines.append(
            '      enable_prefix_caching: true'
        )

    extra_options = ""
    if extra_lines:
        extra_options = "\n".join(extra_lines) + "\n"

    graph_content = GRAPH_PBTXT_TEMPLATE.format(
        models_path=model_dir,
        max_num_seqs=ovms_config.max_num_seqs,
        cache_size=ovms_config.cache_size,
        device=device,
        extra_options=extra_options,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(graph_content)

    logger.debug(
        f"Written graph.pbtxt: {output_path} "
        f"(max_num_seqs={ovms_config.max_num_seqs}, "
        f"cache_size={ovms_config.cache_size}GB, device={device})",
        extra={"component": "ovms_configurator"},
    )


def _write_config_json(
    output_path: str,
    model_name: str,
    model_dir: str,
) -> None:
    """
    Write the OVMS config.json pointing to the MediaPipe graph.

    OVMS uses this config to discover and load models at startup.
    The mediapipe_config_list entry points OVMS to the graph.pbtxt
    which in turn defines the LLM serving pipeline.
    """
    config = {
        "model_config_list": [],
        "mediapipe_config_list": [
            {
                "name": model_name,
                "base_path": model_dir,
            }
        ],
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.debug(
        f"Written config.json: {output_path} (model_name={model_name})",
        extra={"component": "ovms_configurator"},
    )


def generate_ovms_command(
    config_path: str,
    rest_port: int = 8001,
    grpc_port: int = 8002,
    log_level: str = "INFO",
) -> list[str]:
    """
    Generate the OVMS launch command.

    Args:
        config_path: Path to the generated config.json.
        rest_port: REST API port (loopback only).
        grpc_port: gRPC API port (loopback only).
        log_level: OVMS log level.

    Returns:
        Command line arguments as a list of strings.
    """
    cmd = [
        "/ovms/bin/ovms",
        "--config_path", config_path,
        "--rest_port", str(rest_port),
        "--grpc_port", str(grpc_port),
        "--rest_bind_address", "127.0.0.1",
        "--grpc_bind_address", "127.0.0.1",
        "--log_level", log_level,
        "--file_system_poll_wait_seconds", "0",  # Disable model polling
    ]

    return cmd
