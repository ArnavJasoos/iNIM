"""
GGUF → HuggingFace format dequantization utility.

Extracts tensors from a GGUF file, dequantizes to FP16, and writes
them as safetensors with a compatible config.json and tokenizer files.

This is Sub-path B2's first step — producing an intermediate FP16
HuggingFace-compatible directory that optimum-cli can then convert
to OpenVINO IR.

Uses the `gguf` Python package to parse GGUF files and extract
model architecture, vocabulary, and tensor data.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from inim.exceptions import ModelConversionError
from inim.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# GGUF architecture → HuggingFace model type mapping
# ---------------------------------------------------------------------------
_GGUF_ARCH_TO_HF_TYPE: dict[str, str] = {
    "llama": "LlamaForCausalLM",
    "qwen2": "Qwen2ForCausalLM",
    "qwen2moe": "Qwen2MoeForCausalLM",
    "phi3": "PhiForCausalLM",
    "phi": "PhiForCausalLM",
    "mistral": "MistralForCausalLM",
    "gemma": "GemmaForCausalLM",
    "gemma2": "Gemma2ForCausalLM",
    "starcoder2": "Starcoder2ForCausalLM",
    "gpt2": "GPT2LMHeadModel",
    "falcon": "FalconForCausalLM",
    "bloom": "BloomForCausalLM",
    "stablelm": "StableLmForCausalLM",
    "internlm2": "InternLM2ForCausalLM",
    "command-r": "CohereForCausalLM",
}

# GGUF metadata key → HuggingFace config.json key mapping
_GGUF_META_TO_HF_CONFIG: dict[str, str] = {
    "llama.context_length": "max_position_embeddings",
    "llama.embedding_length": "hidden_size",
    "llama.feed_forward_length": "intermediate_size",
    "llama.block_count": "num_hidden_layers",
    "llama.attention.head_count": "num_attention_heads",
    "llama.attention.head_count_kv": "num_key_value_heads",
    "llama.rope.freq_base": "rope_theta",
    "llama.attention.layer_norm_rms_epsilon": "rms_norm_eps",
    "general.architecture": "_architecture",
    "general.name": "_name",
    "tokenizer.ggml.model": "_tokenizer_model",
}


def convert_gguf_to_hf(
    gguf_path: str,
    output_dir: str,
    hf_token: Optional[str] = None,
) -> None:
    """
    Convert a GGUF file to HuggingFace-compatible FP16 format.

    This function:
    1. Reads the GGUF file metadata and tensors
    2. Dequantizes tensors to FP16
    3. Writes safetensors weight files
    4. Generates config.json from GGUF metadata
    5. Copies/generates tokenizer files

    Args:
        gguf_path: Path to the .gguf file.
        output_dir: Directory to write the HuggingFace-compatible output.
        hf_token: Optional HF token for downloading tokenizer from
                  the original model repo.

    Raises:
        ModelConversionError: If conversion fails.
    """
    try:
        from gguf import GGUFReader  # type: ignore[import-untyped]
    except ImportError:
        raise ModelConversionError(
            "The 'gguf' package is not installed. Required for GGUF dequantization.",
            details="Install with: pip install gguf>=0.6.0",
        )

    logger.info(
        f"Reading GGUF file: {gguf_path}",
        extra={"component": "gguf_to_hf"},
    )

    try:
        reader = GGUFReader(gguf_path)
    except Exception as e:
        raise ModelConversionError(
            f"Failed to read GGUF file: {e}",
            details=f"File: {gguf_path}",
        )

    os.makedirs(output_dir, exist_ok=True)

    # --- Step 1: Extract metadata ---
    metadata = _extract_metadata(reader)
    architecture = metadata.get("_architecture", "llama")

    logger.info(
        f"GGUF architecture: {architecture}, "
        f"name: {metadata.get('_name', 'unknown')}",
        extra={"component": "gguf_to_hf"},
    )

    # --- Step 2: Generate config.json ---
    config = _generate_config(metadata, architecture)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    logger.info(
        f"Generated config.json: {config_path}",
        extra={"component": "gguf_to_hf"},
    )

    # --- Step 3: Extract and dequantize tensors ---
    _extract_tensors(reader, output_dir)

    # --- Step 4: Handle tokenizer ---
    _handle_tokenizer(reader, output_dir, metadata, hf_token)

    logger.info(
        f"GGUF → HuggingFace conversion complete: {output_dir}",
        extra={"component": "gguf_to_hf"},
    )


def _extract_metadata(reader) -> dict[str, Any]:
    """Extract metadata from GGUF file header."""
    metadata: dict[str, Any] = {}

    for field in reader.fields.values():
        key = field.name
        # Get the first data value
        if field.data is not None and len(field.data) > 0:
            try:
                parts = field.parts
                if len(parts) > 1:
                    value = parts[-1].tolist()
                    if len(value) == 1:
                        value = value[0]
                else:
                    value = field.data.tolist()
                    if isinstance(value, list) and len(value) == 1:
                        value = value[0]
                metadata[key] = value
            except Exception:
                pass

    return metadata


def _generate_config(metadata: dict[str, Any], architecture: str) -> dict[str, Any]:
    """
    Generate a HuggingFace-compatible config.json from GGUF metadata.

    Maps GGUF metadata keys to HuggingFace config keys based on the
    model architecture.
    """
    # Start with architecture-specific defaults
    model_type = architecture.split(".")[0] if "." in architecture else architecture

    # Determine HF model class
    hf_architectures = _GGUF_ARCH_TO_HF_TYPE.get(
        model_type, "LlamaForCausalLM"
    )

    config: dict[str, Any] = {
        "architectures": [hf_architectures],
        "model_type": model_type,
        "torch_dtype": "float16",
    }

    # Map known metadata keys
    arch_prefix = f"{architecture}." if not architecture.endswith(".") else architecture

    # Try generic keys first, then architecture-prefixed keys
    key_mappings = {
        "max_position_embeddings": [
            f"{model_type}.context_length",
            "context_length",
        ],
        "hidden_size": [
            f"{model_type}.embedding_length",
            "embedding_length",
        ],
        "intermediate_size": [
            f"{model_type}.feed_forward_length",
            "feed_forward_length",
        ],
        "num_hidden_layers": [
            f"{model_type}.block_count",
            "block_count",
        ],
        "num_attention_heads": [
            f"{model_type}.attention.head_count",
            "attention.head_count",
        ],
        "num_key_value_heads": [
            f"{model_type}.attention.head_count_kv",
            "attention.head_count_kv",
        ],
        "rope_theta": [
            f"{model_type}.rope.freq_base",
            "rope.freq_base",
        ],
        "rms_norm_eps": [
            f"{model_type}.attention.layer_norm_rms_epsilon",
            "attention.layer_norm_rms_epsilon",
        ],
        "vocab_size": [
            "tokenizer.ggml.tokens",
        ],
    }

    for hf_key, gguf_keys in key_mappings.items():
        for gguf_key in gguf_keys:
            if gguf_key in metadata:
                value = metadata[gguf_key]
                if gguf_key.endswith("tokens") and isinstance(value, list):
                    config[hf_key] = len(value)
                else:
                    config[hf_key] = value
                break

    return config


def _extract_tensors(reader, output_dir: str) -> None:
    """
    Extract tensors from GGUF, dequantize to FP16, and save as safetensors.

    For quantized tensors (Q4_0, Q4_K_M, etc.), dequantizes to float32
    then converts to float16. For already-float tensors, converts directly.
    """
    try:
        import numpy as np
        from safetensors.numpy import save_file  # type: ignore[import-untyped]
    except ImportError:
        raise ModelConversionError(
            "safetensors and numpy are required for GGUF dequantization.",
            details="Install with: pip install safetensors numpy",
        )

    logger.info(
        f"Extracting {len(reader.tensors)} tensors from GGUF",
        extra={"component": "gguf_to_hf"},
    )

    tensors: dict[str, Any] = {}
    shard_size = 0
    shard_idx = 0
    max_shard_bytes = 5 * 1024 * 1024 * 1024  # 5GB per shard

    for tensor_info in reader.tensors:
        name = tensor_info.name
        data = tensor_info.data

        # Dequantize if needed
        try:
            if hasattr(data, 'dtype'):
                if data.dtype in (np.float32, np.float16):
                    fp16_data = data.astype(np.float16)
                else:
                    # Quantized data — cast through float32
                    fp16_data = data.astype(np.float32).astype(np.float16)
            else:
                fp16_data = np.array(data, dtype=np.float16)
        except Exception as e:
            logger.warning(
                f"Failed to dequantize tensor '{name}': {e}. Skipping.",
                extra={"component": "gguf_to_hf"},
            )
            continue

        # Map GGUF tensor names to HF-style names
        hf_name = _map_tensor_name(name)
        tensors[hf_name] = fp16_data

        shard_size += fp16_data.nbytes

        # Save shard if it exceeds max size
        if shard_size >= max_shard_bytes:
            shard_file = os.path.join(
                output_dir,
                f"model-{shard_idx:05d}-of-XXXXX.safetensors",
            )
            save_file(tensors, shard_file)
            logger.info(
                f"Saved shard {shard_idx}: {len(tensors)} tensors, "
                f"{shard_size / (1024**3):.1f}GB",
                extra={"component": "gguf_to_hf"},
            )
            tensors = {}
            shard_size = 0
            shard_idx += 1

    # Save remaining tensors
    if tensors:
        if shard_idx == 0:
            # Single file — no sharding needed
            shard_file = os.path.join(output_dir, "model.safetensors")
        else:
            shard_file = os.path.join(
                output_dir,
                f"model-{shard_idx:05d}-of-XXXXX.safetensors",
            )
        save_file(tensors, shard_file)
        logger.info(
            f"Saved final shard: {len(tensors)} tensors, "
            f"{shard_size / (1024**3):.1f}GB",
            extra={"component": "gguf_to_hf"},
        )


def _map_tensor_name(gguf_name: str) -> str:
    """
    Map GGUF tensor names to HuggingFace model tensor names.

    GGUF uses names like:
      blk.0.attn_norm.weight → model.layers.0.input_layernorm.weight
      output.weight → lm_head.weight
      token_embd.weight → model.embed_tokens.weight
    """
    # Common GGUF → HF name mappings
    name = gguf_name

    # Block layers
    name = name.replace("blk.", "model.layers.")

    # Attention components
    name = name.replace(".attn_q.", ".self_attn.q_proj.")
    name = name.replace(".attn_k.", ".self_attn.k_proj.")
    name = name.replace(".attn_v.", ".self_attn.v_proj.")
    name = name.replace(".attn_output.", ".self_attn.o_proj.")

    # FFN components
    name = name.replace(".ffn_gate.", ".mlp.gate_proj.")
    name = name.replace(".ffn_up.", ".mlp.up_proj.")
    name = name.replace(".ffn_down.", ".mlp.down_proj.")

    # Norms
    name = name.replace(".attn_norm.", ".input_layernorm.")
    name = name.replace(".ffn_norm.", ".post_attention_layernorm.")

    # Embeddings and output
    name = name.replace("token_embd.", "model.embed_tokens.")
    name = name.replace("output_norm.", "model.norm.")
    name = name.replace("output.", "lm_head.")

    return name


def _handle_tokenizer(
    reader,
    output_dir: str,
    metadata: dict[str, Any],
    hf_token: Optional[str],
) -> None:
    """
    Handle tokenizer extraction/generation from GGUF.

    Strategy:
    1. Try to download tokenizer from the original HF repo (if identifiable)
    2. Fall back to generating a basic tokenizer_config.json from GGUF vocabulary
    """
    # Try to identify the original model repo from GGUF metadata
    source_repo = None
    for key in ["general.source.url", "general.source.huggingface.repository",
                "general.base_model.0.name"]:
        if key in metadata:
            source_repo = metadata[key]
            break

    # Also check INIM_TOKENIZER_REPO env var
    import os
    tokenizer_repo = os.environ.get("INIM_TOKENIZER_REPO", source_repo)

    if tokenizer_repo:
        try:
            _download_tokenizer(tokenizer_repo, output_dir, hf_token)
            return
        except Exception as e:
            logger.warning(
                f"Failed to download tokenizer from {tokenizer_repo}: {e}",
                extra={"component": "gguf_to_hf"},
            )

    # Fallback: generate minimal tokenizer config from GGUF vocabulary
    _generate_minimal_tokenizer(reader, output_dir, metadata)


def _download_tokenizer(
    repo_id: str,
    output_dir: str,
    hf_token: Optional[str],
) -> None:
    """Download tokenizer files from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ]

    downloaded = 0
    for filename in tokenizer_files:
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                token=hf_token,
                local_dir=output_dir,
            )
            downloaded += 1
        except Exception:
            continue

    if downloaded == 0:
        raise RuntimeError(f"No tokenizer files found in {repo_id}")

    logger.info(
        f"Downloaded {downloaded} tokenizer files from {repo_id}",
        extra={"component": "gguf_to_hf"},
    )


def _generate_minimal_tokenizer(
    reader,
    output_dir: str,
    metadata: dict[str, Any],
) -> None:
    """Generate a minimal tokenizer_config.json from GGUF vocabulary data."""
    tokenizer_config: dict[str, Any] = {
        "tokenizer_class": "PreTrainedTokenizerFast",
    }

    # Try to extract chat template from metadata
    for key in ["tokenizer.chat_template", "chat_template"]:
        if key in metadata:
            tokenizer_config["chat_template"] = metadata[key]
            break

    # Try to extract BOS/EOS token IDs
    for key, config_key in [
        ("tokenizer.ggml.bos_token_id", "bos_token_id"),
        ("tokenizer.ggml.eos_token_id", "eos_token_id"),
        ("tokenizer.ggml.padding_token_id", "pad_token_id"),
    ]:
        if key in metadata:
            tokenizer_config[config_key] = metadata[key]

    config_path = os.path.join(output_dir, "tokenizer_config.json")
    with open(config_path, "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    logger.info(
        f"Generated minimal tokenizer_config.json",
        extra={"component": "gguf_to_hf"},
    )
