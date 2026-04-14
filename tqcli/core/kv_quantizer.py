"""KV cache quantization selection for TurboQuant.

Selects the optimal TurboQuant KV cache compression level based on
available KV cache memory budget and inference engine.

TurboQuant compresses the KV cache at runtime using PolarQuant +
Walsh-Hadamard rotation. The model weights are unchanged — only the
attention key/value cache is compressed during inference.

Compression levels:
  turbo4: 4.25 bpv, 3.8x compression, near-lossless (+0.23% PPL)
  turbo3: 3.5 bpv, 4.6x compression, minimal loss (+1.06% PPL)
  turbo2: 2.5 bpv, 6.4x compression, noticeable loss (+6.48% PPL)

References:
  - TurboQuant paper: arxiv.org/abs/2504.19874 (ICLR 2026)
  - llama.cpp fork: github.com/TheTom/llama-cpp-turboquant
  - vLLM fork: github.com/mitkox/vllm-turboquant
"""

from __future__ import annotations

from enum import Enum


class KVQuantLevel(Enum):
    """KV cache compression levels."""
    NONE = "none"       # Default (q8_0 for llama.cpp, auto for vLLM)
    TURBO4 = "turbo4"   # 4.25 bpv, 3.8x, near-lossless
    TURBO3 = "turbo3"   # 3.5 bpv, 4.6x, minimal loss
    TURBO2 = "turbo2"   # 2.5 bpv, 6.4x, quality trade-off


# Compression ratios relative to q8_0 (8.5 bpv)
KV_COMPRESSION_RATIO = {
    KVQuantLevel.NONE: 1.0,
    KVQuantLevel.TURBO4: 3.8,
    KVQuantLevel.TURBO3: 4.6,
    KVQuantLevel.TURBO2: 6.4,
}

# Perplexity impact (% increase vs q8_0 baseline)
KV_PPL_IMPACT = {
    KVQuantLevel.NONE: 0.0,
    KVQuantLevel.TURBO4: 0.23,
    KVQuantLevel.TURBO3: 1.06,
    KVQuantLevel.TURBO2: 6.48,
}


def select_kv_quant(
    available_kv_mb: float,
    engine: str = "llama.cpp",
    user_choice: str = "auto",
) -> KVQuantLevel:
    """Select KV cache compression level based on available memory.

    Args:
        available_kv_mb: Available memory for KV cache in MB.
        engine: Inference engine ("llama.cpp" or "vllm").
        user_choice: User's explicit choice or "auto" for automatic.

    Returns:
        KVQuantLevel to use.
    """
    # Explicit user choice
    if user_choice != "auto":
        try:
            return KVQuantLevel(user_choice)
        except ValueError:
            pass

    # Auto-select based on available KV memory
    if available_kv_mb >= 200:
        return KVQuantLevel.NONE  # Plenty of room, no compression needed
    elif available_kv_mb >= 50:
        return KVQuantLevel.TURBO4  # 3.8x, near-lossless
    elif available_kv_mb >= 20:
        return KVQuantLevel.TURBO3  # 4.6x, minimal loss
    else:
        return KVQuantLevel.TURBO2  # 6.4x, quality warning


def estimate_context_tokens(
    available_kv_mb: float,
    param_billions: float,
    kv_level: KVQuantLevel,
) -> int:
    """Estimate achievable context length with given KV compression.

    Args:
        available_kv_mb: Available memory for KV cache in MB.
        param_billions: Model parameter count in billions.
        kv_level: KV cache compression level.

    Returns:
        Estimated maximum context tokens.
    """
    # Base KV usage: ~0.14 MB per token per billion params at q8_0
    base_kv_per_token_mb = 0.14 * param_billions
    # Apply compression
    compressed_kv_per_token_mb = base_kv_per_token_mb / KV_COMPRESSION_RATIO[kv_level]
    if compressed_kv_per_token_mb <= 0:
        return 0
    return int(available_kv_mb / compressed_kv_per_token_mb)


def get_llama_kv_params(level: KVQuantLevel) -> dict:
    """Get llama.cpp parameters for the given KV compression level.

    Returns dict with cache_type_k and cache_type_v values to pass
    to the llama.cpp backend (via --cache-type-k / --cache-type-v).
    """
    mapping = {
        KVQuantLevel.NONE: {"cache_type_k": "f16", "cache_type_v": "f16"},
        KVQuantLevel.TURBO4: {"cache_type_k": "turbo4", "cache_type_v": "turbo4"},
        KVQuantLevel.TURBO3: {"cache_type_k": "turbo3", "cache_type_v": "turbo3"},
        KVQuantLevel.TURBO2: {"cache_type_k": "turbo2", "cache_type_v": "turbo2"},
    }
    return mapping.get(level, mapping[KVQuantLevel.NONE])


def get_vllm_kv_params(level: KVQuantLevel) -> dict:
    """Get vLLM parameters for the given KV compression level.

    Returns dict with kv_cache_dtype and related settings to pass
    to the vLLM backend (via --kv-cache-dtype / --enable-turboquant).
    """
    mapping = {
        KVQuantLevel.NONE: {},
        KVQuantLevel.TURBO4: {
            "kv_cache_dtype": "turboquant35",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
        KVQuantLevel.TURBO3: {
            "kv_cache_dtype": "turboquant35",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
        KVQuantLevel.TURBO2: {
            "kv_cache_dtype": "turboquant25",
            "enable_turboquant": True,
            "attention_backend": "TRITON_ATTN",
        },
    }
    return mapping.get(level, {})


def get_kv_quant_info(level: KVQuantLevel) -> dict:
    """Get human-readable info about a KV compression level."""
    info = {
        KVQuantLevel.NONE: {
            "bits_per_value": 8.5,
            "compression": "1x (no compression)",
            "quality": "Baseline",
            "description": "Default KV cache (q8_0/f16)",
        },
        KVQuantLevel.TURBO4: {
            "bits_per_value": 4.25,
            "compression": "3.8x",
            "quality": "Near-lossless (+0.23% PPL)",
            "description": "TurboQuant 4-bit: PolarQuant + QJL",
        },
        KVQuantLevel.TURBO3: {
            "bits_per_value": 3.5,
            "compression": "4.6x",
            "quality": "Minimal loss (+1.06% PPL)",
            "description": "TurboQuant 3-bit: PolarQuant + QJL",
        },
        KVQuantLevel.TURBO2: {
            "bits_per_value": 2.5,
            "compression": "6.4x",
            "quality": "Noticeable loss (+6.48% PPL)",
            "description": "TurboQuant 2-bit: PolarQuant (no QJL)",
        },
    }
    return info.get(level, info[KVQuantLevel.NONE])
