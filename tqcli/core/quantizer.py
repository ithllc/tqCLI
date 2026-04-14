"""Quantization selection engine for tqCLI.

Determines the optimal quantization method for a model based on available
hardware, then provides the correct vLLM/llama.cpp configuration parameters.

Supported methods:
  - bitsandbytes INT4 (NF4): On-the-fly quantization at load time for vLLM.
    Reduces BF16 models to ~25-30% of original size.  No pre-quantization needed.
  - AWQ: Pre-quantized checkpoints (loaded directly, no conversion needed).
  - GGUF k-quant: Pre-quantized llama.cpp files (loaded directly).
"""

from __future__ import annotations

from enum import Enum

from tqcli.core.model_registry import ModelProfile
from tqcli.core.system_info import SystemInfo


class QuantizationMethod(Enum):
    NONE = "none"           # BF16/FP16, no quantization
    BNB_INT4 = "bnb_int4"  # bitsandbytes 4-bit NF4 (on-the-fly)
    BNB_INT8 = "bnb_int8"  # bitsandbytes 8-bit (on-the-fly)
    AWQ = "awq"            # AutoAWQ (pre-quantized checkpoint)
    GPTQ = "gptq"          # AutoGPTQ (pre-quantized checkpoint)
    GGUF = "gguf"          # llama.cpp GGUF k-quant (pre-quantized)


# Approximate bytes per parameter for each format
_BYTES_PER_PARAM = {
    "BF16": 2.0,
    "FP16": 2.0,
    "FP32": 4.0,
    "INT8": 1.1,   # 8-bit + scales
    "INT4": 0.72,  # 4-bit NF4 + scales + zero-points + deq buffers (measured: 2.65 GiB for 4B)
    "AWQ": 0.6,    # 4-bit AWQ + scales (pre-quantized, no deq overhead)
    "Q4_K_M": 0.55,  # GGUF 4-bit k-quant medium
}

# Multimodal encoder overhead in MB (vision + audio for Gemma 4).
# Measured: Gemma 4 E2B loads at 9.89 GiB in BF16, but text-only params
# estimate to ~2.6 GB.  The ~7.3 GB delta is the multimodal stack.
_MULTIMODAL_OVERHEAD_MB = {
    "gemma4": 7000,   # SigLIP vision + audio encoder + VQ-VAE (measured ~7 GB BF16)
}

# vLLM runtime overhead in MB (activation buffers, NCCL, workspace)
_VLLM_RUNTIME_OVERHEAD_MB = 700


def estimate_bf16_model_size(model: ModelProfile) -> int:
    """Estimate the BF16 VRAM footprint of a model in MB.

    For models already quantized (AWQ, GGUF), estimates the quantized size.
    For BF16/FP16 models, estimates full-precision size.
    """
    param_b = _parse_param_count(model.parameter_count)

    if model.quantization in ("AWQ", "GPTQ"):
        bytes_per_param = _BYTES_PER_PARAM["AWQ"]
    elif model.quantization in ("Q4_K_M", "Q3_K_M", "Q4_K_S"):
        bytes_per_param = _BYTES_PER_PARAM["Q4_K_M"]
    elif model.quantization in ("INT8", "FP8"):
        bytes_per_param = _BYTES_PER_PARAM["INT8"]
    else:
        bytes_per_param = _BYTES_PER_PARAM["BF16"]

    weight_mb = int(param_b * bytes_per_param * 1024)  # GB to MB

    # Add multimodal encoder overhead
    if model.multimodal:
        weight_mb += _MULTIMODAL_OVERHEAD_MB.get(model.family, 500)

    return weight_mb


def estimate_quantized_size(model: ModelProfile, method: QuantizationMethod) -> int:
    """Estimate model VRAM after applying quantization, in MB."""
    param_b = _parse_param_count(model.parameter_count)

    if method == QuantizationMethod.BNB_INT4:
        bytes_per_param = _BYTES_PER_PARAM["INT4"]
    elif method == QuantizationMethod.BNB_INT8:
        bytes_per_param = _BYTES_PER_PARAM["INT8"]
    elif method == QuantizationMethod.AWQ:
        bytes_per_param = _BYTES_PER_PARAM["AWQ"]
    else:
        bytes_per_param = _BYTES_PER_PARAM["BF16"]

    weight_mb = int(param_b * bytes_per_param * 1024)

    if model.multimodal:
        # With bitsandbytes, encoders are also quantized to INT4
        base_overhead = _MULTIMODAL_OVERHEAD_MB.get(model.family, 500)
        if method in (QuantizationMethod.BNB_INT4, QuantizationMethod.AWQ):
            weight_mb += int(base_overhead * 0.35)  # ~35% of BF16 for quantized encoders
        else:
            weight_mb += base_overhead

    return weight_mb


def select_quantization(
    model: ModelProfile,
    sys_info: SystemInfo,
) -> QuantizationMethod | None:
    """Select the best quantization method for a model on the given hardware.

    Returns:
        QuantizationMethod if a viable method exists, None if the model
        cannot fit even after maximum quantization.
    """
    # If model is already quantized (AWQ, GGUF), no further quantization needed
    if model.quantization in ("AWQ", "GPTQ", "Q4_K_M", "Q3_K_M"):
        return QuantizationMethod.NONE

    total_vram_mb = sys_info.total_vram_mb

    # Calculate VRAM budget for model weights:
    # total - CUDA context overhead - vLLM runtime - minimal KV cache
    cuda_overhead = 810 if sys_info.is_wsl else 400
    available_for_model_mb = total_vram_mb - cuda_overhead - _VLLM_RUNTIME_OVERHEAD_MB - 50

    bf16_size = estimate_bf16_model_size(model)

    # Can the model fit at full BF16 precision?
    if bf16_size <= available_for_model_mb:
        return QuantizationMethod.NONE

    # Can the model fit with bitsandbytes INT4?
    int4_size = estimate_quantized_size(model, QuantizationMethod.BNB_INT4)
    if int4_size <= available_for_model_mb:
        return QuantizationMethod.BNB_INT4

    # Can the model fit with bitsandbytes INT8?
    int8_size = estimate_quantized_size(model, QuantizationMethod.BNB_INT8)
    if int8_size <= available_for_model_mb:
        return QuantizationMethod.BNB_INT8

    # Model too large even after maximum quantization
    return None


def get_vllm_quantization_params(method: QuantizationMethod) -> dict:
    """Get vLLM LLM() constructor kwargs for the given quantization method.

    These are passed directly to vLLM's LLM() class.
    """
    if method == QuantizationMethod.BNB_INT4:
        return {
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        }
    elif method == QuantizationMethod.BNB_INT8:
        return {
            "quantization": "bitsandbytes",
            "load_format": "bitsandbytes",
        }
    elif method == QuantizationMethod.AWQ:
        return {"quantization": "awq_marlin"}
    elif method == QuantizationMethod.GPTQ:
        return {"quantization": "gptq"}
    return {}


def _parse_param_count(param_str: str) -> float:
    """Parse '4.5B', '31B', etc. into float billions."""
    s = param_str.upper().replace("B", "").strip()
    try:
        return float(s)
    except ValueError:
        return 4.0
