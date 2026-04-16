"""Hardware-aware vLLM configuration builder.

Automatically tunes vLLM parameters based on detected GPU VRAM, model size,
and model capabilities.  Uses the quantizer module to select bitsandbytes
INT4 quantization when BF16 models exceed available VRAM.

References:
  - vLLM Gemma 4 recipe: docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
  - TurboQuant methodology: GoogleTurboQuant workspace
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tqcli.core.model_registry import ModelProfile
from tqcli.core.quantizer import (
    QuantizationMethod,
    estimate_bf16_model_size,
    estimate_quantized_size,
    get_vllm_quantization_params,
    select_quantization,
)
from tqcli.core.system_info import SystemInfo


@dataclass
class VllmTuningProfile:
    """Tuned vLLM parameters for a specific model + hardware combination."""

    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    enforce_eager: bool = False
    quantization: str | None = None
    load_format: str | None = None
    kv_cache_dtype: str = "auto"
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    quantization_method: QuantizationMethod = QuantizationMethod.NONE
    estimated_model_size_mb: int = 0
    # Multimodal settings (Gemma 4)
    mm_processor_kwargs: dict = field(default_factory=dict)
    limit_mm_per_prompt: dict = field(default_factory=dict)
    # Gemma 4 thinking mode
    reasoning_parser: str = ""
    # CPU offloading: spill excess model weight into system RAM
    cpu_offload_gb: float = 0
    # Manual KV cache memory (bypasses vLLM profiler, needed for BNB+offload)
    kv_cache_memory_bytes: int | None = None
    # Max tokens in a single batch (limits profiling memory)
    max_num_batched_tokens: int | None = None
    warnings: list[str] = field(default_factory=list)
    feasible: bool = True
    reason: str = ""


# CUDA context overhead per GPU in MB.
_CUDA_CONTEXT_OVERHEAD_MB = 400

# Minimum KV cache memory in MB
_MIN_KV_CACHE_MB = 30

# vLLM runtime overhead (activations, NCCL, workspace)
_VLLM_RUNTIME_OVERHEAD_MB = 700


def build_vllm_config(
    model: ModelProfile,
    sys_info: SystemInfo,
    requested_max_len: int | None = None,
    kv_quant_choice: str | None = None,
) -> VllmTuningProfile:
    """Build a hardware-tuned vLLM configuration for the given model.

    Automatically selects bitsandbytes INT4 quantization when the BF16
    model exceeds available VRAM.
    """
    profile = VllmTuningProfile()
    total_vram_mb = sys_info.total_vram_mb
    warnings = []

    if total_vram_mb <= 0:
        profile.feasible = False
        profile.reason = "No GPU detected; vLLM requires NVIDIA GPU"
        return profile

    # ── Step 1: Select quantization method ────────────────────────────
    quant_method = select_quantization(model, sys_info)

    if quant_method is None:
        # Model too large for VRAM even with INT4 — try CPU offloading
        # Use INT4 quantization + offload the excess to system RAM
        available_ram_mb = sys_info.ram_available_mb
        int4_size = estimate_quantized_size(model, QuantizationMethod.BNB_INT4)
        cuda_overhead = 810 if sys_info.is_wsl else 400
        available_for_model = total_vram_mb - cuda_overhead - 700 - 50
        excess_mb = int4_size - available_for_model

        if excess_mb > 0 and available_ram_mb > excess_mb + 2000:
            # We have enough system RAM to offload the excess.
            # The offloader measures bf16 param sizes at init time (before
            # BNB quantization), so we need to compute the offload budget
            # in bf16 terms. Offload aggressively: leave only ~1.5 GiB
            # worth of bf16 params on GPU (which becomes ~0.5 GiB INT4).
            quant_method = QuantizationMethod.BNB_INT4
            bf16_size = estimate_bf16_model_size(model)
            bf16_to_keep_gb = min(1.5, bf16_size / 1024)
            offload_gb = round(max((bf16_size / 1024) - bf16_to_keep_gb, 1.0), 1)
            profile.cpu_offload_gb = offload_gb
            # vLLM's memory profiler overestimates peak usage with BNB +
            # CPU offload (non-UVA path). Bypass with manual KV cache size.
            kv_cache_mb = min(max(available_for_model // 4, 32), 128)
            profile.kv_cache_memory_bytes = kv_cache_mb * 1024 * 1024
            # Limit profiling batch size to reduce peak activation memory
            profile.max_num_batched_tokens = 256
            warnings.append(
                f"Model exceeds VRAM (INT4={int4_size} MB > {available_for_model} MB available). "
                f"CPU offloading {offload_gb} GB to system RAM ({available_ram_mb} MB available)."
            )
        else:
            bf16_size = estimate_bf16_model_size(model)
            profile.feasible = False
            profile.reason = (
                f"Model too large even after INT4 quantization: "
                f"BF16={bf16_size} MB, VRAM={total_vram_mb} MB"
            )
            return profile

    profile.quantization_method = quant_method

    # ── Step 2: Calculate model size after quantization ───────────────
    if quant_method == QuantizationMethod.NONE:
        model_weight_mb = estimate_bf16_model_size(model)
        # For already-quantized models (AWQ), use their native quant params
        if model.quantization == "AWQ":
            profile.quantization = "awq_marlin"
        elif model.quantization in ("GPTQ", "FP8"):
            profile.quantization = model.quantization.lower()
    else:
        model_weight_mb = estimate_quantized_size(model, quant_method)
        vllm_params = get_vllm_quantization_params(quant_method)
        profile.quantization = vllm_params.get("quantization")
        profile.load_format = vllm_params.get("load_format")
        bf16_size = estimate_bf16_model_size(model)
        warnings.append(
            f"BF16 model ({bf16_size} MB) will be quantized to {quant_method.value} "
            f"(~{model_weight_mb} MB) via bitsandbytes"
        )

    profile.estimated_model_size_mb = model_weight_mb

    # ── Step 3: Calculate usable VRAM ─────────────────────────────────
    overhead_mb = _CUDA_CONTEXT_OVERHEAD_MB
    if sys_info.is_wsl:
        overhead_mb = 810
    usable_vram_mb = total_vram_mb - overhead_mb

    if usable_vram_mb <= 0:
        profile.feasible = False
        profile.reason = f"Insufficient VRAM: {total_vram_mb} MB total, ~{overhead_mb} MB CUDA overhead"
        return profile

    max_util = usable_vram_mb / total_vram_mb
    profile.gpu_memory_utilization = min(round(max_util, 2), 0.95)

    # ── Step 4: enforce_eager on small GPUs ───────────────────────────
    if total_vram_mb < 8000:
        profile.enforce_eager = True
        warnings.append("enforce_eager=True (saves ~1 GB VRAM on small GPUs)")

    # ── Step 5: Available KV cache memory ─────────────────────────────
    model_in_vram_mb = model_weight_mb
    if profile.cpu_offload_gb > 0:
        # With CPU offloading, only part of the model stays in VRAM
        model_in_vram_mb = max(model_weight_mb - int(profile.cpu_offload_gb * 1024), 0)
    model_loaded_mb = model_in_vram_mb + _VLLM_RUNTIME_OVERHEAD_MB
    if not profile.enforce_eager:
        model_loaded_mb += 500  # torch.compile overhead
    available_for_kv_mb = (profile.gpu_memory_utilization * total_vram_mb) - model_loaded_mb

    if available_for_kv_mb < _MIN_KV_CACHE_MB:
        profile.feasible = False
        profile.reason = (
            f"Insufficient VRAM for KV cache: model+overhead needs ~{model_loaded_mb:.0f} MB, "
            f"only {profile.gpu_memory_utilization * total_vram_mb:.0f} MB usable"
        )
        return profile

    # ── Step 6: KV cache compression ────────────────────────────────
    # Try TurboQuant KV first (much better than FP8), then fall back to FP8
    gpu_cc = ""
    if sys_info.gpus:
        gpu_cc = sys_info.gpus[0].compute_capability
    cc_major = 0
    if gpu_cc:
        try:
            cc_major = int(gpu_cc.split(".")[0])
        except (ValueError, IndexError):
            pass

    # Check TurboQuant KV availability
    from tqcli.core.kv_quantizer import (
        KVQuantLevel,
        check_turboquant_compatibility,
        get_vllm_kv_params,
        select_kv_quant,
    )

    tq_available, _tq_msg = check_turboquant_compatibility(sys_info)
    kv_quant = kv_quant_choice if kv_quant_choice else "auto"

    if kv_quant != "none" and tq_available:
        kv_level = select_kv_quant(
            available_kv_mb=available_for_kv_mb,
            engine="vllm",
            user_choice=kv_quant,
        )
        if kv_level != KVQuantLevel.NONE:
            vllm_kv_params = get_vllm_kv_params(kv_level)
            if vllm_kv_params.get("kv_cache_dtype"):
                profile.kv_cache_dtype = vllm_kv_params["kv_cache_dtype"]
                # TurboQuant KV gives ~4-6x compression
                from tqcli.core.kv_quantizer import KV_COMPRESSION_RATIO
                compression = KV_COMPRESSION_RATIO.get(kv_level, 1.0)
                available_for_kv_mb *= compression
                warnings.append(
                    f"kv_cache_dtype={profile.kv_cache_dtype} "
                    f"(TurboQuant {kv_level.value}: {compression}x KV compression)"
                )
    elif cc_major >= 9 and available_for_kv_mb < 500:
        # Fall back to FP8 KV on Hopper+ if TurboQuant unavailable
        profile.kv_cache_dtype = "fp8"
        available_for_kv_mb *= 2
        warnings.append("kv_cache_dtype=fp8 (doubles KV cache capacity)")

    # ── Step 7: max_model_len ─────────────────────────────────────────
    param_billions = _parse_param_count(model.parameter_count)
    kv_per_token_mb = 0.14 * param_billions
    if profile.kv_cache_dtype == "fp8":
        kv_per_token_mb *= 0.5

    max_tokens_from_kv = int(available_for_kv_mb / max(kv_per_token_mb, 0.01))
    max_tokens_from_kv = max(max_tokens_from_kv, 128)

    if requested_max_len:
        profile.max_model_len = min(requested_max_len, max_tokens_from_kv, model.context_length)
    else:
        profile.max_model_len = min(max_tokens_from_kv, model.context_length, 8192)

    if profile.max_model_len < 256:
        warnings.append(f"Very short context ({profile.max_model_len} tokens) due to limited VRAM")

    # ── Step 8: Multimodal (Gemma 4) ──────────────────────────────────
    if model.multimodal and model.family == "gemma4":
        if total_vram_mb < 12000:
            vision_budget = 70
        elif total_vram_mb < 24000:
            vision_budget = 280
        else:
            vision_budget = 560

        profile.mm_processor_kwargs = {"max_soft_tokens": vision_budget}
        profile.limit_mm_per_prompt = {"image": 2, "audio": 1}
        if total_vram_mb < 8000:
            profile.limit_mm_per_prompt = {"image": 1, "audio": 0}
            warnings.append("Multimodal limited to 1 image, no audio (low VRAM)")

    # ── Step 9: Thinking mode ─────────────────────────────────────────
    if model.supports_thinking and model.family == "gemma4":
        profile.reasoning_parser = "gemma4"

    profile.warnings = warnings
    return profile


def _parse_param_count(param_str: str) -> float:
    """Parse '4.5B', '31B', etc. into float billions."""
    s = param_str.upper().replace("B", "").strip()
    try:
        return float(s)
    except ValueError:
        return 4.0
