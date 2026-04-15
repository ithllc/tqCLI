#!/usr/bin/env python3
"""Gemma 4 E2B vLLM Full Quantization Pipeline Diagnostic.

Traces every stage of the unified quantization pipeline for Gemma 4 E2B BF16
on vLLM to show exactly what happens and why:

  Stage 1: detect_model_precision() → "full_precision" (BF16 safetensors)
  Stage 2: estimate_bf16_model_size() → 11,710 MB (text + multimodal)
  Stage 3: select_quantization() → attempt BNB_INT4 → still too large → None
  Stage 4: plan_quantization_pipeline() → infeasible
  Stage 5: build_vllm_config() → rejected

This test documents the EXACT numbers at each stage so we can see why
Gemma 4 E2B cannot run on 4 GB VRAM via vLLM, and what hardware would work.

Output:
  tests/integration_reports/gemma4_vllm_pipeline_diagnostic.json
  tests/integration_reports/gemma4_vllm_pipeline_diagnostic.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.core.kv_quantizer import (
    KV_COMPRESSION_RATIO,
    KVQuantLevel,
    check_turboquant_compatibility,
    detect_model_precision,
    get_kv_quant_info,
    get_vllm_kv_params,
    plan_quantization_pipeline,
    select_kv_quant,
)
from tqcli.core.model_registry import BUILTIN_PROFILES
from tqcli.core.quantizer import (
    QuantizationMethod,
    _BYTES_PER_PARAM,
    _MULTIMODAL_OVERHEAD_MB,
    _VLLM_RUNTIME_OVERHEAD_MB,
    estimate_bf16_model_size,
    estimate_quantized_size,
    get_vllm_quantization_params,
    select_quantization,
)
from tqcli.core.system_info import detect_system
from tqcli.core.vllm_config import build_vllm_config

REPORT_DIR = Path(__file__).parent / "integration_reports"


def get_profile(model_id):
    for p in BUILTIN_PROFILES:
        if p.id == model_id:
            return p
    return None


def run_diagnostic():
    sys_info = detect_system()
    profile = get_profile("gemma-4-e2b-it-vllm")
    if not profile:
        print("ERROR: gemma-4-e2b-it-vllm not found in BUILTIN_PROFILES")
        return

    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_type": "gemma4_vllm_pipeline_diagnostic",
        "purpose": (
            "Traces every stage of the unified quantization pipeline for "
            "Gemma 4 E2B BF16 on vLLM to document exactly what happens "
            "and why it is infeasible on 4 GB VRAM."
        ),
    }

    # ── System Info ──────────────────────────────────────────────────
    results["system"] = {
        "os": sys_info.os_display,
        "gpu": sys_info.gpus[0].name if sys_info.gpus else "None",
        "vram_mb": sys_info.total_vram_mb,
        "cuda_version": sys_info.gpus[0].cuda_version if sys_info.gpus else "N/A",
        "cuda_toolkit": sys_info.gpus[0].cuda_toolkit_version if sys_info.gpus else "N/A",
        "compute_capability": sys_info.gpus[0].compute_capability if sys_info.gpus else "N/A",
        "is_wsl": sys_info.is_wsl,
    }

    # ── Model Profile ────────────────────────────────────────────────
    results["model_profile"] = {
        "id": profile.id,
        "display_name": profile.display_name,
        "family": profile.family,
        "parameter_count": profile.parameter_count,
        "quantization": profile.quantization,
        "format": profile.format,
        "engine": profile.engine,
        "multimodal": profile.multimodal,
        "supports_thinking": profile.supports_thinking,
        "hf_repo": profile.hf_repo,
        "min_vram_mb": profile.min_vram_mb,
    }

    # ── Stage 1: Detect Model Precision ──────────────────────────────
    precision = detect_model_precision(profile)
    results["stage_1_detect_precision"] = {
        "description": "Detect whether model is full-precision or already weight-quantized",
        "input": {
            "quantization_field": profile.quantization,
            "format_field": profile.format,
        },
        "output": precision,
        "explanation": (
            f"Model quantization='{profile.quantization}', format='{profile.format}' "
            f"→ detected as '{precision}'. BF16 safetensors are full-precision, "
            f"meaning weight quantization (Stage 2) WILL be attempted."
        ),
    }

    # ── Stage 2: Estimate Model Sizes ────────────────────────────────
    param_b = float(profile.parameter_count.upper().replace("B", "").strip())
    bf16_size = estimate_bf16_model_size(profile)
    int4_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT4)
    int8_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT8)

    text_bf16_mb = int(param_b * _BYTES_PER_PARAM["BF16"] * 1024)
    text_int4_mb = int(param_b * _BYTES_PER_PARAM["INT4"] * 1024)
    mm_overhead_bf16 = _MULTIMODAL_OVERHEAD_MB.get(profile.family, 500)
    mm_overhead_int4 = int(mm_overhead_bf16 * 0.35)

    results["stage_2_size_estimates"] = {
        "description": "Estimate VRAM footprint at each quantization level",
        "parameter_count_billions": param_b,
        "multimodal": profile.multimodal,
        "breakdown": {
            "bf16": {
                "text_weights_mb": text_bf16_mb,
                "multimodal_overhead_mb": mm_overhead_bf16,
                "total_mb": bf16_size,
                "note": f"{param_b}B params × 2.0 bytes/param × 1024 = {text_bf16_mb} MB text + {mm_overhead_bf16} MB multimodal",
            },
            "bnb_int4": {
                "text_weights_mb": text_int4_mb,
                "multimodal_overhead_mb": mm_overhead_int4,
                "total_mb": int4_size,
                "note": f"{param_b}B params × 0.72 bytes/param × 1024 = {text_int4_mb} MB text + {mm_overhead_int4} MB multimodal (35% of BF16)",
            },
            "bnb_int8": {
                "text_weights_mb": int(param_b * _BYTES_PER_PARAM["INT8"] * 1024),
                "multimodal_overhead_mb": mm_overhead_bf16,
                "total_mb": int8_size,
                "note": "INT8 does not reduce multimodal overhead",
            },
        },
        "key_insight": (
            f"The multimodal encoders (SigLIP vision + audio + VQ-VAE) add "
            f"{mm_overhead_bf16} MB in BF16, reduced to {mm_overhead_int4} MB with INT4. "
            f"Text-only INT4 would be just {text_int4_mb} MB — small enough for 4 GB VRAM. "
            f"The multimodal overhead is what makes it infeasible."
        ),
    }

    # ── Stage 3: Select Weight Quantization ──────────────────────────
    cuda_overhead = 810 if sys_info.is_wsl else 400
    vllm_runtime = _VLLM_RUNTIME_OVERHEAD_MB
    available_for_model = sys_info.total_vram_mb - cuda_overhead - vllm_runtime - 50

    quant_method = select_quantization(profile, sys_info)
    results["stage_3_select_quantization"] = {
        "description": "Select best weight quantization method for available VRAM",
        "vram_budget": {
            "total_vram_mb": sys_info.total_vram_mb,
            "cuda_overhead_mb": cuda_overhead,
            "vllm_runtime_mb": vllm_runtime,
            "min_kv_cache_mb": 50,
            "available_for_model_mb": available_for_model,
        },
        "checks": [
            {
                "method": "BF16 (no quantization)",
                "size_mb": bf16_size,
                "fits": bf16_size <= available_for_model,
                "verdict": f"{bf16_size} MB > {available_for_model} MB → DOES NOT FIT",
            },
            {
                "method": "BNB_INT4",
                "size_mb": int4_size,
                "fits": int4_size <= available_for_model,
                "verdict": f"{int4_size} MB > {available_for_model} MB → DOES NOT FIT"
                if int4_size > available_for_model
                else f"{int4_size} MB <= {available_for_model} MB → FITS",
            },
            {
                "method": "BNB_INT8",
                "size_mb": int8_size,
                "fits": int8_size <= available_for_model,
                "verdict": f"{int8_size} MB > {available_for_model} MB → DOES NOT FIT",
            },
        ],
        "selected_method": str(quant_method) if quant_method else "None (model too large)",
        "explanation": (
            f"Even with INT4 quantization ({int4_size} MB), the model exceeds "
            f"the {available_for_model} MB budget. The {mm_overhead_int4} MB "
            f"multimodal encoder overhead (35% of {mm_overhead_bf16} MB BF16) "
            f"is the primary blocker."
        ),
    }

    # ── Stage 4: Unified Pipeline Decision ───────────────────────────
    pipeline = plan_quantization_pipeline(
        model=profile,
        sys_info=sys_info,
        kv_quant_choice="turbo3",
        engine="vllm",
    )
    results["stage_4_pipeline_decision"] = {
        "description": "Unified quantization pipeline decision (weight quant + KV compression)",
        "model_precision": pipeline.model_precision,
        "needs_weight_quant": pipeline.needs_weight_quant,
        "weight_quant_method": pipeline.weight_quant_method,
        "weight_quant_reason": pipeline.weight_quant_reason,
        "needs_kv_compression": pipeline.needs_kv_compression,
        "kv_level": pipeline.kv_level.value,
        "kv_reason": pipeline.kv_reason,
        "stages_applied": pipeline.stages_applied,
        "summary": pipeline.summary,
        "explanation": (
            f"Pipeline detected '{pipeline.model_precision}' → attempted weight quantization "
            f"but no method fits ({pipeline.weight_quant_reason}). "
            f"KV compression was still planned ({pipeline.kv_reason}) but cannot be applied "
            f"if the model can't load."
        ),
    }

    # ── Stage 5: vLLM Config Builder ─────────────────────────────────
    vllm_config = build_vllm_config(
        profile, sys_info,
        requested_max_len=2048,
        kv_quant_choice="turbo3",
    )
    results["stage_5_vllm_config"] = {
        "description": "Hardware-aware vLLM configuration builder result",
        "feasible": vllm_config.feasible,
        "reason": vllm_config.reason,
        "gpu_memory_utilization": vllm_config.gpu_memory_utilization,
        "max_model_len": vllm_config.max_model_len,
        "quantization": vllm_config.quantization,
        "kv_cache_dtype": vllm_config.kv_cache_dtype,
        "enforce_eager": vllm_config.enforce_eager,
        "estimated_model_size_mb": vllm_config.estimated_model_size_mb,
        "warnings": vllm_config.warnings,
        "explanation": (
            f"build_vllm_config() returned feasible={vllm_config.feasible}. "
            f"Reason: {vllm_config.reason}"
        ),
    }

    # ── TurboQuant KV Compatibility ──────────────────────────────────
    tq_available, tq_msg = check_turboquant_compatibility(sys_info)
    results["turboquant_kv_status"] = {
        "available": tq_available,
        "message": tq_msg,
        "note": (
            "TurboQuant KV compression IS available on this system, but cannot "
            "be used because the model weights themselves don't fit in VRAM. "
            "KV compression only reduces the KV cache memory, not model weight memory."
        ),
    }

    # ── Minimum Hardware Analysis ────────────────────────────────────
    def calc_min_vram(model_mb, wsl=True):
        overhead = 810 if wsl else 400
        return model_mb + overhead + 700 + 50  # model + CUDA + runtime + min KV

    min_vram_int4 = calc_min_vram(int4_size)
    min_vram_bf16 = calc_min_vram(bf16_size)
    min_vram_int4_text_only = calc_min_vram(text_int4_mb)

    results["minimum_hardware_analysis"] = {
        "description": "Minimum VRAM required for Gemma 4 E2B on vLLM",
        "scenarios": [
            {
                "scenario": "INT4 text-only (no multimodal encoders)",
                "model_size_mb": text_int4_mb,
                "min_vram_mb_wsl": calc_min_vram(text_int4_mb, wsl=True),
                "min_vram_mb_linux": calc_min_vram(text_int4_mb, wsl=False),
                "fits_4gb": calc_min_vram(text_int4_mb, wsl=True) <= 4096,
                "min_gpu": "RTX 3050 4GB (Linux, not WSL2)" if calc_min_vram(text_int4_mb, wsl=False) <= 4096 else "RTX 2060 6GB",
            },
            {
                "scenario": "INT4 with multimodal (current estimate)",
                "model_size_mb": int4_size,
                "min_vram_mb_wsl": calc_min_vram(int4_size, wsl=True),
                "min_vram_mb_linux": calc_min_vram(int4_size, wsl=False),
                "fits_4gb": False,
                "min_gpu": "RTX 3060 6GB or RTX 4060 8GB",
            },
            {
                "scenario": "BF16 full precision with multimodal",
                "model_size_mb": bf16_size,
                "min_vram_mb_wsl": calc_min_vram(bf16_size, wsl=True),
                "min_vram_mb_linux": calc_min_vram(bf16_size, wsl=False),
                "fits_4gb": False,
                "min_gpu": "RTX 4090 24GB or A100 40GB",
            },
        ],
        "your_system": {
            "gpu": sys_info.gpus[0].name if sys_info.gpus else "None",
            "vram_mb": sys_info.total_vram_mb,
            "verdict": (
                f"RTX A2000 (4 GB) cannot run Gemma 4 E2B on vLLM because "
                f"even with INT4 quantization, the multimodal encoders push "
                f"the total to {int4_size} MB, exceeding the {available_for_model} MB budget. "
                f"Text-only INT4 ({text_int4_mb} MB) would fit on native Linux "
                f"({calc_min_vram(text_int4_mb, wsl=False)} MB needed) but not WSL2 "
                f"({calc_min_vram(text_int4_mb, wsl=True)} MB needed > 4096 MB)."
            ),
        },
        "recommendation": (
            "For Gemma 4 E2B on your hardware, use llama.cpp with the GGUF Q4_K_M model "
            "(2,890 MB on disk, loads with ~2.8 GB VRAM + turbo3 KV). "
            "For vLLM Gemma 4 E2B, minimum 6 GB VRAM (RTX 3060) is required."
        ),
    }

    # ── Comparison: What Works on This Hardware ──────────────────────
    results["what_works_on_4gb"] = {
        "llama_cpp": {
            "gemma_4_e2b_Q4_K_M": {
                "status": "WORKS",
                "size_mb": 2890,
                "kv_compression": "turbo3 (4.6x)",
                "tested": True,
                "tok_s": "2-4",
            },
            "qwen3_4b_Q4_K_M": {
                "status": "WORKS",
                "size_mb": 2382,
                "kv_compression": "turbo3 (4.6x)",
                "tested": True,
                "tok_s": "6-9",
            },
        },
        "vllm": {
            "qwen3_4b_AWQ": {
                "status": "WORKS",
                "size_mb": 2558,
                "kv_compression": "turboquant35",
                "tested": True,
                "tok_s": "0.5-1.0",
            },
            "gemma_4_e2b_BF16_INT4": {
                "status": "DOES NOT FIT",
                "estimated_int4_mb": int4_size,
                "available_mb": available_for_model,
                "blocker": f"Multimodal encoders add {mm_overhead_int4} MB even with INT4",
                "min_vram_needed": f"{min_vram_int4} MB (WSL2)",
            },
        },
    }

    return results


def generate_markdown(results: dict) -> str:
    lines = []
    lines.append("# Gemma 4 E2B vLLM Full Quantization Pipeline Diagnostic")
    lines.append("")
    lines.append(f"**Generated:** {results['generated']}")
    lines.append(f"**Purpose:** {results['purpose']}")
    lines.append("")

    # System
    s = results["system"]
    lines.append("## System")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    for k, v in s.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Model Profile
    m = results["model_profile"]
    lines.append("## Model Profile")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    for k, v in m.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Stage 1
    st1 = results["stage_1_detect_precision"]
    lines.append("## Stage 1: Detect Model Precision")
    lines.append(f"**Result:** `{st1['output']}`")
    lines.append(f"\n{st1['explanation']}")
    lines.append("")

    # Stage 2
    st2 = results["stage_2_size_estimates"]
    lines.append("## Stage 2: Estimate Model Sizes")
    lines.append("")
    lines.append("| Level | Text Weights | Multimodal Overhead | Total |")
    lines.append("|-------|-------------|--------------------:|------:|")
    for level, data in st2["breakdown"].items():
        lines.append(
            f"| {level} | {data['text_weights_mb']} MB | "
            f"{data['multimodal_overhead_mb']} MB | "
            f"**{data['total_mb']} MB** |"
        )
    lines.append("")
    lines.append(f"> **Key Insight:** {st2['key_insight']}")
    lines.append("")

    # Stage 3
    st3 = results["stage_3_select_quantization"]
    lines.append("## Stage 3: Select Weight Quantization")
    lines.append("")
    lines.append(f"**VRAM Budget:** {st3['vram_budget']['total_vram_mb']} MB total "
                 f"- {st3['vram_budget']['cuda_overhead_mb']} MB CUDA "
                 f"- {st3['vram_budget']['vllm_runtime_mb']} MB runtime "
                 f"- {st3['vram_budget']['min_kv_cache_mb']} MB KV "
                 f"= **{st3['vram_budget']['available_for_model_mb']} MB available**")
    lines.append("")
    lines.append("| Method | Size | Fits? | Verdict |")
    lines.append("|--------|-----:|:-----:|---------|")
    for check in st3["checks"]:
        fits = "YES" if check["fits"] else "NO"
        lines.append(f"| {check['method']} | {check['size_mb']} MB | {fits} | {check['verdict']} |")
    lines.append("")
    lines.append(f"**Selected:** `{st3['selected_method']}`")
    lines.append(f"\n{st3['explanation']}")
    lines.append("")

    # Stage 4
    st4 = results["stage_4_pipeline_decision"]
    lines.append("## Stage 4: Unified Pipeline Decision")
    lines.append("")
    lines.append(f"| Property | Value |")
    lines.append(f"|----------|-------|")
    lines.append(f"| Model Precision | {st4['model_precision']} |")
    lines.append(f"| Needs Weight Quant | {st4['needs_weight_quant']} |")
    lines.append(f"| Weight Method | {st4['weight_quant_method'] or 'N/A (too large)'} |")
    lines.append(f"| Needs KV Compression | {st4['needs_kv_compression']} |")
    lines.append(f"| KV Level | {st4['kv_level']} |")
    lines.append(f"| Stages Applied | {st4['stages_applied'] or 'None'} |")
    lines.append(f"| Summary | {st4['summary']} |")
    lines.append("")
    lines.append(f"{st4['explanation']}")
    lines.append("")

    # Stage 5
    st5 = results["stage_5_vllm_config"]
    lines.append("## Stage 5: vLLM Config Builder")
    lines.append("")
    lines.append(f"**Feasible:** `{st5['feasible']}`")
    lines.append(f"**Reason:** {st5['reason']}")
    lines.append("")

    # TurboQuant
    tq = results["turboquant_kv_status"]
    lines.append("## TurboQuant KV Status")
    lines.append(f"**Available:** {tq['available']}")
    lines.append(f"\n> {tq['note']}")
    lines.append("")

    # Minimum Hardware
    hw = results["minimum_hardware_analysis"]
    lines.append("## Minimum Hardware for Gemma 4 E2B on vLLM")
    lines.append("")
    lines.append("| Scenario | Model Size | Min VRAM (WSL2) | Min VRAM (Linux) | Fits 4 GB? | Min GPU |")
    lines.append("|----------|-----------|----------------|-----------------|:----------:|---------|")
    for sc in hw["scenarios"]:
        fits = "YES" if sc["fits_4gb"] else "NO"
        lines.append(
            f"| {sc['scenario']} | {sc['model_size_mb']} MB | "
            f"{sc['min_vram_mb_wsl']} MB | {sc['min_vram_mb_linux']} MB | "
            f"{fits} | {sc['min_gpu']} |"
        )
    lines.append("")
    lines.append(f"**Your System:** {hw['your_system']['verdict']}")
    lines.append("")
    lines.append(f"**Recommendation:** {hw['recommendation']}")
    lines.append("")

    # What works
    ww = results["what_works_on_4gb"]
    lines.append("## What Works on 4 GB VRAM")
    lines.append("")
    lines.append("### llama.cpp (TurboQuant fork)")
    lines.append("| Model | Status | Size | KV | tok/s |")
    lines.append("|-------|--------|-----:|:--:|:-----:|")
    for model_id, data in ww["llama_cpp"].items():
        lines.append(f"| {model_id} | {data['status']} | {data['size_mb']} MB | {data['kv_compression']} | {data['tok_s']} |")
    lines.append("")
    lines.append("### vLLM (TurboQuant fork)")
    lines.append("| Model | Status | Details |")
    lines.append("|-------|--------|---------|")
    for model_id, data in ww["vllm"].items():
        if data["status"] == "WORKS":
            lines.append(f"| {model_id} | {data['status']} | {data['size_mb']} MB, {data['kv_compression']}, {data['tok_s']} tok/s |")
        else:
            lines.append(f"| {model_id} | {data['status']} | {data['blocker']} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEMMA 4 E2B vLLM PIPELINE DIAGNOSTIC")
    print("=" * 60)

    results = run_diagnostic()

    # Write JSON
    json_path = REPORT_DIR / "gemma4_vllm_pipeline_diagnostic.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"JSON: {json_path}")

    # Write Markdown
    md_path = REPORT_DIR / "gemma4_vllm_pipeline_diagnostic.md"
    md_path.write_text(generate_markdown(results))
    print(f"MD:   {md_path}")

    # Print summary
    print()
    for stage_key in sorted(k for k in results if k.startswith("stage_")):
        stage = results[stage_key]
        desc = stage.get("description", "")
        print(f"  {stage_key}: {desc}")
        if "output" in stage:
            print(f"    → {stage['output']}")
        if "selected_method" in stage:
            print(f"    → {stage['selected_method']}")
        if "feasible" in stage:
            print(f"    → feasible={stage['feasible']}: {stage.get('reason', '')}")
        if "summary" in stage:
            print(f"    → {stage['summary']}")

    print()
    hw = results["minimum_hardware_analysis"]
    print(f"  Your system: {hw['your_system']['gpu']} ({hw['your_system']['vram_mb']} MB)")
    print(f"  Verdict: {hw['your_system']['verdict'][:120]}...")
    print(f"  Recommendation: {hw['recommendation'][:120]}...")
