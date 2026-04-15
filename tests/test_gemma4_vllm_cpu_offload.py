#!/usr/bin/env python3
"""Gemma 4 E2B vLLM + CPU Offloading + BNB_INT4 + TurboQuant KV Test.

Tests the full unified quantization pipeline with CPU offloading:
  detect full_precision → weight:bnb_int4 → cpu_offload 2.1 GB → kv:turboquant35

This proves that Gemma 4 E2B CAN run on 4 GB VRAM via vLLM by spilling
excess model weights (multimodal encoders) to system RAM.

Output:
  tests/integration_reports/gemma4_vllm_cpu_offload_report.json
  tests/integration_reports/gemma4_vllm_cpu_offload_report.md
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig
from tqcli.core.engine import ChatMessage
from tqcli.core.kv_quantizer import (
    check_turboquant_compatibility,
    detect_model_precision,
    plan_quantization_pipeline,
)
from tqcli.core.model_registry import BUILTIN_PROFILES
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.quantizer import (
    QuantizationMethod,
    estimate_bf16_model_size,
    estimate_quantized_size,
    select_quantization,
)
from tqcli.core.system_info import detect_system
from tqcli.core.thinking import (
    ThinkingConfig,
    ThinkingFormat,
    build_system_prompt_with_thinking,
    extract_thinking,
)
from tqcli.core.vllm_config import build_vllm_config

REPORT_DIR = Path(__file__).parent / "integration_reports"


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


def get_system_info_dict():
    info = detect_system()
    return {
        "os": info.os_display,
        "gpu": info.gpus[0].name if info.gpus else "None",
        "vram_mb": info.total_vram_mb,
        "ram_total_mb": info.ram_total_mb,
        "ram_available_mb": info.ram_available_mb,
        "cuda_toolkit": info.gpus[0].cuda_toolkit_version if info.gpus else "N/A",
        "compute_capability": info.gpus[0].compute_capability if info.gpus else "N/A",
        "is_wsl": info.is_wsl,
    }


def run_test():
    results = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "test_name": "Gemma 4 E2B vLLM + CPU Offloading + BNB_INT4 + TurboQuant KV",
        "system": get_system_info_dict(),
    }

    sys_info = detect_system()
    config = TqConfig.load()
    config.ensure_dirs()
    steps = []

    # ── Find model profile ───────────────────────────────────────────
    profile = None
    for p in BUILTIN_PROFILES:
        if p.id == "gemma-4-e2b-it-vllm":
            profile = p
            break

    if not profile:
        steps.append(StepResult(name="find_model", passed=False, details="Profile not found"))
        results["steps"] = [s.__dict__ for s in steps]
        return results

    steps.append(StepResult(
        name="find_model_profile",
        passed=True,
        details=f"Found {profile.id}: {profile.display_name} ({profile.parameter_count}, {profile.quantization})",
        metrics={
            "model_id": profile.id,
            "parameter_count": profile.parameter_count,
            "quantization": profile.quantization,
            "format": profile.format,
            "multimodal": profile.multimodal,
        },
    ))

    # ── Step 1: Detect precision ─────────────────────────────────────
    precision = detect_model_precision(profile)
    steps.append(StepResult(
        name="detect_precision",
        passed=precision == "full_precision",
        details=f"Detected: {precision} (quant={profile.quantization}, format={profile.format})",
        metrics={"precision": precision},
    ))

    # ── Step 2: Size estimates ───────────────────────────────────────
    bf16_size = estimate_bf16_model_size(profile)
    int4_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT4)
    steps.append(StepResult(
        name="size_estimates",
        passed=True,
        details=f"BF16={bf16_size} MB, INT4={int4_size} MB, VRAM={sys_info.total_vram_mb} MB, RAM={sys_info.ram_available_mb} MB",
        metrics={
            "bf16_mb": bf16_size,
            "int4_mb": int4_size,
            "vram_mb": sys_info.total_vram_mb,
            "ram_available_mb": sys_info.ram_available_mb,
        },
    ))

    # ── Step 3: Select quantization (should return None without offload) ──
    quant_method = select_quantization(profile, sys_info)
    steps.append(StepResult(
        name="select_quantization_without_offload",
        passed=quant_method is None,
        details=f"select_quantization() returned: {quant_method} (expected None — too large for VRAM alone)",
        metrics={"method": str(quant_method), "expected": "None"},
    ))

    # ── Step 4: Build vLLM config WITH CPU offloading ────────────────
    start = time.time()
    tune = build_vllm_config(
        profile, sys_info,
        requested_max_len=2048,
        kv_quant_choice="turbo3",
    )
    elapsed = time.time() - start

    steps.append(StepResult(
        name="build_vllm_config_with_offload",
        passed=tune.feasible,
        duration_s=elapsed,
        details=(
            f"feasible={tune.feasible} | cpu_offload_gb={tune.cpu_offload_gb} | "
            f"quantization={tune.quantization} | kv_cache_dtype={tune.kv_cache_dtype} | "
            f"max_model_len={tune.max_model_len}"
        ),
        metrics={
            "feasible": tune.feasible,
            "cpu_offload_gb": tune.cpu_offload_gb,
            "quantization": tune.quantization,
            "load_format": tune.load_format,
            "kv_cache_dtype": tune.kv_cache_dtype,
            "max_model_len": tune.max_model_len,
            "gpu_memory_utilization": tune.gpu_memory_utilization,
            "enforce_eager": tune.enforce_eager,
            "estimated_model_size_mb": tune.estimated_model_size_mb,
            "warnings": tune.warnings,
            "reason": tune.reason,
        },
    ))

    results["pipeline_config"] = {
        "weight_quantization": "bnb_int4",
        "cpu_offload_gb": tune.cpu_offload_gb,
        "kv_compression": tune.kv_cache_dtype,
        "max_model_len": tune.max_model_len,
        "pipeline_path": "detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35",
    }

    if not tune.feasible:
        steps.append(StepResult(
            name="config_infeasible",
            passed=False,
            details=f"vLLM config not feasible: {tune.reason}",
        ))
        results["steps"] = [s.__dict__ for s in steps]
        results["passed"] = False
        return results

    # ── Step 5: Download model if needed ─────────────────────────────
    models_dir = config.models_dir
    model_dir = models_dir / profile.id
    if not (model_dir.is_dir() and (model_dir / "config.json").exists()):
        start = time.time()
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=profile.hf_repo,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            elapsed = time.time() - start
            steps.append(StepResult(
                name="download_model",
                passed=True,
                duration_s=elapsed,
                details=f"Downloaded {profile.hf_repo} to {model_dir} in {elapsed:.1f}s",
            ))
        except Exception as e:
            steps.append(StepResult(
                name="download_model",
                passed=False,
                duration_s=time.time() - start,
                details=f"Download failed: {e}",
            ))
            results["steps"] = [s.__dict__ for s in steps]
            results["passed"] = False
            return results
    else:
        steps.append(StepResult(
            name="model_available",
            passed=True,
            details=f"Model already downloaded at {model_dir}",
        ))

    # ── Step 6: Load model with CPU offloading + INT4 + TurboQuant KV ──
    start = time.time()
    engine = None
    try:
        from tqcli.core.vllm_backend import VllmBackend

        engine = VllmBackend.from_tuning_profile(tune)
        engine.load_model(str(model_dir))
        elapsed = time.time() - start
        steps.append(StepResult(
            name="load_model_with_cpu_offload",
            passed=True,
            duration_s=elapsed,
            details=(
                f"Loaded Gemma 4 E2B via vLLM in {elapsed:.1f}s | "
                f"BNB_INT4 + cpu_offload={tune.cpu_offload_gb} GB + "
                f"kv={tune.kv_cache_dtype}"
            ),
            metrics={
                "load_time_s": round(elapsed, 2),
                "cpu_offload_gb": tune.cpu_offload_gb,
                "quantization": tune.quantization,
                "kv_cache_dtype": tune.kv_cache_dtype,
            },
        ))
    except Exception as e:
        elapsed = time.time() - start
        steps.append(StepResult(
            name="load_model_with_cpu_offload",
            passed=False,
            duration_s=elapsed,
            details=f"Load failed: {e}",
            metrics={"error": str(e)},
        ))
        results["steps"] = [s.__dict__ for s in steps]
        results["passed"] = False
        return results

    # ── Step 7: Chat turn (thinking mode) ────────────────────────────
    if engine:
        monitor = PerformanceMonitor(config.performance)
        think_cfg = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=True)
        sys_prompt = build_system_prompt_with_thinking("Be concise.", think_cfg)
        history = [ChatMessage(role="system", content=sys_prompt)]
        history.append(ChatMessage(role="user", content="What is 15% of 240?"))

        start = time.time()
        try:
            full_response = ""
            final_stats = None
            for chunk, stats in engine.chat_stream(history):
                if stats:
                    final_stats = stats
                    break
                full_response += chunk

            elapsed = time.time() - start
            thinking_text, clean_response = extract_thinking(full_response, ThinkingFormat.GEMMA4)

            metrics = {
                "has_thinking": len(thinking_text.strip()) > 0,
                "thinking_length": len(thinking_text),
                "response_length": len(clean_response),
            }
            if final_stats:
                monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
                metrics.update({
                    "tokens_per_second": round(final_stats.tokens_per_second, 2),
                    "completion_tokens": final_stats.completion_tokens,
                })

            steps.append(StepResult(
                name="chat_thinking_turn",
                passed=len(clean_response.strip()) > 0,
                duration_s=elapsed,
                details=f"Response: {clean_response[:200]}...",
                metrics=metrics,
            ))
        except Exception as e:
            steps.append(StepResult(
                name="chat_thinking_turn",
                passed=False,
                duration_s=time.time() - start,
                details=f"Chat error: {e}",
            ))

        # ── Step 8: Simple chat (no thinking) ────────────────────────
        history2 = [ChatMessage(role="system", content="Be concise.")]
        history2.append(ChatMessage(role="user", content="What is the capital of France?"))

        start = time.time()
        try:
            full_response = ""
            final_stats = None
            for chunk, stats in engine.chat_stream(history2):
                if stats:
                    final_stats = stats
                    break
                full_response += chunk

            elapsed = time.time() - start
            metrics = {"response_length": len(full_response)}
            if final_stats:
                metrics.update({
                    "tokens_per_second": round(final_stats.tokens_per_second, 2),
                    "completion_tokens": final_stats.completion_tokens,
                })

            steps.append(StepResult(
                name="chat_simple_turn",
                passed=len(full_response.strip()) > 0,
                duration_s=elapsed,
                details=f"Response: {full_response[:200]}...",
                metrics=metrics,
            ))
        except Exception as e:
            steps.append(StepResult(
                name="chat_simple_turn",
                passed=False,
                duration_s=time.time() - start,
                details=f"Chat error: {e}",
            ))

        # ── Step 9: Unload ───────────────────────────────────────────
        engine.unload_model()
        steps.append(StepResult(name="unload_model", passed=True, details="Model unloaded"))

    # ── Collect results ──────────────────────────────────────────────
    results["steps"] = [s.__dict__ for s in steps]
    results["pass_count"] = sum(1 for s in steps if s.passed)
    results["fail_count"] = sum(1 for s in steps if not s.passed)
    results["total_steps"] = len(steps)
    results["passed"] = all(s.passed for s in steps)
    results["total_duration_s"] = round(sum(s.duration_s for s in steps), 2)

    return results


def generate_markdown(results: dict) -> str:
    lines = []
    lines.append("# Gemma 4 E2B vLLM + CPU Offloading + BNB_INT4 + TurboQuant KV")
    lines.append("")
    lines.append(f"**Generated:** {results['generated']}")
    lines.append(f"**Test:** {results['test_name']}")
    lines.append(f"**Result:** {'PASS' if results.get('passed') else 'FAIL'} "
                 f"({results.get('pass_count', 0)}/{results.get('total_steps', 0)} steps)")
    lines.append(f"**Duration:** {results.get('total_duration_s', 0)}s")
    lines.append("")

    # System
    s = results.get("system", {})
    lines.append("## System")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    for k, v in s.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Pipeline Config
    pc = results.get("pipeline_config", {})
    if pc:
        lines.append("## Pipeline Configuration")
        lines.append(f"**Path:** `{pc.get('pipeline_path', 'N/A')}`")
        lines.append("")
        lines.append("| Setting | Value |")
        lines.append("|---------|-------|")
        for k, v in pc.items():
            if k != "pipeline_path":
                lines.append(f"| {k} | {v} |")
        lines.append("")

    # Steps
    lines.append("## Step Results")
    lines.append("")
    lines.append("| # | Step | Result | Duration | Details |")
    lines.append("|---|------|--------|----------|---------|")
    for i, step in enumerate(results.get("steps", []), 1):
        status = "PASS" if step["passed"] else "FAIL"
        details = step["details"][:100].replace("|", "\\|").replace("\n", " ")
        dur = f"{step['duration_s']:.2f}s" if step["duration_s"] > 0 else "-"
        lines.append(f"| {i} | {step['name']} | {status} | {dur} | {details} |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GEMMA 4 E2B vLLM + CPU OFFLOADING TEST")
    print("BNB_INT4 + cpu_offload + turboquant35 KV")
    print("=" * 60)

    results = run_test()

    # Write reports
    json_path = REPORT_DIR / "gemma4_vllm_cpu_offload_report.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    md_path = REPORT_DIR / "gemma4_vllm_cpu_offload_report.md"
    md_path.write_text(generate_markdown(results))

    # Print summary
    print(f"\nResult: {'PASS' if results.get('passed') else 'FAIL'} "
          f"({results.get('pass_count', 0)}/{results.get('total_steps', 0)} steps)")
    print(f"Duration: {results.get('total_duration_s', 0)}s")

    pc = results.get("pipeline_config", {})
    if pc:
        print(f"\nPipeline: {pc.get('pipeline_path', 'N/A')}")
        print(f"CPU offload: {pc.get('cpu_offload_gb', 0)} GB")
        print(f"KV compression: {pc.get('kv_compression', 'N/A')}")

    for step in results.get("steps", []):
        status = "PASS" if step["passed"] else "FAIL"
        print(f"  [{status}] {step['name']}: {step['details'][:120]}")

    print(f"\nJSON: {json_path}")
    print(f"MD:   {md_path}")
