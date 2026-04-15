#!/usr/bin/env python3
"""Combined integration test runner for tqCLI.

Runs BOTH llama.cpp and vLLM integration tests (Tests 5-7 only: thinking,
tool calling, and combined) with TurboQuant KV cache compression.

Validates the unified quantization pipeline:
  - Pre-quantized models (GGUF Q4_K_M, AWQ) → KV cache compression ONLY
  - Full-precision models (BF16) → weight quantization + KV cache compression

Outputs:
  - tests/integration_reports/turboquant_kv_unified_report.json  (audit)
  - tests/integration_reports/turboquant_kv_unified_report.md    (human-readable)

Usage:
  python tests/test_integration_combined.py                  # Run all (5-7 on both engines)
  python tests/test_integration_combined.py --engine llama   # llama.cpp only
  python tests/test_integration_combined.py --engine vllm    # vLLM only
  python tests/test_integration_combined.py --test 5         # Thinking only (both engines)
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

# Ensure tqcli is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

REPORT_DIR = Path(__file__).parent / "integration_reports"


@dataclass
class EngineResults:
    engine_name: str
    results: list = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


def get_system_info() -> dict:
    """Get system info dict for report."""
    from tqcli.core.system_info import detect_system
    info = detect_system()
    return {
        "os": info.os_display,
        "arch": info.arch,
        "cpu_cores": info.cpu_cores_logical,
        "cpu_physical": info.cpu_cores_physical,
        "ram_total_mb": info.ram_total_mb,
        "ram_available_mb": info.ram_available_mb,
        "gpu": info.gpus[0].name if info.gpus else "None",
        "vram_mb": info.total_vram_mb,
        "recommended_engine": info.recommended_engine,
        "recommended_quant": info.recommended_quant,
        "max_model_gb": info.max_model_size_estimate_gb,
        "is_wsl": info.is_wsl,
    }


def get_vllm_version() -> str:
    try:
        import vllm
        return vllm.__version__
    except Exception:
        return "not installed"


def get_llama_cpp_version() -> str:
    try:
        import llama_cpp
        return getattr(llama_cpp, "__version__", "installed (version unknown)")
    except Exception:
        return "not installed"


# ─── llama.cpp Test Runner ──────────────────────────────────────────────


def run_llama_tests(test_nums: list[int]) -> EngineResults:
    """Run llama.cpp integration tests 5-7."""
    engine_results = EngineResults(engine_name="llama.cpp (TurboQuant fork)")
    start = time.time()

    try:
        from tests.test_integration import (
            run_test_5_thinking_turbo3,
            run_test_6_tool_calling_turbo3,
            run_test_7_combined_turbo3,
        )

        test_map = {
            5: ("Test 5: Thinking + turbo3 KV (llama.cpp)", run_test_5_thinking_turbo3),
            6: ("Test 6: Tool Calling + turbo3 KV (llama.cpp)", run_test_6_tool_calling_turbo3),
            7: ("Test 7: Combined + turbo3 KV (llama.cpp)", run_test_7_combined_turbo3),
        }

        for num in test_nums:
            if num in test_map:
                name, func = test_map[num]
                print(f"\n{'=' * 60}")
                print(f"RUNNING {name}")
                print(f"{'=' * 60}")
                try:
                    result = func()
                    engine_results.results.append(result)
                    status = "PASS" if result.passed else "FAIL"
                    print(f"  {name}: {status} ({result.pass_count}/{len(result.steps)} steps)")
                except Exception as e:
                    tb = traceback.format_exc()
                    engine_results.errors.append(f"{name}: {e}\n{tb}")
                    print(f"  {name}: ERROR - {e}")

    except ImportError as e:
        engine_results.errors.append(f"Import error: {e}")
        print(f"  llama.cpp tests import failed: {e}")

    engine_results.duration_s = time.time() - start
    return engine_results


# ─── vLLM Test Runner ──────────────────────────────────────────────────


def run_vllm_tests(test_nums: list[int]) -> EngineResults:
    """Run vLLM integration tests 5-7."""
    engine_results = EngineResults(engine_name="vLLM (TurboQuant fork)")
    start = time.time()

    try:
        from tests.test_integration_vllm import (
            run_test_5_thinking_turboquant_vllm,
            run_test_6_tool_calling_turboquant_vllm,
            run_test_7_combined_turboquant_vllm,
        )

        test_map = {
            5: ("Test 5: Thinking + turboquant35 KV (vLLM)", run_test_5_thinking_turboquant_vllm),
            6: ("Test 6: Tool Calling + turboquant35 KV (vLLM)", run_test_6_tool_calling_turboquant_vllm),
            7: ("Test 7: Combined + turboquant35 KV (vLLM)", run_test_7_combined_turboquant_vllm),
        }

        for num in test_nums:
            if num in test_map:
                name, func = test_map[num]
                print(f"\n{'=' * 60}")
                print(f"RUNNING {name}")
                print(f"{'=' * 60}")
                try:
                    result = func()
                    engine_results.results.append(result)
                    status = "PASS" if result.passed else "FAIL"
                    print(f"  {name}: {status} ({result.pass_count}/{len(result.steps)} steps)")
                except Exception as e:
                    tb = traceback.format_exc()
                    engine_results.errors.append(f"{name}: {e}\n{tb}")
                    print(f"  {name}: ERROR - {e}")

    except ImportError as e:
        engine_results.errors.append(f"Import error: {e}")
        print(f"  vLLM tests import failed: {e}")

    engine_results.duration_s = time.time() - start
    return engine_results


# ─── Report Generation ──────────────────────────────────────────────────


def _serialize_results(engine_results: EngineResults) -> list[dict]:
    """Serialize TestResult objects to dicts."""
    serialized = []
    for r in engine_results.results:
        serialized.append({
            "test_name": r.test_name,
            "model_id": r.model_id,
            "model_family": r.model_family,
            "engine": r.engine,
            "started": r.started,
            "finished": r.finished,
            "total_duration_s": r.total_duration_s,
            "passed": r.passed,
            "pass_count": r.pass_count,
            "fail_count": r.fail_count,
            "steps": [
                {
                    "name": s.name,
                    "passed": s.passed,
                    "duration_s": s.duration_s,
                    "details": s.details,
                    "metrics": s.metrics,
                }
                for s in r.steps
            ],
        })
    return serialized


def generate_json_report(
    system_info: dict,
    llama_results: EngineResults | None,
    vllm_results: EngineResults | None,
) -> dict:
    """Generate structured JSON report for auditing."""
    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "report_type": "turboquant_kv_unified_integration",
        "tqcli_version": "0.5.0",
        "system": system_info,
        "engines": {
            "llama_cpp": get_llama_cpp_version(),
            "vllm": get_vllm_version(),
        },
        "test_scope": "Tests 5-7: Thinking + Tool Calling + Combined with TurboQuant KV",
        "pipeline_validation": {
            "description": (
                "Validates the unified quantization pipeline detects pre-quantized models "
                "(GGUF Q4_K_M, AWQ) and applies KV cache compression ONLY, while "
                "full-precision models (BF16) receive both weight quantization and KV compression."
            ),
            "expected_paths": {
                "pre_quantized_gguf": "KV compression only (turbo3)",
                "pre_quantized_awq": "KV compression only (turboquant35)",
                "full_precision_bf16": "weight:bnb_int4 + kv:turbo3",
            },
        },
    }

    all_results = []
    total_pass = 0
    total_fail = 0
    total_steps = 0

    if llama_results:
        report["llama_cpp"] = {
            "engine": llama_results.engine_name,
            "duration_s": round(llama_results.duration_s, 2),
            "errors": llama_results.errors,
            "tests": _serialize_results(llama_results),
        }
        for r in llama_results.results:
            all_results.append(r)
            total_pass += r.pass_count
            total_fail += r.fail_count
            total_steps += len(r.steps)

    if vllm_results:
        report["vllm"] = {
            "engine": vllm_results.engine_name,
            "duration_s": round(vllm_results.duration_s, 2),
            "errors": vllm_results.errors,
            "tests": _serialize_results(vllm_results),
        }
        for r in vllm_results.results:
            all_results.append(r)
            total_pass += r.pass_count
            total_fail += r.fail_count
            total_steps += len(r.steps)

    report["summary"] = {
        "total_tests": len(all_results),
        "total_steps": total_steps,
        "steps_passed": total_pass,
        "steps_failed": total_fail,
        "pass_rate": round(total_pass / max(total_steps, 1) * 100, 1),
        "all_passed": total_fail == 0 and total_steps > 0,
    }

    return report


def generate_markdown_report(
    system_info: dict,
    llama_results: EngineResults | None,
    vllm_results: EngineResults | None,
) -> str:
    """Generate human-readable markdown report."""
    lines = []
    lines.append("# tqCLI Unified Integration Test Report — TurboQuant KV")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**tqCLI Version:** 0.5.0")
    lines.append(f"**Scope:** Tests 5-7 (Thinking + Tool Calling + Combined) with TurboQuant KV")
    lines.append(f"**Engines:** llama.cpp ({get_llama_cpp_version()}), vLLM ({get_vllm_version()})")
    lines.append("")

    # System info
    lines.append("## System Information")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    for k, v in system_info.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Pipeline validation
    lines.append("## Quantization Pipeline Validation")
    lines.append("")
    lines.append("The unified quantization pipeline detects model precision and applies the appropriate stages:")
    lines.append("")
    lines.append("| Model Type | Weight Quantization | KV Cache Compression | Pipeline Path |")
    lines.append("|------------|--------------------|--------------------|---------------|")
    lines.append("| GGUF Q4_K_M (llama.cpp) | SKIP (pre-quantized) | turbo3 (4.6x) | KV-only |")
    lines.append("| AWQ INT4 (vLLM) | SKIP (pre-quantized) | turboquant35 | KV-only |")
    lines.append("| BF16 safetensors (vLLM) | BNB_INT4 (on-the-fly) | turboquant35 | Full pipeline |")
    lines.append("")

    # Collect all results
    all_engines = []
    if llama_results:
        all_engines.append(llama_results)
    if vllm_results:
        all_engines.append(vllm_results)

    total_pass = 0
    total_fail = 0
    total_steps = 0
    total_tests = 0

    for eng in all_engines:
        for r in eng.results:
            total_pass += r.pass_count
            total_fail += r.fail_count
            total_steps += len(r.steps)
            total_tests += 1

    # Overall summary
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total Tests | {total_tests} |")
    lines.append(f"| Total Steps | {total_steps} |")
    lines.append(f"| Passed | {total_pass} |")
    lines.append(f"| Failed | {total_fail} |")
    lines.append(f"| Pass Rate | {total_pass / max(total_steps, 1) * 100:.1f}% |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Per-engine results
    for eng in all_engines:
        lines.append(f"## {eng.engine_name}")
        lines.append(f"**Duration:** {eng.duration_s:.1f}s")
        lines.append("")

        if eng.errors:
            lines.append("### Errors")
            for err in eng.errors:
                lines.append(f"- `{err[:200]}`")
            lines.append("")

        for r in eng.results:
            status = "**PASS**" if r.passed else "**FAIL**"
            lines.append(f"### {r.test_name}")
            lines.append("")
            lines.append(
                f"**Model:** `{r.model_id}` | **Engine:** {r.engine} | "
                f"**Result:** {status} ({r.pass_count}/{len(r.steps)} steps) | "
                f"**Duration:** {r.total_duration_s:.1f}s"
            )
            lines.append("")
            lines.append("| # | Step | Result | Duration | Details |")
            lines.append("|---|------|--------|----------|---------|")
            for i, s in enumerate(r.steps, 1):
                st = "PASS" if s.passed else "FAIL"
                details = s.details[:100].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {i} | {s.name} | {st} | {s.duration_s:.2f}s | {details} |")
            lines.append("")

            # Pipeline metrics
            pipe_steps = [s for s in r.steps if s.name == "quantization_pipeline"]
            if pipe_steps:
                lines.append("#### Pipeline Decision")
                for s in pipe_steps:
                    m = s.metrics
                    lines.append(
                        f"- **Precision:** {m.get('model_precision', 'N/A')} | "
                        f"**Weight quant:** {m.get('needs_weight_quant', 'N/A')} "
                        f"({m.get('weight_quant_method', 'none')}) | "
                        f"**KV level:** {m.get('kv_level', 'N/A')} | "
                        f"**KV-only:** {m.get('kv_only_for_prequantized', 'N/A')}"
                    )
                lines.append("")

            # Performance metrics
            perf_steps = [s for s in r.steps if s.metrics.get("tokens_per_second")]
            if perf_steps:
                lines.append("#### Performance")
                lines.append("| Step | tok/s | Tokens | Thinking |")
                lines.append("|------|-------|--------|----------|")
                for s in perf_steps:
                    m = s.metrics
                    thinking = "YES" if m.get("has_thinking_block") else "NO" if "has_thinking_block" in m else "N/A"
                    lines.append(
                        f"| {s.name} | {m.get('tokens_per_second', 'N/A')} | "
                        f"{m.get('completion_tokens', 'N/A')} | {thinking} |"
                    )
                lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


# ─── Main ───────────────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="tqCLI unified integration tests: TurboQuant KV + Thinking + Tool Calling",
    )
    parser.add_argument(
        "--test", type=int, choices=[5, 6, 7], default=None,
        help="Run specific test (5=thinking, 6=tool calling, 7=combined)",
    )
    parser.add_argument(
        "--engine", choices=["llama", "vllm", "both"], default="both",
        help="Which engine(s) to test (default: both)",
    )
    parser.add_argument(
        "--output-dir", default=str(REPORT_DIR),
        help="Output directory for reports",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    test_nums = [args.test] if args.test else [5, 6, 7]

    print("=" * 60)
    print("tqCLI UNIFIED INTEGRATION TESTS")
    print("TurboQuant KV + Thinking + Tool Calling")
    print(f"Tests: {test_nums} | Engines: {args.engine}")
    print("=" * 60)

    system_info = get_system_info()
    llama_results = None
    vllm_results = None

    # Run llama.cpp tests
    if args.engine in ("llama", "both"):
        print("\n" + "=" * 60)
        print("LLAMA.CPP ENGINE TESTS (TurboQuant turbo3 KV)")
        print("=" * 60)
        llama_results = run_llama_tests(test_nums)
        print(f"\nllama.cpp: {len(llama_results.results)} tests, "
              f"{sum(r.pass_count for r in llama_results.results)} steps passed, "
              f"{sum(r.fail_count for r in llama_results.results)} failed, "
              f"{len(llama_results.errors)} errors")

    # Run vLLM tests
    if args.engine in ("vllm", "both"):
        print("\n" + "=" * 60)
        print("VLLM ENGINE TESTS (TurboQuant turboquant35 KV)")
        print("=" * 60)
        vllm_results = run_vllm_tests(test_nums)
        print(f"\nvLLM: {len(vllm_results.results)} tests, "
              f"{sum(r.pass_count for r in vllm_results.results)} steps passed, "
              f"{sum(r.fail_count for r in vllm_results.results)} failed, "
              f"{len(vllm_results.errors)} errors")

    # Generate reports
    json_report = generate_json_report(system_info, llama_results, vllm_results)
    md_report = generate_markdown_report(system_info, llama_results, vllm_results)

    json_path = output_dir / "turboquant_kv_unified_report.json"
    md_path = output_dir / "turboquant_kv_unified_report.md"

    json_path.write_text(json.dumps(json_report, indent=2, default=str))
    md_path.write_text(md_report)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    s = json_report["summary"]
    print(f"Total tests: {s['total_tests']}")
    print(f"Total steps: {s['total_steps']} ({s['steps_passed']} passed, {s['steps_failed']} failed)")
    print(f"Pass rate: {s['pass_rate']}%")
    print(f"All passed: {'YES' if s['all_passed'] else 'NO'}")
    print(f"\nJSON report: {json_path}")
    print(f"MD report:   {md_path}")

    # Exit code: 0 if all passed, 1 if failures
    sys.exit(0 if s["all_passed"] else 1)


if __name__ == "__main__":
    main()
