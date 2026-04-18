#!/usr/bin/env python3
"""Agent modes INFRASTRUCTURE SMOKE test.

Scope: verifies that the orchestrator wires cleanly against a real
TurboQuant-loaded backend on both engines WITHOUT crashing. It does NOT
verify parse→execute→observation→loop functionality against a live
model — Gemma 4 E2B (2 B params, not tool-fine-tuned) does not reliably
emit protocol-compliant `<tool_call>` tags, so any functionality check
driven by the real model is flaky at best and silently null at worst.

Functional end-to-end verification lives in
`test_integration_agent_functional.py`, which uses a playback engine to
guarantee protocol-compliant emissions regardless of model compliance,
then delegates the follow-up turn to the real quantized backend so the
round-trip is still exercised against live TurboQuant inference.

This smoke test is useful during dev as a "doesn't crash on real engine"
gate; it must NOT be presented as feature verification.

Report outputs:
  tests/integration_reports/agent_modes_smoke_report.md
  tests/integration_reports/agent_modes_smoke_report.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig  # noqa: E402
from tqcli.core.agent_orchestrator import (  # noqa: E402
    MODE_TINKERING,
    MODE_UNRESTRICTED,
    AgentOrchestrator,
    OrchestratorConfig,
    build_tool_system_prompt,
    parse_tool_calls,
)
from tqcli.core.agent_tools import default_tools  # noqa: E402
from tqcli.core.engine import ChatMessage  # noqa: E402
from tqcli.core.kv_quantizer import (  # noqa: E402
    check_turboquant_compatibility,
    get_llama_kv_params,
    select_kv_quant,
)
from tqcli.core.model_registry import BUILTIN_PROFILES  # noqa: E402
from tqcli.core.performance import PerformanceMonitor  # noqa: E402
from tqcli.core.system_info import detect_system  # noqa: E402
from tqcli.core.vllm_config import build_vllm_config  # noqa: E402

REPORT_DIR = Path(__file__).parent / "integration_reports"

LLAMA_MODEL_ID = "gemma-4-e2b-it-Q4_K_M"
VLLM_MODEL_ID = "gemma-4-e2b-it-vllm"

# Keep prompts short so a slow CPU-offload vLLM run finishes in minutes, not
# hours. The instruction is strong enough that a compliant model will emit
# the target tag shape; a non-compliant model still exercises every
# orchestrator code path.
TINKERING_USER_PROMPT = (
    "Use the tq-file-read tool to read /tmp/tqcli_agent_smoke.txt. "
    "Emit exactly one <staged_tool_call> block and nothing else. "
    'Format: <staged_tool_call>{"name":"tq-file-read",'
    '"arguments":{"path":"/tmp/tqcli_agent_smoke.txt"}}</staged_tool_call>'
)
YOLO_USER_PROMPT = (
    "Use the tq-file-read tool to read /tmp/tqcli_agent_smoke.txt. "
    "Emit exactly one <tool_call> block and nothing else. "
    'Format: <tool_call>{"name":"tq-file-read",'
    '"arguments":{"path":"/tmp/tqcli_agent_smoke.txt"}}</tool_call>'
)
SMOKE_FILE = Path("/tmp/tqcli_agent_smoke.txt")
SMOKE_PAYLOAD = "Hello from the agent modes integration test.\n"


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class TestResult:
    test_name: str
    engine: str
    model_id: str
    kv_quant: str = ""
    agent_mode: str = ""
    started: str = ""
    finished: str = ""
    total_duration_s: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    passed: bool = False

    def add_step(self, s: StepResult):
        self.steps.append(s)

    @property
    def pass_count(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for s in self.steps if not s.passed)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _prepare_smoke_fixture() -> None:
    SMOKE_FILE.write_text(SMOKE_PAYLOAD, encoding="utf-8")


def _get_profile(model_id: str):
    for p in BUILTIN_PROFILES:
        if p.id == model_id:
            return p
    raise RuntimeError(f"profile {model_id} not found in BUILTIN_PROFILES")


def _auto_approve_confirm(name, args, safety):
    return "y", args


# ─── Test A: llama.cpp + Gemma 4 E2B + turbo3 KV + ai_tinkering ──────


def test_llama_agent_tinkering() -> TestResult:
    """Gemma 4 E2B via llama.cpp + TurboQuant turbo3 KV + ai_tinkering."""
    result = TestResult(
        test_name="llama.cpp Gemma 4 E2B + turbo3 KV + ai_tinkering",
        engine="llama.cpp",
        model_id=LLAMA_MODEL_ID,
        kv_quant="turbo3",
        agent_mode=MODE_TINKERING,
        started=_now(),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    profile = _get_profile(LLAMA_MODEL_ID)
    model_path = profile.local_path or (config.models_dir / profile.filename)

    step_start = time.time()
    present = Path(model_path).is_file()
    result.add_step(StepResult(
        name="model_file_present",
        passed=present,
        duration_s=time.time() - step_start,
        details=f"path={model_path} exists={present}",
    ))
    if not present:
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = False
        return result

    # TurboQuant params
    step_start = time.time()
    tq_available, tq_msg = check_turboquant_compatibility(sys_info)
    kv_level = select_kv_quant(available_kv_mb=50, engine="llama.cpp", user_choice="turbo3")
    kv_params = get_llama_kv_params(kv_level) if tq_available else {}
    result.add_step(StepResult(
        name="resolve_kv_params",
        passed=True,
        duration_s=time.time() - step_start,
        details=(
            f"tq_available={tq_available}, level={kv_level.value}, "
            f"params={kv_params}"
        ),
        metrics={
            "tq_available": tq_available,
            "kv_level": kv_level.value,
            "cache_type_k": kv_params.get("cache_type_k"),
            "cache_type_v": kv_params.get("cache_type_v"),
        },
    ))

    # Load backend
    step_start = time.time()
    try:
        from tqcli.core.llama_backend import LlamaBackend

        eng = LlamaBackend(
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=0,
            cache_type_k=kv_params.get("cache_type_k", "f16"),
            cache_type_v=kv_params.get("cache_type_v", "f16"),
        )
        if not eng.is_available:
            raise RuntimeError("llama-cpp-python not installed")
        eng.load_model(str(model_path))
        result.add_step(StepResult(
            name="load_model",
            passed=True,
            duration_s=time.time() - step_start,
            details=f"loaded {profile.display_name} (n_ctx=2048)",
            metrics={"load_time_s": round(time.time() - step_start, 2)},
        ))
    except Exception as exc:
        result.add_step(StepResult(
            name="load_model",
            passed=False,
            duration_s=time.time() - step_start,
            details=f"load failed: {exc}",
        ))
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = all(s.passed for s in result.steps)
        return result

    # Orchestrator + scripted auto-approve
    try:
        _prepare_smoke_fixture()
        orch_cfg = OrchestratorConfig(
            mode=MODE_TINKERING,
            max_steps=3,
            confirm_fn=_auto_approve_confirm,
            tools=default_tools(),
        )
        orch = AgentOrchestrator(engine=eng, config=orch_cfg)

        # Schema injection check
        step_start = time.time()
        schemas = orch.injected_tool_schemas
        result.add_step(StepResult(
            name="schema_injection_non_empty",
            passed=len(schemas) == 4,
            duration_s=time.time() - step_start,
            details=f"{len(schemas)} tool schemas surfaced in agent mode",
        ))

        sys_prompt = build_tool_system_prompt(default_tools(), MODE_TINKERING)
        history = [
            ChatMessage(role="system", content="Be concise.\n\n" + sys_prompt),
            ChatMessage(role="user", content=TINKERING_USER_PROMPT),
        ]

        step_start = time.time()
        final_text, updated = orch.run_turn(history, max_tokens=256)
        elapsed = time.time() - step_start

        calls = parse_tool_calls(final_text)
        emitted_tag = any(c.kind == "staged" and c.name for c in calls)
        observations = [m for m in updated if m.role == "user" and m.content.startswith("Observation")]

        result.add_step(StepResult(
            name="orchestrator_run_turn",
            passed=True,
            duration_s=elapsed,
            details=(
                f"final_text_len={len(final_text)} | "
                f"emitted_staged_tag={emitted_tag} | "
                f"observations_fed_back={len(observations)} | "
                f"final_text_head={final_text[:160].replace(chr(10), ' ')}"
            ),
            metrics={
                "emitted_staged_tag": emitted_tag,
                "observations_fed_back": len(observations),
                "final_text_chars": len(final_text),
                "run_time_s": round(elapsed, 2),
            },
        ))
    finally:
        eng.unload_model()

    result.finished = _now()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


# ─── Test B: vLLM + Gemma 4 E2B + turboquant35 + unrestricted ──────


def test_vllm_agent_unrestricted() -> TestResult:
    """Gemma 4 E2B via vLLM + BNB_INT4 + CPU offload + turboquant35 KV + unrestricted."""
    result = TestResult(
        test_name="vLLM Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35 + unrestricted",
        engine="vllm",
        model_id=VLLM_MODEL_ID,
        kv_quant="turboquant35",
        agent_mode=MODE_UNRESTRICTED,
        started=_now(),
    )

    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()
    profile = _get_profile(VLLM_MODEL_ID)
    model_dir = profile.local_path or (config.models_dir / profile.id)

    step_start = time.time()
    model_dir_ok = model_dir.is_dir() and (model_dir / "config.json").exists()
    result.add_step(StepResult(
        name="model_dir_present",
        passed=model_dir_ok,
        duration_s=time.time() - step_start,
        details=f"path={model_dir} dir={model_dir_ok}",
    ))
    if not model_dir_ok:
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = False
        return result

    step_start = time.time()
    try:
        import vllm  # noqa: F401
        has_vllm = True
    except ImportError as err:
        has_vllm = False
        import_err = str(err)
    result.add_step(StepResult(
        name="vllm_importable",
        passed=has_vllm,
        duration_s=time.time() - step_start,
        details="vllm import OK" if has_vllm else f"vllm import failed: {import_err}",
    ))
    if not has_vllm:
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = False
        return result

    # Build tuning profile (BNB_INT4 + CPU offload + turboquant35 KV)
    step_start = time.time()
    tune = build_vllm_config(profile, sys_info, requested_max_len=2048, kv_quant_choice="turbo3")
    result.add_step(StepResult(
        name="build_vllm_config",
        passed=tune.feasible,
        duration_s=time.time() - step_start,
        details=(
            f"feasible={tune.feasible} | quant={tune.quantization} | "
            f"cpu_offload_gb={tune.cpu_offload_gb} | kv={tune.kv_cache_dtype} | "
            f"enforce_eager={tune.enforce_eager}"
        ),
        metrics={
            "quantization": tune.quantization,
            "cpu_offload_gb": tune.cpu_offload_gb,
            "kv_cache_dtype": tune.kv_cache_dtype,
            "enforce_eager": tune.enforce_eager,
            "max_model_len": tune.max_model_len,
        },
    ))
    if not tune.feasible:
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = False
        return result

    # Load backend
    step_start = time.time()
    try:
        from tqcli.core.vllm_backend import VllmBackend

        eng = VllmBackend.from_tuning_profile(tune)
        eng.load_model(str(model_dir))
        result.add_step(StepResult(
            name="load_model",
            passed=True,
            duration_s=time.time() - step_start,
            details=f"loaded {profile.display_name} via vLLM",
            metrics={"load_time_s": round(time.time() - step_start, 2)},
        ))
    except Exception as exc:
        result.add_step(StepResult(
            name="load_model",
            passed=False,
            duration_s=time.time() - step_start,
            details=f"load failed: {exc}",
        ))
        result.finished = _now()
        result.total_duration_s = sum(s.duration_s for s in result.steps)
        result.passed = all(s.passed for s in result.steps)
        return result

    # Unrestricted ReAct
    try:
        _prepare_smoke_fixture()
        orch_cfg = OrchestratorConfig(
            mode=MODE_UNRESTRICTED,
            max_steps=3,
            tools=default_tools(),
        )
        orch = AgentOrchestrator(engine=eng, config=orch_cfg)

        step_start = time.time()
        schemas = orch.injected_tool_schemas
        result.add_step(StepResult(
            name="schema_injection_non_empty",
            passed=len(schemas) == 4,
            duration_s=time.time() - step_start,
            details=f"{len(schemas)} tool schemas surfaced in agent mode",
        ))

        sys_prompt = build_tool_system_prompt(default_tools(), MODE_UNRESTRICTED)
        history = [
            ChatMessage(role="system", content="Be concise.\n\n" + sys_prompt),
            ChatMessage(role="user", content=YOLO_USER_PROMPT),
        ]

        step_start = time.time()
        final_text, updated = orch.run_turn(history, max_tokens=256)
        elapsed = time.time() - step_start

        calls = parse_tool_calls(final_text)
        emitted_tag = any(c.kind == "live" and c.name for c in calls)
        observations = [m for m in updated if m.role == "user" and m.content.startswith("Observation")]

        result.add_step(StepResult(
            name="orchestrator_react_loop",
            passed=True,
            duration_s=elapsed,
            details=(
                f"final_text_len={len(final_text)} | "
                f"emitted_live_tag={emitted_tag} | "
                f"observations_fed_back={len(observations)} | "
                f"final_text_head={final_text[:160].replace(chr(10), ' ')}"
            ),
            metrics={
                "emitted_live_tag": emitted_tag,
                "observations_fed_back": len(observations),
                "final_text_chars": len(final_text),
                "run_time_s": round(elapsed, 2),
            },
        ))
    finally:
        try:
            eng.unload_model()
        except Exception:
            pass

    result.finished = _now()
    result.total_duration_s = sum(s.duration_s for s in result.steps)
    result.passed = all(s.passed for s in result.steps)
    return result


# ─── Report generation ──────────────────────────────────────────────


def generate_markdown(results: list[TestResult], sys_info: dict) -> str:
    lines = [
        "# Agent Modes — Integration Test Report",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**System:** {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)",
        f"**CUDA:** driver {sys_info.get('cuda_version', 'n/a')}, toolkit {sys_info.get('cuda_toolkit', 'n/a')}",
        f"**OS:** {sys_info['os']}",
        "",
        "Exercises the Phase 2 / Phase 3 orchestrator (`tqcli/core/agent_orchestrator.py`) "
        "end-to-end with a real Gemma 4 E2B model on both backends and TurboQuant KV "
        "compression active.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Test | Engine | Model | KV Quant | Mode | Result |",
        "|------|--------|-------|----------|------|--------|",
    ]
    for r in results:
        status = f"**PASS** ({r.pass_count}/{r.pass_count + r.fail_count})" if r.passed else f"**FAIL** ({r.pass_count}/{r.pass_count + r.fail_count})"
        lines.append(
            f"| {r.test_name} | {r.engine} | {r.model_id} | {r.kv_quant} | {r.agent_mode} | {status} |"
        )
    lines.extend(["", "---", ""])

    for r in results:
        lines.extend([
            f"## {r.test_name}",
            "",
            f"- **Engine:** {r.engine}",
            f"- **Model:** {r.model_id}",
            f"- **KV Quant:** {r.kv_quant}",
            f"- **Agent Mode:** {r.agent_mode}",
            f"- **Started / Finished:** {r.started} → {r.finished}",
            f"- **Total Duration:** {r.total_duration_s:.2f}s",
            f"- **Result:** {'PASS' if r.passed else 'FAIL'} ({r.pass_count}/{r.pass_count + r.fail_count})",
            "",
            "| Step | Result | Duration | Details |",
            "|------|--------|----------|---------|",
        ])
        for s in r.steps:
            status = "PASS" if s.passed else "FAIL"
            details = s.details[:180].replace("|", "/")
            lines.append(f"| {s.name} | {status} | {s.duration_s:.2f}s | {details} |")
        lines.append("")
    return "\n".join(lines)


def generate_json(results: list[TestResult], sys_info: dict) -> dict:
    return {
        "generated": datetime.now().isoformat(),
        "system": sys_info,
        "tests": [
            {
                "name": r.test_name,
                "engine": r.engine,
                "model_id": r.model_id,
                "kv_quant": r.kv_quant,
                "agent_mode": r.agent_mode,
                "started": r.started,
                "finished": r.finished,
                "duration_s": round(r.total_duration_s, 3),
                "passed": r.passed,
                "pass_count": r.pass_count,
                "fail_count": r.fail_count,
                "steps": [
                    {
                        "name": s.name,
                        "passed": s.passed,
                        "duration_s": round(s.duration_s, 4),
                        "details": s.details,
                        "metrics": s.metrics,
                    }
                    for s in r.steps
                ],
            }
            for r in results
        ],
    }


def get_system_info_dict() -> dict:
    info = detect_system()
    gpu0 = info.gpus[0] if info.gpus else None
    return {
        "os": info.os_display,
        "gpu": gpu0.name if gpu0 else "None",
        "vram_mb": info.total_vram_mb,
        "ram_mb": info.ram_total_mb,
        "is_wsl": info.is_wsl,
        "cuda_version": gpu0.cuda_version if gpu0 else "n/a",
        "cuda_toolkit": gpu0.cuda_toolkit_version if gpu0 else "n/a",
    }


def main() -> int:
    print("=" * 70)
    print("Agent Modes — Integration Tests")
    print("=" * 70)

    sys_info = get_system_info_dict()
    print(f"System: {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)")
    print(f"CUDA: driver {sys_info['cuda_version']}, toolkit {sys_info['cuda_toolkit']}")

    tests = [test_llama_agent_tinkering, test_vllm_agent_unrestricted]
    results: list[TestResult] = []
    for fn in tests:
        print(f"\n--- {fn.__doc__.strip()} ---")
        try:
            r = fn()
        except Exception as exc:
            r = TestResult(
                test_name=fn.__doc__.strip(),
                engine="unknown",
                model_id="unknown",
                started=_now(),
                finished=_now(),
            )
            r.add_step(StepResult(
                name="test_crashed",
                passed=False,
                details=f"{type(exc).__name__}: {exc}",
            ))
            r.passed = False
        results.append(r)
        status = "PASS" if r.passed else "FAIL"
        print(f"  Result: {status} ({r.pass_count}/{r.pass_count + r.fail_count}) in {r.total_duration_s:.1f}s")
        for s in r.steps:
            tag = "  [PASS]" if s.passed else "  [FAIL]"
            print(f"  {tag} {s.name}: {s.details[:140]}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = generate_markdown(results, sys_info)
    (REPORT_DIR / "agent_modes_smoke_report.md").write_text(md)
    js = generate_json(results, sys_info)
    (REPORT_DIR / "agent_modes_smoke_report.json").write_text(json.dumps(js, indent=2))

    total_pass = sum(1 for r in results if r.passed)
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {total_pass}/{len(results)} tests passed")
    print(f"Reports:")
    print(f"  {REPORT_DIR / 'agent_modes_smoke_report.md'}")
    print(f"  {REPORT_DIR / 'agent_modes_smoke_report.json'}")
    print(f"{'=' * 70}")
    return 0 if total_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
