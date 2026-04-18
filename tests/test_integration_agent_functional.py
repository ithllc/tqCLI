#!/usr/bin/env python3
"""FUNCTIONAL end-to-end integration test for tri-state agentic autonomy.

Covers the ACTUAL feature surface — not just engine load + schema count.
The previous `test_integration_agent_smoke.py` is kept as an
infrastructure smoke test (does not crash); this file verifies the
parse→execute→observation→real-inference-on-observation round trip
with concrete assertions.

Matrix (15 scenarios):

  Zero-shot data points (not pass gates, recorded for model-choice
  evidence):
    T0a  ai_tinkering   llama.cpp + Gemma 4 E2B + turbo3
    T0b  ai_tinkering   llama.cpp + Qwen 3 4B   + turbo3
    T0c  unrestricted   vLLM      + Gemma 4 E2B + turboquant35 + CPU offload
    T0d  unrestricted   vLLM      + Qwen 3 4B   + turboquant35

  Orchestrator-logic (model-independent, runs once on the fast engine):
    T2   ai_tinkering   llama.cpp + Gemma 4 — DENY actionable tool
    T3   ai_tinkering   llama.cpp + Gemma 4 — EDIT actionable tool
    T5   unrestricted   llama.cpp + Gemma 4 — dead tool name

  Round-trip integration (both modes × both engines × both models):
    T1_lg  ai_tinkering   llama.cpp + Gemma 4 — APPROVE safe tool + secret-word
    T1_lq  ai_tinkering   llama.cpp + Qwen 3  — APPROVE safe tool + secret-word
    T1_vg  ai_tinkering   vLLM      + Gemma 4 — APPROVE safe tool + secret-word
    T1_vq  ai_tinkering   vLLM      + Qwen 3  — APPROVE safe tool + secret-word
    T4_lg  unrestricted   llama.cpp + Gemma 4 — ReAct chain + secret-word
    T4_lq  unrestricted   llama.cpp + Qwen 3  — ReAct chain + secret-word
    T4_vg  unrestricted   vLLM      + Gemma 4 — ReAct chain + secret-word
    T4_vq  unrestricted   vLLM      + Qwen 3  — ReAct chain + secret-word

Design:
  - `PlaybackEngine(real_engine, [scripted])` wraps a live TurboQuant
    backend. Call #1 yields scripted protocol-compliant tag bytes
    (what a tool-trained model WOULD emit). Call #2+ delegates to the
    real engine with the orchestrator's observation-extended history.
  - Fixture `/tmp/tqcli_agent_fixture.txt` contains a per-run UUID.
    Round-trip tests assert the real model's follow-up output contains
    that UUID — only possible if the Observation turn was actually
    ingested by the live KV cache.
  - Tool spies wrap the default tools so every execute() is recorded
    with exact args; denial/skip paths assert call_count == 0.
  - Filesystem side-effect validation on T3 (Edit): the edited command
    `touch /tmp/tq_t3_mark` creates a real file; the original dangerous
    command `touch /tmp/tq_t3_ORIGINAL` must NOT be created. Asserts
    both on disk, not just the spy.

Reports:
  tests/integration_reports/agent_modes_functional_report.md
  tests/integration_reports/agent_modes_functional_report.json
"""

from __future__ import annotations

import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqcli.config import TqConfig  # noqa: E402
from tqcli.core.agent_orchestrator import (  # noqa: E402
    MODE_TINKERING,
    MODE_UNRESTRICTED,
    AgentOrchestrator,
    OrchestratorConfig,
)
from tqcli.core.agent_tools import (  # noqa: E402
    AgentTool,
    FileReadTool,
    FileWriteTool,
    InteractivePromptTool,
    TerminalExecTool,
)
from tqcli.core.engine import ChatMessage, InferenceEngine, InferenceStats  # noqa: E402
from tqcli.core.kv_quantizer import (  # noqa: E402
    check_turboquant_compatibility,
    get_llama_kv_params,
    select_kv_quant,
)
from tqcli.core.model_registry import BUILTIN_PROFILES, ModelRegistry  # noqa: E402
from tqcli.core.system_info import detect_system  # noqa: E402
from tqcli.core.vllm_config import build_vllm_config  # noqa: E402

REPORT_DIR = Path(__file__).parent / "integration_reports"

GEMMA_LLAMA = "gemma-4-e2b-it-Q4_K_M"
QWEN_LLAMA = "qwen3-4b-Q4_K_M"
GEMMA_VLLM = "gemma-4-e2b-it-vllm"
QWEN_VLLM = "qwen3-4b-vllm"

FIXTURE = Path("/tmp/tqcli_agent_fixture.txt")
T3_MARK_EDITED = Path("/tmp/tq_t3_mark")
T3_MARK_ORIGINAL = Path("/tmp/tq_t3_ORIGINAL")


# ─── Playback engine ────────────────────────────────────────────────


class PlaybackEngine(InferenceEngine):
    """Wrap a real engine, replay scripted chunks first, then delegate."""

    def __init__(self, real_engine: InferenceEngine, scripted: list[str]):
        self.real = real_engine
        self.scripted = list(scripted)
        self.call_count = 0
        self.real_call_count = 0
        self.scripted_call_count = 0

    @property
    def engine_name(self) -> str:
        return f"playback+{self.real.engine_name}"

    @property
    def is_available(self) -> bool:
        return self.real.is_available

    @property
    def is_loaded(self) -> bool:
        return self.real.is_loaded

    def load_model(self, model_path: str, **kw) -> None:
        self.real.load_model(model_path, **kw)

    def unload_model(self) -> None:
        self.real.unload_model()

    def chat(self, messages, **kw):
        raise NotImplementedError("playback engine only exposes chat_stream")

    def complete(self, prompt: str, **kw):
        raise NotImplementedError

    def chat_stream(
        self, messages, **kw
    ) -> Iterator[tuple[str, InferenceStats | None]]:
        self.call_count += 1
        if self.scripted:
            self.scripted_call_count += 1
            reply = self.scripted.pop(0)
            yield reply, None
            yield "", InferenceStats(
                prompt_tokens=1,
                completion_tokens=max(1, len(reply) // 4),
                total_tokens=1 + max(1, len(reply) // 4),
                total_time_s=0.01,
            )
            return
        self.real_call_count += 1
        yield from self.real.chat_stream(messages, **kw)


# ─── Spy tools ──────────────────────────────────────────────────────


class SpyTool(AgentTool):
    """Wrap an underlying AgentTool; record calls + delegate."""

    def __init__(self, inner: AgentTool):
        self._inner = inner
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def description(self) -> str:
        return self._inner.description

    @property
    def safety(self) -> str:
        return self._inner.safety

    @property
    def arg_schema(self) -> dict:
        return self._inner.arg_schema

    def execute(self, args: dict) -> str:
        self.calls.append(dict(args))
        return self._inner.execute(args)


def make_spy_tools() -> tuple[dict[str, SpyTool], list[SpyTool]]:
    spies = {
        "tq-file-read": SpyTool(FileReadTool()),
        "tq-file-write": SpyTool(FileWriteTool()),
        "tq-terminal-exec": SpyTool(TerminalExecTool()),
        "tq-interactive-prompt": SpyTool(InteractivePromptTool()),
    }
    return spies, list(spies.values())


# ─── Results dataclasses ────────────────────────────────────────────


@dataclass
class StepResult:
    name: str
    passed: bool
    duration_s: float = 0.0
    details: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass
class TestResult:
    test_id: str
    title: str
    engine_label: str
    model: str
    mode: str
    kv_quant: str = ""
    started: str = ""
    finished: str = ""
    total_duration_s: float = 0.0
    steps: list[StepResult] = field(default_factory=list)
    passed: bool = False
    is_data_point: bool = False  # zero-shot tests: not a pass gate

    def add(self, s: StepResult):
        self.steps.append(s)

    @property
    def pass_count(self) -> int:
        return sum(1 for s in self.steps if s.passed)

    @property
    def fail_count(self) -> int:
        return sum(1 for s in self.steps if not s.passed)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# ─── Fixture setup ──────────────────────────────────────────────────


def make_fixture() -> str:
    secret = f"ALPHACHARLIE-{uuid.uuid4().hex[:12]}"
    FIXTURE.write_text(f"secret_word={secret}\n", encoding="utf-8")
    return secret


def clear_t3_marks():
    for p in (T3_MARK_EDITED, T3_MARK_ORIGINAL):
        try:
            p.unlink()
        except FileNotFoundError:
            pass


# ─── Engine loaders (one cold load per engine/model) ───────────────


def load_llama_engine(model_id: str, profile):
    from tqcli.core.llama_backend import LlamaBackend

    sys_info = detect_system()
    tq_avail, _ = check_turboquant_compatibility(sys_info)
    kv_level = select_kv_quant(available_kv_mb=50, engine="llama.cpp", user_choice="turbo3")
    kv_params = get_llama_kv_params(kv_level) if tq_avail else {}
    eng = LlamaBackend(
        n_ctx=2048,
        n_gpu_layers=-1,
        n_threads=0,
        cache_type_k=kv_params.get("cache_type_k", "f16"),
        cache_type_v=kv_params.get("cache_type_v", "f16"),
    )
    eng.load_model(str(profile.local_path))
    return eng, kv_params


def load_vllm_engine(
    model_id: str,
    profile,
    kv_quant_choice: str = "turbo3",
    max_len: int = 2048,
):
    """Load a vLLM model with the requested KV quant level.

    The TurboQuant vLLM fork requires a `turboquant_kv.json` metadata file
    next to the model. Gemma 4 E2B's vLLM dir ships it; Qwen 3 4B's does
    not. Callers who know their model lacks the metadata should pass
    `kv_quant_choice="none"` and record in the report that this is an
    environment-level prerequisite gap, NOT an agent-feature failure.

    `max_len` is tuned per-model to fit 4 GB VRAM: Qwen 3 4B is larger
    than Gemma 4 E2B and the KV cache budget at max_model_len=2048
    exceeds the residual VRAM after BNB_INT4 + CPU offload, so Qwen
    callers pass a smaller value (896 matches vLLM's own suggestion).
    """
    from tqcli.core.vllm_backend import VllmBackend

    sys_info = detect_system()
    tune = build_vllm_config(
        profile, sys_info, requested_max_len=max_len, kv_quant_choice=kv_quant_choice
    )
    if not tune.feasible:
        raise RuntimeError(f"vLLM config infeasible: {tune.reason}")
    eng = VllmBackend.from_tuning_profile(tune)
    eng.load_model(str(profile.local_path))
    return eng, tune


_SCAN_DONE = False


def _ensure_registry_scanned():
    global _SCAN_DONE
    if _SCAN_DONE:
        return
    cfg = TqConfig.load()
    cfg.ensure_dirs()
    reg = ModelRegistry(cfg.models_dir)
    reg.scan_local_models()  # mutates BUILTIN_PROFILES in place with local_path
    _SCAN_DONE = True


def get_profile(model_id: str):
    _ensure_registry_scanned()
    for p in BUILTIN_PROFILES:
        if p.id == model_id:
            if p.local_path is None:
                raise RuntimeError(
                    f"{model_id} has no local_path after scan — model not installed at "
                    f"{TqConfig.load().models_dir}"
                )
            return p
    raise RuntimeError(f"{model_id} not in BUILTIN_PROFILES")


# ─── Scenario implementations ──────────────────────────────────────


def _pb_tag(kind: str, name: str, args: dict) -> str:
    """Protocol-compliant tag string."""
    payload = json.dumps({"name": name, "arguments": args})
    tag = "staged_tool_call" if kind == "staged" else "tool_call"
    return f"<{tag}>{payload}</{tag}>"


def _prompt_for_tag(mode: str) -> str:
    """User message that strongly instructs the real model to emit a tag."""
    tag = "staged_tool_call" if mode == MODE_TINKERING else "tool_call"
    return (
        f"Use the tq-file-read tool to read {FIXTURE}. "
        f"Reply with EXACTLY one <{tag}> block containing JSON and nothing "
        f"else.\nFormat: <{tag}>" +
        '{"name":"tq-file-read","arguments":{"path":"%s"}}' % FIXTURE +
        f"</{tag}>"
    )


def run_T0_zero_shot(test_id: str, engine_label: str, model_id: str,
                     mode: str, engine: InferenceEngine, kv_desc: str,
                     title_suffix: str = "") -> TestResult:
    """Data point: does the real model emit a tag without playback?"""
    r = TestResult(
        test_id=test_id,
        title=f"T0 zero-shot ({mode}, {engine_label} {model_id}){title_suffix}",
        engine_label=engine_label, model=model_id, mode=mode,
        kv_quant=kv_desc, started=_now(), is_data_point=True,
    )
    tag_name = "staged_tool_call" if mode == MODE_TINKERING else "tool_call"
    history = [
        ChatMessage(role="system", content="Be concise. Emit exactly one tool_call tag."),
        ChatMessage(role="user", content=_prompt_for_tag(mode)),
    ]
    t0 = time.time()
    buf = ""
    try:
        for chunk, stats in engine.chat_stream(history, max_tokens=256):
            if stats:
                break
            buf += chunk
        elapsed = time.time() - t0
        emitted = f"<{tag_name}>" in buf
        # Data point — passes regardless of whether tag was emitted; report
        # captures the raw output so a reader can audit compliance themselves.
        r.add(StepResult(
            name="zero_shot_data_point",
            passed=True,
            duration_s=elapsed,
            details=(
                f"emitted_{tag_name}={emitted} | chars={len(buf)} | "
                f"head={buf[:160].replace(chr(10),' ')}"
            ),
            metrics={
                "emitted_tag": emitted,
                "chars": len(buf),
                "tag_name": tag_name,
                "raw_output": buf,
            },
        ))
    except Exception as exc:
        r.add(StepResult(
            name="zero_shot_data_point", passed=False,
            duration_s=time.time() - t0,
            details=f"engine error: {type(exc).__name__}: {exc}",
        ))

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


def run_T1_approve_safe(test_id: str, engine_label: str, model_id: str,
                        engine: InferenceEngine, kv_desc: str,
                        turboquant_assertion: Callable[[TestResult], None] | None = None) -> TestResult:
    """ai_tinkering: APPROVE actionable tool (tq-file-read) + secret-word round-trip.

    Note: `tq-file-read` is classified `actionable` (not `safe`) because a
    read can exfiltrate secrets like `~/.ssh` or `.env`. So ai_tinkering
    DOES surface a confirmation for it — this test approves it and verifies
    the approve→execute→observation→live-follow-up chain end-to-end.
    """
    r = TestResult(
        test_id=test_id,
        title=f"T1 approve actionable ({engine_label} {model_id}, ai_tinkering)",
        engine_label=engine_label, model=model_id, mode=MODE_TINKERING,
        kv_quant=kv_desc, started=_now(),
    )
    secret = make_fixture()
    scripted = _pb_tag("staged", "tq-file-read", {"path": str(FIXTURE)})
    pb = PlaybackEngine(engine, [scripted])
    spies, spy_list = make_spy_tools()

    confirm_calls = []

    def approve(name, args, safety):
        confirm_calls.append((name, args, safety))
        return "y", args

    cfg = OrchestratorConfig(
        mode=MODE_TINKERING, max_steps=2, tools=spy_list,
        confirm_fn=approve,
    )
    orch = AgentOrchestrator(engine=pb, config=cfg)
    history = [
        ChatMessage(role="system", content="You are helpful."),
        ChatMessage(role="user", content="Please read the fixture."),
    ]
    hist_before = len(history)

    t0 = time.time()
    try:
        final_text, hist = orch.run_turn(history, max_tokens=256)
        elapsed = time.time() - t0

        # Assertion A: spy fidelity — file-read called exactly once with the fixture path.
        freads = spies["tq-file-read"].calls
        r.add(StepResult(
            name="spy_file_read_called_once",
            passed=(len(freads) == 1 and freads[0].get("path") == str(FIXTURE)),
            duration_s=0.0,
            details=f"calls={freads}",
            metrics={"call_count": len(freads)},
        ))

        # Assertion B: confirm_fn fired exactly once for actionable file-read
        # and the orchestrator honored the 'y' response (not 'n', not 'edit').
        r.add(StepResult(
            name="actionable_tool_confirmed_once",
            passed=(len(confirm_calls) == 1 and confirm_calls[0][0] == "tq-file-read"),
            duration_s=0.0,
            details=f"confirm_fn_calls={confirm_calls}",
        ))

        # Assertion C: history grew by at least 2 (assistant + observation)
        # and the observation turn is the expected shape.
        obs_turns = [m for m in hist if m.role == "user" and m.content.startswith("Observation:")]
        r.add(StepResult(
            name="observation_appended",
            passed=(len(obs_turns) >= 1 and f"secret_word={secret}" in obs_turns[0].content),
            duration_s=0.0,
            details=(
                f"hist_len={len(hist)} / hist_before={hist_before} / "
                f"obs_count={len(obs_turns)} / obs_head={obs_turns[0].content[:160] if obs_turns else 'none'}"
            ),
            metrics={"observation_turns": len(obs_turns)},
        ))

        # Assertion D: playback engine invoked real engine at least once AFTER the scripted turn.
        r.add(StepResult(
            name="real_engine_invoked_after_observation",
            passed=(pb.real_call_count >= 1),
            duration_s=0.0,
            details=(
                f"pb.call_count={pb.call_count} / scripted={pb.scripted_call_count} / "
                f"real={pb.real_call_count}"
            ),
            metrics={"real_calls": pb.real_call_count, "scripted_calls": pb.scripted_call_count},
        ))

        # Assertion E (THE assertion): secret word ingested into live KV.
        secret_in_followup = secret in final_text
        r.add(StepResult(
            name="secret_word_in_real_followup",
            passed=secret_in_followup,
            duration_s=elapsed,
            details=(
                f"secret={secret} | found_in_final={secret_in_followup} | "
                f"final_len={len(final_text)} | final_head={final_text[:240].replace(chr(10),' ')}"
            ),
            metrics={
                "secret_found": secret_in_followup,
                "secret_word": secret,
                "final_text_chars": len(final_text),
            },
        ))

        if turboquant_assertion is not None:
            turboquant_assertion(r)

    except Exception as exc:
        r.add(StepResult(
            name="harness_crashed", passed=False,
            duration_s=time.time() - t0,
            details=f"{type(exc).__name__}: {exc}",
        ))

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


def run_T2_deny(engine, kv_desc: str, model_id: str) -> TestResult:
    """ai_tinkering DENY actionable tool."""
    r = TestResult(
        test_id="T2", title="T2 deny actionable (llama.cpp Gemma 4, ai_tinkering)",
        engine_label="llama.cpp", model=model_id, mode=MODE_TINKERING,
        kv_quant=kv_desc, started=_now(),
    )
    scripted = _pb_tag(
        "staged", "tq-terminal-exec",
        {"command": "echo SHOULD_NOT_RUN"},
    )
    pb = PlaybackEngine(engine, [scripted])
    spies, spy_list = make_spy_tools()
    confirm_calls = []

    def deny(name, args, safety):
        confirm_calls.append((name, args, safety))
        return "n", args

    cfg = OrchestratorConfig(
        mode=MODE_TINKERING, max_steps=3, tools=spy_list, confirm_fn=deny,
    )
    orch = AgentOrchestrator(engine=pb, config=cfg)
    history = [
        ChatMessage(role="system", content="ok"),
        ChatMessage(role="user", content="run that command"),
    ]

    t0 = time.time()
    try:
        _, hist = orch.run_turn(history, max_tokens=64)
        elapsed = time.time() - t0

        r.add(StepResult(
            name="confirm_fn_called_once",
            passed=(len(confirm_calls) == 1),
            details=f"confirm_calls={len(confirm_calls)}",
        ))
        r.add(StepResult(
            name="terminal_exec_never_called",
            passed=(len(spies["tq-terminal-exec"].calls) == 0),
            details=f"calls={spies['tq-terminal-exec'].calls}",
        ))
        obs = [m for m in hist if m.role == "user" and m.content.startswith("Observation:")]
        r.add(StepResult(
            name="observation_is_denial",
            passed=(len(obs) >= 1 and "User denied" in obs[0].content),
            details=f"obs_head={obs[0].content[:160] if obs else 'none'}",
        ))
        r.add(StepResult(
            name="loop_terminated_after_denial",
            passed=(pb.real_call_count == 0 and pb.scripted_call_count == 1),
            duration_s=elapsed,
            details=f"scripted={pb.scripted_call_count} real={pb.real_call_count}",
        ))
    except Exception as exc:
        r.add(StepResult(
            name="harness_crashed", passed=False,
            duration_s=time.time() - t0,
            details=f"{type(exc).__name__}: {exc}",
        ))

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


def run_T3_edit(engine, kv_desc: str, model_id: str) -> TestResult:
    """ai_tinkering EDIT actionable tool — rewrite to benign command."""
    r = TestResult(
        test_id="T3", title="T3 edit actionable (llama.cpp Gemma 4, ai_tinkering)",
        engine_label="llama.cpp", model=model_id, mode=MODE_TINKERING,
        kv_quant=kv_desc, started=_now(),
    )
    clear_t3_marks()
    scripted = _pb_tag(
        "staged", "tq-terminal-exec",
        {"command": f"touch {T3_MARK_ORIGINAL}"},
    )
    pb = PlaybackEngine(engine, [scripted])
    spies, spy_list = make_spy_tools()

    def edit_to_benign(name, args, safety):
        return "y", {"command": f"touch {T3_MARK_EDITED}"}

    cfg = OrchestratorConfig(
        mode=MODE_TINKERING, max_steps=2, tools=spy_list, confirm_fn=edit_to_benign,
    )
    orch = AgentOrchestrator(engine=pb, config=cfg)
    history = [
        ChatMessage(role="system", content="ok"),
        ChatMessage(role="user", content="do the thing"),
    ]

    t0 = time.time()
    try:
        _, hist = orch.run_turn(history, max_tokens=64)
        elapsed = time.time() - t0

        # Spy: called once with EDITED args
        spy_calls = spies["tq-terminal-exec"].calls
        r.add(StepResult(
            name="exec_called_once_with_edited_args",
            passed=(len(spy_calls) == 1 and str(T3_MARK_EDITED) in spy_calls[0].get("command", "")),
            details=f"calls={spy_calls}",
        ))
        # Filesystem: edited mark exists, original does NOT
        r.add(StepResult(
            name="edited_mark_exists",
            passed=T3_MARK_EDITED.exists(),
            details=f"path={T3_MARK_EDITED} exists={T3_MARK_EDITED.exists()}",
        ))
        r.add(StepResult(
            name="original_mark_never_created",
            passed=(not T3_MARK_ORIGINAL.exists()),
            duration_s=elapsed,
            details=f"path={T3_MARK_ORIGINAL} exists={T3_MARK_ORIGINAL.exists()}",
        ))
    except Exception as exc:
        r.add(StepResult(
            name="harness_crashed", passed=False,
            duration_s=time.time() - t0,
            details=f"{type(exc).__name__}: {exc}",
        ))
    finally:
        clear_t3_marks()

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


def run_T4_react(test_id: str, engine_label: str, model_id: str,
                 engine: InferenceEngine, kv_desc: str,
                 turboquant_assertion: Callable[[TestResult], None] | None = None) -> TestResult:
    """unrestricted ReAct chain + secret-word."""
    r = TestResult(
        test_id=test_id,
        title=f"T4 unrestricted ReAct ({engine_label} {model_id})",
        engine_label=engine_label, model=model_id, mode=MODE_UNRESTRICTED,
        kv_quant=kv_desc, started=_now(),
    )
    secret = make_fixture()
    scripted = _pb_tag("live", "tq-file-read", {"path": str(FIXTURE)})
    pb = PlaybackEngine(engine, [scripted])
    spies, spy_list = make_spy_tools()
    confirm_calls = []

    def never_called(name, args, safety):
        confirm_calls.append((name, args, safety))
        raise AssertionError("unrestricted must not prompt")

    cfg = OrchestratorConfig(
        mode=MODE_UNRESTRICTED, max_steps=2, tools=spy_list, confirm_fn=never_called,
    )
    orch = AgentOrchestrator(engine=pb, config=cfg)
    history = [
        ChatMessage(role="system", content="ok"),
        ChatMessage(role="user", content="read it"),
    ]

    t0 = time.time()
    try:
        final_text, hist = orch.run_turn(history, max_tokens=256)
        elapsed = time.time() - t0

        r.add(StepResult(
            name="confirm_fn_never_called",
            passed=(len(confirm_calls) == 0),
            details=f"confirm_calls={len(confirm_calls)}",
        ))
        r.add(StepResult(
            name="file_read_called_once",
            passed=(len(spies["tq-file-read"].calls) == 1),
            details=f"calls={spies['tq-file-read'].calls}",
        ))
        obs = [m for m in hist if m.role == "user" and m.content.startswith("Observation:")]
        r.add(StepResult(
            name="observation_carries_secret",
            passed=(len(obs) >= 1 and f"secret_word={secret}" in obs[0].content),
            details=f"obs_head={obs[0].content[:160] if obs else 'none'}",
        ))
        r.add(StepResult(
            name="real_engine_followup_invoked",
            passed=(pb.real_call_count >= 1),
            details=f"real={pb.real_call_count} scripted={pb.scripted_call_count}",
        ))
        secret_ingested = secret in final_text
        r.add(StepResult(
            name="secret_word_in_real_followup",
            passed=secret_ingested,
            duration_s=elapsed,
            details=(
                f"secret={secret} / found_in_final={secret_ingested} / "
                f"final_head={final_text[:240].replace(chr(10),' ')}"
            ),
            metrics={"secret_found": secret_ingested, "secret_word": secret},
        ))
        if turboquant_assertion is not None:
            turboquant_assertion(r)

    except Exception as exc:
        r.add(StepResult(
            name="harness_crashed", passed=False,
            duration_s=time.time() - t0,
            details=f"{type(exc).__name__}: {exc}",
        ))

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


def run_T5_dead_tool(engine, kv_desc: str, model_id: str) -> TestResult:
    """unrestricted: nonexistent tool → observation captures the error."""
    r = TestResult(
        test_id="T5", title="T5 dead tool name (llama.cpp Gemma 4, unrestricted)",
        engine_label="llama.cpp", model=model_id, mode=MODE_UNRESTRICTED,
        kv_quant=kv_desc, started=_now(),
    )
    scripted = _pb_tag("live", "nonexistent_tool", {"x": 1})
    pb = PlaybackEngine(engine, [scripted])
    spies, spy_list = make_spy_tools()
    cfg = OrchestratorConfig(mode=MODE_UNRESTRICTED, max_steps=2, tools=spy_list)
    orch = AgentOrchestrator(engine=pb, config=cfg)
    history = [
        ChatMessage(role="system", content="ok"),
        ChatMessage(role="user", content="do something"),
    ]

    t0 = time.time()
    try:
        _, hist = orch.run_turn(history, max_tokens=128)
        elapsed = time.time() - t0

        # No real tool was called.
        total_tool_calls = sum(len(s.calls) for s in spy_list)
        r.add(StepResult(
            name="no_real_tool_invoked",
            passed=(total_tool_calls == 0),
            details=f"total_spy_calls={total_tool_calls}",
        ))
        obs = [m for m in hist if m.role == "user" and m.content.startswith("Observation:")]
        r.add(StepResult(
            name="observation_is_error",
            passed=(len(obs) >= 1 and "unknown tool" in obs[0].content),
            details=f"obs_head={obs[0].content[:160] if obs else 'none'}",
        ))
        r.add(StepResult(
            name="orchestrator_did_not_crash",
            passed=True,
            duration_s=elapsed,
            details="ran to completion",
        ))
    except Exception as exc:
        r.add(StepResult(
            name="harness_crashed", passed=False,
            duration_s=time.time() - t0,
            details=f"{type(exc).__name__}: {exc}",
        ))

    r.finished = _now()
    r.total_duration_s = sum(s.duration_s for s in r.steps)
    r.passed = all(s.passed for s in r.steps)
    return r


# ─── Orchestration — engine groups ─────────────────────────────────


def _llama_kv_desc(kv_params: dict) -> str:
    return f"turbo3 (cache_type_k={kv_params.get('cache_type_k')}, cache_type_v={kv_params.get('cache_type_v')})"


def _vllm_kv_desc(tune) -> str:
    return f"{tune.kv_cache_dtype} (quant={tune.quantization}, cpu_offload_gb={tune.cpu_offload_gb})"


def _vllm_turboquant_asserter(tune):
    def inner(r: TestResult):
        r.add(StepResult(
            name="turboquant_kv_active",
            passed=(tune.kv_cache_dtype == "turboquant35"),
            details=f"tune.kv_cache_dtype={tune.kv_cache_dtype}",
        ))
    return inner


def _llama_turboquant_asserter(kv_params: dict):
    def inner(r: TestResult):
        ok = kv_params.get("cache_type_k") == "turbo3" and kv_params.get("cache_type_v") == "turbo3"
        r.add(StepResult(
            name="turboquant_kv_active",
            passed=ok,
            details=f"kv_params={kv_params}",
        ))
    return inner


def run_llama_group() -> list[TestResult]:
    results: list[TestResult] = []
    # Gemma 4 E2B path
    gemma_profile = get_profile(GEMMA_LLAMA)
    print(f"\n[llama.cpp] loading {GEMMA_LLAMA}...", flush=True)
    eng_g, kv_params_g = load_llama_engine(GEMMA_LLAMA, gemma_profile)
    kv_desc_g = _llama_kv_desc(kv_params_g)
    tq_assert_g = _llama_turboquant_asserter(kv_params_g)

    try:
        print("[llama.cpp Gemma] T0a zero-shot", flush=True)
        results.append(run_T0_zero_shot("T0a", "llama.cpp", GEMMA_LLAMA, MODE_TINKERING, eng_g, kv_desc_g))
        print("[llama.cpp Gemma] T1_lg approve safe + secret", flush=True)
        results.append(run_T1_approve_safe("T1_lg", "llama.cpp", GEMMA_LLAMA, eng_g, kv_desc_g, tq_assert_g))
        print("[llama.cpp Gemma] T2 deny", flush=True)
        results.append(run_T2_deny(eng_g, kv_desc_g, GEMMA_LLAMA))
        print("[llama.cpp Gemma] T3 edit", flush=True)
        results.append(run_T3_edit(eng_g, kv_desc_g, GEMMA_LLAMA))
        print("[llama.cpp Gemma] T4_lg react + secret", flush=True)
        results.append(run_T4_react("T4_lg", "llama.cpp", GEMMA_LLAMA, eng_g, kv_desc_g, tq_assert_g))
        print("[llama.cpp Gemma] T5 dead tool", flush=True)
        results.append(run_T5_dead_tool(eng_g, kv_desc_g, GEMMA_LLAMA))
    finally:
        eng_g.unload_model()

    # Qwen 3 4B path
    qwen_profile = get_profile(QWEN_LLAMA)
    print(f"\n[llama.cpp] loading {QWEN_LLAMA}...", flush=True)
    eng_q, kv_params_q = load_llama_engine(QWEN_LLAMA, qwen_profile)
    kv_desc_q = _llama_kv_desc(kv_params_q)
    tq_assert_q = _llama_turboquant_asserter(kv_params_q)
    try:
        print("[llama.cpp Qwen] T0b zero-shot", flush=True)
        results.append(run_T0_zero_shot("T0b", "llama.cpp", QWEN_LLAMA, MODE_TINKERING, eng_q, kv_desc_q))
        print("[llama.cpp Qwen] T1_lq approve safe + secret", flush=True)
        results.append(run_T1_approve_safe("T1_lq", "llama.cpp", QWEN_LLAMA, eng_q, kv_desc_q, tq_assert_q))
        print("[llama.cpp Qwen] T4_lq react + secret", flush=True)
        results.append(run_T4_react("T4_lq", "llama.cpp", QWEN_LLAMA, eng_q, kv_desc_q, tq_assert_q))
    finally:
        eng_q.unload_model()

    return results


def run_vllm_group() -> list[TestResult]:
    results: list[TestResult] = []
    # Gemma 4 E2B
    gemma_profile = get_profile(GEMMA_VLLM)
    print(f"\n[vLLM] loading {GEMMA_VLLM}...", flush=True)
    try:
        eng_g, tune_g = load_vllm_engine(GEMMA_VLLM, gemma_profile)
    except Exception as exc:
        r = TestResult(
            test_id="vllm_gemma_load_failed",
            title=f"vLLM {GEMMA_VLLM} load failed",
            engine_label="vllm", model=GEMMA_VLLM, mode="n/a",
            started=_now(), finished=_now(),
        )
        r.add(StepResult(name="vllm_load", passed=False,
                         details=f"{type(exc).__name__}: {exc}"))
        r.passed = False
        return [r]

    kv_desc_g = _vllm_kv_desc(tune_g)
    tq_assert_g = _vllm_turboquant_asserter(tune_g)
    try:
        print("[vLLM Gemma] T0c zero-shot", flush=True)
        results.append(run_T0_zero_shot("T0c", "vllm", GEMMA_VLLM, MODE_UNRESTRICTED, eng_g, kv_desc_g))
        print("[vLLM Gemma] T1_vg approve safe + secret", flush=True)
        results.append(run_T1_approve_safe("T1_vg", "vllm", GEMMA_VLLM, eng_g, kv_desc_g, tq_assert_g))
        print("[vLLM Gemma] T4_vg react + secret", flush=True)
        results.append(run_T4_react("T4_vg", "vllm", GEMMA_VLLM, eng_g, kv_desc_g, tq_assert_g))
    finally:
        try:
            eng_g.unload_model()
        except Exception:
            pass

    # Qwen 3 4B on vLLM — TurboQuant-vLLM fork requires turboquant_kv.json
    # alongside the model dir; Qwen 3 4B's snapshot does not ship that
    # metadata, so we fall back to `kv_quant_choice="none"` to exercise
    # the agent orchestrator path. This is an environment prerequisite gap
    # (file the user can generate / request from the fork maintainer), NOT
    # an agent-feature defect.
    qwen_profile = get_profile(QWEN_VLLM)
    print(f"\n[vLLM] loading {QWEN_VLLM} (kv_quant=none, max_len=896; metadata missing for turboquant35 + 4GB VRAM budget)...", flush=True)
    try:
        eng_q, tune_q = load_vllm_engine(QWEN_VLLM, qwen_profile, kv_quant_choice="none", max_len=896)
    except Exception as exc:
        r = TestResult(
            test_id="vllm_qwen_load_failed",
            title=f"vLLM {QWEN_VLLM} load failed",
            engine_label="vllm", model=QWEN_VLLM, mode="n/a",
            started=_now(), finished=_now(),
        )
        r.add(StepResult(name="vllm_load", passed=False,
                         details=f"{type(exc).__name__}: {exc}"))
        r.passed = False
        return results + [r]

    kv_desc_q = _vllm_kv_desc(tune_q) + "  [kv:none — turboquant_kv.json not present for Qwen3 4B]"
    # No turboquant assertion for Qwen vLLM — running with kv:none by design
    # (environment prerequisite gap). Agent orchestrator semantics are still
    # fully exercised.
    try:
        print("[vLLM Qwen] T0d zero-shot", flush=True)
        results.append(run_T0_zero_shot("T0d", "vllm", QWEN_VLLM, MODE_UNRESTRICTED, eng_q, kv_desc_q))
        print("[vLLM Qwen] T1_vq approve actionable + secret", flush=True)
        results.append(run_T1_approve_safe("T1_vq", "vllm", QWEN_VLLM, eng_q, kv_desc_q, None))
        print("[vLLM Qwen] T4_vq react + secret", flush=True)
        results.append(run_T4_react("T4_vq", "vllm", QWEN_VLLM, eng_q, kv_desc_q, None))
    finally:
        try:
            eng_q.unload_model()
        except Exception:
            pass

    return results


# ─── Report generation ─────────────────────────────────────────────


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


def generate_markdown(results: list[TestResult], sys_info: dict) -> str:
    lines = [
        "# Agent Modes — FUNCTIONAL Integration Test Report",
        "",
        f"**Generated:** {datetime.now().isoformat(timespec='seconds')}",
        f"**System:** {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)",
        f"**CUDA:** driver {sys_info.get('cuda_version')}, toolkit {sys_info.get('cuda_toolkit')}",
        f"**OS:** {sys_info['os']}",
        "",
        "Exercises the full parse→execute→observation→live-inference loop of "
        "`tqcli/core/agent_orchestrator.py` with concrete assertions — spy "
        "fidelity, history integrity, filesystem side-effects, and "
        "secret-word ingestion into the live KV cache. Zero-shot tests (T0*) "
        "are DATA POINTS capturing real-model tag-emission compliance; they "
        "do not gate the suite.",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Test | Engine | Model | Mode | KV Quant | Kind | Result |",
        "|------|--------|-------|------|----------|------|--------|",
    ]
    for r in results:
        kind = "data-point" if r.is_data_point else "functional"
        status = f"**PASS** ({r.pass_count}/{r.pass_count + r.fail_count})" if r.passed else f"**FAIL** ({r.pass_count}/{r.pass_count + r.fail_count})"
        lines.append(
            f"| {r.test_id} | {r.engine_label} | {r.model} | {r.mode} | "
            f"{r.kv_quant} | {kind} | {status} |"
        )
    lines.extend(["", "---", ""])
    for r in results:
        lines.extend([
            f"## {r.test_id}: {r.title}",
            "",
            f"- **Engine:** {r.engine_label}",
            f"- **Model:** {r.model}",
            f"- **Mode:** {r.mode}",
            f"- **KV Quant:** {r.kv_quant}",
            f"- **Kind:** {'DATA POINT (no pass gate)' if r.is_data_point else 'FUNCTIONAL (pass gated by assertions)'}",
            f"- **Started / Finished:** {r.started} → {r.finished}",
            f"- **Duration:** {r.total_duration_s:.2f}s",
            f"- **Result:** {'PASS' if r.passed else 'FAIL'} ({r.pass_count}/{r.pass_count + r.fail_count})",
            "",
            "| Step | Result | Duration | Details |",
            "|------|--------|----------|---------|",
        ])
        for s in r.steps:
            status = "PASS" if s.passed else "FAIL"
            det = s.details[:220].replace("|", "/")
            lines.append(f"| {s.name} | {status} | {s.duration_s:.2f}s | {det} |")
        lines.append("")
    return "\n".join(lines)


def generate_json(results: list[TestResult], sys_info: dict) -> dict:
    return {
        "generated": datetime.now().isoformat(),
        "system": sys_info,
        "tests": [
            {
                "test_id": r.test_id,
                "title": r.title,
                "engine": r.engine_label,
                "model": r.model,
                "mode": r.mode,
                "kv_quant": r.kv_quant,
                "is_data_point": r.is_data_point,
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


def main() -> int:
    print("=" * 72, flush=True)
    print("Agent Modes — FUNCTIONAL Integration Tests", flush=True)
    print("=" * 72, flush=True)

    sys_info = get_system_info_dict()
    print(f"System: {sys_info['gpu']} ({sys_info['vram_mb']} MB VRAM)", flush=True)
    print(f"CUDA: driver {sys_info['cuda_version']}, toolkit {sys_info['cuda_toolkit']}", flush=True)

    results: list[TestResult] = []
    try:
        results.extend(run_llama_group())
    except Exception as exc:
        print(f"[llama.cpp group crashed] {type(exc).__name__}: {exc}", flush=True)
    try:
        results.extend(run_vllm_group())
    except Exception as exc:
        print(f"[vLLM group crashed] {type(exc).__name__}: {exc}", flush=True)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    md = generate_markdown(results, sys_info)
    (REPORT_DIR / "agent_modes_functional_report.md").write_text(md)
    js = generate_json(results, sys_info)
    (REPORT_DIR / "agent_modes_functional_report.json").write_text(json.dumps(js, indent=2))

    print(f"\n{'=' * 72}", flush=True)
    total_functional = [r for r in results if not r.is_data_point]
    total_data = [r for r in results if r.is_data_point]
    pass_functional = sum(1 for r in total_functional if r.passed)
    pass_data = sum(1 for r in total_data if r.passed)
    print(f"Functional tests: {pass_functional}/{len(total_functional)} passed", flush=True)
    print(f"Data-point runs:  {pass_data}/{len(total_data)} completed", flush=True)
    for r in results:
        tag = "  [PASS]" if r.passed else "  [FAIL]"
        print(f"{tag} {r.test_id} {r.title} — {r.pass_count}/{r.pass_count + r.fail_count} steps", flush=True)
    print(f"Reports:", flush=True)
    print(f"  {REPORT_DIR / 'agent_modes_functional_report.md'}", flush=True)
    print(f"  {REPORT_DIR / 'agent_modes_functional_report.json'}", flush=True)
    print(f"{'=' * 72}", flush=True)

    # Suite passes iff every FUNCTIONAL test passed. Data points are informational.
    return 0 if pass_functional == len(total_functional) and len(total_functional) > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
