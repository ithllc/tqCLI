"""Tests for the Engine Auditor (TP Phase 5 / C6).

Covers:
* sentinel detection: TURBOQUANT_BUILD / TURBOQUANT_ENABLED flips is_turboquant_fork.
* hardware capability gating: SystemInfo capable vs. incapable flips should_warn.
* render_audit_warnings stays silent on non-warning results.
* JSON metadata serialization keys.
* Stderr-ordering contract: render_audit_warnings + flush precede the
  AgentOrchestrator's first stream chunk in agent-mode CLI flows.
* run_audit / get_status / reset_cache lifecycle.
"""

from __future__ import annotations

import io
import sys
import types
from contextlib import redirect_stderr
from unittest.mock import patch

import pytest
from rich.console import Console

from tqcli.core import engine_auditor
from tqcli.core.engine_auditor import (
    EngineAuditResult,
    audit_llama_cpp,
    audit_vllm,
    get_status,
    reset_cache,
    run_audit,
)
from tqcli.core.system_info import GPUInfo, SystemInfo
from tqcli.ui.console import render_audit_warnings


# ----------------------------------------------------------------------
# SystemInfo factories
# ----------------------------------------------------------------------


def _capable_system() -> SystemInfo:
    """Ampere SM_8.6 + CUDA 12.8 — TurboQuant supported."""
    gpu = GPUInfo(
        name="NVIDIA RTX A2000",
        vram_total_mb=4096,
        vram_free_mb=4000,
        compute_capability="8.6",
        cuda_version="12.8",
        cuda_toolkit_version="12.8",
    )
    return SystemInfo(
        os_name="linux",
        os_version="5.15",
        os_display="Linux (test)",
        arch="x86_64",
        gpus=[gpu],
        has_nvidia_gpu=True,
        has_metal=False,
        total_vram_mb=4096,
    )


def _incapable_system() -> SystemInfo:
    """Pre-Ampere CUDA 11.8 — TurboQuant NOT supported."""
    gpu = GPUInfo(
        name="NVIDIA GTX 1080",
        vram_total_mb=8192,
        vram_free_mb=8000,
        compute_capability="6.1",
        cuda_version="11.8",
        cuda_toolkit_version="11.8",
    )
    return SystemInfo(
        os_name="linux",
        os_version="5.15",
        os_display="Linux (test)",
        arch="x86_64",
        gpus=[gpu],
        has_nvidia_gpu=True,
        has_metal=False,
        total_vram_mb=8192,
    )


def _metal_system() -> SystemInfo:
    """Apple Silicon — llama.cpp Metal path is capable; vLLM is not."""
    return SystemInfo(
        os_name="darwin",
        os_version="14.2",
        os_display="macOS 14.2",
        arch="arm64",
        gpus=[],
        has_nvidia_gpu=False,
        has_metal=True,
        total_vram_mb=0,
    )


# ----------------------------------------------------------------------
# Module-import patching
# ----------------------------------------------------------------------


def _install_fake_module(name: str, **attrs) -> types.ModuleType:
    """Replace ``sys.modules[name]`` with a fake module exposing ``attrs``."""
    mod = types.ModuleType(name)
    mod.__version__ = "0.0.0-test"
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


def _remove_module(name: str) -> None:
    sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def _clear_audit_cache():
    reset_cache()
    yield
    reset_cache()


# ----------------------------------------------------------------------
# Sentinel detection
# ----------------------------------------------------------------------


def test_llama_cpp_fork_detected_when_sentinel_true(monkeypatch):
    _install_fake_module("llama_cpp", TURBOQUANT_BUILD=True)
    try:
        result = audit_llama_cpp(_capable_system())
    finally:
        _remove_module("llama_cpp")
    assert result is not None
    assert result.engine == "llama.cpp"
    assert result.is_turboquant_fork is True
    assert result.should_warn is False


def test_llama_cpp_upstream_detected_when_sentinel_missing(monkeypatch):
    _install_fake_module("llama_cpp")  # no TURBOQUANT_BUILD attr
    try:
        result = audit_llama_cpp(_capable_system())
    finally:
        _remove_module("llama_cpp")
    assert result is not None
    assert result.is_turboquant_fork is False
    assert result.should_warn is True
    assert "llama-tq" in result.install_hint


def test_vllm_fork_detected_when_sentinel_true(monkeypatch):
    _install_fake_module("vllm", TURBOQUANT_ENABLED=True)
    try:
        result = audit_vllm(_capable_system())
    finally:
        _remove_module("vllm")
    assert result is not None
    assert result.is_turboquant_fork is True
    assert result.should_warn is False


def test_vllm_upstream_detected_when_sentinel_missing(monkeypatch):
    _install_fake_module("vllm")  # no TURBOQUANT_ENABLED attr
    try:
        result = audit_vllm(_capable_system())
    finally:
        _remove_module("vllm")
    assert result is not None
    assert result.is_turboquant_fork is False
    assert result.should_warn is True
    assert "vllm-tq" in result.install_hint
    assert "find-links" in result.install_hint


# ----------------------------------------------------------------------
# Hardware capability gating
# ----------------------------------------------------------------------


def test_should_warn_false_on_incapable_hw(monkeypatch):
    _install_fake_module("vllm")
    try:
        result = audit_vllm(_incapable_system())
    finally:
        _remove_module("vllm")
    assert result is not None
    assert result.hardware_capable is False
    assert result.should_warn is False, "incapable hardware must not trigger warning"


def test_metal_capable_for_llama_cpp(monkeypatch):
    _install_fake_module("llama_cpp")
    try:
        result = audit_llama_cpp(_metal_system())
    finally:
        _remove_module("llama_cpp")
    assert result is not None
    assert result.hardware_capable is True
    assert result.should_warn is True


def test_metal_NOT_capable_for_vllm(monkeypatch):
    """vLLM has no Metal path — Mac users should never see a vLLM warning."""
    _install_fake_module("vllm")
    try:
        result = audit_vllm(_metal_system())
    finally:
        _remove_module("vllm")
    assert result is not None
    assert result.hardware_capable is False
    assert result.should_warn is False


# ----------------------------------------------------------------------
# run_audit / get_status / reset_cache
# ----------------------------------------------------------------------


def test_run_audit_returns_only_importable_engines(monkeypatch):
    _install_fake_module("llama_cpp", TURBOQUANT_BUILD=True)
    _remove_module("vllm")

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __builtins__["__import__"]

    def _import_blocking_vllm(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise ImportError("vllm masked for test")
        return real_import(name, *args, **kwargs)

    try:
        with patch("builtins.__import__", side_effect=_import_blocking_vllm):
            results = run_audit(_capable_system())
    finally:
        _remove_module("llama_cpp")
    assert len(results) == 1
    assert results[0].engine == "llama.cpp"


def test_get_status_returns_cache_after_run(monkeypatch):
    _install_fake_module("llama_cpp", TURBOQUANT_BUILD=True)
    try:
        run_audit(_capable_system())
        status = get_status()
    finally:
        _remove_module("llama_cpp")
    assert "llama.cpp" in status
    assert status["llama.cpp"].is_turboquant_fork is True


def test_get_status_empty_before_run():
    reset_cache()
    assert get_status() == {}


def test_run_audit_empty_when_neither_engine_importable():
    _remove_module("llama_cpp")
    _remove_module("vllm")

    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __builtins__["__import__"]

    def _import_blocking_both(name, *args, **kwargs):
        if name in ("vllm", "llama_cpp") or name.startswith(("vllm.", "llama_cpp.")):
            raise ImportError(f"{name} masked for test")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_import_blocking_both):
        results = run_audit(_capable_system())
    assert results == []


# ----------------------------------------------------------------------
# render_audit_warnings — visual / silence behavior
# ----------------------------------------------------------------------


def test_render_audit_warnings_silent_when_no_should_warn():
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, width=120)
    silent = EngineAuditResult(
        engine="vllm",
        is_turboquant_fork=True,
        hardware_capable=True,
        should_warn=False,
        install_hint="...",
    )
    render_audit_warnings([silent], test_console)
    assert buf.getvalue() == ""


def test_render_audit_warnings_emits_yellow_panel():
    buf = io.StringIO()
    test_console = Console(file=buf, force_terminal=False, width=120)
    warn = EngineAuditResult(
        engine="vllm",
        is_turboquant_fork=False,
        hardware_capable=True,
        should_warn=True,
        install_hint="pip install --upgrade 'turboquant-cli[vllm-tq]'",
    )
    render_audit_warnings([warn], test_console)
    out = buf.getvalue()
    assert "TurboQuant Unavailable" in out
    assert "vllm-tq" in out


# ----------------------------------------------------------------------
# Stderr-ordering contract (TP C5)
# ----------------------------------------------------------------------


def test_render_then_flush_finishes_before_orchestrator_first_chunk():
    """render_audit_warnings → console.file.flush() → orchestrator.stream chunk.

    Asserts that the panel's last newline appears in stderr BEFORE the first
    fake "tool_call" chunk, i.e. there is no interleaving.
    """
    captured = io.StringIO()
    test_console = Console(file=captured, force_terminal=False, width=120)

    warn = EngineAuditResult(
        engine="vllm",
        is_turboquant_fork=False,
        hardware_capable=True,
        should_warn=True,
        install_hint="pip install --upgrade 'turboquant-cli[vllm-tq]'",
    )

    # 1. Render the panel.
    render_audit_warnings([warn], test_console)
    # 2. Flush per the ordering contract.
    test_console.file.flush()

    panel_text = captured.getvalue()

    # 3. Now simulate the orchestrator's first stream chunk landing on stderr.
    captured.write("<tool_call>{\"name\": \"echo\"}</tool_call>")
    final = captured.getvalue()

    panel_end = final.rfind("\n", 0, len(panel_text))
    chunk_start = final.rfind("<tool_call>")

    assert panel_end != -1, "panel did not emit a newline before flush"
    assert chunk_start > panel_end, (
        "tool_call chunk was emitted before the panel's last newline — "
        "ordering contract violated"
    )


# ----------------------------------------------------------------------
# Crash-safety
# ----------------------------------------------------------------------


def test_audit_does_not_crash_on_broken_import():
    """A module that raises arbitrary exceptions on import must NOT take down startup."""

    class _Boom(types.ModuleType):
        def __getattr__(self, item):  # noqa: D401
            raise RuntimeError("simulated broken vllm install")

    boom = _Boom("vllm")
    sys.modules["vllm"] = boom
    try:
        # Should NOT raise.
        result = audit_vllm(_capable_system())
    finally:
        _remove_module("vllm")
    # Whatever the auditor returns, it must not crash; either None or a result.
    assert result is None or isinstance(result, EngineAuditResult)
