"""Engine Auditor — fork-vs-upstream detection at startup.

Cross-references two facts at every CLI invocation:

1. **Engine identity** — does the loaded ``llama_cpp`` / ``vllm`` package carry
   the TurboQuant fork sentinel (``TURBOQUANT_BUILD`` / ``TURBOQUANT_ENABLED``)?
2. **Hardware capability** — would this box actually run the TurboQuant
   kernels (CUDA toolkit ≥ 12.8 + SM ≥ 8.6, or Apple Metal for llama.cpp)?

When the user has capable hardware but is running upstream, the audit emits a
yellow Rich panel (rendered by ``tqcli.ui.console.render_audit_warnings``) with
the exact ``pip install`` command to fix it. When either the fork is correctly
installed, the hardware can't run TurboQuant regardless, or neither engine is
importable, the audit stays silent.

The auditor never raises — every importable side-effect is wrapped — so a
broken ``vllm`` install can't crash ``tqcli`` startup. See TP Phase 5 (C3-C5)
and the Risk Register entry on Engine Auditor panel ↔ orchestrator stream
ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

from tqcli.core.kv_quantizer import check_turboquant_compatibility
from tqcli.core.system_info import SystemInfo


@dataclass
class EngineAuditResult:
    """One-engine audit outcome.

    Attributes:
        engine: ``"llama.cpp"`` or ``"vllm"``.
        is_turboquant_fork: True iff the imported module exposes the TurboQuant
            sentinel attribute (``TURBOQUANT_BUILD`` / ``TURBOQUANT_ENABLED``).
        hardware_capable: True iff this box can run TurboQuant kernels —
            CUDA toolkit ≥ 12.8 + SM ≥ 8.6, or Apple Metal (llama.cpp only).
        should_warn: True iff hardware is capable but the installed package
            is upstream (no TurboQuant build).
        install_hint: The exact ``pip install`` command to install the fork.
    """

    engine: str
    is_turboquant_fork: bool
    hardware_capable: bool
    should_warn: bool
    install_hint: str


# Module-level cache for run_audit() / get_status(); populated on first call.
# Avoids re-importing `vllm` / `llama_cpp` on every CLI tick — vLLM's import
# triggers CUDA context init (300+ ms) which is an unacceptable per-invocation
# tax.
_AUDIT_CACHE: dict[str, EngineAuditResult] | None = None


def _safe_import_attr(module_name: str, attr: str) -> object | None:
    """Import ``module_name`` and read ``attr`` without crashing on errors.

    A broken ``vllm`` install can raise arbitrary exceptions on import (CUDA
    init, missing shared libs, Pydantic schema mismatches). Catch everything
    and treat any failure as "module unavailable" — the CLI must still start.
    """
    try:
        module = __import__(module_name)
    except BaseException:  # noqa: BLE001 — broken installs raise all kinds of things
        return None
    try:
        return getattr(module, attr, None)
    except BaseException:  # noqa: BLE001 — sentinel access shouldn't crash startup
        return None


def _module_importable(module_name: str) -> bool:
    """Probe whether a module imports at all, without surfacing errors."""
    try:
        __import__(module_name)
        return True
    except BaseException:  # noqa: BLE001
        return False


def _llama_capable(system: SystemInfo) -> bool:
    """llama.cpp can use TurboQuant on Apple Metal or any CUDA-capable GPU
    where ``check_turboquant_compatibility`` passes."""
    if system.has_metal:
        return True
    if not system.has_nvidia_gpu:
        return False
    available, _ = check_turboquant_compatibility(system)
    return available


def _vllm_capable(system: SystemInfo) -> bool:
    """vLLM TurboQuant is NVIDIA-only and gated by the same CUDA / SM checks."""
    if not system.has_nvidia_gpu:
        return False
    available, _ = check_turboquant_compatibility(system)
    return available


def audit_llama_cpp(system: SystemInfo) -> EngineAuditResult | None:
    """Audit ``llama_cpp``. Returns None if the package is not importable."""
    if not _module_importable("llama_cpp"):
        return None
    sentinel = _safe_import_attr("llama_cpp", "TURBOQUANT_BUILD")
    is_fork = sentinel is True
    capable = _llama_capable(system)
    return EngineAuditResult(
        engine="llama.cpp",
        is_turboquant_fork=is_fork,
        hardware_capable=capable,
        should_warn=(capable and not is_fork),
        install_hint="pip install --upgrade 'turboquant-cli[llama-tq]'",
    )


def audit_vllm(system: SystemInfo) -> EngineAuditResult | None:
    """Audit ``vllm``. Returns None if the package is not importable."""
    if not _module_importable("vllm"):
        return None
    sentinel = _safe_import_attr("vllm", "TURBOQUANT_ENABLED")
    is_fork = sentinel is True
    capable = _vllm_capable(system)
    return EngineAuditResult(
        engine="vllm",
        is_turboquant_fork=is_fork,
        hardware_capable=capable,
        should_warn=(capable and not is_fork),
        install_hint=(
            "pip install --upgrade 'turboquant-cli[vllm-tq]' "
            "--find-links https://github.com/tqcli/vllm-turboquant/releases/latest"
        ),
    )


def run_audit(system: SystemInfo) -> list[EngineAuditResult]:
    """Run all engine audits, populate the module cache, return non-None results."""
    global _AUDIT_CACHE
    results: list[EngineAuditResult] = []
    cache: dict[str, EngineAuditResult] = {}
    for fn in (audit_llama_cpp, audit_vllm):
        try:
            result = fn(system)
        except BaseException:  # noqa: BLE001 — never let the auditor crash startup
            result = None
        if result is not None:
            results.append(result)
            cache[result.engine] = result
    _AUDIT_CACHE = cache
    return results


def get_status() -> dict[str, EngineAuditResult]:
    """Return cached audit results keyed by engine name.

    Internal API for future agent tools to report authoritative TurboQuant
    status to the LLM without re-importing ``vllm`` / ``llama_cpp``. Returns
    an empty dict if ``run_audit()`` has not been called yet — callers must
    have invoked the audit at startup.
    """
    return dict(_AUDIT_CACHE) if _AUDIT_CACHE is not None else {}


def reset_cache() -> None:
    """Clear the cached audit. Test-only — not part of the public API."""
    global _AUDIT_CACHE
    _AUDIT_CACHE = None
