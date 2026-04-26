"""Rich console output for tqCLI."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from tqcli import __version__
from tqcli.core.engine import InferenceStats
from tqcli.core.model_registry import ModelProfile
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.engine_auditor import EngineAuditResult
from tqcli.core.router import RouteDecision

console = Console()


def setup_json_logging():
    """Redirect all third-party logging and progress bars to stderr."""
    import logging
    import os
    import sys
    from functools import partial

    # 1. Environment variables (inherited by vLLM subprocesses).
    # NOTE: We deliberately do NOT set TQDM_DISABLE=1. vLLM's internals read
    # `tqdm.format_dict['rate']` for progress accounting, and a disabled tqdm
    # returns `rate=None` which triggers ZeroDivisionError inside vLLM. The
    # file=sys.stderr monkey-patch below is enough to keep stdout clean.
    os.environ["VLLM_CONFIGURE_LOGGING"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"

    # 2. Redirect root logger to stderr
    logging.basicConfig(level=logging.ERROR, stream=sys.stderr, force=True)

    # 3. Capture loggers already initialized by module-level imports
    third_party = (
        "vllm",
        "torch",
        "transformers",
        "bitsandbytes",
        "accelerate",
        "PIL",
        "urllib3",
        "nvml",
    )
    log_manager = logging.Logger.manager
    for name in list(log_manager.loggerDict.keys()):
        if name.startswith(third_party):
            logger = logging.getLogger(name)
            logger.handlers = []
            logger.propagate = True
            logger.setLevel(logging.ERROR)

    # 4. Patch tqdm just in case TQDM_DISABLE isn't respected
    try:
        import tqdm

        tqdm.tqdm = partial(tqdm.tqdm, file=sys.stderr)
    except ImportError:
        pass


def print_banner():
    banner = Text()
    banner.append("tqCLI", style="bold cyan")
    banner.append(f" v{__version__}", style="dim")
    banner.append(" — TurboQuant Local Inference", style="white")
    console.print(Panel(banner, border_style="cyan"))


def print_system_info(info):
    from tqcli.core.kv_quantizer import check_turboquant_compatibility

    table = Table(title="System Info", border_style="cyan", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")

    table.add_row("OS", info.os_display)
    table.add_row("CPU", f"{info.cpu_name} ({info.cpu_cores_physical}c/{info.cpu_cores_logical}t)")
    table.add_row("RAM", f"{info.ram_total_mb:,} MB total / {info.ram_available_mb:,} MB available")

    if info.gpus:
        for gpu in info.gpus:
            cuda_info = ""
            if gpu.cuda_version:
                cuda_info = f", CUDA {gpu.cuda_version}"
            if gpu.cuda_toolkit_version:
                cuda_info += f", toolkit {gpu.cuda_toolkit_version}"
            table.add_row("GPU", f"{gpu.name} ({gpu.vram_total_mb:,} MB VRAM{cuda_info})")
    elif info.has_metal:
        table.add_row("GPU", "Apple Silicon (Metal)")
    else:
        table.add_row("GPU", "None detected")

    table.add_row("Engine", f"{info.recommended_engine} (recommended)")
    table.add_row("Max Model", f"~{info.max_model_size_estimate_gb} GB")
    table.add_row("Quant", f"{info.recommended_quant} recommended")

    # TurboQuant KV cache compatibility
    tq_available, tq_msg = check_turboquant_compatibility(info)
    if tq_available:
        table.add_row("TurboQuant KV", f"[green]available[/green]")
    else:
        # Extract the short reason (first sentence)
        short_msg = tq_msg.split(". ")[0] + "." if ". " in tq_msg else tq_msg
        table.add_row("TurboQuant KV", f"[yellow]unavailable[/yellow] ({short_msg})")

    if info.is_wsl:
        table.add_row("Environment", f"WSL{info.wsl_version}")

    console.print(table)


def print_route_decision(decision: RouteDecision):
    console.print(f"  [dim]Router:[/dim] {decision.reason}", highlight=False)


def print_stats_bar(stats: InferenceStats):
    tps = stats.tokens_per_second
    if tps >= 20:
        color = "green"
    elif tps >= 10:
        color = "yellow"
    else:
        color = "red"

    console.print(
        f"\n  [{color}]{tps:.1f} tok/s[/{color}] | "
        f"{stats.completion_tokens} tokens | "
        f"{stats.total_time_s:.2f}s",
        highlight=False,
    )


def print_performance_warning(monitor: PerformanceMonitor):
    stats = monitor.get_stats_display()
    console.print(
        Panel(
            f"[yellow]Performance Warning[/yellow]\n"
            f"Rolling avg: {stats['rolling_tps']} tok/s (threshold: {stats['threshold_tps']})\n"
            f"Consider: tqcli handoff --target claude-code",
            border_style="yellow",
        )
    )


def print_handoff_alert(filepath):
    console.print(
        Panel(
            f"[red]Performance Below Threshold[/red]\n\n"
            f"Handoff file generated:\n"
            f"  {filepath}\n\n"
            f"To continue with a frontier model:\n"
            f"  claude  (then reference the handoff file)",
            border_style="red",
        )
    )


def print_model_list(models: list[ModelProfile], title: str = "Models"):
    table = Table(title=title, border_style="cyan")
    table.add_column("ID", style="bold")
    table.add_column("Params")
    table.add_column("Quant")
    table.add_column("Engine")
    table.add_column("Strengths")
    table.add_column("Status")

    for m in models:
        strengths = ", ".join(s.value for s in m.strengths)
        status = "[green]installed[/green]" if m.local_path else "[dim]available[/dim]"
        table.add_row(m.id, m.parameter_count, m.quantization, m.engine, strengths, status)

    console.print(table)


def print_skill_list(skills):
    table = Table(title="Available Skills", border_style="cyan")
    table.add_column("Skill", style="bold")
    table.add_column("Description")
    table.add_column("Scripts", justify="center")

    for skill in skills:
        scripts = str(len(skill.scripts)) if skill.has_scripts else "-"
        table.add_row(skill.name, skill.description[:80], scripts)

    console.print(table)


def render_audit_warnings(
    results: list[EngineAuditResult],
    target_console: Console | None = None,
) -> None:
    """Render one yellow Rich panel per ``should_warn=True`` audit result.

    Stays silent when no engine result triggers a warning. Writes to the
    supplied ``target_console`` (defaults to the module-level ``console``).
    Callers in agent modes (``--ai-tinkering`` / unrestricted) MUST flush the
    underlying file before constructing the ``AgentOrchestrator`` — see TP C5
    ordering contract — to keep this panel from interleaving with streamed
    tool-call tags.
    """
    from rich.markup import escape as _rich_escape

    out = target_console if target_console is not None else console
    for r in results:
        if not r.should_warn:
            continue
        # The install_hint contains literal `[llama-tq]` / `[vllm-tq]` tokens
        # that Rich would otherwise interpret as markup tags and strip.
        hint = _rich_escape(r.install_hint)
        if r.engine == "vllm":
            body = (
                "Your GPU supports TurboQuant KV compression but your "
                "installed vLLM is upstream (no turboquant35 kernel).\n\n"
                "[bold]Fix:[/bold]\n"
                f"  {hint}\n\n"
                "Continuing with kv:none fallback."
            )
            title = "TurboQuant Unavailable (vLLM)"
        else:
            body = (
                "Your hardware supports TurboQuant KV compression but the "
                "installed llama.cpp is upstream (no turboN kernels).\n\n"
                "[bold]Fix:[/bold]\n"
                f"  {hint}\n\n"
                "Continuing with kv:none fallback."
            )
            title = "TurboQuant Unavailable (llama.cpp)"
        out.print(Panel(body, title=title, border_style="yellow"))


def audit_to_dict(result: EngineAuditResult) -> dict:
    """Serialize an EngineAuditResult for ``--json`` stderr metadata."""
    return {
        "engine": result.engine,
        "is_turboquant_fork": result.is_turboquant_fork,
        "hardware_capable": result.hardware_capable,
        "should_warn": result.should_warn,
        "install_hint": result.install_hint,
    }

