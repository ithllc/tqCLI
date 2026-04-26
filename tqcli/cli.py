"""tqCLI — TurboQuant CLI main entry point.

Usage:
    tqcli                     # Interactive chat (auto-selects model)
    tqcli chat                # Interactive chat mode
    tqcli chat --model <id>   # Chat with specific model
    tqcli system info         # Show system hardware and capabilities
    tqcli model list          # List available/installed models
    tqcli model pull <id>     # Download a model from HuggingFace
    tqcli model remove <id>   # Remove a downloaded model
    tqcli benchmark           # Benchmark loaded models
    tqcli security audit      # Run security checks
    tqcli skills              # List available skills
    tqcli handoff             # Generate handoff file for frontier CLI
    tqcli serve start/stop    # Multi-process inference server
    tqcli workers spawn N     # Spawn multi-process workers
    tqcli config show         # Show current configuration
    tqcli config init         # Initialize default configuration

Flags:
    --stop-trying-to-control-everything-and-just-let-go
        Bypass resource guards, confirmation prompts, and safety checks.
        Equivalent to Claude Code's --dangerously-skip-permissions
        or Gemini CLI's --yolo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tqcli import __version__


@click.group(invoke_without_command=True)
@click.version_option(__version__, prog_name="tqcli")
@click.option(
    "--stop-trying-to-control-everything-and-just-let-go",
    "unrestricted",
    is_flag=True,
    default=False,
    help="Bypass resource guards, confirmation prompts, and safety checks.",
)
@click.pass_context
def main(ctx, unrestricted):
    """tqCLI — TurboQuant CLI for local LLM inference with smart routing."""
    import os as _os

    ctx.ensure_object(dict)
    ctx.obj["unrestricted"] = unrestricted
    if unrestricted:
        from tqcli.core.unrestricted import show_unrestricted_warning
        from tqcli.ui.console import console
        show_unrestricted_warning(console)

    # Engine Auditor — runs once per CLI invocation, populates the cache, and
    # reports a yellow panel when the user has capable hardware but upstream
    # llama.cpp / vLLM. Skip entirely when TQCLI_SUPPRESS_AUDIT=1 (CI / scripts
    # that don't want the panel). Rendering is deferred for the `chat` command
    # because that command may turn on --json mode (stderr metadata instead of
    # the panel). See TP Phase 5.
    if _os.environ.get("TQCLI_SUPPRESS_AUDIT") != "1":
        try:
            from tqcli.core.engine_auditor import run_audit
            from tqcli.core.system_info import detect_system as _detect_system
            _audit_sys_info = _detect_system()
            _audit_results = run_audit(_audit_sys_info)
            ctx.obj["_audit_results"] = _audit_results
            ctx.obj["_audit_sys_info"] = _audit_sys_info
            if ctx.invoked_subcommand != "chat":
                from tqcli.ui.console import console as _console, render_audit_warnings
                render_audit_warnings(_audit_results, _console)
        except BaseException:  # noqa: BLE001 — audit must never crash startup
            ctx.obj["_audit_results"] = []
            ctx.obj["_audit_sys_info"] = None

    if ctx.invoked_subcommand is None:
        ctx.invoke(chat)


# ── Chat ──────────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default=None, help="Model ID to use (bypasses router)")
@click.option("--engine", "-e", type=click.Choice(["auto", "llama.cpp", "vllm", "server"]), default="auto")
@click.option("--context-length", "-c", type=int, default=None)
@click.option("--server-url", default=None, help="Connect to a running inference server (multi-process mode)")
@click.option("--kv-quant", type=click.Choice(["auto", "none", "turbo4", "turbo3", "turbo2"]),
              default="auto", help="TurboQuant KV cache compression level")
@click.option("--ai-tinkering", "ai_tinkering", is_flag=True, default=False,
              help="Shared Determinism: stage tool calls and require [Y/n/Edit] approval.")
@click.option("--max-agent-steps", type=int, default=10,
              help="Max ReAct loop iterations in unrestricted mode (default 10).")
@click.option("--prompt", "-p", default=None, help="Single-shot prompt for headless mode")
@click.option("--image", "-i", multiple=True, help="Repeatable image path for multimodal input")
@click.option("--audio", "-a", multiple=True, help="Repeatable audio path for multimodal input")
@click.option("--messages", type=click.Path(exists=True), help="JSON file containing conversation history")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON to stdout, route non-JSON chatter to stderr")
@click.option("--max-tokens", type=int, default=1024, help="Maximum tokens to generate")
@click.pass_context
def chat(ctx, model, engine, context_length, server_url, kv_quant, ai_tinkering, max_agent_steps, prompt, image, audio, messages, as_json, max_tokens):
    """Start an interactive or headless chat session.

    Interactive (default): start a terminal chat UI.
    Headless: provide --prompt or --messages for single-shot output.
    """
    import json as json_lib
    import sys
    from tqcli.config import TqConfig
    from tqcli.core.engine import ChatMessage
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.performance import PerformanceMonitor
    from tqcli.core.router import ModelRouter
    from tqcli.core.security import SecurityManager
    from tqcli.core.system_info import detect_system
    from tqcli.core.unrestricted import is_unrestricted
    from tqcli.ui.console import console, print_banner
    from tqcli.ui.interactive import InteractiveSession

    unrestricted = is_unrestricted(ctx)
    # Tri-state autonomy: unrestricted > ai_tinkering > manual.
    if unrestricted:
        agent_mode = "unrestricted"
    elif ai_tinkering:
        agent_mode = "ai_tinkering"
    else:
        agent_mode = "manual"
    # --ai-tinkering blocks on a TTY confirmation prompt; it cannot coexist
    # with --json (which is explicitly for automated/headless pipelines).
    # Fail fast instead of silently hanging.
    if as_json and agent_mode == "ai_tinkering":
        raise click.UsageError(
            "--ai-tinkering requires an interactive TTY for [Y/n/Edit] "
            "confirmation and cannot be combined with --json. Use "
            "--stop-trying-to-control-everything-and-just-let-go for "
            "autonomous headless runs."
        )
    if as_json:
        # Route non-JSON chatter to stderr
        from tqcli.ui.console import setup_json_logging
        console.file = sys.stderr
        setup_json_logging()

    # Engine-audit reporting — main() ran the audit; this is where the chat
    # command renders. In --json mode the audit goes to stderr as a JSON
    # metadata line (one object per line) so a downstream parser can pick it
    # up. In TTY mode it shows a yellow Rich panel via render_audit_warnings.
    _audit_results = ctx.obj.get("_audit_results", []) if ctx.obj else []
    if _audit_results:
        if as_json:
            import json as _json_audit
            for r in _audit_results:
                line = _json_audit.dumps({
                    "engine_audit": {
                        "engine": r.engine,
                        "is_turboquant_fork": r.is_turboquant_fork,
                        "hardware_capable": r.hardware_capable,
                        "should_warn": r.should_warn,
                        "install_hint": r.install_hint,
                    }
                })
                sys.stderr.write(line + "\n")
            sys.stderr.flush()
        else:
            from tqcli.ui.console import render_audit_warnings
            render_audit_warnings(_audit_results, console)

    if not as_json or not (prompt or messages):
        print_banner()
    
    config = TqConfig.load()
    config.ensure_dirs()
    if unrestricted:
        config.unrestricted = True
        config.performance.auto_handoff = False  # disabled in unrestricted mode

    if context_length:
        config.context_length = context_length

    # Security initialization
    sec = SecurityManager(config.security)
    for msg in sec.initialize():
        console.print(f"  [dim]{msg}[/dim]")

    # System detection
    sys_info = detect_system()
    console.print(f"  [dim]System: {sys_info.os_display} | RAM: {sys_info.ram_available_mb:,} MB[/dim]")
    if sys_info.gpus:
        console.print(f"  [dim]GPU: {sys_info.gpus[0].name} ({sys_info.total_vram_mb:,} MB VRAM)[/dim]")

    # Server/multi-process mode
    if engine == "server" or server_url:
        from tqcli.core.server_client import ServerClientBackend
        url = server_url or f"http://{config.multiprocess.server_host}:{config.multiprocess.server_port}"
        eng = ServerClientBackend(base_url=url, model_name=model or "local")
        console.print(f"  [dim]Mode: multi-process (server at {url})[/dim]")
        try:
            eng.load_model(model or "local")
        except RuntimeError as e:
            console.print(f"\n[red]{e}[/red]")
            return
        monitor = PerformanceMonitor(config.performance)
        # Ordering contract (TP C5): in agent modes, the audit panel must be
        # fully flushed to stderr before the orchestrator emits its first
        # streamed token tag, otherwise Rich's buffered output can interleave
        # with <tool_call> tags and produce a garbled stream.
        if agent_mode != "manual":
            try:
                console.file.flush()
            except Exception:
                pass
        session = InteractiveSession(
            config, eng, None, monitor,
            agent_mode=agent_mode, max_agent_steps=max_agent_steps,
        )
        try:
            session.run()
        finally:
            eng.unload_model()
            sec.log_event("session_end")
        return

    # Single-process mode (original behavior)
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()

    if not available:
        console.print("\n[yellow]No models installed.[/yellow]")
        console.print("Download one with:")
        for p in registry.get_all_profiles()[:3]:
            console.print(f"  tqcli model pull {p.id}")
        return

    # Router
    router = ModelRouter(registry, sys_info.ram_available_mb, sys_info.total_vram_mb)
    if model:
        router.set_override(model)

    # Engine selection
    selected_engine = engine if engine != "auto" else config.preferred_engine
    if selected_engine == "auto":
        selected_engine = sys_info.recommended_engine

    if selected_engine == "vllm":
        from tqcli.core.vllm_backend import VllmBackend
        from tqcli.core.vllm_config import build_vllm_config
        # Auto-tune vLLM parameters based on hardware and model
        target_for_tune = available[0]
        if model:
            target_for_tune = registry.get_profile(model) or target_for_tune
        tune = build_vllm_config(target_for_tune, sys_info, requested_max_len=context_length, kv_quant_choice=kv_quant)
        if not tune.feasible and not unrestricted:
            console.print(f"[red]vLLM not feasible: {tune.reason}[/red]")
            console.print("Use --stop-trying-to-control-everything-and-just-let-go to bypass.")
            return
        for w in tune.warnings:
            console.print(f"  [dim]vLLM: {w}[/dim]")
        eng = VllmBackend.from_tuning_profile(tune)
    else:
        from tqcli.core.llama_backend import LlamaBackend
        from tqcli.core.kv_quantizer import (
            KVQuantLevel, check_turboquant_compatibility, get_llama_kv_params, select_kv_quant,
        )
        # Check TurboQuant compatibility before applying KV compression
        tq_available, tq_msg = check_turboquant_compatibility(sys_info)
        effective_kv_quant = kv_quant

        if kv_quant not in ("none", "auto") and not tq_available:
            # User explicitly requested turbo KV but system is incompatible
            console.print(f"  [yellow]{tq_msg}[/yellow]")
            effective_kv_quant = "none"
        elif kv_quant == "auto" and not tq_available:
            # Auto mode: silently fall back with a dim warning
            console.print(f"  [dim]TurboQuant KV: unavailable — {tq_msg.split('. ')[0]}.[/dim]")
            effective_kv_quant = "none"

        # Determine KV cache compression level
        kv_level = select_kv_quant(available_kv_mb=50, engine="llama.cpp", user_choice=effective_kv_quant)
        kv_params = get_llama_kv_params(kv_level)
        if kv_level != KVQuantLevel.NONE:
            console.print(f"  [dim]TurboQuant KV: {kv_level.value} ({kv_params})[/dim]")
        eng = LlamaBackend(
            n_ctx=config.context_length,
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.threads,
            cache_type_k=kv_params.get("cache_type_k", "f16"),
            cache_type_v=kv_params.get("cache_type_v", "f16"),
        )

    if not eng.is_available:
        console.print(f"\n[red]Engine '{selected_engine}' not installed.[/red]")
        if selected_engine == "vllm":
            console.print("Install with: pip install vllm")
        else:
            console.print("Install with: pip install llama-cpp-python")
        return

    # Resource check before loading
    target = available[0]
    if model:
        target = registry.get_profile(model) or target
    if not unrestricted:
        ok, issues = sec.check_before_load(target.id)
        if not ok:
            console.print("[red]Resource check failed:[/red]")
            for issue in issues:
                console.print(f"  {issue}")
            console.print("\nUse --stop-trying-to-control-everything-and-just-let-go to bypass.")
            return

    if target.local_path:
        console.print(f"\n  Loading [bold]{target.display_name}[/bold]...")
        console.print(f"  [dim]Mode: single-process (in-process inference)[/dim]")
        sec.log_event("model_load", {"model": target.id})
        eng.load_model(str(target.local_path), multimodal=target.multimodal)
    else:
        console.print(f"\n[red]Model {target.id} not found locally.[/red]")
        return

    # Performance monitor
    monitor = PerformanceMonitor(config.performance)

    # Ordering contract (TP C5): flush console.file to stderr before the
    # orchestrator's first stream chunk in agent modes. Same rationale as
    # the server-mode block above.
    if agent_mode != "manual":
        try:
            console.file.flush()
        except Exception:
            pass

    # Launch interactive session
    session = InteractiveSession(
        config, eng, router, monitor,
        model_family=target.family,
        agent_mode=agent_mode,
        max_agent_steps=max_agent_steps,
    )
    if agent_mode != "manual":
        console.print(f"  [bold yellow]Agent mode: {agent_mode}[/bold yellow]")
    
    if prompt or messages:
        # Headless mode
        try:
            if messages:
                with open(messages, "r") as f:
                    history_dicts = json_lib.load(f)
                    session.history = [ChatMessage(**m) for m in history_dicts]
            
            # Run the turn
            user_prompt = prompt or ""
            session.chat_turn(
                user_prompt, 
                images=list(image), 
                audio=list(audio), 
                show_ui=not as_json,
                max_tokens=max_tokens
            )
            
            if as_json:
                from tqcli.core.thinking import extract_thinking
                stats = session.last_stats
                thinking_text, clean_response = extract_thinking(session.last_response, session._thinking_fmt)
                res = {
                    "model": target.id,
                    "engine": selected_engine,
                    "response": clean_response,
                    "thinking": thinking_text or None,
                    "usage": {
                        "prompt_tokens": stats.prompt_tokens if stats else 0,
                        "completion_tokens": stats.completion_tokens if stats else 0,
                        "total_tokens": stats.total_tokens if stats else 0,
                    },
                    "performance": {
                        "tokens_per_second": stats.tokens_per_second if stats else 0.0,
                        "total_time_s": stats.total_time_s if stats else 0.0,
                    },
                    "metadata": {
                        "kv_quant": kv_quant,
                        "images": list(image),
                        "audio": list(audio),
                    }
                }
                print(json_lib.dumps(res, indent=2))
            else:
                # If not JSON but headless, response is already printed by chat_turn(show_ui=True)
                pass
                
        except Exception as e:
            if as_json:
                print(json_lib.dumps({
                    "error": str(e),
                    "error_type": e.__class__.__name__
                }, indent=2))
            else:
                console.print(f"[red]Error: {e}[/red]")
            sys.exit(1)
        finally:
            eng.unload_model()
            sec.log_event("session_end")
        return

    # Interactive mode
    try:
        session.run()
    finally:
        eng.unload_model()
        sec.log_event("session_end")


# ── System ────────────────────────────────────────────────────────────


@main.group()
def system():
    """System information and diagnostics."""
    pass


@system.command("info")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def system_info(as_json):
    """Display OS, hardware, and inference engine capabilities."""
    import json as json_mod

    from tqcli.core.system_info import detect_system
    from tqcli.ui.console import print_system_info, setup_json_logging

    from tqcli.core.kv_quantizer import check_turboquant_compatibility

    if as_json:
        setup_json_logging()

    info = detect_system()
    tq_available, tq_msg = check_turboquant_compatibility(info)
    if as_json:
        data = {
            "os": info.os_name,
            "os_display": info.os_display,
            "arch": info.arch,
            "is_wsl": info.is_wsl,
            "cpu_cores": info.cpu_cores_logical,
            "ram_total_mb": info.ram_total_mb,
            "ram_available_mb": info.ram_available_mb,
            "gpus": [
                {
                    "name": g.name,
                    "vram_mb": g.vram_total_mb,
                    "cuda_version": g.cuda_version,
                    "cuda_toolkit_version": g.cuda_toolkit_version,
                }
                for g in info.gpus
            ],
            "recommended_engine": info.recommended_engine,
            "recommended_quant": info.recommended_quant,
            "max_model_gb": info.max_model_size_estimate_gb,
            "turboquant_kv": {"available": tq_available, "message": tq_msg},
        }
        click.echo(json_mod.dumps(data, indent=2))
    else:
        print_system_info(info)


# ── Model ─────────────────────────────────────────────────────────────


@main.group()
def model():
    """Model management — list, pull, remove."""
    pass


@model.command("list")
def model_list():
    """List all known and installed models."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import print_model_list

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    print_model_list(registry.get_all_profiles())


@model.command("pull")
@click.argument("model_id")
def model_pull(model_id):
    """Download a model from HuggingFace Hub."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.security import SecurityManager
    from tqcli.ui.console import console

    config = TqConfig.load()
    config.ensure_dirs()
    registry = ModelRegistry(config.models_dir)
    profile = registry.get_profile(model_id)

    if not profile:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        console.print("Available models:")
        for p in registry.get_all_profiles():
            console.print(f"  {p.id}")
        return

    # Security check
    sec = SecurityManager(config.security)
    ok, issues = sec.check_before_load(model_id)
    if not ok:
        console.print("[red]Resource check failed:[/red]")
        for issue in issues:
            console.print(f"  {issue}")
        return

    console.print(f"Downloading [bold]{profile.display_name}[/bold]...")
    console.print(f"  From: {profile.hf_repo}")

    if profile.engine == "vllm" and profile.format in ("safetensors", "awq"):
        # vLLM models: download the full repo snapshot into a named directory
        console.print(f"  Mode: Full repo snapshot (vLLM)")
        console.print(f"  To:   {config.models_dir}/{profile.id}/")
        try:
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id=profile.hf_repo,
                local_dir=str(config.models_dir / profile.id),
                local_dir_use_symlinks=False,
            )
            sec.log_event("model_downloaded", {"model": model_id, "path": path})
            console.print(f"\n[green]Downloaded:[/green] {path}")
        except Exception as e:
            console.print(f"\n[red]Download failed:[/red] {e}")
    else:
        # GGUF models: download single file
        console.print(f"  File: {profile.filename}")
        console.print(f"  To:   {config.models_dir}/")
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=profile.hf_repo,
                filename=profile.filename,
                local_dir=str(config.models_dir),
                local_dir_use_symlinks=False,
            )
            sec.log_event("model_downloaded", {"model": model_id, "path": path})
            console.print(f"\n[green]Downloaded:[/green] {path}")
        except Exception as e:
            console.print(f"\n[red]Download failed:[/red] {e}")


@model.command("remove")
@click.argument("model_id")
@click.pass_context
def model_remove(ctx, model_id):
    """Remove a downloaded model."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    profile = registry.get_profile(model_id)

    if not profile or not profile.local_path:
        console.print(f"[red]Model not found locally: {model_id}[/red]")
        return

    path = profile.local_path
    from tqcli.core.unrestricted import is_unrestricted
    if path.is_dir():
        # vLLM model directory — calculate total size
        import shutil
        size_mb = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
        if is_unrestricted(ctx) or click.confirm(f"Remove {profile.display_name} ({size_mb:.0f} MB)?"):
            shutil.rmtree(path)
            console.print(f"[green]Removed:[/green] {path}")
    else:
        size_mb = path.stat().st_size / (1024 * 1024)
        if is_unrestricted(ctx) or click.confirm(f"Remove {profile.display_name} ({size_mb:.0f} MB)?"):
            path.unlink()
            console.print(f"[green]Removed:[/green] {path}")


@model.command("quantize")
@click.argument("model_id")
@click.option("--method", type=click.Choice(["auto", "bnb", "awq", "gguf"]), default="auto")
@click.option("--bits", type=click.Choice(["4", "8"]), default="4")
def model_quantize(model_id, method, bits):
    """Show quantization info and recommendations for a model.

    Displays the model's current format, estimated BF16 size, recommended
    quantization method for the current hardware, and expected quantized size.
    On-the-fly quantization (bitsandbytes) is applied automatically at load time.
    """
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.quantizer import (
        QuantizationMethod,
        estimate_bf16_model_size,
        estimate_quantized_size,
        select_quantization,
    )
    from tqcli.core.system_info import detect_system
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    profile = registry.get_profile(model_id)
    if not profile:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        return

    sys_info = detect_system()
    bf16_size = estimate_bf16_model_size(profile)
    recommended = select_quantization(profile, sys_info)

    console.print(f"\n[bold]Quantization Info: {profile.display_name}[/bold]\n")
    console.print(f"  Current format:    {profile.quantization} ({profile.format})")
    console.print(f"  Parameter count:   {profile.parameter_count}")
    console.print(f"  Estimated BF16:    {bf16_size:,} MB")
    console.print(f"  Available VRAM:    {sys_info.total_vram_mb:,} MB")

    if recommended == QuantizationMethod.NONE:
        console.print(f"\n  [green]No quantization needed — model fits at current precision.[/green]")
    elif recommended == QuantizationMethod.BNB_INT4:
        q_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT4)
        console.print(f"\n  [yellow]Recommended: bitsandbytes INT4[/yellow]")
        console.print(f"  Quantized size:    ~{q_size:,} MB ({q_size/bf16_size*100:.0f}% of BF16)")
        console.print(f"  Method:            On-the-fly at load time (no pre-quantization)")
        console.print(f"\n  This will be applied automatically when you run:")
        console.print(f"    tqcli chat --engine vllm --model {model_id}")
    elif recommended == QuantizationMethod.BNB_INT8:
        q_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT8)
        console.print(f"\n  [yellow]Recommended: bitsandbytes INT8[/yellow]")
        console.print(f"  Quantized size:    ~{q_size:,} MB ({q_size/bf16_size*100:.0f}% of BF16)")
    elif recommended is None:
        console.print(f"\n  [red]Model too large for this GPU even after INT4 quantization.[/red]")
        int4_size = estimate_quantized_size(profile, QuantizationMethod.BNB_INT4)
        console.print(f"  INT4 size:         ~{int4_size:,} MB (still exceeds VRAM)")


@model.command("calibrate-kv")
@click.argument("model_id")
@click.option("--recipe", type=click.Choice(["turboquant25", "turboquant35"]),
              default="turboquant35",
              help="TurboQuant recipe to calibrate for (default: turboquant35).")
@click.option("--force", is_flag=True,
              help="Overwrite an existing turboquant_kv.json.")
def model_calibrate_kv(model_id: str, recipe: str, force: bool) -> None:
    """Pre-calibrate TurboQuant KV metadata for a vLLM model.

    Runs activation-based calibration on a diverse prompt corpus to
    produce turboquant_kv.json with per-layer, per-KV-head outlier
    channel indices. Pre-warming this avoids lazy calibration at first
    chat invocation and makes the calibration failure mode explicit.
    """
    from pathlib import Path
    import sys

    from tqcli.config import TqConfig
    from tqcli.core.kv_metadata_generator import (
        check_calibration_preconditions,
        generate_turboquant_metadata,
    )
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    profile = registry.get_profile(model_id)
    if profile is None:
        console.print(f"[red]Unknown model: {model_id}[/red]")
        console.print("Installed models:")
        for p in registry.get_all_profiles():
            if p.local_path:
                console.print(f"  {p.id}")
        sys.exit(2)

    if not profile.local_path:
        console.print(
            f"[red]{model_id} is not installed locally.[/red] "
            f"Run `tqcli model pull {model_id}` first."
        )
        sys.exit(2)

    model_dir = Path(profile.local_path)
    if model_dir.is_file():
        model_dir = model_dir.parent

    if profile.engine != "vllm":
        console.print(
            f"[yellow]{model_id} uses engine={profile.engine}, not vllm. "
            f"TurboQuant KV metadata generation is vLLM-only; llama.cpp uses "
            f"runtime cache_type_k / cache_type_v without metadata.[/yellow]"
        )
        sys.exit(2)

    metadata_path = model_dir / "turboquant_kv.json"
    if metadata_path.is_file() and not force:
        console.print(
            f"[yellow]Metadata already exists at {metadata_path}[/yellow]"
        )
        console.print("Pass --force to regenerate.")
        sys.exit(0)

    ok, reason = check_calibration_preconditions(model_dir, recipe)
    if not ok:
        console.print(f"[red]Calibration refused:[/red] {reason}")
        sys.exit(3)

    console.print(f"Calibrating [bold]{profile.display_name}[/bold] for recipe [bold]{recipe}[/bold]")
    console.print(f"  Model dir: {model_dir}")
    console.print(f"  Output:    {metadata_path}")
    console.print("  Running 30-prompt activation calibration...")
    try:
        generated = generate_turboquant_metadata(
            model_dir=model_dir,
            kv_cache_dtype=recipe,
        )
    except Exception as exc:
        console.print(f"[red]Calibration failed:[/red] {type(exc).__name__}: {exc}")
        sys.exit(4)

    size_kb = generated.stat().st_size / 1024.0
    console.print(f"[green]Wrote {generated} ({size_kb:.1f} KB)[/green]")


# ── Benchmark ─────────────────────────────────────────────────────────


@main.command()
@click.option("--model", "-m", default=None, help="Model ID to benchmark")
@click.option("--all-models", is_flag=True, help="Benchmark all installed models")
def benchmark(model, all_models):
    """Run performance benchmarks on installed models."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.ui.console import console

    config = TqConfig.load()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()

    if not available:
        console.print("[yellow]No models installed. Use 'tqcli model pull <id>' first.[/yellow]")
        return

    console.print("[bold]Running benchmarks...[/bold]\n")
    # Delegate to benchmark script
    import subprocess

    script = Path(__file__).parent.parent / ".claude" / "skills" / "tq-benchmark" / "scripts" / "run_benchmark.py"
    cmd = [sys.executable, str(script)]
    if model:
        cmd.extend(["--model", model])
    elif all_models:
        cmd.append("--all")
    subprocess.run(cmd)


# ── Security ──────────────────────────────────────────────────────────


@main.group()
def security():
    """Security tools and audit."""
    pass


@security.command("audit")
@click.option("--json", "as_json", is_flag=True)
@click.option("--fix", is_flag=True, help="Auto-fix safe issues")
def security_audit(as_json, fix):
    """Run security audit on the tqCLI environment."""
    import subprocess

    script = Path(__file__).parent.parent / ".claude" / "skills" / "tq-security-audit" / "scripts" / "run_audit.py"
    cmd = [sys.executable, str(script)]
    if as_json:
        cmd.append("--json")
    if fix:
        cmd.append("--fix")
    subprocess.run(cmd)


# ── Skills ────────────────────────────────────────────────────────────


@main.group("skill", invoke_without_command=True)
@click.pass_context
def skill_group(ctx):
    """Skill management — list and create skills."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(skill_list)


@skill_group.command("list")
def skill_list():
    """List available tqCLI skills."""
    from tqcli.skills.loader import SkillLoader
    from tqcli.ui.console import print_skill_list

    project_skills = Path(__file__).parent.parent / ".claude" / "skills"
    user_skills = Path.home() / ".tqcli" / "skills"
    loader = SkillLoader([project_skills, user_skills])
    skills = loader.list_skills()
    print_skill_list(skills)


@skill_group.command("create")
@click.argument("name")
@click.option("--description", "-d", default="", help="Skill description")
def skill_create(name, description):
    """Create a new skill directory with SKILL.md and template script."""
    from tqcli.ui.console import console

    skills_dir = Path.home() / ".tqcli" / "skills"
    skill_dir = skills_dir / name
    if skill_dir.exists():
        console.print(f"[red]Skill '{name}' already exists at {skill_dir}[/red]")
        return

    skill_dir.mkdir(parents=True)
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir()

    desc = description or f"Custom skill: {name}"
    # Create SKILL.md
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(f"""---
name: {name}
description: {desc}
---

# {name}

{desc}

## Usage
Run with: `tqcli skill run {name}`
""")

    # Create template script
    script_file = scripts_dir / f"run_{name.replace('-', '_')}.py"
    script_file.write_text(f'''#!/usr/bin/env python3
"""Skill: {name} — {desc}"""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description="{desc}")
    parser.add_argument("--output", "-o", default=None, help="Output file path")
    args = parser.parse_args()

    result = {{
        "skill": "{name}",
        "status": "completed",
        "message": "Skill {name} executed successfully",
    }}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
''')

    console.print(f"[green]Skill created:[/green] {skill_dir}")
    console.print(f"  SKILL.md:  {skill_md}")
    console.print(f"  Script:    {script_file}")
    console.print(f"\nList skills with: tqcli skill list")


@skill_group.command("generate")
@click.option("--prd", "prd_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--plan", "plan_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--name", "skill_name", required=True, help="Target skill name (also directory slug)")
@click.option("--model", "-m", default=None, help="Model ID (defaults to first available)")
@click.option("--engine", "-e", type=click.Choice(["auto", "llama.cpp", "vllm"]), default="auto")
@click.option("--kv-quant", type=click.Choice(["auto", "none", "turbo4", "turbo3", "turbo2"]),
              default="turbo4", help="KV compression; turbo4 is the safest default for code gen")
@click.option("--max-tokens", type=int, default=1500)
@click.option("--yes", is_flag=True, help="Skip the interactive review prompt")
@click.option("--overwrite", is_flag=True, help="Overwrite an existing skill directory")
@click.pass_context
def skill_generate(ctx, prd_path, plan_path, skill_name, model, engine, kv_quant, max_tokens, yes, overwrite):
    """Generate a tqCLI skill from a PRD + Technical Plan using the local LLM.

    The generated skill lands in ~/.tqcli/skills/<name>/ after you review it.
    """
    from pathlib import Path as _Path

    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.skill_generator import generate_skill, slugify, write_skill
    from tqcli.core.system_info import detect_system
    from tqcli.core.unrestricted import is_unrestricted
    from tqcli.ui.console import console

    unrestricted = is_unrestricted(ctx)
    config = TqConfig.load()
    config.ensure_dirs()

    sys_info = detect_system()
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()
    if not available:
        console.print("[red]No models installed.[/red] Run `tqcli model pull <id>` first.")
        return

    target = available[0]
    if model:
        target = registry.get_profile(model) or target
    if not target.local_path:
        console.print(f"[red]Model {target.id} not installed locally.[/red]")
        return

    selected_engine = engine if engine != "auto" else (target.engine or sys_info.recommended_engine)

    # Build the engine the same way the chat command does (minus the router / REPL).
    if selected_engine == "vllm":
        from tqcli.core.vllm_backend import VllmBackend
        from tqcli.core.vllm_config import build_vllm_config
        tune = build_vllm_config(target, sys_info, requested_max_len=None, kv_quant_choice=kv_quant)
        if not tune.feasible and not unrestricted:
            console.print(f"[red]vLLM not feasible: {tune.reason}[/red]")
            return
        for w in tune.warnings:
            console.print(f"  [dim]vLLM: {w}[/dim]")
        eng = VllmBackend.from_tuning_profile(tune)
    else:
        from tqcli.core.kv_quantizer import (
            KVQuantLevel, check_turboquant_compatibility, get_llama_kv_params, select_kv_quant,
        )
        from tqcli.core.llama_backend import LlamaBackend
        tq_available, tq_msg = check_turboquant_compatibility(sys_info)
        effective = kv_quant
        if kv_quant not in ("none", "auto") and not tq_available:
            console.print(f"  [yellow]{tq_msg}[/yellow]")
            effective = "none"
        kv_level = select_kv_quant(available_kv_mb=50, engine="llama.cpp", user_choice=effective)
        kv_params = get_llama_kv_params(kv_level)
        if kv_level != KVQuantLevel.NONE:
            console.print(f"  [dim]TurboQuant KV: {kv_level.value} ({kv_params})[/dim]")
        eng = LlamaBackend(
            n_ctx=max(config.context_length, 8192),
            n_gpu_layers=config.n_gpu_layers,
            n_threads=config.threads,
            cache_type_k=kv_params.get("cache_type_k", "f16"),
            cache_type_v=kv_params.get("cache_type_v", "f16"),
        )

    if not eng.is_available:
        console.print(f"[red]Engine '{selected_engine}' not installed.[/red]")
        return

    console.print(f"  Loading [bold]{target.display_name}[/bold] ({selected_engine})...")
    eng.load_model(str(target.local_path), multimodal=target.multimodal)

    try:
        console.print(f"  Generating skill [bold]{skill_name}[/bold] from PRD + Plan...")
        result = generate_skill(
            eng,
            prd_path=_Path(prd_path),
            plan_path=_Path(plan_path),
            skill_name=skill_name,
            max_tokens=max_tokens,
        )
    finally:
        eng.unload_model()

    if not result.valid:
        console.print("[red]Generation validation failed:[/red]")
        for err in result.errors:
            console.print(f"  - {err}")
        console.print(f"  Raw model output: {len(result.raw_response)} chars.")
        if not yes and not click.confirm("Write the files anyway for manual inspection?", default=False):
            return

    console.print(f"\n  Target directory: [bold]{result.target_dir}[/bold]")
    console.print(f"  Files generated: {len(result.files)}")
    for gf in result.files:
        flag = "[green]OK[/green]" if (not gf.is_python or gf.ast_ok) else f"[red]{gf.error}[/red]"
        console.print(f"    - {gf.relative_path} ({len(gf.content)} chars) {flag}")

    if not yes:
        for gf in result.files:
            console.print(f"\n[bold]--- {gf.relative_path} ---[/bold]")
            preview = gf.content if len(gf.content) < 2000 else gf.content[:2000] + "\n... (truncated)"
            console.print(preview)
        if not click.confirm(f"\nWrite to {result.target_dir}?", default=True):
            console.print("[yellow]Aborted — no files written.[/yellow]")
            return

    try:
        path = write_skill(result, overwrite=overwrite)
        console.print(f"[green]Skill written:[/green] {path}")
        console.print(f"Run with: tqcli skill run {slugify(skill_name)}")
    except FileExistsError as e:
        console.print(f"[red]{e}[/red] Pass --overwrite to replace.")


@skill_group.command("run")
@click.argument("name")
@click.argument("args", nargs=-1)
def skill_run(name, args):
    """Run a skill's script."""
    from tqcli.skills.loader import SkillLoader
    from tqcli.ui.console import console

    project_skills = Path(__file__).parent.parent / ".claude" / "skills"
    user_skills = Path.home() / ".tqcli" / "skills"
    loader = SkillLoader([project_skills, user_skills])
    skill = loader.get_skill(name)

    if not skill:
        console.print(f"[red]Skill '{name}' not found.[/red]")
        console.print("Available skills:")
        for s in loader.list_skills():
            console.print(f"  {s.name}")
        return

    if not skill.has_scripts:
        console.print(f"[yellow]Skill '{name}' has no scripts.[/yellow]")
        return

    import subprocess
    for script in skill.scripts:
        console.print(f"Running {script.name}...")
        result = subprocess.run(
            [sys.executable, str(script)] + list(args),
            capture_output=False,
        )
        if result.returncode != 0:
            console.print(f"[red]Script {script.name} failed (exit code {result.returncode})[/red]")
        else:
            console.print(f"[green]Script {script.name} completed.[/green]")


# Backward compat alias
@main.command("skills", hidden=True)
def list_skills_compat():
    """List available tqCLI skills (alias for 'skill list')."""
    from tqcli.skills.loader import SkillLoader
    from tqcli.ui.console import print_skill_list

    project_skills = Path(__file__).parent.parent / ".claude" / "skills"
    user_skills = Path.home() / ".tqcli" / "skills"
    loader = SkillLoader([project_skills, user_skills])
    skills = loader.list_skills()
    print_skill_list(skills)


# ── Handoff ───────────────────────────────────────────────────────────


@main.command()
@click.option("--task", "-t", required=True, help="Task description")
@click.option(
    "--target",
    type=click.Choice(["auto", "claude-code", "gemini-cli", "aider", "openai"]),
    default="auto",
)
def handoff(task, target):
    """Generate a handoff file to transfer work to a frontier model CLI."""
    from tqcli.config import TqConfig
    from tqcli.core.handoff import generate_handoff
    from tqcli.core.performance import PerformanceMonitor
    from tqcli.ui.console import console

    config = TqConfig.load()
    monitor = PerformanceMonitor(config.performance)
    # Simulate slow performance for handoff context
    monitor.record(tokens=20, elapsed_s=10.0)
    monitor.record(tokens=15, elapsed_s=8.0)

    output_dir = Path.home() / ".tqcli" / "handoffs"
    filepath = generate_handoff(
        monitor=monitor,
        conversation_history=[{"role": "user", "content": task}],
        task_description=task,
        output_dir=output_dir,
        target_cli=target,
    )
    console.print(f"[green]Handoff file generated:[/green] {filepath}")
    console.print(f"\nTo continue with {target}:")
    if target in ("auto", "claude-code"):
        console.print("  claude")
        console.print(f"  Then reference: @{filepath.name}")


# ── Config ────────────────────────────────────────────────────────────


@main.group()
def config():
    """Configuration management."""
    pass


@config.command("show")
def config_show():
    """Show current tqCLI configuration."""
    from tqcli.config import TqConfig
    from tqcli.ui.console import console

    cfg = TqConfig.load()
    console.print("[bold]tqCLI Configuration[/bold]\n")
    console.print(f"  Models dir:       {cfg.models_dir}")
    console.print(f"  Preferred engine: {cfg.preferred_engine}")
    console.print(f"  Default quant:    {cfg.default_quantization}")
    console.print(f"  Context length:   {cfg.context_length}")
    console.print(f"  GPU layers:       {cfg.n_gpu_layers}")
    console.print(f"  Threads:          {cfg.threads or 'auto'}")
    console.print(f"\n  [bold]Performance[/bold]")
    console.print(f"  Min tok/s:        {cfg.performance.min_tokens_per_second}")
    console.print(f"  Warning tok/s:    {cfg.performance.warning_tokens_per_second}")
    console.print(f"  Auto handoff:     {cfg.performance.auto_handoff}")
    console.print(f"\n  [bold]Security[/bold]")
    console.print(f"  Venv:             {cfg.security.use_venv}")
    console.print(f"  Sandbox:          {cfg.security.sandbox_enabled}")
    console.print(f"  Audit log:        {cfg.security.audit_log}")
    console.print(f"  Max memory:       {cfg.security.max_memory_percent}%")
    console.print(f"  Max GPU memory:   {cfg.security.max_gpu_memory_percent}%")
    console.print(f"\n  [bold]Multi-Process[/bold]")
    console.print(f"  Server host:      {cfg.multiprocess.server_host}")
    console.print(f"  Server port:      {cfg.multiprocess.server_port}")
    console.print(f"  Max workers:      {cfg.multiprocess.max_workers}")
    console.print(f"  Auto-start:       {cfg.multiprocess.auto_start_server}")


@config.command("init")
def config_init():
    """Initialize default tqCLI configuration."""
    from tqcli.config import TqConfig
    from tqcli.ui.console import console

    cfg = TqConfig()
    cfg.save()
    cfg.ensure_dirs()
    console.print("[green]Configuration initialized at ~/.tqcli/config.yaml[/green]")
    console.print("Edit the file to customize settings, or use 'tqcli config show' to review.")


# ── Serve (Multi-Process Server) ─────────────────────────────────────


@main.group()
def serve():
    """Multi-process inference server management."""
    pass


@serve.command("start")
@click.option("--model", "-m", default=None, help="Model ID to serve")
@click.option("--engine", "-e", type=click.Choice(["auto", "llama.cpp", "vllm"]), default="auto")
@click.option("--port", "-p", type=int, default=8741, help="Server port")
@click.pass_context
def serve_start(ctx, model, engine, port):
    """Start a shared inference server for multi-process mode."""
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.multiprocess import assess_multiprocess
    from tqcli.core.server import InferenceServer, ServerConfig
    from tqcli.core.system_info import detect_system
    from tqcli.core.unrestricted import is_unrestricted
    from tqcli.ui.console import console

    unrestricted = is_unrestricted(ctx)
    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()

    # Find model
    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()
    if not available:
        console.print("[yellow]No models installed.[/yellow]")
        return

    target = available[0]
    if model:
        target = registry.get_profile(model) or target
    if not target.local_path:
        console.print(f"[red]Model {target.id} not installed.[/red]")
        return

    # Assess feasibility
    plan = assess_multiprocess(
        sys_info=sys_info,
        model_path=str(target.local_path),
        model_size_mb=target.min_ram_mb,
        requested_workers=1,
        preferred_engine=engine,
        unrestricted=unrestricted,
    )

    if not plan.feasible and not unrestricted:
        console.print(f"[red]Cannot start server: {plan.reason}[/red]")
        for w in plan.warnings:
            console.print(f"  [yellow]{w}[/yellow]")
        console.print("\nUse --stop-trying-to-control-everything-and-just-let-go to bypass.")
        return

    for w in plan.warnings:
        console.print(f"  [yellow]Warning: {w}[/yellow]")

    # Start server — auto-tune for low-VRAM GPUs when using vLLM
    use_eager = plan.engine == "vllm" and sys_info.total_vram_mb < 6000
    gpu_util = 0.80 if use_eager else config.security.max_gpu_memory_percent / 100.0
    server_config = ServerConfig(
        engine=plan.engine,
        model_path=str(target.local_path),
        host="127.0.0.1",
        port=port,
        context_length=min(config.context_length, 256) if use_eager else config.context_length,
        n_gpu_layers=config.n_gpu_layers,
        threads=config.threads,
        gpu_memory_utilization=gpu_util,
        enforce_eager=use_eager,
    )
    server = InferenceServer(server_config)

    console.print(f"Starting {plan.engine} inference server...")
    console.print(f"  Model:  {target.display_name}")
    console.print(f"  Engine: {plan.engine}")
    console.print(f"  Port:   {port}")

    try:
        status = server.start()
        console.print(f"\n[green]Server running![/green]")
        console.print(f"  PID:  {status.pid}")
        console.print(f"  URL:  http://{status.host}:{status.port}")
        console.print(f"\nConnect workers with:")
        console.print(f"  tqcli chat --engine server")
        console.print(f"  tqcli chat --server-url http://127.0.0.1:{port}")
        console.print(f"\nStop with: tqcli serve stop")
    except RuntimeError as e:
        console.print(f"\n[red]Failed to start server: {e}[/red]")


@serve.command("status")
def serve_status():
    """Show inference server status."""
    from tqcli.core.server import InferenceServer, ServerConfig
    from tqcli.ui.console import console

    server = InferenceServer(ServerConfig(engine="", model_path=""))
    status = server.status()

    if not status.running:
        console.print("No inference server running.")
        console.print("Start one with: tqcli serve start")
        return

    health = server.health_check()
    console.print("[bold]Inference Server Status[/bold]\n")
    console.print(f"  Status:  [green]running[/green]")
    console.print(f"  PID:     {status.pid}")
    console.print(f"  Engine:  {status.engine}")
    console.print(f"  Model:   {status.model}")
    console.print(f"  URL:     http://{status.host}:{status.port}")
    console.print(f"  Health:  {'[green]OK[/green]' if health else '[red]UNREACHABLE[/red]'}")
    console.print(f"  Uptime:  {status.uptime_s:.0f}s")


@serve.command("stop")
def serve_stop():
    """Stop the running inference server."""
    from tqcli.core.server import InferenceServer, ServerConfig
    from tqcli.ui.console import console

    server = InferenceServer(ServerConfig(engine="", model_path=""))
    if server.is_running():
        console.print("Stopping inference server...")
        server.stop()
        console.print("[green]Server stopped.[/green]")
    else:
        console.print("No server running.")


# ── Workers (Multi-Process Workers) ──────────────────────────────────


@main.group()
def workers():
    """Multi-process worker management."""
    pass


@workers.command("spawn")
@click.argument("count", type=int, default=2)
@click.option("--model", "-m", default=None)
@click.option("--engine", "-e", type=click.Choice(["auto", "llama.cpp", "vllm"]), default="auto")
@click.pass_context
def workers_spawn(ctx, count, model, engine):
    """Spawn N worker processes connecting to the inference server.

    Starts a server automatically if one isn't running.
    """
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.multiprocess import MultiProcessCoordinator, assess_multiprocess
    from tqcli.core.server import InferenceServer, ServerConfig
    from tqcli.core.system_info import detect_system
    from tqcli.core.unrestricted import is_unrestricted
    from tqcli.ui.console import console

    unrestricted = is_unrestricted(ctx)
    config = TqConfig.load()
    config.ensure_dirs()
    sys_info = detect_system()

    registry = ModelRegistry(config.models_dir)
    registry.scan_local_models()
    available = registry.get_available_models()
    if not available:
        console.print("[yellow]No models installed.[/yellow]")
        return

    target = available[0]
    if model:
        target = registry.get_profile(model) or target
    if not target.local_path:
        console.print(f"[red]Model not installed.[/red]")
        return

    # Assess
    plan = assess_multiprocess(
        sys_info=sys_info,
        model_path=str(target.local_path),
        model_size_mb=target.min_ram_mb,
        requested_workers=count,
        preferred_engine=engine,
        unrestricted=unrestricted,
    )

    if not plan.feasible and not unrestricted:
        console.print(f"[red]Cannot spawn {count} workers: {plan.reason}[/red]")
        for w in plan.warnings:
            console.print(f"  [yellow]{w}[/yellow]")
        console.print(f"\n  Max workers for your hardware: {plan.max_workers}")
        console.print("  Use --stop-trying-to-control-everything-and-just-let-go to bypass.")
        return

    for w in plan.warnings:
        console.print(f"  [yellow]Warning: {w}[/yellow]")

    effective_count = plan.recommended_workers if not unrestricted else count

    console.print(f"[bold]Spawning {effective_count} workers[/bold]")
    console.print(f"  Engine: {plan.engine}")
    console.print(f"  Model:  {target.display_name}")

    # Check if server is running, start if not
    server = InferenceServer(ServerConfig(engine="", model_path=""))
    if not server.is_running():
        console.print(f"\n  Starting inference server first...")
        ctx.invoke(serve_start, model=target.id if model else None, engine=engine, port=8741)
        if not server.is_running():
            # Server might still be starting, give it a moment
            import time
            time.sleep(3)

    plan.model_id = target.id
    coordinator = MultiProcessCoordinator(config, plan)

    console.print()
    for i in range(effective_count):
        worker = coordinator.spawn_worker(i + 1)
        console.print(f"  Worker {worker.id} started (PID {worker.pid})")

    console.print(f"\n[green]{effective_count} workers active.[/green]")
    console.print(f"Each worker is an independent tqcli chat session connected to the server.")
    console.print(f"\nStop with: tqcli serve stop (stops server + all workers)")


@workers.command("list")
def workers_list():
    """List active worker processes."""
    import psutil
    from tqcli.ui.console import console

    # Find tqcli processes with --engine server
    tqcli_workers = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline)
            if "tqcli" in cmd_str and "--engine" in cmd_str and "server" in cmd_str:
                tqcli_workers.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if not tqcli_workers:
        console.print("No active workers found.")
        console.print("Spawn workers with: tqcli workers spawn 2")
        return

    console.print(f"[bold]Active Workers ({len(tqcli_workers)})[/bold]\n")
    for w in tqcli_workers:
        console.print(f"  PID {w['pid']}")


@workers.command("stop")
def workers_stop():
    """Stop all worker processes."""
    import psutil
    from tqcli.ui.console import console

    stopped = 0
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            cmd_str = " ".join(cmdline)
            if "tqcli" in cmd_str and "--engine" in cmd_str and "server" in cmd_str:
                proc.terminate()
                stopped += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if stopped:
        console.print(f"[green]Stopped {stopped} worker(s).[/green]")
    else:
        console.print("No active workers found.")
