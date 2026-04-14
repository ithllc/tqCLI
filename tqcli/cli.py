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
    ctx.ensure_object(dict)
    ctx.obj["unrestricted"] = unrestricted
    if unrestricted:
        from tqcli.core.unrestricted import show_unrestricted_warning
        from tqcli.ui.console import console
        show_unrestricted_warning(console)
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
@click.pass_context
def chat(ctx, model, engine, context_length, server_url, kv_quant):
    """Start an interactive chat session.

    Single-process (default): loads model in-process.
    Multi-process: connect to a shared server with --engine server or --server-url.
    """
    from tqcli.config import TqConfig
    from tqcli.core.model_registry import ModelRegistry
    from tqcli.core.performance import PerformanceMonitor
    from tqcli.core.router import ModelRouter
    from tqcli.core.security import SecurityManager
    from tqcli.core.system_info import detect_system
    from tqcli.core.unrestricted import is_unrestricted
    from tqcli.ui.console import console, print_banner
    from tqcli.ui.interactive import InteractiveSession

    unrestricted = is_unrestricted(ctx)
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
        session = InteractiveSession(config, eng, None, monitor)
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
        tune = build_vllm_config(target_for_tune, sys_info, requested_max_len=context_length)
        if not tune.feasible and not unrestricted:
            console.print(f"[red]vLLM not feasible: {tune.reason}[/red]")
            console.print("Use --stop-trying-to-control-everything-and-just-let-go to bypass.")
            return
        for w in tune.warnings:
            console.print(f"  [dim]vLLM: {w}[/dim]")
        eng = VllmBackend.from_tuning_profile(tune)
    else:
        from tqcli.core.llama_backend import LlamaBackend
        from tqcli.core.kv_quantizer import KVQuantLevel, get_llama_kv_params, select_kv_quant
        # Determine KV cache compression level
        kv_level = select_kv_quant(available_kv_mb=50, engine="llama.cpp", user_choice=kv_quant)
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

    # Launch interactive session
    session = InteractiveSession(config, eng, router, monitor, model_family=target.family)
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
    from tqcli.ui.console import print_system_info

    info = detect_system()
    if as_json:
        data = {
            "os": info.os_name,
            "os_display": info.os_display,
            "arch": info.arch,
            "is_wsl": info.is_wsl,
            "cpu_cores": info.cpu_cores_logical,
            "ram_total_mb": info.ram_total_mb,
            "ram_available_mb": info.ram_available_mb,
            "gpus": [{"name": g.name, "vram_mb": g.vram_total_mb} for g in info.gpus],
            "recommended_engine": info.recommended_engine,
            "recommended_quant": info.recommended_quant,
            "max_model_gb": info.max_model_size_estimate_gb,
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
