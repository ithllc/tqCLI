"""Interactive chat mode for tqCLI."""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.text import Text

from tqcli.config import TqConfig
from tqcli.core.engine import ChatMessage, InferenceEngine
from tqcli.core.handoff import generate_handoff
from tqcli.core.performance import PerformanceMonitor
from tqcli.core.router import ModelRouter
from tqcli.core.thinking import (
    ThinkingConfig,
    ThinkingFormat,
    build_system_prompt_with_thinking,
    detect_thinking_format,
    is_inside_thinking_block,
    strip_thinking_blocks,
)
from tqcli.ui.console import (
    console,
    print_handoff_alert,
    print_performance_warning,
    print_route_decision,
    print_stats_bar,
)

SYSTEM_PROMPT = (
    "You are a helpful AI assistant running locally via tqCLI (TurboQuant CLI). "
    "You are a quantized model optimized for fast local inference. "
    "Be concise and accurate."
)


class InteractiveSession:
    """Manages an interactive chat session with streaming, routing, and performance monitoring."""

    def __init__(
        self,
        config: TqConfig,
        engine: InferenceEngine,
        router: ModelRouter | None = None,
        monitor: PerformanceMonitor | None = None,
        model_family: str = "",
    ):
        self.config = config
        self.engine = engine
        self.router = router
        self.monitor = monitor or PerformanceMonitor(config.performance)
        self._thinking_fmt = detect_thinking_format(model_family)
        self._thinking_config = ThinkingConfig(format=self._thinking_fmt, enabled=False)
        self.history: list[ChatMessage] = [ChatMessage(role="system", content=SYSTEM_PROMPT)]
        self._conversation_dicts: list[dict] = []

    def chat_turn(self, user_input: str, images: list[str] | None = None, audio: list[str] | None = None) -> str:
        # Qwen 3 thinking mode: user can override per-turn with /think or /no_think
        use_thinking = False
        effective_input = user_input
        if user_input.strip().startswith("/think "):
            use_thinking = True
            effective_input = user_input.strip()[7:]
        elif user_input.strip().startswith("/no_think "):
            use_thinking = False
            effective_input = user_input.strip()[10:]

        # Parse /image and /audio prefixes from input
        parsed_images = list(images or [])
        parsed_audio = list(audio or [])
        while effective_input.strip().startswith("/image "):
            parts = effective_input.strip()[7:].split(None, 1)
            if parts:
                parsed_images.append(parts[0])
                effective_input = parts[1] if len(parts) > 1 else ""
        while effective_input.strip().startswith("/audio "):
            parts = effective_input.strip()[7:].split(None, 1)
            if parts:
                parsed_audio.append(parts[0])
                effective_input = parts[1] if len(parts) > 1 else ""

        msg = ChatMessage(
            role="user", content=effective_input,
            images=parsed_images or None, audio=parsed_audio or None,
        )
        self.history.append(msg)
        self._conversation_dicts.append({"role": "user", "content": effective_input})

        # Route if router is available and multiple models exist
        if self.router:
            try:
                decision = self.router.route(effective_input)
                # Use router's thinking recommendation unless user overrode
                if not user_input.strip().startswith(("/think ", "/no_think ")):
                    use_thinking = decision.use_thinking
                print_route_decision(decision)
                # Update thinking format if model changed
                self._thinking_fmt = detect_thinking_format(decision.model.family)
                self._thinking_config = ThinkingConfig(
                    format=self._thinking_fmt, enabled=use_thinking
                )
                if use_thinking:
                    fmt_name = "Qwen3" if self._thinking_fmt == ThinkingFormat.QWEN3 else "Gemma4"
                    console.print(f"  [dim]Thinking mode: enabled ({fmt_name} format)[/dim]")
                # If routed to a different model than currently loaded, switch
                if decision.model.local_path and str(decision.model.local_path) != getattr(
                    self.engine, "_model_path", ""
                ):
                    console.print(f"  [dim]Switching to {decision.model.display_name}...[/dim]")
                    self.engine.unload_model()
                    self.engine.load_model(str(decision.model.local_path))
            except RuntimeError:
                pass  # No models available, just use what's loaded

        # Inject thinking tokens into system prompt if enabled
        if self._thinking_config.is_active and self.history and self.history[0].role == "system":
            self.history[0] = ChatMessage(
                role="system",
                content=build_system_prompt_with_thinking(SYSTEM_PROMPT, self._thinking_config),
            )

        # Stream response
        full_response = ""
        final_stats = None

        console.print()
        with Live(Text(""), console=console, refresh_per_second=15) as live:
            buffer = ""
            for chunk, stats in self.engine.chat_stream(self.history):
                if stats:
                    final_stats = stats
                    break
                buffer += chunk
                full_response += chunk
                # Render: show thinking blocks dimmed, strip completed ones
                if is_inside_thinking_block(buffer, self._thinking_fmt):
                    # Inside an unclosed thinking block — show dimmed
                    clean_part = strip_thinking_blocks(buffer, self._thinking_fmt)
                    display = Text(clean_part)
                    # Find the unclosed portion and append dimmed
                    remainder = buffer[len(clean_part):] if len(clean_part) < len(buffer) else ""
                    if remainder:
                        display.append(remainder, style="dim")
                    live.update(display)
                else:
                    clean = strip_thinking_blocks(buffer, self._thinking_fmt)
                    live.update(Text(clean))

        self.history.append(ChatMessage(role="assistant", content=full_response))
        self._conversation_dicts.append({"role": "assistant", "content": full_response})

        # Record performance
        if final_stats:
            self.monitor.record(final_stats.completion_tokens, final_stats.completion_time_s)
            print_stats_bar(final_stats)

            # Check performance thresholds
            if self.monitor.is_warning:
                print_performance_warning(self.monitor)
            elif self.monitor.should_handoff and self.config.performance.auto_handoff:
                self._do_handoff(user_input)

        return full_response

    def _do_handoff(self, last_task: str):
        output_dir = Path.home() / ".tqcli" / "handoffs"
        filepath = generate_handoff(
            monitor=self.monitor,
            conversation_history=self._conversation_dicts,
            task_description=last_task,
            output_dir=output_dir,
        )
        print_handoff_alert(filepath)

    def run(self):
        console.print("[bold cyan]tqCLI Interactive Chat[/bold cyan]")
        console.print("[dim]Type /quit to exit, /stats for performance, /handoff to generate handoff[/dim]")
        console.print("[dim]/think <msg> to force reasoning, /no_think <msg> to skip it[/dim]\n")

        while True:
            try:
                user_input = console.input("[bold green]> [/bold green]")
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not user_input.strip():
                continue

            cmd = user_input.strip().lower()
            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break
            elif cmd == "/stats":
                stats = self.monitor.get_stats_display()
                for k, v in stats.items():
                    console.print(f"  {k}: {v}")
                continue
            elif cmd == "/handoff":
                self._do_handoff("User-requested handoff")
                continue
            elif cmd == "/help":
                console.print("  /quit     — Exit chat")
                console.print("  /stats    — Show performance statistics")
                console.print("  /handoff  — Generate handoff file for frontier model CLI")
                console.print("  /think    — Prefix a message to force thinking mode (Qwen 3)")
                console.print("  /no_think — Prefix a message to skip thinking mode")
                console.print("  /help     — Show this help")
                continue

            self.chat_turn(user_input)
