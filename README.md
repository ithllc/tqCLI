# tqCLI (TurboQuant CLI)

**Version:** 0.7.0 — [release notes in CHANGELOG.md](CHANGELOG.md). Distributed on PyPI as `turboquant-cli`.

A cross-platform CLI for **local LLM inference** using quantized models, with smart routing, real-time performance monitoring, TurboQuant KV cache compression, and automatic handoff to frontier model CLIs when local inference falls below acceptable thresholds.

Built with [TurboQuant](https://arxiv.org/abs/2504.19874) methodologies — applying quantization best practices from Google Research's ICLR 2026 paper on lossless 3-bit KV cache compression.

## What's new in 0.7.0

- **TurboQuant fork wheels are now distributed packages.** `pip install turboquant-cli[llama-tq]` pulls cibuildwheel matrix wheels for `llama-cpp-python-turboquant` from PyPI (Linux/macOS/Windows × Py 3.10–3.12 × CPU/CUDA/Metal). `pip install turboquant-cli[vllm-tq]` pulls `vllm-turboquant` from the GitHub Release on `tqcli/vllm-turboquant` (Ampere/Ada/Hopper). Blackwell consumers (sm_100/120/121) install `[vllm-tq-blackwell]` which targets the dedicated CUDA 13.0 build with PTX hedge for Rubin.
- **Engine Auditor** (`tqcli/core/engine_auditor.py`) detects fork-vs-upstream on every CLI start and emits a yellow Rich panel with the exact `pip install` command when your hardware supports TurboQuant but the installed engine is upstream. Stays silent on capable+fork installs and on hardware that cannot run TurboQuant. Suppress with `TQCLI_SUPPRESS_AUDIT=1`. In `--json` mode the audit is emitted as a one-line stderr metadata blob, never on stdout. In agent modes (`--ai-tinkering` / unrestricted), the audit panel is flushed BEFORE the orchestrator's first stream chunk so tool-call tags can't interleave.
- **Distribution name on PyPI is `turboquant-cli`.** The import name is unchanged: `import tqcli` still works, the entry-point `tqcli` script is unchanged. (dateutil pattern: `pip install python-dateutil` -> `import dateutil`.)
- **License:** Apache-2.0 across the umbrella package and the vLLM fork. The llama.cpp fork inherits MIT from upstream.
- **GitHub Sponsors** wired via `.github/FUNDING.yml` -> `ithllc` (the LLC behind tqCLI).

## What's new in 0.6.2

- **LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM capture wrappers** in the KV-metadata calibrator — Qwen 3 was joined by Llama 3 / Mistral / Phi-3 with shared accumulators and head_dim derivation when configs omit it explicitly.

## What's new in 0.6.1

- **TurboQuant KV metadata auto-calibration** ([#27](https://github.com/ithllc/tqCLI/issues/27)) — models that don't ship a calibrated `turboquant_kv.json` (e.g. Qwen 3 4B BF16 on vLLM) are now automatically calibrated on first load via a 30-prompt activation sweep. Mirrors the fork's own reference outlier-selection math; captures post-RoPE K and V; refuses AWQ/GPTQ and variable-head-dim sources with clear reasons.
- **Expanded paragraph-length calibration corpus** ([#28](https://github.com/ithllc/tqCLI/issues/28)) — 30 domain-diverse prompts (~5,100 Qwen3 tokens); enforced by unit test via real tokenizer.
- **Explicit `tqcli model calibrate-kv <id>`** ([#29](https://github.com/ithllc/tqCLI/issues/29)) — pre-warm TurboQuant metadata without going through a chat session. Useful for CI and pre-production environments.
- **Perplexity validation gate** ([#30](https://github.com/ithllc/tqCLI/issues/30)) — opt-in integration test asserts PPL(turboquant35) / PPL(kv:auto) ≤ 1.05 on a 10-prompt corpus. Current measured ratio: **0.9997** (indistinguishable from baseline).
- **Agent-modes integration suite** re-validated end-to-end on Qwen 3 4B vLLM with real TurboQuant (no more `kv:none` fallback). `agent_modes_functional_report.md`: 11/11 functional + 4/4 data-point PASS.

## What's new in 0.6.0

- **Tri-state agentic autonomy** for `tqcli chat` — `manual` (default), `--ai-tinkering` (stages each tool call for `[Y/n/Edit]` approval), and the existing `--stop-trying-to-control-everything-and-just-let-go` mode upgraded into a full ReAct loop bounded by `--max-agent-steps`. Four core agent tools: `tq-file-read`, `tq-file-write`, `tq-terminal-exec`, `tq-interactive-prompt`. See [`docs/architecture/agent_orchestrator.md`](docs/architecture/agent_orchestrator.md).
- **Verified end-to-end** with Gemma 4 E2B on both engines — `tests/integration_reports/agent_modes_report.md`.

## What's new in 0.5.0

- **AI Skills Builder** ([#23](https://github.com/ithllc/tqCLI/issues/23)) — `tqcli skill generate --prd A.md --plan B.md --name my-skill` turns a PRD + Technical Plan into a working `~/.tqcli/skills/<name>/` scaffold using the local LLM (TurboQuant-aware, llama.cpp + vLLM).
- **Headless chat + vLLM multimodal** ([#24](https://github.com/ithllc/tqCLI/issues/24)) — `tqcli chat` gains `--prompt`, `--image`, `--audio`, `--messages`, `--json`, `--max-tokens`; `VllmBackend` now passes images to `self._llm.generate()` via `multi_modal_data={"image": [...]}` so Gemma 4 E2B handles image inputs under TurboQuant KV + CPU offload.
- **TurboQuant KV cache compression** on both llama.cpp and vLLM backends — 3.8×–6.4× cache compression with near-lossless to minimal quality trade-off (gated by CUDA 12.8+; graceful `kv:none` fallback otherwise).
- **Unified quantization pipeline** — detects model precision and applies weight quant (bnb_int4/AWQ/GGUF) + KV compression as an explicit plan; no more silent surprises at load time.
- **Gemma 4 E2B on 4 GB VRAM** via vLLM BNB_INT4 + CPU offload (9.9 GB) + turboquant35 — verified end-to-end in the integration suite.
- **Multi-process mode** with shared server + workers (llama.cpp sequential or vLLM continuous batching).
- **Closed issues shipped:** [#13](https://github.com/ithllc/tqCLI/issues/13), [#20](https://github.com/ithllc/tqCLI/issues/20), [#22](https://github.com/ithllc/tqCLI/issues/22).

See [`docs/examples/USAGE.md`](docs/examples/USAGE.md) for verified end-to-end flows.

## What It Does

- **Run quantized LLMs locally** on Mac, Linux, or Windows via [llama.cpp](https://github.com/ggerganov/llama.cpp) or [vLLM](https://github.com/vllm-project/vllm)
- **Smart routing** automatically dispatches prompts to the best model for the task (coding, reasoning, general)
- **Multi-process mode** with a shared inference server and multiple worker processes (llama.cpp sequential or vLLM continuous batching)
- **Performance monitoring** tracks tokens/second in real-time with live stats display
- **Automatic handoff** generates context files to transfer work to Claude Code, Gemini CLI, or Aider when local inference is too slow
- **Security isolation** via Python venvs, resource guards, and audit logging
- **Unrestricted mode** (`--stop-trying-to-control-everything-and-just-let-go`) to bypass resource guards for advanced users
- **Skills system** with discoverable, extensible skills (like Claude Code's skill architecture)

## Supported Models

| Model Family | Variants | Strengths | Best For |
|-------------|----------|-----------|----------|
| **Google Gemma 4** | E2B, E4B, 26B MoE, 31B Dense | General reasoning, multimodal, 140+ languages, 256K context | Q&A, analysis, general tasks |
| **Qwen3** | 4B, 8B, 32B, 30B-A3B MoE | Thinking mode, reasoning, 128K context, 119 languages | Reasoning, general chat, math |
| **Qwen3-Coder** | Coder-Next 80B MoE, Coder-30B-A3B | Code generation, agentic coding, 256K context | Coding, debugging, SWE tasks |

All models available as GGUF quantized files on HuggingFace for llama.cpp. Qwen 3 models feature built-in **thinking mode** — deep reasoning chains that the router enables automatically for complex tasks.

## Platform Support

| Platform | llama.cpp | vLLM | Notes |
|----------|-----------|------|-------|
| macOS (Apple Silicon) | Full (Metal GPU) | No | Unified memory enables large models |
| macOS (Intel) | CPU only | No | |
| Linux + NVIDIA GPU | Full (CUDA) | Full | Best vLLM experience |
| Linux (CPU only) | Full | No | |
| Windows (WSL2 + NVIDIA) | Full (CUDA) | Full | Recommended for Windows |
| Windows (Native) | CPU only | No | WSL2 recommended |

## Quick Start

### Install

```bash
# Most users: install from PyPI (distribution name is `turboquant-cli`).
# llama.cpp fork wheel pulls automatically from PyPI.
pip install 'turboquant-cli[llama-tq]'

# vLLM fork (Ampere/Ada/Hopper). Wheel lives on the GitHub Release, so we
# pass `--find-links` to pip:
pip install 'turboquant-cli[vllm-tq]' \
  --find-links https://github.com/tqcli/vllm-turboquant/releases/latest

# Blackwell hardware (sm_100 / sm_120 / sm_121) — opt in to the Blackwell
# wheel built against CUDA 13.0:
pip install 'turboquant-cli[vllm-tq-blackwell]' \
  --find-links https://github.com/tqcli/vllm-turboquant/releases/latest

# Both engines — Ampere/Ada/Hopper default. (Note: `[all]` does NOT include
# `[vllm-tq-blackwell]`; Blackwell users must opt in explicitly so a wheel
# resolver does not pull sm_100+ kernels onto an Ada box.)
pip install 'turboquant-cli[all]' \
  --find-links https://github.com/tqcli/vllm-turboquant/releases/latest

# macOS users: install only `[llama-tq]`. There is no Darwin wheel for
# vLLM — `[all]` will fail to resolve `vllm-turboquant` on a Mac.

# Editable / development install (cloning the source tree):
git clone https://github.com/tqcli/tqcli.git
cd tqcli
pip install -e '.[llama-tq]'   # or .[vllm-tq], .[all], .[dev]
```

### Initialize

```bash
# Set up configuration and directories
tqcli config init

# Check your system capabilities
tqcli system info
```

### Download a Model

```bash
# List available models
tqcli model list

# Download based on your hardware (see system info output)
tqcli model pull gemma-4-e4b-it-Q4_K_M            # 4.5B edge model (needs ~4GB)
tqcli model pull qwen3-8b-Q4_K_M                  # 8B general w/ thinking mode (needs ~6GB)
tqcli model pull qwen3-coder-30b-a3b-instruct-Q4_K_M  # 30B MoE coder (needs ~18GB)
```

### Chat

```bash
# Interactive chat (auto-selects best model via router)
tqcli

# Chat with a specific model
tqcli chat --model qwen3-coder-30b-a3b-instruct-Q4_K_M

# Enable TurboQuant KV cache compression (CUDA 12.8+ required)
tqcli chat --model qwen3-4b-Q4_K_M --kv-quant turbo3

# Pre-warm TurboQuant metadata on a vLLM model (0.6.1+; optional — auto-runs on first chat load)
tqcli model calibrate-kv qwen3-4b-vllm

# In-session commands:
#   /stats    — Show performance statistics
#   /image    — Attach an image (multimodal models only)
#   /audio    — Attach an audio file
#   /think    — Force Qwen 3 thinking mode for the next turn
#   /handoff  — Generate handoff file for frontier CLI
#   /help     — Show help
#   /quit     — Exit
```

## Smart Routing

When multiple models are installed, tqCLI automatically routes prompts to the best model:

```
User: "Write a Python function to parse JSON"
  -> Routes to Qwen3-Coder (coding: 0.95 score)

User: "Prove that the square root of 2 is irrational"
  -> Routes to Qwen3 32B (reasoning: 0.92 score) +thinking mode

User: "Summarize this article for me"
  -> Routes to Gemma 4 31B (general: 0.95 score)
```

Qwen 3 models support **thinking mode** — the router automatically enables it for coding, math, and reasoning tasks. Override per-message with `/think` or `/no_think` in the chat session.

The router classifies prompts using keyword and pattern heuristics, then ranks available models by their domain-specific strength scores. If only one model is loaded, classification is skipped.

## Multi-Process Mode

For parallel workloads, tqCLI can run a shared inference server with multiple worker processes:

```bash
# Start a shared inference server
tqcli serve start --model qwen3-coder-30b-a3b-instruct-Q4_K_M

# Spawn worker processes that connect to the server
tqcli workers spawn 3

# Or connect a single chat session to the server
tqcli chat --engine server
```

tqCLI auto-selects the best server engine for your hardware:

| Engine | How It Handles Multiple Workers | When to Use |
|--------|-------------------------------|-------------|
| **llama.cpp server** | Sequential queue (workers wait in line) | Cross-platform, any hardware |
| **vLLM server** | Continuous batching + PagedAttention (true parallel) | Linux + 8 GB+ VRAM |

The coordinator assesses your hardware before spawning workers and will refuse to start more than your system can handle (unless you use unrestricted mode).

```bash
# Check what your system can support
tqcli serve status
tqcli workers list
tqcli serve stop       # stop server + all workers
```

## Performance Monitoring & Handoff

tqCLI tracks tokens/second during every inference:

```
  32.8 tok/s | 128 tokens | 3.90s
```

When performance drops below threshold (default: 5 tok/s), tqCLI generates a **handoff file** — a self-contained markdown document with:
- Task description and conversation history
- Performance stats explaining why the handoff occurred  
- CLI-specific instructions for continuing with Claude Code, Gemini CLI, or Aider

```bash
# Manual handoff
tqcli handoff -t "Implement authentication middleware" --target claude-code

# Auto-handoff (enable in config)
# Set performance.auto_handoff: true in ~/.tqcli/config.yaml
```

## Security

tqCLI runs in an isolated environment by default:

- **Virtual environment** — Separate Python venv at `~/.tqcli/venv`
- **Resource guards** — Memory and GPU limits prevent system overload
- **Audit logging** — All model loads, downloads, and security events logged
- **Environment detection** — Identifies WSL2, containers, bare-metal

```bash
# Run security audit
tqcli security audit

# Auto-fix safe issues
tqcli security audit --fix
```

## Skills

tqCLI includes a skills system inspired by Claude Code:

| Skill | Description |
|-------|-------------|
| `tq-system-info` | Detect OS, hardware, recommend engine and quantization |
| `tq-model-manager` | Download, list, and remove quantized models |
| `tq-benchmark` | Benchmark models for tokens/second performance |
| `tq-security-audit` | Audit environment isolation and security posture |
| `tq-handoff-generator` | Generate handoff files for frontier model CLIs |
| `tq-multi-process` | Multi-process orchestration with shared inference server |
| `tq-model-updater` | Research and update model registry with latest HF repos |

```bash
tqcli skills  # List all available skills
```

## Configuration

Configuration lives at `~/.tqcli/config.yaml`:

```yaml
models_dir: ~/.tqcli/models
preferred_engine: auto          # auto, llama.cpp, vllm
default_quantization: Q4_K_M
context_length: 4096
n_gpu_layers: -1                # -1 = all layers to GPU

performance:
  min_tokens_per_second: 5.0    # Below this = handoff trigger
  warning_tokens_per_second: 10.0
  auto_handoff: false

security:
  use_venv: true
  sandbox_enabled: true
  audit_log: true
  max_memory_percent: 80.0
  max_gpu_memory_percent: 90.0

router:
  enabled: true

multiprocess:
  server_host: 127.0.0.1
  server_port: 8741
  max_workers: 3
  auto_start_server: true
```

## Unrestricted Mode

For experienced users who know their hardware better than our heuristics:

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go chat
tqcli --stop-trying-to-control-everything-and-just-let-go serve start
tqcli --stop-trying-to-control-everything-and-just-let-go workers spawn 5
```

This is equivalent to Claude Code's `--dangerously-skip-permissions` and Gemini CLI's `--yolo`. It bypasses resource guards, confirmation prompts, and safety checks. Audit logging remains active.

When combined with `tqcli chat`, this flag upgrades the chat loop into a **ReAct agent**: the model emits `<tool_call>{...}</tool_call>` blocks that fire immediately (file read/write, shell exec, human prompt), up to `--max-agent-steps` iterations (default 10).

## Agent Modes

Three-tier autonomy for `tqcli chat`:

```bash
# Manual (default) — plain chat. CI-safe; no tool schemas injected.
tqcli chat

# AI Tinkering — model stages tool calls, you approve each actionable one.
tqcli chat --ai-tinkering

# Unrestricted (yolo) — model executes tool calls autonomously in a ReAct loop.
tqcli --stop-trying-to-control-everything-and-just-let-go chat --max-agent-steps 10
```

Core agent tools (`tq-file-read`, `tq-file-write`, `tq-terminal-exec`, `tq-interactive-prompt`) are injected as OpenAI-style JSON schemas into the system prompt only when an agent mode is active. Manual mode passes an **empty** tool list so headless JSON-stdout pipelines remain deterministic.

## TurboQuant Reference

This project applies methodologies from [Google's TurboQuant research](https://arxiv.org/abs/2504.19874):

- **PolarQuant**: Polar coordinate transformation for lossless KV cache compression
- **QJL**: Quantized Johnson-Lindenstrauss sign-bit error correction
- **Result**: 6x+ KV cache compression at 3-bit with zero accuracy loss

The quantization decision tree in tqCLI (engine selection, quant level recommendation, model sizing) is informed by the comparative analysis in the [GoogleTurboQuant](https://github.com/ithllc/GoogleTurboQuant) workspace.

## Documentation

| Destination | What's there |
|-------------|--------------|
| [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) | Step-by-step install, first model, first chat |
| [docs/examples/USAGE.md](docs/examples/USAGE.md) | End-to-end flows: image/audio input, llama.cpp + vLLM TurboQuant, multi-process CRM |
| [docs/architecture/](docs/architecture/README.md) | Subsystem-by-subsystem design with Mermaid diagrams |
| [docs/architecture/turboquant_kv.md](docs/architecture/turboquant_kv.md) | KV compression levels, per-engine dtype mapping, CUDA gating |
| [docs/architecture/quantization_pipeline.md](docs/architecture/quantization_pipeline.md) | Detect precision → weight quant → KV stages |
| [docs/architecture/multi_process.md](docs/architecture/multi_process.md) | Server + workers model, `assess_multiprocess`, engine concurrency |
| [docs/contributing/](docs/contributing/) | Branch protection + skill-quality compensation strategy |
| [docs/design/WHY_PYTHON.md](docs/design/WHY_PYTHON.md) | Language-choice rationale |
| [docs/prompts/](docs/prompts/) | Resume + Gemini research + future-work implementation prompts |
| [tests/integration_reports/turboquant_kv_comparison_report.md](tests/integration_reports/turboquant_kv_comparison_report.md) | Latest integration run: 7/7 tests, 137/137 step assertions |

## License

MIT License. See [LICENSE](LICENSE) for details.
