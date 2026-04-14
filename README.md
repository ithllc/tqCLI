# tqCLI (TurboQuant CLI)

A cross-platform CLI for **local LLM inference** using quantized models, with smart routing, real-time performance monitoring, and automatic handoff to frontier model CLIs when local inference falls below acceptable thresholds.

Built with [TurboQuant](https://arxiv.org/abs/2504.19874) methodologies — applying quantization best practices from Google Research's ICLR 2026 paper on lossless 3-bit KV cache compression.

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
# Clone the repository
git clone https://github.com/ithllc/tqCLI.git
cd tqCLI

# Install with llama.cpp backend (most platforms)
pip install -e ".[llama]"

# Or with vLLM backend (Linux/WSL2 + NVIDIA GPU)
pip install -e ".[vllm]"

# Or both
pip install -e ".[all]"
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
tqcli chat --model qwen2.5-coder-7b-instruct-Q4_K_M

# In-session commands:
#   /stats    — Show performance statistics
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
tqcli serve start --model qwen2.5-coder-7b-instruct-Q4_K_M

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

## TurboQuant Reference

This project applies methodologies from [Google's TurboQuant research](https://arxiv.org/abs/2504.19874):

- **PolarQuant**: Polar coordinate transformation for lossless KV cache compression
- **QJL**: Quantized Johnson-Lindenstrauss sign-bit error correction
- **Result**: 6x+ KV cache compression at 3-bit with zero accuracy loss

The quantization decision tree in tqCLI (engine selection, quant level recommendation, model sizing) is informed by the comparative analysis in the [GoogleTurboQuant](https://github.com/ithllc/GoogleTurboQuant) workspace.

## License

MIT License. See [LICENSE](LICENSE) for details.
