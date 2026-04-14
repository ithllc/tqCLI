# Getting Started with tqCLI

This guide walks you through installing tqCLI, downloading your first model, and running local inference on your specific platform.

## Prerequisites

- **Python 3.10 or higher** — check with `python3 --version`
- **Git** — for cloning the repository
- **8 GB+ RAM** — minimum for running a 7B quantized model
- **(Optional) NVIDIA GPU** — for GPU-accelerated inference

## Step 1: Install tqCLI

### From Source (recommended during alpha)

```bash
git clone https://github.com/ithllc/tqCLI.git
cd tqCLI

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows PowerShell

# Install tqCLI
pip install -e ".[llama]"   # with llama.cpp backend
# pip install -e ".[vllm]"  # with vLLM backend (Linux + NVIDIA only)
# pip install -e ".[all]"   # both backends
```

### Verify Installation

```bash
tqcli --version
# tqcli, version 0.1.0
```

## Step 2: Check Your System

```bash
tqcli system info
```

This tells you:
- Your OS and whether you're in WSL2
- CPU cores and available RAM
- GPU model and VRAM (if NVIDIA)
- Which inference engine is recommended
- Maximum model size that fits your hardware
- Recommended quantization level

**Example output on a WSL2 system with NVIDIA GPU:**
```
=== System Info ===
OS:          Linux (Ubuntu 22.04) [WSL2]
CPU:         x86_64 (8c/16t)
RAM:         31,956 MB total / 28,504 MB available
GPU:         NVIDIA RTX A2000 Laptop GPU (4,096 MB VRAM)
Engine:      llama.cpp (recommended)
Max Model:   ~3.4 GB
Quant:       Q3_K_M recommended
```

## Step 3: Initialize Configuration

```bash
tqcli config init
```

This creates `~/.tqcli/config.yaml` with sensible defaults. Review it:

```bash
tqcli config show
```

## Step 4: Download a Model

Check what's available and pick based on your system info:

```bash
tqcli model list
```

### Choosing the Right Model

**If you have 20+ GB available memory:**
```bash
tqcli model pull gemma-4-31b-it-Q4_K_M         # Best general-purpose (256K context)
tqcli model pull qwen3-32b-Q4_K_M              # Best reasoning with thinking mode
```

**If you have 6-8 GB available:**
```bash
tqcli model pull qwen3-8b-Q4_K_M              # General + thinking mode (128K context)
tqcli model pull gemma-4-e4b-it-Q4_K_M        # Gemma edge model (128K context)
```

**If you have 3-4 GB available (edge/constrained):**
```bash
tqcli model pull qwen3-4b-Q4_K_M              # Small but capable (rivals Qwen2.5-72B!)
tqcli model pull gemma-4-e2b-it-Q4_K_M        # Smallest Gemma with multimodal
```

**If you have limited VRAM (4 GB) but plenty of RAM (16+ GB):**

The model will run partially on GPU and partially on RAM. This works — it's slower than full GPU but faster than CPU-only. tqCLI's llama.cpp backend handles this automatically.

**For coding specifically:**
```bash
tqcli model pull qwen3-coder-30b-a3b-instruct-Q4_K_M  # MoE coder (3B active params)
```

### Understanding Memory

```
Model on disk (GGUF file)  ≈  What it needs in memory
Q4_K_M 7B model:   ~4.5 GB file → ~5.5-7 GB in memory (model + KV cache + overhead)
Q4_K_M 12B model:  ~7.2 GB file → ~9-11 GB in memory
```

**Memory breakdown:**
- Model weights: the GGUF file size
- KV cache: ~0.5-2 GB depending on context length
- Engine overhead: ~0.5 GB
- OS and other processes: leave 2-4 GB headroom

## Step 5: Chat

```bash
# Auto-selects the best model for each prompt
tqcli

# Or specify a model
tqcli chat --model qwen2.5-coder-7b-instruct-Q4_K_M
```

### In-Session Commands

| Command | What It Does |
|---------|-------------|
| `/stats` | Show tokens/second performance statistics |
| `/handoff` | Generate a handoff file for Claude Code or Gemini CLI |
| `/help` | Show available commands |
| `/quit` | Exit the chat session |

### Understanding the Performance Display

After each response, tqCLI shows:

```
  32.8 tok/s | 128 tokens | 3.90s
```

- **tok/s** — tokens generated per second. Higher is better.
  - Green (20+): excellent
  - Yellow (10-20): good
  - Red (<10): slow — consider handoff
- **tokens** — how many tokens were generated
- **time** — total wall clock time

## Step 6: Run a Security Audit

```bash
tqcli security audit
```

This checks your environment isolation, resource limits, and file permissions.

## Platform-Specific Notes

### macOS (Apple Silicon — M1/M2/M3/M4)

- llama.cpp uses **Metal** for GPU acceleration automatically
- Unified memory means the GPU and CPU share RAM — a 32 GB M2 Max can load a 27B Q4_K_M model
- vLLM is **not supported** on macOS (requires CUDA)

```bash
pip install -e ".[llama]"  # Metal acceleration included
```

### macOS (Intel)

- CPU-only inference (no GPU acceleration)
- Limited to smaller models (7B Q4_K_M comfortably)

### Linux (with NVIDIA GPU)

- Best platform for both llama.cpp (CUDA) and vLLM
- For vLLM: needs 16+ GB VRAM for 7B models

```bash
# Check CUDA is working
nvidia-smi

# Install with both backends
pip install -e ".[all]"
```

### Windows (WSL2) — Recommended for Windows

- Install WSL2 with Ubuntu: `wsl --install -d Ubuntu-22.04`
- Install NVIDIA drivers on **Windows host** (not inside WSL)
- Inside WSL, CUDA passthrough gives you GPU access

```bash
# Inside WSL2
nvidia-smi  # Should show your GPU
pip install -e ".[llama]"
```

### Windows (Native)

- CPU-only with llama.cpp (limited performance)
- WSL2 is strongly recommended instead

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `tqcli: command not found` | Activate your venv: `source .venv/bin/activate` |
| `No models installed` | Run `tqcli model pull <model_id>` |
| `Engine not installed` | Run `pip install llama-cpp-python` or `pip install vllm` |
| Very slow inference | Check `tqcli system info` — you may need a smaller model or lower quant |
| GPU not detected | Verify `nvidia-smi` works. On WSL2, update Windows NVIDIA drivers. |
| Out of memory | Use a smaller quantization: Q3_K_M or Q2_K |

## Multi-Process Mode

If you want to run multiple chat sessions against the same model (or use skills that coordinate multiple workers):

```bash
# Start a shared inference server
tqcli serve start

# Open multiple worker sessions (each in its own terminal)
tqcli chat --engine server     # terminal 1
tqcli chat --engine server     # terminal 2

# Or spawn workers automatically
tqcli workers spawn 2

# Check status
tqcli serve status

# Shut everything down
tqcli serve stop
```

The system auto-detects whether to use llama.cpp server (any platform) or vLLM server (Linux with 8+ GB VRAM). vLLM is preferred for multi-process because it batches concurrent requests on the GPU.

## Unrestricted Mode

If tqCLI's resource guards are blocking you and you know your system can handle it:

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go chat
tqcli --stop-trying-to-control-everything-and-just-let-go workers spawn 5
```

This bypasses memory checks, confirmation prompts, and worker limits. Audit logging stays on.

## Next Steps

- **Benchmark your setup**: `tqcli benchmark`
- **Try multiple models**: download 2-3 models and let the router pick the best one per prompt
- **Try multi-process**: `tqcli serve start` then `tqcli chat --engine server`
- **Read the architecture**: `docs/ARCHITECTURE.md`
- **Contribute**: see [CONTRIBUTING.md](../CONTRIBUTING.md)
