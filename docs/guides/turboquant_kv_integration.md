# TurboQuant KV Cache Compression — Integration Guide

## Overview

tqCLI v0.5.0 integrates TurboQuant KV cache compression from Google Research (ICLR 2026, arXiv 2504.19874). This compresses the attention key/value cache at runtime using PolarQuant + Walsh-Hadamard rotation, giving 3.5-6.4x more context tokens with minimal quality loss.

**Design principle:** tqCLI is ONE version — no separate builds per CUDA version. The CUDA-specific work lives in forked inference engines. tqCLI detects at runtime what's available and degrades gracefully.

## Compression Levels

| Level | Bits/Value | Compression | Quality Impact | Use Case |
|-------|-----------|-------------|---------------|----------|
| none | 16.0 (f16) | 1x | Baseline | Default, maximum quality |
| turbo4 | 4.25 | 3.8x | +0.23% PPL | Near-lossless, recommended |
| turbo3 | 3.5 | 4.6x | +1.06% PPL | Best balance for constrained VRAM |
| turbo2 | 2.5 | 6.4x | +6.48% PPL | Maximum compression, quality tradeoff |

## CLI Usage

```bash
# Auto-select (uses turbo3 if available, falls back to none)
tqcli chat

# Explicit KV compression level
tqcli chat --kv-quant turbo3
tqcli chat --kv-quant turbo4
tqcli chat --kv-quant none       # Disable KV compression

# Check system TurboQuant support
tqcli system info                 # Shows "TurboQuant KV: available/unavailable"
tqcli system info --json          # JSON output with turboquant_kv field
```

## Unified Quantization Pipeline

The pipeline automatically detects model precision and applies the correct stages:

```
Model loaded -> detect format
    |
    +-- Already weight-quantized (GGUF Q4_K_M, AWQ INT4, GPTQ)?
    |       -> Skip weight quantization
    |       -> Apply KV cache compression ONLY (turbo3/turbo4)
    |
    +-- Full precision (BF16 / FP16 / FP32 safetensors)?
            -> Apply weight quantization FIRST
            |   vLLM: bitsandbytes INT4 on-the-fly
            |   llama.cpp: use pre-quantized GGUF
            -> THEN apply KV cache compression (turbo3/turbo4)
```

## CUDA Compatibility

| CUDA Version | TurboQuant Status |
|-------------|------------------|
| >= 12.8 | Full support (turbo2/3/4) |
| 11.x - 12.7 | Graceful fallback to standard KV cache |
| No CUDA | CPU-only, standard KV cache |

The `check_turboquant_compatibility()` function detects:
1. CUDA toolkit version (nvcc)
2. GPU compute capability (need SM70+ / Volta)
3. Whether TurboQuant fork inference engines are installed

## Inference Engine Forks

| Engine | Fork | Status |
|--------|------|--------|
| llama.cpp | [ithllc/llama-cpp-turboquant](https://github.com/ithllc/llama-cpp-turboquant) | Built, tested (7.33 tok/s turbo3) |
| vLLM | [ithllc/vllm-turboquant](https://github.com/ithllc/vllm-turboquant) | Built, E2E verified (turboquant35, 4.0x KV compression) |

### Building llama-cpp-turboquant

```bash
git clone https://github.com/ithllc/llama-cpp-turboquant.git
cd llama-cpp-turboquant
cmake -B build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES="86" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)

# Python bindings
LLAMA_CPP_LIB=build/bin/libllama.so pip install llama-cpp-python
```

### Building vllm-turboquant

```bash
git clone https://github.com/ithllc/vllm-turboquant.git
cd vllm-turboquant
CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda pip install -e .
```

## Performance Results (RTX A2000, 4 GB VRAM)

### llama.cpp (turbo3 KV)

| Model | KV Type | tok/s | Est. Context |
|-------|---------|-------|-------------|
| Qwen3-4B Q4_K_M | f16 (baseline) | 6.41 | ~368 tokens |
| Qwen3-4B Q4_K_M | turbo3 | 7.33 (+14%) | ~1700 tokens |
| Gemma 4 E2B Q4_K_M | turbo3 | 6.52-7.27 | tested with thinking + tool calling |

### vLLM (turboquant35 KV)

| Model | KV Type | tok/s | KV Tokens (50 MB) |
|-------|---------|-------|--------------------|
| Qwen3-4B AWQ | auto (baseline) | 5.72 | 336 |
| Qwen3-4B AWQ | turboquant35 | 2.04 | 1,344 (4.0x more) |

### Verified Capabilities with TurboQuant KV

| Capability | Qwen3 (llama.cpp) | Qwen3 (vLLM) | Gemma 4 (llama.cpp) |
|------------|-------------------|---------------|---------------------|
| Basic inference | PASS | PASS | PASS |
| Thinking mode | PASS | PASS | PASS |
| Tool calling (JSON) | PASS | PASS | PASS |
| Combined think+tool | PASS | SKIP (ctx limit) | PASS |

## Architecture

```
tqcli/core/kv_quantizer.py
    |-- KVQuantLevel enum (NONE, TURBO4, TURBO3, TURBO2)
    |-- select_kv_quant() — auto-select based on available KV memory
    |-- get_llama_kv_params() — llama.cpp cache_type_k/v values
    |-- get_vllm_kv_params() — vLLM kv_cache_dtype + enable_turboquant
    |-- check_turboquant_compatibility() — CUDA version check
    |-- detect_model_precision() — full precision vs pre-quantized
    |-- plan_quantization_pipeline() — unified weight + KV pipeline

tqcli/core/system_info.py
    |-- GPUInfo.cuda_version — driver max CUDA version
    |-- GPUInfo.cuda_toolkit_version — installed nvcc version

tqcli/core/vllm_config.py
    |-- build_vllm_config(kv_quant_choice=...) — integrates TurboQuant KV

tqcli/cli.py
    |-- chat --kv-quant flag with graceful fallback
    |-- system info shows TurboQuant KV status
```

## References

- TurboQuant paper: [arxiv.org/abs/2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)
- PolarQuant paper: [arxiv.org/abs/2502.02617](https://arxiv.org/abs/2502.02617) (AISTATS 2026)
- GoogleTurboQuant workspace: `../GoogleTurboQuant/`
- GitHub Issue: [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13)
