# PRD: TurboQuant KV Cache Compression for tqCLI

**Document Version:** 1.0
**Date:** 2026-04-14
**Status:** Approved
**GitHub Issue:** [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13)
**Author:** ithllc

---

## 1. Introduction

### 1.1 Problem Statement

tqCLI on a 4 GB VRAM GPU (RTX A2000) can load models via weight quantization (GGUF Q4_K_M, AWQ INT4) but achieves only ~368 tokens of context — too short for meaningful conversation. The bottleneck is the KV cache, which grows linearly with context length at 8.5 bits per value (q8_0 default).

TurboQuant KV cache compression (ICLR 2026, arXiv 2504.19874) achieves 4-8x compression of the KV cache at runtime with minimal quality loss. Working community implementations exist for both llama.cpp and vLLM.

### 1.2 Elevator Pitch

Integrate TurboQuant KV cache compression into tqCLI's llama.cpp and vLLM backends, increasing usable context from ~368 tokens to ~1,700-2,950 tokens on 4 GB VRAM hardware. Uses existing model files — no new downloads or conversions required.

### 1.3 Success Criteria

| Metric | Target |
|--------|--------|
| Context tokens (turbo3, Qwen 3 4B, 4 GB VRAM) | >= 1,500 tokens (vs 368 baseline) |
| Perplexity degradation (turbo3 vs q8_0) | < 2% |
| Inference speed (turbo3 vs q8_0) | >= 90% of baseline |
| Existing tests still pass | All 14 basic + all integration tests |
| New `--kv-quant` flag works | CLI flag accepted and applied |

---

## 2. Target Audience

Same as tqCLI: local inference users with consumer NVIDIA GPUs (4-16 GB VRAM). TurboQuant is most impactful on VRAM-constrained systems where KV cache is the bottleneck.

---

## 3. Scope

### In Scope
1. **llama.cpp backend**: Build llama-cpp-python against TheTom/turboquant_plus fork; expose `--cache-type-k` and `--cache-type-v` via new `--kv-quant` CLI flag
2. **vLLM backend**: Install mitkox/vllm-turboquant; expose `--kv-cache-dtype turboquant35` and `--enable-turboquant`
3. **CLI flag**: `tqcli chat --kv-quant turbo3` (simplified interface wrapping both backends)
4. **Hardware-aware defaults**: Auto-select KV quant level based on VRAM budget
5. **Integration tests**: 4 new tests comparing q8_0 baseline vs turbo3/turbo4 KV cache
6. **Comparison report**: Side-by-side context capacity, speed, and quality metrics

### Out of Scope
- Weight quantization (already implemented in v0.4.0)
- TurboQuant weight compression (separate feature)
- Custom CUDA kernel development (using community implementations)
- Upstream contributions to llama.cpp or vLLM

---

## 4. Key Features

### Feature 1: llama.cpp TurboQuant KV Cache

**Implementation:** Build llama-cpp-python against TheTom's fork which adds `turbo4`, `turbo3`, `turbo2` GGML types for KV cache.

**Usage:**
```bash
tqcli chat --model qwen3-4b-Q4_K_M --kv-quant turbo3
# Internally passes: --cache-type-k turbo3 --cache-type-v turbo3
```

**KV Quant Levels (llama.cpp):**
| Level | Bits/Value | Compression | Quality |
|-------|-----------|-------------|---------|
| turbo4 | 4.25 | 3.8x | Near-lossless (+0.23% PPL) |
| turbo3 | 3.5 | 4.6x | Minimal loss (+1.06% PPL) |
| turbo2 | 2.5 | 6.4x | Noticeable loss (+6.48% PPL) |

### Feature 2: vLLM TurboQuant KV Cache

**Implementation:** Install mitkox/vllm-turboquant fork which adds turboquant35 and turboquant25 KV cache dtypes.

**Usage:**
```bash
tqcli chat --engine vllm --model qwen3-4b-AWQ --kv-quant turbo35
# Internally passes: --kv-cache-dtype turboquant35 --enable-turboquant --attention-backend TRITON_ATTN
```

### Feature 3: Hardware-Aware KV Quant Selection

The tuner auto-selects KV quant level based on available KV cache memory:
- KV budget > 200 MB → q8_0 (no compression needed)
- KV budget 50-200 MB → turbo4 (3.8x, near-lossless)
- KV budget 20-50 MB → turbo3 (4.6x, minimal loss)
- KV budget < 20 MB → turbo2 (6.4x, quality trade-off warning)

---

## 5. User Stories

### US-1: Longer Conversations on Consumer GPU
**As a** user with 4 GB VRAM, **I want** TurboQuant to compress my KV cache, **so that** I can have 1,700+ token conversations instead of being cut off at 368 tokens.

### US-2: Quality-Compression Control
**As a** power user, **I want** to choose between turbo4/turbo3/turbo2, **so that** I can trade quality for context length based on my task.

### US-3: Automatic Best Setting
**As a** casual user, **I want** `tqcli chat` to auto-select the best KV quant for my hardware, **so that** I don't need to understand compression levels.

---

## 6. Technical Requirements

### Dependencies

**llama.cpp backend:**
- TheTom/turboquant_plus fork (Apache 2.0)
- Build with: `cmake -B build -DGGML_CUDA=ON` (SM86+ for our GPU)
- llama-cpp-python built against this fork

**vLLM backend:**
- mitkox/vllm-turboquant fork
- Build from source with CUDA 12.8
- Requires: `--attention-backend TRITON_ATTN`

### Architecture Changes

**Modified modules:**
- `tqcli/core/llama_backend.py` — Add cache_type_k, cache_type_v params
- `tqcli/core/vllm_backend.py` — Add turboquant KV cache dtype
- `tqcli/core/vllm_config.py` — KV quant selection in tuner
- `tqcli/cli.py` — Add `--kv-quant` flag to chat/serve commands
- `tqcli/core/system_info.py` — Detect TurboQuant availability

**New module:**
- `tqcli/core/kv_quantizer.py` — KV cache quant level selection and mapping

---

## 7. Success Metrics

| Metric | Measurement | Target |
|--------|------------|--------|
| Context capacity increase | Tokens at turbo3 vs q8_0 | >= 4x |
| Quality preservation | PPL turbo3 vs q8_0 | < 2% degradation |
| Speed impact | tok/s turbo3 vs q8_0 | >= 90% |
| Test pass rate | Integration tests | 100% |

---

## Next Steps

1. Generate technical plan via `/technical-planner`
2. Write test cases
3. Run integration tests (deferred to tonight)
