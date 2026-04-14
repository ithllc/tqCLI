# tqCLI TurboQuant KV Cache Compression Test Cases

**Status:** NOT EXECUTED — test cases documented, awaiting implementation
**Feature:** TurboQuant KV cache compression (ICLR 2026, arXiv 2504.19874)
**GitHub Issue:** [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13)
**Prerequisites:**
- llama-cpp-python built against TheTom/turboquant_plus fork
- mitkox/vllm-turboquant installed from source (CUDA 12.8)

> These tests validate that TurboQuant KV cache compression works on 4 GB VRAM
> hardware (RTX A2000 SM86) with both inference engines, using standard model
> files (same GGUF and safetensors as previous tests — no new model downloads).

---

## Hardware Requirements

| Property | Value |
|----------|-------|
| GPU | NVIDIA with SM86+ (Ampere or newer) |
| VRAM | 4 GB minimum |
| CUDA | 12.8+ |
| OS | Linux / WSL2 |

---

## Test 1: llama.cpp Gemma 4 E4B Q4_K_M + TurboQuant turbo3 KV

**Model:** `gemma-4-e4b-it-Q4_K_M` (same GGUF as previous tests)
**Engine:** llama.cpp (TurboQuant fork)
**Weight Quantization:** Q4_K_M (pre-quantized GGUF, ~4,981 MB)
**KV Cache Quantization:** turbo3 (3.5 bpv, 4.6x compression)

### Steps

| # | Step | Expected |
|---|------|----------|
| 1 | Download model | Same GGUF file from unsloth repo |
| 2 | Load model with `cache_type_k=turbo3, cache_type_v=turbo3` | Model loads, turbo3 KV active |
| 3 | Verify KV compression active | Log shows turbo3 cache type |
| 4 | Chat turn 1: "What is the capital of France?" | Correct answer (Paris) |
| 5 | Chat turn 2: "What is its population?" | Reasonable answer (~2M) |
| 6 | Capture metrics | tok/s, context tokens achievable, VRAM |
| 7 | Compare context capacity vs q8_0 baseline | >= 3x more tokens |
| 8 | Unload model | Clean unload |

### Metrics to Capture
- Context tokens achievable with turbo3 vs q8_0 baseline
- Tokens per second (prompt eval + generation)
- Response quality (factual correctness)
- Load time

---

## Test 2: llama.cpp Qwen 3 4B Q4_K_M + TurboQuant turbo3 KV

**Model:** `qwen3-4b-Q4_K_M` (same GGUF as previous tests)
**Engine:** llama.cpp (TurboQuant fork)
**Weight Quantization:** Q4_K_M (pre-quantized GGUF, ~2,382 MB)
**KV Cache Quantization:** turbo3 (3.5 bpv, 4.6x compression)

### Steps

| # | Step | Expected |
|---|------|----------|
| 1 | Download model | Same GGUF from Qwen repo |
| 2 | Load with turbo3 KV | Model loads successfully |
| 3 | Chat turn 1: "What is 2 + 2?" | Correct answer (4), may include thinking |
| 4 | Chat turn 2: "Multiply by 10" | Correct answer (40) |
| 5 | Capture metrics | tok/s, context capacity, VRAM |
| 6 | Compare vs q8_0 baseline | >= 4x more context tokens |
| 7 | Test turbo4 KV | Verify turbo4 also works (3.8x compression) |
| 8 | Test turbo2 KV | Verify turbo2 works (6.4x, quality warning) |
| 9 | Unload model | Clean unload |

### Metrics to Capture
- Context tokens: q8_0 vs turbo4 vs turbo3 vs turbo2
- tok/s at each level
- Response correctness at each level

---

## Test 3: vLLM Qwen 3 4B AWQ + TurboQuant turboquant35 KV

**Model:** `qwen3-4b-AWQ` (same AWQ checkpoint as previous vLLM tests)
**Engine:** vLLM (mitkox/vllm-turboquant fork)
**Weight Quantization:** AWQ INT4 (pre-quantized, ~2,558 MB)
**KV Cache Quantization:** turboquant35 (3.5 bpv equivalent)

### Prerequisites
```bash
# Install mitkox fork from source
git clone https://github.com/mitkox/vllm-turboquant.git
cd vllm-turboquant
CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda VLLM_USE_PRECOMPILED=0 pip install -e .
```

### Steps

| # | Step | Expected |
|---|------|----------|
| 1 | Verify vllm-turboquant installed | `import vllm; print(vllm.__version__)` |
| 2 | Download AWQ model | Same model from previous tests |
| 3 | Load with turboquant35 KV + TRITON_ATTN | Model loads, turbo KV active |
| 4 | Chat turn 1: "What is 2 + 2?" | Correct answer |
| 5 | Chat turn 2: "Multiply by 10" | Correct answer |
| 6 | Capture metrics | tok/s, context capacity |
| 7 | Compare vs standard KV (auto dtype) | More context tokens |
| 8 | Unload model | Clean unload |

### Configuration
```python
engine = VllmBackend(
    max_model_len=2048,  # Should be achievable with turboquant
    gpu_memory_utilization=0.80,
    quantization="awq_marlin",
    kv_cache_dtype="turboquant35",
    enforce_eager=True,
)
```

---

## Test 4: Baseline Comparison (No TurboQuant KV)

**Purpose:** Establish q8_0/auto KV cache baseline for comparison with Tests 1-3.

### Steps

| # | Step | Expected |
|---|------|----------|
| 1 | llama.cpp Gemma 4 E4B Q4_K_M + q8_0 KV | Record context tokens, tok/s |
| 2 | llama.cpp Qwen 3 4B Q4_K_M + q8_0 KV | Record context tokens, tok/s |
| 3 | vLLM Qwen 3 4B AWQ + auto KV | Record context tokens, tok/s |
| 4 | Generate comparison table | turbo3 vs q8_0 for all models |

---

## Expected Comparison Results

| Model | Engine | KV Type | Est. Context | Est. tok/s | Quality |
|-------|--------|---------|-------------|-----------|---------|
| Gemma 4 E4B | llama.cpp | q8_0 | ~100 | 2-4 | Baseline |
| Gemma 4 E4B | llama.cpp | turbo3 | ~460 | 2-4 | +1% PPL |
| Qwen 3 4B | llama.cpp | q8_0 | ~368 | 6-9 | Baseline |
| Qwen 3 4B | llama.cpp | turbo3 | ~1,700 | 6-9 | +1% PPL |
| Qwen 3 4B | llama.cpp | turbo2 | ~2,350 | 6-9 | +6% PPL |
| Qwen 3 4B | vLLM AWQ | auto | ~226 | 6-8 | Baseline |
| Qwen 3 4B | vLLM AWQ | turbo35 | ~1,040 | 6-8 | +1% PPL |

---

## Pre-Test Checklist

- [ ] llama-cpp-python built against TheTom/turboquant_plus fork
- [ ] `turbo3` cache type accepted by llama_cpp.Llama()
- [ ] mitkox/vllm-turboquant installed from source
- [ ] `turboquant35` KV dtype accepted by vLLM LLM()
- [ ] Existing GGUF models available (~/.tqcli/models/)
- [ ] Existing AWQ model available (~/.tqcli/models/)
- [ ] CUDA 12.8 available
- [ ] Flash attention enabled (`-fa` for llama.cpp)

## Output Files
- `tests/integration_reports/turboquant_kv_comparison_report.md`
- `tests/integration_reports/turboquant_kv_comparison_report.json`
