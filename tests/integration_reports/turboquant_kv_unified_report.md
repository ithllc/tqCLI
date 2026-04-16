# tqCLI Unified Integration Test Report — TurboQuant KV

**Generated:** 2026-04-15 20:12:18
**tqCLI Version:** 0.5.0
**Scope:** Tests 5-7 (Thinking + Tool Calling + Combined) with TurboQuant KV
**Engines:** llama.cpp (0.3.20), vLLM (0.1.dev5+g896f08bf6)

## System Information

| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| arch | x86_64 |
| cpu_cores | 16 |
| cpu_physical | 8 |
| ram_total_mb | 31956 |
| ram_available_mb | 12147 |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| recommended_engine | llama.cpp |
| recommended_quant | Q3_K_M |
| max_model_gb | 3.4 |
| is_wsl | True |

## Quantization Pipeline Validation

The unified quantization pipeline detects model precision and applies the appropriate stages:

| Model Type | Weight Quantization | KV Cache Compression | Pipeline Path |
|------------|--------------------|--------------------|---------------|
| GGUF Q4_K_M (llama.cpp) | SKIP (pre-quantized) | turbo3 (4.6x) | KV-only |
| AWQ INT4 (vLLM) | SKIP (pre-quantized) | turboquant35 | KV-only |
| BF16 safetensors (vLLM) | BNB_INT4 (on-the-fly) | turboquant35 | Full pipeline |

## Overall Summary

| Metric | Value |
|--------|-------|
| Total Tests | 1 |
| Total Steps | 13 |
| Passed | 12 |
| Failed | 1 |
| Pass Rate | 92.3% |

---

## vLLM (TurboQuant fork)
**Duration:** 372.2s

### Test 5: Thinking Mode + turboquant35 KV (vLLM)

**Model:** `qwen3-4b-AWQ` | **Engine:** vLLM (TurboQuant fork) | **Result:** **FAIL** (12/13 steps) | **Duration:** 370.1s

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | turboquant_compatibility | PASS | 0.00s | TurboQuant KV cache available (llama.cpp, vLLM) |
| 2 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/qwen3-4b-AWQ (2558 MB) |
| 3 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: weight_quantized \| Weight quant needed: False \| KV compression: tur |
| 4 | verify_kv_only_pipeline | PASS | 0.00s | Weight quant needed: False (expected: False for pre-quantized) \| KV compression: True (expected: Tru |
| 5 | load_model_turbo_kv | PASS | 70.36s | Loaded Qwen3 4B (AWQ INT4, vLLM) via vLLM with kv_cache_dtype=turboquant35 in 70.4s |
| 6 | qwen3_thinking_turn | PASS | 61.05s | Thinking: NO (0 chars) \| Response: To calculate 15% of 240, I can follow the:  15. First, convert th |
| 7 | qwen3_no_think_turn | PASS | 34.80s | Thinking: 0 chars (expected minimal) \| Response: </think>  </think>  To find 10% of 5 5 500, I can m |
| 8 | qwen3_reasoning_turn | PASS | 67.57s | Thinking: NO (0 chars) \| Response: The ball costs 5 cents.   **Explanation  The problem is a classic |
| 9 | download_model | PASS | 0.00s | Already downloaded at /root/.tqcli/models/gemma-4-e2b-it-vllm (9803 MB) |
| 10 | quantization_pipeline | PASS | 0.00s | Pipeline: kv:turbo3 \| Precision: full_precision \| Weight quant needed: False \| KV compression: turbo |
| 11 | verify_full_pipeline | PASS | 0.00s | Weight quant needed: False \| Weight method:  \| KV compression: True \| KV level: turbo3 \| Precision:  |
| 12 | load_model_turbo_kv | FAIL | 136.31s | Failed to load with TurboQuant KV: Engine core initialization failed. See root cause above. Failed c |
| 13 | gemma4_hw_limitation | PASS | 0.00s | Gemma 4 E2B BF16 + BNB_INT4 may not fit 4096 MB VRAM. Expected on < 6 GB hardware. |

#### Pipeline Decision
- **Precision:** weight_quantized | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True
- **Precision:** full_precision | **Weight quant:** False () | **KV level:** turbo3 | **KV-only:** True

#### Performance
| Step | tok/s | Tokens | Thinking |
|------|-------|--------|----------|
| qwen3_thinking_turn | 1.84 | 112 | NO |
| qwen3_no_think_turn | 1.52 | 53 | NO |
| qwen3_reasoning_turn | 1.64 | 111 | NO |

---
