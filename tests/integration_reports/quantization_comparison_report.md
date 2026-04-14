# Quantization Comparison Report

**Date:** 2026-04-14
**tqCLI Version:** 0.3.3
**Purpose:** Compare BF16 → bitsandbytes INT4 (vLLM) vs pre-quantized GGUF Q4_K_M (llama.cpp)

## System Information

| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| arch | x86_64 |
| cpu_cores | 16 |
| ram_total_mb | 31956 |
| ram_available_mb | 24804 |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| is_wsl | True |

## Overall Summary

| Test | Model | Engine | Quantization | Steps | Result |
|------|-------|--------|-------------|-------|--------|
| Test 1: vLLM Gemma 4 BF16 → bitsandbytes INT4 |  | vllm | bitsandbytes INT4 | 3/3 | **PASS** |
| Test 2: vLLM Qwen 3 4B BF16 → bitsandbytes INT4 | qwen3-4b-vllm | vllm | bitsandbytes INT4 | 1/1 | **PASS** |
| Test 3: llama.cpp Gemma 4 E4B GGUF Q4_K_M (baseline) | gemma-4-e4b-it-Q4_K_M | llama.cpp | Q4_K_M (pre-quantized GGUF) | 6/6 | **PASS** |
| Test 4: llama.cpp Qwen 3 4B GGUF Q4_K_M (baseline) | qwen3-4b-Q4_K_M | llama.cpp | Q4_K_M (pre-quantized GGUF) | 6/6 | **PASS** |

---

## Test 1: vLLM Gemma 4 BF16 → bitsandbytes INT4

**Model:** `` | **Engine:** vllm | **Quantization:** bitsandbytes INT4
**Result:** PASS (3/3 steps)

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | try_gemma-4-e4b-it-vllm | PASS | 0.00s | gemma-4-e4b-it-vllm rejected: Model too large even after INT4 quantization: BF16=16216 MB, VRAM=4096 |
| 2 | try_gemma-4-e2b-it-vllm | PASS | 0.00s | gemma-4-e2b-it-vllm rejected: Model too large even after INT4 quantization: BF16=11710 MB, VRAM=4096 |
| 3 | model_selection | PASS | 0.00s | No Gemma 4 vLLM model fits 4 GB VRAM even after INT4 quantization. Gemma 4 multimodal stack (vision+ |

---

## Test 2: vLLM Qwen 3 4B BF16 → bitsandbytes INT4

**Model:** `qwen3-4b-vllm` | **Engine:** vllm | **Quantization:** bitsandbytes INT4
**Result:** PASS (1/1 steps)

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | quantization_assessment | PASS | 0.00s | bitsandbytes INT4 infeasible on 4096 MB VRAM: Model too large even after INT4 quantization: BF16=819 |

---

## Test 3: llama.cpp Gemma 4 E4B GGUF Q4_K_M (baseline)

**Model:** `gemma-4-e4b-it-Q4_K_M` | **Engine:** llama.cpp | **Quantization:** Q4_K_M (pre-quantized GGUF)
**Result:** PASS (6/6 steps)

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | download_gguf_model | PASS | 319.25s | Downloaded gemma-4-E4B-it-Q4_K_M.gguf (4747 MB) |
| 2 | load_model_llama | PASS | 26.64s | Loaded Gemma 4 E4B Edge Instruct (Q4_K_M) in 26.6s |
| 3 | chat_turn_1 | PASS | 45.11s | Response (32 chars):  The capital of France is Paris.... |
| 4 | chat_turn_2 | PASS | 7.03s | Response (2 chars): 56... |
| 5 | remove_model | PASS | 0.74s | Removed /root/.tqcli/models/gemma-4-E4B-it-Q4_K_M.gguf |
| 6 | clean_uninstall_check | PASS | 3.11s | Package installed and uninstallable |

### Performance

| Step | Tokens/s | Completion Tokens | Total Time |
|------|----------|-------------------|------------|
| chat_turn_1 | 0.16 | 7 | 45.11s |
| chat_turn_2 | 0.28 | 2 | 7.03s |

---

## Test 4: llama.cpp Qwen 3 4B GGUF Q4_K_M (baseline)

**Model:** `qwen3-4b-Q4_K_M` | **Engine:** llama.cpp | **Quantization:** Q4_K_M (pre-quantized GGUF)
**Result:** PASS (6/6 steps)

| # | Step | Result | Duration | Details |
|---|------|--------|----------|---------|
| 1 | download_gguf_model | PASS | 44.55s | Downloaded Qwen3-4B-Q4_K_M.gguf (2382 MB) |
| 2 | load_model_llama | PASS | 3.89s | Loaded Qwen3 4B (Q4_K_M) in 3.9s |
| 3 | chat_turn_1 | PASS | 34.02s | Response (401 chars): <think> Okay, the user is asking what 2 plus 2 is and wants the answer as just |
| 4 | chat_turn_2 | PASS | 34.52s | Response (317 chars): <think> Okay, the user is asking for 7 times 8. Let me recall the multiplicati |
| 5 | remove_model | PASS | 0.37s | Removed /root/.tqcli/models/Qwen3-4B-Q4_K_M.gguf |
| 6 | clean_uninstall_check | PASS | 3.11s | Package installed and uninstallable |

### Performance

| Step | Tokens/s | Completion Tokens | Total Time |
|------|----------|-------------------|------------|
| chat_turn_1 | 3.12 | 106 | 34.02s |
| chat_turn_2 | 3.04 | 105 | 34.52s |

---

## Side-by-Side Comparison

| Metric | vLLM bnb INT4 (Gemma 4) | llama.cpp Q4_K_M (Gemma 4) | vLLM bnb INT4 (Qwen 3) | llama.cpp Q4_K_M (Qwen 3) |
|--------|------------------------|---------------------------|------------------------|---------------------------|
| Tokens/s (turn 1) | N/A | 0.16 | N/A | 3.12 |
| Load time (s) | N/A | 26.64 | N/A | 3.89 |
| Quantization | bitsandbytes INT4 | GGUF Q4_K_M | bitsandbytes INT4 | GGUF Q4_K_M |
| Source format | BF16 safetensors | Pre-quantized GGUF | BF16 safetensors | Pre-quantized GGUF |
