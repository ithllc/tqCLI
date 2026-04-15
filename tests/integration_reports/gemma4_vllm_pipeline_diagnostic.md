# Gemma 4 E2B vLLM Full Quantization Pipeline Diagnostic

**Generated:** 2026-04-15T14:06:25
**Purpose:** Traces every stage of the unified quantization pipeline for Gemma 4 E2B BF16 on vLLM to document exactly what happens and why it is infeasible on 4 GB VRAM.

## System
| Property | Value |
|----------|-------|
| os | Linux (Ubuntu 22.04.4 LTS) (WSL2) |
| gpu | NVIDIA RTX A2000 Laptop GPU |
| vram_mb | 4096 |
| cuda_version | 13.0 |
| cuda_toolkit | 12.8 |
| compute_capability | 8.6 |
| is_wsl | True |

## Model Profile
| Property | Value |
|----------|-------|
| id | gemma-4-e2b-it-vllm |
| display_name | Gemma 4 E2B Edge Instruct (vLLM BF16) |
| family | gemma4 |
| parameter_count | 2.3B |
| quantization | BF16 |
| format | safetensors |
| engine | vllm |
| multimodal | True |
| supports_thinking | True |
| hf_repo | google/gemma-4-e2b-it |
| min_vram_mb | 3500 |

## Stage 1: Detect Model Precision
**Result:** `full_precision`

Model quantization='BF16', format='safetensors' → detected as 'full_precision'. BF16 safetensors are full-precision, meaning weight quantization (Stage 2) WILL be attempted.

## Stage 2: Estimate Model Sizes

| Level | Text Weights | Multimodal Overhead | Total |
|-------|-------------|--------------------:|------:|
| bf16 | 4710 MB | 7000 MB | **11710 MB** |
| bnb_int4 | 1695 MB | 2450 MB | **4145 MB** |
| bnb_int8 | 2590 MB | 7000 MB | **9590 MB** |

> **Key Insight:** The multimodal encoders (SigLIP vision + audio + VQ-VAE) add 7000 MB in BF16, reduced to 2450 MB with INT4. Text-only INT4 would be just 1695 MB — small enough for 4 GB VRAM. The multimodal overhead is what makes it infeasible.

## Stage 3: Select Weight Quantization

**VRAM Budget:** 4096 MB total - 810 MB CUDA - 700 MB runtime - 50 MB KV = **2536 MB available**

| Method | Size | Fits? | Verdict |
|--------|-----:|:-----:|---------|
| BF16 (no quantization) | 11710 MB | NO | 11710 MB > 2536 MB → DOES NOT FIT |
| BNB_INT4 | 4145 MB | NO | 4145 MB > 2536 MB → DOES NOT FIT |
| BNB_INT8 | 9590 MB | NO | 9590 MB > 2536 MB → DOES NOT FIT |

**Selected:** `None (model too large)`

Even with INT4 quantization (4145 MB), the model exceeds the 2536 MB budget. The 2450 MB multimodal encoder overhead (35% of 7000 MB BF16) is the primary blocker.

## Stage 4: Unified Pipeline Decision

| Property | Value |
|----------|-------|
| Model Precision | full_precision |
| Needs Weight Quant | False |
| Weight Method | N/A (too large) |
| Needs KV Compression | True |
| KV Level | turbo3 |
| Stages Applied | ['kv:turbo3'] |
| Summary | kv:turbo3 |

Pipeline detected 'full_precision' → attempted weight quantization but no method fits (Model is BF16 — too large for 4096 MB VRAM even with INT4 quantization (may need smaller model or more VRAM)). KV compression was still planned (Applying turbo3 KV compression (4.6x, Minimal loss (+1.06% PPL))) but cannot be applied if the model can't load.

## Stage 5: vLLM Config Builder

**Feasible:** `False`
**Reason:** Model too large even after INT4 quantization: BF16=11710 MB, VRAM=4096 MB

## TurboQuant KV Status
**Available:** True

> TurboQuant KV compression IS available on this system, but cannot be used because the model weights themselves don't fit in VRAM. KV compression only reduces the KV cache memory, not model weight memory.

## Minimum Hardware for Gemma 4 E2B on vLLM

| Scenario | Model Size | Min VRAM (WSL2) | Min VRAM (Linux) | Fits 4 GB? | Min GPU |
|----------|-----------|----------------|-----------------|:----------:|---------|
| INT4 text-only (no multimodal encoders) | 1695 MB | 3255 MB | 2845 MB | YES | RTX 3050 4GB (Linux, not WSL2) |
| INT4 with multimodal (current estimate) | 4145 MB | 5705 MB | 5295 MB | NO | RTX 3060 6GB or RTX 4060 8GB |
| BF16 full precision with multimodal | 11710 MB | 13270 MB | 12860 MB | NO | RTX 4090 24GB or A100 40GB |

**Your System:** RTX A2000 (4 GB) cannot run Gemma 4 E2B on vLLM because even with INT4 quantization, the multimodal encoders push the total to 4145 MB, exceeding the 2536 MB budget. Text-only INT4 (1695 MB) would fit on native Linux (2845 MB needed) but not WSL2 (3255 MB needed > 4096 MB).

**Recommendation:** For Gemma 4 E2B on your hardware, use llama.cpp with the GGUF Q4_K_M model (2,890 MB on disk, loads with ~2.8 GB VRAM + turbo3 KV). For vLLM Gemma 4 E2B, minimum 6 GB VRAM (RTX 3060) is required.

## What Works on 4 GB VRAM

### llama.cpp (TurboQuant fork)
| Model | Status | Size | KV | tok/s |
|-------|--------|-----:|:--:|:-----:|
| gemma_4_e2b_Q4_K_M | WORKS | 2890 MB | turbo3 (4.6x) | 2-4 |
| qwen3_4b_Q4_K_M | WORKS | 2382 MB | turbo3 (4.6x) | 6-9 |

### vLLM (TurboQuant fork)
| Model | Status | Details |
|-------|--------|---------|
| qwen3_4b_AWQ | WORKS | 2558 MB, turboquant35, 0.5-1.0 tok/s |
| gemma_4_e2b_BF16_INT4 | DOES NOT FIT | Multimodal encoders add 2450 MB even with INT4 |
