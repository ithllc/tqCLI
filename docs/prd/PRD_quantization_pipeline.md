# PRD: tqCLI Quantization Pipeline

**Document Version:** 1.0
**Date:** 2026-04-14
**Status:** Approved
**GitHub Issue:** [ithllc/tqCLI#10](https://github.com/ithllc/tqCLI/issues/10)
**Author:** ithllc

---

## 1. Introduction

### 1.1 Problem Statement

tqCLI promises local LLM inference using TurboQuant methodologies, but currently **performs zero quantization**. The CLI downloads pre-quantized models from HuggingFace (relying on third-party quantization) or downloads raw BF16 models that cannot fit on consumer GPUs. Gemma 4 E2B loads at 9.89 GiB in BF16 on a 4 GB VRAM GPU — a model that should compress to ~2.5 GB with INT4 quantization.

### 1.2 Elevator Pitch

Add a complete quantization pipeline to tqCLI that downloads BF16 base models and quantizes them locally before inference, using bitsandbytes (on-the-fly INT4), AutoAWQ (persistent AWQ checkpoints), and llama.cpp (GGUF k-quant conversion). This enables Gemma 4 and Qwen 3 models to run on consumer GPUs with 4-8 GB VRAM.

### 1.3 Success Criteria

| Metric | Target |
|--------|--------|
| Gemma 4 E4B runs on 4 GB VRAM via vLLM | Yes, via bitsandbytes INT4 |
| Qwen 3 4B BF16 runs on 4 GB VRAM via vLLM | Yes, via bitsandbytes INT4 |
| `tqcli model quantize` command exists | CLI command with method/bits selection |
| Integration tests pass with BF16 source models | Both llama.cpp and vLLM backends |
| Quantized model size < 40% of BF16 | INT4 achieves ~25% of BF16 |
| Inference quality preserved | Coherent responses to test prompts |

---

## 2. Target Audience

### Primary Users
- **Local inference enthusiasts** with consumer NVIDIA GPUs (4-8 GB VRAM)
- **Developers** testing LLM applications who need fast local inference without cloud APIs
- **Researchers** evaluating quantization quality vs. performance tradeoffs

### Hardware Profile (Reference System)
- GPU: NVIDIA RTX A2000 Laptop (4 GB VRAM, Ampere, CC 8.6)
- RAM: 32 GB
- OS: Linux (Ubuntu 22.04 WSL2)
- CUDA: 12.8 (via PyTorch 2.9+)

---

## 3. Scope & Constraints

### In Scope
1. **bitsandbytes INT4 on-the-fly quantization** for vLLM (load-time quantization, no persistent checkpoint)
2. **Hardware-aware quantization** level selection (bits, method based on VRAM)
3. **`tqcli model quantize` CLI command** for explicit quantization
4. **Integration tests** using BF16 source models for both vLLM and llama.cpp
5. **Comparison report** of BF16 vs. quantized inference results

### Out of Scope
- TurboQuant KV cache quantization (research paper, no public implementation)
- AutoAWQ persistent checkpoint creation (Phase 2 — requires calibration data and GPU time)
- GGUF conversion pipeline via llama.cpp tools (Phase 2 — requires llama.cpp build tools)
- Training or fine-tuning
- Custom quantization algorithm development

### Constraints
- vLLM 0.19.0 with `transformers>=5.5.0` required for Gemma 4
- bitsandbytes requires CUDA GPU (no CPU-only support)
- 4 GB VRAM is the minimum target — must work within this budget
- WSL2 multiprocessing requires `spawn` method (no `fork`)

---

## 4. Key Features

### Feature 1: bitsandbytes On-the-Fly INT4 Quantization (vLLM)

**Description:** When loading a BF16 model via vLLM, automatically apply bitsandbytes 4-bit quantization during model loading. This requires no pre-quantization step — the model is quantized as it loads into GPU memory.

**User Interaction:**
```bash
# Automatic: tuner detects BF16 model won't fit, applies bnb quantization
tqcli chat --engine vllm --model gemma-4-e4b-it-vllm

# Explicit: user requests quantization method
tqcli chat --engine vllm --model gemma-4-e4b-it-vllm --quantization bitsandbytes
```

**Technical Implementation:**
- Pass `quantization="bitsandbytes"` and `load_format="bitsandbytes"` to vLLM `LLM()` constructor
- Set `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)`
- Update `VllmTuningProfile` to select bitsandbytes when BF16 model exceeds VRAM
- Recalculate `max_model_len` based on quantized model size (~25% of BF16)

**Acceptance Criteria:**
- Gemma 4 E4B loads via vLLM on 4 GB VRAM
- Qwen 3 4B BF16 loads via vLLM on 4 GB VRAM
- Coherent responses to test chat prompts
- Model size in VRAM < 40% of BF16 size

### Feature 2: Hardware-Aware Quantization Selection

**Description:** The `vllm_config.py` tuner automatically selects quantization method and level based on model size vs. available VRAM.

**Decision Logic:**
```
IF model_bf16_size <= available_vram:
    → Load at BF16 (no quantization needed)
ELIF model_bf16_size * 0.30 <= available_vram:
    → Use bitsandbytes INT4 (reduces to ~25-30% of BF16)
ELSE:
    → Model too large even after quantization, reject
```

**Acceptance Criteria:**
- Tuner automatically selects bitsandbytes for models that exceed VRAM in BF16
- Tuner rejects models that won't fit even after INT4 quantization
- No user intervention required for common cases

### Feature 3: `tqcli model quantize` CLI Command

**Description:** Explicit CLI command to quantize a downloaded model.

**Interface:**
```bash
tqcli model quantize <model_id> [--method bnb|awq|gguf] [--bits 4|8] [--output <path>]
```

**Phase 1 Scope:** Report quantization status and recommended method. Apply bitsandbytes at load time (not persistent).

**Acceptance Criteria:**
- Command shows model's current quantization status
- Command shows recommended quantization for current hardware
- Integrates with model registry

### Feature 4: Integration Tests with BF16 Source Models

**Description:** New integration tests that download BF16 base models, quantize them, and run inference — validating the entire pipeline.

**Test Matrix:**

| Test | Model | Engine | Quantization | Expected |
|------|-------|--------|-------------|----------|
| vLLM Gemma 4 E4B BF16→INT4 | google/gemma-4-E4B-it | vLLM | bitsandbytes INT4 | PASS on 4 GB |
| vLLM Qwen 3 4B BF16→INT4 | Qwen/Qwen3-4B | vLLM | bitsandbytes INT4 | PASS on 4 GB |
| llama.cpp Gemma 4 E4B pre-quant | unsloth GGUF Q4_K_M | llama.cpp | Q4_K_M (pre-quantized) | PASS (baseline) |
| llama.cpp Qwen 3 4B pre-quant | Qwen GGUF Q4_K_M | llama.cpp | Q4_K_M (pre-quantized) | PASS (baseline) |

**Comparison Report:** Side-by-side metrics (tok/s, model size, response quality) for BF16→INT4 vs. pre-quantized GGUF.

---

## 5. User Stories

### US-1: Auto-Quantize on Load
**As a** user with a 4 GB VRAM GPU,
**I want** tqCLI to automatically quantize Gemma 4 E4B when I load it,
**So that** it fits in my GPU memory without me needing to know about quantization.

### US-2: Explicit Quantization Command
**As a** power user,
**I want** to run `tqcli model quantize gemma-4-e4b-it-vllm --method bnb --bits 4`,
**So that** I can control the quantization method and verify it before inference.

### US-3: Hardware-Aware Model Selection
**As a** user who runs `tqcli chat --engine vllm`,
**I want** the system to automatically select the best model AND quantization level for my hardware,
**So that** I get the best possible quality within my VRAM budget.

### US-4: Quantization Comparison
**As a** researcher,
**I want** to see a comparison report of BF16-quantized vs. pre-quantized GGUF performance,
**So that** I can evaluate quality vs. performance tradeoffs.

### US-5: BF16 Download and Quantize
**As a** user who wants the latest model,
**I want** to download the official BF16 model from Google/Qwen and have tqCLI quantize it locally,
**So that** I'm not dependent on third-party quantized model availability on HuggingFace.

---

## 6. Technical Requirements

### 6.1 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `bitsandbytes` | >=0.43.0 | INT4/INT8 on-the-fly quantization |
| `accelerate` | >=0.30.0 | Required by bitsandbytes for model loading |
| `vllm` | >=0.19.0 | Inference engine with bnb support |
| `transformers` | >=5.5.0 | Gemma 4 architecture support |

### 6.2 Architecture Changes

**New Module:** `tqcli/core/quantizer.py`
- `QuantizationMethod` enum: `BNB_INT4`, `BNB_INT8`, `AWQ`, `GPTQ`, `GGUF_Q4_K_M`
- `select_quantization(model, sys_info) -> QuantizationMethod`
- `get_bnb_config(method) -> BitsAndBytesConfig`

**Modified Modules:**
- `tqcli/core/vllm_config.py` — Integrate bitsandbytes into tuning profile
- `tqcli/core/vllm_backend.py` — Pass quantization and load_format to vLLM
- `tqcli/core/model_registry.py` — Add BF16 base model profiles that are candidates for quantization
- `tqcli/cli.py` — Add `tqcli model quantize` command

### 6.3 Performance Requirements

| Metric | Requirement |
|--------|------------|
| Model load time (BF16→INT4) | < 120 seconds |
| VRAM usage after quantization | < 40% of BF16 size |
| Inference tok/s | > 1 tok/s (minimum usable) |
| Response coherence | Correct answers to simple factual questions |

### 6.4 Security Requirements

- No model weights transmitted externally during quantization
- Quantization runs entirely on local hardware
- Audit log records quantization events

---

## 7. Success Metrics

| Metric | Measurement | Target |
|--------|------------|--------|
| Gemma 4 E4B fits 4 GB VRAM | VRAM usage during inference | < 3.5 GB |
| Qwen 3 4B BF16 fits 4 GB VRAM | VRAM usage during inference | < 3.0 GB |
| Integration tests pass | All test steps PASS | 100% |
| Quantized inference speed | Tokens per second | > 1 tok/s |
| Response quality | Correct factual answers | Matches BF16 baseline |
| User experience | No manual quantization steps | Fully automatic |

---

## 8. Phases

### Phase 1 (This Implementation)
- bitsandbytes INT4 on-the-fly for vLLM
- Hardware-aware tuner integration
- `tqcli model quantize` command (info + bnb)
- Integration tests with BF16 source models
- Comparison report

### Phase 2 (Future)
- AutoAWQ persistent checkpoint creation
- GGUF conversion + llama-quantize pipeline
- Quantization progress UI with Rich
- Calibration dataset support for AWQ/GPTQ

### Phase 3 (When Available)
- TurboQuant KV cache quantization integration
- Custom quantization profiles per model family

---

## Next Steps

Run `/technical-planner` to generate the implementation plan from this PRD.
