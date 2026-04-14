# Technical Implementation Plan: tqCLI Quantization Pipeline

**Date:** 2026-04-14
**PRD:** [docs/prd/PRD_quantization_pipeline.md](../prd/PRD_quantization_pipeline.md)
**GitHub Issue:** [ithllc/tqCLI#10](https://github.com/ithllc/tqCLI/issues/10)
**Status:** Ready for implementation

---

## Overview

Add bitsandbytes INT4 on-the-fly quantization to tqCLI's vLLM backend, enabling BF16 models (Gemma 4 E4B, Qwen 3 4B) to run on 4 GB VRAM GPUs. The implementation modifies 6 existing files, creates 2 new files, and adds integration tests.

## Architecture

```
User runs: tqcli chat --engine vllm --model gemma-4-e4b-it-vllm

                 CLI (cli.py)
                     |
            vllm_config.py (tuner)
            /                  \
    Model fits BF16?        Model too large?
        |                       |
    Load directly          quantizer.py selects method
        |                       |
    VllmBackend            bitsandbytes INT4
        |                       |
    vLLM LLM()             vLLM LLM(quantization="bitsandbytes",
        |                       load_format="bitsandbytes")
    Inference                   |
                            Inference (~25% VRAM of BF16)
```

## Dependency Graph

```
Phase 1 (Dependencies)
    |
Phase 2 (Quantizer Module)
    |
Phase 3 (vLLM Integration)
    |
Phase 4 (CLI Command)
    |
Phase 5 (Integration Tests + Report)
```

All phases are sequential — each depends on the previous.

---

## Phase 1: Dependencies and Configuration

### Objective
Install bitsandbytes and accelerate; update pyproject.toml; verify compatibility.

### Implementation Steps

**1.1** Update `pyproject.toml` optional dependencies:
```toml
[project.optional-dependencies]
llama = ["llama-cpp-python[server]>=0.3.0"]
vllm = ["vllm>=0.6.0", "bitsandbytes>=0.43.0", "accelerate>=0.30.0"]
all = ["tqcli[llama,vllm]"]
```

**1.2** Install dependencies:
```bash
pip3 install bitsandbytes>=0.43.0 accelerate>=0.30.0
```

**1.3** Verify bitsandbytes works with CUDA:
```python
import bitsandbytes as bnb
print(bnb.__version__)  # Should be >= 0.43.0
```

**1.4** Verify vLLM accepts `quantization="bitsandbytes"`:
```python
from vllm import LLM
# Test with a small model to confirm the flag is accepted
```

### Files Modified
- `pyproject.toml` — Add bitsandbytes, accelerate to vllm extras

### Dependencies
None (first phase)

### Risks
- bitsandbytes CUDA compatibility on WSL2 — mitigate by testing early
- vLLM 0.19.0 may have specific bitsandbytes version requirements

---

## Phase 2: Quantizer Module

### Objective
Create `tqcli/core/quantizer.py` — the quantization selection and configuration engine.

### Implementation Steps

**2.1** Create `QuantizationMethod` enum:
```python
class QuantizationMethod(Enum):
    NONE = "none"           # BF16/FP16, no quantization
    BNB_INT4 = "bnb_int4"  # bitsandbytes 4-bit (NF4)
    BNB_INT8 = "bnb_int8"  # bitsandbytes 8-bit
    AWQ = "awq"            # AutoAWQ (pre-quantized checkpoints)
    GPTQ = "gptq"          # AutoGPTQ (pre-quantized checkpoints)
    GGUF = "gguf"          # llama.cpp GGUF k-quant
```

**2.2** Implement `estimate_bf16_model_size(model: ModelProfile) -> int` (returns MB):
- Parse parameter count from profile
- Account for multimodal encoders (vision ~1.5 GB, audio ~0.5 GB for Gemma 4)
- Return estimated BF16 VRAM footprint

**2.3** Implement `select_quantization(model, sys_info) -> QuantizationMethod`:
```python
def select_quantization(model, sys_info):
    bf16_size_mb = estimate_bf16_model_size(model)
    available_mb = sys_info.total_vram_mb * 0.75  # 75% for model, rest for KV cache + overhead
    
    if bf16_size_mb <= available_mb:
        return QuantizationMethod.NONE  # Fits at full precision
    
    int4_size_mb = bf16_size_mb * 0.30  # INT4 is ~25-30% of BF16
    if int4_size_mb <= available_mb:
        return QuantizationMethod.BNB_INT4
    
    return None  # Model too large even after quantization
```

**2.4** Implement `get_vllm_quantization_params(method) -> dict`:
- Returns the vLLM constructor kwargs for the chosen method
- For BNB_INT4: `{"quantization": "bitsandbytes", "load_format": "bitsandbytes"}`

### Files Created
- `tqcli/core/quantizer.py`

### Dependencies
Phase 1 (bitsandbytes installed)

---

## Phase 3: vLLM Backend Integration

### Objective
Wire the quantizer into the vLLM config tuner and backend so BF16 models are automatically quantized.

### Implementation Steps

**3.1** Update `VllmTuningProfile` in `vllm_config.py`:
- Add `quantization_method: QuantizationMethod` field
- Add `load_format: str` field (for bitsandbytes)
- Add `estimated_quantized_size_mb: int` field

**3.2** Update `build_vllm_config()` in `vllm_config.py`:
- After Step 1 (estimate model weight), call `select_quantization()`
- If quantization selected, recalculate model_weight_mb using quantized size
- Set `profile.quantization` and `profile.load_format` accordingly
- Recalculate `max_model_len` with the reduced model footprint

Key logic:
```python
from tqcli.core.quantizer import select_quantization, estimate_bf16_model_size

bf16_size = estimate_bf16_model_size(model)
quant_method = select_quantization(model, sys_info)

if quant_method == QuantizationMethod.BNB_INT4:
    model_weight_mb = bf16_size * 0.30  # INT4 reduces to ~30%
    profile.quantization = "bitsandbytes"
    profile.load_format = "bitsandbytes"
    warnings.append(f"BF16 model ({bf16_size} MB) quantized to INT4 via bitsandbytes (~{model_weight_mb:.0f} MB)")
elif quant_method == QuantizationMethod.NONE:
    model_weight_mb = bf16_size  # Full precision
```

**3.3** Update `VllmBackend.load_model()` in `vllm_backend.py`:
- Accept `load_format` kwarg and pass to vLLM `LLM()` constructor
- When `quantization="bitsandbytes"`, also set `load_format="bitsandbytes"`

**3.4** Update `VllmBackend.from_tuning_profile()`:
- Pass load_format from profile to constructor

**3.5** Update `VllmBackend.__init__()`:
- Add `load_format: str | None = None` parameter

### Files Modified
- `tqcli/core/vllm_config.py`
- `tqcli/core/vllm_backend.py`

### Dependencies
Phase 2 (quantizer module exists)

---

## Phase 4: CLI Command and Model Registry

### Objective
Add `tqcli model quantize` command and ensure BF16 model profiles are properly registered.

### Implementation Steps

**4.1** Verify BF16 model profiles exist in registry for:
- `gemma-4-e4b-it-vllm` (google/gemma-4-e4b-it) — BF16 safetensors
- `gemma-4-e2b-it-vllm` (google/gemma-4-e2b-it) — BF16 safetensors
- Add `qwen3-4b-vllm` profile for `Qwen/Qwen3-4B` — BF16 safetensors (NOT the AWQ variant)

**4.2** Add `tqcli model quantize` CLI command:
```python
@model.command("quantize")
@click.argument("model_id")
@click.option("--method", type=click.Choice(["auto", "bnb", "awq", "gguf"]), default="auto")
@click.option("--bits", type=click.Choice(["4", "8"]), default="4")
def model_quantize(model_id, method, bits):
    """Show quantization info and recommendations for a model."""
```

Phase 1 implementation: Info-only command that shows:
- Model's current format (BF16/AWQ/GGUF)
- Estimated BF16 size
- Recommended quantization method for current hardware
- Expected quantized size
- Whether on-the-fly quantization will be used at load time

**4.3** Update `model_pull` to handle BF16 vLLM models (already done — snapshot_download works)

### Files Modified
- `tqcli/cli.py` — Add `model quantize` command
- `tqcli/core/model_registry.py` — Add Qwen 3 4B BF16 profile

### Dependencies
Phase 2 (quantizer for recommendations)

---

## Phase 5: Integration Tests and Comparison Report

### Objective
Create integration tests that download BF16 models, quantize via bitsandbytes, run inference, and produce a comparison report.

### Implementation Steps

**5.1** Create `tests/test_integration_quantization.py`:

**Test 1: vLLM Gemma 4 E4B BF16 → bitsandbytes INT4**
1. Hardware selection (vLLM Gemma 4 profiles)
2. Download BF16 model (google/gemma-4-E4B-it)
3. Verify tuner selects bitsandbytes quantization
4. Load model via vLLM with bitsandbytes INT4
5. Two-turn chat test
6. Capture metrics (tok/s, VRAM, load time)
7. Remove model
8. Clean uninstall check

**Test 2: vLLM Qwen 3 4B BF16 → bitsandbytes INT4**
1. Hardware selection
2. Download BF16 model (Qwen/Qwen3-4B)
3. Verify tuner selects bitsandbytes quantization
4. Load model via vLLM with bitsandbytes INT4
5. Two-turn chat test
6. Capture metrics
7. Remove model
8. Clean uninstall check

**Test 3: llama.cpp Gemma 4 E4B pre-quantized GGUF (baseline)**
1. Download pre-quantized GGUF (unsloth/gemma-4-E4B-it-GGUF Q4_K_M)
2. Load via llama.cpp
3. Two-turn chat test
4. Capture metrics
5. Remove model

**Test 4: llama.cpp Qwen 3 4B pre-quantized GGUF (baseline)**
1. Download pre-quantized GGUF (Qwen/Qwen3-4B-GGUF Q4_K_M)
2. Load via llama.cpp
3. Two-turn chat test
4. Capture metrics
5. Remove model

**5.2** Generate comparison report:
- Output: `tests/integration_reports/quantization_comparison_report.md`
- Output: `tests/integration_reports/quantization_comparison_report.json`
- Compare: vLLM bitsandbytes INT4 vs. llama.cpp GGUF Q4_K_M
- Metrics: tok/s, model load time, VRAM usage, response quality

**5.3** Report format:
```markdown
# Quantization Comparison Report
## BF16 → INT4 (bitsandbytes) vs Pre-Quantized GGUF (Q4_K_M)

| Metric | vLLM bnb INT4 (Gemma 4) | llama.cpp Q4_K_M (Gemma 4) | vLLM bnb INT4 (Qwen 3) | llama.cpp Q4_K_M (Qwen 3) |
```

### Files Created
- `tests/test_integration_quantization.py`
- `tests/integration_reports/quantization_comparison_report.md` (generated)
- `tests/integration_reports/quantization_comparison_report.json` (generated)

### Dependencies
Phases 1-4 (all code changes complete)

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| bitsandbytes doesn't work with vLLM 0.19.0 | Medium | High | Test early in Phase 1; fallback to GPTQ |
| Gemma 4 E4B still too large after INT4 | Low | Medium | Fall back to E2B; adjust test expectations |
| WSL2 multiprocessing issues with bnb | Medium | Medium | Use `enforce_eager=True`, test spawn method |
| transformers 5.5.0 incompatibility | Low | High | Already tested — works despite pip warning |
| Inference quality degradation at INT4 | Low | Low | Compare responses qualitatively in report |

---

## Success Criteria

1. `python3 tests/test_integration_quantization.py` passes all 4 tests
2. Gemma 4 E4B loads on 4 GB VRAM via bitsandbytes INT4
3. Qwen 3 4B BF16 loads on 4 GB VRAM via bitsandbytes INT4
4. Comparison report generated with side-by-side metrics
5. All existing tests (test_basic.py, test_integration.py, test_integration_vllm.py) continue to pass
6. Changes committed and pushed to origin/main

---

## File Manifest

| File | Action | Phase |
|------|--------|-------|
| `pyproject.toml` | Modify — add bitsandbytes, accelerate | 1 |
| `tqcli/core/quantizer.py` | Create — quantization selection engine | 2 |
| `tqcli/core/vllm_config.py` | Modify — integrate quantizer into tuner | 3 |
| `tqcli/core/vllm_backend.py` | Modify — pass load_format to vLLM | 3 |
| `tqcli/core/model_registry.py` | Modify — add Qwen 3 4B BF16 profile | 4 |
| `tqcli/cli.py` | Modify — add model quantize command | 4 |
| `tests/test_integration_quantization.py` | Create — 4-test quantization suite | 5 |
| `tests/integration_reports/quantization_comparison_report.md` | Generated | 5 |
| `tests/integration_reports/quantization_comparison_report.json` | Generated | 5 |

---

## Next Steps

Run `/project-manager` to execute this plan.
