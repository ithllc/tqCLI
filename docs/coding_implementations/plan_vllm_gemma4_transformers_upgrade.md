# Technical Implementation Plan: vLLM TurboQuant Fork Gemma 4 Support

**PRD:** `docs/coding_implementations/prd_vllm_gemma4_transformers_upgrade.md`
**Issue:** ithllc/tqCLI#20
**Date:** 2026-04-15

---

## Overview

Unblock Gemma 4 E2B on the TurboQuant vLLM fork by upgrading Transformers from 4.57.6 (no `gemma4` architecture) to 5.5.x (full Gemma 4 support). The CPU offloading pipeline is already implemented and verified — this plan addresses only the dependency gap, regression validation, and E2E verification.

## Architecture

No tqCLI code changes are expected. The fix is a dependency upgrade:

```
Transformers 4.57.6 ──(upgrade)──> 5.5.x
         │                            │
         ├── gemma3 ✓                 ├── gemma3 ✓ (regression check)
         ├── qwen3 ✓                  ├── qwen3 ✓ (regression check)
         └── gemma4 ✗ BLOCKED         └── gemma4 ✓ UNBLOCKED
```

**Critical path:** Phase 1 → Phase 2 (gate) → Phase 3 (conditional) → Phase 4 → Phase 5 → Phase 6

---

## Phase 1: Upgrade Transformers

### Objective
Upgrade the `transformers` package from 4.57.6 to >=5.5.0 and verify `gemma4` is recognized.

### Implementation Steps
1. Record current state: `pip show transformers vllm bitsandbytes`
2. Upgrade: `pip install --upgrade transformers`
3. Verify gemma4 architecture exists:
   ```python
   python3 -c "from transformers import AutoConfig; print('gemma4' in AutoConfig._model_type_to_module_name)"
   ```
4. Verify Gemma 4 model classes are importable:
   ```python
   python3 -c "from transformers import Gemma4ForConditionalGeneration; print('OK')"
   ```

### Files
- No file changes — pip upgrade only

### Dependencies
- None (first phase)

### Rollback
```bash
pip install transformers==4.57.6
```

### Risk
- **Major version jump (4.x → 5.x):** May break vLLM fork's internal Transformers API calls
- **Mitigation:** Phase 2 is a hard gate — no further work until Qwen3 passes

---

## Phase 2: Regression Test — Qwen3 AWQ (GATE)

### Objective
Verify Qwen3 AWQ + turboquant35 KV cache still works on vLLM after the Transformers upgrade. This is a **hard gate** — failure triggers Phase 3.

### Implementation Steps
1. Run Qwen3 AWQ thinking test:
   ```bash
   python3 tests/test_integration_combined.py --test 5 --engine vllm
   ```
   Expected: PASS (Qwen3-1.7B AWQ + turboquant35 thinking mode)

2. Run Qwen3 AWQ tool calling test:
   ```bash
   python3 tests/test_integration_combined.py --test 6 --engine vllm
   ```
   Expected: PASS

3. Run Qwen3 AWQ chat test:
   ```bash
   python3 tests/test_integration_combined.py --test 7 --engine vllm
   ```
   Expected: PASS

### Gate Logic
- **All 3 PASS:** Skip Phase 3, proceed to Phase 4
- **Any FAIL:** Enter Phase 3 (fork rebuild)

### Files
- `tests/test_integration_combined.py` (read-only execution)

### Dependencies
- Phase 1 must complete first

---

## Phase 3: Rebuild vLLM TurboQuant Fork (CONDITIONAL)

### Objective
Rebuild the vLLM TurboQuant fork from source against the upgraded Transformers. Only execute if Phase 2 detects regressions.

### Implementation Steps
1. Clone the fork:
   ```bash
   cd /tmp
   git clone https://github.com/ithllc/vllm-turboquant.git
   cd vllm-turboquant
   ```

2. Build from source with CUDA 12.8:
   ```bash
   CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 pip install . --no-build-isolation
   ```
   Note: `MAX_JOBS=2` to avoid OOM on 32 GB RAM during build.

3. If build fails on pyproject.toml license field (known issue from #15):
   - Fix `license` field to use SPDX format
   - Re-run build

4. Verify fork version:
   ```bash
   pip show vllm | grep Version
   ```
   Expected: `0.1.dev5+...cu128` (or newer commit hash)

5. Re-run Phase 2 regression tests — all must pass before continuing

### Files
- No tqCLI code changes
- vLLM fork build artifacts installed to site-packages

### Dependencies
- Phase 2 must FAIL to trigger this phase
- Phase 1 (Transformers upgrade) must stay in place

### Risk
- Build takes ~25-30 min on this hardware
- cmake/CUDA compatibility issues (documented fixes from #15)

---

## Phase 4: Gemma 4 E2B vLLM E2E Verification

### Objective
Load and run Gemma 4 E2B on vLLM with the full pipeline: BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV.

### Implementation Steps
1. Run the dedicated Gemma 4 CPU offload test:
   ```bash
   python3 tests/test_gemma4_vllm_cpu_offload.py
   ```
   Expected results per stage:
   - Stage 1 (imports): PASS
   - Stage 2 (system info): PASS
   - Stage 3 (config builder): PASS — `feasible=True, cpu_offload_gb=2.1`
   - Stage 4 (model load): PASS — `gemma4` recognized, model loaded
   - Stage 5 (single chat): PASS — coherent response
   - Stage 6 (multi-turn): PASS — contextual follow-up

2. If Stage 4 fails with a new error (not architecture recognition):
   - Run the pipeline diagnostic for stage-by-stage isolation:
     ```bash
     python3 tests/test_gemma4_vllm_pipeline_diagnostic.py
     ```
   - Diagnose whether the issue is in vLLM, bitsandbytes, or Transformers

3. Capture results to report:
   - `tests/integration_reports/gemma4_vllm_cpu_offload_report.json`
   - `tests/integration_reports/gemma4_vllm_cpu_offload_report.md`

### Files
- `tests/test_gemma4_vllm_cpu_offload.py` (execution)
- `tests/test_gemma4_vllm_pipeline_diagnostic.py` (fallback)
- `tests/integration_reports/gemma4_vllm_cpu_offload_report.*` (output)

### Dependencies
- Phase 2 must PASS (or Phase 3 must resolve regressions)

### Key Configuration (from `vllm_config.py`)
```python
VllmTuningProfile(
    feasible=True,
    quantization="bitsandbytes",
    load_format="bitsandbytes",
    cpu_offload_gb=2.1,
    kv_cache_dtype="turboquant35",
    max_model_len=2048,
    enforce_eager=True,
    gpu_memory_utilization=0.80,
)
```

---

## Phase 5: Full Integration Test Suite

### Objective
Run the complete integration suite to confirm no regressions across all models and both engines.

### Implementation Steps
1. Run all tests:
   ```bash
   python3 tests/test_integration_combined.py
   ```
   Expected: 6 tests, all PASS

2. Test matrix:

   | Test | Engine | Model | Mode | Expected |
   |------|--------|-------|------|----------|
   | 5 | llama.cpp | Gemma 4 E2B | thinking | PASS |
   | 5 | vLLM | Qwen3 AWQ | thinking | PASS |
   | 6 | llama.cpp | Gemma 4 E2B | tool calling | PASS |
   | 6 | vLLM | Qwen3 AWQ | tool calling | PASS |
   | 7 | llama.cpp | Gemma 4 E2B | chat | PASS |
   | 7 | vLLM | Qwen3 AWQ | chat | PASS |

### Files
- `tests/test_integration_combined.py` (execution)
- `tests/integration_reports/turboquant_kv_comparison_report.md` (update with new results)

### Dependencies
- Phase 4 must PASS

---

## Phase 6: Benchmark Gemma 4 E2B on vLLM

### Objective
Capture performance numbers for Gemma 4 E2B on vLLM and compare against the llama.cpp Q4_K_M + turbo3 baseline.

### Implementation Steps
1. Run `/tq-benchmark` for Gemma 4 E2B on vLLM:
   - Configuration: BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV
   - Metrics: tok/s, time-to-first-token, total throughput

2. Record llama.cpp baseline (already captured):
   - Gemma 4 E2B Q4_K_M + turbo3: ~7.33 tok/s

3. Add results to comparison report:
   - `tests/integration_reports/turboquant_kv_comparison_report.md`

4. Expected performance range:
   - vLLM with BNB_INT4 + cpu_offload likely slower than llama.cpp Q4_K_M due to CPU↔GPU memory transfers
   - Acceptable if >3 tok/s (above handoff threshold)

### Files
- `tests/integration_reports/turboquant_kv_comparison_report.md` (update)

### Dependencies
- Phase 5 must PASS

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Transformers 5.x breaks vLLM fork | Medium | High | Phase 3 rebuild; rollback to 4.57.6 |
| bitsandbytes incompatible with Transformers 5.x | Low | High | Check bnb release notes; pin version if needed |
| Gemma 4 E2B OOMs on 4 GB VRAM despite offload | Low | Medium | Reduce max_model_len, increase cpu_offload_gb |
| vLLM fork rebuild fails | Low | Medium | Follow #15 fixes (pyproject.toml, cmake) |
| cpu_offload degrades performance below threshold | Medium | Low | Acceptable if >3 tok/s; document in benchmark |

## Success Criteria

1. `python3 -c "from transformers import Gemma4ForConditionalGeneration"` succeeds
2. Qwen3 AWQ Tests 5-7 on vLLM all PASS (no regression)
3. Gemma 4 E2B loads and runs on vLLM with cpu_offload + turboquant35
4. Full integration suite: 6/6 PASS
5. Benchmark data captured and added to comparison report
6. Issue #20 closed with implementation comment

---

**Next step:** Execute Phase 1 — `pip install --upgrade transformers`
