# Resume Prompt: Gemma 4 E2B vLLM — Transformers Upgrade + CPU Offloading Fix

**Issue:** ithllc/tqCLI#20
**Date Created:** 2026-04-15
**Status:** CPU offload pipeline logic implemented, blocked on Transformers not recognizing `gemma4` architecture in vLLM fork

---

## Context

tqCLI's unified quantization pipeline now supports CPU offloading for models that exceed GPU VRAM. For Gemma 4 E2B on a 4 GB RTX A2000 (WSL2):

- **Pipeline path:** `detect full_precision → weight:bnb_int4 → cpu_offload 2.1 GB → kv:turboquant35`
- **Config builder returns:** `feasible=True, cpu_offload_gb=2.1, quantization=bitsandbytes, kv_cache_dtype=turboquant35`
- **Model load fails:** Transformers does not recognize `gemma4` architecture (the TurboQuant vLLM fork v0.1.dev5 was built against older Transformers)

The CPU offloading code changes are already in place:
- `tqcli/core/vllm_config.py` — `cpu_offload_gb` field + auto-detection when INT4 exceeds VRAM but system RAM is available
- `tqcli/core/vllm_backend.py` — `cpu_offload_gb` param passed to vLLM `LLM()` constructor

llama.cpp is NOT affected — it uses GGUF files directly and has no Transformers dependency.

## Structured Approach (Execute in Order)

### Phase 1: Log & Plan (use skills)

1. **`/issue-manager`** — Review issue #20 and add a sub-issue or update for the Transformers upgrade dependency. Document:
   - Current Transformers version installed (`pip show transformers`)
   - Minimum version needed for `gemma4` architecture support
   - Risk assessment: will upgrading break existing Qwen3 AWQ vLLM tests?

2. **`/prd-generator`** — Generate a focused PRD for "vLLM TurboQuant Fork Gemma 4 Support" covering:
   - Scope: Transformers upgrade + fork rebuild + Gemma 4 E2B E2E verification
   - Success criteria: Gemma 4 E2B loads on vLLM with BNB_INT4 + cpu_offload + turboquant35 KV
   - Hardware matrix: RTX A2000 4 GB (WSL2), target 6 GB+ GPUs
   - Out of scope: AWQ/GPTQ pre-quantized Gemma 4 (none exist on HuggingFace)

3. **`/technical-planner`** — Create phased implementation plan from the PRD:
   - Phase 1: Upgrade Transformers (`pip install transformers>=5.5.0`)
   - Phase 2: Regression test Qwen3 AWQ on vLLM (must still work)
   - Phase 3: Rebuild vllm-turboquant from source if needed (follow #15 pattern)
   - Phase 4: Test Gemma 4 E2B load with CPU offload
   - Phase 5: Run full integration test suite (`python3 tests/test_integration_combined.py`)
   - Phase 6: Benchmark Gemma 4 E2B vLLM performance (`/tq-benchmark`)

### Phase 2: Execute the Fix

4. **Check current Transformers version:**
   ```bash
   pip show transformers | grep Version
   ```

5. **Upgrade Transformers:**
   ```bash
   pip install --upgrade transformers
   ```

6. **Regression test — Qwen3 AWQ must still work:**
   ```bash
   python3 tests/test_integration_combined.py --test 5 --engine vllm
   ```
   Expected: PASS (Qwen3 AWQ + turboquant35 KV thinking test)

7. **If Qwen3 regression fails** — the vLLM fork needs a rebuild:
   ```bash
   cd /tmp
   git clone https://github.com/ithllc/vllm-turboquant.git
   cd vllm-turboquant
   CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 pip install . --no-build-isolation
   ```
   Then re-run step 6.

8. **Test Gemma 4 E2B with CPU offloading:**
   ```bash
   python3 tests/test_gemma4_vllm_cpu_offload.py
   ```
   Expected: All steps PASS including model load + chat turns

9. **Run full integration suite:**
   ```bash
   python3 tests/test_integration_combined.py
   ```
   Expected: 6 tests, all PASS

### Phase 3: Document & Close

10. **`/tq-benchmark`** — Benchmark Gemma 4 E2B on vLLM:
    - tok/s with BNB_INT4 + cpu_offload + turboquant35
    - Compare to llama.cpp Q4_K_M + turbo3 baseline
    - Add results to `tests/integration_reports/turboquant_kv_comparison_report.md`

11. **`/architecture-doc-review`** — Update documentation:
    - CLAUDE.md: Add CPU offloading to quick commands
    - Update test cases doc with Gemma 4 E2B vLLM results

12. **`/project-manager`** — Close out:
    - Post implementation comment on issue #20
    - Close issue #20
    - Commit all changes and push

## Key Files

| File | Role |
|------|------|
| `tqcli/core/vllm_config.py` | CPU offload logic (already implemented) |
| `tqcli/core/vllm_backend.py` | cpu_offload_gb param (already implemented) |
| `tests/test_gemma4_vllm_cpu_offload.py` | E2E test for Gemma 4 + CPU offload |
| `tests/test_gemma4_vllm_pipeline_diagnostic.py` | Pipeline stage-by-stage diagnostic |
| `tests/test_integration_combined.py` | Full unified test runner |
| `tests/integration_reports/gemma4_vllm_cpu_offload_report.json` | Last test results (load failed on Transformers) |

## Prior Art (How #15 Was Structured)

Issue #15 (Build vllm-turboquant from source for CUDA 12.8) followed this pattern:
1. Identified dependency gap (stock vLLM lacks turboquant35 KV dtype)
2. Forked mitkox/vllm-turboquant → ithllc/vllm-turboquant
3. Built from source with CUDA 12.8 (`CUDA_HOME=/usr/local/cuda-12.8`)
4. Fixed build issues (pyproject.toml license field, cmake version)
5. E2E verified with Qwen3 AWQ + turboquant35
6. Pushed fixes to fork repo (#17)
7. Documented in resume prompt and integration reports

This fix follows the same structure but the dependency is Transformers version, not CUDA toolkit.

## Expected Final State

- Gemma 4 E2B runs on vLLM with 4 GB VRAM via BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV
- Qwen3 AWQ still works (no regression)
- All 6 integration tests pass (Tests 5-7 on both engines)
- Issue #20 closed with implementation comment
- Benchmark data captured for Gemma 4 E2B vLLM performance
