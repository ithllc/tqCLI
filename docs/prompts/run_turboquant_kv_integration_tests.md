# Prompts for Running TurboQuant KV Cache Integration Tests

**Date:** 2026-04-14
**GitHub Issue:** [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13)
**Purpose:** Copy-paste these prompts to Claude Code to execute the TurboQuant KV cache integration tests tonight.

---

## Prompt 1: Build llama.cpp TurboQuant Backend

```
Execute Phase 1 of the technical plan at docs/technical_plans/TP_turboquant_kv_cache.md for GitHub issue ithllc/tqCLI#13.

Build llama-cpp-python against TheTom/turboquant_plus fork:
1. Clone https://github.com/TheTom/turboquant_plus.git (branch: feature/turboquant-kv-cache)
2. Build with CUDA support (DGGML_CUDA=ON) for SM86 (RTX A2000)
3. Install the Python bindings (llama-cpp-python) against this fork
4. Verify turbo3/turbo4 cache types are accepted by the Llama() constructor
5. If llama-cpp-python doesn't expose cache_type_k/cache_type_v, investigate the fork's Python bindings or use the compiled binary directly

If you encounter issues, file them via gh issue create on ithllc/tqCLI, fix them, and comment the solution on the issue. Update tqcli/core/llama_backend.py with cache_type_k and cache_type_v parameters.
```

---

## Prompt 2: Build vLLM TurboQuant Backend

```
Execute Phase 2 of the technical plan at docs/technical_plans/TP_turboquant_kv_cache.md for GitHub issue ithllc/tqCLI#13.

Install mitkox/vllm-turboquant from source:
1. Clone https://github.com/mitkox/vllm-turboquant.git
2. Build from source with CUDA_HOME=/usr/local/cuda-12.8, VLLM_TARGET_DEVICE=cuda
3. Verify that kv_cache_dtype="turboquant35" is accepted by vLLM LLM()
4. Verify that --attention-backend TRITON_ATTN works on our RTX A2000 (SM86)

If you encounter issues, file them via gh issue create on ithllc/tqCLI, fix them, and comment the solution. Update tqcli/core/vllm_backend.py and tqcli/core/vllm_config.py with turboquant KV support.
```

---

## Prompt 3: Create KV Quantizer Module and CLI Flag

```
Execute Phase 3 of the technical plan at docs/technical_plans/TP_turboquant_kv_cache.md for GitHub issue ithllc/tqCLI#13.

1. Create tqcli/core/kv_quantizer.py with:
   - KVQuantLevel enum (NONE, TURBO4, TURBO3, TURBO2)
   - select_kv_quant(available_kv_mb, engine) function
   - get_llama_kv_params(level) and get_vllm_kv_params(level) mappers
2. Add --kv-quant flag to tqcli chat command (choices: auto, none, turbo4, turbo3, turbo2)
3. Wire the flag through to LlamaBackend (cache_type_k/v) and VllmBackend (kv_cache_dtype)
4. Verify all 14 basic tests still pass
5. Test the --kv-quant flag manually with a loaded model

If you encounter issues, file them via gh issue create, fix, and comment.
```

---

## Prompt 4: Write and Run Integration Tests

```
Execute Phase 4 of the technical plan at docs/technical_plans/TP_turboquant_kv_cache.md for GitHub issue ithllc/tqCLI#13.

Create tests/test_integration_turboquant_kv.py based on the test cases at tests/integration_reports/turboquant_kv_test_cases.md:

Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV cache
Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV cache (also test turbo4 and turbo2)
Test 3: vLLM Qwen 3 4B AWQ + turboquant35 KV cache
Test 4: Baseline comparison (q8_0/auto KV for all models)

Each test should:
- Download the model (reuse from previous tests if cached)
- Load with TurboQuant KV cache type
- Run 2 chat turns (factual questions)
- Capture metrics: tok/s, context capacity, VRAM usage
- Generate comparison report at tests/integration_reports/turboquant_kv_comparison_report.md and .json

Run all tests. If any test fails, file a GitHub issue via gh issue create on ithllc/tqCLI, fix the issue, comment the solution, then continue. Use your issue-manager and project-manager skill sets as needed.
```

---

## Prompt 5: Post-Integration Tasks

```
All TurboQuant KV cache integration tests have been run for GitHub issue ithllc/tqCLI#13. Now perform post-integration tasks:

1. Review all GitHub issues filed during this test run
2. For each issue, verify the fix was applied and comment the solution if not already done
3. Determine if fixes structurally changed the application
4. Update ALL affected documentation:
   - CLAUDE.md — Add kv_quantizer.py, --kv-quant flag to workspace structure
   - tests/integration_reports/llama_cpp_test_cases.md — Add TurboQuant KV test cases
   - tests/integration_reports/vllm_test_cases.md — Add TurboQuant KV test cases
   - README.md — Add TurboQuant KV cache section with usage examples
5. Close all open issues related to this test run
6. Bump version to v0.5.0
7. Commit all changes with a descriptive message
8. Push to origin/main
9. Report the final summary including: test results, issues filed/fixed, documentation updated, comparison report highlights
```

---

## Execution Order

Run prompts in order: 1 → 2 → 3 → 4 → 5

Each prompt is self-contained and assumes the previous prompt's work is complete. If a prompt fails partway through, re-run it — it should be idempotent.

**Estimated time:** 60-90 minutes total (dominated by model downloads and compilation)
