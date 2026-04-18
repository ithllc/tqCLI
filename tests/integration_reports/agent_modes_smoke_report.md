# Agent Modes — Integration Test Report

**Generated:** 2026-04-18T13:26:13
**System:** NVIDIA RTX A2000 Laptop GPU (4096 MB VRAM)
**CUDA:** driver n/a, toolkit n/a
**OS:** Linux (Ubuntu 22.04.4 LTS) (WSL2)

Exercises the Phase 2 / Phase 3 orchestrator (`tqcli/core/agent_orchestrator.py`) end-to-end with a real Gemma 4 E2B model on both backends and TurboQuant KV compression active.

---

## Summary

| Test | Engine | Model | KV Quant | Mode | Result |
|------|--------|-------|----------|------|--------|
| llama.cpp Gemma 4 E2B + turbo3 KV + ai_tinkering | llama.cpp | gemma-4-e2b-it-Q4_K_M | turbo3 | ai_tinkering | **PASS** (5/5) |
| vLLM Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35 + unrestricted | vllm | gemma-4-e2b-it-vllm | turboquant35 | unrestricted | **PASS** (6/6) |

---

## llama.cpp Gemma 4 E2B + turbo3 KV + ai_tinkering

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **KV Quant:** turbo3
- **Agent Mode:** ai_tinkering
- **Started / Finished:** 2026-04-18T13:17:04 → 2026-04-18T13:17:27
- **Total Duration:** 22.43s
- **Result:** PASS (5/5)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| model_file_present | PASS | 0.00s | path=/root/.tqcli/models/gemma-4-E2B-it-Q4_K_M.gguf exists=True |
| resolve_kv_params | PASS | 0.00s | tq_available=True, level=turbo3, params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |
| load_model | PASS | 2.24s | loaded Gemma 4 E2B Edge Instruct (Q4_K_M) (n_ctx=2048) |
| schema_injection_non_empty | PASS | 0.00s | 4 tool schemas surfaced in agent mode |
| orchestrator_run_turn | PASS | 20.19s | final_text_len=3 / emitted_staged_tag=False / observations_fed_back=1 / final_text_head=OK. |

## vLLM Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35 + unrestricted

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **KV Quant:** turboquant35
- **Agent Mode:** unrestricted
- **Started / Finished:** 2026-04-18T13:17:27 → 2026-04-18T13:26:13
- **Total Duration:** 522.38s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| model_dir_present | PASS | 0.00s | path=/root/.tqcli/models/gemma-4-e2b-it-vllm dir=True |
| vllm_importable | PASS | 0.00s | vllm import OK |
| build_vllm_config | PASS | 0.00s | feasible=True / quant=bitsandbytes / cpu_offload_gb=9.9 / kv=turboquant35 / enforce_eager=True |
| load_model | PASS | 202.41s | loaded Gemma 4 E2B Edge Instruct (vLLM BF16) via vLLM |
| schema_injection_non_empty | PASS | 0.00s | 4 tool schemas surfaced in agent mode |
| orchestrator_react_loop | PASS | 319.97s | final_text_len=0 / emitted_live_tag=False / observations_fed_back=0 / final_text_head= |
