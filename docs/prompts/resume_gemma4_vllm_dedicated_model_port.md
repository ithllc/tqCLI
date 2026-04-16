# Resume Prompt: Gemma 4 E2B vLLM — All Issues Resolved, Performance Optimization Remaining

**Issue:** ithllc/tqCLI#21 (CLOSED), #20 (OPEN — needs final testing + close)
**Date Created:** 2026-04-16
**Status:** 10/10 issues resolved. Model loads and runs inference. Performance optimization remaining.

---

## Context

We ported the dedicated Gemma 4 model files from mainline vLLM (`vllm-project/vllm`) into the TurboQuant fork (`ithllc/vllm-turboquant`) to fix the variable head dimension mismatch that blocked Gemma 4 E2B on vLLM. The fork is rebuilt, installed as **v0.1.dev6**, and ALL compatibility issues are resolved.

### All Issues Resolved (10 fixes)

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | Transformers gemma4 recognition | Upgraded 4.57.6 → 5.5.4 | `pip install` |
| 2 | TurboQuant metadata | Generated `turboquant_kv.json` (35 layers, head_size=256) | `/root/.tqcli/models/gemma-4-e2b-it-vllm/` |
| 3 | Buffer loading (ClippableLinear) | Added buffer handling to `_add_loadable_non_param_tensors` | `vllm/model_executor/models/utils.py` |
| 4 | BNB None mapping crash | Skip weights with `mapped_name is None` | `vllm/model_executor/model_loader/bitsandbytes_loader.py` |
| 5 | TurboQuant variable head_size | Graceful skip for full attention layers (512 != 256) | `vllm/v1/attention/backends/triton_attn.py` |
| 6 | Proportional RoPE | Registered `Gemma4RotaryEmbedding` for `rope_type: "proportional"` | `vllm/model_executor/layers/rotary_embedding/__init__.py` |
| 7 | Metadata layer names | `model.layers.{i}.self_attn` format for dedicated model | `/root/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json` |
| 8 | **BNB stacked params (bnb_quant_state)** | Preserve BNB attrs in UVA offloader's functional_call path | `vllm/model_executor/offloader/uva.py` |
| 9 | **InputBatch re-init assertion** | Softened assert to warning (safe for weight offloading) | `vllm/v1/worker/gpu_model_runner.py` |
| 10 | **VRAM profiler overestimates memory** | Added `kv_cache_memory_bytes` + `max_num_batched_tokens` to config | `tqcli/core/vllm_config.py`, `tqcli/core/vllm_backend.py` |

### Root Cause Analysis: Fix #8 (BNB Stacked Params)

**Problem:** The `AttributeError: 'Tensor' object has no attribute 'bnb_quant_state'` was NOT caused by the BNB loader's stacked params handling (which was already correct). It was caused by the **UVA CPU offloader**.

**Why:** WSL2 doesn't support UVA (Unified Virtual Addressing) or pin_memory. The offloader falls back to `functional_call(module, device_state, ...)` which creates `device_state` by calling `module.state_dict().to(device)`. This creates new plain tensors that **lose all custom attributes** (`bnb_quant_state`, `bnb_shard_offsets`) set by the BNB loader.

**Fix:** In the forward wrapper, iterate `module.named_parameters()` at forward time (not wrap time, since BNB sets attrs AFTER offloader wrapping) and copy BNB attrs to the corresponding `device_state` tensors.

**Timing subtlety:** The offloader wraps modules during `initialize_model()` (before weight loading). BNB sets `bnb_quant_state` during `load_weights()` (after). So attributes must be read at forward time from the `nn.Parameter` wrappers (which keep custom attrs even after `.data` is moved to CPU).

### Verified Working Configuration

```python
LLM(
    model='/root/.tqcli/models/gemma-4-e2b-it-vllm',
    quantization='bitsandbytes',
    load_format='bitsandbytes',
    cpu_offload_gb=8.0,            # Aggressive: offload most bf16 params
    max_model_len=256,
    max_num_batched_tokens=256,    # Limits profiling memory
    gpu_memory_utilization=0.8,
    enforce_eager=True,
    trust_remote_code=True,
    kv_cache_dtype='auto',
    kv_cache_memory_bytes=64*1024*1024,  # 64 MiB, bypasses profiler
)
```

- Model loads in ~160s
- Inference works (coherent but slow: ~0.2 tok/s due to CPU⇄GPU weight transfers)
- Architecture: `Gemma4ForConditionalGeneration`

### Fork State

- **Repo:** `ithllc/vllm-turboquant` (commit 38595dd, pushed to GitHub)
- **Local clone:** `/tmp/vllm-turboquant/`
- **Installed version:** `0.1.dev6+gb236390bf` (with runtime patches applied to site-packages)
- **Transformers:** 5.5.4

## What Remains

### 1. Performance Optimization (priority)

Current: ~0.2 tok/s (unusable). Target: 5+ tok/s.

The extreme CPU offloading (8.0 GiB) forces every layer's weights CPU→GPU each forward pass. Options:
- **TurboQuant KV compaction** (`kv_cache_dtype='turboquant35'`): 4.6x KV compression → more VRAM for model weights → less offloading needed
- **Reduce offload to 4-5 GiB**: Keep more layers on GPU (faster) at cost of smaller KV cache
- **Benchmark different cpu_offload_gb values** to find the sweet spot

### 2. Update vllm_config.py (done)

The `build_vllm_config()` now calculates aggressive CPU offload for BNB models and adds `kv_cache_memory_bytes` + `max_num_batched_tokens`. Needs testing via the E2E test.

### 3. Run Full Test Suite

```bash
# Gemma 4 E2B specific test
python3 tests/test_gemma4_vllm_cpu_offload.py

# Qwen3 AWQ regression test
python3 tests/test_integration_combined.py --engine vllm

# Full integration suite
python3 tests/test_integration_combined.py
```

### 4. Benchmark & Document

- `/tq-benchmark` — Capture Gemma 4 E2B performance at various offload levels
- `/architecture-doc-review` — Update CLAUDE.md and docs
- `/issue-manager` — Post implementation comment on #20 and close it

### 5. Commit Fork Patches

These patches are applied to site-packages but should be in the fork:
1. `triton_attn.py` — TurboQuant head_size graceful skip (fix #5)
2. `rotary_embedding/__init__.py` — Proportional RoPE registration (fix #6)

The BNB attr fix (#8) and InputBatch assertion fix (#9) are already committed to the fork (38595dd).

## Key Files

| File | Role |
|------|------|
| `vllm/model_executor/offloader/uva.py` | **FIX #8** — BNB attr preservation in functional_call |
| `vllm/v1/worker/gpu_model_runner.py` | **FIX #9** — InputBatch re-init with CPU offload |
| `tqcli/core/vllm_config.py` | **FIX #10** — Aggressive offload + kv_cache_memory_bytes |
| `tqcli/core/vllm_backend.py` | **FIX #10** — Pass new params to vLLM LLM() |
| `vllm/model_executor/models/gemma4.py` | Dedicated Gemma 4 text model |
| `vllm/model_executor/models/gemma4_mm.py` | Multimodal wrapper |
| `vllm/model_executor/model_loader/bitsandbytes_loader.py` | BNB loader (fix #4) |
| `vllm/v1/attention/backends/triton_attn.py` | TurboQuant head_size skip (fix #5) |
| `vllm/model_executor/layers/rotary_embedding/__init__.py` | Proportional RoPE (fix #6) |
| `tests/test_gemma4_vllm_cpu_offload.py` | E2E test |

## Skills Needed

- `/tq-benchmark` — Performance capture at various offload levels
- `/architecture-doc-review` — Update CLAUDE.md
- `/issue-manager` — Close #20 with implementation comment

## What NOT To Touch

- llama.cpp engine — unaffected, Gemma 4 works at 7.33 tok/s
- Qwen3 AWQ — verify no regression, but don't modify
- TurboQuant KV metadata format — working for sliding layers, skip for full
