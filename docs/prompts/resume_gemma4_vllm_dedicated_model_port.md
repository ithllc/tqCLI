# Resume Prompt: Gemma 4 E2B vLLM — Dedicated Model Port + BNB Stacked Params Fix

**Issue:** ithllc/tqCLI#21 (depends on #20)
**Date Created:** 2026-04-16
**Status:** 7/8 issues resolved, blocked on BNB stacked params quantization

---

## Context

We ported the dedicated Gemma 4 model files from mainline vLLM (`vllm-project/vllm`) into the TurboQuant fork (`ithllc/vllm-turboquant`) to fix the variable head dimension mismatch that blocked Gemma 4 E2B on vLLM. The fork is rebuilt, installed as **v0.1.dev6**, and 7 of 8 compatibility issues are resolved.

### What's Done (7 fixes)

| # | Issue | Fix | Where |
|---|---|---|---|
| 1 | Transformers gemma4 recognition | Upgraded 4.57.6 → 5.5.4 | `pip install` |
| 2 | TurboQuant metadata | Generated `turboquant_kv.json` (35 layers, head_size=256) | `/root/.tqcli/models/gemma-4-e2b-it-vllm/` |
| 3 | Buffer loading (ClippableLinear) | Added buffer handling to `_add_loadable_non_param_tensors` | `vllm/model_executor/models/utils.py` |
| 4 | BNB None mapping crash | Skip weights with `mapped_name is None` | `vllm/model_executor/model_loader/bitsandbytes_loader.py` |
| 5 | TurboQuant variable head_size | Graceful skip for full attention layers (512 != 256) | `vllm/v1/attention/backends/triton_attn.py` |
| 6 | Proportional RoPE | Registered `Gemma4RotaryEmbedding` for `rope_type: "proportional"` | `vllm/model_executor/layers/rotary_embedding/__init__.py` |
| 7 | Metadata layer names | `model.layers.{i}.self_attn` format for dedicated model | `/root/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json` |

### What's Blocked (1 remaining)

**#8: BNB stacked params** — The dedicated model's `load_weights()` stacks individual HF weights (`q_proj`, `k_proj`, `v_proj`) into a merged vLLM parameter (`qkv_proj`) via `QKVParallelLinear`. BNB-quantized tensors lose their `bnb_quant_state` attribute during this stacking.

```
AttributeError: 'Tensor' object has no attribute 'bnb_quant_state'
```

**Root cause:** The fork's BNB loader (v0.1.dev5 vintage) was designed for the generic Transformers backend which doesn't use stacked params. Mainline vLLM has an updated BNB loader that handles stacked params for dedicated models.

**Verified working so far:**
- Model architecture resolves to `Gemma4ForConditionalGeneration` (dedicated model, not Transformers backend)
- TurboQuant enabled on 28/35 sliding attention layers, 7 full attention layers gracefully skipped
- Proportional RoPE registered and working
- BNB weight loading completes (6.47 GiB, 88s) — but stacked params lack quantization state
- Model loading itself succeeds — failure is during profiling/dummy forward pass

### Fork State

- **Repo:** `ithllc/vllm-turboquant` (commit b236390, pushed to GitHub)
- **Local clone:** `/tmp/vllm-turboquant/`
- **Installed version:** `0.1.dev6+gb236390bf`
- **Transformers:** 5.5.4 (re-upgraded after build downgraded it)

### Additional patches applied to installed site-packages (NOT yet in fork repo)

These were applied directly to `/usr/local/lib/python3.11/site-packages/vllm/` during debugging:

1. `triton_attn.py` — TurboQuant head_size graceful skip (fix #5)
2. `rotary_embedding/__init__.py` — Proportional RoPE registration (fix #6)

These ARE in the fork clone at `/tmp/vllm-turboquant/` but the fork wasn't rebuilt after adding them. They need to be committed and the fork rebuilt, OR applied as runtime patches.

## Structured Approach (Execute in Order)

### Phase 1: Fix BNB Stacked Params

1. **Fetch mainline BNB loader** to compare:
   ```bash
   gh api repos/vllm-project/vllm/contents/vllm/model_executor/model_loader/bitsandbytes_loader.py \
     --jq '.content' | base64 -d > /tmp/gemma4_port/mainline_bnb_loader.py
   ```

2. **Diff the BNB stacked weight handling:**
   - Compare `_unquantized_generator` in mainline vs fork
   - Look for `stacked_params_mapping` or `QKVParallelLinear` handling
   - Check if mainline quantizes AFTER stacking (vs fork which may quantize BEFORE)

3. **Port the fix** — likely one of:
   - Update `_unquantized_generator` to handle merged weight patterns
   - Or add post-load quantization for stacked params
   - Or port mainline's `_apply_4bit_weight` which may handle missing `bnb_quant_state`

4. **Test:** Run `python3 tests/test_gemma4_vllm_cpu_offload.py`

### Phase 2: Commit & Rebuild

5. **Commit all pending patches to fork:**
   ```bash
   cd /tmp/vllm-turboquant
   # Commit: triton_attn.py, rotary_embedding/__init__.py, bnb fix
   git add -A && git commit -m "Fix BNB stacked params, TurboQuant head_size skip, proportional RoPE"
   git push origin main
   ```

6. **Rebuild fork** (if BNB fix requires code changes, not just runtime patches):
   ```bash
   cd /tmp/vllm-turboquant
   CUDA_HOME=/usr/local/cuda-12.8 VLLM_TARGET_DEVICE=cuda MAX_JOBS=2 \
     python3 -m pip install . --no-build-isolation
   ```
   Note: Build takes ~4 hours at MAX_JOBS=2. Only rebuild if C++ changes needed. Python-only changes can be applied directly to site-packages.

7. **Re-upgrade Transformers** (build will downgrade it):
   ```bash
   python3 -m pip install --upgrade transformers
   ```

### Phase 3: Test & Benchmark

8. **Test Gemma 4 E2B:**
   ```bash
   python3 tests/test_gemma4_vllm_cpu_offload.py
   ```

9. **Regression test Qwen3 AWQ:**
   ```bash
   python3 tests/test_integration_combined.py --engine vllm
   ```

10. **Full integration suite:**
    ```bash
    python3 tests/test_integration_combined.py
    ```

11. **Benchmark** (`/tq-benchmark`):
    - Gemma 4 E2B on vLLM: BNB_INT4 + cpu_offload 2.1 GB + turboquant35
    - Compare against llama.cpp baseline (7.33 tok/s turbo3)

### Phase 4: Document & Close

12. **Update reports:**
    - `tests/integration_reports/gemma4_vllm_cpu_offload_report.md`
    - `tests/integration_reports/turboquant_kv_comparison_report.md`

13. **Close issues:**
    - Post implementation comment on #21 and #20
    - Close both issues

14. **Use skills:**
    - `/tq-benchmark` — Performance capture
    - `/architecture-doc-review` — Update CLAUDE.md and docs
    - `/issue-manager` — Close issues with implementation comments

## Key Files

| File | Role |
|------|------|
| `vllm/model_executor/models/gemma4.py` | Dedicated Gemma 4 text model (ported from mainline) |
| `vllm/model_executor/models/gemma4_mm.py` | Multimodal wrapper (ported from mainline) |
| `vllm/model_executor/layers/rotary_embedding/gemma4_rope.py` | Proportional RoPE (ported) |
| `vllm/model_executor/layers/rotary_embedding/__init__.py` | RoPE factory — needs `proportional` registration |
| `vllm/model_executor/models/registry.py` | Gemma4 registration entries |
| `vllm/transformers_utils/model_arch_config_convertor.py` | `Gemma4ModelArchConfigConvertor` |
| `vllm/model_executor/models/utils.py` | Buffer loading fix |
| `vllm/model_executor/model_loader/bitsandbytes_loader.py` | BNB loader — **THE FILE TO FIX** |
| `vllm/v1/attention/backends/triton_attn.py` | TurboQuant head_size graceful skip |
| `/root/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json` | TurboQuant metadata |
| `tests/test_gemma4_vllm_cpu_offload.py` | E2E test |
| `tests/test_integration_combined.py` | Full unified test runner |

## What NOT To Touch

- llama.cpp engine — unaffected, Gemma 4 works at 7.33 tok/s
- Qwen3 AWQ — already verified no regression with Transformers 5.5.4
- tqCLI pipeline logic — already correct (cpu_offload, build_vllm_config)
- TurboQuant KV metadata format — working for sliding layers, skip for full

## Expected Final State

- Gemma 4 E2B runs on vLLM with BNB_INT4 + cpu_offload 2.1 GB + turboquant35 KV (28/35 layers)
- Multi-turn chat produces coherent output
- Qwen3 AWQ still works (no regression)
- All integration tests pass
- Fork pushed to `ithllc/vllm-turboquant` with all fixes
- Issues #20 and #21 closed
- Benchmark data captured
