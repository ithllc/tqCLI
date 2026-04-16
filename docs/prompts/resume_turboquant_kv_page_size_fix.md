# Resume Prompt: TurboQuant KV Page Size Unification for Variable head_dim Models

**Issue:** ithllc/tqCLI#22
**Date Created:** 2026-04-16
**Status:** Research complete, implementation needed
**Depends on:** Fix #5 (triton_attn.py graceful skip, fork commit d9a9157)

---

## Context

TurboQuant KV cache compression (`kv_cache_dtype='turboquant35'`) fails for models with mixed head dimensions because `unify_kv_cache_spec_page_size()` in `kv_cache_utils.py` requires exact divisibility between page sizes. Gemma 4 has head_dim=256 (28 sliding layers) and head_dim=512 (7 full attention layers), producing page sizes of 30,720 and 59,392 bytes — ratio 1.933, not an integer.

Qwen3 (uniform head_dim=128) is unaffected.

### Page Size Math

| head_dim | packed_dim | page_size_bytes | block_size=16 |
|----------|-----------|-----------------|---------------|
| 256 | 120 | 30,720 | Standard |
| 512 | 232 | 59,392 | Standard |

- `59,392 % 30,720 = 28,672` (NOT divisible)
- `GCD = 2,048`
- `LCM = 890,880` (870 KB — very large blocks)

### Execution Flow (Where It Crashes)

```
get_kv_cache_groups()  [kv_cache_utils.py:1245]
  → is_kv_cache_spec_uniform()  → NO
  → UniformTypeKVCacheSpecs.from_specs()  → NO
  → unify_kv_cache_spec_page_size()  → CRASH at line 942
  ──── never reaches ────
  → triton_attn.py:616 fix #5 (per-layer graceful skip)
```

### What Fix #5 Does vs What's Needed

Fix #5 (already committed to fork d9a9157) handles 256/512 mismatch at the **attention forward pass** level — falls back to bf16 KV for 512 layers. But the crash happens **earlier** during KV cache initialization, before any forward pass runs.

## Candidate Solutions (Researched)

### Option A: LCM-Based Unification
Modify `unify_kv_cache_spec_page_size()` to use LCM when divisibility fails.

```python
# In unify_kv_cache_spec_page_size, replace:
if max_page_size % layer_page_size != 0:
    raise NotImplementedError(...)
# With:
if max_page_size % layer_page_size != 0:
    lcm = math.lcm(max_page_size, layer_page_size)
    max_page_size = lcm  # Update the target
    # Rescale ALL layers' block_sizes to match LCM
```

- LCM = 890,880 bytes per block
- block_size for 256 layers: 464 tokens, for 512 layers: 240 tokens
- **Pros:** Minimal code change (only `unify_kv_cache_spec_page_size`)
- **Cons:** Large blocks waste 72% memory on short sequences. Bad for 4GB VRAM.
- **Qwen3 impact:** None (Qwen3 has uniform pages, skips unification entirely)
- **Other hardware:** Wasteful but functional. Better GPUs tolerate the waste.

### Option B: Pre-Split KV Cache Dtype at Spec Level
Assign `kv_cache_dtype='auto'` to layers whose head_dim doesn't match TurboQuant metadata, at the KV cache spec creation level (not just at forward time).

- 28 layers: turboquant35, page=30,720
- 7 layers: bf16, page=262,144
- `262,144 / 30,720 = 8.533` — STILL not divisible
- **Verdict:** Doesn't solve the math alone. Combine with Option A as fallback.

### Option C: Option B + A Hybrid
Pre-split dtypes (B), then LCM fallback (A) for remaining incompatibility.
- LCM(30720, 262144) = 3,932,160 (3.75 MB per block!)
- **Verdict:** WORSE than Option A alone. Not viable.

### Option D (Recommended): Padded Group Allocation
Modify the KV cache grouping to support groups with different page sizes by padding smaller allocations to match the largest:

1. Group layers by compatible page sizes (layers with page=30,720 in one group, page=59,392 in another)
2. Allocate all blocks at the max page size (59,392)
3. Smaller-page layers use only the first 30,720 bytes of each block, rest is padding

- Memory overhead: 28 layers × (59392-30720) / 59392 = 48% padding for 256 layers
- But each block is only 59,392 bytes (58 KB), not 890 KB like LCM
- **Pros:** Reasonable block sizes (16 tokens), no LCM explosion, works for all hardware
- **Cons:** ~48% memory waste in the 256-layer group. Requires changes to `_get_kv_cache_groups_uniform_page_size` and possibly KVCacheManager allocation.
- **Qwen3 impact:** None (uniform pages, no padding needed)

### Recommendation

**Start with Option A (LCM)** for correctness and simplicity. It's a ~10-line change to one function. The large block sizes are a concern for 4GB VRAM but acceptable for initial functionality. If benchmarks show unacceptable memory waste, follow up with Option D.

The LCM approach is also the most portable — it works for ANY future model with variable head dimensions, not just Gemma 4.

## Pre-Implementation Checklist

1. [ ] **Commit fix #5 to fork** — DONE (d9a9157)
2. [ ] **Test Qwen3 + turboquant35 baseline** — Confirm Qwen3 still works with fix #5 (no regression)
3. [ ] **Implement Option A** in `kv_cache_utils.py:914-951`
4. [ ] **Test Gemma 4 E2B + turboquant35** — Should pass with LCM
5. [ ] **Test Qwen3 + turboquant35** — Confirm no regression (uniform pages skip LCM)
6. [ ] **Benchmark memory waste** — Measure actual block allocation with LCM
7. [ ] If waste is unacceptable, implement Option D

## Skills Needed

- **`/tq-benchmark`** — Measure memory usage and tok/s with LCM block sizes
- **`/tq-system-info`** — Verify hardware compatibility across different GPUs
- **Gemini MCP (`consult_gemini`)** — Research TurboQuant packed_dim alignment constraints, verify LCM doesn't violate TurboQuant's group alignment (16-dim)
- **`/architecture-doc-review`** — Update CLAUDE.md and architecture docs
- **`/issue-manager`** — Update #22 with implementation progress
- **`/technical-planner`** — If Option D is needed, plan the KVCacheManager changes

## Key Files

| File | Role |
|------|------|
| `vllm/v1/core/kv_cache_utils.py:914-951` | **THE FUNCTION TO FIX** — `unify_kv_cache_spec_page_size()` |
| `vllm/v1/core/kv_cache_utils.py:1245-1264` | `get_kv_cache_groups()` — caller |
| `vllm/v1/core/kv_cache_utils.py:959-1018` | `_get_kv_cache_groups_uniform_page_size()` — downstream grouping |
| `vllm/v1/attention/ops/turboquant_kv_cache.py:376` | `get_turboquant_packed_dim()` — packed_dim computation |
| `vllm/v1/kv_cache_interface.py:97-107` | `page_size_bytes` property |
| `vllm/v1/attention/backends/triton_attn.py:616-629` | Fix #5 (committed d9a9157) |

## Gemini MCP Setup

The tqCLI Gemini MCP server has been reconfigured for TurboQuant research. Config at `.claude/mcp-servers/gemini-mcp/config.json`. Use `consult_gemini` tool for:
- TurboQuant packed_dim alignment verification
- KV cache page size strategies from vLLM community
- Variable head_dim model support in other inference engines

The ARGOS_POC_LOCAL Gemini MCP uses service account `argos-proof-of-concept-6da2ec3a22f0.json`. The tqCLI MCP server should use the same auth pattern but configured for the tqCLI project context.

## GoogleTurboQuant Reference

- `../GoogleTurboQuant/docs/research/turboquant_research.md` — Core algorithm (PolarQuant + QJL two-stage)
- `../GoogleTurboQuant/docs/architecture/system_overview.md` — System design
- Key constraint: group alignment = 16 dimensions. LCM-based block_size scaling doesn't affect dimension alignment (it scales tokens per block, not dimensions).

## What NOT To Touch

- Qwen3 AWQ pipeline — unaffected (uniform head_dim)
- llama.cpp engine — separate TurboQuant implementation
- BNB + CPU offload fixes (#8, #9, #10) — orthogonal, already working
- TurboQuant metadata format — `turboquant_kv.json` is correct for 256 layers

## Expected Final State

- Gemma 4 E2B runs with `kv_cache_dtype='turboquant35'` on 28/35 layers
- 7 full attention layers (512) either use LCM-scaled blocks or fall back to bf16
- Qwen3 AWQ + turboquant35 continues to work (no regression)
- Memory usage is acceptable for 4GB VRAM with CPU offloading
- Fork changes committed and pushed to `ithllc/vllm-turboquant`
