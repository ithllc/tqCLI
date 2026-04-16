# Resume Prompt: TurboQuant KV Page Size Fix for Variable head_dim Models

**Issue:** ithllc/tqCLI#22
**Date Created:** 2026-04-16
**Status:** Research complete, implementation ready
**Depends on:** Fix #5 (triton_attn.py graceful skip, fork commit d9a9157)

---

## Context

TurboQuant KV cache (`kv_cache_dtype='turboquant35'`) crashes during KV cache initialization for models with mixed head dimensions. Gemma 4 has head_dim=256 (28 sliding layers) and head_dim=512 (7 full attention layers), producing page sizes of 30,720 and 59,392 bytes. The `unify_kv_cache_spec_page_size()` function requires `max % min == 0`, which fails (ratio = 1.933).

Qwen3 (uniform head_dim=128) is unaffected — it skips unification entirely.

## Solution: Use vLLM's `page_size_padded` Field

vLLM already has infrastructure for this exact problem. The `AttentionSpec` dataclass has a `page_size_padded` field (line 86 of `kv_cache_interface.py`) that separates allocation size from data size:

```python
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    page_size_padded: int | None = None  # ← Padding support

    @property
    def page_size_bytes(self) -> int:
        real_page_size = self.real_page_size_bytes
        if self.page_size_padded is not None:
            assert self.page_size_padded >= real_page_size
            return self.page_size_padded  # ← Returns padded for allocation
        return real_page_size

    @property
    def real_page_size_bytes(self) -> int:
        # Returns actual data size (used by attention kernels)
```

This pattern is already used by Mamba hybrid models (`mamba_page_size_padded` in `cache.py:103`). The attention kernels access data via `packed_dim` stride, NOT `page_size_bytes`, so padding doesn't affect kernel correctness.

### Why This Is The Correct Solution (Not LCM)

**LCM approach (rejected):**
- LCM(30720, 59392) = 890,880 bytes per block per layer
- block_size for 256 layers: 464 tokens, for 512 layers: 240 tokens
- On 4GB VRAM with 64 MiB KV cache: only 2 blocks fit, wasting 72% per block
- Defeats tqCLI's purpose of KV cache compaction on constrained hardware

**Padding approach (correct):**
- Allocate all blocks at max_page_size (59,392 bytes per layer)
- 256 layers use 30,720 bytes, padding 28,672 bytes unused
- block_size stays at 16 tokens (fine granularity)
- On 4GB VRAM with 64 MiB KV cache: ~32 blocks fit = 512 tokens context
- 38% allocation waste (vs 72% for LCM), and finer granularity

**Qwen3 safety:** Qwen3 has uniform pages → `len(page_sizes) <= 1` → returns early at line 930. The padding code path is never entered.

## Sample Implementation

### Change to `vllm/v1/core/kv_cache_utils.py`

Replace the `NotImplementedError` block (lines 941-945) with padding:

```python
def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. If the page size of all layers
    are the same, return the original KVCacheSpec. If not same, unify the page
    size by increasing the block size of layers with smaller page size, or by
    padding smaller pages to match the maximum when block_size scaling is not
    possible (e.g., TurboQuant with variable head_dim).
    """
    page_sizes = {layer.page_size_bytes for layer in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    new_kv_cache_spec = {}
    for layer_name, layer_spec in kv_cache_spec.items():
        if layer_spec.page_size_bytes == max_page_size:
            new_kv_cache_spec[layer_name] = layer_spec
        else:
            layer_page_size = layer_spec.page_size_bytes
            if max_page_size % layer_page_size == 0:
                # Clean integer ratio — scale block_size (existing behavior)
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            else:
                # Non-integer ratio (e.g., TurboQuant with mixed head_dim).
                # Pad smaller pages to match the max page size. The extra
                # bytes are never accessed by attention kernels (they use
                # real_page_size_bytes / packed_dim for data access).
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec
```

### Changes Required

| File | Change |
|------|--------|
| `vllm/v1/core/kv_cache_utils.py:941-945` | Replace `raise NotImplementedError` with `replace(layer_spec, page_size_padded=max_page_size)` |

That's it. One file, ~5 lines changed. The `page_size_padded` infrastructure handles everything else.

### Why Only One File

- `page_size_padded` on `AttentionSpec` already exists (kv_cache_interface.py:86)
- `page_size_bytes` property already returns padded value (kv_cache_interface.py:88-94)
- Block allocation uses `page_size_bytes` (padded) → correct allocation size
- Attention kernels use `packed_dim` stride → never touch padding bytes
- `_get_kv_cache_groups_uniform_page_size` downstream works because all `page_size_bytes` are now equal

## Pre-Implementation Checklist

1. [ ] **Verify Qwen3 + turboquant35 baseline** — Confirm it passes (uniform pages, skips this code path)
2. [ ] **Apply the fix** to `kv_cache_utils.py` in site-packages
3. [ ] **Test Gemma 4 E2B + turboquant35 + BNB + CPU offload** — Should pass
4. [ ] **Test Qwen3 + turboquant35** — Confirm no regression
5. [ ] **Commit to fork** (`ithllc/vllm-turboquant`)
6. [ ] **Benchmark** memory usage and tok/s with the fix
7. [ ] **Close or update issue #22**

## Skills Needed

- **Gemini MCP (`consult_gemini`)** — Verify TurboQuant packed_dim alignment with padded allocation
- **`/tq-benchmark`** — Performance comparison with and without TurboQuant KV
- **`/issue-manager`** — Update #22 with results

## Key Files

| File | Role |
|------|------|
| `vllm/v1/core/kv_cache_utils.py:914-951` | **THE FUNCTION TO FIX** |
| `vllm/v1/kv_cache_interface.py:86-94` | `page_size_padded` infrastructure (existing, no changes needed) |
| `vllm/v1/attention/backends/triton_attn.py:398-401` | KV cache shape uses `packed_dim` (not page_size) — confirms padding safety |
| `vllm/v1/attention/backends/triton_attn.py:616-629` | Fix #5 graceful skip (already committed d9a9157) |

## Fork State

- **Repo:** `ithllc/vllm-turboquant` (commit d9a9157)
- **Installed version:** `0.1.dev6+gb236390bf` (with runtime patches in site-packages)
- **Transformers:** 5.5.4

## Expected Final State

- Gemma 4 E2B runs with `kv_cache_dtype='turboquant35'` + BNB_INT4 + cpu_offload
- 28 sliding layers (head_dim=256): TurboQuant KV at 4.6x compression
- 7 full attention layers (head_dim=512): TurboQuant KV with padded allocation
- Fix #5 (triton_attn.py) provides per-layer fallback if metadata doesn't match
- Qwen3 AWQ + turboquant35 continues to work (uniform pages, no padding triggered)
- Reduced CPU offloading → better than 0.2 tok/s performance
