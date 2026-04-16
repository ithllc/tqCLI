# Resume Prompt: TurboQuant KV Page Size Fix for Variable head_dim Models

**Issue:** ithllc/tqCLI#22
**Date Created:** 2026-04-16
**Status:** Deep research complete. Two-file fix identified. Ready for implementation.
**Depends on:** Fix #5 (triton_attn.py graceful skip, fork commit d9a9157)

---

## Context

TurboQuant KV cache (`kv_cache_dtype='turboquant35'`) crashes during KV cache initialization for models with mixed head dimensions. Gemma 4 has head_dim=256 (28 sliding layers) and head_dim=512 (7 full attention layers), producing page sizes of 30,720 and 59,392 bytes.

Qwen3 (uniform head_dim=128) is unaffected — skips unification entirely.

## Solution: `page_size_padded` + Reshape Slice

vLLM's `AttentionSpec` has a `page_size_padded` field designed for non-uniform page sizes. However, using it alone creates a second problem: the KV cache reshape step fails because the padded tensor is larger than the shape expects.

### The Two Problems

**Problem 1 — Page size unification (kv_cache_utils.py:942):**
```
59,392 % 30,720 = 28,672  →  NotImplementedError
```
Fix: Set `page_size_padded=max_page_size` on smaller layers.

**Problem 2 — KV cache reshape (gpu_model_runner.py:6614-6616):**
```python
kv_cache_raw_tensors[layer_name]  # int8, num_blocks × 59,392 (padded)
    .view(dtype)                   # uint8, num_blocks × 59,392 elements
    .view(kv_cache_shape)          # shape product = num_blocks × 30,720
                                   # 59,392 ≠ 30,720 → RuntimeError!
```
Fix: Slice the padded tensor to `real_page_size_bytes` before reshaping.

### Important: Mamba Precedent Note

The `page_size_padded` field exists on `AttentionSpec` and is used in production for **Mamba-to-attention padding** in hybrid models (Jamba). However, it has NOT been used for **attention-to-attention padding** in production. This fix is the first such use.

The mechanism is identical (same field, same allocation path), but the **reshape step** is different: Mamba layers have their own reshape path (`MambaSpec` branch at line 6619-6639) that uses `torch.as_strided` with explicit storage offsets. Attention layers use `.view(dtype).view(shape)` which requires exact size matching.

This means the reshape needs a small modification for padded attention layers — the Mamba workaround doesn't apply here.

### TurboQuant dtype: `torch.uint8`

Key finding: `STR_DTYPE_TO_TORCH_DTYPE["turboquant35"] = torch.uint8` (torch_utils.py:45). So:
- `AttentionSpec.dtype = torch.uint8` for TurboQuant layers
- `.view(uint8)` on int8 tensor = reinterpret cast (same element count)
- Shape dimensions are in BYTES, not bf16 elements
- `packed_dim` (64, 120, 232) represents bytes per KV per head

## Sample Implementation

### File 1: `vllm/v1/core/kv_cache_utils.py` (lines 941-945)

Replace `raise NotImplementedError` with padding:

```python
def unify_kv_cache_spec_page_size(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """
    Unify the page size of the given KVCacheSpec. Uses block_size scaling
    when page sizes are evenly divisible, or page_size_padded when they
    are not (e.g., TurboQuant with variable head_dim).
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
                # Pad to max page size. The attention kernel accesses data
                # via packed_dim stride, so the extra bytes are never read.
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
    return new_kv_cache_spec
```

### File 2: `vllm/v1/worker/gpu_model_runner.py` (~line 6577-6618)

In `_reshape_kv_cache_tensors`, after computing `num_blocks`, slice padded tensors to their real data size before reshaping:

```python
for layer_name in group.layer_names:
    if layer_name in self.runner_only_attn_layers:
        continue
    raw_tensor = kv_cache_raw_tensors[layer_name]
    assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
    num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
    if isinstance(kv_cache_spec, AttentionSpec):
        has_attn = True

        # If page_size is padded, slice raw tensor to real data size
        # before reshaping. The padding bytes are allocated but never
        # accessed by attention kernels.
        real_page_size = kv_cache_spec.real_page_size_bytes
        if (hasattr(kv_cache_spec, 'page_size_padded')
                and kv_cache_spec.page_size_padded is not None
                and kv_cache_spec.page_size_padded > real_page_size):
            # Reshape: [total_padded_bytes] → [num_blocks, padded_page]
            # then slice → [num_blocks, real_page] → flatten
            raw_tensor = (raw_tensor
                .view(num_blocks, kv_cache_spec.page_size_bytes)
                [:, :real_page_size]
                .contiguous()
                .view(-1))

        num_blocks_per_kv_block = (
            kv_cache_spec.block_size // kernel_block_size
        )
        # ... rest of existing reshape code unchanged ...
```

### Why This Two-File Approach Works

1. **Allocation** (`kv_cache_utils.py`): All layers report `page_size_bytes=59,392`. The block allocator sees uniform sizes. ✓
2. **Reshape** (`gpu_model_runner.py`): Padded layers get sliced to `real_page_size_bytes=30,720` before `.view(kv_cache_shape)`. Shape product matches. ✓
3. **Kernel access** (`triton_attn.py`): Attention kernels use `packed_dim` stride on uint8 data. Never touch padding. ✓
4. **Grouping** (`_get_kv_cache_groups_uniform_page_size`): Layers group by exact spec equality. 256-head layers (with padding) and 512-head layers (without) stay in separate groups. Merge within each group succeeds. ✓
5. **Qwen3**: Uniform pages → `len(page_sizes) <= 1` → returns early. Neither file's new code executes. ✓

### Memory Impact

For 64 MiB KV cache with Gemma 4 E2B:
- Block size: 59,392 bytes per layer per block (16 tokens)
- 35 layers × 59,392 = ~2.08 MiB per block
- ~30 blocks = 480 tokens context
- Padding waste: 28 layers × 28,672 bytes = ~784 KB per block (38%)
- Effective TurboQuant compression on 28/35 layers: ~3.3x average across all layers

## Pre-Implementation Checklist

1. [ ] **Qwen3 + turboquant35 baseline** — Confirm it passes before any changes
2. [ ] **Apply File 1 fix** (kv_cache_utils.py)
3. [ ] **Apply File 2 fix** (gpu_model_runner.py)
4. [ ] **Test Gemma 4 E2B + turboquant35 + BNB + CPU offload**
5. [ ] **Test Qwen3 + turboquant35** — Confirm no regression
6. [ ] **Commit to fork** (`ithllc/vllm-turboquant`)
7. [ ] **Benchmark** memory usage and tok/s
8. [ ] **Update issue #22**

## Skills Needed

- **Gemini MCP (`consult_gemini`)** — Verify TurboQuant packed_dim alignment with padded+sliced allocation
- **`/tq-benchmark`** — Measure tok/s improvement with TurboQuant KV vs bf16 KV
- **`/issue-manager`** — Update #22 with results

## Key Files

| File | Change | Lines |
|------|--------|-------|
| `vllm/v1/core/kv_cache_utils.py` | Replace NotImplementedError with page_size_padded | 941-945 |
| `vllm/v1/worker/gpu_model_runner.py` | Slice padded tensors before reshape | ~6577-6618 |
| `vllm/v1/kv_cache_interface.py` | No change (page_size_padded already exists) | 86-94 |
| `vllm/v1/attention/backends/triton_attn.py` | No change (fix #5 already committed) | 616-629 |
| `vllm/utils/torch_utils.py` | No change (turboquant35→uint8 mapping exists) | 44-45 |

## GoogleTurboQuant Compliance

Per `../GoogleTurboQuant/docs/research/turboquant_research.md`:
- TurboQuant operates on fixed-dimension groups with 16-dim alignment
- Padding occurs at the page/block level, not the dimension level
- The `packed_dim` computation is per-layer based on `head_size` — padding doesn't change it
- The slice operation preserves the exact `packed_dim` bytes the kernel expects
- No conflict with TurboQuant's PolarQuant + QJL two-stage compression

## Fork State

- **Repo:** `ithllc/vllm-turboquant` (commit d9a9157)
- **Installed:** `0.1.dev6+gb236390bf` (with runtime patches in site-packages)
- **Transformers:** 5.5.4
