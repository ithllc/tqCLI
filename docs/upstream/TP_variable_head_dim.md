# TP: Variable head_dim support in vllm-turboquant

Companion to `PRD_variable_head_dim.md`. Fork-side implementation only.

## Schema v2

```jsonc
{
  "version": 2,
  "recipe": "turboquant35",
  "head_size": 256,                      // kept as DEFAULT for legacy readers
  "model_name": "gemma-4-E2B-it",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "head_size": 256,                  // NEW (optional). If absent, inherit top-level.
      "key_high_precision_indices":   [[...]],
      "value_high_precision_indices": [[...]]
    },
    "model.layers.4.self_attn": {
      "head_size": 512,                  // example global-attention layer
      "key_high_precision_indices":   [[...]],
      "value_high_precision_indices": [[...]]
    }
  }
}
```

### Back-compat rules
- Readers MUST accept `version: 1` (top-level `head_size` only) unchanged.
- Readers parsing `version: 2` MUST resolve a layer's effective head_size as: `layer.head_size or metadata.head_size`.
- Writers SHOULD emit `version: 2` only when at least one layer's `head_size` differs from the top-level field.

### Inner-list length
For layers with `head_size != top_level_head_size`, the per-layer `key_high_precision_indices` inner-list length MUST equal `round(layer_head_size × outlier_ratio / 16) × 16` (NOT the top-level value).

## Implementation

### File 1: `vllm/v1/attention/ops/turboquant_metadata.py`

```python
# Add to TurboQuantLayerMetadata
@dataclass(frozen=True)
class TurboQuantLayerMetadata:
    key_high_precision_indices: tuple[tuple[int, ...], ...]
    value_high_precision_indices: tuple[tuple[int, ...], ...]
    head_size: int | None = None          # NEW. None => inherit from parent.

def load_turboquant_metadata(path: Path) -> TurboQuantMetadata:
    raw = json.loads(path.read_text())
    version = raw.get("version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported turboquant metadata version: {version}")
    top_head_size = int(raw["head_size"])
    layers: dict[str, TurboQuantLayerMetadata] = {}
    for name, body in raw["layers"].items():
        layer_head_size = body.get("head_size")  # v2 optional
        effective = layer_head_size if layer_head_size is not None else top_head_size
        _validate_indices(body, effective, raw["recipe"])
        layers[name] = TurboQuantLayerMetadata(
            key_high_precision_indices=_tuplify(body["key_high_precision_indices"]),
            value_high_precision_indices=_tuplify(body["value_high_precision_indices"]),
            head_size=layer_head_size,
        )
    return TurboQuantMetadata(
        version=version,
        recipe=raw["recipe"],
        head_size=top_head_size,
        transform_version=raw["transform_version"],
        codebook_version=raw["codebook_version"],
        layers=layers,
        ...
    )
```

`_validate_indices(body, effective_head_size, recipe, num_kv_heads)` MUST check:
- **Outer list length** equals `num_kv_heads` (prevents out-of-bounds access in the Triton backend).
- **Inner list length** matches `round(effective_head_size × ratio / 16) × 16`.

### File 2: `vllm/v1/attention/backends/triton_attn.py`

Current lines 616-629 (paraphrased) compare `layer.head_size` to `metadata.head_size`. Replace with:

```python
layer_meta = metadata.layers.get(layer_name)
effective_head_size = (
    layer_meta.head_size if layer_meta and layer_meta.head_size is not None
    else metadata.head_size
)
if layer.head_size != effective_head_size:
    # existing bf16-fallback path stays as-is for unknown dims
    logger.warning(
        f"Layer {layer_name} head_size {layer.head_size} has no matching "
        f"metadata entry ({effective_head_size}); falling back to bf16 KV."
    )
    return _bf16_fallback_impl(...)

# Use layer-specific indices with layer-specific head_size
self._setup_turboquant(layer_meta, effective_head_size)
```

### File 3: Hadamard rotation-matrix cache

Verify in `vllm/v1/attention/ops/hadamard.py` (or equivalent) that the cache key is `(device, dim, seed_offset)`. If it's currently `(device, seed_offset)` only, fix. The cache grows O(n_distinct_head_dims) — bounded small.

### File 4: `examples/turboquant_kv_gemma4_v2.json`

Ship a sample Gemma 4 E2B v2 file with 28 head_dim=256 layers + 7 head_dim=512 layers. Use known-good calibrated indices (heuristic acceptable; mark as "example only, not calibrated for production").

## Testing

### Unit tests (fork)
1. `test_metadata_v1_roundtrip` — existing Qwen 3 4B metadata loads identically under the new reader.
2. `test_metadata_v2_mixed_head_dim` — load the new Gemma 4 sample, assert per-layer head_size is respected.
3. `test_metadata_v2_defaults_to_top` — omitting per-layer head_size in a v2 file inherits the top-level value.
4. `test_metadata_v2_rejects_bad_indices_length` — per-layer inner-list length mismatches raise `ValueError`.

### Integration test (fork)
Load `google/gemma-4-e2b-it` with the v2 sample on a CUDA 12.8 + SM ≥ 8.6 box:

```python
from vllm import LLM
llm = LLM(
    model="google/gemma-4-e2b-it",
    quantization="bitsandbytes",
    kv_cache_dtype="turboquant35",
    turboquant_metadata_path="examples/turboquant_kv_gemma4_v2.json",
    cpu_offload_gb=9.9,
    max_model_len=4096,
)
# Assert every attention layer reports kv_cache_dtype == 'turboquant35'
for layer in llm.llm_engine.model_executor.model.model.layers:
    assert layer.self_attn.kv_cache_dtype == "turboquant35", layer
```

### PPL gate
Run the fork's existing 100-prompt sanity set twice: once with v1 partial metadata, once with v2 full metadata. v2 PPL must be ≤ 1.05× v1 PPL (usually better — more compression, less quality loss).

## Risk register

| Risk | Mitigation |
|------|-----------|
| Rotation-matrix cache thrashing with >1 head_dim | Verify cache key; likely already correct. |
| Kernel launch overhead from heterogeneous layer sizes | Negligible — kernel launch is already per-layer. |
| v2 files accidentally loaded by old fork installs | Version field is read first; old loader raises clean error, users see actionable message. |
| Sample JSON's indices are miscalibrated | Clearly label `"calibration": "heuristic-for-example-only"` in the file. |

## Out of scope (re-stated)

- tqCLI-side changes. Those land after this upstream ships.
- Changes to the Triton kernels.
- Changes to the Lloyd-Max codebook.
- Multi-node cache synchronization.
