# PRD: Variable head_dim support in vllm-turboquant

**Target repo:** `github.com/ithllc/vllm-turboquant`
**Authored by:** tqCLI project (downstream consumer)
**Closes (downstream):** tqCLI #32
**Opens (upstream):** issue to be filed on `ithllc/vllm-turboquant` linking to this doc + TP
**Status:** proposal — awaits fork-maintainer review

## 1. Problem

Gemma 4 (and any transformer with heterogeneous attention-layer `head_dim`) cannot enjoy full-model TurboQuant KV compression under the current fork. Gemma 4 E2B has:

- 28 sliding-window layers at `head_dim = 256`
- 7 global-attention layers at `head_dim = 512`

The fork's metadata schema carries a single top-level `head_size` field. Any layer whose runtime `head_size` doesn't match is handled by the mismatch-warn path at `vllm/v1/attention/backends/triton_attn.py:616-629`, which silently falls back to bf16 KV for that layer. The user-visible effect: Gemma 4 E2B runs with only 28/35 attention layers compressed — 80% coverage, not 100%.

The fork's runtime ALREADY handles the mismatch gracefully (no crash, correct output). What's missing is a way for the metadata file AND the runtime to declare compression for multiple head_dims in one model.

## 2. Why this matters now

- **Memory budget on low-VRAM devices.** On a 4 GB VRAM box, the 7 bf16 layers hold disproportionate KV pressure relative to the 28 compressed layers. Users optimizing for fit-in-VRAM want every layer compressed.
- **Compound architectures are becoming the norm.** Gemma 2/3/4, many mixture-of-experts variants, and research architectures freely mix head dims. The single-head_size assumption is a growing liability.
- **tqCLI v0.6.1 just shipped auto-calibration for Qwen 3 (single head_dim).** The calibration path is tested and ready. The only thing preventing tqCLI from shipping Gemma 4 auto-calibration is the fork's single-head_size schema — not the calibration math.

## 3. Scope

### In scope
- Extend the `turboquant_kv.json` schema to declare per-layer `head_size` (schema version bump to `v2`).
- Update `vllm/v1/attention/ops/turboquant_metadata.py` loader to parse both v1 (single `head_size`) and v2 (per-layer override) files.
- Update `vllm/v1/attention/backends/triton_attn.py` to resolve a layer's `head_size` from the metadata's per-layer entry when present, instead of comparing to the top-level field.
- Extend the Hadamard rotation matrix cache to key on (device, dim, seed_offset) — already correctly keyed; no behavior change, just verify that multiple dims coexist without eviction thrashing.
- Add a v2 sample file for Gemma 4 E2B to the fork's `examples/` directory.

### Out of scope
- Changes to the Triton kernels themselves (they already parametrize on `head_size`).
- New codebook versions.
- Changes to the outlier selection policy.
- Cross-GPU cache sharing.

## 4. Acceptance criteria

1. A v2 `turboquant_kv.json` with per-layer `head_size` entries loads without error via `load_turboquant_metadata()`.
2. v1 (legacy, single `head_size`) files continue to load unchanged — no regression on Qwen 3 4B, Gemma 4 sliding-only metadata, or any shipped Gemma 4 E2B deployment.
3. Gemma 4 E2B loaded with a v2 file compresses ALL 35 attention layers (28 at head_dim=256, 7 at head_dim=512), verified by reading the per-layer `kv_cache_dtype` after loading the model.
4. Perplexity on a 100-prompt sanity set is within 1.05× of v1 partial-coverage baseline (should be better, since more layers are now compressed).
5. Rotation-matrix cache statistics show both `dim=256` and `dim=512` matrices coexist without repeated allocation across the inference loop.

## 5. Non-goals for this spec

This PRD does not design the tqCLI calibrator changes required to emit v2 files. That's a tqCLI-side follow-up issue, landed only after this upstream spec ships. tqCLI's current `check_calibration_preconditions` hard-refuses variable-head_dim models precisely because the schema can't express them — that guardrail stays in place until this upstream work lands.

## 6. Rollout

1. Fork maintainer reviews + lands this PRD.
2. Fork maintainer implements per the companion TP (`TP_variable_head_dim.md`).
3. Fork cuts a new tagged release.
4. tqCLI opens a new issue: "enable Gemma 4 auto-calibration via v2 metadata" — separate from #32.
5. tqCLI's `check_calibration_preconditions` is updated to allow variable-head_dim AND to emit v2 files when the installed fork version supports the v2 schema.
6. tqCLI #32 is closed as superseded by the upstream delivery.
