# Fix: Qwen 3 4B on vLLM + TurboQuant KV Compression

**Status:** open
**Scope:** tqCLI 0.6.0+
**Blocks:** the Qwen 3 4B line of the agent-modes functional integration suite (T1_vq, T4_vq) from asserting TurboQuant KV active. Currently asserts agent orchestrator correctness only, with `kv:none` fallback documented inline in `tests/integration_reports/agent_modes_functional_report.md`.

---

## 1. What actually happened

During the 0.6.0 functional integration suite (`tests/test_integration_agent_functional.py`), the vLLM + Qwen 3 4B configuration failed to load with TurboQuant KV compression enabled (`kv_quant_choice="turbo3"` → `kv_cache_dtype="turboquant35"`). The raw exception:

```
ValueError: TurboQuant KV cache requires metadata. Pass
`turboquant_metadata_path` or place `turboquant_kv.json` under the
local model path.
```

Raised at `vllm/v1/attention/backends/triton_attn.py:600` inside `TritonAttentionImpl.__init__` when `kv_cache_dtype.startswith("turboquant")` and `discover_turboquant_metadata_path()` returns `None`.

**Why it returned None:** `~/.tqcli/models/qwen3-4b-vllm/` does NOT contain a `turboquant_kv.json` file. The Gemma 4 E2B vLLM snapshot (`~/.tqcli/models/gemma-4-e2b-it-vllm/turboquant_kv.json`) DOES ship one. The TurboQuant-vLLM fork (`github.com/ithllc/vllm-turboquant`) refuses to enable turboquant35 without per-model calibrated metadata because the runtime cannot infer the correct "high precision channel" selection without it.

## 2. Why the suite currently falls back to `kv:none`

To land the 0.6.0 release with a clean 11/11 functional pass, `tests/test_integration_agent_functional.py` passes `kv_quant_choice="none"` specifically for the Qwen 3 4B vLLM load path:

```python
# tests/test_integration_agent_functional.py (run_vllm_group)
eng_q, tune_q = load_vllm_engine(QWEN_VLLM, qwen_profile, kv_quant_choice="none", max_len=896)
```

The `max_len=896` is unrelated: it's a VRAM-budget tune because Qwen 3 4B (32 attention heads, 8 KV heads, 36 layers) uses more residual VRAM than Gemma 4 E2B under BNB_INT4 + CPU offload on a 4 GB GPU. vLLM's own error message suggested that value.

The `kv_quant_choice="none"` is a **temporary** workaround. It exercises the agent orchestrator's parse→execute→observation→live-inference round trip on real Qwen 3 4B inference, but without TurboQuant KV compression active on that combination. The report explicitly labels this combination `kv:none` and cites this prompt as the tracked fix.

## 3. What `turboquant_kv.json` actually contains

Schema is defined in `/usr/local/lib/python3.11/site-packages/vllm/v1/attention/ops/turboquant_metadata.py`. Representative Gemma 4 E2B file (9,320 lines):

```json
{
  "version": 1,
  "recipe": "turboquant35",
  "head_size": 256,
  "model_name": "gemma-4-E2B-it",
  "transform_version": "structured_hadamard_v1",
  "codebook_version": "lloyd_beta_v1",
  "layers": {
    "model.layers.0.self_attn": {
      "key_high_precision_indices":   [[0, 1, 2, ..., 127]],
      "value_high_precision_indices": [[0, 1, 2, ..., 127]]
    },
    "model.layers.1.self_attn": { ... },
    ...
    "model.layers.34.self_attn": { ... }
  }
}
```

Field semantics:

| Field | Meaning |
|---|---|
| `version` | `1` (only supported version) |
| `recipe` | `"turboquant25"` or `"turboquant35"`. Outlier ratio 0.25 vs 0.50. |
| `head_size` | Attention head dimension. Must be `% 16 == 0`. |
| `model_name` | Freeform. Informational. |
| `transform_version` | `"structured_hadamard_v1"`. The Hadamard rotation recipe applied inside vLLM before the KV cache layer. Metadata and transform are orthogonal in the file, but the indices only make sense in post-rotation coordinates. |
| `codebook_version` | `"lloyd_beta_v1"`. Tag for the fork-baked Lloyd-Max scalar codebook compiled into the Triton kernels. Fixed; never regenerated per-model. Just declare it. |
| `layers` | Object keyed by attention-layer name, e.g. `"model.layers.{i}.self_attn"`. Must cover every attention layer the loader looks up at load time. |
| `layers.<name>.key_high_precision_indices` | **Outer list length = `num_kv_heads`** (NOT `num_attention_heads`). For Qwen 3 4B with GQA, that's 8, not 32. Each inner list is `outlier_count` channel indices in `[0, head_size)`. `outlier_count = round(head_size × ratio / 16) × 16`. |
| `layers.<name>.value_high_precision_indices` | Same shape. Independent channel selection for V. |

For Qwen 3 4B: `head_size=128`, `num_kv_heads=8`, `num_hidden_layers=36`, `turboquant35` outlier ratio 0.50 → `outlier_count = round(128 × 0.50 / 16) × 16 = 64`. So each inner list has 64 integers in `[0, 128)`, the outer list has 8 entries, and there are 36 layers.

## 4. Why `range(outlier_count)` is NOT a safe default

The vLLM fork exposes a `build_default_turboquant_metadata()` helper that emits `tuple(range(outlier_count))` as the high-precision indices for every head. **Do not ship that as a production fix.** Gemini review 2026-04-18:

> Tier 1 ignores the mathematical reality of the Hadamard-transformed space. You are essentially trying to perform heart surgery with a blindfold on, hoping the "important parts" are in the first few inches of the chest. They aren't.

The Hadamard rotation decorrelates channels but does NOT concentrate activation magnitude at low-indexed channels. It only makes outliers learnable via calibration. Default metadata → silent quality collapse (coherent-looking gibberish, repetition loops, rare-unicode token storms). It is not a detectable failure; it is a slow-burn regression across real prompts. Do not ship it.

## 5. Fix path

There are two real options — a pragmatic one-shot heuristic and a proper calibration. Either produces a shippable `turboquant_kv.json`. Pick one; do not ship the default-range version.

### Option A — Weight-magnitude heuristic (fast, usable) — **recommended for 0.6.1**

Per-layer, per-KV-head, pick the top-`outlier_count` channels by L2 norm of the corresponding K / V projection weight rows AFTER applying the same Hadamard rotation the fork applies at runtime. This is ~5 minutes of scripting and produces materially better indices than `range()` because the channels with the largest weight magnitude are more likely to carry the largest activations — not guaranteed, but a defensible heuristic when activation tracing isn't available.

Implementation plan — new module `tqcli/core/kv_metadata_generator.py`:

1. Load the model snapshot's safetensors files using `safetensors.safe_open`.
2. For each attention layer `i` in `range(num_hidden_layers)`, read:
   - `model.layers.{i}.self_attn.k_proj.weight` — shape `[num_kv_heads × head_size, hidden_size]`.
   - `model.layers.{i}.self_attn.v_proj.weight` — same shape.
3. Reshape K weight to `[num_kv_heads, head_size, hidden_size]`. Same for V.
4. Apply a structured Hadamard along the `head_size` axis. The fork uses a "structured_hadamard_v1" transform — inspect `vllm.v1.attention.ops` for the exact primitive, or use `scipy.linalg.hadamard(head_size)` for a standard Hadamard (tqCLI must match the fork's rotation, verify by diff on Gemma's metadata vs. Gemma's weights before claiming parity).
5. Per head, compute row-wise L2 norm along the `hidden_size` axis → one scalar per `head_size` channel.
6. Select the indices of the top-`outlier_count` channels, sorted ascending.
7. Emit JSON matching the schema.
8. Save to `~/.tqcli/models/qwen3-4b-vllm/turboquant_kv.json`.
9. Wire into `tqcli model pull` so future snapshots auto-generate metadata if the model has no `turboquant_kv.json` and the recipe requires one.

Acceptance criteria:
- The file loads via `load_turboquant_metadata()` without raising.
- `tqcli chat --model qwen3-4b-vllm --engine vllm --kv-quant turbo3 --prompt "What is the capital of France?" --json` produces "Paris" (or a coherent sentence containing it), not gibberish.
- Perplexity on a 100-prompt sanity set is no worse than 1.5× the `kv:none` baseline. (If it is worse, escalate to Option B.)
- Add a new integration test `T1_vq_turboquant` / `T4_vq_turboquant` that re-runs T1_vq and T4_vq with `kv_quant_choice="turbo3"` on Qwen 3 4B vLLM and asserts `tune.kv_cache_dtype == "turboquant35"` plus the secret-word round-trip.

### Option B — Activation-based Lloyd calibration (proper, slow)

Match Google's published `lloyd_beta_v1` codebook derivation. Ballpark 2-3 days of work to build a minimum-viable calibrator from vLLM forward hooks.

1. Load the model with `kv_cache_dtype="auto"` (no TurboQuant) — this is the baseline precision.
2. Run a 100-200 prompt calibration corpus through the model under forward hooks on every `*.self_attn.k_proj` and `*.self_attn.v_proj` output.
3. Accumulate per-layer, per-head, per-channel absolute activation statistics (mean, max, or 99th-percentile). Store as a float tensor of shape `[n_layers, 2 (k/v), num_kv_heads, head_size]`.
4. Apply the fork's Hadamard rotation on the accumulated tensor (post-projection, pre-cache is the right surface).
5. Per (layer, head, k/v), select top-`outlier_count` indices by the chosen statistic.
6. Emit the same JSON schema as Option A.
7. Acceptance criterion: perplexity within 1.05× of `kv:none` on the sanity set — i.e., near-lossless, which is what `turboquant35` is supposed to deliver per Google's paper.

Option B dominates Option A on quality. Option A dominates Option B on time-to-ship.

### Option C — Request upstream metadata

Open an issue on `github.com/ithllc/vllm-turboquant` asking the fork maintainer to publish calibrated metadata for the Qwen 3 family alongside the existing Gemma 4 metadata. Zero work on our side; depends on the maintainer's priorities and roadmap.

## 6. Verification checklist

Before declaring the fix landed:

1. `~/.tqcli/models/qwen3-4b-vllm/turboquant_kv.json` exists and validates with `vllm.v1.attention.ops.turboquant_metadata.load_turboquant_metadata()` without exception.
2. `tqcli chat --model qwen3-4b-vllm --engine vllm --kv-quant turbo3 --prompt "...smoke prompt..." --json` loads and responds with coherent text (no "the the the" repetition, no mojibake, no NaN-like empty output).
3. `tests/test_integration_agent_functional.py` is updated to run T1_vq and T4_vq with `kv_quant_choice="turbo3"` on Qwen 3 4B vLLM, and both still pass with the `turboquant_kv_active` assertion added (6/6 instead of the current 5/5).
4. `tests/integration_reports/agent_modes_functional_report.md` regenerated; the Qwen vLLM rows no longer carry the `[kv:none — turboquant_kv.json not present for Qwen3 4B]` annotation.
5. One-paragraph entry in `CHANGELOG.md` under 0.6.1 Added or Fixed, describing the calibration approach used (A, B, or C).
6. Memory update to `project_turboquant_kv_progress.md` noting Qwen 3 4B on vLLM is now turboquant35-enabled.

## 7. Out of scope for this prompt

- Fixing the head-dim mismatch for Gemma 4's other layer sizes (`head_size=512` observed at layer 4 disabling TurboQuant per-layer — separate issue #20-class concern).
- Qwen 3 8B / 14B / 32B TurboQuant support. Same path applies but with different `num_kv_heads`, `head_size`, and `num_hidden_layers`.
- Metadata for the `qwen3-4b-AWQ` pre-quantized variant (AWQ + TurboQuant KV compatibility is a separate investigation).
- CPU-offload tuning beyond `max_len=896` for the 4 GB VRAM reference box.

## 8. References

- `vllm/v1/attention/ops/turboquant_metadata.py` — schema, loader, `build_default_turboquant_metadata` (the anti-pattern), `discover_turboquant_metadata_path`.
- `vllm/v1/attention/backends/triton_attn.py:495-600` — the call site that requires the file.
- `tests/test_integration_agent_functional.py:load_vllm_engine` — where to reapply `kv_quant_choice="turbo3"` after the metadata lands.
- `tests/integration_reports/agent_modes_functional_report.md` — current 5/5 Qwen vLLM rows to upgrade to 6/6.
- `tests/integration_reports/turboquant_kv_comparison_report.md` — test-7 Gemma 4 E2B vLLM run for a known-good turboquant35 reference.
- Gemini review transcript 2026-04-18 (not checked in; summary in §4 above).
- arxiv.org/abs/2504.19874 — TurboQuant paper (for Option B methodology).
- arxiv.org/abs/2502.02617 — PolarQuant paper (complementary).
