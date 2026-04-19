# PRD: TurboQuant KV Calibrator Architecture Extension (Llama 3 / Mistral / Phi-3)

**Closes:** tqCLI #31
**Target release:** tqCLI 0.6.2
**Prerequisite:** tqCLI 0.6.1 (Qwen 3 calibrator shipped 2026-04-19)

## 1. Problem

`tqcli/core/kv_metadata_generator.py::_CAPTURE_INSTALLERS` currently recognizes only `Qwen3ForCausalLM`. Every other HF architecture hits the precondition check at line 544-549 and refuses calibration with `Architecture 'X' has no capture wrapper registered.` This blocks users running Llama 3, Mistral, or Phi-3 variants from getting auto-generated `turboquant_kv.json` via `tqcli model calibrate-kv`.

## 2. Target audience
- Users running Llama 3 series (including SmolLM2, Llama-3.2-1B/3B, Llama-3.1-8B derivatives).
- Users running Mistral 7B and architecturally-compatible small variants.
- Users running Phi-3 mini (3.8B) — particularly relevant since Phi-3 uses fused qkv_proj which requires a distinct wrapper shape.

## 3. Scope

### In scope
- Add capture wrappers for `LlamaForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM` to `_CAPTURE_INSTALLERS`.
- Extend `_extract_architecture_params` to derive `head_dim = hidden_size // num_attention_heads` when `config.head_dim` is absent (most new Llama/Mistral/Phi-3 configs).
- End-to-end calibration verification on one hardware-compatible model per family:
  - Llama 3: `HuggingFaceTB/SmolLM2-135M-Instruct` (`LlamaForCausalLM`, 135M, ~270 MB)
  - Mistral: `Locutusque/TinyMistral-248M` (`MistralForCausalLM`, 248M, ~500 MB)
  - Phi-3: `microsoft/Phi-3-mini-4k-instruct` (`Phi3ForCausalLM`, 3.8B, ~7.6 GB)
- Unit tests validating each wrapper's `patched_forward` captures K/V into the correct accumulator dict.
- Integration smoke test: `tqcli model calibrate-kv <model> --recipe turboquant35` emits a valid `turboquant_kv.json` for each of the three test models.

### Out of scope (explicitly)
- Perplexity-gate validation on large (>3B) models — deferred to user's own compute because a 7B Mistral calibration on 4 GB VRAM + 31 GB RAM box runs into hours.
- Gated (meta-llama/, mistralai/) model testing — those need HF token approval outside the session. We cover the architecture classes via open-access mirrors (SmolLM2 for LlamaForCausalLM, TinyMistral for MistralForCausalLM). The architecture-level validation transfers.
- Gemma family — separate (blocked on upstream #32).
- Phi-1 / Phi-2 — older model families, distinct from Phi-3's fused-qkv architecture.
- Mixture-of-Experts variants (Mixtral) — different attention architecture scope.

## 4. User stories

- **As a user running Phi-3-mini on vLLM**, I want `tqcli model calibrate-kv phi-3-mini` to generate a valid `turboquant_kv.json` so I can run `--kv-quant turbo3` without hitting the "metadata required" error at `triton_attn.py:600`.
- **As a user running SmolLM2 or a small Llama-3 derivative**, I want the same generator to work for my model without me writing a capture wrapper.
- **As the maintainer**, I want `_CAPTURE_INSTALLERS` to be extensible with ~50 lines per architecture so adding future families (Qwen2, etc.) is mechanical.

## 5. Acceptance criteria

1. `_CAPTURE_INSTALLERS` contains all four entries: `Qwen3ForCausalLM`, `LlamaForCausalLM`, `MistralForCausalLM`, `Phi3ForCausalLM`.
2. `check_calibration_preconditions` returns `(True, "OK: ...")` for configs where `head_dim` must be derived.
3. `tqcli model calibrate-kv ~/.tqcli/models/smollm2-135m-instruct --recipe turboquant35` produces a `turboquant_kv.json` whose schema validates against `load_turboquant_metadata` (vLLM fork) without error.
4. Same for `tinymistral-248m` and `phi-3-mini-4k-instruct` model dirs.
5. Unit tests cover (a) each wrapper's forward-pass correctness, (b) head_dim derivation, (c) fused-qkv slicing for Phi-3, (d) sliding_window parameter not affecting capture shape for Mistral.
6. Integration smoke: a generated JSON for SmolLM2-135M loads successfully via `vllm.v1.attention.ops.turboquant_metadata.load_turboquant_metadata()`.

## 6. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Transformers version drift renames the attention class or its internal forward signature | Pin behavior by capturing via monkey-patch, not subclassing. Unit tests flag breakage on version upgrade. Document supported transformers version range. |
| Phi-3 fused-qkv slice sizes mis-computed (num_heads vs num_kv_heads) | Unit test asserts split sizes sum to `qkv_proj.out_features` and each slice passes through RoPE without shape error. |
| Mistral sliding_window affects K capture unexpectedly | Capture happens BEFORE the attention interface is called — sliding_window is a mask-side concern, not a tensor-shape concern. Test: run TinyMistral calibration with sliding_window=32 (model's default) and verify captured K shape equals `(n_kv_heads, head_dim)`. |
| TinyMistral head_dim=32 gives outlier_count=16 (borderline) | Acceptable for architecture validation; not a quality claim. Production Mistral-7B has head_dim=128 → outlier_count=64, same as Qwen3. |
| Downloading Phi-3-mini (~7.6 GB) blows disk quotas | 800 GB free disk confirmed; download once to `~/.tqcli/models/`. |

## 7. Success metric

Running `tqcli model calibrate-kv <model-id>` for a SmolLM2 / TinyMistral / Phi-3-mini snapshot produces a schema-valid `turboquant_kv.json` on the first try, with no architecture-refusal from `check_calibration_preconditions`.
