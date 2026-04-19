# TP: TurboQuant KV Calibrator Architecture Extension

Companion to `docs/prd/PRD_calibrator_arch_extension.md`. Closes tqCLI #31. Target release: **0.6.2**.

## Architecture

Each wrapper monkey-patches the HF `*Attention.forward` for one architecture, captures post-RoPE K and raw V into per-layer second-moment dicts, then delegates to the normal forward. The pattern is identical to the shipped Qwen3 wrapper at `tqcli/core/kv_metadata_generator.py:392-466`; only the forward-replication body differs per family.

## Phase 1: head_dim derivation fix

### File: `tqcli/core/kv_metadata_generator.py`

Update `_extract_architecture_params` (currently lines 481-498) to derive `head_dim` when not explicitly set:

```python
def _extract_architecture_params(config: dict) -> tuple[str, int, int, int]:
    arch = (config.get("architectures") or ["unknown"])[0]
    head_dim = config.get("head_dim")
    num_kv_heads = config.get("num_key_value_heads")
    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    num_attn_heads = config.get("num_attention_heads")

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        head_dim = head_dim if head_dim is not None else text_config.get("head_dim")
        num_kv_heads = num_kv_heads if num_kv_heads is not None else text_config.get("num_key_value_heads")
        num_layers = num_layers if num_layers is not None else text_config.get("num_hidden_layers")
        hidden_size = hidden_size if hidden_size is not None else text_config.get("hidden_size")
        num_attn_heads = num_attn_heads if num_attn_heads is not None else text_config.get("num_attention_heads")

    # Derive head_dim when not explicitly provided (Llama 3 / Mistral / Phi-3 / SmolLM2 pattern).
    if head_dim is None and hidden_size and num_attn_heads:
        head_dim = hidden_size // num_attn_heads

    return arch, head_dim, num_kv_heads, num_layers
```

## Phase 2: Llama 3 capture wrapper

### File: `tqcli/core/kv_metadata_generator.py`

Add after `_install_qwen3_capture`:

```python
def _install_llama_capture() -> _CaptureHandle:
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_llama.LlamaAttention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        # LlamaAttention has no q_norm/k_norm — direct projection.
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(
            self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_llama.LlamaAttention.forward = patched_forward

    def restore():
        modeling_llama.LlamaAttention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)
```

Extract the accumulator into a shared helper (called from all wrappers):

```python
def _accumulate_kv_scores(
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scores_k: dict[int, torch.Tensor],
    scores_v: dict[int, torch.Tensor],
    token_counts: dict[int, int],
) -> None:
    with torch.no_grad():
        k32 = key_states.detach().to(torch.float32)
        v32 = value_states.detach().to(torch.float32)
        bsz, n_kv, seq, hdim = k32.shape
        k_flat = k32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
        v_flat = v32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
        k_sum = k_flat.square().sum(dim=0).to(torch.float64).cpu()
        v_sum = v_flat.square().sum(dim=0).to(torch.float64).cpu()
        if layer_idx in scores_k:
            scores_k[layer_idx] += k_sum
            scores_v[layer_idx] += v_sum
            token_counts[layer_idx] += bsz * seq
        else:
            scores_k[layer_idx] = k_sum
            scores_v[layer_idx] = v_sum
            token_counts[layer_idx] = bsz * seq
```

Refactor `_install_qwen3_capture` to use `_accumulate_kv_scores` (no behavior change, same output).

## Phase 3: Mistral capture wrapper

Nearly identical to Llama 3. Key difference: MistralAttention passes `sliding_window` to the attention interface; capture is before the interface, so sliding_window is passed through unchanged.

```python
def _install_mistral_capture() -> _CaptureHandle:
    from transformers.models.mistral import modeling_mistral
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_mistral.MistralAttention.forward

    def patched_forward(
        self, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling, sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_mistral.MistralAttention.forward = patched_forward
    def restore():
        modeling_mistral.MistralAttention.forward = original
    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)
```

## Phase 4: Phi-3 capture wrapper

Phi-3 uses fused `qkv_proj` (single Linear outputting `[Q|K|V]` concatenated). Slice then proceed.

```python
def _install_phi3_capture() -> _CaptureHandle:
    from transformers.models.phi3 import modeling_phi3
    from transformers.models.phi3.modeling_phi3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_phi3.Phi3Attention.forward

    def patched_forward(
        self, hidden_states, position_embeddings, attention_mask,
        past_key_values=None, **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        qkv = self.qkv_proj(hidden_states)
        # Split sizes: [n_heads*hd, n_kv*hd, n_kv*hd]
        q_size = self.config.num_attention_heads * self.head_dim
        kv_size = self.config.num_key_value_heads * self.head_dim
        query_states = qkv[..., :q_size].view(hidden_shape).transpose(1, 2)
        key_states = qkv[..., q_size : q_size + kv_size].view(hidden_shape).transpose(1, 2)
        value_states = qkv[..., q_size + kv_size :].view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self, query_states, key_states, value_states, attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_phi3.Phi3Attention.forward = patched_forward
    def restore():
        modeling_phi3.Phi3Attention.forward = original
    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)
```

## Phase 5: Registry

```python
_CAPTURE_INSTALLERS: dict[str, Callable[[], _CaptureHandle]] = {
    "Qwen3ForCausalLM": _install_qwen3_capture,
    "LlamaForCausalLM": _install_llama_capture,
    "MistralForCausalLM": _install_mistral_capture,
    "Phi3ForCausalLM": _install_phi3_capture,
}
```

## Phase 6: Tests

### File: `tests/test_kv_metadata_archs.py`

1. `test_head_dim_derived_when_missing` — config without `head_dim` but with `hidden_size` + `num_attention_heads` resolves via `_extract_architecture_params`.
2. `test_llama_wrapper_captures_post_rope_k` — load SmolLM2-135M locally, run one forward, assert `scores_k` has 30 entries with shape `(3, 64)`.
3. `test_mistral_wrapper_captures_post_rope_k` — load TinyMistral-248M, run one forward, assert `scores_k` has 12 entries with shape `(8, 32)`.
4. `test_phi3_fused_qkv_slicing` — mock `qkv_proj` output, assert `q`, `k`, `v` shapes match expected splits.
5. `test_phi3_wrapper_captures_post_rope_k` — load Phi-3-mini-4k-instruct, run one forward, assert `scores_k` has 32 entries with shape `(32, 96)`.
6. `test_precondition_accepts_new_archs` — dummy configs for each arch pass `check_calibration_preconditions`.

### Integration smoke (`tests/test_cli_calibrate_kv_archs.py`)

For each of the three test models that's already downloaded:
- Run `tqcli model calibrate-kv <model-id> --recipe turboquant35`.
- Assert exit code 0 and `turboquant_kv.json` emitted.
- Load the emitted file via `vllm.v1.attention.ops.turboquant_metadata.load_turboquant_metadata()`; assert no error.

## Phase 7: Model downloads

Use existing `tqcli model pull` path. Confirm registry has entries for:
- `smollm2-135m-instruct` → `HuggingFaceTB/SmolLM2-135M-Instruct`
- `tinymistral-248m` → `Locutusque/TinyMistral-248M`
- `phi-3-mini-4k-instruct` → `microsoft/Phi-3-mini-4k-instruct`

If the model registry doesn't have them, add entries or download directly via `huggingface_hub.snapshot_download` to `~/.tqcli/models/`.

## Phase 8: Docs + CHANGELOG

- `docs/architecture/turboquant_kv.md`: add "Supported architectures" section listing all four.
- `CHANGELOG.md`: 0.6.2 entry — Added LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM to `_CAPTURE_INSTALLERS`. Fixed: head_dim derivation from `hidden_size / num_attention_heads` when not explicit.
- Version bump: `pyproject.toml` + `tqcli/__init__.py` → `0.6.2`.

## Verification

1. `pytest tests/test_kv_metadata_archs.py -v` — 6/6 green.
2. `tqcli model pull smollm2-135m-instruct && tqcli model calibrate-kv smollm2-135m-instruct` — succeeds.
3. Same for tinymistral-248m and phi-3-mini-4k-instruct.
4. Load each emitted JSON via vLLM's metadata loader — no error.

## Out of scope for this release (explicit)

- Full PPL validation on 7B+ models (hours on 4 GB VRAM).
- Gated model access (meta-llama/*, mistralai/*).
- Mixtral (MoE architecture).
- Gemma family (blocked by upstream #32).
