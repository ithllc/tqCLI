# tqCLI TurboQuant KV Comparison — Full-Lifecycle Test Cases

**Status:** Planned — generates `turboquant_kv_comparison_report.md` / `.json`
**Feature:** TurboQuant KV cache compression across both inference engines
**Scope:** llama.cpp (turbo3/turbo4/turbo2) + vLLM (turboquant35) + CPU-offload path
**GitHub Issues:** [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13), [#20](https://github.com/ithllc/tqCLI/issues/20), [#22](https://github.com/ithllc/tqCLI/issues/22)
**Driving test scripts:**
- `tests/test_integration_turboquant_kv.py` — pipeline logic + llama.cpp + vLLM KV tests
- `tests/test_gemma4_vllm_cpu_offload.py` — Gemma 4 E2B on vLLM with BNB_INT4 + CPU offload + turboquant35

**Prerequisites:**
- llama-cpp-python built against `ithllc/llama-cpp-turboquant` fork (CUDA 12.8+)
- `ithllc/vllm-turboquant` installed from source (CUDA 12.8+), with the four-file page-size unification patches (see `patches/vllm-turboquant/issue_22_page_size_fix.md`)
- Transformers >= 5.5.0 for Gemma 4 architecture recognition
- `turboquant_kv.json` metadata present for each vLLM model directory

> This suite supersedes `turboquant_kv_test_cases.md` by (a) adding the
> Gemma 4 E2B vLLM CPU-offload path from `test_gemma4_vllm_cpu_offload.py`,
> (b) restoring the full-lifecycle coverage from `llama_cpp_test_cases.md`
> (image input, audio input, skill creation/verification, multi-process
> assessment, model removal, clean-uninstall check), and (c) explicitly
> capturing tokens/second and KV compression ratio for every engine +
> KV-dtype combination so the comparison report is complete.

---

## Hardware Requirements

| Property | Value |
|----------|-------|
| GPU | NVIDIA with SM86+ (Ampere or newer) |
| VRAM | 4 GB minimum (CPU offload covers Gemma 4 E2B on vLLM) |
| RAM | 16 GB minimum, 32 GB recommended (for vLLM CPU offload) |
| CUDA | 12.8+ |
| OS | Linux / WSL2 (CPU offload UVA patch required on WSL2) |

---

## Section A — Pipeline Logic Tests (both engines)

Driver: `tests/test_integration_turboquant_kv.py`

### A.1 llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV
**Model:** `gemma-4-e4b-it-Q4_K_M` (GGUF, ~4,981 MB)
**Pipeline:** `detect pre_quantized → kv:turbo3`

| # | Step | Expected |
|---|------|----------|
| 1 | `detect_model_precision` | Reports `pre_quantized` (Q4_K_M, gguf) |
| 2 | `plan_quantization_pipeline` | Returns `kv:turbo3`, no weight stage |
| 3 | `get_llama_kv_params(TURBO3)` | `cache_type_k=turbo3, cache_type_v=turbo3` |
| 4 | `check_turboquant_compatibility` | PASS (CUDA 12.8, SM86) |

### A.2 llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV
**Model:** `qwen3-4b-Q4_K_M` (GGUF, ~2,382 MB)
**Pipeline:** `detect pre_quantized → kv:turbo3`

| # | Step | Expected |
|---|------|----------|
| 1 | Registry lookup | Profile found, GGUF format |
| 2 | Precision detection | `pre_quantized` |
| 3 | Pipeline plan | `kv:turbo3` only (no weight stage) |
| 4 | Cache-param generation | Matches turbo3 spec |
| 5 | turbo4 fallback check | `get_llama_kv_params(TURBO4)` OK |
| 6 | turbo2 fallback check | `get_llama_kv_params(TURBO2)` OK |
| 7 | Compression-ratio table | `KV_COMPRESSION_RATIO[TURBO3]==4.6` |

### A.3 vLLM Qwen 3 4B BF16 + bnb_int4 + turboquant35 KV
**Model:** `qwen3-4b-vllm` (BF16 safetensors)
**Pipeline:** `detect full_precision → weight:bnb_int4 → kv:turboquant35`

| # | Step | Expected |
|---|------|----------|
| 1 | Precision detection | `full_precision` |
| 2 | `select_quantization` | Returns `bnb_int4` (fits after INT4) |
| 3 | Pipeline plan | `weight:bnb_int4 → kv:turboquant35` |
| 4 | `get_vllm_kv_params` | `kv_cache_dtype=turboquant35` |
| 5 | `build_vllm_config` | `feasible=True`, TRITON_ATTN |

### A.4 vLLM Qwen 3 4B AWQ + turboquant35 KV
**Model:** `qwen3-4b-AWQ` (AWQ INT4 safetensors)
**Pipeline:** `detect pre_quantized → kv:turboquant35`

| # | Step | Expected |
|---|------|----------|
| 1 | Precision detection | `pre_quantized` (AWQ) |
| 2 | Pipeline plan | `kv:turboquant35` only |
| 3 | `get_vllm_kv_params` | `kv_cache_dtype=turboquant35` |
| 4 | `build_vllm_config` | `feasible=True`, `quantization=awq_marlin` |
| 5 | `turboquant_kv.json` present | head_dim=128, 36 layers |

### A.5 Baseline (no KV compression)

| # | Step | Expected |
|---|------|----------|
| 1 | llama.cpp q8_0 path | `cache_type_k=q8_0` |
| 2 | vLLM auto path | `kv_cache_dtype=auto` |
| 3-8 | Pipeline plans `kv:none` | No KV stage emitted |

### A.6 CUDA compatibility + graceful degradation

| # | Check | Expected |
|---|-------|----------|
| 1 | `parse_cuda_version("12.8")` | `(12, 8)` |
| 2 | `parse_cuda_version("13.0")` | `(13, 0)` |
| 3 | `check_turboquant_compatibility` on CUDA 12.8/SM86 | PASS |
| 4 | Unsupported CUDA → fallback | Degrades to `kv:none`, warns user |
| 5-13 | `get_kv_quant_info` strings | Match compression table |

---

## Section B — llama.cpp End-to-End KV Benchmarks

Driver: `tests/test_integration_turboquant_kv.py` (real inference path via `LlamaBackend`)

### B.1 Qwen 3 4B Q4_K_M: f16 vs turbo3

| # | Step | Expected |
|---|------|----------|
| 1 | Load with `cache_type_k=f16, cache_type_v=f16` | Baseline load |
| 2 | 2-turn chat ("What is 2 + 2?" → "Multiply by 10") | Correct answers (4, 40) |
| 3 | Capture Turn 1/Turn 2 tok/s | ~6.4 / ~7.4 tok/s |
| 4 | Reload with `cache_type_k=turbo3, cache_type_v=turbo3` | turbo3 active |
| 5 | Repeat 2-turn chat | Same content quality |
| 6 | Capture Turn 1/Turn 2 tok/s | ~7.3 / ~7.4 tok/s (+14% prompt eval) |
| 7 | Measure KV cache VRAM | ~112 MB (4.6× less than baseline) |
| 8 | Context capacity estimate | ~1,700 tokens vs ~368 baseline |

### B.2 Gemma 4 E4B Q4_K_M: f16 vs turbo3

| # | Step | Expected |
|---|------|----------|
| 1 | Load f16 baseline, chat 2 turns (Paris → population) | Correct answers |
| 2 | Record tok/s | 2-5 tok/s |
| 3 | Reload with turbo3 | turbo3 active |
| 4 | Repeat chat | Same quality |
| 5 | Record tok/s + KV MB | Record for comparison table |

### B.3 turbo4 and turbo2 quality sweep (Qwen 3 4B)

| # | Step | Expected |
|---|------|----------|
| 1 | Load turbo4 | 3.8× compression, ≤ +0.23% PPL |
| 2 | Load turbo2 | 6.4× compression, quality warning printed |
| 3 | Chat turn on each | Answers still correct for simple prompts |

---

## Section C — vLLM End-to-End KV Benchmarks

Driver: `tests/test_integration_turboquant_kv.py` + `tests/test_gemma4_vllm_cpu_offload.py`

### C.1 Qwen 3 4B AWQ: auto vs turboquant35

| # | Step | Expected |
|---|------|----------|
| 1 | Load AWQ + `kv_cache_dtype=auto` (FlashAttn v2) | Baseline load |
| 2 | 2-turn chat | Correct answers |
| 3 | Record tok/s + KV tokens at 50 MB | ~5.7/6.8 tok/s, 336 tokens |
| 4 | Reload with `kv_cache_dtype=turboquant35` (Triton attn) | turboquant35 on all 36 layers |
| 5 | 2-turn chat | Correct answers |
| 6 | Record tok/s + KV tokens | ~2.0/1.2 tok/s, **1,344 tokens (4.0×)** |
| 7 | Verify no regression after page-size patch | 30/33 prior failures resolved |

### C.2 Gemma 4 E2B vLLM + BNB_INT4 + CPU offload + turboquant35

Driver: `tests/test_gemma4_vllm_cpu_offload.py` (10-step test)

**Pipeline:** `detect full_precision → weight:bnb_int4 → cpu_offload → kv:turboquant35`

| # | Step | Expected |
|---|------|----------|
| 1 | `find_model_profile("gemma-4-e2b-it-vllm")` | Profile resolved |
| 2 | `detect_model_precision` | `full_precision` (BF16 safetensors) |
| 3 | Size estimates | BF16≈11,710 MB; INT4≈4,145 MB; VRAM=4,096 MB |
| 4 | `select_quantization` without offload | Returns `None` (too large) |
| 5 | `build_vllm_config` with offload | `feasible=True`, `cpu_offload_gb=9.9`, `kv_cache_dtype=turboquant35`, `max_model_len=2048` |
| 6 | Model present at `~/.tqcli/models/gemma-4-e2b-it-vllm` | Safetensors + `turboquant_kv.json` (head_dim=256, 35 layers) |
| 7 | Load model | ~375 s on RTX A2000 4 GB WSL2 |
| 8 | Thinking turn ("What is 15% of 240?") | Answer `36`, coherent reasoning |
| 9 | Simple turn ("Capital of France?") | Answer contains `Paris` |
| 10 | Unload | Clean GPU memory release |

**Expected KV compression coverage:**
- 28 sliding-window layers (head_dim=256) → **turboquant35** (4.6×)
- 7 full-attention layers (head_dim=512) → bf16 fallback (no calibration metadata)
- KV budget: 4,368 tokens under 64 MiB, max_concurrency 4.21× at 2,048 ctx
- Throughput: ~0.2 tok/s (CPU-offload dominated; not a regression)

### C.3 Qwen 3 4B AWQ regression after page-size patches

| # | Step | Expected |
|---|------|----------|
| 1 | Re-run C.1 after the four `vllm-turboquant` patches | Load in ~41.7 s |
| 2 | Verify no regression | tok/s and KV tokens match pre-patch numbers |

---

## Section D — Thinking + Tool Calling under TurboQuant KV

Validates that KV compression does not corrupt reasoning chains or structured JSON output.

### D.1 Thinking mode

| Sub-test | Model | Engine | KV | Expected |
|----------|-------|--------|----|----------|
| 5a | Qwen3 4B Q4_K_M | llama.cpp | turbo3 | `<think>` coherent, `/no_think` produces empty block |
| 5b | Qwen3 4B AWQ | vLLM | turboquant35 | `<think>` coherent, answers correct |
| 5c | Gemma 4 E2B Q4_K_M | llama.cpp | turbo3 | Step-by-step reasoning |
| 5d | Gemma 4 E2B vLLM | vLLM | turboquant35 | Thinking turn via `test_gemma4_vllm_cpu_offload.py` step 8 |

### D.2 Tool / function calling

| Sub-test | Model | Engine | KV | Expected |
|----------|-------|--------|----|----------|
| 6a | Qwen3 4B Q4_K_M | llama.cpp | turbo3 | `<tool_call>` valid JSON `{get_weather, Tokyo}` |
| 6b | Qwen3 4B AWQ | vLLM | turboquant35 | JSON body valid, closing tag may truncate at 128-ctx |
| 6c | Gemma 4 E2B Q4_K_M | llama.cpp | turbo3 | Valid `<\|tool_call>` structure |

### D.3 Combined thinking → tool call → answer

| Sub-test | Model | Engine | KV | Expected |
|----------|-------|--------|----|----------|
| 7a | Qwen3 4B Q4_K_M | llama.cpp | turbo3 | `<think>` → `<tool_call>` → natural answer |
| 7c | Gemma 4 E2B Q4_K_M | llama.cpp | turbo3 | Reasoning → weather call → umbrella answer |

Skipped on 4 GB VRAM: 7b (vLLM 128-ctx too tight), 7d (Gemma 4 GPTQ doesn't fit).

---

## Section E — Full Lifecycle (parity with `llama_cpp_test_cases.md`)

Runs the commands below once per `(engine, model, kv_level)` combination that completed Sections B/C successfully. Captures tok/s and KV compression ratio at each step.

### E.1 Install

```bash
pip install -e ".[server]"
tqcli --version
tqcli system info --json
```

**Expected:** Version >= 0.5.0, `system info` shows TurboQuant KV status and selected engine.

### E.2 Pull model

```bash
tqcli model pull <model-id>
tqcli model list
```

**Expected:** Correct HF repo, quantization, format, engine recorded; GGUF single-file or vLLM directory with `turboquant_kv.json`.

### E.3 KV-compressed chat (two turns)

```bash
tqcli chat --model <model-id> --kv-quant <level>
```

**Expected per turn:**
- Correct factual/arithmetic answer
- Performance monitor prints tok/s and time-to-first-token
- KV compression indicator shown (`turbo3`, `turbo4`, `turbo2`, or `turboquant35`)

### E.4 Image input (multimodal models only)

```bash
# in chat:
/image tests/fixtures/test_image.png What colors do you see?
```

**Expected:** Gemma 4 E4B / E2B identify colors; Qwen 3 returns text-only failure gracefully.

### E.5 Audio input

```bash
# in chat:
/audio tests/fixtures/test_audio.wav Describe what you hear.
```

**Expected:** Graceful "no audio capability" for all current quantized models. Acceptable behavior.

### E.6 Skill creation + verification

```bash
tqcli skill create tq-kv-<model>-<kv> -d "TurboQuant KV smoke skill"
tqcli skill run tq-kv-<model>-<kv>
tqcli skill list
```

**Expected:** SKILL.md + script created, skill executes and returns JSON, appears in list.

### E.7 Multi-process feasibility assessment

```python
from tqcli.core.multiprocess import assess_multiprocess
from tqcli.core.system_info import detect_system

plan = assess_multiprocess(
    sys_info=detect_system(),
    model_path="<path>",
    model_size_mb=<size>,
    requested_workers=3,
    preferred_engine="<llama.cpp|vllm>",
    unrestricted=True,
)
print(plan.feasible, plan.engine, plan.max_workers)
```

**Expected:**
- llama.cpp: sequential queue, max 4 workers
- vLLM: continuous batching, max workers bounded by remaining VRAM after KV budget

### E.8 Server / worker smoke (yolo mode)

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go serve start -m <model-id> --kv-quant <level>
tqcli serve status
tqcli chat --engine server
tqcli serve stop
```

**Expected:** Server starts on 8741, worker connects, KV-level propagates, clean stop.

### E.9 Remove model + clean uninstall

```bash
tqcli model remove <model-id>
tqcli model list          # gone
pip show tqcli             # still installed
pip uninstall tqcli -y
pip show tqcli             # not found
```

**Expected:** Model directory deleted, registry entry cleared, package removed cleanly.

---

## Metrics Captured (drives the comparison report)

For every `(engine, model, kv_level)` combination the report must include:

| Column | Source |
|--------|--------|
| Turn 1 tok/s | `PerformanceMonitor` around `engine.chat()` |
| Turn 2 tok/s | Second turn of the same session |
| Time to first token | `PerformanceMonitor` |
| KV cache MB | llama.cpp load log / vLLM KV budget line |
| KV tokens served | vLLM `kv_cache_memory_bytes` / n_ctx |
| Compression ratio | `KV_COMPRESSION_RATIO[level]` + measured |
| Quality check | Answer correctness + coherence of `<think>` / tool JSON |
| Load time | Wall-clock from `engine.load()` |
| Lifecycle pass/fail | Sections E.1–E.9 |

---

## Model Download Requirements

| Model | Source | File / Layout | Size | Engine |
|-------|--------|---------------|------|--------|
| Qwen3 4B Q4_K_M | `Qwen/Qwen3-4B-GGUF` | `Qwen3-4B-Q4_K_M.gguf` | 2,382 MB | llama.cpp |
| Qwen3 4B AWQ | `Qwen/Qwen3-4B-AWQ` | repo snapshot + `turboquant_kv.json` | 2,558 MB | vLLM |
| Gemma 4 E4B Q4_K_M | `unsloth/gemma-4-E4B-it-GGUF` | `gemma-4-E4B-it-Q4_K_M.gguf` | 4,981 MB | llama.cpp |
| Gemma 4 E2B Q4_K_M | `unsloth/gemma-4-E2B-it-GGUF` | `gemma-4-E2B-it-Q4_K_M.gguf` | 2,890 MB | llama.cpp |
| Gemma 4 E2B BF16 (vLLM) | `google/gemma-4-e2b-it` | safetensors + generated `turboquant_kv.json` (head_dim=256, 35 layers) | 11,710 MB BF16 / 4,145 MB INT4 | vLLM (BNB_INT4 + CPU offload 9.9 GB) |

---

## Pre-Test Checklist

- [ ] CUDA 12.8 toolkit installed (`nvcc --version` shows 12.8+)
- [ ] `ithllc/llama-cpp-turboquant` fork built, `turbo3` cache type accepted
- [ ] `ithllc/vllm-turboquant` installed, four page-size patches applied
- [ ] Transformers >= 5.5.0 (Gemma 4 architecture)
- [ ] `turboquant_kv.json` generated for each vLLM model
- [ ] `--kv-quant` flag wired with graceful fallback on incompatible systems
- [ ] `tqcli system info` shows TurboQuant KV status
- [ ] Flash attention enabled (`-fa`) for llama.cpp
- [ ] Disk space available (~25 GB for full model set)

## Output Files

- `tests/integration_reports/turboquant_kv_comparison_report.md`
- `tests/integration_reports/turboquant_kv_comparison_report.json`

The report aggregates Sections A–E. The Gemma 4 E2B vLLM run (Section C.2) must appear alongside the Qwen 3 comparisons so the engine-vs-engine and KV-level-vs-KV-level views are complete.
