# tqCLI Usage Examples

Verified end-to-end flows. Every scenario labelled **Verified** below was
executed on the reference hardware (NVIDIA RTX A2000 4 GB / WSL2 / CUDA
12.8) and has a corresponding row in
`tests/integration_reports/turboquant_kv_comparison_report.md`.
Scenarios previously labelled **Planned** against
[`docs/prompts/implement_headless_chat_and_vllm_multimodal.md`](../prompts/implement_headless_chat_and_vllm_multimodal.md)
now ship in 0.5.0.

## Reference hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA RTX A2000 Laptop (4,096 MB VRAM) |
| RAM | 32 GB |
| CUDA toolkit | 12.8 |
| OS | Ubuntu 22.04 on WSL2 (Windows 11) |
| TurboQuant forks | `tqcli/llama-cpp-turboquant`, `tqcli/vllm-turboquant` |

---

## 1. Install + first-run bootstrap — **Verified**

```bash
# 0.7.0+: install from PyPI under the `turboquant-cli` distribution name
# (import name remains `tqcli`).
python3 -m venv .venv && source .venv/bin/activate
pip install 'turboquant-cli[all]' \
  --find-links https://github.com/tqcli/vllm-turboquant/releases/latest

tqcli --version
# tqcli, version 0.7.0

tqcli config init
tqcli system info --json | jq '.gpu, .vram_mb, .cuda_toolkit_version'
```

Expected output shape:

```json
"NVIDIA RTX A2000 Laptop"
4096
"12.8"
```

If `tqcli system info --json` returns `turboquant_kv: { "available": true }`
you can use any `--kv-quant` level. On older CUDA it reports
`available: false` and the planner falls back to `kv:none` automatically.

---

## 2. Image input (multimodal, llama.cpp) — **Verified**

Applies to Gemma 4 E4B (`gemma-4-e4b-it-Q4_K_M`) and Gemma 4 E2B
(`gemma-4-e2b-it-Q4_K_M`) on llama.cpp. Requires the matching `mmproj`
CLIP file on disk (auto-downloaded with the model).

```bash
tqcli model pull gemma-4-e4b-it-Q4_K_M
tqcli chat --model gemma-4-e4b-it-Q4_K_M --kv-quant turbo3
```

In the interactive session:

```
You> /image tests/fixtures/test_image.png What colors do you see?
Gemma 4 E4B> The image shows a red square surrounded by a blue border.

  2.8 tok/s | 120 tokens | 42.8s
```

Notes:
- `/image PATH TEXT` prefix is parsed by `tqcli/ui/interactive.py:60`.
- Image encoding adds ~8–10 s on 4 GB VRAM.
- Audio slash-command (`/audio`) is supported but Gemma 4 replies with a
  graceful "no audio capability" message at this quant level — see §5.

---

## 3. Audio input (graceful no-op) — **Verified**

Qwen 3 and Gemma 4 quantized profiles do not ship audio decoders. The
CLI accepts `/audio PATH` but the model responds with a polite refusal.
Per §E.5 of the integration test cases this is the **expected** behavior
and counts as a pass.

```
You> /audio tests/fixtures/test_audio.wav Describe what you hear.
Gemma 4 E4B> I don't have audio processing capability in this configuration.
             If you'd like to share what the recording contains, I can help you analyze the description.
```

No special handling needed — the flow above just works.

---

## 4. llama.cpp + TurboQuant Qwen 3 4B — **Verified**

```bash
tqcli model pull qwen3-4b-Q4_K_M
tqcli chat --model qwen3-4b-Q4_K_M --kv-quant turbo3
```

Two-turn chat matching the §B.1 integration fixture:

```
You> What is 2 + 2?
Qwen 3 4B> <think>The user is asking a simple arithmetic question...</think>
           4

  7.3 tok/s | 42 tokens | 5.8s

You> Now multiply that by 10.
Qwen 3 4B> <think>Starting from 4 and multiplying by 10...</think>
           40

  7.4 tok/s | 36 tokens | 4.9s
```

Measured against the §B.1 baseline (f16 KV):

| KV level | Turn 1 tok/s | Turn 2 tok/s | KV cache MB | Max ctx tokens |
|----------|-------------:|-------------:|-----------:|---------------:|
| f16 (baseline) | ~6.4 | ~7.4 | ~520 | ~368 |
| turbo3 | ~7.3 | ~7.4 | ~112 | ~1,700 |
| turbo4 | ~7.2 | ~7.3 | ~137 | ~1,400 |

`turbo3` is the default auto-selection on 4 GB VRAM (per `select_kv_quant`).

---

## 5. llama.cpp + TurboQuant Gemma 4 E4B — **Verified**

```bash
tqcli model pull gemma-4-e4b-it-Q4_K_M
tqcli chat --model gemma-4-e4b-it-Q4_K_M --kv-quant turbo3
```

```
You> What is the capital of France?
Gemma 4 E4B> Paris.

  3.4 tok/s | 6 tokens | 1.8s

You> Approximately what is the population of that city?
Gemma 4 E4B> Paris has about 2.1 million residents in the city proper,
             with around 12 million in the greater metropolitan area.

  3.8 tok/s | 48 tokens | 12.6s
```

Applicable thinking mode: Gemma 4 uses `<start_of_think>` / `<end_of_think>`
tokens; tqCLI's `tqcli/core/thinking.py` extracts them automatically.

---

## 6. vLLM + TurboQuant Qwen 3 4B AWQ — **Verified**

```bash
tqcli model pull qwen3-4b-AWQ
tqcli chat --model qwen3-4b-AWQ --engine vllm --kv-quant turbo3
```

Expected first-load output:

```
[triton_attn.py] TurboQuant enabled for layer language_model.model.layers.0.self_attn.attn with turboquant35 on Triton/CUDA.
... (36 layers total, all turboquant35) ...
[kv_cache_utils.py] GPU KV cache size: 1,344 tokens
[kv_cache_utils.py] Maximum concurrency for 128 tokens per request: 10.5x
```

Pipeline plan: `detect pre_quantized → kv:turboquant35` (no weight stage).

| KV level | Turn 1 tok/s | Turn 2 tok/s | KV tokens | Compression |
|----------|-------------:|-------------:|----------:|------------:|
| auto (FA) | ~5.7 | ~6.8 | 336 | 1.0× |
| turboquant35 | ~2.0 | ~1.2 | **1,344** | **4.0×** |

Throughput drops because Triton attention is slower than FlashAttention v2
on small batches — the win is in context capacity, not speed.

---

## 7. vLLM + TurboQuant BF16 Qwen 3 4B (bnb_int4 + turboquant35) — **Verified E2E (0.6.1+)**

Since 0.6.1, `qwen3-4b-vllm` (BF16 safetensors) runs end-to-end with
`turboquant35` KV compression. On first load, tqCLI auto-calibrates the
required `turboquant_kv.json` sidecar from a 30-prompt activation
corpus. After that it's a straight reload.

```bash
tqcli model pull qwen3-4b-vllm

# Option A — let first chat invocation auto-calibrate (blocks for ~2-3 min once)
tqcli chat --model qwen3-4b-vllm --engine vllm --kv-quant turbo3

# Option B — pre-warm the metadata explicitly (recommended for CI / prod)
tqcli model calibrate-kv qwen3-4b-vllm
tqcli chat --model qwen3-4b-vllm --engine vllm --kv-quant turbo3
```

Expected first-run output with Option A:

```
[vllm] TurboQuant metadata missing for qwen3-4b-vllm; running activation calibration (this happens once).
[calibrate] model=qwen3-4b-vllm arch=Qwen3ForCausalLM head_dim=128 num_kv_heads=8 num_layers=36 recipe=turboquant35 outlier_count=64
[calibrate] loading model (bf16, device_map=auto)...
[calibrate]   prompt 1/30: 186 tokens (total 186)
... (30 prompts) ...
[calibrate] wrote /root/.tqcli/models/qwen3-4b-vllm/turboquant_kv.json (527.5 KB, 36 layers, 5083 tokens, elapsed 141.8s)
(EngineCore) INFO: kv_cache_dtype=turboquant35 ...
(EngineCore) INFO: GPU KV cache size: 3,632 tokens
```

Pipeline plan: `detect full_precision → weight:bnb_int4 → kv_metadata_calibrate → kv:turboquant35`.

**Quality gate** (opt-in, `TQCLI_PPL_GATE=1 python3 -m unittest tests.test_kv_ppl_validation`):
baseline PPL 6.8893 vs turboquant35 PPL 6.8872, ratio **0.9997** — indistinguishable
from uncompressed. No silent quality collapse.

Precondition refuses (visible via `tqcli model calibrate-kv <model-id>`):
- `qwen3-4b-AWQ` → refused ("already quantized (awq)")
- `gemma-4-e2b-it-vllm` → refused ("variable head_dim (256 / 512)")
- `gemma-4-e2b-it-Q4_K_M` → refused (not a vLLM model)

This is the fix for [#27](https://github.com/ithllc/tqCLI/issues/27).

---

## 8. vLLM + TurboQuant Gemma 4 E2B + CPU offload — **Verified**

The showpiece: Gemma 4 E2B (2.3 B params, 10.2 GB BF16 on disk) running on
a 4 GB VRAM laptop via BNB_INT4 + CPU offload 9.9 GB + TurboQuant35 KV.
All of this is driven by the integration test
`test_7_gemma4_e2b_vllm_cpu_offload`.

```bash
tqcli model pull gemma-4-e2b-it-vllm
tqcli chat --model gemma-4-e2b-it-vllm --engine vllm --kv-quant turbo3
```

Expected first-load timeline on 4 GB VRAM WSL2:

```
Loading safetensors checkpoint shards: 100% [03:58<00:00, 238.40s/it]
[gpu_model_runner.py] Model loading took 6.33 GiB memory and 316.1 seconds
[uva.py] Total CPU offloaded parameters: 0.88
[kv_cache_utils.py] GPU KV cache size: 4,368 tokens
[kv_cache_utils.py] Maximum concurrency for 2,048 tokens per request: 4.21x
```

Two-turn chat (matches §C.2 fixture):

```
You> What is 15% of 240?
Gemma 4 E2B> * **15% of 240 is 36.**

  0.22 tok/s | 30 tokens | 71.5s

You> What is the capital of France?
Gemma 4 E2B> Paris

  0.13 tok/s | 3 tokens | 22.7s
```

Throughput is CPU-offload bounded. Load time (~500–625 s end-to-end) is
dominated by the UVA offloader moving 0.88 GB of weights into pinned RAM.

TurboQuant layer coverage on Gemma 4 E2B:

- **28 sliding-window layers** (head_dim 256) — turboquant35 active
- **7 full-attention layers** (head_dim 512) — bf16 fallback (no
  calibration metadata at that head_dim)

GitHub issues unblocking this flow: #20 (CPU offload), #22 (page-size
unification), #21 (Gemma 4 port into the vLLM fork).

---

## 9. AI Skills Builder Command — **Shipped in 0.5.0**

Generates a functional `~/.tqcli/skills/<name>/` skill scaffold from a PRD and
a Technical Plan using whichever local LLM the config points at. Runs
TurboQuant-aware on both llama.cpp and vLLM.

```bash
# Generate a skill using Qwen3 4B on llama.cpp with turbo3 KV compression
tqcli skill generate \
    --prd docs/prd/PRD_AI_Skills_Builder.md \
    --plan docs/technical_plans/TP_AI_Skills_Builder.md \
    --name my-generated-skill \
    --model qwen3-4b-Q4_K_M \
    --engine llama.cpp \
    --kv-quant turbo4 \
    --yes
# (Omit --yes to review the generated files interactively before writing.)

# Use vLLM + AWQ instead (higher code quality, requires ~5 GB VRAM)
tqcli skill generate \
    --prd PRD.md --plan PLAN.md --name my-skill \
    --model qwen3-4b-AWQ --engine vllm --kv-quant turbo4

# List + run the new skill
tqcli skill list
tqcli skill run my-generated-skill
```

The command writes `SKILL.md` plus one or more Python scripts under
`~/.tqcli/skills/<name>/scripts/`, validates the Python with `ast.parse()`
before committing, and persists the raw LLM response as
`.raw_response.md` alongside for audit/debugging.

---

## 10. Multi-process CRM build on llama.cpp — **Verified**

Based on the Test 3/4 recipe in
[`tests/integration_reports/llama_cpp_test_cases.md`](../../tests/integration_reports/llama_cpp_test_cases.md).

```bash
# In terminal 1 — start the shared server
tqcli --stop-trying-to-control-everything-and-just-let-go serve start \
    -m qwen3-4b-Q4_K_M \
    --engine llama.cpp

tqcli serve status
# Running: pid=NNNN engine=llama.cpp health=OK

# In terminal 2 — create the three CRM skills
tqcli skill create crm-frontend -d "Generate HTML/CSS/JS frontend for CRM"
tqcli skill create crm-backend  -d "Generate Flask backend API for CRM"
tqcli skill create crm-database -d "Generate SQLite schema for CRM"

# Prepare workspace
mkdir -p /tmp/crm_workspace/{frontend,backend,database}

# In terminals 3/4/5 — each worker connects to the server and generates one artifact
tqcli chat --engine server --model qwen3-4b-Q4_K_M
# inside: ask for frontend; save response to /tmp/crm_workspace/frontend/index.html
# repeat for backend/app.py and database/schema.sql

# Verify
ls -la /tmp/crm_workspace/frontend/index.html /tmp/crm_workspace/backend/app.py /tmp/crm_workspace/database/schema.sql

# Cleanup
tqcli serve stop
```

llama.cpp queues requests sequentially — expect ~1 request in flight at
any time. For true concurrent generation, use the vLLM equivalent once §10
lands.

---

## 11. vLLM multimodal + multi-process CRM — **Shipped in 0.5.0**

Headless chat (`--prompt`, `--image`, `--audio`, `--messages`, `--json`,
`--max-tokens`) and vLLM multimodal pass-through landed with #24.
`VllmBackend` now passes PIL images through
`self._llm.generate(prompts=[{"prompt": ..., "multi_modal_data": {"image": [...]}}])`
while preserving CPU offload, `kv_cache_dtype`, `enable_turboquant`, and
`enforce_eager=True` on 4 GB VRAM.

Two commercial-grade features that don't work headlessly today:

### 10a. Image input on vLLM Gemma 4 E2B

Expected once headless chat + `vllm_backend.py` multimodal pass-through
lands:

```bash
tqcli chat \
    --model gemma-4-e2b-it-vllm \
    --engine vllm \
    --kv-quant turbo3 \
    --prompt "What colors do you see?" \
    --image tests/fixtures/test_image.png \
    --json
```

Expected JSON (shape):

```json
{
  "model": "gemma-4-e2b-it-vllm",
  "engine": "vllm",
  "response": "...red square with blue border...",
  "usage": { "prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60 },
  "performance": { "tokens_per_second": 0.22, "total_time_s": 81.7 },
  "metadata": { "kv_quant": "turboquant35", "images": ["tests/fixtures/test_image.png"] }
}
```

Qwen 3 4B AWQ and `qwen3-4b-vllm` are **text-only** (`multimodal=False` in
the registry) — image flows do not apply and are recorded as N/A in the
comparison report.

### 10b. Multi-process CRM on vLLM (Qwen 3 4B AWQ + Gemma 4 E2B)

Expected once headless chat lands:

```bash
# Start the vLLM server
tqcli --stop-trying-to-control-everything-and-just-let-go serve start \
    -m qwen3-4b-AWQ --engine vllm

# Spawn workers — vLLM continuous batching runs them in parallel on the GPU
tqcli workers spawn 2

# Each worker runs one headless chat generating one CRM artifact
tqcli chat --engine server --prompt "Generate an HTML/CSS/JS frontend..." \
    --json > /tmp/crm/frontend.json
tqcli chat --engine server --prompt "Generate a Flask backend..." \
    --json > /tmp/crm/backend.json
tqcli chat --engine server --prompt "Generate a SQLite schema..." \
    --json > /tmp/crm/database.json

tqcli serve stop
```

The Gemma 4 E2B variant runs the same flow with `-m gemma-4-e2b-it-vllm`
and accepts image inputs via `--image` on the worker side. Multi-process
concurrency is bounded by the VRAM budget — on 4 GB VRAM Gemma 4 E2B
supports `max_workers=1` (verified by `assess_multiprocess` in §E.7).

Test coverage plan is documented in §F and §G of
[`turboquant_kv_comparison_test_cases.md`](../../tests/integration_reports/turboquant_kv_comparison_test_cases.md).

---

## 7. Agent Modes — **Planned (0.6.0)**

Three-tier autonomy on `tqcli chat`:

```bash
# (a) Manual — default. CI-safe, deterministic, no tool schemas injected.
tqcli chat --prompt "Summarize CHANGELOG.md" --json

# (b) AI Tinkering — model stages each tool call, you approve [Y/n/Edit].
tqcli chat --ai-tinkering
> Read pyproject.toml and tell me the installed extras.
# → model emits <staged_tool_call>{"name":"tq-file-read",
#                                    "arguments":{"path":"pyproject.toml"}}</staged_tool_call>
# → CLI prints: "Agent wants to run tq-file-read with {path: pyproject.toml}"
# → Prompt: Proceed? [y/n/edit]
# → On y: runs the tool, feeds Observation back to the model, loops.

# (c) Unrestricted (yolo) — ReAct loop, no prompts, bounded by max_steps.
tqcli --stop-trying-to-control-everything-and-just-let-go \
    chat --max-agent-steps 8 \
    --prompt "Read tests/test_basic.py, fix any failing assert, re-run pytest."
```

Under the hood:

- The orchestrator (`tqcli/core/agent_orchestrator.py`) intercepts
  `<tool_call>` / `<staged_tool_call>` blocks emitted by the local model.
- Tool schemas (`tq-file-read`, `tq-file-write`, `tq-terminal-exec`,
  `tq-interactive-prompt`) are injected as OpenAI-style JSON Schema into
  the system prompt only when a mode is active — manual mode injects
  nothing.
- Observations are threaded back as `role=user` turns prefixed with
  `Observation:\n`, head+tail-truncated to 1,000 chars so shell errors
  (which land at the end of stderr) survive the cap.

See [`docs/architecture/agent_orchestrator.md`](../architecture/agent_orchestrator.md)
for the full diagram and safety model.

---

## Where to look next

- Architecture — [`docs/architecture/`](../architecture/README.md)
- Integration test report — [`tests/integration_reports/turboquant_kv_comparison_report.md`](../../tests/integration_reports/turboquant_kv_comparison_report.md)
- Full lifecycle plan — [`tests/integration_reports/turboquant_kv_comparison_test_cases.md`](../../tests/integration_reports/turboquant_kv_comparison_test_cases.md)
- GitHub issues — <https://github.com/ithllc/tqCLI/issues?state=closed>
