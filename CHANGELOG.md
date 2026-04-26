# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-04-26

### Added

- **TurboQuant fork wheels are now installable.** `pip install
  turboquant-cli[llama-tq]` pulls cibuildwheel matrix wheels for
  `llama-cpp-python-turboquant` from PyPI (Linux/macOS/Windows × Py
  3.10–3.12 × CPU/CUDA/Metal). `pip install turboquant-cli[vllm-tq]`
  pulls `vllm-turboquant` from the GitHub Release on
  `tqcli/vllm-turboquant` (Ampere/Ada/Hopper, single CUDA 12.8 build).
  Blackwell hardware (sm_100 / sm_120 / sm_121) opts in with
  `[vllm-tq-blackwell]`, which targets the dedicated CUDA 13.0 build
  with a PTX hedge for Rubin
  (`TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"`).
- **Engine Auditor** (`tqcli/core/engine_auditor.py`) — runs at every
  CLI start, detects fork-vs-upstream via the `TURBOQUANT_BUILD` /
  `TURBOQUANT_ENABLED` sentinels, and emits a yellow Rich panel with
  the exact `pip install` command when the user has capable hardware
  but upstream packages. Stays silent on capable+fork installs and on
  hardware that cannot run TurboQuant. Suppress with
  `TQCLI_SUPPRESS_AUDIT=1`. In `--json` mode the audit is emitted as a
  one-line stderr metadata blob (`{"engine_audit": {...}}`) instead of
  a Rich panel — `--json` stdout stays parseable. Exposes
  `engine_auditor.get_status()` for future agent tools.
- **Stderr-ordering contract** (`tqcli/cli.py`) — in agent modes
  (`--ai-tinkering` / unrestricted), `console.file.flush()` is called
  BEFORE `InteractiveSession` constructs the `AgentOrchestrator`, so
  the audit panel cannot interleave with the orchestrator's first
  `<tool_call>` stream chunk. Asserted in
  `tests/test_engine_auditor.py::test_render_then_flush_finishes_before_orchestrator_first_chunk`.
- **`scripts/community_verify.sh`** — opt-in verification helper for
  friend-of-the-project Mac verifiers. Prints an explicit consent
  manifest BEFORE collecting any data; refuses to run on declined
  consent. Two modes: `--auto-report` (uses `gh` CLI; never reads or
  ships tokens) and `--manual` (prints a paste-ready markdown block).
  Issue template at `.github/ISSUE_TEMPLATE/community_verify_0_7_0.yml`,
  nightly intake workflow at
  `.github/workflows/community_verify_collect.yml` opens a PR
  scraping the labeled issues into
  `tests/integration_reports/community_verification/0.7.0/`.
- **GitHub Sponsors** — `.github/FUNDING.yml` routes to `ithllc`
  (Ivey Technology Holdings LLC, the legal entity behind tqCLI).
- **Maintainer runbook** — `docs/contributing/RELEASING_WHEELS.md`
  documents the GCP on-demand wheel-build flow for `vllm-turboquant`
  (B200 / sm_100 / sm_121 verified; ~$14/round trip on-demand).
- **`docs/architecture/inference_engines.md`** documents the sentinel
  attributes and the Engine Auditor's role.
- **`docs/architecture/agent_orchestrator.md`** cross-references the
  Engine Auditor's stderr-ordering contract.

### Changed

- **PyPI distribution name is now `turboquant-cli`.** The `tqcli` slug
  was unavailable on PyPI (taken by an unrelated project, TranQuant);
  the import name is unchanged (`import tqcli` still works) — the
  dateutil pattern. `pyproject.toml::project.name` updated; both
  console scripts (`tqcli`, `tq`) unchanged.
- **`pyproject.toml` extras replaced.** `[llama]` / `[vllm]` / `[all]`
  → `[llama-tq]` / `[vllm-tq]` / `[all]` plus the new
  `[vllm-tq-blackwell]`. The old `[llama]` / `[vllm]` keys no longer
  install anything — copy/paste from pre-0.7.0 docs will fail loudly.
  macOS users must install `[llama-tq]` directly; `[all]` has no
  Darwin wheel path for vLLM.
- **Repo URLs migrated to `github.com/tqcli/`** — the `ithllc/` URLs
  redirect permanently; existing clones keep working but new docs
  reference the new org.
- **License: Apache-2.0** at the umbrella package level (matching
  TurboQuant lead author Zandieh's QJL repo and the inherited
  `vllm-turboquant` license). `LICENSE`, `NOTICE`, and `CITATION.cff`
  were added in 0.6.2's pre-flight; this release ships against them.
- **Authoritative architecture target** for the CUDA 13.0 build:
  `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"` —
  Ampere + Ada + Hopper + DC Blackwell + consumer Blackwell + DGX
  Spark/GB10 + PTX hedge for Rubin. CUDA 12.8 cannot compile sm_121.

### Removed

- Upstream `llama-cpp-python` and `vllm` dependency paths. tqCLI now
  ships exclusively against the TurboQuant forks. Users who need
  upstream behavior can `pip uninstall llama-cpp-python-turboquant
  vllm-turboquant && pip install llama-cpp-python vllm` — but the
  Engine Auditor will then warn at every startup until they suppress
  it.

## [0.6.2] - 2026-04-19

### Added

- **LlamaForCausalLM, MistralForCausalLM, Phi3ForCausalLM capture wrappers**
  ([#31](https://github.com/ithllc/tqCLI/issues/31)). `_CAPTURE_INSTALLERS` in
  `tqcli/core/kv_metadata_generator.py` now registers four architectures
  (Qwen3 + Llama 3 + Mistral + Phi-3). Llama and Mistral share a pattern
  (separate q/k/v projections, RoPE, no q_norm/k_norm); Phi-3 uses a fused
  `qkv_proj` that the wrapper slices by (num_attention_heads × head_dim,
  num_key_value_heads × head_dim, num_key_value_heads × head_dim). All three
  validated end-to-end against `HuggingFaceTB/SmolLM2-135M-Instruct`,
  `Locutusque/TinyMistral-248M`, and `microsoft/Phi-3-mini-4k-instruct`:
  calibration emits `turboquant_kv.json` that loads cleanly via
  `vllm.v1.attention.ops.turboquant_metadata.load_turboquant_metadata`.
- **12 architecture-coverage tests** in `tests/test_kv_metadata_archs.py`
  covering registry contents, install/restore lifecycle, head_dim derivation,
  live precondition checks, metadata-shape invariants, and vLLM loader
  round-trip. Live-model tests skip gracefully when the model dir is absent.

### Changed

- **`_extract_architecture_params` now derives `head_dim` from
  `hidden_size // num_attention_heads`** when the config omits an explicit
  `head_dim` field. This pattern is the norm for Llama 3, Mistral, Phi-3, and
  SmolLM2 configs (all three test models required it). Qwen3 / Gemma 4
  configs that set `head_dim` explicitly are unaffected.
- Extracted `_accumulate_kv_scores` shared helper — the Qwen3 wrapper's
  inline accumulator is now the common path reused by Llama/Mistral/Phi-3.
  No behavior change for Qwen3 calibration; existing tests pass unchanged.

## [0.6.1] - 2026-04-19

### Added

- **TurboQuant KV metadata auto-calibration** ([#27](https://github.com/ithllc/tqCLI/issues/27))
  for models that ship without `turboquant_kv.json`. New module
  `tqcli/core/kv_metadata_generator.py` runs an activation-based calibration
  that mirrors the fork's own `build_turboquant_outlier_masks` reference
  (mean-squared activation per-kv-head per-channel, top-k outlier indices,
  sorted). Captures post-RoPE K and V via a monkey-patched
  `Qwen3Attention.forward`; accumulates second moments online in fp64 on CPU.
  Auto-runs on first vLLM load when `kv_cache_dtype.startswith("turboquant")`
  and metadata is missing. Architecture registry + precondition checks refuse
  AWQ/GPTQ sources, variable-head-dim models (Gemma 4), non-16-aligned
  `head_dim`, and unregistered architectures with clear reason strings.
- **Paragraph-length calibration corpus** ([#28](https://github.com/ithllc/tqCLI/issues/28)).
  Replaces 30 short-sentence prompts (~525 Qwen3 tokens) with 30 domain-diverse
  paragraph prompts (~5,100 Qwen3 tokens). Raises generator default
  `max_seq_len` from 512 to 1024. New unit test
  `tests/test_kv_metadata_corpus.py` enforces `MIN_OBSERVED_TOKENS=5_000` via
  real Qwen3 tokenizer, preventing corpus regressions.
- **`tqcli model calibrate-kv <model-id>`** ([#29](https://github.com/ithllc/tqCLI/issues/29))
  — explicit CLI for pre-warming TurboQuant metadata. Flags: `--recipe`
  (turboquant25/turboquant35), `--force`. Exits 2/3/4 with distinct messages
  for different refuse paths. Auto-calibration on vllm load remains as a
  fallback for users who skip the pre-warm step. Unit-tested via
  `click.testing.CliRunner` (8 tests, `tests/test_cli_calibrate_kv.py`).
- **Perplexity validation gate** ([#30](https://github.com/ithllc/tqCLI/issues/30))
  — new opt-in integration test `tests/test_kv_ppl_validation.py`. Loads
  `qwen3-4b-vllm` under `kv_cache_dtype=auto` then `turboquant35`, computes
  forced-sequence PPL via `SamplingParams(prompt_logprobs=1)` over a
  10-prompt corpus, and asserts ratio ≤ 1.05 to detect silent quality
  collapse from bad outlier indices. Gated by `TQCLI_PPL_GATE=1`; runs in
  ~161 s on the 4 GB VRAM reference box. Current calibrated metadata
  scores ratio **0.9997** (baseline PPL 6.8893, turboquant35 PPL 6.8872).

### Changed

- `tests/test_integration_agent_functional.py` — removed the
  `kv_quant_choice="none"` workaround for Qwen 3 4B on vLLM. Now loads
  `qwen3-4b-vllm` with `turbo3` and relies on the auto-calibrate-on-load
  path. Tests T1_vq and T4_vq advance from 5/5 to 6/6 steps: the new 6th
  assertion is `turboquant_kv_active`, which verifies the vLLM runtime
  actually wired `kv_cache_dtype=turboquant35` (not a silent fallback).
  Full E2E: 11/11 functional + 4/4 data-point PASS; zero `kv:none`
  annotations remain in `agent_modes_functional_report.md`.

### Closed (tracking only — not implemented)

- [#31](https://github.com/ithllc/tqCLI/issues/31) — Llama 3 / Mistral / Phi-3
  architecture wrappers. Deferred until the target models are available
  locally for E2E validation. Shipping untested wrappers was rejected by
  Gemini review as risking silent breakage.
- [#32](https://github.com/ithllc/tqCLI/issues/32) — Gemma 4 per-layer
  `head_dim` (256 sliding / 512 global). Requires metadata schema v2 and
  runtime changes in `vllm-turboquant` fork itself; tqCLI's preconditions
  correctly refuse variable-head-dim calibration today, and Gemma 4's 28
  sliding-window layers already benefit from TurboQuant with bf16 fallback
  on the 7 global-attention layers. Filed against fork maintainer.
- [#33](https://github.com/ithllc/tqCLI/issues/33) — Multi-head Latent
  Attention (DeepSeek V3) compatibility. Closed as **won't fix**: TurboQuant's
  per-head Hadamard rotation is mathematically incompatible with MLA's
  low-rank latent compression + per-head up-projection. Recommend `kv_cache_dtype=fp8`
  for MLA models (supported by vLLM upstream).

## [0.6.0] - 2026-04-18

### Added

- **Tri-state agentic autonomy** for `tqcli chat` (closes #25 once filed).
  Introduces a middleware layer (`tqcli/core/agent_orchestrator.py`) that
  intercepts streamed LLM output and routes tool calls through one of three
  modes:
  - **manual** (default) — no tool schemas injected; unchanged behavior.
    `tools` list passed to the LLM is explicitly empty, so existing CI and
    JSON-stdout pipelines stay bit-for-bit identical.
  - **ai_tinkering** (`--ai-tinkering`) — model emits
    `<staged_tool_call>{...}</staged_tool_call>`; the CLI halts, shows name +
    JSON args, and asks `[Y/n/Edit]`. Approved tools run, observations are
    threaded back as a new user turn, and the loop continues so the model can
    chain follow-up calls without a fresh prompt. Safe-marked tools auto-run.
  - **unrestricted** (`--stop-trying-to-control-everything-and-just-let-go`)
    — model emits `<tool_call>{...}</tool_call>` and the ReAct loop fires
    tools immediately, bounded by `--max-agent-steps` (default 10).
- **Core agent tools** (`tqcli/core/agent_tools.py`): `tq-file-read`,
  `tq-file-write`, `tq-terminal-exec`, `tq-interactive-prompt`. Each tool
  emits an OpenAI-compatible JSON Schema and carries a `safety` classifier
  (`safe` / `actionable`) that drives the tinkering confirmation gate.
- **Agent tests** (`tests/test_agent_orchestrator.py`): Phase 4 rubrics plus
  regressions for nested-JSON argument parsing, multi-step tinkering chains,
  denial behavior, and `max_steps` bounds.

### Changed

- `InteractiveSession` accepts `agent_mode` and `max_agent_steps` so the chat
  loop short-circuits to the orchestrator in agentic modes. Manual mode path
  is untouched.
- `FileReadTool.safety` is classified `actionable` (not `safe`) in tinkering
  mode so the CLI gates reads of secrets like `~/.ssh`, `.env`, or
  `/etc/shadow`. Under `--stop-trying-to-control-everything-and-just-let-go`
  the user has explicitly opted in and the gate is bypassed.

## [0.5.0] - 2026-04-17

### Added

- **`tqcli skill generate`** — AI Skills Builder (#23). Reads a PRD + Technical
  Plan and asks the currently configured local LLM to emit a complete skill
  scaffold (`SKILL.md` + Python script(s)) into `~/.tqcli/skills/<name>/`.
  Runs TurboQuant-aware on llama.cpp and vLLM, parses `<file path="">`-tagged
  output (plus a `FILE:` fenced-block fallback), validates Python with
  `ast.parse()` before writing, and defaults to an interactive review prompt.
  New module `tqcli/core/skill_generator.py` and prompt template
  `tqcli/prompts/skill_generation_prompt.md`.
- **Headless chat** on `tqcli chat` (#24): `--prompt`, `--image` (repeatable),
  `--audio` (repeatable), `--messages`, `--json`, `--max-tokens`. When `--json`
  is set, the result is emitted as a structured object on stdout
  (`model/engine/response/thinking/usage/performance/metadata`) and all other
  chatter is routed to stderr; exit non-zero on failure.
- **vLLM multimodal pass-through** (#24): `VllmBackend._messages_to_dicts`
  now emits the content-list form so the tokenizer inserts the Gemma 4
  `<start_of_image>` placeholder, and `chat()` passes PIL images to
  `self._llm.generate()` via `multi_modal_data={"image": [...]}`. Preserves
  `cpu_offload_gb`, `kv_cache_dtype`, `enable_turboquant`, and `enforce_eager`
  wiring.
- New helper `extract_thinking_content()` in `tqcli/core/thinking.py`.
- Tests: `tests/test_skill_generator.py`, `tests/test_headless_chat.py`.

### Changed

- `InteractiveSession.chat_turn()` gains `show_ui` and `max_tokens` parameters
  and surfaces `last_response` / `last_stats` so a single code path covers both
  the interactive REPL and headless single-shot mode.

### Fixed

- `--json` stdout is now strict-parser clean ([#25](https://github.com/ithllc/tqCLI/issues/25)).
  Previously the Rich console was routed to stderr but third-party libraries
  (`vllm`, `torch`, `bitsandbytes`, `transformers`, `accelerate`, `PIL`,
  `urllib3`) still wrote `INFO`/`WARNING` records to stdout, and `tqdm`
  progress bars (e.g. `Processed prompts: ...`) rendered to stdout. Added
  `tqcli.ui.console.setup_json_logging()` which, before engine import:
  sets `VLLM_CONFIGURE_LOGGING=0`, `VLLM_LOGGING_LEVEL=ERROR`, and
  `TQDM_DISABLE=1` (inherited by vLLM `EngineCore` subprocesses); forces the
  root logger to `sys.stderr` via `logging.basicConfig(force=True, ...)`;
  walks `logging.Logger.manager.loggerDict` and clears any handlers already
  installed on third-party loggers; and globally patches `tqdm.tqdm` to
  default `file=sys.stderr`. Verified: `tqcli chat ... --json | jq -e
  '.response'` exits 0 on both llama.cpp and vLLM paths.

## [0.3.1] - 2026-04-14

### Changed

- **Unified thinking mode**: Gemma 4 and Qwen 3 now both have proper thinking control
  - Gemma 4: `<|think|>` token in system instruction, `<|channel>thought...<channel|>` blocks
  - Qwen 3: `<think>...</think>` blocks with `enable_thinking` control
  - New `tqcli/core/thinking.py` module handles both formats transparently
  - Router detects model family and applies correct thinking format automatically
  - System prompt injection: `<|think|>` for Gemma 4, hints for Qwen 3
  - Thinking depth control: "low"/"default"/"high" (Gemma 4 reduces ~20% at low)
  - History stripping: removes thinking blocks between turns (required by Gemma 4)
- All Gemma 4 models now have `supports_thinking=True`

### Added

- `tqcli/core/thinking.py` — unified thinking abstraction for both model families
- `ThinkingFormat` enum: `QWEN3`, `GEMMA4`, `NONE`
- `ThinkingConfig` with format, enabled, and depth fields
- Functions: `strip_thinking_blocks()`, `extract_thinking()`, `is_inside_thinking_block()`, `strip_thinking_from_history()`, `build_system_prompt_with_thinking()`
- 18 new tests for thinking module

## [0.3.0] - 2026-04-14

### Changed

- **Model registry rewrite**: replaced stale Qwen 2.5 and generic Gemma entries with accurate current models
  - **Google Gemma 4**: E2B (2.3B), E4B (4.5B), 26B MoE (3.8B active), 31B Dense — all with multimodal support, 128K-256K context
  - **Qwen 3**: 4B, 8B, 32B dense + 30B-A3B MoE — all with thinking mode, 32K-128K context
  - **Qwen3-Coder**: Coder-Next 80B MoE (3B active, 256K context), Coder-30B-A3B
- Updated strength scores from published benchmarks (Gemma 4 model card, Qwen 3 technical report, qwenlm.github.io)
- MoE models now track `active_params` separately from total parameter count

### Added

- **Qwen 3 thinking mode**: router auto-enables `<think>` reasoning for coding/math/reasoning tasks
- `/think` and `/no_think` per-message overrides in interactive chat
- Thinking block display: `<think>` blocks shown dimmed, cleaned from final output
- `ModelProfile.supports_thinking`, `active_params`, `multimodal` fields
- `tq-model-updater` skill for researching and updating model registry
- `tq-model-updater/scripts/check_models.py` to verify HuggingFace repo availability

## [0.2.0] - 2026-04-14

### Added

- **Multi-process mode**: shared inference server with multiple worker processes
  - `tqcli serve start/stop/status` — manage the inference server
  - `tqcli workers spawn N/list/stop` — manage worker processes
  - `tqcli chat --engine server` — connect to a running server
  - Automatic engine selection: vLLM (continuous batching + PagedAttention) on Linux with 8+ GB VRAM, llama.cpp server (sequential queue) everywhere else
  - Resource assessment before spawning workers — estimates VRAM/RAM usage
  - `tq-multi-process` skill with assessment and orchestration scripts
- **Unrestricted mode**: `--stop-trying-to-control-everything-and-just-let-go`
  - Bypasses resource guards, confirmation prompts, and feasibility checks
  - Equivalent to Claude Code's `--dangerously-skip-permissions` / Gemini CLI's `--yolo`
  - Audit logging remains active (always on)
- **Server client backend**: `ServerClientBackend` inference engine that connects to HTTP servers via OpenAI-compatible API with SSE streaming support
- New core modules: `server.py`, `server_client.py`, `multiprocess.py`, `unrestricted.py`
- `MultiProcessConfig` in config for server host, port, and max workers

## [0.1.0] - 2026-04-13

### Added

- Initial release of tqCLI (TurboQuant CLI)
- Cross-platform CLI with Click commands: `chat`, `system info`, `model list/pull/remove`, `benchmark`, `security audit`, `skills`, `handoff`, `config show/init`
- **Inference backends**: llama.cpp (via llama-cpp-python) and vLLM
- **Model registry**: 6 pre-configured profiles across 3 model families
  - Google Gemma 4 (12B, 27B)
  - Qwen2.5-Coder (7B, 32B)
  - Qwen2.5-Instruct (7B, 32B)
- **Smart router**: keyword-based prompt classification with domain-specific model ranking
- **Performance monitor**: real-time tokens/second tracking with configurable thresholds
- **Handoff system**: generates context files for Claude Code, Gemini CLI, Aider
- **Security layer**: venv isolation, environment detection (WSL2/container/bare-metal), resource guards, audit logging
- **Skills system**: 5 tqCLI skills (system-info, model-manager, benchmark, security-audit, handoff-generator)
- **Rich terminal UI**: tables, panels, streaming output, colored performance stats
- Hardware auto-detection: CPU, RAM, GPU/VRAM, Apple Silicon Metal, WSL2
- YAML configuration at `~/.tqcli/config.yaml`
- 11 passing tests covering all core modules
