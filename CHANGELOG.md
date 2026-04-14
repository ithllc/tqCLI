# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
