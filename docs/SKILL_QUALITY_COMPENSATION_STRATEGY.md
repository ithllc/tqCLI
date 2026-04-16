# Skill-Based Quality Compensation for Constrained Hardware

When tqCLI runs on hardware that forces a smaller model (e.g., E2B on 4 GB VRAM instead of E4B on 8 GB), the skills pipeline can recover most of the quality gap through five complementary strategies.

## The Gap

| Domain | E2B (2.3B) | E4B (4.5B) | Delta |
|--------|-----------|-----------|-------|
| Coding | 0.40 | 0.55 | -27% |
| Reasoning | 0.50 | 0.65 | -23% |
| General | 0.58 | 0.72 | -19% |
| Math | 0.45 | 0.58 | -22% |
| Creative | 0.52 | 0.65 | -20% |
| Instruction | 0.55 | 0.70 | -21% |

Average gap: ~22%. The strategies below can recover 15-25% depending on task type.

## Strategy 1: Cross-Engine Model Cascade

**Key insight:** Gemma 4 E4B Q4_K_M GGUF requires only **3 GB VRAM** on llama.cpp — it already fits on 4 GB hardware.

tqCLI supports both llama.cpp and vLLM engines simultaneously. The smart router can cascade between them:

| Task Type | Primary Model | Engine | Fallback |
|-----------|--------------|--------|----------|
| Multimodal (image/audio) | E2B BF16 | vLLM | Handoff to frontier |
| Text coding | E4B Q4_K_M | llama.cpp | Qwen3-Coder if available |
| Text reasoning | E4B Q4_K_M | llama.cpp | E2B + thinking mode |
| Text general | E4B Q4_K_M | llama.cpp | E2B |

**Impact:** For text-only tasks, this eliminates the gap entirely — the user gets E4B quality at 7+ tok/s via llama.cpp. The E2B vLLM path is reserved for multimodal tasks that require the vision/audio towers.

**How to configure:**
```bash
# Download both models
tqcli model pull gemma-4-e4b-it-Q4_K_M    # llama.cpp, text
tqcli model pull gemma-4-e2b-it-vllm       # vLLM, multimodal

# Router automatically picks the best available model per task
tqcli chat
```

The router's `classify_prompt()` detects task domain, `get_models_for_domain()` ranks by strength scores, and `fits_hardware()` filters by VRAM. With both models downloaded, the router picks E4B for text tasks and E2B for multimodal.

## Strategy 2: Thinking Mode Activation

E2B supports Gemma 4's native thinking mode (`<|channel>thought...<channel|>`). When enabled, the model performs chain-of-thought reasoning before answering.

The router already enables thinking for coding, math, and reasoning tasks automatically:

```python
_THINKING_DOMAINS = {TaskDomain.CODING, TaskDomain.MATH, TaskDomain.REASONING}
```

Thinking mode typically improves accuracy by 10-20% on reasoning-heavy tasks, partially closing the E2B→E4B quality gap in the domains where it matters most.

**Impact:** +10-20% accuracy on coding/math/reasoning. Combined with strategy 1 (use E4B GGUF for these tasks), this can exceed E4B's base scores.

## Strategy 3: Frontier Model Handoff

When local inference quality is insufficient for a critical task, `/tq-handoff-generator` packages the conversation context and task into a portable markdown file for transfer to a frontier model CLI:

- **Claude Code** — `claude @tqcli_handoff.md`
- **Gemini CLI** — paste context
- **Aider** — `aider --message-file tqcli_handoff.md`

The performance monitor (`tqcli/core/performance.py`) tracks tok/s and can trigger automatic handoff suggestions when inference drops below the 5 tok/s threshold.

**Impact:** Unlimited quality ceiling. Complex tasks that exceed E2B's capabilities get delegated to frontier models with full context preservation.

## Strategy 4: Domain-Specific Routing to Specialist Models

The model registry supports multiple model families. On hardware with sufficient RAM (32 GB+), users can download specialist models alongside Gemma 4:

| Domain | Specialist | Score | vs E2B Gain |
|--------|-----------|-------|-------------|
| Coding | Qwen3-Coder 30B-A3B Q4_K_M | 0.90 | +125% |
| Math | Qwen3 4B (thinking mode) | 0.70 | +56% |
| General | E4B Q4_K_M (llama.cpp) | 0.72 | +24% |

The router automatically selects the highest-scoring model that fits the hardware for each domain. No configuration needed — just download and the router handles it.

```bash
tqcli model pull qwen3-coder-30b-a3b-instruct-Q4_K_M
tqcli model pull gemma-4-e4b-it-Q4_K_M
tqcli chat  # Router picks best model per prompt
```

## Strategy 5: TurboQuant KV for Context Quality

Longer context = better quality. TurboQuant KV compression (4.6x) extends effective context from ~256 tokens to ~1,180 tokens on 4 GB VRAM. Longer context helps the model:

- Maintain coherence in multi-turn conversations
- Process longer code snippets
- Reference more document context

This indirectly improves quality scores because many benchmark failures on small models come from context truncation, not model capability.

## Combined Impact Matrix

For a 4 GB VRAM system (RTX A2000 Laptop, WSL2) with 32 GB RAM:

| Task Type | Without Skills | With Skills Pipeline | Method |
|-----------|---------------|---------------------|--------|
| Text coding | E2B: 0.40 | E4B GGUF: 0.55, or Qwen3-Coder: 0.90 | Cross-engine cascade |
| Text reasoning | E2B: 0.50 | E4B GGUF + thinking: ~0.75 | Cascade + thinking |
| Multimodal | E2B: 0.58 | E2B + thinking: ~0.65 | Thinking mode |
| Complex/critical | E2B: varies | Frontier model | Handoff |

**Net result:** The skills pipeline gives 4 GB hardware effective quality scores that match or exceed E4B's base scores for most tasks, with frontier handoff as an escape valve.

## Scaling to Better Hardware

These strategies scale naturally with hardware:

| VRAM | Primary Model | Skills Still Useful? |
|------|--------------|---------------------|
| 4 GB | E2B vLLM + E4B GGUF | Yes: cascade, thinking, handoff |
| 8 GB | E4B vLLM + E4B GGUF | Yes: thinking, specialist routing |
| 12 GB | 26B MoE GGUF | Yes: specialist routing, handoff |
| 24 GB+ | 31B Dense + Qwen3-Coder | Yes: handoff for frontier tasks |

The skills pipeline adds value at every hardware tier — it's not a workaround for weak hardware, it's a general quality multiplier.
