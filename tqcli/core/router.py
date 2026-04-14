"""Smart model router — classifies prompts and dispatches to the best available model.

Routing strategy:
1. Classify the user's prompt into a TaskDomain using keyword/pattern heuristics.
2. Rank available models by their strength_score for that domain.
3. Filter out models that exceed the hardware budget.
4. Pick the top-ranked model that fits.

If only one model is loaded, skip classification and use it directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from tqcli.core.model_registry import ModelProfile, ModelRegistry, TaskDomain


# Keyword lists for classification. Compiled once at import time.
_CODING_PATTERNS = re.compile(
    r"\b("
    r"code|coding|program|function|class|method|debug|refactor|implement|"
    r"python|javascript|typescript|rust|golang|java|c\+\+|html|css|sql|"
    r"api|endpoint|regex|algorithm|data structure|git|commit|merge|"
    r"bug|error|exception|traceback|stack trace|syntax|compile|"
    r"test|unittest|pytest|jest|lint|format|"
    r"import|export|module|package|library|framework|"
    r"variable|loop|condition|return|async|await"
    r")\b",
    re.IGNORECASE,
)

_MATH_PATTERNS = re.compile(
    r"\b("
    r"math|calculate|equation|formula|integral|derivative|"
    r"algebra|geometry|calculus|statistics|probability|"
    r"matrix|vector|tensor|eigenvalue|polynomial|"
    r"sum|product|factorial|logarithm|exponent|"
    r"proof|theorem|lemma|conjecture|"
    r"optimize|minimize|maximize|gradient"
    r")\b",
    re.IGNORECASE,
)

_REASONING_PATTERNS = re.compile(
    r"\b("
    r"reason|reasoning|analyze|analysis|explain why|think step|"
    r"logic|logical|deduce|infer|conclude|"
    r"compare|contrast|evaluate|assess|critique|"
    r"pros and cons|tradeoff|trade-off|"
    r"cause|effect|implication|consequence|"
    r"strategy|plan|approach|methodology|"
    r"research|investigate|examine|study"
    r")\b",
    re.IGNORECASE,
)

_CREATIVE_PATTERNS = re.compile(
    r"\b("
    r"write|story|poem|creative|fiction|narrative|"
    r"imagine|describe|illustrate|"
    r"dialogue|character|scene|plot|"
    r"blog|article|essay|copywriting|"
    r"metaphor|analogy|style|tone|voice"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class RouteDecision:
    model: ModelProfile
    domain: TaskDomain
    confidence: float  # 0.0 - 1.0
    reason: str
    use_thinking: bool = False  # whether to enable Qwen 3 thinking mode


# Domains that benefit from thinking/reasoning chains
_THINKING_DOMAINS = {TaskDomain.CODING, TaskDomain.MATH, TaskDomain.REASONING}


def classify_prompt(text: str) -> tuple[TaskDomain, float]:
    """Classify a prompt into a task domain with a confidence score."""
    text_lower = text.lower()

    scores: dict[TaskDomain, float] = {
        TaskDomain.CODING: 0.0,
        TaskDomain.MATH: 0.0,
        TaskDomain.REASONING: 0.0,
        TaskDomain.CREATIVE: 0.0,
        TaskDomain.GENERAL: 0.3,  # baseline — general is the fallback
        TaskDomain.INSTRUCTION: 0.2,
    }

    coding_hits = len(_CODING_PATTERNS.findall(text_lower))
    math_hits = len(_MATH_PATTERNS.findall(text_lower))
    reasoning_hits = len(_REASONING_PATTERNS.findall(text_lower))
    creative_hits = len(_CREATIVE_PATTERNS.findall(text_lower))

    if coding_hits:
        scores[TaskDomain.CODING] = min(0.5 + coding_hits * 0.15, 1.0)
    if math_hits:
        scores[TaskDomain.MATH] = min(0.5 + math_hits * 0.15, 1.0)
    if reasoning_hits:
        scores[TaskDomain.REASONING] = min(0.4 + reasoning_hits * 0.15, 1.0)
    if creative_hits:
        scores[TaskDomain.CREATIVE] = min(0.4 + creative_hits * 0.15, 1.0)

    # Code blocks are a strong signal
    if "```" in text or "def " in text or "class " in text:
        scores[TaskDomain.CODING] = max(scores[TaskDomain.CODING], 0.85)

    # Direct instruction patterns boost instruction domain
    if any(
        text_lower.startswith(p)
        for p in ("summarize", "translate", "list", "explain", "tell me", "what is", "how to")
    ):
        scores[TaskDomain.INSTRUCTION] = max(scores[TaskDomain.INSTRUCTION], 0.6)

    best_domain = max(scores, key=lambda d: scores[d])
    confidence = scores[best_domain]
    return best_domain, confidence


class ModelRouter:
    """Routes prompts to the best available model based on task classification."""

    def __init__(self, registry: ModelRegistry, ram_mb: int = 0, vram_mb: int = 0):
        self.registry = registry
        self.ram_mb = ram_mb
        self.vram_mb = vram_mb
        self._override_model: str | None = None

    def set_override(self, model_id: str | None) -> None:
        """Force all prompts to a specific model, bypassing routing."""
        self._override_model = model_id

    def route(self, prompt: str) -> RouteDecision:
        available = self.registry.get_available_models()
        if not available:
            raise RuntimeError(
                "No models available. Download a model first with: tqcli model pull <model_id>"
            )

        # Override takes priority
        if self._override_model:
            profile = self.registry.get_profile(self._override_model)
            if profile and profile.local_path:
                return RouteDecision(
                    model=profile,
                    domain=TaskDomain.GENERAL,
                    confidence=1.0,
                    reason=f"User override: {profile.display_name}",
                )

        # Single model — skip classification
        if len(available) == 1:
            model = available[0]
            return RouteDecision(
                model=model,
                domain=TaskDomain.GENERAL,
                confidence=1.0,
                reason=f"Only available model: {model.display_name}",
            )

        # Classify and rank
        domain, confidence = classify_prompt(prompt)

        # Get models ranked by strength for this domain, filtered by hardware
        candidates = []
        for model in self.registry.get_models_for_domain(domain):
            if self.registry.fits_hardware(model, self.ram_mb, self.vram_mb):
                candidates.append(model)

        if not candidates:
            # Fall back to any available model
            best = available[0]
            thinking = best.supports_thinking and domain in _THINKING_DOMAINS
            return RouteDecision(
                model=best,
                domain=domain,
                confidence=confidence * 0.5,
                reason=f"No optimal model fits hardware; falling back to {best.display_name}",
                use_thinking=thinking,
            )

        best = candidates[0]
        score = best.strength_scores.get(domain.value, 0.0)
        # Enable thinking mode for complex domains if the model supports it
        thinking = best.supports_thinking and domain in _THINKING_DOMAINS
        thinking_note = " +thinking" if thinking else ""
        return RouteDecision(
            model=best,
            domain=domain,
            confidence=confidence,
            reason=(
                f"Routed to {best.display_name} — best for {domain.value} "
                f"(score: {score:.2f}, confidence: {confidence:.2f}){thinking_note}"
            ),
            use_thinking=thinking,
        )
