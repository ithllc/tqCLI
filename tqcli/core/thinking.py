"""Unified thinking mode abstraction for Qwen 3 and Gemma 4.

Two model families, two thinking formats, one interface.

Qwen 3:
  - Enable: set enable_thinking=True in tokenizer template
  - Thinking blocks: <think>...</think>
  - Per-turn toggle: /think, /no_think appended to user message
  - Inference settings: temperature=0.6, top_p=0.95, top_k=20

Gemma 4:
  - Enable: add <|think|> to the system instruction
  - Thinking blocks: <|channel>thought\n...\n<channel|>
  - Depth control: system instruction text like "think briefly" reduces ~20%
  - Critical: strip thoughts from history between turns (except during tool calls)
  - Uses <|turn>, <turn|> for dialogue structure

Sources:
  - Qwen 3: qwenlm.github.io/blog/qwen3
  - Gemma 4: ai.google.dev/gemma/docs/core/prompt-formatting-gemma4
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class ThinkingFormat(Enum):
    NONE = "none"
    QWEN3 = "qwen3"       # <think>...</think>
    GEMMA4 = "gemma4"      # <|channel>thought...<channel|>


@dataclass
class ThinkingConfig:
    """Configuration for a model's thinking mode."""

    format: ThinkingFormat
    enabled: bool = False
    depth: str = "default"  # "default", "low", "high" — Gemma 4 supports adaptive depth

    @property
    def is_active(self) -> bool:
        return self.format != ThinkingFormat.NONE and self.enabled


def detect_thinking_format(model_family: str) -> ThinkingFormat:
    """Determine which thinking format a model family uses."""
    family = model_family.lower().replace("-", "")
    if family.startswith("qwen3"):
        return ThinkingFormat.QWEN3
    if family.startswith("gemma4"):
        return ThinkingFormat.GEMMA4
    return ThinkingFormat.NONE


def build_system_prompt_with_thinking(
    base_prompt: str, config: ThinkingConfig
) -> str:
    """Inject thinking mode tokens into the system prompt."""
    if not config.is_active:
        return base_prompt

    if config.format == ThinkingFormat.QWEN3:
        # Qwen 3: thinking is controlled via enable_thinking param in tokenizer,
        # but we can also hint via system prompt
        if config.depth == "low":
            return base_prompt + "\nThink briefly before answering."
        return base_prompt + "\nThink step by step before answering."

    if config.format == ThinkingFormat.GEMMA4:
        # Gemma 4: <|think|> token in system instruction enables thinking
        think_token = "<|think|>"
        if config.depth == "low":
            return f"{think_token}Think briefly. {base_prompt}"
        return f"{think_token}{base_prompt}"

    return base_prompt


# ── Thinking block extraction and stripping ───────────────────────────

# Qwen 3: <think>...</think>
_QWEN3_THINK_PATTERN = re.compile(r"<think>(.*?)</think>\s*", re.DOTALL)
_QWEN3_THINK_OPEN = re.compile(r"<think>(?!.*</think>)", re.DOTALL)

# Gemma 4: <|channel>thought\n...\n<channel|>
_GEMMA4_THINK_PATTERN = re.compile(
    r"<\|channel>thought\s*\n(.*?)\n?<channel\|>\s*", re.DOTALL
)
_GEMMA4_THINK_OPEN = re.compile(r"<\|channel>thought(?!.*<channel\|>)", re.DOTALL)


def strip_thinking_blocks(text: str, fmt: ThinkingFormat) -> str:
    """Remove completed thinking blocks from text for clean display."""
    if fmt == ThinkingFormat.QWEN3:
        return _QWEN3_THINK_PATTERN.sub("", text)
    if fmt == ThinkingFormat.GEMMA4:
        return _GEMMA4_THINK_PATTERN.sub("", text)
    return text


def extract_thinking(text: str, fmt: ThinkingFormat) -> tuple[str, str]:
    """Extract thinking content and clean response separately.

    Returns (thinking_text, clean_response).
    """
    if fmt == ThinkingFormat.QWEN3:
        match = _QWEN3_THINK_PATTERN.search(text)
        if match:
            thinking = match.group(1).strip()
            clean = _QWEN3_THINK_PATTERN.sub("", text).strip()
            return thinking, clean
    elif fmt == ThinkingFormat.GEMMA4:
        match = _GEMMA4_THINK_PATTERN.search(text)
        if match:
            thinking = match.group(1).strip()
            clean = _GEMMA4_THINK_PATTERN.sub("", text).strip()
            return thinking, clean
    return "", text


def is_inside_thinking_block(text: str, fmt: ThinkingFormat) -> bool:
    """Check if the text ends inside an unclosed thinking block (streaming)."""
    if fmt == ThinkingFormat.QWEN3:
        return bool(_QWEN3_THINK_OPEN.search(text))
    if fmt == ThinkingFormat.GEMMA4:
        return bool(_GEMMA4_THINK_OPEN.search(text))
    return False


def strip_thinking_from_history(
    messages: list[dict], fmt: ThinkingFormat
) -> list[dict]:
    """Strip thinking blocks from assistant messages in history.

    Gemma 4 requires this between turns. Qwen 3 also benefits from it
    to keep context clean.
    """
    cleaned = []
    for msg in messages:
        if msg.get("role") == "assistant":
            content = strip_thinking_blocks(msg["content"], fmt)
            cleaned.append({**msg, "content": content})
        else:
            cleaned.append(msg)
    return cleaned
