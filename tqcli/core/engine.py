"""Inference engine abstraction layer."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class InferenceStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_eval_time_s: float = 0.0
    completion_time_s: float = 0.0
    tokens_per_second: float = 0.0
    time_to_first_token_s: float = 0.0
    total_time_s: float = 0.0


@dataclass
class ChatMessage:
    role: str  # "system", "user", "assistant"
    content: str
    images: list[str] | None = None  # list of file paths for image inputs
    audio: list[str] | None = None  # list of file paths for audio inputs


@dataclass
class CompletionResult:
    text: str
    stats: InferenceStats
    model_id: str
    finish_reason: str = "stop"


class InferenceEngine(ABC):
    """Abstract base class for inference backends."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        ...

    @property
    @abstractmethod
    def is_available(self) -> bool:
        ...

    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> None:
        ...

    @abstractmethod
    def unload_model(self) -> None:
        ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        ...

    @abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        ...

    @abstractmethod
    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        ...

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        ...

    def _compute_tps(self, tokens: int, elapsed: float) -> float:
        if elapsed <= 0:
            return 0.0
        return tokens / elapsed
