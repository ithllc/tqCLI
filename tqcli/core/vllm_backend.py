"""vLLM backend for GPU-accelerated serving. Linux/WSL2 + NVIDIA GPU only."""

from __future__ import annotations

import time
from typing import Generator

from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)


class VllmBackend(InferenceEngine):
    def __init__(
        self,
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.80,
        quantization: str | None = None,
        load_format: str | None = None,
        tensor_parallel_size: int = 1,
        kv_cache_dtype: str = "auto",
        enforce_eager: bool = False,
        cpu_offload_gb: float = 0,
    ):
        self._max_model_len = max_model_len
        self._gpu_memory_utilization = gpu_memory_utilization
        self._quantization = quantization
        self._load_format = load_format
        self._tensor_parallel_size = tensor_parallel_size
        self._kv_cache_dtype = kv_cache_dtype
        self._enforce_eager = enforce_eager
        self._cpu_offload_gb = cpu_offload_gb
        self._llm = None
        self._model_name: str = ""
        self._tokenizer = None

    @classmethod
    def from_tuning_profile(cls, profile) -> "VllmBackend":
        """Create a VllmBackend pre-configured from a VllmTuningProfile."""
        return cls(
            max_model_len=profile.max_model_len,
            gpu_memory_utilization=profile.gpu_memory_utilization,
            quantization=profile.quantization,
            load_format=profile.load_format,
            tensor_parallel_size=profile.tensor_parallel_size,
            kv_cache_dtype=profile.kv_cache_dtype,
            enforce_eager=profile.enforce_eager,
            cpu_offload_gb=getattr(profile, "cpu_offload_gb", 0),
        )

    @property
    def engine_name(self) -> str:
        return "vllm"

    @property
    def is_available(self) -> bool:
        try:
            import vllm  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_path: str, **kwargs) -> None:
        from vllm import LLM

        params = {
            "model": model_path,
            "max_model_len": kwargs.get("max_model_len", self._max_model_len),
            "gpu_memory_utilization": kwargs.get(
                "gpu_memory_utilization", self._gpu_memory_utilization
            ),
            "tensor_parallel_size": kwargs.get(
                "tensor_parallel_size", self._tensor_parallel_size
            ),
            "trust_remote_code": True,
        }
        if kwargs.get("enforce_eager", self._enforce_eager):
            params["enforce_eager"] = True
        if self._quantization:
            params["quantization"] = self._quantization
        if self._load_format:
            params["load_format"] = self._load_format
        if self._kv_cache_dtype != "auto":
            params["kv_cache_dtype"] = self._kv_cache_dtype
            # TurboQuant fork requires enable_turboquant=True alongside kv_cache_dtype
            try:
                from vllm.v1.attention.ops.turboquant_kv_cache import (
                    is_turboquant_kv_cache,
                )

                if is_turboquant_kv_cache(self._kv_cache_dtype):
                    params["enable_turboquant"] = True
            except ImportError:
                pass

        if kwargs.get("kv_cache_memory_bytes"):
            params["kv_cache_memory_bytes"] = kwargs["kv_cache_memory_bytes"]

        # CPU offloading: spill model weights that exceed VRAM into system RAM
        offload = kwargs.get("cpu_offload_gb", self._cpu_offload_gb)
        if offload and offload > 0:
            params["cpu_offload_gb"] = offload

        self._llm = LLM(**params)
        self._model_name = model_path
        # Cache the tokenizer for chat template formatting
        self._tokenizer = self._llm.get_tokenizer()

    def unload_model(self) -> None:
        if self._llm is not None:
            del self._llm
            self._llm = None
            self._model_name = ""
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    def _messages_to_dicts(self, messages: list[ChatMessage]) -> list[dict]:
        """Convert ChatMessage list to OpenAI-style message dicts."""
        return [{"role": m.role, "content": m.content} for m in messages]

    def _apply_chat_template(self, messages: list[ChatMessage]) -> str:
        """Format messages using the model's tokenizer chat template."""
        msg_dicts = self._messages_to_dicts(messages)
        if self._tokenizer and hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                msg_dicts,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback for tokenizers without chat template
        return self._fallback_format(messages)

    def _fallback_format(self, messages: list[ChatMessage]) -> str:
        """Fallback prompt formatting when tokenizer has no chat template."""
        parts = []
        for msg in messages:
            parts.append(f"{msg.role}: {msg.content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        if not self._llm:
            raise RuntimeError("No model loaded. Call load_model() first.")

        from vllm import SamplingParams

        prompt = self._apply_chat_template(messages)
        sampling = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop"),
        )

        start = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        completion_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)

        stats = InferenceStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_name,
            finish_reason=output.outputs[0].finish_reason or "stop",
        )

    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        # vLLM streaming requires the async engine; for the CLI we do a full
        # generation and then yield the text in chunks to keep the UI responsive.
        result = self.chat(messages, **kwargs)
        chunk_size = 4
        text = result.text
        for i in range(0, len(text), chunk_size):
            chunk = text[i : i + chunk_size]
            yield chunk, None
        yield "", result.stats

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        if not self._llm:
            raise RuntimeError("No model loaded. Call load_model() first.")

        from vllm import SamplingParams

        sampling = SamplingParams(
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            stop=kwargs.get("stop"),
        )

        start = time.perf_counter()
        outputs = self._llm.generate([prompt], sampling)
        elapsed = time.perf_counter() - start

        output = outputs[0]
        text = output.outputs[0].text
        completion_tokens = len(output.outputs[0].token_ids)
        prompt_tokens = len(output.prompt_token_ids)

        stats = InferenceStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_name,
            finish_reason=output.outputs[0].finish_reason or "stop",
        )
