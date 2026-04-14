"""llama.cpp backend via llama-cpp-python. Cross-platform: macOS, Linux, Windows."""

from __future__ import annotations

import time
from typing import Generator

from tqcli.core.engine import (
    ChatMessage,
    CompletionResult,
    InferenceEngine,
    InferenceStats,
)


class LlamaBackend(InferenceEngine):
    def __init__(
        self,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int = 0,
        verbose: bool = False,
        cache_type_k: str = "f16",
        cache_type_v: str = "f16",
    ):
        self._n_ctx = n_ctx
        self._n_gpu_layers = n_gpu_layers
        self._n_threads = n_threads
        self._verbose = verbose
        self._cache_type_k = cache_type_k
        self._cache_type_v = cache_type_v
        self._model = None
        self._model_path: str = ""
        self._chat_handler = None  # multimodal clip handler

    @property
    def engine_name(self) -> str:
        return "llama.cpp"

    @property
    def is_available(self) -> bool:
        try:
            import llama_cpp  # noqa: F401

            return True
        except ImportError:
            return False

    def load_model(self, model_path: str, **kwargs) -> None:
        from llama_cpp import Llama

        params = {
            "model_path": model_path,
            "n_ctx": kwargs.get("n_ctx", self._n_ctx),
            "n_gpu_layers": kwargs.get("n_gpu_layers", self._n_gpu_layers),
            "verbose": self._verbose,
        }
        if self._n_threads > 0:
            params["n_threads"] = self._n_threads
        # TurboQuant KV cache types (requires TurboQuant fork of llama-cpp-python)
        if self._cache_type_k != "f16":
            params["cache_type_k"] = self._cache_type_k
        if self._cache_type_v != "f16":
            params["cache_type_v"] = self._cache_type_v

        # Multimodal: load clip model only if explicitly requested or model supports it
        clip_path = kwargs.get("clip_model_path")
        multimodal = kwargs.get("multimodal", False)
        if not clip_path and multimodal:
            from pathlib import Path
            model_dir = Path(model_path).parent
            for candidate in model_dir.glob("mmproj*.gguf"):
                clip_path = str(candidate)
                break

        if clip_path and multimodal:
            try:
                from llama_cpp.llama_chat_format import Llava16ChatHandler
                self._chat_handler = Llava16ChatHandler(clip_model_path=clip_path, verbose=self._verbose)
                params["chat_handler"] = self._chat_handler
            except (ImportError, Exception):
                self._chat_handler = None

        self._model = Llama(**params)
        self._model_path = model_path

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            self._model_path = ""

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _build_message_dicts(self, messages: list[ChatMessage]) -> list[dict]:
        """Build message dicts, converting multimodal content to llava format."""
        import base64
        from pathlib import Path

        result = []
        for m in messages:
            if (m.images or m.audio) and self._chat_handler:
                # Build multimodal content array
                content_parts = []
                if m.images:
                    for img_path in m.images:
                        p = Path(img_path)
                        if p.exists():
                            data = base64.b64encode(p.read_bytes()).decode()
                            ext = p.suffix.lower().lstrip(".")
                            mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                                    "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/png")
                            content_parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{data}"}
                            })
                if m.audio:
                    for audio_path in m.audio:
                        p = Path(audio_path)
                        if p.exists():
                            data = base64.b64encode(p.read_bytes()).decode()
                            ext = p.suffix.lower().lstrip(".")
                            mime = {"wav": "audio/wav", "mp3": "audio/mpeg",
                                    "ogg": "audio/ogg", "flac": "audio/flac"}.get(ext, "audio/wav")
                            content_parts.append({
                                "type": "input_audio",
                                "input_audio": {"data": data, "format": ext}
                            })
                content_parts.append({"type": "text", "text": m.content})
                result.append({"role": m.role, "content": content_parts})
            else:
                result.append({"role": m.role, "content": m.content})
        return result

    def chat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        msg_dicts = self._build_message_dicts(messages)
        start = time.perf_counter()

        response = self._model.create_chat_completion(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stop=kwargs.get("stop"),
        )

        elapsed = time.perf_counter() - start
        text = response["choices"][0]["message"]["content"]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_path,
            finish_reason=response["choices"][0].get("finish_reason", "stop"),
        )

    def chat_stream(
        self, messages: list[ChatMessage], **kwargs
    ) -> Generator[tuple[str, InferenceStats | None], None, None]:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        msg_dicts = self._build_message_dicts(messages)
        start = time.perf_counter()
        first_token_time = None
        token_count = 0

        stream = self._model.create_chat_completion(
            messages=msg_dicts,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            top_p=kwargs.get("top_p", 0.9),
            stream=True,
        )

        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += 1
                yield content, None

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else elapsed

        final_stats = InferenceStats(
            completion_tokens=token_count,
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(token_count, elapsed),
            time_to_first_token_s=ttft,
            total_time_s=elapsed,
        )
        yield "", final_stats

    def complete(self, prompt: str, **kwargs) -> CompletionResult:
        if not self._model:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start = time.perf_counter()
        response = self._model(
            prompt,
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
            stop=kwargs.get("stop"),
            echo=False,
        )
        elapsed = time.perf_counter() - start
        text = response["choices"][0]["text"]
        usage = response.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)

        stats = InferenceStats(
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=completion_tokens,
            total_tokens=usage.get("total_tokens", 0),
            completion_time_s=elapsed,
            tokens_per_second=self._compute_tps(completion_tokens, elapsed),
            total_time_s=elapsed,
        )

        return CompletionResult(
            text=text,
            stats=stats,
            model_id=self._model_path,
            finish_reason=response["choices"][0].get("finish_reason", "stop"),
        )
