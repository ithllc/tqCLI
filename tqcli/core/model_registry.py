"""Model registry: catalog of supported models and their capability profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class TaskDomain(Enum):
    CODING = "coding"
    REASONING = "reasoning"
    GENERAL = "general"
    MATH = "math"
    CREATIVE = "creative"
    INSTRUCTION = "instruction"


@dataclass
class ModelProfile:
    """Describes a model's identity and strengths for routing decisions."""

    id: str  # e.g. "gemma-4-12b-Q4_K_M"
    family: str  # e.g. "gemma", "qwen-coder", "qwen-instruct"
    display_name: str
    hf_repo: str  # HuggingFace repo for download
    filename: str  # GGUF filename or model dir name
    parameter_count: str  # "12B", "7B", etc.
    quantization: str  # "Q4_K_M", "AWQ", etc.
    format: str  # "gguf", "awq", "gptq", "safetensors"
    context_length: int = 8192
    strengths: list[TaskDomain] = field(default_factory=list)
    strength_scores: dict[str, float] = field(default_factory=dict)
    min_ram_mb: int = 4000
    min_vram_mb: int = 0
    engine: str = "llama.cpp"  # "llama.cpp" or "vllm"
    local_path: Path | None = None
    supports_thinking: bool = False  # Qwen 3 thinking mode
    active_params: str = ""  # for MoE models, e.g. "3.8B"
    multimodal: bool = False  # supports image/audio input


# Predefined profiles for supported models.
# strength_scores: 0.0 (worst) to 1.0 (best) — used by the router to rank models per task.
#
# Sources for strength scores:
#  - Gemma 4: Google model card (ai.google.dev/gemma/docs/core/model_card_4).
#    31B #3 on Arena AI open-source leaderboard. MMLU-Pro 85.2%, AIME 2026 89.2%.
#    26B MoE MMLU-Pro 82.6%. E4B/E2B are edge-optimized with audio support.
#  - Qwen 3: qwenlm.github.io/blog/qwen3, arXiv 2505.09388.
#    Qwen3-4B rivals Qwen2.5-72B. 32B has 128K context. Thinking mode built-in.
#  - Qwen3-Coder: qwenlm.github.io/blog/qwen3-coder.
#    Coder-Next (80B MoE, 3B active) with 256K context. SWE-bench competitive.
#
# MoE models: min_ram_mb is based on total params (model file size) but
# active_params reflects actual compute during inference.

BUILTIN_PROFILES: list[ModelProfile] = [
    # ── Google Gemma 4 ────────────────────────────────────────────────
    ModelProfile(
        id="gemma-4-31b-it-Q4_K_M",
        family="gemma4",
        display_name="Gemma 4 31B Dense Instruct (Q4_K_M)",
        hf_repo="unsloth/gemma-4-31B-it-GGUF",
        filename="gemma-4-31B-it-Q4_K_M.gguf",
        parameter_count="31B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=256000,
        strengths=[TaskDomain.GENERAL, TaskDomain.REASONING, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.80,
            "reasoning": 0.93,
            "general": 0.95,
            "math": 0.88,
            "creative": 0.88,
            "instruction": 0.93,
        },
        min_ram_mb=20000,
        min_vram_mb=18000,
        engine="llama.cpp",
        supports_thinking=True,  # <|think|> + <|channel>thought...<channel|>
        multimodal=True,
    ),
    ModelProfile(
        id="gemma-4-27b-it-Q4_K_M",
        family="gemma4",
        display_name="Gemma 4 26B MoE Instruct (Q4_K_M)",
        hf_repo="unsloth/gemma-4-26B-A4B-it-GGUF",
        filename="gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        parameter_count="26B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=256000,
        strengths=[TaskDomain.GENERAL, TaskDomain.REASONING, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.75,
            "reasoning": 0.88,
            "general": 0.90,
            "math": 0.83,
            "creative": 0.85,
            "instruction": 0.88,
        },
        min_ram_mb=16000,
        min_vram_mb=14000,
        engine="llama.cpp",
        supports_thinking=True,
        active_params="3.8B",
        multimodal=True,
    ),
    ModelProfile(
        id="gemma-4-e4b-it-Q4_K_M",
        family="gemma4",
        display_name="Gemma 4 E4B Edge Instruct (Q4_K_M)",
        hf_repo="unsloth/gemma-4-E4B-it-GGUF",
        filename="gemma-4-E4B-it-Q4_K_M.gguf",
        parameter_count="4.5B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.GENERAL, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.55,
            "reasoning": 0.65,
            "general": 0.72,
            "math": 0.58,
            "creative": 0.65,
            "instruction": 0.70,
        },
        min_ram_mb=4000,
        min_vram_mb=3000,
        engine="llama.cpp",
        supports_thinking=True,
        multimodal=True,
    ),
    ModelProfile(
        id="gemma-4-e2b-it-Q4_K_M",
        family="gemma4",
        display_name="Gemma 4 E2B Edge Instruct (Q4_K_M)",
        hf_repo="unsloth/gemma-4-E2B-it-GGUF",
        filename="gemma-4-E2B-it-Q4_K_M.gguf",
        parameter_count="2.3B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.GENERAL, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.40,
            "reasoning": 0.50,
            "general": 0.58,
            "math": 0.45,
            "creative": 0.52,
            "instruction": 0.55,
        },
        min_ram_mb=2000,
        min_vram_mb=1500,
        engine="llama.cpp",
        supports_thinking=True,
        multimodal=True,
    ),
    # ── Qwen 3 Coder ─────────────────────────────────────────────────
    ModelProfile(
        id="qwen3-coder-next-Q4_K_M",
        family="qwen3-coder",
        display_name="Qwen3 Coder Next 80B MoE (Q4_K_M)",
        hf_repo="Qwen/Qwen3-Coder-Next-GGUF",
        filename="Qwen3-Coder-Next-Q4_K_M.gguf",
        parameter_count="80B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=256000,
        strengths=[TaskDomain.CODING, TaskDomain.MATH, TaskDomain.REASONING],
        strength_scores={
            "coding": 0.95,
            "reasoning": 0.82,
            "general": 0.70,
            "math": 0.88,
            "creative": 0.50,
            "instruction": 0.72,
        },
        min_ram_mb=48000,
        min_vram_mb=24000,
        engine="llama.cpp",
        active_params="3B",
    ),
    ModelProfile(
        id="qwen3-coder-30b-a3b-instruct-Q4_K_M",
        family="qwen3-coder",
        display_name="Qwen3 Coder 30B-A3B Instruct (Q4_K_M)",
        hf_repo="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        filename="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        parameter_count="30B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=256000,
        strengths=[TaskDomain.CODING, TaskDomain.MATH],
        strength_scores={
            "coding": 0.88,
            "reasoning": 0.75,
            "general": 0.62,
            "math": 0.82,
            "creative": 0.45,
            "instruction": 0.65,
        },
        min_ram_mb=18000,
        min_vram_mb=10000,
        engine="llama.cpp",
        active_params="3B",
    ),
    # ── Qwen 3 General (with thinking mode) ───────────────────────────
    ModelProfile(
        id="qwen3-32b-Q4_K_M",
        family="qwen3",
        display_name="Qwen3 32B (Q4_K_M)",
        hf_repo="Qwen/Qwen3-32B-GGUF",
        filename="Qwen3-32B-Q4_K_M.gguf",
        parameter_count="32B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.REASONING, TaskDomain.INSTRUCTION, TaskDomain.GENERAL],
        strength_scores={
            "coding": 0.78,
            "reasoning": 0.92,
            "general": 0.92,
            "math": 0.85,
            "creative": 0.82,
            "instruction": 0.92,
        },
        min_ram_mb=20000,
        min_vram_mb=18000,
        engine="llama.cpp",
        supports_thinking=True,
    ),
    ModelProfile(
        id="qwen3-8b-Q4_K_M",
        family="qwen3",
        display_name="Qwen3 8B (Q4_K_M)",
        hf_repo="Qwen/Qwen3-8B-GGUF",
        filename="Qwen3-8B-Q4_K_M.gguf",
        parameter_count="8B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.REASONING, TaskDomain.GENERAL, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.68,
            "reasoning": 0.78,
            "general": 0.80,
            "math": 0.72,
            "creative": 0.72,
            "instruction": 0.78,
        },
        min_ram_mb=6000,
        min_vram_mb=5000,
        engine="llama.cpp",
        supports_thinking=True,
    ),
    ModelProfile(
        id="qwen3-4b-Q4_K_M",
        family="qwen3",
        display_name="Qwen3 4B (Q4_K_M)",
        hf_repo="Qwen/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        parameter_count="4B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=32768,
        strengths=[TaskDomain.GENERAL, TaskDomain.INSTRUCTION],
        strength_scores={
            "coding": 0.58,
            "reasoning": 0.68,
            "general": 0.70,
            "math": 0.62,
            "creative": 0.62,
            "instruction": 0.68,
        },
        min_ram_mb=3500,
        min_vram_mb=3000,
        engine="llama.cpp",
        supports_thinking=True,
    ),
    ModelProfile(
        id="qwen3-30b-a3b-Q4_K_M",
        family="qwen3",
        display_name="Qwen3 30B-A3B MoE (Q4_K_M)",
        hf_repo="Qwen/Qwen3-30B-A3B-GGUF",
        filename="Qwen3-30B-A3B-Q4_K_M.gguf",
        parameter_count="30B",
        quantization="Q4_K_M",
        format="gguf",
        context_length=128000,
        strengths=[TaskDomain.REASONING, TaskDomain.GENERAL, TaskDomain.MATH],
        strength_scores={
            "coding": 0.72,
            "reasoning": 0.85,
            "general": 0.85,
            "math": 0.80,
            "creative": 0.75,
            "instruction": 0.82,
        },
        min_ram_mb=18000,
        min_vram_mb=10000,
        engine="llama.cpp",
        supports_thinking=True,
        active_params="3B",
    ),
]


class ModelRegistry:
    """Discovers, registers, and queries available models."""

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self._profiles: dict[str, ModelProfile] = {}
        for p in BUILTIN_PROFILES:
            self._profiles[p.id] = p

    def scan_local_models(self) -> list[ModelProfile]:
        found = []
        if not self.models_dir.exists():
            return found
        for f in self.models_dir.rglob("*.gguf"):
            for profile in self._profiles.values():
                if f.name.lower() == profile.filename.lower():
                    profile.local_path = f
                    found.append(profile)
        return found

    def get_available_models(self) -> list[ModelProfile]:
        return [p for p in self._profiles.values() if p.local_path is not None]

    def get_all_profiles(self) -> list[ModelProfile]:
        return list(self._profiles.values())

    def get_profile(self, model_id: str) -> ModelProfile | None:
        return self._profiles.get(model_id)

    def get_models_for_domain(self, domain: TaskDomain) -> list[ModelProfile]:
        available = self.get_available_models()
        scored = []
        for m in available:
            score = m.strength_scores.get(domain.value, 0.0)
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    def register_model(self, profile: ModelProfile) -> None:
        self._profiles[profile.id] = profile

    def fits_hardware(self, profile: ModelProfile, ram_mb: int, vram_mb: int) -> bool:
        if vram_mb > 0 and profile.min_vram_mb > 0:
            return profile.min_vram_mb <= vram_mb
        return profile.min_ram_mb <= ram_mb
