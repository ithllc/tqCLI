"""Tests for the unified thinking mode abstraction."""

from __future__ import annotations


def test_detect_qwen3_format():
    from tqcli.core.thinking import ThinkingFormat, detect_thinking_format
    assert detect_thinking_format("qwen3") == ThinkingFormat.QWEN3
    assert detect_thinking_format("qwen3-coder") == ThinkingFormat.QWEN3
    assert detect_thinking_format("qwen3") == ThinkingFormat.QWEN3


def test_detect_gemma4_format():
    from tqcli.core.thinking import ThinkingFormat, detect_thinking_format
    assert detect_thinking_format("gemma4") == ThinkingFormat.GEMMA4
    assert detect_thinking_format("gemma4") == ThinkingFormat.GEMMA4


def test_detect_unknown_format():
    from tqcli.core.thinking import ThinkingFormat, detect_thinking_format
    assert detect_thinking_format("llama") == ThinkingFormat.NONE
    assert detect_thinking_format("") == ThinkingFormat.NONE


def test_thinking_config_active():
    from tqcli.core.thinking import ThinkingConfig, ThinkingFormat
    cfg = ThinkingConfig(format=ThinkingFormat.QWEN3, enabled=True)
    assert cfg.is_active is True

    cfg2 = ThinkingConfig(format=ThinkingFormat.NONE, enabled=True)
    assert cfg2.is_active is False

    cfg3 = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=False)
    assert cfg3.is_active is False


def test_build_system_prompt_qwen3():
    from tqcli.core.thinking import ThinkingConfig, ThinkingFormat, build_system_prompt_with_thinking
    cfg = ThinkingConfig(format=ThinkingFormat.QWEN3, enabled=True)
    result = build_system_prompt_with_thinking("You are helpful.", cfg)
    assert "Think step by step" in result

    cfg_low = ThinkingConfig(format=ThinkingFormat.QWEN3, enabled=True, depth="low")
    result_low = build_system_prompt_with_thinking("You are helpful.", cfg_low)
    assert "Think briefly" in result_low


def test_build_system_prompt_gemma4():
    from tqcli.core.thinking import ThinkingConfig, ThinkingFormat, build_system_prompt_with_thinking
    cfg = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=True)
    result = build_system_prompt_with_thinking("You are helpful.", cfg)
    assert "<|think|>" in result

    cfg_low = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=True, depth="low")
    result_low = build_system_prompt_with_thinking("You are helpful.", cfg_low)
    assert "<|think|>" in result_low
    assert "Think briefly" in result_low


def test_build_system_prompt_disabled():
    from tqcli.core.thinking import ThinkingConfig, ThinkingFormat, build_system_prompt_with_thinking
    cfg = ThinkingConfig(format=ThinkingFormat.GEMMA4, enabled=False)
    result = build_system_prompt_with_thinking("You are helpful.", cfg)
    assert "<|think|>" not in result
    assert result == "You are helpful."


def test_strip_qwen3_thinking():
    from tqcli.core.thinking import ThinkingFormat, strip_thinking_blocks
    text = "<think>Let me reason about this...</think>The answer is 42."
    clean = strip_thinking_blocks(text, ThinkingFormat.QWEN3)
    assert clean == "The answer is 42."


def test_strip_gemma4_thinking():
    from tqcli.core.thinking import ThinkingFormat, strip_thinking_blocks
    text = "<|channel>thought\nLet me reason about this...\n<channel|>The answer is 42."
    clean = strip_thinking_blocks(text, ThinkingFormat.GEMMA4)
    assert clean == "The answer is 42."


def test_extract_qwen3_thinking():
    from tqcli.core.thinking import ThinkingFormat, extract_thinking
    text = "<think>Step 1: consider the problem</think>The result is 7."
    thinking, clean = extract_thinking(text, ThinkingFormat.QWEN3)
    assert thinking == "Step 1: consider the problem"
    assert clean == "The result is 7."


def test_extract_gemma4_thinking():
    from tqcli.core.thinking import ThinkingFormat, extract_thinking
    text = "<|channel>thought\nStep 1: consider the problem\n<channel|>The result is 7."
    thinking, clean = extract_thinking(text, ThinkingFormat.GEMMA4)
    assert thinking == "Step 1: consider the problem"
    assert clean == "The result is 7."


def test_is_inside_thinking_block_qwen3():
    from tqcli.core.thinking import ThinkingFormat, is_inside_thinking_block
    assert is_inside_thinking_block("<think>partial thought", ThinkingFormat.QWEN3) is True
    assert is_inside_thinking_block("<think>done</think>response", ThinkingFormat.QWEN3) is False
    assert is_inside_thinking_block("no thinking here", ThinkingFormat.QWEN3) is False


def test_is_inside_thinking_block_gemma4():
    from tqcli.core.thinking import ThinkingFormat, is_inside_thinking_block
    assert is_inside_thinking_block("<|channel>thought\npartial", ThinkingFormat.GEMMA4) is True
    assert is_inside_thinking_block("<|channel>thought\ndone\n<channel|>resp", ThinkingFormat.GEMMA4) is False
    assert is_inside_thinking_block("no thinking here", ThinkingFormat.GEMMA4) is False


def test_strip_thinking_from_history():
    from tqcli.core.thinking import ThinkingFormat, strip_thinking_from_history
    history = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "<think>Simple math</think>The answer is 4."},
        {"role": "user", "content": "Thanks"},
    ]
    cleaned = strip_thinking_from_history(history, ThinkingFormat.QWEN3)
    assert "<think>" not in cleaned[1]["content"]
    assert cleaned[1]["content"] == "The answer is 4."
    # User messages unchanged
    assert cleaned[0]["content"] == "What is 2+2?"


def test_all_gemma4_models_support_thinking():
    from tqcli.core.model_registry import BUILTIN_PROFILES
    gemma_models = [p for p in BUILTIN_PROFILES if p.family == "gemma4"]
    assert len(gemma_models) == 4
    for m in gemma_models:
        assert m.supports_thinking is True, f"{m.id} should support thinking"


def test_all_qwen3_models_support_thinking():
    from tqcli.core.model_registry import BUILTIN_PROFILES
    qwen_general = [p for p in BUILTIN_PROFILES if p.family == "qwen3"]
    for m in qwen_general:
        assert m.supports_thinking is True, f"{m.id} should support thinking"
