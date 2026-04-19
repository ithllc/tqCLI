"""Architecture-coverage tests for the TurboQuant KV calibrator.

Covers LlamaForCausalLM, MistralForCausalLM, and Phi3ForCausalLM wrappers
added in 0.6.2, plus the head_dim derivation fix.
"""

from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from tqcli.core.kv_metadata_generator import (
    _CAPTURE_INSTALLERS,
    _extract_architecture_params,
    check_calibration_preconditions,
)

MODELS_DIR = Path(os.environ.get("TQCLI_MODELS_DIR", str(Path.home() / ".tqcli/models")))


class TestArchitectureRegistry(unittest.TestCase):
    def test_registry_contains_four_archs(self) -> None:
        self.assertEqual(
            sorted(_CAPTURE_INSTALLERS),
            ["LlamaForCausalLM", "MistralForCausalLM", "Phi3ForCausalLM", "Qwen3ForCausalLM"],
        )

    def test_every_installer_installs_and_restores(self) -> None:
        for arch, installer in _CAPTURE_INSTALLERS.items():
            with self.subTest(arch=arch):
                handle = installer()
                try:
                    self.assertTrue(callable(handle.restore))
                    self.assertIsInstance(handle.scores_k, dict)
                finally:
                    handle.restore()


class TestHeadDimDerivation(unittest.TestCase):
    def test_derives_when_missing(self) -> None:
        cfg = {
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 576,
            "num_attention_heads": 9,
            "num_key_value_heads": 3,
            "num_hidden_layers": 30,
        }
        arch, head_dim, num_kv, num_layers = _extract_architecture_params(cfg)
        self.assertEqual(head_dim, 64)
        self.assertEqual(arch, "LlamaForCausalLM")
        self.assertEqual(num_kv, 3)
        self.assertEqual(num_layers, 30)

    def test_explicit_head_dim_preserved(self) -> None:
        cfg = {
            "architectures": ["Qwen3ForCausalLM"],
            "head_dim": 128,
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 36,
        }
        _, head_dim, _, _ = _extract_architecture_params(cfg)
        self.assertEqual(head_dim, 128)

    def test_phi3_fused_qkv_dims(self) -> None:
        cfg = {
            "architectures": ["Phi3ForCausalLM"],
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "num_hidden_layers": 32,
        }
        _, head_dim, num_kv, _ = _extract_architecture_params(cfg)
        self.assertEqual(head_dim, 96)
        self.assertEqual(num_kv, 32)


class TestPreconditionsAcceptNewArchs(unittest.TestCase):
    """Live checks against downloaded models where available."""

    def _model_dir(self, name: str) -> Path | None:
        p = MODELS_DIR / name
        return p if (p / "config.json").is_file() else None

    def test_smollm2_accepted(self) -> None:
        d = self._model_dir("smollm2-135m-instruct")
        if d is None:
            self.skipTest("smollm2-135m-instruct not downloaded")
        ok, reason = check_calibration_preconditions(d, "turboquant35")
        self.assertTrue(ok, reason)
        self.assertIn("LlamaForCausalLM", reason)

    def test_tinymistral_accepted(self) -> None:
        d = self._model_dir("tinymistral-248m")
        if d is None:
            self.skipTest("tinymistral-248m not downloaded")
        ok, reason = check_calibration_preconditions(d, "turboquant35")
        self.assertTrue(ok, reason)
        self.assertIn("MistralForCausalLM", reason)

    def test_phi3_mini_accepted(self) -> None:
        d = self._model_dir("phi-3-mini-4k-instruct")
        if d is None:
            self.skipTest("phi-3-mini-4k-instruct not downloaded")
        ok, reason = check_calibration_preconditions(d, "turboquant35")
        self.assertTrue(ok, reason)
        self.assertIn("Phi3ForCausalLM", reason)


class TestGeneratedMetadataSchema(unittest.TestCase):
    """If a calibration artifact exists, verify shape invariants."""

    def _read_meta(self, name: str) -> dict | None:
        p = MODELS_DIR / name / "turboquant_kv.json"
        return json.loads(p.read_text()) if p.is_file() else None

    def _check_shape(self, meta: dict, expected_layers: int, expected_kv_outer: int,
                     expected_head_size: int, expected_outlier: int) -> None:
        self.assertEqual(meta["version"], 1)
        self.assertEqual(meta["recipe"], "turboquant35")
        self.assertEqual(meta["head_size"], expected_head_size)
        self.assertEqual(len(meta["layers"]), expected_layers)
        first = next(iter(meta["layers"].values()))
        self.assertEqual(len(first["key_high_precision_indices"]), expected_kv_outer)
        self.assertEqual(len(first["key_high_precision_indices"][0]), expected_outlier)
        self.assertEqual(len(first["value_high_precision_indices"]), expected_kv_outer)
        self.assertEqual(len(first["value_high_precision_indices"][0]), expected_outlier)

    def test_smollm2_metadata_shape(self) -> None:
        meta = self._read_meta("smollm2-135m-instruct")
        if meta is None:
            self.skipTest("smollm2 metadata not generated")
        self._check_shape(meta, expected_layers=30, expected_kv_outer=3,
                          expected_head_size=64, expected_outlier=32)

    def test_tinymistral_metadata_shape(self) -> None:
        meta = self._read_meta("tinymistral-248m")
        if meta is None:
            self.skipTest("tinymistral metadata not generated")
        self._check_shape(meta, expected_layers=12, expected_kv_outer=8,
                          expected_head_size=32, expected_outlier=16)

    def test_phi3_metadata_shape(self) -> None:
        meta = self._read_meta("phi-3-mini-4k-instruct")
        if meta is None:
            self.skipTest("phi-3 metadata not generated")
        self._check_shape(meta, expected_layers=32, expected_kv_outer=32,
                          expected_head_size=96, expected_outlier=48)

    def test_vllm_loader_accepts_generated_metadata(self) -> None:
        try:
            from vllm.v1.attention.ops.turboquant_metadata import load_turboquant_metadata
        except ImportError:
            self.skipTest("vllm-turboquant not installed")
        for name in ("smollm2-135m-instruct", "tinymistral-248m", "phi-3-mini-4k-instruct"):
            p = MODELS_DIR / name / "turboquant_kv.json"
            if not p.is_file():
                continue
            with self.subTest(model=name):
                md = load_turboquant_metadata(p)
                self.assertGreater(len(md.layers), 0)


if __name__ == "__main__":
    unittest.main()
