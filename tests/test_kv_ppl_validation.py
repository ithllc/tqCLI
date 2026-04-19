"""Perplexity validation gate for TurboQuant KV compression.

Detects silent quality collapse from bad outlier indices by comparing
forced-sequence perplexity of qwen3-4b-vllm under kv_cache_dtype='auto'
(baseline) vs. kv_cache_dtype='turboquant35'. The gate asserts that
TurboQuant PPL is within 5% of baseline on a fixed 10-prompt corpus.

Gated by TQCLI_PPL_GATE=1 because each run reloads the model twice on a
4 GB VRAM reference box, costing ~15 minutes wall-clock.

All assertions derive from real log-prob computation; no hardcoded pass.
"""

from __future__ import annotations

import math
import os
import unittest
from pathlib import Path

VALIDATION_PROMPTS = [
    "The industrial revolution, which began in Britain around 1760, fundamentally altered human society. "
    "Mechanization of textile production first transformed small cottage industries into centralized factories. "
    "Steam power then extended the revolution to mining, transportation, and metallurgy, allowing production at scales "
    "previously unimaginable. Urbanization accelerated as rural workers migrated to industrial cities in search of wages, "
    "though conditions in early factory towns were often appalling by modern standards.",
    "Photosynthesis is the biochemical process by which plants, algae, and cyanobacteria convert light energy into "
    "chemical energy stored in glucose molecules. The overall reaction takes carbon dioxide and water as inputs, "
    "producing glucose and oxygen as outputs. It occurs in two stages: the light-dependent reactions, which split "
    "water molecules and generate ATP and NADPH in the thylakoid membrane, and the Calvin cycle, which fixes "
    "carbon dioxide into sugars in the stroma of the chloroplast.",
    "Quantum entanglement is a physical phenomenon that occurs when pairs of particles become correlated in such a "
    "way that the quantum state of each particle cannot be described independently of the others, even across large "
    "spatial distances. Measurements performed on one particle are instantly correlated with the state of the other, "
    "an observation that Einstein famously described as 'spooky action at a distance.' Despite the strange appearance, "
    "entanglement does not permit faster-than-light communication.",
    "The history of the English language is typically divided into three periods: Old English, spoken from roughly "
    "the 5th century to the 11th century; Middle English, from the Norman conquest of 1066 until approximately 1500; "
    "and Modern English, from 1500 onward. Each transition brought dramatic changes in vocabulary, grammar, and "
    "pronunciation. The Norman invasion alone introduced thousands of French-derived words, particularly in domains "
    "of law, religion, cuisine, and government, fundamentally reshaping the language's character.",
    "Convolutional neural networks became the dominant architecture for computer vision tasks during the 2010s, "
    "achieving human-level performance on benchmarks like ImageNet. A typical convolutional network alternates "
    "between convolution layers that learn spatial filters, pooling layers that reduce resolution while preserving "
    "translation invariance, and fully-connected layers at the top that produce final classifications. More recent "
    "architectures like ResNets introduced skip connections to address the vanishing gradient problem that had "
    "limited the depth of earlier models.",
    "The general theory of relativity, published by Einstein in 1915, describes gravity not as a force in the "
    "Newtonian sense but as a consequence of the curvature of spacetime caused by mass and energy. The theory "
    "predicts phenomena that Newtonian physics cannot, including the bending of light by gravitational fields, "
    "the gravitational redshift of light escaping massive bodies, and the precession of Mercury's perihelion. "
    "It remains the best-tested theory of gravity, verified to extraordinary precision by experiments ranging "
    "from laboratory interferometry to observations of binary pulsars.",
    "In typical relational databases, transactions provide four guarantees summarized by the ACID acronym: "
    "atomicity ensures that a transaction either completes entirely or leaves no trace; consistency ensures "
    "that the database moves from one valid state to another according to integrity constraints; isolation "
    "ensures that concurrent transactions do not interfere with each other; durability ensures that committed "
    "changes survive system failures. Different isolation levels trade consistency guarantees against concurrency "
    "performance, with serializable isolation being the strongest but slowest.",
    "Climate feedback loops are mechanisms that can either amplify or dampen the initial climate response to a "
    "forcing. The ice-albedo feedback is a well-known positive feedback: as polar ice melts due to warming, the "
    "darker ocean surface absorbs more solar radiation, accelerating further warming. The Planck feedback is a "
    "negative feedback: a warmer planet radiates more energy to space according to the Stefan-Boltzmann law, "
    "partially offsetting the warming. Accurate climate projection depends on correctly quantifying the net "
    "balance of all such feedbacks.",
    "Mitochondria, the cellular organelles responsible for energy production through oxidative phosphorylation, "
    "carry their own distinct DNA separate from the nucleus. This mitochondrial DNA is inherited exclusively "
    "from the mother in most animals, a property that has made it useful in tracing human evolutionary history "
    "through matrilineal lineages. The endosymbiotic theory proposes that mitochondria descended from ancient "
    "free-living bacteria that were engulfed by a eukaryotic ancestor, a hypothesis supported by structural "
    "and genetic similarities to modern alphaproteobacteria.",
    "The Byzantine Generals Problem, formulated by Leslie Lamport and colleagues in 1982, asks how distributed "
    "systems can reach consensus in the presence of faulty or malicious participants. The classic statement "
    "involves generals coordinating an attack via messengers, where some generals may betray the plan. The "
    "problem showed that consensus is achievable if and only if at most one-third of participants are faulty. "
    "Modern blockchain protocols generalize these ideas to achieve consensus across untrusted, permissionless "
    "networks at global scale.",
]


def _sum_logprobs_for_prompt(llm, sampling_params, prompt: str) -> tuple[float, int]:
    """Return (sum_logprob, num_tokens_scored) for the given prompt.

    Uses vLLM's prompt_logprobs to extract the log-probability of each
    actually-appearing prompt token given its context. Skips position 0
    (no context) and any None entries.
    """
    outputs = llm.generate([prompt], sampling_params)
    out = outputs[0]
    prompt_token_ids = out.prompt_token_ids
    prompt_logprobs = out.prompt_logprobs
    if prompt_logprobs is None:
        raise RuntimeError("vLLM returned no prompt_logprobs; check SamplingParams.")
    total = 0.0
    scored = 0
    for pos, entry in enumerate(prompt_logprobs):
        if entry is None or pos == 0:
            continue
        token_id = prompt_token_ids[pos]
        logprob_obj = entry.get(token_id)
        if logprob_obj is None:
            continue
        total += float(logprob_obj.logprob)
        scored += 1
    return total, scored


def _compute_corpus_ppl(llm, prompts: list[str]) -> tuple[float, int]:
    """Run vLLM over a prompt list, return (perplexity, total_scored_tokens)."""
    from vllm import SamplingParams

    sampling = SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1)
    total_logprob = 0.0
    total_tokens = 0
    for p in prompts:
        lp, n = _sum_logprobs_for_prompt(llm, sampling, p)
        total_logprob += lp
        total_tokens += n
    if total_tokens == 0:
        raise RuntimeError("No tokens scored across corpus.")
    nll = -total_logprob / total_tokens
    return math.exp(nll), total_tokens


@unittest.skipUnless(
    os.environ.get("TQCLI_PPL_GATE") == "1",
    "Set TQCLI_PPL_GATE=1 to run the PPL validation gate (~15 min runtime).",
)
class TestTurboQuantPPLGate(unittest.TestCase):
    def setUp(self) -> None:
        self.model_dir = Path.home() / ".tqcli/models/qwen3-4b-vllm"
        if not (self.model_dir / "config.json").is_file():
            self.skipTest(f"qwen3-4b-vllm not installed at {self.model_dir}")
        try:
            import vllm  # noqa: F401
        except ImportError:
            self.skipTest("vllm not installed in this environment.")

    def _make_llm(self, kv_cache_dtype: str):
        from vllm import LLM

        params = dict(
            model=str(self.model_dir),
            max_model_len=768,
            gpu_memory_utilization=0.70,
            quantization="bitsandbytes",
            load_format="bitsandbytes",
            cpu_offload_gb=6.5,
            enforce_eager=True,
            trust_remote_code=True,
        )
        if kv_cache_dtype != "auto":
            params["kv_cache_dtype"] = kv_cache_dtype
            if kv_cache_dtype.startswith("turboquant"):
                params["enable_turboquant"] = True
        return LLM(**params)

    def test_turboquant35_ppl_within_5pct_of_baseline(self) -> None:
        # Baseline: kv:auto
        print("[ppl-gate] loading baseline (kv:auto)...", flush=True)
        llm_baseline = self._make_llm("auto")
        try:
            ppl_base, n_base = _compute_corpus_ppl(llm_baseline, VALIDATION_PROMPTS)
            print(f"[ppl-gate] baseline PPL={ppl_base:.4f} over {n_base} tokens", flush=True)
        finally:
            del llm_baseline
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # TurboQuant35
        print("[ppl-gate] loading turboquant35...", flush=True)
        llm_tq = self._make_llm("turboquant35")
        try:
            ppl_tq, n_tq = _compute_corpus_ppl(llm_tq, VALIDATION_PROMPTS)
            print(f"[ppl-gate] turboquant35 PPL={ppl_tq:.4f} over {n_tq} tokens", flush=True)
        finally:
            del llm_tq
            import gc, torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.assertEqual(
            n_base, n_tq,
            f"Scored-token count must match: baseline={n_base} vs turboquant={n_tq}",
        )
        ratio = ppl_tq / ppl_base
        print(
            f"[ppl-gate] ratio turboquant35 / baseline = {ratio:.4f} "
            f"(pass threshold: <= 1.05)",
            flush=True,
        )
        self.assertLessEqual(
            ratio,
            1.05,
            f"TurboQuant35 PPL {ppl_tq:.4f} is more than 5% worse than baseline "
            f"{ppl_base:.4f} (ratio {ratio:.4f}). Likely silent quality collapse "
            f"from poor outlier indices.",
        )


if __name__ == "__main__":
    unittest.main()
