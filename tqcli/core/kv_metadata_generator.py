"""Generate turboquant_kv.json metadata via activation-based calibration.

Mirrors the vLLM-turboquant fork's own reference selector,
`vllm.v1.attention.ops.turboquant_kv_cache.build_turboquant_outlier_masks`,
which scores each channel by the mean-squared activation across tokens and
picks the top-`outlier_count` indices per KV head. This module accumulates the
same statistic online over a calibration corpus and writes the resulting
per-layer, per-head outlier indices to `turboquant_kv.json` in the model
directory.

Design rationale:
    Pre-computed metadata is higher quality than runtime first-batch auto
    calibration (more prompts → tighter variance estimate, reproducible
    across servers). We capture POST-RoPE K (since the KV cache stores
    post-RoPE keys) and V (no RoPE on values) at bf16 precision with CPU
    offload to fit small-VRAM hosts.

Refused cases:
    - Pre-quantized source weights (AWQ, GPTQ, bnb): activation statistics
      from quantized weights are biased by the weight quantization, not the
      true FP distribution the runtime will see.
    - Variable head_dim within a model (Gemma 4 style sliding/global).
    - head_dim not a multiple of 16.
    - Architectures without a registered capture wrapper.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_GROUP_ALIGNMENT,
    TURBOQUANT_OUTLIER_RATIOS,
    get_turboquant_outlier_count,
)

# Minimum observed tokens across the full corpus (enforced by the unit test).
# 5_000 is a 10x uplift over the 525-token baseline shipped in #27 and gives
# ~40 samples per (kv_head, channel) cell for Qwen3 4B (8 KV heads x 128 channels).
# Long-term target is 10-15k; pushing beyond 5k is a follow-up corpus extension.
MIN_OBSERVED_TOKENS = 5_000

# 30 paragraph-length calibration prompts balanced across code, math/logic,
# prose/narrative, technical explanation, dialog, and misc. Current tokenized
# size on Qwen3 ≈ 5.1k; see tests/test_kv_metadata_corpus.py.
DEFAULT_CALIBRATION_PROMPTS: list[str] = [
    # --- Code (5) ---
    (
        "Walk me through how a red-black tree insertion works step by step. Start from an empty tree, "
        "insert the keys 10, 20, 30, 15, 25, 5, 1 in sequence, and at each step describe which invariant "
        "(root is black, no two reds in a row, equal black-height on every root-to-leaf path) might be "
        "violated and what rotation or recoloring fixes it. Include a brief Python skeleton for the insert "
        "routine with helper functions for left_rotate, right_rotate, and fix_up_after_insert, even if the "
        "bodies are stubs. Explain why red-black trees guarantee O(log n) worst-case operations while "
        "regular binary search trees only give O(log n) in the average case. Finish with one sentence on "
        "how red-black trees relate to 2-3-4 trees and why a left-leaning variant is easier to implement."
    ),
    (
        "Review this Go function for correctness and idiomatic style, then rewrite it: "
        "`func Contains(s []int, x int) bool { for i := 0; i < len(s); i++ { if s[i] == x { return true } }; return false }`. "
        "Point out three specific issues an experienced Go reviewer would raise (range loop over index loop, "
        "generic opportunity if on a recent Go version, potential inlining hints). Rewrite the function using "
        "generics so it accepts any comparable type, and include a short benchmark comparison outline that uses "
        "testing.B to measure per-element cost for slices of length 1, 10, 100, 1000, and 10000. Conclude with "
        "when a hash set would beat this linear scan and at what crossover size that typically happens on modern x86."
    ),
    (
        "Design a rate limiter for a REST API that must allow 100 requests per minute per API key with a strict "
        "ceiling (no burst). Describe three candidate algorithms — fixed-window counter, sliding-window log, "
        "and token bucket — with their accuracy and memory trade-offs. Pick one and give a concrete Redis-backed "
        "implementation sketch in Python using Redis Lua scripts to keep the check atomic. Explain why naive "
        "INCR + EXPIRE has a double-count bug at window boundaries and how the sliding-window variant fixes it. "
        "Include how you'd expose per-key limit headers (X-RateLimit-Remaining, X-RateLimit-Reset) in the response."
    ),
    (
        "Write a Rust module that implements a bounded multi-producer single-consumer channel using std sync "
        "primitives only (no crossbeam, no async). Show the public send, try_send, and recv signatures with "
        "appropriate error types. Explain the inner state: Mutex<VecDeque<T>>, capacity, and two Condvar "
        "primitives for not-full and not-empty signalling. Walk through a concrete deadlock scenario that "
        "would occur if you used only one Condvar, and why two are required. Finish with a sentence on "
        "when an SPSC lock-free ring buffer would be strictly better and at what producer-rate threshold "
        "the lock contention on Mutex becomes the bottleneck in practice."
    ),
    (
        "Refactor this SQL query for a PostgreSQL OLTP workload so it uses an index-only scan: "
        "`SELECT user_id, COUNT(*) FROM orders WHERE created_at > NOW() - INTERVAL '24 hours' GROUP BY user_id`. "
        "Specify the exact CREATE INDEX statement needed, explain why a BRIN index would be inappropriate here, "
        "and describe how to check via EXPLAIN (ANALYZE, BUFFERS) whether Postgres actually uses the index. "
        "Include guidance on when adding an INCLUDE column is preferable to adding another key column. Mention "
        "the interaction with autovacuum's visibility map since index-only scans require tuples to be marked "
        "all-visible."
    ),

    # --- Math / logic (5) ---
    (
        "Derive the closed-form solution to linear regression using ordinary least squares. Start from the "
        "sum-of-squared-residuals loss function L = ||y - Xβ||^2, expand it, take the gradient with respect to β, "
        "set the gradient to zero, and solve. Show every step of the matrix algebra including the identity "
        "∇_β (β^T A β) = (A + A^T) β when A is symmetric. State two conditions on X under which the solution "
        "β* = (X^T X)^{-1} X^T y is well-defined, and describe what regularization choice (ridge vs. lasso) "
        "you would apply if the design matrix is rank-deficient. Conclude with the geometric interpretation: "
        "Xβ* is the orthogonal projection of y onto the column space of X."
    ),
    (
        "Prove by induction that the sum 1 + 2 + 3 + ... + n equals n(n+1)/2 for every positive integer n. "
        "State the base case, the inductive hypothesis, and the inductive step clearly. Then give an independent "
        "second proof via Gauss's pairing argument (pair the first and last terms, second and second-to-last, etc.). "
        "Discuss when induction is preferable to direct proof and vice versa. Finally, generalize to the sum of "
        "squares 1 + 4 + 9 + ... + n^2 = n(n+1)(2n+1)/6 and sketch the inductive proof without working it out fully, "
        "pointing out where the difficulty lies relative to the linear case."
    ),
    (
        "Explain the Monty Hall problem and prove rigorously that switching doors wins with probability 2/3 when "
        "the host always reveals a losing door. Set up the sample space explicitly, enumerate all six equally-likely "
        "outcomes (choice of car door × choice of contestant's initial door), and compute the probability of winning "
        "under each strategy. Address the common objection that 'after the reveal, there are two doors so it should "
        "be 50/50.' Show what changes if the host is adversarial (reveals your door whenever you're wrong) or "
        "uninformed (reveals a random non-contestant door, possibly the car). These two variants give 0 and 1/2 "
        "respectively — prove one of them."
    ),
    (
        "Compute the eigenvalues and eigenvectors of the 2x2 matrix A = [[4, -2], [1, 1]] from scratch. First "
        "write down the characteristic polynomial det(A - λI) = (4-λ)(1-λ) - (-2)(1), expand it, and solve the "
        "resulting quadratic for λ. For each eigenvalue, substitute back into (A - λI)v = 0 and find a non-zero "
        "solution v. Verify your answer by checking Av = λv element-wise. Finally, use the eigendecomposition "
        "to compute A^10 without directly multiplying the matrix ten times — show the factorization A = PDP^{-1} "
        "and the resulting formula A^10 = PD^10 P^{-1}."
    ),
    (
        "Solve the following combinatorial puzzle: in how many ways can you tile a 2xn rectangle using 1x2 dominoes? "
        "Set up the recurrence by conditioning on the last tile placed — if it's vertical, you have a 2x(n-1) problem; "
        "if it's two stacked horizontals, you have a 2x(n-2) problem. This gives T(n) = T(n-1) + T(n-2) with T(0) = 1 "
        "and T(1) = 1, which is the Fibonacci sequence. Prove by strong induction that T(n) = F(n+1) where F(k) is "
        "the k-th Fibonacci number. Finish with a generating-function derivation of the same result and explain "
        "why Binet's formula gives a closed-form answer using the golden ratio."
    ),

    # --- Prose / narrative (5) ---
    (
        "Write a short story opening (about 200 words) set aboard a generation ship 300 years into its 900-year "
        "voyage from Earth to Tau Ceti. The protagonist, Mira, is the seventh generation of a family line that will "
        "never see either planet. She has just discovered that the ship's onboard historian, whom she idolizes, "
        "has been systematically rewriting the ship's public archives — not to falsify the past, but to soften the "
        "accounts of the founding generation's compromises. The story should introduce the setting, hint at the "
        "discovery without revealing it overtly, and end on a line of dialogue that forces Mira to choose between "
        "telling her family or confronting the historian privately. Use present tense. Favor concrete imagery "
        "over abstract introspection."
    ),
    (
        "Describe, in the style of a turn-of-the-century nature essay, a late-autumn morning in a New England "
        "hardwood forest the week after the first killing frost. Focus on the sensory details: the slant of low "
        "sun through leafless birches, the smell of decaying oak leaves mixed with damp earth, the sound of a "
        "solitary downy woodpecker working a dead branch forty feet overhead. Mention at least three species of "
        "trees by common name and what their bark reveals in winter when identification by leaves is no longer "
        "possible. Finish with a reflection on the particular quality of silence in a deciduous forest in the "
        "weeks between leaf-fall and the first snow, when the woods feel briefly emptied but are in fact still "
        "full of small mammals and winter birds."
    ),
    (
        "Draft a letter from a grandmother to her newly-adult granddaughter giving practical advice on managing "
        "money in her twenties. The letter should be warm but not sentimental. Cover: (1) the difference between "
        "emergency savings and investment accounts and why you need both before thinking about anything else, "
        "(2) the concept of lifestyle creep and one concrete rule of thumb for avoiding it as income grows, "
        "(3) the psychological trap of checking market prices daily for long-term holdings, (4) the one-line rule "
        "'never carry a credit card balance,' and (5) a final observation about how money stops being a daily "
        "anxiety at whatever income level you stop comparing yours to other people's. Sign it with an affectionate "
        "closing and a P.S. about something unrelated and specific."
    ),
    (
        "Continue this fragment for about 250 words: 'The train had been stopped at the rural station for forty "
        "minutes before the conductor finally walked through the car with the news.' Maintain the ambiguous, "
        "slightly dreamlike tone of the opening. Introduce one other passenger who the protagonist notices for a "
        "specific reason — perhaps the book they're reading, or a gesture repeated three times. Keep the cause of "
        "the delay deliberately off-stage; it is mentioned, perhaps, but never described. Dialogue is welcome but "
        "should be sparse and not used to explain anything. The piece should end before the train moves again, "
        "at a moment of sudden, inexplicable clarity about something the protagonist has been putting off deciding."
    ),
    (
        "Write a fictional diary entry from someone serving in the Antarctic Survey at Halley VI Research Station "
        "during the deep winter, mid-July, when the sun has been below the horizon for two months. Describe one "
        "specific task that occupied the day — perhaps recalibrating the lidar array after a snowdrift partially "
        "buried the mount — in enough technical detail to feel authentic but not dry. Include a short paragraph "
        "on what the station's nine overwintering staff did for dinner that night, and a final paragraph on the "
        "quality of the aurora observed afterward, in terms that communicate both the physical phenomenon and what "
        "it feels like to witness at latitude 75°S after seven weeks of continuous darkness. Keep the narrator's "
        "emotional register understated; scientists on the ice tend to downplay."
    ),

    # --- Technical explanation (5) ---
    (
        "Explain how modern solid-state drives manage wear leveling internally, assuming the reader knows what "
        "a flash cell is but nothing about FTL (flash translation layer) design. Cover: (1) why NAND cells have "
        "a finite program/erase cycle count — typically 1,000 for TLC and 100 for QLC — and what physically wears "
        "out when you program them repeatedly, (2) how the FTL maintains a logical-to-physical block address map "
        "so that overwriting LBA 42 does not actually overwrite the same physical cells, (3) the distinction "
        "between dynamic wear leveling (spreading hot writes across free blocks) and static wear leveling "
        "(proactively migrating cold data from low-erase-count blocks so those blocks become available for hot "
        "writes), and (4) how garbage collection interacts with the TRIM command to avoid write amplification. "
        "Finish with one sentence on why high-end enterprise SSDs report wear as a percentage of rated total "
        "bytes written rather than elapsed calendar time."
    ),
    (
        "Describe how the Border Gateway Protocol (BGP) propagates routes across the public internet and what "
        "makes a route 'preferred' when multiple paths exist. Start with the distinction between iBGP and eBGP, "
        "the role of autonomous system (AS) numbers, and the basic mechanics of an UPDATE message carrying NLRI "
        "plus path attributes. Explain the standard BGP path-selection algorithm in order of tie-breakers: highest "
        "LOCAL_PREF, shortest AS_PATH, lowest origin type, lowest MED, eBGP over iBGP, lowest IGP metric to next "
        "hop, then oldest route. Give a real-world example of why an operator might prefer LOCAL_PREF manipulation "
        "over AS_PATH prepending to shape inbound vs. outbound traffic. Conclude with the 2008 Pakistan-YouTube "
        "incident — a misconfigured route got accepted globally in minutes — as a motivating example of why RPKI "
        "origin validation matters today."
    ),
    (
        "Walk through how CRISPR-Cas9 gene editing works at the molecular level, targeting a reader who "
        "understands high school biology. Start with the bacterial immune system origin: phage DNA fragments are "
        "stored in CRISPR arrays, transcribed to crRNA, and paired with a Cas9 endonuclease to cleave matching "
        "phage DNA on subsequent infection. Explain how researchers adapted this — by fusing the crRNA and "
        "tracrRNA into a single guide RNA (sgRNA) that's straightforward to design — into a programmable "
        "double-strand-break tool. Describe the two DNA repair pathways that cells use at the break site: "
        "non-homologous end joining (error-prone, used for gene knockout) and homology-directed repair "
        "(template-dependent, used for precise edits). Mention protospacer adjacent motif (PAM) recognition as "
        "a constraint on what sequences Cas9 can target. Finish with one sentence on off-target effects and "
        "the improvements base editing and prime editing offer over direct double-strand breaks."
    ),
    (
        "Explain how a modern public-key infrastructure (PKI) authenticates HTTPS connections, targeting a reader "
        "who understands symmetric encryption but has never inspected a certificate chain. Walk through the TLS "
        "1.3 handshake: ClientHello with supported cipher suites and ephemeral Diffie-Hellman key share, server "
        "response with certificate chain and its own key share, ClientKeyShare and Finished messages. Explain why "
        "the server's RSA or ECDSA key in its leaf certificate is used only to sign the handshake transcript, not "
        "to encrypt the session — the actual session key comes from the ECDHE exchange, which gives forward "
        "secrecy. Describe what OCSP stapling is and why browsers preferred it over traditional OCSP lookups for "
        "revocation checking. Mention Certificate Transparency logs, their Merkle tree structure, and how they "
        "make certificate misissuance detectable after the fact. Close with why CT-compliance is now mandatory "
        "in major browsers and what happens when a root CA fails a compliance audit."
    ),
    (
        "Describe the physiology of human cardiac muscle contraction from the arrival of the action potential "
        "at the cell membrane to the mechanical shortening of the sarcomere. Include: depolarization via "
        "voltage-gated sodium channels, the subsequent calcium influx through L-type calcium channels during "
        "the plateau phase, calcium-induced calcium release from the sarcoplasmic reticulum via ryanodine "
        "receptors, calcium binding to troponin-C on the actin filament, the resulting conformational change "
        "that exposes myosin binding sites, the cross-bridge cycle with ATP binding and hydrolysis powering the "
        "power stroke, and finally SERCA pumping calcium back into the SR to trigger relaxation. Contrast this "
        "with skeletal muscle where the action potential directly triggers calcium release without requiring "
        "extracellular calcium influx. Finish with why cardiac muscle exhibits a long refractory period (~250 ms) "
        "and how this prevents tetanic contraction — critical for sustained pumping."
    ),

    # --- Dialog (5) ---
    (
        "Write a short dialog between two senior backend engineers pairing on a production incident. Alex is "
        "the on-call engineer and has paged Jordan, the service owner, at 3 AM because the API's p99 latency "
        "suddenly jumped from 80 ms to 2.4 seconds about fifteen minutes ago. Alex shares what they've already "
        "ruled out (no recent deploys, no traffic spike, no database CPU elevation). Jordan suggests checking the "
        "connection pool saturation and whether a slow downstream dependency is holding threads. The dialog "
        "should capture the back-and-forth of narrowing a hypothesis: Alex proposes, Jordan pushes back with "
        "counter-evidence, then they land on a likely culprit (a garbage-collection pause pattern suggesting "
        "a memory leak) and discuss whether to restart the affected pods or collect a heap dump first. Keep "
        "the tone professional but weary — it's 3 AM. End with Alex heading off to take the heap dump."
    ),
    (
        "Compose a dialog between two friends at a bookstore, one of whom recently quit a high-paying tech job "
        "to go to nursing school. Sam, the questioner, is skeptical but wants to understand; Robin, the career "
        "changer, wants to explain without sounding defensive. Cover the practical concerns: the pay cut of "
        "roughly 60%, the four-year timeline including prerequisites, the physical demands, and the uncertainty "
        "of whether they'll actually enjoy bedside nursing once they're doing it. Robin should mention a specific "
        "moment that catalyzed the decision (something small — helping an elderly neighbor through a post-surgical "
        "recovery) rather than a grand philosophical revelation. Sam should land somewhere between envious and "
        "worried by the end. Include one line of incidental banter about a book on display to break up the "
        "seriousness."
    ),
    (
        "Write an exchange between a parent and their nine-year-old child over a quiet Sunday breakfast. The "
        "child has just asked, in the unprompted way that children do, 'Why do people die?' The parent has to "
        "answer honestly without either frightening the child or pretending to know more than they do. Rather "
        "than a single long answer, the conversation should unfold as a back-and-forth — the child asks follow-up "
        "questions that reveal what they're really worried about (probably, once it becomes clear, a specific "
        "grandparent). The parent should resist the urge to reassure prematurely. Include one moment of humor "
        "that arises naturally — perhaps the child's wildly inaccurate guess about biology — that doesn't break "
        "the warmth of the exchange. End with a practical gesture: the parent suggesting they call the grandparent "
        "later that afternoon."
    ),
    (
        "Script a dialog between a translator and a technical writer reviewing the English-to-Japanese "
        "localization of a software manual. The translator, Keiko, has flagged three passages where the "
        "source English uses idioms that don't map cleanly — 'hit the ground running,' 'low-hanging fruit,' "
        "and 'move the needle.' The writer, Daniel, initially suggests just keeping the literal translation. "
        "Keiko explains, with patience, why each of these would read awkwardly or nonsensically in Japanese "
        "business prose, and offers two alternative rewordings per idiom that preserve meaning without the "
        "metaphor. Daniel should learn something. The dialog should reveal the craft of technical translation: "
        "it's not substitution, it's rewriting at the meaning layer. End with Keiko mentioning a glossary she "
        "maintains of English idioms she's replaced and why she's never kept one for idioms that went the other direction."
    ),
    (
        "Write a short exchange between a therapist and a new client in their first session together. The client, "
        "Dana, has come in at a partner's urging and is skeptical that therapy will help. The therapist, Dr. Rao, "
        "needs to establish rapport without either oversharing or performing neutrality. Dana should resist the "
        "obvious opening question — 'what brings you here today?' — with a dismissive deflection. Dr. Rao should "
        "gently acknowledge the resistance without confrontation, then shift to a practical question about what "
        "Dana would want to be different in six months if therapy did work. Dana's answer should be more specific "
        "than expected, revealing at least the shape of what they came in for. The session doesn't resolve "
        "anything — a first session shouldn't. End with Dr. Rao suggesting a topic for next week and Dana "
        "saying, with a small note of surprise, that they might actually come back."
    ),

    # --- Misc (5) ---
    (
        "Draft a detailed, realistic incident post-mortem for a fictional SaaS outage. Product: a collaborative "
        "document-editing platform. Impact: approximately 40 minutes of inability to save documents, affecting "
        "around 60% of active users in US and EU regions; no data loss. Root cause: a database migration to "
        "add a new index on the documents table took a lock level that blocked writes, due to a misconfigured "
        "ALTER TABLE that should have used CREATE INDEX CONCURRENTLY. Structure the post-mortem with: summary, "
        "timeline of events (noting when the first user report came in, when oncall acknowledged, when the "
        "migration was rolled back, when writes resumed), contributing factors beyond the immediate trigger "
        "(change review didn't catch the omission of CONCURRENTLY; staging environment had too little data to "
        "reveal the lock contention), corrective actions with owners and deadlines, and a final 'what went well' "
        "section that acknowledges the rollback procedure was exercised quickly."
    ),
    (
        "Write a thorough product review of a hypothetical pair of wireless noise-cancelling headphones priced "
        "at $349. Cover audio quality (bass vs. mid/treble balance, soundstage for classical vs. electronic music), "
        "active noise cancellation effectiveness in three environments (airplane cabin, coffee shop, urban "
        "sidewalk with traffic), microphone quality for video calls, comfort over a four-hour session, battery "
        "life claims vs. measured performance, the companion app experience, multipoint Bluetooth pairing "
        "stability with a laptop + phone simultaneously, and the transparency/ambient-sound mode's naturalness. "
        "End with a direct comparison to the two leading competitors in that price bracket and a one-sentence "
        "recommendation noting who this headphone is best suited for and who would be better served by an "
        "alternative. Maintain the register of a reputable audio reviewer — specific, skeptical of marketing claims, "
        "willing to mark down for small flaws that cumulatively matter."
    ),
    (
        "Produce a two-week itinerary for a first-time traveler visiting Japan in mid-April, budget-conscious but "
        "not backpacker-style (willing to spend on good food, not on luxury hotels). Cover Tokyo (days 1-4), a "
        "day trip to Kamakura, a travel day to Kyoto with overnight in Hakone (days 5-6), Kyoto (days 7-9), a day "
        "trip to Nara, Osaka (day 11), a travel day to Hiroshima and overnight (day 12), day trip to Miyajima, "
        "then return to Tokyo via Shinkansen on day 14. Include JR Pass strategy, cash-vs-IC-card for transit, "
        "two specific restaurant recommendations per city (one affordable, one splurge), and a note on which "
        "sights require advance reservations (Studio Ghibli Museum especially). Include the cherry blossom timing "
        "caveat — mid-April often catches late bloom in cooler regions — and how to check forecasts. Keep the "
        "tone practical: a friend writing for a friend, not a travel-agency brochure."
    ),
    (
        "Compose a full-length cover letter for a hypothetical software engineer applying to a staff engineering "
        "role at a mid-size B2B SaaS company. The applicant has seven years of experience, most recently leading "
        "a team of six on the company's data platform team. The target role emphasizes technical leadership, "
        "mentorship, and cross-team collaboration over hands-on coding. The letter should cover: one specific "
        "technical accomplishment with a quantifiable outcome (e.g., reduced mean pipeline latency by 40% through "
        "reworking the job scheduler), one specific leadership moment (e.g., coaching a junior engineer through "
        "a difficult oncall rotation), why this company specifically rather than a generic 'looking for a new "
        "challenge,' and an authentic closing that avoids hollow enthusiasm. Length: slightly over one page. "
        "Voice: confident without being boastful, specific without being grandiose."
    ),
    (
        "Write a thoughtful 400-word reflection on what it means to read slowly in an age of feed-based reading. "
        "Start with a specific observation about your own reading habits — perhaps how many books you've "
        "abandoned this year and why. Distinguish between the technical literacy of skimming for information "
        "and the older, more durable kind of reading that involves sitting with a single sentence long enough to "
        "find what else is in it. Argue that slow reading is not a nostalgic preference but a cognitive "
        "discipline, related to the ability to notice one's own inattention. Mention one specific author whose "
        "prose rewards slowing down (Marilynne Robinson is a strong candidate) and one whose prose does not — "
        "not in a pejorative sense but because the writing is designed to be absorbed at speed (Elmore Leonard). "
        "End with a practical suggestion: what a person who wants to read slower might actually change in how "
        "they choose books, arrange their reading environment, or track their own reading over a week."
    ),
]


# ---------------------------------------------------------------------------
# Architecture registry — each entry patches a specific model's attention
# forward to capture post-RoPE K and V as second-moment accumulators.
# ---------------------------------------------------------------------------


@dataclass
class _CaptureHandle:
    """Lifetime-manager for a monkey-patched attention forward."""

    restore: Callable[[], None]
    scores_k: dict[int, torch.Tensor]
    scores_v: dict[int, torch.Tensor]
    token_counts: dict[int, int]


def _accumulate_kv_scores(
    layer_idx: int,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scores_k: dict[int, torch.Tensor],
    scores_v: dict[int, torch.Tensor],
    token_counts: dict[int, int],
) -> None:
    with torch.no_grad():
        k32 = key_states.detach().to(torch.float32)
        v32 = value_states.detach().to(torch.float32)
        bsz, n_kv, seq, hdim = k32.shape
        k_flat = k32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
        v_flat = v32.permute(0, 2, 1, 3).reshape(-1, n_kv, hdim)
        k_sum = k_flat.square().sum(dim=0).to(torch.float64).cpu()
        v_sum = v_flat.square().sum(dim=0).to(torch.float64).cpu()
        if layer_idx in scores_k:
            scores_k[layer_idx] += k_sum
            scores_v[layer_idx] += v_sum
            token_counts[layer_idx] += bsz * seq
        else:
            scores_k[layer_idx] = k_sum
            scores_v[layer_idx] = v_sum
            token_counts[layer_idx] = bsz * seq


def _install_qwen3_capture() -> _CaptureHandle:
    from transformers.models.qwen3 import modeling_qwen3
    from transformers.models.qwen3.modeling_qwen3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_qwen3.Qwen3Attention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(
            self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_qwen3.Qwen3Attention.forward = patched_forward

    def restore():
        modeling_qwen3.Qwen3Attention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)


def _install_llama_capture() -> _CaptureHandle:
    from transformers.models.llama import modeling_llama
    from transformers.models.llama.modeling_llama import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_llama.LlamaAttention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(
            self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_llama.LlamaAttention.forward = patched_forward

    def restore():
        modeling_llama.LlamaAttention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)


def _install_mistral_capture() -> _CaptureHandle:
    from transformers.models.mistral import modeling_mistral
    from transformers.models.mistral.modeling_mistral import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_mistral.MistralAttention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(
            self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_mistral.MistralAttention.forward = patched_forward

    def restore():
        modeling_mistral.MistralAttention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)


def _install_phi3_capture() -> _CaptureHandle:
    from transformers.models.phi3 import modeling_phi3
    from transformers.models.phi3.modeling_phi3 import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    scores_k: dict[int, torch.Tensor] = {}
    scores_v: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    original = modeling_phi3.Phi3Attention.forward

    def patched_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q_size = self.config.num_attention_heads * self.head_dim
        kv_size = self.config.num_key_value_heads * self.head_dim
        qkv = self.qkv_proj(hidden_states)
        query_states = qkv[..., :q_size].view(hidden_shape).transpose(1, 2)
        key_states = qkv[..., q_size : q_size + kv_size].view(hidden_shape).transpose(1, 2)
        value_states = qkv[..., q_size + kv_size :].view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        _accumulate_kv_scores(
            self.layer_idx, key_states, value_states, scores_k, scores_v, token_counts
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        attn_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attn_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=getattr(self, "sliding_window", None),
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    modeling_phi3.Phi3Attention.forward = patched_forward

    def restore():
        modeling_phi3.Phi3Attention.forward = original

    return _CaptureHandle(restore=restore, scores_k=scores_k, scores_v=scores_v, token_counts=token_counts)


# Registry maps HF model architecture strings to capture installers.
# Add new entries as other families are validated.
_CAPTURE_INSTALLERS: dict[str, Callable[[], _CaptureHandle]] = {
    "Qwen3ForCausalLM": _install_qwen3_capture,
    "LlamaForCausalLM": _install_llama_capture,
    "MistralForCausalLM": _install_mistral_capture,
    "Phi3ForCausalLM": _install_phi3_capture,
}


# ---------------------------------------------------------------------------
# Pre-flight validation
# ---------------------------------------------------------------------------


def _extract_architecture_params(config: dict) -> tuple[str, int, int, int]:
    """Return (architecture, head_dim, num_kv_heads, num_hidden_layers).

    Handles nested text_config (Gemma-style multimodal) when values aren't at
    the top level. Derives head_dim from hidden_size / num_attention_heads
    when the config omits head_dim (Llama 3 / Mistral / Phi-3 / SmolLM2).
    """
    arch = (config.get("architectures") or ["unknown"])[0]
    head_dim = config.get("head_dim")
    num_kv_heads = config.get("num_key_value_heads")
    num_layers = config.get("num_hidden_layers")
    hidden_size = config.get("hidden_size")
    num_attn_heads = config.get("num_attention_heads")

    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        head_dim = head_dim if head_dim is not None else text_config.get("head_dim")
        num_kv_heads = num_kv_heads if num_kv_heads is not None else text_config.get("num_key_value_heads")
        num_layers = num_layers if num_layers is not None else text_config.get("num_hidden_layers")
        hidden_size = hidden_size if hidden_size is not None else text_config.get("hidden_size")
        num_attn_heads = num_attn_heads if num_attn_heads is not None else text_config.get("num_attention_heads")

    if head_dim is None and hidden_size and num_attn_heads:
        head_dim = hidden_size // num_attn_heads

    return arch, head_dim, num_kv_heads, num_layers


def check_calibration_preconditions(
    model_dir: Path | str,
    kv_cache_dtype: str,
) -> tuple[bool, str]:
    """Return (ok, reason). Reason always carries enough info to act on."""
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False, f"No config.json at {model_dir}"

    try:
        config = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"config.json malformed: {exc}"

    # Refuse pre-quantized source weights — activation stats would be biased.
    if config.get("quantization_config"):
        method = config["quantization_config"].get("quant_method", "unknown")
        return False, (
            f"Source weights are already quantized ({method}). Activation calibration "
            f"on pre-quantized weights biases the variance estimate. Requires dequantized "
            f"source or a different calibration path (out of scope)."
        )

    arch, head_dim, num_kv_heads, num_layers = _extract_architecture_params(config)

    if head_dim is None or num_kv_heads is None or num_layers is None:
        return False, (
            f"Could not resolve head_dim / num_kv_heads / num_hidden_layers from "
            f"config.json (got {head_dim} / {num_kv_heads} / {num_layers})."
        )

    if head_dim % TURBOQUANT_GROUP_ALIGNMENT != 0:
        return False, f"head_dim {head_dim} is not a multiple of {TURBOQUANT_GROUP_ALIGNMENT}."

    text_config = config.get("text_config") or {}
    global_head_dim = text_config.get("global_head_dim")
    if global_head_dim is not None and global_head_dim != head_dim:
        return False, (
            f"Variable head_dim detected (head_dim={head_dim}, global_head_dim={global_head_dim}). "
            f"Mixed-head-dim architectures need per-layer metadata, not supported."
        )

    if arch not in _CAPTURE_INSTALLERS:
        return False, (
            f"Architecture {arch!r} has no capture wrapper registered. "
            f"Supported: {sorted(_CAPTURE_INSTALLERS)}. Add a handler in "
            f"_CAPTURE_INSTALLERS to enable."
        )

    if kv_cache_dtype not in TURBOQUANT_OUTLIER_RATIOS:
        return False, (
            f"Unknown kv_cache_dtype {kv_cache_dtype!r}. Supported: "
            f"{sorted(TURBOQUANT_OUTLIER_RATIOS)}."
        )

    try:
        outlier_count = get_turboquant_outlier_count(head_dim, kv_cache_dtype)
    except ValueError as exc:
        return False, str(exc)

    return True, (
        f"OK: arch={arch}, head_dim={head_dim}, num_kv_heads={num_kv_heads}, "
        f"num_layers={num_layers}, outlier_count={outlier_count}"
    )


# ---------------------------------------------------------------------------
# Calibration entry point
# ---------------------------------------------------------------------------


def generate_turboquant_metadata(
    model_dir: Path | str,
    kv_cache_dtype: str,
    calibration_prompts: list[str] | None = None,
    max_seq_len: int = 1024,
    output_path: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Calibrate a model and write turboquant_kv.json to its directory.

    Returns the output path. Raises ValueError on precondition failure.
    """
    model_dir = Path(model_dir)
    ok, reason = check_calibration_preconditions(model_dir, kv_cache_dtype)
    if not ok:
        raise ValueError(f"Cannot calibrate {model_dir}: {reason}")

    if calibration_prompts is None:
        calibration_prompts = DEFAULT_CALIBRATION_PROMPTS

    output_path = Path(output_path) if output_path else (model_dir / "turboquant_kv.json")

    def _log(msg: str) -> None:
        if progress is not None:
            progress(msg)
        else:
            print(msg, flush=True)

    config = json.loads((model_dir / "config.json").read_text())
    arch, head_dim, num_kv_heads, num_layers = _extract_architecture_params(config)
    outlier_count = get_turboquant_outlier_count(head_dim, kv_cache_dtype)

    _log(
        f"[calibrate] model={model_dir.name} arch={arch} head_dim={head_dim} "
        f"num_kv_heads={num_kv_heads} num_layers={num_layers} recipe={kv_cache_dtype} "
        f"outlier_count={outlier_count}"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    handle = _CAPTURE_INSTALLERS[arch]()

    model = None
    total_tokens_processed = 0
    start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        _log(f"[calibrate] loading model (bf16, device_map=auto)...")
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        model.eval()

        with torch.no_grad():
            for idx, prompt in enumerate(calibration_prompts, start=1):
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                seq_len = int(inputs["input_ids"].shape[-1])
                model(**inputs, use_cache=False)
                total_tokens_processed += seq_len
                _log(
                    f"[calibrate]   prompt {idx}/{len(calibration_prompts)}: "
                    f"{seq_len} tokens (total {total_tokens_processed})"
                )

        elapsed = time.perf_counter() - start

        metadata = {
            "version": 1,
            "recipe": kv_cache_dtype,
            "head_size": head_dim,
            "model_name": config.get("_name_or_path") or model_dir.name,
            "transform_version": "structured_hadamard_v1",
            "codebook_version": "lloyd_beta_v1",
            "calibration": {
                "method": "activation_second_moment_top_k",
                "objective": "per_kv_head_per_channel",
                "num_prompts": len(calibration_prompts),
                "max_seq_len": max_seq_len,
                "num_observed_tokens": total_tokens_processed,
                "dtype": "bfloat16",
                "device": str(model.device),
            },
            "layers": {},
        }

        for layer_idx in range(num_layers):
            if layer_idx not in handle.scores_k:
                raise RuntimeError(
                    f"Layer {layer_idx} produced no activations during calibration — "
                    f"check that the capture wrapper is wired correctly."
                )
            k_score = handle.scores_k[layer_idx]  # [num_kv_heads, head_dim]
            v_score = handle.scores_v[layer_idx]

            if k_score.shape != (num_kv_heads, head_dim):
                raise RuntimeError(
                    f"Layer {layer_idx} K score has shape {k_score.shape}, "
                    f"expected {(num_kv_heads, head_dim)}."
                )

            k_top = torch.sort(
                torch.topk(k_score, k=outlier_count, dim=-1).indices, dim=-1
            ).values.tolist()
            v_top = torch.sort(
                torch.topk(v_score, k=outlier_count, dim=-1).indices, dim=-1
            ).values.tolist()

            metadata["layers"][f"model.layers.{layer_idx}.self_attn"] = {
                "key_high_precision_indices": k_top,
                "value_high_precision_indices": v_top,
            }

        output_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        size_kb = output_path.stat().st_size / 1024.0
        _log(
            f"[calibrate] wrote {output_path} ({size_kb:.1f} KB, {num_layers} layers, "
            f"{total_tokens_processed} tokens, elapsed {elapsed:.1f}s)"
        )

    finally:
        handle.restore()
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return output_path


def ensure_turboquant_metadata(
    model_dir: Path | str,
    kv_cache_dtype: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, bool]:
    """If metadata is missing and prerequisites hold, generate it.

    Returns (path, generated_bool). If metadata exists, skips; if
    precondition check refuses, raises ValueError with the reason.
    """
    model_dir = Path(model_dir)
    metadata_path = model_dir / "turboquant_kv.json"
    if metadata_path.is_file():
        return metadata_path, False

    generated_path = generate_turboquant_metadata(
        model_dir=model_dir,
        kv_cache_dtype=kv_cache_dtype,
        progress=progress,
    )
    return generated_path, True
