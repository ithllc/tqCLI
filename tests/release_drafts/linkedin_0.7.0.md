# tqCLI 0.7.0 — LinkedIn Launch Drafts

> Drafted via the **`tq-linkedin`** skill at `.claude/skills/tq-linkedin/`.
> Output is markdown for the user to paste manually — this skill never auto-posts.
> User should pick ONE variant and edit to taste before posting.

---

## Variant 1 — Short (<500 chars)

tqCLI 0.7.0 is out.

TurboQuant fork wheels for llama.cpp and vLLM now install with one command — including a CUDA 13.0 build with a PTX hedge for Rubin.

```
pip install 'turboquant-cli[llama-tq]'
```

https://github.com/tqcli/tqcli/releases/tag/v0.7.0

#LocalLLM #OpenSource #LLMInference

---

## Variant 2 — Medium (500–1500 chars)

Shipped tqCLI 0.7.0 today.

The headline: TurboQuant fork wheels for `llama-cpp-python` and `vLLM` are now installable in one command. Before today, anyone wanting TurboQuant KV cache compression had to clone two forks and compile vLLM from source against CUDA 13.0 — a 6-hour build on a 16-vCPU box. Now it is `pip install 'turboquant-cli[llama-tq]'` (PyPI) or `pip install 'turboquant-cli[vllm-tq]' --find-links <release>` (GitHub Release for the vLLM fork).

Under the hood: cibuildwheel matrix on PyPI for the llama.cpp fork (Linux + macOS + Windows × Py 3.10–3.12 × CPU + CUDA + Metal), plus a `vllm-turboquant` build with `TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"` — Ampere + Ada + Hopper + DC Blackwell + consumer Blackwell + DGX Spark / GB10 + PTX hedge for Rubin. Engine Auditor at startup detects fork-vs-upstream and prints the exact `pip install` command if you are running upstream on capable hardware.

Built by Ivey Technology Holdings LLC. Sponsors at https://github.com/sponsors/ithllc help cover the rebuild compute (~$20 per cut).

Release notes: https://github.com/tqcli/tqcli/releases/tag/v0.7.0

#LocalLLM #OpenSource #LLMInference #vLLM #CUDA

---

## Variant 3 — Long (1500–3000 chars)

tqCLI 0.7.0 shipped today. Here is what is in it and why it matters.

**What is new**

TurboQuant KV cache compression is now installable as a single `pip install`. Two engine forks (`llama-cpp-python-turboquant` and `vllm-turboquant`) ship as wheels: the llama.cpp fork on PyPI as a full cibuildwheel matrix, the vLLM fork as a GitHub Release built against CUDA 13.0 on a rented GCP VM. Blackwell hardware (sm_100 / sm_120 / sm_121 — DC, consumer, DGX Spark) opts into a separate `[vllm-tq-blackwell]` extra so a wheel resolver does not push sm_100+ kernels onto an Ada box.

**Why this release**

Before today, running TurboQuant on vLLM meant cloning the fork, installing CUDA 13.0 by hand, exporting a long `TORCH_CUDA_ARCH_LIST`, and compiling for ~6 hours on a 16-vCPU box. That kept TurboQuant locked behind an "I will set up an afternoon for this" gate. The 0.7.0 release moves the build cost from every-user-once to maintainer-occasional: I rent a GCP VM, build three Python-version wheels in parallel, attach them to a GitHub Release, and the wheel goes out at one-pip-install latency.

A few engineering details I am proud of:

- The Engine Auditor reads a single sentinel attribute on each engine module (`TURBOQUANT_BUILD` / `TURBOQUANT_ENABLED`). If the sentinel is missing on capable hardware, it prints a yellow panel with the exact `pip install` to fix it — no version-string parsing.
- A stderr-ordering contract in agent modes: `console.file.flush()` before the AgentOrchestrator's first stream chunk so the audit panel cannot interleave with `<tool_call>` tags.
- A community-verify script that prints an explicit consent manifest BEFORE collecting anything. Two modes: `--auto-report` (uses `gh` CLI; never reads or ships tokens) or `--manual` (paste-into-issue).

**How to try it**

```
pip install 'turboquant-cli[llama-tq]'                          # any platform
pip install 'turboquant-cli[vllm-tq]' \\
  --find-links https://github.com/tqcli/vllm-turboquant/releases/latest
```

**What's next**

Mac verifiers: if you have an M-series, an Intel Mac, or a Mac mini and 15 minutes, run `scripts/community_verify.sh --manual` and paste the output into the issue template at the repo. Total verification spend across all hardware cells (Ampere, Ada, Blackwell consumer, B200, GB10, three Mac chips) was about $5.35 — the rest came out of community goodwill.

Sponsors at https://github.com/sponsors/ithllc help cover the rebuild compute (~$20 per cut, realistic cadence of 2–4 cuts per year).

Built by Ivey Technology Holdings LLC. Released under Apache-2.0.

Release notes: https://github.com/tqcli/tqcli/releases/tag/v0.7.0

#LocalLLM #OpenSource #LLMInference #vLLM #llamacpp

---

## Notes for the user before posting

- Pick ONE variant. The medium variant is usually the right cadence for a 0.x.0 release.
- LinkedIn deprioritizes posts with 8+ hashtags; each variant above stays at ≤5.
- If you want emojis, opt in explicitly when you edit — defaults are text-only.
- Tag specific verifiers + early users in a comment after posting (more reach than tagging in the post body).
- Posting from a personal profile reads better than the LLC company page for OSS releases.
