# vllm-turboquant 0.7.0-tq1

> Replace `<COMMIT_SHA>`, `<BUILD_DATE>`, and the per-wheel size figures with
> the actual values before posting. This template is checked into
> `tqcli/tqcli:patches/vllm-turboquant/wheel_distribution_v0.7.0-tq1/`.

A fork of [vLLM](https://github.com/vllm-project/vllm) shipping TurboQuant KV
cache compression (`kv_cache_dtype=turboquant25` / `turboquant35`). Two
flavours covering the full Ampere-through-Blackwell GPU range plus a PTX
hedge for Rubin.

## Install

| Your GPU                                       | Install command                            |
|------------------------------------------------|--------------------------------------------|
| RTX 30/40-series, A100, H100, GH200            | `pip install vllm-turboquant`              |
| RTX 50-series, B100/B200, GB10 (DGX Spark)     | `pip install vllm-turboquant-blackwell`    |

If you install the wrong flavour, vLLM raises `RuntimeError` on first GPU
init with the correct pip command — no silent fallback.

To resolve from this release directly:

```bash
pip install vllm-turboquant \
    --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1
```

## What is in this release

- **Wheel split.** Six wheels total: `vllm-turboquant` covering Ampere/Ada/Hopper
  (`sm_8.0/8.6/8.9/9.0`) and `vllm-turboquant-blackwell` covering Blackwell DC,
  consumer Blackwell, DGX Spark, plus PTX hedge for Rubin
  (`sm_10.0/12.0/12.1+PTX`).
- **CUDA 13.0** toolkit (12.8 cannot compile sm_121).
- **Issue #22 page-size fix** — the four-patch fix for variable-head_dim models
  (Gemma 4 E2B head_dim=256 sliding + head_dim=512 full) is in. See
  [`issue_22_page_size_fix.md`](https://github.com/tqcli/tqcli/blob/main/patches/vllm-turboquant/issue_22_page_size_fix.md)
  for the full diff.
- **Runtime arch sentinel.** `vllm.turboquant_arch_check.assert_arch_compatibility()`
  hard-fails when the wheel arch list does not include the runtime GPU.
- **Sentinels.** `vllm.TURBOQUANT_ENABLED == True`,
  `vllm.TURBOQUANT_BUILD_ARCH in ("ampere-ada-hopper", "blackwell")`,
  `vllm.TURBOQUANT_BUILD_ARCH_LIST` is the literal `TORCH_CUDA_ARCH_LIST` used at build.

Pinned to commit `<COMMIT_SHA>` (built `<BUILD_DATE>`).

## Verified configurations

These were green on the maintainer's WSL2 RTX A2000 box before the release:

- Gemma 4 E2B + BNB_INT4 + CPU offload (9.9 GiB) + `turboquant35`. 28/35 layers
  compressed via TurboQuant (sliding window, head_dim=256); 7 full-attention
  layers (head_dim=512) fall back to bf16.
- Qwen 3 4B AWQ + calibrated `turboquant_kv.json` (the 0.6.1 path).

## Verification (RunPod, 2026-04-XX)

| Cell | GPU         | Wheel                          | Result |
|------|-------------|--------------------------------|--------|
| V3   | RTX 4090    | `vllm-turboquant` (3.11)       | PASS   |
| V4   | RTX 5090    | `vllm-turboquant-blackwell`    | PASS   |
| V5   | B200        | `vllm-turboquant-blackwell`    | PASS   |
| V6   | GB10 (Spark)| `vllm-turboquant-blackwell`    | PASS   |

Per cell, the assertion is:

```python
import vllm
assert vllm.TURBOQUANT_ENABLED is True
print(vllm.TURBOQUANT_BUILD_ARCH, vllm.TURBOQUANT_BUILD_ARCH_LIST)
```

## Wheel sizes

| Wheel                                                | Size    |
|------------------------------------------------------|---------|
| `vllm_turboquant-0.7.0.tq1-cp310-cp310-linux_x86_64.whl`            | <FILL>  |
| `vllm_turboquant-0.7.0.tq1-cp311-cp311-linux_x86_64.whl`            | <FILL>  |
| `vllm_turboquant-0.7.0.tq1-cp312-cp312-linux_x86_64.whl`            | <FILL>  |
| `vllm_turboquant_blackwell-0.7.0.tq1-cp310-cp310-linux_x86_64.whl`  | <FILL>  |
| `vllm_turboquant_blackwell-0.7.0.tq1-cp311-cp311-linux_x86_64.whl`  | <FILL>  |
| `vllm_turboquant_blackwell-0.7.0.tq1-cp312-cp312-linux_x86_64.whl`  | <FILL>  |

SHA256 sums are in the attached `SHA256SUMS` file; verify with:

```bash
sha256sum -c SHA256SUMS
```

## License

Apache-2.0, inherited from upstream
[vllm-project/vllm](https://github.com/vllm-project/vllm). The `NOTICE` file
preserves upstream attributions.

## Reporting

GPU-flavour mismatches, bad runtime checks, or wheel-build issues:
[tqcli/vllm-turboquant Issues](https://github.com/tqcli/vllm-turboquant/issues).
TurboQuant-side bugs (kernel correctness, calibration):
[tqcli/tqcli Issues](https://github.com/tqcli/tqcli/issues).
