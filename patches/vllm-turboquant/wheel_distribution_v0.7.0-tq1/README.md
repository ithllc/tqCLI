# vllm-turboquant Wheel Distribution — Workstream B Artifacts (v0.7.0-tq1)

**Status:** Staged for application to `github.com/tqcli/vllm-turboquant` fork.
**Date:** 2026-04-26
**Source playbook:** `docs/prompts/ship_turboquant_wheels.md` Section 1, Workstream B.

This directory holds the artifacts produced by Workstream B of the 0.7.0 wheel-
distribution effort. They are staged here in the `tqcli/tqcli` repo because the
worker that produced them does not have push access to the fork; the maintainer
applies them by copying the matching paths into the fork repo.

## Locked decisions (2026-04-26)

1. **Wheel split.** Six wheels total — `vllm-turboquant` (sm 8.0/8.6/8.9/9.0)
   plus `vllm-turboquant-blackwell` (sm 10.0/12.0/12.1+PTX), three Python
   versions each (3.10/3.11/3.12).
2. **Sequential single-VM build** on GCP `n2-standard-8`. ~30h wall time, ~$12
   compute. Stays within default 8-vCPU regional quota.
3. **GCP** project `tqcli-wheel-build`, billing `01124B-E52669-78A9D0`, bucket
   `gs://tqcli-wheel-build/`, $50 budget alert active.
4. **Runtime sentinel** `TURBOQUANT_BUILD_ARCH` plus
   `check_arch_compatibility()` hard-fails on mismatch.
5. **CUDA toolkit 13.0+** (12.8 cannot compile sm_121).

## Files in this directory

| Path                                | Maps to (in fork repo)                          | Purpose                                                                         |
|-------------------------------------|-------------------------------------------------|---------------------------------------------------------------------------------|
| `fork_changes.md`                   | (description only)                              | Authoritative diff description for `pyproject.toml`, `__init__.py`, `README.md` |
| `vllm/turboquant_arch_check.py`     | `vllm/turboquant_arch_check.py` (new)           | Runtime arch-vs-build sentinel check                                            |
| `vllm/__init__.py.snippet`          | append to `vllm/__init__.py`                    | Sentinel + arch fields populated at build time                                  |
| `scripts/build_wheel_gcp.sh`        | `scripts/build_wheel_gcp.sh` (new)              | Sequential six-wheel GCP build                                                  |
| `docs/RELEASING.md`                 | `docs/RELEASING.md` (new)                       | Maintainer runbook                                                              |
| `release_body.md`                   | (paste into `gh release create`)                | GitHub Release body for `v0.7.0-tq1`                                            |
| `verification_runpod.md`            | (operational doc)                               | V3/V4/V5 RunPod verification commands                                           |

## Workstream B step to file mapping

| Playbook step                              | Artifact                                              |
|--------------------------------------------|-------------------------------------------------------|
| 1 — pyproject.toml templated `name`        | `fork_changes.md` section pyproject.toml              |
| 2 — `vllm/__init__.py` sentinels           | `vllm/__init__.py.snippet`                            |
| 3 — `vllm/turboquant_arch_check.py`        | `vllm/turboquant_arch_check.py`                       |
| 4 — README + install table                 | `fork_changes.md` section README.md banner            |
| 5 — golden commit identification           | `fork_changes.md` section Pinning the golden commit   |
| 6 — tag `v0.7.0-tq1`                       | `docs/RELEASING.md` section Tag                       |
| 7 — `scripts/build_wheel_gcp.sh`           | `scripts/build_wheel_gcp.sh`                          |
| 8 — wheel size measurement                 | `docs/RELEASING.md` section Wheel size gate           |
| 9 — `gh release create v0.7.0-tq1`         | `release_body.md` + `docs/RELEASING.md`               |
| 10 — `docs/RELEASING.md`                   | `docs/RELEASING.md`                                   |
| 11 — clean-VM verification                 | `verification_runpod.md`                              |

## Deferred to maintainer (cannot be automated from this worktree)

- Identifying the exact golden commit SHA on the fork's `main` branch.
- Running the GCP build (`gcloud auth`, VM provisioning, ~30h wall time).
- Tagging `v0.7.0-tq1` on the fork.
- Creating the GitHub Release and uploading the six `.whl` files.
- Running V3/V4/V5 RunPod verification.

## Out of scope for this worktree

- The umbrella `tqcli` package (`tqcli/`, `pyproject.toml`, etc.) — that is
  Workstream C.
- The `llama-cpp-python-turboquant` fork — that is Workstream A.
- Any change to `kv_quantizer.py`, `vllm_backend.py`, or other `tqcli/core/*`
  modules in this repo.
