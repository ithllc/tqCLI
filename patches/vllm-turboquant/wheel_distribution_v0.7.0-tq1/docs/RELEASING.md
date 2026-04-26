# Releasing `vllm-turboquant` / `vllm-turboquant-blackwell` Wheels

Maintainer runbook for cutting a release from the GCP build pipeline (sequential
single-VM, ~30h wall time, ~$12 compute). This is the canonical reference; if
the build script disagrees with this doc, the script wins.

---

## 0. Prerequisites (one-time)

- `gcloud` CLI authenticated against the `tqcli-wheel-build` project.
- `gh` CLI authenticated; push access to `tqcli/vllm-turboquant`.
- `gsutil` available (ships with `gcloud`).
- A working internet connection that can hold open `gcloud compute ssh` for
  multi-hour stretches (or use `tmux`/`screen` on the local box).

GCP environment (already provisioned 2026-04-26):

| Item            | Value                                 |
|-----------------|---------------------------------------|
| Project         | `tqcli-wheel-build`                   |
| Billing account | `01124B-E52669-78A9D0`                |
| Bucket          | `gs://tqcli-wheel-build/`             |
| Budget alert    | $50, 50%/90%/100% of actual spend     |
| Default region  | `us-central1`                         |

---

## 1. Pin the golden commit on the fork

The release pins the commit AFTER Issue #22's four-patch page-size fix landed
(see `patches/vllm-turboquant/issue_22_page_size_fix.md` in `tqcli/tqcli`).
Verification gates BEFORE tagging:

1. `Gemma 4 E2B + BNB_INT4 + CPU offload (9.9 GB) + turboquant35` runs green
   on RTX A2000 4 GB. Section C.2 of
   `tests/integration_reports/turboquant_kv_comparison_report.md` is the
   canonical expected-output.
2. `Qwen 3 4B + calibrated turboquant_kv.json` (the 0.6.1 path) runs green on
   the same box.

Both must be green on the candidate SHA. Once they are:

```bash
cd ~/dev/vllm-turboquant
git checkout main && git pull
git tag -a v0.7.0-tq1 -m "vllm-turboquant 0.7.0-tq1 (Gemma 4 + BNB_INT4 + CPU offload + turboquant35 + Qwen 3 calibrator)"
git push origin v0.7.0-tq1
```

The release tag is the build script's anchor. After this, the build is fully
non-interactive.

---

## 2. Run the build

```bash
cd ~/dev/vllm-turboquant
bash scripts/build_wheel_gcp.sh
```

The script:

1. Provisions an `n2-standard-8` VM in `us-central1-a` (idempotent — reuses
   if already up).
2. Bootstraps the toolchain (CUDA 13.0, Python 3.10/3.11/3.12, ccache).
3. Clones the fork, checks out `v0.7.0-tq1`.
4. Loops over six builds in this order:
   1. `vllm-turboquant` for Python 3.10
   2. `vllm-turboquant` for Python 3.11
   3. `vllm-turboquant` for Python 3.12
   4. `vllm-turboquant-blackwell` for Python 3.10
   5. `vllm-turboquant-blackwell` for Python 3.11
   6. `vllm-turboquant-blackwell` for Python 3.12
5. Pushes each wheel + sha256 to `gs://tqcli-wheel-build/0.7.0-tq1/`.
6. Tears the VM down on success.

Wall time: ~30 hours. Compute cost: ~$11.64.

### If a build fails

```bash
# Leave the VM up for inspection
bash scripts/build_wheel_gcp.sh --keep-vm-on-error
gcloud compute ssh vllm-tq-builder --project=tqcli-wheel-build --zone=us-central1-a
# Investigate, then resume from the failed entry:
bash scripts/build_wheel_gcp.sh --resume-from blackwell-3.11
```

---

## 3. Wheel size gate (Workstream B step 8)

The helper script (`scripts/_build_one_wheel.sh`) hard-fails if any wheel
exceeds 2 GiB. If you hit this:

1. **STOP** — do not work around it. The 2 GiB ceiling is GitHub Releases'
   per-asset limit; LFS has its own quota and is not a free fallback.
2. Escalate to user. Two pre-approved fallback paths:
   - **GitHub LFS** for the oversize wheel(s) (separate quota; same 2 GiB
     per-file limit, so this only helps if LFS quota is the actual blocker).
   - **Further granularity** — split `vllm-turboquant-blackwell` into a DC
     wheel (sm_10.0) and a consumer/spark wheel (sm_12.0/12.1). The build
     script's matrix and helper take per-flavour `TORCH_CUDA_ARCH_LIST`, so
     this requires editing the matrix only, not the wheel-build logic.

For the 0.7.0-tq1 reference build the upstream vLLM wheel size sits at
~1.6 GiB; the TurboQuant additions add ~150 MiB of compiled kernels, so we
expect each flavour wheel at roughly 1.7-1.8 GiB. If the Blackwell flavour
crosses 2 GiB despite the split, suspect duplicate PTX-vs-SASS embedding for
sm_12.1+PTX — strip the PTX with `auditwheel` or drop the `+PTX` suffix.

---

## 4. Cut the GitHub Release

After the build script reports SUCCESS:

```bash
mkdir -p ./dist-release
gsutil cp gs://tqcli-wheel-build/0.7.0-tq1/* ./dist-release/

# Combined SHA256SUMS for the release body
( cd ./dist-release && sha256sum *.whl | tee SHA256SUMS )

gh release create v0.7.0-tq1 \
    --repo tqcli/vllm-turboquant \
    --title "vllm-turboquant 0.7.0-tq1" \
    --notes-file ./dist-release/RELEASE_BODY.md \
    ./dist-release/*.whl ./dist-release/SHA256SUMS
```

Use the body template at
`patches/vllm-turboquant/wheel_distribution_v0.7.0-tq1/release_body.md` in the
`tqcli/tqcli` repo. Substitute the actual commit SHA, build date, and per-wheel
sizes before publishing.

---

## 5. Verify the release on real GPUs (RunPod)

See `verification_runpod.md` for the V3/V4/V5 cells. Total cost ~$7.69 on
Community Cloud; budget for ~5h of operator time.

After all three RunPod cells pass, post the verification block on the GitHub
Release page (sub-section "Verification"). The body template has a
placeholder section.

---

## 6. Update the umbrella `tqcli` package

Workstream C pins the wheels in `tqcli/pyproject.toml`. The pin format:

```toml
[project.optional-dependencies]
vllm-tq          = ["vllm-turboquant==0.7.0.tq1", "bitsandbytes>=0.43.0", "accelerate>=0.30.0"]
vllm-tq-blackwell = ["vllm-turboquant-blackwell==0.7.0.tq1", "bitsandbytes>=0.43.0", "accelerate>=0.30.0"]
```

Users on Blackwell hardware install the `vllm-tq-blackwell` extra. The Engine
Auditor flags a mismatch on first run.

---

## 7. Rollback

If the release breaks installs in the wild:

1. Mark the GitHub Release as a pre-release (do NOT delete — mid-download
   users need stability).
2. Tag a `v0.7.1-tq1` patch from the next clean SHA.
3. Re-run `scripts/build_wheel_gcp.sh` with `TQ_RELEASE_TAG=v0.7.1-tq1`.
4. The umbrella `tqcli` package can hotfix-pin the new build via 0.7.1.

If the bug is specifically in the runtime arch check (false-positive on a
valid GPU), do NOT yank — that breaks the whole release. Ship a 0.7.1-tq1
that fixes `turboquant_arch_check.py` and re-builds; the broken check is in
the pure-Python module so a fix is a single file change away.

---

## 8. Cost ledger (per release)

| Item                                     | Cost        |
|------------------------------------------|-------------|
| GCP n2-standard-8, ~30h sequential       | ~$11.64     |
| GCS storage (~10 GiB, weeks)             | <$0.30/mo   |
| GCS egress (6 wheels x ~1.7 GiB)         | ~$0.60      |
| RunPod V3/V4/V5 verification             | ~$7.69      |
| **Total per release**                    | **~$20**    |

Realistic cadence: 4-8 releases the first year, settling to 2-4 per year. The
$50 budget alert covers 2x the per-release cost; if it fires, suspect a stuck
VM (the `trap_handler` in `build_wheel_gcp.sh` should prevent this).
