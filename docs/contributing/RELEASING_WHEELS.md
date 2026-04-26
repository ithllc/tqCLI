# Releasing TurboQuant Wheels

Maintainer runbook for cutting a new wheel set for the two TurboQuant forks
(`tqcli/llama-cpp-turboquant`, `tqcli/vllm-turboquant`). Companion to
[`docs/prd/PRD_turboquant_wheel_distribution.md`](../prd/PRD_turboquant_wheel_distribution.md)
and [`docs/technical_plans/TP_turboquant_wheel_distribution.md`](../technical_plans/TP_turboquant_wheel_distribution.md).

## When to cut a new wheel

- The TurboQuant fork has a meaningful kernel change (new KV recipe, fix to
  an existing recipe, perf regression patch).
- A new CUDA toolkit drops that the fork should compile against (e.g.
  CUDA 13.0 was the trigger for 0.7.0 because sm_121 / DGX Spark needs it).
- Upstream `vllm` or `llama.cpp` has shipped a security or stability fix
  worth rebasing onto.

Cosmetic fork commits (READMEs, comments, test fixtures) do NOT need a new
wheel — wait for a code-touching change.

## llama-cpp-python-turboquant — automated via cibuildwheel

Workflow lives at `tqcli/llama-cpp-turboquant/.github/workflows/wheels.yml`.
PyPI Trusted Publishing is wired up via OIDC; no API tokens are stored.

```bash
# In a clone of `tqcli/llama-cpp-turboquant`:
git checkout main && git pull
git tag v0.3.0-tq2          # bump the tq suffix; lockstep with llama.cpp upstream
git push --tags             # triggers wheels.yml
```

The workflow runs the cibuildwheel matrix (Linux + macOS arm64 + macOS
x86_64 + Windows) × Python 3.10 / 3.11 / 3.12 × CPU + CUDA 12.8 + Metal.
On green, `pypa/gh-action-pypi-publish@release/v1` ships every wheel to
PyPI. The same wheels are mirrored to a GitHub Release page via
`softprops/action-gh-release@v2` for users who prefer `--find-links`.

**Verification (clean venv, minute one after publish):**

```bash
python -m venv /tmp/v && source /tmp/v/bin/activate
pip install llama-cpp-python-turboquant
python -c "import llama_cpp; assert llama_cpp.TURBOQUANT_BUILD is True"
```

## vllm-turboquant — manual GCP build

GitHub Actions free runners do not have enough RAM to compile vLLM with the
full Blackwell arch list. The build happens on a rented GCP
`n2-standard-16` on-demand VM (~$4.66 / VM, three VMs in parallel for the
three Python versions, ~6 hr each, ~$14 total).

### Pre-flight

1. **Pin the golden commit** on the fork. Run the Section C.2 integration
   set on the candidate SHA:
   - Gemma 4 E2B + BNB_INT4 + CPU offload + turboquant35
   - Qwen 3 4B + calibrated `turboquant_kv.json`

   Both must be green. Tag `v0.7.0-tq{N}` on the fork.
2. **Run TP C2a dependency-harmony check** before the rebuild —
   `pip install -e ".[vllm-tq]" --find-links <release> && pytest tests/test_agent_orchestrator.py -x`.
   If the new wheel's pinned `torch` / `numpy` / `pydantic` downgrades an
   agent-orchestrator dep, escalate before tagging.
3. **Gemini cross-check** — run the Section 2.A prompt in
   `docs/prompts/ship_turboquant_wheels.md` to surface any new
   compute-capability requirements landed in upstream vllm in the last 30
   days.

### Build

```bash
# scripts/build_wheel_gcp.sh on the vllm-turboquant fork:
gcloud compute instances create tq-build-py312 \
    --machine-type=n2-standard-16 \
    --image-family=ubuntu-2404-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size=200GB --zone=us-central1-a

# On the VM, after ssh'ing in:
sudo apt-get install -y cuda-toolkit-13-0 gcc-11
python3.12 -m venv .v && source .v/bin/activate
export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0 12.1+PTX"
export MAX_JOBS=4 NVCC_THREADS=4 VLLM_TARGET_DEVICE=cuda
pip install --upgrade pip build wheel setuptools "torch>=2.6" ninja
python -m build --wheel --outdir dist/
sha256sum dist/*.whl > dist/SHA256SUMS

# Upload + tear down
gsutil cp dist/* gs://tqcli-wheel-build/0.7.0-tq{N}/
gcloud compute instances delete tq-build-py312 --zone=us-central1-a -q
```

Run three VMs in parallel for Python 3.10, 3.11, 3.12. Total wall time
~6 hr; total cost ~$14.

### Wheel-size gate

If any `.whl` exceeds **2 GB**, STOP and escalate. GitHub Releases caps
asset size at 2 GB / file. Two paths:

- **(a) LFS host the wheel** — works only up to 2 GB / file regardless.
- **(b) Split into `vllm-turboquant` (Ampere/Ada/Hopper) + `vllm-turboquant-blackwell`
  (sm_100 / sm_120 / sm_121)** — the [vllm-tq-blackwell] extra in
  `pyproject.toml` already anticipates this split.

Do NOT decide unilaterally; the umbrella package's `[all]` extra changes
behavior depending on which path is taken.

### Release

```bash
gh release create v0.7.0-tq{N} --repo tqcli/vllm-turboquant \
  --title "vllm-turboquant 0.7.0-tq{N} (post-Issue-22, CUDA 13.0)" \
  --notes-file RELEASE_NOTES.md \
  dist/*.whl dist/SHA256SUMS
```

Release body must paste Section C.2 numbers verbatim as the provenance
story.

### Verification

```bash
# Vast.ai RTX 4090 on-demand, 3 hr (~$0.87)
pip install vllm-turboquant \
  --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq{N}
python -c "import vllm; assert vllm.TURBOQUANT_ENABLED is True"
tqcli chat --model gemma-4-e2b-it-vllm --engine vllm \
  --kv-quant turbo3 --prompt "Paris?" --json
# Expect Section C.2 metadata: cpu_offload_gb, kv_cache_dtype=turboquant35
```

For Blackwell the same drill, but on Vast.ai RTX 5090 (sm_120, ~$0.51 / 1 hr)
+ Lambda Labs B200 (sm_100, ~$3.49 / 1 hr) + ASUS Ascent GX10 (sm_121,
on-hand, $0).

## Bumping the tqCLI pin

After both forks are published:

```python
# pyproject.toml
[project.optional-dependencies]
vllm-tq = [
    "vllm-turboquant==0.7.0.postYYYYMMDD",   # <-- update YYYYMMDD
    ...
]
vllm-tq-blackwell = [
    "vllm-turboquant-blackwell==0.7.0.postYYYYMMDD",
    ...
]
```

Then run TP C2a one last time on the umbrella package. If green, tag
`v0.7.{N+1}` on `tqcli/tqcli` and let the publish-pypi.yml workflow handle
the upload. Update CHANGELOG.md before tagging.
