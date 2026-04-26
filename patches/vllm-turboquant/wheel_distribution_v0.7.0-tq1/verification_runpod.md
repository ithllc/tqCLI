# RunPod Verification — vllm-turboquant 0.7.0-tq1

Workstream B step 11 + Section 3 cells V3, V4, V5. Total cost on RunPod
Community Cloud (per 2026-04-26 GraphQL price check): ~$7.69. Account already
validated (balance $500). The `runpodctl` CLI supports headless `pod create`,
which is what we use here.

The maintainer's `RUNPOD_API_KEY` lives outside the repo. Never commit it.

## Cell V3 — RTX 4090 (sm_8.9, Ada) — `vllm-turboquant`

```bash
runpodctl create pods \
    --name vllm-tq-v3-rtx4090 \
    --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
    --gpuType "NVIDIA GeForce RTX 4090" \
    --gpuCount 1 \
    --secureCloud=false \
    --containerDiskInGb 60 \
    --communityCloud=true
```

Once the pod is running, `runpodctl exec` (or web SSH):

```bash
pip install vllm-turboquant \
    --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1
python -c "
import vllm
from vllm.turboquant_arch_check import check_arch_compatibility
print('TURBOQUANT_ENABLED:', vllm.TURBOQUANT_ENABLED)
print('TURBOQUANT_BUILD_ARCH:', vllm.TURBOQUANT_BUILD_ARCH)
print('TURBOQUANT_BUILD_ARCH_LIST:', vllm.TURBOQUANT_BUILD_ARCH_LIST)
print('check_arch_compatibility():', check_arch_compatibility())
"
```

**Pass criterion (V3):**
- `TURBOQUANT_ENABLED: True`
- `TURBOQUANT_BUILD_ARCH: ampere-ada-hopper`
- `TURBOQUANT_BUILD_ARCH_LIST: 8.0 8.6 8.9 9.0`
- `check_arch_compatibility(): None`

End-to-end smoke (Gemma 4 E2B + turboquant35):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model google/gemma-2-2b-it \
    --kv-cache-dtype turboquant35 \
    --max-model-len 4096 &
sleep 60
curl -sS http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"google/gemma-2-2b-it","prompt":"Two plus two?","max_tokens":16}'
```

Expected wall time on RTX 4090: ~3 hours including pull + warmup. Cost ~$1.02.

## Cell V4 — RTX 5090 (sm_12.0, Blackwell consumer) — `vllm-turboquant-blackwell`

```bash
runpodctl create pods \
    --name vllm-tq-v4-rtx5090 \
    --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
    --gpuType "NVIDIA GeForce RTX 5090" \
    --gpuCount 1 \
    --secureCloud=false \
    --containerDiskInGb 60 \
    --communityCloud=true
```

```bash
pip install vllm-turboquant-blackwell \
    --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1
python -c "
import vllm
print(vllm.TURBOQUANT_ENABLED, vllm.TURBOQUANT_BUILD_ARCH, vllm.TURBOQUANT_BUILD_ARCH_LIST)
"
```

**Pass criterion (V4):**
- `True ampere-ada-hopper` would be the wrong wheel (FAIL).
- Required: `True blackwell 10.0 12.0 12.1+PTX`.

Cross-flavour mismatch test (paste this on the same RTX 5090 pod after V4
PASSes):

```bash
pip install --force-reinstall vllm-turboquant \
    --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1
python -c "
from vllm.turboquant_arch_check import assert_arch_compatibility
try:
    assert_arch_compatibility()
except RuntimeError as e:
    print('OK (raised as expected):', e)
"
```

**Pass criterion (V4 mismatch):** `RuntimeError` is raised; the message
contains `vllm-turboquant-blackwell` as the install hint.

Expected wall time ~1 hour. Cost ~$0.69.

## Cell V5 — B200 (sm_10.0, Blackwell DC) — `vllm-turboquant-blackwell`

```bash
runpodctl create pods \
    --name vllm-tq-v5-b200 \
    --imageName runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04 \
    --gpuType "NVIDIA B200" \
    --gpuCount 1 \
    --secureCloud=false \
    --containerDiskInGb 80 \
    --communityCloud=true
```

Same install + assertion block as V4. Pass criterion: `blackwell` arch +
`check_arch_compatibility()` returns `None` on sm_10.0.

Expected wall time ~1 hour. Cost ~$5.98 (Community Cloud, 2026-04-26 price).

## Cell V6 — GB10 (sm_12.1, DGX Spark) — own ASUS Ascent GX10

RunPod does not stock GB10 as of 2026-04-26. Run on the maintainer's own
ASUS Ascent GX10 (acquired 2026-04-25):

```bash
pip install vllm-turboquant-blackwell \
    --find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1
python -c "
import torch, vllm
from vllm.turboquant_arch_check import check_arch_compatibility
print('cap:', torch.cuda.get_device_capability(0))
print('TURBOQUANT_BUILD_ARCH:', vllm.TURBOQUANT_BUILD_ARCH)
print('check:', check_arch_compatibility())
"
```

Expected: `cap: (12, 1)`, `TURBOQUANT_BUILD_ARCH: blackwell`, `check: None`.
This is the only path that exercises the `+PTX` JIT — sm_12.1 falls back to
PTX from the sm_12.0 SASS unless the wheel embedded sm_12.1 SASS, which our
arch list does (`12.1+PTX` includes both).

Cost: $0 (own hardware).

## Summary table

| Cell | GPU                        | Cost       | Wall time | Assertion target               |
|------|----------------------------|------------|-----------|--------------------------------|
| V3   | RTX 4090                   | ~$1.02     | ~3 hrs    | ampere-ada-hopper PASS         |
| V4   | RTX 5090                   | ~$0.69     | ~1 hr     | blackwell PASS + mismatch FAIL |
| V5   | B200                       | ~$5.98     | ~1 hr     | blackwell PASS                 |
| V6   | GB10 (own ASUS Ascent GX10)| $0         | ~1 hr     | blackwell+PTX PASS             |
| **Total**                       | **~$7.69** | **~6 hrs** |                                |

## After verification

Edit `release_body.md` to fill in:
- Per-wheel sizes (from `gsutil ls -l gs://tqcli-wheel-build/0.7.0-tq1/`)
- The Verification table's `Result` column with PASS/FAIL
- The actual build date

Then post on the GitHub Release page (Edit -> paste body -> Save).

## Cleanup

```bash
runpodctl stop pod vllm-tq-v3-rtx4090
runpodctl stop pod vllm-tq-v4-rtx5090
runpodctl stop pod vllm-tq-v5-b200
runpodctl remove pod vllm-tq-v3-rtx4090
runpodctl remove pod vllm-tq-v4-rtx5090
runpodctl remove pod vllm-tq-v5-b200
```

Confirm RunPod balance after cleanup; we should have spent roughly $7.69
out of the $500 balance.
