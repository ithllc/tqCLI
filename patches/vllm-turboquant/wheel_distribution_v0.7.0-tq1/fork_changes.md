# Fork Changes — `github.com/tqcli/vllm-turboquant` for v0.7.0-tq1

This file consolidates every textual change Workstream B requires the
maintainer to land on the fork. The build script (`scripts/build_wheel_gcp.sh`)
also rewrites several of these at build time on a per-flavour basis; that is
called out below.

---

## 1. `pyproject.toml` — build-time-templated `name`

The fork ships **two** distribution names from the same source tree:

| Flavour            | Distribution name              | TORCH_CUDA_ARCH_LIST           |
|--------------------|--------------------------------|--------------------------------|
| ampere-ada-hopper  | `vllm-turboquant`              | `8.0 8.6 8.9 9.0`              |
| blackwell          | `vllm-turboquant-blackwell`    | `10.0 12.0 12.1+PTX`           |

Set the `[project]` table to a sentinel that the build script rewrites in place:

```toml
[project]
name = "VLLM_TURBOQUANT_NAME_PLACEHOLDER"
version = "0.7.0.tq1"
description = "vLLM fork with TurboQuant KV cache compression. Install vllm-turboquant for Ampere/Ada/Hopper, vllm-turboquant-blackwell for Blackwell (sm_10.0/12.0/12.1)."
license = { text = "Apache-2.0" }
```

The build script does:

```sh
sed -i "s/VLLM_TURBOQUANT_NAME_PLACEHOLDER/${DIST_NAME}/g" pyproject.toml
```

before each `python -m build --wheel` invocation, then `git checkout pyproject.toml`
between flavours so the placeholder is restored.

**Do not** flip `name` in git; the placeholder is the canonical source state.

The `vllm` import package directory (`vllm/`) keeps its name in both flavours
so existing user code (`from vllm import LLM`) is identical. The `Requires-Dist`
metadata is identical between flavours; only the wheel name differs.

---

## 2. `vllm/__init__.py` — TurboQuant sentinels

Append the contents of `vllm/__init__.py.snippet` to the fork's
`vllm/__init__.py`. The `TURBOQUANT_BUILD_ARCH` and `TURBOQUANT_BUILD_ARCH_LIST`
fields are intentionally empty in source; the build script writes the real
values in place per flavour.

After `python -m build`, the resulting wheel's `vllm/__init__.py` will look like
(for a blackwell build):

```python
TURBOQUANT_ENABLED = True
TURBOQUANT_KV_DTYPES = ("turboquant25", "turboquant35")
TURBOQUANT_BUILD_ARCH = "blackwell"
TURBOQUANT_BUILD_ARCH_LIST = "10.0 12.0 12.1+PTX"
```

The script restores the placeholder after each build via `git checkout`.

---

## 3. `vllm/turboquant_arch_check.py` — runtime sentinel check

Drop the file `vllm/turboquant_arch_check.py` from this directory into the
fork's `vllm/` package. It exposes:

- `check_arch_compatibility() -> Optional[str]` — returns a remediation message
  on mismatch, `None` on match / no GPU / editable install.
- `assert_arch_compatibility() -> None` — raises `RuntimeError` on mismatch.

**Wire-in point.** Add a single call in `vllm/engine/llm_engine.py` (or the V1
equivalent `vllm/v1/engine/core.py`) just after CUDA initialization and before
the first worker is spawned:

```python
from vllm.turboquant_arch_check import assert_arch_compatibility

assert_arch_compatibility()
```

Place it as early as possible so a user who installed the wrong flavour fails
in milliseconds with a clear pip command, instead of waiting for kernel
compilation to crash deep in the worker.

---

## 4. README.md banner + GPU install table

Replace the fork's top-of-README banner with:

```markdown
# vllm-turboquant

This is a fork of [vLLM](https://github.com/vllm-project/vllm) with TurboQuant
KV cache compression (kv_cache_dtype `turboquant25` / `turboquant35`). Not
affiliated with the upstream vLLM project.

## Install (pick the wheel that matches your GPU)

| Your GPU                                       | Install command                            |
|------------------------------------------------|--------------------------------------------|
| RTX 30/40-series, A100, H100, GH200            | `pip install vllm-turboquant`              |
| RTX 50-series, B100/B200, GB10 (DGX Spark)     | `pip install vllm-turboquant-blackwell`    |

`pip install` resolves wheels from the GitHub Release attached to each tag —
add `--find-links https://github.com/tqcli/vllm-turboquant/releases/expanded_assets/v0.7.0-tq1`
if you want the exact 0.7.0-tq1 build.

The `vllm` Python import name is identical in both flavours; only the wheel
distribution differs. If you install the wrong flavour, the engine raises
`RuntimeError` on first GPU init with the correct pip command.

## License

Apache-2.0 (inherited from upstream vllm-project/vllm). The `NOTICE` file
preserves upstream attributions.
```

---

## 5. Pinning the golden commit

Workstream B step 5 pins the commit AFTER the Issue #22 four-patch page-size
fix (see `../issue_22_page_size_fix.md`). Verification gates:

1. **Gemma 4 E2B + BNB_INT4 + CPU offload (9.9 GB) + turboquant35** runs
   end-to-end on RTX A2000 (4 GB VRAM). Section C.2 of
   `tests/integration_reports/turboquant_kv_comparison_report.md` in the
   `tqcli/tqcli` repo defines the exact prompt and expected metadata.
2. **Qwen 3 4B + calibrated `turboquant_kv.json`** (the 0.6.1 path) runs green
   on the same box.

Both must pass on the candidate SHA before tagging. The script that drives the
two-test verification lives in the `tqcli/tqcli` repo; run it with the fork
checked out as a sibling and `pip install -e ../vllm-turboquant` active.

The maintainer identifies the SHA at release time. The release body should
quote it explicitly (`commit: <abbrev>`), as a provenance anchor — users mid-
download a year from now should still be able to reproduce the exact build.

---

## 6. Files map for application to fork

```
vllm/__init__.py.snippet       -> append to vllm/__init__.py (in fork)
vllm/turboquant_arch_check.py  -> vllm/turboquant_arch_check.py (in fork)
scripts/build_wheel_gcp.sh     -> scripts/build_wheel_gcp.sh (in fork, chmod +x)
docs/RELEASING.md              -> docs/RELEASING.md (in fork)
README.banner above            -> manual edit to fork's README.md
pyproject.toml change above    -> manual edit to fork's pyproject.toml
```

After the maintainer applies these and tags `v0.7.0-tq1`, the build script
takes over.
