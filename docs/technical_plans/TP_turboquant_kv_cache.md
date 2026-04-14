# Technical Implementation Plan: TurboQuant KV Cache Compression

**Date:** 2026-04-14
**PRD:** [docs/prd/PRD_turboquant_kv_cache.md](../prd/PRD_turboquant_kv_cache.md)
**GitHub Issue:** [ithllc/tqCLI#13](https://github.com/ithllc/tqCLI/issues/13)
**Status:** Ready for implementation

---

## Overview

Integrate TurboQuant KV cache compression into tqCLI by building against community forks of llama.cpp and vLLM that implement the ICLR 2026 TurboQuant algorithm. This is a runtime-only change — existing model files (GGUF, safetensors) are used as-is; only the KV cache storage is compressed.

---

## Phase 1: llama.cpp TurboQuant Backend

### Objective
Build llama-cpp-python against TheTom/turboquant_plus fork and expose KV cache type parameters.

### Steps

**1.1** Clone and build the TurboQuant fork:
```bash
git clone https://github.com/TheTom/turboquant_plus.git
cd turboquant_plus
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

**1.2** Build llama-cpp-python against the fork:
```bash
# Option A: Point CMAKE_ARGS to the fork
CMAKE_ARGS="-DGGML_CUDA=ON" FORCE_CMAKE=1 \
  pip install llama-cpp-python --no-binary :all: \
  --config-settings="cmake.args=-DLLAMA_CPP_DIR=/path/to/turboquant_plus"

# Option B: Build from the fork's Python bindings directly
cd turboquant_plus
pip install -e ".[server]"
```

**1.3** Verify turbo types are available:
```python
from llama_cpp import Llama
# Check if cache_type_k parameter is accepted
llm = Llama(model_path="model.gguf", cache_type_k="turbo3", cache_type_v="turbo3")
```

**1.4** Update `LlamaBackend.__init__()`:
```python
def __init__(self, ..., cache_type_k: str = "f16", cache_type_v: str = "f16"):
    self._cache_type_k = cache_type_k
    self._cache_type_v = cache_type_v
```

**1.5** Update `LlamaBackend.load_model()` to pass cache types:
```python
params["cache_type_k"] = self._cache_type_k
params["cache_type_v"] = self._cache_type_v
```

### Files Modified
- `tqcli/core/llama_backend.py` — Add cache_type params
- `pyproject.toml` — Update llama-cpp-python dependency (may need fork URL)

### Risk
- llama-cpp-python may not expose `cache_type_k`/`cache_type_v` — need to verify
- If not exposed, must shell out to the fork's compiled binary

---

## Phase 2: vLLM TurboQuant Backend

### Objective
Install mitkox/vllm-turboquant and wire turboquant KV cache dtype into VllmBackend.

### Steps

**2.1** Install vllm-turboquant from source:
```bash
git clone https://github.com/mitkox/vllm-turboquant.git
cd vllm-turboquant
export CUDA_HOME=/usr/local/cuda-12.8
export VLLM_TARGET_DEVICE=cuda
export VLLM_USE_PRECOMPILED=0
pip install -e .
```

**2.2** Verify turboquant dtype works:
```python
from vllm import LLM
llm = LLM(model="model_path",
           kv_cache_dtype="turboquant35",
           attention_backend="TRITON_ATTN")
```

**2.3** Update `VllmTuningProfile`:
```python
kv_cache_dtype: str = "auto"  # Can now be "turboquant35", "turboquant25"
enable_turboquant: bool = False
attention_backend: str | None = None  # "TRITON_ATTN" when turboquant
```

**2.4** Update `build_vllm_config()` to select turboquant KV:
```python
if kv_budget_mb < 200 and turboquant_available:
    profile.kv_cache_dtype = "turboquant35"
    profile.enable_turboquant = True
    profile.attention_backend = "TRITON_ATTN"
```

**2.5** Update `VllmBackend.load_model()` to pass turboquant flags.

### Files Modified
- `tqcli/core/vllm_backend.py`
- `tqcli/core/vllm_config.py`

### Risk
- mitkox fork tested on A6000 (SM86) — same SM as our A2000 but 48 GB VRAM
- May need memory tuning for 4 GB
- Source build takes 20-60 minutes

---

## Phase 3: KV Quantizer Module and CLI

### Objective
Create KV cache quant selection logic and `--kv-quant` CLI flag.

### Steps

**3.1** Create `tqcli/core/kv_quantizer.py`:
```python
class KVQuantLevel(Enum):
    NONE = "none"       # q8_0 / f16 default
    TURBO4 = "turbo4"   # 4.25 bpv, 3.8x compression
    TURBO3 = "turbo3"   # 3.5 bpv, 4.6x compression
    TURBO2 = "turbo2"   # 2.5 bpv, 6.4x compression

def select_kv_quant(available_kv_mb: int, engine: str) -> KVQuantLevel:
    """Auto-select KV quant based on available memory."""
    if available_kv_mb >= 200:
        return KVQuantLevel.NONE
    elif available_kv_mb >= 50:
        return KVQuantLevel.TURBO4
    elif available_kv_mb >= 20:
        return KVQuantLevel.TURBO3
    else:
        return KVQuantLevel.TURBO2

def get_llama_kv_params(level: KVQuantLevel) -> dict:
    mapping = {
        KVQuantLevel.NONE: {"cache_type_k": "f16", "cache_type_v": "f16"},
        KVQuantLevel.TURBO4: {"cache_type_k": "turbo4", "cache_type_v": "turbo4"},
        KVQuantLevel.TURBO3: {"cache_type_k": "turbo3", "cache_type_v": "turbo3"},
        KVQuantLevel.TURBO2: {"cache_type_k": "turbo2", "cache_type_v": "turbo2"},
    }
    return mapping[level]

def get_vllm_kv_params(level: KVQuantLevel) -> dict:
    mapping = {
        KVQuantLevel.NONE: {},
        KVQuantLevel.TURBO4: {"kv_cache_dtype": "turboquant35", "enable_turboquant": True},
        KVQuantLevel.TURBO3: {"kv_cache_dtype": "turboquant35", "enable_turboquant": True},
        KVQuantLevel.TURBO2: {"kv_cache_dtype": "turboquant25", "enable_turboquant": True},
    }
    return mapping[level]
```

**3.2** Add `--kv-quant` flag to `tqcli chat`:
```python
@click.option("--kv-quant", type=click.Choice(["auto", "none", "turbo4", "turbo3", "turbo2"]),
              default="auto", help="KV cache compression level (TurboQuant)")
```

**3.3** Wire into chat command:
- For llama.cpp: pass `cache_type_k`/`cache_type_v` to LlamaBackend
- For vLLM: pass `kv_cache_dtype`/`enable_turboquant` to VllmBackend

### Files Created
- `tqcli/core/kv_quantizer.py`

### Files Modified
- `tqcli/cli.py` — Add `--kv-quant` flag

---

## Phase 4: Integration Tests

### Objective
Test TurboQuant KV cache compression on both backends with BF16 source models.

### Test File
`tests/test_integration_turboquant_kv.py`

### Test Matrix

| Test | Model | Engine | Weight Quant | KV Quant | Expected |
|------|-------|--------|-------------|----------|----------|
| 1 | Gemma 4 E4B Q4_K_M | llama.cpp | Q4_K_M (GGUF) | turbo3 | PASS (4.6x KV compression) |
| 2 | Qwen 3 4B Q4_K_M | llama.cpp | Q4_K_M (GGUF) | turbo3 | PASS (4.6x KV compression) |
| 3 | Qwen 3 4B AWQ | vLLM | AWQ INT4 | turboquant35 | PASS (if fork installed) |
| 4 | Baseline comparison | Both | Same as above | q8_0/none | PASS (reference metrics) |

### Comparison Report
Output: `tests/integration_reports/turboquant_kv_comparison_report.md`

Metrics per test:
- Context capacity (tokens achievable)
- Load time
- Tokens per second (prompt eval + generation)
- Perplexity estimate (response coherence check)
- VRAM usage

---

## Phase 5: Documentation

### Files to Update
- `CLAUDE.md` — Add kv_quantizer.py, --kv-quant flag
- `README.md` — Add TurboQuant KV cache section
- `tests/integration_reports/vllm_test_cases.md` — Add turboquant test cases
- `tests/integration_reports/llama_cpp_test_cases.md` — Add turboquant test cases
- `docs/prd/PRD_turboquant_kv_cache.md` — Update with results

---

## File Manifest

| File | Action | Phase |
|------|--------|-------|
| `tqcli/core/llama_backend.py` | Modify — add cache_type params | 1 |
| `pyproject.toml` | Modify — update llama-cpp-python source | 1 |
| `tqcli/core/vllm_backend.py` | Modify — add turboquant kv dtype | 2 |
| `tqcli/core/vllm_config.py` | Modify — add turboquant to tuner | 2 |
| `tqcli/core/kv_quantizer.py` | Create — KV quant selection engine | 3 |
| `tqcli/cli.py` | Modify — add --kv-quant flag | 3 |
| `tests/test_integration_turboquant_kv.py` | Create — 4-test KV comparison | 4 |
| `CLAUDE.md`, `README.md`, test case docs | Modify — document changes | 5 |
