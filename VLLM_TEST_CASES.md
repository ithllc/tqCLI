# tqCLI vLLM Backend Test Cases

**Status:** NOT EXECUTED — test cases documented for future execution
**Backend:** vLLM (>=0.6.0)
**Prerequisites:** Linux with NVIDIA GPU (8+ GB VRAM recommended), CUDA toolkit installed

> These test cases mirror the llama.cpp tests (Tests 1-4) but use the vLLM inference engine.
> vLLM requires: Linux + NVIDIA GPU with CUDA support. WSL2 with passthrough GPU is supported.

---

## Prerequisites

```bash
# Install tqCLI with vLLM backend
pip install -e ".[vllm]"

# Verify vLLM is available
python -c "import vllm; print(vllm.__version__)"

# Verify GPU is detected
nvidia-smi
```

### Hardware Requirements for vLLM Tests

| Model | Min VRAM | Recommended VRAM | Notes |
|-------|----------|------------------|-------|
| Gemma 4 E4B (4.5B) | 6 GB | 8 GB | SafeTensors format for vLLM |
| Gemma 4 E2B (2.3B) | 4 GB | 6 GB | SafeTensors format for vLLM |
| Qwen 3 4B | 5 GB | 8 GB | SafeTensors format for vLLM |
| Qwen 3 8B | 10 GB | 12 GB | Better quality at larger VRAM |

> **Important:** vLLM uses SafeTensors/AWQ/GPTQ formats, NOT GGUF. The model registry needs vLLM-specific profiles with different `hf_repo`, `filename`, `format`, and `engine` fields.

---

## Test 1 (vLLM): Gemma 4 Full Lifecycle

### 1.1 Install tqCLI with vLLM

```bash
pip install -e ".[vllm]"
tqcli --version
tqcli system info --json
```

**Expected:** Version 0.3.1, system info shows `recommended_engine: "vllm"` (if NVIDIA GPU with 8+ GB VRAM)

### 1.2 Download Gemma 4 Model (vLLM format)

```bash
# vLLM uses the original HuggingFace model (SafeTensors), not GGUF
# The model registry needs a vLLM-specific profile:
#   hf_repo: "google/gemma-4-e4b-it" (or AWQ quantized variant)
#   format: "safetensors" or "awq"
#   engine: "vllm"

tqcli model pull gemma-4-e4b-it-vllm
```

**Verify:**
- tqCLI selects hardware-appropriate model based on VRAM
- Model is downloaded to `~/.tqcli/models/`
- For 8GB VRAM: E4B or E2B
- For 16GB+ VRAM: 26B MoE or 31B

### 1.3 Verify Quantization

```bash
tqcli model list
```

**Expected:** Model shows quantization method appropriate for vLLM:
- AWQ (Activation-aware Weight Quantization) — TurboQuant compatible
- GPTQ — alternative quantization
- FP16/BF16 — full precision if VRAM allows

### 1.4 Two-Turn Chat Test

```bash
tqcli chat --engine vllm --model gemma-4-e4b-it-vllm
```

**Turn 1:** "What is the capital of France? Answer in one sentence."
- **Expected:** Correct factual answer mentioning Paris

**Turn 2:** "What is the population of that city? Just give the number."
- **Expected:** Approximate population of Paris (~2.1 million or ~12 million metro)

**Metrics to capture:**
- Tokens per second (vLLM should be significantly faster than llama.cpp)
- Time to first token
- Total generation time

### 1.5 Image Input Test

```bash
# In chat session:
/image /path/to/test_image.png What colors do you see in this image?
```

**Expected:** Model identifies colors in the test image (red square with blue border)

**Note:** vLLM multimodal support requires the `--image-input-type` flag and model must support vision (Gemma 4 E4B does)

### 1.6 Audio Input Test

```bash
# In chat session:
/audio /path/to/test_audio.wav Describe what you hear.
```

**Expected:** Model responds about audio (may indicate no audio processing in this configuration)

### 1.7 Generate and Verify Skill

```bash
tqcli skill create test-gemma4-vllm-skill -d "Test skill for Gemma 4 on vLLM"
tqcli skill run test-gemma4-vllm-skill
tqcli skill list
```

**Expected:** Skill created, script executes successfully

### 1.8 Remove Model and Clean Uninstall

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go model remove gemma-4-e4b-it-vllm
pip show tqcli  # Should show installed
pip uninstall tqcli -y
pip show tqcli  # Should fail (not found)
```

---

## Test 2 (vLLM): Qwen 3 Full Lifecycle

### 2.1 Install and Download

```bash
pip install -e ".[vllm]"
# Qwen 3 4B for vLLM:
#   hf_repo: "Qwen/Qwen3-4B" (SafeTensors) or "Qwen/Qwen3-4B-AWQ" (quantized)
tqcli model pull qwen3-4b-vllm
```

### 2.2 Verify Quantization

```bash
tqcli model list
```

**Expected:** AWQ or GPTQ quantization shown for vLLM model

### 2.3 Two-Turn Chat Test

```bash
tqcli chat --engine vllm --model qwen3-4b-vllm
```

**Turn 1:** "What is 2 + 2? Answer with just the number."
- **Expected:** "4" (Qwen 3 may include `<think>` blocks)

**Turn 2:** "Now multiply that result by 10. Answer with just the number."
- **Expected:** "40"

**Metrics to capture:**
- Tokens/second with vLLM (expected 20-50+ tok/s vs 6-9 tok/s with llama.cpp)
- Time to first token (vLLM has higher startup but faster sustained generation)

### 2.4 Image Input Test

**Expected:** FAIL — Qwen 3 is text-only (`multimodal=False`). Should fail gracefully.

### 2.5 Audio Input Test

**Expected:** FAIL — Qwen 3 is text-only. Should fail gracefully.

### 2.6 Generate Skill, Remove Model, Uninstall

Same as Test 1 steps 1.7-1.8 but with `test-qwen3-vllm-skill`.

---

## Test 3 (vLLM): Gemma 4 Multi-Process + Yolo Mode + CRM Build

### 3.1 Setup

```bash
pip install -e ".[vllm]"
tqcli model pull gemma-4-e4b-it-vllm
```

### 3.2 Assess Multi-Process Feasibility

```python
from tqcli.core.multiprocess import assess_multiprocess
from tqcli.core.system_info import detect_system

sys_info = detect_system()
plan = assess_multiprocess(
    sys_info=sys_info,
    model_path="/path/to/model",
    model_size_mb=4000,
    requested_workers=3,
    preferred_engine="vllm",
    unrestricted=True,
)
print(f"Feasible: {plan.feasible}, Engine: {plan.engine}, Max workers: {plan.max_workers}")
```

**Expected:** vLLM with PagedAttention enables more concurrent workers than llama.cpp

### 3.3 Start vLLM Server (Yolo Mode)

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go serve start -m gemma-4-e4b-it-vllm -e vllm
```

**Expected:** vLLM OpenAI-compatible server starts on port 8741
- Uses continuous batching (not sequential like llama.cpp)
- PagedAttention for efficient KV cache sharing

### 3.4 Check Server Status

```bash
tqcli serve status
```

**Expected:** Shows running status with PID, engine=vllm, health=OK

### 3.5 Generate CRM Skills and Workspace

```bash
tqcli skill create crm-frontend -d "Generate HTML/CSS/JS frontend for CRM"
tqcli skill create crm-backend -d "Generate Flask backend API for CRM"
tqcli skill create crm-database -d "Generate SQLite schema for CRM"

# Create workspace at root of LLM_MODELS_PYTHON_CODE_SRC
mkdir -p /llm_models_python_code_src/crm_workspace_vllm/{frontend,backend,database}
```

### 3.6 Verify CRM Workspace

```bash
ls -la /llm_models_python_code_src/crm_workspace_vllm/frontend/index.html
ls -la /llm_models_python_code_src/crm_workspace_vllm/backend/app.py
ls -la /llm_models_python_code_src/crm_workspace_vllm/database/schema.sql
```

### 3.7 Cleanup

```bash
rm -rf /llm_models_python_code_src/crm_workspace_vllm
tqcli serve stop
tqcli --stop-trying-to-control-everything-and-just-let-go model remove gemma-4-e4b-it-vllm
pip uninstall tqcli -y
```

---

## Test 4 (vLLM): Qwen 3 Multi-Process + Yolo Mode + CRM Build

### 4.1 Setup

```bash
pip install -e ".[vllm]"
tqcli model pull qwen3-4b-vllm
```

### 4.2 Assess and Start vLLM Server

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go serve start -m qwen3-4b-vllm -e vllm
tqcli serve status
```

### 4.3 Spawn Workers

```bash
tqcli --stop-trying-to-control-everything-and-just-let-go workers spawn 3
tqcli workers list
```

**Expected:** 3 workers connected to vLLM server, handling requests concurrently (not sequentially like llama.cpp)

### 4.4 Generate CRM Skills and Workspace

Same as Test 3 steps 3.5-3.6 but using Qwen 3.

### 4.5 Performance Comparison Point

During CRM generation, capture:
- Server throughput (requests/second)
- Per-worker latency
- Total VRAM usage
- Compare continuous batching (vLLM) vs sequential queue (llama.cpp)

### 4.6 Cleanup

```bash
rm -rf /llm_models_python_code_src/crm_workspace_vllm
tqcli workers stop
tqcli serve stop
tqcli --stop-trying-to-control-everything-and-just-let-go model remove qwen3-4b-vllm
pip uninstall tqcli -y
```

---

## Pre-Test Checklist (vLLM)

Before executing these tests, verify:

- [ ] NVIDIA GPU with 8+ GB VRAM available
- [ ] CUDA toolkit installed (`nvcc --version`)
- [ ] `pip install vllm>=0.6.0` succeeds
- [ ] `python -c "import vllm"` works
- [ ] Model registry has vLLM-specific profiles (SafeTensors/AWQ format)
- [ ] vLLM OpenAI API server module works (`python -m vllm.entrypoints.openai.api_server --help`)

## Expected Performance Comparison (vLLM vs llama.cpp)

| Metric | llama.cpp (measured) | vLLM (expected) |
|--------|---------------------|-----------------|
| Gemma 4 E4B tok/s | 2-4 | 15-30 |
| Qwen 3 4B tok/s | 6-9 | 20-50 |
| Multi-process mode | Sequential queue | Continuous batching |
| KV Cache | Per-request | PagedAttention (shared) |
| Max concurrent workers | 4 (CPU-limited) | 6-10 (VRAM-limited) |
| Time to first token | 2-10s | 0.5-2s |

## Model Registry Updates Needed for vLLM

The current model registry only has GGUF profiles. For vLLM tests, add profiles like:

```python
ModelProfile(
    id="gemma-4-e4b-it-vllm",
    family="gemma4",
    display_name="Gemma 4 E4B Edge Instruct (vLLM)",
    hf_repo="google/gemma-4-e4b-it",  # SafeTensors format
    filename="",  # vLLM loads entire repo
    parameter_count="4.5B",
    quantization="FP16",  # or "AWQ" if using quantized
    format="safetensors",
    context_length=128000,
    engine="vllm",
    min_ram_mb=4000,
    min_vram_mb=6000,  # vLLM needs more VRAM than llama.cpp
    supports_thinking=True,
    multimodal=True,
),
ModelProfile(
    id="qwen3-4b-vllm",
    family="qwen3",
    display_name="Qwen3 4B (vLLM)",
    hf_repo="Qwen/Qwen3-4B",  # SafeTensors format
    filename="",
    parameter_count="4B",
    quantization="FP16",
    format="safetensors",
    context_length=32768,
    engine="vllm",
    min_ram_mb=3500,
    min_vram_mb=5000,
    supports_thinking=True,
),
```
