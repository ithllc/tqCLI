# Resume Prompt: TurboQuant KV Cache Integration

**Last Session:** 2026-04-14 15:07 EDT
**GitHub Issue:** ithllc/tqCLI#13
**Last Commit:** cf5a646 (resume prompt + WIP kv_quantizer)

---

## Copy-paste this prompt to resume:

```
Resume the TurboQuant KV cache integration for GitHub issue ithllc/tqCLI#13. The session was paused at 3:07 PM on April 14.

Check your memory file at /root/.claude/projects/-llm-models-python-code-src-tqCLI/memory/project_turboquant_kv_progress.md for full status.

IMPORTANT CHANGE FROM ORIGINAL PLAN: We need to FORK both TurboQuant inference engine projects under the ithllc GitHub org and rewrite them for our CUDA 12.8 / PyTorch 2.9 / vLLM 0.19.0 / transformers 5.5.4 stack. The upstream forks have CUDA version incompatibilities.

What's already done:
- tqcli/core/kv_quantizer.py created (KVQuantLevel enum, select_kv_quant, get_llama_kv_params, get_vllm_kv_params)
- --kv-quant CLI flag wired into tqcli chat (turbo4/turbo3/turbo2 choices)
- LlamaBackend has cache_type_k/cache_type_v params
- TheTom/llama-cpp-turboquant cloned at /tmp/llama-cpp-turboquant/ (CPU build works, turbo types verified)
- Qwen3-4B-Q4_K_M.gguf model at /root/.tqcli/models/

Steps to execute (in order):

1. **Install CUDA 12.8 toolkit** (full, with nvcc — not just runtime):
   ```bash
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
   dpkg -i cuda-keyring_1.1-1_all.deb
   apt-get update && apt-get install -y cuda-toolkit-12-8
   export PATH=/usr/local/cuda-12.8/bin:$PATH
   nvcc --version  # Should show 12.8
   ```

2. **Fork and fix llama-cpp-turboquant**:
   - Fork TheTom/llama-cpp-turboquant → ithllc/llama-cpp-turboquant on GitHub
   - Clone: `gh repo fork TheTom/llama-cpp-turboquant --org ithllc --clone`
   - Fix CUDA compilation issues for CUDA 12.8 + SM86 (the C++ template errors from nvcc 11.5)
   - Build: `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="86" -DCMAKE_BUILD_TYPE=Release`
   - Test: `./build/bin/llama-cli -m model.gguf --cache-type-k turbo3 --cache-type-v turbo3 -ngl 99`
   - Push fixes to ithllc/llama-cpp-turboquant

3. **Fork and fix vllm-turboquant**:
   - Fork mitkox/vllm-turboquant → ithllc/vllm-turboquant
   - Clone: `gh repo fork mitkox/vllm-turboquant --org ithllc --clone`
   - Fix for vLLM 0.19.0 + transformers 5.5.4 + CUDA 12.8 compatibility
   - Build from source: `CUDA_HOME=/usr/local/cuda-12.8 pip install -e .`
   - Test: verify turboquant35 KV dtype works
   - Push fixes to ithllc/vllm-turboquant

4. **Implement unified quantization pipeline** — CRITICAL DESIGN CHANGE:
   The quantization pipeline in tqcli must detect model precision and apply
   the correct compression stages automatically. Update tqcli/core/quantizer.py
   and tqcli/core/kv_quantizer.py (or merge them) to implement this logic:

   ```
   Model loaded → detect format
       │
       ├── Already weight-quantized (GGUF Q4_K_M, AWQ INT4, GPTQ)?
       │       → Skip weight quantization
       │       → Apply KV cache compression ONLY (turbo3/turbo4)
       │
       └── Full precision (BF16 / FP16 / FP32 safetensors)?
               → Apply weight quantization FIRST
               │   • vLLM: bitsandbytes INT4 on-the-fly
               │   • llama.cpp: would need GGUF conversion (Phase 2)
               → THEN apply KV cache compression (turbo3/turbo4)
   ```

   The detection logic should check:
   - `model.quantization` field in the registry ("BF16", "FP16" = full precision)
   - `model.format` field ("safetensors" = likely full precision, "gguf" = already quantized, "awq" = already quantized)
   - File inspection as fallback (check config.json for quantization_config)

   This unified pipeline means a user running `tqcli chat --model gemma-4-e4b-it-vllm`
   gets BOTH weight quantization AND KV cache compression automatically, while a user
   running `tqcli chat --model qwen3-4b-Q4_K_M` gets KV cache compression only
   (weights already compressed).

   The `--kv-quant` flag should still allow override (e.g., `--kv-quant none` to
   disable KV compression, or `--kv-quant turbo2` for maximum compression).

   Update the integration tests to verify this pipeline:
   - BF16 model → should get weight quant + KV cache compression
   - Pre-quantized GGUF → should get KV cache compression only
   - Pre-quantized AWQ → should get KV cache compression only

5. **Wire VllmBackend for turboquant KV** (not yet done):
   - Update tqcli/core/vllm_backend.py to pass turboquant kv_cache_dtype
   - Update tqcli/core/vllm_config.py to include KV quant in tuning profile
   - Ensure the unified pipeline from step 4 drives both weight + KV decisions

6. **Write test_integration_turboquant_kv.py** based on test cases at:
   tests/integration_reports/turboquant_kv_test_cases.md
   
   IMPORTANT: Tests must verify the unified pipeline logic:
   - Test 1: llama.cpp Gemma 4 E4B Q4_K_M + turbo3 KV (weight already quantized → KV only)
   - Test 2: llama.cpp Qwen 3 4B Q4_K_M + turbo3 KV (weight already quantized → KV only)
   - Test 3: vLLM Qwen 3 4B BF16 + bnb INT4 + turboquant35 KV (full precision → BOTH stages)
   - Test 4: vLLM Qwen 3 4B AWQ + turboquant35 KV (weight already quantized → KV only)
   - Test 5: Baseline comparison (no weight quant changes, no KV compression)
   
   Each test must LOG which pipeline stages were applied (weight quant, KV cache, or both)
   so the report clearly shows the decision logic working.

7. **Run all 5 integration tests** and generate comparison report at:
   tests/integration_reports/turboquant_kv_comparison_report.md

8. **Post-integration**: Review issues, update ALL docs (CLAUDE.md, README.md, test case docs, llama_cpp_test_cases.md, vllm_test_cases.md), close issues, bump to v0.5.0, commit, push.

If you encounter issues at any step, file them via gh issue create on ithllc/tqCLI, fix them, and comment the solution on the issue. Use your issue-manager and project-manager skill sets as needed.

Our hardware: NVIDIA RTX A2000 Laptop (4 GB VRAM, SM86 Ampere), WSL2 Ubuntu 22.04.
Our stack: Python 3.11, PyTorch 2.9.1+cu128, vLLM 0.19.0, transformers 5.5.4, CUDA driver 581.95 (supports CUDA 13.0).
```
