# Product Requirements Document (PRD): vLLM TurboQuant Fork Gemma 4 Support

## 1. Introduction
**vLLM TurboQuant Fork Gemma 4 Support** is a dependency upgrade and verification effort to enable Gemma 4 E2B inference on the TurboQuant vLLM fork. The current Transformers library (4.57.6) does not recognize the `gemma4` architecture, blocking model load even though the tqCLI CPU offloading pipeline logic is fully implemented and verified. This PRD covers the upgrade from Transformers 4.x to 5.x, optional vLLM fork rebuild, and E2E verification of Gemma 4 E2B on constrained VRAM hardware.

## 2. Target Audience
- **tqCLI developers:** Need Gemma 4 E2B running on vLLM with TurboQuant KV cache compression
- **Low-VRAM users (4 GB GPUs):** Benefit from CPU offloading to run multimodal models that exceed GPU memory
- **TurboQuant researchers:** Need both llama.cpp and vLLM engines verified for Gemma 4 to validate KV compression across backends

## 3. Scope & Constraints
- **Hardware matrix:** NVIDIA RTX A2000 Laptop GPU (4 GB VRAM, SM86) under WSL2; target 6 GB+ GPUs for non-offload path
- **Deployment:** Local-first inference on developer workstations
- **CUDA:** 12.8 (required by TurboQuant vLLM fork)
- **System RAM:** 32 GB (24 GB available for CPU offload spill)

**In Scope:**
- Transformers upgrade from 4.57.6 to 5.5.x
- Qwen3 AWQ regression verification on vLLM after upgrade
- vLLM TurboQuant fork rebuild from source if regression detected
- Gemma 4 E2B E2E load + inference with BNB_INT4 + cpu_offload + turboquant35 KV
- Full integration test suite pass (6 tests across both engines)

**Out of Scope:**
- AWQ/GPTQ pre-quantized Gemma 4 variants (none exist on HuggingFace as of 2026-04-15)
- llama.cpp engine changes (GGUF path has no Transformers dependency)
- Gemma 4 31B or 26B-A4B MoE variants (too large for 4 GB VRAM even with offload)
- Multimodal (image/video/audio) inference — text-only verification for this PRD
- Changes to the CPU offloading pipeline logic (already implemented and verified)

## 4. Key Features

### 4.1. Transformers Major Version Upgrade (4.x → 5.x)
- Upgrade `transformers` package to >=5.5.0 to gain `gemma4` architecture recognition
- **Risk:** Major version jump may introduce breaking API changes affecting vLLM fork or bitsandbytes integration
- **User Interaction:** None — transparent dependency upgrade

### 4.2. Qwen3 AWQ Regression Gate
- Before proceeding, verify Qwen3 AWQ + turboquant35 KV still works on vLLM after the Transformers upgrade
- **Gate:** If regression detected, halt and rebuild vLLM fork before continuing
- **User Interaction:** Automated test (Test 5 on vLLM engine)

### 4.3. vLLM TurboQuant Fork Rebuild (Conditional)
- If the Transformers 5.x upgrade breaks vLLM fork compatibility, rebuild from source
- Clone `ithllc/vllm-turboquant`, build with `CUDA_HOME=/usr/local/cuda-12.8`
- Follow the #15 rebuild pattern (pyproject.toml fixes, cmake version handling)
- **User Interaction:** None if not needed; ~30 min build time if triggered

### 4.4. Gemma 4 E2B vLLM E2E Verification
- Load Gemma 4 E2B (google/gemma-4-E2B) on vLLM with:
  - Weight quantization: bitsandbytes INT4
  - CPU offload: 2.1 GB to system RAM
  - KV cache: turboquant35
  - Max model length: 2048
- Verify multi-turn chat inference produces coherent output
- **AI/Models:** google/gemma-4-E2B (2.6B effective parameters, multimodal)

### 4.5. Integration Test Suite Validation
- Run all 6 integration tests (Tests 5-7 on both llama.cpp and vLLM engines)
- Confirm no regressions across thinking mode, tool calling, and standard chat

## 5. User Stories
1. *As a tqCLI user with a 4 GB GPU, I want Gemma 4 E2B to load on vLLM so I can use a multimodal model locally without cloud API costs.*
2. *As a developer, I want Transformers upgraded without breaking existing Qwen3 AWQ inference so both model families work on the same vLLM fork.*
3. *As a TurboQuant researcher, I want verified E2E results for Gemma 4 on vLLM with turboquant35 KV so I can compare compression ratios across engines.*
4. *As a CI maintainer, I want all 6 integration tests passing after the upgrade so I know nothing regressed.*

## 6. Technical Requirements
- **Runtime:** Python 3.10+, CUDA 12.8, NVIDIA driver 570+
- **Packages:** transformers>=5.5.0, vllm 0.1.dev5 (TurboQuant fork), bitsandbytes>=0.49.2
- **VRAM budget:** 4,096 MB total — 810 MB CUDA overhead (WSL2) — 700 MB vLLM runtime — 50 MB min KV = 2,536 MB for model weights
- **CPU offload:** Gemma 4 E2B INT4 = 4,145 MB → 1,609 MB excess → 2.1 GB offloaded to system RAM
- **KV cache:** turboquant35 dtype (4.6x compression, +1% PPL)
- **Privacy:** All inference local, no data leaves the machine

## 7. Success Metrics
- **Model load:** Gemma 4 E2B loads successfully on vLLM with BNB_INT4 + cpu_offload_gb=2.1 + turboquant35 KV
- **Inference quality:** Multi-turn chat produces coherent, on-topic responses
- **Regression:** Qwen3 AWQ + turboquant35 Tests 5-7 on vLLM all PASS after upgrade
- **Integration suite:** 6/6 tests PASS (Tests 5-7 on both engines)
- **Performance:** Benchmark captured (tok/s with Gemma 4 E2B on vLLM vs llama.cpp baseline)

---

**Next steps:** Run `/technical-planner` to generate the phased implementation plan from this PRD.
