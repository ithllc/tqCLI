# Agent Modes — FUNCTIONAL Integration Test Report

**Generated:** 2026-04-18T18:30:49
**System:** NVIDIA RTX A2000 Laptop GPU (4096 MB VRAM)
**CUDA:** driver 13.0, toolkit 12.8
**OS:** Linux (Ubuntu 22.04.4 LTS) (WSL2)

Exercises the full parse→execute→observation→live-inference loop of `tqcli/core/agent_orchestrator.py` with concrete assertions — spy fidelity, history integrity, filesystem side-effects, and secret-word ingestion into the live KV cache. Zero-shot tests (T0*) are DATA POINTS capturing real-model tag-emission compliance; they do not gate the suite.

---

## Summary

| Test | Engine | Model | Mode | KV Quant | Kind | Result |
|------|--------|-------|------|----------|------|--------|
| T0a | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | data-point | **PASS** (1/1) |
| T1_lg | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T2 | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (4/4) |
| T3 | llama.cpp | gemma-4-e2b-it-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (3/3) |
| T4_lg | llama.cpp | gemma-4-e2b-it-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T5 | llama.cpp | gemma-4-e2b-it-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (3/3) |
| T0b | llama.cpp | qwen3-4b-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | data-point | **PASS** (1/1) |
| T1_lq | llama.cpp | qwen3-4b-Q4_K_M | ai_tinkering | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T4_lq | llama.cpp | qwen3-4b-Q4_K_M | unrestricted | turbo3 (cache_type_k=turbo3, cache_type_v=turbo3) | functional | **PASS** (6/6) |
| T0c | vllm | gemma-4-e2b-it-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | data-point | **PASS** (1/1) |
| T1_vg | vllm | gemma-4-e2b-it-vllm | ai_tinkering | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | functional | **PASS** (6/6) |
| T4_vg | vllm | gemma-4-e2b-it-vllm | unrestricted | turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9) | functional | **PASS** (6/6) |
| T0d | vllm | qwen3-4b-vllm | unrestricted | auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B] | data-point | **PASS** (1/1) |
| T1_vq | vllm | qwen3-4b-vllm | ai_tinkering | auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B] | functional | **PASS** (5/5) |
| T4_vq | vllm | qwen3-4b-vllm | unrestricted | auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B] | functional | **PASS** (5/5) |

---

## T0a: T0 zero-shot (ai_tinkering, llama.cpp gemma-4-e2b-it-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-18T18:09:33 → 2026-04-18T18:09:41
- **Duration:** 8.21s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 8.21s | emitted_staged_tool_call=True / chars=117 / head=<staged_tool_call>{ "name":"tq-file-read", "arguments":{ "path":"/tmp/tqcli_agent_fixture.txt" } }</staged_tool_call> |

## T1_lg: T1 approve actionable (llama.cpp gemma-4-e2b-it-Q4_K_M, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:09:41 → 2026-04-18T18:09:50
- **Duration:** 9.62s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-5d1b1d9d09e9
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 9.62s | secret=ALPHACHARLIE-5d1b1d9d09e9 / found_in_final=True / final_len=273 / final_head=Thank you for providing the observation.  The secret word is: **ALPHACHARLIE-5d1b1d9d09e9**  How would you like me to proceed with this  |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T2: T2 deny actionable (llama.cpp Gemma 4, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:09:50 → 2026-04-18T18:09:50
- **Duration:** 0.00s
- **Result:** PASS (4/4)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_called_once | PASS | 0.00s | confirm_calls=1 |
| terminal_exec_never_called | PASS | 0.00s | calls=[] |
| observation_is_denial | PASS | 0.00s | obs_head=Observation:
User denied execution. Request alternatives. |
| loop_terminated_after_denial | PASS | 0.00s | scripted=1 real=0 |

## T3: T3 edit actionable (llama.cpp Gemma 4, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:09:50 → 2026-04-18T18:09:56
- **Duration:** 5.39s
- **Result:** PASS (3/3)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| exec_called_once_with_edited_args | PASS | 0.00s | calls=[{'command': 'touch /tmp/tq_t3_mark'}] |
| edited_mark_exists | PASS | 0.00s | path=/tmp/tq_t3_mark exists=True |
| original_mark_never_created | PASS | 5.39s | path=/tmp/tq_t3_ORIGINAL exists=False |

## T4_lg: T4 unrestricted ReAct (llama.cpp gemma-4-e2b-it-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:09:56 → 2026-04-18T18:10:01
- **Duration:** 4.95s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-e472127f2261
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 4.95s | secret=ALPHACHARLIE-e472127f2261 / found_in_final=True / final_head=Okay, I have read the observation.  The secret word is: **ALPHACHARLIE-e472127f2261** |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T5: T5 dead tool name (llama.cpp Gemma 4, unrestricted)

- **Engine:** llama.cpp
- **Model:** gemma-4-e2b-it-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:10:01 → 2026-04-18T18:10:11
- **Duration:** 10.62s
- **Result:** PASS (3/3)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| no_real_tool_invoked | PASS | 0.00s | total_spy_calls=0 |
| observation_is_error | PASS | 0.00s | obs_head=Observation:
ERROR: unknown tool 'nonexistent_tool' |
| orchestrator_did_not_crash | PASS | 10.62s | ran to completion |

## T0b: T0 zero-shot (ai_tinkering, llama.cpp qwen3-4b-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-18T18:10:16 → 2026-04-18T18:10:45
- **Duration:** 28.93s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 28.93s | emitted_staged_tool_call=False / chars=1121 / head=<think> Okay, the user wants me to use the tq-file-read tool to read the file at /tmp/tqcli_agent_fixture.txt. They specified that the response should be exactl |

## T1_lq: T1 approve actionable (llama.cpp qwen3-4b-Q4_K_M, ai_tinkering)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** ai_tinkering
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:10:45 → 2026-04-18T18:11:09
- **Duration:** 23.92s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-c003f369de3c
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 23.92s | secret=ALPHACHARLIE-c003f369de3c / found_in_final=True / final_len=771 / final_head=<think> Okay, the user asked me to read the fixture, and I used the tool to read the file. The observation shows that the secret word is |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T4_lq: T4 unrestricted ReAct (llama.cpp qwen3-4b-Q4_K_M)

- **Engine:** llama.cpp
- **Model:** qwen3-4b-Q4_K_M
- **Mode:** unrestricted
- **KV Quant:** turbo3 (cache_type_k=turbo3, cache_type_v=turbo3)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:11:09 → 2026-04-18T18:11:35
- **Duration:** 26.50s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-5bf0d6f09a46
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 26.50s | secret=ALPHACHARLIE-5bf0d6f09a46 / found_in_final=True / final_head=<think> Okay, the user asked me to read a file, and I used the tq-file-read tool to read the file at /tmp/tqcli_agent_fixture.txt. The observation from  |
| turboquant_kv_active | PASS | 0.00s | kv_params={'cache_type_k': 'turbo3', 'cache_type_v': 'turbo3'} |

## T0c: T0 zero-shot (unrestricted, vllm gemma-4-e2b-it-vllm)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-18T18:14:05 → 2026-04-18T18:14:51
- **Duration:** 45.90s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 45.90s | emitted_tool_call=True / chars=102 / head=    <tool_call>{"name":"tq-file-read","arguments":{"path":"/tmp/tqcli_agent_fixture.txt"}}</tool_call> |

## T1_vg: T1 approve actionable (vllm gemma-4-e2b-it-vllm, ai_tinkering)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** ai_tinkering
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:14:51 → 2026-04-18T18:15:12
- **Duration:** 20.44s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-54fe6b605dd3
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 20.44s | secret=ALPHACHARLIE-54fe6b605dd3 / found_in_final=True / final_len=27 / final_head="ALPHACHARLIE-54fe6b605dd3" |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |

## T4_vg: T4 unrestricted ReAct (vllm gemma-4-e2b-it-vllm)

- **Engine:** vllm
- **Model:** gemma-4-e2b-it-vllm
- **Mode:** unrestricted
- **KV Quant:** turboquant35 (quant=bitsandbytes, cpu_offload_gb=9.9)
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:15:12 → 2026-04-18T18:18:35
- **Duration:** 203.10s
- **Result:** PASS (6/6)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-d42047fe16a5
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 203.10s | secret=ALPHACHARLIE-d42047fe16a5 / found_in_final=True / final_head=<thought The user has provided an observation containing a secret word. The goal is to process this information. Since the previous turn was a tool call |
| turboquant_kv_active | PASS | 0.00s | tune.kv_cache_dtype=turboquant35 |

## T0d: T0 zero-shot (unrestricted, vllm qwen3-4b-vllm)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** unrestricted
- **KV Quant:** auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B]
- **Kind:** DATA POINT (no pass gate)
- **Started / Finished:** 2026-04-18T18:19:47 → 2026-04-18T18:23:22
- **Duration:** 215.10s
- **Result:** PASS (1/1)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| zero_shot_data_point | PASS | 215.10s | emitted_tool_call=False / chars=1180 / head=<think> Okay, the user wants me to use the tq-file-read tool to read the file located at /tmp/tqcli_agent_fixture.txt. They specified that I need to reply with  |

## T1_vq: T1 approve actionable (vllm qwen3-4b-vllm, ai_tinkering)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** ai_tinkering
- **KV Quant:** auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B]
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:23:22 → 2026-04-18T18:26:58
- **Duration:** 216.09s
- **Result:** PASS (5/5)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| spy_file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| actionable_tool_confirmed_once | PASS | 0.00s | confirm_fn_calls=[('tq-file-read', {'path': '/tmp/tqcli_agent_fixture.txt'}, 'actionable')] |
| observation_appended | PASS | 0.00s | hist_len=5 / hist_before=2 / obs_count=1 / obs_head=Observation:
secret_word=ALPHACHARLIE-329425010f50
 |
| real_engine_invoked_after_observation | PASS | 0.00s | pb.call_count=2 / scripted=1 / real=1 |
| secret_word_in_real_followup | PASS | 216.09s | secret=ALPHACHARLIE-329425010f50 / found_in_final=True / final_len=1159 / final_head=<think> Okay, the user asked me to read the fixture, and I used the tq-file-read tool to get the content. The observation shows that th |

## T4_vq: T4 unrestricted ReAct (vllm qwen3-4b-vllm)

- **Engine:** vllm
- **Model:** qwen3-4b-vllm
- **Mode:** unrestricted
- **KV Quant:** auto (quant=bitsandbytes, cpu_offload_gb=6.5)  [kv:none — turboquant_kv.json not present for Qwen3 4B]
- **Kind:** FUNCTIONAL (pass gated by assertions)
- **Started / Finished:** 2026-04-18T18:26:58 → 2026-04-18T18:30:48
- **Duration:** 229.86s
- **Result:** PASS (5/5)

| Step | Result | Duration | Details |
|------|--------|----------|---------|
| confirm_fn_never_called | PASS | 0.00s | confirm_calls=0 |
| file_read_called_once | PASS | 0.00s | calls=[{'path': '/tmp/tqcli_agent_fixture.txt'}] |
| observation_carries_secret | PASS | 0.00s | obs_head=Observation:
secret_word=ALPHACHARLIE-c1b898410475
 |
| real_engine_followup_invoked | PASS | 0.00s | real=1 scripted=1 |
| secret_word_in_real_followup | PASS | 229.86s | secret=ALPHACHARLIE-c1b898410475 / found_in_final=True / final_head=<think> Okay, the user asked me to read a file, and I did that using the tq-file-read action. The observation from the file is that the secret_word is s |
