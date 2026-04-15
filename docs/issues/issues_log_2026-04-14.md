# Issues Log — 2026-04-14

**Project:** tqCLI TurboQuant KV Cache Integration
**Phase:** Post-v0.5.0 — end-to-end verification

## Summary Table

| # | GitHub | Title | Type | Severity | Status | Depends On |
|---|--------|-------|------|----------|--------|------------|
| 1 | #14 | Build llama-cpp-python against turboquant fork | Enhancement | Major | **CLOSED** | #13 |
| 2 | #15 | Build vllm-turboquant from source for CUDA 12.8 | Enhancement | Major | **CLOSED** | #13 |
| 3 | #16 | End-to-end TurboQuant KV inference tests | Testing | Major | **CLOSED** | #14, #15 |
| 4 | #17 | Push CUDA 12.8 fixes to fork repos | Architecture | Moderate | **CLOSED** | #14, #15 |
| 5 | #18 | TurboQuant KV tok/s benchmarks and comparison report | Testing | Moderate | **CLOSED** | #14, #15, #16 |

## Dependency Graph

```
#13 (v0.5.0 code) ─── DONE
    │
    ├── #14 (llama-cpp-python build)
    │       │
    ├── #15 (vllm-turboquant build)
    │       │
    │   ┌───┴───┐
    │   ▼       ▼
    │  #16 (E2E tests)
    │   │
    │   ▼
    │  #18 (benchmarks)
    │
    └── #17 (push fixes to forks)
```

## Execution Order
1. #14 and #15 in parallel (build dependencies)
2. #17 after builds verified (push fixes)
3. #16 after #14 and #15 (end-to-end tests)
4. #18 after #16 (benchmark comparison)
