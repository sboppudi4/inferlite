# Building an LLM inference server in Python: what continuous batching actually does

## Draft thesis

I built InferLite to understand the internals behind vLLM-style serving: continuous batching,
KV cache management, and request-level scheduling under multi-tenant load. The goal was not to
beat optimized CUDA kernels, but to build a readable inference engine that exposes real systems
trade-offs and can be benchmarked honestly.

## Sections to complete

1. Why request-level batching is not enough.
2. Continuous batching scheduler loop explained with annotated pseudocode.
3. Contiguous KV cache vs paged blocks: where memory waste comes from.
4. Multi-tenant policy: tiers, rate limits, and fairness under bursts.
5. Benchmark setup:
   - inferlite vs vllm vs naive pipeline baseline
   - steady and bursty workloads
   - fixed model/hardware controls
6. Results:
   - throughput, p95/p99 latency, TTFT, TPOT
   - where InferLite is slower and why
7. Why Python is still valuable for learning infra internals (and where it clearly loses).
