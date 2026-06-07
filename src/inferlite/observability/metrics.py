from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_LATENCY_SECONDS = Histogram(
    "inferlite_request_latency_seconds",
    "End-to-end request latency",
    ["endpoint", "tier"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
)

TTFT_SECONDS = Histogram(
    "inferlite_ttft_seconds",
    "Time to first token",
    ["endpoint", "tier"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
)

TPOT_SECONDS = Histogram(
    "inferlite_tpot_seconds",
    "Time per output token",
    ["endpoint", "tier"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
)

BATCH_SIZE = Histogram(
    "inferlite_batch_size",
    "Batch size used for generation calls",
    ["endpoint"],
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

QUEUE_DEPTH = Gauge("inferlite_queue_depth", "Scheduler pending queue depth")
KV_CACHE_UTILIZATION = Gauge("inferlite_kv_cache_utilization", "KV cache utilization ratio")

PREFILL_TOKENS_TOTAL = Counter("inferlite_prefill_tokens_total", "Prompt tokens processed")
DECODE_TOKENS_TOTAL = Counter("inferlite_decode_tokens_total", "Completion tokens processed")

