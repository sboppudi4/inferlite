# Building an LLM inference server in Python: what continuous batching actually does

I built InferLite to answer a question that kept nagging me while reading about vLLM and
TGI: *what does an inference server actually do between the moment an HTTP request arrives and the
moment the next token comes out?* The headline features — continuous batching, paged KV cache,
priority scheduling — are easy to name and hard to picture. So I wrote a readable Python
implementation of each one, with benchmarks, and documented where it wins and where it loses.

This is the write-up. It is deliberately honest about the limitations, which are documented in full
in [`docs/design.md`](./design.md).

## 1. Why request-level batching is not enough

The obvious way to serve a model is one request at a time. The slightly less obvious way is *static
batching*: collect a handful of prompts, run them through a single `model.generate()` call, return
all the results.

Static batching has a structural flaw. Every request in a batch is held until the **longest**
generation in that batch finishes. An 8-token reply and a 256-token reply in the same batch both
occupy a slot for 256 steps. Worse, a request that arrives one step after the batch starts has to
wait for the whole batch to drain before it can even begin.

That "wait for the slowest tablemate" tax is what continuous batching removes.

## 2. The continuous batching scheduler

Continuous batching makes admission and completion decisions *between every decode step* instead of
at batch boundaries. In InferLite that lives in
[`scheduler/continuous.py`](../src/inferlite/scheduler/continuous.py).

The core state is an active pool (a dict, capped at `max_active_requests`) and a pending min-heap.
The heap key is the interesting part:

```python
(-int(request.priority), request.arrived_at, next(self._seq), request)
```

Three keys, each doing a job: negated priority so the highest tier pops first; arrival time so
within a tier it's FIFO-fair; and a monotonic counter so two requests are never compared directly
(request objects aren't orderable — this key is load-bearing, not decoration).

At the top of every step, `_admit_pending()` pulls runnable requests into the active pool. Because
this runs every single token step, a request that shows up mid-generation joins almost immediately.
That's the whole trick.

### Preemption

When the pool is full and a higher-priority request is waiting, the scheduler preempts — but only
when the newcomer's priority is **strictly higher** than the lowest-priority active request. The
victim is pushed back onto the heap with its decode progress preserved, so it resumes rather than
restarts. The strict-priority gate is what stops two free-tier requests from endlessly evicting each
other.

## 3. Contiguous vs paged KV cache: where the memory goes

Both cache implementations model the same physical cost — bytes of key/value tensor per token:

```
bytes_per_token = 2 (K and V) × num_layers × num_heads × head_dim × dtype_bytes
```

The difference is reservation strategy.
[`ContiguousKVCache`](../src/inferlite/cache/contiguous.py) rounds each request up to a 64-token
chunk, so a 129-token request reserves 192 tokens and wastes the tail.
[`PagedKVCache`](../src/inferlite/cache/paged.py) hands out 16-token blocks from a shared free list,
so worst-case waste is one block (15 tokens) instead of one chunk (63).

I measured it (CPU-only, reproducible with `seed=7`, 512 requests, a 32-layer/32-head/128-dim fp16
model shape):

| workload | cache | utilization | wasted KV memory |
|---|---|---:|---:|
| seq 8–2048 | contiguous | 97.0% | 7,917 MB |
| seq 8–2048 | paged | **99.3%** | **1,798 MB** |
| seq 8–256  | contiguous | 80.5% | 7,979 MB |
| seq 8–256  | paged | **94.6%** | **1,867 MB** |

Paging cuts wasted KV memory by about **77%** in both cases. The effect is biggest on short,
variable-length requests (utilization 80% → 95%); when sequences are long the tail waste amortizes
and the gap shrinks. That's the regime distinction the textbooks gloss over, and you can watch it
happen by changing one flag. Full table and method: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

## 4. Multi-tenancy is just the scheduling order

There are three tiers — `PAID`, `FREE`, `BATCH` — and they aren't a bolt-on subsystem. The tier maps
straight onto scheduler priority, which feeds both the admission heap key and the preemption gate.
A paid request jumps the queue and can preempt running free/batch work; batch work only runs when
nothing better is waiting. Rate limiting is a per-key fixed-window counter, kept deliberately simple.

## 5. What I'd build next, and what Python can't do

I'm explicit about the seams (see [`docs/design.md`](./design.md) §5):

- The scheduler models continuous batching at the token-accounting level and is tested against a
  fake decode function. The real HF model call is still a static-batch `generate()`. Fusing the
  per-step scheduler with true token-by-token decoding is the main piece of unfinished work.
- The paged allocator is correct and measured, but isn't yet wired into the live decode loop.
- There's no prefix/KV block sharing, and preemption doesn't model the cost of evicting KV blocks.

And the honest framing: **Python is the wrong language to win this benchmark.** InferLite is expected
to lose to vLLM on absolute throughput. That was never the point. The point was to be able to read
the scheduler and the allocator, change a block size, and *watch the trade-off move* — which is
exactly what the fragmentation numbers above let you do.

## Read the code

- Scheduler: [`src/inferlite/scheduler/continuous.py`](../src/inferlite/scheduler/continuous.py)
- Caches: [`paged.py`](../src/inferlite/cache/paged.py) vs [`contiguous.py`](../src/inferlite/cache/contiguous.py)
- Design notes and limitations: [`docs/design.md`](./design.md)
- How the numbers are produced: [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md)
