# InferLite Design Notes

This document explains the design decisions behind InferLite's scheduler, KV cache, and
multi-tenant policy, the trade-offs each one makes, and the limitations I chose to accept. It is
meant to be read alongside the code in `src/inferlite/`.

The goal of the project is not to beat optimized serving stacks. It is to make the control-plane
decisions an inference server makes — *which request runs, when it is admitted, how much memory it
reserves* — explicit and readable.

---

## 1. Why request-level batching is not enough

The naive baseline (`engine/naive_runner.py`) runs one request at a time through a Hugging Face
`pipeline()`. Even the slightly better static batcher (`engine/model_runner.py`) groups a fixed set
of prompts into a single `model.generate()` call.

Static batching has one structural problem: **every request in the batch is held until the longest
generation in that batch finishes.** A 256-token request and an 8-token request in the same batch
both occupy a slot for 256 steps. A request that arrives one step after the batch starts waits for
the entire batch to drain before it can run.

Continuous batching fixes this by making admission and completion decisions *between every decode
step* rather than at batch boundaries. That is the single idea the scheduler exists to model.

---

## 2. The continuous batching scheduler

Source: `scheduler/continuous.py`, types in `scheduler/types.py`.

### Data structures

- `_active: dict[str, GenerationRequest]` — requests currently advancing one token per step, capped
  at `max_active_requests`.
- `_pending_heap: list[tuple[int, float, int, GenerationRequest]]` — a min-heap of queued requests.

The heap key is `(-int(priority), arrived_at, monotonic_seq)`:

1. `-priority` — Python's `heapq` is a min-heap, so negating priority makes the **highest** priority
   pop first.
2. `arrived_at` — within the same priority, the **earliest-arrived** request wins (FIFO fairness).
3. `monotonic_seq` — a strictly increasing counter that breaks remaining ties deterministically and
   prevents the heap from ever trying to compare two `GenerationRequest` objects (which are not
   orderable). This third key is load-bearing, not cosmetic.

### The admission loop

`_admit_pending()` runs at the top of every `run_step()`. This is the heart of continuous batching:
because admission happens every token step, a request that arrives mid-generation can join almost
immediately instead of waiting for the current batch to finish.

The loop:

1. Skips finished or zero-budget requests at the head of the heap.
2. If the active pool has room, admits the head request.
3. If the pool is full, attempts preemption (below). If preemption fails, the request is pushed back
   and the loop stops for this step — there is no point scanning deeper, because nothing behind a
   non-admittable head can be admitted either under the current policy.

### Preemption

`_try_preempt_for()` allows a queued request to displace an active one, but only when the queued
request has **strictly higher** priority than the lowest-priority active request. The displaced
"victim" is pushed back onto the pending heap with its state intact (`generated_tokens` and
`remaining_tokens` are preserved), so it resumes later rather than restarting.

`_lowest_priority_active()` selects the victim as the request with the lowest `(priority,
arrived_at)`. Two deliberate choices live here:

- **Strictly-higher-priority gate.** A paid request preempts a free one; a free request never
  preempts another free one. This prevents preemption thrashing within a tier.
- **State-preserving eviction.** Because decode state is kept on the request object, preemption is
  cheap to model. In a real engine this is exactly where you would also have to evict/restore the
  victim's KV cache blocks — see the limitation in §5.

### Decode is pluggable

`run_step()` calls an injected `decode_step_fn(active_batch)` and then decrements each active
request's `remaining_tokens` by one. The scheduler does **not** itself call the model. This is a
deliberate separation: the scheduler is the control plane, and it is tested (see
`tests/test_scheduler_continuous.py`) against a fake decode function so the admission/preemption
logic can be verified without a GPU. Wiring this loop to true token-by-token HF generation is listed
as future work in §5.

---

## 3. KV cache: contiguous vs paged

Sources: `cache/contiguous.py`, `cache/paged.py`, shared config in `cache/types.py`.

Both caches model the same physical quantity — bytes of key/value tensor memory per token:

```
bytes_per_token = 2 (K and V) × num_layers × num_heads × head_dim × dtype_bytes
```

The difference is **how capacity is reserved**, and therefore how much memory is wasted.

### Contiguous (the baseline)

`ContiguousKVCache` rounds each request up to a `chunk_size_tokens` boundary (default 64). A
129-token request reserves 192 tokens of capacity. The waste is the tail of the last chunk — up to
`chunk_size_tokens - 1` tokens per request. With many short, variable-length requests this waste
dominates, which is the classic motivation for paging.

### Paged (the improvement)

`PagedKVCache` carves memory into fixed blocks (`block_size`, default 16) and hands out a page table
(`page_ids`) per request. A request reserves `ceil(seq_len / block_size)` blocks, so the worst-case
internal waste drops to `block_size - 1` tokens — roughly a 4× reduction versus the 64-token chunk
baseline. Freed blocks return to a shared free list and can be reused by any request, so memory is
not stranded behind a single tenant.

Both caches expose a `utilization()` (`bytes_used / bytes_reserved`) and a `stats()` dict reporting
`waste_bytes`. Those metrics are the point: they let the benchmark harness *measure* fragmentation
rather than assert it.

### The block-size trade-off

`block_size` is the central tuning knob:

- **Smaller blocks** → less internal fragmentation (tighter `utilization`), but more blocks per
  request, a larger page table, and more allocator bookkeeping.
- **Larger blocks** → cheaper bookkeeping, but more wasted tokens in each request's final block.

16 is a reasonable middle that matches what production paged-attention implementations use. The
allocator is intentionally simple: a free list of block ids, `pop()` on allocate, `extend()` on
free. There is no block sharing across requests (no prefix/copy-on-write sharing) — see §5.

---

## 4. Multi-tenant policy

Sources: `auth/store.py`, `auth/rate_limit.py`, `scheduler/types.py` (`Priority`).

Three tiers map directly onto scheduler priority:

```
Priority.PAID  = 2   # preempts everything below
Priority.FREE  = 1
Priority.BATCH = 0   # best-effort, preempted first
```

Because priority flows into both the admission heap key and the preemption gate, tenant tier is not
a separate subsystem — it *is* the scheduling order. A paid request jumps the queue and can preempt
a running free/batch request; batch work only runs when nothing better is waiting.

Rate limiting (`PerKeyRateLimiter`) is a per-key fixed-window counter (requests/minute) backed by an
in-memory `deque` of timestamps. This is deliberately the simplest correct thing; its limitations
(window-edge bursts, no cross-instance sharing) are called out in §5.

---

## 5. Known limitations and honest trade-offs

I would rather document these than imply they don't exist.

1. **The decode path is static-batch, the scheduler is continuous.** `model_runner.py` issues one
   `model.generate()` for a batch and shares `max_new_tokens` across it, so the *real* model call
   still has the "wait for the longest" property. The continuous-batching behavior is modeled and
   tested at the scheduler/token-accounting level; fusing the per-step scheduler with true
   token-by-token HF decoding (feeding the model one step at a time and re-batching each step) is the
   main piece of future work.
2. **The paged cache is not wired into the live decode loop yet.** It is a correct, tested allocator
   with measurable fragmentation, but the API path does not currently allocate/free real blocks per
   running request. The cache is used as a measurement and design artifact.
3. **No KV block sharing.** There is no prefix caching or copy-on-write block sharing across
   requests, which a production engine uses to deduplicate shared system prompts.
4. **Preemption ignores KV cost.** Modeling preemption as a dict move is cheap; a real engine must
   also evict and later restore the victim's KV blocks, which has real latency.
5. **Rate limiter is in-memory and per-process.** Running multiple instances would require a shared
   store (Redis/Postgres). The fixed window also permits a 2× burst across a window boundary; a
   sliding window or token bucket would be stricter.
6. **TTFT in non-streaming benchmark runs is approximated**, as noted in `benchmarks/README.md`.

---

## 6. Where to read next

- Scheduler behavior and tie-breaking: `scheduler/continuous.py`
- Fragmentation math: `cache/paged.py` vs `cache/contiguous.py`
- Tenant priority wiring: `scheduler/types.py` + `auth/`
- How claims are measured: `benchmarks/README.md` and `benchmarks/RESULTS.md`
