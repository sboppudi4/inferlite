from __future__ import annotations

import asyncio
import heapq
import itertools
from collections.abc import Awaitable, Callable

from inferlite.scheduler.types import GenerationRequest, ScheduledBatch

DecodeStepFn = Callable[[list[GenerationRequest]], Awaitable[None]]


class ContinuousBatchScheduler:
    """Token-level continuous batching scheduler.

    Key behavior:
    - Requests are admitted between decode steps, not only at batch boundaries.
    - Active requests each advance by one token per decode step.
    - Higher-priority queued requests can preempt lower-priority active requests.
    """

    def __init__(
        self,
        *,
        max_active_requests: int,
        decode_step_fn: DecodeStepFn,
        enable_preemption: bool = True,
    ) -> None:
        if max_active_requests <= 0:
            raise ValueError("max_active_requests must be > 0")
        self.max_active_requests = max_active_requests
        self.decode_step_fn = decode_step_fn
        self.enable_preemption = enable_preemption

        self._active: dict[str, GenerationRequest] = {}
        self._pending_heap: list[tuple[int, float, int, GenerationRequest]] = []
        self._seq = itertools.count()
        self._step_id = 0
        self._lock = asyncio.Lock()

    async def submit(self, request: GenerationRequest) -> None:
        async with self._lock:
            heapq.heappush(
                self._pending_heap,
                (-int(request.priority), request.arrived_at, next(self._seq), request),
            )

    def queue_depth(self) -> int:
        return len(self._pending_heap)

    def active_request_ids(self) -> list[str]:
        return list(self._active.keys())

    def _lowest_priority_active(self) -> GenerationRequest | None:
        if not self._active:
            return None
        # Break ties by preserving existing active requests (FIFO within same priority).
        return min(self._active.values(), key=lambda r: (int(r.priority), r.arrived_at))

    def _try_preempt_for(self, candidate: GenerationRequest) -> bool:
        if not self.enable_preemption:
            return False
        victim = self._lowest_priority_active()
        if victim is None:
            return False
        if int(candidate.priority) <= int(victim.priority):
            return False

        # Move victim back to queue (state preserved), then activate higher-priority request.
        self._active.pop(victim.request_id, None)
        heapq.heappush(
            self._pending_heap,
            (-int(victim.priority), victim.arrived_at, next(self._seq), victim),
        )
        self._active[candidate.request_id] = candidate
        return True

    def _admit_pending(self) -> None:
        """Admit queued requests into the active set before each decode step.

        This method is the heart of continuous batching: admission happens at every token step,
        which allows newly arrived requests to join quickly instead of waiting for long generations
        to finish.
        """
        while self._pending_heap:
            _, _, _, candidate = self._pending_heap[0]
            if candidate.finished or candidate.remaining_tokens <= 0:
                heapq.heappop(self._pending_heap)
                continue

            if len(self._active) < self.max_active_requests:
                heapq.heappop(self._pending_heap)
                self._active[candidate.request_id] = candidate
                continue

            # Pool full: preempt only when queued work has strictly higher priority.
            popped = heapq.heappop(self._pending_heap)
            _ = popped
            if not self._try_preempt_for(candidate):
                # Could not preempt; put candidate back and stop trying this step.
                heapq.heappush(
                    self._pending_heap,
                    (-int(candidate.priority), candidate.arrived_at, next(self._seq), candidate),
                )
                break

    def _collect_finished(self) -> list[GenerationRequest]:
        finished: list[GenerationRequest] = []
        for req in list(self._active.values()):
            if req.remaining_tokens <= 0:
                req.finished = True
                self._active.pop(req.request_id, None)
                finished.append(req)
        return finished

    async def run_step(self) -> ScheduledBatch:
        async with self._lock:
            self._admit_pending()
            active_batch = list(self._active.values())
            self._step_id += 1
            batch = ScheduledBatch(
                step_id=self._step_id,
                request_ids=[r.request_id for r in active_batch],
                batch_size=len(active_batch),
            )

        if active_batch:
            await self.decode_step_fn(active_batch)

        async with self._lock:
            for req in active_batch:
                if req.request_id in self._active:
                    req.generated_tokens += 1
                    req.remaining_tokens -= 1
            self._collect_finished()
        return batch

    async def drain(self, max_steps: int = 10_000) -> list[ScheduledBatch]:
        """Run until no active/pending requests remain or max_steps is reached."""
        batches: list[ScheduledBatch] = []
        for _ in range(max_steps):
            batch = await self.run_step()
            batches.append(batch)
            if batch.batch_size == 0 and self.queue_depth() == 0:
                break
        return batches

