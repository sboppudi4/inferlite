from __future__ import annotations

import asyncio

from inferlite.scheduler import ContinuousBatchScheduler, GenerationRequest, Priority


async def _noop_decode_step(active: list[GenerationRequest]) -> None:
    _ = active
    await asyncio.sleep(0)


def test_continuous_batching_admits_new_requests_between_steps() -> None:
    async def run() -> None:
        scheduler = ContinuousBatchScheduler(
            max_active_requests=2,
            decode_step_fn=_noop_decode_step,
            enable_preemption=False,
        )
        await scheduler.submit(
            GenerationRequest(
                "r1", "hello", max_new_tokens=3, priority=Priority.FREE, arrived_at=0.0
            )
        )
        first = await scheduler.run_step()
        assert first.request_ids == ["r1"]

        await scheduler.submit(
            GenerationRequest(
                "r2", "world", max_new_tokens=2, priority=Priority.FREE, arrived_at=1.0
            )
        )
        second = await scheduler.run_step()
        assert set(second.request_ids) == {"r1", "r2"}

    asyncio.run(run())


def test_higher_priority_can_preempt_when_pool_full() -> None:
    async def run() -> None:
        scheduler = ContinuousBatchScheduler(
            max_active_requests=1,
            decode_step_fn=_noop_decode_step,
            enable_preemption=True,
        )
        low = GenerationRequest(
            "low", "a", max_new_tokens=5, priority=Priority.BATCH, arrived_at=0.0
        )
        high = GenerationRequest(
            "high", "b", max_new_tokens=1, priority=Priority.PAID, arrived_at=1.0
        )
        await scheduler.submit(low)
        s1 = await scheduler.run_step()
        assert s1.request_ids == ["low"]

        await scheduler.submit(high)
        s2 = await scheduler.run_step()
        assert s2.request_ids == ["high"]
        assert "low" in scheduler.active_request_ids() or scheduler.queue_depth() > 0

    asyncio.run(run())


def test_scheduler_drains_all_requests() -> None:
    async def run() -> None:
        scheduler = ContinuousBatchScheduler(
            max_active_requests=2,
            decode_step_fn=_noop_decode_step,
            enable_preemption=True,
        )
        reqs = [
            GenerationRequest("r1", "a", max_new_tokens=2, priority=Priority.FREE, arrived_at=0.0),
            GenerationRequest("r2", "b", max_new_tokens=3, priority=Priority.BATCH, arrived_at=0.1),
            GenerationRequest("r3", "c", max_new_tokens=1, priority=Priority.PAID, arrived_at=0.2),
        ]
        for r in reqs:
            await scheduler.submit(r)
        _ = await scheduler.drain()
        assert scheduler.queue_depth() == 0
        assert scheduler.active_request_ids() == []
        assert all(r.finished for r in reqs)

    asyncio.run(run())

