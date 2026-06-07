from __future__ import annotations

import time
from collections import defaultdict, deque


class PerKeyRateLimiter:
    """Simple per-key fixed-window limiter (requests/minute)."""

    def __init__(self) -> None:
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, *, key_id: str, rpm: int) -> bool:
        now = time.time()
        window_start = now - 60.0
        q = self._events[key_id]
        while q and q[0] < window_start:
            q.popleft()
        if len(q) >= rpm:
            return False
        q.append(now)
        return True

