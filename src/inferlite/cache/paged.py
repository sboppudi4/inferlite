from __future__ import annotations

from dataclasses import dataclass

from inferlite.cache.types import KVCacheConfig, RequestAllocation


@dataclass(frozen=True)
class PagedRequestState:
    request_id: str
    seq_len: int
    page_ids: list[int]


class PagedKVCache:
    """Paged KV cache with a block allocator and request page tables."""

    def __init__(self, config: KVCacheConfig, total_blocks: int) -> None:
        if total_blocks <= 0:
            raise ValueError("total_blocks must be > 0")
        self.config = config
        self.total_blocks = total_blocks
        self._free_blocks: list[int] = list(range(total_blocks))
        self._states: dict[str, PagedRequestState] = {}

    def _bytes_per_token(self) -> int:
        return (
            2
            * self.config.num_layers
            * self.config.num_heads
            * self.config.head_dim
            * self.config.dtype_bytes
        )

    def _bytes_per_block(self) -> int:
        return self._bytes_per_token() * self.config.block_size

    def allocate(self, request_id: str, seq_len: int) -> RequestAllocation:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        blocks_needed = (seq_len + self.config.block_size - 1) // self.config.block_size
        if len(self._free_blocks) < blocks_needed:
            raise MemoryError("insufficient free KV blocks")

        page_ids = [self._free_blocks.pop() for _ in range(blocks_needed)]
        self._states[request_id] = PagedRequestState(
            request_id=request_id, seq_len=seq_len, page_ids=page_ids
        )

        token_capacity = blocks_needed * self.config.block_size
        bytes_per_token = self._bytes_per_token()
        return RequestAllocation(
            request_id=request_id,
            seq_len=seq_len,
            token_capacity=token_capacity,
            bytes_used=seq_len * bytes_per_token,
            bytes_reserved=token_capacity * bytes_per_token,
        )

    def free(self, request_id: str) -> None:
        state = self._states.pop(request_id, None)
        if state is None:
            return
        self._free_blocks.extend(state.page_ids)

    def request_pages(self, request_id: str) -> list[int]:
        state = self._states.get(request_id)
        if state is None:
            return []
        return list(state.page_ids)

    def free_block_count(self) -> int:
        return len(self._free_blocks)

    def utilization(self) -> float:
        reserved = 0
        used = 0
        bytes_per_token = self._bytes_per_token()
        for state in self._states.values():
            used += state.seq_len * bytes_per_token
            capacity = len(state.page_ids) * self.config.block_size
            reserved += capacity * bytes_per_token
        if reserved == 0:
            return 1.0
        return used / reserved

    def stats(self) -> dict[str, int | float]:
        bytes_per_token = self._bytes_per_token()
        used = 0
        reserved = 0
        for state in self._states.values():
            used += state.seq_len * bytes_per_token
            reserved += len(state.page_ids) * self.config.block_size * bytes_per_token
        return {
            "active_requests": len(self._states),
            "free_blocks": len(self._free_blocks),
            "total_blocks": self.total_blocks,
            "bytes_per_block": self._bytes_per_block(),
            "bytes_reserved": reserved,
            "bytes_used": used,
            "waste_bytes": reserved - used,
            "utilization": self.utilization(),
        }

