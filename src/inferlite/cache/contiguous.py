from __future__ import annotations

from inferlite.cache.types import KVCacheConfig, RequestAllocation


class ContiguousKVCache:
    """Baseline KV cache model with contiguous reservation per request.

    In practice this style of cache tends to over-reserve when sequence lengths vary because
    each request gets a large contiguous region up-front (or rounded up in chunks).
    """

    def __init__(self, config: KVCacheConfig, chunk_size_tokens: int = 64) -> None:
        self.config = config
        self.chunk_size_tokens = chunk_size_tokens
        self._allocations: dict[str, RequestAllocation] = {}

    def _bytes_per_token(self) -> int:
        # K and V tensors for each layer/head/dim.
        return (
            2
            * self.config.num_layers
            * self.config.num_heads
            * self.config.head_dim
            * self.config.dtype_bytes
        )

    def allocate(self, request_id: str, seq_len: int) -> RequestAllocation:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        rounded_capacity = ((seq_len + self.chunk_size_tokens - 1) // self.chunk_size_tokens) * (
            self.chunk_size_tokens
        )
        bytes_per_token = self._bytes_per_token()
        allocation = RequestAllocation(
            request_id=request_id,
            seq_len=seq_len,
            token_capacity=rounded_capacity,
            bytes_used=seq_len * bytes_per_token,
            bytes_reserved=rounded_capacity * bytes_per_token,
        )
        self._allocations[request_id] = allocation
        return allocation

    def free(self, request_id: str) -> None:
        self._allocations.pop(request_id, None)

    def utilization(self) -> float:
        reserved = sum(x.bytes_reserved for x in self._allocations.values())
        if reserved == 0:
            return 1.0
        used = sum(x.bytes_used for x in self._allocations.values())
        return used / reserved

    def stats(self) -> dict[str, int | float]:
        reserved = sum(x.bytes_reserved for x in self._allocations.values())
        used = sum(x.bytes_used for x in self._allocations.values())
        return {
            "active_requests": len(self._allocations),
            "bytes_reserved": reserved,
            "bytes_used": used,
            "waste_bytes": reserved - used,
            "utilization": self.utilization(),
        }

