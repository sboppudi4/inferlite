"""KV cache module (contiguous + paged in Phase 2)."""

from inferlite.cache.contiguous import ContiguousKVCache
from inferlite.cache.paged import PagedKVCache
from inferlite.cache.types import KVCacheConfig, RequestAllocation

__all__ = ["ContiguousKVCache", "PagedKVCache", "KVCacheConfig", "RequestAllocation"]
