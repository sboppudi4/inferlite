from __future__ import annotations

import pytest

from inferlite.cache import ContiguousKVCache, KVCacheConfig, PagedKVCache


def _config() -> KVCacheConfig:
    return KVCacheConfig(num_layers=4, num_heads=8, head_dim=16, dtype_bytes=2, block_size=8)


def test_contiguous_allocation_rounding() -> None:
    cache = ContiguousKVCache(_config(), chunk_size_tokens=16)
    alloc = cache.allocate("r1", seq_len=17)
    assert alloc.token_capacity == 32
    assert alloc.bytes_reserved >= alloc.bytes_used
    assert cache.stats()["active_requests"] == 1


def test_paged_allocation_and_free_reuses_blocks() -> None:
    cache = PagedKVCache(_config(), total_blocks=10)
    a1 = cache.allocate("r1", seq_len=9)  # 2 blocks
    pages_before_free = cache.request_pages("r1")
    assert len(pages_before_free) == 2
    assert a1.token_capacity == 16

    cache.free("r1")
    assert cache.request_pages("r1") == []
    free_after = cache.free_block_count()
    assert free_after == 10

    a2 = cache.allocate("r2", seq_len=8)  # 1 block
    assert a2.token_capacity == 8
    assert cache.free_block_count() == 9


def test_paged_raises_when_out_of_blocks() -> None:
    cache = PagedKVCache(_config(), total_blocks=1)
    cache.allocate("r1", seq_len=8)
    with pytest.raises(MemoryError):
        cache.allocate("r2", seq_len=9)


def test_paged_and_contiguous_bytes_used_match_for_same_seq_len() -> None:
    cfg = _config()
    contiguous = ContiguousKVCache(cfg, chunk_size_tokens=8)
    paged = PagedKVCache(cfg, total_blocks=100)
    seq_len = 19
    c = contiguous.allocate("r1", seq_len)
    p = paged.allocate("r1", seq_len)
    assert c.bytes_used == p.bytes_used
    assert p.bytes_reserved == p.token_capacity * (
        2 * cfg.num_layers * cfg.num_heads * cfg.head_dim * cfg.dtype_bytes
    )


@pytest.mark.parametrize("seq_len", [0, -1])
def test_allocators_reject_nonpositive_seq_len(seq_len: int) -> None:
    with pytest.raises(ValueError):
        ContiguousKVCache(_config()).allocate("r1", seq_len)
    with pytest.raises(ValueError):
        PagedKVCache(_config(), total_blocks=4).allocate("r1", seq_len)


@pytest.mark.parametrize("total_blocks", [0, -3])
def test_paged_rejects_nonpositive_total_blocks(total_blocks: int) -> None:
    with pytest.raises(ValueError):
        PagedKVCache(_config(), total_blocks=total_blocks)


def test_paged_free_unknown_request_is_noop() -> None:
    cache = PagedKVCache(_config(), total_blocks=4)
    cache.free("never-allocated")  # must not raise or leak phantom blocks
    assert cache.free_block_count() == 4
    assert cache.request_pages("never-allocated") == []


def test_utilization_of_empty_cache_is_one() -> None:
    assert ContiguousKVCache(_config()).utilization() == 1.0
    assert PagedKVCache(_config(), total_blocks=4).utilization() == 1.0


def test_paged_internal_fragmentation_bounded_by_block_size() -> None:
    cfg = _config()  # block_size=8
    cache = PagedKVCache(cfg, total_blocks=16)
    alloc = cache.allocate("r1", seq_len=17)  # 3 blocks -> capacity 24, waste = 7 tokens
    wasted_tokens = (alloc.bytes_reserved - alloc.bytes_used) // (
        2 * cfg.num_layers * cfg.num_heads * cfg.head_dim * cfg.dtype_bytes
    )
    assert wasted_tokens == 7
    assert wasted_tokens < cfg.block_size

