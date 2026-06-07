from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KVCacheConfig:
    num_layers: int
    num_heads: int
    head_dim: int
    dtype_bytes: int = 2  # fp16 default
    block_size: int = 16


@dataclass(frozen=True)
class RequestAllocation:
    request_id: str
    seq_len: int
    token_capacity: int
    bytes_used: int
    bytes_reserved: int

