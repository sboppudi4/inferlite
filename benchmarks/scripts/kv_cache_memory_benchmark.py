from __future__ import annotations

import argparse
import random
from pathlib import Path

from inferlite.cache import ContiguousKVCache, KVCacheConfig, PagedKVCache


def run_trial(num_requests: int, seed: int, max_seq_len: int) -> dict[str, float | int]:
    random.seed(seed)
    cfg = KVCacheConfig(num_layers=32, num_heads=32, head_dim=128, dtype_bytes=2, block_size=16)
    seq_lens = [random.randint(8, max_seq_len) for _ in range(num_requests)]

    contiguous = ContiguousKVCache(cfg, chunk_size_tokens=64)
    # Large enough block pool for the simulated workload.
    paged = PagedKVCache(cfg, total_blocks=200_000)

    for i, seq_len in enumerate(seq_lens):
        request_id = f"req-{i}"
        contiguous.allocate(request_id, seq_len)
        paged.allocate(request_id, seq_len)

    c_stats = contiguous.stats()
    p_stats = paged.stats()
    return {
        "num_requests": num_requests,
        "max_seq_len": max_seq_len,
        "contiguous_reserved_mb": round(c_stats["bytes_reserved"] / (1024 * 1024), 2),
        "contiguous_waste_mb": round(c_stats["waste_bytes"] / (1024 * 1024), 2),
        "paged_reserved_mb": round(p_stats["bytes_reserved"] / (1024 * 1024), 2),
        "paged_waste_mb": round(p_stats["waste_bytes"] / (1024 * 1024), 2),
        "contiguous_utilization": round(float(c_stats["utilization"]), 4),
        "paged_utilization": round(float(p_stats["utilization"]), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--out",
        type=str,
        default="benchmarks/results/kv_cache_memory_comparison.csv",
    )
    args = parser.parse_args()

    result = run_trial(args.requests, args.seed, args.max_seq_len)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(result.keys())
    values = [str(result[h]) for h in headers]
    out_path.write_text(",".join(headers) + "\n" + ",".join(values) + "\n", encoding="utf-8")
    print(f"Wrote results to {out_path}")


if __name__ == "__main__":
    main()

