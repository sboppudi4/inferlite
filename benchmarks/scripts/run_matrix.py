from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from benchmarks.scripts.run_benchmark import run_once, write_row


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inferlite-url", type=str, default="http://localhost:8000")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8001")
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument(
        "--workloads",
        nargs="+",
        default=[
            "benchmarks/configs/default_workload.json",
            "benchmarks/configs/bursty_workload.json",
        ],
    )
    parser.add_argument("--out", type=str, default="benchmarks/results/benchmark_summary.csv")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        out.unlink()

    for workload in args.workloads:
        inferlite_row = await run_once(
            base_url=args.inferlite_url,
            backend="inferlite",
            workload_path=workload,
            api_key=args.api_key,
        )
        write_row(args.out, inferlite_row)

        naive_row = await run_once(
            base_url=args.inferlite_url,
            backend="naive",
            workload_path=workload,
            api_key=args.api_key,
        )
        write_row(args.out, naive_row)

        vllm_row = await run_once(
            base_url=args.vllm_url,
            backend="vllm",
            workload_path=workload,
            api_key=args.api_key,
        )
        write_row(args.out, vllm_row)

    print(f"Wrote benchmark matrix results to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

