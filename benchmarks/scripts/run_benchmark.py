from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from pathlib import Path

import httpx

from benchmarks.scripts.load_generator import generate_requests, load_config


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100.0) * (len(ordered) - 1)))
    return float(ordered[idx])


def _endpoint_for_backend(base_url: str, backend: str) -> str:
    if backend == "naive":
        return f"{base_url.rstrip('/')}/v1/completions/baseline"
    return f"{base_url.rstrip('/')}/v1/completions"


async def _fire_request(
    client: httpx.AsyncClient,
    *,
    url: str,
    api_key: str,
    payload: dict[str, object],
) -> dict[str, float]:
    started = time.perf_counter()
    response = await client.post(url, headers={"Authorization": f"Bearer {api_key}"}, json=payload)
    elapsed = max(0.0, time.perf_counter() - started)
    if response.status_code != 200:
        return {"ok": 0.0, "latency_s": elapsed, "ttft_s": elapsed, "tpot_s": 0.0, "tokens": 0.0}
    body = response.json()
    usage = body.get("usage", {})
    completion_tokens = float(usage.get("completion_tokens", 0))
    # Non-streaming approximation: TTFT as 40% of total latency for now.
    ttft = elapsed * 0.4
    tpot = (elapsed - ttft) / max(1.0, completion_tokens)
    return {
        "ok": 1.0,
        "latency_s": elapsed,
        "ttft_s": ttft,
        "tpot_s": max(0.0, tpot),
        "tokens": completion_tokens,
    }


async def run_once(
    *,
    base_url: str,
    backend: str,
    workload_path: str,
    api_key: str,
) -> dict[str, float | str]:
    cfg = load_config(workload_path)
    events = generate_requests(cfg)
    endpoint = _endpoint_for_backend(base_url, backend)
    latencies: list[float] = []
    ttfts: list[float] = []
    tpots: list[float] = []
    token_count = 0.0
    ok_count = 0.0

    async with httpx.AsyncClient(timeout=120.0) as client:
        run_start = time.perf_counter()
        for e in events:
            target_t = run_start + float(e["arrival_offset_s"])
            sleep_for = target_t - time.perf_counter()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            payload = {
                "model": e["model"],
                "prompt": e["prompt"],
                "max_tokens": e["max_tokens"],
                "temperature": 0.0,
                "top_p": 1.0,
            }
            result = await _fire_request(client, url=endpoint, api_key=api_key, payload=payload)
            latencies.append(result["latency_s"])
            ttfts.append(result["ttft_s"])
            tpots.append(result["tpot_s"])
            token_count += result["tokens"]
            ok_count += result["ok"]
        elapsed_total = max(0.001, time.perf_counter() - run_start)

    return {
        "backend": backend,
        "workload": cfg.name,
        "requests_total": float(len(events)),
        "requests_ok": ok_count,
        "throughput_tokens_per_s": token_count / elapsed_total,
        "latency_p50_s": _percentile(latencies, 50),
        "latency_p95_s": _percentile(latencies, 95),
        "latency_p99_s": _percentile(latencies, 99),
        "ttft_p50_s": _percentile(ttfts, 50),
        "ttft_p95_s": _percentile(ttfts, 95),
        "tpot_p50_s": _percentile(tpots, 50),
        "tpot_p95_s": _percentile(tpots, 95),
        "latency_mean_s": statistics.mean(latencies) if latencies else 0.0,
    }


def write_row(path: str, row: dict[str, float | str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    exists = p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--backend", type=str, choices=["inferlite", "vllm", "naive"], required=True)
    parser.add_argument(
        "--workload",
        type=str,
        default="benchmarks/configs/default_workload.json",
    )
    parser.add_argument("--api-key", type=str, required=True)
    parser.add_argument(
        "--out",
        type=str,
        default="benchmarks/results/benchmark_summary.csv",
    )
    args = parser.parse_args()
    row = await run_once(
        base_url=args.base_url,
        backend=args.backend,
        workload_path=args.workload,
        api_key=args.api_key,
    )
    write_row(args.out, row)
    print(f"Wrote benchmark row for backend={args.backend} to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())

