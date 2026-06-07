from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except Exception:
        return 0.0


def plot_summary(csv_path: str, out_dir: str) -> None:
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError("No benchmark rows found.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    backends = [r["backend"] for r in rows]
    tps = [_to_float(r, "throughput_tokens_per_s") for r in rows]
    p95 = [_to_float(r, "latency_p95_s") for r in rows]
    ttft95 = [_to_float(r, "ttft_p95_s") for r in rows]

    plt.figure(figsize=(9, 4))
    plt.bar(backends, tps)
    plt.title("Throughput (tokens/s)")
    plt.ylabel("tokens/s")
    plt.tight_layout()
    plt.savefig(out / "throughput_tokens_per_s.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(backends, p95)
    plt.title("Latency p95 (s)")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig(out / "latency_p95_s.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 4))
    plt.bar(backends, ttft95)
    plt.title("TTFT p95 (s)")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig(out / "ttft_p95_s.png", dpi=150)
    plt.close()


def write_markdown_report(csv_path: str, out_path: str) -> None:
    rows = _read_rows(csv_path)
    lines = [
        "# Benchmark Summary",
        "",
        "| backend | workload | req_total | req_ok | tok/s | p95 latency (s) | p95 TTFT (s) | p95 TPOT (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {backend} | {workload} | {requests_total} | {requests_ok} | {throughput_tokens_per_s} | "
            "{latency_p95_s} | {ttft_p95_s} | {tpot_p95_s} |".format(**r)
        )
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="benchmarks/results/benchmark_summary.csv")
    parser.add_argument("--out-dir", type=str, default="benchmarks/results")
    args = parser.parse_args()

    plot_summary(args.csv, args.out_dir)
    write_markdown_report(args.csv, str(Path(args.out_dir) / "benchmark_report.md"))
    print(f"Wrote plots and report to {args.out_dir}")


if __name__ == "__main__":
    main()

