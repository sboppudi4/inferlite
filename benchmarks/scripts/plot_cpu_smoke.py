from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INFERLITE_COLOR = "#2dd4bf"  # teal
NAIVE_COLOR = "#94a3b8"  # slate


def _read_rows(csv_path: Path) -> dict[str, dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return {row["backend"]: row for row in csv.DictReader(f)}


def render(csv_path: Path, out_path: Path) -> None:
    rows = _read_rows(csv_path)
    if "inferlite" not in rows or "naive" not in rows:
        raise SystemExit(
            f"{csv_path} needs both 'inferlite' and 'naive' rows. "
            "Run `make bench-cpu` (or the two run_benchmark.py invocations) first."
        )

    metrics = [
        ("p50 latency", "latency_p50_s"),
        ("p95 latency", "latency_p95_s"),
        ("p95 TTFT", "ttft_p95_s"),
    ]
    labels = [m[0] for m in metrics]
    inferlite_ms = [float(rows["inferlite"][k]) * 1000 for _, k in metrics]
    naive_ms = [float(rows["naive"][k]) * 1000 for _, k in metrics]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.axisbelow": True,
        }
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = range(len(labels))
    width = 0.38
    left = [i - width / 2 for i in x]
    right = [i + width / 2 for i in x]
    bars_n = ax.bar(left, naive_ms, width, color=NAIVE_COLOR, label="naive (1 req at a time)")
    bars_i = ax.bar(right, inferlite_ms, width, color=INFERLITE_COLOR,
                    label="inferlite (static batch)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("latency (ms) — lower is better")
    for bars in (bars_n, bars_i):
        ax.bar_label(bars, fmt="%.0f ms", padding=3, fontsize=9)
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("InferLite vs. naive baseline — CPU smoke run (distilgpt2, qps≈2)")
    fig.text(
        0.5, -0.02,
        "Sequential load (batch size 1), so continuous batching can't engage — backends are "
        "within noise, as expected on CPU. The batching win needs concurrent in-flight requests "
        "on a GPU.",
        ha="center", va="top", fontsize=8.5, wrap=True, color="#475569",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the CPU smoke latency comparison (inferlite vs naive)."
    )
    parser.add_argument("--csv", type=str, default="benchmarks/results/cpu_summary.csv")
    parser.add_argument("--out", type=str, default="docs/assets/cpu_smoke_latency.png")
    args = parser.parse_args()
    render(Path(args.csv), Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
