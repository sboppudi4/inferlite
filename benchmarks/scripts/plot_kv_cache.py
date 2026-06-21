from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make `inferlite` importable when run from the repo root without an editable install.
_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from benchmarks.scripts.kv_cache_memory_benchmark import run_trial  # noqa: E402

CONTIGUOUS_COLOR = "#94a3b8"  # slate
PAGED_COLOR = "#2dd4bf"  # teal
LABEL_CONTIGUOUS = "contiguous (64-tok chunks)"
LABEL_PAGED = "paged (16-tok blocks)"


def _gb(mb: float) -> float:
    return mb / 1024.0


def render(out_path: Path, *, requests: int = 512, seed: int = 7) -> dict[str, dict]:
    short = run_trial(num_requests=requests, seed=seed, max_seq_len=256)
    long = run_trial(num_requests=requests, seed=seed, max_seq_len=2048)
    workloads = ["seq 8–256\n(short, variable)", "seq 8–2048\n(long, variable)"]

    util_contig = [short["contiguous_utilization"] * 100, long["contiguous_utilization"] * 100]
    util_paged = [short["paged_utilization"] * 100, long["paged_utilization"] * 100]
    waste_contig = [_gb(short["contiguous_waste_mb"]), _gb(long["contiguous_waste_mb"])]
    waste_paged = [_gb(short["paged_waste_mb"]), _gb(long["paged_waste_mb"])]

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

    fig, (ax_util, ax_waste) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(workloads))
    width = 0.38

    def _grouped(ax, contig, paged, *, fmt: str):
        left = [i - width / 2 for i in x]
        right = [i + width / 2 for i in x]
        bars_c = ax.bar(left, contig, width, label=LABEL_CONTIGUOUS, color=CONTIGUOUS_COLOR)
        bars_p = ax.bar(right, paged, width, label=LABEL_PAGED, color=PAGED_COLOR)
        ax.set_xticks(list(x))
        ax.set_xticklabels(workloads)
        for bars in (bars_c, bars_p):
            ax.bar_label(bars, fmt=fmt, padding=3, fontsize=9)
        return bars_c, bars_p

    handles = _grouped(ax_util, util_contig, util_paged, fmt="%.1f%%")
    ax_util.set_title("KV-cache utilization — higher is better")
    ax_util.set_ylabel("utilization (%)")
    ax_util.set_ylim(0, 112)

    _grouped(ax_waste, waste_contig, waste_paged, fmt="%.1f GB")
    ax_waste.set_title("Wasted KV memory — lower is better")
    ax_waste.set_ylabel("reserved-but-unused (GB)")
    ax_waste.set_ylim(0, max(waste_contig) * 1.28)

    fig.suptitle(
        "Paged vs. contiguous KV cache — 512 requests, fp16, 32 layers × 32 heads × 128 dim",
        fontsize=12.5,
        y=1.0,
    )
    fig.legend(
        handles,
        [LABEL_CONTIGUOUS, LABEL_PAGED],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"short": short, "long": long}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the paged-vs-contiguous KV fragmentation result."
    )
    parser.add_argument("--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="docs/assets/kv_cache_fragmentation.png")
    args = parser.parse_args()
    out = Path(args.out)
    render(out, requests=args.requests, seed=args.seed)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
