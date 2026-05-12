"""
Generate benchmark charts from results.csv.

Usage:
  python src/plot.py
  python src/plot.py --input results/results.csv --output results/

Outputs:
  results/pareto_frontier.png      — recall vs latency scatter + Pareto frontier
  results/latency_distribution.png — p50/p95/p99 grouped bar chart
  results/disk_size_chart.png      — horizontal bar chart of disk usage
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Style ─────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e94560",
    "axes.labelcolor": "#eee",
    "text.color": "#eee",
    "xtick.color": "#ccc",
    "ytick.color": "#ccc",
    "grid.color": "#333",
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = ["#e94560", "#0f3460", "#533483", "#16c79a", "#f5a623"]


# ── Pareto frontier ──────────────────────────────────────────────────────
def _pareto_frontier(recall: np.ndarray, latency: np.ndarray):
    """Return indices forming the Pareto frontier (max recall, min latency)."""
    # Sort by latency ascending
    order = np.argsort(latency)
    pareto_idx = []
    best_recall = -1.0
    for i in order:
        if recall[i] > best_recall:
            best_recall = recall[i]
            pareto_idx.append(i)
    return pareto_idx


def plot_pareto(df: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 7))

    recalls = df["recall_at_10"].values
    latencies = df["latency_p50_ms"].values
    names = df["db_name"].values

    # scatter
    for i, (r, l, n) in enumerate(zip(recalls, latencies, names)):
        ax.scatter(l, r, s=200, c=COLORS[i % len(COLORS)], zorder=5, edgecolors="white", linewidth=1.5)
        ax.annotate(
            n, (l, r), textcoords="offset points",
            xytext=(12, 8), fontsize=11, fontweight="bold",
            color=COLORS[i % len(COLORS)],
        )

    # pareto line
    pidx = _pareto_frontier(recalls, latencies)
    if len(pidx) > 1:
        pidx_sorted = sorted(pidx, key=lambda i: latencies[i])
        ax.plot(
            latencies[pidx_sorted], recalls[pidx_sorted],
            "--", color="#e94560", alpha=0.6, linewidth=2, label="Pareto frontier",
        )

    ax.set_xlabel("Query Latency p50 (ms)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Recall@10", fontsize=13, fontweight="bold")
    ax.set_title("Pareto Frontier: Recall vs Latency", fontsize=16, fontweight="bold", pad=15)
    ax.grid(True, linestyle="--")
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {output}")


# ── Latency distribution ─────────────────────────────────────────────────
def plot_latency(df: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(df))
    width = 0.25
    names = df["db_name"].values

    bars_p50 = ax.bar(x - width, df["latency_p50_ms"], width, label="p50", color="#16c79a")
    bars_p95 = ax.bar(x, df["latency_p95_ms"], width, label="p95", color="#f5a623")
    bars_p99 = ax.bar(x + width, df["latency_p99_ms"], width, label="p99", color="#e94560")

    # value labels
    for bars in [bars_p50, bars_p95, bars_p99]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontweight="bold")
    ax.set_ylabel("Latency (ms)", fontsize=13, fontweight="bold")
    ax.set_title("Query Latency Distribution (p50 / p95 / p99)", fontsize=15, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--")
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {output}")


# ── Disk size ─────────────────────────────────────────────────────────────
def plot_disk(df: pd.DataFrame, output: Path):
    fig, ax = plt.subplots(figsize=(10, 5))

    names = df["db_name"].values
    disk_mb = df["disk_mb"].values
    y = np.arange(len(names))

    bars = ax.barh(y, disk_mb, color=COLORS[: len(names)], edgecolor="white", linewidth=0.8, height=0.6)

    for bar, val in zip(bars, disk_mb):
        ax.text(bar.get_width() + max(disk_mb) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} MB", va="center", fontweight="bold", fontsize=11)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontweight="bold", fontsize=12)
    ax.set_xlabel("Disk Size (MB)", fontsize=13, fontweight="bold")
    ax.set_title("Index Disk Size per Vector DB", fontsize=15, fontweight="bold", pad=15)
    ax.grid(True, axis="x", linestyle="--")
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ {output}")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate benchmark charts")
    parser.add_argument("--input", default="results/results.csv", help="Input CSV")
    parser.add_argument("--output", default="results/", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"📊  Loaded {len(df)} results from {args.input}\n")

    plot_pareto(df, out_dir / "pareto_frontier.png")
    plot_latency(df, out_dir / "latency_distribution.png")
    plot_disk(df, out_dir / "disk_size_chart.png")

    # Also print a nice markdown table
    print("\n📋  Results Table (Markdown):\n")
    print("| DB | Recall@10 | MRR@10 | p50 ms | p95 ms | p99 ms | Index (s) | Disk (MB) |")
    print("|---|---|---|---|---|---|---|---|")
    for _, row in df.iterrows():
        print(
            f"| {row['db_name']} "
            f"| {row['recall_at_10']:.4f} "
            f"| {row['mrr_at_10']:.4f} "
            f"| {row['latency_p50_ms']:.3f} "
            f"| {row['latency_p95_ms']:.3f} "
            f"| {row['latency_p99_ms']:.3f} "
            f"| {row['index_time_sec']:.2f} "
            f"| {row['disk_mb']:.1f} |"
        )

    print(f"\n🎉  All charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
