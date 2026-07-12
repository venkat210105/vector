"""Generates the comparison charts embedded in docs/BENCHMARKS.md, using the
results already recorded there. Plots saved numbers rather than re-running
the (slow, ~10 minute at N=15,000) benchmark just to make an image -- if you
re-run `python -m benchmarks.run_benchmark` and get different numbers,
update the constants below to match before regenerating.

Run with: python -m benchmarks.make_charts
"""
import os

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "docs/images"

# Fixed color-per-entity mapping, consistent across every chart.
COLOR_FLAT = "#2a78d6"
COLOR_HNSW = "#008300"
COLOR_FAISS = "#e34948"
COLOR_BEFORE_FIX = "#4a3aa7"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def _style_ax(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_MUTED)
    ax.tick_params(colors=INK_SECONDARY, labelsize=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)


def chart_recall_vs_ef_search() -> None:
    ef_values = [10, 20, 50, 100, 200, 400]
    recall = [0.66, 0.80, 0.97, 1.00, 1.00, 1.00]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(ef_values, recall, color=COLOR_HNSW, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(ef_values)
    ax.set_xticklabels([str(v) for v in ef_values])
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("ef_search", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("recall@10", color=INK_SECONDARY, fontsize=11)
    ax.set_title("Recall@10 vs ef_search (N=3,000)", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    for x, y in zip(ef_values, recall):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK_SECONDARY)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/recall_vs_ef_search.png")
    plt.close(fig)


def chart_latency_comparison() -> None:
    labels = ["FlatIndex", "HNSW (ours)", "faiss HNSW"]
    p50 = [0.466, 5.518, 0.175]
    colors = [COLOR_FLAT, COLOR_HNSW, COLOR_FAISS]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars = ax.bar(labels, p50, color=colors, width=0.55, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("p50 latency (ms/query, log scale)", color=INK_SECONDARY, fontsize=11)
    ax.set_title("Query latency at N=3,000 (ef_search=100)", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    for bar, val in zip(bars, p50):
        ax.annotate(f"{val:.3f} ms", (bar.get_x() + bar.get_width() / 2, val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, color=INK_SECONDARY)
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/latency_comparison.png")
    plt.close(fig)


def chart_distance_computation_ratio() -> None:
    n_values = [500, 1000, 3000]
    ratio = [99.6, 87.4, 52.2]

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.plot(n_values, ratio, color=COLOR_HNSW, linewidth=2, marker="o", markersize=6, zorder=3)
    ax.set_ylim(0, 108)
    ax.set_xlabel("N (dataset size)", color=INK_SECONDARY, fontsize=11)
    ax.set_ylabel("HNSW distance calls, as % of brute force", color=INK_SECONDARY, fontsize=11)
    ax.set_title("Algorithmic advantage grows with N", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    for x, y in zip(n_values, ratio):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color=INK_SECONDARY)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/distance_computation_ratio.png")
    plt.close(fig)


def chart_latency_vs_n() -> None:
    n_labels = ["N=3,000", "N=15,000"]
    flat_p50 = [0.466, 3.263]
    hnsw_p50 = [5.518, 6.328]

    x = np.arange(len(n_labels))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars1 = ax.bar(x - width / 2, flat_p50, width, label="FlatIndex", color=COLOR_FLAT, zorder=3)
    bars2 = ax.bar(x + width / 2, hnsw_p50, width, label="HNSW (ours)", color=COLOR_HNSW, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(n_labels)
    ax.set_ylabel("p50 latency (ms/query)", color=INK_SECONDARY, fontsize=11)
    ax.set_title("FlatIndex stays faster even at 5x the scale", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    ax.set_ylim(0, max(hnsw_p50) * 1.18)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f} ms", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, loc="upper left", fontsize=10, labelcolor=INK_SECONDARY)
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/latency_vs_n.png")
    plt.close(fig)


def chart_batching_before_after() -> None:
    """The optimization story: batching distance computations across a
    candidate frontier (docs/SETBACKS.md), with a size threshold added
    after the first attempt regressed insert time. Shows both metrics
    together since the fix's effect on them was not the same direction
    or magnitude -- a single "average speedup" number would hide that.
    """
    metrics = ["Insert\n(ms/point)", "Query p50\n(ms)"]
    before = [12.85, 7.312]
    after = [11.35, 5.518]

    x = np.arange(len(metrics))
    width = 0.32

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    bars1 = ax.bar(x - width / 2, before, width, label="before fix", color=COLOR_BEFORE_FIX, zorder=3)
    bars2 = ax.bar(x + width / 2, after, width, label="after fix (hybrid threshold)", color=COLOR_HNSW, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("milliseconds", color=INK_SECONDARY, fontsize=11)
    ax.set_title("Batching distance computations: before vs after", color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    ax.set_ylim(0, max(before) * 1.18)
    for bars in (bars1, bars2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9, color=INK_SECONDARY)
    ax.legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK_SECONDARY)
    _style_ax(ax)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/batching_before_after.png")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    chart_recall_vs_ef_search()
    chart_latency_comparison()
    chart_distance_computation_ratio()
    chart_latency_vs_n()
    chart_batching_before_after()
    print(f"Charts written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
