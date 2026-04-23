"""Stacked-bar trajectory of night-day diff distribution by ICU day.

Three panels (propofol / fent-eq / midaz-eq). X = day_n bin (1..7, '8+').
Each bar is 100% tall, split into four diverging categories around the
per-drug threshold T. Clinical hypothesis: the red ("> T") segment should
shrink left-to-right as patients stabilize.

Usage:
    uv run python code/descriptive/pct_uptitrated_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_BIN_COLORS,
    DIFF_COLS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    THRESHOLDS,
    apply_style,
    cap_day,
    categorize_diff,
    load_analytical,
    prepare_diffs,
    save_fig,
)


def main() -> None:
    apply_style()
    df = cap_day(prepare_diffs(load_analytical()), max_day=7)
    bins = list(df["_nth_day_bin"].cat.categories)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)

    category_order: list[str] = []

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        thr = THRESHOLDS[drug]
        x = np.arange(len(bins))
        ns = []

        # Pre-compute a matrix of (bin × category) → pct
        pct_matrix = np.zeros((len(bins), 4))
        for bi, b in enumerate(bins):
            sub = df.loc[df["_nth_day_bin"] == b, col].dropna()
            cat = categorize_diff(sub, thr)
            counts = cat.value_counts().reindex(cat.cat.categories, fill_value=0)
            total = int(counts.sum())
            ns.append(total)
            if not category_order:
                category_order = list(counts.index)
            if total > 0:
                pct_matrix[bi, :] = (counts.to_numpy() / total) * 100.0

        # Plot stacks segment-by-segment from bottom up.
        cum = np.zeros(len(bins))
        for seg_idx in range(4):
            seg = pct_matrix[:, seg_idx]
            ax.bar(x, seg, bottom=cum, width=0.7,
                   color=DIFF_BIN_COLORS[seg_idx], edgecolor="white", linewidth=0.3,
                   label=category_order[seg_idx] if drug == DRUGS[0] else None)
            # Annotate segments that are tall enough to fit text
            for xi, pct in enumerate(seg):
                if pct >= 4.0:
                    ax.text(
                        xi, cum[xi] + pct / 2, f"{pct:.0f}",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if seg_idx in (0, 3) else "black",
                    )
            cum += seg

        # N-per-bin annotation below each bar
        for xi, n in enumerate(ns):
            ax.text(xi, -4, f"n={n:,}", ha="center", va="top",
                    fontsize=7, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel("ICU day")
        ax.set_ylim(-8, 104)
        ax.set_title(
            f"{DRUG_LABELS[drug]}   (T = {THRESHOLDS[drug]} {DRUG_UNITS[drug]})"
        )

    axes[0].set_ylabel("% of patient-days")

    # Legend on right of figure, order top-down to mirror the visual stack
    handles, labels = axes[0].get_legend_handles_labels()
    order_top_down = list(reversed(category_order))
    h_by_label = dict(zip(labels, handles))
    fig.legend(
        [h_by_label[l] for l in order_top_down],
        order_top_down,
        title="Diff category (vs threshold T)",
        loc="center left", bbox_to_anchor=(0.99, 0.5), frameon=False,
    )

    fig.suptitle(
        "Distribution of night-minus-day dose-rate diff by ICU day",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "pct_uptitrated_by_icu_day")


if __name__ == "__main__":
    main()
