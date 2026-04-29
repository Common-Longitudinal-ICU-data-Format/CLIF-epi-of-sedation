"""Violin plot of diff distribution per ICU-day bin.

Complements `night_day_diff_mean_by_icu_day.py`: that plot shows whether the
mean diff shrinks over time; this one shows whether the full distribution
(spread, tails, asymmetry) also contracts. Same 1..7 / '8+' binning.

Y-axis is clipped to the 1st–99th percentile of the pooled-across-days diff
so that long tails from day 1 don't squash later bins.

Usage:
    uv run python code/descriptive/night_day_diff_spread_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    apply_style,
    cap_day,
    load_exposure,
    prepare_diffs,
    save_fig,
)


def main() -> None:
    apply_style()
    df = cap_day(prepare_diffs(load_exposure()), max_day=7)
    bins = list(df["_nth_day_bin"].cat.categories)

    # Bottom region reserved for the inline footnote.
    fig, axes = plt.subplots(1, 3, figsize=(15, 9.5), sharex=True)

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        pooled = df[col].dropna()
        lo, hi = np.percentile(pooled, [1, 99])

        data_per_bin = [
            df.loc[df["_nth_day_bin"] == b, col].dropna().clip(lower=lo, upper=hi).to_numpy()
            for b in bins
        ]
        ns = [len(d) for d in data_per_bin]

        # Filter out any bin with fewer than 2 points (violin requires variance);
        # matplotlib raises if asked to violin a constant array.
        positions = np.arange(len(bins))
        valid = [i for i, d in enumerate(data_per_bin) if len(d) >= 2 and np.ptp(d) > 0]
        vp = ax.violinplot(
            [data_per_bin[i] for i in valid],
            positions=positions[valid],
            widths=0.8,
            showmedians=True,
            showextrema=False,
        )
        color = DRUG_COLORS[drug]
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.55)
            body.set_edgecolor("dimgray")
        if "cmedians" in vp:
            vp["cmedians"].set_color("black")
            vp["cmedians"].set_linewidth(1.0)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{b}\nn={n:,}" for b, n in zip(bins, ns)])
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"{DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_ylim(lo, hi)
        ax.set_title(DRUG_LABELS[drug])

    fig.suptitle(
        "Spread of night-minus-day dose rate by ICU day",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.34)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "One panel per drug. Each violin shape = the distribution of `diff` for one ICU-day bin\n"
        "(see x-axis: 1, 2, …, 7, 8+). The wider the violin at a given y-value, the more patient-days\n"
        "have a diff at that level.\n"
        "\n"
        "  black horizontal stroke inside each violin → cohort median diff for that bin.\n"
        "  black dashed (y = 0)                       → the equal-shifts reference.\n"
        "  X-axis label `n=N`                         → number of patient-days in that bin.\n"
        "\n"
        "Y-axis is clipped to the 1st-99th percentile of pooled-across-days diff so long tails from\n"
        "early days don't squash later bins.\n"
        "\n"
        "USE THIS FIGURE TO ANSWER\n"
        "  Does the spread of diff narrow as patients stabilize? If the violins shrink toward zero\n"
        "  going left-to-right, sedation patterns become more synchronized between shifts. If they\n"
        "  stay wide, day-to-day variability persists across the stay.\n"
        "\n"
        "  This is a complement to `night_day_diff_mean_by_icu_day.png` which shows only the mean.\n"
        "  A flat-mean trajectory can hide a shrinking-spread story or vice versa.\n"
        "\n"
        "GLOSSARY\n"
        "  diff — (per-hour rate during night-shift hours) − (per-hour rate during day-shift hours).\n"
    )
    fig.text(
        0.04, 0.001, footnote,
        ha="left", va="bottom", fontsize=8, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, "night_day_diff_spread_by_icu_day")


if __name__ == "__main__":
    main()
