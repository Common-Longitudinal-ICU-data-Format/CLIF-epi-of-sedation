"""Violin plot of diff distribution per ICU-day bin.

Complements `night_day_diff_mean_by_icu_day.py`: that plot shows whether the
mean diff shrinks over time; this one shows whether the full distribution
(spread, tails, asymmetry) also contracts. Restricted to ICU days 1..7
under full-24h coverage (n_hours_day = n_hours_night = 12).

Y-axis is clipped to the 1st–99th percentile of the pooled-across-days diff
so that long tails from day 1 don't squash later bins. ±T threshold lines
are drawn as horizontal references so the 6-group cutpoints stay visible.

Usage:
    uv run python code/descriptive/night_day_diff_violin_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    THRESHOLDS,
    apply_style,
    load_model_input,
    save_fig,
)


MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = [str(i) for i in range(MIN_DAY, MAX_DAY + 1)]


def main() -> None:
    apply_style()
    df = load_model_input()
    # Restrict to full-24h ICU days 1..7 (drops day 0 partial,
    # extubation-day partial, and days 8+). Replaces the old
    # `_drop_last_day_per_patient + cap_day` pipeline; the registry's
    # `_is_full_24h_day` flag is the canonical "complete 12+12 hr
    # coverage" filter.
    in_range = df["_nth_day"].between(MIN_DAY, MAX_DAY)
    df = df.loc[df["_is_full_24h_day"] & in_range].copy()
    df["_nth_day_bin"] = pd.Categorical(
        df["_nth_day"].astype(int).astype(str),
        categories=DAY_BINS, ordered=True,
    )
    bins = DAY_BINS

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharex=True)

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        thr = THRESHOLDS[drug]
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
        # ±T threshold reference lines (the 6-group cutpoints).
        if lo <= thr <= hi:
            ax.axhline(thr, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
        if lo <= -thr <= hi:
            ax.axhline(-thr, color="blue", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{b}\nn={n:,}" for b, n in zip(bins, ns)])
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"{DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_ylim(lo, hi)
        ax.set_title(f"{DRUG_LABELS[drug]}  (T = ±{thr} {DRUG_UNITS[drug]})")

    fig.suptitle(
        "Spread of night-minus-day dose rate by ICU day (violin)\n"
        "Median = black tick. ±T (red/blue dashed) mark the 6-group cutpoints. "
        "y-axis clipped to 1st-99th percentile of pooled diff.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.03,
        "Each violin = distribution of diff for one ICU-day bin (width ∝ density at that y-level). "
        "Sister figure to `night_day_diff_combined_by_icu_day.png` — narrowing violins left-to-right "
        "indicate shift-to-shift dosing converging as patients stabilize. "
        "Cohort: full-24h ICU-day rows from `model_input_by_id_imvday.parquet` "
        "(`_is_full_24h_day = TRUE AND _nth_day BETWEEN 1 AND 7`). Day 0 partial intubation day, "
        "trajectory-final partial day, and days 8+ are dropped at the load filter so each violin "
        "reflects a fully-comparable 12+12 hr coverage population. "
        "Glossary: docs/descriptive_figures.md §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_violin_by_icu_day")


if __name__ == "__main__":
    main()
