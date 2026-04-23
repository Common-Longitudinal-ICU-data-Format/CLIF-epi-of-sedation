"""Histogram of the continuous night-minus-day dose-rate diff for each sedative.

Three horizontal panels, one per drug. Shows the overall distribution of the
exposure-of-interest (`*_dif` columns in analytical_dataset.parquet). Vertical
markers: x=0 (dashed black, "no change") and x=threshold (dashed red, the
clinical "uptitrated" cutoff). The positive tail above threshold is shaded to
visually emphasize the up-titration subcohort.

Usage:
    uv run python code/descriptive/night_day_diff_distribution.py
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
    THRESHOLDS,
    apply_style,
    load_analytical,
    prepare_diffs,
    save_fig,
    threshold_label,
)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_analytical())
    print(f"Loaded {len(df):,} patient-days from {df['hospitalization_id'].nunique():,} hospitalizations")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        series = df[col].dropna()
        thr = THRESHOLDS[drug]

        # Crop to 1st–99th percentile so the x-axis isn't blown out by a
        # handful of extreme patient-days. The threshold still appears
        # inside the crop for all three drugs in expected data.
        lo, hi = np.percentile(series, [1, 99])
        clipped = series[(series >= lo) & (series <= hi)]

        ax.hist(clipped, bins=60, color=DRUG_COLORS[drug], edgecolor="white", linewidth=0.3)

        # Shade positive region above threshold to mark the "up-titrated" tail.
        ax.axvspan(thr, hi, color="red", alpha=0.08)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.2, alpha=0.8,
                   label=f"threshold {threshold_label(drug)}")

        n_above = int((series > thr).sum())
        pct = 100.0 * n_above / len(series) if len(series) else 0.0
        ax.set_title(
            f"{DRUG_LABELS[drug]}  —  {n_above:,} / {len(series):,} "
            f"patient-days above threshold ({pct:.1f}%)"
        )
        ax.set_xlabel(f"Night − day rate ({DRUG_UNITS[drug]})")
        ax.set_ylabel("Patient-days")
        ax.legend(loc="upper left")

    fig.suptitle(
        "Distribution of night-minus-day sedative dose rate (per patient-day)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "night_day_diff_distribution")


if __name__ == "__main__":
    main()
