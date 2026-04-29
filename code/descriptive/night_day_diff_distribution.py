"""Histogram of the continuous night-minus-day dose-rate diff for each sedative.

Three horizontal panels, one per drug. Shows the overall distribution of the
exposure-of-interest (`*_dif` columns in exposure_dataset.parquet — full
hospital-stay coverage including day 0 and last day). Vertical markers:
x=0 (dashed black, "equal across shifts") and x=threshold (dashed red, the
"markedly higher at night" cutoff). The positive tail above threshold is
shaded to highlight the night-higher subcohort.

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
    load_exposure,
    prepare_diffs,
    save_fig,
    threshold_label,
)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    print(f"Loaded {len(df):,} patient-days from {df['hospitalization_id'].nunique():,} hospitalizations")

    # Bottom region reserved for the inline footnote.
    fig, axes = plt.subplots(1, 3, figsize=(15, 8.0))

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

        # Shade positive region above threshold to mark the night-higher tail.
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
    fig.subplots_adjust(bottom=0.40)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "One histogram per drug. Each bar = number of patient-days with `diff` falling in that bin.\n"
        "Histogram is clipped to the 1st-99th percentile of diff so a few extreme outliers don't\n"
        "squash the visible range.\n"
        "\n"
        "VERTICAL LINES\n"
        "  black dashed (x = 0)  — the equal-shifts reference. Bars to the right have night > day;\n"
        "                           bars to the left have day > night.\n"
        "  red dashed   (x = T)  — the per-drug clinical threshold. Patient-days with diff > T (the\n"
        "                           shaded red region to its right) are the \"markedly higher at night\"\n"
        "                           bucket used by other figures.\n"
        "\n"
        "TITLE NUMBERS\n"
        "  N above threshold / total — count and percent of patient-days with diff > T.\n"
        "  Important: the matching D > N count (diff < -T) is NOT shown here — see\n"
        "  `dose_pattern_subgroup_*.csv` or `paradox_summary.png` for both sides at once.\n"
        "\n"
        "GLOSSARY\n"
        "  diff — (per-hour rate during night-shift hours) − (per-hour rate during day-shift hours).\n"
        "  T    — drug-specific cutoff: 10 mcg/kg/min (propofol), 25 mcg/hr (fentanyl-eq), 1 mg/hr\n"
        "         (midazolam-eq).\n"
    )
    fig.text(
        0.04, 0.001, footnote,
        ha="left", va="bottom", fontsize=8, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, "night_day_diff_distribution")


if __name__ == "__main__":
    main()
