"""Per-patient-day histogram of the night-minus-day dose-rate diff.

Three horizontal panels, one per sedative. Shows the overall distribution of
the exposure-of-interest (`*_dif` columns in `exposure_dataset.parquet` —
full hospital-stay coverage including day 0 and last day).

Vertical markers:
  - x = 0 (dashed black) — equal-shifts reference.
  - x = ±T (dashed red/blue) — per-drug clinical thresholds; positive shading
    above +T (night-higher tail) and negative shading below −T (day-higher
    tail), matching the 6-group cutpoints used elsewhere.

Each panel title reports counts above +T AND below −T so both directional
tails are readable at a glance (the previous version only showed +T).

This is the patient-day-level histogram. The hospitalization-level analog
lives at `night_day_diff_hist_by_hosp.py`.

Usage:
    uv run python code/descriptive/night_day_diff_hist.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.descriptive.night_day_diff_hist")

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
)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    logger.info(
        f"Loaded {len(df):,} patient-days from "
        f"{df['hospitalization_id'].nunique():,} hospitalizations"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0))

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        series = df[col].dropna()
        thr = THRESHOLDS[drug]

        # Crop to 1st–99th percentile so a few extreme patient-days don't
        # blow out the visible range. Both ±T thresholds appear inside the
        # crop in expected data.
        lo, hi = np.percentile(series, [1, 99])
        clipped = series[(series >= lo) & (series <= hi)]

        ax.hist(clipped, bins=60, color=DRUG_COLORS[drug],
                edgecolor="white", linewidth=0.3)

        # Shaded tails on BOTH sides so the 6-group cutpoints are visible.
        ax.axvspan(thr, hi, color="red", alpha=0.08)
        ax.axvspan(lo, -thr, color="blue", alpha=0.08)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.2, alpha=0.85,
                   label=f"+T = {thr} {DRUG_UNITS[drug]}")
        ax.axvline(-thr, color="blue", linestyle="--", linewidth=1.2, alpha=0.85,
                   label=f"−T = −{thr} {DRUG_UNITS[drug]}")

        n_above = int((series > thr).sum())
        n_below = int((series < -thr).sum())
        n_total = len(series)
        pct_above = 100.0 * n_above / n_total if n_total else 0.0
        pct_below = 100.0 * n_below / n_total if n_total else 0.0
        ax.set_title(
            f"{DRUG_LABELS[drug]}\n"
            f"+T: {n_above:,} ({pct_above:.1f}%) | "
            f"−T: {n_below:,} ({pct_below:.1f}%) | "
            f"total {n_total:,}",
            fontsize=10,
        )
        ax.set_xlabel(f"Night − day rate ({DRUG_UNITS[drug]})")
        ax.set_ylabel("Patient-days")
        ax.legend(loc="upper left", fontsize=7.5)

    fig.suptitle(
        "Night-minus-day dose rate: histogram per patient-day\n"
        "Red shading: diff > +T (markedly higher at night). "
        "Blue shading: diff < −T (markedly higher at day).",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.04,
        "Each bar = patient-days with diff in that bin. Histogram clipped to 1st-99th percentile. "
        "Counts in panel titles report patient-days above +T and below −T (the 6-group "
        "Markedly-night and Markedly-day buckets). "
        "Cohort: patient-day rate-diffs from the qualifying first IMV streak ≥ 24h. "
        "Single-shift days dropped (rate-diff = NaN; can't be histogrammed). "
        "Glossary in docs/descriptive_figures.md §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_hist")


if __name__ == "__main__":
    main()
