"""Histogram of per-hospitalization mean night-minus-day diff (encounter-block).

Sister figure to `night_day_diff_hist.py`. Where the per-patient-day
histogram pools ALL patient-days, this version aggregates UP to the
hospitalization level: each hospitalization contributes ONE diff value per
sedative — the mean of that hospitalization's on-sedative patient-days.
This is the headline aggregation recommended in
`docs/descriptive_figures.md §5.4` for the manuscript.

Important: a patient with multiple admissions contributes multiple data
points (CLIF's per-encounter convention; matches `sed_dose_by_shift.csv`).

Three horizontal panels, one per sedative. Vertical markers: 0 (dashed
black), ±T (dashed red/blue) — both directional thresholds drawn so the
6-group cutpoints are visually anchored. Mean (solid green) and median
(dashed orange) of the per-hospitalization distribution.

Usage:
    uv run python code/descriptive/night_day_diff_hist_by_hosp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DAY_COLS,
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    NIGHT_COLS,
    THRESHOLDS,
    apply_style,
    load_exposure,
    prepare_diffs,
    save_fig,
)


def _per_hospitalization_mean_diff(df: pd.DataFrame, drug: str) -> pd.Series:
    """One row per hospitalization_id — mean diff across on-sedative days.

    "On-sedative" = at least one shift had a nonzero rate (so off-drug
    holidays don't pull the mean to 0). NaN rate-diff days within a
    hospitalization are dropped INDIVIDUALLY (not the whole hospitalization)
    so single-shift rows on a single day don't disqualify the encounter.
    Hospitalizations with NO on-sedative days are excluded entirely.
    """
    diff_col = DIFF_COLS[drug]
    day_col = DAY_COLS[drug]
    night_col = NIGHT_COLS[drug]

    on_drug = df[
        (df[day_col].fillna(0) > 0) | (df[night_col].fillna(0) > 0)
    ].copy()
    return (
        on_drug.dropna(subset=[diff_col])
        .groupby("hospitalization_id")[diff_col]
        .mean()
    )


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    n_hosp_total = df["hospitalization_id"].nunique()
    print(f"Loaded {len(df):,} patient-days from {n_hosp_total:,} hospitalizations")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    for ax, drug in zip(axes, DRUGS):
        per_hosp = _per_hospitalization_mean_diff(df, drug)
        n_hosp = len(per_hosp)
        n_excluded = n_hosp_total - n_hosp
        thr = THRESHOLDS[drug]
        if n_hosp == 0:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        lo, hi = np.percentile(per_hosp, [1, 99])
        clipped = per_hosp[(per_hosp >= lo) & (per_hosp <= hi)]

        ax.hist(clipped, bins=50, color=DRUG_COLORS[drug],
                edgecolor="white", linewidth=0.3)

        # Shaded tails on BOTH sides (6-group cutpoints).
        ax.axvspan(thr, hi, color="red", alpha=0.08)
        ax.axvspan(lo, -thr, color="blue", alpha=0.08)

        mean = float(per_hosp.mean())
        median = float(per_hosp.median())
        ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.axvline(mean, color="green", linestyle="-", linewidth=1.6,
                   label=f"Mean = {mean:+.3f}")
        ax.axvline(median, color="darkorange", linestyle="--", linewidth=1.4,
                   label=f"Median = {median:+.3f}")
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.2, alpha=0.85,
                   label=f"+T = {thr} {DRUG_UNITS[drug]}")
        ax.axvline(-thr, color="blue", linestyle="--", linewidth=1.2, alpha=0.85,
                   label=f"−T = −{thr} {DRUG_UNITS[drug]}")

        n_above = int((per_hosp > thr).sum())
        n_below = int((per_hosp < -thr).sum())
        pct_above = 100.0 * n_above / n_hosp
        pct_below = 100.0 * n_below / n_hosp
        ax.set_title(
            f"{DRUG_LABELS[drug]}\n"
            f"+T: {n_above:,} ({pct_above:.1f}%) | "
            f"−T: {n_below:,} ({pct_below:.1f}%)\n"
            f"n hosp on-sedative = {n_hosp:,}  "
            f"(excluded = {n_excluded:,})",
            fontsize=9,
        )
        ax.set_xlabel(f"Per-hospitalization mean night−day rate ({DRUG_UNITS[drug]})")
        ax.set_ylabel("Hospitalizations")
        ax.legend(loc="upper left", fontsize=7)

    fig.suptitle(
        "Night-minus-day dose rate: per-hospitalization mean (encounter-block aggregation)\n"
        "Each observation = one hospitalization (a re-admitted patient contributes multiple). "
        "Sister figure to `night_day_diff_hist.png` (per-patient-day).",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.04,
        "This is the headline aggregation in docs/descriptive_figures.md §5.4. The cohort mean here may swing "
        "in the opposite direction from the per-patient-day mean — that's the aggregation-paradox signal "
        "(short-stay night-heavy days → fewer hospitalization-level data points). "
        "Cohort: per-hospitalization mean rate-diff. Hospitalizations included if they have ≥ 1 day with "
        "finite rate-diff; single-shift days drop within the per-hosp mean but don't disqualify the "
        "hospitalization. Glossary: §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_hist_by_hosp")


if __name__ == "__main__":
    main()
