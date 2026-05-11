"""Pairwise Pearson correlation across the analytical-cohort continuous variables.

Per-figure descriptive QC artifact for spotting collinearity blocks among the
modeling covariates. Migrated from the retired `code/07_descriptive.py`
(2026-05-11): same outputs at the same path, just produced via the modern
per-figure convention with the federation-friendly `_shared` helpers.

Covers:
  - demographics + time: `age`, `_nth_day`
  - severity: `sofa_total`, `cci_score`, `elix_score`
  - day-vs-night dose diffs: `DIFF_COLS` (prop, fenteq, midazeq)
  - day-shift doses: `DAY_COLS`  (`_*_day_*_total`)
  - night-shift doses: `NIGHT_COLS` (`_*_night_*_total`)
  - shift covariates: NEE (7am/7pm), pH (7am/7pm), P/F (7am/7pm)

Columns not present in the parquet are silently skipped (defensive idiom
inherited from old 07) so a schema-evolved column rename here doesn't crash
the figure.

Outputs (both at `output_to_share/{SITE_NAME}/descriptive/`):
  - `pairwise_corr_matrix.csv` (with row index = variable name)
  - `pairwise_corr_matrix.png` (vlag-colormap heatmap, 14×10 inches)

Usage:
    uv run python code/descriptive/pairwise_corr_matrix.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DAY_COLS,
    DIFF_COLS,
    DRUGS,
    NIGHT_COLS,
    apply_style,
    load_modeling,
    save_csv,
    save_fig,
)


def main() -> None:
    apply_style()
    df = load_modeling()  # eligibility-cohort filter already applied

    # Variable list: same scientific intent as the old 07 corr matrix
    # but sourced through DAY/NIGHT/DIFF_COLS dicts so column-rename
    # churn doesn't bit-rot this script again.
    continuous_vars: list[str] = [
        "age",
        "_nth_day",
        "sofa_total",
        "cci_score",
        "elix_score",
        *(DIFF_COLS[d] for d in DRUGS),
        *(DAY_COLS[d] for d in DRUGS),
        *(NIGHT_COLS[d] for d in DRUGS),
        "nee_7am",
        "nee_7pm",
        "_ph_7am",
        "_ph_7pm",
        "_pf_7am",
        "_pf_7pm",
    ]

    # Defensive — skip columns missing from the current parquet schema.
    cols = [c for c in continuous_vars if c in df.columns]
    corr = df[cols].corr(method="pearson")

    save_csv(corr, "pairwise_corr_matrix", index=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        linewidths=0.5,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Pairwise Pearson Correlation (Continuous Variables)")
    fig.tight_layout()
    save_fig(fig, "pairwise_corr_matrix")


if __name__ == "__main__":
    main()
