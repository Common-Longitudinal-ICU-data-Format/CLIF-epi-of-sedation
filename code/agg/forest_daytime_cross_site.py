"""Cross-site forest plot: daytime dose-rate effects (supplement figure).

Mirror of `forest_night_day_cross_site.py` but with the three DAYTIME
absolute-rate predictors instead of the night–day diff predictors. Same
3×3 layout, same spec encoding, same outcomes — destined for the
manuscript's supplement so reviewers can see whether the dose-level signal
diverges from the night–day variation signal.

Layout (3 rows × 3 cols):
  - Rows: sedative — daytime propofol, daytime fentanyl eq, daytime midaz eq.
  - Cols: outcome — SBT eligible, SBT delivered (multiday), Successful extub.
  - Per-panel y-positions: one per site (top = first alphabetically).
  - Per-site: TWO dots side-by-side at tight ±0.08 jitter.
        - filled circle  → daydose_wt
        - open  circle   → clinical_wt
  - X-axis: log-scaled OR with major ticks at sparse round values and
            minor gridlines every 0.05.
  - Reference dashed vertical line at OR=1.

OR scaling: each site's `08_models.py` reports the OR for going from the
10th→90th percentile of the predictor's NON-ZERO subset (the "among the
exposed" framing — see `08_models.py:826–827`), so the daytime-row
scaling is conceptually consistent with the night-day diff scaling.

Outputs:
  - output_to_agg/forest_daytime_cross_site.csv
  - output_to_agg/figures/forest_daytime_cross_site.png

Usage:
    uv run python code/agg/forest_daytime_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/forest_daytime_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _forest_helpers import (  # noqa: E402
    add_or_reference_line,
    apply_or_xaxis,
    or_xlim_from_data,
    stack_per_site,
)
from _shared import (  # noqa: E402
    SITE_PALETTE,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# ── Figure-fixed selectors ────────────────────────────────────────────────
OUTCOMES: list[tuple[str, str]] = [
    ("sbt_elig_next_day",          "SBT eligible (next day)"),
    ("sbt_done_multiday_next_day", "SBT delivered (multiday)"),
    ("success_extub_next_day",     "Successful extubation"),
]

SPECS: list[tuple[str, str, dict]] = [
    ("daydose_wt",  "daydose + weight",  {"marker": "o", "fillstyle": "full"}),
    ("clinical_wt", "clinical + weight", {"marker": "o", "fillstyle": "none"}),
]

# 3 sedatives, daytime continuous-rate version.
PREDICTORS: list[tuple[str, str]] = [
    ("_prop_day_mcg_kg_min", "Daytime propofol\n(mcg/kg/min)"),
    ("_fenteq_day_mcg_hr",   "Daytime fentanyl eq\n(mcg/hr)"),
    ("_midazeq_day_mg_hr",   "Daytime midazolam eq\n(mg/hr)"),
]


def _filter(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    return df_all[
        df_all["outcome"].isin([o for o, _ in OUTCOMES])
        & (df_all["model_type"] == "gee")
        & df_all["spec"].isin([s for s, *_ in SPECS])
        & df_all["predictor"].isin([p for p, _ in PREDICTORS])
    ].copy()


def _render(df: pd.DataFrame) -> plt.Figure:
    sites = sorted(df["site"].unique().tolist()) if not df.empty else []
    n_sites = len(sites)

    n_rows, n_cols = len(PREDICTORS), len(OUTCOMES)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(13.0, 8.5),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    xlim = or_xlim_from_data(df)
    spec_jitter = (
        np.linspace(-0.08, 0.08, len(SPECS))
        if len(SPECS) > 1 else np.zeros(1)
    )

    for ri, (pred_key, _pred_label) in enumerate(PREDICTORS):
        for ci, (outcome_key, outcome_label) in enumerate(OUTCOMES):
            ax = axes[ri, ci]
            add_or_reference_line(ax)

            for si, s in enumerate(sites):
                y_base = (n_sites - 1) - si
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                for sj, (spec_key, _spec_label, marker_kw) in enumerate(SPECS):
                    cell = df[
                        (df["site"] == s)
                        & (df["spec"] == spec_key)
                        & (df["predictor"] == pred_key)
                        & (df["outcome"] == outcome_key)
                    ]
                    if cell.empty:
                        continue
                    r = cell.iloc[0]
                    if not (
                        np.isfinite(r["OR"])
                        and np.isfinite(r["OR_lo"])
                        and np.isfinite(r["OR_hi"])
                    ):
                        continue
                    y = y_base + spec_jitter[sj]
                    fillstyle = marker_kw.get("fillstyle", "full")
                    mfc = color if fillstyle == "full" else "white"
                    ax.errorbar(
                        r["OR"], y,
                        xerr=[[r["OR"] - r["OR_lo"]], [r["OR_hi"] - r["OR"]]],
                        fmt=marker_kw.get("marker", "o"),
                        color=color, ecolor=color,
                        markerfacecolor=mfc, markeredgecolor=color,
                        markersize=6, markeredgewidth=1.2,
                        capsize=2, elinewidth=1.0,
                    )

            ax.set_ylim(-0.6, n_sites - 0.4)
            apply_or_xaxis(ax, xlim)

            if ri == 0:
                ax.set_title(outcome_label, fontsize=11)
            if ri == n_rows - 1:
                ax.set_xlabel("Odds ratio (10th → 90th percentile shift)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(_pred_label, fontsize=10.5, labelpad=46)

    # Site y-tick labels — set ONCE; sharey propagates to every shared axis.
    axes[0, 0].set_yticks(list(range(n_sites)))
    axes[0, 0].set_yticklabels(
        [site_label(s) for s in reversed(sites)], fontsize=9,
    )

    # Legend: site colors + spec marker style.
    site_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markeredgecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markersize=8, label=site_label(s))
        for i, s in enumerate(sites)
    ]
    spec_handles = []
    for _spec_key, spec_label_, marker_kw in SPECS:
        fillstyle = marker_kw.get("fillstyle", "full")
        mfc = "0.4" if fillstyle == "full" else "white"
        spec_handles.append(
            plt.Line2D([0], [0], marker=marker_kw.get("marker", "o"),
                       linestyle="", markerfacecolor=mfc,
                       markeredgecolor="0.4", markersize=8, markeredgewidth=1.2,
                       label=spec_label_)
        )

    fig.legend(
        handles=site_handles + spec_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(site_handles) + len(spec_handles),
        frameon=False, fontsize=10,
    )

    fig.suptitle(
        "Effect of daytime dose rate (prior day) on next-day outcomes",
        fontsize=13, y=1.06,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    df_all = stack_per_site()
    if df_all.empty:
        return

    df_fig = _filter(df_all)
    have_specs = set(df_fig["spec"].unique().tolist())
    missing = [s for s, *_ in SPECS if s not in have_specs]
    if missing:
        print(
            f"  WARN: spec(s) {missing} absent from forest_data.csv. "
            "Re-run 08_models.py per site to refresh."
        )

    save_agg_csv(
        df_fig[["site", "outcome", "model_type", "spec", "predictor",
                "OR", "OR_lo", "OR_hi"]],
        "forest_daytime_cross_site",
    )
    fig = _render(df_fig)
    save_agg_fig(fig, "forest_daytime_cross_site")
    plt.close(fig)


if __name__ == "__main__":
    main()
