"""Cross-site forest plot: night–day diff dose-rate effects.

Stacks each site's `output_to_share/{site}/models/forest_data.csv` and
renders a 3×3 figure of per-site ORs (10→90 percentile shift) for the
3 manuscript outcomes × 3 night–day diff predictors × 2 weight-adjusted
specs.

Layout (3 rows × 3 cols):
  - Rows: sedative — Δ propofol, Δ fentanyl eq, Δ midazolam eq.
  - Cols: outcome — SBT eligible, SBT delivered (multiday), Successful extub.
  - Per-panel y-positions: one per site (top = first alphabetically).
  - Per-site: TWO dots side-by-side at tight ±0.08 jitter.
        - filled circle  → daydose_wt
        - open  circle   → clinical_wt
  - X-axis: log-scaled OR with major ticks at sparse round values
            (0.5/0.6/.../1.0/.../2.0) and minor gridlines every 0.05.
  - Reference dashed vertical line at OR=1.

Site colors via SITE_PALETTE (alphabetical mapping → consistent with
night_day_diff_mean_cross_site.py). Site labels respect ANONYMIZE_SITES
through site_label().

No pooled marker — same deferral convention as
`night_day_diff_mean_cross_site.py`.

Outputs:
  - output_to_agg/forest_night_day_cross_site.csv   (long-format stacked data)
  - output_to_agg/figures/forest_night_day_cross_site.png

Usage:
    uv run python code/agg/forest_night_day_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/forest_night_day_cross_site.py
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
# Each tuple: (outcome_key, model_type, display_label). Model type is
# per-outcome because successful extubation is a terminal one-time event
# (cluster-robust logit is the right spec; GEE's working-correlation
# structure doesn't apply) while SBT delivery is a daily-repeated binary
# event where GEE's exchangeable structure is meaningful.
OUTCOMES: list[tuple[str, str, str]] = [
    ("sbt_done_multiday_next_day", "gee",   "SBT delivered (multiday)"),
    ("success_extub_next_day",     "logit", "Successful extubation"),
]

# Spec → (display label, marker spec). `mfc=None` reuses face color = site color
# (filled). `mfc='white'` makes the open variant clearly hollow.
SPECS: list[tuple[str, str, dict]] = [
    ("daydose_wt",  "daydose + weight",  {"marker": "o", "fillstyle": "full"}),
    ("clinical_wt", "clinical + weight", {"marker": "o", "fillstyle": "none"}),
]

# 3 sedatives → 3 panel rows (one row per sedative).
PREDICTORS: list[tuple[str, str]] = [
    ("prop_dif_mcg_kg_min", "Δ propofol\n(mcg/kg/min)"),
    ("fenteq_dif_mcg_hr",   "Δ fentanyl eq\n(mcg/hr)"),
    ("midazeq_dif_mg_hr",   "Δ midazolam eq\n(mg/hr)"),
]


def _filter(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    # Match (outcome, model_type) pairs since each outcome carries its own
    # preferred model type (sbt_done_multiday → gee; success_extub → logit).
    allowed_om = {(o, mt) for o, mt, _ in OUTCOMES}
    om_pairs = list(zip(df_all["outcome"], df_all["model_type"]))
    keep = df_all[
        pd.Series([p in allowed_om for p in om_pairs], index=df_all.index)
        & df_all["spec"].isin([s for s, *_ in SPECS])
        & df_all["predictor"].isin([p for p, _ in PREDICTORS])
    ].copy()
    return keep


def _render(df: pd.DataFrame) -> plt.Figure:
    sites = sorted(df["site"].unique().tolist()) if not df.empty else []
    n_sites = len(sites)

    n_rows, n_cols = len(PREDICTORS), len(OUTCOMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10.0, 8.5),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    xlim = or_xlim_from_data(df)

    # Per-spec horizontal jitter inside one site row. ±0.08 keeps the two
    # spec dots visually paired (close together) but distinguishable.
    spec_jitter = (
        np.linspace(-0.08, 0.08, len(SPECS))
        if len(SPECS) > 1 else np.zeros(1)
    )

    for ri, (pred_key, _pred_label) in enumerate(PREDICTORS):
        for ci, (outcome_key, outcome_mt, outcome_label) in enumerate(OUTCOMES):
            ax = axes[ri, ci]
            add_or_reference_line(ax)

            # Site y-positions: site 0 at top, site 1 below, etc. We invert
            # later by setting reasonable y-ticks.
            for si, s in enumerate(sites):
                y_base = (n_sites - 1) - si
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                for sj, (spec_key, spec_label, marker_kw) in enumerate(SPECS):
                    cell = df[
                        (df["site"] == s)
                        & (df["spec"] == spec_key)
                        & (df["predictor"] == pred_key)
                        & (df["outcome"] == outcome_key)
                        & (df["model_type"] == outcome_mt)
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
                    # Filled marker: face = site color. Hollow: face = white,
                    # edge = site color (so the ring still encodes the site).
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
                # Sedative-row label sits as the y-axis label of column 0.
                # labelpad=46 leaves room for the site tick labels.
                ax.set_ylabel(_pred_label, fontsize=10.5, labelpad=46)

    # ── Y-tick labels (site names) — set ONCE; sharey propagates ──────────
    # Aligns with y_base = (n_sites - 1) - si: site index 0 (alphabetical
    # first) sits at the top, so the labels need to be reversed.
    axes[0, 0].set_yticks(list(range(n_sites)))
    axes[0, 0].set_yticklabels(
        [site_label(s) for s in reversed(sites)], fontsize=9,
    )

    # Legend: site colors (filled circles) + spec marker style (gray markers).
    site_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markeredgecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markersize=8, label=site_label(s))
        for i, s in enumerate(sites)
    ]
    spec_handles = []
    for spec_key, spec_label, marker_kw in SPECS:
        fillstyle = marker_kw.get("fillstyle", "full")
        mfc = "0.4" if fillstyle == "full" else "white"
        spec_handles.append(
            plt.Line2D([0], [0], marker=marker_kw.get("marker", "o"),
                       linestyle="", markerfacecolor=mfc,
                       markeredgecolor="0.4", markersize=8, markeredgewidth=1.2,
                       label=spec_label)
        )

    fig.legend(
        handles=site_handles + spec_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(site_handles) + len(spec_handles),
        frameon=False, fontsize=10,
    )

    fig.suptitle(
        "Effect of night-minus-day dose rate (prior day) on next-day outcomes",
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
        "forest_night_day_cross_site",
    )
    fig = _render(df_fig)
    save_agg_fig(fig, "forest_night_day_cross_site")
    plt.close(fig)


if __name__ == "__main__":
    main()
