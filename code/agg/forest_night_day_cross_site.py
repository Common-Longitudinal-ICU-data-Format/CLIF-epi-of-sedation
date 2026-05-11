"""Cross-site forest plot: night–day diff dose-rate effects.

Stacks each site's `output_to_share/{site}/models/models_coeffs.csv` and
renders a 3×2 figure of per-site ORs (10→90 percentile shift) for the
2 manuscript outcomes × 3 night–day diff predictors × 2 manuscript
linear specs.

Layout (3 rows × 2 cols):
  - Rows: sedative — Δ propofol, Δ fentanyl eq, Δ midazolam eq.
  - Cols: outcome — SBT delivered (multiday), Successful extub.
  - Per-panel y-positions: one per site (top = first alphabetically).
  - Per-site: TWO dots side-by-side at tight ±0.08 jitter.
        - filled circle  → daydose
        - open  circle   → daydose_physio
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

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.forest_night_day")

sys.path.insert(0, str(Path(__file__).parent))

from _forest_helpers import (  # noqa: E402
    add_or_reference_line,
    apply_or_xaxis,
    stack_per_site,
)
from _shared import (  # noqa: E402
    SITE_PALETTE,
    add_audit_badge,
    add_salient_headline,
    save_agg_csv,
    save_agg_fig,
    site_label,
)
# Pull the pool primitives from meta_analysis_cross_site so this script
# can render the Pooled diamond without relying on the meta_pooled.csv
# file existing first (Makefile glob runs scripts alphabetically;
# forest_* lands before meta_analysis_*).
from meta_analysis_cross_site import (  # noqa: E402
    PRESENTATIONS,
    build_pooled_table,
)


# ── Figure-fixed selectors ────────────────────────────────────────────────
# Each tuple: (outcome_key, model_type, display_label). model_type matches
# the meta-analysis pool scope (logit_asym for extub, gee for SBT) so the
# per-site dots and the Pooled diamond reflect the SAME model-type fit.
OUTCOMES: list[tuple[str, str, str]] = [
    ("sbt_done_multiday_next_day", "gee",        "SBT delivered (multiday)"),
    ("success_extub_next_day",     "logit_asym", "Successful extubation"),
]

# Spec → (display label, marker spec). Filled = daydose, hollow = daydose_physio.
SPECS: list[tuple[str, str, dict]] = [
    ("daydose",        "daydose",        {"marker": "o", "fillstyle": "full"}),
    ("daydose_physio", "daydose_physio", {"marker": "o", "fillstyle": "none"}),
]

# 3 sedatives → 3 panel rows. SHORT labels — units are in the headline
# italic line, so no "(per X mcg/kg/min)" suffix here (avoids the long-
# label overlap problem the prior renderer hit).
PREDICTORS: list[tuple[str, str]] = [
    ("prop_dif_mcg_kg_min", "Δ propofol"),
    ("fenteq_dif_mcg_hr",   "Δ fentanyl eq"),
    ("midazeq_dif_mg_hr",   "Δ midazolam eq"),
]

# Per-presentation labels for the headline italic line + x-axis label.
PRES_UNITS_LINE: dict[str, str] = {
    "per_unit": "OR per +10 mcg/kg/min · +25 mcg/hr · +1 mg/hr (prop / fent eq / midaz eq)",
    "p10_p90":  "OR per (10th→90th percentile shift); shift sizes vary by site",
}
PRES_AXIS_LABEL: dict[str, str] = {
    "per_unit": "Odds ratio (per fixed clinical unit, log scale)",
    "p10_p90":  "Odds ratio (10th→90th percentile shift, log scale)",
}
# Tight presentation-specific xlim clamp — the prior universal floor of
# 0.05 let one wide CI balloon the visible range. Manuscript-standard
# ranges per presentation:
PRES_XLIM_BOUNDS: dict[str, tuple[float, float]] = {
    "per_unit": (0.6, 1.6),
    "p10_p90":  (0.4, 2.5),
}


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


def _render(
    per_site_df: pd.DataFrame,
    pooled_df: pd.DataFrame,
    presentation: str,
) -> plt.Figure:
    """Forest with per-site dots + Pooled diamond row (presentation = per_unit / p10_p90)."""
    cols = PRESENTATIONS[presentation]
    sites = sorted(per_site_df["site"].unique().tolist()) if not per_site_df.empty else []
    n_sites = len(sites)
    n_y = n_sites + 1  # +1 for Pooled row at the bottom

    n_rows, n_cols = len(PREDICTORS), len(OUTCOMES)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10.0, 9.5),
        sharex=True, sharey=True,
    )
    axes = np.atleast_2d(axes)

    # xlim from data INCLUDING pooled CIs, scoped to this figure's
    # outcomes / specs / predictors only (the prior pooled-renderer bug
    # was using unfiltered data and ballooning the visible range).
    pres_pool = pooled_df[pooled_df["presentation"] == presentation]
    los = np.concatenate([
        per_site_df[cols["or_lo"]].to_numpy(dtype=float),
        pres_pool["pooled_or_lo"].to_numpy(dtype=float),
    ])
    his = np.concatenate([
        per_site_df[cols["or_hi"]].to_numpy(dtype=float),
        pres_pool["pooled_or_hi"].to_numpy(dtype=float),
    ])
    los = los[np.isfinite(los)]
    his = his[np.isfinite(his)]
    if len(los) and len(his):
        lo = float(min(los.min(), 1.0))
        hi = float(max(his.max(), 1.0))
        span = hi - lo
        pad = max(span * 0.08, 0.005)
        floor, ceil = PRES_XLIM_BOUNDS[presentation]
        xlim = (max(lo - pad, floor), min(hi + pad, ceil))
    else:
        xlim = (0.85, 1.05)

    # ±0.08 jitter for the 2 specs WITHIN each y-row (matches the prior convention).
    spec_jitter = (
        np.linspace(-0.08, 0.08, len(SPECS))
        if len(SPECS) > 1 else np.zeros(1)
    )

    pooled_y = 0  # bottom of panel
    site_y = {s: (n_sites - i) for i, s in enumerate(sites)}  # site 0 at top

    for ri, (pred_key, _pred_label) in enumerate(PREDICTORS):
        for ci, (outcome_key, outcome_mt, outcome_label) in enumerate(OUTCOMES):
            ax = axes[ri, ci]
            add_or_reference_line(ax)

            # Per-site dots (one per site × spec, jittered).
            for si, s in enumerate(sites):
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                for sj, (spec_key, _spec_label, marker_kw) in enumerate(SPECS):
                    cell = per_site_df[
                        (per_site_df["site"] == s)
                        & (per_site_df["spec"] == spec_key)
                        & (per_site_df["predictor"] == pred_key)
                        & (per_site_df["outcome"] == outcome_key)
                        & (per_site_df["model_type"] == outcome_mt)
                    ]
                    if cell.empty:
                        continue
                    r = cell.iloc[0]
                    _or = r[cols["or"]]
                    _lo = r[cols["or_lo"]]
                    _hi = r[cols["or_hi"]]
                    if not (np.isfinite(_or) and np.isfinite(_lo) and np.isfinite(_hi)):
                        continue
                    fillstyle = marker_kw.get("fillstyle", "full")
                    mfc = color if fillstyle == "full" else "white"
                    y = site_y[s] + spec_jitter[sj]
                    ax.errorbar(
                        _or, y,
                        xerr=[[_or - _lo], [_hi - _or]],
                        fmt=marker_kw.get("marker", "o"),
                        color=color, ecolor=color,
                        markerfacecolor=mfc, markeredgecolor=color,
                        markersize=6, markeredgewidth=1.2,
                        capsize=2, elinewidth=1.0,
                    )

            # Pooled diamonds (one per spec, jittered) at the bottom row.
            for sj, (spec_key, _spec_label, marker_kw) in enumerate(SPECS):
                pooled_row = pres_pool[
                    (pres_pool["outcome"] == outcome_key)
                    & (pres_pool["model_type"] == outcome_mt)
                    & (pres_pool["spec"] == spec_key)
                    & (pres_pool["predictor"] == pred_key)
                ]
                if pooled_row.empty:
                    continue
                pr = pooled_row.iloc[0]
                p_or = pr["pooled_or"]
                p_lo = pr["pooled_or_lo"]
                p_hi = pr["pooled_or_hi"]
                if not (np.isfinite(p_or) and np.isfinite(p_lo) and np.isfinite(p_hi)):
                    continue
                fillstyle = marker_kw.get("fillstyle", "full")
                mfc = "black" if fillstyle == "full" else "white"
                y = pooled_y + spec_jitter[sj]
                ax.errorbar(
                    p_or, y,
                    xerr=[[p_or - p_lo], [p_hi - p_or]],
                    fmt="D", color="black", ecolor="black",
                    markerfacecolor=mfc, markeredgecolor="black",
                    markersize=8, markeredgewidth=1.3,
                    capsize=2, elinewidth=1.1,
                )

            ax.set_ylim(-0.6, n_y - 0.4)
            apply_or_xaxis(ax, xlim)

            if ri == 0:
                ax.set_title(outcome_label, fontsize=11)
            if ri == n_rows - 1:
                ax.set_xlabel(PRES_AXIS_LABEL[presentation], fontsize=9)
            if ci == 0:
                # Short label (no units) — keeps row labels from colliding.
                ax.set_ylabel(_pred_label, fontsize=11, labelpad=18)

    # Y-tick labels: matplotlib y=0 → bottom (Pooled). Index `i` of
    # set_yticklabels corresponds to y=i, so labels go bottom-up:
    # ["Pooled", site_label(sites[-1]), ..., site_label(sites[0])].
    yticklabels = ["Pooled"] + [site_label(s) for s in reversed(sites)]
    axes[0, 0].set_yticks(list(range(n_y)))
    axes[0, 0].set_yticklabels(yticklabels, fontsize=9)

    # Legend: sites + Pooled marker + spec marker style.
    site_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markeredgecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markersize=8, label=site_label(s))
        for i, s in enumerate(sites)
    ]
    pooled_handle = plt.Line2D(
        [0], [0], marker="D", linestyle="",
        markerfacecolor="black", markeredgecolor="black",
        markersize=9, label="Pooled (DL random-effects)",
    )
    spec_handles = []
    for spec_key, spec_label_, marker_kw in SPECS:
        fillstyle = marker_kw.get("fillstyle", "full")
        mfc = "0.4" if fillstyle == "full" else "white"
        spec_handles.append(
            plt.Line2D([0], [0], marker=marker_kw.get("marker", "o"),
                       linestyle="", markerfacecolor=mfc,
                       markeredgecolor="0.4", markersize=8, markeredgewidth=1.2,
                       label=spec_label_)
        )
    fig.legend(
        handles=site_handles + [pooled_handle] + spec_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=len(site_handles) + 1 + len(spec_handles),
        frameon=False, fontsize=10,
    )

    # Three-tier salient headline (mirrors sed_dose_by_hr_of_day pattern).
    n_studies_max = pres_pool["n_studies"].max() if not pres_pool.empty else 0
    n_studies = int(n_studies_max) if pd.notna(n_studies_max) else 0
    add_salient_headline(
        fig,
        title="Effect of night-vs-day dose-rate difference on next-day liberation",
        subtitle=(
            f"outcomes={{SBT delivered (multiday), Successful extub}} · "
            f"specs={{daydose, daydose_physio}} · "
            f"presentation={presentation} · k={n_studies} sites"
        ),
        units_line=PRES_UNITS_LINE[presentation],
    )

    fig.tight_layout()

    # AUDIT badge for the per-percentile sibling (per-unit is primary).
    if presentation == "p10_p90":
        add_audit_badge(fig, ha="left")

    return fig


def main() -> None:
    df_all = stack_per_site()
    if df_all.empty:
        return

    df_fig = _filter(df_all)
    if df_fig.empty:
        return
    have_specs = set(df_fig["spec"].unique().tolist())
    missing = [s for s, *_ in SPECS if s not in have_specs]
    if missing:
        logger.info(
            f"  WARN: spec(s) {missing} absent from models_coeffs.csv. "
            "Re-run 08_models.py per site to refresh."
        )

    # Compute the cross-site DL pool from per-site coefficients (uses
    # POOL_OUTCOMES / POOL_SPECS / POOL_PREDICTORS module-globals from
    # meta_analysis_cross_site so the pooled rows include all manuscript
    # outcomes; we just look up the ones this figure needs).
    pooled = build_pooled_table(stack_per_site(row_type="exposure"))

    save_agg_csv(
        df_fig[["site", "outcome", "model_type", "spec", "predictor",
                "or_per_unit", "or_per_unit_lo", "or_per_unit_hi",
                "or_p10_p90", "or_p10_p90_lo", "or_p10_p90_hi"]],
        "forest_night_day_cross_site",
    )

    # Two PNGs per figure family — primary (per_unit) + audit (p10_p90).
    for presentation, suffix in [("per_unit", ""), ("p10_p90", "_audit")]:
        fig = _render(df_fig, pooled, presentation)
        save_agg_fig(fig, f"forest_night_day_cross_site{suffix}")
        plt.close(fig)


if __name__ == "__main__":
    main()
