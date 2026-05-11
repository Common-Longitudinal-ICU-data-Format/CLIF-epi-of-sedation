"""Cross-site forest plot: SBT-delivery algorithm sensitivity (internal).

Internal sensitivity figure comparing the 5 alternative SBT-delivery
detection variants currently fitted in 08_models.py. Layout puts each
variant in its own column so reviewers can scan whether the dose-effect
estimates are robust to the SBT-detection algorithm choice.

Layout (1 row × 5 cols):
  - Cols: SBT-delivery variant — sbt_done_prefix, _multiday, _subira,
          _abc, _v2 (the full set fitted under MODEL_CONFIGS at
          `08_models.py:380–385`).
  - Per-panel y-rows (6 total, top-to-bottom):
        Group A — night–day diff predictors:
          1. Δ propofol (mcg/kg/min)
          2. Δ fentanyl eq (mcg/hr)
          3. Δ midazolam eq (mg/hr)
        Group B — daytime continuous-rate predictors:
          4. Daytime propofol (mcg/kg/min)
          5. Daytime fentanyl eq (mcg/hr)
          6. Daytime midazolam eq (mg/hr)
        A faint horizontal divider sits between groups A and B.
  - Spec: `daydose_physio` only. (Single spec → no marker variation; color
          alone encodes site.)
  - X-axis: log-scaled OR with sparse adaptive major ticks + minor
          gridlines every 0.05.
  - Reference dashed vertical line at OR=1.

Outputs:
  - output_to_agg/forest_sbt_sensitivity_cross_site.csv
  - output_to_agg/figures/forest_sbt_sensitivity_cross_site.png

Usage:
    uv run python code/agg/forest_sbt_sensitivity_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/forest_sbt_sensitivity_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.forest_sbt_sensitivity")

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
from meta_analysis_cross_site import (  # noqa: E402
    PRESENTATIONS,
    build_pooled_table,
)


# ── Figure-fixed selectors ────────────────────────────────────────────────
# Columns: 5 SBT-delivery variants currently fitted in 08_models.py.
OUTCOMES: list[tuple[str, str]] = [
    ("sbt_done_prefix_next_day",   "sbt_done_prefix"),
    ("sbt_done_multiday_next_day", "sbt_done_multiday"),
    ("sbt_done_subira_next_day",   "sbt_done_subira"),
    ("sbt_done_abc_next_day",      "sbt_done_abc"),
    ("sbt_done_v2_next_day",       "sbt_done_v2"),
]

# Single spec for this figure — daydose_physio (renamed 2026-05-11 from
# clinical_wt; weight_kg now lives in BASELINE so the explicit _wt suffix
# is redundant).
SPEC: str = "daydose_physio"

# Presentation: per_unit OR (manuscript-standard fixed clinical units).
# This figure is ALREADY an audit view (algorithm-robustness check), so
# only one presentation is rendered; the per_unit scope keeps it on the
# same OR axis as the primary forests.
PRESENTATION: str = "per_unit"

# 6 predictor rows: top group = diffs, bottom group = daytime continuous.
# Short labels (no unit suffix — those are in the headline italic line).
# `group_idx` (0 = diff group, 1 = daytime group) drives the divider line.
PREDICTORS: list[tuple[str, str, int]] = [
    ("prop_dif_mcg_kg_min",   "Δ propofol",         0),
    ("fenteq_dif_mcg_hr",     "Δ fentanyl eq",      0),
    ("midazeq_dif_mg_hr",     "Δ midazolam eq",     0),
    ("_prop_day_mcg_kg_min",  "Daytime propofol",   1),
    ("_fenteq_day_mcg_hr",    "Daytime fentanyl eq", 1),
    ("_midazeq_day_mg_hr",    "Daytime midazolam eq", 1),
]


def _filter(df_all: pd.DataFrame) -> pd.DataFrame:
    if df_all.empty:
        return df_all
    return df_all[
        df_all["outcome"].isin([o for o, _ in OUTCOMES])
        & (df_all["model_type"] == "gee")
        & (df_all["spec"] == SPEC)
        & df_all["predictor"].isin([p for p, _, _ in PREDICTORS])
    ].copy()


def _render(per_site_df: pd.DataFrame, pooled_df: pd.DataFrame) -> plt.Figure:
    """Forest with per-site dots + Pooled diamond at each predictor y-row."""
    cols = PRESENTATIONS[PRESENTATION]
    sites = sorted(per_site_df["site"].unique().tolist()) if not per_site_df.empty else []
    n_sites = len(sites)
    n_preds = len(PREDICTORS)
    n_y_per_pred = n_sites + 1  # sites + Pooled diamond per predictor row

    n_cols = len(OUTCOMES)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(16.0, 8.0),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    # xlim from data INCLUDING pooled CIs (scope to figure-relevant rows).
    pres_pool = pooled_df[
        (pooled_df["presentation"] == PRESENTATION)
        & (pooled_df["spec"] == SPEC)
        & (pooled_df["model_type"] == "gee")
        & (pooled_df["outcome"].isin([o for o, _ in OUTCOMES]))
    ]
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
        xlim = (max(lo - pad, 0.6), min(hi + pad, 1.6))
    else:
        xlim = (0.85, 1.05)

    # 3-position jitter per predictor row: site 0 (top), site 1 (mid), Pooled (bottom).
    yj = (np.linspace(-0.20, 0.20, n_y_per_pred)
          if n_y_per_pred > 1 else np.zeros(1))
    # matplotlib y up = top, so REVERSE so site 0 (alphabetical first) is at TOP
    yj_top_to_bottom = yj[::-1]

    for ci, (outcome_key, outcome_label) in enumerate(OUTCOMES):
        ax = axes[ci]
        add_or_reference_line(ax)

        # Faint horizontal divider between diff group (top 3 rows) and
        # daytime group (bottom 3 rows).
        n_top = sum(1 for _, _, g in PREDICTORS if g == 0)
        divider_y = (n_preds - n_top) - 0.5
        ax.axhline(divider_y, color="0.7", linewidth=0.7, linestyle=":", zorder=1)

        for pi, (pred_key, _pred_label, _group) in enumerate(PREDICTORS):
            y_base = (n_preds - 1) - pi
            # Per-site dots at the top n_sites jitter slots.
            for si, s in enumerate(sites):
                cell = per_site_df[
                    (per_site_df["site"] == s)
                    & (per_site_df["predictor"] == pred_key)
                    & (per_site_df["outcome"] == outcome_key)
                ]
                if cell.empty:
                    continue
                r = cell.iloc[0]
                _or = r[cols["or"]]
                _lo = r[cols["or_lo"]]
                _hi = r[cols["or_hi"]]
                if not (np.isfinite(_or) and np.isfinite(_lo) and np.isfinite(_hi)):
                    continue
                y = y_base + yj_top_to_bottom[si]
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                ax.errorbar(
                    _or, y,
                    xerr=[[_or - _lo], [_hi - _or]],
                    fmt="o", color=color, ecolor=color,
                    markerfacecolor=color, markeredgecolor=color,
                    markersize=5, capsize=2, elinewidth=1.0,
                )
            # Pooled diamond at the bottom jitter slot.
            pooled_row = pres_pool[
                (pres_pool["outcome"] == outcome_key)
                & (pres_pool["predictor"] == pred_key)
            ]
            if not pooled_row.empty:
                pr = pooled_row.iloc[0]
                p_or = pr["pooled_or"]
                p_lo = pr["pooled_or_lo"]
                p_hi = pr["pooled_or_hi"]
                if np.isfinite(p_or) and np.isfinite(p_lo) and np.isfinite(p_hi):
                    y = y_base + yj_top_to_bottom[n_sites]  # last slot = Pooled
                    ax.errorbar(
                        p_or, y,
                        xerr=[[p_or - p_lo], [p_hi - p_or]],
                        fmt="D", color="black", ecolor="black",
                        markerfacecolor="black", markeredgecolor="black",
                        markersize=6, markeredgewidth=1.2,
                        capsize=2, elinewidth=1.0,
                    )

        ax.set_ylim(-0.6, n_preds - 0.4)
        apply_or_xaxis(ax, xlim)

        ax.set_title(outcome_label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Odds ratio (per fixed clinical unit, log scale)", fontsize=9)

    # Y-tick labels: predictor names (one per predictor row).
    pred_labels = [lbl for _, lbl, _ in reversed(PREDICTORS)]
    axes[0].set_yticks(list(range(n_preds)))
    axes[0].set_yticklabels(pred_labels, fontsize=10)

    # Legend: sites + Pooled.
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
    fig.legend(
        handles=site_handles + [pooled_handle],
        loc="upper center", bbox_to_anchor=(0.5, 1.005),
        ncol=len(site_handles) + 1, frameon=False, fontsize=10,
    )

    n_studies_max = pres_pool["n_studies"].max() if not pres_pool.empty else 0
    n_studies = int(n_studies_max) if pd.notna(n_studies_max) else 0
    add_salient_headline(
        fig,
        title="SBT-delivery algorithm robustness check",
        subtitle=(
            f"outcomes={{5 sbt_done_* algorithms}} · spec={SPEC} · "
            f"model_type=gee · presentation={PRESENTATION} · k={n_studies} sites"
        ),
        units_line=(
            "Stable estimates across the 5 columns indicate the SBT signal "
            "is detection-method-robust"
        ),
    )
    fig.tight_layout()
    # AUDIT badge — this figure is the algorithm-robustness audit view.
    add_audit_badge(fig, ha="left")
    return fig


def main() -> None:
    df_all = stack_per_site()
    if df_all.empty:
        return

    df_fig = _filter(df_all)
    if df_fig.empty:
        logger.info(
            f"  WARN: no rows match (spec={SPEC}, model_type=gee). "
            "Re-run 08_models.py per site to refresh models_coeffs.csv."
        )
        return

    pooled = build_pooled_table(stack_per_site(row_type="exposure"))

    save_agg_csv(
        df_fig[["site", "outcome", "model_type", "spec", "predictor",
                "or_per_unit", "or_per_unit_lo", "or_per_unit_hi",
                "or_p10_p90", "or_p10_p90_lo", "or_p10_p90_hi"]],
        "forest_sbt_sensitivity_cross_site",
    )
    fig = _render(df_fig, pooled)
    save_agg_fig(fig, "forest_sbt_sensitivity_cross_site")
    plt.close(fig)


if __name__ == "__main__":
    main()
