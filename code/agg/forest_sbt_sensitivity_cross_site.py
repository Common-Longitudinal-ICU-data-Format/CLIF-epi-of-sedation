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
  - Spec: `clinical_wt` only. (Single spec → no marker variation; color
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
# Columns: 5 SBT-delivery variants currently fitted in 08_models.py.
OUTCOMES: list[tuple[str, str]] = [
    ("sbt_done_prefix_next_day",   "sbt_done_prefix"),
    ("sbt_done_multiday_next_day", "sbt_done_multiday"),
    ("sbt_done_subira_next_day",   "sbt_done_subira"),
    ("sbt_done_abc_next_day",      "sbt_done_abc"),
    ("sbt_done_v2_next_day",       "sbt_done_v2"),
]

# Single spec for this figure — clinical+weight only.
SPEC: str = "clinical_wt"

# 6 predictor rows: top group = diffs, bottom group = daytime continuous.
# `group_idx` (0 = diff group, 1 = daytime group) drives the divider line.
PREDICTORS: list[tuple[str, str, int]] = [
    ("prop_dif_mcg_kg_min",   "Δ propofol\n(mcg/kg/min)",   0),
    ("fenteq_dif_mcg_hr",     "Δ fentanyl eq\n(mcg/hr)",    0),
    ("midazeq_dif_mg_hr",     "Δ midazolam eq\n(mg/hr)",    0),
    ("_prop_day_mcg_kg_min",  "Daytime propofol\n(mcg/kg/min)", 1),
    ("_fenteq_day_mcg_hr",    "Daytime fentanyl eq\n(mcg/hr)",  1),
    ("_midazeq_day_mg_hr",    "Daytime midazolam eq\n(mg/hr)",  1),
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


def _render(df: pd.DataFrame) -> plt.Figure:
    sites = sorted(df["site"].unique().tolist()) if not df.empty else []
    n_sites = len(sites)
    n_preds = len(PREDICTORS)

    n_cols = len(OUTCOMES)
    fig, axes = plt.subplots(
        1, n_cols, figsize=(16.0, 7.5),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    xlim = or_xlim_from_data(df)
    # Per-site jitter inside one predictor y-row. ±0.18 (matches the
    # per-site forest plot's spec-jitter convention) so the 2 site dots
    # are clearly separated within a single predictor row.
    site_jitter = (
        np.linspace(-0.18, 0.18, n_sites)
        if n_sites > 1 else np.zeros(1)
    )

    for ci, (outcome_key, outcome_label) in enumerate(OUTCOMES):
        ax = axes[ci]
        add_or_reference_line(ax)

        # Faint horizontal divider between the diff group (top) and the
        # daytime group (bottom). Diff group occupies y_base = n_preds-1
        # down to n_preds-3 (3 diff rows); daytime group below.
        # Divider y-position: midpoint between the lowest diff row and
        # the highest daytime row.
        n_top = sum(1 for _, _, g in PREDICTORS if g == 0)
        divider_y = (n_preds - n_top) - 0.5
        ax.axhline(divider_y, color="0.7", linewidth=0.7, linestyle=":", zorder=1)

        for pi, (pred_key, _pred_label, _group) in enumerate(PREDICTORS):
            y_base = (n_preds - 1) - pi
            for si, s in enumerate(sites):
                cell = df[
                    (df["site"] == s)
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
                y = y_base + site_jitter[si]
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                ax.errorbar(
                    r["OR"], y,
                    xerr=[[r["OR"] - r["OR_lo"]], [r["OR_hi"] - r["OR"]]],
                    fmt="o", color=color, ecolor=color,
                    markerfacecolor=color, markeredgecolor=color,
                    markersize=5, capsize=2, elinewidth=1.0,
                )

        ax.set_ylim(-0.6, n_preds - 0.4)
        apply_or_xaxis(ax, xlim)

        ax.set_title(outcome_label, fontsize=11)
        ax.set_xlabel("Odds ratio (10th → 90th percentile shift)", fontsize=9)

    # Y-tick labels (predictor names) — set ONCE; sharey propagates.
    # Predictor labels already carry "Δ" (diff group) / "Daytime" prefixes
    # which, combined with the dotted horizontal divider rendered per axis,
    # makes the grouping self-evident — no separate rotated group labels
    # needed (they'd overlap the predictor text at this panel width).
    pred_labels = [lbl for _, lbl, _ in reversed(PREDICTORS)]
    axes[0].set_yticks(list(range(n_preds)))
    axes[0].set_yticklabels(pred_labels, fontsize=9)

    # Site legend.
    site_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markeredgecolor=SITE_PALETTE[i % len(SITE_PALETTE)],
                   markersize=8, label=site_label(s))
        for i, s in enumerate(sites)
    ]
    fig.legend(
        handles=site_handles,
        loc="upper center", bbox_to_anchor=(0.5, 1.02),
        ncol=len(site_handles), frameon=False, fontsize=10,
        title="Site",
    )

    fig.suptitle(
        "Cross-site sensitivity: SBT-delivery algorithm "
        f"variants ({SPEC} spec)",
        fontsize=13, y=1.06,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    df_all = stack_per_site()
    if df_all.empty:
        return

    df_fig = _filter(df_all)
    if df_fig.empty:
        print(
            f"  WARN: no rows match (spec={SPEC}, model_type=gee). "
            "Re-run 08_models.py per site to refresh forest_data.csv."
        )
        return

    save_agg_csv(
        df_fig[["site", "outcome", "model_type", "spec", "predictor",
                "OR", "OR_lo", "OR_hi"]],
        "forest_sbt_sensitivity_cross_site",
    )
    fig = _render(df_fig)
    save_agg_fig(fig, "forest_sbt_sensitivity_cross_site")
    plt.close(fig)


if __name__ == "__main__":
    main()
