"""Cross-site overlay of RCS marginal-effect curves.

Stacks each site's `output_to_share/{site}/models/marginal_effects_grid.csv`
and renders one 2×3 figure per (outcome × model_type × spec) combo in
`MANUSCRIPT_TARGETS`, with per-site predicted-probability curves overlaid.

Layout (per figure, mirrors the per-site marginal-effects PNG):
  - 2 rows × 3 cols
  - Row 0: daytime continuous rates (propofol / fentanyl eq / midazolam eq)
  - Row 1: night–day diff rates (same 3 drugs)
  - Each panel: per-site curve (color via SITE_PALETTE) with a thin CI
    ribbon (alpha=0.15) so multiple sites stay legible.
  - Y-axis [0, 1]; X-axis = each site's own 2.5–97.5 percentile range
    (curves can be different lengths if sites have different cohort
    distributions — what's plotted is each site's faithful in-range curve).
  - Reference style: ggplot-like (gray bg, white grid, no spines), matching
    `_ggplot_ax` in `code/08_models.py`.

The `MANUSCRIPT_TARGETS` list controls which figures get rendered. Edit
to expand or trim. The script also saves a single companion long-format
CSV with every site's curve (for inspection / supplemental exhibits).

Outputs:
  - output_to_agg/marginal_effects_rcs_cross_site.csv
  - output_to_agg/figures/marginal_effects_{outcome_short}_{model_type}_{spec}_cross_site.png
    (one PNG per element of `MANUSCRIPT_TARGETS` × specs in the data)

Usage:
    uv run python code/agg/marginal_effects_rcs_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/marginal_effects_rcs_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    SITE_PALETTE,
    list_sites,
    load_site_marginal_effects,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# ── Manuscript scope ──────────────────────────────────────────────────────
# Render one cross-site figure for each (outcome, model_type) in this list,
# crossed with each spec found in the data. With 3 outcomes × 1 model_type ×
# 2 RCS specs = 6 PNGs by default. Add v2 outcomes / logit if needed.
MANUSCRIPT_TARGETS: list[tuple[str, str]] = [
    ("sbt_elig_next_day",          "gee"),
    ("sbt_done_multiday_next_day", "gee"),
    ("success_extub_next_day",     "gee"),
]

# Y-axis label per outcome (mirrors `Y_LABEL` in `code/08_models.py`).
Y_LABEL: dict[str, str] = {
    "sbt_elig_next_day":          "Probability of SBT Eligibility",
    "sbt_done_multiday_next_day": "Probability of Passing SBT (multiday)",
    "sbt_done_v2_next_day":       "Probability of Passing SBT (v2)",
    "success_extub_next_day":     "Probability of Successful Extubation",
    "success_extub_v2_next_day":  "Probability of Successful Extubation (v2)",
}

# Filename short form per outcome (mirrors `OUTCOME_SHORT` in `code/08_models.py`).
OUTCOME_SHORT: dict[str, str] = {
    "sbt_elig_next_day":          "sbt_elig",
    "sbt_done_multiday_next_day": "sbt_done_multiday",
    "sbt_done_v2_next_day":       "sbt_done_v2",
    "success_extub_next_day":     "success_extub",
    "success_extub_v2_next_day":  "success_extub_v2",
}


def _ggplot_ax(ax) -> None:
    """Style an axes ggplot-style — matches `_ggplot_ax` in 08_models.py."""
    ax.set_facecolor("#ebebeb")
    ax.grid(color="white", linewidth=0.8, which="major", zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors="#4d4d4d", labelsize=8)
    ax.xaxis.label.set_color("#4d4d4d")
    ax.yaxis.label.set_color("#4d4d4d")


def _stack_per_site() -> pd.DataFrame:
    """Concat each discovered site's marginal_effects_grid.csv with a `site` column."""
    sites = list_sites()
    if not sites:
        print("No sites found under output_to_share/. Nothing to plot.")
        return pd.DataFrame()
    print(f"Discovered sites: {sites}")
    frames: list[pd.DataFrame] = []
    for s in sites:
        try:
            df = load_site_marginal_effects(s)
        except FileNotFoundError:
            print(
                f"  SKIP {s}: marginal_effects_grid.csv missing — "
                "re-run 08_models.py for this site."
            )
            continue
        df["site"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _render_one(df: pd.DataFrame, outcome: str, model_type: str, spec: str) -> plt.Figure:
    """2×3 figure with per-site curves overlaid for one (outcome × mt × spec)."""
    cell = df[
        (df["outcome"] == outcome)
        & (df["model_type"] == model_type)
        & (df["spec"] == spec)
    ]
    sites = sorted(cell["site"].unique().tolist())

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.5))
    fig.patch.set_facecolor("white")

    panel_letters = ["A", "B", "C", "D", "E", "F"]
    pidx = 0

    for row_idx in range(2):
        for col_idx in range(3):
            ax = axes[row_idx, col_idx]
            _ggplot_ax(ax)

            panel = cell[
                (cell["panel_row"] == row_idx) & (cell["panel_col"] == col_idx)
            ]
            if panel.empty:
                pidx += 1
                continue
            xlabel = panel["xlabel"].iloc[0]

            for si, s in enumerate(sites):
                site_curve = panel[panel["site"] == s].sort_values("x_actual")
                if site_curve.empty:
                    continue
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                x = site_curve["x_actual"].to_numpy()
                p = site_curve["prob"].to_numpy()
                lo = site_curve["ci_lo"].to_numpy()
                hi = site_curve["ci_hi"].to_numpy()
                ax.fill_between(x, lo, hi, color=color, alpha=0.15, zorder=2)
                # Legend label only on the first panel so the figure-level
                # legend has one entry per site.
                ax.plot(
                    x, p, color=color, linewidth=1.6, zorder=3,
                    label=site_label(s) if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(
                Y_LABEL.get(outcome, "Predicted Probability"), fontsize=9,
            )
            ax.set_ylim(0, 1)
            ax.text(
                -0.12, 1.08, panel_letters[pidx],
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
            )
            pidx += 1

    # Single figure-level legend (one entry per site).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.03),
            ncol=len(handles), frameon=False, fontsize=10, title="Site",
        )

    fig.suptitle(
        f"{Y_LABEL.get(outcome, 'Probability')} by Sedative Exposure\n"
        f"({spec} spec, {model_type.upper()})",
        fontsize=12, y=1.07,
    )
    fig.tight_layout()
    return fig


def main() -> None:
    df_all = _stack_per_site()
    if df_all.empty:
        return

    save_agg_csv(df_all, "marginal_effects_rcs_cross_site")

    # Filter to manuscript targets, then render once per (outcome × mt × spec)
    # combo present in the data.
    target_set = set(MANUSCRIPT_TARGETS)
    df_in_scope = df_all[
        df_all.apply(lambda r: (r["outcome"], r["model_type"]) in target_set, axis=1)
    ]
    if df_in_scope.empty:
        print(
            f"  WARN: no rows match MANUSCRIPT_TARGETS={MANUSCRIPT_TARGETS}. "
            "Re-run 08_models.py per site to populate marginal_effects_grid.csv."
        )
        return

    keys = (
        df_in_scope[["outcome", "model_type", "spec"]]
        .drop_duplicates()
        .sort_values(["outcome", "model_type", "spec"])
        .itertuples(index=False, name=None)
    )
    for outcome, model_type, spec in keys:
        outcome_short = OUTCOME_SHORT.get(outcome, outcome)
        fig = _render_one(df_all, outcome, model_type, spec)
        save_agg_fig(
            fig,
            f"marginal_effects_{outcome_short}_{model_type}_{spec}_cross_site",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
