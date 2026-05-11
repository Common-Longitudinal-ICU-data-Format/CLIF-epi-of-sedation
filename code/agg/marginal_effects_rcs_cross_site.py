"""Cross-site overlay of RCS marginal-effect curves.

Stacks each site's `output_to_share/{site}/models/marginal_effects_grid.csv`
and renders two complementary figure families, distinguished by file-name
suffix:

  - `*_primary_cross_site.png` — manuscript / presentation view.
    ONE figure per spec, combining ALL manuscript outcomes (rows) ×
    3 sedatives (cols) in a single compact panel grid. Diff predictors
    only — the within-patient day-vs-night sedation rate variation
    that carries the primary scientific story. With 2 manuscript
    outcomes × 2 RCS specs = 2 PNGs total.

  - `*_audit_cross_site.png` — verification view. 2 rows × 3 cols
    per (outcome × model_type × spec) adding the daytime continuous-rate
    predictors above the diff row. Lets us audit whether the primary
    diff effect is being driven by absolute dose level rather than the
    day-vs-night contrast. 2 outcomes × 2 specs = 4 PNGs.

See `.dev/CLAUDE.md` "Cross-site agg figures: primary vs audit versions"
for the project-wide convention.

Per-outcome model_type: `MANUSCRIPT_TARGETS` carries `(outcome, model_type)`
tuples. SBT delivery uses GEE (daily-repeated binary event); successful
extubation uses cluster-robust logit (terminal one-time event). The two
families are unified in the primary figure since model_type is implicit
in the outcome.

Each panel: per-site curve (color via SITE_PALETTE) with a thin CI
ribbon (alpha=0.15). Y-axis is clipped to [0, 0.5] so any decline is
visually salient. X-axis is each site's own 2.5–97.5 percentile range.
Style: ggplot-like, matching `_ggplot_ax` in `code/08_models.py`.

Outputs:
  - output_to_agg/marginal_effects_rcs_cross_site.csv (one shared CSV)
  - output_to_agg/figures/marginal_effects_{spec}_primary_cross_site.png
    (combined-outcomes primary; 2 PNGs at 2 specs)
  - output_to_agg/figures/marginal_effects_{outcome_short}_{model_type}_{spec}_audit_cross_site.png
    (one per outcome × spec; 4 PNGs)

Usage:
    uv run python code/agg/marginal_effects_rcs_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/marginal_effects_rcs_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.marginal_effects_rcs")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    SITE_PALETTE,
    add_audit_badge,
    add_salient_headline,
    list_sites,
    load_site_marginal_effects,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# Per-outcome bold-title color for the audit branch (single-outcome figs).
# Mirrors the convention from sed_dose_by_hr_of_day's drug-title-color
# override — distinct dark hue per outcome so a reader can flip between
# audit PNGs and instantly tell which outcome they're looking at.
_OUTCOME_TITLE_COLORS: dict[str, str] = {
    "sbt_done_multiday_next_day": "#1f4e79",  # deep blue
    "sbt_done_prefix_next_day":   "#1f4e79",
    "sbt_done_subira_next_day":   "#1f4e79",
    "sbt_done_abc_next_day":      "#1f4e79",
    "sbt_done_v2_next_day":       "#1f4e79",
    "success_extub_next_day":     "#a61d24",  # firebrick
    "success_extub_v2_next_day":  "#a61d24",
}


# ── Manuscript scope ──────────────────────────────────────────────────────
# Two manuscript base specs (linear-spec names; the RCS variants are
# `*_rcs_diff` or `*_rcs_full`).
# Renamed 2026-05-11: weight_kg moved into BASELINE so the `_wt` suffix
# is gone; `clinical_wt` → `daydose_physio` since the spec adds
# physiologic markers (pH/PF/NEE) on top of daydose.
BASE_SPECS: list[str] = ["daydose", "daydose_physio"]

# Three versions of the primary marginal-effects figure. Each version
# specifies which RCS spec suffix to use and which logit fit to pull
# the extubation curves from. SBT delivered always uses `gee`.
#
# - v1_main: cr() on all 6 vars (`_full`) + asymptotic-SE logit. The
#   manuscript / presentation view; both rows render with clean
#   thin CI ribbons.
# - v2_sa_clusterse: cr() on diff only (`_diff`) + cluster-robust
#   logit. SA illustrating the cluster-robust failure mode — extub
#   row will show degenerate full-y-axis CI ribbons by design.
# - v3_sa_asymse: cr() on diff only (`_diff`) + asymptotic-SE logit.
#   SA documenting that the parsimony argument is orthogonal to the
#   SE choice; CI ribbons clean like v1.
VERSIONS: list[tuple[str, dict]] = [
    ("v1_main",
     {"spec_suffix": "full", "extub_mt": "logit_asym",
      "short_desc": "full RCS, asymptotic SE"}),
    ("v2_sa_clusterse",
     {"spec_suffix": "diff", "extub_mt": "logit",
      "short_desc": "RCS diff, cluster-robust SE"}),
    ("v3_sa_asymse",
     {"spec_suffix": "diff", "extub_mt": "logit_asym",
      "short_desc": "RCS diff, asymptotic SE"}),
]

# Outcomes always rendered in this order (top to bottom of the primary
# figure). SBT delivered always uses gee; extub model_type comes from
# the version config.
OUTCOME_ORDER = ["sbt_done_multiday_next_day", "success_extub_next_day"]
SBT_MT = "gee"


def _rows_for_version(cfg: dict) -> list[tuple[str, str]]:
    """Return [(outcome, model_type), ...] for the version's primary rows."""
    return [
        ("sbt_done_multiday_next_day", SBT_MT),
        ("success_extub_next_day",     cfg["extub_mt"]),
    ]

# Source-grid panel row indices. The per-site grid stores `panel_row`
# (0 = daytime continuous rates, 1 = night–day diffs). Both render
# functions below pull from these.
_DAYTIME_ROW = 0
_DIFF_ROW = 1

# Long-form y-axis label per outcome. Used by the audit figure
# (single-outcome) where the suptitle already names the outcome and
# the y-axis can carry the metric. The primary figure uses a generic
# "Predicted probability" axis label combined with `OUTCOME_HEADER`
# bold row titles for clarity.
Y_LABEL: dict[str, str] = {
    "sbt_done_multiday_next_day": "Probability of Passing SBT (multiday)",
    "sbt_done_v2_next_day":       "Probability of Passing SBT (v2)",
    "success_extub_next_day":     "Probability of Successful Extubation",
    "success_extub_v2_next_day":  "Probability of Successful Extubation (v2)",
}

# Short header label per outcome for the bold row title in primary
# figures. Kept compact so the layout doesn't squeeze panel content.
OUTCOME_HEADER: dict[str, str] = {
    "sbt_done_multiday_next_day": "SBT Delivered (multiday)",
    "success_extub_next_day":     "Successful Extubation",
}

# Filename short form per outcome (mirrors `OUTCOME_SHORT` in `code/08_models.py`).
OUTCOME_SHORT: dict[str, str] = {
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
        logger.info("No sites found under output_to_share/. Nothing to plot.")
        return pd.DataFrame()
    logger.info(f"Discovered sites: {sites}")
    frames: list[pd.DataFrame] = []
    for s in sites:
        try:
            df = load_site_marginal_effects(s)
        except FileNotFoundError:
            logger.info(
                f"  SKIP {s}: marginal_effects_grid.csv missing — "
                "re-run 08_models.py for this site."
            )
            continue
        df["site"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _render_primary_combined(
    df: pd.DataFrame,
    base_spec: str,
    version_name: str,
    version_cfg: dict,
) -> plt.Figure:
    """Combined primary figure for one (base_spec × version) combo.

    Layout: 2 outcome rows × 3 sedative cols, diff predictors only.
    Each outcome row's center column carries a bold row header (the
    outcome name) so the structure is scannable at a glance.

    The RCS spec resolved internally as `f"{base_spec}_rcs_{spec_suffix}"`
    where `spec_suffix` comes from `version_cfg`. SBT delivered always
    uses gee; extub model_type comes from `version_cfg["extub_mt"]`.

    Designed as a single-figure presentation artifact carrying the full
    manuscript narrative for one spec × version combination.
    """
    sites = sorted(df["site"].unique().tolist())
    spec = f"{base_spec}_rcs_{version_cfg['spec_suffix']}"
    rows = _rows_for_version(version_cfg)

    n_outcomes = len(rows)
    # Per-row figure height ~3.5 in + ~1 in for title/legend chrome +
    # ~0.5 in extra so the bold row headers don't crowd the panels.
    fig_h = 3.5 + 3.5 * n_outcomes
    fig, axes = plt.subplots(
        n_outcomes, 3, figsize=(13.0, fig_h), squeeze=False,
    )
    fig.patch.set_facecolor("white")

    panel_letters_full = ["A", "B", "C", "D", "E", "F"]
    pidx = 0

    for ri, (outcome, model_type) in enumerate(rows):
        cell = df[
            (df["outcome"] == outcome)
            & (df["model_type"] == model_type)
            & (df["spec"] == spec)
            & (df["panel_row"] == _DIFF_ROW)
        ]
        for col_idx in range(3):
            ax = axes[ri, col_idx]
            _ggplot_ax(ax)

            panel = cell[cell["panel_col"] == col_idx]
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
                ax.fill_between(x, lo, hi, color=color, alpha=0.18, zorder=2)
                # Legend label only on the top-left panel — figure-level
                # legend gets one entry per site.
                ax.plot(
                    x, p, color=color, linewidth=1.8, zorder=3,
                    label=site_label(s) if (ri == 0 and col_idx == 0) else None,
                )

            ax.set_xlabel(xlabel, fontsize=9)
            # Generic y-axis label — the bold row header (set below)
            # carries the outcome identification.
            ax.set_ylabel("Predicted probability", fontsize=9)
            ax.set_ylim(0, 0.5)
            ax.text(
                -0.12, 1.08, panel_letters_full[pidx],
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
            )
            pidx += 1

        # Bold row header on the center panel of each outcome row.
        # Reviewers can identify which outcome any row covers without
        # parsing y-axis labels.
        axes[ri, 1].set_title(
            OUTCOME_HEADER.get(outcome, outcome),
            fontsize=14, fontweight="bold", pad=18,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.01),
            ncol=len(handles), frameon=False, fontsize=10, title="Site",
        )

    add_salient_headline(
        fig,
        title="Marginal effect of sedation on liberation: cross-site overlay",
        subtitle=(
            f"base_spec={base_spec} · "
            f"version={version_name} (cr() basis = {version_cfg['spec_suffix']}) · "
            f"extub model_type={version_cfg['extub_mt']}"
        ),
        units_line="Predicted probability vs predictor; one curve per site, ribbons = 95% CI",
    )
    fig.tight_layout()
    return fig


def _render_one(
    df: pd.DataFrame,
    outcome: str,
    model_type: str,
    spec: str,
    panel_rows: list[int],
) -> plt.Figure:
    """Figure with per-site curves overlaid; renders only `panel_rows`.

    `panel_rows` indexes into the per-site grid's `panel_row` column
    (0 = daytime continuous rates, 1 = night–day diffs). The output
    axes grid has `len(panel_rows)` rows × 3 cols. With `squeeze=False`
    the returned `axes` is always 2D so the per-cell loop is uniform
    regardless of row count.
    """
    cell = df[
        (df["outcome"] == outcome)
        & (df["model_type"] == model_type)
        & (df["spec"] == spec)
    ]
    sites = sorted(cell["site"].unique().tolist())

    n_rows = len(panel_rows)
    # Figsize scales with row count: ~3.5 in per row + ~1 in for axis
    # title/legend chrome. Single-row "primary" view is shorter; 2-row
    # "audit" view keeps the prior height.
    fig_h = 4.5 if n_rows == 1 else 7.5
    fig, axes = plt.subplots(n_rows, 3, figsize=(13.0, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    panel_letters_full = ["A", "B", "C", "D", "E", "F"]
    pidx = 0

    for out_row_idx, src_row_idx in enumerate(panel_rows):
        for col_idx in range(3):
            ax = axes[out_row_idx, col_idx]
            _ggplot_ax(ax)

            panel = cell[
                (cell["panel_row"] == src_row_idx)
                & (cell["panel_col"] == col_idx)
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
                # Legend label only on the first rendered panel (top-left of
                # the OUTPUT axes, not the source 2×3 grid) so the figure-level
                # legend has one entry per site.
                ax.plot(
                    x, p, color=color, linewidth=1.6, zorder=3,
                    label=site_label(s) if (out_row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(
                Y_LABEL.get(outcome, "Predicted Probability"), fontsize=9,
            )
            # Y-axis clipped to [0, 0.5]: manuscript outcomes have predicted
            # probabilities well below 0.5 at both cohorts, so halving the
            # visible range doubles the apparent slope of any decline.
            ax.set_ylim(0, 0.5)
            ax.text(
                -0.12, 1.08, panel_letters_full[pidx],
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

    n_sites = cell["site"].nunique() if not cell.empty else 0
    add_salient_headline(
        fig,
        title=Y_LABEL.get(outcome, "Probability"),
        subtitle=(
            f"model_type={model_type} · spec={spec} · k={n_sites} sites · "
            "cr() basis re-evaluated at site grids"
        ),
        units_line="Predicted probability vs predictor; one curve per site, ribbons = 95% CI",
        title_color=_OUTCOME_TITLE_COLORS.get(outcome, "#1f1f1f"),
    )
    fig.tight_layout()
    return fig


def main() -> None:
    df_all = _stack_per_site()
    if df_all.empty:
        return

    save_agg_csv(df_all, "marginal_effects_rcs_cross_site")

    # Build the union of (outcome, model_type, spec) tuples that any
    # version × base_spec needs. Used both to (a) render audit figures
    # for each unique combo and (b) sanity-check that the per-site
    # data actually contains rows for everything the versions reference.
    needed_combos: set[tuple[str, str, str]] = set()
    for _name, cfg in VERSIONS:
        for base_spec in BASE_SPECS:
            spec = f"{base_spec}_rcs_{cfg['spec_suffix']}"
            for outcome, mt in _rows_for_version(cfg):
                needed_combos.add((outcome, mt, spec))

    # Sanity-check: warn if any needed combo is missing from the data.
    have_combos = set(map(tuple, df_all[["outcome", "model_type", "spec"]].drop_duplicates().values))
    missing = needed_combos - have_combos
    if missing:
        logger.info(
            f"  WARN: {len(missing)} needed (outcome, model_type, spec) combos "
            f"are missing from per-site data. Re-run 08_models.py per site. "
            f"First few: {sorted(missing)[:3]}"
        )

    # Primary: one combined figure per (base_spec, version). 3 versions ×
    # 2 base specs = 6 PNGs.
    for version_name, cfg in VERSIONS:
        for base_spec in BASE_SPECS:
            fig = _render_primary_combined(df_all, base_spec, version_name, cfg)
            save_agg_fig(
                fig,
                f"marginal_effects_{base_spec}_rcs_{cfg['spec_suffix']}"
                f"_{version_name}_primary_cross_site",
            )
            plt.close(fig)

    # Audit: one figure per unique (outcome × model_type × spec) tuple any
    # version uses. With 3 versions × 2 base_specs the unique-combo count
    # is 10 (SBT-gee × 4 specs + extub-logit × 2 specs + extub-logit_asym × 4 specs).
    # Each shows daytime + diff predictors for verification.
    for outcome, model_type, spec in sorted(needed_combos):
        outcome_short = OUTCOME_SHORT.get(outcome, outcome)
        fig = _render_one(
            df_all, outcome, model_type, spec,
            panel_rows=[_DAYTIME_ROW, _DIFF_ROW],
        )
        # AUDIT badge top-LEFT (away from the spec-name title at top-right
        # of the right column). Companion to the `_audit_cross_site.png`
        # filename suffix.
        add_audit_badge(fig, ha="left")
        save_agg_fig(
            fig,
            f"marginal_effects_{outcome_short}_{model_type}_{spec}_audit_cross_site",
        )
        plt.close(fig)


if __name__ == "__main__":
    main()
