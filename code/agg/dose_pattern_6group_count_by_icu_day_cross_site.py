"""Cross-site cohort-attrition stacked bar with 6-group dose-pattern composition.

Federated companion to
`code/descriptive/dose_pattern_6group_count_by_icu_day.py`. Reads each site's
pre-aggregated `output_to_share/{site}/descriptive/dose_pattern_6group_count_by_icu_day.csv`
(no raw-parquet access — see `.dev/CLAUDE.md` "Federation contract"). Each
per-site CSV is a long-format `(drug, nth_day, pattern_label, count)`
table already filtered to full-24h ICU days 1..7; this script just stacks
sites, pools, and renders TWO PNGs sharing one long-format CSV:

  - `*_pooled.png` — single 1×3 row: `Master cohort (all sites)`.
    Patient-day counts pooled across sites at the (drug, x_bin, pattern)
    level. Mathematically `pooled_count(b, p) = Σ_site count_site(b, p)`.
    Federated-summary view that stays readable as more sites join.
  - `*_per_site.png` — `N_sites × 3` grid, one row per site, alphabetical
    (matches `SITE_PALETTE` indexing). Row labels respect `ANONYMIZE_SITES`
    via `site_label()`.
  - **Cols (3)** in both PNGs: drugs in canonical order — propofol,
    fentanyl eq, midaz eq.

Color-encoding 7 categories already exhausts the visual palette, so site is
encoded by **vertical position** (not color).

Outputs:
  - `output_to_agg/dose_pattern_6group_count_by_icu_day_cross_site.csv`
    Long format: `site, drug, x_bin, pattern, count`. Per-site rows are
    keyed by the real site name; pooled rows use `site = "ALL"`.
  - `output_to_agg/figures/dose_pattern_6group_count_by_icu_day_cross_site_pooled.png`
  - `output_to_agg/figures/dose_pattern_6group_count_by_icu_day_cross_site_per_site.png`

Usage:
    uv run python code/agg/dose_pattern_6group_count_by_icu_day_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/dose_pattern_6group_count_by_icu_day_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.dose_pattern_6group_count_by_icu_day")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    COUNT_BAR_STACK_ORDER,
    DOSE_PATTERN_COLORS,
    DOSE_PATTERN_LABELS,
    DRUG_LABELS,
    DRUGS,
    THRESHOLDS,
    apply_style,
    list_sites,
    load_site_descriptive_csv,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# ── Constants (match the per-site script's conventions) ─────────────────
MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = [str(i) for i in range(MIN_DAY, MAX_DAY + 1)]
SEGMENT_LABEL_THRESHOLD_FRAC = 0.04
SEGMENT_LABEL_MIN_COUNT = 30  # absolute count floor for "X%" annotations

# Sentinel value used for pooled rows in the long-format CSV. Distinct
# from any real site name (alphabetic site dirs only) so a join/groupby
# never confuses the two.
POOLED_KEY = "ALL"
POOLED_DISPLAY = "Master cohort\n(all sites)"

def _load_site_long(site: str) -> pd.DataFrame | None:
    """Read a site's pre-aggregated 6-group count CSV.

    Federation-clean: pulls from
    `output_to_share/{site}/descriptive/dose_pattern_6group_count_by_icu_day.csv`
    (no PHI, no IDs, no raw rows). Adapts the per-site column names
    (`nth_day`, `pattern_label`) to the cross-site script's downstream
    names (`x_bin`, `pattern`) and reindexes to the canonical 7 × 6 grid
    so every cell is present (zero-filled if missing).

    Returns None if the per-site CSV is absent — the caller logs + skips.
    """
    try:
        df = load_site_descriptive_csv(site, "dose_pattern_6group_count_by_icu_day")
    except FileNotFoundError:
        logger.info(
            f"  SKIP {site}: output_to_share/{site}/descriptive/"
            f"dose_pattern_6group_count_by_icu_day.csv not found — "
            f"re-run code/descriptive/dose_pattern_6group_count_by_icu_day.py."
        )
        return None

    out_frames: list[pd.DataFrame] = []
    for drug in DRUGS:
        d = df.loc[df["drug"] == drug].copy()
        d["x_bin"] = pd.Categorical(
            d["nth_day"].astype(int).astype(str),
            categories=DAY_BINS, ordered=True,
        )
        d["pattern"] = pd.Categorical(
            d["pattern_label"], categories=list(DOSE_PATTERN_LABELS), ordered=True,
        )
        # Reindex to the canonical (x_bin × pattern) grid; per-site CSV
        # already covers this range but the reindex is cheap insurance.
        grid = (
            d.set_index(["x_bin", "pattern"])["count"]
            .unstack(fill_value=0)
            .reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
            .reindex(index=DAY_BINS, fill_value=0)
        )
        long = grid.stack().reset_index()
        long.columns = ["x_bin", "pattern", "count"]
        long.insert(0, "drug", drug)
        long.insert(0, "site", site)
        out_frames.append(long)
    return pd.concat(out_frames, ignore_index=True)


def _add_pooled(per_site_long: pd.DataFrame) -> pd.DataFrame:
    """Append a pooled `site == "ALL"` row set to the per-site long frame.

    Pooling is a straight sum of counts across sites at the
    (drug, x_bin, pattern) level — equivalent to treating all
    patient-days as a single combined cohort with unit weight per row.
    """
    pooled = (
        per_site_long.groupby(["drug", "x_bin", "pattern"], observed=False)["count"]
        .sum()
        .reset_index()
    )
    pooled.insert(0, "site", POOLED_KEY)
    return pd.concat([pooled, per_site_long], ignore_index=True)


# ── Figure rendering ────────────────────────────────────────────────────
def _render(
    long_df: pd.DataFrame,
    row_keys: list[str],
    *,
    suptitle: str,
) -> plt.Figure:
    """Build an `len(row_keys) × 3` stacked-bar grid for a subset of rows.

    `row_keys` may contain the sentinel `POOLED_KEY` ("ALL") to render
    the pooled "Master cohort" row, and/or real site names for per-site
    rows. Each row's 3 drug panels share y-axis (`sharey="row"`); rows
    use independent y-scales because cohort sizes differ.
    """
    n_rows = len(row_keys)
    n_cols = len(DRUGS)
    # Per-row vertical real estate stays roughly constant as rows are added,
    # plus a fixed header/legend allowance.
    fig_height = 3.0 + 3.0 * n_rows
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(17, fig_height),
        sharex=True, sharey="row",
    )
    axes = np.atleast_2d(axes)

    x_positions = np.arange(len(DAY_BINS))

    for ri, row_key in enumerate(row_keys):
        is_pooled = row_key == POOLED_KEY
        row_label = (
            POOLED_DISPLAY if is_pooled else site_label(row_key)
        )

        for ci, drug in enumerate(DRUGS):
            ax = axes[ri, ci]

            # Pivot (long → 7×6) for this (row, drug) panel.
            cell = long_df[
                (long_df["site"] == row_key) & (long_df["drug"] == drug)
            ]
            counts = (
                cell.pivot(index="x_bin", columns="pattern", values="count")
                .reindex(index=DAY_BINS, columns=list(DOSE_PATTERN_LABELS), fill_value=0)
                .astype(float)
            )
            bar_totals = counts.sum(axis=1)

            # Stack 6 categories: 5 colored measurable-diff bands, then the
            # gray drug-holiday cap on top.
            cum = np.zeros(len(DAY_BINS))
            for label in COUNT_BAR_STACK_ORDER:
                seg = counts[label].to_numpy()
                color = DOSE_PATTERN_COLORS[label]
                ax.bar(
                    x_positions, seg, bottom=cum,
                    color=color, edgecolor="white",
                    linewidth=0.4, label=label,
                )
                # "X%" annotations on segments large enough to read
                # (matches per-site script `s >= 30` floor).
                for x, c, total, s in zip(x_positions, cum, bar_totals.to_numpy(), seg):
                    if total <= 0:
                        continue
                    frac = s / total
                    if frac >= SEGMENT_LABEL_THRESHOLD_FRAC and s >= SEGMENT_LABEL_MIN_COUNT:
                        ax.text(
                            x, c + s / 2, f"{frac * 100:.0f}%",
                            ha="center", va="center", fontsize=6.5,
                            color="white" if color in ("#2166ac", "#b2182b") else "black",
                        )
                cum += seg

            # Horizontal ruler line at the boundary between the 5 measurable
            # bands and the gray drug-holiday cap.
            on_drug_top = (
                bar_totals - counts["Not receiving that day"]
            ).to_numpy()
            ax.hlines(
                on_drug_top, x_positions - 0.45, x_positions + 0.45,
                color="dimgray", linewidth=0.8, zorder=3,
            )

            # Column titles on top row only.
            if ri == 0:
                ax.set_title(
                    f"{DRUG_LABELS[drug]}  (T = ±{THRESHOLDS[drug]})",
                    fontsize=11,
                )
            # X labels on bottom row only.
            if ri == n_rows - 1:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(DAY_BINS)
                ax.set_xlabel("ICU day (full-24h coverage only)")
            # Row label + y-axis label on leftmost column. Pooled row gets
            # a bold label so it reads as the "headline" view.
            if ci == 0:
                fontweight = "bold" if is_pooled else "normal"
                ax.set_ylabel(
                    f"{row_label}\nPatient-days",
                    fontsize=10, fontweight=fontweight,
                )

    # Single legend at the figure level — one entry per category.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DOSE_PATTERN_COLORS[label], ec="white")
        for label in DOSE_PATTERN_LABELS
    ]
    # Suptitle BEFORE tight_layout so the layout reserves space for it
    # (otherwise it overlaps the panel column titles in short figures).
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    # Legend placement: reserve a fixed ~0.9 inches of bottom whitespace
    # regardless of n_rows so the legend has consistent room to clear the
    # x-axis tick labels on the bottom panel row. Without this, the
    # 1-row pooled figure (figsize=(17, 6)) has only ~0.18 inches below
    # the bottom panel and the legend overlaps the x-tick labels.
    bottom_reserve_in = 0.9
    bottom_frac = bottom_reserve_in / fig_height
    fig.subplots_adjust(bottom=bottom_frac)
    fig.legend(
        handles, list(DOSE_PATTERN_LABELS),
        loc="lower center", ncol=3, frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )
    return fig


# ── Entry point ─────────────────────────────────────────────────────────
def main() -> None:
    apply_style()

    sites = list_sites()
    if not sites:
        logger.info("No sites found under output_to_share/. Nothing to plot.")
        return
    logger.info(f"Discovered sites: {sites}")

    per_site_frames: list[pd.DataFrame] = []
    valid_sites: list[str] = []
    for s in sites:
        long = _load_site_long(s)
        if long is None:
            continue
        per_site_frames.append(long)
        valid_sites.append(s)

    if not per_site_frames:
        logger.info("No usable per-site data; nothing to render.")
        return

    per_site_long = pd.concat(per_site_frames, ignore_index=True)
    full = _add_pooled(per_site_long)

    save_agg_csv(full, "dose_pattern_6group_count_by_icu_day_cross_site")

    # Pooled view (1 × 3): the federated-summary headline.
    fig_pooled = _render(
        full, [POOLED_KEY],
        suptitle=(
            "Master cohort (all sites): cohort attrition with 6-group "
            "dose-pattern composition\n"
            "Patient-days summed across all sites at each (drug, ICU-day, "
            "pattern) cell."
        ),
    )
    save_agg_fig(
        fig_pooled,
        "dose_pattern_6group_count_by_icu_day_cross_site_pooled",
    )
    plt.close(fig_pooled)

    # Per-site view (N_sites × 3): one row per discovered site.
    fig_per_site = _render(
        full, valid_sites,
        suptitle=(
            "Per-site cohort attrition with 6-group dose-pattern composition\n"
            "One row per site; sharey within each row, independent across rows."
        ),
    )
    save_agg_fig(
        fig_per_site,
        "dose_pattern_6group_count_by_icu_day_cross_site_per_site",
    )
    plt.close(fig_per_site)


if __name__ == "__main__":
    main()
