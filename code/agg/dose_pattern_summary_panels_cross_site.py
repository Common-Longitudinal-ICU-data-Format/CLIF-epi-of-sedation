"""Cross-site 3×3 QA summary: mean diff + two proportion panels per drug.

Public-facing aggregated QA figure. Reads each site's pre-aggregated
`output_to_share/{site}/descriptive/night_day_dose_stats_by_icu_day.csv`
(federation-clean — see `.dev/CLAUDE.md` "Federation contract"). The
master cohort is the union of all per-site rows; per-site and pooled
estimates share one figure.

Layout: 3 rows × 3 cols (one drug per column: propofol, fent-eq,
midaz-eq).

  - **Row 1**: per-site mean(night − day) ± Student-t 95% CI errorbars,
    plus bold black "Master cohort (all sites)" point+errorbar per ICU
    day. Reference line at y = 0.

  - **Row 2**: per-site proportion `n_diff_gt_0_all / n_full_24h` with
    Wilson 95% CI errorbars, plus pooled (master) proportion with
    Wilson CI. Denominator = every patient-day surviving on IMV that
    day with full-24h coverage, regardless of whether the drug was
    given. Reference line at y = 0.5.

  - **Row 3**: same as Row 2 but denominator = patient-days where the
    drug was actively administered (`_X_any = 1`).

Pooling is fixed-effects "master cohort" — algebraically equivalent to
treating every patient-day from every site as one big cohort with unit
weight per row:

  - Proportion: `pooled_k = Σ k`, `pooled_n = Σ n`, Wilson CI on (Σk, Σn).
  - Mean: `pooled_mean = Σ(n·mean) / Σn`;
          `pooled_var = (Σ sum_sq − (Σ sum)² / Σn) / (Σn − 1)`;
          Student-t CI on `(Σn, pooled_mean, √pooled_var)`.

NOT DerSimonian–Laird random-effects pooling (that stays deferred per
memory `plan_agg_scaffold.md`). Color stability via `SITE_PALETTE`
indexed by alphabetical `list_sites()`. `ANONYMIZE_SITES=1` is honored
via `site_label()`. Pooled series renders solid black with a larger
marker so it reads as the headline.

Outputs:
  - `output_to_agg/figures/dose_pattern_summary_panels_cross_site.png`
  - `output_to_agg/dose_pattern_summary_panels_cross_site.csv` — long
    format `metric, drug, nth_day, n, k, point, ci_lo, ci_hi` (pooled
    master-cohort rows only). Manuscript text cites from this CSV.

Usage:
    uv run python code/agg/dose_pattern_summary_panels_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/dose_pattern_summary_panels_cross_site.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.dose_pattern_summary_panels")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    SITE_PALETTE,
    apply_style,
    list_sites,
    load_site_descriptive_csv,
    save_agg_csv,
    save_agg_fig,
    site_label,
    student_t_ci_from_summary,
    wilson_ci,
)


MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = list(range(MIN_DAY, MAX_DAY + 1))

POOLED_COLOR = "#000000"          # solid black; not in SITE_PALETTE
POOLED_LABEL = "Master cohort (all sites)"
JITTER = 0.06                     # per-site x-offset around the integer tick


# ── Pooled-cohort helpers ──────────────────────────────────────────────
def _pool_proportion(
    cells: pd.DataFrame, num_col: str, den_col: str,
) -> tuple[int, int, float, float, float]:
    """Sum k and n across sites at one (drug, nth_day) cell; Wilson CI.

    Returns `(n, k, point, lo, hi)`. NaN-coerced when n == 0.
    """
    k = int(cells[num_col].sum())
    n = int(cells[den_col].sum())
    point, lo, hi = wilson_ci(k, n)
    return n, k, point, lo, hi


def _pool_mean(cells: pd.DataFrame) -> tuple[int, float, float, float, float]:
    """Fixed-effects pool of mean(diff) across sites at one (drug, nth_day).

    Uses `sum_diff_all` and `sum_sq_diff_all` from the per-site CSV so
    the pooled SD is computed without re-reading raw rows:

      pooled_mean = Σ sum / Σ n
      pooled_var  = (Σ sum_sq − (Σ sum)² / Σ n) / (Σ n − 1)

    Returns `(n, point, lo, hi)` where `point = pooled_mean`. NaN-coerced
    when n == 0; lo/hi == point when n == 1.
    """
    n = int(cells["n_full_24h"].sum())
    if n == 0:
        nan = float("nan")
        return 0, nan, nan, nan
    s = float(cells["sum_diff_all"].sum())
    s2 = float(cells["sum_sq_diff_all"].sum())
    mean = s / n
    if n < 2:
        return n, mean, mean, mean
    var = max(0.0, (s2 - s * s / n) / (n - 1))
    sd = math.sqrt(var)
    _, lo, hi = student_t_ci_from_summary(n, mean, sd)
    return n, mean, lo, hi


# ── Row renderers ──────────────────────────────────────────────────────
def _draw_top_row(
    axes_row, per_site_stats: dict[str, pd.DataFrame], sites: list[str],
) -> list[dict]:
    """Top row: mean diff with Student-t 95% CI errorbars (per-site + pooled).

    Returns pooled rows (one dict per (drug, nth_day)) for the agg CSV.
    """
    pooled_rows: list[dict] = []
    n_sites = len(sites)
    # Symmetric jitter so the cluster centers on the integer tick.
    site_offsets = (np.arange(n_sites) - (n_sites - 1) / 2.0) * JITTER

    for ci, drug in enumerate(DRUGS):
        ax = axes_row[ci]
        for si, s in enumerate(sites):
            stats_df = per_site_stats[s]
            cell = stats_df.loc[stats_df["drug"] == drug].set_index("nth_day")
            color = SITE_PALETTE[si % len(SITE_PALETTE)]

            xs, ys, los, his = [], [], [], []
            for d in DAY_BINS:
                if d not in cell.index:
                    continue
                row = cell.loc[d]
                m, lo, hi = student_t_ci_from_summary(
                    int(row["n_full_24h"]),
                    float(row["mean_diff_all"]),
                    float(row["sd_diff_all"]),
                )
                xs.append(d + site_offsets[si])
                ys.append(m); los.append(lo); his.append(hi)

            ys_a, los_a, his_a = np.array(ys), np.array(los), np.array(his)
            yerr = np.vstack([ys_a - los_a, his_a - ys_a])
            ax.errorbar(
                xs, ys_a, yerr=yerr, fmt="o", color=color,
                markersize=5, linewidth=1.2, capsize=2.5,
                label=site_label(s) if ci == 0 else None,
            )

        # Pooled master-cohort series at un-jittered integer ticks.
        pxs, pys, plos, phis = [], [], [], []
        for d in DAY_BINS:
            cells = pd.concat(
                [
                    df.loc[(df["drug"] == drug) & (df["nth_day"] == d)]
                    for df in per_site_stats.values()
                ],
                ignore_index=True,
            )
            if cells.empty:
                continue
            n, m, lo, hi = _pool_mean(cells)
            pxs.append(d); pys.append(m); plos.append(lo); phis.append(hi)
            pooled_rows.append({
                "metric": "mean_diff",
                "drug": drug,
                "nth_day": d,
                "n": n,
                "k": pd.NA,
                "point": m,
                "ci_lo": lo,
                "ci_hi": hi,
            })

        if pys:
            pys_a, plos_a, phis_a = np.array(pys), np.array(plos), np.array(phis)
            yerr = np.vstack([pys_a - plos_a, phis_a - pys_a])
            ax.errorbar(
                pxs, pys_a, yerr=yerr, fmt="o", color=POOLED_COLOR,
                markersize=7, linewidth=1.8, capsize=3.5, zorder=5,
                label=POOLED_LABEL if ci == 0 else None,
            )

        ax.axhline(0, color="dimgray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(DAY_BINS)
        ax.set_xticklabels([str(d) for d in DAY_BINS])
        ax.set_ylabel(f"Mean diff ({DRUG_UNITS[drug]})")
        ax.set_title(f"{DRUG_LABELS[drug]}", fontsize=11)

    return pooled_rows


def _draw_proportion_row(
    axes_row, per_site_stats: dict[str, pd.DataFrame], sites: list[str],
    num_col: str, den_col: str, metric_label: str, ylabel: str,
) -> list[dict]:
    """Generic row for Wilson-CI proportion panels (middle + bottom).

    Returns pooled rows (one dict per (drug, nth_day)) for the agg CSV.
    """
    pooled_rows: list[dict] = []
    n_sites = len(sites)
    site_offsets = (np.arange(n_sites) - (n_sites - 1) / 2.0) * JITTER

    for ci, drug in enumerate(DRUGS):
        ax = axes_row[ci]
        for si, s in enumerate(sites):
            stats_df = per_site_stats[s]
            cell = stats_df.loc[stats_df["drug"] == drug].set_index("nth_day")
            color = SITE_PALETTE[si % len(SITE_PALETTE)]

            xs, ys, los, his = [], [], [], []
            for d in DAY_BINS:
                if d not in cell.index:
                    continue
                row = cell.loc[d]
                k = int(row[num_col])
                n = int(row[den_col])
                if n == 0:
                    continue
                p, lo, hi = wilson_ci(k, n)
                xs.append(d + site_offsets[si])
                ys.append(p); los.append(lo); his.append(hi)

            ys_a, los_a, his_a = np.array(ys), np.array(los), np.array(his)
            yerr = np.vstack([ys_a - los_a, his_a - ys_a])
            ax.errorbar(
                xs, ys_a, yerr=yerr, fmt="o", color=color,
                markersize=5, linewidth=1.2, capsize=2.5,
            )

        # Pooled master-cohort proportion.
        pxs, pys, plos, phis = [], [], [], []
        for d in DAY_BINS:
            cells = pd.concat(
                [
                    df.loc[(df["drug"] == drug) & (df["nth_day"] == d)]
                    for df in per_site_stats.values()
                ],
                ignore_index=True,
            )
            if cells.empty:
                continue
            n, k, p, lo, hi = _pool_proportion(cells, num_col, den_col)
            if n == 0:
                continue
            pxs.append(d); pys.append(p); plos.append(lo); phis.append(hi)
            pooled_rows.append({
                "metric": metric_label,
                "drug": drug,
                "nth_day": d,
                "n": n,
                "k": k,
                "point": p,
                "ci_lo": lo,
                "ci_hi": hi,
            })

        if pys:
            pys_a, plos_a, phis_a = np.array(pys), np.array(plos), np.array(phis)
            yerr = np.vstack([pys_a - plos_a, phis_a - pys_a])
            ax.errorbar(
                pxs, pys_a, yerr=yerr, fmt="o", color=POOLED_COLOR,
                markersize=7, linewidth=1.8, capsize=3.5, zorder=5,
            )

        ax.axhline(0.5, color="dimgray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(DAY_BINS)
        ax.set_xticklabels([str(d) for d in DAY_BINS])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(ylabel)

    return pooled_rows


# ── Entry point ────────────────────────────────────────────────────────
def main() -> None:
    apply_style()

    sites = list_sites()
    if not sites:
        logger.info("No sites found under output_to_share/. Nothing to plot.")
        return
    logger.info(f"Discovered sites: {sites}")

    per_site_stats: dict[str, pd.DataFrame] = {}
    for s in sites:
        try:
            per_site_stats[s] = load_site_descriptive_csv(
                s, "night_day_dose_stats_by_icu_day",
            )
        except FileNotFoundError:
            logger.info(
                f"  SKIP {s}: output_to_share/{s}/descriptive/"
                f"night_day_dose_stats_by_icu_day.csv not found — "
                f"re-run code/descriptive/night_day_diff_combined_by_icu_day.py."
            )

    if not per_site_stats:
        logger.info("No usable per-site stats; nothing to render.")
        return

    valid_sites = list(per_site_stats.keys())

    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex="col")

    pooled_top = _draw_top_row(axes[0], per_site_stats, valid_sites)
    pooled_mid = _draw_proportion_row(
        axes[1], per_site_stats, valid_sites,
        num_col="n_diff_gt_0_all", den_col="n_full_24h",
        metric_label="prop_diff_gt_0_imv",
        ylabel="% IMV patient-days\nwith night > day",
    )
    pooled_bot = _draw_proportion_row(
        axes[2], per_site_stats, valid_sites,
        num_col="n_diff_gt_0_on_drug", den_col="n_on_drug",
        metric_label="prop_diff_gt_0_on_drug",
        ylabel="% on-drug patient-days\nwith night > day",
    )

    # Bottom-row x labels only.
    for ax in axes[2]:
        ax.set_xlabel("ICU day (full-24h coverage only)")

    # Single figure-level legend at the bottom (per-site + pooled).
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        "Cross-site QA: night-minus-day sedation patterns by ICU day "
        "(full-24h coverage, days 1–7)\n"
        "Row 1: mean diff ± Student-t 95% CI  •  "
        "Row 2: % IMV cohort with night > day ± Wilson 95% CI  •  "
        "Row 3: % on-drug cohort with night > day ± Wilson 95% CI",
        fontsize=11, y=0.995,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.legend(
        handles, labels,
        loc="lower center", ncol=min(len(handles), 6),
        frameon=False, fontsize=9,
        bbox_to_anchor=(0.5, 0.005),
    )

    save_agg_fig(fig, "dose_pattern_summary_panels_cross_site")
    plt.close(fig)

    # Persist pooled estimates for direct citation in manuscript text.
    pooled_df = pd.DataFrame(pooled_top + pooled_mid + pooled_bot)
    save_agg_csv(pooled_df, "dose_pattern_summary_panels_cross_site")


if __name__ == "__main__":
    main()
