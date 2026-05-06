"""Cross-site hour-of-day sedation dose figures.

For each site (`output/{site}/sed_dose_by_hr.parquet` joined to
`sed_dose_daily.parquet`) we compute 5-cohort × 24-hour aggregates of
three rate metrics per drug, stack them, and render one PNG per drug
showing how the diurnal pattern compares across sites.

Cohorts (columns of each PNG):
  - `day1`     — first ICU day, **full 24-hr only** (n_hours_day == 12
                 AND n_hours_night == 12). Patients extubated or
                 exiting the cohort mid-day-1 are excluded so the
                 hour-by-hour curve is not contaminated by survival
                 bias (~7-14% of `_nth_day=1` patient-hours come from
                 partial days).
  - `day2`     — second ICU day, full 24-hr only.
  - `day3`     — third ICU day, full 24-hr only.
  - `matched`  — ALL days where BOTH shifts had any IMV coverage
                 (n_hours_day > 0 AND n_hours_night > 0). Pools across
                 all ICU days. Same row set the per-site `*_mean_matched`
                 columns use.
  - `strict`   — ALL days with full 12+12h coverage
                 (n_hours_day == 12 AND n_hours_night == 12). Pools
                 across all ICU days; `day1`/`day2`/`day3` are
                 day-stratified subsets of this cohort.

A single hourly row can belong to MULTIPLE cohorts (e.g., a `_nth_day=1`
row from a full-24-hr day contributes to `day1`, `matched`, and
`strict`). DuckDB's UNNEST handles this in a single scan.

Rates (columns of each PNG):
  - `mean_on_drug`  — avg dose rate among hours with rate > 0
  - `mean_all_imv`  — avg dose rate across all IMV hours (zeros included)
  - `pct_on_drug`   — % of IMV patient-hours with rate > 0

X-axis: hour-of-day in 7am→7pm→7am order so the day shift sits in the
left half and night shift in the right half (matches
`07_descriptive.py:395-397`). Red dashed vertical at the day↔night
boundary (7pm).

Outputs:
  - output_to_agg/sed_dose_by_hr_of_day_cross_site.csv        (long-ish wide)
  - output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_prop.png
  - output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_fenteq.png
  - output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_midazeq.png

Usage:
    uv run python code/agg/sed_dose_by_hr_of_day_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/sed_dose_by_hr_of_day_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DRUG_LABELS,
    DRUG_UNITS,
    SITE_PALETTE,
    apply_style,
    list_sites,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# ── Figure-fixed selectors ────────────────────────────────────────────────
# Cohort axis (columns after the row/col flip). Order = left-to-right.
# day1/2/3 are now restricted to FULL 24-hr ICU days only — patients
# extubated or exiting mid-day are excluded so the diurnal curve is not
# contaminated by survival bias at later hours of the day.
COHORTS: list[tuple[str, str]] = [
    ("day1",    "Day 1\n(full 24h)"),
    ("day2",    "Day 2\n(full 24h)"),
    ("day3",    "Day 3\n(full 24h)"),
    ("matched", "Matched\n(any-shift coverage)"),
    ("strict",  "Strict\n(any day, 12 + 12h)"),
]

# Rate axis (cols). The 3rd col uses a different y-scale (% vs absolute
# rate) — `_render` uses sharey='col' so the column groups stay coherent.
# Each rate spec is keyed by drug; the script substitutes drug-specific
# column names from the wide CSV at render time.
RATES: list[tuple[str, str, str]] = [
    # (rate_key, column-suffix-in-wide-csv, column-title)
    ("mean_on_drug", "_mean_on_drug", "Avg rate (on-drug only)"),
    ("mean_all_imv", "_mean_all_imv", "Avg rate (all IMV-hrs)"),
    ("pct_on_drug",  "_pct_on_drug",  "% of pt-hrs on drug"),
]

# Drugs (one PNG per drug). The wide CSV uses these as column-name prefixes.
DRUGS: list[tuple[str, str, str]] = [
    # (key, csv-prefix, drug-key for DRUG_LABELS / DRUG_UNITS lookup)
    ("prop",    "propofol_mcg_kg_min", "prop"),
    ("fenteq",  "fenteq_mcg",          "fenteq"),
    ("midazeq", "midazeq_mg",          "midazeq"),
]

# Hour-of-day reordering: 7am → 7pm → 7am (matches per-site figure).
# Hour 19 (7pm) lands at x-position 12 in the reordered list. The
# day↔night boundary line is drawn AT x=12 (right under the labelled
# "19" tick) rather than at x=11.5: x=12 is where the first night-shift
# hour (19:00–20:00) begins on the axis, and aligning the line with the
# labelled tick avoids the visual offset that x=11.5 creates.
_HOUR_ORDER = list(range(7, 24)) + list(range(0, 7))
_DAY_NIGHT_BOUNDARY_IDX = 12  # 7pm tick position
_HOUR_INDEX = {h: i for i, h in enumerate(_HOUR_ORDER)}


# ── Per-site DuckDB aggregation ──────────────────────────────────────────
def _aggregate_site(site: str) -> pd.DataFrame:
    """Return a DataFrame with cohort × _hr × per-drug stats for one site.

    Single-scan UNNEST: one hourly row may emit up to 5 cohort labels
    (it's in `day1` AND `matched` AND `strict` if all conditions hold);
    DuckDB UNNEST expands that array, NULL entries are filtered out via
    WHERE before the GROUP BY. Propofol's hourly column (`prop_mcg_kg_total`)
    is in mcg/kg/HR upstream; we divide by 60 to land at the
    clinician-preferred mcg/kg/MIN units used everywhere downstream.
    """
    sql = f"""
        WITH daily AS (
            FROM read_parquet('output/{site}/sed_dose_daily.parquet')
            SELECT hospitalization_id, _nth_day, n_hours_day, n_hours_night
        ),
        hours AS (
            FROM read_parquet('output/{site}/sed_dose_by_hr.parquet')
            SELECT
                hospitalization_id,
                _nth_day,
                _hr,
                prop_mcg_kg_total,
                _fenteq_mcg_total,
                _midazeq_mg_total
        ),
        cohort_assigned AS (
            FROM hours h
            LEFT JOIN daily d USING (hospitalization_id, _nth_day)
            SELECT
                h.*,
                -- day1/2/3 require full 12+12h coverage so partial-day
                -- patients (extubation / cohort exit during the day)
                -- don't bias later hours of the diurnal curve. matched
                -- and strict still pool ACROSS all ICU days.
                cohort: UNNEST([
                    CASE WHEN h._nth_day = 1
                         AND d.n_hours_day = 12 AND d.n_hours_night = 12
                         THEN 'day1' END,
                    CASE WHEN h._nth_day = 2
                         AND d.n_hours_day = 12 AND d.n_hours_night = 12
                         THEN 'day2' END,
                    CASE WHEN h._nth_day = 3
                         AND d.n_hours_day = 12 AND d.n_hours_night = 12
                         THEN 'day3' END,
                    CASE WHEN d.n_hours_day > 0 AND d.n_hours_night > 0
                         THEN 'matched' END,
                    CASE WHEN d.n_hours_day = 12 AND d.n_hours_night = 12
                         THEN 'strict' END
                ])
        )
        FROM cohort_assigned
        SELECT
            cohort
            , _hr
            -- avg-on-drug (excludes 0-dose hours)
            , propofol_mcg_kg_min_mean_on_drug:
                AVG(CASE WHEN prop_mcg_kg_total > 0
                         THEN prop_mcg_kg_total / 60.0 END)
            , fenteq_mcg_mean_on_drug:
                AVG(CASE WHEN _fenteq_mcg_total > 0
                         THEN _fenteq_mcg_total END)
            , midazeq_mg_mean_on_drug:
                AVG(CASE WHEN _midazeq_mg_total > 0
                         THEN _midazeq_mg_total END)
            -- avg-all-imv (zeros included)
            , propofol_mcg_kg_min_mean_all_imv:
                AVG(COALESCE(prop_mcg_kg_total, 0) / 60.0)
            , fenteq_mcg_mean_all_imv:
                AVG(COALESCE(_fenteq_mcg_total, 0))
            , midazeq_mg_mean_all_imv:
                AVG(COALESCE(_midazeq_mg_total, 0))
            -- counts
            , n_on_drug_propofol: COUNT(*) FILTER (WHERE prop_mcg_kg_total > 0)
            , n_on_drug_fenteq:   COUNT(*) FILTER (WHERE _fenteq_mcg_total  > 0)
            , n_on_drug_midazeq:  COUNT(*) FILTER (WHERE _midazeq_mg_total  > 0)
            , n_imv: COUNT(*)
        WHERE cohort IS NOT NULL
        GROUP BY cohort, _hr
        ORDER BY cohort, _hr
    """
    df = duckdb.sql(sql).df()
    df.insert(0, "site", site)
    return df


def _stack_per_site() -> pd.DataFrame:
    """Concat per-site aggregates; skip sites missing the required parquets."""
    sites = list_sites()
    if not sites:
        print("No sites found under output_to_share/. Nothing to plot.")
        return pd.DataFrame()
    print(f"Discovered sites: {sites}")

    frames: list[pd.DataFrame] = []
    for s in sites:
        hr_path = Path(f"output/{s}/sed_dose_by_hr.parquet")
        daily_path = Path(f"output/{s}/sed_dose_daily.parquet")
        if not (hr_path.exists() and daily_path.exists()):
            print(f"  SKIP {s}: missing parquets — re-run 02_exposure.py for this site.")
            continue
        frames.append(_aggregate_site(s))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _add_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute pct_on_drug = n_on_drug / n_imv * 100 per drug."""
    out = df.copy()
    safe_n = out["n_imv"].where(out["n_imv"] > 0, np.nan)
    for csv_prefix, n_col in [
        ("propofol_mcg_kg_min", "n_on_drug_propofol"),
        ("fenteq_mcg",          "n_on_drug_fenteq"),
        ("midazeq_mg",          "n_on_drug_midazeq"),
    ]:
        out[f"{csv_prefix}_pct_on_drug"] = out[n_col] / safe_n * 100.0
    return out


# ── Render one PNG per drug ──────────────────────────────────────────────
def _format_panel(ax, *, top_row: bool, left_col: bool,
                  col_title: str | None, row_label: str | None,
                  ylabel: str | None) -> None:
    """Common per-panel styling shared by every cell of every figure."""
    ax.axvline(_DAY_NIGHT_BOUNDARY_IDX, color="firebrick",
               linestyle="--", linewidth=0.9, alpha=0.6, zorder=0)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4, zorder=0)
    if top_row and col_title:
        ax.set_title(col_title, fontsize=11)
    if left_col and row_label:
        ax.set_ylabel(row_label + (f"\n({ylabel})" if ylabel else ""),
                      fontsize=10)
    elif left_col and ylabel:
        ax.set_ylabel(ylabel, fontsize=10)


def _render_drug(df: pd.DataFrame, drug_key: str, csv_prefix: str,
                 drug_label_key: str) -> plt.Figure:
    """Build one 3×5 forest-of-curves figure for a single drug.

    Rows = rate type (3), cols = cohort (5). `sharey="row"` so each rate
    row uses its own y-axis (drug-specific units for `mean_on_drug` and
    `mean_all_imv` rows; 0–100 for the `pct_on_drug` row).
    """
    sites = sorted(df["site"].unique().tolist())
    n_rows, n_cols = len(RATES), len(COHORTS)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16.0, 9.0),
        sharex=True, sharey="row",
    )
    axes = np.atleast_2d(axes)

    drug_pretty = DRUG_LABELS[drug_label_key]
    drug_unit = DRUG_UNITS[drug_label_key]

    # X tick positions: label every 2nd hour. With the reordering starting
    # at 07, this hits 07, 09, 11, 13, 15, 17, 19, 21, 23, 01, 03, 05 —
    # the 7pm boundary tick "19" is visible (index 12 is even).
    xtick_idx = [i for i, _ in enumerate(_HOUR_ORDER) if i % 2 == 0]
    xtick_lbl = [f"{_HOUR_ORDER[i]:02d}" for i in xtick_idx]

    for ri, (_rate_key, rate_suffix, rate_title) in enumerate(RATES):
        for ci, (cohort_key, cohort_label) in enumerate(COHORTS):
            ax = axes[ri, ci]
            value_col = f"{csv_prefix}{rate_suffix}"

            for si, s in enumerate(sites):
                cell = df[(df["site"] == s) & (df["cohort"] == cohort_key)]
                if cell.empty or value_col not in cell.columns:
                    continue
                cell = cell.set_index("_hr").reindex(_HOUR_ORDER)
                ys = cell[value_col].to_numpy(dtype=float)
                xs = np.arange(len(_HOUR_ORDER))
                color = SITE_PALETTE[si % len(SITE_PALETTE)]
                # Label only on the (top-left) panel — single fig-level
                # legend entry per site.
                label = site_label(s) if (ri == 0 and ci == 0) else None
                ax.plot(
                    xs, ys,
                    color=color, linewidth=1.6,
                    marker="o", markersize=3.5,
                    label=label,
                )

            # Y-axis units differ between absolute-rate rows and pct row.
            ylabel = drug_unit if rate_suffix != "_pct_on_drug" else "%"
            # Build the row label as "Rate title\n(units)" for the leftmost
            # column only — single multi-line ylabel keeps the row identity
            # explicit without needing an extra rotated text element.
            row_label = f"{rate_title}\n({ylabel})" if ci == 0 else None
            _format_panel(
                ax,
                top_row=(ri == 0),
                left_col=(ci == 0),
                col_title=cohort_label,
                row_label=row_label,
                ylabel=None,  # already folded into row_label above
            )

            if rate_suffix == "_pct_on_drug":
                ax.set_ylim(0, 100)

            if ri == n_rows - 1:
                ax.set_xticks(xtick_idx)
                ax.set_xticklabels(xtick_lbl, fontsize=9)
                ax.set_xlabel("Hour of day (local, 7am →)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.01),
            ncol=len(handles), frameon=False, fontsize=10,
            title="Site",
        )

    fig.suptitle(
        f"Cross-site diurnal pattern — {drug_pretty} "
        "(per-site lines, no pooling)",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    return fig


# ── Entry point ──────────────────────────────────────────────────────────
def main() -> None:
    apply_style()
    raw = _stack_per_site()
    if raw.empty:
        return
    df = _add_pct_columns(raw)

    save_agg_csv(df, "sed_dose_by_hr_of_day_cross_site")

    for drug_key, csv_prefix, drug_label_key in DRUGS:
        fig = _render_drug(df, drug_key, csv_prefix, drug_label_key)
        save_agg_fig(fig, f"sed_dose_by_hr_of_day_cross_site_{drug_key}")
        plt.close(fig)


if __name__ == "__main__":
    main()
