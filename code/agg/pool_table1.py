"""Pooled Table 1 across all discovered sites — federation-clean rewrite.

Reads each site's three new long-format CSVs from
`output_to_share/{site}/models/`:
  - `table1_continuous.csv`  (n, sum, sum_sq → exact pooled mean/SD)
  - `table1_categorical.csv` (n, total_n per (variable, level))
  - `table1_histograms.csv`  (per-bin counts → pooled median/Q1/Q3
    via inverse-CDF interpolation)

Produces `output_to_agg/table1_by_site.csv` with one row per Table 1
entry. Schema:

    row_label, category,
    <site_1>, <site_2>, ..., Pooled,
    Missing_<site_1>, Missing_<site_2>, ..., Missing_pooled

Pooling rules:
  - Continuous reported as `mean (SD)` (= variables in NORMAL_VARS):
    fixed-effects pool of `(n, sum, sum_sq)` via
    `pooled_mean_sd_from_summary`. Algebraically equivalent to recomputing
    on the concatenated raw vectors.
  - Continuous reported as `median [Q1, Q3]` (= NONNORMAL_VARS): pooled
    quantiles via `pooled_quantile_from_histograms` on summed bins.
    Integer-aware mode for `cci_score`, `sofa_1st24h`, `n_days_full_24h`
    so the pooled values match `np.percentile(values, q)` exactly on the
    raw integer sample.
  - Categorical: sum counts per (variable, category), recompute pct
    against pooled `total_n`. Apply `BINARY_DISPLAY_LEVEL` to suppress
    the redundant level for binary vars (so `ever_pressor` shows only
    the `Yes` row).

Site column headers respect ANONYMIZE_SITES (via site_label()).

Usage:
    uv run python code/agg/pool_table1.py
    ANONYMIZE_SITES=1 uv run python code/agg/pool_table1.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.pool_table1")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from _shared import (  # noqa: E402
    list_sites,
    load_site_cohort_stats,
    load_site_table1_categorical,
    load_site_table1_continuous,
    load_site_table1_histograms,
    pooled_mean_sd_from_summary,
    pooled_quantile_from_histograms,
    save_agg_csv,
    site_label,
)
from _table1_schema import (  # noqa: E402
    BINARY_DISPLAY_LEVEL,
    CATEGORICAL_ORDER,
    CATEGORICAL_VARS_PER_PATIENT_DAY,
    CATEGORICAL_VARS_PER_STAY,
    CONTINUOUS_VARS,
    INTEGER_VARS,
    NONNORMAL_VARS,
    NORMAL_VARS,
)


def _fmt_mean_sd(mean: float, sd: float) -> str:
    if mean != mean:  # NaN
        return ""
    if sd != sd:
        return f"{mean:.1f}"
    return f"{mean:.1f} ({sd:.1f})"


def _fmt_median_iqr(med: float, q1: float, q3: float) -> str:
    if med != med:
        return ""
    return f"{med:.1f} [{q1:.1f}, {q3:.1f}]"


def _fmt_count_pct(n: int, pct: float) -> str:
    if pct != pct:
        return f"{int(n):,}"
    return f"{int(n):,} ({pct:.1f})"


def main() -> None:
    sites = list_sites()
    if not sites:
        logger.info("No sites found under output_to_share/. Nothing to pool.")
        return
    logger.info(f"Discovered sites: {sites}")

    site_labels = [site_label(s) for s in sites]
    if len(set(site_labels)) != len(site_labels):
        raise RuntimeError(f"Site label collision among {site_labels}")

    # ── Load per-site CSVs ────────────────────────────────────────────
    cont_per_site:  dict[str, pd.DataFrame] = {}
    cat_per_site:   dict[str, pd.DataFrame] = {}
    hist_per_site:  dict[str, pd.DataFrame] = {}
    cohort_stats:   dict[str, dict] = {}
    for s in sites:
        cont_per_site[s]  = load_site_table1_continuous(s)
        cat_per_site[s]   = load_site_table1_categorical(s)
        hist_per_site[s]  = load_site_table1_histograms(s)
        cohort_stats[s]   = load_site_cohort_stats(s).iloc[0].to_dict()

    # Tag each long-format frame with its site so groupby can pool.
    for s in sites:
        cat_per_site[s]  = cat_per_site[s].assign(_site=s)
        hist_per_site[s] = hist_per_site[s].assign(_site=s)

    cat_all  = pd.concat(cat_per_site.values(),  ignore_index=True)
    hist_all = pd.concat(hist_per_site.values(), ignore_index=True)

    # ── Build Table 1 row by row ──────────────────────────────────────
    out_rows: list[dict] = []

    # Header row: n  (= number of hospitalizations)
    site_n: dict[str, int] = {
        s: int(cohort_stats[s].get("n_hospitalizations", 0))
        for s in sites
    }
    pooled_n = sum(site_n.values())
    out_rows.append({
        "row_label": "n",
        "category":  "",
        **{site_labels[i]: f"{site_n[s]:,}" for i, s in enumerate(sites)},
        "Pooled":    f"{pooled_n:,}",
        **{f"Missing_{site_labels[i]}": "" for i in range(len(sites))},
        "Missing_pooled": "",
    })

    # Continuous rows
    for var in CONTINUOUS_VARS:
        # Per-site values + pooled
        site_cells:    dict[str, str] = {}
        site_missing:  dict[str, int] = {}
        per_site_summaries: list[dict] = []
        for s in sites:
            row = cont_per_site[s].loc[cont_per_site[s].variable == var]
            if row.empty:
                site_cells[s] = ""
                site_missing[s] = 0
                continue
            r = row.iloc[0].to_dict()
            site_missing[s] = int(r.get("n_missing", 0) or 0)
            per_site_summaries.append({
                "n":      int(r["n"]),
                "sum":    float(r["sum"]),
                "sum_sq": float(r["sum_sq"]),
            })
            if var in NORMAL_VARS:
                site_cells[s] = _fmt_mean_sd(float(r["mean"]), float(r["sd"]))
            else:
                site_cells[s] = _fmt_median_iqr(
                    float(r["median"]), float(r["q1"]), float(r["q3"])
                )

        if var in NORMAL_VARS:
            pm, psd, _ = pooled_mean_sd_from_summary(per_site_summaries)
            pooled_cell = _fmt_mean_sd(pm, psd)
            row_label = f"{var}, mean (SD)"
        else:
            hist_var = hist_all.loc[hist_all.variable == var]
            integer_mode = var in INTEGER_VARS
            pmed = pooled_quantile_from_histograms(hist_var, 0.50, integer=integer_mode)
            pq1  = pooled_quantile_from_histograms(hist_var, 0.25, integer=integer_mode)
            pq3  = pooled_quantile_from_histograms(hist_var, 0.75, integer=integer_mode)
            pooled_cell = _fmt_median_iqr(pmed, pq1, pq3)
            row_label = f"{var}, median [Q1, Q3]"

        pooled_missing = sum(site_missing.values())
        out_rows.append({
            "row_label": row_label,
            "category":  "",
            **{site_labels[i]: site_cells[s] for i, s in enumerate(sites)},
            "Pooled":    pooled_cell,
            **{f"Missing_{site_labels[i]}": site_missing[s] for i, s in enumerate(sites)},
            "Missing_pooled": pooled_missing,
        })

    # Categorical rows (per-stay then per-patient-day, in order)
    for var in CATEGORICAL_VARS_PER_STAY + CATEGORICAL_VARS_PER_PATIENT_DAY:
        var_rows = cat_all.loc[cat_all.variable == var]
        if var_rows.empty:
            continue

        # Determine display order for categories.
        if var in CATEGORICAL_ORDER:
            order = CATEGORICAL_ORDER[var]
            extra = sorted(set(var_rows.category) - set(order))
            display_order = list(order) + extra
        elif var in BINARY_DISPLAY_LEVEL:
            # Binary: only show the user-chosen level (e.g., "Yes")
            display_order = [BINARY_DISPLAY_LEVEL[var]]
        else:
            display_order = sorted(var_rows.category.unique().tolist())

        # Per-site total_n (for the categorical row's Missing column)
        per_site_total_n  = {s: 0 for s in sites}
        per_site_missing  = {s: 0 for s in sites}
        for s in sites:
            site_var_rows = var_rows.loc[var_rows._site == s]
            if not site_var_rows.empty:
                per_site_total_n[s] = int(site_var_rows.iloc[0]["total_n"])
                per_site_missing[s] = int(site_var_rows.iloc[0].get("n_missing", 0) or 0)

        denom_unit = (
            var_rows.iloc[0]["denominator_unit"]
            if "denominator_unit" in var_rows.columns
            else "patients"
        )
        # Row-label suffix to flag the per-patient-day denominator clearly.
        row_label_base = f"{var}, n (%)"
        if denom_unit == "patient-days":
            row_label_base = f"{var}, n (%) [per patient-day]"

        first_row_for_var = True
        for cat in display_order:
            cat_subset = var_rows.loc[var_rows.category == cat]

            # Per-site cell values
            site_cells: dict[str, str] = {}
            for s in sites:
                site_cat = cat_subset.loc[cat_subset._site == s]
                if site_cat.empty:
                    site_cells[s] = "0 (0.0)"
                else:
                    r = site_cat.iloc[0]
                    site_cells[s] = _fmt_count_pct(int(r["n"]), float(r["pct"]))

            # Pooled
            pooled_n_cat = int(cat_subset["n"].sum())
            pooled_total = sum(per_site_total_n.values())
            pooled_pct = (
                100.0 * pooled_n_cat / pooled_total if pooled_total > 0 else float("nan")
            )
            pooled_cell = _fmt_count_pct(pooled_n_cat, pooled_pct)

            # Missing column only on the first row of each variable (avoid
            # repeating the same value on every category row).
            missing_per_site = (
                {f"Missing_{site_labels[i]}": per_site_missing[s] for i, s in enumerate(sites)}
                if first_row_for_var else
                {f"Missing_{site_labels[i]}": "" for i in range(len(sites))}
            )
            missing_pooled = sum(per_site_missing.values()) if first_row_for_var else ""

            out_rows.append({
                "row_label": row_label_base if first_row_for_var else "",
                "category":  cat,
                **{site_labels[i]: site_cells[s] for i, s in enumerate(sites)},
                "Pooled":    pooled_cell,
                **missing_per_site,
                "Missing_pooled": missing_pooled,
            })
            first_row_for_var = False

    # ── Assemble + write ──────────────────────────────────────────────
    out = pd.DataFrame(out_rows)
    save_agg_csv(out, "table1_by_site")
    logger.info(
        f"Sites pooled: {site_labels}  totals: {site_n}  pooled n: {pooled_n}"
    )


if __name__ == "__main__":
    main()
