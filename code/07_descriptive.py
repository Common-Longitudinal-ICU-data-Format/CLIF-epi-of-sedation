# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(sql_output="native")

with app.setup:
    import marimo as mo
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from clifpy.utils.logging_config import get_logger
    logger = get_logger("epi_sedation.descriptive")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 07 Descriptive Analysis

    Correlation matrix, day-vs-night t-tests, and hourly dose visualizations.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    import pandas as pd

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    # Site-scoped output dirs (see Makefile SITE= flag).
    # Path B++ refactor: descriptive (night-vs-day) outputs land FLAT in
    # {site}/descriptive/ — PNGs and CSVs side-by-side, no nested figures/.
    os.makedirs(f"output_to_share/{SITE_NAME}/descriptive", exist_ok=True)
    logger.info(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, pd


@app.cell
def _(SITE_NAME, pd):
    cohort_merged_final = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")
    logger.info(f"Modeling dataset: {len(cohort_merged_final)} rows")
    return (cohort_merged_final,)


@app.cell
def _(SITE_NAME, pd):
    sed_dose_by_hr = pd.read_parquet(f"output/{SITE_NAME}/seddose_by_id_imvhr.parquet")
    logger.info(f"sed_dose_by_hr: {len(sed_dose_by_hr)} rows")
    return (sed_dose_by_hr,)


@app.cell
def _(SITE_NAME, pd):
    # sed_dose_daily: one row per (hospitalization_id, _nth_day) with day/night
    # shift totals AND n_hours_day/n_hours_night (added in 02_exposure.py).
    # Used by the "Dose by Shift" cell below for per-patient hourly-rate
    # computation that correctly handles single-shift bias.
    sed_dose_daily = pd.read_parquet(f"output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    logger.info(f"sed_dose_daily: {len(sed_dose_daily)} rows")
    return (sed_dose_daily,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Correlation Matrix
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final):
    import matplotlib.pyplot as _plt
    import seaborn as _sns
    continuous_vars = [
        'age', '_nth_day', 'sofa_total', 'cci_score', 'elix_score',
        'prop_dif_mcg_kg_min', 'fenteq_dif_mcg_hr', 'midazeq_dif_mg_hr',
        '_prop_day_mcg_kg_min', '_prop_night_mcg_kg_min', '_fenteq_day_mcg_hr', '_fenteq_night_mcg_hr',
        '_midazeq_day_mg_hr', '_midazeq_night_mg_hr', 'nee_7am', 'nee_7pm',
        '_ph_7am', '_ph_7pm', '_pf_7am', '_pf_7pm',
    ]
    continuous_vars_df = cohort_merged_final[[col for col in continuous_vars if col in cohort_merged_final.columns]]
    corr_matrix = continuous_vars_df.corr(method='pearson')
    _plt.figure(figsize=(14, 10))
    _sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
    _plt.title('Pairwise Pearson Correlation Matrix (Continuous Variables)')
    _plt.tight_layout()
    _corr_path = f'output_to_share/{SITE_NAME}/descriptive/pairwise_corr_matrix.csv'
    corr_matrix.to_csv(_corr_path)
    logger.info(f"Saved {_corr_path}")
    # PNG companion to the CSV — used to visually scan for collinearity blocks
    # when investigating counterintuitive coefficient signs (e.g., the
    # `_dif_*` ↔ `_day_*` linear-combination structure).
    _corr_png = _corr_path.replace('.csv', '.png')
    _plt.savefig(_corr_png, dpi=120, bbox_inches='tight')
    logger.info(f"Saved {_corr_png}")
    _plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Day vs Night T-tests
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd, sed_dose_daily):
    # Dose by Shift — paired + unpaired day-vs-night sedation rates.
    #
    # Per-hospitalization weighted-mean hourly dose rate:
    #     rate = sum(dose across all days) / sum(hours across all days)
    #
    # Two cohort variants (`cohort` column in the output CSV):
    #
    #   `full`    — all cohort days included (default; this is a hosp-level
    #               rate-difference, not a per-day diff, so by the
    #               descriptive_figures.md §6.0 governing rule single-shift
    #               days legitimately accumulate into the lifetime sums).
    #   `matched` — drop single-shift days BEFORE the per-hosp groupby so
    #               each hospitalization's lifetime totals only sum
    #               full-coverage days. Diagnostic companion that matches
    #               the row set used by paradox_summary_6group's
    #               patient-day mean and the matched hour-of-day curve.
    #               Hospitalizations whose ENTIRE qualifying streak fits
    #               in a single shift drop out of the matched cohort.
    #
    # Primary test: paired t-test (each patient contributes both night and
    # day rates; paired removes between-patient variance). Unpaired Welch's
    # t-test shown as a robustness check — it ignores the pairing and is
    # therefore more conservative.
    from scipy import stats as _stats

    # Cohort hosp IDs from analytical_dataset (post-NMB exclusion)
    _cohort_hosp_ids = cohort_merged_final['hospitalization_id'].unique()

    _sed_filtered_full = sed_dose_daily[
        sed_dose_daily['hospitalization_id'].isin(_cohort_hosp_ids)
    ]
    # Matched cohort: drop single-shift days (one shift had 0 hours).
    _sed_filtered_matched = _sed_filtered_full[
        (_sed_filtered_full['n_hours_day'] > 0)
        & (_sed_filtered_full['n_hours_night'] > 0)
    ]

    def _safe_rate(num, denom):
        """num/denom with 0 for denom==0 (avoids NaN; shouldn't occur in practice)."""
        return num.where(denom > 0, other=0) / denom.where(denom > 0, other=1)

    def _per_pt_rates(_sed_frame):
        """Aggregate per-hospitalization weighted-mean dose rates.

        Sums dose AND hours across the supplied row set, then divides — so
        single-shift days legitimately accumulate when present and are
        absent when filtered out upstream (matched variant).
        """
        _hosp_index = _sed_frame['hospitalization_id'].unique()
        _per_pt = (
            _sed_frame.groupby('hospitalization_id')
            .agg({
                'fenteq_day_mcg': 'sum', 'fenteq_night_mcg': 'sum',
                'midazeq_day_mg': 'sum', 'midazeq_night_mg': 'sum',
                # Phase 2: propofol totals are now in mcg/kg.
                'prop_day_mcg_kg': 'sum', 'prop_night_mcg_kg': 'sum',
                'n_hours_day': 'sum', 'n_hours_night': 'sum',
            })
            .reindex(_hosp_index)
            .fillna(0)
        )
        _per_pt['fenteq_rate_night'] = _safe_rate(_per_pt['fenteq_night_mcg'], _per_pt['n_hours_night'])
        _per_pt['fenteq_rate_day']   = _safe_rate(_per_pt['fenteq_day_mcg'],   _per_pt['n_hours_day'])
        _per_pt['midazeq_rate_night'] = _safe_rate(_per_pt['midazeq_night_mg'], _per_pt['n_hours_night'])
        _per_pt['midazeq_rate_day']   = _safe_rate(_per_pt['midazeq_day_mg'],   _per_pt['n_hours_day'])
        # Phase 2: propofol rate in mcg/kg/min (sum mcg/kg / hours / 60).
        _per_pt['prop_rate_night'] = _safe_rate(_per_pt['prop_night_mcg_kg'], _per_pt['n_hours_night']) / 60.0
        _per_pt['prop_rate_day']   = _safe_rate(_per_pt['prop_day_mcg_kg'],   _per_pt['n_hours_day']) / 60.0
        return _per_pt

    def _paired_stats(night, day):
        diff = night - day
        n = len(diff)
        mean_diff = diff.mean()
        se = diff.std(ddof=1) / (n ** 0.5)
        t_crit = _stats.t.ppf(0.975, df=n - 1)
        return (
            mean_diff,
            mean_diff - t_crit * se,
            mean_diff + t_crit * se,
            _stats.ttest_rel(night, day).pvalue,
        )

    def _unpaired_stats(night, day):
        var_n, var_d = night.var(ddof=1), day.var(ddof=1)
        n_n, n_d = len(night), len(day)
        mean_diff = night.mean() - day.mean()
        se = (var_n / n_n + var_d / n_d) ** 0.5
        df_welch = (var_n / n_n + var_d / n_d) ** 2 / (
            (var_n / n_n) ** 2 / (n_n - 1) + (var_d / n_d) ** 2 / (n_d - 1)
        )
        t_crit = _stats.t.ppf(0.975, df=df_welch)
        return (
            mean_diff,
            mean_diff - t_crit * se,
            mean_diff + t_crit * se,
            _stats.ttest_ind(night, day, equal_var=False).pvalue,
        )

    def _fmt_mean_sd(m, s):
        return f"{m:.2f} ({s:.2f})"

    def _fmt_diff_ci(d, lo, hi):
        return f"{d:.2f} ({lo:.2f} to {hi:.2f})"

    def _fmt_p(p):
        if pd.isna(p):
            return "n/a"
        if p < 0.001:
            return "<0.001"
        return f"{p:.3f}"

    _rows = []
    for _cohort_name, _sed_frame in [('full', _sed_filtered_full),
                                       ('matched', _sed_filtered_matched)]:
        _per_pt = _per_pt_rates(_sed_frame)
        logger.info(f"[{_cohort_name}] per-patient rates: {len(_per_pt)} hospitalizations")
        for _label, _night_col, _day_col in [
            ('Fentanyl equivalents dose rate (mcg/hr)', 'fenteq_rate_night', 'fenteq_rate_day'),
            ('Midazolam equivalents dose rate (mg/hr)', 'midazeq_rate_night', 'midazeq_rate_day'),
            ('Propofol dose rate (mcg/kg/min)',          'prop_rate_night',   'prop_rate_day'),
        ]:
            _night = _per_pt[_night_col]
            _day = _per_pt[_day_col]
            for _spec_name, _stat_fn in [('paired', _paired_stats), ('unpaired', _unpaired_stats)]:
                _mean_diff, _lo, _hi, _p = _stat_fn(_night, _day)
                _rows.append({
                    'Variable': _label,
                    'cohort': _cohort_name,
                    'spec': _spec_name,
                    'n_hosp': len(_per_pt),
                    'Nighttime (7p-7a), mean (SD)': _fmt_mean_sd(_night.mean(), _night.std(ddof=1)),
                    'Daytime (7a-7p), mean (SD)':   _fmt_mean_sd(_day.mean(), _day.std(ddof=1)),
                    'mean difference (95% CI)':      _fmt_diff_ci(_mean_diff, _lo, _hi),
                    'p-value':                       _fmt_p(_p),
                })
    sed_dose_by_shift = pd.DataFrame(_rows)
    _shift_path = f'output_to_share/{SITE_NAME}/descriptive/sed_dose_by_shift.csv'
    sed_dose_by_shift.to_csv(_shift_path, index=False)
    logger.info(f"Saved {_shift_path}")
    logger.info(sed_dose_by_shift.to_string(index=False))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly Dose Visualization
    """)
    return


@app.cell
def _(SITE_NAME, sed_dose_by_hr, sed_dose_daily):
    # Three aggregations per sedative per clock hour, plus denominators:
    #
    #   _mean_on_drug — AVG over patient-hours WITH dose > 0 (the CASE WHEN
    #                   filters out zero-dose rows by emitting NULL when the
    #                   condition fails, then AVG ignores those NULLs).
    #                   This is the "on-drug" view: for patients actively
    #                   receiving the sedative at hour H, what's their
    #                   typical drip rate?
    #
    #   _mean_all_imv — AVG over ALL IMV-on patient-hours including off-drug
    #                   ones at 0. NOTE: this is identical to the original
    #                   `AVG(prop_mcg_kg_total)` aggregation, because the
    #                   upstream pipeline (02_exposure.py forward-fill +
    #                   COALESCE → 0) already makes off-drug IMV-hours arrive
    #                   here as 0 (not NULL). This is the "all-IMV" view:
    #                   what's the average drip rate across all ventilated
    #                   patient-hours, including patients off this sedative
    #                   right then? = population sedation burden.
    #                   all_imv_mean = on_drug_mean × (n_on_drug / n_imv).
    #
    #   _mean_matched — AVG over IMV-on patient-hours from the MATCHED cohort
    #                   only: full-coverage non-single-shift patient-days
    #                   (n_hours_day > 0 AND n_hours_night > 0). This matches
    #                   the row set used by the per-patient-day mean in
    #                   `paradox_summary_6group.png`'s exposure cohort. On
    #                   full-coverage days, hour-pooled and patient-day-mean
    #                   are algebraically identical, so this curve should
    #                   agree in direction with the patient-day mean. The
    #                   gap between this curve and `_mean_all_imv` reveals
    #                   the contribution of single-shift boundary days.
    #
    #   n_on_drug_*   — count of patient-hours that hour with dose > 0.
    #   n_imv         — count of patient-hours that hour on IMV at all
    #                   (denominator-transparency for the bottom panel).
    #   n_imv_matched — count of patient-hours that hour from the matched
    #                   (full-coverage) cohort.
    #
    # Legacy column names (propofol_mcg_kg / _fenteq_mcg / _midazeq_mg) are
    # retained as aliases of `_mean_all_imv` (the legacy semantics — NOT
    # `_mean_on_drug`) so any downstream reader keeps working without churn.
    sed_dose_by_hr_of_day = mo.sql(
        f"""
        -- Matched-cohort patient-day filter: only days where BOTH shifts had
        -- nonzero coverage. This matches the row set used by the per-patient-
        -- day mean (which drops single-shift days where rate-diff is NaN).
        -- The JOIN below applies this filter to the matched aggregations.
        WITH matched_pdays AS (
            SELECT hospitalization_id, _nth_day
            FROM sed_dose_daily
            WHERE n_hours_day > 0 AND n_hours_night > 0
        )
        , full_agg AS (
            FROM sed_dose_by_hr
            SELECT _hr
                -- On-drug-only means (denominator = patient-hours with dose > 0).
                -- Propofol divided by 60 to convert from mcg/kg/hr (upstream
                -- parquet's native unit, set in 02_exposure.py line 366) to
                -- mcg/kg/min (clinician-preferred, matches _shared.DRUG_UNITS).
                , propofol_mcg_kg_min_mean_on_drug: AVG(CASE WHEN prop_mcg_kg_total > 0 THEN prop_mcg_kg_total / 60.0 END)
                , fenteq_mcg_mean_on_drug:          AVG(CASE WHEN _fenteq_mcg_total  > 0 THEN _fenteq_mcg_total  END)
                , midazeq_mg_mean_on_drug:          AVG(CASE WHEN _midazeq_mg_total  > 0 THEN _midazeq_mg_total  END)
                -- Across-all-IMV means, FULL cohort (includes boundary
                -- single-shift days). Off-drug arrives as 0 from upstream.
                , propofol_mcg_kg_min_mean_all_imv: AVG(COALESCE(prop_mcg_kg_total, 0) / 60.0)
                , fenteq_mcg_mean_all_imv:          AVG(COALESCE(_fenteq_mcg_total,  0))
                , midazeq_mg_mean_all_imv:          AVG(COALESCE(_midazeq_mg_total,  0))
                , n_on_drug_propofol: COUNT(*) FILTER (WHERE prop_mcg_kg_total > 0)
                , n_on_drug_fenteq:   COUNT(*) FILTER (WHERE _fenteq_mcg_total  > 0)
                , n_on_drug_midazeq:  COUNT(*) FILTER (WHERE _midazeq_mg_total  > 0)
                , n_imv: COUNT(*)
                -- Legacy column aliases — match the OLD `AVG(...)` semantics
                -- (which included zero-dose IMV-hours at 0). Propofol legacy
                -- column was `propofol_mcg_kg` — renamed to
                -- `propofol_mcg_kg_min` to be explicit about the unit.
                , propofol_mcg_kg_min: AVG(COALESCE(prop_mcg_kg_total, 0) / 60.0)
                , _fenteq_mcg:         AVG(COALESCE(_fenteq_mcg_total,  0))
                , _midazeq_mg:         AVG(COALESCE(_midazeq_mg_total,  0))
            GROUP BY _hr
        )
        , matched_agg AS (
            -- Across-all-IMV means restricted to MATCHED cohort: only
            -- patient-hours from full-coverage non-single-shift days.
            FROM sed_dose_by_hr h
            JOIN matched_pdays m USING (hospitalization_id, _nth_day)
            SELECT _hr
                , propofol_mcg_kg_min_mean_matched: AVG(COALESCE(h.prop_mcg_kg_total, 0) / 60.0)
                , fenteq_mcg_mean_matched:          AVG(COALESCE(h._fenteq_mcg_total,  0))
                , midazeq_mg_mean_matched:          AVG(COALESCE(h._midazeq_mg_total,  0))
                , n_imv_matched: COUNT(*)
            GROUP BY _hr
        )
        FROM full_agg f
        LEFT JOIN matched_agg m USING (_hr)
        SELECT *
        ORDER BY _hr
        """
    )
    _df = sed_dose_by_hr_of_day.df()
    _hr_csv_path = f'output_to_share/{SITE_NAME}/descriptive/sed_dose_by_hr_of_day.csv'
    _df.to_csv(_hr_csv_path, index=False)
    logger.info(f"Saved {_hr_csv_path} ({len(_df.columns)} columns)")
    return (sed_dose_by_hr_of_day,)


@app.cell
def _(SITE_NAME, sed_dose_by_hr_of_day):
    import matplotlib.pyplot as plt
    import numpy as np

    _df = sed_dose_by_hr_of_day.df()

    # Reorder x-axis so the day shift (7am-7pm) leads and the night shift
    # (7pm-7am) follows: 7, 8, ..., 23, 0, 1, ..., 6. The vertical red
    # 7-PM cutoff sits exactly at index 12 in the reordered axis.
    desired_order = list(range(7, 24)) + list(range(0, 7))
    _df = _df.set_index('_hr').reindex(desired_order).reset_index()
    x = np.arange(len(desired_order))

    # Per-sedative palette + axis labels. Names match _shared.DRUG_*
    # conventions even though this cell doesn't import them (07_descriptive
    # is a marimo notebook predating the descriptive layer's _shared.py).
    # Tuple: (display_label, mean-column stem, denominator-column key, color, unit).
    sedatives = [
        ("Propofol",       "propofol_mcg_kg_min", "propofol", "skyblue",         "mcg/kg/min"),
        ("Fentanyl eq.",   "fenteq_mcg",          "fenteq",   "salmon",          "mcg/hr"),
        ("Midazolam eq.",  "midazeq_mg",          "midazeq",  "mediumseagreen",  "mg/hr"),
    ]

    # 2 rows × 3 cols. Top: two mean curves per sedative. Bottom: on-drug
    # fraction bars (denominator transparency). Shared x-axis. Single
    # figure-level legend at the bottom so it doesn't overlap the curves.
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1.2]},
    )

    LABEL_ALL_IMV  = "Mean across all IMV-hours, FULL cohort (off-drug = 0)  ← legacy curve"
    LABEL_MATCHED  = "Mean across all IMV-hours, MATCHED cohort (full-coverage days only)"
    LABEL_ON_DRUG  = "Mean among on-drug hours only  (excludes zeros)"

    for col_i, (label, stem, denom_key, color, unit) in enumerate(sedatives):
        ax_top = axes[0, col_i]
        ax_bot = axes[1, col_i]

        on_drug = _df[f'{stem}_mean_on_drug'].to_numpy()
        all_imv = _df[f'{stem}_mean_all_imv'].to_numpy()
        matched = _df[f'{stem}_mean_matched'].to_numpy()
        n_on_drug = _df[f'n_on_drug_{denom_key}'].to_numpy()
        n_imv = _df['n_imv'].to_numpy()
        on_drug_pct = 100.0 * n_on_drug / np.where(n_imv > 0, n_imv, 1)

        # Top: three overlaid mean curves.
        # Solid heavier line: among-on-drug-only (highest magnitude — active
        # drip rate among patients receiving the drug).
        # Dashed: across-all-IMV, FULL cohort (legacy curve, includes boundary
        # single-shift days; off-drug at 0).
        # Dotted: across-all-IMV, MATCHED cohort (full-coverage non-partial-
        # shift days only — same row set as patient-day-mean). The dashed-
        # vs-dotted gap = the single-shift boundary day contribution.
        ax_top.plot(x, on_drug, color=color, linewidth=2.0, marker='o',
                    markersize=4, label=LABEL_ON_DRUG)
        ax_top.plot(x, all_imv, color=color, linewidth=1.6, linestyle='--',
                    marker='s', markersize=3.5, alpha=0.85,
                    label=LABEL_ALL_IMV)
        ax_top.plot(x, matched, color=color, linewidth=1.6, linestyle=':',
                    marker='^', markersize=3.5, alpha=0.85,
                    label=LABEL_MATCHED)
        ax_top.set_ylabel(f'{label}\n({unit})')
        ax_top.set_title(label, fontsize=11)
        ax_top.grid(True, axis='y', alpha=0.3)
        # Red 7-PM vline at the day→night boundary (index 12 in reordered).
        ax_top.axvline(11.5, color='red', linestyle='--', linewidth=1.5, alpha=0.75)
        ax_top.set_ylim(bottom=0)

        # Bottom: on-drug fraction bar (denominator transparency).
        ax_bot.bar(x, on_drug_pct, color=color, width=0.7, alpha=0.7,
                   edgecolor='white', linewidth=0.5)
        ax_bot.set_ylabel('% IMV-hours\non this sedative')
        ax_bot.set_xlabel('Hour of day (clock)')
        ax_bot.set_ylim(0, 100)
        ax_bot.axvline(11.5, color='red', linestyle='--', linewidth=1.5, alpha=0.75)
        ax_bot.grid(True, axis='y', alpha=0.3)

    # X-tick labels show clock hours in reorder.
    for ax in axes[1, :]:
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in desired_order], fontsize=8)

    # Single figure-level legend at the bottom (between top panels and
    # bottom panels). Built from a synthetic gray-line proxy so the user
    # reads the line *style* (solid vs dashed) without needing to map
    # back to a specific sedative color.
    legend_handles = [
        plt.Line2D([0], [0], color="dimgray", linewidth=2.0, marker='o',
                   markersize=5, label=LABEL_ON_DRUG),
        plt.Line2D([0], [0], color="dimgray", linewidth=1.6, linestyle='--',
                   marker='s', markersize=4.5, alpha=0.85, label=LABEL_ALL_IMV),
        plt.Line2D([0], [0], color="dimgray", linewidth=1.6, linestyle=':',
                   marker='^', markersize=4.5, alpha=0.85, label=LABEL_MATCHED),
        plt.Line2D([0], [0], color="red", linestyle='--', linewidth=1.5,
                   alpha=0.75, label="7 PM (day→night boundary)"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center",
        bbox_to_anchor=(0.5, 0.39), ncol=2, frameon=False, fontsize=9,
    )

    fig.suptitle(
        f'Sedative dose by hour of day — {SITE_NAME.upper()}\n'
        'Top row: solid = on-drug-only mean. Dashed = across-all-IMV (FULL cohort, includes boundary '
        'single-shift days). Dotted = across-all-IMV (MATCHED cohort, full-coverage days only — '
        'same rows as the patient-day-mean uses). Bottom row: % of IMV patient-hours on this sedative.',
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    # Reserve mid-figure room for the legend between top and bottom rows.
    fig.subplots_adjust(hspace=0.65)
    fig.text(
        0.5, -0.02,
        'The dashed-vs-dotted gap shows what fraction of the legacy hour-of-day curve is driven by partial-'
        'shift boundary patient-hours (intub-after-7-PM contributes night-hours with no day-shift offset). '
        'The matched-cohort dotted curve uses the SAME row set as `paradox_summary_6group.png`\'s patient-'
        'day mean — for full-coverage days, hour-pooled and patient-day-mean are algebraically identical. '
        'Glossary: docs/descriptive_figures.md §3 + worked example in §6.2.',
        ha='center', va='top', fontsize=8, color='dimgray', wrap=True,
    )

    _hr_png_path = f'output_to_share/{SITE_NAME}/descriptive/sed_dose_by_hr_of_day.png'
    plt.savefig(_hr_png_path, bbox_inches='tight', dpi=250)
    logger.info(f"Saved {_hr_png_path}")
    fig
    return


if __name__ == "__main__":
    app.run()
