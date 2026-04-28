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
    os.makedirs(f"output_to_share/{SITE_NAME}/figures", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, pd


@app.cell
def _(SITE_NAME, pd):
    cohort_merged_final = pd.read_parquet(f"output/{SITE_NAME}/analytical_dataset.parquet")
    print(f"Analytical dataset: {len(cohort_merged_final)} rows")
    return (cohort_merged_final,)


@app.cell
def _(SITE_NAME, pd):
    sed_dose_by_hr = pd.read_parquet(f"output/{SITE_NAME}/sed_dose_by_hr.parquet")
    print(f"sed_dose_by_hr: {len(sed_dose_by_hr)} rows")
    return (sed_dose_by_hr,)


@app.cell
def _(SITE_NAME, pd):
    # sed_dose_daily: one row per (hospitalization_id, _nth_day) with day/night
    # shift totals AND n_hours_day/n_hours_night (added in 02_exposure.py).
    # Used by the "Dose by Shift" cell below for per-patient hourly-rate
    # computation that correctly handles partial-shift bias.
    sed_dose_daily = pd.read_parquet(f"output/{SITE_NAME}/sed_dose_daily.parquet")
    print(f"sed_dose_daily: {len(sed_dose_daily)} rows")
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
    _corr_path = f'output_to_share/{SITE_NAME}/pairwise_corr_matrix.csv'
    corr_matrix.to_csv(_corr_path)
    print(f"Saved {_corr_path}")
    # PNG companion to the CSV — used to visually scan for collinearity blocks
    # when investigating counterintuitive coefficient signs (e.g., the
    # `_dif_*` ↔ `_day_*` linear-combination structure).
    _corr_png = _corr_path.replace('.csv', '.png')
    _plt.savefig(_corr_png, dpi=120, bbox_inches='tight')
    print(f"Saved {_corr_png}")
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
    # This correctly handles partial-shift bias (extubation day has <12h
    # on the last shift) — totals would be artificially low for the partial
    # side. Rates are directly comparable across patients regardless of
    # IMV duration.
    #
    # Primary test: paired t-test (each patient contributes both night and
    # day rates; paired removes between-patient variance). Unpaired Welch's
    # t-test shown as a robustness check — it ignores the pairing and is
    # therefore more conservative.
    from scipy import stats as _stats

    # Cohort hosp IDs from analytical_dataset (post-NMB exclusion)
    _cohort_hosp_ids = cohort_merged_final['hospitalization_id'].unique()

    # Aggregate per patient: sum totals AND hours across all days
    _sed_filtered = sed_dose_daily[
        sed_dose_daily['hospitalization_id'].isin(_cohort_hosp_ids)
    ]
    _per_pt = (
        _sed_filtered.groupby('hospitalization_id')
        .agg({
            'fenteq_day_mcg': 'sum', 'fenteq_night_mcg': 'sum',
            'midazeq_day_mg': 'sum', 'midazeq_night_mg': 'sum',
            # Phase 2: propofol totals are now in mcg/kg.
            'prop_day_mcg_kg': 'sum', 'prop_night_mcg_kg': 'sum',
            'n_hours_day': 'sum', 'n_hours_night': 'sum',
        })
        .reindex(_cohort_hosp_ids)
        .fillna(0)
    )

    def _safe_rate(num, denom):
        """num/denom with 0 for denom==0 (avoids NaN; shouldn't occur in practice)."""
        return num.where(denom > 0, other=0) / denom.where(denom > 0, other=1)

    _per_pt['fenteq_rate_night'] = _safe_rate(_per_pt['fenteq_night_mcg'], _per_pt['n_hours_night'])
    _per_pt['fenteq_rate_day']   = _safe_rate(_per_pt['fenteq_day_mcg'],   _per_pt['n_hours_day'])
    _per_pt['midazeq_rate_night'] = _safe_rate(_per_pt['midazeq_night_mg'], _per_pt['n_hours_night'])
    _per_pt['midazeq_rate_day']   = _safe_rate(_per_pt['midazeq_day_mg'],   _per_pt['n_hours_day'])
    # Phase 2: propofol rate in mcg/kg/min (sum mcg/kg / hours / 60).
    _per_pt['prop_rate_night'] = _safe_rate(_per_pt['prop_night_mcg_kg'], _per_pt['n_hours_night']) / 60.0
    _per_pt['prop_rate_day']   = _safe_rate(_per_pt['prop_day_mcg_kg'],   _per_pt['n_hours_day']) / 60.0

    print(f"Per-patient rates: {len(_per_pt)} hospitalizations")

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
                'spec': _spec_name,
                'Nighttime (7p-7a), mean (SD)': _fmt_mean_sd(_night.mean(), _night.std(ddof=1)),
                'Daytime (7a-7p), mean (SD)':   _fmt_mean_sd(_day.mean(), _day.std(ddof=1)),
                'mean difference (95% CI)':      _fmt_diff_ci(_mean_diff, _lo, _hi),
                'p-value':                       _fmt_p(_p),
            })
    sed_dose_by_shift = pd.DataFrame(_rows)
    _shift_path = f'output_to_share/{SITE_NAME}/sed_dose_by_shift.csv'
    sed_dose_by_shift.to_csv(_shift_path, index=False)
    print(f"Saved {_shift_path}")
    print(sed_dose_by_shift.to_string(index=False))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly Dose Visualization
    """)
    return


@app.cell
def _(SITE_NAME, sed_dose_by_hr):
    sed_dose_by_hr_of_day = mo.sql(
        f"""
        -- Average dose by hour of day
        FROM sed_dose_by_hr
        SELECT _hr
        -- Phase 2: prop_mcg_kg_total is mcg/kg delivered in the hour
        -- (= mcg/kg/hr rate). AVG across patients is mcg/kg/hr.
        , propofol_mcg_kg: AVG(prop_mcg_kg_total)
        , _fenteq_mcg: AVG(_fenteq_mcg_total)
        , _midazeq_mg: AVG(_midazeq_mg_total)
        GROUP BY _hr
        ORDER BY _hr
        """
    )
    _df = sed_dose_by_hr_of_day.df()
    _hr_csv_path = f'output_to_share/{SITE_NAME}/sed_dose_by_hr_of_day.csv'
    _df.to_csv(_hr_csv_path, index=False)
    print(f"Saved {_hr_csv_path}")
    return (sed_dose_by_hr_of_day,)


@app.cell
def _(SITE_NAME, sed_dose_by_hr_of_day):
    import matplotlib.pyplot as plt
    import numpy as np

    _df = sed_dose_by_hr_of_day.df()
    hours = _df['_hr']
    propofol = _df['propofol_mcg_kg']
    fenteq = _df['_fenteq_mcg']
    midazeq = _df['_midazeq_mg']

    # Reorder so that x-axis goes from 7,8,...,23,0,1,...,6
    desired_order = list(range(7, 24)) + list(range(0, 7))
    hours_ordered = []
    propofol_ordered = []
    fenteq_ordered = []
    midazeq_ordered = []

    for h in desired_order:
        if h in list(hours):
            idx = list(hours).index(h)
            hours_ordered.append(hours.iloc[idx])
            propofol_ordered.append(propofol.iloc[idx])
            fenteq_ordered.append(fenteq.iloc[idx])
            midazeq_ordered.append(midazeq.iloc[idx])

    hours_ordered = np.array(hours_ordered)
    propofol_ordered = np.array(propofol_ordered)
    fenteq_ordered = np.array(fenteq_ordered)
    midazeq_ordered = np.array(midazeq_ordered)

    fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    x = np.arange(len(hours_ordered))
    bar_width = 0.6

    axs[0].bar(x, propofol_ordered, color='skyblue', width=bar_width)
    axs[0].set_ylabel('Propofol (mcg/kg/hr)')
    axs[0].set_title('Mean Propofol Rate by Hour of Day (continuous infusion)')
    axs[0].grid(True, axis='y')

    axs[1].bar(x, fenteq_ordered, color='salmon', width=bar_width)
    axs[1].set_ylabel('Fentanyl Eq (mcg)')
    axs[1].set_title('Mean Total Fentanyl Equivalent Dose by Hour of Day')
    axs[1].grid(True, axis='y')

    axs[2].bar(x, midazeq_ordered, color='mediumseagreen', width=bar_width)
    axs[2].set_ylabel('Midazolam Eq (mg)')
    axs[2].set_title('Mean Total Midazolam Equivalent Dose by Hour of Day')
    axs[2].set_xlabel('Hour of Day')
    axs[2].grid(True, axis='y')

    for ax in axs:
        for cutoff in [19]:
            if cutoff in hours_ordered:
                cutoff_pos = np.where(hours_ordered == cutoff)[0][0]
                ax.axvline(cutoff_pos - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.8)
            else:
                insert_pos = np.searchsorted(hours_ordered, cutoff)
                ax.axvline(insert_pos - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.8)

    plt.xticks(x, hours_ordered.astype(int))
    plt.tight_layout()
    plt.suptitle(f'Cumulative Sedative Doses by Hour of Day — {SITE_NAME}', fontsize=18, y=1.04)
    _hr_png_path = f'output_to_share/{SITE_NAME}/figures/sed_dose_by_hr_of_day.png'
    plt.savefig(_hr_png_path, bbox_inches='tight', dpi=250)
    print(f"Saved {_hr_png_path}")
    fig
    return


if __name__ == "__main__":
    app.run()
