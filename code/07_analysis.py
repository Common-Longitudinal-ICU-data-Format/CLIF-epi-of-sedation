# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
#     "statsmodels>=0.14.5",
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
    # 07 Analysis

    GEE, logistic regression, correlation matrix, and visualizations.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    import pandas as pd

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    os.makedirs("output_to_share/figures", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, pd


@app.cell
def _(pd):
    cohort_merged_final = pd.read_parquet("output/analytical_dataset.parquet")
    print(f"Analytical dataset: {len(cohort_merged_final)} rows")
    return (cohort_merged_final,)


@app.cell
def _(pd):
    sed_dose_by_hr = pd.read_parquet("output/sed_dose_by_hr.parquet")
    print(f"sed_dose_by_hr: {len(sed_dose_by_hr)} rows")
    return (sed_dose_by_hr,)


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
        'age', 'propofol_diff', 'fentanyl_eq_diff', 'midazolam_eq_diff',
        '_propofol_day', '_propofol_night', '_fentanyl_eq_day', '_fentanyl_eq_night',
        '_midazolam_eq_day', '_midazolam_eq_night', 'nee_7am', 'nee_7pm',
        '_ph_7am', '_ph_7pm', '_pf_7am', '_pf_7pm',
    ]
    continuous_vars_df = cohort_merged_final[[col for col in continuous_vars if col in cohort_merged_final.columns]]
    corr_matrix = continuous_vars_df.corr(method='pearson')
    _plt.figure(figsize=(14, 10))
    _sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
    _plt.title('Pairwise Pearson Correlation Matrix (Continuous Variables)')
    _plt.tight_layout()
    corr_matrix.to_csv('output_to_share/pairwise_corr_matrix.csv')
    print("Saved output_to_share/pairwise_corr_matrix.csv")
    _plt.gcf()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Day vs Night T-tests
    """)
    return


@app.cell
def _(SITE_NAME, pd, sed_dose_by_hr):
    import scipy.stats as stats
    test_cols = ['propofol_mg_total', '_fentanyl_eq_mcg_total', '_midazolam_eq_mg_total']
    shift_day = sed_dose_by_hr[sed_dose_by_hr['_shift'] == 'day']
    shift_night = sed_dose_by_hr[sed_dose_by_hr['_shift'] == 'night']
    t_pvals = {}
    for col in test_cols:
        tstat, pval = stats.ttest_ind(shift_day[col], shift_night[col], nan_policy='omit', equal_var=False)
        t_pvals[col] = pval

    sed_dose_by_shift = sed_dose_by_hr.groupby('_shift')[test_cols].mean().reset_index()
    sed_dose_by_shift.rename(columns={'_shift': 'shift'}, inplace=True)
    pval_row = pd.Series({'shift': 'ttest_pval', **t_pvals})
    out_df = pd.concat([sed_dose_by_shift, pd.DataFrame([pval_row])], ignore_index=True)
    out_df.to_csv('output_to_share/sed_dose_by_shift.csv', index=False)
    print("Saved output_to_share/sed_dose_by_shift.csv")
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
        , propofol_mg: AVG(propofol_mg_total)
        , _fentanyl_eq_mcg: AVG(_fentanyl_eq_mcg_total)
        , _midazolam_eq_mg: AVG(_midazolam_eq_mg_total)
        GROUP BY _hr
        ORDER BY _hr
        """
    )
    _df = sed_dose_by_hr_of_day.df()
    _df.to_csv('output_to_share/sed_dose_by_hr_of_day.csv', index=False)
    print("Saved output_to_share/sed_dose_by_hr_of_day.csv")
    return (sed_dose_by_hr_of_day,)


@app.cell
def _(SITE_NAME, sed_dose_by_hr_of_day):
    import matplotlib.pyplot as plt
    import numpy as np

    _df = sed_dose_by_hr_of_day.df()
    hours = _df['_hr']
    propofol = _df['propofol_mg']
    fentanyl_eq = _df['_fentanyl_eq_mcg']
    midazolam_eq = _df['_midazolam_eq_mg']

    # Reorder so that x-axis goes from 7,8,...,23,0,1,...,6
    desired_order = list(range(7, 24)) + list(range(0, 7))
    hours_ordered = []
    propofol_ordered = []
    fentanyl_eq_ordered = []
    midazolam_eq_ordered = []

    for h in desired_order:
        if h in list(hours):
            idx = list(hours).index(h)
            hours_ordered.append(hours.iloc[idx])
            propofol_ordered.append(propofol.iloc[idx])
            fentanyl_eq_ordered.append(fentanyl_eq.iloc[idx])
            midazolam_eq_ordered.append(midazolam_eq.iloc[idx])

    hours_ordered = np.array(hours_ordered)
    propofol_ordered = np.array(propofol_ordered)
    fentanyl_eq_ordered = np.array(fentanyl_eq_ordered)
    midazolam_eq_ordered = np.array(midazolam_eq_ordered)

    fig, axs = plt.subplots(3, 1, figsize=(13, 12), sharex=True)

    x = np.arange(len(hours_ordered))
    bar_width = 0.6

    axs[0].bar(x, propofol_ordered, color='skyblue', width=bar_width)
    axs[0].set_ylabel('Propofol (mg)')
    axs[0].set_title('Mean Total Propofol Dose by Hour of Day')
    axs[0].grid(True, axis='y')

    axs[1].bar(x, fentanyl_eq_ordered, color='salmon', width=bar_width)
    axs[1].set_ylabel('Fentanyl Eq (mcg)')
    axs[1].set_title('Mean Total Fentanyl Equivalent Dose by Hour of Day')
    axs[1].grid(True, axis='y')

    axs[2].bar(x, midazolam_eq_ordered, color='mediumseagreen', width=bar_width)
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
    plt.savefig('output_to_share/figures/sed_dose_by_hr_of_day.png', bbox_inches='tight')
    print("Saved output_to_share/figures/sed_dose_by_hr_of_day.png")
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## GEE: SBT Done Next Day
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    import statsmodels.api as _sm
    sbt_done_formula = """sbt_done_next_day ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm + age
    """
    gee_model = _smf.gee(formula=sbt_done_formula, groups='hospitalization_id', data=cohort_merged_final, family=_sm.families.Binomial())
    gee_result = gee_model.fit()
    print(gee_result.summary())
    _summary_df = gee_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv('output_to_share/gee_summary.csv', index=False)
    _cov_matrix = gee_result.cov_params()
    _cov_matrix.to_csv('output_to_share/gee_covmat.csv')
    print("Saved output_to_share/gee_summary.csv, gee_covmat.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logistic Regression: Successful Extubation Next Day
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    success_extub_formula = """success_extub_next_day ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm + age
    """
    logit_model = _smf.logit(formula=success_extub_formula, data=cohort_merged_final)
    logit_result = logit_model.fit(cov_type='cluster', cov_kwds={'groups': cohort_merged_final['hospitalization_id']})
    print(logit_result.summary())
    _summary_df = logit_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv('output_to_share/logit_summary.csv', index=False)
    _cov_matrix = logit_result.cov_params()
    _cov_matrix.to_csv('output_to_share/logit_covmat.csv')
    print("Saved output_to_share/logit_summary.csv, logit_covmat.csv")
    return


if __name__ == "__main__":
    app.run()
