# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
#     "tableone>=0.9.5",
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
    # 06 Table One

    Generates Table 1 summaries: overall (day 1) and by shift (day vs night).
    Outputs CSV + JSON dual format for cross-site aggregation.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    from clifpy import Hospitalization
    import pandas as pd
    import duckdb
    import tableone
    import json

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    os.makedirs("output_to_share", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, apply_outlier_handling, duckdb, json, pd, tableone


@app.cell
def _(pd):
    analytical_df = pd.read_parquet("output/analytical_dataset.parquet")
    print(f"Analytical dataset: {len(analytical_df)} rows, {analytical_df['hospitalization_id'].nunique()} hospitalizations")
    return (analytical_df,)


@app.cell
def _(pd):
    sed_dose_agg = pd.read_parquet("output/sed_dose_agg.parquet")
    print(f"sed_dose_agg: {len(sed_dose_agg)} rows")
    return (sed_dose_agg,)


@app.cell
def _(pd):
    covs_shift = pd.read_parquet("output/covariates_shift.parquet")
    print(f"covs_shift: {len(covs_shift)} rows")
    return (covs_shift,)


@app.cell
def _(CONFIG_PATH, apply_outlier_handling):
    hosp = Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'hospitalization_id', 'discharge_dttm', 'discharge_category', 'age_at_admission'],
    )
    apply_outlier_handling(hosp, outlier_config_path='config/outlier_config.yaml')
    hosp_df = hosp.df
    return (hosp_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Filter to Day 1
    """)
    return


@app.cell
def _(analytical_df):
    cohort_merged_for_t1 = mo.sql(
        f"""
        -- Day 1 rows only for Table 1
        FROM analytical_df
        SELECT *
        WHERE _nth_day = 1
        """
    )
    return (cohort_merged_for_t1,)


@app.cell
def _(cohort_merged_for_t1, covs_shift, hosp_df, sed_dose_agg):
    cohort_merged_for_t1_w_by_shift = mo.sql(
        f"""
        -- Join day-1 with shift-level doses and covariates for by-shift Table 1
        WITH t1 AS (
        FROM cohort_merged_for_t1 g
        LEFT JOIN sed_dose_agg s USING (hospitalization_id, _nth_day)
        LEFT JOIN hosp_df h USING (hospitalization_id)
        SELECT g.hospitalization_id, g._nth_day
            , s._shift
            , s.propofol_mg_total
            , s.fentanyl_eq_mcg_total
            , s.midazolam_eq_mg_total
            , h.patient_id
        )
        , t2 AS (
        FROM t1
        LEFT JOIN covs_shift c USING (hospitalization_id, _nth_day, _shift)
        SELECT *
        )
        SELECT *
        FROM t2
        ORDER BY hospitalization_id, _nth_day, _shift
        """
    )
    return (cohort_merged_for_t1_w_by_shift,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Generate Tables
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_for_t1_w_by_shift, pd):
    _df = cohort_merged_for_t1_w_by_shift.df()
    n_unique_patients = _df['patient_id'].nunique()
    pd.DataFrame({'n_unique_patients': [n_unique_patients]})\
        .to_csv('output_to_share/cohort_stats.csv', index=False)
    print(f"Unique patients: {n_unique_patients}")
    return


@app.cell
def _(SITE_NAME, cohort_merged_for_t1, tableone):
    _df = cohort_merged_for_t1.df()
    outcome_vars = ['_sbt_done_today', '_success_extub_today']
    diff_doses = ['propofol_diff', 'fentanyl_eq_diff', 'midazolam_eq_diff']
    _cont_vars = ['age'] + diff_doses
    _cat_vars = outcome_vars
    table1_overall = tableone.TableOne(data=_df, continuous=_cont_vars, categorical=_cat_vars)
    table1_overall.to_csv('output_to_share/table1.csv')
    print("Saved output_to_share/table1.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Day-Night Comparison Table
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_for_t1_w_by_shift, tableone):
    _df = cohort_merged_for_t1_w_by_shift.df()
    sed_vars = ['propofol_mg_total', 'fentanyl_eq_mcg_total', 'midazolam_eq_mg_total']
    _cont_vars = sed_vars + ['ph', 'pf', '_nee']
    _cat_vars = ['ph_level', 'pf_level']
    table1_by_shift = tableone.TableOne(
        data=_df, continuous=_cont_vars, groupby='_shift', categorical=_cat_vars, pval=True
    )
    table1_by_shift.to_csv('output_to_share/table1_by_shift.csv')
    print("Saved output_to_share/table1_by_shift.csv")
    return


# TODO: Add structured JSON export for cross-site aggregation (table1.json)


if __name__ == "__main__":
    app.run()
