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

    # Site-scoped output dirs (see Makefile SITE= flag).
    os.makedirs(f"output_to_share/{SITE_NAME}", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, apply_outlier_handling, duckdb, json, pd, tableone


@app.cell
def _(SITE_NAME, pd):
    analytical_df = pd.read_parquet(f"output/{SITE_NAME}/analytical_dataset.parquet")
    print(f"Analytical dataset: {len(analytical_df)} rows, {analytical_df['hospitalization_id'].nunique()} hospitalizations")
    return (analytical_df,)


@app.cell
def _(SITE_NAME, pd):
    sed_dose_agg = pd.read_parquet(f"output/{SITE_NAME}/sed_dose_agg.parquet")
    print(f"sed_dose_agg: {len(sed_dose_agg)} rows")
    return (sed_dose_agg,)


@app.cell
def _(SITE_NAME, pd):
    covs_shift = pd.read_parquet(f"output/{SITE_NAME}/covariates_shift.parquet")
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
            , s.prop_mg_total
            , s.fenteq_mcg_total
            , s.midazeq_mg_total
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
    n_hospitalizations = _df['hospitalization_id'].nunique()
    n_unique_patients = _df['patient_id'].nunique()
    pd.DataFrame({
        'site': [SITE_NAME],
        'n_hospitalizations': [n_hospitalizations],
        'n_unique_patients': [n_unique_patients],
    }).to_csv(f'output_to_share/{SITE_NAME}/cohort_stats.csv', index=False)
    print(f"Cohort: {n_hospitalizations} hospitalizations from {n_unique_patients} unique patients")
    return


@app.cell
def _(SITE_NAME, cohort_merged_for_t1, hosp_df, tableone):
    # Pull patient_id from hosp_df so we can report unique-patient count alongside Table 1
    _df = cohort_merged_for_t1.df().merge(
        hosp_df[['hospitalization_id', 'patient_id']],
        on='hospitalization_id',
        how='left',
    )
    # Binary var relabeling: convert 0/1 to No/Yes so the row labels in Table 1 read like
    # clinical paper conventions rather than "0" / "1". Combined with order+limit below,
    # this collapses each binary var to a single "Yes" row (the 0 row is redundant).
    _df['ever_pressor'] = _df['ever_pressor'].map({0: 'No', 1: 'Yes'})
    _df['sepsis_ase'] = _df['sepsis_ase'].map({0: 'No', 1: 'Yes'})

    n_hosp = _df['hospitalization_id'].nunique()
    n_pat = _df['patient_id'].nunique()
    print(f"Table 1 cohort: N={n_hosp} hospitalizations from {n_pat} unique patients")

    # NEW Table 1: standard ICU epidemiology baseline characteristics
    _cont_vars = [
        'age',
        'bmi',
        'cci_score',
        'sofa_1st24h',
        'pf_1st24h_min',
        'imv_duration_hrs',
    ]
    # Vars to display as median [Q1, Q3] instead of mean (SD).
    # cci/sofa are integer + right-skewed; pf is heavily skewed (ARDS tail);
    # imv duration has 24h floor + long right tail. age/bmi stay mean (SD).
    _nonnormal_vars = [
        'cci_score',
        'sofa_1st24h',
        'pf_1st24h_min',
        'imv_duration_hrs',
    ]
    _cat_vars = [
        'sex_category',
        'icu_type',
        'ever_pressor',
        'sepsis_ase',
    ]

    # OLD Table 1 vars (kept for easy reactivation, do not delete):
    # outcome_vars = ['_sbt_done_today', '_success_extub_today']
    # diff_doses = ['prop_dif', 'fenteq_dif', 'midazeq_dif']
    # _cont_vars = ['age', 'sofa_total', 'cci_score'] + diff_doses
    # _cat_vars = outcome_vars + ['sex_category', 'icu_type']

    table1_overall = tableone.TableOne(
        data=_df,
        continuous=_cont_vars,
        categorical=_cat_vars,
        nonnormal=_nonnormal_vars,
        # Show only the "Yes" row for binary vars (the "No" row is redundant since
        # it's simply total - yes). `order` puts Yes first, `limit=1` drops the rest.
        # Verified via tableone/formatting.py:89-128 that apply_limits respects order.
        order={'ever_pressor': ['Yes', 'No'], 'sepsis_ase': ['Yes', 'No']},
        limit={'ever_pressor': 1, 'sepsis_ase': 1},
    )
    _t1_path = f'output_to_share/{SITE_NAME}/table1.csv'
    table1_overall.to_csv(_t1_path)
    print(f"Saved {_t1_path}")
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
    sed_vars = ['prop_mg_total', 'fenteq_mcg_total', 'midazeq_mg_total']
    _cont_vars = sed_vars + ['ph', 'pf', '_nee']
    _cat_vars = ['ph_level', 'pf_level']
    table1_by_shift = tableone.TableOne(
        data=_df, continuous=_cont_vars, groupby='_shift', categorical=_cat_vars, pval=True
    )
    # ARCHIVED 2026-04-08: replaced by the paired+unpaired per-patient rate
    # analysis in code/07_descriptive.py (outputs sed_dose_by_shift.csv).
    # The tableone computation above is kept intact for git history / easy
    # restore; only the CSV write is disabled so the outdated
    # output_to_share/table1_by_shift.csv is no longer regenerated.
    # table1_by_shift.to_csv('output_to_share/table1_by_shift.csv')
    # print("Saved output_to_share/table1_by_shift.csv")
    return


# TODO: Add structured JSON export for cross-site aggregation (table1.json)


if __name__ == "__main__":
    app.run()
