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
    # Path B++ refactor: modeling outputs live under {site}/models/ so that
    # descriptive (night-vs-day) artifacts and model artifacts sit in
    # parallel thematic subdirs.
    os.makedirs(f"output_to_share/{SITE_NAME}/models", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, SITE_NAME, apply_outlier_handling, duckdb, json, pd, tableone


@app.cell
def _(SITE_NAME, pd):
    analytical_df = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")
    print(f"Modeling dataset: {len(analytical_df)} rows, {analytical_df['hospitalization_id'].nunique()} hospitalizations")
    return (analytical_df,)


@app.cell
def _(SITE_NAME):
    # Phase 2 (2026-05-07): `sed_dose_agg.parquet` was deleted as a redundant
    # within-script intermediate of `02_exposure.py`. Its long-format
    # (hosp, day, shift) per-shift dose totals are now derived on-the-fly
    # by UNPIVOTing the wide-form `seddose_by_id_imvday.parquet` (separate
    # `*_day_*` / `*_night_*` columns). Same variable name preserved so
    # downstream cells continue to read `sed_dose_agg` from the cross-cell
    # DAG without any join changes.
    sed_dose_agg = mo.sql(f"""
        FROM read_parquet('output/{SITE_NAME}/seddose_by_id_imvday.parquet')
        SELECT hospitalization_id, _nth_day
            , _shift: 'day'
            , prop_mcg_kg: prop_day_mcg_kg
            , fenteq_mcg:  fenteq_day_mcg
            , midazeq_mg:  midazeq_day_mg
            , n_hours:     n_hours_day
        UNION ALL
        FROM read_parquet('output/{SITE_NAME}/seddose_by_id_imvday.parquet')
        SELECT hospitalization_id, _nth_day
            , _shift: 'night'
            , prop_mcg_kg: prop_night_mcg_kg
            , fenteq_mcg:  fenteq_night_mcg
            , midazeq_mg:  midazeq_night_mg
            , n_hours:     n_hours_night
    """)
    return (sed_dose_agg,)


@app.cell
def _(SITE_NAME, pd):
    covs_shift = pd.read_parquet(f"output/{SITE_NAME}/covariates_by_id_shift.parquet")
    print(f"covs_shift: {len(covs_shift)} rows")
    return (covs_shift,)


@app.cell
def _(SITE_NAME, pd):
    # Per-stay registry from 04_covariates: LOS rollups + exit_mechanism.
    # Joined onto the day-1 Table 1 frame so the new rows render alongside
    # the existing baseline characteristics (one row per hospitalization).
    cohort_meta_by_id = pd.read_parquet(
        f"output/{SITE_NAME}/cohort_meta_by_id.parquet"
    )
    print(
        f"cohort_meta_by_id: {len(cohort_meta_by_id)} hospitalizations; "
        f"exit_mechanism distribution: "
        f"{cohort_meta_by_id['exit_mechanism'].value_counts().to_dict()}"
    )
    return (cohort_meta_by_id,)


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
            , s.prop_mcg_kg
            , s.fenteq_mcg
            , s.midazeq_mg
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
    }).to_csv(f'output_to_share/{SITE_NAME}/models/cohort_stats.csv', index=False)
    print(f"Cohort: {n_hospitalizations} hospitalizations from {n_unique_patients} unique patients")
    return


@app.cell
def _(SITE_NAME, cohort_merged_for_t1, cohort_meta_by_id, hosp_df, tableone):
    # Pull patient_id from hosp_df so we can report unique-patient count alongside Table 1
    _df = cohort_merged_for_t1.df().merge(
        hosp_df[['hospitalization_id', 'patient_id']],
        on='hospitalization_id',
        how='left',
    )
    # Merge the per-stay registry — adds LOS-in-full-24h-days and the
    # categorical `exit_mechanism` for rendering in Table 1.
    _df = _df.merge(
        cohort_meta_by_id[
            ['hospitalization_id', 'n_days_full_24h', 'exit_mechanism']
        ],
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
        'n_days_full_24h',
    ]
    # Vars to display as median [Q1, Q3] instead of mean (SD).
    # cci/sofa are integer + right-skewed; pf is heavily skewed (ARDS tail);
    # imv duration has 24h floor + long right tail. age/bmi stay mean (SD).
    # n_days_full_24h is integer + right-skewed (typical LOS distribution).
    _nonnormal_vars = [
        'cci_score',
        'sofa_1st24h',
        'pf_1st24h_min',
        'imv_duration_hrs',
        'n_days_full_24h',
    ]
    _cat_vars = [
        'sex_category',
        'icu_type',
        'ever_pressor',
        'sepsis_ase',
        'exit_mechanism',
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
        order={
            'ever_pressor': ['Yes', 'No'],
            'sepsis_ase': ['Yes', 'No'],
            # Display exit_mechanism in clinical-narrative order: terminal events
            # first (trach, died_on_imv), then attempted-liberation outcomes,
            # then census-end discharge, finally unknown.
            'exit_mechanism': [
                'tracheostomy',
                'died_on_imv',
                'palliative_extubation',
                'failed_extubation',
                'successful_extubation',
                'discharge_on_imv',
                'unknown',
            ],
        },
        limit={'ever_pressor': 1, 'sepsis_ase': 1},
    )
    _t1_path = f'output_to_share/{SITE_NAME}/models/table1.csv'
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
    sed_vars = ['prop_mcg_kg', 'fenteq_mcg', 'midazeq_mg']
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
