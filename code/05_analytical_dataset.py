# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "clifpy>=0.3.1",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
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
    # 05 Analytical Dataset

    Merges outcomes, exposure, covariates, and demographics.
    Computes next-day outcomes via LEAD.
    Applies NMB exclusion.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    import pandas as pd
    import duckdb

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    print(f"Site: {SITE_NAME}")
    return CONFIG_PATH, apply_outlier_handling, pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Intermediate Outputs
    """)
    return


@app.cell
def _(pd):
    sbt_outcomes_daily = pd.read_parquet("output/sbt_outcomes_daily.parquet")
    print(f"sbt_outcomes_daily: {len(sbt_outcomes_daily)} rows")
    return (sbt_outcomes_daily,)


@app.cell
def _(pd):
    sed_dose_daily = pd.read_parquet("output/sed_dose_daily.parquet")
    print(f"sed_dose_daily: {len(sed_dose_daily)} rows")
    return (sed_dose_daily,)


@app.cell
def _(pd):
    covs_daily = pd.read_parquet("output/covariates_daily.parquet")
    print(f"covs_daily: {len(covs_daily)} rows")
    return (covs_daily,)


@app.cell
def _(pd):
    nmb_excluded = pd.read_parquet("output/nmb_excluded.parquet")
    print(f"nmb_excluded patient-days: {len(nmb_excluded)}")
    return (nmb_excluded,)


@app.cell
def _(CONFIG_PATH, apply_outlier_handling):
    from clifpy import Hospitalization
    hosp = Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'hospitalization_id', 'discharge_dttm', 'discharge_category', 'age_at_admission'],
    )
    apply_outlier_handling(hosp, outlier_config_path='config/outlier_config.yaml')
    hosp_df = hosp.df
    print(f"hosp_df: {len(hosp_df)} rows")
    return (hosp_df,)


@app.cell
def _(CONFIG_PATH):
    from clifpy import Patient
    _patient = Patient.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'sex_category'],
    )
    patient_df = _patient.df
    print(f"patient_df: {len(patient_df)} rows")
    return (patient_df,)


@app.cell
def _(pd):
    sofa_daily = pd.read_parquet("output/sofa_daily.parquet")
    icu_type_df = pd.read_parquet("output/icu_type.parquet")
    cci_df = pd.read_parquet("output/cci.parquet")
    elix_df = pd.read_parquet("output/elix.parquet")
    covariates_t1 = pd.read_parquet("output/covariates_t1.parquet")
    print(
        f"sofa_daily: {len(sofa_daily)}, icu_type: {len(icu_type_df)}, "
        f"cci: {len(cci_df)}, elix: {len(elix_df)}, "
        f"covariates_t1: {len(covariates_t1)}"
    )
    return cci_df, covariates_t1, elix_df, icu_type_df, sofa_daily


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge
    """)
    return


@app.cell
def _(
    cci_df,
    covariates_t1,
    covs_daily,
    elix_df,
    hosp_df,
    icu_type_df,
    patient_df,
    sbt_outcomes_daily,
    sed_dose_daily,
    sofa_daily,
):
    cohort_merged = mo.sql(
        f"""
        -- Merge all intermediate outputs into analytical dataset.
        -- covariates_t1 is hospitalization-level (from 04_covariates Cell H) and broadcasts
        -- its values across all patient-days for each hospitalization.
        FROM sbt_outcomes_daily o
        LEFT JOIN sed_dose_daily s USING (hospitalization_id, _nth_day)
        LEFT JOIN covs_daily c USING (hospitalization_id, _nth_day)
        LEFT JOIN hosp_df h USING (hospitalization_id)
        LEFT JOIN patient_df p USING (patient_id)
        LEFT JOIN icu_type_df i USING (hospitalization_id)
        LEFT JOIN sofa_daily sf USING (hospitalization_id, _nth_day)
        LEFT JOIN cci_df cc USING (hospitalization_id)
        LEFT JOIN elix_df ex USING (hospitalization_id)
        LEFT JOIN covariates_t1 t1 USING (hospitalization_id)
        SELECT o.hospitalization_id
        , o._nth_day
        , _sbt_done_today: o.sbt_done
        , _success_extub_today: o._success_extub
        , sbt_done_next_day: LEAD(o.sbt_done) OVER w
        , success_extub_next_day: LEAD(o._success_extub) OVER w
        -- NOTE: Dose columns below are per-hour RATES (mg/hr or mcg/hr), not
        -- per-shift totals. Conversion: totals ÷ 12 (hours per shift). This is
        -- valid because the filter below (`_nth_day > 0 AND ... IS NOT NULL`)
        -- guarantees complete 24h 7am-anchored days — partial shifts at
        -- intubation / extubation are already excluded. The rate unit makes
        -- dose coefficients in 08_models.py directly comparable across
        -- patients and avoids partial-shift bias in descriptives.
        , _prop_day: COALESCE(s.prop_day, 0) / 12.0
        , _prop_night: COALESCE(s.prop_night, 0) / 12.0
        , _fenteq_day: COALESCE(s.fenteq_day, 0) / 12.0
        , _fenteq_night: COALESCE(s.fenteq_night, 0) / 12.0
        , _midazeq_day: COALESCE(s.midazeq_day, 0) / 12.0
        , _midazeq_night: COALESCE(s.midazeq_night, 0) / 12.0
        , prop_dif: (COALESCE(s.prop_night, 0) - COALESCE(s.prop_day, 0)) / 12.0
        , fenteq_dif: (COALESCE(s.fenteq_night, 0) - COALESCE(s.fenteq_day, 0)) / 12.0
        , midazeq_dif: (COALESCE(s.midazeq_night, 0) - COALESCE(s.midazeq_day, 0)) / 12.0
        , COLUMNS('(7am)|(7pm)')
        , age: h.age_at_admission
        , p.sex_category
        , i.icu_type
        , sofa_total: COALESCE(sf.sofa_total,0)
        , cci_score: COALESCE(cc.cci_score, 0)
        , elix_score: COALESCE(ex.elix_score, 0)
        -- Table 1 hospitalization-level covariates (from 04_covariates.py covariates_t1.parquet)
        , t1.bmi
        , t1.height_cm
        , t1.weight_kg
        , t1.sofa_1st24h
        , t1.sofa_cv_97_1st24h
        , t1.sofa_coag_1st24h
        , t1.sofa_liver_1st24h
        , t1.sofa_resp_1st24h
        , t1.sofa_cns_1st24h
        , t1.sofa_renal_1st24h
        , ever_pressor: COALESCE(t1.ever_pressor, 0)
        , t1.pf_1st24h_min
        , t1.pf_1st24h_source
        , t1.imv_duration_hrs
        , sepsis_ase: COALESCE(t1.sepsis_ase, 0)
        , t1._first_icu_dttm
        WINDOW w AS (PARTITION BY o.hospitalization_id ORDER BY o._nth_day)
        ORDER BY o.hospitalization_id, o._nth_day
        """
    )
    return (cohort_merged,)


@app.cell
def _(cohort_merged):
    _merged_df = cohort_merged.df()
    _merged_df.dropna(subset=['age'], inplace=True)
    cohort_merged_clean = _merged_df
    print(f"After dropping null age: {len(cohort_merged_clean)} rows")
    return (cohort_merged_clean,)


@app.cell
def _(cohort_merged_clean, nmb_excluded):
    cohort_merged_final = mo.sql(
        f"""
        -- NOTE: Hospitalization-level NMB exclusion (exclude entire hosp if any day had >1h NMB).
        -- Original spec was patient-day level; changed per PI — NMB patients are a distinct population.
        -- To revert to patient-day exclusion, replace the ANTI JOIN with:
        --   ANTI JOIN nmb_excluded USING (hospitalization_id, _nth_day)
        FROM cohort_merged_clean
        ANTI JOIN (SELECT DISTINCT hospitalization_id FROM nmb_excluded) USING (hospitalization_id)
        SELECT *
        WHERE _nth_day > 0 AND sbt_done_next_day IS NOT NULL AND success_extub_next_day IS NOT NULL
        """
    )
    return (cohort_merged_final,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save
    """)
    return


@app.cell
def _(cohort_merged_final):
    _df = cohort_merged_final.df()
    _df.to_parquet("output/analytical_dataset.parquet", index=False)
    print(f"Saved output/analytical_dataset.parquet ({len(_df)} rows, {_df['hospitalization_id'].nunique()} hospitalizations)")
    return


if __name__ == "__main__":
    app.run()
