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
    # 04 Covariates

    Computes shift-level and daily covariates: pH, P/F ratio, vasopressors/NEE.
    Reads hourly grids from 01_cohort and resp_processed for FiO2.
    Writes `output/covariates_shift.parquet` and `output/covariates_daily.parquet`.
    """)
    return


@app.cell
def _():
    from clifpy import ClifOrchestrator
    import pandas as pd
    import duckdb
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    from _utils import remove_meds_duplicates

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"
    co = ClifOrchestrator(config_path=CONFIG_PATH)

    os.makedirs("output", exist_ok=True)
    return (
        CONFIG_PATH,
        apply_outlier_handling,
        convert_dose_units_by_med_category,
        duckdb,
        get_config_or_params,
        pd,
        remove_meds_duplicates,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params):
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    print(f"Site: {SITE_NAME}")
    return (SITE_NAME,)


@app.cell
def _(pd):
    cohort_hrly_grids = pd.read_parquet("output/cohort_hrly_grids.parquet")
    print(f"Hourly grid rows: {len(cohort_hrly_grids)}")
    return (cohort_hrly_grids,)


@app.cell
def _(cohort_hrly_grids):
    cohort_shift_change_grids = cohort_hrly_grids[cohort_hrly_grids['_hr'].isin([7, 19])]
    print(f"Shift-change grid rows (7am/7pm only): {len(cohort_shift_change_grids)}")
    return (cohort_shift_change_grids,)


@app.cell
def _(cohort_hrly_grids):
    cohort_hosp_ids = cohort_hrly_grids['hospitalization_id'].unique().tolist()
    print(f"Cohort hospitalizations: {len(cohort_hosp_ids)}")
    return (cohort_hosp_ids,)


@app.cell
def _(SITE_NAME, pd):
    resp_processed_path = f"output/{SITE_NAME}_resp_processed_bf.parquet"
    resp_p = pd.read_parquet(resp_processed_path)
    print(f"resp_p: {len(resp_p)} rows (from {resp_processed_path})")
    return (resp_p,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## pH
    """)
    return


@app.cell
def _(apply_outlier_handling, cohort_hosp_ids, duckdb):
    from clifpy import Labs

    labs = Labs.from_file(
        config_path='config/config.json',
        columns=[
            'hospitalization_id', 'lab_order_dttm', 'lab_result_dttm',
            'lab_category', 'lab_value_numeric',
        ],
        filters={
            'hospitalization_id': cohort_hosp_ids,
            'lab_category': ['ph_arterial', 'ph_venous'],
        },
    )
    apply_outlier_handling(labs, outlier_config_path='config/outlier_config.yaml')
    labs_df = labs.df
    _q = """
    PIVOT_WIDER labs_df
    ON lab_category
    USING MAX(lab_value_numeric)
    """
    labs_w = duckdb.sql(_q).df()
    return Labs, labs_w


@app.cell
def _(cohort_shift_change_grids, labs_w):
    ph_df = mo.sql(
        f"""
        -- ASOF join pH labs to shift-change grids; categorize pH level
        FROM cohort_shift_change_grids g
        ASOF LEFT JOIN labs_w l ON
            g.hospitalization_id = l.hospitalization_id
            AND l.lab_order_dttm <= g.event_dttm
        SELECT g.*
            , l.lab_order_dttm
            , l.lab_result_dttm
            , l.ph_arterial
            , l.ph_venous
            , ph: COALESCE(ph_arterial, ph_venous + 0.05)
            , _time_diff: g.event_dttm - l.lab_order_dttm
            , _within_12_hours: CASE WHEN _time_diff <= INTERVAL '12 hour' THEN 1 ELSE 0 END
            , ph_level: CASE
                WHEN _within_12_hours = 0 THEN 'missing'
                WHEN ph < 7.20 THEN 'ph_lt72'
                WHEN ph >= 7.20 AND ph < 7.30 THEN 'ph_72_73'
                WHEN ph >= 7.30 AND ph < 7.40 THEN 'ph_73_74'
                WHEN ph >= 7.40 AND ph < 7.45 THEN 'ph_74_745'
                WHEN ph >= 7.45 THEN 'ph_ge745'
                ELSE 'missing'
            END
        ORDER BY g.hospitalization_id, g.event_dttm
        """
    )
    return (ph_df,)


@app.cell
def _(cohort_shift_change_grids, ph_df):
    assert len(ph_df) == len(cohort_shift_change_grids), 'ph_df length altered by ASOF join'
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## P/F Ratio
    """)
    return


@app.cell
def _(Labs, apply_outlier_handling, cohort_hosp_ids, duckdb):
    po2 = Labs.from_file(
        config_path='config/config.json',
        columns=[
            'hospitalization_id', 'lab_order_dttm', 'lab_result_dttm',
            'lab_category', 'lab_value_numeric',
        ],
        filters={
            'hospitalization_id': cohort_hosp_ids,
            'lab_category': ['po2_arterial'],
        },
    )
    apply_outlier_handling(po2, outlier_config_path='config/outlier_config.yaml')
    po2_df = po2.df
    _q = """
    PIVOT_WIDER po2_df
    ON lab_category
    USING MAX(lab_value_numeric)
    """
    po2_w = duckdb.sql(_q).df()
    return (po2_w,)


@app.cell
def _(cohort_shift_change_grids, po2_w, resp_p):
    pf_df = mo.sql(
        f"""
        -- ASOF join FiO2 + PO2 to shift-change grids; compute P/F ratio and level
        FROM cohort_shift_change_grids g
        ASOF LEFT JOIN resp_p r ON
            g.hospitalization_id = r.hospitalization_id
            AND r.recorded_dttm <= g.event_dttm
        ASOF LEFT JOIN po2_w p ON
            g.hospitalization_id = p.hospitalization_id
            AND p.lab_order_dttm <= g.event_dttm
        SELECT g.*
            , fio2_dttm: r.recorded_dttm
            , fio2_set: r.fio2_set
            , po2_dttm: p.lab_order_dttm
            , po2_arterial: p.po2_arterial
            , pf: po2_arterial / fio2_set
            , pf_level: CASE
                WHEN pf IS NULL THEN 'missing'
                WHEN pf < 100 THEN 'pf_lt100'
                WHEN pf >= 100 AND pf < 200 THEN 'pf_100_200'
                WHEN pf >= 200 AND pf < 300 THEN 'pf_200_300'
                WHEN pf >= 300 THEN 'pf_ge300'
                ELSE 'missing'
            END
        ORDER BY g.hospitalization_id, g.event_dttm
        """
    )
    return (pf_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Vasopressors
    """)
    return


@app.cell
def _(apply_outlier_handling, cohort_hosp_ids):
    from clifpy import Vitals

    vitals = Vitals.from_file(
        config_path='config/config.json',
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        filters={
            'vital_category': ['weight_kg'],
            'hospitalization_id': cohort_hosp_ids,
        },
    )
    apply_outlier_handling(vitals, outlier_config_path='config/outlier_config.yaml')
    vitals_df = vitals.df
    return (vitals_df,)


@app.cell
def _(
    apply_outlier_handling,
    cohort_hosp_ids,
    convert_dose_units_by_med_category,
    remove_meds_duplicates,
    vitals_df,
):
    from clifpy import MedicationAdminContinuous

    _vaso_categories = [
        "norepinephrine",
        "epinephrine",
        "phenylephrine",
        "dopamine",
        "vasopressin",
        "angiotensin",
    ]

    try:
        cont_veso = MedicationAdminContinuous.from_file(
            config_path="config/config.json",
            columns=[
                "hospitalization_id",
                "admin_dttm",
                "med_name",
                "med_category",
                "med_dose",
                "med_dose_unit",
                "mar_action_name",
                "mar_action_category",
            ],
            filters={
                "med_category": _vaso_categories,
                "hospitalization_id": cohort_hosp_ids,
            },
        )
    except Exception:
        print("Loading without mar_action_category instead")
        cont_veso = MedicationAdminContinuous.from_file(
            config_path="config/config.json",
            columns=[
                "hospitalization_id",
                "admin_dttm",
                "med_name",
                "med_category",
                "med_dose",
                "med_dose_unit",
                "mar_action_name",
            ],
            filters={
                "med_category": _vaso_categories,
                "hospitalization_id": cohort_hosp_ids,
            },
        )

    # NOTE: Preferred units match NEE formula expectations
    cont_veso_preferred_units = {
        "dopamine": "mcg/kg/min",
        "norepinephrine": "mcg/kg/min",
        "epinephrine": "mcg/kg/min",
        "phenylephrine": "mcg/kg/min",
        "angiotensin": "mcg/kg/min",
        "vasopressin": "u/min",
    }

    cont_veso_deduped = remove_meds_duplicates(cont_veso.df)
    _n_removed = len(cont_veso.df) - len(cont_veso_deduped)
    print(
        f"Removed {_n_removed} ({_n_removed / len(cont_veso.df):.2%}) duplicates by MAR action"
    )

    cont_veso_converted, cont_veso_convert_summary = convert_dose_units_by_med_category(
        cont_veso_deduped,
        vitals_df=vitals_df,
        preferred_units=cont_veso_preferred_units,
        override=True,
    )
    cont_veso_converted.rename(
        columns={
            "med_dose": "med_dose_original",
            "med_dose_unit": "med_dose_unit_original",
            "med_dose_converted": "med_dose",
            "med_dose_unit_converted": "med_dose_unit",
        },
        inplace=True,
    )

    cont_veso.df = cont_veso_converted
    apply_outlier_handling(cont_veso, outlier_config_path="config/outlier_config.yaml")
    cont_veso_converted = cont_veso.df
    return (cont_veso_converted,)


@app.cell
def _(cohort_shift_change_grids, cont_veso_converted):
    vaso_df = mo.sql(
        f"""
        -- ASOF join each vasopressor to shift-change grids; compute NEE
        FROM cohort_shift_change_grids g
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'dopamine'
        ) m1 ON g.hospitalization_id = m1.hospitalization_id
            AND m1.admin_dttm <= g.event_dttm
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'norepinephrine'
        ) m2 ON g.hospitalization_id = m2.hospitalization_id
            AND m2.admin_dttm <= g.event_dttm
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'epinephrine'
        ) m3 ON g.hospitalization_id = m3.hospitalization_id
            AND m3.admin_dttm <= g.event_dttm
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'phenylephrine'
        ) m4 ON g.hospitalization_id = m4.hospitalization_id
            AND m4.admin_dttm <= g.event_dttm
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'angiotensin'
        ) m5 ON g.hospitalization_id = m5.hospitalization_id
            AND m5.admin_dttm <= g.event_dttm
        ASOF LEFT JOIN (
            FROM cont_veso_converted
            WHERE med_category = 'vasopressin'
        ) m6 ON g.hospitalization_id = m6.hospitalization_id
            AND m6.admin_dttm <= g.event_dttm
        SELECT g.*
            , dopamine: COALESCE(m1.med_dose, 0)
            , norepinephrine: COALESCE(m2.med_dose, 0)
            , epinephrine: COALESCE(m3.med_dose, 0)
            , phenylephrine: COALESCE(m4.med_dose, 0)
            , angiotensin: COALESCE(m5.med_dose, 0)
            , vasopressin: COALESCE(m6.med_dose, 0)
            , _nee: norepinephrine + epinephrine + phenylephrine / 10.0 + dopamine / 100.0 + vasopressin * 2.5 + angiotensin * 10
        ORDER BY g.hospitalization_id, g.event_dttm
        """
    )
    return (vaso_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Daily SOFA

    Computes daily SOFA scores using `compute_sofa_polars` (local copy of pyCLIF).
    Creates per-day observation windows from hourly grids, then aggregates
    worst values within each window to produce 6 subscores + total.
    """)
    return


@app.cell
def _():
    import polars as pl

    _grids = pl.read_parquet("output/cohort_hrly_grids.parquet")
    sofa_cohort = (
        _grids
        .group_by(['hospitalization_id', '_nth_day'])
        .agg([
            pl.col('event_dttm').min().alias('start_dttm'),
            pl.col('event_dttm').max().alias('end_dttm'),
        ])
        .with_columns(
            patient_day_id=pl.concat_str([
                pl.col('hospitalization_id'),
                pl.lit('__'),
                pl.col('_nth_day').cast(pl.Utf8),
            ])
        )
    )
    print(f"SOFA cohort: {sofa_cohort.height} patient-days")
    return (sofa_cohort,)


@app.cell
def _(CONFIG_PATH, get_config_or_params, sofa_cohort):
    from _sofa import compute_sofa_polars

    _cfg = get_config_or_params(CONFIG_PATH)
    sofa_raw = compute_sofa_polars(
        data_directory=_cfg['data_directory'],
        cohort_df=sofa_cohort,
        filetype=_cfg.get('filetype', 'parquet'),
        id_name='patient_day_id',
        timezone=_cfg.get('timezone'),
    )
    print(f"SOFA raw: {sofa_raw.height} rows, {sofa_raw.width} columns")
    return (sofa_raw,)


@app.cell
def _(sofa_raw):
    import polars as _pl

    sofa_daily = sofa_raw.with_columns(
        [
            _pl.col("patient_day_id")
            .str.split("__")
            .list.get(0)
            .alias("hospitalization_id"),
            _pl.col("patient_day_id")
            .str.split("__")
            .list.get(1)
            .cast(_pl.Float64)
            .cast(_pl.Int64)
            .alias("_nth_day"),
        ]
    ).select(
        [
            "hospitalization_id",
            "_nth_day",
            "sofa_total",
            "sofa_cv_97",
            "sofa_coag",
            "sofa_liver",
            "sofa_resp",
            "sofa_cns",
            "sofa_renal",
        ]
    )
    sofa_daily.to_pandas().to_parquet("output/sofa_daily.parquet")
    print(f"Saved: output/sofa_daily.parquet ({sofa_daily.height} patient-days)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge Covariates
    """)
    return


@app.cell
def _(cohort_shift_change_grids, pf_df, ph_df, vaso_df):
    covs = mo.sql(
        f"""
        -- Join pH, P/F, and vasopressor covariates at shift-change level
        FROM cohort_shift_change_grids g
        LEFT JOIN ph_df ph USING (hospitalization_id, event_dttm)
        LEFT JOIN pf_df pf USING (hospitalization_id, event_dttm)
        LEFT JOIN vaso_df v USING (hospitalization_id, event_dttm)
        SELECT g.*
            , ph.ph_level
            , ph.ph
            , pf.pf_level
            , pf.pf
            , v._nee
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (covs,)


@app.cell
def _(covs):
    covs_daily = mo.sql(
        f"""
        -- Aggregate shift-level covariates to daily (7am/7pm values per day)
        FROM covs
        SELECT hospitalization_id
            , _nth_day
            , ph_level_7am: COALESCE(ANY_VALUE(CASE WHEN _hr = 7 THEN ph_level END), 'missing')
            , ph_level_7pm: COALESCE(ANY_VALUE(CASE WHEN _hr = 19 THEN ph_level END), 'missing')
            , pf_level_7am: COALESCE(ANY_VALUE(CASE WHEN _hr = 7 THEN pf_level END), 'missing')
            , pf_level_7pm: COALESCE(ANY_VALUE(CASE WHEN _hr = 19 THEN pf_level END), 'missing')
            , nee_7am: COALESCE(ANY_VALUE(CASE WHEN _hr = 7 THEN _nee END), 0)
            , nee_7pm: COALESCE(ANY_VALUE(CASE WHEN _hr = 19 THEN _nee END), 0)
            , _ph_7am: ANY_VALUE(CASE WHEN _hr = 7 THEN ph END)
            , _ph_7pm: ANY_VALUE(CASE WHEN _hr = 19 THEN ph END)
            , _pf_7am: ANY_VALUE(CASE WHEN _hr = 7 THEN pf END)
            , _pf_7pm: ANY_VALUE(CASE WHEN _hr = 19 THEN pf END)
        GROUP BY hospitalization_id, _nth_day
        ORDER BY hospitalization_id, _nth_day
        """
    )
    return (covs_daily,)


# ── Comorbidity Indices ────────────────────────────────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Comorbidity Indices (CCI & Elixhauser)
    """)
    return


@app.cell
def _(CONFIG_PATH, cohort_hosp_ids):
    from clifpy import HospitalDiagnosis
    from clifpy.utils import calculate_cci
    from clifpy.utils.comorbidity import calculate_elix

    _dx = HospitalDiagnosis.from_file(
        config_path=CONFIG_PATH,
        filters={'hospitalization_id': cohort_hosp_ids},
    )
    cci_df = calculate_cci(_dx, hierarchy=True)
    elix_df = calculate_elix(_dx, hierarchy=True)
    print(f"CCI: {len(cci_df)} rows, Elixhauser: {len(elix_df)} rows")

    cci_df.to_parquet("output/cci.parquet", index=False)
    elix_df.to_parquet("output/elix.parquet", index=False)
    print("Saved: output/cci.parquet, output/elix.parquet")
    return


# ── Save ───────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs
    """)
    return


@app.cell
def _(covs, covs_daily):
    _covs_shift_df = covs.df()
    _covs_shift_df.to_parquet("output/covariates_shift.parquet")
    print(f"Saved: output/covariates_shift.parquet ({len(_covs_shift_df)} rows)")

    _covs_daily_df = covs_daily.df()
    _covs_daily_df.to_parquet("output/covariates_daily.parquet")
    print(f"Saved: output/covariates_daily.parquet ({len(_covs_daily_df)} rows)")
    return


if __name__ == "__main__":
    app.run()
