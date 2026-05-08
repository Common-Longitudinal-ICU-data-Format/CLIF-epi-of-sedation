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

    # Cache-bypass flags for expensive Table 1 covariate recomputes
    RERUN_SOFA_24H = False
    RERUN_ASE = False


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
    from _utils import remove_meds_duplicates, retag_to_local_tz

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"
    co = ClifOrchestrator(config_path=CONFIG_PATH)
    return (
        CONFIG_PATH,
        apply_outlier_handling,
        convert_dose_units_by_med_category,
        duckdb,
        get_config_or_params,
        pd,
        remove_meds_duplicates,
        retag_to_local_tz,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params):
    # Site-scoped output dir (see Makefile SITE= flag).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    print(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return SITE_NAME, SITE_TZ


@app.cell
def _(SITE_NAME, pd):
    cohort_hrly_grids = pd.read_parquet(f"output/{SITE_NAME}/cohort_hrly_grids.parquet")
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
    resp_processed_path = f"output/{SITE_NAME}/resp_processed_bf.parquet"
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
def _(SITE_NAME):
    import polars as pl

    _grids = pl.read_parquet(f"output/{SITE_NAME}/cohort_hrly_grids.parquet")
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
def _(SITE_NAME, sofa_raw):
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
    _sofa_path = f"output/{SITE_NAME}/sofa_daily.parquet"
    sofa_daily.to_pandas().to_parquet(_sofa_path)
    print(f"Saved: {_sofa_path} ({sofa_daily.height} patient-days)")
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
def _(CONFIG_PATH, SITE_NAME, cohort_hosp_ids):
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

    cci_df.to_parquet(f"output/{SITE_NAME}/cci.parquet", index=False)
    elix_df.to_parquet(f"output/{SITE_NAME}/elix.parquet", index=False)
    print(f"Saved: output/{SITE_NAME}/cci.parquet, output/{SITE_NAME}/elix.parquet")
    return


# ── Table 1 Hospitalization-Level Covariates ──────────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Table 1 Hospitalization-Level Covariates

    Hospitalization-level variables for Table 1 (one row per hospitalization).
    Anchored to the first 24 h of ICU admit per patient. Writes
    `output/covariates_t1.parquet` which is joined into the analytical
    dataset by `05_analytical_dataset.py`.
    """)
    return


@app.cell
def _(CONFIG_PATH, SITE_TZ, cohort_hosp_ids, duckdb):
    # Cell A — first ICU admit dttm per cohort hospitalization.
    # NOTE: For patients intubated in ED/OR then transferred to ICU,
    # this 24 h window starts AFTER intubation. Per user spec (first 24 h of ICU admit).
    from clifpy import Adt

    _adt_icu = Adt.from_file(
        config_path=CONFIG_PATH,
        columns=['hospitalization_id', 'in_dttm', 'location_category'],
        filters={
            'hospitalization_id': cohort_hosp_ids,
            'location_category': ['icu'],
        },
    )
    adt_icu_df = _adt_icu.df
    # clifpy's `from_file` loaders return *_dttm columns NAIVE but with
    # wall-clock already in the config's site tz (per loader log line
    # "Timezone: <site_tz>"). Attach the tz tag now via tz_localize —
    # this is a metadata claim ("this naive wall-clock IS site-local"),
    # NOT a tz_convert (which would relabel a UTC instant). The downstream
    # DuckDB SQL then sees TIMESTAMPTZ and propagates the tz through
    # MIN() into _first_icu_dttm / _first_icu_24h_end.
    # localize_naive_to_site_tz handles DST fall-back ambiguity (UCMC has
    # ICU admits at 01:00-02:00 on fall-back Sundays in 2019 / 2021).
    from _utils import localize_naive_to_site_tz
    adt_icu_df['in_dttm'] = localize_naive_to_site_tz(adt_icu_df['in_dttm'], SITE_TZ)

    first_icu_admit = duckdb.sql("""
        FROM adt_icu_df
        SELECT hospitalization_id
            , _first_icu_dttm: MIN(in_dttm)
            , _first_icu_24h_end: MIN(in_dttm) + INTERVAL 24 HOUR
        GROUP BY hospitalization_id
        ORDER BY hospitalization_id
    """).df()
    print(f"first_icu_admit: {len(first_icu_admit)} hospitalizations")
    return (first_icu_admit,)


@app.cell
def _(CONFIG_PATH, first_icu_admit, get_config_or_params):
    # Cell B — SOFA over first 24 h of ICU admit (cached).
    # Reuses the existing local _sofa.compute_sofa_polars but with a different cohort_df:
    # one row per hospitalization with [start_dttm, end_dttm] = [first_icu, first_icu+24h]
    # instead of the per-patient-day windows used by the existing sofa_daily computation.
    # Imports aliased with `_` prefix to avoid marimo multi-cell definition collisions
    # with the existing SOFA cell above.
    import polars as _pl
    from _sofa import compute_sofa_polars as _compute_sofa_polars

    _sofa_24h_path = "output/sofa_first_24h.parquet"
    if (not os.path.exists(_sofa_24h_path)) or RERUN_SOFA_24H:
        _cfg = get_config_or_params(CONFIG_PATH)
        _sofa_24h_cohort = _pl.from_pandas(
            first_icu_admit.rename(columns={
                '_first_icu_dttm': 'start_dttm',
                '_first_icu_24h_end': 'end_dttm',
            })[['hospitalization_id', 'start_dttm', 'end_dttm']]
        )
        _sofa_24h = _compute_sofa_polars(
            data_directory=_cfg['data_directory'],
            cohort_df=_sofa_24h_cohort,
            filetype=_cfg.get('filetype', 'parquet'),
            id_name='hospitalization_id',
            timezone=_cfg.get('timezone'),
        )
        # Rename to avoid collision with existing per-day `sofa_total` in analytical_dataset
        _sofa_24h = _sofa_24h.rename({
            'sofa_total': 'sofa_1st24h',
            'sofa_cv_97': 'sofa_cv_97_1st24h',
            'sofa_coag': 'sofa_coag_1st24h',
            'sofa_liver': 'sofa_liver_1st24h',
            'sofa_resp': 'sofa_resp_1st24h',
            'sofa_cns': 'sofa_cns_1st24h',
            'sofa_renal': 'sofa_renal_1st24h',
        })
        _sofa_24h.write_parquet(_sofa_24h_path)
        print(f"Computed + saved {_sofa_24h_path} ({_sofa_24h.height} rows)")
    else:
        _sofa_24h = _pl.read_parquet(_sofa_24h_path)
        print(f"Loaded cached {_sofa_24h_path} ({_sofa_24h.height} rows)")

    sofa_24h_pd = _sofa_24h.to_pandas()
    return (sofa_24h_pd,)


@app.cell
def _(CONFIG_PATH, apply_outlier_handling, cohort_hosp_ids, duckdb):
    # Cell C — Load vitals needed for BMI (height_cm + weight_kg) and P/F fallback (spo2).
    # Single combined load for efficiency; kept separate from the existing weight-only
    # `vitals_df` at the top of the Vasopressors section so we don't perturb the dose
    # unit converter that depends on that DataFrame's exact shape.
    # `_Vitals` alias avoids marimo multi-cell definition collision with the existing
    # Vitals import in the vasopressor cell.
    from clifpy import Vitals as _Vitals

    _vitals_t1 = _Vitals.from_file(
        config_path=CONFIG_PATH,
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        filters={
            'vital_category': ['height_cm', 'weight_kg', 'spo2'],
            'hospitalization_id': cohort_hosp_ids,
        },
    )
    apply_outlier_handling(_vitals_t1, outlier_config_path='config/outlier_config.yaml')
    vitals_t1_df = _vitals_t1.df

    # First non-null height_cm and weight_kg per hospitalization → BMI.
    # Height guarded to 50–250 cm (reject impossible values even post-outlier-handling).
    bmi_df = duckdb.sql("""
        WITH first_h AS (
            FROM vitals_t1_df
            SELECT hospitalization_id
                , height_cm: vital_value
            WHERE vital_category = 'height_cm'
              AND vital_value IS NOT NULL
              AND vital_value BETWEEN 50 AND 250
            QUALIFY ROW_NUMBER() OVER (PARTITION BY hospitalization_id ORDER BY recorded_dttm) = 1
        )
        , first_w AS (
            FROM vitals_t1_df
            SELECT hospitalization_id
                , weight_kg: vital_value
            WHERE vital_category = 'weight_kg'
              AND vital_value IS NOT NULL
              AND vital_value > 0
            QUALIFY ROW_NUMBER() OVER (PARTITION BY hospitalization_id ORDER BY recorded_dttm) = 1
        )
        FROM first_h h
        FULL JOIN first_w w USING (hospitalization_id)
        SELECT hospitalization_id
            , h.height_cm
            , w.weight_kg
            , bmi: w.weight_kg / ((h.height_cm / 100.0) * (h.height_cm / 100.0))
        ORDER BY hospitalization_id
    """).df()
    print(f"bmi_df: {len(bmi_df)} rows, {bmi_df['bmi'].notna().sum()} non-null BMIs")
    return bmi_df, vitals_t1_df


@app.cell
def _(bmi_df, cohort_shift_change_grids, vitals_t1_df):
    # Cell C2 — Per-patient-day weight at start of each patient-day (7am anchor).
    # ASOF backward-join most recent vitals weight_kg to each 7am grid row; coalesce
    # with admission weight (bmi_df.weight_kg = first recorded weight per hospitalization)
    # when no prior reading exists. Consumed by the mcg/kg/min propofol descriptives
    # in code/descriptive/.
    weight_daily = mo.sql(
        f"""
        WITH day_starts AS (
            FROM cohort_shift_change_grids
            SELECT hospitalization_id, _nth_day, event_dttm
            WHERE _hr = 7 AND _nth_day > 0
        )
        , weight_events AS (
            FROM vitals_t1_df
            SELECT hospitalization_id
                , recorded_dttm
                , weight_kg: vital_value
            WHERE vital_category = 'weight_kg' AND vital_value IS NOT NULL
        )
        , asof_weight AS (
            FROM day_starts d
            ASOF LEFT JOIN weight_events w ON
                d.hospitalization_id = w.hospitalization_id
                AND w.recorded_dttm <= d.event_dttm
            SELECT d.hospitalization_id
                , d._nth_day
                , weight_asof: w.weight_kg
        )
        FROM asof_weight a
        LEFT JOIN bmi_df b USING (hospitalization_id)
        SELECT a.hospitalization_id
            , a._nth_day
            , weight_kg_asof_day_start: COALESCE(a.weight_asof, b.weight_kg)
        ORDER BY a.hospitalization_id, a._nth_day
        """
    )
    return (weight_daily,)


@app.cell
def _(cont_veso_converted, duckdb):
    # Cell D — 'ever on vasopressors' binary per hospitalization.
    # DECISION POINT (revisit): Currently uses CONTINUOUS vasopressor infusions only
    # (medication_admin_continuous via cont_veso_converted). Bolus / push-dose vasopressors
    # from medication_admin_intermittent are NOT included. Standard ICU literature uses
    # continuous-only for 'ever on pressors' because it reflects sustained shock;
    # intermittent captures peri-intubation push doses that are less specific. To add
    # intermittent later: load MedicationAdminIntermittent with the same vaso med_categories,
    # dedupe + outlier-handle, compute the same MAX(med_dose > 0) per hosp, and OR the two
    # flags together at the SELECT.
    ever_pressor_df = duckdb.sql("""
        FROM cont_veso_converted
        SELECT hospitalization_id
            , ever_pressor: MAX(CASE WHEN med_dose > 0 THEN 1 ELSE 0 END)
        GROUP BY hospitalization_id
        ORDER BY hospitalization_id
    """).df()
    _n_pressor = int(ever_pressor_df['ever_pressor'].sum())
    print(
        f"ever_pressor_df: {len(ever_pressor_df)} rows, {_n_pressor} with ever_pressor=1"
    )
    return (ever_pressor_df,)


@app.cell
def _(duckdb, first_icu_admit, po2_w, resp_p, vitals_t1_df):
    # Cell E — Worst P/F (or S/F-imputed P/F via Rice equation) in first 24 h of ICU admit.
    # Rice formula: PF_imputed = 64 + 0.84 * (SpO2 / FiO2 * 100), valid for SpO2 <= 97
    # (above 97, the saturation curve flattens and S/F is unreliable).
    # Event-driven: no fresh hourly grid — ASOF-join FiO2 from resp_p to the event times
    # of PaO2 labs and SpO2 vitals that fall within the ICU-24h window.
    pf_24h_df = duckdb.sql("""
        WITH hosp_window AS (
            FROM first_icu_admit
            SELECT hospitalization_id, _first_icu_dttm, _first_icu_24h_end
        )
        , pao2_in_window AS (
            FROM po2_w p
            JOIN hosp_window h USING (hospitalization_id)
            SELECT p.hospitalization_id
                , event_dttm: p.lab_order_dttm
                , pao2: p.po2_arterial
            WHERE p.lab_order_dttm BETWEEN h._first_icu_dttm AND h._first_icu_24h_end
              AND p.po2_arterial IS NOT NULL
        )
        , spo2_in_window AS (
            FROM vitals_t1_df s
            JOIN hosp_window h USING (hospitalization_id)
            SELECT s.hospitalization_id
                , event_dttm: s.recorded_dttm
                , spo2: s.vital_value
            WHERE s.vital_category = 'spo2'
              AND s.recorded_dttm BETWEEN h._first_icu_dttm AND h._first_icu_24h_end
              AND s.vital_value IS NOT NULL
              AND s.vital_value <= 97  -- Rice valid range
        )
        , fio2_at_pao2 AS (
            FROM pao2_in_window p
            ASOF LEFT JOIN resp_p r
                ON p.hospitalization_id = r.hospitalization_id
                AND r.recorded_dttm <= p.event_dttm
            SELECT p.hospitalization_id
                , pf: p.pao2 / r.fio2_set
                , source: 'pao2'
            WHERE r.fio2_set IS NOT NULL AND r.fio2_set > 0
        )
        , fio2_at_spo2 AS (
            FROM spo2_in_window s
            ASOF LEFT JOIN resp_p r
                ON s.hospitalization_id = r.hospitalization_id
                AND r.recorded_dttm <= s.event_dttm
            SELECT s.hospitalization_id
                , pf: 64 + 0.84 * (s.spo2 / r.fio2_set * 100)
                , source: 'spo2'
            WHERE r.fio2_set IS NOT NULL AND r.fio2_set > 0
        )
        , combined AS (
            FROM fio2_at_pao2 SELECT hospitalization_id, pf, source
            UNION ALL
            FROM fio2_at_spo2 SELECT hospitalization_id, pf, source
        )
        FROM combined
        SELECT hospitalization_id
            , pf_1st24h_min: MIN(pf)
            , pf_1st24h_source: ANY_VALUE(source ORDER BY pf ASC)  -- source of the worst pf
        WHERE pf > 0
        GROUP BY hospitalization_id
        ORDER BY hospitalization_id
    """).df()
    _med_pf = pf_24h_df['pf_1st24h_min'].median() if len(pf_24h_df) else float('nan')
    print(f"pf_24h_df: {len(pf_24h_df)} rows, median pf_1st24h_min: {_med_pf:.1f}")
    return (pf_24h_df,)


@app.cell
def _(SITE_NAME, pd):
    # Cell F — IMV duration (already computed in 01_cohort.py).
    # NOTE: This is the duration of the FIRST QUALIFYING IMV STREAK per hospitalization,
    # which is also the cohort exposure window (per cohort definition in 01_cohort.py:272-283:
    # "First IMV streak ≥24 hours"). _duration_hrs = (end - start) where start = intubation
    # event and end = COALESCE(_trach_dttm, _next_start_dttm, _last_observed_dttm). Subsequent
    # re-intubations after extubation are NOT included (those would be separate streaks
    # not in the cohort).
    _imv_streaks_t1 = pd.read_parquet(f"output/{SITE_NAME}/cohort_imv_streaks.parquet")
    imv_dur_df = _imv_streaks_t1[['hospitalization_id', '_duration_hrs']].rename(
        columns={'_duration_hrs': 'imv_duration_hrs'}
    )
    _med_imv = imv_dur_df['imv_duration_hrs'].median()
    print(f"imv_dur_df: {len(imv_dur_df)} rows, median imv_duration_hrs: {_med_imv:.1f}")
    return (imv_dur_df,)


@app.cell
def _(CONFIG_PATH, SITE_NAME, cohort_hosp_ids, duckdb, pd):
    # Cell G — Sepsis CDC Adult Sepsis Event (ASE) via clifpy (cached).
    # compute_ase independently loads many CLIF tables (Hospitalization, MedAdmin-cont,
    # MedAdmin-intermittent, Labs, MicrobiologyCulture, Adt, RespiratorySupport) and can
    # take 5–15 minutes for ~thousand hospitalizations. Cached to output/{site}/ase.parquet
    # and re-used on subsequent notebook runs unless RERUN_ASE flag is True.
    _ase_path = f"output/{SITE_NAME}/ase.parquet"
    if (not os.path.exists(_ase_path)) or RERUN_ASE:
        from clifpy.utils import compute_ase
        ase_full = compute_ase(
            hospitalization_ids=cohort_hosp_ids,
            config_path=CONFIG_PATH,
            apply_rit=True,
            rit_only_hospital_onset=True,
            include_lactate=False,  # CDC strict definition (matches function default)
            verbose=True,
        )
        ase_full.to_parquet(_ase_path, index=False)
        print(f"Computed + saved {_ase_path} ({len(ase_full)} rows)")
    else:
        ase_full = pd.read_parquet(_ase_path)
        print(f"Loaded cached {_ase_path} ({len(ase_full)} rows)")

    # Aggregate to per-hospitalization binary: any ASE episode = sepsis_ase = 1
    ase_df = duckdb.sql("""
        FROM ase_full
        SELECT hospitalization_id
            , sepsis_ase: MAX(CAST(sepsis AS INTEGER))
        GROUP BY hospitalization_id
        ORDER BY hospitalization_id
    """).df()
    _n_sepsis = int(ase_df['sepsis_ase'].sum())
    print(f"ase_df: {len(ase_df)} rows, {_n_sepsis} with sepsis_ase=1")
    return (ase_df,)


@app.cell
def _(
    SITE_NAME,
    SITE_TZ,
    ase_df,
    bmi_df,
    duckdb,
    ever_pressor_df,
    first_icu_admit,
    imv_dur_df,
    pf_24h_df,
    retag_to_local_tz,
    sofa_24h_pd,
):
    # Cell H — Combine all T1 covariates into one row-per-hospitalization dataframe
    # and save to output/{site}/covariates_t1.parquet for 05_analytical_dataset.py to join.
    covariates_t1 = duckdb.sql("""
        FROM first_icu_admit f
        LEFT JOIN bmi_df USING (hospitalization_id)
        LEFT JOIN sofa_24h_pd USING (hospitalization_id)
        LEFT JOIN ever_pressor_df USING (hospitalization_id)
        LEFT JOIN pf_24h_df USING (hospitalization_id)
        LEFT JOIN imv_dur_df USING (hospitalization_id)
        LEFT JOIN ase_df USING (hospitalization_id)
        SELECT f.hospitalization_id
            , f._first_icu_dttm
            , bmi_df.height_cm
            , bmi_df.weight_kg
            , bmi_df.bmi
            , sofa_24h_pd.sofa_1st24h
            , sofa_24h_pd.sofa_cv_97_1st24h
            , sofa_24h_pd.sofa_coag_1st24h
            , sofa_24h_pd.sofa_liver_1st24h
            , sofa_24h_pd.sofa_resp_1st24h
            , sofa_24h_pd.sofa_cns_1st24h
            , sofa_24h_pd.sofa_renal_1st24h
            , ever_pressor: COALESCE(ever_pressor_df.ever_pressor, 0)
            , pf_24h_df.pf_1st24h_min
            , pf_24h_df.pf_1st24h_source
            , imv_dur_df.imv_duration_hrs
            , sepsis_ase: COALESCE(ase_df.sepsis_ase, 0)
        ORDER BY f.hospitalization_id
    """).df()
    _t1_path = f"output/{SITE_NAME}/covariates_t1.parquet"
    # Retag _first_icu_dttm to SITE_TZ so the on-disk tz tag is the site's
    # configured tz, not the OS session tz that DuckDB stamped at .df() time.
    covariates_t1 = retag_to_local_tz(covariates_t1, ["_first_icu_dttm"], SITE_TZ)
    covariates_t1.to_parquet(_t1_path, index=False)
    print(
        f"Saved: {_t1_path} "
        f"({len(covariates_t1)} hospitalizations, {covariates_t1.shape[1]} columns)"
    )
    return (covariates_t1,)


# ── Save ───────────────────────────────────────────────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs
    """)
    return


@app.cell
def _(SITE_NAME, SITE_TZ, covs, covs_daily, retag_to_local_tz):
    _covs_shift_df = covs.df()
    # covariates_shift has event_dttm (shift-grain). covariates_daily is a
    # pure aggregate (groups by hospitalization_id, _nth_day) so no retag.
    _covs_shift_df = retag_to_local_tz(_covs_shift_df, ["event_dttm"], SITE_TZ)
    _covs_shift_path = f"output/{SITE_NAME}/covariates_shift.parquet"
    _covs_shift_df.to_parquet(_covs_shift_path)
    print(f"Saved: {_covs_shift_path} ({len(_covs_shift_df)} rows)")

    _covs_daily_df = covs_daily.df()
    _covs_daily_path = f"output/{SITE_NAME}/covariates_daily.parquet"
    _covs_daily_df.to_parquet(_covs_daily_path)
    print(f"Saved: {_covs_daily_path} ({len(_covs_daily_df)} rows)")
    return


@app.cell
def _(SITE_NAME, weight_daily):
    _weight_daily_df = weight_daily.df()
    _weight_path = f"output/{SITE_NAME}/weight_daily.parquet"
    _weight_daily_df.to_parquet(_weight_path)
    _n_nonnull = _weight_daily_df["weight_kg_asof_day_start"].notna().sum()
    print(
        f"Saved: {_weight_path} "
        f"({len(_weight_daily_df)} rows, {_n_nonnull} non-null weights)"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Per-Stay Metadata Registry

    Build `cohort_meta_by_id` — the canonical per-stay registry.
    One row per hospitalization. Aggregates the day-grain registry to a
    stay-grain summary (`n_days_total`, `n_days_full_24h`,
    `has_first_partial`, `has_last_partial`), joins in the IMV streak
    boundaries (`imv_first_dttm`, `imv_last_dttm`, `imv_dur_hrs`), and
    derives a mutually-exclusive `exit_mechanism` categorical from the
    existing per-event flags in `sbt_outcomes_daily.parquet` plus the
    discharge_category from CLIF Hospitalization.

    Categorization order (first match wins): `tracheostomy` →
    `died_on_imv` → `palliative_extubation` → `failed_extubation` →
    `successful_extubation` → `discharge_on_imv` → `unknown`. Phase 1
    uses the existing `_fail_extub` / `_withdrawl_lst` / `_success_extub`
    flags from `03_outcomes.py` (whose underlying reintubation window
    is currently 24 h hardcoded — Phase 5 will plumb the new
    `reintub_window_hrs` config through to those flag derivations).
    """)
    return


@app.cell
def _(
    CONFIG_PATH,
    SITE_NAME,
    SITE_TZ,
    cohort_hosp_ids,
    duckdb,
):
    # Imports are `_`-prefixed to keep them cell-local — `localize_naive_to_site_tz`
    # is also imported by an earlier cell, and marimo enforces single
    # cross-cell binding.
    from clifpy import Hospitalization as _Hospitalization
    from _utils import localize_naive_to_site_tz as _loc_tz

    # Load discharge metadata for the cohort. Naive *_dttm gets tz-localized
    # to SITE_TZ (clifpy convention: from_file returns naive wall-clock that
    # is already site-local; we just attach the tz tag).
    _hosp = _Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=['hospitalization_id', 'discharge_category', 'discharge_dttm'],
        filters={'hospitalization_id': cohort_hosp_ids},
    )
    hosp_meta_df = _hosp.df.copy()
    hosp_meta_df['discharge_dttm'] = _loc_tz(
        hosp_meta_df['discharge_dttm'], SITE_TZ,
    )

    cohort_meta_by_id = duckdb.sql(f"""
        WITH per_hosp_meta AS (
            -- Aggregate the day-grain registry → per-stay rollups.
            FROM read_parquet('output/{SITE_NAME}/cohort_meta_by_id_imvday.parquet')
            SELECT
                hospitalization_id
                , encounter_block: MIN(encounter_block)
                , n_days_total:     COUNT(*)
                , n_days_full_24h:  COUNT(*) FILTER (WHERE _is_full_24h_day)
                , has_first_partial: BOOL_OR(_is_first_day)
                , has_last_partial:  BOOL_OR(_is_last_day)
            GROUP BY hospitalization_id
        )
        , per_hosp_outcomes AS (
            -- Roll daily outcome flags → per-stay flags via MAX. The daily
            -- file already replicates these flags onto every row of a hosp,
            -- so MAX is a no-op aggregator (matches the convention used by
            -- 05_modeling_dataset's analytical_dataset assembly).
            FROM read_parquet('output/{SITE_NAME}/sbt_outcomes_daily.parquet')
            SELECT
                hospitalization_id
                , ever_extubated:     COALESCE(MAX(_extub_1st),    0)
                , ever_trach:         COALESCE(MAX(_trach_1st),    0)
                , ever_success_extub: COALESCE(MAX(_success_extub), 0)
                , ever_failed_extub:  COALESCE(MAX(_fail_extub),   0)
                , ever_withdrawl:     COALESCE(MAX(_withdrawl_lst), 0)
            GROUP BY hospitalization_id
        )
        FROM read_parquet('output/{SITE_NAME}/cohort_imv_streaks.parquet') s
        LEFT JOIN per_hosp_meta m USING (hospitalization_id)
        LEFT JOIN per_hosp_outcomes o USING (hospitalization_id)
        LEFT JOIN hosp_meta_df h USING (hospitalization_id)
        SELECT
            s.hospitalization_id
            , m.encounter_block
            , m.n_days_total
            , m.n_days_full_24h
            , m.has_first_partial
            , m.has_last_partial
            , imv_first_dttm: s._start_dttm
            , imv_last_dttm:  s._end_dttm
            , imv_dur_hrs:    s._duration_hrs
            , successful_extubation:    COALESCE(o.ever_success_extub, 0) = 1
            , reintubated_within_window: COALESCE(o.ever_failed_extub, 0) = 1
            -- Mutually-exclusive exit_mechanism. Order matters: trach wins
            -- over death-on-IMV when both happen (trach is the cohort exit
            -- event); withdrawal-of-care wins over fail/success (different
            -- clinical category — the patient was extubated in anticipation
            -- of death, not as a recovery attempt).
            , exit_mechanism: CASE
                WHEN o.ever_trach = 1 THEN 'tracheostomy'
                WHEN COALESCE(o.ever_extubated, 0) = 0
                    AND TRIM(LOWER(h.discharge_category)) = 'expired' THEN 'died_on_imv'
                WHEN o.ever_withdrawl = 1 THEN 'palliative_extubation'
                WHEN o.ever_failed_extub = 1 THEN 'failed_extubation'
                WHEN o.ever_success_extub = 1 THEN 'successful_extubation'
                WHEN COALESCE(o.ever_extubated, 0) = 0 THEN 'discharge_on_imv'
                ELSE 'unknown'
            END
        ORDER BY s.hospitalization_id
    """).df()

    _n = len(cohort_meta_by_id)
    _n_unknown = int((cohort_meta_by_id['exit_mechanism'] == 'unknown').sum())
    print(
        f"cohort_meta_by_id: {_n:,} rows; "
        f"unknown exit_mechanism count = {_n_unknown:,} (should be 0 in healthy data)"
    )
    return (cohort_meta_by_id,)


@app.cell
def _(SITE_NAME, SITE_TZ, cohort_meta_by_id, retag_to_local_tz):
    # Persist with site-local tz tags on the boundary timestamps (mirrors
    # cohort_imv_streaks save convention).
    _df = retag_to_local_tz(
        cohort_meta_by_id, ["imv_first_dttm", "imv_last_dttm"], SITE_TZ,
    )
    _path = f"output/{SITE_NAME}/cohort_meta_by_id.parquet"
    _df.to_parquet(_path, index=False)

    _exit_counts = (
        _df['exit_mechanism']
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )
    print(f"Saved: {_path} ({len(_df):,} hospitalizations)")
    print(f"  exit_mechanism distribution: {_exit_counts}")
    return


if __name__ == "__main__":
    app.run()
