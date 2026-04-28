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
    # sys.path.insert(0, str(Path(__file__).parent))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 Sedation Dose Calculation

    Computes hourly, daily, and aggregated sedation doses (continuous + intermittent)
    for the IMV cohort identified in 01_cohort.
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
    # Site-scoped output dir (see Makefile SITE= flag).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return (SITE_NAME,)


@app.cell
def _(SITE_NAME, pd):
    cohort_hrly_grids_f = pd.read_parquet(f"output/{SITE_NAME}/cohort_hrly_grids.parquet")
    print(f"Hourly grid rows: {len(cohort_hrly_grids_f)}")
    return (cohort_hrly_grids_f,)


@app.cell
def _(cohort_hrly_grids_f):
    cohort_hosp_ids = cohort_hrly_grids_f['hospitalization_id'].unique().tolist()
    print(f"Cohort hospitalizations: {len(cohort_hosp_ids)}")
    return (cohort_hosp_ids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous Sedation

    Load vitals (weight_kg) for unit conversion, then load, dedup, convert,
    and pivot continuous sedation administrations.
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
            'hospitalization_id': cohort_hosp_ids
        }
    )
    apply_outlier_handling(vitals, outlier_config_path='config/outlier_config.yaml')
    vitals_df = vitals.df
    print(f"Vitals (weight_kg) rows: {len(vitals_df)}")
    return (vitals_df,)


@app.cell
def _(cohort_hosp_ids):
    from clifpy import MedicationAdminContinuous

    cont_sed = MedicationAdminContinuous.from_file(
        config_path='config/config.json',
        columns=[
            'hospitalization_id', 'admin_dttm', 'med_name', 'med_category',
            'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category',
        ],
        filters={
            'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'],
            'hospitalization_id': cohort_hosp_ids,
        }
    )
    print(f"Continuous sedation records: {len(cont_sed.df)}")
    return (cont_sed,)


@app.cell
def _(cont_sed, remove_meds_duplicates):
    cont_sed_deduped = remove_meds_duplicates(cont_sed.df)
    _n_removed = len(cont_sed.df) - len(cont_sed_deduped)
    print(f"Removed {_n_removed} ({_n_removed / len(cont_sed.df):.2%}) duplicates by MAR action")
    return (cont_sed_deduped,)


@app.cell
def _(cont_sed_deduped, duckdb, vitals_df):
    # Phase 2 weight override: pre-attach a project-controlled `weight_kg`
    # column on each admin row BEFORE handing off to clifpy. Per
    # clifpy/utils/unit_converter.py:717-719, clifpy only runs its own ASOF
    # when `weight_kg` isn't already a column on med_df, so this override
    # skips clifpy's no-fallback per-admin ASOF entirely (the path that
    # produced the silent /kg-factor-dropped bug — see
    # code/qc/weight_audit_README.md §5).
    #
    # Strategy: per-admin ASOF backward-join to most-recent prior weight
    # (same temporal logic as clifpy), with fallback to the patient's
    # first-ever weight (admission fallback). The weight-QC drop list at
    # 01_cohort.py guarantees every kept patient has ≥1 weight, so the
    # admission fallback is never NULL.
    cont_sed_with_weight = duckdb.sql("""
        WITH weights AS (
            FROM vitals_df
            SELECT hospitalization_id, recorded_dttm
                , weight_kg: vital_value
            WHERE vital_category = 'weight_kg' AND vital_value IS NOT NULL
        )
        , first_w AS (
            FROM weights
            SELECT hospitalization_id, _admit_weight: weight_kg
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY hospitalization_id ORDER BY recorded_dttm
            ) = 1
        )
        , asof_w AS (
            FROM cont_sed_deduped m
            ASOF LEFT JOIN weights v
              ON m.hospitalization_id = v.hospitalization_id
              AND v.recorded_dttm <= m.admin_dttm
            SELECT m.*
                , _asof_weight: v.weight_kg
        )
        FROM asof_w a
        LEFT JOIN first_w f USING (hospitalization_id)
        SELECT a.* EXCLUDE (_asof_weight)
            , weight_kg: COALESCE(a._asof_weight, f._admit_weight)
            , _weight_source: CASE
                WHEN a._asof_weight IS NOT NULL THEN 'per_admin_asof'
                WHEN f._admit_weight IS NOT NULL THEN 'admission_fallback'
                ELSE 'null'
                END
        ORDER BY a.hospitalization_id, a.admin_dttm
    """).df()

    _n_total = len(cont_sed_with_weight)
    _n_null = int(cont_sed_with_weight['weight_kg'].isna().sum())
    _n_fallback = int((cont_sed_with_weight['_weight_source'] == 'admission_fallback').sum())
    print(f"Pre-attached weight on {_n_total:,} admins")
    print(f"  per_admin_asof: {_n_total - _n_fallback - _n_null:,}")
    print(f"  admission_fallback: {_n_fallback:,}")
    if _n_null > 0:
        print(f"  WARNING: {_n_null:,} admins NULL after fallback — "
              "these patients should have been in weight-QC drop list. "
              "Check 01_cohort.py applied the drop.")
    return (cont_sed_with_weight,)


@app.cell
def _(
    cont_sed,
    cont_sed_with_weight,
    convert_dose_units_by_med_category,
    vitals_df,
):
    # Preferred unit for propofol changed to mcg/kg/min (Phase 2): pump-native
    # /kg/min input passes through with only amount/time scaling — no weight
    # multiplication/division at the conversion layer for the dominant
    # charting form. Combined with the pre-attached weight column above,
    # this eliminates the silent /kg-factor-dropped bug entirely.
    _cont_sed_preferred_units = {
        'propofol': 'mcg/kg/min',
        'midazolam': 'mg/min',
        'fentanyl': 'mcg/min',
        'hydromorphone': 'mg/min',
        'lorazepam': 'mg/min',
    }
    _cont_sed_converted, _cont_sed_convert_summary = convert_dose_units_by_med_category(
        cont_sed_with_weight,
        vitals_df=vitals_df,
        preferred_units=_cont_sed_preferred_units,
        override=True,
    )
    print(f"{len(_cont_sed_converted)} rows after unit conversion")

    _cont_sed_converted.rename(
        columns={
            'med_dose': 'med_dose_original',
            'med_dose_unit': 'med_dose_unit_original',
            'med_dose_converted': 'med_dose',
            'med_dose_unit_converted': 'med_dose_unit',
        },
        inplace=True,
    )

    cont_sed.df = _cont_sed_converted
    return


@app.cell
def _(apply_outlier_handling, cont_sed):
    apply_outlier_handling(cont_sed, outlier_config_path='config/outlier_config.yaml')
    cont_sed_converted = cont_sed.df
    print(f"{len(cont_sed_converted)} rows in cont_sed_converted")
    return (cont_sed_converted,)


@app.cell
def _(cont_sed_converted, t1, t2):
    cont_sed_w = mo.sql(
        f"""
        -- Pivot continuous sedation to wide format (one column per drug_unit)
        -- NOTE: stop events forced to dose=0 so forward-fill doesn't propagate stale rates (audit H1)
        WITH t1 AS (
        FROM cont_sed_converted
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit: med_category || '_' || REPLACE(med_dose_unit, '/', '_') || '_cont'
            , med_dose: CASE WHEN mar_action_category IN ('stop', 'not_given') THEN 0 ELSE med_dose END
        )
        , t2 AS (
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        )
        SELECT * FROM t2 ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_w,)


@app.cell
def _(cohort_hrly_grids_f, cont_sed_w):
    # NOTE: Materialize with .df() because we add pandas columns (_dh, _hr) afterward
    cont_sed_wg = mo.sql(
        f"""
        -- FULL JOIN continuous sedation wide table with hourly grid
        FROM cohort_hrly_grids_f g
        FULL JOIN cont_sed_w m USING (hospitalization_id, event_dttm)
        ORDER BY hospitalization_id, event_dttm
        """
    ).df()
    cont_sed_wg['_dh'] = cont_sed_wg['event_dttm'].dt.floor('h', ambiguous='NaT')
    cont_sed_wg['_hr'] = cont_sed_wg['event_dttm'].dt.hour
    print(f"cont_sed_wg rows: {len(cont_sed_wg)}")
    return (cont_sed_wg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Continuous Dose by Hour

    Inlined from `cont_sed_dose_by_hr.sql`: forward-fill, compute duration,
    coalesce nulls, multiply dose x duration, aggregate by hour.
    """)
    return


@app.cell
def _(cont_sed_wg, duckdb):
    cont_sed_t1 = duckdb.sql(
        """
        -- Forward-fill continuous sedation rates and compute duration between events
        FROM cont_sed_wg g
        SELECT hospitalization_id, event_dttm, _dh, _hr
            , LAST_VALUE(COLUMNS('_cont') IGNORE NULLS) OVER (
                PARTITION BY hospitalization_id ORDER BY event_dttm
            )
            , _duration: EXTRACT(EPOCH FROM (LEAD(event_dttm, 1, event_dttm) OVER w - event_dttm)) / 60.0
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        """
    )
    return (cont_sed_t1,)


@app.cell
def _(cont_sed_t1, duckdb):
    cont_sed_t2 = duckdb.sql(
        """
        -- Coalesce null sedation rates to 0
        FROM cont_sed_t1
        SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
            , COALESCE(COLUMNS('_cont'), 0)
        """
    )
    return (cont_sed_t2,)


@app.cell
def _(cont_sed_t2, duckdb):
    cont_sed_t3 = duckdb.sql(
        """
        -- Multiply dose rate by duration to get total dose per interval
        FROM cont_sed_t2
        SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
            , COLUMNS('_cont') * _duration
        """
    )
    return (cont_sed_t3,)


@app.cell
def _(cont_sed_t3, duckdb):
    # Aggregate continuous dose by hour, then rename cols to reflect the
    # post-aggregation unit. Rates are mg/min (or mcg/min) pre-sum, but after
    # SUM-per-hour the values represent total mg (or mcg) delivered in that
    # hour — so `_mg_min_cont` → `_mg_hr_cont` (2026-04-24 unit convention).
    # pandas.rename silently skips keys absent from the frame, so it is safe
    # even when the cohort has no records for a given drug.
    cont_sed_dose_by_hr = duckdb.sql(
        """
        FROM cont_sed_t3
        SELECT hospitalization_id, _dh, _hr
            , SUM(COLUMNS('_cont'))
        GROUP BY hospitalization_id, _dh, _hr
        ORDER BY hospitalization_id, _dh
        """
    ).df().rename(columns={
        # Propofol: mcg/kg/min input × _duration (min) summed by hour
        # → mcg/kg total in that hour = mcg/kg/hr rate.
        'propofol_mcg_kg_min_cont':  'propofol_mcg_kg_hr_cont',
        # Other drugs unchanged: unweighted mg or mcg per minute → per hour.
        'fentanyl_mcg_min_cont':     'fentanyl_mcg_hr_cont',
        'midazolam_mg_min_cont':     'midazolam_mg_hr_cont',
        'hydromorphone_mg_min_cont': 'hydromorphone_mg_hr_cont',
        'lorazepam_mg_min_cont':     'lorazepam_mg_hr_cont',
    })
    return (cont_sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Intermittent Sedation

    Load, dedup, convert, pivot, and aggregate intermittent sedation administrations.
    """)
    return


@app.cell
def _(cohort_hosp_ids):
    from clifpy import MedicationAdminIntermittent

    intm_sed = MedicationAdminIntermittent.from_file(
        config_path='config/config.json',
        columns=[
            'hospitalization_id', 'admin_dttm', 'med_name', 'med_category',
            'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category',
        ],
        filters={
            'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'],
            'hospitalization_id': cohort_hosp_ids,
        }
    )
    print(f"Intermittent sedation records: {len(intm_sed.df)}")
    return (intm_sed,)


@app.cell
def _(
    apply_outlier_handling,
    convert_dose_units_by_med_category,
    intm_sed,
    remove_meds_duplicates,
    vitals_df,
):
    _intm_sed_deduped = remove_meds_duplicates(intm_sed.df)
    _n_removed = len(intm_sed.df) - len(_intm_sed_deduped)
    print(f"Removed {_n_removed} ({_n_removed / len(intm_sed.df):.2%}) duplicates by MAR action")

    _intm_sed_preferred_units = {
        'propofol': 'mg',
        'midazolam': 'mg',
        'fentanyl': 'mcg',
        'hydromorphone': 'mg',
        'lorazepam': 'mg',
    }
    _intm_sed_converted, _intm_sed_convert_summary = convert_dose_units_by_med_category(
        _intm_sed_deduped,
        vitals_df=vitals_df,
        preferred_units=_intm_sed_preferred_units,
        override=True,
    )
    print(f"{len(_intm_sed_converted)} rows after unit conversion")

    _intm_sed_converted.rename(
        columns={
            'med_dose': 'med_dose_original',
            'med_dose_unit': 'med_dose_unit_original',
            'med_dose_converted': 'med_dose',
            'med_dose_unit_converted': 'med_dose_unit',
        },
        inplace=True,
    )

    intm_sed.df = _intm_sed_converted
    apply_outlier_handling(intm_sed, outlier_config_path='config/outlier_config.yaml')
    intm_sed_converted = intm_sed.df
    print(f"{len(intm_sed_converted)} rows in intm_sed_converted")
    return (intm_sed_converted,)


@app.cell
def _(intm_sed_converted, t1, t2):
    intm_sed_w = mo.sql(
        f"""
        -- Pivot intermittent sedation to wide format; zero out not_given doses
        WITH t1 AS (
        FROM intm_sed_converted
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit: med_category || '_' || REPLACE(med_dose_unit, '/', '_') || '_intm'
            , med_dose: CASE WHEN mar_action_category = 'not_given' THEN 0 ELSE med_dose END
        )
        , t2 AS (
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        )
        SELECT * FROM t2 ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_w,)


@app.cell
def _(cohort_hrly_grids_f, intm_sed_w):
    # NOTE: Materialize with .df() because we add pandas columns (_dh, _hr) afterward
    intm_sed_wg = mo.sql(
        f"""
        -- FULL JOIN intermittent sedation wide table with hourly grid
        FROM cohort_hrly_grids_f g
        FULL JOIN intm_sed_w m USING (hospitalization_id, event_dttm)
        ORDER BY hospitalization_id, event_dttm
        """
    ).df()
    intm_sed_wg['_dh'] = intm_sed_wg['event_dttm'].dt.floor('h', ambiguous='NaT')
    intm_sed_wg['_hr'] = intm_sed_wg['event_dttm'].dt.hour
    print(f"intm_sed_wg rows: {len(intm_sed_wg)}")
    return (intm_sed_wg,)


@app.cell
def _(duckdb, intm_sed_wg):
    # Aggregate intermittent dose by hour, then rename cols to match the
    # post-aggregation "per hour" unit (2026-04-24 unit convention). After
    # SUM-per-hour, `propofol_mg_intm` represents total mg delivered that
    # hour from intermittent admin, so the column name gains the `_hr_`
    # segment. pandas.rename silently skips keys that aren't present.
    intm_sed_dose_by_hr = duckdb.sql(
        """
        FROM intm_sed_wg
        SELECT hospitalization_id, _dh
            , SUM(COALESCE(COLUMNS('_intm'), 0))
        GROUP BY hospitalization_id, _dh
        ORDER BY hospitalization_id, _dh
        """
    ).df().rename(columns={
        'propofol_mg_intm':      'propofol_mg_hr_intm',
        'fentanyl_mcg_intm':     'fentanyl_mcg_hr_intm',
        'midazolam_mg_intm':     'midazolam_mg_hr_intm',
        'hydromorphone_mg_intm': 'hydromorphone_mg_hr_intm',
        'lorazepam_mg_intm':     'lorazepam_mg_hr_intm',
    })
    return (intm_sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge Continuous + Intermittent

    Join hourly grids with continuous and intermittent doses,
    compute drug totals and equivalency doses.
    """)
    return


@app.cell
def _(cohort_hrly_grids_f, cont_sed_dose_by_hr, intm_sed_dose_by_hr):
    sed_dose_by_hr = mo.sql(
        f"""
        -- Join hourly grid with continuous and intermittent doses; compute equivalencies
        WITH t1 AS (
        FROM cohort_hrly_grids_f g
        LEFT JOIN intm_sed_dose_by_hr i USING (hospitalization_id, _dh)
        LEFT JOIN cont_sed_dose_by_hr c USING (hospitalization_id, _dh)
        SELECT *
        )
        , t2 AS (
        FROM t1
        SELECT *
            , fentanyl_mcg_total: fentanyl_mcg_hr_intm + fentanyl_mcg_hr_cont
            , hydromorphone_mg_total: hydromorphone_mg_hr_intm + hydromorphone_mg_hr_cont
            , lorazepam_mg_total: lorazepam_mg_hr_intm + lorazepam_mg_hr_cont
            , midazolam_mg_total: midazolam_mg_hr_intm + midazolam_mg_hr_cont
            -- Phase 2: propofol is now in mcg/kg/hr (continuous-only). The
            -- intermittent propofol path (`propofol_mg_hr_intm`) is bolus
            -- mg/dose with no defined duration, so converting to mcg/kg/min
            -- is ill-defined. ICU bolus propofol is rare (mostly procedural)
            -- compared to continuous pump infusion, which dominates the
            -- exposure signal. We track intm propofol separately (still as
            -- mg/hr) but exclude it from the rate-based descriptive.
            , prop_mcg_kg_total: propofol_mcg_kg_hr_cont
            , _midazeq_mg_total: lorazepam_mg_total * 2 + midazolam_mg_total
            , _fenteq_mcg_total: hydromorphone_mg_total * 50 + fentanyl_mcg_total
        )
        SELECT * FROM t2 ORDER BY hospitalization_id, _dh
        """
    )
    return (sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Daily Aggregation

    Aggregate per (hospitalization_id, _nth_day, _shift), then pivot to
    day/night columns for the analytical dataset.
    """)
    return


@app.cell
def _(duckdb, sed_dose_by_hr):
    sed_dose_agg = duckdb.sql(
        """
        -- Aggregate dose totals AND hour counts per hospitalization, day, shift.
        -- n_hours is used downstream to convert totals to per-hour dose rates,
        -- which avoids partial-shift bias in the Dose by Shift descriptive table
        -- and makes model dose coefficients interpretable as rates (see
        -- 05_analytical_dataset.py for the ÷12 rate conversion).
        -- Column-name convention (2026-04-24): totals carry their unit as
        -- suffix (`_mg` for propofol+midaz totals, `_mcg` for fentanyl totals);
        -- per-hour rates downstream carry `_mg_hr` / `_mcg_hr`.
        FROM sed_dose_by_hr
        SELECT hospitalization_id, _nth_day, _shift
            -- Phase 2: propofol now sums to total mcg/kg over the shift
            -- (continuous infusion only). Downstream 05 divides by hours
            -- and minutes to produce the per-min rate `_prop_day_mcg_kg_min`.
            , SUM(prop_mcg_kg_total) AS prop_mcg_kg
            , SUM(_fenteq_mcg_total) AS fenteq_mcg
            , SUM(_midazeq_mg_total) AS midazeq_mg
            , COUNT(*) AS n_hours
        GROUP BY hospitalization_id, _nth_day, _shift
        ORDER BY hospitalization_id, _nth_day, _shift
        """
    ).df()
    print(f"sed_dose_agg rows: {len(sed_dose_agg)}")
    return (sed_dose_agg,)


@app.cell
def _(sed_dose_agg):
    # NOTE: Pivot to wide day/night columns using pandas .pivot()
    # n_hours_day/night flow through so downstream can compute per-hour dose rates
    # that correctly handle partial shifts (intubation / extubation day edges).
    # Unit-suffixed column names (e.g. prop_day_mg = total mg over 12h day shift)
    # match the 2026-04-24 naming convention; see 05_analytical_dataset.py.
    sed_dose_daily = sed_dose_agg.pivot(
        index=['hospitalization_id', '_nth_day'],
        columns='_shift',
        values=['prop_mcg_kg', 'fenteq_mcg', 'midazeq_mg', 'n_hours'],
    ).reset_index()

    # Flatten MultiIndex columns to clean names. The pivot preserves the
    # order: value_col1×shifts, value_col2×shifts, ... so pairs are always
    # (value_col, 'day') then (value_col, 'night').
    sed_dose_daily.columns = [
        'hospitalization_id', '_nth_day',
        # Phase 2: propofol totals are now mcg/kg over the shift (was mg).
        'prop_day_mcg_kg', 'prop_night_mcg_kg',
        'fenteq_day_mcg', 'fenteq_night_mcg',
        'midazeq_day_mg', 'midazeq_night_mg',
        'n_hours_day', 'n_hours_night',
    ]
    # Drop any columns that ended up as None (e.g. if a shift is missing)
    sed_dose_daily = sed_dose_daily.loc[:, [c for c in sed_dose_daily.columns if c is not None]]
    print(f"sed_dose_daily rows: {len(sed_dose_daily)}")
    return (sed_dose_daily,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs
    """)
    return


@app.cell
def _(SITE_NAME, sed_dose_agg, sed_dose_by_hr, sed_dose_daily):
    sed_dose_daily.to_parquet(f"output/{SITE_NAME}/sed_dose_daily.parquet", index=False)
    sed_dose_agg.to_parquet(f"output/{SITE_NAME}/sed_dose_agg.parquet", index=False)
    sed_dose_by_hr.df().to_parquet(f"output/{SITE_NAME}/sed_dose_by_hr.parquet", index=False)

    print(f"Saved: output/{SITE_NAME}/sed_dose_daily.parquet")
    print(f"Saved: output/{SITE_NAME}/sed_dose_agg.parquet")
    print(f"Saved: output/{SITE_NAME}/sed_dose_by_hr.parquet")
    return


if __name__ == "__main__":
    app.run()
