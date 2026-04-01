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
    return


@app.cell
def _(pd):
    cohort_hrly_grids_f = pd.read_parquet("output/cohort_hrly_grids.parquet")
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
def _(
    cont_sed,
    cont_sed_deduped,
    convert_dose_units_by_med_category,
    vitals_df,
):
    _cont_sed_preferred_units = {
        'propofol': 'mg/min',
        'midazolam': 'mg/min',
        'fentanyl': 'mcg/min',
        'hydromorphone': 'mg/min',
        'lorazepam': 'mg/min',
    }
    _cont_sed_converted, _cont_sed_convert_summary = convert_dose_units_by_med_category(
        cont_sed_deduped,
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
    cont_sed_dose_by_hr = duckdb.sql(
        """
        -- Aggregate continuous dose by hour
        FROM cont_sed_t3
        SELECT hospitalization_id, _dh, _hr
            , SUM(COLUMNS('_cont'))
        GROUP BY hospitalization_id, _dh, _hr
        ORDER BY hospitalization_id, _dh
        """
    )
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
    intm_sed_dose_by_hr = duckdb.sql(
        """
        -- Aggregate intermittent dose by hour
        FROM intm_sed_wg
        SELECT hospitalization_id, _dh
            , SUM(COALESCE(COLUMNS('_intm'), 0))
        GROUP BY hospitalization_id, _dh
        ORDER BY hospitalization_id, _dh
        """
    )
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
            , fentanyl_mcg_total: fentanyl_mcg_intm + fentanyl_mcg_min_cont
            , hydromorphone_mg_total: hydromorphone_mg_intm + hydromorphone_mg_min_cont
            , lorazepam_mg_total: lorazepam_mg_intm + lorazepam_mg_min_cont
            , midazolam_mg_total: midazolam_mg_intm + midazolam_mg_min_cont
            , prop_mg_total: propofol_mg_intm + propofol_mg_min_cont
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
        -- Aggregate dose totals per hospitalization, day, shift
        FROM sed_dose_by_hr
        SELECT hospitalization_id, _nth_day, _shift
            , SUM(prop_mg_total) AS prop_mg_total
            , SUM(_fenteq_mcg_total) AS fenteq_mcg_total
            , SUM(_midazeq_mg_total) AS midazeq_mg_total
        GROUP BY hospitalization_id, _nth_day, _shift
        ORDER BY hospitalization_id, _nth_day, _shift
        """
    ).df()
    print(f"sed_dose_agg rows: {len(sed_dose_agg)}")
    return (sed_dose_agg,)


@app.cell
def _(sed_dose_agg):
    # NOTE: Pivot to wide day/night columns using pandas .pivot()
    sed_dose_daily = sed_dose_agg.pivot(
        index=['hospitalization_id', '_nth_day'],
        columns='_shift',
        values=['prop_mg_total', 'fenteq_mcg_total', 'midazeq_mg_total'],
    ).reset_index()

    # Flatten MultiIndex columns to clean names
    sed_dose_daily.columns = [
        'hospitalization_id', '_nth_day',
        'prop_day', 'prop_night',
        'fenteq_day', 'fenteq_night',
        'midazeq_day', 'midazeq_night',
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
def _(sed_dose_agg, sed_dose_by_hr, sed_dose_daily):
    sed_dose_daily.to_parquet("output/sed_dose_daily.parquet", index=False)
    sed_dose_agg.to_parquet("output/sed_dose_agg.parquet", index=False)
    sed_dose_by_hr.df().to_parquet("output/sed_dose_by_hr.parquet", index=False)

    print("Saved: output/sed_dose_daily.parquet")
    print("Saved: output/sed_dose_agg.parquet")
    print("Saved: output/sed_dose_by_hr.parquet")
    return


if __name__ == "__main__":
    app.run()
