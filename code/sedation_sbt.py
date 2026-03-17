# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "clifpy>=0.3.1",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
#     "statsmodels>=0.14.5",
#     "tableone>=0.9.5",
#     "scipy",
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App()

with app.setup:
    import marimo as mo
    import os
    from pathlib import Path
    RERUN_WATERFALL = False


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Init
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Import
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

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH="config/config.json"
    co = ClifOrchestrator(config_path=CONFIG_PATH)

    # Ensure the output subdirs exists
    os.makedirs("output/intermediate", exist_ok=True)
    os.makedirs("output/final", exist_ok=True)
    return (
        CONFIG_PATH,
        apply_outlier_handling,
        co,
        convert_dose_units_by_med_category,
        duckdb,
        get_config_or_params,
        pd,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params):
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    print(SITE_NAME)
    return (SITE_NAME,)


@app.cell
def _():
    from datetime import datetime
    CURRENT_TIME_STR = datetime.now().strftime('%Y%m%d_%H%M%S')
    return (CURRENT_TIME_STR,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Utils
    """)
    return


@app.function
def read_sql(path: str) -> str:
    """Read a SQL file and return its contents as a string."""
    with open(path) as f:
        return f.read()


@app.cell
def _(duckdb, pd):
    def add_day_shift_id(
        df: pd.DataFrame, timestamp_name="event_dttm"
    ) -> pd.DataFrame:
        df["_dh"] = df[timestamp_name].dt.floor("h", ambiguous="NaT")
        df["_hr"] = df[timestamp_name].dt.hour
        _q = """
        WITH day_starts AS (
            FROM df
            SELECT *
                , _shift: CASE WHEN _hr >= 7 AND _hr < 19 THEN 'day' ELSE 'night' END
                , _is_day_start: CASE
                    WHEN _hr = 7 AND COALESCE(LAG(_hr) OVER w, -1) != 7 THEN 1
                    ELSE 0 END
            WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _dh)       
        )
        FROM day_starts
        -- INNER JOIN cohort_hosp_ids_df USING (hospitalization_id)
        SELECT *
            , _nth_day: SUM(_is_day_start) OVER w
            , _day_shift: 'day' || _nth_day::INT::TEXT || '_' || _shift
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _dh)       
        ORDER BY hospitalization_id, _dh
        """
        return duckdb.sql(_q).df()

    return (add_day_shift_id,)


@app.cell
def _(duckdb, pd):
    def remove_meds_duplicates(meds_df: pd.DataFrame) -> pd.DataFrame:
        if 'mar_action_category' not in meds_df.columns:
            print('mar_action_category not available, deduping by mar_action_name instead')
            _q = f"""
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY 
                -- apply mar action dedup logic
                CASE WHEN mar_action_name IS NULL THEN 10
                    WHEN regexp_matches(mar_action_name, 'verify', 'i') THEN 9
                    WHEN regexp_matches(mar_action_name, '(stopped)|(held)|(paused)|(completed)', 'i') THEN 8
                    ELSE 1 END,
                -- if tied at the same mar action, deprioritize zero or null doses
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                -- prioritize larger doses
                med_dose DESC 
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
        else:
            _q = f"""
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY 
                -- apply mar action dedup logic
                CASE WHEN mar_action_category IS NULL THEN 10
                    WHEN mar_action_category in ('verify', 'not_given') THEN 9
                    WHEN mar_action_category = 'stop' THEN 8
                    WHEN mar_action_category = 'going' THEN 7
                    ELSE 1 END,
                -- if tied at the same mar action, deprioritize zero or null doses
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                -- prioritize larger doses
                med_dose DESC 
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
        return duckdb.sql(_q).to_df()

    return (remove_meds_duplicates,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Cohort ID
    """)
    return


@app.cell
def _():
    from clifpy import Adt, Hospitalization

    adt = Adt.from_file(
        config_path = 'config/config.json',
        columns = ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category'],
        filters = {
            'location_category': ['icu']
        }
        )

    hosp_ids_w_icu_stays = adt.df['hospitalization_id'].unique().tolist()
    return Adt, Hospitalization, hosp_ids_w_icu_stays


@app.cell
def _(Adt, Hospitalization, hosp_ids_w_icu_stays):
    from clifpy.utils.stitching_encounters import stitch_encounters

    # Load your dataframes
    hosp_w_icu_stays = Hospitalization.from_file(
        config_path = 'config/config.json',
        filters = {
            'hospitalization_id': hosp_ids_w_icu_stays
        }
    )
    adt_w_icu_stays = Adt.from_file(
        config_path = 'config/config.json',
        filters = {
            'hospitalization_id': hosp_ids_w_icu_stays
        }
        # columns = ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category'],
        )

    # Perform stitching
    hosp_stitched, adt_stitched, encounter_mapping = stitch_encounters(
        hospitalization=hosp_w_icu_stays.df,
        adt=adt_w_icu_stays.df,
        time_interval=12  # 12-hour window
    )
    return


@app.cell
def _(
    CONFIG_PATH,
    SITE_NAME,
    apply_outlier_handling,
    hosp_ids_w_icu_stays,
    pd,
):
    from clifpy import RespiratorySupport

    resp_processed_path = (
        f"output/intermediate/{SITE_NAME}_resp_processed_bf.parquet"
    )

    if not os.path.exists(resp_processed_path) or RERUN_WATERFALL:
        cohort_resp = RespiratorySupport.from_file(
            config_path=CONFIG_PATH,
            columns=[
                "hospitalization_id",
                "recorded_dttm",
                "device_name",
                "device_category",
                "mode_name",
                "mode_category",
                "fio2_set",
                "peep_set",
                "pressure_support_set",
                "resp_rate_set",
                "tidal_volume_set",
                "peak_inspiratory_pressure_set",
                "tracheostomy",
            ],
            filters={"hospitalization_id": hosp_ids_w_icu_stays},
        )
        apply_outlier_handling(
            cohort_resp, outlier_config_path="config/outlier_config.yaml"
        )
        cohort_resp_p = cohort_resp.waterfall(bfill=True)
        cohort_resp_p.df.to_parquet(resp_processed_path)
        resp_p = cohort_resp_p.df
    else:
        print(f"Loading {resp_processed_path}")
        resp_p = pd.read_parquet(resp_processed_path)
    return (resp_p,)


@app.cell
def _(resp_p):
    resp_p['tracheostomy'] = resp_p['tracheostomy'].fillna(0).astype(int)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Time grids
    """)
    return


@app.cell
def _():
    all_streaks = mo.sql(read_sql('code/cohort_id.sql'))
    return (all_streaks,)


@app.cell
def _(all_streaks):
    # build the observation-window hourly time grids
    # start and end time for each hospitalization_id is the start and end of the first IMV streak of 24 hours or longer
    # the end of the imv streak means extubation, which we evaulates whether as successful or not
    cohort_imv_streaks_f = mo.sql(
        f"""
        FROM all_streaks
        SELECT hospitalization_id, _streak_id, _start_dttm, _end_dttm, _duration_hrs
        WHERE _at_least_24h = 1 -- has to last for at least 24 hours
        AND _on_imv = 1 -- has to be an IMV streak
        AND _streak_id = 1 -- has to bethe first IMV streak
        """
    )
    return (cohort_imv_streaks_f,)


@app.cell
def _(cohort_imv_streaks_f):
    cohort_hosp_ids = cohort_imv_streaks_f['hospitalization_id'].unique().tolist()
    return (cohort_hosp_ids,)


@app.cell
def _(cohort_imv_streaks_f):
    cohort_imv_streaks_f['_start_hr'] = cohort_imv_streaks_f['_start_dttm'].dt.floor('h', ambiguous='NaT')
    cohort_imv_streaks_f['_end_hr'] = cohort_imv_streaks_f['_end_dttm'].dt.ceil('h', ambiguous='NaT')
    cohort_hrly_grids = mo.sql(
        f"""
    SELECT 
    hospitalization_id,
    unnest(generate_series(_start_hr, _end_hr, INTERVAL '1 hour')) AS event_dttm
    FROM cohort_imv_streaks_f
    ORDER BY hospitalization_id, event_dttm
    """
    )
    return (cohort_hrly_grids,)


@app.cell
def _(add_day_shift_id, cohort_hrly_grids):
    cohort_hrly_grids_f = add_day_shift_id(cohort_hrly_grids)
    assert len(cohort_hrly_grids_f) == len(cohort_hrly_grids), 'length altered'
    return (cohort_hrly_grids_f,)


@app.cell
def _(cohort_hrly_grids_f):
    cohort_shift_change_grids = cohort_hrly_grids_f[cohort_hrly_grids_f['_hr'].isin([7, 19])]
    return (cohort_shift_change_grids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exclude: neuromuscular blocking agent
    """)
    return


@app.cell
def _():
    from clifpy import MedicationAdminContinuous

    # neuromuscular blocking agent
    nmb = MedicationAdminContinuous.from_file(
        config_path = 'config/config.json',
        columns = ['hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit'],
        filters = {
            'med_category': ['cisatracurium', 'vecuronium', 'rocuronium']
        }
    )

    nmb_df = nmb.df
    return (MedicationAdminContinuous,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Patient & Hosp
    """)
    return


@app.cell
def _(Hospitalization, apply_outlier_handling):
    from clifpy import Patient
    pt = Patient.from_file(config_path='config/config.json', columns=['patient_id', 'death_dttm'])
    pt_df = pt.df
    hosp = Hospitalization.from_file(config_path='config/config.json', columns=['patient_id', 'hospitalization_id', 'discharge_dttm', 'discharge_category', 'age_at_admission'])
    apply_outlier_handling(hosp, outlier_config_path='config/outlier_config.yaml')
    hosp_df = hosp.df
    return (hosp_df,)


@app.cell
def _(cohort_imv_streaks_f, hosp_df):
    pt_to_hosp_id_mapper = mo.sql(
        f"""
        FROM hosp_df
        INNER JOIN cohort_imv_streaks_f USING (hospitalization_id)
        SELECT DISTINCT patient_id, hospitalization_id
        """
    )
    return (pt_to_hosp_id_mapper,)


@app.cell
def _(pt_to_hosp_id_mapper):
    cohort_pt_ids = pt_to_hosp_id_mapper['patient_id'].tolist()
    return (cohort_pt_ids,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # SBT outcomes
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Vitals
    """)
    return


@app.cell
def _(co):
    vitals_path = co.data_directory + '/clif_vitals.parquet'
    last_vitals_df = mo.sql(
        f"""
    -- find the latest recorded vital for each hospitalization
    FROM '{vitals_path}'
    SELECT hospitalization_id
    , MAX(recorded_dttm) AS recorded_dttm
    GROUP BY hospitalization_id
    """
    )
    return


@app.cell
def _(apply_outlier_handling, cohort_hosp_ids):
    from clifpy import Vitals
    vitals = Vitals.from_file(
        config_path = 'config/config.json',
        columns = ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        filters = {
            'vital_category': [
                'weight_kg' # 'spo2', , 'respiratory_rate', 'heart_rate'
                ],
            'hospitalization_id': cohort_hosp_ids
        }
        )
    apply_outlier_handling(vitals, outlier_config_path = 'config/outlier_config.yaml')
    vitals_df = vitals.df
    return (vitals_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Code status
    """)
    return


@app.cell
def _(cohort_pt_ids, duckdb):
    from clifpy import CodeStatus
    cs = CodeStatus.from_file(config_path='config/config.json', columns=['patient_id', 'start_dttm', 'code_status_category'], filters={'patient_id': cohort_pt_ids})
    cs_df = cs.df
    _q = """
    FROM cs_df
    LEFT JOIN pt_to_hosp_id_mapper USING (patient_id)
    SELECT hospitalization_id, start_dttm, code_status_category
    """
    cs_df = duckdb.sql(_q).df()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Resp
    """)
    return


@app.cell
def _():
    sbt_outcomes = mo.sql(read_sql('code/sbt.sql'))
    # assert len(sbt_outcomes_f) == len(resp_p), 'length altered'
    return (sbt_outcomes,)


@app.cell
def _(sbt_outcomes):
    mo.sql(
        f"""
    -- look for represenative examples to eyeball
    FROM sbt_outcomes
    SELECT hospitalization_id
    , MAX(_trach_1st) AS _trach_1st
    , MAX(_fail_extub) AS _fail_extub
    , COUNT(*) AS _n
    GROUP BY hospitalization_id
    HAVING _trach_1st = 1 AND _fail_extub = 1 AND _N > 100
    ORDER BY _n
    LIMIT 10
    """
    )
    return


@app.cell
def _(sbt_outcomes):
    sbt_outcomes['_dh'] = sbt_outcomes['event_dttm'].dt.floor('h', ambiguous='NaT')
    sbt_outcomes_hrly = mo.sql(
        f"""
    FROM sbt_outcomes
    SELECT hospitalization_id, _dh
    -- , _nth_day
    , sbt_done: COALESCE(MAX(sbt_done), 0)
    -- , _extub_1st: COALESCE(MAX(_extub_1st), 0)
    , success_extub: COALESCE(MAX(_success_extub), 0)
    , trach_1st: COALESCE(MAX(_trach_1st), 0)
    GROUP BY hospitalization_id, _dh
    ORDER BY hospitalization_id, _dh
    """
    )
    return (sbt_outcomes_hrly,)


@app.cell
def _(cohort_hrly_grids_f, sbt_outcomes_hrly):
    cohort_sbt_outcomes_hrly = mo.sql(
        f"""
        FROM cohort_hrly_grids_f
        LEFT JOIN sbt_outcomes_hrly USING (hospitalization_id, _dh)
        SELECT *
        ORDER BY hospitalization_id, _dh
        """
    )
    return (cohort_sbt_outcomes_hrly,)


@app.cell
def _(cohort_sbt_outcomes_hrly):
    cohort_sbt_outcomes_daily = mo.sql(
        f"""
        FROM cohort_sbt_outcomes_hrly
        SELECT hospitalization_id, _nth_day
        , sbt_done: COALESCE(MAX(sbt_done), 0)
        , success_extub: COALESCE(MAX(success_extub), 0)
        , trach_1st: COALESCE(MAX(trach_1st), 0)
        , n_hrs: COUNT(*)
        GROUP BY hospitalization_id, _nth_day
        ORDER BY hospitalization_id, _nth_day
        """
    )
    return (cohort_sbt_outcomes_daily,)


@app.cell
def _(cohort_sbt_outcomes_daily):
    cohort_sbt_outcomes_by_pt = mo.sql(
        f"""
    FROM cohort_sbt_outcomes_daily
    SELECT
    hospitalization_id
    , sbt_done: COALESCE(MAX(sbt_done), 0)
    , success_extub: COALESCE(MAX(success_extub), 0)
    GROUP BY hospitalization_id
    """
    )
    print(f"success_extub rate: {cohort_sbt_outcomes_by_pt['success_extub'].mean()}")
    print(f"sbt_done rate per day: {cohort_sbt_outcomes_daily['sbt_done'].mean()}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Trajectory
    """)
    return


@app.cell(disabled=True)
def _():
    # sbt_outcomes_f_w_ids = add_day_shift_id(sbt_outcomes_f)
    # assert len(sbt_outcomes_f_w_ids) == len(sbt_outcomes_f), 'length altered'
    return


@app.cell(disabled=True)
def _():
    # q = """
    # WITH agg as (
    #     FROM sbt_outcomes_f_w_ids
    #     SELECT hospitalization_id
    #         , _nth_day
    #         , sbt_done: COALESCE(MAX(sbt_done), 0)
    #         , _extub_1st: COALESCE(MAX(_extub_1st), 0)
    #         , success_extub: COALESCE(MAX(_success_extub), 0)
    #         , _intub: COALESCE(MAX(_intub), 0)
    #         , _trach_1st: COALESCE(MAX(_trach_1st), 0)
    #         , _fail_extub: COALESCE(MAX(_fail_extub), 0)
    #         , _withdrawl_lst: COALESCE(MAX(_withdrawl_lst), 0)
    #         , _death_after_extub_wo_reintub: COALESCE(MAX(_death_after_extub_wo_reintub), 0)
    #         , discharge: ANY_VALUE(discharge_category)
    #         , code_status: ANY_VALUE(code_status_category ORDER BY cs_start_dttm DESC)

    #     --WHERE hospitalization_id in ('20001361', '20004088', '20005024')
    #     GROUP BY hospitalization_id, _nth_day
    # )
    # , aug as (
    #     FROM agg
    #     SELECT *
    #         , _exit_sum: success_extub + _trach_1st + _fail_extub + _withdrawl_lst + _death_after_extub_wo_reintub
    #         , _exit: _exit_sum::BOOL::INT
    # )
    # SELECT *
    # FROM aug
    # --WHERE hospitalization_id IN ('20001361', '20004088', '20005024', '20006409', '21341369', '20134240', '20008807', '20014600')
    # ORDER BY hospitalization_id, _nth_day
    # """
    # resp_traj_by_days = duckdb.sql(q).df()
    # # resp_traj_by_days.head()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Covariates
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## pH
    """)
    return


@app.cell
def _(apply_outlier_handling, cohort_hosp_ids, duckdb):
    from clifpy import Labs
    labs = Labs.from_file(config_path='config/config.json', columns=['hospitalization_id', 'lab_order_dttm', 'lab_result_dttm', 'lab_category', 'lab_value_numeric'], filters={'hospitalization_id': cohort_hosp_ids, 'lab_category': ['ph_arterial', 'ph_venous']})
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
    FROM cohort_shift_change_grids g
    ASOF LEFT JOIN labs_w l ON
    g.hospitalization_id = l.hospitalization_id 
    AND l.lab_order_dttm <= g.event_dttm
    SELECT g.*
    , l.lab_order_dttm
    , l.lab_result_dttm
    , l.ph_arterial
    , l.ph_venous -- change it to ph_venous: NULL if ph_venous is not available
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
    assert len(ph_df) == len(cohort_shift_change_grids), 'length altered'
    return (ph_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## P/F ratio
    """)
    return


@app.cell
def _(Labs, apply_outlier_handling, cohort_hosp_ids, duckdb):
    po2 = Labs.from_file(config_path='config/config.json', columns=['hospitalization_id', 'lab_order_dttm', 'lab_result_dttm', 'lab_category', 'lab_value_numeric'], filters={'hospitalization_id': cohort_hosp_ids, 'lab_category': ['po2_arterial']})
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
            WHEN pf is NULL THEN 'missing'
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
def _(
    MedicationAdminContinuous,
    apply_outlier_handling,
    cohort_hosp_ids,
    convert_dose_units_by_med_category,
    remove_meds_duplicates,
    vitals_df,
):
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
                "med_category": [
                    "norepinephrine",
                    "epinephrine",
                    "phenylephrine",
                    "dopamine",
                    "vasopressin",
                    "angiotensin",
                ],
                "hospitalization_id": cohort_hosp_ids,
            },
        )
    except Exception as e:
        print(f"Loading without mar_action_category instead")
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
                "med_category": [
                    "norepinephrine",
                    "epinephrine",
                    "phenylephrine",
                    "dopamine",
                    "vasopressin",
                    "angiotensin",
                ],
                "hospitalization_id": cohort_hosp_ids,
            },
        )
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
    cont_veso_converted, cont_veso_convert_summary = (
        convert_dose_units_by_med_category(
            cont_veso_deduped,
            vitals_df=vitals_df,
            preferred_units=cont_veso_preferred_units,
            override=True,
        )
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
    apply_outlier_handling(
        cont_veso, outlier_config_path="config/outlier_config.yaml"
    )
    cont_veso_converted = (
        cont_veso.df
    )  # 'dobutamine': 'mcg/kg/min',  # 'milrinone': 'mcg/kg/min',
    return (cont_veso_converted,)


@app.cell
def _(cohort_shift_change_grids, cont_veso_converted):
    vaso_df = mo.sql(
        f"""
        FROM
            cohort_shift_change_grids g
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'dopamine'
            ) m1 ON g.hospitalization_id = m1.hospitalization_id
            AND m1.admin_dttm <= g.event_dttm
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'norepinephrine'
            ) m2 ON g.hospitalization_id = m2.hospitalization_id
            AND m2.admin_dttm <= g.event_dttm
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'epinephrine'
            ) m3 ON g.hospitalization_id = m3.hospitalization_id
            AND m3.admin_dttm <= g.event_dttm
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'phenylephrine'
            ) m4 ON g.hospitalization_id = m4.hospitalization_id
            AND m4.admin_dttm <= g.event_dttm
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'angiotensin'
            ) m5 ON g.hospitalization_id = m5.hospitalization_id
            AND m5.admin_dttm <= g.event_dttm
            ASOF LEFT JOIN (
                SELECT
                    *
                FROM
                    cont_veso_converted
                WHERE
                    med_category = 'vasopressin'
            ) m6 ON g.hospitalization_id = m6.hospitalization_id
            AND m6.admin_dttm <= g.event_dttm
        SELECT
            g.*,
            dopamine: COALESCE(m1.med_dose, 0),
            norepinephrine: COALESCE(m2.med_dose, 0),
            epinephrine: COALESCE(m3.med_dose, 0),
            phenylephrine: COALESCE(m4.med_dose, 0),
            angiotensin: COALESCE(m5.med_dose, 0),
            vasopressin: COALESCE(m6.med_dose, 0),
            _nee: norepinephrine + epinephrine + phenylephrine / 10.0 + dopamine / 100.0 + vasopressin * 2.5 + angiotensin * 10
        ORDER BY
            g.hospitalization_id,
            g.event_dttm
        """
    )
    return (vaso_df,)


@app.cell
def _(cohort_shift_change_grids, ph_df):
    assert len(ph_df) == len(cohort_shift_change_grids), 'length altered'
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge covariates
    """)
    return


@app.cell
def _(cohort_shift_change_grids, pf_df, ph_df, vaso_df):
    covs = mo.sql(
        f"""
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
    covs_daily = mo.sql(
        f"""
    FROM covs
    SELECT hospitalization_id
    --, event_dttm: MIN(event_dttm)
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
    return covs, covs_daily


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Sedation dose
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Continuous
    """)
    return


@app.cell
def _(
    MedicationAdminContinuous,
    cohort_hosp_ids,
    convert_dose_units_by_med_category,
    remove_meds_duplicates,
    vitals_df,
):
    cont_sed = MedicationAdminContinuous.from_file(config_path='config/config.json', columns=['hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category'], filters={'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'], 'hospitalization_id': cohort_hosp_ids})
    cont_sed_preferred_units = {'propofol': 'mg/min', 'midazolam': 'mg/min', 'fentanyl': 'mcg/min', 'hydromorphone': 'mg/min', 'lorazepam': 'mg/min'}
    cont_sed_deduped = remove_meds_duplicates(cont_sed.df)
    _n_removed = len(cont_sed.df) - len(cont_sed_deduped)
    print(f'Removed {_n_removed} ({_n_removed / len(cont_sed.df):.2%}) duplicates by MAR action')
    _cont_sed_converted, _cont_sed_convert_summary = convert_dose_units_by_med_category(cont_sed_deduped, vitals_df=vitals_df, preferred_units=cont_sed_preferred_units, override=True)
    print(f'{len(_cont_sed_converted)} rows in intm_sed_converted')
    _cont_sed_converted.rename(columns={'med_dose': 'med_dose_original', 'med_dose_unit': 'med_dose_unit_original', 'med_dose_converted': 'med_dose', 'med_dose_unit_converted': 'med_dose_unit'}, inplace=True)
    # apply_outlier_handling(cont_sed, outlier_config_path = 'config/outlier_config.yaml')
    # cont_sed.df.drop(columns=['mar_action_category'], inplace=True)
    cont_sed.df = _cont_sed_converted  #'dexmedetomidine': 'mcg/min',  #'ketamine': 'mg/min',  #'morphine': 'mg/min',  #'remifentanil': 'mcg/min',  #'pentobarbital': 'mg/min',
    return (cont_sed,)


@app.cell
def _(apply_outlier_handling, cont_sed):
    apply_outlier_handling(cont_sed, outlier_config_path='config/outlier_config.yaml')
    cont_sed_converted = cont_sed.df
    print(f'{len(cont_sed_converted)} rows in cont_sed_converted')
    return (cont_sed_converted,)


@app.cell
def _(cont_sed_converted, t1, t2):
    cont_sed_w = mo.sql(
        f"""
        -- converting to wide format
        WITH t1 AS (
        SELECT hospitalization_id
            , admin_dttm as event_dttm
            , med_category_unit: med_category || '_' || REPLACE(med_dose_unit, '/', '_') || '_cont'
            , med_dose
        FROM cont_sed_converted
        )
        , t2 AS (
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        )
        SELECT *
        FROM t2
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_w,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Intermittent
    """)
    return


@app.cell
def _(
    apply_outlier_handling,
    cohort_hosp_ids,
    cont_sed_converted,
    convert_dose_units_by_med_category,
    remove_meds_duplicates,
    vitals_df,
):
    from clifpy import MedicationAdminIntermittent
    intm_sed = MedicationAdminIntermittent.from_file(config_path='config/config.json', columns=['hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category'], filters={'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'], 'hospitalization_id': cohort_hosp_ids})
    intm_sed_preferred_units = {'propofol': 'mg', 'midazolam': 'mg', 'fentanyl': 'mcg', 'hydromorphone': 'mg', 'lorazepam': 'mg'}
    intm_sed_deduped = remove_meds_duplicates(intm_sed.df)
    _n_removed = len(intm_sed.df) - len(intm_sed_deduped)
    print(f'Removed {_n_removed} ({_n_removed / len(intm_sed.df):.2%}) duplicates by MAR action')
    intm_sed_converted, intm_sed_convert_summary = convert_dose_units_by_med_category(intm_sed_deduped, vitals_df=vitals_df, preferred_units=intm_sed_preferred_units, override=True)
    print(f'{len(cont_sed_converted)} rows in intm_sed_converted')
    intm_sed_converted.rename(columns={'med_dose': 'med_dose_original', 'med_dose_unit': 'med_dose_unit_original', 'med_dose_converted': 'med_dose', 'med_dose_unit_converted': 'med_dose_unit'}, inplace=True)
    intm_sed.df = intm_sed_converted
    apply_outlier_handling(intm_sed, outlier_config_path='config/outlier_config.yaml')
    print(f'{len(intm_sed_converted)} rows in intm_sed_converted')
    intm_sed_converted = intm_sed.df
    return (intm_sed_converted,)


@app.cell
def _(intm_sed_converted, t1, t2):
    intm_sed_w = mo.sql(
        f"""
        -- converting to wide format
        WITH t1 AS (
        SELECT hospitalization_id
            , admin_dttm as event_dttm
            , med_category_unit: med_category || '_' || REPLACE(med_dose_unit, '/', '_') || '_intm'
            , med_dose: CASE WHEN mar_action_category = 'not_given' THEN 0 ELSE med_dose END
        FROM intm_sed_converted
        )
        , t2 AS (
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        )
        SELECT *
        FROM t2
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_w,)


@app.cell
def _(cohort_hrly_grids_f, cont_sed_w):
    cont_sed_wg = mo.sql(
        f"""
    -- create the hourly grid for the wide sedation table
    FROM cohort_hrly_grids_f g
    FULL JOIN cont_sed_w m USING (hospitalization_id, event_dttm)
    ORDER BY hospitalization_id, event_dttm
    """
    )
    cont_sed_wg['_dh'] = cont_sed_wg['event_dttm'].dt.floor('h', ambiguous='NaT')
    cont_sed_wg['_hr'] = cont_sed_wg['event_dttm'].dt.hour
    # wide table with hourly grids inserted
    print(len(cont_sed_wg))
    return


@app.cell
def _(cohort_hrly_grids_f, intm_sed_w):
    intm_sed_wg = mo.sql(
        f"""
    -- create the hourly grid for the wide sedation table
    FROM cohort_hrly_grids_f g
    FULL JOIN intm_sed_w m USING (hospitalization_id, event_dttm)
    ORDER BY hospitalization_id, event_dttm
    """
    )
    intm_sed_wg['_dh'] = intm_sed_wg['event_dttm'].dt.floor('h', ambiguous='NaT')
    intm_sed_wg['_hr'] = intm_sed_wg['event_dttm'].dt.hour
    print(len(intm_sed_wg))
    return (intm_sed_wg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hrly sum
    """)
    return


@app.cell
def _():
    cont_sed_dose_by_hr = mo.sql(read_sql('code/cont_sed_dose_by_hr.sql'))
    print(len(cont_sed_dose_by_hr))
    return


@app.cell
def _(intm_sed_wg):
    intm_sed_dose_by_hr = mo.sql(
        f"""
    FROM intm_sed_wg
    SELECT hospitalization_id, _dh
    , SUM(COALESCE(COLUMNS('_intm'), 0))
    GROUP BY hospitalization_id, _dh
    ORDER BY hospitalization_id, _dh
    """
    )
    print(len(intm_sed_dose_by_hr))
    return


@app.cell
def _():
    sed_dose_by_hr = mo.sql(
        f"""
        -- join the cont and intm hourly cumm dose table
        WITH t1 as (
        FROM cohort_hrly_grids_f g
        LEFT JOIN intm_sed_dose_by_hr i USING (hospitalization_id, _dh)
        LEFT JOIN cont_sed_dose_by_hr c USING (hospitalization_id, _dh)
        SELECT *
        )
        , t2 as (
        SELECT *
            , fentanyl_mcg_total: fentanyl_mcg_intm + fentanyl_mcg_min_cont
            , hydromorphone_mg_total: hydromorphone_mg_intm + hydromorphone_mg_min_cont
            , lorazepam_mg_total: lorazepam_mg_intm + lorazepam_mg_min_cont
            , midazolam_mg_total: midazolam_mg_intm + midazolam_mg_min_cont
            , propofol_mg_total: propofol_mg_intm + propofol_mg_min_cont
            , _midazolam_eq_mg_total: lorazepam_mg_total * 2 + midazolam_mg_total
            , _fentanyl_eq_mcg_total: hydromorphone_mg_total * 50 + fentanyl_mcg_total
        FROM t1
        )
        SELECT *
        FROM t2
        ORDER BY hospitalization_id, _dh
        #assert len(sed_dose_by_hr) == len(cont_sed_dose_by_hr), 'length altered for cont sed'
        #assert len(sed_dose_by_hr) == len(intm_sed_dose_by_hr), 'length altered for intm sed'
        """
    )
    return (sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By shift
    """)
    return


@app.cell
def _(sed_dose_by_hr):
    # sed_dose_by_shift.to_csv(f'output/final/{SITE_NAME}_sed_dose_by_shift_{CURRENT_TIME_STR}.csv', index=False)
    sed_dose_by_shift = mo.sql(
        f"""
        FROM sed_dose_by_hr
        SELECT _shift
        , propofol_mg: AVG(propofol_mg_total)
        , _fentanyl_eq_mcg: AVG(_fentanyl_eq_mcg_total)
        , _midazolam_eq_mg: AVG(_midazolam_eq_mg_total)
        GROUP BY _shift
        ORDER BY _shift
        """
    )
    return (sed_dose_by_shift,)


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, pd, sed_dose_by_hr, sed_dose_by_shift):
    import scipy.stats as stats
    test_cols = ['propofol_mg_total', '_fentanyl_eq_mcg_total', '_midazolam_eq_mg_total']
    shift_day = sed_dose_by_hr[sed_dose_by_hr['_shift'] == 'day']
    # Columns to test, keep as raw *_total column names for clarity and consistency
    shift_night = sed_dose_by_hr[sed_dose_by_hr['_shift'] == 'night']
    t_pvals = {}
    for col in test_cols:
        tstat, pval = stats.ttest_ind(shift_day[col], shift_night[col], nan_policy='omit', equal_var=False)
        t_pvals[col] = pval
    out_df = sed_dose_by_shift.rename(columns={'propofol_mg': 'propofol_mg_total', '_fentanyl_eq_mcg': '_fentanyl_eq_mcg_total', '_midazolam_eq_mg': '_midazolam_eq_mg_total', '_shift': 'shift'}).loc[:, ['shift', 'propofol_mg_total', '_fentanyl_eq_mcg_total', '_midazolam_eq_mg_total']]
    # Get the dose data for each shift
    pval_row = pd.Series({'shift': 'ttest_pval', 'propofol_mg_total': t_pvals['propofol_mg_total'], '_fentanyl_eq_mcg_total': t_pvals['_fentanyl_eq_mcg_total'], '_midazolam_eq_mg_total': t_pvals['_midazolam_eq_mg_total']})
    out_df = pd.concat([out_df, pd.DataFrame([pval_row])], ignore_index=True)
    # Perform t-tests and store p-values using *_total column names
    # Prepare summary table: output columns match sed_dose_by_shift, but we keep *_total column names for ttest row
    # Add a ttest_pval row, matching columns present in out_df
    # Save
    out_df.to_csv(f'output/final/{SITE_NAME}_sed_dose_by_shift_with_ttest_{CURRENT_TIME_STR}.csv', index=False)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By hour of day
    """)
    return


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, sed_dose_by_hr):
    sed_dose_by_hr_of_day = mo.sql(
        f"""
    FROM sed_dose_by_hr
    SELECT _hr
    , propofol_mg: AVG(propofol_mg_total)
    , _fentanyl_eq_mcg: AVG(_fentanyl_eq_mcg_total)
    , _midazolam_eq_mg: AVG(_midazolam_eq_mg_total)
    GROUP BY _hr
    ORDER BY _hr
    """
    )
    sed_dose_by_hr_of_day.to_csv(f'output/final/{SITE_NAME}_sed_dose_by_hr_of_day_{CURRENT_TIME_STR}.csv', index=False)
    return (sed_dose_by_hr_of_day,)


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, sed_dose_by_hr_of_day):
    import matplotlib.pyplot as plt
    import numpy as np

    # Extract variables for plotting
    hours = sed_dose_by_hr_of_day['_hr']
    propofol = sed_dose_by_hr_of_day['propofol_mg']
    fentanyl_eq = sed_dose_by_hr_of_day['_fentanyl_eq_mcg']
    midazolam_eq = sed_dose_by_hr_of_day['_midazolam_eq_mg']

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

    # Propofol
    axs[0].bar(x, propofol_ordered, color='skyblue', width=bar_width)
    axs[0].set_ylabel('Propofol (mg)')
    axs[0].set_title('Mean Total Propofol Dose by Hour of Day')
    axs[0].grid(True, axis='y')

    # Fentanyl Eq
    axs[1].bar(x, fentanyl_eq_ordered, color='salmon', width=bar_width)
    axs[1].set_ylabel('Fentanyl Eq (mcg)')
    axs[1].set_title('Mean Total Fentanyl Equivalent Dose by Hour of Day')
    axs[1].grid(True, axis='y')

    # Midazolam Eq
    axs[2].bar(x, midazolam_eq_ordered, color='mediumseagreen', width=bar_width)
    axs[2].set_ylabel('Midazolam Eq (mg)')
    axs[2].set_title('Mean Total Midazolam Equivalent Dose by Hour of Day')
    axs[2].set_xlabel('Hour of Day (_hr)')
    axs[2].grid(True, axis='y')

    # Add cutoff lines at _hr=7 and _hr=19 to each axis (find their positions in the reordered hours)
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

    # Add a title with the site_name variable
    plt.suptitle(f'Cumulative Sedative Doses by Hour of Day — {SITE_NAME}', fontsize=18, y=1.04)
    # Save the figure to file in output/final/{site_name}_sed_dose_by_hr_of_day_{current time down to sec}
    os.makedirs('output/final', exist_ok=True)
    save_path = f'output/final/{SITE_NAME}_sed_dose_by_hr_of_day_{CURRENT_TIME_STR}.png'
    plt.savefig(save_path, bbox_inches='tight')

    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## By day
    """)
    return


@app.cell
def _(duckdb):
    _q = """
    -- Aggregate per hospitalization, day, shift
    FROM sed_dose_by_hr
    SELECT
    hospitalization_id,
    _nth_day,
    _shift,
    SUM(propofol_mg_total) AS propofol_mg_total,
    SUM(_fentanyl_eq_mcg_total) AS fentanyl_eq_mcg_total,
    SUM(_midazolam_eq_mg_total) AS midazolam_eq_mg_total
    GROUP BY hospitalization_id, _nth_day, _shift
    ORDER BY hospitalization_id, _nth_day, _shift
    """
    sed_dose_agg = duckdb.sql(_q).df()
    sed_dose_daily = sed_dose_agg.pivot(index=['hospitalization_id', '_nth_day'], columns='_shift', values=['propofol_mg_total', 'fentanyl_eq_mcg_total', 'midazolam_eq_mg_total']).reset_index()
    sed_dose_daily.columns = ['hospitalization_id', '_nth_day', 'propofol_day' if ('propofol_mg_total', 'day') in sed_dose_daily.columns else None, 'propofol_night' if ('propofol_mg_total', 'night') in sed_dose_daily.columns else None, 'fentanyl_eq_day' if ('fentanyl_eq_mcg_total', 'day') in sed_dose_daily.columns else None, 'fentanyl_eq_night' if ('fentanyl_eq_mcg_total', 'night') in sed_dose_daily.columns else None, 'midazolam_eq_day' if ('midazolam_eq_mg_total', 'day') in sed_dose_daily.columns else None, 'midazolam_eq_night' if ('midazolam_eq_mg_total', 'night') in sed_dose_daily.columns else None]
    # Optionally flatten MultiIndex columns and rename as requested
    # Remove 'None' columns in case day or night is missing
    # For full reproducibility, here is a more rigid column assignment to avoid None if both shifts exist:
    # sed_dose_wide.columns = [
    #     'hospitalization_id', '_nth_day',
    #     'propofol_day', 'propofol_night',
    #     'fentanyl_eq_day', 'feantanyl_eq_night',
    #     'midazolam_eq_day', 'midazolam_eq_night'
    # ]
    sed_dose_daily = sed_dose_daily.loc[:, [c for c in sed_dose_daily.columns if c is not None]]
    return sed_dose_agg, sed_dose_daily


@app.cell(disabled=True)
def _():
    # q = """
    # PIVOT_WIDER sed_dose_by_day_shift
    # ON _shift
    # USING SUM(propofol_mg_total)
    # ORDER BY hospitalization_id, _nth_day
    # """
    # sed_dose_by_day_shift_w = duckdb.sql(q).df()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Merge into analytical dataset
    """)
    return


@app.cell
def _(cohort_sbt_outcomes_daily, covs_daily, hosp_df, sed_dose_daily):
    cohort_merged = mo.sql(
        f"""
    FROM cohort_sbt_outcomes_daily o
    LEFT JOIN sed_dose_daily s USING (hospitalization_id, _nth_day)
    LEFT JOIN covs_daily c USING (hospitalization_id, _nth_day)
    LEFT JOIN hosp_df h USING (hospitalization_id)
    SELECT o.hospitalization_id
    , o._nth_day
    , o.n_hrs
    , _sbt_done_today: o.sbt_done
    , _success_extub_today: o.success_extub
    , sbt_done_next_day: LEAD(o.sbt_done) OVER w
    , success_extub_next_day: LEAD(o.success_extub) OVER w
    --, o.trach_1st
    , _propofol_day: COALESCE(s.propofol_day, 0)
    , _propofol_night: COALESCE(s.propofol_night, 0)
    , _fentanyl_eq_day: COALESCE(s.fentanyl_eq_day, 0)
    , _fentanyl_eq_night: COALESCE(s.fentanyl_eq_night, 0)
    , _midazolam_eq_day: COALESCE(s.midazolam_eq_day, 0)
    , _midazolam_eq_night: COALESCE(s.midazolam_eq_night, 0)
    , propofol_diff: COALESCE(s.propofol_night, 0) - COALESCE(s.propofol_day, 0)
    , fentanyl_eq_diff: COALESCE(s.fentanyl_eq_night, 0) - COALESCE(s.fentanyl_eq_day, 0)
    , midazolam_eq_diff: COALESCE(s.midazolam_eq_night, 0) - COALESCE(s.midazolam_eq_day, 0)
    , COLUMNS('(7am)|(7pm)')
    , age: h.age_at_admission
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _nth_day)
    ORDER BY o.hospitalization_id, o._nth_day
    """
    )
    cohort_merged.dropna(subset=['age'], inplace=True)
    return (cohort_merged,)


@app.cell
def _(cohort_merged):
    cohort_merged_final = mo.sql(
        f"""
        FROM cohort_merged
        SELECT *
        WHERE _nth_day > 0 AND sbt_done_next_day IS NOT NULL AND success_extub_next_day IS NOT NULL
        """
    )
    return (cohort_merged_final,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Table one
    """)
    return


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME):
    import tableone

    def gen_and_save_tableone(file_name, **kwargs):
        """
        Wrapper for tableone.TableOne that automatically saves results to a CSV file.

        Args:
            data: DataFrame to create table one from
            file_name: Name for the output file (without extension)
            **kwargs: All other arguments passed to tableone.TableOne

        Returns:
            tableone.TableOne object
        """
        table = tableone.TableOne(**kwargs)
        table.to_csv(f'output/final/{SITE_NAME}_{file_name}_{CURRENT_TIME_STR}.csv')
        return table

    return (gen_and_save_tableone,)


@app.cell
def _(cohort_merged_final):
    cohort_merged_for_t1 = mo.sql(
        f"""
        FROM cohort_merged_final
        SELECT * -- EXCLUDE(hospitalization_id)
        WHERE _nth_day = 1
        """
    )
    return (cohort_merged_for_t1,)


@app.cell
def _(cohort_merged_for_t1, covs, hosp_df, sed_dose_agg):
    cohort_merged_for_t1_w_by_shift = mo.sql(
        f"""
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
    LEFT JOIN covs c USING (hospitalization_id, _nth_day, _shift)
    SELECT *
    )
    SELECT *
    FROM t2
    ORDER BY hospitalization_id, _nth_day, _shift
    """
    )
    assert len(cohort_merged_for_t1) * 2 == len(cohort_merged_for_t1_w_by_shift)
    return (cohort_merged_for_t1_w_by_shift,)


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, cohort_merged_for_t1_w_by_shift, pd):
    n_unique_patients = cohort_merged_for_t1_w_by_shift['patient_id'].nunique()
    pd.DataFrame({'n_unique_patients': [n_unique_patients]})\
        .to_csv(f'output/final/{SITE_NAME}_cohort_stats_{CURRENT_TIME_STR}.csv', index=False)
    return


@app.cell
def _(cohort_merged_for_t1, gen_and_save_tableone):
    outcome_vars = ['_sbt_done_today', '_success_extub_today']
    diff_doses = ['propofol_diff', 'fentanyl_eq_diff', 'midazolam_eq_diff']
    _cont_vars = ['age'] + diff_doses
    _cat_vars = outcome_vars
    gen_and_save_tableone(file_name='table_one_day_1', data=cohort_merged_for_t1, continuous=_cont_vars, categorical=_cat_vars)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## day-night comparison table
    """)
    return


@app.cell
def _(cohort_merged_for_t1_w_by_shift, gen_and_save_tableone):
    sed_vars = ['propofol_mg_total', 'fentanyl_eq_mcg_total', 'midazolam_eq_mg_total']
    _cont_vars = sed_vars + ['ph', 'pf', '_nee']
    _cat_vars = ['ph_level', 'pf_level']
    nonnorm_vars = ['ph', 'pf']
    gen_and_save_tableone(file_name='table_one_day_1_by_shift', pval=True, data=cohort_merged_for_t1_w_by_shift, continuous=_cont_vars, groupby='_shift', categorical=_cat_vars)  # nonnormal=nonnorm_vars
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Regression
    """)
    return


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, cohort_merged_final):
    import matplotlib.pyplot as _plt
    import seaborn as _sns
    continuous_vars = ['age', 'propofol_diff', 'fentanyl_eq_diff', 'midazolam_eq_diff', '_propofol_day', '_propofol_night', '_fentanyl_eq_day', '_fentanyl_eq_night', '_midazolam_eq_day', '_midazolam_eq_night', 'nee_7am', 'nee_7pm', '_ph_7am', '_ph_7pm', '_pf_7am', '_pf_7pm']
    continuous_vars_df = cohort_merged_final[[col for col in continuous_vars if col in cohort_merged_final.columns]]
    # Define your list of continuous variables as in your earlier code
    corr_matrix = continuous_vars_df.corr(method='pearson')
    _plt.figure(figsize=(14, 10))
    _sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='vlag', linewidths=0.5, cbar_kws={'label': 'Pearson Correlation'})
    _plt.title('Pairwise Pearson Correlation Matrix (Continuous Variables)')
    _plt.tight_layout()
    # Only select columns that exist in the dataframe (to avoid KeyError)
    # Compute pairwise Pearson correlations between continuous variables
    # Save the correlation matrix to CSV in output/final
    corr_matrix.to_csv(f'output/final/{SITE_NAME}_pairwise_corr_matrix_{CURRENT_TIME_STR}.csv')
    _plt.gcf()
    return


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, cohort_merged_final, pd):
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
    # GEE regression
    _summary_pd.to_csv(f'output/final/{SITE_NAME}_gee_summary_{CURRENT_TIME_STR}.csv', index=False)
    _cov_matrix = gee_result.cov_params()
    # Save the summary as CSV for readability
    # Convert the summary table to DataFrame
    # Save covariance matrix of the model
    _cov_matrix.to_csv(f'output/final/{SITE_NAME}_gee_covmat_{CURRENT_TIME_STR}.csv')
    return


@app.cell
def _(CURRENT_TIME_STR, SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    # Logit regression with clustered standard errors
    success_extub_formula = """success_extub_next_day ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm + age
    """
    logit_model = _smf.logit(formula=success_extub_formula, data=cohort_merged_final)
    logit_result = logit_model.fit(cov_type='cluster', cov_kwds={'groups': cohort_merged_final['hospitalization_id']})
    print(logit_result.summary())
    _summary_df = logit_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output/final/{SITE_NAME}_logit_summary_{CURRENT_TIME_STR}.csv', index=False)
    _cov_matrix = logit_result.cov_params()
    # Save the summary as CSV for readability
    # Convert the summary table to DataFrame
    # Save covariance matrix of the model
    _cov_matrix.to_csv(f'output/final/{SITE_NAME}_logit_covmat_{CURRENT_TIME_STR}.csv')
    return


if __name__ == "__main__":
    app.run()
