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
    # 03 SBT Detection & Extubation Outcomes

    Detects SBT (Spontaneous Breathing Trial) states from respiratory support data,
    identifies extubation events, and computes extubation outcomes
    (success, failure, withdrawal of care, death after extubation).
    """)
    return


@app.cell
def _():
    from clifpy import ClifOrchestrator
    from clifpy.utils.config import get_config_or_params
    import pandas as pd
    import duckdb

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"
    co = ClifOrchestrator(config_path=CONFIG_PATH)

    os.makedirs("output", exist_ok=True)
    return CONFIG_PATH, co, get_config_or_params, pd


@app.cell
def _(CONFIG_PATH, get_config_or_params):
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    print(f"Site: {SITE_NAME}")
    return (SITE_NAME,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Respiratory Support (processed)
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    resp_processed_path = f"output/intermediate/{SITE_NAME}_resp_processed_bf.parquet"
    assert os.path.exists(resp_processed_path), (
        f"Missing {resp_processed_path} — run 01_cohort.py first"
    )
    resp_p = pd.read_parquet(resp_processed_path)
    resp_p['tracheostomy'] = resp_p['tracheostomy'].fillna(0).astype(int)
    print(f"resp_p: {len(resp_p)} rows from {resp_processed_path}")
    return (resp_p,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Cohort Hourly Grids
    """)
    return


@app.cell
def _(pd):
    cohort_hrly_grids_f = pd.read_parquet("output/cohort_hrly_grids.parquet")
    print(f"cohort_hrly_grids_f: {len(cohort_hrly_grids_f)} rows")
    return (cohort_hrly_grids_f,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load CLIF Tables (Hospitalization, CodeStatus, Vitals)
    """)
    return


@app.cell
def _(CONFIG_PATH):
    from clifpy import Patient, Hospitalization, CodeStatus

    patient = Patient.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'race_category', 'ethnicity_category', 'sex_category'],
    )

    hosp = Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=[
            'hospitalization_id', 'patient_id',
            'discharge_category', 'discharge_dttm',
        ],
    )
    return hosp, patient


@app.cell
def _(hosp):
    hosp_df = hosp.df[['hospitalization_id', 'discharge_category', 'discharge_dttm']].copy()
    print(f"hosp_df: {len(hosp_df)} rows")
    return (hosp_df,)


@app.cell
def _(CONFIG_PATH, hosp):
    cs = CodeStatus.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'code_status_category', 'start_dttm'],
    )

    # Map patient_id -> hospitalization_id via hospitalization table
    _pid_to_hid = hosp.df[['hospitalization_id', 'patient_id']].drop_duplicates()
    cs_df = cs.df.merge(_pid_to_hid, on='patient_id', how='inner')
    cs_df = cs_df[['hospitalization_id', 'code_status_category', 'start_dttm']].copy()
    cs_df = cs_df.sort_values(['hospitalization_id', 'start_dttm']).reset_index(drop=True)
    print(f"cs_df: {len(cs_df)} rows (mapped to hospitalization_id)")
    return (cs_df,)


@app.cell
def _(co, pd):
    # Compute last vitals datetime per hospitalization
    _vitals_path = co.tables_metadata['vitals']['path']
    _vitals = pd.read_parquet(
        _vitals_path,
        columns=['hospitalization_id', 'recorded_dttm'],
    )
    last_vitals_df = (
        _vitals
        .groupby('hospitalization_id', as_index=False)['recorded_dttm']
        .max()
    )
    print(f"last_vitals_df: {len(last_vitals_df)} hospitalizations")
    return (last_vitals_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SBT Detection (gaps-and-islands)

    Each CTE from the original `sbt.sql` is inlined as a separate cell.
    """)
    return


@app.cell
def _(resp_p):
    sbt_t1 = mo.sql(
        f"""
        -- Detect SBT state, intubation, extubation, and trach flip events
        FROM resp_p
        SELECT
            device_category, device_name, mode_category, mode_name
            , fio2_set, peep_set, pressure_support_set, tracheostomy
            , hospitalization_id, recorded_dttm
            , _sbt_state: CASE
                WHEN (mode_category IN ('pressure support/cpap')
                      AND peep_set <= 8 AND pressure_support_set <= 8)
                    OR regexp_matches(device_name, 't1[\\s_-]?piece')
                    THEN 1 ELSE 0 END
            , _intub: CASE
                WHEN LAG(device_category) OVER w IS DISTINCT FROM 'imv'
                    AND device_category = 'imv' THEN 1 ELSE 0 END
            , _extub: CASE
                WHEN LAG(device_category) OVER w = 'imv'
                    AND device_category IS DISTINCT FROM 'imv'
                THEN 1 ELSE 0 END
            , _trach_flip_to_1: CASE
                WHEN LAG(tracheostomy) OVER w = 0
                    AND tracheostomy = 1 THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t1,)


@app.cell
def _(sbt_t1):
    sbt_t2 = mo.sql(
        f"""
        -- Gaps-and-islands: cumulative extubation and trach flags
        FROM sbt_t1
        SELECT *
            , _chg_sbt_state: CASE
                WHEN _sbt_state IS DISTINCT FROM LAG(_sbt_state) OVER w
                THEN 1 ELSE 0 END
            , _extub_cum: SUM(_extub) OVER w
            , _extub_1st: CASE
                WHEN _extub = 1 AND _extub_cum = 1 THEN 1 ELSE 0 END
            , _trach_flip_cum: SUM(_trach_flip_to_1) OVER w
            , _trach_1st: CASE
                WHEN _trach_flip_to_1 = 1 AND _trach_flip_cum = 1 THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t2,)


@app.cell
def _(last_vitals_df, sbt_t1, sbt_t2):
    sbt_t3 = mo.sql(
        f"""
        -- Assign block IDs, detect failed extubation, check last vitals within 24h
        FROM sbt_t2
        SELECT *
            , _block_id: SUM(_chg_sbt_state) OVER w
            , _fail_extub: CASE
                WHEN sbt_t2._extub_1st = 1 AND EXISTS (
                    SELECT 1
                    FROM sbt_t1
                    WHERE sbt_t1.hospitalization_id = sbt_t2.hospitalization_id
                      AND sbt_t1._intub = 1
                      AND sbt_t1.recorded_dttm > sbt_t2.recorded_dttm
                      AND sbt_t1.recorded_dttm <= sbt_t2.recorded_dttm + INTERVAL 24 HOUR
                ) THEN 1 ELSE 0 END
            , _last_vitals_within_24h_of_extub: CASE
                WHEN sbt_t2._extub_1st = 1 AND EXISTS (
                    SELECT 1
                    FROM last_vitals_df lv
                    WHERE lv.hospitalization_id = sbt_t2.hospitalization_id
                      AND lv.recorded_dttm >= sbt_t2.recorded_dttm
                      AND lv.recorded_dttm <= sbt_t2.recorded_dttm + INTERVAL 24 HOUR
                ) THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t3,)


@app.cell
def _(sbt_t3):
    sbt_all_blocks = mo.sql(
        f"""
        -- Aggregate per SBT block: start/end modes and timestamps
        FROM sbt_t3
        SELECT hospitalization_id, _block_id, _sbt_state
            , _start_mode: ANY_VALUE(mode_category ORDER BY recorded_dttm)
            , _end_mode: ANY_VALUE(mode_category ORDER BY recorded_dttm)
            , _start_dttm: MIN(recorded_dttm)
            , _last_dttm: MAX(recorded_dttm)
        GROUP BY hospitalization_id, _block_id, _sbt_state
        """
    )
    return (sbt_all_blocks,)


@app.cell
def _(sbt_all_blocks):
    sbt_all_blocks_w_duration = mo.sql(
        f"""
        -- Compute block duration using next block start as end boundary
        FROM sbt_all_blocks
        SELECT *
            , _next_start_dttm: LEAD(_start_dttm) OVER w
            , _end_dttm: COALESCE(_next_start_dttm, _last_dttm)
            , _duration_mins: date_diff('minute', _start_dttm, _end_dttm)
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _block_id)
        """
    )
    return (sbt_all_blocks_w_duration,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SBT Outcomes

    Final join: SBT blocks + code status (ASOF) + hospitalization discharge info.
    Computes sbt_done, success_extub, fail_extub, withdrawal, death_after_extub.
    """)
    return


@app.cell
def _(cs_df, hosp_df, sbt_all_blocks_w_duration, sbt_t3):
    sbt_outcomes = mo.sql(
        f"""
        -- Final SBT outcomes with code status and discharge info
        FROM sbt_t3
        LEFT JOIN sbt_all_blocks_w_duration AS b
            ON sbt_t3.hospitalization_id = b.hospitalization_id
            AND sbt_t3._block_id = b._block_id
        ASOF LEFT JOIN cs_df AS c
            ON c.hospitalization_id = sbt_t3.hospitalization_id
            AND c.start_dttm <= sbt_t3.recorded_dttm
        ASOF LEFT JOIN hosp_df AS h
            ON sbt_t3.hospitalization_id = h.hospitalization_id
            AND sbt_t3.recorded_dttm <= h.discharge_dttm
        SELECT sbt_t3.fio2_set, sbt_t3.peep_set, sbt_t3.pressure_support_set, sbt_t3.tracheostomy
            , _block_duration_mins: COALESCE(b._duration_mins, 0)
            , sbt_t3.device_category, sbt_t3.device_name, sbt_t3.mode_category, sbt_t3.mode_name
            , sbt_t3.hospitalization_id, event_dttm: sbt_t3.recorded_dttm
            , sbt_done: CASE WHEN _block_duration_mins >= 30 AND sbt_t3._sbt_state = 1 THEN 1 ELSE 0 END
            , _extub_1st, _intub, sbt_t3._trach_1st, _fail_extub
            , c.code_status_category, cs_start_dttm: c.start_dttm
            , h.discharge_category, discharge_dttm: h.discharge_dttm
            , _last_vitals_within_24h_of_extub
            , _withdrawl_lst: CASE
                WHEN _extub_1st = 1
                    AND TRIM(LOWER(code_status_category)) != 'full'
                    AND TRIM(LOWER(discharge_category)) IN ('hospice', 'expired')
                THEN 1 ELSE 0 END
            , _success_extub: CASE
                WHEN _extub_1st = 1 AND _withdrawl_lst = 0 AND _fail_extub = 0
                THEN 1 ELSE 0 END
            , _death_after_extub_wo_reintub: CASE
                WHEN _extub_1st = 1
                    AND _last_vitals_within_24h_of_extub = 1
                    AND _fail_extub = 0
                    AND TRIM(LOWER(discharge_category)) IN ('hospice', 'expired')
                THEN 1 ELSE 0 END
        WHERE (sbt_t3.tracheostomy = 0 OR sbt_t3._trach_1st = 1)
        ORDER BY sbt_t3.hospitalization_id, event_dttm
        """
    )
    return (sbt_outcomes,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly & Daily Aggregation

    Floor event_dttm to hour, aggregate SBT outcomes per hour,
    join back to cohort hourly grid, then roll up to daily.
    """)
    return


@app.cell
def _(sbt_outcomes):
    sbt_outcomes_hrly = mo.sql(
        f"""
        -- Aggregate SBT outcomes per hospitalization-hour
        FROM sbt_outcomes
        SELECT hospitalization_id
            , _dh: date_trunc('hour', event_dttm)
            , sbt_done: MAX(sbt_done)
            , _success_extub: MAX(_success_extub)
            , _trach_1st: MAX(_trach_1st)
            , _fail_extub: MAX(_fail_extub)
            , _extub_1st: MAX(_extub_1st)
            , _withdrawl_lst: MAX(_withdrawl_lst)
            , _death_after_extub_wo_reintub: MAX(_death_after_extub_wo_reintub)
        GROUP BY hospitalization_id, _dh
        """
    )
    return (sbt_outcomes_hrly,)


@app.cell
def _(cohort_hrly_grids_f, sbt_outcomes_hrly):
    cohort_sbt_outcomes_hrly = mo.sql(
        f"""
        -- Left join SBT hourly outcomes onto cohort hourly grid
        FROM cohort_hrly_grids_f g
        LEFT JOIN sbt_outcomes_hrly s
            ON g.hospitalization_id = s.hospitalization_id
            AND g._dh = s._dh
        SELECT g.hospitalization_id, g.event_dttm, g._dh, g._nth_day, g._shift, g._day_shift
            , sbt_done: COALESCE(s.sbt_done, 0)
            , _success_extub: COALESCE(s._success_extub, 0)
            , _trach_1st: COALESCE(s._trach_1st, 0)
            , _fail_extub: COALESCE(s._fail_extub, 0)
            , _extub_1st: COALESCE(s._extub_1st, 0)
            , _withdrawl_lst: COALESCE(s._withdrawl_lst, 0)
            , _death_after_extub_wo_reintub: COALESCE(s._death_after_extub_wo_reintub, 0)
        ORDER BY g.hospitalization_id, g.event_dttm
        """
    )
    return (cohort_sbt_outcomes_hrly,)


@app.cell
def _(cohort_sbt_outcomes_hrly):
    cohort_sbt_outcomes_daily = mo.sql(
        f"""
        -- Aggregate SBT outcomes to daily level
        FROM cohort_sbt_outcomes_hrly
        SELECT hospitalization_id, _nth_day
            , sbt_done: MAX(sbt_done)
            , _success_extub: MAX(_success_extub)
            , _trach_1st: MAX(_trach_1st)
            , _fail_extub: MAX(_fail_extub)
            , _extub_1st: MAX(_extub_1st)
            , _withdrawl_lst: MAX(_withdrawl_lst)
            , _death_after_extub_wo_reintub: MAX(_death_after_extub_wo_reintub)
        GROUP BY hospitalization_id, _nth_day
        ORDER BY hospitalization_id, _nth_day
        """
    )
    return (cohort_sbt_outcomes_daily,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs
    """)
    return


@app.cell
def _(cohort_sbt_outcomes_daily):
    _out = cohort_sbt_outcomes_daily.df()
    _out.to_parquet("output/sbt_outcomes_daily.parquet")
    print(f"Saved: output/sbt_outcomes_daily.parquet ({len(_out)} rows, {_out['hospitalization_id'].nunique()} hospitalizations)")
    return


if __name__ == "__main__":
    app.run()
