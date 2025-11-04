import marimo

__generated_with = "0.15.1"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Epidemiology of Sedation in Mechanical Ventilation""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Init""")
    return


@app.cell
def _():
    import os
    os.chdir("..")
    os.getcwd()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Import""")
    return


@app.cell
def _():
    from importlib import reload
    import pandas as pd
    import duckdb
    from utils import pyCLIF as pc
    reload(pc)
    from utils.waterfall import process_resp_support_waterfall
    import ipytest
    import tableone
    from utils.data_cleaner import remove_outliers_with_timing
    import datetime
    import warnings
    return (
        datetime,
        duckdb,
        pc,
        pd,
        process_resp_support_waterfall,
        reload,
        remove_outliers_with_timing,
        tableone,
        warnings,
    )


@app.cell
def _(datetime, pc):
    helper = pc.load_config()
    site = helper['site_name'].lower()
    print(f"your site name is: {site}")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return helper, site, timestamp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Cohort Identification""")
    return


@app.cell
def _(pc):
    adt = pc.load_data("clif_adt")
    hospitalization = pc.load_data("clif_hospitalization")
    return adt, hospitalization


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Create ICU-stay level unique id""")
    return


@app.cell
def _(adt, hospitalization, pc, warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        stitched_encounters = pc.stitch_encounters(hospitalization, adt)
    return


@app.cell
def _(duckdb):
    _query = '\nSELECT DISTINCT patient_id, hospitalization_id, encounter_block\nFROM stitched_encounters\n'
    hosp_to_enc_blk_mapper = duckdb.sql(_query).to_df()
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT hospitalization_id\n    , encounter_block\n    , date_trunc('hour', in_dttm) as in_date_hr\n    , 1 as new_icu_stay\nFROM stitched_encounters\nWHERE location_category = 'icu'\n"
    new_icu_start_hours = duckdb.sql(_query).to_df()
    hosp_ids_w_icu_stays = new_icu_start_hours['hospitalization_id'].unique().tolist()
    return (hosp_ids_w_icu_stays,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Hour 24 and 72""")
    return


@app.cell
def _(hosp_ids_w_icu_stays, pc):
    resp = pc.load_data(
        table = "clif_respiratory_support",
        filters = {
            "hospitalization_id": hosp_ids_w_icu_stays
        }
        )
    return (resp,)


@app.cell
def _(process_resp_support_waterfall, resp, site, warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        resp_f = process_resp_support_waterfall(resp)
        resp_f.to_parquet(f"output/intermediate/{site}_resp_processed.parquet")

    # resp_f = pd.read_parquet(f"output/intermediate/{site}_resp_processed.parquet")
    return (resp_f,)


@app.cell
def _():
    # for MIMIC dev
    focal_hosp_ids = [
        '21738444', 
        '20004088', 
        '20006154', 
        '20018306'
        ]

    # query = f"""
    # SELECT SUM(CASE WHEN location_category = 'icu' THEN 1 ELSE 0 END) as total_icu_stays
    # FROM adt
    # WHERE hospitalization_id IN ({",".join(focal_hosp_ids)})
    # """
    # duckdb.sql(query).to_df()
    return (focal_hosp_ids,)


@app.cell
def _(duckdb, focal_hosp_ids, resp_f):
    resp_f['date_hr'] = resp_f['recorded_dttm'].dt.floor('h')
    _query = f"\nSELECT t1.hospitalization_id\n    , t3.encounter_block\n    , t1.date_hr\n    , MAX(CASE WHEN t2.new_icu_stay = 1 THEN 1 ELSE 0 END) as new_icu_start_from_adt\n    , MAX(CASE WHEN t1.device_category = 'imv' THEN 1 ELSE 0 END) as on_imv\n    , MAX(CASE WHEN t1.tracheostomy is True OR t1.tracheostomy = 1 THEN 1 ELSE 0 END) as trach_ever\n    , ROW_NUMBER() OVER (PARTITION BY t1.hospitalization_id ORDER BY t1.date_hr) as rn_by_hosp\n    , CASE WHEN (\n        rn_by_hosp = 1 -- new hospitalization\n        OR new_icu_start_from_adt = 1 -- new icu stay\n    ) THEN 1 ELSE 0 END as new_icu_stay\nFROM resp_f as t1\nLEFT JOIN new_icu_start_hours AS t2\n    ON t1.hospitalization_id = t2.hospitalization_id\n    AND t1.date_hr = t2.in_date_hr\nLEFT JOIN hosp_to_enc_blk_mapper AS t3\n    ON t1.hospitalization_id = t3.hospitalization_id\n-- WHERE t1.hospitalization_id IN ({','.join(focal_hosp_ids)})\nGROUP BY t1.hospitalization_id, t1.date_hr, t3.encounter_block\nORDER BY t1.hospitalization_id, t1.date_hr\n"
    df1 = duckdb.sql(_query).to_df()
    return


@app.cell
def _(duckdb):
    _query = '\n-- generate unique icu stay ids\nWITH t1 AS (\n    SELECT hospitalization_id\n        , encounter_block\n        , date_hr\n        , on_imv\n        , new_icu_stay\n        , SUM(new_icu_stay) OVER (ORDER BY hospitalization_id, date_hr) as icu_stay_id\n    FROM df1\n    -- keep only hospitalizations that have at least one hour on imv and no tracheostomy\n    WHERE hospitalization_id IN (\n        SELECT DISTINCT hospitalization_id\n        FROM df1\n        GROUP BY hospitalization_id\n        HAVING MAX(on_imv) = 1 AND MAX(trach_ever) = 0\n    )\n),\n-- generate unique imv streak ids\nt2 AS (\n    SELECT hospitalization_id\n        , icu_stay_id\n        -- , encounter_block\n        , date_hr\n        , on_imv\n        , ROW_NUMBER() OVER (PARTITION BY icu_stay_id ORDER BY date_hr) as rn_overall\n        , ROW_NUMBER() OVER (PARTITION BY icu_stay_id, on_imv ORDER BY date_hr) as rn_by_imv_status\n        , rn_overall - rn_by_imv_status as imv_streak_id\n    FROM t1\n    -- keep only icu stays that have at least one hour on imv\n    WHERE icu_stay_id IN (\n        SELECT DISTINCT icu_stay_id\n        FROM t1\n        GROUP BY icu_stay_id\n        HAVING MAX(on_imv) = 1\n    )\n    ORDER BY hospitalization_id, icu_stay_id, date_hr\n),\n-- mark the 24th and 72th hour of each imv streak\nt3 AS (\n    SELECT hospitalization_id, icu_stay_id, date_hr\n        , imv_streak_id, on_imv\n        , SUM(on_imv) OVER (PARTITION BY icu_stay_id, imv_streak_id ORDER BY date_hr) as imv_hrs_in_streak\n        , CASE WHEN (imv_hrs_in_streak = 24) THEN 1 ELSE 0 END as hr_24_on_imv\n        , CASE WHEN (imv_hrs_in_streak = 72) THEN 1 ELSE 0 END as hr_72_on_imv\n        -- calculate hour since first intubation within each icu stay\n        , MIN(CASE WHEN on_imv = 1 THEN date_hr END) OVER (PARTITION BY icu_stay_id) as first_imv_hr_in_icu_stay\n        -- can only calculate diff in secs, so convert to hrs\n        ,  EXTRACT(EPOCH FROM (date_hr - first_imv_hr_in_icu_stay)) / 3600 + 1 as hrs_since_first_imv\n    FROM t2\n    ORDER BY hospitalization_id, icu_stay_id, date_hr\n    )\n-- exclude cases with reintubation within 72 hours\nSELECT hospitalization_id, icu_stay_id, date_hr\n    , imv_streak_id, on_imv, imv_hrs_in_streak, hrs_since_first_imv\n    , hr_24_on_imv, hr_72_on_imv\n    , COUNT(DISTINCT CASE WHEN hrs_since_first_imv BETWEEN 0 AND 72 THEN imv_streak_id END) \n        OVER (PARTITION BY icu_stay_id) as n_imv_streaks_in_72_hrs\n    , CASE WHEN n_imv_streaks_in_72_hrs <= 2 AND hr_24_on_imv = 1 THEN 1 ELSE 0 END as hr_24_on_imv_noreintub\n    , CASE WHEN n_imv_streaks_in_72_hrs = 1 AND hr_72_on_imv = 1 THEN 1 ELSE 0 END as hr_72_on_imv_noreintub\nFROM t3\nORDER BY hospitalization_id, icu_stay_id, date_hr\n'
    df2 = duckdb.sql(_query).to_df()
    return


@app.cell
def _():
    # %%ipytest

    # # sanity tests against the MIMIC-IV data
    # @pytest.mark.parametrize("hospitalization_id,date_hr,expected_hr,expected_flag,expected_flag_noreintub", [
    #     # on imv for 24-hrs twice during the same hospitalization -- so would be excluded if no reintubation within 72 hrs
    #     (21738444, "2186-09-14 17:00:00-06:00", 24, 1, 0),  
    #     (21738444, "2186-09-14 18:00:00-06:00", 24, 0, 0),  
    #     (21738444, "2186-09-16 18:00:00-06:00", 24, 1, 0), # second streak within the hosp
    #     # not on imv for the first few hrs but long streak afterwards
    #     (20004088, "2159-09-30 09:00:00-06:00", 24, 1, 1),
    #     (20004088, "2159-10-02 09:00:00-06:00", 72, 1, 1),
    #     # very short streaks: 20006154
    #     # 3 icu stays within the same hospitalization
    #     (20018306, "2136-05-16 05:00:00-06:00", 24, 1, 1),
    #     (20018306, "2136-07-01 19:00:00-06:00", 24, 1, 1),
    #     # (20018306, "2136-06-01 03:00:00-06:00", 24, 0), # in a icu stay that was filtered out in the df because of no imv ever
    # ])
    # def test_if_on_imv_at_hr_x(hospitalization_id, date_hr, expected_hr, expected_flag, expected_flag_noreintub):
    #     query = f"""
    #     SELECT hr_{expected_hr}_on_imv, hr_{expected_hr}_on_imv_noreintub
    #     FROM df2
    #     WHERE hospitalization_id = {hospitalization_id}
    #     AND date_hr = '{date_hr}'
    #     """
    #     result = duckdb.sql(query).to_df()
    #     observed_flag = result[f'hr_{expected_hr}_on_imv'].iloc[0]
    #     observed_flag_noreintub = result[f'hr_{expected_hr}_on_imv_noreintub'].iloc[0]
    #     assert observed_flag == expected_flag
    #     assert observed_flag_noreintub == expected_flag_noreintub
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### The Cohort""")
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT ROW_NUMBER() OVER () as row_id\n    , hospitalization_id\n    --, encounter_block\n    --, icu_stay_id\n    , date_hr\n    , CASE WHEN hr_24_on_imv_noreintub = 1 THEN 'hr_24'\n        WHEN hr_72_on_imv_noreintub = 1 THEN 'hr_72'\n        ELSE NULL END as cohort_flag\nFROM df2\nLEFT JOIN hosp_to_enc_blk_mapper USING (hospitalization_id)\nWHERE hr_24_on_imv_noreintub = 1 OR hr_72_on_imv_noreintub = 1\n"
    cohort = duckdb.sql(_query).to_df()
    cohort_hosp_ids = cohort['hospitalization_id'].unique().tolist()
    return (cohort_hosp_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Demographics""")
    return


@app.cell
def _(cohort_hosp_ids, pc):
    hosp_required_columns = [
        "patient_id", 
        "hospitalization_id", 
        "age_at_admission"
    ]

    cohort_hosp = pc.load_data(
        table = "clif_hospitalization",
        columns = hosp_required_columns,
        filters = {
            "hospitalization_id": cohort_hosp_ids
        }
    )

    cohort_pt_ids = cohort_hosp['patient_id'].unique().tolist()

    pt_required_columns = [
        "patient_id",
        "race_category",
        "ethnicity_category",
        "sex_category"
    ]

    cohort_pt = pc.load_data(
        table = "clif_patient",
        columns = pt_required_columns,
        filters = {
            "patient_id": cohort_pt_ids
        }
    )
    return


@app.cell
def _(duckdb):
    _query = '\nSELECT *\nFROM cohort_hosp\nLEFT JOIN cohort_pt\nUSING (patient_id)\n'
    cohort_demogs = duckdb.sql(_query).to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Vitals""")
    return


@app.cell
def _(cohort_hosp_ids, pc):
    vitals_required_columns = [
        "hospitalization_id",
        "recorded_dttm",
        "vital_category",
        "vital_value"
    ]

    vitals = pc.load_data(
        table = "clif_vitals",
        filters = {
            "hospitalization_id": cohort_hosp_ids
        },
        columns = vitals_required_columns
    )

    vitals['date_hr'] = vitals['recorded_dttm'].dt.floor('h')
    return (vitals,)


@app.cell
def _(remove_outliers_with_timing, vitals):
    vitals_1 = remove_outliers_with_timing(vitals, 'vitals', 'vital_value', file_path='config/outliers.json')
    return


@app.cell
def _(duckdb):
    _query = f'\nSELECT *\nFROM cohort c\nCROSS JOIN (SELECT DISTINCT vital_category FROM vitals) v\nORDER BY hospitalization_id, date_hr, vital_category\n'
    cohort_hrs_cross_vital_categories = duckdb.sql(_query).to_df()
    return


@app.cell
def _(duckdb):
    _query = '\n-- fill any missing values in the cohort hours with the nearest in time (in the past or future)\nSELECT hospitalization_id\n    , date_hr\n    , cohort_flag\n    , vital_category\n    , MEAN(vital_value) as mean_value\n    \n    , LAG(mean_value) OVER (PARTITION BY hospitalization_id, vital_category ORDER BY date_hr) as mean_value_lag\n    , LAST_VALUE(mean_value IGNORE NULLS) OVER (\n        PARTITION BY hospitalization_id, vital_category \n        ORDER BY date_hr\n        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW\n    ) as mean_value_final\n    -- , COALESCE(mean_value, mean_value_last) as mean_value_final\nFROM cohort_hrs_cross_vital_categories\nFULL OUTER JOIN vitals USING (hospitalization_id, date_hr, vital_category)\nGROUP BY hospitalization_id, date_hr, cohort_flag, vital_category\nORDER BY hospitalization_id, vital_category, date_hr, cohort_flag\n'
    vitals_hrly = duckdb.sql(_query).to_df()
    return


@app.cell
def _(duckdb):
    _query = '\nSELECT hospitalization_id\n    , date_hr\n    , cohort_flag\n    , vital_category\n    , mean_value\n    , mean_value_final\nFROM vitals_hrly\nWHERE cohort_flag IS NOT NULL\nORDER BY hospitalization_id, vital_category, date_hr, cohort_flag\n'
    vitals_cohort_hrs = duckdb.sql(_query).to_df()
    return (vitals_cohort_hrs,)


@app.cell
def _(vitals_cohort_hrs):
    vitals_cohort_hrs_w = vitals_cohort_hrs.pivot_table(
        index=['hospitalization_id', 'date_hr', 'cohort_flag'], 
        columns='vital_category', 
        values='mean_value_final', 
        fill_value=0
    ).reset_index()
    vitals_cohort_hrs_w.columns.name = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Assessments""")
    return


@app.cell
def _(cohort_hosp_ids, pc):
    pa_required_columns = [
        "hospitalization_id", 
        "recorded_dttm",
        "assessment_category", 
        "numerical_value"
    ]

    pa = pc.load_data(
        table = "clif_patient_assessments",
        columns = pa_required_columns,
        filters = {
            "hospitalization_id": cohort_hosp_ids,
            "assessment_category": ["gcs_total", "rass", "RASS"]
        }
    )

    pa['date_hr'] = pa['recorded_dttm'].dt.floor('h')
    return


@app.cell
def _():
    # pa_required_columns = [
    #     "hospitalization_id", 
    #     "recorded_dttm",
    #     "assessment_category", 
    #     "numerical_value"
    # ]

    # gcs = pc.load_data(
    #     table = "clif_patient_assessments",
    #     columns = pa_required_columns,
    #     filters = {
    #         "hospitalization_id": cohort_hosp_ids,
    #         "assessment_category": ["gcs_total", #"rass", "RASS", 
    #                                 "gcs_verbal", "gcs_motor", "gcs_eye"]
    #     }
    # )

    # gcs['date_hr'] = gcs['recorded_dttm'].dt.floor('h')

    # gcs_w = gcs.pivot_table(
    #     index=['hospitalization_id', 'recorded_dttm'],
    #     columns='assessment_category',
    #     values='numerical_value',
    #     aggfunc='mean'
    # ).reset_index()

    # # Flatten column names
    # gcs_w.columns.name = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Intubaton start hour""")
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT *\n    -- note that we subtract 23 instead of 24 to be inclusive of all minutes in the starting hr\n    , CASE WHEN cohort_flag = 'hr_24' THEN date_hr - INTERVAL '20 hours' \n        WHEN cohort_flag = 'hr_72' THEN date_hr - INTERVAL '68 hours'\n        ELSE NULL END as date_hr_intub\nFROM cohort c\n"
    cohort_intub = duckdb.sql(_query).to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Most recent GCS""")
    return


@app.cell
def _(duckdb):
    def find_most_recent_gcs(target_dttm_name: str='date_hr_intub', window_in_hr: int=22):
        """
        Find the most recent GCS assessment to the target time.
        """
        _query = f"\n    WITH t1 AS (\n        SELECT *\n            , CASE WHEN cohort_flag = 'hr_24' THEN date_hr - INTERVAL '{window_in_hr} hours' \n                WHEN cohort_flag = 'hr_72' THEN date_hr - INTERVAL '{window_in_hr + 48} hours'\n                ELSE NULL END as date_hr_intub\n        FROM cohort c\n    )\n    SELECT c.*\n        , p.numerical_value as gcs_total\n        , p.recorded_dttm as gcs_recorded_dttm\n        -- rn = 1 for the gcs w/ the latest recorded_dttm (and thus most recent)\n        , ROW_NUMBER() OVER (\n            PARTITION BY c.row_id, c.hospitalization_id, c.{target_dttm_name}\n            ORDER BY p.recorded_dttm DESC\n            ) as rn\n    FROM t1 c\n    LEFT JOIN pa p\n        ON c.hospitalization_id = p.hospitalization_id \n        AND p.assessment_category = 'gcs_total'\n        AND p.numerical_value IS NOT NULL\n        AND p.recorded_dttm <= c.{target_dttm_name}\n    QUALIFY (rn = 1) -- OR (gcs_total IS NULL) -- include cohort even if no gcs found\n    ORDER BY row_id, c.{target_dttm_name}, p.recorded_dttm\n    "
        return duckdb.sql(_query).to_df()
    pa_gcs_intub_hr = find_most_recent_gcs(target_dttm_name='date_hr_intub')
    pa_gcs_cohort_hr = find_most_recent_gcs(target_dttm_name='date_hr')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Most recent RASS""")
    return


@app.cell
def _(duckdb):
    def find_most_recent(clif_category: str, hr_from_cohort_dttm: int=1):
        """
        Find the most recent assessment to the target time.
        """
        _query = f"\n    WITH t1 AS (\n        SELECT *\n            , CASE WHEN cohort_flag = 'hr_24' THEN date_hr - INTERVAL '{hr_from_cohort_dttm} hours' \n                WHEN cohort_flag = 'hr_72' THEN date_hr - INTERVAL '{hr_from_cohort_dttm + 48} hours'\n                ELSE NULL END as date_hr_intub\n            , date_hr + INTERVAL '{hr_from_cohort_dttm} hours' as target_dttm\n        FROM cohort c\n    )\n    SELECT c.*\n        , p.numerical_value as value\n        , p.recorded_dttm\n        -- rn = 1 for the gcs w/ the latest recorded_dttm (and thus most recent)\n        , ROW_NUMBER() OVER (\n            PARTITION BY c.row_id, c.hospitalization_id, c.target_dttm\n            ORDER BY p.recorded_dttm DESC\n            ) as rn\n    FROM t1 c\n    LEFT JOIN pa p\n        ON c.hospitalization_id = p.hospitalization_id \n        AND p.assessment_category in ('{clif_category}', '{clif_category.lower()}', '{clif_category.upper()}')\n        AND p.numerical_value IS NOT NULL\n        AND p.recorded_dttm <= c.target_dttm\n    QUALIFY (rn = 1) --OR (value IS NULL) -- include cohort even if no clif_category found\n    ORDER BY row_id, c.target_dttm, p.recorded_dttm\n    "
        return duckdb.sql(_query).to_df()
    cohort_rass = find_most_recent(clif_category='rass', hr_from_cohort_dttm=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Medication""")
    return


@app.cell
def _(cohort_hosp_ids, pc):
    sed_med_categories = [
        "midazolam", "lorazepam", "hydromorphone", "fentanyl", "propofol", "dexmedetomidine", "ketamine"
    ]

    vaso_med_categories = [
        "epinephrine", "norepinephrine", "phenylephrine", "vasopressin", "angiotensin", 
        "dopamine", "dobutamine"
    ]

    mac_required_columns = [
        "hospitalization_id", 
        "admin_dttm",
        "med_category",
        "med_group",
        "med_dose",
        "med_dose_unit",
        "mar_action_name"
    ]

    mac = pc.load_data(
        table = "clif_medication_admin_continuous",
        columns = mac_required_columns,
        filters = {
            "hospitalization_id": cohort_hosp_ids,
            "med_category": sed_med_categories + vaso_med_categories
        }
    )

    mac['date_hr'] = mac['admin_dttm'].dt.floor('h')

    mac = mac[~mac['mar_action_name'].str.contains('bolus', case=False)]
    return sed_med_categories, vaso_med_categories


@app.cell
def _():
    # # Pivot MAC to wide format
    # mac_pivot = mac.pivot_table(
    #     index=['hospitalization_id', 'admin_dttm'],
    #     columns='med_category',
    #     values='med_dose',
    #     # aggfunc='sum',
    #     fill_value=0
    # ).reset_index()

    # # Flatten column names
    # mac_pivot.columns.name = None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Check dosage unit""")
    return


@app.cell
def _(duckdb, site, timestamp):
    _query = '\nSELECT med_category, med_dose_unit, COUNT(*) as n\nFROM mac\nGROUP BY med_category, med_dose_unit\nORDER BY med_category, n DESC\n'
    med_units_count = duckdb.sql(_query).to_df()
    med_units_count.to_csv(f'output/final/{site}_med_units_count_{timestamp}.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Most recent patient weight""")
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT m.*\n    , v.vital_value as weight_kg\n    , v.recorded_dttm as weight_recorded_dttm\n    -- rn = 1 for the weight w/ the latest recorded_dttm (and thus most recent)\n    , ROW_NUMBER() OVER (\n        PARTITION BY m.hospitalization_id, m.admin_dttm, m.med_category\n        ORDER BY v.recorded_dttm DESC\n        ) as rn\nFROM mac m\nLEFT JOIN vitals v \n    ON m.hospitalization_id = v.hospitalization_id \n    AND v.vital_category = 'weight_kg' AND v.vital_value IS NOT NULL\n    AND v.recorded_dttm <= m.admin_dttm  -- only past weights\nQUALIFY (rn = 1) -- OR (weight_kg IS NULL) -- include meds even if no weight found\nORDER BY m.hospitalization_id, m.admin_dttm, m.med_category, rn\n"
    mac_w_wt = duckdb.sql(_query).to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Standardize dosage unit""")
    return


@app.cell
def _(duckdb, pd):
    def standardize_dose_unit(df_name: str) -> pd.DataFrame:
        """
        Standardize everything to mcg/min.
        Assumes the presentation of the following columns:
        - med_dose_unit: the original unit of the dose
        - med_dose: the original dose
        - weight_kg: the (imputed, most recent) weight of the patient
        """
        _query = f"\n    SELECT *\n        , LOWER(med_dose_unit) AS med_dose_unit_lower\n        , CASE WHEN regexp_matches(med_dose_unit_lower, '/h(r|our)?\\b') THEN 1/60.0\n            WHEN regexp_matches(med_dose_unit_lower, '/m(in|inute)?\\b') THEN 1.0\n            ELSE NULL END as time_multiplier\n        , CASE WHEN contains(med_dose_unit_lower, '/kg/') THEN weight_kg\n            ELSE 1 END AS pt_weight_adjustment\n        , CASE WHEN contains(med_dose_unit_lower, 'mcg/') THEN 1.0\n            WHEN contains(med_dose_unit_lower, 'mg/') THEN 1000.0\n            WHEN contains(med_dose_unit_lower, 'ng/') THEN 0.001\n            WHEN contains(med_dose_unit_lower, 'milli') THEN 0.001\n            WHEN contains(med_dose_unit_lower, 'units/') THEN 1\n            ELSE NULL END as dose_mass_multiplier\n        , med_dose * time_multiplier * pt_weight_adjustment * dose_mass_multiplier as med_dose_converted\n        , CASE WHEN contains(med_dose_unit_lower, 'units/') THEN 'units/min'\n            ELSE 'mcg/min' END as med_dose_unit_converted\n    FROM {df_name}\n    "
        return duckdb.sql(_query).to_df()
    mac_converted = standardize_dose_unit('mac_w_wt')
    return (mac_converted,)


@app.cell
def _(mac_converted):
    mac_converted.value_counts('mar_action_name')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Remove outliers""")
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT *\n    , CASE WHEN med_category in ('midazolam', 'lorazepam') AND med_dose_converted > (10*1000/60.0) THEN 1 \n        WHEN med_category in ('hydromorphone') AND med_dose_converted > (4*1000/60.0) THEN 1 \n        WHEN med_category in ('fentanyl') AND med_dose_converted > (700/60.0) THEN 1 \n        WHEN med_category in ('propofol') AND med_dose_converted > (4000*weight_kg/60.0) THEN 1 \n        WHEN med_category in ('dexmedetomidine') AND med_dose_converted > (1.5*weight_kg/60.0) THEN 1 \n        WHEN med_category in ('ketamine') AND med_dose_converted > (6000*weight_kg/60.0) THEN 1 \n        WHEN med_category in ('norepinephrine', 'epinephrine') AND med_dose_converted > weight_kg THEN 1 \n        WHEN med_category in ('phenylephrine') AND med_dose_converted > (5*weight_kg) THEN 1 \n        WHEN med_category in ('dopamine') AND med_dose_converted > (20*weight_kg) THEN 1 \n        WHEN med_category in ('dobutamine') AND med_dose_converted > (40*weight_kg) THEN 1 \n        WHEN med_category in ('vasopressin') AND med_dose_converted > (0.05) THEN 1 \n        WHEN med_category in ('angiotensin') AND med_dose_converted > (0.04*weight_kg) THEN 1 \n\n        ELSE 0 END as is_outlier\nFROM mac_converted\n"
    mac_no_outliers = duckdb.sql(_query).to_df()
    _mask = mac_no_outliers['is_outlier'] == 1
    mac_no_outliers = mac_no_outliers[~_mask]
    print(f'Removed {_mask.sum()} ({_mask.mean():.2%}) outliers')
    return (mac_no_outliers,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Remove duplicates""")
    return


@app.cell
def _(duckdb, mac_no_outliers, pd):
    mac_no_outliers.drop_duplicates(subset=['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose_converted'], inplace=True)

    def remove_meds_duplicates(meds_df_name: str) -> pd.DataFrame:
        _query = f"\n    SELECT *\n        , LOWER(mar_action_name) as mar_action_lower\n        , ROW_NUMBER() OVER (\n            PARTITION BY hospitalization_id, admin_dttm, med_category\n            ORDER BY \n                CASE\n                    WHEN contains(mar_action_lower, 'verify') THEN 10 -- CAST('inf' AS DOUBLE)\n                    -- WHEN contains(mar_action_lower, 'stopped') THEN 9\n                    ELSE 1\n                END,\n                CASE -- deprioritize zero or null doses\n                    WHEN med_dose_converted > 0 THEN 1\n                    ELSE 2\n                END,\n                med_dose_converted desc -- prioritize larger doses\n        ) as rn_dedup\n    FROM {meds_df_name}\n    -- QUALIFY rn_dedup = 1\n    ORDER BY hospitalization_id, admin_dttm, med_category;\n    "
        return duckdb.sql(_query).to_df()
    mac_deduped = remove_meds_duplicates('mac_converted')
    mac_deduped.drop(columns=['med_dose', 'med_dose_unit'], inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Scaffold cohort with med_categories""")
    return


@app.cell
def _(duckdb):
    _query = f'\nSELECT *\nFROM cohort c\nCROSS JOIN (SELECT DISTINCT med_category FROM mac) m\nORDER BY hospitalization_id, date_hr, med_category\n'
    cohort_hrs_cross_med_categories = duckdb.sql(_query).to_df()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Calculate cumulative dosage""")
    return


@app.cell
def _(duckdb):
    _query = "\n-- insert the cohort hours into the mac record\nSELECT hospitalization_id\n    , cohort_flag\n    , med_category\n    , date_hr\n    , admin_dttm\n    , mar_action_name\n    , med_dose_converted as med_dose\n    , med_dose_unit_converted as med_dose_unit\n    , weight_kg\n    , LAST_VALUE(admin_dttm IGNORE NULLS) OVER (\n        PARTITION BY hospitalization_id, med_category \n        ORDER BY date_hr, admin_dttm \n        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING\n    ) as admin_dttm_last\n    , LAST_VALUE(med_dose IGNORE NULLS) OVER (\n        PARTITION BY hospitalization_id, med_category \n        ORDER BY date_hr, admin_dttm \n        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING\n    ) as med_dose_last\n    , LAST_VALUE(mar_action_name IGNORE NULLS) OVER (\n        PARTITION BY hospitalization_id, med_category \n        ORDER BY date_hr, admin_dttm \n        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING\n    ) as mar_action_name_last\n    -- add helper flags for first and last streak of the same med within the hour -- which needs special handling\n    , CASE WHEN admin_dttm = MIN(admin_dttm) OVER (PARTITION BY hospitalization_id, med_category, date_hr) \n        THEN 1 ELSE 0 END as is_first_streak\n    , CASE WHEN admin_dttm = MAX(admin_dttm) OVER (PARTITION BY hospitalization_id, med_category, date_hr) \n        THEN 1 ELSE 0 END as is_last_streak\n    , date_hr + INTERVAL '1 hour' as date_hr_next\nFROM cohort_hrs_cross_med_categories c\nFULL OUTER JOIN mac_deduped m USING (hospitalization_id, date_hr, med_category)\nORDER BY hospitalization_id, med_category, date_hr, admin_dttm\n"
    mac_hrly = duckdb.sql(_query).to_df()
    return


@app.cell
def _():
    # query = """
    # SELECT *

    # FROM mac_hrly
    # ORDER BY hospitalization_id, med_category, date_hr, admin_dttm
    # """
    # # forward filled
    # mac_hrly_ff = duckdb.sql(query).to_df()
    return


@app.cell
def _(duckdb):
    _query = '\n-- keep only the med admins that are within the cohort hours\nSELECT *\nFROM mac_hrly\nWHERE cohort_flag IS NOT NULL -- AND admin_dttm_last IS NOT NULL\nORDER BY hospitalization_id, med_category, date_hr, admin_dttm\n'
    mac_cohort_hrs = duckdb.sql(_query).to_df()
    return (mac_cohort_hrs,)


@app.cell
def _(mac_cohort_hrs):
    _mask = mac_cohort_hrs['cohort_flag'] == 'hr_24'
    _mask.sum()
    return


@app.cell
def _(duckdb, vaso_med_categories):
    _query = f'\n-- calculate the cumulative dosage within the hr\nSELECT hospitalization_id, cohort_flag, med_category, date_hr\n    , SUM(CASE \n        -- if no mac admin record within the cohort hour, use the last observed dose and assume it runs the entire 60 mins\n        WHEN admin_dttm IS NULL\n            THEN 60.0 * COALESCE(med_dose_last, 0)\n        -- otherwise, calculate the cumulative dosage within the hr\n        WHEN is_first_streak = 1\n            THEN EXTRACT(EPOCH FROM (admin_dttm - date_hr))/60.0 * med_dose_last \n        WHEN is_first_streak != 1 AND admin_dttm IS NOT NULL\n            THEN EXTRACT(EPOCH FROM (admin_dttm - admin_dttm_last))/60.0 * med_dose_last \n        WHEN is_last_streak = 1\n            THEN EXTRACT(EPOCH FROM (date_hr_next - admin_dttm))/60.0 * med_dose\n        ELSE 0 END) as total_dosage\n    , COUNT(DISTINCT CASE \n        WHEN total_dosage > 0 AND med_category in {vaso_med_categories}\n        THEN med_category END) OVER (PARTITION BY hospitalization_id, date_hr) as n_pressors\n    , CASE WHEN total_dosage > 0 THEN total_dosage ELSE NULL END as total_dosage_na\nFROM mac_cohort_hrs\nGROUP BY hospitalization_id, med_category, date_hr, cohort_flag, med_dose_unit\nORDER BY hospitalization_id, med_category, date_hr\n'
    mac_cohort_dosage = duckdb.sql(_query).to_df()
    return (mac_cohort_dosage,)


@app.cell
def _(mac_cohort_dosage):
    mac_cohort_dosage_w = mac_cohort_dosage.pivot_table(
        index=['hospitalization_id', 'date_hr', 'cohort_flag', 'n_pressors'], 
        columns='med_category', 
        values='total_dosage', 
        fill_value=0
    ).reset_index()
    mac_cohort_dosage_w.columns.name = None
    return


@app.cell
def _():
    # query = """
    # SELECT hospitalization_id
    #     , date_hr
    #     , cohort_flag
    #     , med_category
    #     , COALESCE(total_dosage, 0) as total_dosage
    #     , med_dose_unit
    # FROM cohort_hrs_cross_med_categories c
    # FULL OUTER JOIN mac_cohort_dosage m USING (hospitalization_id, date_hr, med_category, cohort_flag)
    # ORDER BY hospitalization_id, cohort_flag, med_category, date_hr
    # """
    # df = duckdb.sql(query).to_df()
    return


@app.cell
def _(mac_cohort_dosage):
    mac_cohort_dosage['med_category'].unique().tolist()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## SOFA score""")
    return


@app.cell
def _(duckdb):
    _query = "\nSELECT row_id\n    , date_hr - INTERVAL '23 hours' as start_dttm\n    , date_hr + INTERVAL '1 hours'as stop_dttm\nFROM cohort\n"
    sofa_input = duckdb.sql(_query).to_df()
    return (sofa_input,)


@app.cell
def _(duckdb):
    _query = '\nSELECT row_id\n    , hospitalization_id\nFROM cohort\n'
    id_mappings = duckdb.sql(_query).to_df()
    return (id_mappings,)


@app.cell
def _(reload):
    from utils import sofa_score
    reload(sofa_score)
    return (sofa_score,)


@app.cell
def _(helper, id_mappings, pc, site, sofa_input, sofa_score, warnings):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning)
        sofa_score_1 = sofa_score.compute_sofa(ids_w_dttm=sofa_input, tables_path=helper['tables_path'], use_hospitalization_id=False, id_mapping=id_mappings, output_filepath=f'output/intermediate/{site}_sofa.parquet', helper_module=pc)
    return (sofa_score_1,)


@app.cell
def _(sofa_score_1):
    sofa_score_1.columns
    return


@app.cell
def _(sofa_score_1, vaso_med_categories):
    sofa_score_1.drop(columns=vaso_med_categories, inplace=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Merge results""")
    return


@app.cell
def _(vaso_med_categories):
    vaso_med_categories
    return


@app.cell
def _(duckdb):
    _query = f"\nSELECT row_id\n    , hospitalization_id, date_hr, cohort_flag\n    , EXTRACT(hour FROM date_hr) as hr\n    , CASE \n        WHEN EXTRACT(hour FROM date_hr) >= 7 AND EXTRACT(hour FROM date_hr) < 19 \n        THEN 'day' ELSE 'night' END AS shift\n    , d.age_at_admission as age\n    , d.race_category as race\n    , d.ethnicity_category as ethnicity\n    , d.sex_category as sex\n    , v.*\n    , r.value as rass\n    , CASE WHEN rass in (-4, -5) THEN 1 \n        WHEN rass IS NULL THEN NULL\n        ELSE 0 END as rass_deep_sedation\n  -- sedatives\n    , LEAST(m.midazolam, 10*1000) as _midazolam\n    , LEAST(m.lorazepam, 10*1000) as _lorazepam\n    , LEAST(m.hydromorphone, 4*1000) as _hydromorphone\n    , LEAST(m.fentanyl, 700) as _fentanyl -- too low?\n    , LEAST(m.propofol, 6000*v.weight_kg) as _propofol\n    , LEAST(m.dexmedetomidine, 1.5*v.weight_kg) as _dexmedetomidine\n    , LEAST(m.ketamine, 6000*v.weight_kg) as _ketamine\n    -- vasopressors\n    , LEAST(m.norepinephrine, 1*60*v.weight_kg) as _norepinephrine\n    , LEAST(m.epinephrine, 1*60*v.weight_kg) as _epinephrine\n    , LEAST(m.phenylephrine, 5*60*v.weight_kg) as _phenylephrine\n    , LEAST(m.dopamine, 20*60*v.weight_kg) as _dopamine\n    , LEAST(m.dobutamine, 40*60*v.weight_kg) as _dobutamine\n    , LEAST(m.vasopressin, 0.05*60) as _vasopressin\n    , LEAST(m.angiotensin, 0.04*60*v.weight_kg) as _angiotensin\n    -- converted equivalents\n    , (_hydromorphone * 0.05 + _fentanyl) AS fentanyl_eq\n    , _lorazepam * 2 + _midazolam AS midazolam_eq\n    , _norepinephrine + _epinephrine + _phenylephrine / 10.0 + _dopamine / 100.0 \n        + _vasopressin * 2.5 + _angiotensin * 10 as ne_eq\n    , n_pressors\n    , s.sofa_total as sofa\n    , COALESCE(s.po2_arterial_recent, s.pao2_imputed_recent) AS pao2\n    , s.fio2_recent * 100 AS fio2\n    , COALESCE(s.p_f, s.p_f_imputed) / 100 AS p_f\nFROM cohort c\nLEFT JOIN cohort_demogs d USING (hospitalization_id)\nLEFT JOIN vitals_cohort_hrs_w v USING (hospitalization_id, date_hr, cohort_flag)\nLEFT JOIN cohort_rass r USING (row_id)\nLEFT JOIN mac_cohort_dosage_w m USING (hospitalization_id, date_hr, cohort_flag)\nLEFT JOIN sofa_score s USING (row_id) \nORDER BY row_id\n"
    cohort_results = duckdb.sql(_query).to_df()
    cohort_results_24 = cohort_results[cohort_results['cohort_flag'] == 'hr_24'].copy()
    cohort_results_72 = cohort_results[cohort_results['cohort_flag'] == 'hr_72'].copy()
    return cohort_results, cohort_results_24, cohort_results_72


@app.cell
def _(cohort_results, site):
    cohort_results.to_parquet(f"output/intermediate/{site}_cohort_results.parquet")
    return


@app.cell
def _(cohort_results_24, cohort_results_72):
    mask_24 =cohort_results_24['_hydromorphone'] > 0
    mask_72 =cohort_results_72['_hydromorphone'] > 0

    print(f"24 hours: {mask_24.sum()} out of {len(mask_24)}")
    print(f"72 hours: {mask_72.sum()} out of {len(mask_72)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Table one""")
    return


@app.cell
def _(cohort_results):
    cohort_results.columns
    return


@app.cell
def _(sed_med_categories, vaso_med_categories):
    vitals_vars = [
        'sbp', 'dbp', 'map', 'heart_rate', 'respiratory_rate', 'spo2', 'temp_c', 'weight_kg', 'height_cm'
        ]

    resp_vars = ['pao2', 'fio2', 'p_f']

    sed_med_vars = ["_" + med for med in sed_med_categories]
    vaso_med_vars = ["_" + med for med in vaso_med_categories]

    cat_vars = ['race', 'ethnicity', 'sex', 'rass_deep_sedation']
    eq_vars = ['fentanyl_eq', 'midazolam_eq', 'ne_eq']
    cont_vars = ['age', 'rass', 'sofa', 'n_pressors'] \
        + resp_vars + vitals_vars + eq_vars + sed_med_vars + vaso_med_vars
    return (
        cat_vars,
        cont_vars,
        eq_vars,
        resp_vars,
        sed_med_vars,
        vaso_med_vars,
        vitals_vars,
    )


@app.cell
def _(site, tableone, timestamp):
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
        table = tableone.TableOne(**kwargs, pval=True)
        table.to_csv(f'output/final/{site}_{file_name}_{timestamp}.csv')
        return table
    return (gen_and_save_tableone,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Customized""")
    return


@app.cell
def _():
    # query = """
    # SELECT cohort_flag, SUM(CASE WHEN _midazolam > 0 THEN 1 ELSE 0 END) as n_patients
    # FROM cohort_results
    # GROUP BY cohort_flag
    # """
    # df4 = duckdb.sql(query).to_df()

    # df4.head()
    return


@app.cell
def _(cohort_results, eq_vars, sed_med_vars, vaso_med_vars):
    # Pivot cohort_results to long format
    cohort_results_l = cohort_results.melt(
        id_vars=['row_id', 'hospitalization_id', 'date_hr', 'cohort_flag', 'hr', 'shift', 
                 'age', 'race', 'ethnicity', 'sex'],
        value_vars= sed_med_vars + vaso_med_vars + eq_vars,
        var_name='variable',
        value_name='value'
    )

    cohort_results_l['value_na'] = cohort_results_l['value'].where(cohort_results_l['value'] > 0)
    return


@app.cell
def _(duckdb, site, timestamp):
    _query = '\n-- count the no. of patients on each sedative\nSELECT DISTINCT cohort_flag, variable\n    --, CASE WHEN value > 0 THEN value ELSE NULL END as value_na\n    , COUNT(DISTINCT CASE WHEN value_na IS NOT NULL THEN row_id ELSE NULL END) \n        OVER (PARTITION BY cohort_flag, variable) as n_pt\n    , COUNT(DISTINCT row_id) OVER (PARTITION BY cohort_flag) as total_pt\n    , n_pt / total_pt as pct_pt\n    , QUANTILE_CONT(value_na, 0.25) \n        OVER (PARTITION BY cohort_flag, variable) as q1_value\n    , QUANTILE_CONT(value_na, 0.5) \n        OVER (PARTITION BY cohort_flag, variable) as median_value\n    , QUANTILE_CONT(value_na, 0.75) \n        OVER (PARTITION BY cohort_flag, variable) as q3_value\nFROM cohort_results_l\nORDER BY variable, cohort_flag\n'
    pct_pt_on_med = duckdb.sql(_query).to_df()
    pct_pt_on_med.to_csv(f'output/final/{site}_pct_pt_on_med_{timestamp}.csv', index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### By cohort""")
    return


@app.cell
def _(
    cat_vars,
    cohort_results,
    cont_vars,
    gen_and_save_tableone,
    resp_vars,
    vitals_vars,
):
    gen_and_save_tableone(
        file_name='table_one_by_cohort_hr',
        data=cohort_results, 
        continuous=cont_vars, 
        categorical=cat_vars, 
        groupby='cohort_flag',
        nonnormal=['age', 'rass', 'sofa'] + resp_vars + vitals_vars
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### By race""")
    return


@app.cell
def _(
    cohort_results,
    cont_vars,
    gen_and_save_tableone,
    resp_vars,
    vitals_vars,
):
    gen_and_save_tableone(
        file_name='table_one_by_race',
        data=cohort_results, 
        continuous=cont_vars, 
        categorical=['ethnicity', 'sex'], 
        groupby='race',
        nonnormal=['age', 'rass', 'sofa'] + resp_vars + vitals_vars
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### By shift""")
    return


@app.cell
def _(cat_vars, cohort_results_24, cont_vars, gen_and_save_tableone):
    gen_and_save_tableone(
        file_name='table_one_by_shift_hr24',
        data=cohort_results_24, 
        continuous=cont_vars, 
        categorical=cat_vars, 
        groupby='shift'
        # nonnormal=['age', 'rass', 'sofa'] + resp_vars + vitals_vars,    
        )
    return


@app.cell
def _(cat_vars, cohort_results_72, cont_vars, gen_and_save_tableone):
    gen_and_save_tableone(
        file_name='table_one_by_shift_hr72',
        data=cohort_results_72, 
        continuous=cont_vars, 
        categorical=cat_vars, 
        groupby='shift',
        # nonnormal=['age', 'rass', 'sofa'] + resp_vars + vitals_vars,
        )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Regression""")
    return


@app.cell
def _(cohort_results_24):
    cohort_results_24.columns
    return


@app.cell
def _(cohort_results_24, cohort_results_72, pd, site, timestamp):
    import statsmodels.formula.api as smf

    def run_regression_analysis(cohort_hr, outcome_predictors):
        """
        Run regression analysis for sedation outcomes

        Parameters:
        cohort_hr (str): Either '24' or '72' to specify which cohort to use
        outcome_predictors (dict): Dictionary mapping outcomes to their predictors
                                  e.g., {"outcome_1": ["predictor_1", "predictor_2"]}

        Returns:
        dict: Dictionary containing fitted regression models
        """

        # Map cohort_hr to corresponding dataframe
        if cohort_hr == '24':
            data = cohort_results_24
        elif cohort_hr == '72':
            data = cohort_results_72
        else:
            raise ValueError("cohort_hr must be either '24' or '72'")

        categorical_vars = ['shift', 'race', 'ethnicity', 'sex']
        results = {}

        # Get all unique predictors and outcomes for data cleaning
        all_predictors = set()
        all_outcomes = list(outcome_predictors.keys())

        for predictors in outcome_predictors.values():
            all_predictors.update(predictors)

        all_predictors = list(all_predictors)

        # Create a clean dataset by dropping rows with NA values in key variables
        regression_data = data.dropna(subset=all_predictors + all_outcomes)

        print(f"Original dataset size: {len(data)}")
        print(f"Dataset size after dropping NAs: {len(regression_data)}")

        # Run regression for each outcome
        for outcome, predictors in outcome_predictors.items():
            print(f"\n{'='*60}")
            print(f"Regression for {outcome} using formula API (cohort_hr={cohort_hr})")
            print('='*60)

            # Create formula string
            # C() indicates categorical variable
            categorical_terms = [f"C({p})" for p in predictors if p in categorical_vars]
            continuous_terms = [p for p in predictors if p not in categorical_vars]

            formula = f"{outcome} ~ " + " + ".join(categorical_terms + continuous_terms)

            # Fit model
            model = smf.ols(formula=formula, data=regression_data)
            results[outcome] = model.fit()

            print(results[outcome].summary())

        # Save regression coefficients and statistics as CSV
        for outcome in outcome_predictors.keys():
            # Extract coefficient table
            coef_df = pd.DataFrame({
                'coefficient': results[outcome].params,
                'std_err': results[outcome].bse,
                't_value': results[outcome].tvalues,
                'p_value': results[outcome].pvalues,
                'conf_int_lower': results[outcome].conf_int()[0],
                'conf_int_upper': results[outcome].conf_int()[1]
            })

            # Add model statistics
            model_stats = pd.DataFrame({
                'statistic': ['r_squared', 'adj_r_squared', 'f_statistic', 'f_pvalue', 'aic', 'bic', 'n_obs'],
                'value': [
                    results[outcome].rsquared,
                    results[outcome].rsquared_adj,
                    results[outcome].fvalue,
                    results[outcome].f_pvalue,
                    results[outcome].aic,
                    results[outcome].bic,
                    results[outcome].nobs
                ]
            })

            # Save coefficient table
            coef_path = f'output/final/{site}_reg_coeff_{outcome}_hr{cohort_hr}_{timestamp}.csv'
            coef_df.to_csv(coef_path)

            # Save model statistics
            stats_path = f'output/final/{site}_reg_stats_{outcome}_hr{cohort_hr}_{timestamp}.csv'
            model_stats.to_csv(stats_path, index=False)

        print(f"\nResults saved to:")
        print(f"  CSV coefficients: {coef_path}")
        print(f"  CSV statistics: {stats_path}")

        return results

    ols_predictors = [
            'shift', 
            'age', 
            'race', 'ethnicity', 'sex', 
            'dbp', 'map', 'sbp', 
            'height_cm', 'weight_kg',
            'heart_rate', 'respiratory_rate', 'spo2', 'temp_c', 
            'ne_eq', 'n_pressors', 
            'rass', 
            'sofa', 
            'p_f'
        ]

    # Define outcome-predictor mappings
    outcome_predictors = {
        'fentanyl_eq': ols_predictors,
        '_fentanyl': ols_predictors,
        'midazolam_eq': ols_predictors,
        '_propofol': ols_predictors
    }

    # Run analysis for both cohorts
    results_24 = run_regression_analysis('24', outcome_predictors)
    results_72 = run_regression_analysis('72', outcome_predictors)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
