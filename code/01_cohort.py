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
    RERUN_WATERFALL = False


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 Cohort Identification

    Identifies ICU patients with first IMV streak >= 24 hours.
    Builds hourly time grids with day/shift annotations.
    Computes NMB exclusion flags.
    """)
    return


@app.cell
def _():
    from clifpy import ClifOrchestrator
    import pandas as pd
    import duckdb
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"
    co = ClifOrchestrator(config_path=CONFIG_PATH)
    return CONFIG_PATH, apply_outlier_handling, get_config_or_params, pd


@app.cell
def _(CONFIG_PATH, get_config_or_params):
    # Site-scoped output directories — every per-site run lives under
    # output/{site}/ and output_to_share/{site}/ so multiple sites coexist
    # on disk (see Makefile's SITE= flag and the "Multi-site support"
    # section in .dev/CLAUDE.md).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    os.makedirs(f"output_to_share/{SITE_NAME}", exist_ok=True)
    os.makedirs(f"output_to_share/{SITE_NAME}/figures", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return (SITE_NAME,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load ADT & Hospitalization
    """)
    return


@app.cell
def _():
    from clifpy import Adt, Hospitalization

    adt = Adt.from_file(
        config_path='config/config.json',
        columns=['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category', 'location_type'],
        filters={
            'location_category': ['icu']
        }
    )

    adt_df = adt.df
    hosp_ids_w_icu_stays = adt_df['hospitalization_id'].unique().tolist()
    print(f"Hospitalizations with ICU stays: {len(hosp_ids_w_icu_stays)}")
    return Adt, Hospitalization, adt_df, hosp_ids_w_icu_stays


@app.cell
def _(Adt, Hospitalization, hosp_ids_w_icu_stays):
    from clifpy.utils.stitching_encounters import stitch_encounters

    hosp_w_icu_stays = Hospitalization.from_file(
        config_path='config/config.json',
        filters={
            'hospitalization_id': hosp_ids_w_icu_stays
        }
    )
    adt_w_icu_stays = Adt.from_file(
        config_path='config/config.json',
        filters={
            'hospitalization_id': hosp_ids_w_icu_stays
        }
    )

    # NOTE: Encounter stitching computed but not yet wired into downstream pipeline
    hosp_stitched, adt_stitched, encounter_mapping = stitch_encounters(
        hospitalization=hosp_w_icu_stays.df,
        adt=adt_w_icu_stays.df,
        time_interval=12  # 12-hour window
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Respiratory Support
    """)
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

    resp_processed_path = f"output/{SITE_NAME}/resp_processed_bf.parquet"

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
        print(f"Loading cached {resp_processed_path}")
        resp_p = pd.read_parquet(resp_processed_path)

    print(f"resp_p: {len(resp_p)} rows")
    return (resp_p,)


@app.cell
def _(resp_p):
    resp_p['tracheostomy'] = resp_p['tracheostomy'].fillna(0).astype(int)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## IMV Streak Detection

    Inlined from `cohort_id.sql`. Each CTE is a separate cell for interactive inspection.
    """)
    return


@app.cell
def _(resp_p):
    cohort_t1 = mo.sql(
        f"""
        -- Detect IMV transitions: intubation and extubation events
        FROM resp_p
        SELECT hospitalization_id
            , event_dttm: recorded_dttm
            , device_category
            , tracheostomy
            , _on_imv: CASE WHEN device_category = 'imv' THEN 1 ELSE 0 END
            , _chg_imv: CASE
                -- getting off imv (extub)
                WHEN (_on_imv = 0 AND LAG(_on_imv) OVER w = 1)
                -- getting on imv (intub)
                OR (_on_imv = 1 AND _on_imv IS DISTINCT FROM LAG(_on_imv) OVER w)
                THEN 1 ELSE 0 END
            , _trach_flip_to_1: CASE
                WHEN tracheostomy = 1 AND tracheostomy IS DISTINCT FROM LAG(tracheostomy) OVER w THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        """
    )
    return (cohort_t1,)


@app.cell
def _(cohort_t1):
    cohort_t2 = mo.sql(
        f"""
        -- Assign streak IDs and track first tracheostomy
        FROM cohort_t1
        SELECT *
            , _streak_id: SUM(_chg_imv) OVER w
            , _trach_flip_cumsum: SUM(_trach_flip_to_1) OVER w
            , _trach_1st: CASE
                WHEN _trach_flip_to_1 = 1 AND _trach_flip_cumsum = 1 THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        """
    )
    return (cohort_t2,)


@app.cell
def _(cohort_t2):
    all_streaks = mo.sql(
        f"""
        -- Aggregate per streak: start, end, tracheostomy timing
        FROM cohort_t2
        SELECT hospitalization_id
            , _streak_id
            , _start_dttm: MIN(event_dttm)
            , _last_observed_dttm: MAX(event_dttm)
            , _trach_dttm: MIN(CASE WHEN _trach_1st = 1 THEN event_dttm END)
            , _on_imv: MAX(_on_imv)
        GROUP BY hospitalization_id, _streak_id
        """
    )
    return (all_streaks,)


@app.cell
def _(all_streaks):
    all_streaks_w_lead = mo.sql(
        f"""
        -- Compute streak duration and flag ≥24h streaks
        FROM all_streaks
        SELECT *
            , _next_start_dttm: LEAD(_start_dttm) OVER w
            , _end_dttm: COALESCE(_trach_dttm, _next_start_dttm, _last_observed_dttm)
            , _duration_hrs: date_diff('minute', _start_dttm, _end_dttm) / 60
            , _at_least_24h: CASE WHEN _duration_hrs >= 24 THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _streak_id)
        ORDER BY hospitalization_id, _streak_id
        """
    )
    return (all_streaks_w_lead,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## First Qualifying IMV Streak
    """)
    return


@app.cell
def _(all_streaks_w_lead):
    cohort_imv_streaks = mo.sql(
        f"""
        -- First IMV streak ≥24 hours per hospitalization
        FROM all_streaks_w_lead
        SELECT hospitalization_id, _streak_id, _start_dttm, _end_dttm, _duration_hrs
        WHERE _at_least_24h = 1
        AND _on_imv = 1
        AND _streak_id = 1
        """
    )
    return (cohort_imv_streaks,)


@app.cell
def _(SITE_NAME, all_streaks_w_lead, cohort_imv_streaks, hosp_ids_w_icu_stays, pd):
    import duckdb as _ddb
    from pathlib import Path as _Path

    cohort_hosp_ids_pre_weight = cohort_imv_streaks.df()['hospitalization_id'].unique().tolist()

    # Intermediate CONSORT counts from all_streaks_w_lead
    _streaks_df = all_streaks_w_lead.df()

    _n_with_any_imv = _streaks_df.loc[_streaks_df['_on_imv'] == 1, 'hospitalization_id'].nunique()
    _first_imv = _streaks_df[(_streaks_df['_streak_id'] == 1) & (_streaks_df['_on_imv'] == 1)]
    _n_first_imv_lt24 = len(_first_imv[_first_imv['_at_least_24h'] == 0])
    _n_trach_truncated = len(_first_imv[_first_imv['_trach_dttm'].notna() & (_first_imv['_at_least_24h'] == 0)])

    # Weight-QC exclusion (Phase 2 of weight-audit work). Reads the drop list
    # produced by `make weight-audit SITE=<site>`. If the audit hasn't run
    # yet, the file won't exist and we skip the exclusion — first-run pipelines
    # complete without weight QC, then the user runs the audit, then re-runs
    # this script to apply the drop. See code/qc/weight_audit_README.md.
    _drop_path = _Path(f"output/{SITE_NAME}/qc/weight_qc_drop_list.parquet")
    if _drop_path.exists():
        _drop_df = pd.read_parquet(_drop_path)
        _weight_drop_ids = set(_drop_df['hospitalization_id'].astype(str))
        weight_drop_breakdown = _drop_df['_drop_reason'].value_counts().to_dict()
        cohort_hosp_ids = [
            h for h in cohort_hosp_ids_pre_weight
            if str(h) not in _weight_drop_ids
        ]
        _n_weight_excluded = len(cohort_hosp_ids_pre_weight) - len(cohort_hosp_ids)
        print(f"Weight-QC exclusion: {_n_weight_excluded} hospitalizations dropped")
        for _reason, _n in weight_drop_breakdown.items():
            print(f"  {_reason}: {_n}")
    else:
        weight_drop_breakdown = {}
        cohort_hosp_ids = list(cohort_hosp_ids_pre_weight)
        _n_weight_excluded = 0
        print(f"No weight-QC drop list at {_drop_path} — skipping weight QC")
        print(f"  (run `make weight-audit SITE={SITE_NAME}` then re-run 01_cohort.py to apply)")

    consort_counts = {
        'n_icu': len(hosp_ids_w_icu_stays),
        'n_any_imv': _n_with_any_imv,
        'n_no_imv': len(hosp_ids_w_icu_stays) - _n_with_any_imv,
        'n_first_imv_ge24': len(cohort_hosp_ids_pre_weight),
        'n_first_imv_lt24': _n_first_imv_lt24,
        'n_trach_truncated': _n_trach_truncated,
        'n_weight_excluded': _n_weight_excluded,
        'n_post_weight': len(cohort_hosp_ids),
        'weight_drop_breakdown': weight_drop_breakdown,
    }
    print(f"Cohort hospitalizations (first IMV ≥24h, post-weight-QC): {len(cohort_hosp_ids)}")
    print(f"  Excluded — no IMV: {consort_counts['n_no_imv']}")
    print(f"  Excluded — first IMV <24h: {consort_counts['n_first_imv_lt24']} (of which {consort_counts['n_trach_truncated']} tracheostomy-truncated)")
    print(f"  Excluded — weight-QC: {_n_weight_excluded}")
    return cohort_hosp_ids, consort_counts


@app.cell
def _(adt_df, cohort_hosp_ids):
    icu_type_df = mo.sql(
        f"""
        -- First ICU type per cohort hospitalization (earliest ADT record)
        FROM adt_df
        SELECT hospitalization_id
            , icu_type: FIRST(location_type ORDER BY in_dttm)
        WHERE hospitalization_id IN (SELECT UNNEST({cohort_hosp_ids}))
        GROUP BY hospitalization_id
        """
    )
    return (icu_type_df,)


@app.cell
def _(icu_type_df):
    print(f"ICU types extracted: {len(icu_type_df)} hospitalizations")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly Time Grids
    """)
    return


@app.cell
def _(cohort_imv_streaks):
    cohort_hrly_grids = mo.sql(
        f"""
        -- Generate hourly time grid from start to end of each qualifying IMV streak
        FROM (
            FROM cohort_imv_streaks
            SELECT hospitalization_id
                , _start_hr: date_trunc('hour', _start_dttm)
                , _end_hr: date_trunc('hour', _end_dttm) + INTERVAL '1 hour'
        )
        SELECT hospitalization_id
            , unnest(generate_series(_start_hr, _end_hr, INTERVAL '1 hour')) AS event_dttm
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cohort_hrly_grids,)


@app.cell
def _(cohort_hrly_grids):
    from _utils import add_day_shift_id

    _grids_df = cohort_hrly_grids.df()
    cohort_hrly_grids_f = add_day_shift_id(_grids_df)
    assert len(cohort_hrly_grids_f) == len(_grids_df), 'length altered by add_day_shift_id'
    print(f"Hourly grid rows: {len(cohort_hrly_grids_f)}")
    return (cohort_hrly_grids_f,)


@app.cell
def _(cohort_hrly_grids_f):
    cohort_shift_change_grids = cohort_hrly_grids_f[cohort_hrly_grids_f['_hr'].isin([7, 19])]
    print(f"Shift-change grid rows (7am/7pm only): {len(cohort_shift_change_grids)}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## NMB Exclusion

    Exclude patient-days with >1 hour of neuromuscular blockade.
    Agents: cisatracurium, vecuronium, rocuronium (verified in mCIDE, med_group: paralytics).
    """)
    return


@app.cell
def _():
    from clifpy import MedicationAdminContinuous

    nmb = MedicationAdminContinuous.from_file(
        config_path='config/config.json',
        columns=['hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit'],
        filters={
            'med_category': ['cisatracurium', 'vecuronium', 'rocuronium']
        }
    )

    nmb_df = nmb.df
    print(f"NMB records: {len(nmb_df)}")
    return (nmb_df,)


@app.cell
def _(nmb_df):
    nmb_w_duration = mo.sql(
        f"""
        -- Compute duration (minutes) between consecutive NMB administrations
        FROM nmb_df
        SELECT hospitalization_id
            , admin_dttm
            , med_dose
            , _duration_min: EXTRACT(EPOCH FROM (
                LEAD(admin_dttm, 1, admin_dttm) OVER w - admin_dttm
              )) / 60.0
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY admin_dttm)
        """
    )
    return (nmb_w_duration,)


@app.cell
def _(cohort_hrly_grids_f, nmb_w_duration):
    nmb_hrly = mo.sql(
        f"""
        -- ASOF join NMB to hourly grid to get NMB status at each hour
        FROM cohort_hrly_grids_f g
        ASOF LEFT JOIN nmb_w_duration n
            ON g.hospitalization_id = n.hospitalization_id
            AND n.admin_dttm <= g.event_dttm
        SELECT g.hospitalization_id
            , g._nth_day
            , n.med_dose
            , n._duration_min
        """
    )
    return (nmb_hrly,)


@app.cell
def _(nmb_hrly):
    nmb_excluded_patient_days = mo.sql(
        f"""
        -- Flag patient-days with >1 hour total NMB for exclusion
        FROM nmb_hrly
        SELECT hospitalization_id, _nth_day
            , _nmb_total_min: SUM(CASE WHEN med_dose > 0 THEN _duration_min ELSE 0 END)
        GROUP BY hospitalization_id, _nth_day
        HAVING _nmb_total_min > 60
        """
    )
    return (nmb_excluded_patient_days,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## CONSORT Flow & Save Outputs
    """)
    return


@app.cell
def _(
    SITE_NAME,
    cohort_hosp_ids,
    cohort_hrly_grids_f,
    cohort_imv_streaks,
    consort_counts,
    icu_type_df,
    nmb_excluded_patient_days,
):
    import json

    # CONSORT flow
    _nmb_excluded_df = nmb_excluded_patient_days.df()
    _n_nmb_patient_days = len(_nmb_excluded_df)
    _n_nmb_hosp_ids = _nmb_excluded_df['hospitalization_id'].nunique() if len(_nmb_excluded_df) > 0 else 0
    _cc = consort_counts

    # Weight-QC exclusion already applied (or skipped if drop list missing) in
    # the cohort_hosp_ids cell above. The CONSORT step here just reports the
    # totals from consort_counts.
    _weight_breakdown_str = (
        ", ".join(f"{k}: {v}" for k, v in _cc['weight_drop_breakdown'].items())
        if _cc['weight_drop_breakdown'] else "n/a — drop list not present at run time"
    )

    consort_flow = {
        "site": SITE_NAME,
        "steps": [
            {
                "step": 1,
                "description": "Hospitalizations with ICU stays",
                "n_remaining": _cc['n_icu'],
            },
            {
                "step": 2,
                "description": "With any invasive mechanical ventilation",
                "n_remaining": _cc['n_any_imv'],
                "n_excluded": _cc['n_no_imv'],
                "exclusion_reason": "No IMV recorded",
            },
            {
                "step": 3,
                "description": "First IMV streak >= 24 hours",
                "n_remaining": _cc['n_first_imv_ge24'],
                "n_excluded": _cc['n_first_imv_lt24'],
                "exclusion_reason": f"First IMV streak <24h ({_cc['n_trach_truncated']} tracheostomy-truncated)",
            },
            {
                "step": 4,
                "description": "Pass weight-QC (zero-weight, jump rule)",
                "n_remaining": _cc['n_post_weight'],
                "n_excluded": _cc['n_weight_excluded'],
                "exclusion_reason": f"Weight-QC drop list ({_weight_breakdown_str})",
                "weight_qc_breakdown": _cc['weight_drop_breakdown'],
            },
            {
                "step": 5,
                "description": "Exclude hospitalizations with any NMB >1h",
                "n_remaining": _cc['n_post_weight'] - _n_nmb_hosp_ids,
                "n_excluded": _n_nmb_hosp_ids,
                "exclusion_reason": f"Any patient-day with NMB >1h ({_n_nmb_patient_days:,} patient-days across {_n_nmb_hosp_ids:,} hosp)",
            },
        ],
    }

    _consort_json_path = f"output_to_share/{SITE_NAME}/consort_inclusion.json"
    with open(_consort_json_path, "w") as f:
        json.dump(consort_flow, f, indent=2)
    print(f"CONSORT flow saved to {_consort_json_path}")

    # Save intermediate outputs (site-scoped: output/{site}/...).
    # cohort_imv_streaks and cohort_hrly_grids_f are filtered to the
    # post-weight-QC cohort so every downstream consumer (02 / 04 / 05 / qc)
    # only sees kept patients. The pre-weight-QC streaks/grid are still in
    # memory (used by NMB exclusion above) but not persisted.
    _kept_set = set(cohort_hosp_ids)
    _streaks_df_full = cohort_imv_streaks.df()
    _streaks_df_kept = _streaks_df_full[
        _streaks_df_full['hospitalization_id'].isin(_kept_set)
    ]
    _grids_df_kept = cohort_hrly_grids_f[
        cohort_hrly_grids_f['hospitalization_id'].isin(_kept_set)
    ]

    _streaks_df_kept.to_parquet(f"output/{SITE_NAME}/cohort_imv_streaks.parquet")
    _grids_df_kept.to_parquet(f"output/{SITE_NAME}/cohort_hrly_grids.parquet")
    _nmb_excluded_df.to_parquet(f"output/{SITE_NAME}/nmb_excluded.parquet")
    icu_type_df.df().to_parquet(f"output/{SITE_NAME}/icu_type.parquet")

    print(f"Saved: output/{SITE_NAME}/cohort_imv_streaks.parquet ({len(cohort_hosp_ids)} hospitalizations)")
    print(f"Saved: output/{SITE_NAME}/cohort_hrly_grids.parquet ({len(_grids_df_kept)} rows)")
    print(f"Saved: output/{SITE_NAME}/nmb_excluded.parquet ({_n_nmb_patient_days} patient-days)")
    print(f"Saved: output/{SITE_NAME}/icu_type.parquet ({len(icu_type_df)} hospitalizations)")

    # CONSORT flowchart PNG
    from _utils import plot_consort
    _consort_png_path = f"output_to_share/{SITE_NAME}/consort_inclusion.png"
    plot_consort(consort_flow, _consort_png_path)
    print(f"Saved: {_consort_png_path}")
    return (consort_flow,)


@app.cell
def _(consort_flow):
    from _utils import consort_to_markdown
    mo.md(f"""
    ## CONSORT Flow

    {consort_to_markdown(consort_flow)}
    """)
    return


if __name__ == "__main__":
    app.run()
