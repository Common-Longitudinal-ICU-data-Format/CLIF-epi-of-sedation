# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "clifpy>=0.3.1",
#     "duckdb>=1.4.1",
#     "polars>=1.34.0",
# ]
# ///

import marimo

__generated_with = "0.22.5"
app = marimo.App(sql_output="native")

with app.setup:
    import marimo as mo
    import os
    import sys
    import logging
    import polars as pl
    from pathlib import Path
    # sys.path.insert(0, str(Path(__file__).parent))
    RERUN_WATERFALL = False
    # Hoisted into setup so any cell can call them without re-importing
    # (marimo flags duplicate top-level imports across cells as
    # cross-cell shadowing). retag_to_local_tz is metadata-only tz_convert
    # at parquet boundaries; add_day_shift_id is the SQL-based local-hour
    # derivation (tested in tests/test_timezone.py); plot_consort and
    # consort_to_markdown render the CONSORT artifacts.
    from _utils import (
        retag_to_local_tz,
        add_day_shift_id,
        plot_consort,
        consort_to_markdown,
    )
    from clifpy.utils.logging_config import get_logger
    logger = get_logger("epi_sedation.cohort")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 01 Cohort Identification

    Identifies ICU patients with first IMV streak >= 24 hours.
    Builds hourly time grids with day/shift annotations.
    Computes NMB exclusion flags.

    Pipeline follows `pyCLIF/docs/duckdb_perf_guide.md` §11: stay lazy as
    `DuckDBPyRelation`s, materialize only at framework boundaries (waterfall,
    parquet writes for tz-tagged outputs). Vendored DuckDB outlier handler
    (`code/_outlier_handler.py`) replaces clifpy's pandas-based
    `apply_outlier_handling` so the resp chain stays unmaterialized through
    the outlier step.

    Cohort definition (per-hospitalization, first IMV streak ≥24h with at-
    least-one-ICU-stay eligibility) is preserved unchanged from prior
    versions. Encounter-stitching collapse and patient-level filter were
    explored and deferred — see `docs/analysis_plan.md` "Cohort definition
    — alternatives considered" for the rationale + empirical numbers.
    """)
    return


@app.cell
def _():
    from clifpy import load_data, setup_logging
    import duckdb
    from clifpy.utils.config import get_config_or_params
    from _outlier_handler import apply_outlier_handling_duckdb

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"
    return (
        CONFIG_PATH,
        apply_outlier_handling_duckdb,
        duckdb,
        get_config_or_params,
        load_data,
        setup_logging,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params, setup_logging):
    # Site-scoped output directories — every per-site run lives under
    # output/{site}/ and output_to_share/{site}/ so multiple sites coexist
    # on disk (see Makefile's SITE= flag and the "Multi-site support"
    # section in .dev/CLAUDE.md).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    # Path B++ refactor: descriptive (night-vs-day) artifacts under
    # {site}/descriptive/, modeling-cohort artifacts (incl. CONSORT) under
    # {site}/models/. Both flat — no nested figures/.
    os.makedirs(f"output_to_share/{SITE_NAME}/descriptive", exist_ok=True)
    os.makedirs(f"output_to_share/{SITE_NAME}/models", exist_ok=True)
    # Per-site log separation: logs land at output/{site}/logs/clifpy_all.log
    # + clifpy_errors.log. setup_logging is idempotent; we call it explicitly
    # here (replaces the ClifOrchestrator instantiation side effect that the
    # prior version relied on) so the output_directory is site-scoped.
    setup_logging(output_directory=f"output/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return SITE_NAME, SITE_TZ


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load ADT
    """)
    return


@app.cell
def _(duckdb, load_data):
    # Lazy ADT load with column + ICU-category pushdown into the parquet scan.
    adt_rel = load_data(
        'adt',
        config_path='config/config.json',
        return_rel=True,
        columns=['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category', 'location_type'],
        filters={'location_category': ['icu']},
    )
    hosp_ids_w_icu_stays = [
        r[0] for r in duckdb.sql(
            "FROM adt_rel SELECT DISTINCT hospitalization_id"
        ).fetchall()
    ]
    logger.info(f"Hospitalizations with ICU stays: {len(hosp_ids_w_icu_stays):,}")
    return adt_rel, hosp_ids_w_icu_stays


@app.cell
def _(adt_rel):
    adt_rel
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
    SITE_TZ,
    apply_outlier_handling_duckdb,
    hosp_ids_w_icu_stays,
    load_data,
):
    from clifpy import RespiratorySupport

    resp_processed_path = f"output/{SITE_NAME}/resp_processed_bf.parquet"

    if not os.path.exists(resp_processed_path) or RERUN_WATERFALL:
        # Load via clifpy.utils.io.load_data with site_tz="" to skip the
        # default UTC→site_tz conversion. clifpy's standard `from_file` path
        # would convert to site_tz via DuckDB's `timezone(site_tz, col)`,
        # which RETURNS A NAIVE TIMESTAMP (no tz metadata) — leaving us with
        # tz-naive Central wall-clock values. clifpy's waterfall scaffold
        # builder then forces tz-aware UTC, the dtype mismatch silently
        # downgrades recorded_dttm to object after concat, and pandas 2.3+
        # raises a strict-Categorical TypeError on the subsequent sort_values.
        # Bypass that path: load raw UTC (tz-aware) directly so the waterfall
        # gets compatible dtypes throughout.
        _resp_columns = [
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
        ]
        import json as _json
        with open(CONFIG_PATH) as _cfg_f:
            _cfg = _json.load(_cfg_f)
        # Lazy load + vendored DuckDB outlier handler (per duckdb_perf_guide
        # §11.1 — replaces clifpy's pandas-based apply_outlier_handling).
        _resp_rel = load_data(
            "respiratory_support",
            config_path=CONFIG_PATH,
            return_rel=True,
            columns=_resp_columns,
            filters={"hospitalization_id": hosp_ids_w_icu_stays},
            site_tz="",  # skip auto-conversion; values stay UTC tz-aware
        )
        _resp_rel = apply_outlier_handling_duckdb(
            _resp_rel, 'respiratory_support', 'config/outlier_config.yaml',
        )
        # Materialize at the waterfall boundary — RespiratorySupport.waterfall
        # is Table-bound (operates on .df, pandas). Vendoring waterfall is
        # out of scope.
        cohort_resp = RespiratorySupport(
            data_directory=_cfg["data_directory"],
            filetype=_cfg.get("filetype", "parquet"),
            timezone="UTC",
            data=_resp_rel.df(),
        )
        cohort_resp_p = cohort_resp.waterfall(bfill=True)
        # Re-tag UTC tz-aware recorded_dttm → site-local tz-aware before
        # persisting. Pre-waterfall data must stay UTC (waterfall reentry
        # compat); the retag at the parquet-write boundary is the project
        # convention for `output/{site}/*.parquet`. See _utils.py for the
        # helper and tests/test_timezone.py for invariants.
        cohort_resp_p.df = retag_to_local_tz(
            cohort_resp_p.df, ["recorded_dttm"], SITE_TZ
        )
        cohort_resp_p.df.to_parquet(resp_processed_path)
        resp_p = cohort_resp_p.df
    else:
        # Cached load via Polars to preserve the on-disk site-tz tag (pyarrow
        # round-trips the tag through parquet metadata; DuckDB's `.df()` would
        # rewrite the tag to session tz). `.to_pandas()` keeps the tz info.
        logger.info(f"Loading cached {resp_processed_path}")
        resp_p = pl.read_parquet(resp_processed_path).to_pandas()

    logger.info(f"resp_p: {len(resp_p):,} rows")
    return (resp_p,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## IMV Streak Detection

    Inlined from `cohort_id.sql`. Each CTE is a separate cell for interactive
    inspection (per the "one CTE per cell" pattern in the global CLAUDE.md).
    The `tracheostomy` NaN→0 coercion is folded into cohort_t1's SELECT via
    `COALESCE(...)::INT` rather than mutating `resp_p` in pandas first.
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
            , tracheostomy: COALESCE(tracheostomy, 0)::INT
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
def _(
    SITE_NAME,
    all_streaks_w_lead,
    cohort_imv_streaks,
    duckdb,
    hosp_ids_w_icu_stays,
):
    # Pre-weight cohort = qualifying IMV streaks (one row per qualifying hosp).
    # Pulled via DuckDB SQL to keep the streak relation lazy; only the small
    # ID list materializes here.
    cohort_hosp_ids_pre_weight = [
        r[0] for r in duckdb.sql(
            "FROM cohort_imv_streaks SELECT DISTINCT hospitalization_id"
        ).fetchall()
    ]

    # Intermediate CONSORT counts via scalar SQL queries on the lazy
    # all_streaks_w_lead relation (per duckdb_perf_guide §11.6 — scalar
    # diagnostics without breaking the lazy DAG).
    _n_with_any_imv = duckdb.sql("""
        FROM all_streaks_w_lead
        SELECT COUNT(DISTINCT hospitalization_id)
        WHERE _on_imv = 1
    """).fetchone()[0]
    _n_first_imv_lt24 = duckdb.sql("""
        FROM all_streaks_w_lead
        SELECT COUNT(*)
        WHERE _streak_id = 1 AND _on_imv = 1 AND _at_least_24h = 0
    """).fetchone()[0]
    _n_trach_truncated = duckdb.sql("""
        FROM all_streaks_w_lead
        SELECT COUNT(*)
        WHERE _streak_id = 1 AND _on_imv = 1 AND _at_least_24h = 0
            AND _trach_dttm IS NOT NULL
    """).fetchone()[0]

    # Weight-QC exclusion (Phase 2 of weight-audit work). Reads the drop list
    # produced by `make weight-audit SITE=<site>`. If the audit hasn't run
    # yet, the file won't exist and we skip the exclusion — first-run pipelines
    # complete without weight QC, then the user runs the audit, then re-runs
    # this script to apply the drop. See code/qc/weight_audit_README.md.
    _drop_path = Path(f"output/{SITE_NAME}/qc/weight_qc_drop_list.parquet")
    if _drop_path.exists():
        _drop_df = pl.read_parquet(_drop_path)
        _weight_drop_set = set(
            _drop_df.select(pl.col('hospitalization_id').cast(pl.Utf8)).to_series().to_list()
        )
        _vc = _drop_df['_drop_reason'].value_counts(sort=True)
        weight_drop_breakdown = dict(
            zip(_vc['_drop_reason'].to_list(), _vc['count'].to_list())
        )
        cohort_hosp_ids = [
            h for h in cohort_hosp_ids_pre_weight
            if str(h) not in _weight_drop_set
        ]
        _n_weight_excluded = len(cohort_hosp_ids_pre_weight) - len(cohort_hosp_ids)
        logger.info(f"Weight-QC exclusion: {_n_weight_excluded:,} hospitalizations dropped")
        for _reason, _n_for_reason in weight_drop_breakdown.items():
            logger.info(f"  {_reason}: {_n_for_reason:,}")
    else:
        weight_drop_breakdown = {}
        cohort_hosp_ids = list(cohort_hosp_ids_pre_weight)
        _n_weight_excluded = 0
        logger.warning(f"No weight-QC drop list at {_drop_path} — skipping weight QC")
        logger.warning(f"  (run `make weight-audit SITE={SITE_NAME}` then re-run 01_cohort.py to apply)")

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
    logger.info(f"Cohort hospitalizations (first IMV ≥24h, post-weight-QC): {len(cohort_hosp_ids):,}")
    logger.info(f"  Excluded — no IMV: {consort_counts['n_no_imv']:,}")
    logger.info(
        f"  Excluded — first IMV <24h: {consort_counts['n_first_imv_lt24']:,} "
        f"(of which {consort_counts['n_trach_truncated']:,} tracheostomy-truncated)"
    )
    logger.info(f"  Excluded — weight-QC: {_n_weight_excluded:,}")
    return cohort_hosp_ids, consort_counts


@app.cell
def _(adt_rel, cohort_hosp_ids):
    icu_type_df = mo.sql(
        f"""
        -- First ICU type per cohort hospitalization (earliest ADT record)
        FROM adt_rel
        SELECT hospitalization_id
            , icu_type: FIRST(location_type ORDER BY in_dttm)
        WHERE hospitalization_id IN (SELECT UNNEST({cohort_hosp_ids}))
        GROUP BY hospitalization_id
        """
    )
    return (icu_type_df,)


@app.cell
def _(duckdb, icu_type_df):
    _n_icu_types = duckdb.sql("FROM icu_type_df SELECT COUNT(*)").fetchone()[0]
    logger.info(f"ICU types extracted: {_n_icu_types:,} hospitalizations")
    return


@app.cell
def _(cohort_imv_streaks):
    cohort_imv_streaks
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly Time Grids

    Per the "one CTE per cell" pattern: bound computation is split off so
    the intermediate is inspectable in the marimo UI. The two cells fuse
    into one execution at the terminal `.df()` / parquet write — zero perf
    overhead in production.
    """)
    return


@app.cell
def _(cohort_imv_streaks):
    cohort_streak_bounds = mo.sql(
        f"""
        -- Per-streak hour bounds (rounded to the hour boundary, end exclusive +1h).
        FROM cohort_imv_streaks
        SELECT hospitalization_id
            , _start_hr: date_trunc('hour', _start_dttm)
            , _end_hr: date_trunc('hour', _end_dttm) + INTERVAL '1 hour'
        """
    )
    return (cohort_streak_bounds,)


@app.cell
def _(cohort_streak_bounds):
    cohort_hrly_grids = mo.sql(
        f"""
        -- Generate hourly time grid from start to end of each qualifying IMV streak
        FROM cohort_streak_bounds
        SELECT hospitalization_id
            , unnest(generate_series(_start_hr, _end_hr, INTERVAL '1 hour')) AS event_dttm
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cohort_hrly_grids,)


@app.cell
def _(SITE_TZ, cohort_hrly_grids):
    # add_day_shift_id derives _dh / _hr / _shift / _is_day_start / _nth_day /
    # _day_shift via explicit `AT TIME ZONE site_tz` in SQL — session-tz
    # invariant (see tests/test_timezone.py for invariants and DST coverage).
    # The function returns pandas; downstream consumers via DuckDB
    # replacement scan handle pandas natively.
    _grids_df = cohort_hrly_grids.df()
    cohort_hrly_grids_f = add_day_shift_id(_grids_df, site_tz=SITE_TZ)
    assert len(cohort_hrly_grids_f) == len(_grids_df), 'length altered by add_day_shift_id'
    logger.info(f"Hourly grid rows: {len(cohort_hrly_grids_f):,}")
    return (cohort_hrly_grids_f,)


@app.cell
def _(cohort_hrly_grids_f):
    cohort_hrly_grids_f
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
def _(load_data):
    nmb_rel = load_data(
        'medication_admin_continuous',
        config_path='config/config.json',
        return_rel=True,
        columns=['hospitalization_id', 'admin_dttm', 'med_name', 'med_category', 'med_dose', 'med_dose_unit'],
        filters={'med_category': ['cisatracurium', 'vecuronium', 'rocuronium']},
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = nmb_rel.count("*").fetchone()[0]
        logger.debug(f"NMB records: {_n:,}")
    else:
        logger.info("NMB records: lazy relation built")
    return (nmb_rel,)


@app.cell
def _(nmb_rel):
    nmb_w_duration = mo.sql(
        f"""
        -- Compute duration (minutes) between consecutive NMB administrations
        FROM nmb_rel
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

    Terminal materialization. Tz-tagged outputs (`cohort_imv_streaks` with
    `_start_dttm`/`_end_dttm`, `cohort_hrly_grids` with `event_dttm`) write
    via Polars `dt.convert_time_zone` + `write_parquet` to preserve the
    site-local tz tag on disk. Non-tz outputs (`nmb_excluded`, `icu_type`)
    use DuckDB's native `.to_parquet()` directly. Mirrors the 02_exposure
    save pattern.
    """)
    return


@app.cell
def _(
    SITE_NAME,
    SITE_TZ,
    cohort_hosp_ids,
    cohort_hrly_grids_f,
    cohort_imv_streaks,
    consort_counts,
    duckdb,
    icu_type_df,
    nmb_excluded_patient_days,
):
    import json

    # NMB exclusion summary — scalar SQL on the lazy relation.
    _n_nmb_patient_days = duckdb.sql(
        "FROM nmb_excluded_patient_days SELECT COUNT(*)"
    ).fetchone()[0]
    _n_nmb_hosp_ids = duckdb.sql(
        "FROM nmb_excluded_patient_days SELECT COUNT(DISTINCT hospitalization_id)"
    ).fetchone()[0]
    _cc = consort_counts

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

    _consort_json_path = f"output_to_share/{SITE_NAME}/models/consort_inclusion.json"
    with open(_consort_json_path, "w") as f:
        json.dump(consort_flow, f, indent=2)
    logger.info(f"CONSORT flow saved to {_consort_json_path}")

    # Filter cohort artifacts to the kept (post-weight-QC) cohort. SEMI JOIN
    # in DuckDB is the lazy way; results materialize at .pl() / .to_parquet().
    _streaks_kept_rel = duckdb.sql(f"""
        FROM cohort_imv_streaks
        WHERE hospitalization_id IN (SELECT UNNEST({cohort_hosp_ids}))
    """)
    _grids_kept_rel = duckdb.sql(f"""
        FROM cohort_hrly_grids_f
        WHERE hospitalization_id IN (SELECT UNNEST({cohort_hosp_ids}))
    """)

    # Tz-tagged outputs via Polars (preserves site-local tz tag on disk;
    # DuckDB native .to_parquet would normalize to UTC tag — see
    # duckdb_perf_guide.md §11.4).
    (
        _streaks_kept_rel
        .pl()
        .with_columns(
            pl.col('_start_dttm').dt.convert_time_zone(SITE_TZ),
            pl.col('_end_dttm').dt.convert_time_zone(SITE_TZ),
        )
        .write_parquet(f"output/{SITE_NAME}/cohort_imv_streaks.parquet")
    )
    (
        _grids_kept_rel
        .pl()
        .with_columns(pl.col('event_dttm').dt.convert_time_zone(SITE_TZ))
        .write_parquet(f"output/{SITE_NAME}/cohort_hrly_grids.parquet")
    )

    # Non-tz outputs: DuckDB native .to_parquet() directly on the relation.
    nmb_excluded_patient_days.to_parquet(f"output/{SITE_NAME}/nmb_excluded.parquet")
    icu_type_df.to_parquet(f"output/{SITE_NAME}/icu_type.parquet")

    _n_grids_rows = _grids_kept_rel.count("*").fetchone()[0]
    _n_icu = icu_type_df.count("*").fetchone()[0]
    logger.info(
        f"Saved: output/{SITE_NAME}/cohort_imv_streaks.parquet "
        f"({len(cohort_hosp_ids):,} hospitalizations)"
    )
    logger.info(f"Saved: output/{SITE_NAME}/cohort_hrly_grids.parquet ({_n_grids_rows:,} rows)")
    logger.info(f"Saved: output/{SITE_NAME}/nmb_excluded.parquet ({_n_nmb_patient_days:,} patient-days)")
    logger.info(f"Saved: output/{SITE_NAME}/icu_type.parquet ({_n_icu:,} hospitalizations)")

    # CONSORT flowchart PNG
    _consort_png_path = f"output_to_share/{SITE_NAME}/models/consort_inclusion.png"
    plot_consort(consort_flow, _consort_png_path)
    logger.info(f"Saved: {_consort_png_path}")
    return (consort_flow,)


@app.cell
def _(consort_flow):
    mo.md(f"""
    ## CONSORT Flow

    {consort_to_markdown(consort_flow)}
    """)
    return


if __name__ == "__main__":
    app.run()
