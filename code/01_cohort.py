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
    # cross-cell shadowing). to_utc is the canonical tz-normalization helper
    # used at every parquet-write boundary; add_day_shift_id is the
    # SQL-based local-hour derivation (tested in tests/test_timezone.py);
    # plot_consort and consort_to_markdown render the CONSORT artifacts.
    from _utils import (
        to_utc,
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
    # hospital_id is included so the same relation can feed both ICU filtering
    # and clifpy's stitch_encounters (which requires it in adt) without a
    # second parquet scan.
    adt_rel = load_data(
        'adt',
        config_path='config/config.json',
        return_rel=True,
        columns=['hospitalization_id', 'hospital_id', 'in_dttm', 'out_dttm', 'location_category', 'location_type'],
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
    ## Encounter-Stitch Dedup Filter

    Drop within-12h discharge→readmit paperwork artifacts so downstream "first
    IMV streak" reflects a genuinely independent ICU episode. Keep only the
    first hospitalization (by `admission_dttm`) per stitched encounter block.

    `encounter_block` is consumed-and-discarded inside this cell —
    `hospitalization_id` stays the cohort key, so downstream cells see exactly
    the same schema as before. Runtime toggle:
    `COHORT_STITCH_DEDUP_ON=0` short-circuits to identity passthrough; CONSORT
    step 2 is then omitted via the `n_dropped_stitch is None` sentinel
    (mirrors the conditional range-rule step at step 7+).
    """)
    return


@app.cell
def _(adt_rel, hosp_ids_w_icu_stays, load_data):
    # Encounter-stitch DEDUP filter — see markdown above. Toggle:
    # COHORT_STITCH_DEDUP_ON=0 short-circuits to identity passthrough
    # (n_dropped_stitch=None → CONSORT step 2 is omitted). Default ON.
    _stitch_on = os.environ.get('COHORT_STITCH_DEDUP_ON', '1') == '1'

    if not _stitch_on:
        cohort_hosp_ids_post_stitch = list(hosp_ids_w_icu_stays)
        n_dropped_stitch = None  # sentinel — CONSORT skips step 2
        stitch_dropped_hosp_ids: list = []
        stitch_block_sizes = None  # sentinel — QA cell skips
        n_unmapped_singletons = 0
        # encounter_mapping = None propagates as the OFF sentinel — the
        # downstream `cohort_meta_by_id_imvday` builder treats this as "no
        # mapping" and emits encounter_block = NULL on every row.
        encounter_mapping = None
        logger.info("Encounter stitch-dedup: OFF (COHORT_STITCH_DEDUP_ON=0)")
    else:
        from clifpy.utils.stitching_encounters import stitch_encounters

        _hosp_rel = load_data(
            'hospitalization',
            config_path='config/config.json',
            return_rel=True,
            columns=[
                'patient_id', 'hospitalization_id', 'admission_dttm',
                'discharge_dttm', 'age_at_admission',
                'admission_type_category', 'discharge_category',
            ],
            filters={'hospitalization_id': hosp_ids_w_icu_stays},
        )
        _hosp_df = _hosp_rel.df()
        _adt_df = adt_rel.df()

        # stitch_encounters is pandas-only — clifpy I/O bridge. Returns
        # (hosp_stitched, adt_stitched, encounter_mapping); only the mapping
        # is needed (cols: hospitalization_id, encounter_block).
        _, _, _encounter_mapping = stitch_encounters(
            _hosp_df, _adt_df, time_interval=12,
        )

        # CRITICAL: clifpy's stitch_encounters can silently drop hospitalizations
        # from the returned encounter_mapping (e.g., when admission_dttm /
        # discharge_dttm have null/quirky values that confuse the propagation
        # step). These hosps are NOT stitched to anything — they should be
        # KEPT as singletons by the dedup filter, not dropped along with the
        # genuinely-stitched-second-or-later hosps. Surface the missing count
        # separately in QA so federated sites can investigate clifpy data
        # quality without it confounding the cohort-attrition story.
        _mapped_ids = set(_encounter_mapping['hospitalization_id'])
        _unmapped_ids = [h for h in hosp_ids_w_icu_stays if h not in _mapped_ids]
        n_unmapped_singletons = len(_unmapped_ids)

        # For each MAPPED hosp, pick the first by admission_dttm in its
        # encounter block. Then re-add unmapped hosps as singletons.
        _merged = _encounter_mapping.merge(
            _hosp_df[['hospitalization_id', 'admission_dttm']],
            on='hospitalization_id',
        )
        _first_per_block = (
            _merged
            .sort_values(['encounter_block', 'admission_dttm'])
            .drop_duplicates('encounter_block', keep='first')
        )
        _kept_from_mapping = _first_per_block['hospitalization_id'].tolist()
        cohort_hosp_ids_post_stitch = _kept_from_mapping + _unmapped_ids
        n_dropped_stitch = (
            len(hosp_ids_w_icu_stays) - len(cohort_hosp_ids_post_stitch)
        )

        # Surface the dropped hosp_ids list + block-size series for the
        # QA-summary cell that follows. Both are cross-cell DAG outputs (no
        # underscore prefix). IDs stay in-memory only — never persisted.
        _kept_set = set(cohort_hosp_ids_post_stitch)
        stitch_dropped_hosp_ids = [
            h for h in hosp_ids_w_icu_stays if h not in _kept_set
        ]
        stitch_block_sizes = _encounter_mapping.groupby('encounter_block').size()

        # Surface the (hospitalization_id, encounter_block) mapping as a
        # cross-cell variable so downstream cells (notably the
        # `cohort_meta_by_id_imvday` / `cohort_meta_by_id` builders) can
        # propagate `encounter_block` without re-running stitch_encounters.
        # Restrict to the two ID columns to keep memory footprint minimal.
        encounter_mapping = _encounter_mapping[
            ['hospitalization_id', 'encounter_block']
        ].copy()

        logger.info(
            f"Encounter stitch-dedup (12h window): {n_dropped_stitch:,} "
            f"hospitalizations stitched out (kept first per encounter block; "
            f"{len(cohort_hosp_ids_post_stitch):,} remain). "
            f"Unmapped (kept as singletons): {n_unmapped_singletons:,}"
        )

    return (
        cohort_hosp_ids_post_stitch,
        encounter_mapping,
        n_dropped_stitch,
        n_unmapped_singletons,
        stitch_dropped_hosp_ids,
        stitch_block_sizes,
    )


@app.cell
def _(
    SITE_NAME,
    load_data,
    n_dropped_stitch,
    n_unmapped_singletons,
    stitch_block_sizes,
    stitch_dropped_hosp_ids,
):
    # Stitch QA summary — persists per-site stitch behavior + cohort-impact
    # decomposition for cross-site QA pooling. Written to
    # output_to_share/{site}/qc/stitch_summary.json. Skipped when toggle is
    # OFF (no stitch behavior to characterize). No IDs surfaced — only
    # group-level counts per CLIF data privacy rules.
    if n_dropped_stitch is None:
        logger.info("Stitch QA summary: skipped (COHORT_STITCH_DEDUP_ON=0)")
    else:
        import json as _json

        os.makedirs(f"output_to_share/{SITE_NAME}/qc", exist_ok=True)

        # Block-size distribution from encounter_mapping. Group-level only.
        _n_blocks_total = int(len(stitch_block_sizes))
        _n_blocks_singleton = int((stitch_block_sizes == 1).sum())
        _n_blocks_size_2 = int((stitch_block_sizes == 2).sum())
        _n_blocks_size_ge3 = int((stitch_block_sizes >= 3).sum())
        _bs_describe = {
            k: float(v) for k, v in stitch_block_sizes.describe().to_dict().items()
        }

        # Decompose dropped hosps by IMV status. Coarse heuristic for the
        # ≥24h split: max(recorded_dttm) - min(recorded_dttm) on IMV-tagged
        # rows ≥ 24h. This is an UPPER BOUND on actual IMV duration (no gap
        # / restart handling). Empirically at sites with reliable charting
        # (UCMC/MIMIC) n_dropped_with_imv = 0 so the split is moot; the
        # finer ≥24h split is included to surface anomalies at federated
        # sites where the bucket may be non-trivial.
        #
        # Pandas (not DuckDB) for the aggregation: the dropped-hosp set is
        # tiny (60-150 hosps), and DuckDB's replacement scan doesn't resolve
        # underscore-prefixed cell-local relations, so we materialize the
        # filtered relation to pandas and aggregate there.
        if len(stitch_dropped_hosp_ids) > 0:
            _resp_dropped_df = load_data(
                'respiratory_support',
                config_path='config/config.json',
                return_rel=True,
                columns=['hospitalization_id', 'recorded_dttm', 'device_category'],
                filters={'hospitalization_id': stitch_dropped_hosp_ids},
            ).df()
            _resp_imv = _resp_dropped_df[
                _resp_dropped_df['device_category'] == 'imv'
            ]
            _imv_per_hosp = _resp_imv.groupby('hospitalization_id').agg(
                _imv_min=('recorded_dttm', 'min'),
                _imv_max=('recorded_dttm', 'max'),
            )
            _imv_per_hosp['_imv_dur_hr'] = (
                (_imv_per_hosp['_imv_max'] - _imv_per_hosp['_imv_min'])
                .dt.total_seconds() / 3600
            )
            _n_with_imv = int(len(_imv_per_hosp))
            _n_imv_ge24h = int((_imv_per_hosp['_imv_dur_hr'] >= 24).sum())
            _n_imv_lt24h = _n_with_imv - _n_imv_ge24h
            _n_no_imv_total = len(stitch_dropped_hosp_ids) - _n_with_imv
        else:
            _n_imv_ge24h = 0
            _n_imv_lt24h = 0
            _n_no_imv_total = 0

        # Auto-generated interpretation. Triggers a clearer warning string
        # if any qualifying-IMV hosps were stitched out (the only scenario
        # where the filter has cohort cost).
        if _n_imv_ge24h == 0:
            _interp = (
                f"All {n_dropped_stitch} stitched-out hospitalizations had "
                f"either no IMV records ({_n_no_imv_total}) or IMV <24h "
                f"({_n_imv_lt24h}). Net effect on final analytic cohort: 0."
            )
        else:
            _interp = (
                f"{_n_imv_ge24h} of {n_dropped_stitch} stitched-out "
                f"hospitalizations had IMV >=24h. These would have qualified "
                f"for the analytic cohort but were dropped as paperwork-"
                f"artifact re-admissions. INVESTIGATE before pooling."
            )

        _pre_stitch = (
            _n_blocks_total + n_dropped_stitch + n_unmapped_singletons
        )
        _post_stitch = _n_blocks_total + n_unmapped_singletons
        _summary = {
            "site": SITE_NAME,
            "time_interval_hours": 12,
            "toggle_state": "ON",
            "cohort_size": {
                "pre_stitch": _pre_stitch,
                "post_stitch": _post_stitch,
                "n_dropped": n_dropped_stitch,
            },
            "block_size_distribution": {
                "n_blocks_total": _n_blocks_total,
                "n_blocks_singleton": _n_blocks_singleton,
                "n_blocks_size_2": _n_blocks_size_2,
                "n_blocks_size_ge3": _n_blocks_size_ge3,
                "describe": _bs_describe,
                "comment": (
                    "describes encounter_mapping returned by clifpy's "
                    "stitch_encounters; excludes unmapped hosps (see "
                    "clifpy_data_quality.n_unmapped_singletons)"
                ),
            },
            "clifpy_data_quality": {
                "n_unmapped_singletons": n_unmapped_singletons,
                "comment": (
                    "Hospitalizations missing from clifpy's returned "
                    "encounter_mapping (likely null/quirky admission_dttm "
                    "or discharge_dttm). Kept as singletons by the dedup "
                    "filter, not stitched. Investigate at federated sites "
                    "where this count is non-trivial relative to total ICU "
                    "stays — may indicate upstream date-parsing issues."
                ),
            },
            "dropped_hosp_decomposition": {
                "n_dropped_no_imv": _n_no_imv_total,
                "n_dropped_imv_lt24h": _n_imv_lt24h,
                "n_dropped_imv_ge24h": _n_imv_ge24h,
                "duration_method": (
                    "max(recorded_dttm) - min(recorded_dttm) on IMV rows; "
                    "coarse upper bound (no gap/restart handling)"
                ),
            },
            "interpretation": _interp,
        }

        _path = f"output_to_share/{SITE_NAME}/qc/stitch_summary.json"
        with open(_path, "w") as _f:
            _json.dump(_summary, _f, indent=2)
        logger.info(f"Saved: {_path}")
        logger.info(
            f"  Stitch QA: dropped={n_dropped_stitch:,} "
            f"(no_imv={_n_no_imv_total:,}, imv_lt24h={_n_imv_lt24h:,}, "
            f"imv_ge24h={_n_imv_ge24h:,}); "
            f"blocks={_n_blocks_total:,} (singleton={_n_blocks_singleton:,}, "
            f"size_2={_n_blocks_size_2:,}, size_ge3={_n_blocks_size_ge3:,})"
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
    SITE_TZ,
    apply_outlier_handling_duckdb,
    cohort_hosp_ids_post_stitch,
    load_data,
):
    from clifpy import RespiratorySupport

    resp_processed_path = f"output/{SITE_NAME}/cohort_resp_processed_bf.parquet"

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
            filters={"hospitalization_id": cohort_hosp_ids_post_stitch},
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
        # Re-tag recorded_dttm to UTC display tag at the parquet-write
        # boundary (project convention: every *_dttm column on disk is
        # UTC tz-aware — see docs/timezone_audit.md). Same UTC instants;
        # tz_convert is metadata-only.
        cohort_resp_p.df = to_utc(cohort_resp_p.df, ["recorded_dttm"])
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
    all_streaks_w_lead,
    cohort_hosp_ids_post_stitch,
    cohort_imv_streaks,
    duckdb,
    hosp_ids_w_icu_stays,
    n_dropped_stitch,
):
    # Cell A — pre-weight cohort + CONSORT scalar counts.
    # Pulled via DuckDB SQL to keep the streak relation lazy; only the small
    # ID list materializes here. Per-criterion scalars come from independent
    # scalar SQL queries (per duckdb_perf_guide §11.6 — scalar diagnostics
    # without breaking the lazy DAG).
    cohort_hosp_ids_pre_weight = [
        r[0] for r in duckdb.sql(
            "FROM cohort_imv_streaks SELECT DISTINCT hospitalization_id"
        ).fetchall()
    ]

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

    # Re-anchor n_no_imv on n_post_stitch so the CONSORT cascade stays
    # additive (each step's n_excluded is a marginal exclusion against the
    # PRIOR step's n_remaining). When stitch is OFF, n_post_stitch == n_icu,
    # so the math reduces to the prior expression.
    _n_post_stitch = len(cohort_hosp_ids_post_stitch)
    cohort_pre_weight_counts = {
        'n_icu': len(hosp_ids_w_icu_stays),
        'n_post_stitch': _n_post_stitch,
        'n_dropped_stitch': n_dropped_stitch,  # None when stitch toggle OFF
        'n_any_imv': _n_with_any_imv,
        'n_no_imv': _n_post_stitch - _n_with_any_imv,
        'n_first_imv_ge24': len(cohort_hosp_ids_pre_weight),
        'n_first_imv_lt24': _n_first_imv_lt24,
        'n_trach_truncated': _n_trach_truncated,
    }
    return cohort_hosp_ids_pre_weight, cohort_pre_weight_counts


@app.cell
def _(cohort_hosp_ids_pre_weight, duckdb, load_data):
    # Cell B — weight-presence exclusion (upfront, runtime).
    # Drop hospitalizations with ZERO non-null weight_kg rows in vitals so the
    # first-pass cohort already excludes hosps that can never produce weight-
    # dependent dose conversions. The audit's value-quality checks (jump,
    # range) live separately in qc/weight_audit.py and feed back via
    # weight_qc_drop_list.parquet on pass 2 (next cell).
    weight_presence_rel = load_data(
        'vitals',
        config_path='config/config.json',
        return_rel=True,
        columns=['hospitalization_id', 'vital_category', 'vital_value'],
        filters={
            'vital_category': ['weight_kg'],
            'hospitalization_id': cohort_hosp_ids_pre_weight,
        },
    )
    cohort_hosp_ids_w_weight = [
        r[0] for r in duckdb.sql("""
            FROM weight_presence_rel
            SELECT DISTINCT hospitalization_id
            WHERE vital_value IS NOT NULL
        """).fetchall()
    ]
    n_dropped_no_weight = (
        len(cohort_hosp_ids_pre_weight) - len(cohort_hosp_ids_w_weight)
    )
    logger.info(
        f"Weight-presence exclusion: {n_dropped_no_weight:,} hospitalizations "
        f"dropped (zero weight_kg rows in vitals)"
    )
    return cohort_hosp_ids_w_weight, n_dropped_no_weight


@app.cell
def _(
    SITE_NAME,
    cohort_hosp_ids_w_weight,
    cohort_pre_weight_counts,
    n_dropped_no_weight,
):
    # Cell C — weight-QC drop list re-entry split per criterion.
    # Phase 2 of weight-audit: reads `weight_qc_drop_list.parquet` produced
    # by `make weight-audit SITE=<site>`, splits the drops by `_drop_reason`
    # prefix (zero_weight* / jump_* / range_*), and applies them
    # incrementally. Each criterion's marginal exclusion count flows into
    # CONSORT as its own step (steps 5-7), giving per-rule visibility.
    # If the file isn't present yet (first-pass run), counts are zero and
    # the existing cohort_hosp_ids_w_weight passes through unchanged.
    _drop_path = Path(f"output/{SITE_NAME}/qc/weight_qc_drop_list.parquet")
    if _drop_path.exists():
        _drop_df = pl.read_parquet(_drop_path)
        # Group drop ids by reason prefix (handles parameterized reason
        # strings like 'jump_gt_20kg_within_24h' / 'range_gt_30kg').
        _drop_by_kind = {'zero': set(), 'jump': set(), 'range': set()}
        for _row in _drop_df.iter_rows(named=True):
            _hid = str(_row['hospitalization_id'])
            _reason = _row['_drop_reason']
            if _reason.startswith('zero_weight'):
                _drop_by_kind['zero'].add(_hid)
            elif _reason.startswith('jump_'):
                _drop_by_kind['jump'].add(_hid)
            elif _reason.startswith('range_'):
                _drop_by_kind['range'].add(_hid)
            else:
                logger.warning(f"Unrecognized weight-QC drop reason: {_reason}")

        # Apply incrementally; each step's count is the marginal exclusion.
        _post_c1_resid = [
            h for h in cohort_hosp_ids_w_weight
            if str(h) not in _drop_by_kind['zero']
        ]
        _post_jump = [
            h for h in _post_c1_resid
            if str(h) not in _drop_by_kind['jump']
        ]
        _post_range = [
            h for h in _post_jump
            if str(h) not in _drop_by_kind['range']
        ]
        cohort_hosp_ids = _post_range

        n_excl_c1_residual = len(cohort_hosp_ids_w_weight) - len(_post_c1_resid)
        n_excl_jump = len(_post_c1_resid) - len(_post_jump)
        n_excl_range = len(_post_jump) - len(_post_range)

        # Surface the parameterized threshold strings so the CONSORT
        # exclusion_reason text shows the active threshold.
        _all_reasons = _drop_df['_drop_reason'].unique().to_list()
        jump_threshold_str = next(
            (r for r in _all_reasons if r.startswith('jump_')), None
        )
        range_threshold_str = next(
            (r for r in _all_reasons if r.startswith('range_')), None
        )

        logger.info(
            f"Weight-QC value checks: C1-residual={n_excl_c1_residual:,}, "
            f"jump={n_excl_jump:,}, range={n_excl_range:,}"
        )
    else:
        cohort_hosp_ids = list(cohort_hosp_ids_w_weight)
        n_excl_c1_residual = 0
        n_excl_jump = 0
        n_excl_range = 0
        jump_threshold_str = None
        range_threshold_str = None
        logger.warning(
            f"No weight-QC drop list at {_drop_path} — "
            "skipping C1-residual/jump/range"
        )
        logger.warning(
            f"  (run `make weight-audit SITE={SITE_NAME}` "
            "then re-run 01_cohort.py to apply)"
        )

    consort_counts = {
        **cohort_pre_weight_counts,
        'n_dropped_no_weight': n_dropped_no_weight,
        'n_post_weight_presence': len(cohort_hosp_ids_w_weight),
        'n_excl_c1_residual': n_excl_c1_residual,
        'n_post_c1_residual': len(cohort_hosp_ids_w_weight) - n_excl_c1_residual,
        'n_excl_jump': n_excl_jump,
        'n_post_jump': (
            len(cohort_hosp_ids_w_weight) - n_excl_c1_residual - n_excl_jump
        ),
        'n_excl_range': n_excl_range,
        'n_post_range': len(cohort_hosp_ids),
        'jump_threshold_str': jump_threshold_str,
        'range_threshold_str': range_threshold_str,
    }

    logger.info(
        f"Cohort hospitalizations (after all weight QC + IMV + ICU): "
        f"{len(cohort_hosp_ids):,}"
    )
    logger.info(f"  Excluded — no IMV: {consort_counts['n_no_imv']:,}")
    logger.info(
        f"  Excluded — first IMV <24h: {consort_counts['n_first_imv_lt24']:,} "
        f"(of which {consort_counts['n_trach_truncated']:,} tracheostomy-truncated)"
    )
    logger.info(f"  Excluded — no weight_kg row: {n_dropped_no_weight:,}")
    logger.info(f"  Excluded — all weights out of clamp: {n_excl_c1_residual:,}")
    logger.info(f"  Excluded — weight jump rule: {n_excl_jump:,}")
    logger.info(f"  Excluded — weight range rule: {n_excl_range:,}")
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
    ## Patient-Day Metadata Registry

    Build `cohort_meta_by_id_imvday` — the canonical per-day registry. One
    row per `(hospitalization_id, _nth_day)`. Pure cohort metadata: shift-
    hour counts, day boundary timestamps, derived `day_type` ∈
    {`first_partial`, `full`, `last_partial`}, and the canonical
    `_is_full_24h_day` flag (= `n_hrs_day == 12 AND n_hrs_night == 12`).
    No doses, no outcomes — those join in downstream.

    `encounter_block` is LEFT-JOINed in from the stitch cell's
    `encounter_mapping`; NULL when stitch is OFF or the hosp is an
    unmapped singleton (clifpy data-quality edge case).

    Saved at terminal-materialization time as
    `output/{site}/cohort_meta_by_id_imvday.parquet`, filtered to the
    final cohort.
    """)
    return


@app.cell
def _(
    cohort_hrly_grids_f,
    encounter_mapping,
    nmb_excluded_patient_days,
):
    # Aggregate hourly grid → one row per (hosp, _nth_day). Window functions
    # over PARTITION BY hospitalization_id give us min/max _nth_day per hosp
    # so we can label the partial-vs-full day_type. Within a kept IMV streak
    # the hourly grid is contiguous, so middle days are full by construction
    # — only the first and last days can be partial. Edge case: a patient
    # extubated at 6:59am produces a max-`_nth_day` row that is still full
    # (24 chargeable hours under the 7am-cross convention); day_type='full'
    # is correct for those.
    #
    # `encounter_block` propagation: when stitch is ON, `encounter_mapping`
    # is a (hospitalization_id, encounter_block) pandas df; when OFF it's
    # None. To keep the SQL a single mo.sql call (rather than branching
    # between two duckdb.sql variants), we synthesize an empty-but-schema-
    # stable mapping df when stitch is OFF — the LEFT JOIN unifies both
    # cases and the encounter_block column is NULL on the OFF path.
    #
    # Phase 2 fold-in (2026-05-07): NMB exclusion info from
    # `cohort_nmb_excluded.parquet` is LEFT-JOINed in additively as
    # `nmb_excluded` (bool — TRUE where any NMB exclusion applies) +
    # `nmb_total_min` (float — total NMB minutes that day, NULL on
    # non-flagged days). Standalone file stays on disk; canonical read
    # source for the per-day NMB flag becomes the registry.
    import pandas as _pd
    if encounter_mapping is None:
        encounter_map_df = _pd.DataFrame({
            "hospitalization_id": _pd.Series([], dtype="object"),
            "encounter_block":    _pd.Series([], dtype="Int64"),
        })
    else:
        encounter_map_df = encounter_mapping

    # Materialize the NMB-excluded patient-days as a pandas DataFrame.
    # DuckDB's replacement scan in marimo script mode reliably resolves
    # pandas DFs by name; cross-cell DuckDBPyRelation handoffs into a
    # consumer mo.sql block also resolve, but materializing here keeps
    # this cell self-contained.
    nmb_excluded_pd = nmb_excluded_patient_days.df()

    cohort_meta_by_id_imvday = mo.sql(f"""
        WITH per_day AS (
            FROM cohort_hrly_grids_f
            SELECT
                hospitalization_id
                , _nth_day
                , n_hrs_day:   COUNT(*) FILTER (WHERE _shift = 'day')
                , n_hrs_night: COUNT(*) FILTER (WHERE _shift = 'night')
                , day_start_dttm: MIN(event_dttm)
                , day_end_dttm:   MAX(event_dttm)
            GROUP BY hospitalization_id, _nth_day
        )
        , with_extremes AS (
            FROM per_day
            SELECT *
                , _is_full_24h_day: (n_hrs_day = 12 AND n_hrs_night = 12)
                , _max_nth: MAX(_nth_day) OVER (PARTITION BY hospitalization_id)
                , _min_nth: MIN(_nth_day) OVER (PARTITION BY hospitalization_id)
        )
        , typed AS (
            FROM with_extremes
            SELECT
                hospitalization_id
                , _nth_day
                , day_type: CASE
                    WHEN NOT _is_full_24h_day AND _nth_day = _min_nth THEN 'first_partial'
                    WHEN NOT _is_full_24h_day AND _nth_day = _max_nth THEN 'last_partial'
                    ELSE 'full'
                END
                , _is_full_24h_day
                , n_hrs_day
                , n_hrs_night
                , day_start_dttm
                , day_end_dttm
        )
        -- Compute the `_nth_day` of each patient's last full day in a separate
        -- CTE because `day_type` was derived in `typed` above; window functions
        -- can't reference an alias from the same SELECT.
        , with_last_full AS (
            FROM typed
            SELECT *
                , _max_full_nth: MAX(CASE WHEN day_type = 'full' THEN _nth_day END)
                                 OVER (PARTITION BY hospitalization_id)
        )
        , joined AS (
            FROM with_last_full t
            LEFT JOIN encounter_map_df em USING (hospitalization_id)
            SELECT
                t.hospitalization_id
                , em.encounter_block
                , t._nth_day
                , t.day_type
                -- _is_first_day = patient had a partial intubation day AND this
                -- row is it. Narrow semantic — patients whose IMV starts at 7am
                -- exactly have NO first_partial row and therefore no row flagged.
                , _is_first_day: (t.day_type = 'first_partial')
                -- _is_last_partial_day = the truncated extubation day (the row
                -- removed by analyses that require full 24-h coverage).
                , _is_last_partial_day:  (t.day_type = 'last_partial')
                -- _is_last_full_day = the patient's final full-24h day, i.e.
                -- the last row that survives the modeling-cohort filter. For
                -- patients with no full days at all, this is False everywhere.
                , _is_last_full_day: (t.day_type = 'full'
                                       AND t._nth_day = t._max_full_nth)
                , t._is_full_24h_day
                , t.n_hrs_day
                , t.n_hrs_night
                , t.day_start_dttm
                , t.day_end_dttm
        )
        FROM joined j
        LEFT JOIN nmb_excluded_pd n USING (hospitalization_id, _nth_day)
        SELECT
            j.*
            , nmb_excluded:  (n._nmb_total_min IS NOT NULL)
            , nmb_total_min: n._nmb_total_min
        ORDER BY j.hospitalization_id, j._nth_day
    """)

    _n_rows = cohort_meta_by_id_imvday.count("*").fetchone()[0]
    logger.info(
        f"cohort_meta_by_id_imvday: {_n_rows:,} (hosp×day) rows "
        f"(pre-cohort filter)"
    )
    return (cohort_meta_by_id_imvday,)


@app.cell
def _(cohort_meta_by_id_imvday):
    cohort_meta_by_id_imvday
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
    cohort_meta_by_id_imvday,
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

    # Per-criterion CONSORT steps. Two conditional steps coexist via the
    # n_*=None sentinel pattern:
    #   - Step 2 (encounter-stitch dedup) — emitted only if the stitch toggle
    #     is ON (n_dropped_stitch is not None).
    #   - Range rule (step ~7) — emitted only if the audit ran with
    #     WEIGHT_QC_RANGE_RULE_ON=1 (range_threshold_str is not None).
    # Step numbers are assigned by the renumber loop at the end so neither
    # conditional shifts a hardcoded literal.
    _steps = [
        {
            "step": 1,  # placeholder — final step numbers assigned below
            "description": "Hospitalizations with ICU stays",
            "n_remaining": _cc['n_icu'],
        },
    ]

    # Optional step 2: encounter-stitch dedup. Skipped when the toggle is off.
    if _cc['n_dropped_stitch'] is not None:
        _steps.append({
            "step": 2,
            "description": "After encounter-stitch dedup (12h window)",
            "n_remaining": _cc['n_post_stitch'],
            "n_excluded": _cc['n_dropped_stitch'],
            "exclusion_reason": "Within-12h discharge→readmit paperwork artifact (kept first hosp per stitched encounter block)",
        })

    _steps.extend([
        {
            "step": 0,
            "description": "With any invasive mechanical ventilation",
            "n_remaining": _cc['n_any_imv'],
            "n_excluded": _cc['n_no_imv'],
            "exclusion_reason": "No IMV recorded",
        },
        {
            "step": 0,
            "description": "First IMV streak >= 24 hours",
            "n_remaining": _cc['n_first_imv_ge24'],
            "n_excluded": _cc['n_first_imv_lt24'],
            "exclusion_reason": f"First IMV streak <24h ({_cc['n_trach_truncated']} tracheostomy-truncated)",
        },
        {
            "step": 0,
            "description": "Has >= 1 weight_kg record (presence check)",
            "n_remaining": _cc['n_post_weight_presence'],
            "n_excluded": _cc['n_dropped_no_weight'],
            "exclusion_reason": "Zero weight_kg rows in vitals (runtime upfront check)",
        },
        {
            "step": 0,
            "description": "Has >= 1 weight in [30, 300] kg (post-clamp residual)",
            "n_remaining": _cc['n_post_c1_residual'],
            "n_excluded": _cc['n_excl_c1_residual'],
            "exclusion_reason": "All weight_kg values outside clamp [30, 300] kg (audit C1 residual)",
        },
        {
            "step": 0,
            "description": "No consecutive weight jump within 24h",
            "n_remaining": _cc['n_post_jump'],
            "n_excluded": _cc['n_excl_jump'],
            "exclusion_reason": (
                f"Implausible weight jump (audit C2; reason='{_cc['jump_threshold_str']}')"
                if _cc['jump_threshold_str'] else
                "Implausible weight jump (audit C2; threshold unknown — drop list absent)"
            ),
        },
    ])

    # Optional weight-range rule. Skipped when audit ran without the rule.
    if _cc['range_threshold_str'] is not None:
        _steps.append({
            "step": 0,
            "description": "No within-stay weight range exceeded",
            "n_remaining": _cc['n_post_range'],
            "n_excluded": _cc['n_excl_range'],
            "exclusion_reason": f"Within-stay weight range exceeded (audit C3; reason='{_cc['range_threshold_str']}')",
        })

    # NMB step (always last). n_remaining is anchored to the actual final
    # cohort size minus NMB-excluded hosps.
    _steps.append({
        "step": 0,
        "description": "Exclude hospitalizations with any NMB >1h",
        "n_remaining": len(cohort_hosp_ids) - _n_nmb_hosp_ids,
        "n_excluded": _n_nmb_hosp_ids,
        "exclusion_reason": f"Any patient-day with NMB >1h ({_n_nmb_patient_days:,} patient-days across {_n_nmb_hosp_ids:,} hosp)",
    })

    # Assign final step numbers in order — single source of truth, robust to
    # which conditional steps fired.
    for _i, _s in enumerate(_steps, start=1):
        _s["step"] = _i

    consort_flow = {
        "site": SITE_NAME,
        "steps": _steps,
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
    _meta_imvday_kept_rel = duckdb.sql(f"""
        FROM cohort_meta_by_id_imvday
        WHERE hospitalization_id IN (SELECT UNNEST({cohort_hosp_ids}))
    """)

    # UTC-on-disk: every *_dttm column is written as UTC tz-aware
    # (see docs/timezone_audit.md). tz_convert is metadata-only — same
    # UTC instants, just the display tag is UTC. Polars `write_parquet`
    # round-trips the tag through Arrow.
    (
        _streaks_kept_rel
        .pl()
        .with_columns(
            pl.col('_start_dttm').dt.convert_time_zone("UTC"),
            pl.col('_end_dttm').dt.convert_time_zone("UTC"),
        )
        .write_parquet(f"output/{SITE_NAME}/cohort_imv_streaks.parquet")
    )
    (
        _grids_kept_rel
        .pl()
        .with_columns(pl.col('event_dttm').dt.convert_time_zone("UTC"))
        .write_parquet(f"output/{SITE_NAME}/cohort_meta_by_id_imvhr.parquet")
    )
    (
        _meta_imvday_kept_rel
        .pl()
        .with_columns(
            pl.col('day_start_dttm').dt.convert_time_zone("UTC"),
            pl.col('day_end_dttm').dt.convert_time_zone("UTC"),
        )
        .write_parquet(f"output/{SITE_NAME}/cohort_meta_by_id_imvday.parquet")
    )

    # Non-tz outputs: DuckDB native .to_parquet() directly on the relation.
    nmb_excluded_patient_days.to_parquet(f"output/{SITE_NAME}/cohort_nmb_excluded.parquet")
    icu_type_df.to_parquet(f"output/{SITE_NAME}/cohort_icu_type.parquet")

    _n_grids_rows = _grids_kept_rel.count("*").fetchone()[0]
    _n_meta_imvday_rows = _meta_imvday_kept_rel.count("*").fetchone()[0]
    _n_icu = icu_type_df.count("*").fetchone()[0]
    logger.info(
        f"Saved: output/{SITE_NAME}/cohort_imv_streaks.parquet "
        f"({len(cohort_hosp_ids):,} hospitalizations)"
    )
    logger.info(f"Saved: output/{SITE_NAME}/cohort_meta_by_id_imvhr.parquet ({_n_grids_rows:,} rows)")
    logger.info(
        f"Saved: output/{SITE_NAME}/cohort_meta_by_id_imvday.parquet "
        f"({_n_meta_imvday_rows:,} hosp×day rows)"
    )
    logger.info(f"Saved: output/{SITE_NAME}/cohort_nmb_excluded.parquet ({_n_nmb_patient_days:,} patient-days)")
    logger.info(f"Saved: output/{SITE_NAME}/cohort_icu_type.parquet ({_n_icu:,} hospitalizations)")

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
