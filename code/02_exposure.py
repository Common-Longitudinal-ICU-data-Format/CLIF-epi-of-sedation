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
app = marimo.App(width="columns", sql_output="native")

with app.setup:
    import marimo as mo
    import os
    import sys
    import logging
    from pathlib import Path
    # sys.path.insert(0, str(Path(__file__).parent))

    # Module-level logger singleton (per logging_integration_guide.md §2-3:
    # hardcoded short name, not __name__). Visible in every cell. The
    # `clifpy.epi_sedation.exposure` resolved name groups this script's
    # log lines under the project namespace.
    from clifpy.utils.logging_config import get_logger
    logger = get_logger("epi_sedation.exposure")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 02 Sedation Dose Calculation

    Computes hourly, daily, and aggregated sedation doses (continuous + intermittent)
    for the IMV cohort identified in 01_cohort.

    Pipeline follows `pyCLIF/docs/duckdb_perf_guide.md`: stay lazy as
    `DuckDBPyRelation`s, materialize only at the parquet-write boundary.
    Vendored DuckDB outlier handler (`code/_outlier_handler.py`) replaces
    `apply_outlier_handling` so the chain doesn't have to round-trip through
    `Table.from_file`'s pandas attribute.
    """)
    return


@app.cell
def _():
    from clifpy import load_data, setup_logging
    import duckdb
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category
    from clifpy.utils.config import get_config_or_params
    from _utils import remove_meds_duplicates
    from _outlier_handler import apply_outlier_handling_duckdb

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # All timestamps in this script are tz-aware: cohort_meta_by_id_imvhr.parquet
    # arrives site-tz tagged (per 01_cohort.py's retag_to_local_tz boundary);
    # clifpy-loaded admin/recorded times are UTC tz-aware. Local-hour
    # extraction uses explicit `AT TIME ZONE '{SITE_TZ}'` per query (see
    # cont_sed_wg / intm_sed_wg cells), so DuckDB's session timezone is
    # never read — no `SET TimeZone` here. The pytest at
    # tests/test_timezone.py asserts session-tz invariance.

    CONFIG_PATH = "config/config.json"
    return (
        CONFIG_PATH,
        apply_outlier_handling_duckdb,
        convert_dose_units_by_med_category,
        duckdb,
        get_config_or_params,
        load_data,
        remove_meds_duplicates,
        setup_logging,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params, setup_logging):
    # Site-scoped output dir (see Makefile SITE= flag).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    # Per-site log separation: each site writes to output/{site}/logs/
    # clifpy_all.log + clifpy_errors.log. setup_logging is idempotent; we
    # call it explicitly here (instead of via ClifOrchestrator's __init__
    # side effect) so the output_directory is site-scoped.
    setup_logging(output_directory=f"output/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return SITE_NAME, SITE_TZ


@app.cell
def _(SITE_NAME, duckdb):
    # Lazy parquet read — DuckDB scans on demand. The grid is site-tz tagged
    # at write time by 01_cohort's retag_to_local_tz boundary, so event_dttm
    # carries the correct local tz on disk; downstream SQL still uses
    # AT TIME ZONE for explicit local-hour extraction.
    cohort_hrly_grids_f = duckdb.sql(
        f"FROM 'output/{SITE_NAME}/cohort_meta_by_id_imvhr.parquet' SELECT *"
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = cohort_hrly_grids_f.count("*").fetchone()[0]
        logger.debug(f"Hourly grid rows: {_n:,}")
    else:
        logger.info("Hourly grid loaded as lazy relation")
    return (cohort_hrly_grids_f,)


@app.cell
def _(cohort_hrly_grids_f, duckdb):
    cohort_hosp_ids = [
        r[0] for r in duckdb.sql(
            "FROM cohort_hrly_grids_f SELECT DISTINCT hospitalization_id"
        ).fetchall()
    ]
    logger.info(f"Cohort hospitalizations: {len(cohort_hosp_ids):,}")
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
def _(apply_outlier_handling_duckdb, cohort_hosp_ids, load_data):
    # `load_data(return_rel=True)` returns a bare DuckDBPyRelation — lazy,
    # session-tz-independent (per duckdb_perf_guide §11.1, the vendored
    # DuckDB outlier handler replaces clifpy's Table-object-bound
    # apply_outlier_handling so the whole chain stays unmaterialized).
    vitals_rel = load_data(
        'vitals',
        config_path='config/config.json',
        return_rel=True,
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        filters={
            'vital_category': ['weight_kg'],
            'hospitalization_id': cohort_hosp_ids,
        },
    )
    vitals_rel = apply_outlier_handling_duckdb(
        vitals_rel, 'vitals', 'config/outlier_config.yaml',
    )
    logger.info("Vitals (weight_kg): lazy relation built")
    return (vitals_rel,)


@app.cell
def _(cohort_hosp_ids, duckdb, load_data):
    # Same pattern as vitals: lazy load with clifpy column/filter pushdown,
    # then NULL-dose drop in SQL. clifpy's `from_file` filters dict only
    # supports IN-list semantics, so the IS NOT NULL gate is applied here
    # rather than at the load layer. The cohort + med_category pushdown
    # above already narrows the parquet scan dramatically, so the remaining
    # cost is a single-column predicate on the filtered subset.
    cont_sed_rel = load_data(
        'medication_admin_continuous',
        config_path='config/config.json',
        return_rel=True,
        columns=[
            'hospitalization_id', 'admin_dttm', 'med_name', 'med_category',
            'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category',
        ],
        filters={
            'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'],
            'hospitalization_id': cohort_hosp_ids,
        }
    )
    cont_sed_rel = duckdb.sql("FROM cont_sed_rel WHERE med_dose IS NOT NULL AND mar_action_category != 'not_given'")
    if logger.isEnabledFor(logging.DEBUG):
        _n = cont_sed_rel.count("*").fetchone()[0]
        logger.debug(f"Continuous sedation: {_n:,} non-null-dose rows")
    else:
        logger.info("Continuous sedation: lazy relation built")
    return (cont_sed_rel,)


@app.cell
def _(cont_sed_rel):
    cont_sed_rel
    return


@app.cell
def _(cont_sed_rel, duckdb, remove_meds_duplicates):
    cont_sed_deduped = remove_meds_duplicates(cont_sed_rel)
    # Summary count is always logged (single scalar query, cheap). The
    # per-mar_action_name combo breakdown below is DEBUG-gated.
    _n_before = cont_sed_rel.count("*").fetchone()[0]
    _n_after = cont_sed_deduped.count("*").fetchone()[0]
    _n_removed = _n_before - _n_after
    _pct = _n_removed / _n_before * 100 if _n_before else 0.0
    logger.info(
        f"cont_sed dedup: removed {_n_removed:,} ({_pct:.2f}%) "
        f"({_n_before:,} → {_n_after:,})"
    )
    if logger.isEnabledFor(logging.DEBUG):
        # MAR-dedup QC (F5): surface top mar_action_name combos in the
        # duplicate clusters. Lazy scalar query — never materializes the
        # row-level frame. Adapted from
        # pyCLIF/dev/check_mar_duplicates_dev.ipynb.
        _rows = duckdb.sql("""
            WITH dups AS (
                FROM cont_sed_rel
                SELECT *
                    , _grp_size: COUNT(*) OVER (
                        PARTITION BY hospitalization_id, admin_dttm, med_category
                    )
                QUALIFY _grp_size > 1
            )
            , combos AS (
                FROM dups
                SELECT
                    hospitalization_id, admin_dttm, med_category
                    , combo: STRING_AGG(mar_action_name, '; ' ORDER BY mar_action_name)
                GROUP BY 1, 2, 3
            )
            FROM combos
            SELECT combo, n_groups: COUNT(*)
            GROUP BY combo
            ORDER BY n_groups DESC
            LIMIT 10
        """).fetchall()
        for _combo, _n_groups in _rows:
            logger.debug(f"  cont_sed dup combo [{_combo}]: {_n_groups:,} groups")
    return (cont_sed_deduped,)


@app.cell
def _(cont_sed_deduped, vitals_rel):
    cont_sed_with_weight = mo.sql(
        f"""
        -- Phase 2 weight override: pre-attach a project-controlled `weight_kg`
        -- column on each admin row BEFORE handing off to clifpy. Per
        -- clifpy/utils/unit_converter.py:717-719, clifpy only runs its own ASOF
        -- when `weight_kg` isn't already a column on med_df, so this override
        -- skips clifpy's no-fallback per-admin ASOF entirely (the path that
        -- produced the silent /kg-factor-dropped bug — see
        -- code/qc/weight_audit_README.md §5).
        --
        -- Strategy: per-admin ASOF backward-join to most-recent prior weight
        -- (same temporal logic as clifpy), with fallback to the patient's
        -- first-ever weight (admission fallback). The weight-QC drop list at
        -- 01_cohort.py guarantees every kept patient has ≥1 weight, so the
        -- admission fallback is never NULL.
        --
        -- Stays lazy as a DuckDBPyRelation — feeds directly into
        -- convert_dose_units_by_med_category(return_rel=True) below.
        WITH weights AS (
                FROM vitals_rel
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
        """
    )
    return (cont_sed_with_weight,)


@app.cell
def _(cont_sed_with_weight, duckdb):
    # Diagnostics on the lazy weight-attached relation (per perf-guide §11.6:
    # scalar diagnostics without breaking the lazy DAG). The SUM-over-CASE
    # query touches the relation independently; cont_sed_with_weight
    # continues downstream untouched.
    _diag = duckdb.sql("""
        FROM cont_sed_with_weight
        SELECT
            COUNT(*) AS n_total
            , SUM(CASE WHEN weight_kg IS NULL THEN 1 ELSE 0 END) AS n_null
            , SUM(CASE WHEN _weight_source = 'admission_fallback' THEN 1 ELSE 0 END) AS n_fallback
    """).fetchone()
    _n_total, _n_null, _n_fallback = _diag
    logger.info(f"Pre-attached weight on {_n_total:,} admins")
    logger.info(f"  per_admin_asof: {_n_total - _n_fallback - _n_null:,}")
    logger.info(f"  admission_fallback: {_n_fallback:,}")
    if _n_null > 0:
        # WARNING for the fallback-failure path. Don't embed the literal
        # "WARNING:" prefix — the EmojiFormatter handles per-level prefixes.
        logger.warning(
            f"{_n_null:,} admins NULL after fallback — these patients should "
            "have been in weight-QC drop list. Check 01_cohort.py applied the drop."
        )
    return


@app.cell
def _(cont_sed_with_weight):
    cont_sed_with_weight
    return


@app.cell
def _(cont_sed_with_weight, convert_dose_units_by_med_category):
    # Preferred unit for propofol changed to mcg/kg/min (Phase 2): pump-native
    # /kg/min input passes through with only amount/time scaling — no weight
    # multiplication/division at the conversion layer for the dominant
    # charting form. Combined with the pre-attached weight column above,
    # this eliminates the silent /kg-factor-dropped bug entirely.
    #
    # `return_rel=True` keeps both outputs lazy:
    #   - cont_sed_converted_rel: lazy relation feeding the outlier-handling
    #     boundary in the next cell.
    #   - cont_sed_convert_summary: lazy relation surfacing per-category
    #     unit-mapping counts. Exposed publicly for inspection.
    _cont_sed_preferred_units = {
        'propofol': 'mcg/kg/min',
        'midazolam': 'mg/min',
        'fentanyl': 'mcg/min',
        'hydromorphone': 'mg/min',
        'lorazepam': 'mg/min',
    }
    cont_sed_converted_rel, cont_sed_convert_summary = convert_dose_units_by_med_category(
        cont_sed_with_weight,
        preferred_units=_cont_sed_preferred_units,
        override=True,
        return_rel=True,
    )
    return cont_sed_convert_summary, cont_sed_converted_rel


@app.cell
def _(cont_sed_convert_summary):
    cont_sed_convert_summary
    return


@app.cell
def _(cont_sed_converted_rel):
    # Promote `_converted` columns to canonical names (med_dose,
    # med_dose_unit) and preserve originals as `_original`. Pure SQL,
    # lazy. The vendored outlier handler in the next cell operates on the
    # promoted (preferred-unit) values — matching the YAML config keys.
    cont_sed_renamed = mo.sql(
        f"""
        FROM cont_sed_converted_rel
        SELECT
            * EXCLUDE (med_dose, med_dose_unit, med_dose_converted, med_dose_unit_converted)
            , med_dose_original: med_dose
            , med_dose_unit_original: med_dose_unit
            , med_dose: med_dose_converted
            , med_dose_unit: med_dose_unit_converted
        """
    )
    return (cont_sed_renamed,)


@app.cell
def _(apply_outlier_handling_duckdb, cont_sed_renamed):
    # Vendored DuckDB outlier handler — operates on a DuckDBPyRelation,
    # returns a lazy relation. Replaces clifpy's pandas-based
    # apply_outlier_handling (per duckdb_perf_guide §11.1).
    cont_sed_outliered = apply_outlier_handling_duckdb(
        cont_sed_renamed,
        'medication_admin_continuous',
        'config/outlier_config.yaml',
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = cont_sed_outliered.count("*").fetchone()[0]
        logger.debug(f"cont_sed post-outlier: {_n:,} rows")
    else:
        logger.info("cont_sed post-outlier: lazy relation built")
    return (cont_sed_outliered,)


@app.cell
def _(cont_sed_outliered):
    cont_sed_outliered
    return


@app.cell
def _():
    # Pivot continuous sedation to wide format. F4 fix: the column-name
    # builder embeds the post-aggregation unit (`_hr_`) directly so the
    # SUM-per-hour step downstream produces final names without any
    # pandas .rename() pass (per duckdb_perf_guide §11.2 — dynamic
    # PIVOT_WIDER columns block post-pivot SQL aliasing).
    #
    # Mid-pipeline naming caveat: between forward-fill and SUM-per-hour,
    # column names will say `_hr_cont` while values are still per-minute
    # rates. Mirrors a pre-F4 inverse lie (`_min_cont` after
    # multiplication-by-duration). Names match values only at the final
    # per-hour aggregation step.
    #
    # NOTE: stop events forced to dose=0 so forward-fill doesn't propagate
    # stale rates (audit H1).
    cont_sed_w = mo.sql(
        f"""
        WITH t1 AS (
        FROM cont_sed_outliered
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit:
                med_category || '_'
                || REPLACE(REPLACE(med_dose_unit, '/min', ''), '/', '_')
                || '_hr_cont'
            , med_dose: CASE WHEN mar_action_category IN ('stop', 'not_given') THEN 0 ELSE med_dose END
        )
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_w,)


@app.cell
def _(SITE_TZ, cohort_hrly_grids_f, cont_sed_w):
    # FULL JOIN with hourly grid: the grid side anchors the timeline at hour
    # boundaries so forward-fill below can propagate dose rates across hours
    # with no admin event. We subset the grid to just the join keys —
    # downstream cells only need (hosp_id, event_dttm, _dh, _hr, *_cont).
    # `_shift`/`_nth_day`/etc. on cohort_hrly_grids_f are re-joined later at
    # the per-hour merge step (sed_dose_by_hr).
    #
    # `_dh`/`_hr` derived from `event_dttm AT TIME ZONE '{SITE_TZ}'` —
    # explicit local-tz extraction, session-tz-independent. `event_dttm` may
    # arrive as site-tz tagged (from cohort_meta_by_id_imvhr.parquet, post
    # 01_cohort retag) or as UTC tz-aware (from cont_sed_w via clifpy load);
    # DuckDB normalizes both to TIMESTAMPTZ internally and `AT TIME ZONE`
    # operates on the underlying instant, so the result is correct
    # regardless of which path supplied the row.
    cont_sed_wg = mo.sql(
        f"""
        WITH grid AS (
            FROM cohort_hrly_grids_f
            SELECT hospitalization_id, event_dttm
        )
        FROM grid g
        FULL JOIN cont_sed_w m USING (hospitalization_id, event_dttm)
        SELECT
            * EXCLUDE (event_dttm)
            , event_dttm
            , _dh: date_trunc('hour', event_dttm AT TIME ZONE '{SITE_TZ}')
            , _hr: extract('hour' FROM event_dttm AT TIME ZONE '{SITE_TZ}')::INT
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_wg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Continuous Dose by Hour

    Forward-fill rates within hospitalization, coalesce nulls to 0,
    multiply rate × duration to get per-event doses, then aggregate
    by hour.

    Each step stays lazy as a `DuckDBPyRelation`; the chain is fused by
    the optimizer when `cont_sed_dose_by_hr` materializes.
    """)
    return


@app.cell
def _(cont_sed_wg):
    # Forward-fill continuous sedation rates and compute duration between events.
    # `_duration` for the last event of a partition is 0 (LEAD's third arg
    # falls back to event_dttm itself, yielding zero diff).
    cont_sed_filled = mo.sql(
        f"""
        FROM cont_sed_wg g
        SELECT hospitalization_id, event_dttm, _dh, _hr
            , LAST_VALUE(COLUMNS('_cont') IGNORE NULLS) OVER (
                PARTITION BY hospitalization_id ORDER BY event_dttm
            )
            , _duration: EXTRACT(EPOCH FROM (LEAD(event_dttm, 1, event_dttm) OVER w - event_dttm)) / 60.0
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        """
    )
    return (cont_sed_filled,)


@app.cell
def _(cont_sed_filled):
    # Coalesce null sedation rates to 0 (stop events + pre-first-admin grid rows).
    cont_sed_zeroed = mo.sql(
        f"""
        FROM cont_sed_filled
        SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
            , COALESCE(COLUMNS('_cont'), 0)
        """
    )
    return (cont_sed_zeroed,)


@app.cell
def _(cont_sed_zeroed):
    # Multiply rate by duration (minutes) → total dose per inter-event interval.
    cont_sed_per_event = mo.sql(
        f"""
        FROM cont_sed_zeroed
        SELECT hospitalization_id, event_dttm, _dh, _hr, _duration
            , COLUMNS('_cont') * _duration
        """
    )
    return (cont_sed_per_event,)


@app.cell
def _(cont_sed_per_event):
    # Aggregate continuous dose by hour. Column names already carry the
    # post-aggregation `_hr_` unit suffix (set at pivot-construction time
    # in cont_sed_w), so no rename is needed — the SQL output IS the final
    # relation. Per duckdb_perf_guide §11.2 (dynamic PIVOT_WIDER columns
    # block post-pivot SQL aliasing; push the desired names into the
    # upstream column-builder).
    cont_sed_dose_by_hr = mo.sql(
        f"""
        FROM cont_sed_per_event
        SELECT hospitalization_id, _dh, _hr
            , SUM(COLUMNS('_cont'))
        GROUP BY hospitalization_id, _dh, _hr
        ORDER BY hospitalization_id, _dh
        """
    )
    return (cont_sed_dose_by_hr,)


@app.cell
def _(cont_sed_dose_by_hr):
    cont_sed_dose_by_hr
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Intermittent Sedation

    Same pattern as continuous, but `mar_action_category` filters and the
    unit-conversion targets differ.
    """)
    return


@app.cell
def _(cohort_hosp_ids, duckdb, load_data):
    intm_sed_rel = load_data(
        'medication_admin_intermittent',
        config_path='config/config.json',
        return_rel=True,
        columns=[
            'hospitalization_id', 'admin_dttm', 'med_name', 'med_category',
            'med_dose', 'med_dose_unit', 'mar_action_name', 'mar_action_category',
        ],
        filters={
            'med_category': ['hydromorphone', 'fentanyl', 'lorazepam', 'midazolam', 'propofol'],
            'hospitalization_id': cohort_hosp_ids,
            # 'mar_action_category': ['given', 'bolus', 'other'],
        }
    )
    intm_sed_rel = duckdb.sql("FROM intm_sed_rel WHERE med_dose IS NOT NULL")
    if logger.isEnabledFor(logging.DEBUG):
        _n = intm_sed_rel.count("*").fetchone()[0]
        logger.debug(f"Intermittent sedation: {_n:,} non-null-dose rows")
    else:
        logger.info("Intermittent sedation: lazy relation built")
    return (intm_sed_rel,)


@app.cell
def _(duckdb, intm_sed_rel, remove_meds_duplicates):
    intm_sed_deduped = remove_meds_duplicates(intm_sed_rel)
    # Summary always at INFO so MIMIC's high removal rate is visible by
    # default; the per-combo breakdown is DEBUG-gated.
    _n_before = intm_sed_rel.count("*").fetchone()[0]
    _n_after = intm_sed_deduped.count("*").fetchone()[0]
    _n_removed = _n_before - _n_after
    _pct = _n_removed / _n_before * 100 if _n_before else 0.0
    logger.info(
        f"intm_sed dedup: removed {_n_removed:,} ({_pct:.2f}%) "
        f"({_n_before:,} → {_n_after:,})"
    )
    if logger.isEnabledFor(logging.DEBUG):
        # MIMIC's intermittent table has historically shown ~25% removal
        # rate (vs UCMC ~few%), suggesting systematic same-timestamp
        # duplicates from the dual-charting pattern (see
        # pyCLIF/dev/check_mar_duplicates_dev.ipynb). Surface the top
        # mar_action_name combos here.
        _rows = duckdb.sql("""
            WITH dups AS (
                FROM intm_sed_rel
                SELECT *
                    , _grp_size: COUNT(*) OVER (
                        PARTITION BY hospitalization_id, admin_dttm, med_category
                    )
                QUALIFY _grp_size > 1
            )
            , combos AS (
                FROM dups
                SELECT
                    hospitalization_id, admin_dttm, med_category
                    , combo: STRING_AGG(mar_action_name, '; ' ORDER BY mar_action_name)
                GROUP BY 1, 2, 3
            )
            FROM combos
            SELECT combo, n_groups: COUNT(*)
            GROUP BY combo
            ORDER BY n_groups DESC
            LIMIT 10
        """).fetchall()
        for _combo, _n_groups in _rows:
            logger.debug(f"  intm_sed dup combo [{_combo}]: {_n_groups:,} groups")
    return (intm_sed_deduped,)


@app.cell
def _(convert_dose_units_by_med_category, intm_sed_deduped, vitals_rel):
    # Intermittent: bolus mg/dose with no defined duration. Targets are
    # weight-free amount units; vitals_rel is needed when a source row carries
    # a /kg qualifier. Stay lazy via return_rel=True; convert_summary is the
    # surface that confirms the unit map for each category.
    _intm_sed_preferred_units = {
        'propofol': 'mg',
        'midazolam': 'mg',
        'fentanyl': 'mcg',
        'hydromorphone': 'mg',
        'lorazepam': 'mg',
    }
    intm_sed_converted_rel, intm_sed_convert_summary = convert_dose_units_by_med_category(
        intm_sed_deduped,
        vitals_df=vitals_rel,
        preferred_units=_intm_sed_preferred_units,
        override=True,
        return_rel=True,
    )
    return intm_sed_convert_summary, intm_sed_converted_rel


@app.cell
def _(intm_sed_convert_summary):
    intm_sed_convert_summary
    return


@app.cell
def _(intm_sed_converted_rel):
    # Same alias-rename as cont_sed: promote `_converted` columns to
    # canonical names (med_dose, med_dose_unit) for the outlier handler.
    intm_sed_renamed = mo.sql(
        f"""
        FROM intm_sed_converted_rel
        SELECT
            * EXCLUDE (med_dose, med_dose_unit, med_dose_converted, med_dose_unit_converted)
            , med_dose_original: med_dose
            , med_dose_unit_original: med_dose_unit
            , med_dose: med_dose_converted
            , med_dose_unit: med_dose_unit_converted
        """
    )
    return (intm_sed_renamed,)


@app.cell
def _(apply_outlier_handling_duckdb, intm_sed_renamed):
    intm_sed_outliered = apply_outlier_handling_duckdb(
        intm_sed_renamed,
        'medication_admin_intermittent',
        'config/outlier_config.yaml',
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = intm_sed_outliered.count("*").fetchone()[0]
        logger.debug(f"intm_sed post-outlier: {_n:,} rows")
    else:
        logger.info("intm_sed post-outlier: lazy relation built")
    return (intm_sed_outliered,)


@app.cell
def _(intm_sed_outliered):
    intm_sed_outliered
    return


@app.cell
def _():
    # Pivot intermittent sedation to wide format. F4 fix: the column-name
    # builder embeds `_hr_intm` directly so the SUM-per-hour step
    # downstream produces final names without rename. After SUM-per-hour,
    # `propofol_mg_hr_intm` represents total mg delivered that hour from
    # intermittent admin.
    intm_sed_w = mo.sql(
        f"""
        WITH t1 AS (
        FROM intm_sed_outliered
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit:
                med_category || '_'
                || REPLACE(med_dose_unit, '/', '_')
                || '_hr_intm'
            , med_dose: CASE WHEN mar_action_category = 'not_given' THEN 0 ELSE med_dose END
        )
        PIVOT_WIDER t1
        ON med_category_unit
        USING FIRST(med_dose)
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_w,)


@app.cell
def _(intm_sed_w):
    intm_sed_w.df()
    return


@app.cell
def _(SITE_TZ, cohort_hrly_grids_f, intm_sed_w):
    # FULL JOIN with hourly grid — same pattern as cont_sed_wg (subset grid
    # to join keys, derive `_dh`/`_hr` from `event_dttm AT TIME ZONE site_tz`
    # for explicit, session-tz-independent local-hour extraction).
    intm_sed_wg = mo.sql(
        f"""
        WITH grid AS (
            FROM cohort_hrly_grids_f
            SELECT hospitalization_id, event_dttm
        )
        FROM grid g
        FULL JOIN intm_sed_w m USING (hospitalization_id, event_dttm)
        SELECT
            * EXCLUDE (event_dttm)
            , event_dttm
            , _dh: date_trunc('hour', event_dttm AT TIME ZONE '{SITE_TZ}')
            , _hr: extract('hour' FROM event_dttm AT TIME ZONE '{SITE_TZ}')::INT
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_wg,)


@app.cell
def _(intm_sed_wg):
    # Aggregate intermittent dose by hour. Column names already carry the
    # post-aggregation `_hr_intm` unit suffix (set at pivot-construction
    # time in intm_sed_w), so no rename is needed.
    intm_sed_dose_by_hr = mo.sql(
        f"""
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
        ORDER BY hospitalization_id, _dh
        """
    )
    return (sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Daily Aggregation

    Aggregate per (hospitalization_id, _nth_day, _shift), then pivot to
    day/night columns via SQL `FILTER` clauses (replaces pandas `.pivot()`
    so the daily roll-up stays in SQL).
    """)
    return


@app.cell
def _(sed_dose_by_hr):
    # Aggregate dose totals AND hour counts per hospitalization, day, shift.
    # n_hours is used downstream to convert totals to per-hour dose rates,
    # which avoids single-shift bias in the Dose by Shift descriptive table
    # and makes model dose coefficients interpretable as rates (see
    # 05_analytical_dataset.py for the ÷12 rate conversion).
    # Column-name convention (2026-04-24): totals carry their unit as
    # suffix (`_mg` for propofol+midaz totals, `_mcg` for fentanyl totals);
    # per-hour rates downstream carry `_mg_hr` / `_mcg_hr`.
    sed_dose_agg = mo.sql(
        f"""
        FROM sed_dose_by_hr
        SELECT hospitalization_id, _nth_day, _shift
            -- Phase 2: propofol now sums to total mcg/kg over the shift
            -- (continuous infusion only). Downstream 05 divides by hours
            -- and minutes to produce the per-min rate `_prop_day_mcg_kg_min`.
            , prop_mcg_kg: SUM(prop_mcg_kg_total)
            , fenteq_mcg:  SUM(_fenteq_mcg_total)
            , midazeq_mg:  SUM(_midazeq_mg_total)
            , n_hours:     COUNT(*)
        GROUP BY hospitalization_id, _nth_day, _shift
        ORDER BY hospitalization_id, _nth_day, _shift
        """
    )
    return (sed_dose_agg,)


@app.cell
def _(sed_dose_agg):
    # Pivot _shift to wide day/night columns via conditional aggregation.
    # FILTER (WHERE _shift = 'day' / 'night') replaces the pandas .pivot()
    # +column-flatten dance so the entire daily roll-up stays in SQL.
    #
    # n_hours_day/night flow through so downstream can compute per-hour dose
    # rates that correctly handle single-shift days (intubation / extubation
    # day edges). Unit-suffixed column names (e.g. prop_day_mcg_kg = total
    # mcg/kg over 12h day shift) match the 2026-04-24 naming convention; see
    # 05_analytical_dataset.py.
    sed_dose_daily = mo.sql(
        f"""
        FROM sed_dose_agg
        SELECT hospitalization_id, _nth_day
            , prop_day_mcg_kg:   SUM(prop_mcg_kg) FILTER (WHERE _shift = 'day')
            , prop_night_mcg_kg: SUM(prop_mcg_kg) FILTER (WHERE _shift = 'night')
            , fenteq_day_mcg:    SUM(fenteq_mcg)  FILTER (WHERE _shift = 'day')
            , fenteq_night_mcg:  SUM(fenteq_mcg)  FILTER (WHERE _shift = 'night')
            , midazeq_day_mg:    SUM(midazeq_mg)  FILTER (WHERE _shift = 'day')
            , midazeq_night_mg:  SUM(midazeq_mg)  FILTER (WHERE _shift = 'night')
            , n_hours_day:       SUM(n_hours)     FILTER (WHERE _shift = 'day')
            , n_hours_night:     SUM(n_hours)     FILTER (WHERE _shift = 'night')
        GROUP BY hospitalization_id, _nth_day
        ORDER BY hospitalization_id, _nth_day
        """
    )
    return (sed_dose_daily,)


@app.cell
def _(sed_dose_daily):
    sed_dose_daily
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs

    Terminal materialization. `sed_dose_daily` writes via DuckDB native
    `.to_parquet()` (no `*_dttm` columns). `sed_dose_by_hr` has `event_dttm`
    from the cohort grid join — DuckDB's parquet writer normalizes
    TIMESTAMPTZ to a UTC tag, which would lose our site-local tag on disk,
    so it routes through Polars (preserves the tz tag via Arrow). See
    `pyCLIF/docs/duckdb_perf_guide.md §11.4`.

    Phase 2 (2026-05-07): `sed_dose_agg.parquet` is no longer written.
    Its per-shift dose totals were already mirrored as separate day/night
    columns in `seddose_by_id_imvday.parquet`, so the aggregate file was
    a pure intra-script intermediate with no external readers.
    """)
    return


@app.cell
def _(SITE_NAME, SITE_TZ, sed_dose_by_hr, sed_dose_daily):
    import polars as pl  # cell-local — used only for the tz-bearing parquet write

    sed_dose_daily.to_parquet(f"output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    (
        sed_dose_by_hr
        .pl()
        .with_columns(pl.col("event_dttm").dt.convert_time_zone(SITE_TZ))
        .write_parquet(f"output/{SITE_NAME}/seddose_by_id_imvhr.parquet")
    )

    logger.info(f"Saved: output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    logger.info(f"Saved: output/{SITE_NAME}/seddose_by_id_imvhr.parquet")
    return


if __name__ == "__main__":
    app.run()
