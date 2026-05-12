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
app = marimo.App(width="full", sql_output="native")

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

    # Single source of truth for the 5 expected sedatives + their preferred
    # post-conversion units. Drives BOTH:
    #   (a) clifpy's `convert_dose_units_by_med_category(preferred_units=...)`
    #       — what unit each drug is normalized to.
    #   (b) `PIVOT_WIDER ... ON med_category_unit IN (value_list)` — what
    #       columns the pivot is GUARANTEED to produce, even if a site
    #       has zero rows for one of the drugs (would otherwise crash
    #       downstream SQL with `Referenced column not found`).
    # Adding/removing a drug here automatically flows through to both
    # call sites + the IN list, with no separate "expected cols" registry
    # to keep in sync.
    CONT_SED_PREFERRED_UNITS = {
        'propofol':      'mcg/kg/min',
        'midazolam':     'mg/min',
        'fentanyl':      'mcg/min',
        'hydromorphone': 'mg/min',
        'lorazepam':     'mg/min',
    }
    INTM_SED_PREFERRED_UNITS = {
        'propofol':      'mcg/kg',
        'midazolam':     'mg',
        'fentanyl':      'mcg',
        'hydromorphone': 'mg',
        'lorazepam':     'mg',
    }

    def _expected_pivot_cols(preferred_units, suffix):
        """Build the expected pivot column names from a preferred_units dict.

        Mirrors the pivot's column-name builder:
            <drug> || '_' || replace(<unit>, '/', '_') || '_' || <suffix>
        Returns the list passed to `PIVOT_WIDER ... ON col IN (...)` so
        the pivot output schema is deterministic regardless of which
        drugs appear in the underlying data.
        """
        return [
            f"{drug}_{unit.replace('/', '_')}_{suffix}"
            for drug, unit in preferred_units.items()
        ]

    CONT_EXPECTED_PIVOT_COLS = _expected_pivot_cols(CONT_SED_PREFERRED_UNITS, 'cont')
    INTM_EXPECTED_PIVOT_COLS = _expected_pivot_cols(INTM_SED_PREFERRED_UNITS, 'intm')


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
    from clifpy import setup_logging
    import duckdb
    from clifpy.utils.unit_converter import convert_dose_units_by_med_category
    from clifpy.utils.config import get_config_or_params
    from _utils import (
        add_dh_hr,
        mar_action_not_given_filter_sql,
        mar_action_zero_dose_sql,
        normalize_categories,
        remove_meds_duplicates,
    )
    from _outlier_handler import apply_outlier_handling_duckdb

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # All timestamps in this script are UTC TIMESTAMPTZ throughout: clifpy
    # parquet files store *_dttm as UTC tz-aware, and we read them via
    # raw DuckDB (`FROM '<data_dir>/clif_<table>.parquet' SELECT ...`)
    # rather than clifpy.load_data — which would silently convert to naive
    # site-local. cohort_meta_by_id_imvhr.parquet event_dttm is also UTC.
    # Local-hour extraction is centralized in `_utils.add_dh_hr` (single
    # source of truth — same helper produces the cohort grid's `_dh`/`_hr`
    # via `add_day_shift_id`), so DuckDB's session timezone never affects
    # the result.

    CONFIG_PATH = "config/config.json"
    return (
        CONFIG_PATH,
        apply_outlier_handling_duckdb,
        convert_dose_units_by_med_category,
        duckdb,
        get_config_or_params,
        mar_action_not_given_filter_sql,
        mar_action_zero_dose_sql,
        normalize_categories,
        remove_meds_duplicates,
        setup_logging,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params, setup_logging):
    # Site-scoped output dir (see Makefile SITE= flag).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    DATA_DIR = cfg['data_directory']
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    # Per-site log separation: each site writes to output/{site}/logs/
    # clifpy_all.log + clifpy_errors.log. setup_logging is idempotent; we
    # call it explicitly here (instead of via ClifOrchestrator's __init__
    # side effect) so the output_directory is site-scoped.
    setup_logging(output_directory=f"output_to_share/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return DATA_DIR, SITE_NAME, SITE_TZ


@app.cell
def _(SITE_NAME, duckdb):
    # Lazy parquet read — DuckDB scans on demand. The grid is UTC tagged
    # at write time by 01_cohort's to_utc boundary, so event_dttm carries
    # canonical UTC on disk; downstream SQL uses AT TIME ZONE for explicit
    # local-hour extraction.
    cohort_meta_by_id_imvhr = duckdb.sql(
        f"FROM 'output/{SITE_NAME}/cohort_meta_by_id_imvhr.parquet' SELECT *"
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = cohort_meta_by_id_imvhr.count("*").fetchone()[0]
        logger.debug(f"Hourly grid rows: {_n:,}")
    else:
        logger.info("Hourly grid loaded as lazy relation")
    return (cohort_meta_by_id_imvhr,)


@app.cell
def _(cohort_meta_by_id_imvhr, duckdb):
    # Lazy one-column relation for downstream INNER JOINs. Stays as a
    # DuckDBPyRelation through the chain (no .fetchall() → Python list
    # → giant SQL literal round-trip). Scales to 1M+ hospitalizations
    # without per-query statement-length blowup.
    cohort_hosp_ids_rel = duckdb.sql(
        "FROM cohort_meta_by_id_imvhr SELECT DISTINCT hospitalization_id"
    )
    _n_hosp = cohort_hosp_ids_rel.count("*").fetchone()[0]
    logger.info(f"Cohort hospitalizations: {_n_hosp:,}")
    return (cohort_hosp_ids_rel,)


@app.cell
def _(cohort_meta_by_id_imvhr):
    # Per-patient IMV-start timestamp — the hour-sharp boundary where the
    # cohort's first IMV grid row sits. Used downstream by the ASOF starter
    # cell to inject a "rate at intubation" event so the first IMV hour
    # reflects any propofol/etc. carry-over from the pre-intubation period
    # (ED, OR, floor) rather than starting from rate=0 until the next
    # charted admin event.
    #
    # No daily-registry join needed — the hourly grid alone is the source
    # of truth for IMV time. Verified: every `event_dttm` in the grid is
    # hour-sharp at local-tz (minute=second=0), so injecting a synthetic
    # admin row at this timestamp matches the grid row exactly through the
    # FULL JOIN below.
    imv_starts = mo.sql(
        f"""
        FROM cohort_meta_by_id_imvhr
        SELECT hospitalization_id
            , imv_start_dttm: MIN(event_dttm)
        GROUP BY hospitalization_id
        """
    )
    return (imv_starts,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous Sedation

    Load vitals (weight_kg) for unit conversion, then load, dedup, convert,
    and pivot continuous sedation administrations.
    """)
    return


@app.cell
def _(DATA_DIR, apply_outlier_handling_duckdb, cohort_hosp_ids_rel, duckdb):
    # Inline raw DuckDB read of clifpy's parquet — bypasses
    # `clifpy.load_data` (which silently does UTC→naive-site-local via
    # `timezone(site_tz, col)` and breaks the project's downstream
    # `AT TIME ZONE site_tz + extract` pattern). Returns recorded_dttm as
    # UTC TIMESTAMPTZ — same as what's actually on disk in the parquet.
    # Predicate + projection pushdown engages via parquet zonemap.
    # See docs/timezone_audit.md.
    vitals_rel = duckdb.sql(f"""
        FROM '{DATA_DIR}/clif_vitals.parquet' v
        JOIN cohort_hosp_ids_rel USING (hospitalization_id)
        SELECT
            hospitalization_id
            , v.recorded_dttm
            , v.vital_category
            , v.vital_value
        WHERE v.vital_category = 'weight_kg'
    """)
    vitals_rel = apply_outlier_handling_duckdb(
        vitals_rel, 'vitals', 'config/outlier_config.yaml',
    )
    logger.info("Vitals (weight_kg): lazy relation built (raw DuckDB read, UTC TIMESTAMPTZ)")
    return (vitals_rel,)


@app.cell
def _(DATA_DIR, duckdb):
    # B5: detect mar_action_category presence on each medication table.
    # Asymmetric contract:
    #   - continuous: column is OPTIONAL — fall back to mar_action_name regex.
    #     FFILL semantics are forgiving (a missed stop event over-FFILLs by
    #     minutes, not hours).
    #   - intermittent: column is REQUIRED — bolus dose accounting depends on
    #     the structured 'not_given' zeroing, and free-text mar_action_name
    #     varies too widely across sites to regex reliably.
    _cont_path = f"{DATA_DIR}/clif_medication_admin_continuous.parquet"
    _intm_path = f"{DATA_DIR}/clif_medication_admin_intermittent.parquet"
    _cont_cols = {r[0] for r in duckdb.sql(f"DESCRIBE FROM '{_cont_path}'").fetchall()}
    _intm_cols = {r[0] for r in duckdb.sql(f"DESCRIBE FROM '{_intm_path}'").fetchall()}
    HAS_MAR_CAT_CONT = 'mar_action_category' in _cont_cols
    HAS_MAR_CAT_INTM = 'mar_action_category' in _intm_cols
    if not HAS_MAR_CAT_INTM:
        logger.error(
            "medication_admin_intermittent is missing required column "
            "'mar_action_category'. This column is required for accurate "
            "bolus dose accounting (zeroing of charted-but-not-administered "
            "doses). Refer to the CLIF medication table specification."
        )
        raise RuntimeError(
            "medication_admin_intermittent missing 'mar_action_category' — "
            "see CLIF spec; mar_action_name regex fallback is not safe for "
            "intermittent dose totals."
        )
    if HAS_MAR_CAT_CONT:
        logger.info("medication_admin_continuous: mar_action_category present (preferred path)")
    else:
        logger.warning(
            "medication_admin_continuous: mar_action_category missing — "
            "falling back to mar_action_name regex for stop/not_given detection. "
            "Acceptable for continuous FFILL semantics."
        )
    return (HAS_MAR_CAT_CONT,)


@app.cell
def _(
    DATA_DIR,
    HAS_MAR_CAT_CONT,
    duckdb,
    mar_action_not_given_filter_sql,
    normalize_categories,
):
    # Inline raw DuckDB read — bypasses clifpy.load_data (see vitals cell
    # comment + docs/timezone_audit.md). admin_dttm flows downstream as
    # UTC TIMESTAMPTZ. Predicate + projection pushdown engages.
    # B5: mar_action_category SELECT and WHERE both branch on column presence.
    _mar_cat_select = (
        ", c.mar_action_category" if HAS_MAR_CAT_CONT
        else ", CAST(NULL AS VARCHAR) AS mar_action_category"
    )
    _not_given_filter = mar_action_not_given_filter_sql(HAS_MAR_CAT_CONT)
    cont_sed_rel = duckdb.sql(f"""
        FROM '{DATA_DIR}/clif_medication_admin_continuous.parquet' c
        JOIN cohort_hosp_ids_rel USING (hospitalization_id)
        SELECT
            hospitalization_id
            , c.admin_dttm
            , c.med_name
            , c.med_category
            , c.med_dose
            , c.med_dose_unit
            , c.mar_action_name
            {_mar_cat_select}
        WHERE c.med_category IN ('propofol', 'fentanyl', 'midazolam', 'lorazepam', 'hydromorphone')
            AND c.med_dose IS NOT NULL
            AND {_not_given_filter}
    """)
    # Normalize category casing (cross-site safety). If mar_action_category
    # is the synthetic NULL column, normalize is a no-op for it.
    cont_sed_rel = normalize_categories(
        cont_sed_rel, ['med_category', 'mar_action_category']
    )
    if logger.isEnabledFor(logging.DEBUG):
        _n = cont_sed_rel.count("*").fetchone()[0]
        logger.debug(f"Continuous sedation: {_n:,} non-null-dose rows")
    else:
        logger.info("Continuous sedation: lazy relation built")
    return (cont_sed_rel,)


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
    #
    # `CONT_SED_PREFERRED_UNITS` is hoisted to setup scope so it's the
    # single source of truth for both this conversion AND the pivot's
    # IN list (line ~650 cont_sed_w) that guarantees the expected
    # columns exist even for sites without all 5 drugs.
    cont_sed_converted_rel, cont_sed_convert_summary = convert_dose_units_by_med_category(
        cont_sed_with_weight,
        preferred_units=CONT_SED_PREFERRED_UNITS,
        override=True,
        return_rel=True,
    )
    return cont_sed_convert_summary, cont_sed_converted_rel


@app.cell
def _(cont_sed_convert_summary):
    cont_sed_convert_summary
    return


@app.cell
def _(SITE_NAME, cont_sed_convert_summary):
    # Persist the convert summary to output_to_share/{site}/qc/ so sites'
    # uploaded bundles include per-(drug, source-unit) conversion counts.
    # Federation-safe: group-level counts only, no IDs.
    _qc_dir = Path(f"output_to_share/{SITE_NAME}/qc")
    _qc_dir.mkdir(parents=True, exist_ok=True)
    cont_sed_convert_summary.pl().write_csv(
        _qc_dir / "cont_sed_convert_summary.csv"
    )
    logger.info(f"Saved: {_qc_dir}/cont_sed_convert_summary.csv")
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
def _(SITE_TZ, cont_sed_outliered):
    # Pre-derive _dh (local-tz hour-floor) and _hr on the admin side via
    # the canonical helper. Lets the downstream `_within_imv` JOIN reduce
    # to a clean `USING (hospitalization_id, _dh)` instead of inlining a
    # `date_trunc(... AT TIME ZONE ...)` in the ON clause — single source
    # of truth for the local-tz extraction lives in _utils.add_dh_hr.
    from _utils import add_dh_hr as _add_dh_hr
    cont_sed_with_dh = _add_dh_hr(cont_sed_outliered, 'admin_dttm', site_tz=SITE_TZ)
    return (cont_sed_with_dh,)


@app.cell
def _(cohort_meta_by_id_imvhr, cont_sed_with_dh):
    # Filter admin events to those whose hour bucket falls inside the IMV
    # grid. INNER JOIN on `_dh` — admins whose `_dh` (pre-derived above)
    # isn't in the grid get dropped. Eliminates off-IMV admins that would
    # otherwise let `LEAD(event_dttm)` in `cont_sed_filled` jump across an
    # extubation gap and credit a giant `_duration` to a single in-IMV
    # hour (the bug that the previous-round duration-cap was masking).
    #
    # `SELECT DISTINCT hospitalization_id, _dh` on the grid side guards
    # against DST fall-back day uniqueness: the grid has TWO rows with
    # the same `_dh` on fall-back hours (e.g., 2024-11-03 01:00 CDT and
    # 01:00 CST). Plain INNER JOIN would double admin events; DISTINCT
    # collapses the grid side to one key per hour, preserving prior
    # SEMI JOIN semantics.
    cont_sed_within_imv = mo.sql(
        f"""
        FROM cont_sed_with_dh c
        JOIN (
            FROM cohort_meta_by_id_imvhr SELECT DISTINCT hospitalization_id, _dh
        ) g USING (hospitalization_id, _dh)
        SELECT c.*
        """
    )
    if logger.isEnabledFor(logging.DEBUG):
        # Both cont_sed_with_dh and cont_sed_within_imv are DuckDBPyRelation
        # (sql_output="native"). Use .count("*").fetchone() for row counts —
        # a cheap scalar query that doesn't materialize the relation.
        _n_before = cont_sed_with_dh.count("*").fetchone()[0]
        _n_after = cont_sed_within_imv.count("*").fetchone()[0]
        logger.debug(
            f"cont_sed within-IMV filter: {_n_before:,} → {_n_after:,} rows "
            f"({(_n_before - _n_after) / _n_before * 100 if _n_before else 0.0:.2f}% off-IMV dropped)"
        )
    return (cont_sed_within_imv,)


@app.cell
def _(cont_sed_outliered, imv_starts):
    # ASOF backward join: for each patient's IMV-start timestamp, look up
    # the most recent (post-conversion, post-outlier-handling) admin row
    # with admin_dttm <= imv_start_dttm. Inject as a synthetic admin event
    # at the IMV-start hour boundary (e.g., 5:00:00) so the first IMV hour
    # carries the carry-over rate from before intubation rather than
    # starting at 0 until the next charted admin lands.
    #
    # Sources from `cont_sed_outliered` (NOT `cont_sed_within_imv`) — we
    # specifically need the pre-IMV history that the within-IMV filter
    # would otherwise discard.
    #
    # `mar_action_name` is overwritten to a synthetic provenance tag so a
    # downstream consumer can identify these rows; `mar_action_category`
    # is preserved verbatim so the cont_sed_long CASE still zeros out
    # 'stop' carry-overs (patient was OFF the drug at IMV start).
    cont_sed_starter_rates = mo.sql(
        f"""
        FROM imv_starts e
        ASOF LEFT JOIN cont_sed_outliered c
            ON c.hospitalization_id = e.hospitalization_id
            AND c.admin_dttm <= e.imv_start_dttm
        SELECT
            e.hospitalization_id
            , admin_dttm: e.imv_start_dttm
            , c.med_name
            , c.med_category
            , c.med_dose
            , c.med_dose_unit
            , c.med_dose_original
            , c.med_dose_unit_original
            , mar_action_name: 'asof_starter'
            , c.mar_action_category
        WHERE c.med_dose IS NOT NULL
        """
    )
    if logger.isEnabledFor(logging.DEBUG):
        # DuckDBPyRelation (sql_output="native") — scalar count via .count().
        _n_starter = cont_sed_starter_rates.count("*").fetchone()[0]
        logger.debug(f"cont_sed ASOF starter rates: {_n_starter:,} rows injected")
    return (cont_sed_starter_rates,)


@app.cell
def _(
    HAS_MAR_CAT_CONT,
    cont_sed_starter_rates,
    cont_sed_within_imv,
    mar_action_zero_dose_sql,
):
    # UNION ALL of the within-IMV admin events and the per-patient starter
    # rates (one synthetic admin row per patient at their IMV-start
    # timestamp). The `BY NAME` qualifier matches columns by name rather
    # than position, so column order doesn't have to align between the
    # two relations.
    # B5: zero-dose CASE uses helper so sites without mar_action_category
    # fall back to mar_action_name regex.
    _zero_cond = mar_action_zero_dose_sql(HAS_MAR_CAT_CONT)
    cont_sed_long = mo.sql(
        f"""
        WITH unioned AS (
            FROM cont_sed_within_imv
            UNION ALL BY NAME
            FROM cont_sed_starter_rates
        )
        FROM unioned
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit:
                med_category || '_'
                || REPLACE(med_dose_unit, '/', '_')
                || '_cont'
            , med_dose: CASE WHEN {_zero_cond} THEN 0 ELSE med_dose END
        """
    )
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
    #
    # `ON med_category_unit IN (...)` — explicit value list pinned from
    # CONT_EXPECTED_PIVOT_COLS (derived from CONT_SED_PREFERRED_UNITS in
    # setup scope). Guarantees all 5 expected drug columns are produced
    # even at sites with zero rows for one of the drugs; missing drugs
    # become NULL columns that COALESCE down to 0 through the existing
    # cont chain. Without this, downstream SQL referencing the drug by
    # name (cont_sed_per_event:757-761, merge cell:1099-1119) would crash
    # with `Referenced column not found`.
    _cont_in_list = ", ".join(f"'{c}'" for c in CONT_EXPECTED_PIVOT_COLS)
    cont_sed_w = mo.sql(
        f"""
        PIVOT_WIDER cont_sed_long
        ON med_category_unit IN ({_cont_in_list})
        USING AVG(med_dose)
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_w,)


@app.cell
def _(cont_sed_w, duckdb):
    # Diagnostics: log WARN per all-NULL drug column. Cheap scalar SQL on
    # the lazy relation (5 COUNT-FILTER queries). Pure observability — no
    # materialization, no downstream effect. A NULL-only column means the
    # site has zero post-conversion rows for that drug; downstream COALESCE
    # treats it as zero exposure but operators may want to verify upstream
    # ETL (e.g., a misspelled med_category mapping) rather than silently
    # accept it.
    for _col in CONT_EXPECTED_PIVOT_COLS:
        _n_non_null = duckdb.sql(
            f"FROM cont_sed_w SELECT COUNT(*) FILTER (WHERE {_col} IS NOT NULL)"
        ).fetchone()[0]
        if _n_non_null == 0:
            _drug = _col.split('_')[0]
            logger.warning(
                f"Drug column '{_col}' is all NULL — this site has zero "
                f"continuous {_drug} admin rows in the cohort. Downstream "
                f"treats as zero exposure. Verify upstream ETL "
                f"(med_category mapping, outlier filter) if this is "
                f"unexpected."
            )
    return


@app.cell
def _(cohort_meta_by_id_imvhr, cont_sed_w):
    # FULL JOIN with hourly grid: the grid side anchors the timeline at hour
    # boundaries so forward-fill below can propagate dose rates across hours
    # with no admin event. We subset the grid to just the join keys —
    # downstream cells only need (hosp_id, event_dttm, _dh, _hr, *_cont).
    # `_shift`/`_nth_day`/etc. on cohort_meta_by_id_imvhr are re-joined later at
    # the per-hour merge step (seddose_by_id_imvhr).
    #
    # This cell is the bare FULL JOIN; the next cell adds `_dh`/`_hr` via
    # the canonical helper. Split per CLAUDE.md's "one operation per cell"
    # pattern — the optimizer fuses adjacent cells at the terminal sink.
    cont_sed_wg_raw = mo.sql(
        f"""
        WITH grid AS (
            FROM cohort_meta_by_id_imvhr
            SELECT hospitalization_id, event_dttm
        )
        FROM grid g
        FULL JOIN cont_sed_w m USING (hospitalization_id, event_dttm)
        SELECT * EXCLUDE (event_dttm), event_dttm
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (cont_sed_wg_raw,)


@app.cell
def _(SITE_TZ, cont_sed_wg_raw):
    # Add `_dh`/`_hr` to the joined relation via the canonical helper.
    # `event_dttm` may arrive as site-tz tagged or UTC tz-aware; DuckDB
    # normalizes both to TIMESTAMPTZ internally and `AT TIME ZONE` (inside
    # add_dh_hr) operates on the underlying instant, so the result is
    # correct regardless of which path supplied the row.
    from _utils import add_dh_hr as _add_dh_hr
    cont_sed_wg = _add_dh_hr(cont_sed_wg_raw, 'event_dttm', site_tz=SITE_TZ)
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
    cont_sed_filled = mo.sql(
        f"""
        -- Forward-fill continuous sedation rates and compute duration between events.
        -- `_duration` for the last event of a partition is 0 (LEAD's third arg
        -- falls back to event_dttm itself, yielding zero diff).
        --
        -- No `LEAST(_duration, 60)` cap is needed here: cont_sed_within_imv
        -- restricts admin events to those whose hour bucket is in the IMV
        -- grid, and cont_sed_wg's FULL JOIN with the grid puts a row at
        -- every IMV hour boundary. So LEAD(event_dttm) is always ≤ 60 min
        -- away within IMV time, and at the partition's last row LEAD's
        -- third-arg fallback yields _duration = 0 — there are no off-IMV
        -- trailing rows for the multiplication to blow up.
        FROM cont_sed_wg g
        SELECT hospitalization_id, event_dttm, _dh, _hr
            , LAST_VALUE(COLUMNS('_cont') IGNORE NULLS) OVER (
                PARTITION BY hospitalization_id ORDER BY event_dttm
            )
            , _duration: EXTRACT(EPOCH FROM (LEAD(event_dttm, 1, event_dttm) OVER w - event_dttm)) / 60.0
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        ORDER BY hospitalization_id, event_dttm
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
            -- now the actual units are mcg, mg, or mcg/kg
            , fentanyl_mcg_cont: fentanyl_mcg_min_cont * _duration
            , hydromorphone_mg_cont: hydromorphone_mg_min_cont * _duration
            , lorazepam_mg_cont: lorazepam_mg_min_cont * _duration
            , midazolam_mg_cont: midazolam_mg_min_cont * _duration
            , propofol_mcg_kg_cont: propofol_mcg_kg_min_cont * _duration
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
            , _duration_within_hr: SUM(_duration)
        GROUP BY hospitalization_id, _dh, _hr
        ORDER BY hospitalization_id, _dh
        """
    )
    return (cont_sed_dose_by_hr,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Intermittent Sedation

    Same pattern as continuous, but `mar_action_category` filters and the
    unit-conversion targets differ.
    """)
    return


@app.cell
def _(DATA_DIR, cohort_hosp_ids_rel, duckdb, normalize_categories):
    # Inline raw DuckDB read — same pattern as cont_sed_rel above. admin_dttm
    # flows downstream as UTC TIMESTAMPTZ. See docs/timezone_audit.md.
    # B5: mar_action_category is REQUIRED here — the detection cell above
    # (HAS_MAR_CAT_INTM) has already asserted presence and raised otherwise.
    intm_sed_rel = duckdb.sql(f"""
        FROM '{DATA_DIR}/clif_medication_admin_intermittent.parquet' c
        JOIN cohort_hosp_ids_rel USING (hospitalization_id)
        SELECT
            hospitalization_id
            , c.admin_dttm
            , c.med_name
            , c.med_category
            , c.med_dose
            , c.med_dose_unit
            , c.mar_action_name
            , c.mar_action_category
        WHERE c.med_category IN ('propofol', 'fentanyl', 'midazolam', 'lorazepam', 'hydromorphone')
            AND c.med_dose IS NOT NULL
            AND c.mar_action_category != 'not_given'
    """)
    intm_sed_rel = normalize_categories(
        intm_sed_rel, ['med_category', 'mar_action_category']
    )
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
    # Intermittent bolus doses. Propofol targets `mcg/kg` (weight-adjusted)
    # so the converted intm column drops into `prop_mcg_kg_min_total`'s sum
    # alongside continuous; the other four drugs stay at weight-free amount
    # units because they roll into fenteq/midazeq equivalency sums in
    # absolute units, not per-kg. `vitals_df=vitals_rel` provides the
    # per-bolus ASOF weight that clifpy's mcg/kg converter needs.
    #
    # `INTM_SED_PREFERRED_UNITS` is hoisted to setup scope (see header) —
    # single source of truth for both this conversion AND the intm pivot's
    # IN list at intm_sed_w.
    intm_sed_converted_rel, intm_sed_convert_summary = convert_dose_units_by_med_category(
        intm_sed_deduped,
        vitals_df=vitals_rel,
        preferred_units=INTM_SED_PREFERRED_UNITS,
        override=True,
        return_rel=True,
    )
    return intm_sed_convert_summary, intm_sed_converted_rel


@app.cell
def _(intm_sed_convert_summary):
    intm_sed_convert_summary
    return


@app.cell
def _(SITE_NAME, intm_sed_convert_summary):
    # Persist the intm convert summary alongside the cont one. Same QC
    # purpose: cross-site verification of unit-mapping per drug.
    _qc_dir = Path(f"output_to_share/{SITE_NAME}/qc")
    _qc_dir.mkdir(parents=True, exist_ok=True)
    intm_sed_convert_summary.pl().write_csv(
        _qc_dir / "intm_sed_convert_summary.csv"
    )
    logger.info(f"Saved: {_qc_dir}/intm_sed_convert_summary.csv")
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
def _(SITE_TZ, intm_sed_outliered):
    # Pre-derive _dh / _hr on the intm admin side via the canonical
    # helper (mirror of cont_sed_with_dh upstream). Lets the downstream
    # `_within_imv` JOIN reduce to a clean `USING (hospitalization_id,
    # _dh)`.
    from _utils import add_dh_hr as _add_dh_hr
    intm_sed_with_dh = _add_dh_hr(intm_sed_outliered, 'admin_dttm', site_tz=SITE_TZ)
    return (intm_sed_with_dh,)


@app.cell
def _(cohort_meta_by_id_imvhr, intm_sed_with_dh):
    # Mirror of `cont_sed_within_imv` — filter intermittent admin events
    # to those whose hour bucket falls inside the IMV grid. Intermittent
    # doesn't have the `LEAD`-based duration leak (each bolus is its own
    # row, no forward-fill, no rate × duration), so this filter is mainly
    # for symmetry with the cont path. No ASOF starter for intm: boluses
    # are point-in-time events, not rates that need to "carry forward".
    #
    # `SELECT DISTINCT hospitalization_id, _dh` guards against DST
    # fall-back day uniqueness on the grid side (see cont_sed_within_imv
    # comment for the full explanation).
    intm_sed_within_imv = mo.sql(
        f"""
        FROM intm_sed_with_dh c
        JOIN (
            FROM cohort_meta_by_id_imvhr SELECT DISTINCT hospitalization_id, _dh
        ) g USING (hospitalization_id, _dh)
        SELECT c.*
        """
    )
    return


@app.cell
def _():
    # Pivot intermittent sedation to wide format. F4 fix: the column-name
    # builder embeds `_hr_intm` directly so the SUM-per-hour step
    # downstream produces final names without rename. After SUM-per-hour,
    # `propofol_mcg_kg_intm` (post F2: weight-adjusted) and the other
    # `*_intm` columns represent the amount delivered that hour from
    # intermittent admin.
    #
    # `ON med_category_unit IN (...)` — explicit value list pinned from
    # INTM_EXPECTED_PIVOT_COLS (derived from INTM_SED_PREFERRED_UNITS in
    # setup scope). Same robustness guarantee as the cont pivot: missing
    # drugs become NULL columns, downstream COALESCE treats as zero.
    _intm_in_list = ", ".join(f"'{c}'" for c in INTM_EXPECTED_PIVOT_COLS)
    intm_sed_w = mo.sql(
        f"""
        WITH t1 AS (
        FROM intm_sed_within_imv
        SELECT hospitalization_id
            , admin_dttm AS event_dttm
            , med_category_unit:
                med_category || '_'
                || REPLACE(med_dose_unit, '/', '_')
                || '_intm'
            , med_dose: CASE WHEN mar_action_category = 'not_given' THEN 0 ELSE med_dose END
        )
        PIVOT_WIDER t1
        ON med_category_unit IN ({_intm_in_list})
        USING AVG(med_dose)
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_w,)


@app.cell
def _(duckdb, intm_sed_w):
    # Diagnostics: WARN per all-NULL intm drug column. Mirror of the cont
    # cell above — cheap scalar COUNT-FILTER queries, no materialization.
    for _col in INTM_EXPECTED_PIVOT_COLS:
        _n_non_null = duckdb.sql(
            f"FROM intm_sed_w SELECT COUNT(*) FILTER (WHERE {_col} IS NOT NULL)"
        ).fetchone()[0]
        if _n_non_null == 0:
            _drug = _col.split('_')[0]
            logger.warning(
                f"Drug column '{_col}' is all NULL — this site has zero "
                f"intermittent {_drug} admin rows in the cohort. Downstream "
                f"treats as zero exposure. Verify upstream ETL "
                f"(med_category mapping, outlier filter) if this is "
                f"unexpected."
            )
    return


@app.cell
def _(cohort_meta_by_id_imvhr, intm_sed_w):
    # FULL JOIN with hourly grid — same pattern as cont_sed_wg_raw.
    # `_dh`/`_hr` are added in the next cell via the canonical helper.
    intm_sed_wg_raw = mo.sql(
        f"""
        WITH grid AS (
            FROM cohort_meta_by_id_imvhr
            SELECT hospitalization_id, event_dttm
        )
        FROM grid g
        FULL JOIN intm_sed_w m USING (hospitalization_id, event_dttm)
        SELECT * EXCLUDE (event_dttm), event_dttm
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (intm_sed_wg_raw,)


@app.cell
def _(SITE_TZ, intm_sed_wg_raw):
    from _utils import add_dh_hr as _add_dh_hr
    intm_sed_wg = _add_dh_hr(intm_sed_wg_raw, 'event_dttm', site_tz=SITE_TZ)
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
def _(cohort_meta_by_id_imvhr, cont_sed_dose_by_hr, intm_sed_dose_by_hr):
    # Merge per-hour cont + intm onto the hourly grid; expose per-hour
    # AVG RATES with explicit unit names (no cumulative-amount columns).
    #
    # Inputs from cont_sed_dose_by_hr are "amount delivered in this 60-min
    # window" (rate × duration, summed). Dividing by 60 (for /min rates) or
    # passing through (for /hr rates, since amount-in-an-hour numerically
    # equals avg-rate-in-mcg/hr) yields the per-hour avg rate. intm inputs
    # are amount-in-the-hour (sum of bolus mg/mcg), which equals the per-
    # hour rate in mg/hr or mcg/hr by the same argument.
    #
    # Both `_cont` (continuous-only) and `_total` (cont + intm) variants
    # are exposed so consumers can pick the analytical scope.
    #
    # Propofol: cont is mcg/kg/min infusion rate; intm bolus mg is converted
    # to mcg/kg upstream via clifpy's weight-adjusted unit converter (using
    # the per-bolus ASOF weight). Both terms are then in per-hour mcg/kg
    # units (cont = rate × duration summed; intm = sum of bolus mcg/kg),
    # so they add cleanly. Dividing by 60 re-rates the hourly sum to a
    # mcg/kg/min equivalent — the standard simplification of treating
    # bolus dose as distributed across the hour.
    seddose_by_id_imvhr = mo.sql(
        f"""
        WITH joined AS (
            FROM cohort_meta_by_id_imvhr g
            LEFT JOIN intm_sed_dose_by_hr i USING (hospitalization_id, _dh)
            LEFT JOIN cont_sed_dose_by_hr c USING (hospitalization_id, _dh)
            -- intm_sed_dose_by_hr has no _hr column (it groups on _dh
            -- only); cont_sed_dose_by_hr does. Drop cont's _hr to keep
            -- the registry's canonical one — without this the parquet
            -- ends up with a stray _hr_1 column.
            SELECT g.* EXCLUDE (_hr)
                , g._hr
                , c.* EXCLUDE (hospitalization_id, _dh, _hr)
                , i.* EXCLUDE (hospitalization_id, _dh)
        )
        FROM joined
        SELECT *
            -- Continuous-only per-hour avg rates.
            , prop_mcg_kg_min_cont:  COALESCE(propofol_mcg_kg_cont, 0) / 60.0
            , fenteq_mcg_hr_cont:    COALESCE(hydromorphone_mg_cont, 0) * 50
                                      + COALESCE(fentanyl_mcg_cont, 0)
            , midazeq_mg_hr_cont:    COALESCE(lorazepam_mg_cont, 0) * 2
                                      + COALESCE(midazolam_mg_cont, 0)
            -- Total (cont + intm) per-hour avg rates. Propofol intm is now
            -- weight-adjusted to mcg/kg upstream, so cont (already mcg/kg
            -- per hour) and intm (sum of bolus mcg/kg per hour) add directly;
            -- /60 re-rates to mcg/kg/min averaged over the hour.
            , prop_mcg_kg_min_total: (
                COALESCE(propofol_mcg_kg_cont, 0)
                + COALESCE(propofol_mcg_kg_intm, 0)
              ) / 60.0
            , fenteq_mcg_hr_total:   (COALESCE(hydromorphone_mg_cont, 0)
                                       + COALESCE(hydromorphone_mg_intm, 0)) * 50
                                      + COALESCE(fentanyl_mcg_cont, 0)
                                      + COALESCE(fentanyl_mcg_intm, 0)
            , midazeq_mg_hr_total:   (COALESCE(lorazepam_mg_cont, 0)
                                       + COALESCE(lorazepam_mg_intm, 0)) * 2
                                      + COALESCE(midazolam_mg_cont, 0)
                                      + COALESCE(midazolam_mg_intm, 0)
        ORDER BY hospitalization_id, event_dttm
        """
    )
    return (seddose_by_id_imvhr,)


@app.cell
def _(seddose_by_id_imvhr, duckdb):
    # M1: per-hour clinical-ceiling clamp (cap-to-ceiling, not NULL).
    #
    # WHAT: caps per-hour avg rates at a clinical-implausibility ceiling.
    # Cap-to-ceiling rather than NULL because the downstream
    # `05_modeling_dataset.py` COALESCE(..., 0) wrapper would otherwise
    # silently re-interpret a clamped patient-shift as "no exposure," which
    # is worse than either no-clamp (real signal preserved) or cap-to-ceiling
    # (directional signal preserved, magnitude bounded).
    #
    # WHY per-hour, not per-shift: a partial-shift weight error inflates
    # only some hours; per-shift averaging dilutes that signal past
    # detection. Per-hour clamping lets the surviving hours contribute to
    # the shift average via AVG-ignores-NULL semantics (we cap, not NULL,
    # so this nuance doesn't bite — but the per-hour granularity is still
    # the right place to bound outliers).
    #
    # CEILINGS: chosen at the per-row outlier-handler boundary
    # (config/outlier_config.yaml). For propofol the per-row clamp is
    # 200 mcg/kg/min, so a per-hour AVG of post-clamp values mathematically
    # cannot exceed 200. Setting the M1 ceiling AT that boundary makes the
    # clamp a defense-in-depth net for cases where the per-row clamp is
    # bypassed (e.g., a future change to outlier_config.yaml) without
    # creating false positives on real high-end sustained dosing.
    #
    # TOGGLE: `SEDDOSE_CLAMP=0` makes this cell a pass-through, useful for
    # producing an unclamped comparison run. The raw parquet sibling
    # written at the end of this script lets users diff clamped vs raw at
    # the per-hour level without a second pipeline run.
    import os as _os
    _seddose_clamp_on = _os.getenv("SEDDOSE_CLAMP", "1") == "1"
    _ceilings = {
        'prop_mcg_kg_min_cont':  200,
        'prop_mcg_kg_min_total': 200,
        'fenteq_mcg_hr_cont':   1000,
        'fenteq_mcg_hr_total':  1000,
        'midazeq_mg_hr_cont':     50,
        'midazeq_mg_hr_total':    50,
    }
    if not _seddose_clamp_on:
        logger.info("M1 clinical-ceiling clamp DISABLED (SEDDOSE_CLAMP=0)")
        seddose_by_id_imvhr_clamped = seddose_by_id_imvhr
    else:
        # Step 4: enriched clamp logging. For each ceiling column compute
        # in one scalar SQL pass: count above ceiling, total rows, plus
        # p50/p95/max of the PRE-CLAMP distribution among violators. The
        # pre-clamp percentiles tell operators if the overage is mild
        # (close to ceiling → likely real edge case) vs wild (≫ ceiling
        # → likely weight-error or unit-conversion bug). On-disk parquet
        # caps via LEAST() so stored values never exceed the ceiling —
        # these percentiles describe the ORIGINAL magnitudes.
        _total = duckdb.sql(
            "FROM seddose_by_id_imvhr SELECT COUNT(*)"
        ).fetchone()[0]
        _clamp_rows = []  # for the qc/m1_clamp_summary.csv
        for _col, _ceil in _ceilings.items():
            _stats = duckdb.sql(f"""
                FROM seddose_by_id_imvhr
                SELECT
                    COUNT(*) FILTER (WHERE {_col} > {_ceil})                       AS n_above
                    , quantile_cont({_col}, 0.50) FILTER (WHERE {_col} > {_ceil})  AS p50_above
                    , quantile_cont({_col}, 0.95) FILTER (WHERE {_col} > {_ceil})  AS p95_above
                    , MAX({_col})                  FILTER (WHERE {_col} > {_ceil}) AS max_above
            """).fetchone()
            _n_above, _p50_above, _p95_above, _max_above = _stats
            _pct = _n_above / _total * 100 if _total else 0
            _clamp_rows.append({
                'column': _col, 'ceiling': _ceil,
                'n_above': _n_above, 'n_total': _total, 'pct_above': round(_pct, 4),
                'p50_above': _p50_above, 'p95_above': _p95_above, 'max_above': _max_above,
            })
            if _n_above > 0:
                logger.warning(
                    f"M1 clamp [{_col}]: {_n_above:,} / {_total:,} ({_pct:.3f}%) "
                    f"above ceiling {_ceil}. Pre-clamp distribution among "
                    f"violators — p50={_p50_above:.1f}, p95={_p95_above:.1f}, "
                    f"max={_max_above:.1f}. (On disk: LEAST(col, {_ceil}) — "
                    f"parquet caps at {_ceil}; these percentiles describe "
                    f"ORIGINAL magnitudes to help diagnose severity.) "
                    f"p50 close to {_ceil} → likely real edge case; p50 ≫ "
                    f"{_ceil} → likely weight-error or unit-conversion bug. "
                    f"Investigate if proportion > 0.5%."
                )
            else:
                logger.info(
                    f"M1 clamp [{_col}]: 0 / {_total:,} above ceiling {_ceil} "
                    f"— no cap applied."
                )
        # Persist structured summary CSV — federation-safe (per-ceiling
        # aggregates only, no IDs).
        import polars as _pl_clamp_qc
        _qc_dir = Path(f"output_to_share/{SITE_NAME}/qc")
        _qc_dir.mkdir(parents=True, exist_ok=True)
        _pl_clamp_qc.DataFrame(_clamp_rows).write_csv(
            _qc_dir / "m1_clamp_summary.csv"
        )
        logger.info(f"Saved: {_qc_dir}/m1_clamp_summary.csv")
        _replace_clauses = ", ".join(
            f"LEAST({col}, {ceil}) AS {col}" for col, ceil in _ceilings.items()
        )
        # Stay lazy — return a DuckDBPyRelation. Downstream sed_dose_agg
        # consumes it via mo.sql replacement scan; the terminal write cell
        # materializes to Polars only at the parquet-write boundary.
        seddose_by_id_imvhr_clamped = duckdb.sql(f"""
            FROM seddose_by_id_imvhr
            SELECT * REPLACE ({_replace_clauses})
        """)
        logger.info("M1 clinical-ceiling clamp APPLIED (SEDDOSE_CLAMP=1)")
    return (seddose_by_id_imvhr_clamped,)


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
def _(seddose_by_id_imvhr_clamped):
    # Per-shift avg rate = AVG of hourly rates within the shift. Algebraic
    # equivalence (each hour is the same 60-min length):
    #   total drug = Σ(rate_h × 60); shift duration = n_hours × 60
    #   shift avg rate = total / duration
    #                  = (60 · Σ rate_h) / (n_hours · 60)
    #                  = AVG(rate_h)
    # so the SUM-then-divide-by-(n_hours · 60) pattern used previously
    # collapses to plain AVG(). n_hours stays in the output as coverage
    # metadata (single-shift days at intubation/extubation edges) but is
    # no longer an arithmetic input downstream.
    #
    # Both `_cont` (continuous-only) and `_total` (cont + intm) variants
    # propagate; consumers pick analytical scope at read time.
    # Reads the clamped per-hour relation from the M1 cell above. Set
    # SEDDOSE_CLAMP=0 to bypass clamping (M1 becomes a pass-through, this
    # AVG runs over raw per-hour values).
    sed_dose_agg = mo.sql(
        f"""
        FROM seddose_by_id_imvhr_clamped
        SELECT hospitalization_id, _nth_day, _shift
            , prop_mcg_kg_min_cont:  AVG(prop_mcg_kg_min_cont)
            , prop_mcg_kg_min_total: AVG(prop_mcg_kg_min_total)
            , fenteq_mcg_hr_cont:    AVG(fenteq_mcg_hr_cont)
            , fenteq_mcg_hr_total:   AVG(fenteq_mcg_hr_total)
            , midazeq_mg_hr_cont:    AVG(midazeq_mg_hr_cont)
            , midazeq_mg_hr_total:   AVG(midazeq_mg_hr_total)
            , n_hours:               COUNT(*)
        GROUP BY hospitalization_id, _nth_day, _shift
        ORDER BY hospitalization_id, _nth_day, _shift
        """
    )
    return (sed_dose_agg,)


@app.cell
def _(sed_dose_agg):
    # Pivot _shift to wide day/night columns. Each (hospitalization_id,
    # _nth_day, _shift) group has exactly one row in sed_dose_agg, so
    # MAX(...) FILTER (WHERE _shift = 'day') just picks that row's value
    # for the day shift (and likewise night) — same idiom as before but
    # on rate columns instead of cumulative ones, which is why the
    # aggregation operator changes from SUM to MAX (you can't sum a single
    # rate value, but the choice is moot since there's only one row to
    # collapse).
    #
    # n_hours_day / n_hours_night carry through as coverage metadata —
    # downstream `05_modeling_dataset.py` no longer divides by them
    # because the rate columns are already shift-avg rates.
    #
    # M1 clamping (if SEDDOSE_CLAMP=1) was applied upstream at the per-hour
    # layer, so sed_dose_agg here already reflects clamped inputs.
    seddose_by_id_imvday = mo.sql(
        f"""
        FROM sed_dose_agg
        SELECT hospitalization_id, _nth_day
            , prop_day_mcg_kg_min_cont:    MAX(prop_mcg_kg_min_cont)  FILTER (WHERE _shift = 'day')
            , prop_night_mcg_kg_min_cont:  MAX(prop_mcg_kg_min_cont)  FILTER (WHERE _shift = 'night')
            , prop_day_mcg_kg_min_total:   MAX(prop_mcg_kg_min_total) FILTER (WHERE _shift = 'day')
            , prop_night_mcg_kg_min_total: MAX(prop_mcg_kg_min_total) FILTER (WHERE _shift = 'night')
            , fenteq_day_mcg_hr_cont:      MAX(fenteq_mcg_hr_cont)    FILTER (WHERE _shift = 'day')
            , fenteq_night_mcg_hr_cont:    MAX(fenteq_mcg_hr_cont)    FILTER (WHERE _shift = 'night')
            , fenteq_day_mcg_hr_total:     MAX(fenteq_mcg_hr_total)   FILTER (WHERE _shift = 'day')
            , fenteq_night_mcg_hr_total:   MAX(fenteq_mcg_hr_total)   FILTER (WHERE _shift = 'night')
            , midazeq_day_mg_hr_cont:      MAX(midazeq_mg_hr_cont)    FILTER (WHERE _shift = 'day')
            , midazeq_night_mg_hr_cont:    MAX(midazeq_mg_hr_cont)    FILTER (WHERE _shift = 'night')
            , midazeq_day_mg_hr_total:     MAX(midazeq_mg_hr_total)   FILTER (WHERE _shift = 'day')
            , midazeq_night_mg_hr_total:   MAX(midazeq_mg_hr_total)   FILTER (WHERE _shift = 'night')
            , n_hours_day:                 SUM(n_hours) FILTER (WHERE _shift = 'day')
            , n_hours_night:               SUM(n_hours) FILTER (WHERE _shift = 'night')
        GROUP BY hospitalization_id, _nth_day
        ORDER BY hospitalization_id, _nth_day
        """
    )
    return (seddose_by_id_imvday,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save Outputs

    Terminal materialization. UTC-everywhere on disk: every `*_dttm`
    column is written as UTC TIMESTAMPTZ. `seddose_by_id_imvday` has no
    `*_dttm` columns. `seddose_by_id_imvhr` has `event_dttm` (UTC from
    the cohort grid join).

    Under `sql_output="native"`, every upstream `mo.sql` cell returns a
    lazy `DuckDBPyRelation` — the full cont/intm pipelines stay
    unmaterialized through pivot, forward-fill, per-hour aggregation,
    M1 clamp, per-shift aggregation, and pivot-to-day/night. The write
    cell below materializes to Polars exactly ONCE per output file via
    `.pl()`, then applies `convert_time_zone("UTC")` (Polars metadata
    op, no row scan) and calls `.write_parquet(...)` (Polars preserves
    the tz tag). See `docs/timezone_audit.md`.

    Phase 2 (2026-05-07): `sed_dose_agg.parquet` is no longer written.
    Its per-shift dose totals were already mirrored as separate day/night
    columns in `seddose_by_id_imvday.parquet`, so the aggregate file was
    a pure intra-script intermediate with no external readers.
    """)
    return


@app.cell
def _(SITE_NAME, seddose_by_id_imvday, seddose_by_id_imvhr, seddose_by_id_imvhr_clamped):
    import polars as pl  # cell-local — used only for the tz-bearing parquet write

    # Under sql_output="native" (post Step 2), the inputs are all lazy
    # DuckDBPyRelations. Materialize to Polars right at the write boundary
    # via `.pl()` so we can apply `convert_time_zone("UTC")` (Polars
    # metadata op, no row scan) before `.write_parquet(...)`. Polars'
    # parquet writer preserves the tz tag, matching the project's
    # UTC-everywhere convention (docs/timezone_audit.md).
    seddose_by_id_imvday.pl().write_parquet(
        f"output/{SITE_NAME}/seddose_by_id_imvday.parquet"
    )

    # Canonical per-hour parquet: clamped (or pass-through if SEDDOSE_CLAMP=0).
    # This is what every downstream consumer reads (descriptive scripts,
    # 05_modeling_dataset.py via the daily roll-up).
    (
        seddose_by_id_imvhr_clamped.pl()
        .with_columns(pl.col("event_dttm").dt.convert_time_zone("UTC"))
        .write_parquet(f"output/{SITE_NAME}/seddose_by_id_imvhr.parquet")
    )

    # Raw sibling: pre-clamp data, always written so users can diff the two
    # parquets to see the clamp's impact at the patient-hour level without
    # re-running the pipeline with SEDDOSE_CLAMP=0. When SEDDOSE_CLAMP=0 the
    # two files have identical contents (M1 cell is a pass-through).
    (
        seddose_by_id_imvhr.pl()
        .with_columns(pl.col("event_dttm").dt.convert_time_zone("UTC"))
        .write_parquet(f"output/{SITE_NAME}/seddose_by_id_imvhr_raw.parquet")
    )

    logger.info(f"Saved: output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    logger.info(f"Saved: output/{SITE_NAME}/seddose_by_id_imvhr.parquet (event_dttm in UTC, clamp-aware)")
    logger.info(f"Saved: output/{SITE_NAME}/seddose_by_id_imvhr_raw.parquet (event_dttm in UTC, always pre-clamp)")
    return


if __name__ == "__main__":
    app.run()
