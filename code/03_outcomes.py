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

    # Module-level logger (matches the pattern in 02_exposure.py).
    from clifpy.utils.logging_config import get_logger
    logger = get_logger("epi_sedation.outcomes")


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
    from clifpy.utils.config import get_config_or_params
    from _logging_setup import setup_logging
    from _utils import normalize_categories, to_utc
    import pandas as pd
    import duckdb

    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    CONFIG_PATH = "config/config.json"

    os.makedirs("output", exist_ok=True)
    return (
        CONFIG_PATH,
        duckdb,
        get_config_or_params,
        normalize_categories,
        pd,
        setup_logging,
        to_utc,
    )


@app.cell
def _(CONFIG_PATH, get_config_or_params, setup_logging):
    # Site-scoped output dir (see Makefile SITE= flag).
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    # Reintubation classification window. Drives `_fail_extub` and
    # `_fail_extub_v2` below; downstream `04_covariates.py` consumes the
    # already-derived flag to assign `exit_mechanism`. Default 48h
    # matches the ventilator-liberation literature (Esteban, Thille,
    # ABC-trial); set to 24 in the per-site config for sensitivity.
    REINTUB_WINDOW_HRS = cfg['reintub_window_hrs']
    # V2 outcome family (success_extub_v2, sbt_done_v2, exit_mechanism_v2_based)
    # is SENSITIVITY-ONLY — no manuscript primary depends on it. Large sites
    # (≥50K cohort) should set `enable_v2_outcomes: false` to skip the V2
    # state-machine cell entirely (saves ~7 min at NU scale; ~110 min at
    # 200K cohort scale). When false, V2-suffix columns are emitted as
    # constant zero so downstream schema stays stable; 08_models.py skips
    # *_v2_next_day outcome fits explicitly.
    ENABLE_V2_OUTCOMES = bool(cfg.get('enable_v2_outcomes', True))
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    # Per-site dual log files at output/{site}/logs/clifpy_all.log +
    # clifpy_errors.log. Each numbered script runs in its own subprocess,
    # so each must call setup_logging itself (pyCLIF integration guide
    # rule 1). Idempotent; output_directory is site-scoped.
    setup_logging(output_directory=f"output_to_share/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME} (tz: {SITE_TZ}); reintub window: {REINTUB_WINDOW_HRS}h")
    logger.info(f"enable_v2_outcomes: {ENABLE_V2_OUTCOMES}")
    return ENABLE_V2_OUTCOMES, REINTUB_WINDOW_HRS, SITE_NAME, SITE_TZ


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Respiratory Support (processed)
    """)
    return


@app.cell
def _(SITE_NAME, duckdb, normalize_categories):
    resp_processed_path = f"output/{SITE_NAME}/cohort_resp_processed_bf.parquet"
    assert os.path.exists(resp_processed_path), (
        f"Missing {resp_processed_path} — run 01_cohort.py first"
    )
    # Step 2 (scalability): scan via DuckDB instead of materializing the full
    # parquet to pandas. At 1M-source-DB scale this resp_p can be 28-56 GB —
    # pandas OOMs; DuckDB streams. Downstream `FROM resp_p` mo.sql cells work
    # equivalently on DuckDBPyRelation via marimo's replacement scan. The V2
    # state machine cell (if enabled) materializes its own pandas copy via
    # `resp_p.df()`.
    resp_p = duckdb.sql(f"FROM '{resp_processed_path}'")
    # F5 tracheostomy dtype normalization — ported to SQL REPLACE so it
    # composes with the DuckDB relation. Same two-branch logic as the prior
    # pandas implementation: numeric > 0 OR string match in the truthy set.
    # TRY_CAST returns NULL on non-numeric values (instead of throwing), so
    # the numeric branch fails cleanly on VARCHAR sources and the string
    # branch picks them up.
    resp_p = duckdb.sql("""
        FROM resp_p
        SELECT * REPLACE (
            CASE
                WHEN TRY_CAST(tracheostomy AS DOUBLE) > 0 THEN 1
                WHEN LOWER(CAST(tracheostomy AS VARCHAR)) IN ('true', 't', 'yes', 'y') THEN 1
                ELSE 0
            END :: INTEGER AS tracheostomy
        )
    """)
    # Defensive category normalize (polymorphic — DuckDBPyRelation in / out).
    resp_p = normalize_categories(resp_p, ['device_category', 'mode_category'])
    _n_rows = resp_p.count('*').fetchone()[0]
    logger.info(f"resp_p: {_n_rows} rows from {resp_processed_path} (DuckDBPyRelation)")
    return (resp_p,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Cohort Hourly Grids
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    cohort_hrly_grids_f = pd.read_parquet(f"output/{SITE_NAME}/cohort_meta_by_id_imvhr.parquet")
    logger.info(f"cohort_hrly_grids_f: {len(cohort_hrly_grids_f)} rows")
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
def _(SITE_TZ, hosp, normalize_categories, to_utc):
    hosp_df = hosp.df[['hospitalization_id', 'discharge_category', 'discharge_dttm']].copy()
    # clifpy's `from_file` returns discharge_dttm NAIVE site-local; to_utc
    # localizes to SITE_TZ then converts to UTC so downstream DuckDB joins
    # see TIMESTAMPTZ and the on-disk tag survives the parquet round-trip.
    # to_utc handles DST fall-back ambiguity (UCMC has admits/discharges
    # falling on 01:00-02:00 of fall-back days).
    hosp_df = to_utc(hosp_df, 'discharge_dttm', naive_means=SITE_TZ)
    # Normalize category casing (cross-site safety). The existing
    # TRIM(LOWER(...)) in the SQL below is now redundant for surviving rows
    # but kept as defensive belt-and-suspenders.
    hosp_df = normalize_categories(hosp_df, ['discharge_category'])
    logger.info(f"hosp_df: {len(hosp_df)} rows")
    return (hosp_df,)


@app.cell
def _(CONFIG_PATH, SITE_TZ, hosp, normalize_categories, to_utc):
    cs = CodeStatus.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'code_status_category', 'start_dttm'],
    )

    # Map patient_id -> hospitalization_id via hospitalization table
    _pid_to_hid = hosp.df[['hospitalization_id', 'patient_id']].drop_duplicates()
    cs_df = cs.df.merge(_pid_to_hid, on='patient_id', how='inner')
    cs_df = cs_df[['hospitalization_id', 'code_status_category', 'start_dttm']].copy()
    # Same clifpy convention: start_dttm is naive site-local on load.
    cs_df = to_utc(cs_df, 'start_dttm', naive_means=SITE_TZ)
    # Normalize category casing for cross-site safety (cf. note in hosp_df).
    cs_df = normalize_categories(cs_df, ['code_status_category'])
    cs_df = cs_df.sort_values(['hospitalization_id', 'start_dttm']).reset_index(drop=True)
    logger.info(f"cs_df: {len(cs_df)} rows (mapped to hospitalization_id)")
    return (cs_df,)


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
                    OR regexp_matches(device_name, 't[\\s_-]?piece', 'i')
                    THEN 1 ELSE 0 END
            -- ── Subira-trial-style SBT-state (Subira et al.) ─────────────
            -- Per the published spec: T-piece OR CPAP ≤ 8 cmH2O OR
            -- (PS ≤ 8 AND PEEP ≤ 8). Broader than the primary because
            -- COALESCE(pressure_support_set, 0) catches pure-CPAP rows
            -- (where pressure_support_set IS NULL) — the primary's
            -- non-coalesced check fails on those rows due to NULL→FALSE
            -- semantics in DuckDB CASE.
            , _sbt_state_subira: CASE
                WHEN regexp_matches(device_name, 't[\\s_-]?piece', 'i') THEN 1
                WHEN mode_category = 'pressure support/cpap'
                    AND COALESCE(peep_set, 999) <= 8
                    AND COALESCE(pressure_support_set, 0) <= 8
                THEN 1 ELSE 0 END
            -- ── ABC-trial-style SBT-state (Girard et al. Lancet 2008) ────
            -- Per the published spec: T-piece OR CPAP at 5 cmH2O OR PS < 7
            -- cmH2O. Strictly more conservative than Subira on the PS arm
            -- (< 7 vs ≤ 8) and on the CPAP arm (= 5 exactly vs ≤ 8).
            -- SIMPLIFICATION: the original ABC criterion also requires "no
            -- change in PEEP or FiO2 during SBT" — operationally hard in
            -- row-level SQL (would need a 30-min lookback window comparing
            -- against the prior block's values). Dropped for this
            -- implementation; flag captures the mode-criterion only. See
            -- docs/outcomes_specs.md for the simplification note.
            , _sbt_state_abc: CASE
                WHEN regexp_matches(device_name, 't[\\s_-]?piece', 'i') THEN 1
                WHEN mode_category = 'pressure support/cpap'
                    AND pressure_support_set IS NULL
                    AND peep_set = 5
                THEN 1
                WHEN mode_category = 'pressure support/cpap'
                    AND pressure_support_set < 7
                THEN 1
                ELSE 0 END
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
        -- Gaps-and-islands: cumulative extubation and trach flags;
        -- plus row-level prior-state predicates needed downstream by sbt_outcomes
        -- to enforce spec-literal SBT-onset definition (see docs/outcomes_specs.md).
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
            -- Prior-row SBT state. NULL on first row → treated as not-0 by
            -- equality, so the SBT-onset CASE in sbt_outcomes never fires
            -- on the very first respiratory_support row of a hospitalization
            -- (excludes the patient-arrives-already-in-PS edge case).
            , _lag_sbt_state: LAG(_sbt_state) OVER w
            -- Spec-literal "from controlled mode" predicate (per
            -- CLIF_rule_based_SAT_SBT_signature/docs/sbt_delivery_implementation.md).
            -- mCIDE values are lowercase by convention.
            , _prior_mode_controlled: CASE
                WHEN LAG(mode_category) OVER w IN (
                    'assist control-volume control',
                    'pressure control',
                    'pressure-regulated volume control',
                    'simv'
                ) THEN 1 ELSE 0 END
            -- IMV-streak gaps-and-islands. Used by the sbt_done_imv6h variant
            -- in sbt_outcomes — captures the pySBT.py-style "patient must have
            -- been on IMV for ≥6h before the candidate flip" check. The streak
            -- ID itself (SUM(_imv_chg) OVER w) is computed in sbt_t3, where
            -- _imv_chg is a stable FROM column — DuckDB rejects window-over-
            -- window nesting if both are in the same SELECT via reusable alias.
            , _on_imv: CASE WHEN device_category = 'imv' THEN 1 ELSE 0 END
            , _imv_chg: CASE WHEN _on_imv IS DISTINCT FROM LAG(_on_imv) OVER w THEN 1 ELSE 0 END
            -- Per-variant chg/lag flags for the Subira and ABC operationalizations.
            -- Each variant has its own SBT-state definition (see sbt_t1) and
            -- therefore its own block partition + LAG predicate. The chg flags
            -- are stable FROM columns here so the SUM(...) OVER window in sbt_t3
            -- can use them without nested-window errors.
            , _chg_sbt_state_subira: CASE
                WHEN _sbt_state_subira IS DISTINCT FROM LAG(_sbt_state_subira) OVER w
                THEN 1 ELSE 0 END
            , _lag_sbt_state_subira: LAG(_sbt_state_subira) OVER w
            , _chg_sbt_state_abc: CASE
                WHEN _sbt_state_abc IS DISTINCT FROM LAG(_sbt_state_abc) OVER w
                THEN 1 ELSE 0 END
            , _lag_sbt_state_abc: LAG(_sbt_state_abc) OVER w
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t2,)


@app.cell
def _(REINTUB_WINDOW_HRS, sbt_t1, sbt_t2):
    sbt_t3 = mo.sql(
        f"""
        -- Assign block IDs (per-hospitalization gap-island over _sbt_state)
        -- and IMV-streak IDs (gap-island over device_category='imv'). Both
        -- are SUM(...) OVER window functions over the corresponding _chg
        -- flags from sbt_t2 — _chg flags are stable FROM columns here, so
        -- no nested-window error.
        FROM sbt_t2
        SELECT *
            , _block_id: SUM(_chg_sbt_state) OVER w
            , _block_id_subira: SUM(_chg_sbt_state_subira) OVER w
            , _block_id_abc:    SUM(_chg_sbt_state_abc)    OVER w
            , _imv_streak_id: SUM(_imv_chg) OVER w
            , _fail_extub: CASE
                WHEN sbt_t2._extub_1st = 1 AND EXISTS (
                    SELECT 1
                    FROM sbt_t1
                    WHERE sbt_t1.hospitalization_id = sbt_t2.hospitalization_id
                      AND sbt_t1._intub = 1
                      AND sbt_t1.recorded_dttm > sbt_t2.recorded_dttm
                      AND sbt_t1.recorded_dttm <= sbt_t2.recorded_dttm + INTERVAL '{REINTUB_WINDOW_HRS} HOUR'
                ) THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## v2 Outcome Definitions (ABT-RISE-style alternatives)

    Parallel implementations of intubation/extubation event detection
    (`_*_v2`), SBT delivery (`sbt_done_v2` — 2-min sustained FLIP), and
    SBT eligibility (`sbt_elig`). Existing outcomes are preserved unchanged
    as the working baselines; the v2 family runs alongside as challengers
    so the modeling cell can fit both and pick whichever performs better.

    **Simplifications from the ABT-RISE reference doc**:

    - **Stability gates for `sbt_elig`** use only row-level FiO2 ≤ 0.5 AND
      PEEP ≤ 8 (both available in `cohort_resp_processed_bf.parquet`). The full
      ABT-RISE spec also gates on NEE ≤ 0.2 mcg/kg/min and SpO2 ≥ 88 — those
      are computed at the daily level (`nee_7am`/`nee_7pm`) but not at
      `recorded_dttm` row level in this project. Defer to a follow-up round
      if v2 outcomes show promise.

    - **`sbt_done_v2`** fires on 2-min sustained FLIPs (PS ≤ 8 AND PEEP ≤ 8)
      without the prior-mode-controlled requirement of the spec-literal
      `sbt_done`. The ABT-RISE 5-min secondary is subsumed by the 2-min
      primary at our row-level resolution.

    - **State-machine extubation** (`_extub_1st_v2`) uses the consensus
      windows `CONSENSUS_INTUB_WINDOW_MIN = 15` and
      `CONSENSUS_EXTUB_WINDOW_MIN = 30`, with tracheostomy as a sentinel
      that closes any open episode and stops further counting.
    """)
    return


@app.cell
def _():
    import numpy as np

    CONSENSUS_INTUB_WINDOW_MIN = 15
    CONSENSUS_EXTUB_WINDOW_MIN = 30

    def count_intubations_v2(group):
        """ABT-RISE consensus-window state machine over a per-hosp resp slice.

        Input: a DataFrame slice with columns `recorded_dttm`,
        `device_category`, `tracheostomy` for ONE hospitalization, sorted
        ascending by `recorded_dttm`.

        Output: a DataFrame indexed identically to `group` with three
        binary columns:
          `_intub_event_v2`: 1 on the first row of each new IMV episode
              (after a 15-min consensus window).
          `_extub_event_v2`: 1 on the first row patient leaves IMV
              (after a 30-min consensus window confirming non-reversion).
          `_trach_event_v2`: 1 on the row where tracheostomy first appears.

        Rules (per the reference doc lines 96–179):
          - Direct-to-trach: if a tracheostomy row precedes any IMV row,
            count zero IMV episodes (mark only `_trach_event_v2 = 1`).
          - Tracheostomy sentinel: any later trach event closes the open
            episode (firing `_extub_event_v2` if mid-IMV, then `_trach`)
            and halts further state-machine processing for that patient.
          - Consensus windows: a transition is only confirmed once the
            new state has been sustained for ≥ window_min minutes; if the
            new state reverts within the window, the transition is aborted
            and no event is fired.
        """
        n = len(group)
        out = pd.DataFrame(
            {'_intub_event_v2': 0, '_extub_event_v2': 0, '_trach_event_v2': 0},
            index=group.index,
        )
        if n == 0:
            return out

        is_imv = group['device_category'].eq('imv').to_numpy()
        has_trach = group['tracheostomy'].eq(1).to_numpy()
        dttm = group['recorded_dttm'].to_numpy()

        # Direct-to-trach rule: trach appears before any IMV row.
        first_trach = np.argmax(has_trach) if has_trach.any() else -1
        first_imv = np.argmax(is_imv) if is_imv.any() else -1
        if first_trach >= 0 and (first_imv < 0 or first_trach < first_imv):
            out.iloc[first_trach, out.columns.get_loc('_trach_event_v2')] = 1
            return out

        state = 'pre'  # pre / intub_candidate / on_imv / off_imv_candidate / off_imv
        intub_cand_i = None
        extub_cand_i = None

        for i in range(n):
            if has_trach[i]:
                # Trach sentinel: close any open extub-pending state, then halt.
                if state == 'off_imv_candidate' and extub_cand_i is not None:
                    out.iloc[extub_cand_i, out.columns.get_loc('_extub_event_v2')] = 1
                elif state in ('on_imv', 'intub_candidate'):
                    # Patient went straight from IMV to trach — count as extub
                    # event at the trach row (so the IMV episode has a closure).
                    out.iloc[i, out.columns.get_loc('_extub_event_v2')] = 1
                out.iloc[i, out.columns.get_loc('_trach_event_v2')] = 1
                return out

            if state == 'pre':
                if is_imv[i]:
                    intub_cand_i = i
                    state = 'intub_candidate'
            elif state == 'intub_candidate':
                if not is_imv[i]:
                    state = 'pre'
                    intub_cand_i = None
                else:
                    mins = (dttm[i] - dttm[intub_cand_i]) / np.timedelta64(1, 'm')
                    if mins >= CONSENSUS_INTUB_WINDOW_MIN:
                        out.iloc[intub_cand_i, out.columns.get_loc('_intub_event_v2')] = 1
                        state = 'on_imv'
                        intub_cand_i = None
            elif state == 'on_imv':
                if not is_imv[i]:
                    extub_cand_i = i
                    state = 'off_imv_candidate'
            elif state == 'off_imv_candidate':
                if is_imv[i]:
                    state = 'on_imv'
                    extub_cand_i = None
                else:
                    mins = (dttm[i] - dttm[extub_cand_i]) / np.timedelta64(1, 'm')
                    if mins >= CONSENSUS_EXTUB_WINDOW_MIN:
                        out.iloc[extub_cand_i, out.columns.get_loc('_extub_event_v2')] = 1
                        state = 'off_imv'
                        extub_cand_i = None
            elif state == 'off_imv':
                if is_imv[i]:
                    intub_cand_i = i
                    state = 'intub_candidate'

        # Trailing partial states (window not consummated by end of stream):
        # leave as no-event. Conservative — partial transitions don't fire.
        return out

    return (count_intubations_v2, np)


@app.cell
def _(ENABLE_V2_OUTCOMES, count_intubations_v2, pd, resp_p):
    # Apply state machine per hospitalization. group_keys=False preserves the
    # original index so we can concat back to resp_p without realignment.
    # ENABLE_V2_OUTCOMES=False short-circuit: emit constant-zero v2 event
    # columns and skip the ~7-min pandas state machine entirely. Downstream
    # SQL aggregations (sbt_t5_v2, etc.) continue to execute on the all-zero
    # inputs and emit all-zero v2 outcome columns — schema stays stable for
    # cross-site agg scripts. 08_models.py skips the *_v2_next_day outcome
    # fits explicitly (does NOT attempt to fit on constant-zero outcomes).
    #
    # resp_p is a DuckDBPyRelation (Step 2 of the scalability refactor);
    # materialize to pandas once here. At 1M-source-DB scale this is bounded
    # by the cohort size (~28-56 GB at 100-200K cohort) — still large but
    # consumed only by the V2 cell when enabled, never module-level.
    _resp_pdf = resp_p.df()
    _sorted = _resp_pdf.sort_values(['hospitalization_id', 'recorded_dttm']).reset_index(drop=True)
    if ENABLE_V2_OUTCOMES:
        _events = _sorted.groupby('hospitalization_id', group_keys=False).apply(
            count_intubations_v2
        )
        resp_p_v2 = pd.concat([_sorted, _events], axis=1)
    else:
        resp_p_v2 = _sorted.assign(
            _intub_event_v2=0, _extub_event_v2=0, _trach_event_v2=0,
        )
        logger.info("V2 state machine SKIPPED (enable_v2_outcomes=false)")
    logger.info(
        f"resp_p_v2: {len(resp_p_v2)} rows | "
        f"_intub_event_v2 fired on {int(resp_p_v2['_intub_event_v2'].sum())} rows | "
        f"_extub_event_v2 fired on {int(resp_p_v2['_extub_event_v2'].sum())} rows | "
        f"_trach_event_v2 fired on {int(resp_p_v2['_trach_event_v2'].sum())} rows"
    )
    return (resp_p_v2,)


@app.cell
def _(sbt_t3):
    # IMV-streak duration metadata for the sbt_done_imv6h variant. Split out
    # from sbt_t3 because DuckDB rejects nested window calls — the start_dttm
    # uses MIN() OVER (PARTITION BY ..., _imv_streak_id) and the lag_minutes
    # uses LAG() OVER, both of which would resolve through reusable aliases
    # to nested OVER expressions if defined alongside _imv_streak_id in sbt_t3.
    # The inner WITH clause keeps streak_start_dttm and streak_minutes in one
    # SELECT (date_diff isn't a window so the start_dttm reference is fine);
    # the outer SELECT then computes _lag_imv_streak_minutes off the stable
    # _imv_streak_minutes column.
    sbt_t4 = mo.sql(
        f"""
        WITH streak AS (
            FROM sbt_t3
            SELECT *
                -- Min recorded_dttm per IMV streak. Broadcasts across the streak.
                -- The CASE filters to rows actually on IMV so non-IMV streaks
                -- have NULL start.
                , _imv_streak_start_dttm: MIN(
                    CASE WHEN _on_imv = 1 THEN recorded_dttm END
                  ) OVER (PARTITION BY hospitalization_id, _imv_streak_id)
                -- Minutes the patient has been continuously on IMV at this row.
                -- NULL when not on IMV.
                , _imv_streak_minutes: CASE
                    WHEN _on_imv = 1
                    THEN date_diff('minute', _imv_streak_start_dttm, recorded_dttm)
                    ELSE NULL END
                -- "Did this SBT block start from a controlled mode?" — broadcast
                -- to every row of the block. _prior_mode_controlled (from sbt_t2)
                -- is only 1 on the onset row of an SBT block; MAX-OVER-_block_id
                -- propagates that boolean across the whole block, so the
                -- sbt_done_multiday CASE in sbt_outcomes can fire on every row
                -- of a "real" SBT block without collapsing to onset-only.
                , _block_prior_mode_controlled: MAX(_prior_mode_controlled)
                    OVER (PARTITION BY hospitalization_id, _block_id)
                -- ── Per-variant SBT-block boundaries (Subira / ABC) ──────────
                -- Block start/last-record per `_block_id_<variant>`, broadcast
                -- to every row of the block. Used downstream by sbt_outcomes
                -- to compute per-variant block durations without an extra
                -- aggregation cell. NOTE: this approximates true block end as
                -- the LAST recorded_dttm WITHIN the block, which slightly
                -- under-estimates duration relative to the primary's
                -- "next block start as effective end" trick (see
                -- sbt_all_blocks_w_duration). The under-estimate is bounded by
                -- the resp-record sampling interval (~1h typical), well below
                -- the 30-min variant threshold's noise floor.
                , _block_subira_start_dttm: MIN(recorded_dttm)
                    OVER (PARTITION BY hospitalization_id, _block_id_subira)
                , _block_subira_last_dttm: MAX(recorded_dttm)
                    OVER (PARTITION BY hospitalization_id, _block_id_subira)
                , _block_abc_start_dttm: MIN(recorded_dttm)
                    OVER (PARTITION BY hospitalization_id, _block_id_abc)
                , _block_abc_last_dttm: MAX(recorded_dttm)
                    OVER (PARTITION BY hospitalization_id, _block_id_abc)
        )
        FROM streak
        SELECT *
            -- Duration of the prior row's IMV streak (used by sbt_done_imv6h).
            -- For T-piece SBT (current row not on IMV), the prior row was on
            -- IMV (controlled or supportive) and this LAG gives that prior
            -- streak's minutes-so-far. For PS-with-low SBT (current row still
            -- on IMV), the streak hasn't broken, so LAG gives the duration up
            -- to the prior row.
            , _lag_imv_streak_minutes: LAG(_imv_streak_minutes) OVER w
            -- Per-variant block durations (computed from the broadcast
            -- start/last timestamps in the WITH cell above). Date_diff isn't a
            -- window function so this can live in the outer SELECT alongside
            -- the LAG without nested-window errors.
            , _block_duration_mins_subira: date_diff(
                'minute', _block_subira_start_dttm, _block_subira_last_dttm)
            , _block_duration_mins_abc: date_diff(
                'minute', _block_abc_start_dttm, _block_abc_last_dttm)
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        """
    )
    return (sbt_t4,)


@app.cell
def _(resp_p_v2, sbt_t4):
    # sbt_t5_v2: join sbt_t4's columns with the v2 state-machine event flags
    # from resp_p_v2 + compute v2 cumulative event flags (first event per
    # hospitalization), v2 stability flag (FiO2 ≤ 0.5 AND PEEP ≤ 8 row-level
    # gates — NEE/SpO2 deferred per docstring), and v2 eligibility (stability
    # AND IMV streak ≥ 6 h).
    sbt_t5_v2 = mo.sql(
        f"""
        FROM sbt_t4
        LEFT JOIN resp_p_v2 USING (hospitalization_id, recorded_dttm)
        SELECT sbt_t4.*
            -- COALESCE the v2 event flags so missing-from-resp_p_v2 rows
            -- (shouldn't happen but defensive) become 0.
            , _intub_event_v2: COALESCE(resp_p_v2._intub_event_v2, 0)
            , _extub_event_v2: COALESCE(resp_p_v2._extub_event_v2, 0)
            , _trach_event_v2: COALESCE(resp_p_v2._trach_event_v2, 0)
            -- First-occurrence flags via cumulative SUM windows
            , _extub_cum_v2: SUM(COALESCE(resp_p_v2._extub_event_v2, 0)) OVER w
            , _extub_1st_v2: CASE
                WHEN COALESCE(resp_p_v2._extub_event_v2, 0) = 1
                    AND SUM(COALESCE(resp_p_v2._extub_event_v2, 0)) OVER w = 1
                THEN 1 ELSE 0 END
            , _trach_cum_v2: SUM(COALESCE(resp_p_v2._trach_event_v2, 0)) OVER w
            , _trach_v2: CASE
                WHEN COALESCE(resp_p_v2._trach_event_v2, 0) = 1
                    AND SUM(COALESCE(resp_p_v2._trach_event_v2, 0)) OVER w = 1
                THEN 1 ELSE 0 END
            -- v2 row-level stability gate (FiO2 + PEEP only; NEE/SpO2 deferred).
            -- Treat NULL as fail (COALESCE to a value above the threshold).
            , _stable_v2: CASE
                WHEN COALESCE(sbt_t4.fio2_set, 999) <= 0.5
                    AND COALESCE(sbt_t4.peep_set, 999) <= 8
                THEN 1 ELSE 0 END
            -- v2 eligibility: stability AND IMV streak ≥ 6 h (360 min).
            -- _imv_streak_minutes is NULL on non-IMV rows, so eligibility
            -- only fires while patient is currently on IMV.
            , _eligible_v2: CASE
                WHEN sbt_t4._imv_streak_minutes >= 360
                    AND CASE
                        WHEN COALESCE(sbt_t4.fio2_set, 999) <= 0.5
                            AND COALESCE(sbt_t4.peep_set, 999) <= 8
                        THEN 1 ELSE 0 END = 1
                THEN 1 ELSE 0 END
        WINDOW w AS (PARTITION BY sbt_t4.hospitalization_id ORDER BY sbt_t4.recorded_dttm)
        """
    )
    return (sbt_t5_v2,)


@app.cell
def _(REINTUB_WINDOW_HRS, sbt_t5_v2):
    # Compute v2 fail-extub: re-intub event within REINTUB_WINDOW_HRS after
    # _extub_1st_v2 fires. Mirror of sbt_t3's existing _fail_extub but using
    # v2 event flags.
    sbt_t5_v2_w_fail = mo.sql(
        f"""
        FROM sbt_t5_v2 AS a
        SELECT a.*
            , _fail_extub_v2: CASE
                WHEN a._extub_1st_v2 = 1 AND EXISTS (
                    SELECT 1
                    FROM sbt_t5_v2 AS b
                    WHERE b.hospitalization_id = a.hospitalization_id
                      AND b._intub_event_v2 = 1
                      AND b.recorded_dttm > a.recorded_dttm
                      AND b.recorded_dttm <= a.recorded_dttm + INTERVAL '{REINTUB_WINDOW_HRS} HOUR'
                ) THEN 1 ELSE 0 END
        """
    )
    return (sbt_t5_v2_w_fail,)


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
    Computes sbt_done (spec-literal onset), success_extub, fail_extub, withdrawal.
    """)
    return


@app.cell
def _(cs_df, hosp_df, sbt_all_blocks_w_duration, sbt_t5_v2_w_fail):
    # NOTE: this cell now reads from `sbt_t5_v2_w_fail` (aliased back to
    # `sbt_t4` for column-reference compatibility), which carries all of
    # sbt_t4's columns PLUS the v2 event flags + eligibility flag from the
    # ABT-RISE-style alternative implementations. Existing `sbt_t4.<col>`
    # references continue to resolve; new v2 columns are accessible via the
    # same alias.
    sbt_outcomes = mo.sql(
        f"""
        -- Final SBT outcomes with code status and discharge info
        FROM sbt_t5_v2_w_fail AS sbt_t4
        LEFT JOIN sbt_all_blocks_w_duration AS b
            ON sbt_t4.hospitalization_id = b.hospitalization_id
            AND sbt_t4._block_id = b._block_id
        ASOF LEFT JOIN cs_df AS c
            ON c.hospitalization_id = sbt_t4.hospitalization_id
            AND c.start_dttm <= sbt_t4.recorded_dttm
        ASOF LEFT JOIN hosp_df AS h
            ON sbt_t4.hospitalization_id = h.hospitalization_id
            AND sbt_t4.recorded_dttm <= h.discharge_dttm
        SELECT sbt_t4.fio2_set, sbt_t4.peep_set, sbt_t4.pressure_support_set, sbt_t4.tracheostomy
            , _block_duration_mins: COALESCE(b._duration_mins, 0)
            , sbt_t4.device_category, sbt_t4.device_name, sbt_t4.mode_category, sbt_t4.mode_name
            , sbt_t4.hospitalization_id, event_dttm: sbt_t4.recorded_dttm
            , sbt_t4._block_id
            , sbt_t4._lag_sbt_state, sbt_t4._prior_mode_controlled
            -- Audit columns for the QC dashboard's linked-table view.
            -- Persisting them at the row level lets the auditor verify *why*
            -- each variant flag fired (e.g., `_lag_imv_streak_minutes ≥ 360`
            -- for `sbt_done_imv6h`).
            , sbt_t4._imv_streak_minutes, sbt_t4._lag_imv_streak_minutes
            , sbt_t4._sbt_state_subira, sbt_t4._sbt_state_abc
            , sbt_t4._lag_sbt_state_subira, sbt_t4._lag_sbt_state_abc
            , sbt_t4._block_id_subira, sbt_t4._block_id_abc
            , sbt_t4._block_duration_mins_subira, sbt_t4._block_duration_mins_abc
            -- Block-level "did this block start from controlled IMV?" predicate,
            -- broadcast across all rows of the block in sbt_t4. Drives sbt_done_multiday.
            , sbt_t4._block_prior_mode_controlled
            -- Spec-literal SBT onset (see docs/outcomes_specs.md). Fires only
            -- on the legitimate transition row of a qualifying block:
            --   (a) sustained ≥30 min via _block_duration_mins
            --   (b) current row is in SBT target state (_sbt_state = 1)
            --   (c) immediate predecessor was non-SBT (_lag_sbt_state = 0;
            --       NULL on row 1 doesn't equal 0, so first-row arrivals
            --       never fire — onset-only by construction)
            --   (d) immediate predecessor's mode_category was controlled
            --       (_prior_mode_controlled = 1; spec literal)
            -- These conditions together select exactly one row per qualifying
            -- block, so MAX-based hourly/daily aggregation reflects unique
            -- trials (no spillover into subsequent rows or days).
            , sbt_done: CASE
                WHEN _block_duration_mins >= 30
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._lag_sbt_state = 0
                    AND sbt_t4._prior_mode_controlled = 1
                THEN 1 ELSE 0 END
            -- ── Sensitivity variants (see docs/outcomes_specs.md §
            --    "SBT Detection (current implementation) > Sensitivity siblings")
            -- (a) anyprior: drop the controlled-mode-list requirement; just need
            --     the prior row was non-SBT. Most permissive prior-mode check.
            , sbt_done_anyprior: CASE
                WHEN _block_duration_mins >= 30
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._lag_sbt_state = 0
                THEN 1 ELSE 0 END
            -- (b) imv6h: pySBT.py-style — patient has been continuously on IMV
            --     for ≥6h before the candidate flip. Simplified vs spec by
            --     dropping the explicit 10pm-6am window.
            , sbt_done_imv6h: CASE
                WHEN _block_duration_mins >= 30
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._lag_sbt_state = 0
                    AND sbt_t4._lag_imv_streak_minutes >= 360
                THEN 1 ELSE 0 END
            -- (c) prefix: pre-fix sanity check — the original definition before
            --     this plan. Should reproduce the baseline 52.9% event rate.
            , sbt_done_prefix: CASE
                WHEN _block_duration_mins >= 30
                    AND sbt_t4._sbt_state = 1
                THEN 1 ELSE 0 END
            -- (c') multiday: keeps prefix's multi-day spread (every row of a
            --     qualifying ≥30-min block, NOT onset-only) but restricted to
            --     blocks that came from controlled IMV. Block-level prior-mode
            --     property is broadcast in sbt_t4 via MAX-OVER-_block_id —
            --     row-level _prior_mode_controlled would AND-collapse to
            --     onset-only and defeat the multiday intent. By construction:
            --       sbt_done <= sbt_done_multiday <= sbt_done_prefix  (daily rates)
            , sbt_done_multiday: CASE
                WHEN _block_duration_mins >= 30
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._block_prior_mode_controlled = 1
                THEN 1 ELSE 0 END
            -- (d) 2min: spec-literal prior conditions but ≥2 min duration.
            , sbt_done_2min: CASE
                WHEN _block_duration_mins >= 2
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._lag_sbt_state = 0
                    AND sbt_t4._prior_mode_controlled = 1
                THEN 1 ELSE 0 END
            -- (e) Subira-trial-style operationalization (Subira et al.):
            --     T-piece OR CPAP ≤8 OR (PS ≤8 AND PEEP ≤8), sustained ≥30 min.
            --     Uses its own block partition (`_block_id_subira`) since the
            --     Subira-state set is broader than the primary's (it catches
            --     pure-CPAP rows the primary's non-coalesced AND misses).
            , sbt_done_subira: CASE
                WHEN sbt_t4._block_duration_mins_subira >= 30
                    AND sbt_t4._sbt_state_subira = 1
                    AND sbt_t4._lag_sbt_state_subira = 0
                THEN 1 ELSE 0 END
            -- (f) ABC-trial-style operationalization (Girard et al. 2008):
            --     T-piece OR CPAP=5 OR PS<7, sustained ≥30 min. Strictly more
            --     conservative than Subira on PS arm. Drops the original ABC
            --     "no change in PEEP/FiO2 during SBT" clause as a known
            --     simplification (see docs/outcomes_specs.md).
            , sbt_done_abc: CASE
                WHEN sbt_t4._block_duration_mins_abc >= 30
                    AND sbt_t4._sbt_state_abc = 1
                    AND sbt_t4._lag_sbt_state_abc = 0
                THEN 1 ELSE 0 END
            , _extub_1st, _intub, sbt_t4._trach_1st, _fail_extub
            , c.code_status_category, cs_start_dttm: c.start_dttm
            , h.discharge_category, discharge_dttm: h.discharge_dttm
            , _withdrawl_lst: CASE
                WHEN _extub_1st = 1
                    AND TRIM(LOWER(code_status_category)) != 'full'
                    AND TRIM(LOWER(discharge_category)) IN ('hospice', 'expired')
                THEN 1 ELSE 0 END
            , _success_extub: CASE
                WHEN _extub_1st = 1 AND _withdrawl_lst = 0 AND _fail_extub = 0
                THEN 1 ELSE 0 END
            -- ── v2 outcome family (ABT-RISE-style alternatives, 2026-04-29) ──
            -- sbt_done_v2: 2-min sustained FLIP (PS ≤ 8 AND PEEP ≤ 8) without
            -- the prior_mode_controlled gate of the spec-literal sbt_done.
            -- Reuses the existing _block_duration_mins + _sbt_state from sbt_t1
            -- (the block-partition logic is the same; only the duration cutoff
            -- and the prior-mode gate differ).
            , sbt_done_v2: CASE
                WHEN _block_duration_mins >= 2
                    AND sbt_t4._sbt_state = 1
                    AND sbt_t4._lag_sbt_state = 0
                THEN 1 ELSE 0 END
            -- sbt_elig: per-row eligibility flag (FiO2 ≤ 0.5 + PEEP ≤ 8 +
            -- IMV ≥ 6 h). NEE ≤ 0.2 and SpO2 ≥ 88 gates from ABT-RISE are
            -- deferred — those signals live at daily granularity in this
            -- project (nee_7am / nee_7pm), not at recorded_dttm. Aggregating
            -- per-row eligibility to daily via MAX gives a binary "eligible
            -- at any point this day" outcome.
            , sbt_elig: sbt_t4._eligible_v2
            -- v2 extubation outcome family. The successful-extubation
            -- predicate (`_success_extub_v2`) is identical to the current
            -- `_success_extub` per user direction — only the underlying
            -- first-extubation event detector changes (state-machine v2
            -- vs. cumulative-extub v1).
            , sbt_t4._extub_1st_v2
            , sbt_t4._trach_v2
            , sbt_t4._fail_extub_v2
            , _withdrawl_lst_v2: CASE
                WHEN sbt_t4._extub_1st_v2 = 1
                    AND TRIM(LOWER(code_status_category)) != 'full'
                    AND TRIM(LOWER(discharge_category)) IN ('hospice', 'expired')
                THEN 1 ELSE 0 END
            , _success_extub_v2: CASE
                WHEN sbt_t4._extub_1st_v2 = 1
                    AND CASE
                        WHEN sbt_t4._extub_1st_v2 = 1
                            AND TRIM(LOWER(code_status_category)) != 'full'
                            AND TRIM(LOWER(discharge_category)) IN ('hospice', 'expired')
                        THEN 1 ELSE 0 END = 0
                    AND sbt_t4._fail_extub_v2 = 0
                THEN 1 ELSE 0 END
        WHERE (sbt_t4.tracheostomy = 0 OR sbt_t4._trach_1st = 1 OR sbt_t4._trach_v2 = 1)
        ORDER BY sbt_t4.hospitalization_id, event_dttm
        """
    )
    return (sbt_outcomes,)


@app.cell
def _(SITE_NAME, sbt_outcomes, to_utc):
    # Persist the raw row-level table for audit / QC dashboard consumption.
    # Naming convention: this is the second-to-last aggregation tier (before
    # hourly + daily roll-ups), saved alongside `outcomes_by_id_imvday.parquet`.
    # Retag every *_dttm column to UTC before writing — DuckDB's .df()
    # stamps TIMESTAMPTZ columns with the session tz, which leaks the
    # runner's OS tz into the on-disk schema. Auto-detect over column
    # suffix so derived/renamed *_dttm columns (event_dttm, cs_start_dttm,
    # discharge_dttm, _imv_streak_start_dttm, etc.) all get normalized.
    _out = sbt_outcomes.df()
    _dttm_cols = [c for c in _out.columns if c.endswith('_dttm')]
    _out = to_utc(_out, _dttm_cols)
    _path = f"output/{SITE_NAME}/outcomes_by_event.parquet"
    _out.to_parquet(_path)
    logger.info(f"Saved: {_path} ({len(_out)} rows, {_out['hospitalization_id'].nunique()} hospitalizations)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Hourly & Daily Aggregation

    Floor event_dttm to hour, aggregate SBT outcomes per hour,
    join back to cohort hourly grid, then roll up to daily.
    """)
    return


@app.cell
def _(SITE_TZ, sbt_outcomes):
    sbt_outcomes_hrly = mo.sql(
        f"""
        -- Aggregate SBT outcomes per hospitalization-hour.
        -- `_dh` derived via `event_dttm AT TIME ZONE site_tz` so the result
        -- is naive local — matches cohort_hrly_grids' `_dh` for the LEFT
        -- JOIN below. event_dttm here is UTC tz-aware (sbt_outcomes inherits
        -- from resp_p; cohort_resp_processed_bf.parquet predates the tz reflag in
        -- 01 and stays UTC for waterfall reentry compat). Both sides of the
        -- downstream join must be naive local TIMESTAMP with the same value.
        FROM sbt_outcomes
        SELECT hospitalization_id
            , _dh: date_trunc('hour', event_dttm AT TIME ZONE '{SITE_TZ}')
            , sbt_done: MAX(sbt_done)
            , sbt_done_anyprior: MAX(sbt_done_anyprior)
            , sbt_done_imv6h: MAX(sbt_done_imv6h)
            , sbt_done_prefix: MAX(sbt_done_prefix)
            , sbt_done_multiday: MAX(sbt_done_multiday)
            , sbt_done_2min: MAX(sbt_done_2min)
            , sbt_done_subira: MAX(sbt_done_subira)
            , sbt_done_abc: MAX(sbt_done_abc)
            , _success_extub: MAX(_success_extub)
            , _trach_1st: MAX(_trach_1st)
            , _fail_extub: MAX(_fail_extub)
            , _extub_1st: MAX(_extub_1st)
            , _withdrawl_lst: MAX(_withdrawl_lst)
            -- v2 outcome family
            , sbt_done_v2: MAX(sbt_done_v2)
            , sbt_elig: MAX(sbt_elig)
            , _extub_1st_v2: MAX(_extub_1st_v2)
            , _trach_v2: MAX(_trach_v2)
            , _fail_extub_v2: MAX(_fail_extub_v2)
            , _withdrawl_lst_v2: MAX(_withdrawl_lst_v2)
            , _success_extub_v2: MAX(_success_extub_v2)
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
            , sbt_done_anyprior: COALESCE(s.sbt_done_anyprior, 0)
            , sbt_done_imv6h: COALESCE(s.sbt_done_imv6h, 0)
            , sbt_done_prefix: COALESCE(s.sbt_done_prefix, 0)
            , sbt_done_multiday: COALESCE(s.sbt_done_multiday, 0)
            , sbt_done_2min: COALESCE(s.sbt_done_2min, 0)
            , sbt_done_subira: COALESCE(s.sbt_done_subira, 0)
            , sbt_done_abc: COALESCE(s.sbt_done_abc, 0)
            , _success_extub: COALESCE(s._success_extub, 0)
            , _trach_1st: COALESCE(s._trach_1st, 0)
            , _fail_extub: COALESCE(s._fail_extub, 0)
            , _extub_1st: COALESCE(s._extub_1st, 0)
            , _withdrawl_lst: COALESCE(s._withdrawl_lst, 0)
            -- v2 outcome family
            , sbt_done_v2: COALESCE(s.sbt_done_v2, 0)
            , sbt_elig: COALESCE(s.sbt_elig, 0)
            , _extub_1st_v2: COALESCE(s._extub_1st_v2, 0)
            , _trach_v2: COALESCE(s._trach_v2, 0)
            , _fail_extub_v2: COALESCE(s._fail_extub_v2, 0)
            , _withdrawl_lst_v2: COALESCE(s._withdrawl_lst_v2, 0)
            , _success_extub_v2: COALESCE(s._success_extub_v2, 0)
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
            , sbt_done_anyprior: MAX(sbt_done_anyprior)
            , sbt_done_imv6h: MAX(sbt_done_imv6h)
            , sbt_done_prefix: MAX(sbt_done_prefix)
            , sbt_done_multiday: MAX(sbt_done_multiday)
            , sbt_done_2min: MAX(sbt_done_2min)
            , sbt_done_subira: MAX(sbt_done_subira)
            , sbt_done_abc: MAX(sbt_done_abc)
            , _success_extub: MAX(_success_extub)
            , _trach_1st: MAX(_trach_1st)
            , _fail_extub: MAX(_fail_extub)
            , _extub_1st: MAX(_extub_1st)
            , _withdrawl_lst: MAX(_withdrawl_lst)
            -- v2 outcome family
            , sbt_done_v2: MAX(sbt_done_v2)
            , sbt_elig: MAX(sbt_elig)
            , _extub_1st_v2: MAX(_extub_1st_v2)
            , _trach_v2: MAX(_trach_v2)
            , _fail_extub_v2: MAX(_fail_extub_v2)
            , _withdrawl_lst_v2: MAX(_withdrawl_lst_v2)
            , _success_extub_v2: MAX(_success_extub_v2)
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
def _(SITE_NAME, cohort_sbt_outcomes_daily):
    _out = cohort_sbt_outcomes_daily.df()
    _path = f"output/{SITE_NAME}/outcomes_by_id_imvday.parquet"
    _out.to_parquet(_path)
    logger.info(f"Saved: {_path} ({len(_out)} rows, {_out['hospitalization_id'].nunique()} hospitalizations)")
    return


if __name__ == "__main__":
    app.run()
