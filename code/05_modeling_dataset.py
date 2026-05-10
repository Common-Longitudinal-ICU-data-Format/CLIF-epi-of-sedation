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

    from clifpy.utils.logging_config import get_logger
    logger = get_logger("epi_sedation.modeling_dataset")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 05 Modeling Dataset

    Builds the consolidated per-day modeling input
    `model_input_by_id_imvday.parquet` — one row per
    `(hospitalization_id, _nth_day)` keyed off the canonical patient-day
    registry (`cohort_meta_by_id_imvday.parquet`) with every per-day
    source LEFT-joined onto it.

    Two filter views replace the historic two-parquet split:

    - **Outcome-modeling cohort** (consumers: `06_table1.py`,
      `07_descriptive.py`, `08_models.py`, `08b_models_cascade.py`,
      `descriptive/_shared.py:load_modeling()`,
      `agg/_shared.py:load_site_analytical()`): apply the filter
      `_nth_day > 0 AND sbt_done_next_day IS NOT NULL AND
      success_extub_next_day IS NOT NULL` at read time. Each loader
      already does this — see `code/descriptive/_shared.py:load_modeling`.

    - **Exposure characterization** (consumers: every
      `code/descriptive/*_by_icu_day*.py` figure via
      `descriptive/_shared.py:load_model_input()` and
      `agg/_shared.py:load_site_model_input()`): no filter, full
      registry trajectory. Apply `_is_full_24h_day = TRUE AND _nth_day
      BETWEEN 1 AND 7` at the call site to restrict to fully-comparable
      12+12 hr coverage panels.

    - **Day-0 sensitivity** (was a separate cell in the historic flow):
      apply `_nth_day >= 0 AND <outcome filters>` at read time — a
      2-character switch (`>` → `>=`) on the modeling filter.

    Rate convention: dose columns arrive from `seddose_by_id_imvday.parquet`
    as per-shift avg rates (computed upstream as `AVG(hourly_rate)` over
    the shift's hours in `02_exposure.py`'s `sed_dose_agg`/`seddose_by_id_imvday`
    cells). No /n_hrs/60 arithmetic is needed at this layer — we just
    rename the columns onto the modeling-cohort namespace. `n_hrs_day` /
    `n_hrs_night` flow through as coverage metadata only.

    Hospitalization-level NMB exclusion is applied during the
    consolidated cell below.

    Phase 4 cutover (2026-05-08): the legacy `modeling_dataset.parquet`
    + `exposure_dataset.parquet` producer cells were retired. Existing
    on-disk legacy files (last regenerated 2026-05-08) remain as
    historical references.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    from _utils import to_utc
    import pandas as pd
    import duckdb

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    SITE_TZ = cfg['timezone']
    # Site-scoped output dir (see Makefile SITE= flag). `os` comes from the
    # app.setup block at the top; re-importing here would raise marimo's
    # MultipleDefinitionError.
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    logger.info(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return CONFIG_PATH, SITE_NAME, SITE_TZ, apply_outlier_handling, pd, to_utc


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Intermediate Outputs
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    sbt_outcomes_daily = pd.read_parquet(f"output/{SITE_NAME}/outcomes_by_id_imvday.parquet")
    logger.info(f"sbt_outcomes_daily: {len(sbt_outcomes_daily)} rows")
    return (sbt_outcomes_daily,)


@app.cell
def _(SITE_NAME, pd):
    seddose_by_id_imvday = pd.read_parquet(f"output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    logger.info(f"seddose_by_id_imvday: {len(seddose_by_id_imvday)} rows")
    return (seddose_by_id_imvday,)


@app.cell
def _(SITE_NAME, pd):
    covs_daily = pd.read_parquet(f"output/{SITE_NAME}/covariates_by_id_imvday.parquet")
    logger.info(f"covs_daily: {len(covs_daily)} rows")
    return (covs_daily,)


@app.cell
def _(SITE_NAME, pd):
    nmb_excluded = pd.read_parquet(f"output/{SITE_NAME}/cohort_nmb_excluded.parquet")
    logger.info(f"nmb_excluded patient-days: {len(nmb_excluded)}")
    return (nmb_excluded,)


@app.cell
def _(CONFIG_PATH, apply_outlier_handling):
    from clifpy import Hospitalization
    hosp = Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'hospitalization_id', 'discharge_dttm', 'discharge_category', 'age_at_admission'],
    )
    apply_outlier_handling(hosp, outlier_config_path='config/outlier_config.yaml')
    hosp_df = hosp.df
    logger.info(f"hosp_df: {len(hosp_df)} rows")
    return (hosp_df,)


@app.cell
def _(CONFIG_PATH):
    from clifpy import Patient
    _patient = Patient.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'sex_category'],
    )
    patient_df = _patient.df
    logger.info(f"patient_df: {len(patient_df)} rows")
    return (patient_df,)


@app.cell
def _(SITE_NAME, pd):
    sofa_daily = pd.read_parquet(f"output/{SITE_NAME}/sofa_by_id_imvday.parquet")
    icu_type_df = pd.read_parquet(f"output/{SITE_NAME}/cohort_icu_type.parquet")
    cci_df = pd.read_parquet(f"output/{SITE_NAME}/covariates_cci.parquet")
    elix_df = pd.read_parquet(f"output/{SITE_NAME}/covariates_elix.parquet")
    covariates_t1 = pd.read_parquet(f"output/{SITE_NAME}/covariates_t1.parquet")
    weight_daily = pd.read_parquet(f"output/{SITE_NAME}/weight_by_id_imvday.parquet")
    logger.info(
        f"sofa_daily: {len(sofa_daily)}, icu_type: {len(icu_type_df)}, "
        f"cci: {len(cci_df)}, elix: {len(elix_df)}, "
        f"covariates_t1: {len(covariates_t1)}, weight_daily: {len(weight_daily)}"
    )
    return cci_df, covariates_t1, elix_df, icu_type_df, sofa_daily, weight_daily



@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Consolidated `model_input_by_id_imvday` (Phase 4 — additive cutover)

    Single per-day modeling input, base table = `cohort_meta_by_id_imvday`.
    LEFT-joins every per-day source onto the canonical patient-day registry.
    Two filter views replace the legacy two parquets:

    - **Outcome-modeling cohort** (replaces `modeling_dataset.parquet`):
      `WHERE _nth_day > 0 AND sbt_done_next_day IS NOT NULL AND
      success_extub_next_day IS NOT NULL`.

    - **Exposure characterization** (replaces `exposure_dataset.parquet`):
      no filter — the full registry trajectory.

    - **Day-0 sensitivity** (replaces inline filter in `08_models.py`):
      `WHERE _nth_day >= 0 AND <outcome filters>` — a 2-character switch
      (`>` → `>=`) on the modeling filter.

    Rate convention: dose columns are already per-shift avg rates upstream
    (Phase-3 refactor 2026-05-08 — `02_exposure.py:seddose_by_id_imvday` now
    pivots AVG-of-hourly-rates per shift; `n_hrs_day`/`n_hrs_night` carry
    only as coverage metadata). This SELECT is a pure rename onto the
    modeling-cohort namespace.

    `_is_first_day`, `_is_last_partial_day`, `_is_last_full_day`,
    `_is_full_24h_day`, `n_hrs_day`, `n_hrs_night`, `day_type`,
    `day_start_dttm`, `day_end_dttm`, `encounter_block`, `nmb_excluded`,
    `nmb_total_min` come from the registry directly — no inline
    window-function recomputation.

    Two distinct "last day" semantics are exposed:
    - `_is_last_partial_day` = `day_type='last_partial'` (truncated
      extubation day, removed by analyses requiring full coverage).
    - `_is_last_full_day` = the patient's final full-24h row (last row
      that survives the modeling-cohort filter).

    `_rel_day` is **dropped**. The descriptive scripts now restrict
    panels to days 1–7 full-24h via `_is_full_24h_day = TRUE AND
    _nth_day BETWEEN 1 AND 7` (Phase 5, 2026-05-08), which sidesteps
    the partial-day binning question entirely.

    `_single_shift_day` is kept as a derived column for back-compat —
    `True` whenever either shift had zero coverage hours.
    """)
    return


@app.cell
def _(
    SITE_NAME,
    cci_df,
    covariates_t1,
    covs_daily,
    elix_df,
    hosp_df,
    icu_type_df,
    patient_df,
    sbt_outcomes_daily,
    seddose_by_id_imvday,
    sofa_daily,
    weight_daily,
):
    model_input_relation = mo.sql(
        f"""
        -- Consolidated per-day modeling input. Base table = the canonical
        -- patient-day registry; every other source LEFT-joins onto it.
        -- Day flags (_is_first_day / _is_last_partial_day / _is_last_full_day
        -- / _is_full_24h_day) and hour counts (n_hrs_day / n_hrs_night)
        -- come straight from the registry. Rate divisors use NULLIF(n_hrs_*,
        -- 0) — equivalent to /12.0 on the outcome-modeling filter view
        -- (verified at both sites: byte-identical numerics on the
        -- modeling cohort because partial-shift rows are filtered out).
        FROM read_parquet('output/{SITE_NAME}/cohort_meta_by_id_imvday.parquet') reg
        LEFT JOIN sbt_outcomes_daily o USING (hospitalization_id, _nth_day)
        LEFT JOIN seddose_by_id_imvday s USING (hospitalization_id, _nth_day)
        LEFT JOIN covs_daily c USING (hospitalization_id, _nth_day)
        LEFT JOIN hosp_df h USING (hospitalization_id)
        LEFT JOIN patient_df p USING (patient_id)
        LEFT JOIN icu_type_df i USING (hospitalization_id)
        LEFT JOIN sofa_daily sf USING (hospitalization_id, _nth_day)
        LEFT JOIN cci_df cc USING (hospitalization_id)
        LEFT JOIN elix_df ex USING (hospitalization_id)
        LEFT JOIN covariates_t1 t1 USING (hospitalization_id)
        LEFT JOIN weight_daily wd USING (hospitalization_id, _nth_day)
        SELECT
            -- Registry columns (canonical patient-day metadata)
            reg.hospitalization_id
            , reg._nth_day
            , reg.encounter_block
            , reg.day_type
            , reg._is_first_day
            -- _is_last_partial_day = truncated extubation day (day_type='last_partial').
            -- _is_last_full_day    = patient's final full-24h row (last row that
            --                        survives the modeling-cohort filter).
            -- See registry build cell in 01_cohort.py for full definitions.
            , reg._is_last_partial_day
            , reg._is_last_full_day
            , reg._is_full_24h_day
            , reg.n_hrs_day
            , reg.n_hrs_night
            , reg.day_start_dttm
            , reg.day_end_dttm
            , reg.nmb_excluded
            , reg.nmb_total_min
            -- Derived back-compat flag — `True` if either shift had zero hrs.
            -- For non-first_partial rows this equals ~_is_full_24h_day; for
            -- first_partial rows it's True (only one shift had any coverage).
            , _single_shift_day: (COALESCE(reg.n_hrs_day, 0) = 0
                                    OR COALESCE(reg.n_hrs_night, 0) = 0)
            -- Outcome columns + sensitivity siblings (LEAD over registry order
            -- gives next-day outcomes; rows that don't match sbt_outcomes_daily
            -- contribute NULL, which propagates correctly).
            , _sbt_done_today: o.sbt_done
            , _success_extub_today: o._success_extub
            , sbt_done_next_day: LEAD(o.sbt_done) OVER w
            , success_extub_next_day: LEAD(o._success_extub) OVER w
            , _sbt_done_anyprior_today: o.sbt_done_anyprior
            , sbt_done_anyprior_next_day: LEAD(o.sbt_done_anyprior) OVER w
            , _sbt_done_imv6h_today: o.sbt_done_imv6h
            , sbt_done_imv6h_next_day: LEAD(o.sbt_done_imv6h) OVER w
            , _sbt_done_prefix_today: o.sbt_done_prefix
            , sbt_done_prefix_next_day: LEAD(o.sbt_done_prefix) OVER w
            , _sbt_done_multiday_today: o.sbt_done_multiday
            , sbt_done_multiday_next_day: LEAD(o.sbt_done_multiday) OVER w
            , _sbt_done_2min_today: o.sbt_done_2min
            , sbt_done_2min_next_day: LEAD(o.sbt_done_2min) OVER w
            , _sbt_done_subira_today: o.sbt_done_subira
            , sbt_done_subira_next_day: LEAD(o.sbt_done_subira) OVER w
            , _sbt_done_abc_today: o.sbt_done_abc
            , sbt_done_abc_next_day: LEAD(o.sbt_done_abc) OVER w
            -- v2 outcome family
            , _sbt_elig_today: o.sbt_elig
            , sbt_elig_next_day: LEAD(o.sbt_elig) OVER w
            , _sbt_done_v2_today: o.sbt_done_v2
            , sbt_done_v2_next_day: LEAD(o.sbt_done_v2) OVER w
            , _success_extub_v2_today: o._success_extub_v2
            , success_extub_v2_next_day: LEAD(o._success_extub_v2) OVER w
            -- Per-shift avg rates (already per-shift averages upstream;
            -- COALESCE handles the rare LEFT-JOIN-miss case). Both `_cont`
            -- (continuous-only) and `_total` (cont + intm) variants are
            -- exposed so model specs can choose analytical scope.
            , _prop_day_mcg_kg_min_cont:    COALESCE(s.prop_day_mcg_kg_min_cont, 0)
            , _prop_night_mcg_kg_min_cont:  COALESCE(s.prop_night_mcg_kg_min_cont, 0)
            , _prop_day_mcg_kg_min_total:   COALESCE(s.prop_day_mcg_kg_min_total, 0)
            , _prop_night_mcg_kg_min_total: COALESCE(s.prop_night_mcg_kg_min_total, 0)
            , _fenteq_day_mcg_hr_cont:      COALESCE(s.fenteq_day_mcg_hr_cont, 0)
            , _fenteq_night_mcg_hr_cont:    COALESCE(s.fenteq_night_mcg_hr_cont, 0)
            , _fenteq_day_mcg_hr_total:     COALESCE(s.fenteq_day_mcg_hr_total, 0)
            , _fenteq_night_mcg_hr_total:   COALESCE(s.fenteq_night_mcg_hr_total, 0)
            , _midazeq_day_mg_hr_cont:      COALESCE(s.midazeq_day_mg_hr_cont, 0)
            , _midazeq_night_mg_hr_cont:    COALESCE(s.midazeq_night_mg_hr_cont, 0)
            , _midazeq_day_mg_hr_total:     COALESCE(s.midazeq_day_mg_hr_total, 0)
            , _midazeq_night_mg_hr_total:   COALESCE(s.midazeq_night_mg_hr_total, 0)
            -- Hurdle-binary flags (any 24h exposure on the total scope)
            , _prop_any:   CAST(COALESCE(s.prop_day_mcg_kg_min_total, 0) > 0
                                 OR COALESCE(s.prop_night_mcg_kg_min_total, 0) > 0 AS INTEGER)
            , _fenteq_any: CAST(COALESCE(s.fenteq_day_mcg_hr_total, 0) > 0
                                 OR COALESCE(s.fenteq_night_mcg_hr_total, 0) > 0 AS INTEGER)
            , _midazeq_any: CAST(COALESCE(s.midazeq_day_mg_hr_total, 0) > 0
                                  OR COALESCE(s.midazeq_night_mg_hr_total, 0) > 0 AS INTEGER)
            -- Day-night rate differences (night − day) on the total scope.
            , prop_dif_mcg_kg_min:
                COALESCE(s.prop_night_mcg_kg_min_total, 0)
                - COALESCE(s.prop_day_mcg_kg_min_total, 0)
            , fenteq_dif_mcg_hr:
                COALESCE(s.fenteq_night_mcg_hr_total, 0)
                - COALESCE(s.fenteq_day_mcg_hr_total, 0)
            , midazeq_dif_mg_hr:
                COALESCE(s.midazeq_night_mg_hr_total, 0)
                - COALESCE(s.midazeq_day_mg_hr_total, 0)
            -- Per-day covariate columns (7am/7pm vital snapshots etc.)
            , COLUMNS('(7am)|(7pm)')
            -- Per-stay rollups
            , age: h.age_at_admission
            , p.sex_category
            , i.icu_type
            , sofa_total: COALESCE(sf.sofa_total, 0)
            , cci_score: COALESCE(cc.cci_score, 0)
            , elix_score: COALESCE(ex.elix_score, 0)
            , t1.bmi
            , t1.height_cm
            , t1.weight_kg
            , wd.weight_kg_asof_day_start
            , t1.sofa_1st24h
            , t1.sofa_cv_97_1st24h
            , t1.sofa_coag_1st24h
            , t1.sofa_liver_1st24h
            , t1.sofa_resp_1st24h
            , t1.sofa_cns_1st24h
            , t1.sofa_renal_1st24h
            , ever_pressor: COALESCE(t1.ever_pressor, 0)
            , t1.pf_1st24h_min
            , t1.pf_1st24h_source
            , t1.imv_duration_hrs
            , sepsis_ase: COALESCE(t1.sepsis_ase, 0)
            , t1._first_icu_dttm
        WINDOW w AS (PARTITION BY reg.hospitalization_id ORDER BY reg._nth_day)
        ORDER BY reg.hospitalization_id, reg._nth_day
        """
    )
    return (model_input_relation,)


@app.cell
def _(model_input_relation, nmb_excluded):
    model_input_filtered = mo.sql(
        f"""
        -- Hospitalization-level NMB exclusion (matches the legacy two
        -- parquets) + drop rows whose hosp has no recorded age.
        FROM model_input_relation
        ANTI JOIN (SELECT DISTINCT hospitalization_id FROM nmb_excluded)
            USING (hospitalization_id)
        SELECT *
        WHERE age IS NOT NULL
        """
    )
    return (model_input_filtered,)


@app.cell
def _(SITE_NAME, model_input_filtered, to_utc):
    _df = model_input_filtered.df()
    _df = to_utc(_df, ["_first_icu_dttm"])
    _path = f"output/{SITE_NAME}/model_input_by_id_imvday.parquet"
    _df.to_parquet(_path, index=False)
    _n_full = int(_df['_is_full_24h_day'].sum())
    _n_first_partial = int((_df['day_type'] == 'first_partial').sum())
    _n_last_partial = int((_df['day_type'] == 'last_partial').sum())
    logger.info(
        f"Saved {_path} ({len(_df)} rows, "
        f"{_df['hospitalization_id'].nunique()} hospitalizations: "
        f"{_n_full} full-24h days, {_n_first_partial} first_partial, "
        f"{_n_last_partial} last_partial)"
    )
    return


if __name__ == "__main__":
    app.run()
