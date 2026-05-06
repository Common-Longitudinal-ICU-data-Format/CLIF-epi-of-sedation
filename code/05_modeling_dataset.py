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
    # 05 Modeling + Exposure Datasets

    Builds two sibling parquets from the same cohort base:

    1. **`modeling_dataset.parquet`** (`cohort_merged_final`) — the original
       outcome-modeling view. Filtered to
       `_nth_day > 0 AND sbt_done_next_day IS NOT NULL AND
       success_extub_next_day IS NOT NULL`. Drops day 0 and the last vent-course
       day so every row has a well-defined next-day outcome. Consumers:
       `06_table1.py`, `07_descriptive.py` (Table 1 / cohort summaries),
       `08_models.py` (production GEE / logit fits).

    2. **`exposure_dataset.parquet`** (`cohort_merged_exposure`) — the diurnal-
       characterization view. Filter relaxes to `_nth_day >= 0` (keeps day 0
       AND last day) so the figures in `code/descriptive/` see the full
       hospital-stay coverage. Rate divisors switch from `/12.0` to
       `NULLIF(n_hours_*, 0)` so single-shift rates are correctly hour-
       normalized; zero-hour shifts (intubated-after-7-PM day-0 rows) yield
       NULL rates rather than misleading zeros. Adds three flag columns
       (`_single_shift_day`, `_is_first_day`, `_is_last_day`) so figures
       can stratify visualizations on day 0 / last-day / coverage-artifact
       rows without re-deriving them. Consumers: every `code/descriptive/*.py`
       script (via `_shared.load_exposure()`), and `08_models.py`'s day-0
       sensitivity model (which applies the next-day-outcome filter inline at
       read time so its modeling cohort matches the production cell exactly).

    Both views apply the same NMB hospitalization-level exclusion.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    from _utils import retag_to_local_tz
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
    print(f"Site: {SITE_NAME} (tz: {SITE_TZ})")
    return CONFIG_PATH, SITE_NAME, SITE_TZ, apply_outlier_handling, pd, retag_to_local_tz


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load Intermediate Outputs
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    sbt_outcomes_daily = pd.read_parquet(f"output/{SITE_NAME}/sbt_outcomes_daily.parquet")
    print(f"sbt_outcomes_daily: {len(sbt_outcomes_daily)} rows")
    return (sbt_outcomes_daily,)


@app.cell
def _(SITE_NAME, pd):
    sed_dose_daily = pd.read_parquet(f"output/{SITE_NAME}/sed_dose_daily.parquet")
    print(f"sed_dose_daily: {len(sed_dose_daily)} rows")
    return (sed_dose_daily,)


@app.cell
def _(SITE_NAME, pd):
    covs_daily = pd.read_parquet(f"output/{SITE_NAME}/covariates_daily.parquet")
    print(f"covs_daily: {len(covs_daily)} rows")
    return (covs_daily,)


@app.cell
def _(SITE_NAME, pd):
    nmb_excluded = pd.read_parquet(f"output/{SITE_NAME}/nmb_excluded.parquet")
    print(f"nmb_excluded patient-days: {len(nmb_excluded)}")
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
    print(f"hosp_df: {len(hosp_df)} rows")
    return (hosp_df,)


@app.cell
def _(CONFIG_PATH):
    from clifpy import Patient
    _patient = Patient.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'sex_category'],
    )
    patient_df = _patient.df
    print(f"patient_df: {len(patient_df)} rows")
    return (patient_df,)


@app.cell
def _(SITE_NAME, pd):
    sofa_daily = pd.read_parquet(f"output/{SITE_NAME}/sofa_daily.parquet")
    icu_type_df = pd.read_parquet(f"output/{SITE_NAME}/icu_type.parquet")
    cci_df = pd.read_parquet(f"output/{SITE_NAME}/cci.parquet")
    elix_df = pd.read_parquet(f"output/{SITE_NAME}/elix.parquet")
    covariates_t1 = pd.read_parquet(f"output/{SITE_NAME}/covariates_t1.parquet")
    weight_daily = pd.read_parquet(f"output/{SITE_NAME}/weight_daily.parquet")
    print(
        f"sofa_daily: {len(sofa_daily)}, icu_type: {len(icu_type_df)}, "
        f"cci: {len(cci_df)}, elix: {len(elix_df)}, "
        f"covariates_t1: {len(covariates_t1)}, weight_daily: {len(weight_daily)}"
    )
    return cci_df, covariates_t1, elix_df, icu_type_df, sofa_daily, weight_daily


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Merge
    """)
    return


@app.cell
def _(
    cci_df,
    covariates_t1,
    covs_daily,
    elix_df,
    hosp_df,
    icu_type_df,
    patient_df,
    sbt_outcomes_daily,
    sed_dose_daily,
    sofa_daily,
    weight_daily,
):
    cohort_merged = mo.sql(
        f"""
        -- Merge all intermediate outputs into analytical dataset.
        -- covariates_t1 is hospitalization-level (from 04_covariates Cell H) and broadcasts
        -- its values across all patient-days for each hospitalization.
        FROM sbt_outcomes_daily o
        LEFT JOIN sed_dose_daily s USING (hospitalization_id, _nth_day)
        LEFT JOIN covs_daily c USING (hospitalization_id, _nth_day)
        LEFT JOIN hosp_df h USING (hospitalization_id)
        LEFT JOIN patient_df p USING (patient_id)
        LEFT JOIN icu_type_df i USING (hospitalization_id)
        LEFT JOIN sofa_daily sf USING (hospitalization_id, _nth_day)
        LEFT JOIN cci_df cc USING (hospitalization_id)
        LEFT JOIN elix_df ex USING (hospitalization_id)
        LEFT JOIN covariates_t1 t1 USING (hospitalization_id)
        LEFT JOIN weight_daily wd USING (hospitalization_id, _nth_day)
        SELECT o.hospitalization_id
        , o._nth_day
        , _sbt_done_today: o.sbt_done
        , _success_extub_today: o._success_extub
        , sbt_done_next_day: LEAD(o.sbt_done) OVER w
        , success_extub_next_day: LEAD(o._success_extub) OVER w
        -- Sensitivity-sibling SBT outcomes (see docs/outcomes_specs.md
        -- "Sensitivity siblings"). Each varies the prior-mode operationalization
        -- or sustained-duration threshold; primary `sbt_done` above is the
        -- spec-literal version. The cohort filter at the bottom of this cell
        -- still anchors on `sbt_done_next_day` (primary) — variants ride along
        -- as additional outcome columns for sensitivity comparison in 08_models.py.
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
        -- v2 outcome family (ABT-RISE-style alternative implementations,
        -- 2026-04-29 model-update round). sbt_elig is genuinely new; the
        -- v2 sbt_done / extub variants run alongside the working baselines
        -- so the modeling cell can fit both and the forest plot can compare.
        , _sbt_elig_today: o.sbt_elig
        , sbt_elig_next_day: LEAD(o.sbt_elig) OVER w
        , _sbt_done_v2_today: o.sbt_done_v2
        , sbt_done_v2_next_day: LEAD(o.sbt_done_v2) OVER w
        , _success_extub_v2_today: o._success_extub_v2
        , success_extub_v2_next_day: LEAD(o._success_extub_v2) OVER w
        -- NOTE: Dose columns below are per-hour RATES (mg/hr or mcg/hr),
        -- computed as shift totals ÷ 12 (hours per shift). Valid because the
        -- filter (_nth_day > 0 AND outcome-columns non-null) guarantees
        -- complete 24h 7am-anchored days — single-shift days at intubation /
        -- extubation are already excluded. See
        -- docs/uptitration_paradox_investigation.md §0 for why this filter
        -- matters when characterizing *exposure* (vs predicting outcomes).
        -- Column-name convention (2026-04-24): every dose column has its unit
        -- encoded as a suffix. `_mg_hr` = per-hour rate in mg; `_mcg_hr` = per-
        -- hour rate in mcg. Source shift totals in sed_dose_daily.parquet
        -- likewise carry `_mg` / `_mcg` suffixes (total dose over 12-hour shift).
        -- Phase 2: propofol rates are now in mcg/kg/min (was mg/hr).
        -- The shift total `prop_day_mcg_kg` is total mcg/kg over 12h →
        -- ÷ 12 = mcg/kg/hr → ÷ 60 = mcg/kg/min.
        , _prop_day_mcg_kg_min:   COALESCE(s.prop_day_mcg_kg, 0)   / 12.0 / 60.0
        , _prop_night_mcg_kg_min: COALESCE(s.prop_night_mcg_kg, 0) / 12.0 / 60.0
        , _fenteq_day_mcg_hr:   COALESCE(s.fenteq_day_mcg, 0) / 12.0
        , _fenteq_night_mcg_hr: COALESCE(s.fenteq_night_mcg, 0) / 12.0
        , _midazeq_day_mg_hr:   COALESCE(s.midazeq_day_mg, 0) / 12.0
        , _midazeq_night_mg_hr: COALESCE(s.midazeq_night_mg, 0) / 12.0
        -- Hurdle binaries: did the patient receive ANY of this drug during the
        -- past 24h (day OR night shift)? Pairs with the continuous daytime-rate
        -- column above so the daytime-level effect can be decomposed into "any
        -- 24h exposure" (selection / clinician choice) vs. "daytime dose given
        -- exposed" (intensity). The 24h indicator (vs daytime-only) avoids
        -- misclassifying night-only-sedated patients as non-users.
        , _prop_any:   CAST(COALESCE(s.prop_day_mcg_kg, 0)   > 0 OR COALESCE(s.prop_night_mcg_kg, 0)   > 0 AS INTEGER)
        , _fenteq_any: CAST(COALESCE(s.fenteq_day_mcg, 0)    > 0 OR COALESCE(s.fenteq_night_mcg, 0)    > 0 AS INTEGER)
        , _midazeq_any: CAST(COALESCE(s.midazeq_day_mg, 0)   > 0 OR COALESCE(s.midazeq_night_mg, 0)    > 0 AS INTEGER)
        , prop_dif_mcg_kg_min: (COALESCE(s.prop_night_mcg_kg, 0) - COALESCE(s.prop_day_mcg_kg, 0)) / 12.0 / 60.0
        , fenteq_dif_mcg_hr:  (COALESCE(s.fenteq_night_mcg, 0) - COALESCE(s.fenteq_day_mcg, 0)) / 12.0
        , midazeq_dif_mg_hr:  (COALESCE(s.midazeq_night_mg, 0) - COALESCE(s.midazeq_day_mg, 0)) / 12.0
        -- Absolute totals (per 12-hr shift) — kept alongside the rate columns
        -- above for the sensitivity model in 08_models.py that regresses on
        -- absolute amounts instead of per-hour rates. Pre-2026-04-24 this was
        -- the only parameterization; restored as a SA variant. For full 12-h
        -- shifts (everything in this dataset post-filter), amount = rate × 12
        -- exactly, so the two model fits should agree up to a unit rescaling.
        -- Phase 2: propofol absolute totals now mcg/kg over the 12h shift.
        , _prop_day_mcg_kg:   COALESCE(s.prop_day_mcg_kg, 0)
        , _prop_night_mcg_kg: COALESCE(s.prop_night_mcg_kg, 0)
        , _fenteq_day_mcg:   COALESCE(s.fenteq_day_mcg, 0)
        , _fenteq_night_mcg: COALESCE(s.fenteq_night_mcg, 0)
        , _midazeq_day_mg:   COALESCE(s.midazeq_day_mg, 0)
        , _midazeq_night_mg: COALESCE(s.midazeq_night_mg, 0)
        , prop_dif_mcg_kg: COALESCE(s.prop_night_mcg_kg, 0) - COALESCE(s.prop_day_mcg_kg, 0)
        , fenteq_dif_mcg: COALESCE(s.fenteq_night_mcg, 0) - COALESCE(s.fenteq_day_mcg, 0)
        , midazeq_dif_mg: COALESCE(s.midazeq_night_mg, 0) - COALESCE(s.midazeq_day_mg, 0)
        , COLUMNS('(7am)|(7pm)')
        , age: h.age_at_admission
        , p.sex_category
        , i.icu_type
        , sofa_total: COALESCE(sf.sofa_total,0)
        , cci_score: COALESCE(cc.cci_score, 0)
        , elix_score: COALESCE(ex.elix_score, 0)
        -- Table 1 hospitalization-level covariates (from 04_covariates.py covariates_t1.parquet)
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
        WINDOW w AS (PARTITION BY o.hospitalization_id ORDER BY o._nth_day)
        ORDER BY o.hospitalization_id, o._nth_day
        """
    )
    return (cohort_merged,)


@app.cell
def _(cohort_merged):
    _merged_df = cohort_merged.df()
    _merged_df.dropna(subset=['age'], inplace=True)
    cohort_merged_clean = _merged_df
    print(f"After dropping null age: {len(cohort_merged_clean)} rows")
    return (cohort_merged_clean,)


@app.cell
def _(cohort_merged_clean, nmb_excluded):
    cohort_merged_final = mo.sql(
        f"""
        -- NOTE: Hospitalization-level NMB exclusion (exclude entire hosp if any day had >1h NMB).
        -- Original spec was patient-day level; changed per PI — NMB patients are a distinct population.
        -- To revert to patient-day exclusion, replace the ANTI JOIN with:
        --   ANTI JOIN nmb_excluded USING (hospitalization_id, _nth_day)
        FROM cohort_merged_clean
        ANTI JOIN (SELECT DISTINCT hospitalization_id FROM nmb_excluded) USING (hospitalization_id)
        SELECT *
        WHERE _nth_day > 0 AND sbt_done_next_day IS NOT NULL AND success_extub_next_day IS NOT NULL
        """
    )
    return (cohort_merged_final,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Save
    """)
    return


@app.cell
def _(SITE_NAME, SITE_TZ, cohort_merged_final, retag_to_local_tz):
    _df = cohort_merged_final.df()
    # _first_icu_dttm flows in from covariates_t1; retag so the on-disk tz
    # tag is site-local regardless of who runs the script.
    _df = retag_to_local_tz(_df, ["_first_icu_dttm"], SITE_TZ)
    _path = f"output/{SITE_NAME}/modeling_dataset.parquet"
    _df.to_parquet(_path, index=False)
    print(f"Saved {_path} ({len(_df)} rows, {_df['hospitalization_id'].nunique()} hospitalizations)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Exposure Dataset (full hospital-stay coverage)

    Builds `exposure_dataset.parquet` — a sibling of the modeling dataset that
    retains **day 0** (the partial admission day before the first 7am) AND the
    **last vent-course day** (where `sbt_done_next_day` is NULL because there is
    no next day). Three differences from the modeling chain above:

    1. Rate divisors swap from hardcoded `/12.0` to `NULLIF(s.n_hours_<shift>, 0)`.
       For full 12-h shifts (every middle-of-stay row) this is mathematically
       identical; for single-shift rows (day 0, last day, mid-stay extubation
       gaps) it correctly hour-normalizes per-shift rates and avoids the
       "higher total just because more hours" bias. Zero-hour shifts yield
       NULL rates rather than 0 — the artifact those rows would otherwise
       introduce is then surfaced by `_single_shift_day` (below) instead of
       being silently lumped with real low-dose rows.

    2. Filter relaxes from
       `_nth_day > 0 AND sbt_done_next_day IS NOT NULL AND success_extub_next_day IS NOT NULL`
       to just `_nth_day >= 0`. Day 0 and last day are now both included.

    3. Adds four flag columns so descriptive figures can stratify
       visualizations natively:
       - `_single_shift_day` = `True` when one shift had zero hours of
         coverage (e.g., intubation after 7 PM → `n_hours_day = 0` on day 0).
         Strictly the zero-hour case; short-but-nonzero shifts are unflagged.
       - `_is_first_day` = `True` when `_nth_day == MIN(_nth_day) OVER w`,
         i.e., the row with the smallest `_nth_day` per hospitalization. This
         catches hospitalizations whose IMV streak begins exactly at 7:00 AM
         (whose first row is numbered `_nth_day = 1` not `0` by the upstream
         `add_day_shift_id` helper).
       - `_is_last_day` = `True` when `LEAD(_nth_day) OVER w IS NULL`, i.e.,
         the row with the largest `_nth_day` per hospitalization. Defined on
         the ordering column (not on `sbt_done`) so it can never multi-fire.
       - `_rel_day` = `_nth_day - MIN(_nth_day) OVER w` — days since the
         hospitalization's first cohort row. Use this for ICU-day-N figure
         x-bins to get monotonic cohort attrition.

    All three are computed in this cell; consumers don't need to re-derive
    them. The amount columns (`_prop_day_mcg_kg`, etc.) are kept as-is — for
    single-shift rows they represent total dose over the actual shift hours.

    `08_models.py`'s day-0 sensitivity GEE applies the next-day-outcome filter
    inline at read time (so the model cohort matches the production cell
    exactly).
    """)
    return


@app.cell
def _(
    cci_df,
    covariates_t1,
    covs_daily,
    elix_df,
    hosp_df,
    icu_type_df,
    patient_df,
    sbt_outcomes_daily,
    sed_dose_daily,
    sofa_daily,
    weight_daily,
):
    cohort_merged_exposure = mo.sql(
        f"""
        -- Exposure dataset — sibling of the production modeling cohort,
        -- built for diurnal characterization (descriptive figures). Two
        -- core differences from the modeling chain:
        --   (1) Rate divisors use NULLIF(n_hours_*, 0) instead of /12.0
        --       so single-shift rows are correctly hour-normalized and
        --       zero-hour shifts produce NULL rates (not misleading 0).
        --   (2) Three exposure-characterization flag columns are added
        --       (_single_shift_day, _is_first_day, _is_last_day) so
        --       descriptive figures can stratify on them natively.
        FROM sbt_outcomes_daily o
        LEFT JOIN sed_dose_daily s USING (hospitalization_id, _nth_day)
        LEFT JOIN covs_daily c USING (hospitalization_id, _nth_day)
        LEFT JOIN hosp_df h USING (hospitalization_id)
        LEFT JOIN patient_df p USING (patient_id)
        LEFT JOIN icu_type_df i USING (hospitalization_id)
        LEFT JOIN sofa_daily sf USING (hospitalization_id, _nth_day)
        LEFT JOIN cci_df cc USING (hospitalization_id)
        LEFT JOIN elix_df ex USING (hospitalization_id)
        LEFT JOIN covariates_t1 t1 USING (hospitalization_id)
        LEFT JOIN weight_daily wd USING (hospitalization_id, _nth_day)
        SELECT o.hospitalization_id
        , o._nth_day
        -- Exposure-characterization flags (computed once here so descriptive
        -- figures don't have to re-derive them across N scripts).
        -- _is_first_day catches the 506 (MIMIC) / 376 (UCMC) hospitalizations
        -- whose IMV streak begins exactly at 7:00 AM and therefore have their
        -- first row labeled `_nth_day = 1` (not 0) by `_utils.add_day_shift_id`.
        -- MIN(_nth_day) per hosp recovers the per-streak first day regardless
        -- of whether it's literally numbered 0 or 1.
        -- KNOWN ISSUE: this is a workaround for a bug in _utils.add_day_shift_id;
        -- see docs/descriptive_figures.md §6.9.1 for the full writeup and the
        -- proper-fix path. Revisit before the next manuscript revision.
        , _is_first_day: (o._nth_day = MIN(o._nth_day) OVER w)
        -- _is_last_day uses LEAD(_nth_day) instead of LEAD(sbt_done) — the
        -- former depends only on the partition ordering column and can never
        -- multi-fire, while the latter would over-count if any mid-sequence
        -- row had a NULL sbt_done.
        , _is_last_day: (LEAD(o._nth_day) OVER w IS NULL)
        -- _rel_day = days since each hospitalization's first cohort row.
        -- Use this for ICU-day-N figure x-bins so day 0 is uniformly "first
        -- day in the cohort" rather than "calendar day 0", giving monotonic
        -- cohort attrition. Becomes redundant if §6.9.1's proper fix lands.
        , _rel_day: (o._nth_day - MIN(o._nth_day) OVER w)
        -- TODO (§6.9.2): add `_full_shift_day := (s.n_hours_day = 12 AND
        -- s.n_hours_night = 12)` so the trimmed-shift class becomes flag-
        -- separable from full-shift days. Currently both lump under
        -- `_single_shift_day = False`.
        , _single_shift_day: (COALESCE(s.n_hours_day, 0) = 0
                                OR COALESCE(s.n_hours_night, 0) = 0)
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
        -- v2 outcome family (ABT-RISE-style alternatives, 2026-04-29).
        , _sbt_elig_today: o.sbt_elig
        , sbt_elig_next_day: LEAD(o.sbt_elig) OVER w
        , _sbt_done_v2_today: o.sbt_done_v2
        , sbt_done_v2_next_day: LEAD(o.sbt_done_v2) OVER w
        , _success_extub_v2_today: o._success_extub_v2
        , success_extub_v2_next_day: LEAD(o._success_extub_v2) OVER w
        -- N-hours-aware rate columns. NULLIF(n_hours_*, 0) yields NULL when
        -- the patient had zero hours on that shift (e.g., extubation occurred
        -- exactly at 7am), which then propagates through the arithmetic to a
        -- NULL rate — same downstream behavior as having no recorded dose.
        -- For all-12h-shift rows (i.e., every existing day-1+ row), this
        -- expression equals the production /12.0 exactly.
        -- Phase 2: propofol rates are now in mcg/kg/min.
        , _prop_day_mcg_kg_min:   COALESCE(s.prop_day_mcg_kg, 0)   / NULLIF(s.n_hours_day, 0)   / 60.0
        , _prop_night_mcg_kg_min: COALESCE(s.prop_night_mcg_kg, 0) / NULLIF(s.n_hours_night, 0) / 60.0
        , _fenteq_day_mcg_hr:   COALESCE(s.fenteq_day_mcg, 0)   / NULLIF(s.n_hours_day, 0)
        , _fenteq_night_mcg_hr: COALESCE(s.fenteq_night_mcg, 0) / NULLIF(s.n_hours_night, 0)
        , _midazeq_day_mg_hr:   COALESCE(s.midazeq_day_mg, 0)   / NULLIF(s.n_hours_day, 0)
        , _midazeq_night_mg_hr: COALESCE(s.midazeq_night_mg, 0) / NULLIF(s.n_hours_night, 0)
        -- Hurdle binaries (kept identical to the production query above)
        , _prop_any:   CAST(COALESCE(s.prop_day_mcg_kg, 0)   > 0 OR COALESCE(s.prop_night_mcg_kg, 0)   > 0 AS INTEGER)
        , _fenteq_any: CAST(COALESCE(s.fenteq_day_mcg, 0)    > 0 OR COALESCE(s.fenteq_night_mcg, 0)    > 0 AS INTEGER)
        , _midazeq_any: CAST(COALESCE(s.midazeq_day_mg, 0)   > 0 OR COALESCE(s.midazeq_night_mg, 0)    > 0 AS INTEGER)
        , prop_dif_mcg_kg_min: (COALESCE(s.prop_night_mcg_kg, 0) / NULLIF(s.n_hours_night, 0) / 60.0)
                             - (COALESCE(s.prop_day_mcg_kg, 0)   / NULLIF(s.n_hours_day, 0)   / 60.0)
        , fenteq_dif_mcg_hr: (COALESCE(s.fenteq_night_mcg, 0) / NULLIF(s.n_hours_night, 0))
                           - (COALESCE(s.fenteq_day_mcg, 0)   / NULLIF(s.n_hours_day, 0))
        , midazeq_dif_mg_hr: (COALESCE(s.midazeq_night_mg, 0) / NULLIF(s.n_hours_night, 0))
                           - (COALESCE(s.midazeq_day_mg, 0)   / NULLIF(s.n_hours_day, 0))
        -- Absolute totals — kept identical to production. For day 0 these are
        -- "total over single-shift day" which is the natural absolute-amount SA.
        -- Phase 2: propofol absolute totals now mcg/kg over the shift.
        , _prop_day_mcg_kg:   COALESCE(s.prop_day_mcg_kg, 0)
        , _prop_night_mcg_kg: COALESCE(s.prop_night_mcg_kg, 0)
        , _fenteq_day_mcg:   COALESCE(s.fenteq_day_mcg, 0)
        , _fenteq_night_mcg: COALESCE(s.fenteq_night_mcg, 0)
        , _midazeq_day_mg:   COALESCE(s.midazeq_day_mg, 0)
        , _midazeq_night_mg: COALESCE(s.midazeq_night_mg, 0)
        , prop_dif_mcg_kg: COALESCE(s.prop_night_mcg_kg, 0) - COALESCE(s.prop_day_mcg_kg, 0)
        , fenteq_dif_mcg: COALESCE(s.fenteq_night_mcg, 0) - COALESCE(s.fenteq_day_mcg, 0)
        , midazeq_dif_mg: COALESCE(s.midazeq_night_mg, 0) - COALESCE(s.midazeq_day_mg, 0)
        , COLUMNS('(7am)|(7pm)')
        , age: h.age_at_admission
        , p.sex_category
        , i.icu_type
        , sofa_total: COALESCE(sf.sofa_total,0)
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
        WINDOW w AS (PARTITION BY o.hospitalization_id ORDER BY o._nth_day)
        ORDER BY o.hospitalization_id, o._nth_day
        """
    )
    return (cohort_merged_exposure,)


@app.cell
def _(cohort_merged_exposure):
    _merged_df = cohort_merged_exposure.df()
    _merged_df.dropna(subset=['age'], inplace=True)
    cohort_merged_exposure_clean = _merged_df
    print(f"[day-0] After dropping null age: {len(cohort_merged_exposure_clean)} rows")
    return (cohort_merged_exposure_clean,)


@app.cell
def _(cohort_merged_exposure_clean, nmb_excluded):
    cohort_merged_exposure_final = mo.sql(
        f"""
        -- Same hospitalization-level NMB exclusion as the modeling cohort.
        -- Filter relaxes to just `_nth_day >= 0`: day 0 is now included
        -- AND last-day rows (where `sbt_done_next_day` is NULL) are retained,
        -- since this is the exposure-characterization view. Consumers needing
        -- next-day-outcome rows (e.g., 08_models.py day-0 SA) re-apply the
        -- next-day-NOT-NULL filter inline at read time.
        FROM cohort_merged_exposure_clean
        ANTI JOIN (SELECT DISTINCT hospitalization_id FROM nmb_excluded) USING (hospitalization_id)
        SELECT *
        WHERE _nth_day >= 0
        """
    )
    return (cohort_merged_exposure_final,)


@app.cell
def _(SITE_NAME, SITE_TZ, cohort_merged_exposure_final, retag_to_local_tz):
    _df = cohort_merged_exposure_final.df()
    _df = retag_to_local_tz(_df, ["_first_icu_dttm"], SITE_TZ)
    _path = f"output/{SITE_NAME}/exposure_dataset.parquet"
    _df.to_parquet(_path, index=False)
    _n_day0 = (_df['_nth_day'] == 0).sum()
    _n_last = int(_df['_is_last_day'].sum())
    _n_single = int(_df['_single_shift_day'].sum())
    print(
        f"Saved {_path} ({len(_df)} rows, {_df['hospitalization_id'].nunique()} hospitalizations, "
        f"{_n_day0} day-0 rows, {_n_last} last-day rows, {_n_single} single-shift rows)"
    )
    return


if __name__ == "__main__":
    app.run()
