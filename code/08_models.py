# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.3.1",
#     "statsmodels>=0.14.5",
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
    logger = get_logger("epi_sedation.models")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 08 Models

    Primary models (GEE for SBT, logistic regression for extubation)
    and sensitivity analysis infrastructure.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from _logging_setup import setup_logging
    import pandas as pd

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()
    # Mirror of 03_outcomes.py's flag (see that file for full rationale).
    # When false, the v2 outcome family (success_extub_v2, sbt_done_v2) is
    # all-zero in the modeling dataset, and 08 explicitly skips those fits.
    ENABLE_V2_OUTCOMES = bool(cfg.get('enable_v2_outcomes', True))

    # Site-scoped output dirs (see Makefile SITE= flag).
    # Path B++ refactor: every modeling artifact lands under {site}/models/.
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    os.makedirs(f"output_to_share/{SITE_NAME}/models", exist_ok=True)
    # Per-site dual log files (pyCLIF integration guide rule 1).
    setup_logging(output_directory=f"output_to_share/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME}; enable_v2_outcomes: {ENABLE_V2_OUTCOMES}")
    return ENABLE_V2_OUTCOMES, SITE_NAME, pd


@app.cell
def _(SITE_NAME, pd):
    # Phase 4 cutover (2026-05-08): read consolidated parquet + apply
    # outcome-modeling filter inline. Byte-equivalent to the legacy
    # modeling_dataset.parquet on the surviving cohort. The day-0
    # sensitivity analysis below flips `>` to `>=` to keep partial
    # intubation-day rows (no separate parquet needed).
    _full = pd.read_parquet(f"output/{SITE_NAME}/model_input_by_id_imvday.parquet")
    # Alias the `_total` (cont + intermittent) daytime/night dose columns
    # onto the unsuffixed names the model formulas use. Manuscript primary
    # scope is "all sedation forms" = `_total`. To switch the analysis to
    # continuous-only, point the rename at `_cont` columns instead.
    # (Pipeline produces both _cont and _total; we never use both at once.)
    _DAYDOSE_SCOPE = "_total"  # change to "_cont" for continuous-only SA
    _ALIAS = {
        f"_prop_day_mcg_kg_min{_DAYDOSE_SCOPE}":   "_prop_day_mcg_kg_min",
        f"_prop_night_mcg_kg_min{_DAYDOSE_SCOPE}": "_prop_night_mcg_kg_min",
        f"_fenteq_day_mcg_hr{_DAYDOSE_SCOPE}":     "_fenteq_day_mcg_hr",
        f"_fenteq_night_mcg_hr{_DAYDOSE_SCOPE}":   "_fenteq_night_mcg_hr",
        f"_midazeq_day_mg_hr{_DAYDOSE_SCOPE}":     "_midazeq_day_mg_hr",
        f"_midazeq_night_mg_hr{_DAYDOSE_SCOPE}":   "_midazeq_night_mg_hr",
    }
    _full = _full.rename(columns=_ALIAS)
    cohort_merged_final = _full.loc[
        (_full["_nth_day"] > 0)
        & _full["sbt_done_next_day"].notna()
        & _full["success_extub_next_day"].notna()
    ].reset_index(drop=True)
    logger.info(f"Modeling cohort: {len(cohort_merged_final)} rows (daydose scope: {_DAYDOSE_SCOPE})")
    return (cohort_merged_final,)


@app.cell
def _(cohort_merged_final):
    cohort_merged_final
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Retired cells (2026-05-01)

    Three standalone cells that fitted `sbt_done_next_day` as the primary
    SBT outcome were retired in this round:

    - **Standalone GEE for `sbt_done_next_day`** — produced
      `gee_summary.csv` + `gee_covmat.csv`.
    - **VIF diagnostic for the same SBT formula** — produced
      `vif_sbt_rate.csv`.
    - **Day-0 SA GEE** (refit on `exposure_dataset.parquet` with
      day-0 rows included) — produced `gee_summary_day0.csv` +
      `gee_covmat_day0.csv`.

    `sbt_done_next_day` is no longer in `MODEL_CONFIGS`, so these
    standalone cells produced orphan artifacts and noisy
    cluster-robust-covariance warnings. SBT outcomes the team is now
    reporting (sbt_done_multiday, sbt_done_subira, sbt_done_abc,
    sbt_done_v2, sbt_done_prefix) are fitted by the cross-product loop
    below and appear in `model_comparison_*.csv`, `models_coeffs.csv`,
    and `forest_*.png` artifacts.

    Original code preserved in git history; see
    `~/.claude/plans/2-related-tasks-around-memoized-hartmanis.md`
    for the retire decision context.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nested Model Comparison

    Fits 8 covariate specs across 11 (outcome × model_type) configurations
    for a total of 88 fits per site.

    **Model specs (all include exposures + BASELINE: prop_dif_mcg_kg_min +
    fenteq_dif_mcg_hr + midazeq_dif_mg_hr + age + _nth_day + C(sex_category)
    + C(icu_type) + cci_score + weight_kg):**

    1. **baseline**: BASELINE alone.

    2. **daydose**: baseline + 3 daytime continuous rates.

    3. **sofa**: daydose + `sofa_total`.

    4. **daydose_physio**: daydose + pH/PF/NEE at 7am+7pm
       (renamed 2026-05-11 from `clinical` — captures the organ-system
       physiology snapshot the spec actually adds).

    5. **daydose_rcs_diff** / **daydose_rcs_full**: RCS variants of
       `daydose`. `_diff` applies cr() to the 3 night-day diff vars only
       (parsimony). `_full` applies cr() to all 6 exposure vars.

    6. **daydose_physio_rcs_diff** / **daydose_physio_rcs_full**: same
       RCS variants on the `daydose_physio` adjustment set.

    **Outputs:**

    - `models_coeffs.csv` — long format, one row per coefficient (in EVERY
      fit), with per-unit and per-percentile log-OR + SE columns for
      exposures. Subsumes the prior `sensitivity_analysis.csv`.

    - Wide CSVs (`model_comparison_*.csv`) — rows=covariates, cols=specs, cells=OR (95% CI)

    **Variable scaling:** `VAR_DISPLAY` centralizes scaling and labels. Edit
    `scale` to rescale; edit `label` to relabel.
    """)
    return


@app.cell
def _():
    # Central configuration for variable scaling + display labels.
    # To change a scale (e.g., age per 10 yrs): edit the 'scale' divisor.
    # To change a label: edit 'label'.
    # To add a new scaled variable: add a new entry.
    #
    # UNIT NOTE: Phase 2 (2026-04-27): propofol exposures are now in
    # mcg/kg/min (the bedside pump-display unit) thanks to the pre-attached
    # weight column in 02_exposure.py and the preferred_units change to
    # mcg/kg/min. Other drugs unchanged: fentanyl mcg/hr, midazolam mg/hr.
    # Scales below: propofol "per 10 mcg/kg/min" matches typical pump
    # titration steps (5–25 mcg/kg/min increments).
    VAR_DISPLAY = {
        # Exposures (day-night rate differences). 2026-05-11: standardized
        # to fixed clinical units across diffs and daytime: prop = per 10
        # mcg/kg/min, fent eq = per 25 mcg/hr, midaz eq = per 1 mg/hr.
        # Replaces the prior mixed scales (fent per 10, midaz per 0.1).
        'prop_dif_mcg_kg_min': {'scale': 10, 'label': 'Δ propofol (per 10 mcg/kg/min)'},
        'fenteq_dif_mcg_hr':   {'scale': 25, 'label': 'Δ fentanyl eq (per 25 mcg/hr)'},
        'midazeq_dif_mg_hr':   {'scale': 1,  'label': 'Δ midazolam eq (per 1 mg/hr)'},
        # Daytime absolute rates (same fixed scales as the diffs above)
        '_prop_day_mcg_kg_min': {'scale': 10, 'label': 'Daytime propofol (per 10 mcg/kg/min)'},
        '_fenteq_day_mcg_hr':   {'scale': 25, 'label': 'Daytime fentanyl eq (per 25 mcg/hr)'},
        '_midazeq_day_mg_hr':   {'scale': 1,  'label': 'Daytime midazolam eq (per 1 mg/hr)'},
        # Hurdle binaries: any 24h exposure (day OR night, yes/no). scale=1
        # keeps the column 0/1 — the regression OR (= 10→90 OR when ≥10% of
        # rows are 1) is interpretable as "odds ratio for any exposure vs
        # none." 24h indicator (vs daytime-only) avoids misclassifying
        # night-only-sedated patients as non-users.
        '_prop_any':   {'scale': 1, 'label': 'Any propofol use (day or night, yes/no)'},
        '_fenteq_any': {'scale': 1, 'label': 'Any fentanyl eq use (day or night, yes/no)'},
        '_midazeq_any': {'scale': 1, 'label': 'Any midazolam eq use (day or night, yes/no)'},
        # Other continuous covariates
        'age':          {'scale': 1,   'label': 'Age (per year)'},
        '_nth_day':     {'scale': 1,   'label': 'Day on IMV (per day)'},
        'cci_score':    {'scale': 1,   'label': 'Charlson CCI (per point)'},
        'sofa_total':   {'scale': 1,   'label': 'SOFA total (per point)'},
        'nee_7am':      {'scale': 0.1, 'label': 'NEE 7am (per 0.1 mcg/kg/min)'},
        'nee_7pm':      {'scale': 0.1, 'label': 'NEE 7pm (per 0.1 mcg/kg/min)'},
        # Body habitus. weight_kg is in BASELINE so it appears in every
        # spec; bmi is retained in case a future BMI-adjusted spec is added.
        'weight_kg':    {'scale': 10,  'label': 'Weight (per 10 kg)'},
        'bmi':          {'scale': 5,   'label': 'BMI (per 5 kg/m²)'},
    }
    return (VAR_DISPLAY,)


@app.cell
def _(ENABLE_V2_OUTCOMES, SITE_NAME, VAR_DISPLAY, cohort_merged_final, pd):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import numpy as np
    import re

    # ── Rescale on a COPY so we never mutate cohort_merged_final ──────
    # Using copy() prevents accidental double-scaling if this cell re-runs.
    _df_scaled = cohort_merged_final.copy()
    for _col, _info in VAR_DISPLAY.items():
        if _col in _df_scaled.columns and _info['scale'] != 1:
            _df_scaled[_col] = _df_scaled[_col] / _info['scale']

    # ── Fit functions ──────────────────────────────────────────────────
    # maxiter=500 (statsmodels defaults: Logit=35, GEE=60). The earlier
    # 100-iter ceiling stopped success_extub_v2 × daydose_rcs_diff short of
    # convergence on logit + logit_asym (function value plateaued at ~0.4407
    # with near-zero gradient). 500 gives the optimizer headroom; fits still
    # hitting the cap here are structurally ill-conditioned rather than
    # iteration-bound, and the NO_CONVERGE flag propagates into
    # models_coeffs.csv (fit_status column) for downstream filtering.
    def _fit_gee(formula, data):
        m = smf.gee(formula=formula, groups='hospitalization_id',
                    data=data, family=sm.families.Binomial())
        return m.fit(maxiter=500)

    def _fit_logit(formula, data):
        """Cluster-robust logit (groups=hospitalization_id).

        Methodologically appropriate when within-cluster correlation
        exists (avg cluster size ~5 days here). On cr() basis × binary
        terminal-event outcome the sandwich estimator can produce
        degenerate link-scale variance — see `_fit_logit_asym` for the
        pragmatic alternative used in v1/v3 of the cross-site marginal-
        effects figure. v2 of the cross-site figure uses this version
        deliberately to illustrate the failure mode.
        """
        # Drop rows with NaN in any column referenced by the formula so the
        # cluster-robust SE's `groups` length matches the model's residuals.
        _names = [c for c in data.columns if c in formula]
        if 'hospitalization_id' not in _names:
            _names.append('hospitalization_id')
        _d = data.dropna(subset=_names)
        m = smf.logit(formula=formula, data=_d)
        return m.fit(cov_type='cluster',
                     cov_kwds={'groups': _d['hospitalization_id']},
                     maxiter=500, disp=False)

    def _fit_logit_asym(formula, data):
        """Asymptotic-SE logit (no cluster-robust).

        Anti-conservative when within-cluster correlation exists, but
        produces usable prediction CIs where `_fit_logit` can degenerate.
        Used by v1 (`_full` spec) and v3 (`_diff` spec) of the cross-site
        marginal-effects figure.
        """
        _names = [c for c in data.columns if c in formula]
        if 'hospitalization_id' not in _names:
            _names.append('hospitalization_id')
        _d = data.dropna(subset=_names)
        m = smf.logit(formula=formula, data=_d)
        return m.fit(maxiter=500, disp=False)

    def _is_converged(result):
        # Logit exposes mle_retvals['converged']; GEE exposes result.converged.
        # Default True when neither attribute is present (e.g. future statsmodels
        # API change) so we never spuriously flag a fit as non-converged.
        mle = getattr(result, 'mle_retvals', None)
        if isinstance(mle, dict) and 'converged' in mle:
            return bool(mle['converged'])
        if hasattr(result, 'converged'):
            return bool(result.converged)
        return True

    # ── Dimension 1: nested covariate sets (all include exposures) ────
    # Rate-based exposures (mcg/kg/min for propofol, mcg/hr for fentanyl,
    # mg/hr for midazolam). Amount-based parameterization was retired in the
    # 2026-04-29 model-update round — rate-only going forward.
    #
    # 2026-05-11 grid trim: weight_kg moved into BASELINE (in every spec).
    # Retired siblings: daydose_wt / clinical_wt (algebraic duplicates of
    # daydose / daydose_physio now that weight is in baseline);
    # daydose_anydose (hurdle decomposition); sofa_weight / sofa_bmi
    # (body-habitus). See git history for the retired formulas if needed.
    # Spec rename: clinical → daydose_physio (compositional naming —
    # daydose + physiologic state at shift change).
    _BASELINE_ADJ = ("age + _nth_day + C(sex_category) + C(icu_type) + "
                     "cci_score + weight_kg")
    _PHYSIO_TERMS = (" + ph_level_7am + ph_level_7pm + pf_level_7am + "
                     "pf_level_7pm + nee_7am + nee_7pm")
    BASELINE = ("{{outcome}} ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + "
                "midazeq_dif_mg_hr + " + _BASELINE_ADJ)
    # DAYDOSE: BASELINE + the 3 daytime continuous rates. Single linear
    # coefficient per drug — absorbs BOTH the selection-vs-not-selected
    # contrast AND the dose-given-exposed contrast in one term. Pre-H17
    # form; promoted to default 2026-05-05.
    DAYDOSE = BASELINE + (
        " + _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr"
    )
    SOFA = DAYDOSE + " + sofa_total"
    # DAYDOSE_PHYSIO (was: clinical, renamed 2026-05-11): DAYDOSE + the 3
    # organ-system status markers measured at shift change (pH = acid-base,
    # PF ratio = oxygenation, NEE = vasopressor support). "physio" captures
    # what the spec adds; the daydose_ prefix flags the build-on relationship.
    DAYDOSE_PHYSIO = DAYDOSE + _PHYSIO_TERMS

    # RCS (restricted cubic splines) variants of the manuscript linear specs.
    # Apply cr() to the night–day diff predictors (and optionally the daytime
    # continuous rates in the *_full flavor). 2026-05-07: TWO RCS spec
    # families maintained side by side so the cross-site agg figure can
    # render 3 versions (v1 main / v2 SA clusterse / v3 SA asymse — see
    # code/agg/marginal_effects_rcs_cross_site.py).
    # `*_rcs_full`: cr() on all 6 exposure vars (more flexible).
    # `*_rcs_diff`: cr() on the 3 diff vars only; daytime stays linear
    #               (parsimony — daytime curves are empirically near-linear).
    #
    # KNOT STRATEGY (2026-05-06): hard-coded clinical values, identical
    # across sites. Identical knots → identical cr basis function across
    # sites → cross-site marginal-effect curves overlay on the same x-grid.
    # Coefficients aren't human-interpretable as OR-per-unit for the cr
    # terms; the forest summarizes via 10→90 percentile OR (and via the
    # local per-unit slope at x_ref=0 in models_coeffs.csv).
    _RCS_DIFF_VARS = [
        'prop_dif_mcg_kg_min', 'fenteq_dif_mcg_hr', 'midazeq_dif_mg_hr',
    ]
    _LINEAR_DAYTIME_VARS = [
        '_prop_day_mcg_kg_min', '_fenteq_day_mcg_hr', '_midazeq_day_mg_hr',
    ]
    _RCS_FULL_VARS = _RCS_DIFF_VARS + _LINEAR_DAYTIME_VARS
    RCS_KNOTS_RAW = {
        'prop_dif_mcg_kg_min':    [-10.0, 0.0, 10.0],
        'fenteq_dif_mcg_hr':      [-25.0, 0.0, 25.0],
        'midazeq_dif_mg_hr':      [-1.0,  0.0, 1.0],
        '_prop_day_mcg_kg_min':   [10.0, 20.0, 30.0],
        '_fenteq_day_mcg_hr':     [25.0, 50.0, 75.0],
        '_midazeq_day_mg_hr':     [1.0,  2.0,  3.0],
    }

    def _scaled_knots(v):
        raw = RCS_KNOTS_RAW[v]
        scale = VAR_DISPLAY.get(v, {}).get('scale', 1)
        return [round(k / scale, 6) for k in raw]

    _knots_by_var = {v: _scaled_knots(v) for v in _RCS_FULL_VARS}
    logger.info("RCS knots from clinical defaults (raw → scaled):")
    for _v in _RCS_FULL_VARS:
        logger.info(f"  {_v:<24s}: raw={RCS_KNOTS_RAW[_v]}  scaled={_knots_by_var[_v]}")

    # HURDLE_INDICATORS kept in scope: used by the forest plot's
    # PERCENTILE_REF builder for filtering daytime predictors to the
    # non-zero subset when computing percentiles. The 24h `_*_any` indicators
    # are NO LONGER added as model covariates (no spec includes them since
    # the 2026-05-11 trim removed `daydose_anydose`).
    HURDLE_INDICATORS = {
        '_prop_day_mcg_kg_min': '_prop_any',
        '_fenteq_day_mcg_hr':   '_fenteq_any',
        '_midazeq_day_mg_hr':   '_midazeq_any',
    }

    def _cr_term(v):
        return f"cr({v}, knots={_knots_by_var[v]})"

    _EXPOSURE_TERMS_DIFF = (
        " + ".join(_cr_term(v) for v in _RCS_DIFF_VARS)
        + " + " + " + ".join(_LINEAR_DAYTIME_VARS)
    )
    _EXPOSURE_TERMS_FULL = " + ".join(_cr_term(v) for v in _RCS_FULL_VARS)

    # RCS variants — full parallels of the linear `daydose` and
    # `daydose_physio` specs. Adjustment block matches BASELINE (weight_kg
    # included). No standalone _*_any, no :indicator interaction, no
    # sofa_total. Two flavors per base: _diff (parsimony) and _full (most
    # flexible).
    DAYDOSE_RCS_DIFF = (
        "{{outcome}} ~ " + _EXPOSURE_TERMS_DIFF + " + " + _BASELINE_ADJ
    )
    DAYDOSE_RCS_FULL = (
        "{{outcome}} ~ " + _EXPOSURE_TERMS_FULL + " + " + _BASELINE_ADJ
    )
    DAYDOSE_PHYSIO_RCS_DIFF = DAYDOSE_RCS_DIFF + _PHYSIO_TERMS
    DAYDOSE_PHYSIO_RCS_FULL = DAYDOSE_RCS_FULL + _PHYSIO_TERMS

    COVARIATE_SPECS = [
        {'label': 'baseline',                 'formula': BASELINE},
        {'label': 'daydose',                  'formula': DAYDOSE},
        {'label': 'sofa',                     'formula': SOFA},
        {'label': 'daydose_physio',           'formula': DAYDOSE_PHYSIO},
        {'label': 'daydose_rcs_diff',         'formula': DAYDOSE_RCS_DIFF},
        {'label': 'daydose_rcs_full',         'formula': DAYDOSE_RCS_FULL},
        {'label': 'daydose_physio_rcs_diff',  'formula': DAYDOSE_PHYSIO_RCS_DIFF},
        {'label': 'daydose_physio_rcs_full',  'formula': DAYDOSE_PHYSIO_RCS_FULL},
    ]

    # ── Dimension 2: outcome x model type ─────────────────────────────
    # 2026-05-11: 11 (outcome, model_type) configs × 8 specs = 88 fits/site.
    # Convention: SBT outcomes are GEE-only (cluster-robust logit removed
    # by user direction); extubation outcomes keep gee + logit + logit_asym.
    # `logit` = cluster-robust SE (methodologically right but degenerates
    # on cr() basis); `logit_asym` = ordinary asymptotic SE (anti-
    # conservative but produces usable CIs). Cross-site agg figure renders
    # v1/v3 from `logit_asym` and v2 from `logit`.
    MODEL_CONFIGS = [
        # — extubation (gee + logit + logit_asym) —
        {'outcome': 'success_extub_next_day',     'model_type': 'gee',        'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day',     'model_type': 'logit',      'fit_fn': _fit_logit},
        {'outcome': 'success_extub_next_day',     'model_type': 'logit_asym', 'fit_fn': _fit_logit_asym},
        {'outcome': 'success_extub_v2_next_day',  'model_type': 'gee',        'fit_fn': _fit_gee},
        {'outcome': 'success_extub_v2_next_day',  'model_type': 'logit',      'fit_fn': _fit_logit},
        {'outcome': 'success_extub_v2_next_day',  'model_type': 'logit_asym', 'fit_fn': _fit_logit_asym},
        # — SBT outcomes (gee only) —
        {'outcome': 'sbt_done_prefix_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_multiday_next_day', 'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_subira_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_abc_next_day',      'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_v2_next_day',       'model_type': 'gee',   'fit_fn': _fit_gee},
        # — RETIRED (paste back to revert) —
        # 2026-05-11: sbt_elig_next_day (eligibility, not delivery)
        # 2026-05-01: sbt_done_next_day, sbt_done_anyprior_next_day,
        #             sbt_done_imv6h_next_day, sbt_done_2min_next_day
    ]

    # Outcomes that are GEE-only siblings (skipped from marginal-effect plots
    # to keep the figure roster from exploding; the forest plot covers them).
    # sbt_done_multiday is the manuscript's primary SBT-delivered outcome
    # and renders marginal effects alongside the extub outcomes.
    SBT_VARIANT_OUTCOMES = {
        'sbt_done_prefix_next_day',
        'sbt_done_subira_next_day',
        'sbt_done_abc_next_day',
    }

    # ── Cross-product loop ─────────────────────────────────────────────
    # Key shape: (outcome, model_type) → {spec_label: result}
    # `fit_meta` parallels `fitted` and carries n_obs / n_events / n_clusters
    # captured on the same dropna'd row set the fit consumed. Required by
    # models_coeffs.csv for downstream inverse-variance meta-analysis.
    fitted = {}
    fit_meta = {}
    # Step 6: collect per-fit status into a structured table for end-of-loop
    # summary + qc/model_fit_summary.csv.
    fit_summary_rows = []
    for _config in MODEL_CONFIGS:
        _key = (_config['outcome'], _config['model_type'])
        fitted[_key] = {}
        fit_meta[_key] = {}
        # SKIP_V2 path: when enable_v2_outcomes=false, the *_v2_next_day
        # columns are all-zero placeholders from 03_outcomes.py. Fitting on a
        # constant outcome would produce singular-matrix errors or nonsense
        # coefficients; explicitly skip and record SKIPPED_V2 status so the
        # operator can confirm the flag took effect. Forest + marginal-effect
        # plots for v2 outcomes do not render in this mode.
        if (not ENABLE_V2_OUTCOMES) and '_v2_next_day' in _config['outcome']:
            for _spec in COVARIATE_SPECS:
                fit_summary_rows.append({
                    'outcome': _config['outcome'],
                    'method': _config['model_type'],
                    'spec': _spec['label'],
                    'status': 'SKIPPED_V2',
                    'n_obs': 0,
                    'n_events': 0,
                    'fail_reason': 'enable_v2_outcomes=false',
                })
            logger.info(
                f"  SKIPPED_V2: {_config['outcome']} / {_config['model_type']} "
                f"(all {len(COVARIATE_SPECS)} specs)"
            )
            continue
        for _spec in COVARIATE_SPECS:
            _formula = _spec['formula'].replace('{{outcome}}', _config['outcome'])
            # Replicate the dropna logic used by _fit_logit / _fit_logit_asym
            # so n_events / n_clusters are computed on the same row set.
            _names_in_formula = [c for c in _df_scaled.columns if c in _formula]
            if 'hospitalization_id' not in _names_in_formula:
                _names_in_formula.append('hospitalization_id')
            _d_for_count = _df_scaled.dropna(subset=_names_in_formula)

            # Pre-fit singularity check (Step 3) — LINEAR SPECS ONLY.
            # RCS specs use cr() basis which is INHERENTLY rank-deficient by
            # construction (basis columns span a continuous space with
            # built-in linear dependencies). statsmodels handles this via
            # iterative regularization; a naive np.linalg.matrix_rank check
            # would false-flag every RCS spec as singular even when the fit
            # converges fine. Restrict the check to linear specs where
            # rank-deficiency genuinely predicts fit failure. The truly-
            # singular failures observed at NU (per F11) were on the linear
            # `sofa` spec, not on RCS — so this scope is exactly right.
            # Rank check is O(n_cols^3) — trivial vs the iterative cost.
            _is_rcs = 'rcs' in _spec['label']
            _is_singular = False
            _rank = _ncols = 0
            if not _is_rcs:
                try:
                    import patsy
                    _y_dm, _x_dm = patsy.dmatrices(
                        _formula, _d_for_count, return_type='dataframe'
                    )
                    _x_arr = _x_dm.values
                    _rank = int(np.linalg.matrix_rank(_x_arr, tol=1e-10))
                    _ncols = int(_x_arr.shape[1])
                    _is_singular = _rank < _ncols
                except Exception:
                    # If we can't even build the design matrix, let the
                    # fit_fn handle the failure path; don't pre-judge.
                    _is_singular = False

            if _is_singular:
                logger.warning(
                    f"  SKIP_SINGULAR: {_spec['label']} / {_config['outcome']} / "
                    f"{_config['model_type']} — design matrix rank "
                    f"{_rank}/{_ncols} (deficient)"
                )
                fit_summary_rows.append({
                    'outcome': _config['outcome'],
                    'method': _config['model_type'],
                    'spec': _spec['label'],
                    'status': 'SKIP_SINGULAR',
                    'n_obs': len(_d_for_count),
                    'n_events': int(_d_for_count[_config['outcome']].sum()),
                    'fail_reason': f'design matrix rank {_rank}/{_ncols}',
                })
                continue

            try:
                _result = _config['fit_fn'](_formula, _df_scaled)
                fitted[_key][_spec['label']] = _result
                fit_meta[_key][_spec['label']] = {
                    'n_obs': int(_result.nobs),
                    'n_events': int(_d_for_count[_config['outcome']].sum()),
                    'n_clusters': int(_d_for_count['hospitalization_id'].nunique()),
                }
                _converged = _is_converged(_result)
                _status = 'OK' if _converged else 'NO_CONVERGE'
                _tag = 'OK' if _converged else 'NO_CONVERGE'
                _log = logger.info if _converged else logger.warning
                _log(
                    f"  {_tag}: {_spec['label']} / {_config['outcome']} / "
                    f"{_config['model_type']} (N={int(_result.nobs)})"
                )
                fit_summary_rows.append({
                    'outcome': _config['outcome'],
                    'method': _config['model_type'],
                    'spec': _spec['label'],
                    'status': _status,
                    'n_obs': int(_result.nobs),
                    'n_events': int(_d_for_count[_config['outcome']].sum()),
                    'fail_reason': '' if _converged else 'maxiter reached without convergence',
                })
            except Exception as e:
                logger.info(f"  FAIL: {_spec['label']} / {_config['outcome']} / {_config['model_type']}: {e}")
                fit_summary_rows.append({
                    'outcome': _config['outcome'],
                    'method': _config['model_type'],
                    'spec': _spec['label'],
                    'status': 'FAIL',
                    'n_obs': len(_d_for_count),
                    'n_events': int(_d_for_count[_config['outcome']].sum()),
                    'fail_reason': str(e)[:200],  # truncate long tracebacks
                })

    # Aggregated summary block — per-method OK/FAIL counts so the operator
    # can read primary-vs-sensitivity status at a glance without grepping
    # individual FAIL lines. Primary: SBT outcomes = GEE; success_extub = logit.
    _summary_df = pd.DataFrame(fit_summary_rows)
    logger.info("=" * 70)
    logger.info("Model fit summary (per outcome × method):")
    _outcome_family = lambda o: (
        'SBT' if 'sbt_done' in o else 'success_extub' if 'success_extub' in o else 'other'
    )
    for (_method, _fam), _grp in _summary_df.assign(
        _fam=_summary_df['outcome'].map(_outcome_family)
    ).groupby(['method', '_fam']):
        _n_total = len(_grp)
        _n_ok = (_grp['status'] == 'OK').sum()
        _n_nc = (_grp['status'] == 'NO_CONVERGE').sum()
        _n_fail = (_grp['status'] == 'FAIL').sum()
        _is_primary = (
            (_fam == 'SBT' and _method == 'gee') or
            (_fam == 'success_extub' and _method == 'logit')
        )
        _tag = "PRIMARY" if _is_primary else "sensitivity"
        if _n_fail == 0 and _n_nc == 0:
            logger.info(
                f"  {_method:11s} ({_fam:13s}): {_n_ok}/{_n_total} OK   [{_tag}]"
            )
        else:
            _bits = []
            if _n_fail:
                _failed_specs = _grp.loc[_grp['status'] == 'FAIL', 'spec'].tolist()
                _bits.append(f"FAIL on: {', '.join(_failed_specs)}")
            if _n_nc:
                _nc_specs = _grp.loc[_grp['status'] == 'NO_CONVERGE', 'spec'].tolist()
                _bits.append(f"NO_CONVERGE on: {', '.join(_nc_specs)}")
            logger.info(
                f"  {_method:11s} ({_fam:13s}): {_n_ok}/{_n_total} OK   "
                f"[{_tag}] — {'; '.join(_bits)}"
            )
    logger.info("=" * 70)

    # fit_summary_rows threads through to the models_coeffs.csv builder cell,
    # where its per-fit status stamps every coefficient row and a sentinel
    # row is appended for any FAIL fit (no coefficients otherwise).
    return HURDLE_INDICATORS, MODEL_CONFIGS, SBT_VARIANT_OUTCOMES, fit_meta, fit_summary_rows, fitted, np, re


@app.cell
def _(MODEL_CONFIGS, SITE_NAME, VAR_DISPLAY, fitted, np, pd, re):
    # 2026-05-11: `sensitivity_analysis.csv` (long-format dump of every
    # coefficient) was retired here. Its content moved to the richer
    # `models_coeffs.csv` (built in the forest cell below) which adds
    # per-unit and per-percentile contrasts, n_obs / n_events / n_clusters,
    # and a `row_type` discriminator. Wide `model_comparison_*.csv` tables
    # (human-readable spec comparison, OR with stars) stay here unchanged.

    # ── Wide comparison tables per (outcome, model_type, param) ───────
    def _pretty_label(varname):
        """Map statsmodels parameter name -> human-readable label."""
        if varname in VAR_DISPLAY:
            return VAR_DISPLAY[varname]['label']
        # Match both C(col)[T.level] and col[T.level] (patsy auto-categoricals)
        m = re.match(r'(?:C\()?(\w+)(?:\))?\[T\.(.+)\]$', varname)
        if m:
            col, level = m.groups()
            col_pretty = col.replace('_category', '').replace('_', ' ').title()
            return f"{col_pretty}: {level}"
        return varname

    def build_wide_table(results_dict):
        """results_dict: {spec_label: fitted_result} -> wide DataFrame with OR (95% CI)."""
        all_vars = []
        for r in results_dict.values():
            for v in r.params.index:
                if v not in all_vars:
                    all_vars.append(v)
        _tbl = pd.DataFrame(index=all_vars + ['N'], columns=list(results_dict.keys()))
        for _label, r in results_dict.items():
            for _var in all_vars:
                if _var in r.params.index:
                    _coef = r.params[_var]
                    _ci = r.conf_int().loc[_var]
                    _ci_lo, _ci_hi = _ci.iloc[0], _ci.iloc[1]
                    _or_, _or_lo, _or_hi = np.exp(_coef), np.exp(_ci_lo), np.exp(_ci_hi)
                    _p = r.pvalues[_var]
                    _star = '***' if _p < 0.001 else '**' if _p < 0.01 else '*' if _p < 0.05 else ''
                    _tbl.loc[_var, _label] = f"{_or_:.2f} ({_or_lo:.2f}, {_or_hi:.2f}){_star}"
                else:
                    _tbl.loc[_var, _label] = '—'
            _tbl.loc['N', _label] = f"{int(r.nobs):,}"
        _tbl = _tbl.rename(index={v: _pretty_label(v) for v in all_vars})
        _tbl.index.name = 'Variable'
        return _tbl

    # Filename convention: use full outcome names with the `_next_day`
    # suffix stripped. Retired outcomes preserved as commented-out lines.
    OUTCOME_SHORT = {
        'success_extub_next_day':     'success_extub',
        'success_extub_v2_next_day':  'success_extub_v2',
        'sbt_done_prefix_next_day':   'sbt_done_prefix',
        'sbt_done_multiday_next_day': 'sbt_done_multiday',
        'sbt_done_subira_next_day':   'sbt_done_subira',
        'sbt_done_abc_next_day':      'sbt_done_abc',
        'sbt_done_v2_next_day':       'sbt_done_v2',
        # — RETIRED (paste back to restore) —
        # 2026-05-11: 'sbt_elig_next_day':       'sbt_elig'
        # 2026-05-01: 'sbt_done_next_day':       'sbt_done',
        #             'sbt_done_anyprior_next_day': 'sbt_done_anyprior',
        #             'sbt_done_imv6h_next_day':    'sbt_done_imv6h',
        #             'sbt_done_2min_next_day':     'sbt_done_2min',
    }
    for _config in MODEL_CONFIGS:
        _key = (_config['outcome'], _config['model_type'])
        if _key not in fitted or not fitted[_key]:
            continue
        _outcome_short = OUTCOME_SHORT.get(_config['outcome'], _config['outcome'])
        _fname = f"output_to_share/{SITE_NAME}/models/model_comparison_{_outcome_short}_{_config['model_type']}.csv"
        # Skip any *_rcs spec: cr() basis coefficients aren't human-interpretable
        # as OR-per-unit in the wide table. RCS results are summarized in
        # models_coeffs.csv (per-unit and per-percentile) and rendered as
        # marginal-effect curves.
        _results_for_table = {
            _k: _v for _k, _v in fitted[_key].items()
            if 'rcs' not in _k
        }
        _wide = build_wide_table(_results_for_table)
        _wide.to_csv(_fname)
        logger.info(f"Saved {_fname} ({len(_wide)} rows x {_wide.shape[1]} cols)")
    return (OUTCOME_SHORT,)


# ── Marginal Effect Plots ─────────────────────────────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Marginal Effect Plots

    2×3 figure per (outcome × model_type) showing predicted probability of the
    outcome as each exposure varies over its 2.5–97.5 percentile range, with
    all other covariates held at median (continuous) or mode (categorical).
    Produced from the `sofa` spec (closest to the example paper's adjustment
    set). Uses `result.get_prediction(new_data)` which handles link-scale
    CI transformation internally.

    **Rows**: [daytime rate, day-to-night Δ rate]
    **Cols**: [propofol, fentanyl eq, midazolam eq]
    **Style**: ggplot-like (gray panel bg, white grid, black line, gray CI ribbon)
    """)
    return


@app.cell
def _(HURDLE_INDICATORS, OUTCOME_SHORT, SBT_VARIANT_OUTCOMES, SITE_NAME, VAR_DISPLAY, cohort_merged_final, fitted, np, pd):
    # `np` is inherited from the fitting cell (returned above) to avoid the
    # marimo "multiple definitions" error. `_plt` is private (underscore
    # prefix) so it doesn't collide with any future cell that imports plt.
    import matplotlib.pyplot as _plt

    # Focal variables + human-friendly x-axis labels (actual hourly-rate units).
    # The 2×3 panel layout: row = {daytime rate, day-to-night Δ rate}, col = {prop, fenteq, midazeq}.
    # IMPORTANT: dose columns in cohort_merged_final are already in mg/hr or mcg/hr
    # (05_analytical_dataset.py divides shift totals by 12), so no extra conversion
    # is needed before computing percentiles or constructing the grid.
    FOCAL_VARS = [
        [('_prop_day_mcg_kg_min',  'Mean Daytime Propofol Rate (mcg/kg/min)'),
         ('_fenteq_day_mcg_hr',    'Mean Daytime Fentanyl Eq Rate (mcg/hr)'),
         ('_midazeq_day_mg_hr',    'Mean Daytime Midazolam Eq Rate (mg/hr)')],
        [('prop_dif_mcg_kg_min',   'Day-to-Night Δ Propofol Rate (mcg/kg/min)'),
         ('fenteq_dif_mcg_hr',     'Day-to-Night Δ Fentanyl Eq Rate (mcg/hr)'),
         ('midazeq_dif_mg_hr',     'Day-to-Night Δ Midazolam Eq Rate (mg/hr)')],
    ]

    Y_LABEL = {
        'success_extub_next_day': 'Probability of Successful Extubation',
        'success_extub_v2_next_day': 'Probability of Successful Extubation (v2)',
        'sbt_done_multiday_next_day': 'Probability of Passing SBT (multiday)',
        'sbt_done_v2_next_day': 'Probability of Passing SBT (v2)',
    }

    def _build_reference_row(df_scaled):
        """Median for numeric columns, mode for object/category columns.

        Pandas 2.3 rejects 'str' as a select_dtypes argument (TypeError
        'numpy string dtypes are not allowed, use \\'str\\' or \\'object\\''
        — paradoxically). Use 'object' alone, plus 'category' for any
        clinical-level groupings. Categoricals fall through both buckets
        so .mode() handles them without explicit branching.
        """
        ref = df_scaled.median(numeric_only=True).to_dict()
        for col in df_scaled.select_dtypes(include=['object', 'category']).columns:
            ref[col] = df_scaled[col].mode().iloc[0]
        return ref

    def _marginal_prediction(result, ref_row, focal_col, scaled_grid):
        """Run result.get_prediction for a grid holding everything else constant.

        For daytime continuous-rate predictors paired with a hurdle indicator
        (kept for forward compatibility with hypothetical specs that
        reintroduce a `cr(rate):indicator` interaction; the current
        manuscript RCS specs `daydose_rcs_*`/`daydose_physio_rcs_*` use
        plain `cr(rate, knots=...)` so the indicator force is a no-op for
        them), force the indicator to 1 across the entire grid so the
        basis-times-indicator term evaluates the dose-response shape.
        Otherwise the indicator stays at REF_ROW's median (typically 0),
        the basis term is zero everywhere on the grid, and the curve
        renders flat with a full-y-axis CI ribbon.

        Returns (prob, ci_lo, ci_hi) all in probability space. statsmodels has
        TWO different `summary_frame()` column schemes depending on the model
        family — verified at statsmodels/base/_prediction_inference.py lines
        118 and 326:

        - `PredictionResultsMean` (GEE, GLM, OLS):
          columns = ['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper']
        - `PredictionResultsBase` / `PredictionResultsMonotonic` (Logit, discrete):
          columns = ['predicted', 'se', 'ci_lower', 'ci_upper']

        We detect which scheme is present and use the appropriate keys.
        """
        new_data = pd.DataFrame([ref_row] * len(scaled_grid))
        new_data[focal_col] = scaled_grid
        _ind = HURDLE_INDICATORS.get(focal_col)
        if _ind is not None:
            new_data[_ind] = 1
        pred = result.get_prediction(new_data)
        sf = pred.summary_frame()
        if 'mean' in sf.columns:
            # GEE / GLM path
            prob = np.asarray(sf['mean'])
            ci_lo = np.asarray(sf['mean_ci_lower'])
            ci_hi = np.asarray(sf['mean_ci_upper'])
        elif 'predicted' in sf.columns:
            # Logit / discrete path
            prob = np.asarray(sf['predicted'])
            ci_lo = np.asarray(sf['ci_lower'])
            ci_hi = np.asarray(sf['ci_upper'])
        else:
            raise ValueError(
                f"Unexpected summary_frame columns: {list(sf.columns)}. "
                "Expected either ['mean', 'mean_ci_lower', 'mean_ci_upper'] "
                "or ['predicted', 'ci_lower', 'ci_upper']."
            )
        return prob, ci_lo, ci_hi

    def _ggplot_ax(ax):
        """Style an axes to look like the ggplot default (gray bg, white grid)."""
        ax.set_facecolor('#ebebeb')
        ax.grid(color='white', linewidth=0.8, which='major', zorder=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(colors='#4d4d4d', labelsize=8)
        ax.xaxis.label.set_color('#4d4d4d')
        ax.yaxis.label.set_color('#4d4d4d')

    def plot_marginal_effects(result, outcome, model_type, spec_label):
        """Build and save a 2×3 figure of marginal-effect curves.

        Also returns a long-format DataFrame of the prediction grid (50 points
        per panel × 6 panels = 300 rows) so the caller can stack across all
        (outcome × model_type × spec) runs and save a single
        marginal_effects_grid.csv for cross-site aggregation.
        """
        # Rebuild scaled dataset (must match training-space units for prediction)
        df_scaled = cohort_merged_final.copy()
        for col, info in VAR_DISPLAY.items():
            if col in df_scaled.columns and info['scale'] != 1:
                df_scaled[col] = df_scaled[col] / info['scale']
        ref_row = _build_reference_row(df_scaled)

        fig, axes = _plt.subplots(2, 3, figsize=(12, 7.5))
        fig.patch.set_facecolor('white')

        panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
        pidx = 0
        grid_rows = []

        for row_idx in range(2):
            for col_idx in range(3):
                focal, xlabel = FOCAL_VARS[row_idx][col_idx]
                ax = axes[row_idx, col_idx]
                _ggplot_ax(ax)

                # Observed percentile range in ACTUAL hourly-rate units
                raw = cohort_merged_final[focal].dropna()
                q_lo, q_hi = np.percentile(raw, [2.5, 97.5])
                actual_grid = np.linspace(q_lo, q_hi, 50)
                # Convert to scaled-training-space units for prediction
                scale = VAR_DISPLAY.get(focal, {}).get('scale', 1)
                scaled_grid = actual_grid / scale

                prob, ci_lo, ci_hi = _marginal_prediction(
                    result, ref_row, focal, scaled_grid
                )

                # Collect grid rows for cross-site aggregation. xlabel is
                # included so the agg layer can render axis labels without
                # re-deriving from the focal var name.
                for _xa, _xs, _p, _lo, _hi in zip(
                    actual_grid, scaled_grid, prob, ci_lo, ci_hi,
                ):
                    grid_rows.append({
                        'outcome': outcome,
                        'model_type': model_type,
                        'spec': spec_label,
                        'focal': focal,
                        'xlabel': xlabel,
                        'panel_row': row_idx,
                        'panel_col': col_idx,
                        'x_actual': float(_xa),
                        'x_scaled': float(_xs),
                        'prob': float(_p),
                        'ci_lo': float(_lo),
                        'ci_hi': float(_hi),
                    })

                ax.fill_between(
                    actual_grid, ci_lo, ci_hi,
                    color='#808080', alpha=0.35, zorder=2,
                )
                ax.plot(
                    actual_grid, prob,
                    color='black', linewidth=1.5, zorder=3,
                )

                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(
                    Y_LABEL.get(outcome, 'Predicted Probability'),
                    fontsize=9,
                )
                ax.set_ylim(0, 1)
                # Panel letter in the top-left corner
                ax.text(
                    -0.12, 1.08, panel_letters[pidx],
                    transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top', ha='left',
                )
                pidx += 1

        fig.suptitle(
            f"{Y_LABEL.get(outcome, 'Probability')} by Sedative Exposure\n"
            f"({spec_label} spec, {model_type.upper()})",
            fontsize=11, y=1.00,
        )
        fig.tight_layout()

        _outcome_short = OUTCOME_SHORT.get(outcome, outcome)
        out_path = (
            f"output_to_share/{SITE_NAME}/models/"
            f"marginal_effects_{_outcome_short}_{model_type}_{spec_label}.png"
        )
        fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
        _plt.close(fig)
        logger.info(f"Saved: {out_path}")
        return pd.DataFrame(grid_rows)

    # Generate one 2×3 figure per (outcome, model_type, spec).
    # PLOT_SPECS includes both RCS spec families (`*_rcs_diff` parsimony /
    # `*_rcs_full` flexible) for the 2 manuscript base specs. The cross-site
    # agg figure picks among these per version. SBT sensitivity siblings
    # (sbt_done_prefix / subira / abc) remain skipped from marginal-effect
    # plots; sbt_done_multiday is rendered (manuscript's primary SBT
    # outcome).
    PLOT_SPECS = ['daydose_rcs_diff', 'daydose_rcs_full',
                  'daydose_physio_rcs_diff', 'daydose_physio_rcs_full']
    _grid_frames = []
    for (_outcome, _mt), _spec_dict in fitted.items():
        if _outcome in SBT_VARIANT_OUTCOMES:
            continue
        for _spec_label in PLOT_SPECS:
            if _spec_label in _spec_dict:
                _grid_frames.append(
                    plot_marginal_effects(
                        _spec_dict[_spec_label], _outcome, _mt, _spec_label
                    )
                )
    if _grid_frames:
        _grid_df = pd.concat(_grid_frames, ignore_index=True)
        _grid_path = f"output_to_share/{SITE_NAME}/models/marginal_effects_grid.csv"
        _grid_df.to_csv(_grid_path, index=False)
        logger.info(f"Saved {_grid_path} ({len(_grid_df)} rows)")
    return


# ── models_coeffs.csv build + per-site forest PNG (QC) ────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## models_coeffs.csv + Per-Site Forest PNG

    Builds `models_coeffs.csv` (the federated payload for cross-site meta-
    analysis pooling) and renders one per-site forest PNG per (outcome ×
    model_type) for QC.

    **`models_coeffs.csv`** carries one row per (outcome × model_type ×
    spec × logical predictor) covering EVERY coefficient in EVERY fit.
    `row_type` ∈ {exposure, adjustment_continuous, adjustment_categorical,
    intercept} discriminates rows. For exposures (and continuous
    adjustments), the row carries BOTH per-unit and per-percentile
    log-OR + SE columns plus n_obs / n_events / n_clusters — everything
    a downstream DerSimonian-Laird pooler needs.

    All ORs are computed via the contrast-vector form
    `c = X_hi − X_lo` so the same code path handles linear and RCS
    (`cr()` basis) specs. For RCS specs, per-unit slopes are evaluated at
    a fixed reference point (`x_ref = 0`), which makes them cross-site
    comparable since the cr() basis uses shared knots.

    **Per-site forest PNG**: 6 rows (3 night-day diffs + 3 daytime
    continuous rates), one dot per spec per row, colored by spec. Uses
    the per-percentile (10→90) OR for QC display — clipped to log-scale
    [0.5, 2.0]. Cross-site pooled / per-unit forests live in `code/agg/`.

    **Output**: `output_to_share/{site}/models/models_coeffs.csv` +
    `output_to_share/{site}/models/forest_{outcome_short}_{model_type}.png`
    """)
    return


@app.cell
def _(HURDLE_INDICATORS, MODEL_CONFIGS, OUTCOME_SHORT, SITE_NAME, VAR_DISPLAY,
      cohort_merged_final, fit_meta, fit_summary_rows, fitted, np, pd, re):
    import matplotlib.pyplot as _plt
    from patsy import dmatrix as _dmatrix

    # 6 predictors visualized in the per-site forest PNG: 3 night-day diffs +
    # 3 daytime continuous rates. The 24h `_*_any` indicators were dropped
    # 2026-05-11 along with the `daydose_anydose` spec — no current spec
    # includes them, so they'd render as empty rows.
    FOREST_PREDICTORS = [
        ('prop_dif_mcg_kg_min',  'Δ propofol (mcg/kg/min)'),
        ('fenteq_dif_mcg_hr',    'Δ fentanyl eq (mcg/hr)'),
        ('midazeq_dif_mg_hr',    'Δ midazolam eq (mg/hr)'),
        ('_prop_day_mcg_kg_min', 'Daytime propofol (mcg/kg/min)'),
        ('_fenteq_day_mcg_hr',   'Daytime fentanyl eq (mcg/hr)'),
        ('_midazeq_day_mg_hr',   'Daytime midazolam eq (mg/hr)'),
    ]

    # UNIT_SPEC: per-exposure (unit_size, unit_label, x_ref_raw) for the
    # per-unit OR column in models_coeffs.csv. unit_size is in raw clinical
    # units; identical across sites by design so meta-analysis pools the
    # same shift everywhere.
    # x_ref_raw = 0 universally:
    #   - diffs: 0 = "no day-night difference" (the symmetric center)
    #   - daytime rates: 0 = "no exposure" (the natural anchor)
    # For LINEAR specs the per-unit OR is constant in x_ref. For RCS specs
    # it's a local slope at x_ref; choosing 0 makes contrasts cross-site
    # comparable since the cr() basis is built on shared knots.
    UNIT_SPEC = {
        'prop_dif_mcg_kg_min':   {'unit_size': 10, 'unit_label': 'mcg/kg/min', 'x_ref_raw': 0},
        'fenteq_dif_mcg_hr':     {'unit_size': 25, 'unit_label': 'mcg/hr',     'x_ref_raw': 0},
        'midazeq_dif_mg_hr':     {'unit_size': 1,  'unit_label': 'mg/hr',      'x_ref_raw': 0},
        '_prop_day_mcg_kg_min':  {'unit_size': 10, 'unit_label': 'mcg/kg/min', 'x_ref_raw': 0},
        '_fenteq_day_mcg_hr':    {'unit_size': 25, 'unit_label': 'mcg/hr',     'x_ref_raw': 0},
        '_midazeq_day_mg_hr':    {'unit_size': 1,  'unit_label': 'mg/hr',      'x_ref_raw': 0},
    }
    # ADJ_CONT_UNIT_SPEC: per-unit definitions for adjustment continuous
    # variables. Used only by models_coeffs.csv (one row per coefficient
    # in every fit). unit_size matches VAR_DISPLAY's presentational scale.
    ADJ_CONT_UNIT_SPEC = {
        'age':          {'unit_size': 1,   'unit_label': 'year'},
        '_nth_day':     {'unit_size': 1,   'unit_label': 'day'},
        'cci_score':    {'unit_size': 1,   'unit_label': 'point'},
        'sofa_total':   {'unit_size': 1,   'unit_label': 'point'},
        'weight_kg':    {'unit_size': 10,  'unit_label': 'kg'},
        'bmi':          {'unit_size': 5,   'unit_label': 'kg/m²'},
        'ph_level_7am': {'unit_size': 0.1, 'unit_label': 'pH'},
        'ph_level_7pm': {'unit_size': 0.1, 'unit_label': 'pH'},
        'pf_level_7am': {'unit_size': 50,  'unit_label': 'mmHg'},
        'pf_level_7pm': {'unit_size': 50,  'unit_label': 'mmHg'},
        'nee_7am':      {'unit_size': 0.1, 'unit_label': 'mcg/kg/min'},
        'nee_7pm':      {'unit_size': 0.1, 'unit_label': 'mcg/kg/min'},
    }

    SPEC_ORDER = [
        'baseline', 'daydose', 'sofa', 'daydose_physio',
        'daydose_rcs_diff', 'daydose_rcs_full',
        'daydose_physio_rcs_diff', 'daydose_physio_rcs_full',
    ]
    SPEC_COLORS = {
        'baseline':                 '#5e3c99',  # purple — adjustment-only
        'daydose':                  '#1f77b4',  # blue — manuscript linear spec 1
        'sofa':                     '#2ca02c',  # green — sofa-adjusted
        'daydose_physio':           '#ff7f0e',  # orange — manuscript linear spec 2
        'daydose_rcs_diff':         '#7570b3',  # violet — RCS diff (parsimony)
        'daydose_rcs_full':         '#3a3573',  # dark violet — RCS full
        'daydose_physio_rcs_diff':  '#d95f02',  # burnt orange — RCS diff (parsimony)
        'daydose_physio_rcs_full':  '#7a3700',  # dark burnt orange — RCS full
    }

    # ── Build PERCENTILE_REF: per-predictor (x10_raw, x90_raw, x10_scaled,
    # x90_scaled). Three filtering regimes per predictor type:
    #
    #   1. Daytime continuous-rate predictors (`_prop_day_mcg_kg_min` etc., the
    #      keys of HURDLE_INDICATORS): drop zeros — equivalent to "among the
    #      exposed". Selection effect is captured separately by the `_*_any`
    #      forest row.
    #
    #   2. Night–day diff predictors (`prop_dif_mcg_kg_min` etc.): drop rows
    #      where the patient had NO exposure at all in the past 24h, i.e.
    #      restrict to `_*_any == 1` rows. A diff of zero among
    #      no-drug-at-all patients is uninformative — those rows otherwise
    #      pile up at zero and collapse the percentiles (the original midaz
    #      problem at low-use sites). After this filter, diff = 0 still
    #      appears for patients on a constant rate (informative), but the
    #      "no-drug-trivial-zero" mass is removed.
    #
    #   3. Other predictors (e.g., the `_*_any` indicators themselves): use
    #      the full distribution.
    DIFF_HURDLE_INDICATORS = {
        'prop_dif_mcg_kg_min':  '_prop_any',
        'fenteq_dif_mcg_hr':    '_fenteq_any',
        'midazeq_dif_mg_hr':    '_midazeq_any',
    }
    PERCENTILE_REF = {}
    for _pred, _ in FOREST_PREDICTORS:
        _series = cohort_merged_final[_pred].dropna()
        if len(_series) == 0:
            continue
        if _pred in HURDLE_INDICATORS:
            _vals = _series.to_numpy()
            _vals = _vals[_vals != 0]
            _subset = 'non-zero (daytime exposed)'
        elif _pred in DIFF_HURDLE_INDICATORS:
            _ind_col = DIFF_HURDLE_INDICATORS[_pred]
            _mask = (cohort_merged_final[_ind_col] == 1) & cohort_merged_final[_pred].notna()
            _vals = cohort_merged_final.loc[_mask, _pred].to_numpy()
            _subset = 'any-24h-exposure'
        else:
            _vals = _series.to_numpy()
            _subset = 'all'
        if len(_vals) == 0:
            continue
        _x10, _x90 = np.percentile(_vals, 10), np.percentile(_vals, 90)
        _scale = VAR_DISPLAY.get(_pred, {}).get('scale', 1)
        PERCENTILE_REF[_pred] = {
            'x10_raw': _x10,
            'x90_raw': _x90,
            'x10_scaled': _x10 / _scale,
            'x90_scaled': _x90 / _scale,
            'delta_scaled': (_x90 - _x10) / _scale,
            'subset': _subset,
        }
    logger.info("PERCENTILE_REF (raw clinical units, 10th and 90th percentiles):")
    for _pred, _info in PERCENTILE_REF.items():
        _tag = f"  [{_info['subset']}]" if _info['subset'] != 'all' else ''
        logger.info(f"  {_pred:<24s}: x10={_info['x10_raw']:>+8.3f}, x90={_info['x90_raw']:>+8.3f}{_tag}")

    # Reference row (median for numeric, mode for categorical) over the
    # SCALED dataset — same construction as the marginal-effects cell.
    def _build_reference_row_scaled(df_scaled):
        ref = df_scaled.median(numeric_only=True).to_dict()
        for col in df_scaled.select_dtypes(include=['object', 'category']).columns:
            ref[col] = df_scaled[col].mode().iloc[0]
        return ref

    _df_scaled_forest = cohort_merged_final.copy()
    for _col, _info in VAR_DISPLAY.items():
        if _col in _df_scaled_forest.columns and _info['scale'] != 1:
            _df_scaled_forest[_col] = _df_scaled_forest[_col] / _info['scale']
    REF_ROW = _build_reference_row_scaled(_df_scaled_forest)

    _NAN_CONTRAST = (np.nan, np.nan, np.nan, np.nan, np.nan)

    def _or_contrast(fit, predictor, x_lo_raw, x_hi_raw):
        """Return (log_or, se_log_or, or_, or_lo, or_hi) for a shift
        `predictor: x_lo_raw → x_hi_raw` in RAW clinical units, holding all
        other covariates at REF_ROW. Endpoints are converted to the fit's
        scaled-training-space units internally.

        Works for BOTH linear specs (design matrix changes only on the
        `predictor` column) and RCS specs (multiple cr() basis columns
        change together). The contrast vector c = X_hi − X_lo captures the
        spec automatically; var(log_OR) = c V c' is the delta method.
        """
        if x_lo_raw == x_hi_raw:
            return _NAN_CONTRAST
        scale = VAR_DISPLAY.get(predictor, {}).get('scale', 1)
        x_lo_scaled = x_lo_raw / scale
        x_hi_scaled = x_hi_raw / scale

        nd_lo = pd.DataFrame([REF_ROW])
        nd_hi = pd.DataFrame([REF_ROW])
        nd_lo[predictor] = x_lo_scaled
        nd_hi[predictor] = x_hi_scaled
        # If the predictor is paired with a hurdle indicator (kept here for
        # forward compatibility with hypothetical specs that reintroduce a
        # `cr(rate):indicator` interaction; the current manuscript RCS specs
        # use plain `cr(rate, knots=...)` so this branch is a no-op for them),
        # force indicator=1 at both endpoints so the basis-times-indicator
        # contrast captures the intensity-among-exposed effect.
        _ind = HURDLE_INDICATORS.get(predictor)
        if _ind is not None:
            nd_lo[_ind] = 1
            nd_hi[_ind] = 1

        # Re-evaluate the formula's design matrix on the new rows so cr()
        # basis columns are recomputed at the new predictor value.
        try:
            di = fit.model.data.design_info
            X_lo = np.asarray(_dmatrix(di, nd_lo, return_type='matrix'))[0]
            X_hi = np.asarray(_dmatrix(di, nd_hi, return_type='matrix'))[0]
        except Exception:
            return _NAN_CONTRAST

        beta = fit.params.values
        V = fit.cov_params().values
        contrast = X_hi - X_lo
        # If the predictor isn't in the spec's design matrix, the contrast
        # is all zeros → log_OR = 0 → OR = 1 with zero-width CI. Misleading
        # ("baseline shows null effect"); return NaN so consumers skip it.
        if np.allclose(contrast, 0):
            return _NAN_CONTRAST
        log_or = float(contrast @ beta)
        var_log_or = float(contrast @ V @ contrast)
        if var_log_or < 0:
            return _NAN_CONTRAST
        se = np.sqrt(var_log_or)
        return (
            log_or,
            se,
            float(np.exp(log_or)),
            float(np.exp(log_or - 1.96 * se)),
            float(np.exp(log_or + 1.96 * se)),
        )

    def _or_10_to_90(fit, predictor):
        """Backward-compat shim: 10→90 percentile OR via _or_contrast."""
        info = PERCENTILE_REF.get(predictor)
        if info is None:
            return (np.nan, np.nan, np.nan)
        _, _, or_, or_lo, or_hi = _or_contrast(
            fit, predictor, info['x10_raw'], info['x90_raw']
        )
        return (or_, or_lo, or_hi)

    # ── Build models_coeffs.csv (subsumes sensitivity_analysis.csv) ──
    # One row per (outcome, model_type, spec, logical_predictor) covering
    # EVERY coefficient in EVERY fit. row_type discriminates exposure /
    # adjustment_continuous / adjustment_categorical / intercept rows so
    # consumers can filter cleanly. Per-unit and per-percentile contrast
    # columns are populated where meaningful and NaN otherwise.
    EXPOSURE_SET = set(UNIT_SPEC.keys())
    ADJ_CONT_SET = set(ADJ_CONT_UNIT_SPEC.keys())

    def _strip_predictor(varname):
        """Map statsmodels coefficient name → (logical_name, role).

        role ∈ {'rcs_basis', 'continuous', 'categorical', 'intercept'}.
        For RCS basis columns like `cr(prop_dif_mcg_kg_min, knots=[...])[1]`
        the logical name is the inner variable; multiple basis columns
        collapse to a single logical row in models_coeffs.csv.
        """
        if varname == 'Intercept':
            return ('Intercept', 'intercept')
        m = re.match(r'cr\(([^,]+),', varname)
        if m:
            return (m.group(1).strip(), 'rcs_basis')
        m = re.match(r'(?:C\()?(\w+)(?:\))?\[T\.(.+)\]$', varname)
        if m:
            col, level = m.groups()
            return (f"{col}[T.{level}]", 'categorical')
        return (varname, 'continuous')

    def _classify(logical):
        if logical == 'Intercept':
            return 'intercept'
        if logical in EXPOSURE_SET:
            return 'exposure'
        if logical in ADJ_CONT_SET:
            return 'adjustment_continuous'
        if '[T.' in logical:
            return 'adjustment_categorical'
        return 'adjustment_continuous'

    NAN = float('nan')
    # Lookup (outcome, method, spec) → (status, fail_reason) so the per-fit
    # convergence flag rides every coefficient row of that fit. Built once.
    _fit_status_map = {
        (r['outcome'], r['method'], r['spec']): (r['status'], r['fail_reason'])
        for r in fit_summary_rows
    }
    coeffs_rows = []
    for (_outcome, _mt), _spec_dict in fitted.items():
        for _spec_label, _result in _spec_dict.items():
            _meta = fit_meta.get((_outcome, _mt), {}).get(_spec_label, {})
            _spec_family = 'rcs' if 'rcs' in _spec_label else 'linear'
            _status, _reason = _fit_status_map.get(
                (_outcome, _mt, _spec_label), ('OK', '')
            )
            _seen = set()
            for _vn in _result.params.index:
                _logical, _role = _strip_predictor(_vn)
                if _logical in _seen:
                    continue
                _seen.add(_logical)
                _row_type = _classify(_logical)

                # Defaults (NaN / blank) — overridden per row_type below.
                _unit_size = NAN
                _unit_label = ''
                _x_ref_raw = NAN
                _log_or_pu = NAN; _se_pu = NAN
                _or_pu = NAN;     _or_pu_lo = NAN; _or_pu_hi = NAN
                _x10_raw = NAN;   _x90_raw = NAN
                _log_or_pp = NAN; _se_pp = NAN
                _or_pp = NAN;     _or_pp_lo = NAN; _or_pp_hi = NAN
                _log_or = NAN;    _se_log_or = NAN

                if _row_type == 'exposure':
                    _u = UNIT_SPEC[_logical]
                    _unit_size = _u['unit_size']
                    _unit_label = _u['unit_label']
                    _x_ref_raw = _u['x_ref_raw']
                    _log_or_pu, _se_pu, _or_pu, _or_pu_lo, _or_pu_hi = _or_contrast(
                        _result, _logical, _x_ref_raw, _x_ref_raw + _unit_size
                    )
                    _pinfo = PERCENTILE_REF.get(_logical)
                    if _pinfo is not None:
                        _x10_raw = _pinfo['x10_raw']
                        _x90_raw = _pinfo['x90_raw']
                        _log_or_pp, _se_pp, _or_pp, _or_pp_lo, _or_pp_hi = _or_contrast(
                            _result, _logical, _x10_raw, _x90_raw
                        )
                    # Headline log_or = per-unit (interpretable across linear + RCS)
                    _log_or = _log_or_pu
                    _se_log_or = _se_pu
                elif _row_type == 'adjustment_continuous':
                    _u = ADJ_CONT_UNIT_SPEC.get(_logical, {'unit_size': 1, 'unit_label': ''})
                    _unit_size = _u['unit_size']
                    _unit_label = _u['unit_label']
                    _x_ref_raw = 0
                    _log_or_pu, _se_pu, _or_pu, _or_pu_lo, _or_pu_hi = _or_contrast(
                        _result, _logical, 0, _unit_size
                    )
                    _log_or = _log_or_pu
                    _se_log_or = _se_pu
                elif _row_type in ('adjustment_categorical', 'intercept'):
                    _log_or = float(_result.params[_vn])
                    _se_log_or = float(_result.bse[_vn])

                coeffs_rows.append({
                    'outcome': _outcome,
                    'model_type': _mt,
                    'spec': _spec_label,
                    'spec_family': _spec_family,
                    'predictor': _logical,
                    'row_type': _row_type,
                    'log_or': _log_or,
                    'se_log_or': _se_log_or,
                    'unit_size': _unit_size,
                    'unit_label': _unit_label,
                    'x_ref_raw': _x_ref_raw,
                    'log_or_per_unit': _log_or_pu,
                    'se_per_unit': _se_pu,
                    'or_per_unit': _or_pu,
                    'or_per_unit_lo': _or_pu_lo,
                    'or_per_unit_hi': _or_pu_hi,
                    'x10_raw': _x10_raw,
                    'x90_raw': _x90_raw,
                    'log_or_p10_p90': _log_or_pp,
                    'se_p10_p90': _se_pp,
                    'or_p10_p90': _or_pp,
                    'or_p10_p90_lo': _or_pp_lo,
                    'or_p10_p90_hi': _or_pp_hi,
                    'n_obs': _meta.get('n_obs', 0),
                    'n_events': _meta.get('n_events', 0),
                    'n_clusters': _meta.get('n_clusters', 0),
                    'fit_status': _status,
                    'fail_reason': _reason,
                })
    # FAIL sentinels — fitted[] only has fits that produced coefficients, so
    # exception-failed fits would otherwise vanish from the master CSV. One
    # placeholder row per failed (outcome × method × spec) with coefficient
    # columns NaN; fit_status='FAIL' carries the signal.
    for _r in fit_summary_rows:
        if _r['status'] != 'FAIL':
            continue
        coeffs_rows.append({
            'outcome': _r['outcome'],
            'model_type': _r['method'],
            'spec': _r['spec'],
            'spec_family': 'rcs' if 'rcs' in _r['spec'] else 'linear',
            'predictor': NAN,
            'row_type': NAN,
            'log_or': NAN, 'se_log_or': NAN,
            'unit_size': NAN, 'unit_label': '',
            'x_ref_raw': NAN,
            'log_or_per_unit': NAN, 'se_per_unit': NAN,
            'or_per_unit': NAN, 'or_per_unit_lo': NAN, 'or_per_unit_hi': NAN,
            'x10_raw': NAN, 'x90_raw': NAN,
            'log_or_p10_p90': NAN, 'se_p10_p90': NAN,
            'or_p10_p90': NAN, 'or_p10_p90_lo': NAN, 'or_p10_p90_hi': NAN,
            'n_obs': _r.get('n_obs', 0),
            'n_events': _r.get('n_events', 0),
            'n_clusters': 0,
            'fit_status': 'FAIL',
            'fail_reason': _r['fail_reason'],
        })
    coeffs_df = pd.DataFrame(coeffs_rows)
    _coeffs_path = f'output_to_share/{SITE_NAME}/models/models_coeffs.csv'
    coeffs_df.to_csv(_coeffs_path, index=False)
    logger.info(f"Saved {_coeffs_path} ({len(coeffs_df)} rows)")

    # ── Render one per-site forest PNG per (outcome, model_type) ─────
    # Uses or_p10_p90 columns (10→90 percentile shift) for QC view.
    # Cross-site pooled / per-unit views live in code/agg/.
    def plot_forest(coeffs, outcome, model_type, site, predictors, percentile_ref, out_path):
        fig, ax = _plt.subplots(figsize=(9.5, 5.5))
        n_specs = len(SPEC_ORDER)
        jitter = np.linspace(-0.20, 0.20, n_specs)

        ymin, ymax = -0.6, len(predictors) - 0.4
        for i, (pred, pred_label) in enumerate(predictors):
            y_base = len(predictors) - 1 - i
            for j, spec in enumerate(SPEC_ORDER):
                _row = coeffs[
                    (coeffs['outcome'] == outcome)
                    & (coeffs['model_type'] == model_type)
                    & (coeffs['spec'] == spec)
                    & (coeffs['predictor'] == pred)
                    & (coeffs['row_type'] == 'exposure')
                ]
                if _row.empty:
                    continue
                _r = _row.iloc[0]
                _or = _r['or_p10_p90']
                _lo = _r['or_p10_p90_lo']
                _hi = _r['or_p10_p90_hi']
                if not (np.isfinite(_or) and np.isfinite(_lo) and np.isfinite(_hi)):
                    continue
                _y = y_base + jitter[j]
                ax.errorbar(
                    _or, _y,
                    xerr=[[_or - _lo], [_hi - _or]],
                    fmt='o', color=SPEC_COLORS[spec], markersize=4,
                    capsize=2, elinewidth=1.0, label=spec if i == 0 else None,
                )

        ytick_labels = []
        ytick_pos = []
        for i, (pred, pred_label) in enumerate(predictors):
            y_base = len(predictors) - 1 - i
            info = percentile_ref.get(pred, {})
            x10 = info.get('x10_raw', np.nan)
            x90 = info.get('x90_raw', np.nan)
            ytick_labels.append(f"{pred_label}\nx10={x10:+.2f}, x90={x90:+.2f}")
            ytick_pos.append(y_base)
        ax.set_yticks(ytick_pos)
        ax.set_yticklabels(ytick_labels, fontsize=8)
        ax.set_ylim(ymin, ymax)

        ax.set_xscale('log')
        ax.set_xlim(0.5, 2.0)
        ax.axvline(1.0, color='dimgray', linewidth=0.8, linestyle='--', zorder=0)
        ax.set_xlabel('Odds ratio (10th → 90th percentile shift, log scale, clipped to [0.5, 2.0])', fontsize=9)
        ax.set_title(f"{outcome} — {site} ({model_type.upper()})", fontsize=11)
        ax.grid(True, axis='x', linewidth=0.4, alpha=0.4, zorder=0)

        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
        if dedup:
            ax.legend(
                [h for h, _ in dedup], [l for _, l in dedup],
                loc='upper center', bbox_to_anchor=(0.5, 1.10),
                ncol=len(dedup), fontsize=8, frameon=False,
            )

        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
        _plt.close(fig)

    for _config in MODEL_CONFIGS:
        _outcome = _config['outcome']
        _mt = _config['model_type']
        _outcome_short = OUTCOME_SHORT.get(_outcome, _outcome)
        _out_path = (
            f"output_to_share/{SITE_NAME}/models/"
            f"forest_{_outcome_short}_{_mt}.png"
        )
        plot_forest(
            coeffs_df, _outcome, _mt, SITE_NAME,
            FOREST_PREDICTORS, PERCENTILE_REF, _out_path,
        )
        logger.info(f"Saved: {_out_path}")
    return (FOREST_PREDICTORS, PERCENTILE_REF, coeffs_df)


if __name__ == "__main__":
    app.run()
