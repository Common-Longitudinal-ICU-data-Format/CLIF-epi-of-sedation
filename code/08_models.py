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
    import pandas as pd

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    # Site-scoped output dirs (see Makefile SITE= flag).
    # Path B++ refactor: every modeling artifact lands under {site}/models/.
    os.makedirs(f"output_to_share/{SITE_NAME}/models", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return SITE_NAME, pd


@app.cell
def _(SITE_NAME, pd):
    cohort_merged_final = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")
    print(f"Analytical dataset: {len(cohort_merged_final)} rows")
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
    reporting (sbt_elig, sbt_done_subira, sbt_done_abc, sbt_done_v2,
    sbt_done_prefix) are fitted by the cross-product loop below and
    appear in `model_comparison_*.csv` + `forest_*.png` artifacts.

    Original code preserved in git history; see
    `~/.claude/plans/2-related-tasks-around-memoized-hartmanis.md`
    for the retire decision context.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logistic Regression: Successful Extubation Next Day
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    success_extub_formula = """success_extub_next_day ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr +
    _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    # Drop NaN rows before fitting so the cluster-robust SE's `groups` argument
    # has the same length as the model's residuals (statsmodels' Logit with
    # cov_type='cluster' doesn't auto-align them when missing='drop' is used).
    _logit_cols = [
        'prop_dif_mcg_kg_min', 'fenteq_dif_mcg_hr', 'midazeq_dif_mg_hr',
        '_prop_day_mcg_kg_min', '_midazeq_day_mg_hr', '_fenteq_day_mcg_hr',
        'ph_level_7am', 'ph_level_7pm', 'pf_level_7am', 'pf_level_7pm',
        'nee_7am', 'nee_7pm', 'age', '_nth_day', 'sofa_total', 'cci_score',
        'sex_category', 'icu_type', 'success_extub_next_day',
        'hospitalization_id',
    ]
    _logit_df = cohort_merged_final.dropna(subset=_logit_cols)
    logit_model = _smf.logit(formula=success_extub_formula, data=_logit_df)
    logit_result = logit_model.fit(cov_type='cluster', cov_kwds={'groups': _logit_df['hospitalization_id']})
    print(logit_result.summary())
    _summary_df = logit_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output_to_share/{SITE_NAME}/models/logit_summary.csv', index=False)
    _cov_matrix = logit_result.cov_params()
    _cov_matrix.to_csv(f'output_to_share/{SITE_NAME}/models/logit_covmat.csv')
    print(f"Saved output_to_share/{SITE_NAME}/models/logit_summary.csv, logit_covmat.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nested Model Comparison

    Fits 4 nested models (progressively adding adjustment variables) for each
    (outcome, model_type) combination.

    **Model specs (all include exposures: prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr):**

    1. **baseline**: age + sex + ICU type + CCI

    2. **daydose**: baseline + daytime absolute dose rates (`_prop_day_mcg_kg_min`, `_fenteq_day_mcg_hr`, `_midazeq_day_mg_hr`)

    3. **sofa**: daydose + `sofa_total`

    4. **clinical**: daydose + pH/P/F levels + NEE at 7am/7pm

    `sofa` and `clinical` are siblings (both extend `daydose`).

    **Outputs:**

    - Long CSV (`sensitivity_analysis.csv`) — one row per coefficient

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
        # Exposures (day-night rate differences)
        'prop_dif_mcg_kg_min': {'scale': 10,  'label': 'Δ propofol (per 10 mcg/kg/min)'},
        'fenteq_dif_mcg_hr':   {'scale': 10,  'label': 'Δ fentanyl eq (per 10 mcg/hr)'},
        'midazeq_dif_mg_hr':   {'scale': 0.1, 'label': 'Δ midazolam eq (per 0.1 mg/hr)'},
        # Daytime absolute rates
        '_prop_day_mcg_kg_min': {'scale': 10,  'label': 'Daytime propofol (per 10 mcg/kg/min)'},
        '_fenteq_day_mcg_hr':   {'scale': 10,  'label': 'Daytime fentanyl eq (per 10 mcg/hr)'},
        '_midazeq_day_mg_hr':   {'scale': 0.1, 'label': 'Daytime midazolam eq (per 0.1 mg/hr)'},
        # Hurdle binaries: any daytime exposure (yes/no). scale=1 keeps the
        # column 0/1 — the regression OR (= 10→90 OR when ≥10% of rows are 1)
        # is interpretable as "odds ratio for any exposure vs none."
        '_prop_day_any':   {'scale': 1, 'label': 'Any daytime propofol (yes/no)'},
        '_fenteq_day_any': {'scale': 1, 'label': 'Any daytime fentanyl eq (yes/no)'},
        '_midazeq_day_any': {'scale': 1, 'label': 'Any daytime midazolam eq (yes/no)'},
        # Other continuous covariates
        'age':          {'scale': 1,   'label': 'Age (per year)'},
        '_nth_day':     {'scale': 1,   'label': 'Day on IMV (per day)'},
        'cci_score':    {'scale': 1,   'label': 'Charlson CCI (per point)'},
        'sofa_total':   {'scale': 1,   'label': 'SOFA total (per point)'},
        'nee_7am':      {'scale': 0.1, 'label': 'NEE 7am (per 0.1 mcg/kg/min)'},
        'nee_7pm':      {'scale': 0.1, 'label': 'NEE 7pm (per 0.1 mcg/kg/min)'},
        # Body habitus (sensitivity-spec-only covariates; only appear in the
        # sofa_weight / sofa_bmi rows of the wide CSV, not in forest plots).
        'weight_kg':    {'scale': 10,  'label': 'Weight (per 10 kg)'},
        'bmi':          {'scale': 5,   'label': 'BMI (per 5 kg/m²)'},
    }
    return (VAR_DISPLAY,)


@app.cell
def _(VAR_DISPLAY, cohort_merged_final, pd):
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
    # maxiter=100 raised from statsmodels defaults (Logit=35, GEE=60) so the
    # RCS-extub-logit fit (~37 coefficients) has headroom to converge instead
    # of stopping at the cap.
    def _fit_gee(formula, data):
        m = smf.gee(formula=formula, groups='hospitalization_id',
                    data=data, family=sm.families.Binomial())
        return m.fit(maxiter=100)

    def _fit_logit(formula, data):
        # Drop rows with NaN in any column referenced by the formula so the
        # cluster-robust SE's `groups` length matches the model's residuals.
        # statsmodels' Logit with cov_type='cluster' doesn't auto-align
        # groups when missing rows are dropped internally — the result is a
        # length mismatch in cov_cluster's bincount.
        _names = [c for c in data.columns if c in formula]
        if 'hospitalization_id' not in _names:
            _names.append('hospitalization_id')
        _d = data.dropna(subset=_names)
        m = smf.logit(formula=formula, data=_d)
        return m.fit(cov_type='cluster',
                     cov_kwds={'groups': _d['hospitalization_id']},
                     maxiter=100)

    # ── Dimension 1: nested covariate sets (all include exposures) ────
    # Rate-based exposures (mcg/kg/min for propofol, mcg/hr for fentanyl,
    # mg/hr for midazolam). Amount-based parameterization was retired in the
    # 2026-04-29 model-update round — rate-only going forward.
    BASELINE = ("{{outcome}} ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr + "
                "age + _nth_day + C(sex_category) + C(icu_type) + cci_score")
    # DAYDOSE adds two-part (hurdle) daytime exposure: a binary indicator for
    # ANY daytime drug + the continuous rate. The indicator absorbs the zero-
    # mass selection signal (clinician chose to keep the patient on the drug
    # at all); the continuous rate then reflects dose-response among exposed.
    # See `docs/uptitration_paradox_investigation.md` and plan H17.
    DAYDOSE = BASELINE + (" + _prop_day_any + _fenteq_day_any + _midazeq_day_any"
                          " + _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr")
    SOFA = DAYDOSE + " + sofa_total"
    CLINICAL = DAYDOSE + (" + ph_level_7am + ph_level_7pm + pf_level_7am + "
                          "pf_level_7pm + nee_7am + nee_7pm")

    # RCS (restricted cubic splines) variant of the sofa spec.
    # Wraps the 6 exposure variables in patsy's cr() transform for natural
    # cubic regression splines with 3 internal knots → 4 basis columns per variable.
    # Non-exposure adjustment variables stay linear/categorical.
    # Coefficients aren't human-interpretable as OR-per-unit, so the wide CSV
    # excludes them; the forest plot summarizes via 10→90 percentile OR.
    #
    # NOTE (2026-05-01): Knots are computed from the NON-ZERO subset's
    # quartiles instead of `df=4` (which uses whole-vector quartiles). Daytime
    # exposures are ~62% exact zero in the cohort; whole-vector p25 and p50
    # both collapse to 0, producing a degenerate cr() basis whose cluster-
    # robust covariance has negative eigenvalues for basis directions —
    # surfaced as empty OR rows in `forest_data.csv`. Non-zero quartiles
    # place knots within the active dose range; basis at x=0 becomes a
    # constant column absorbed by the intercept.
    _RCS_VARS = [
        'prop_dif_mcg_kg_min', 'fenteq_dif_mcg_hr', 'midazeq_dif_mg_hr',
        '_prop_day_mcg_kg_min', '_fenteq_day_mcg_hr', '_midazeq_day_mg_hr',
    ]

    def _nz_quartile_knots(s):
        nz = s[s != 0].dropna().to_numpy()
        if len(nz) < 100:
            return None
        return [round(float(k), 6) for k in np.percentile(nz, [25, 50, 75])]

    # Compute knots from `_df_scaled` (the dataset actually fed to fits) so the
    # values are in the same scaled units patsy sees at fit/predict time —
    # raw units would trip cr()'s upper-bound check (knots > observed max).
    _knots_by_var = {v: _nz_quartile_knots(_df_scaled[v]) for v in _RCS_VARS}
    print("RCS knots from non-zero quartiles (in scaled units, per VAR_DISPLAY['scale']):")
    for _v, _k in _knots_by_var.items():
        print(f"  {_v:<24s}: {_k if _k is not None else 'fallback df=4 (<100 nonzero)'}")

    # Map the 3 daytime continuous-rate predictors to their hurdle indicators.
    # The cr() basis on these gets interaction-multiplied by the indicator so the
    # basis evaluates to 0 on indicator=0 rows, removing the 62%-zero pile-up
    # from the basis design and stabilizing cluster-robust V. For diff terms,
    # there's no hurdle indicator (signed predictors), so cr() applies as-is.
    HURDLE_INDICATORS = {
        '_prop_day_mcg_kg_min': '_prop_day_any',
        '_fenteq_day_mcg_hr':   '_fenteq_day_any',
        '_midazeq_day_mg_hr':   '_midazeq_day_any',
    }

    def _cr_term(v):
        k = _knots_by_var[v]
        base = f"cr({v}, df=4)" if k is None else f"cr({v}, knots={k})"
        ind = HURDLE_INDICATORS.get(v)
        return f"{base}:{ind}" if ind else base

    SOFA_RCS = (
        "{{outcome}} ~ "
        + " + ".join(_cr_term(v) for v in _RCS_VARS) + " + "
        "_prop_day_any + _fenteq_day_any + _midazeq_day_any + "
        "age + _nth_day + C(sex_category) + C(icu_type) + cci_score + sofa_total"
    )

    # Body-habitus sensitivity siblings of SOFA. Hypothesis: propofol exposure
    # is already mcg/kg/min (weight-normalized at the source), but if body
    # habitus correlates with sedation practice (titration heuristics use
    # visual habitus more than literal weight) AND with extubation success
    # (BMI extremes both reduce success), the model needs explicit habitus
    # adjustment to remove residual confounding. BMI subcohort attrition at
    # MIMIC is large (height_cm sparsely charted) — _fit_logit's per-spec
    # dropna handles it; the wide CSV's N column will surface it.
    SOFA_WEIGHT = SOFA + " + weight_kg"
    SOFA_BMI    = SOFA + " + bmi"

    COVARIATE_SPECS = [
        {'label': 'baseline',    'formula': BASELINE},
        {'label': 'daydose',     'formula': DAYDOSE},
        {'label': 'sofa',        'formula': SOFA},
        {'label': 'clinical',    'formula': CLINICAL},
        {'label': 'sofa_rcs',    'formula': SOFA_RCS},
        {'label': 'sofa_weight', 'formula': SOFA_WEIGHT},
        {'label': 'sofa_bmi',    'formula': SOFA_BMI},
    ]

    # ── Dimension 2: outcome x model type ─────────────────────────────
    # Working baselines (preserved unchanged this round): sbt_done, sbt_done_abc,
    # success_extub, _success_extub. v2 family (added in 2026-04-29 model-update
    # round) lives alongside as challengers — same 5 covariate specs run against
    # all 15 (outcome, model_type) configs.
    # NOTE (2026-05-01): roster trimmed to the seven outcomes the team is
    # actively reporting. Retired outcomes are kept as commented-out lines
    # below for easy revert — paste them back into MODEL_CONFIGS to restore.
    # Convention: SBT outcomes are GEE-only (cluster-robust logit removed
    # by user direction); extubation outcomes keep both gee + logit.
    MODEL_CONFIGS = [
        # — extubation (gee + logit) —
        {'outcome': 'success_extub_next_day',     'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day',     'model_type': 'logit', 'fit_fn': _fit_logit},
        {'outcome': 'success_extub_v2_next_day',  'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_v2_next_day',  'model_type': 'logit', 'fit_fn': _fit_logit},
        # — SBT outcomes (gee only) —
        {'outcome': 'sbt_done_prefix_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_subira_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_abc_next_day',      'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_elig_next_day',          'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_v2_next_day',       'model_type': 'gee',   'fit_fn': _fit_gee},
        # — RETIRED 2026-05-01 (paste back into MODEL_CONFIGS to revert) —
        # {'outcome': 'sbt_done_next_day',          'model_type': 'gee',   'fit_fn': _fit_gee},
        # {'outcome': 'sbt_done_anyprior_next_day', 'model_type': 'gee',   'fit_fn': _fit_gee},
        # {'outcome': 'sbt_done_imv6h_next_day',    'model_type': 'gee',   'fit_fn': _fit_gee},
        # {'outcome': 'sbt_done_2min_next_day',     'model_type': 'gee',   'fit_fn': _fit_gee},
    ]

    # Outcomes that are GEE-only siblings (skipped from marginal-effect plots
    # to keep the figure roster from exploding; the forest plot covers them).
    # 2026-05-01: trimmed to kept SBT siblings only.
    SBT_VARIANT_OUTCOMES = {
        'sbt_done_prefix_next_day',
        'sbt_done_subira_next_day',
        'sbt_done_abc_next_day',
    }

    # ── Cross-product loop ─────────────────────────────────────────────
    # Key shape: (outcome, model_type) → {spec_label: result}
    fitted = {}
    for _config in MODEL_CONFIGS:
        _key = (_config['outcome'], _config['model_type'])
        fitted[_key] = {}
        for _spec in COVARIATE_SPECS:
            _formula = _spec['formula'].replace('{{outcome}}', _config['outcome'])
            try:
                _result = _config['fit_fn'](_formula, _df_scaled)
                fitted[_key][_spec['label']] = _result
                print(f"  OK: {_spec['label']} / {_config['outcome']} / {_config['model_type']}")
            except Exception as e:
                print(f"  FAIL: {_spec['label']} / {_config['outcome']} / {_config['model_type']}: {e}")
    return HURDLE_INDICATORS, MODEL_CONFIGS, SBT_VARIANT_OUTCOMES, fitted, np, re


@app.cell
def _(MODEL_CONFIGS, SITE_NAME, VAR_DISPLAY, fitted, np, pd, re):
    # ── Long-format CSV (federated-friendly) ──────────────────────────
    # Row identifier: (outcome, model_type, sensitivity).
    def _extract_long(result, label, outcome, model_type):
        _tbl = result.summary().tables[1]
        _row = pd.DataFrame(_tbl.data[1:], columns=_tbl.data[0])
        _row['sensitivity'] = label
        _row['outcome'] = outcome
        _row['model_type'] = model_type
        return _row

    sa_results = []
    for (_outcome, _mt), _spec_dict in fitted.items():
        for _label, _result in _spec_dict.items():
            sa_results.append(_extract_long(_result, _label, _outcome, _mt))
    sa_all = pd.concat(sa_results, ignore_index=True)
    _sa_path = f'output_to_share/{SITE_NAME}/models/sensitivity_analysis.csv'
    sa_all.to_csv(_sa_path, index=False)
    print(f"Saved {_sa_path} ({len(sa_all)} rows)")

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

    # Filename convention (2026-05-01): use full outcome names with the
    # `_next_day` suffix stripped. Older abbreviations (`sbt`, `extub`,
    # etc.) replaced with explicit names per user direction. Retired
    # outcomes preserved as commented-out lines for revert.
    OUTCOME_SHORT = {
        'success_extub_next_day':     'success_extub',
        'success_extub_v2_next_day':  'success_extub_v2',
        'sbt_done_prefix_next_day':   'sbt_done_prefix',
        'sbt_done_subira_next_day':   'sbt_done_subira',
        'sbt_done_abc_next_day':      'sbt_done_abc',
        'sbt_elig_next_day':          'sbt_elig',
        'sbt_done_v2_next_day':       'sbt_done_v2',
        # — RETIRED 2026-05-01 (paste back into the dict to restore) —
        # 'sbt_done_next_day':          'sbt_done',
        # 'sbt_done_anyprior_next_day': 'sbt_done_anyprior',
        # 'sbt_done_imv6h_next_day':    'sbt_done_imv6h',
        # 'sbt_done_2min_next_day':     'sbt_done_2min',
    }
    for _config in MODEL_CONFIGS:
        _key = (_config['outcome'], _config['model_type'])
        if _key not in fitted or not fitted[_key]:
            continue
        _outcome_short = OUTCOME_SHORT.get(_config['outcome'], _config['outcome'])
        _fname = f"output_to_share/{SITE_NAME}/models/model_comparison_{_outcome_short}_{_config['model_type']}.csv"
        # Skip sofa_rcs: cr() basis coefficients aren't human-interpretable
        # as OR-per-unit. The RCS results are still exported in the long-format
        # sensitivity_analysis.csv above, and summarized in the forest plot via
        # the 10→90 percentile-OR rescaling.
        _results_for_table = {
            _k: _v for _k, _v in fitted[_key].items() if _k != 'sofa_rcs'
        }
        _wide = build_wide_table(_results_for_table)
        _wide.to_csv(_fname)
        print(f"Saved {_fname} ({len(_wide)} rows x {_wide.shape[1]} cols)")
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
        'sbt_done_next_day': 'Probability of Passing SBT',
        'success_extub_next_day': 'Probability of Successful Extubation',
        'sbt_elig_next_day': 'Probability of SBT Eligibility',
        'sbt_done_v2_next_day': 'Probability of Passing SBT (v2)',
        'success_extub_v2_next_day': 'Probability of Successful Extubation (v2)',
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
        (sofa_rcs uses `cr(rate, knots=...):indicator`), force the indicator
        to 1 across the entire grid so the basis-times-indicator term actually
        evaluates the dose-response shape. Otherwise the indicator stays at
        REF_ROW's median (typically 0), the basis term is zero everywhere on
        the grid, and the curve renders flat with a full-y-axis CI ribbon.

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
        """Build and save a 2×3 figure of marginal-effect curves."""
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
        print(f"Saved: {out_path}")
        return out_path

    # Generate one 2×3 figure per (outcome, model_type, spec).
    # 2026-05-01: linear `sofa` plots removed — RCS curves carry the
    # nonlinearity signal and the linear form's straightness is implicit.
    # PLOT_SPECS controls which spec(s) to plot.
    # SBT sensitivity siblings (sbt_done_prefix / subira / abc) are skipped
    # — variants are compared via the forest plot + wide CSVs, not curves.
    PLOT_SPECS = ['sofa_rcs']
    for (_outcome, _mt), _spec_dict in fitted.items():
        if _outcome in SBT_VARIANT_OUTCOMES:
            continue
        for _spec_label in PLOT_SPECS:
            if _spec_label in _spec_dict:
                plot_marginal_effects(
                    _spec_dict[_spec_label], _outcome, _mt, _spec_label
                )
    return


# ── Forest Plots (10→90 percentile-OR rescaling) ──────────────────────


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Forest Plots — 10→90 percentile OR per outcome

    One PNG per (outcome × model_type). Each forest plot has 6 rows (3 night–day
    diffs + 3 daytime rates) and 5 horizontal pointranges per row, color-coded
    by spec (`baseline / daydose / sofa / clinical / sofa_rcs`). Every dot is on
    the same OR scale: "OR for a shift from the 10th to 90th percentile of the
    predictor's production-cohort distribution (zeros included; signed diffs
    preserved)" — per the literature pattern (Kamdar et al. 2015).

    **Linear specs** (`baseline / daydose / sofa / clinical`): rescaled
    `OR = exp(β × (x90 − x10))` using the contrast-vector form so the same code
    path works for the RCS spec.

    **RCS spec** (`sofa_rcs`): single OR per predictor for x10→x90 shift,
    computed via `predict_link(x90) − predict_link(x10)` with all other
    covariates held at median/mode. Variance uses the delta method via
    `(X_x90 − X_x10) @ V @ (X_x90 − X_x10)'`.

    **Output**: `output_to_share/{site}/models/forest_{outcome_short}_{model_type}.png`
    """)
    return


@app.cell
def _(HURDLE_INDICATORS, MODEL_CONFIGS, OUTCOME_SHORT, SITE_NAME, VAR_DISPLAY,
      cohort_merged_final, fitted, np, pd):
    import matplotlib.pyplot as _plt
    from patsy import dmatrix as _dmatrix

    # 9 predictors in the user-specified row order: 3 diffs, 3 daytime continuous,
    # 3 daytime hurdle binaries. The hurdle row pairs with the continuous row above
    # — together they decompose the daytime-level effect into "any exposure
    # (selection)" + "dose given exposed (intensity)."
    FOREST_PREDICTORS = [
        ('prop_dif_mcg_kg_min',  'Δ propofol (mcg/kg/min)'),
        ('fenteq_dif_mcg_hr',    'Δ fentanyl eq (mcg/hr)'),
        ('midazeq_dif_mg_hr',    'Δ midazolam eq (mg/hr)'),
        ('_prop_day_mcg_kg_min', 'Daytime propofol (mcg/kg/min)'),
        ('_fenteq_day_mcg_hr',   'Daytime fentanyl eq (mcg/hr)'),
        ('_midazeq_day_mg_hr',   'Daytime midazolam eq (mg/hr)'),
        ('_prop_day_any',        'Any daytime propofol (yes/no)'),
        ('_fenteq_day_any',      'Any daytime fentanyl eq (yes/no)'),
        ('_midazeq_day_any',     'Any daytime midazolam eq (yes/no)'),
    ]
    SPEC_ORDER = ['baseline', 'daydose', 'sofa', 'clinical', 'sofa_rcs',
                  'sofa_weight', 'sofa_bmi']
    # 7 dots per predictor row: 5 original specs + 2 habitus siblings (cyan, olive).
    SPEC_COLORS = {
        'baseline':    '#5e3c99',
        'daydose':     '#1f77b4',
        'sofa':        '#2ca02c',
        'clinical':    '#ff7f0e',
        'sofa_rcs':    '#d62728',
        'sofa_weight': '#17becf',  # cyan — body-habitus sibling 1
        'sofa_bmi':    '#bcbd22',  # olive — body-habitus sibling 2
    }

    # ── Build PERCENTILE_REF: per-predictor (x10_raw, x90_raw, x10_scaled,
    # x90_scaled). Default: full cohort distribution (zeros included). For
    # daytime continuous-rate predictors paired with a hurdle indicator, use
    # the NON-ZERO subset — the forest cell for those predictors then reports
    # "intensity 10→90 OR among the exposed," cleanly separated from the
    # selection effect captured by the indicator's own forest row.
    PERCENTILE_REF = {}
    for _pred, _ in FOREST_PREDICTORS:
        _vals = cohort_merged_final[_pred].dropna().to_numpy()
        if len(_vals) == 0:
            continue
        if _pred in HURDLE_INDICATORS:
            _vals = _vals[_vals != 0]
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
            'subset': 'non-zero' if _pred in HURDLE_INDICATORS else 'all',
        }
    print("PERCENTILE_REF (raw clinical units, 10th and 90th percentiles):")
    for _pred, _info in PERCENTILE_REF.items():
        _tag = f"  [{_info['subset']}]" if _info['subset'] == 'non-zero' else ''
        print(f"  {_pred:<24s}: x10={_info['x10_raw']:>+8.3f}, x90={_info['x90_raw']:>+8.3f}{_tag}")

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

    def _or_10_to_90(fit, predictor):
        """Return (OR, OR_lo, OR_hi) for a 10→90 percentile shift in `predictor`.

        Works for BOTH linear specs (where the design matrix changes only on
        the `predictor` column) and RCS specs (where 3 basis columns change
        together via patsy's cr() expansion). The contrast vector
        c = X_x90 − X_x10 captures the spec automatically; var(log_OR) =
        c V c' is the delta method.
        """
        info = PERCENTILE_REF.get(predictor)
        if info is None or info['x10_raw'] == info['x90_raw']:
            return (np.nan, np.nan, np.nan)

        # Build two new-data rows, predictor at x10_scaled vs x90_scaled.
        nd_x10 = pd.DataFrame([REF_ROW])
        nd_x90 = pd.DataFrame([REF_ROW])
        nd_x10[predictor] = info['x10_scaled']
        nd_x90[predictor] = info['x90_scaled']
        # If the predictor is paired with a hurdle indicator (sofa_rcs spec
        # uses `cr(rate, knots=...):indicator`), force indicator=1 at both
        # endpoints so the basis-times-indicator contrast captures the
        # intensity-among-exposed effect. Otherwise the indicator stays at
        # REF_ROW's median (typically 0), the contrast on the basis-times-
        # indicator term is trivially 0, and OR comes out 1.0 with broken CI.
        _ind = HURDLE_INDICATORS.get(predictor)
        if _ind is not None:
            nd_x10[_ind] = 1
            nd_x90[_ind] = 1

        # Re-evaluate the formula's design matrix on the new rows so cr()
        # basis columns are recomputed at the new predictor value.
        try:
            di = fit.model.data.design_info
            X_x10 = np.asarray(_dmatrix(di, nd_x10, return_type='matrix'))[0]
            X_x90 = np.asarray(_dmatrix(di, nd_x90, return_type='matrix'))[0]
        except Exception:
            return (np.nan, np.nan, np.nan)

        beta = fit.params.values
        V = fit.cov_params().values
        contrast = X_x90 - X_x10
        # If the predictor isn't in the spec's design matrix, the contrast
        # is all zeros → log_OR = 0 → OR = 1 with zero-width CI. That's a
        # misleading display ("baseline shows null effect"); return NaN so
        # the forest plot skips that dot instead.
        if np.allclose(contrast, 0):
            return (np.nan, np.nan, np.nan)
        log_or = float(contrast @ beta)
        var_log_or = float(contrast @ V @ contrast)
        if var_log_or < 0:
            return (np.nan, np.nan, np.nan)
        se = np.sqrt(var_log_or)
        return (
            float(np.exp(log_or)),
            float(np.exp(log_or - 1.96 * se)),
            float(np.exp(log_or + 1.96 * se)),
        )

    # ── Build long-format result table (one row per outcome×model_type×spec×predictor).
    forest_rows = []
    for (_outcome, _mt), _spec_dict in fitted.items():
        for _spec_label, _result in _spec_dict.items():
            for _pred, _ in FOREST_PREDICTORS:
                _or, _or_lo, _or_hi = _or_10_to_90(_result, _pred)
                forest_rows.append({
                    'outcome': _outcome,
                    'model_type': _mt,
                    'spec': _spec_label,
                    'predictor': _pred,
                    'OR': _or,
                    'OR_lo': _or_lo,
                    'OR_hi': _or_hi,
                })
    forest_df = pd.DataFrame(forest_rows)
    _forest_csv = f'output_to_share/{SITE_NAME}/models/forest_data.csv'
    forest_df.to_csv(_forest_csv, index=False)
    print(f"Saved {_forest_csv} ({len(forest_df)} rows)")

    # ── Render one PNG per (outcome, model_type) ─────────────────────
    def plot_forest(rows_df, outcome, model_type, site, predictors, percentile_ref, out_path):
        fig, ax = _plt.subplots(figsize=(9.5, 7.0))
        n_specs = len(SPEC_ORDER)
        # vertical jitter within each predictor row, ±0.18
        jitter = np.linspace(-0.20, 0.20, n_specs)

        ymin, ymax = -0.6, len(predictors) - 0.4
        for i, (pred, pred_label) in enumerate(predictors):
            y_base = len(predictors) - 1 - i  # top-to-bottom: first predictor at top
            for j, spec in enumerate(SPEC_ORDER):
                _row = rows_df[
                    (rows_df['outcome'] == outcome)
                    & (rows_df['model_type'] == model_type)
                    & (rows_df['spec'] == spec)
                    & (rows_df['predictor'] == pred)
                ]
                if _row.empty:
                    continue
                _r = _row.iloc[0]
                if not (np.isfinite(_r['OR']) and np.isfinite(_r['OR_lo']) and np.isfinite(_r['OR_hi'])):
                    continue
                _y = y_base + jitter[j]
                ax.errorbar(
                    _r['OR'], _y,
                    xerr=[[_r['OR'] - _r['OR_lo']], [_r['OR_hi'] - _r['OR']]],
                    fmt='o', color=SPEC_COLORS[spec], markersize=4,
                    capsize=2, elinewidth=1.0, label=spec if i == 0 else None,
                )

        # y-axis: predictor labels with x10/x90 annotation
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

        # Legend (one entry per spec, deduplicated). Place above the plot.
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
            forest_df, _outcome, _mt, SITE_NAME,
            FOREST_PREDICTORS, PERCENTILE_REF, _out_path,
        )
        print(f"Saved: {_out_path}")
    return (FOREST_PREDICTORS, PERCENTILE_REF, forest_df)


if __name__ == "__main__":
    app.run()
