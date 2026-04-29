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
    ## GEE: SBT Done Next Day
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    import statsmodels.api as _sm
    sbt_done_formula = """sbt_done_next_day ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr +
    _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    gee_model = _smf.gee(formula=sbt_done_formula, groups='hospitalization_id', data=cohort_merged_final, family=_sm.families.Binomial())
    gee_result = gee_model.fit()
    print(gee_result.summary())
    _summary_df = gee_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output_to_share/{SITE_NAME}/models/gee_summary.csv', index=False)
    _cov_matrix = gee_result.cov_params()
    _cov_matrix.to_csv(f'output_to_share/{SITE_NAME}/models/gee_covmat.csv')
    print(f"Saved output_to_share/{SITE_NAME}/models/gee_summary.csv, gee_covmat.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Multicollinearity diagnostic (VIF) — `vif_sbt_rate.csv`

    Variance inflation factor for each continuous regressor in the primary
    rate-parameterization GEE. VIF_j = 1 / (1 − R²_j) where R²_j is from
    regressing predictor *j* on all other predictors. Categorical dummies are
    excluded — their VIF interpretation is awkward and they aren't the
    suspected source of the OR-reversal pattern.

    The six dose terms have a built-in linear-combination structure
    (`*_dif_*` = night − day), so high VIF on any of them would flag that
    redundancy as a candidate explanation for the counterintuitive OR > 1 sign
    on `*_dif_*` coefficients post-Phase-4.

    Rule of thumb: VIF > 5 is moderate, > 10 is severe, ≥ 20 is typically
    enough to flip a coefficient sign.
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    from patsy import dmatrix as _dmatrix
    from statsmodels.stats.outliers_influence import (
        variance_inflation_factor as _vif_fn,
    )
    _vif_terms = (
        "prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr + "
        "_prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr + "
        "ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + "
        "nee_7am + nee_7pm + age + _nth_day + sofa_total + cci_score"
    )
    _X = _dmatrix(_vif_terms, data=cohort_merged_final, return_type='dataframe').dropna()
    _vif_rows = []
    for _i, _name in enumerate(_X.columns):
        if _name == 'Intercept':
            continue
        _vif_rows.append({'term': _name, 'vif': _vif_fn(_X.values, _i)})
    _vif_df = pd.DataFrame(_vif_rows).sort_values('vif', ascending=False)
    _vif_path = f'output_to_share/{SITE_NAME}/models/vif_sbt_rate.csv'
    _vif_df.to_csv(_vif_path, index=False)
    print(_vif_df.to_string(index=False))
    print(
        f"Severe (VIF>10): {(_vif_df['vif'] > 10).sum()}; "
        f"Moderate (5<VIF≤10): {((_vif_df['vif'] > 5) & (_vif_df['vif'] <= 10)).sum()}"
    )
    print(f"Saved {_vif_path}")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sensitivity (day-0 included) — `gee_summary_day0.csv`

    Same primary rate-parameterization GEE as the production cell above, but
    fit on `exposure_dataset.parquet` — the exposure-characterization sibling
    that retains day 0 (the partial admission day before the first 7am,
    normally excluded by the `_nth_day > 0` filter) AND last-day rows. Rate
    divisors in that dataset use `NULLIF(n_hours_*, 0)` so single-shift
    normalization is correct; day-1+ rows are numerically identical to
    production. The next-day-outcome filter is applied inline below so the
    day-0 SA model fits the same modeling cohort as production plus the day-0
    rows.

    The diagnostic question this answers: does excluding day 0 contribute to
    the counterintuitive OR > 1 sign on dose-rate coefficients? If signs
    persist on the day-0 fit, the issue is elsewhere (e.g., multicollinearity
    per the VIF cell above). If signs flip, day-0 inclusion materially changes
    the picture.
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    import statsmodels.formula.api as _smf_d0
    import statsmodels.api as _sm_d0
    _df_day0 = pd.read_parquet(f"output/{SITE_NAME}/exposure_dataset.parquet")
    # Apply the next-day-outcome filter that the exposure parquet doesn't
    # apply globally (since exposure characterization wants last-day rows).
    # This restores the day-0 SA cohort = production modeling cohort + day-0
    # rows that have a next day.
    _df_day0 = _df_day0[
        _df_day0['sbt_done_next_day'].notna()
        & _df_day0['success_extub_next_day'].notna()
    ].reset_index(drop=True)
    _n_day0_rows = (_df_day0['_nth_day'] == 0).sum()
    print(f"[day-0] Loaded {len(_df_day0)} rows ({_n_day0_rows} are day-0 rows)")
    sbt_done_formula_d0 = """sbt_done_next_day ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr +
    _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    _gee_d0 = _smf_d0.gee(
        formula=sbt_done_formula_d0,
        groups='hospitalization_id',
        data=_df_day0,
        family=_sm_d0.families.Binomial(),
    ).fit()
    _summary_df = _gee_d0.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output_to_share/{SITE_NAME}/models/gee_summary_day0.csv', index=False)
    _gee_d0.cov_params().to_csv(f'output_to_share/{SITE_NAME}/models/gee_covmat_day0.csv')
    print(f"Saved output_to_share/{SITE_NAME}/models/gee_summary_day0.csv, gee_covmat_day0.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Sensitivity (absolute amounts, per 12-h shift) — `gee_summary_amount.csv`

    Same GEE structure as the rate-based production model above, but exposures
    are total mg / mcg over each 12-hour shift (the pre-2026-04-24
    parameterization). For full 12-hour shifts (everything in this dataset
    post-filter), amount = rate × 12 exactly, so coefficients should match
    the rate model up to a unit rescaling. Any larger discrepancy is a
    data-quality canary.
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf_amt
    import statsmodels.api as _sm_amt
    sbt_done_formula_amt = """sbt_done_next_day ~ prop_dif_mcg_kg + fenteq_dif_mcg + midazeq_dif_mg +
    _prop_day_mcg_kg + _midazeq_day_mg + _fenteq_day_mcg +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    gee_model_amt = _smf_amt.gee(formula=sbt_done_formula_amt, groups='hospitalization_id', data=cohort_merged_final, family=_sm_amt.families.Binomial())
    gee_result_amt = gee_model_amt.fit()
    _summary_df = gee_result_amt.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output_to_share/{SITE_NAME}/models/gee_summary_amount.csv', index=False)
    _cov_matrix = gee_result_amt.cov_params()
    _cov_matrix.to_csv(f'output_to_share/{SITE_NAME}/models/gee_covmat_amount.csv')
    print(f"Saved output_to_share/{SITE_NAME}/models/gee_summary_amount.csv, gee_covmat_amount.csv")
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
    ### Sensitivity (absolute amounts, per 12-h shift) — `logit_summary_amount.csv`

    Mirror of the production logit above using the per-shift total exposures.
    """)
    return


@app.cell
def _(SITE_NAME, cohort_merged_final, pd):
    import statsmodels.formula.api as _smf_amt
    success_extub_formula_amt = """success_extub_next_day ~ prop_dif_mcg_kg + fenteq_dif_mcg + midazeq_dif_mg +
    _prop_day_mcg_kg + _midazeq_day_mg + _fenteq_day_mcg +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    _logit_amt_cols = [
        'prop_dif_mcg_kg', 'fenteq_dif_mcg', 'midazeq_dif_mg',
        '_prop_day_mcg_kg', '_midazeq_day_mg', '_fenteq_day_mcg',
        'ph_level_7am', 'ph_level_7pm', 'pf_level_7am', 'pf_level_7pm',
        'nee_7am', 'nee_7pm', 'age', '_nth_day', 'sofa_total', 'cci_score',
        'sex_category', 'icu_type', 'success_extub_next_day',
        'hospitalization_id',
    ]
    _logit_amt_df = cohort_merged_final.dropna(subset=_logit_amt_cols)
    logit_model_amt = _smf_amt.logit(formula=success_extub_formula_amt, data=_logit_amt_df)
    logit_result_amt = logit_model_amt.fit(cov_type='cluster', cov_kwds={'groups': _logit_amt_df['hospitalization_id']})
    _summary_df = logit_result_amt.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv(f'output_to_share/{SITE_NAME}/models/logit_summary_amount.csv', index=False)
    _cov_matrix = logit_result_amt.cov_params()
    _cov_matrix.to_csv(f'output_to_share/{SITE_NAME}/models/logit_covmat_amount.csv')
    print(f"Saved output_to_share/{SITE_NAME}/models/logit_summary_amount.csv, logit_covmat_amount.csv")
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
        # Exposures (day-night rate differences) — production
        'prop_dif_mcg_kg_min': {'scale': 10,  'label': 'Δ propofol (per 10 mcg/kg/min)'},
        'fenteq_dif_mcg_hr':   {'scale': 10,  'label': 'Δ fentanyl eq (per 10 mcg/hr)'},
        'midazeq_dif_mg_hr':   {'scale': 0.1, 'label': 'Δ midazolam eq (per 0.1 mg/hr)'},
        # Daytime absolute rates (production)
        '_prop_day_mcg_kg_min': {'scale': 10,  'label': 'Daytime propofol (per 10 mcg/kg/min)'},
        '_fenteq_day_mcg_hr':   {'scale': 10,  'label': 'Daytime fentanyl eq (per 10 mcg/hr)'},
        '_midazeq_day_mg_hr':   {'scale': 0.1, 'label': 'Daytime midazolam eq (per 0.1 mg/hr)'},
        # Sensitivity: absolute amounts per 12-h shift. For propofol, the
        # absolute amount is now in mcg/kg (was mg). 720 mcg/kg over 12h
        # ≈ 1 mcg/kg/min × 60 × 12; "per 120 mcg/kg" is comparable to
        # "per 10 mcg/kg/min × 12 min ≈ 120" (rough order-of-magnitude match).
        'prop_dif_mcg_kg': {'scale': 120, 'label': 'Δ propofol (per 120 mcg/kg over 12h)'},
        'fenteq_dif_mcg':  {'scale': 120, 'label': 'Δ fentanyl eq (per 120 mcg over 12h)'},
        'midazeq_dif_mg':  {'scale': 1.2, 'label': 'Δ midazolam eq (per 1.2 mg over 12h)'},
        '_prop_day_mcg_kg': {'scale': 120, 'label': 'Daytime propofol (per 120 mcg/kg)'},
        '_fenteq_day_mcg':  {'scale': 120, 'label': 'Daytime fentanyl eq (per 120 mcg)'},
        '_midazeq_day_mg':  {'scale': 1.2, 'label': 'Daytime midazolam eq (per 1.2 mg)'},
        # Other continuous covariates (unchanged)
        'age':          {'scale': 1,   'label': 'Age (per year)'},
        'cci_score':    {'scale': 1,   'label': 'Charlson CCI (per point)'},
        'sofa_total':   {'scale': 1,   'label': 'SOFA total (per point)'},
        'nee_7am':      {'scale': 0.1, 'label': 'NEE 7am (per 0.1 mcg/kg/min)'},
        'nee_7pm':      {'scale': 0.1, 'label': 'NEE 7pm (per 0.1 mcg/kg/min)'},
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
    def _fit_gee(formula, data):
        m = smf.gee(formula=formula, groups='hospitalization_id',
                    data=data, family=sm.families.Binomial())
        return m.fit()

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
                     cov_kwds={'groups': _d['hospitalization_id']})

    # ── Dimension 1: nested covariate sets (all include exposures) ────
    # PRODUCTION parameterization: rate-based exposures (mg/hr, mcg/hr)
    BASELINE = ("{{outcome}} ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr + "
                "age + C(sex_category) + C(icu_type) + cci_score")
    DAYDOSE = BASELINE + " + _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr"
    SOFA = DAYDOSE + " + sofa_total"
    CLINICAL = DAYDOSE + (" + ph_level_7am + ph_level_7pm + pf_level_7am + "
                          "pf_level_7pm + nee_7am + nee_7pm")

    # RCS (restricted cubic splines) variant of the sofa spec.
    # Wraps the 6 exposure variables in patsy's cr() transform for natural
    # cubic regression splines with df=4 (→ 3 basis columns per variable).
    # Non-exposure adjustment variables stay linear/categorical.
    # Used for marginal-effect plots (curves can bend to reveal dose-response
    # shape) but excluded from the wide comparison CSV — cr() basis coefficients
    # aren't human-interpretable as OR-per-unit. See plan_rcs_exposures.md memory.
    SOFA_RCS = (
        "{{outcome}} ~ "
        "cr(prop_dif_mcg_kg_min, df=4) + cr(fenteq_dif_mcg_hr, df=4) + cr(midazeq_dif_mg_hr, df=4) + "
        "cr(_prop_day_mcg_kg_min, df=4) + cr(_fenteq_day_mcg_hr, df=4) + cr(_midazeq_day_mg_hr, df=4) + "
        "age + C(sex_category) + C(icu_type) + cci_score + sofa_total"
    )

    # SENSITIVITY parameterization: absolute amounts (mg, mcg) per 12-h shift.
    # Mirrors the production specs above with bare _mg / _mcg column names.
    BASELINE_AMOUNT = ("{{outcome}} ~ prop_dif_mcg_kg + fenteq_dif_mcg + midazeq_dif_mg + "
                       "age + C(sex_category) + C(icu_type) + cci_score")
    DAYDOSE_AMOUNT = BASELINE_AMOUNT + " + _prop_day_mcg_kg + _midazeq_day_mg + _fenteq_day_mcg"
    SOFA_AMOUNT = DAYDOSE_AMOUNT + " + sofa_total"
    CLINICAL_AMOUNT = DAYDOSE_AMOUNT + (" + ph_level_7am + ph_level_7pm + pf_level_7am + "
                                       "pf_level_7pm + nee_7am + nee_7pm")
    SOFA_RCS_AMOUNT = (
        "{{outcome}} ~ "
        "cr(prop_dif_mcg_kg, df=4) + cr(fenteq_dif_mcg, df=4) + cr(midazeq_dif_mg, df=4) + "
        "cr(_prop_day_mcg_kg, df=4) + cr(_fenteq_day_mcg, df=4) + cr(_midazeq_day_mg, df=4) + "
        "age + C(sex_category) + C(icu_type) + cci_score + sofa_total"
    )

    # Nested specs grouped by parameterization. Both lists share spec labels
    # ('baseline', 'daydose', ...) so the wide-table builder can compare them
    # side-by-side at the cost of a parameterization suffix on the filename.
    COVARIATE_SPECS_BY_PARAM = {
        'rate': [
            {'label': 'baseline', 'formula': BASELINE},
            {'label': 'daydose',  'formula': DAYDOSE},
            {'label': 'sofa',     'formula': SOFA},
            {'label': 'clinical', 'formula': CLINICAL},
            {'label': 'sofa_rcs', 'formula': SOFA_RCS},
        ],
        'amount': [
            {'label': 'baseline', 'formula': BASELINE_AMOUNT},
            {'label': 'daydose',  'formula': DAYDOSE_AMOUNT},
            {'label': 'sofa',     'formula': SOFA_AMOUNT},
            {'label': 'clinical', 'formula': CLINICAL_AMOUNT},
            {'label': 'sofa_rcs', 'formula': SOFA_RCS_AMOUNT},
        ],
    }

    # ── Dimension 2: outcome x model type ─────────────────────────────
    # Primary outcomes get fit in BOTH rate and amount parameterizations.
    # SBT sensitivity siblings (anyprior / imv6h / prefix / 2min) are
    # rate-only — rate↔amount agreement was already established at perfect
    # ratio = 144.000 = 12² for the spec-literal primary, so triplicate
    # amount fits for siblings would add no diagnostic information.
    MODEL_CONFIGS = [
        {'outcome': 'sbt_done_next_day',          'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day',     'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day',     'model_type': 'logit', 'fit_fn': _fit_logit},
        # SBT sensitivity siblings (rate parameterization only)
        {'outcome': 'sbt_done_anyprior_next_day', 'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_imv6h_next_day',    'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_prefix_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_2min_next_day',     'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_subira_next_day',   'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'sbt_done_abc_next_day',      'model_type': 'gee',   'fit_fn': _fit_gee},
    ]

    # Outcomes restricted to rate parameterization only (skip amount fit).
    SBT_VARIANT_OUTCOMES = {
        'sbt_done_anyprior_next_day',
        'sbt_done_imv6h_next_day',
        'sbt_done_prefix_next_day',
        'sbt_done_2min_next_day',
        'sbt_done_subira_next_day',
        'sbt_done_abc_next_day',
    }

    # ── Cross-product loop: now also iterates over parameterization
    # Key shape: (outcome, model_type, param) → {spec_label: result}
    fitted = {}
    for _param, _specs in COVARIATE_SPECS_BY_PARAM.items():
        for _config in MODEL_CONFIGS:
            # Variants are rate-only (see SBT_VARIANT_OUTCOMES comment above)
            if _config['outcome'] in SBT_VARIANT_OUTCOMES and _param == 'amount':
                continue
            _key = (_config['outcome'], _config['model_type'], _param)
            fitted[_key] = {}
            for _spec in _specs:
                _formula = _spec['formula'].replace('{{outcome}}', _config['outcome'])
                try:
                    _result = _config['fit_fn'](_formula, _df_scaled)
                    fitted[_key][_spec['label']] = _result
                    print(f"  OK: {_param} / {_spec['label']} / {_config['outcome']} / {_config['model_type']}")
                except Exception as e:
                    print(f"  FAIL: {_param} / {_spec['label']} / {_config['outcome']} / {_config['model_type']}: {e}")
    return MODEL_CONFIGS, SBT_VARIANT_OUTCOMES, fitted, np, re


@app.cell
def _(MODEL_CONFIGS, SITE_NAME, VAR_DISPLAY, fitted, np, pd, re):
    # ── Long-format CSV (federated-friendly) ──────────────────────────
    # Now carries a 'parameterization' column ('rate' vs 'amount') so the
    # row identifier is (outcome, model_type, parameterization, sensitivity).
    def _extract_long(result, label, outcome, model_type, parameterization):
        _tbl = result.summary().tables[1]
        _row = pd.DataFrame(_tbl.data[1:], columns=_tbl.data[0])
        _row['sensitivity'] = label
        _row['outcome'] = outcome
        _row['model_type'] = model_type
        _row['parameterization'] = parameterization
        return _row

    sa_results = []
    for (_outcome, _mt, _param), _spec_dict in fitted.items():
        for _label, _result in _spec_dict.items():
            sa_results.append(_extract_long(_result, _label, _outcome, _mt, _param))
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

    # Filename convention: rate parameterization keeps the production file
    # name unsuffixed (`model_comparison_sbt_gee.csv`); amount is sibling
    # suffixed (`model_comparison_sbt_gee_amount.csv`). SBT sensitivity
    # siblings get suffixed short-names so each variant gets its own
    # CSV (otherwise all "sbt"-containing outcomes would collapse).
    _OUTCOME_SHORT = {
        'sbt_done_next_day':          'sbt',
        'sbt_done_anyprior_next_day': 'sbt_anyprior',
        'sbt_done_imv6h_next_day':    'sbt_imv6h',
        'sbt_done_prefix_next_day':   'sbt_prefix',
        'sbt_done_2min_next_day':     'sbt_2min',
        'sbt_done_subira_next_day':   'sbt_subira',
        'sbt_done_abc_next_day':      'sbt_abc',
        'success_extub_next_day':     'extub',
    }
    for _config in MODEL_CONFIGS:
        for _param in ['rate', 'amount']:
            _key = (_config['outcome'], _config['model_type'], _param)
            if _key not in fitted or not fitted[_key]:
                continue
            _outcome_short = _OUTCOME_SHORT.get(_config['outcome'], _config['outcome'])
            _suffix = '' if _param == 'rate' else '_amount'
            _fname = f"output_to_share/{SITE_NAME}/models/model_comparison_{_outcome_short}_{_config['model_type']}{_suffix}.csv"
            # Skip sofa_rcs: cr() basis coefficients aren't human-interpretable
            # as OR-per-unit. The RCS results are still exported in the long-format
            # sensitivity_analysis.csv above, and visualized via marginal-effect plots.
            _results_for_table = {
                _k: _v for _k, _v in fitted[_key].items() if _k != 'sofa_rcs'
            }
            _wide = build_wide_table(_results_for_table)
            _wide.to_csv(_fname)
            print(f"Saved {_fname} ({len(_wide)} rows x {_wide.shape[1]} cols)")
    return


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
def _(SBT_VARIANT_OUTCOMES, SITE_NAME, VAR_DISPLAY, cohort_merged_final, fitted, np, pd):
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

        _outcome_short = 'sbt' if 'sbt' in outcome else 'extub'
        out_path = (
            f"output_to_share/{SITE_NAME}/models/"
            f"marginal_effects_{_outcome_short}_{model_type}_{spec_label}.png"
        )
        fig.savefig(out_path, dpi=250, bbox_inches='tight', facecolor='white')
        _plt.close(fig)
        print(f"Saved: {out_path}")
        return out_path

    # Generate one 2×3 figure per (outcome, model_type, spec).
    # PLOT_SPECS controls which spec(s) to plot. We generate BOTH the linear
    # `sofa` spec AND the RCS `sofa_rcs` spec so reviewers can visually compare
    # them — if the RCS curve looks straight, that's a "no nonlinearity detected"
    # signal without needing a formal p-value threshold. Plotting both specs
    # also gives a stronger methods story than pre-specifying one.
    # To add/remove specs, edit PLOT_SPECS. Filenames encode the spec label.
    # Only the 'rate' parameterization is plotted — amount-parameterized fits
    # produce structurally identical curves up to x-axis units (rate × 12).
    # SBT sensitivity siblings (anyprior / imv6h / prefix / 2min) are also
    # skipped here — variants are compared via CSVs, not figures.
    PLOT_SPECS = ['sofa', 'sofa_rcs']
    for (_outcome, _mt, _param), _spec_dict in fitted.items():
        if _param != 'rate':
            continue
        if _outcome in SBT_VARIANT_OUTCOMES:
            continue
        for _spec_label in PLOT_SPECS:
            if _spec_label in _spec_dict:
                plot_marginal_effects(
                    _spec_dict[_spec_label], _outcome, _mt, _spec_label
                )
    return


if __name__ == "__main__":
    app.run()
