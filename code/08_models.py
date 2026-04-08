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

    os.makedirs("output_to_share", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return (pd,)


@app.cell
def _(pd):
    cohort_merged_final = pd.read_parquet("output/analytical_dataset.parquet")
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
def _(cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    import statsmodels.api as _sm
    sbt_done_formula = """sbt_done_next_day ~ prop_dif + fenteq_dif + midazeq_dif +
    _prop_day + _midazeq_day + _fenteq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    gee_model = _smf.gee(formula=sbt_done_formula, groups='hospitalization_id', data=cohort_merged_final, family=_sm.families.Binomial())
    gee_result = gee_model.fit()
    print(gee_result.summary())
    _summary_df = gee_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv('output_to_share/gee_summary.csv', index=False)
    _cov_matrix = gee_result.cov_params()
    _cov_matrix.to_csv('output_to_share/gee_covmat.csv')
    print("Saved output_to_share/gee_summary.csv, gee_covmat.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Logistic Regression: Successful Extubation Next Day
    """)
    return


@app.cell
def _(cohort_merged_final, pd):
    import statsmodels.formula.api as _smf
    success_extub_formula = """success_extub_next_day ~ prop_dif + fenteq_dif + midazeq_dif +
    _prop_day + _midazeq_day + _fenteq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """
    logit_model = _smf.logit(formula=success_extub_formula, data=cohort_merged_final)
    logit_result = logit_model.fit(cov_type='cluster', cov_kwds={'groups': cohort_merged_final['hospitalization_id']})
    print(logit_result.summary())
    _summary_df = logit_result.summary().tables[1]
    _summary_pd = pd.DataFrame(_summary_df.data[1:], columns=_summary_df.data[0])
    _summary_pd.to_csv('output_to_share/logit_summary.csv', index=False)
    _cov_matrix = logit_result.cov_params()
    _cov_matrix.to_csv('output_to_share/logit_covmat.csv')
    print("Saved output_to_share/logit_summary.csv, logit_covmat.csv")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Nested Model Comparison

    Fits 4 nested models (progressively adding adjustment variables) for each
    (outcome, model_type) combination.

    **Model specs (all include exposures: prop_dif + fenteq_dif + midazeq_dif):**

    1. **baseline**: age + sex + ICU type + CCI

    2. **daydose**: baseline + daytime absolute dose rates (`_prop_day`, `_fenteq_day`, `_midazeq_day`, all in mg/hr or mcg/hr)

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
    # UNIT NOTE: Dose columns (prop_dif, fenteq_dif, midazeq_dif, _prop_day,
    # _fenteq_day, _midazeq_day) are per-hour RATES (mg/hr or mcg/hr), not
    # per-shift totals. The ÷12 conversion happens in 05_analytical_dataset.py.
    # Scales below were retuned so "per N unit" stays clinically meaningful
    # after the rate conversion (old totals-era scales were ~12x larger).
    VAR_DISPLAY = {
        # Exposures (day-night rate differences)
        'prop_dif':     {'scale': 10,  'label': 'Δ propofol (per 10 mg/hr)'},
        'fenteq_dif':   {'scale': 10,  'label': 'Δ fentanyl eq (per 10 mcg/hr)'},
        'midazeq_dif':  {'scale': 0.1, 'label': 'Δ midazolam eq (per 0.1 mg/hr)'},
        # Daytime absolute rates
        '_prop_day':    {'scale': 10,  'label': 'Daytime propofol (per 10 mg/hr)'},
        '_fenteq_day':  {'scale': 10,  'label': 'Daytime fentanyl eq (per 10 mcg/hr)'},
        '_midazeq_day': {'scale': 0.1, 'label': 'Daytime midazolam eq (per 0.1 mg/hr)'},
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
        m = smf.logit(formula=formula, data=data)
        return m.fit(cov_type='cluster',
                     cov_kwds={'groups': data['hospitalization_id']})

    # ── Dimension 1: nested covariate sets (all include exposures) ────
    BASELINE = ("{{outcome}} ~ prop_dif + fenteq_dif + midazeq_dif + "
                "age + C(sex_category) + C(icu_type) + cci_score")
    DAYDOSE = BASELINE + " + _prop_day + _midazeq_day + _fenteq_day"
    SOFA = DAYDOSE + " + sofa_total"
    CLINICAL = DAYDOSE + (" + ph_level_7am + ph_level_7pm + pf_level_7am + "
                          "pf_level_7pm + nee_7am + nee_7pm")

    COVARIATE_SPECS = [
        {'label': 'baseline', 'formula': BASELINE},
        {'label': 'daydose',  'formula': DAYDOSE},
        {'label': 'sofa',     'formula': SOFA},
        {'label': 'clinical', 'formula': CLINICAL},
    ]

    # ── Dimension 2: outcome x model type ─────────────────────────────
    MODEL_CONFIGS = [
        {'outcome': 'sbt_done_next_day',      'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day', 'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day', 'model_type': 'logit', 'fit_fn': _fit_logit},
    ]

    # ── Cross-product loop: keep fitted results for wide-table builder
    fitted = {}  # {(outcome, model_type): {spec_label: result}}
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
    return MODEL_CONFIGS, fitted, np, re


@app.cell
def _(MODEL_CONFIGS, VAR_DISPLAY, fitted, np, pd, re):
    # ── Long-format CSV (federated-friendly) ──────────────────────────
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
    sa_all.to_csv('output_to_share/sensitivity_analysis.csv', index=False)
    print(f"Saved output_to_share/sensitivity_analysis.csv ({len(sa_all)} rows)")

    # ── Wide comparison tables per (outcome, model_type) ──────────────
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

    for _config in MODEL_CONFIGS:
        _key = (_config['outcome'], _config['model_type'])
        if _key not in fitted or not fitted[_key]:
            continue
        _outcome_short = 'sbt' if 'sbt' in _config['outcome'] else 'extub'
        _fname = f"output_to_share/model_comparison_{_outcome_short}_{_config['model_type']}.csv"
        _wide = build_wide_table(fitted[_key])
        _wide.to_csv(_fname)
        print(f"Saved {_fname} ({len(_wide)} rows x {_wide.shape[1]} cols)")
    return


if __name__ == "__main__":
    app.run()
