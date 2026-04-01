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
    sbt_done_formula = """sbt_done_next_day ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
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
    success_extub_formula = """success_extub_next_day ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
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
    ## Sensitivity Analyses

    Cross-products two dimensions:

    - **COVARIATE_SPECS**: formula variants (primary, Elixhauser, no-SOFA, base)

    - **MODEL_CONFIGS**: outcome x model type (SBT=GEE, extub=GEE+logit)

    Adding a new SA = append to the relevant list.
    """)
    return


@app.cell
def _(cohort_merged_final, pd):
    import statsmodels.formula.api as smf
    import statsmodels.api as sm

    # ── Fit functions ──────────────────────────────────────────────────
    def _fit_gee(formula, data):
        m = smf.gee(formula=formula, groups='hospitalization_id',
                    data=data, family=sm.families.Binomial())
        return m.fit()

    def _fit_logit(formula, data):
        m = smf.logit(formula=formula, data=data)
        return m.fit(cov_type='cluster',
                     cov_kwds={'groups': data['hospitalization_id']})

    def _extract(result, label, outcome, model_type):
        _tbl = result.summary().tables[1]
        _df = pd.DataFrame(_tbl.data[1:], columns=_tbl.data[0])
        _df['sensitivity'] = label
        _df['outcome'] = outcome
        _df['model_type'] = model_type
        return _df

    # ── Dimension 1: covariate sets ───────────────────────────────────
    PRIMARY_FORMULA = """{{outcome}} ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day +
    ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm +
    age + _nth_day + sofa_total + cci_score + C(sex_category) + C(icu_type)
    """

    BASE_FORMULA = """{{outcome}} ~ propofol_diff + fentanyl_eq_diff + midazolam_eq_diff +
    _propofol_day + _midazolam_eq_day + _fentanyl_eq_day + age
    """

    COVARIATE_SPECS = [
        {'label': 'primary', 'formula': PRIMARY_FORMULA},
        {'label': 'sa_elix', 'formula': PRIMARY_FORMULA.replace('cci_score', 'elix_score')},
        {'label': 'sa_no_sofa', 'formula': PRIMARY_FORMULA.replace(' + sofa_total', '')},
        {'label': 'sa_base', 'formula': BASE_FORMULA},
    ]

    # ── Dimension 2: outcome x model type ─────────────────────────────
    MODEL_CONFIGS = [
        {'outcome': 'sbt_done_next_day',     'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day', 'model_type': 'gee',   'fit_fn': _fit_gee},
        {'outcome': 'success_extub_next_day', 'model_type': 'logit', 'fit_fn': _fit_logit},
    ]

    # ── Cross-product loop ────────────────────────────────────────────
    sa_results = []
    for spec in COVARIATE_SPECS:
        for config in MODEL_CONFIGS:
            _formula = spec['formula'].replace('{{outcome}}', config['outcome'])
            try:
                _result = config['fit_fn'](_formula, cohort_merged_final)
                sa_results.append(
                    _extract(_result, spec['label'], config['outcome'], config['model_type'])
                )
                print(f"  OK: {spec['label']} / {config['outcome']} / {config['model_type']}")
            except Exception as e:
                print(f"  FAIL: {spec['label']} / {config['outcome']} / {config['model_type']}: {e}")

    sa_all = pd.concat(sa_results, ignore_index=True)
    sa_all.to_csv('output_to_share/sensitivity_analysis.csv', index=False)
    print(f"\nSaved output_to_share/sensitivity_analysis.csv ({len(sa_all)} rows, {sa_all['sensitivity'].nunique()} specs)")
    return


if __name__ == "__main__":
    app.run()
