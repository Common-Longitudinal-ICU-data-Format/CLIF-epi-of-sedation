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

    # RCS (restricted cubic splines) variant of the sofa spec.
    # Wraps the 6 exposure variables in patsy's cr() transform for natural
    # cubic regression splines with df=4 (→ 3 basis columns per variable).
    # Non-exposure adjustment variables stay linear/categorical.
    # Used for marginal-effect plots (curves can bend to reveal dose-response
    # shape) but excluded from the wide comparison CSV — cr() basis coefficients
    # aren't human-interpretable as OR-per-unit. See plan_rcs_exposures.md memory.
    SOFA_RCS = (
        "{{outcome}} ~ "
        "cr(prop_dif, df=4) + cr(fenteq_dif, df=4) + cr(midazeq_dif, df=4) + "
        "cr(_prop_day, df=4) + cr(_fenteq_day, df=4) + cr(_midazeq_day, df=4) + "
        "age + C(sex_category) + C(icu_type) + cci_score + sofa_total"
    )

    COVARIATE_SPECS = [
        {'label': 'baseline', 'formula': BASELINE},
        {'label': 'daydose',  'formula': DAYDOSE},
        {'label': 'sofa',     'formula': SOFA},
        {'label': 'clinical', 'formula': CLINICAL},
        {'label': 'sofa_rcs', 'formula': SOFA_RCS},
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
def _(VAR_DISPLAY, cohort_merged_final, fitted, np, pd):
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
        [('_prop_day',    'Mean Daytime Propofol Rate (mg/hr)'),
         ('_fenteq_day',  'Mean Daytime Fentanyl Eq Rate (mcg/hr)'),
         ('_midazeq_day', 'Mean Daytime Midazolam Eq Rate (mg/hr)')],
        [('prop_dif',     'Day-to-Night Δ Propofol Rate (mg/hr)'),
         ('fenteq_dif',   'Day-to-Night Δ Fentanyl Eq Rate (mcg/hr)'),
         ('midazeq_dif',  'Day-to-Night Δ Midazolam Eq Rate (mg/hr)')],
    ]

    Y_LABEL = {
        'sbt_done_next_day': 'Probability of Passing SBT',
        'success_extub_next_day': 'Probability of Successful Extubation',
    }

    def _build_reference_row(df_scaled):
        """Median for numeric columns, mode for object/string columns.

        `include=['object', 'str']` is explicit to silence Pandas 4 deprecation
        warnings about object/str dtype overlap.
        """
        ref = df_scaled.median(numeric_only=True).to_dict()
        for col in df_scaled.select_dtypes(include=['object', 'str']).columns:
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
            f"output_to_share/figures/"
            f"marginal_effects_{_outcome_short}_{model_type}_{spec_label}.png"
        )
        fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
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
    PLOT_SPECS = ['sofa', 'sofa_rcs']
    for (_outcome, _mt), _spec_dict in fitted.items():
        for _spec_label in PLOT_SPECS:
            if _spec_label in _spec_dict:
                plot_marginal_effects(
                    _spec_dict[_spec_label], _outcome, _mt, _spec_label
                )
    return


if __name__ == "__main__":
    app.run()
