"""Parameterization ablation for the MIMIC daytime-propofol sign flip.

Fits the production extubation logit FOUR times on a single locked MIMIC
cohort, ablating two axes for propofol:
  - rate (per-time) vs amount (total over shift)
  - /kg-normalized vs unweighted

The amount-vs-rate axis is a sanity check: amount = rate × positive constant
(720 for propofol, 12 for fent/midaz), so coefficient signs MUST match.
The /kg axis is where the exposure distribution actually changes — body
habitus enters the variable when /kg is applied, and may flip signs.

Decision rules:
  - Within {kg_rate, kg_amount}: signs MUST match (linear scaling).
  - Within {mg_rate, mg_amount}: signs MUST match (linear scaling).
  - Across {kg_*, mg_*}: if signs differ, /kg normalization is the trigger
    for the historical sign flip and body-habitus adjustment (sofa_weight
    / sofa_bmi specs in 08_models.py) is the right intervention.

Output: a coefficient table printed to stdout. No file writes.

Usage:
    uv run python dev/parameterization_ablation.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SITE = "mimic"
PARQUET = f"output/{SITE}/modeling_dataset.parquet"

COVARIATES = (
    "ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + "
    "nee_7am + nee_7pm + age + _nth_day + sofa_total + cci_score + "
    "C(sex_category) + C(icu_type)"
)
# Fent/midaz held at production rate parameterization across all four fits;
# only propofol is ablated. (Fent/midaz aren't /kg-normalized so the
# /kg axis doesn't apply to them.)
FIXED_OTHER_DRUGS = (
    "_midazeq_day_mg_hr + _fenteq_day_mcg_hr + "
    "midazeq_dif_mg_hr + fenteq_dif_mcg_hr"
)

SPECS = {
    'kg_rate':   "_prop_day_mcg_kg_min + prop_dif_mcg_kg_min",
    'kg_amount': "_prop_day_mcg_kg + prop_dif_mcg_kg",
    'mg_rate':   "_prop_day_mg_hr + prop_dif_mg_hr",
    'mg_amount': "_prop_day_mg + prop_dif_mg",
}
PROP_DAY_TERM = {
    'kg_rate':   '_prop_day_mcg_kg_min',
    'kg_amount': '_prop_day_mcg_kg',
    'mg_rate':   '_prop_day_mg_hr',
    'mg_amount': '_prop_day_mg',
}
PROP_DIF_TERM = {
    'kg_rate':   'prop_dif_mcg_kg_min',
    'kg_amount': 'prop_dif_mcg_kg',
    'mg_rate':   'prop_dif_mg_hr',
    'mg_amount': 'prop_dif_mg',
}


def reconstruct_unweighted(df: pd.DataFrame) -> pd.DataFrame:
    """Add mg-unit columns by un-doing the /kg normalization with admission weight.

    Approximate reconstruction — pre-Phase-2 production used the weight
    attached at the unit-conversion step (per-admin ASOF), not the
    hospitalization-level admission weight used here. For the qualitative
    question "does removing /kg flip the sign?" the approximation suffices.
    """
    w = df['weight_kg']
    df['_prop_day_mg']      = df['_prop_day_mcg_kg']       * w / 1000.0
    df['_prop_night_mg']    = df['_prop_night_mcg_kg']     * w / 1000.0
    df['_prop_day_mg_hr']   = df['_prop_day_mcg_kg_min']   * w / 1000.0 * 60.0
    df['_prop_night_mg_hr'] = df['_prop_night_mcg_kg_min'] * w / 1000.0 * 60.0
    df['prop_dif_mg']       = df['_prop_night_mg']    - df['_prop_day_mg']
    df['prop_dif_mg_hr']    = df['_prop_night_mg_hr'] - df['_prop_day_mg_hr']
    return df


def lock_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing any value used by ANY spec — all four fits then run
    on the same N, so coefficient comparisons aren't confounded by N drift."""
    locked_cols = [
        '_prop_day_mcg_kg_min', '_prop_night_mcg_kg_min', 'prop_dif_mcg_kg_min',
        '_prop_day_mcg_kg', '_prop_night_mcg_kg', 'prop_dif_mcg_kg',
        '_prop_day_mg', '_prop_night_mg', 'prop_dif_mg',
        '_prop_day_mg_hr', '_prop_night_mg_hr', 'prop_dif_mg_hr',
        '_midazeq_day_mg_hr', '_fenteq_day_mcg_hr',
        'midazeq_dif_mg_hr', 'fenteq_dif_mcg_hr',
        'ph_level_7am', 'ph_level_7pm', 'pf_level_7am', 'pf_level_7pm',
        'nee_7am', 'nee_7pm', 'age', '_nth_day', 'sofa_total', 'cci_score',
        'sex_category', 'icu_type',
        'success_extub_next_day', 'hospitalization_id',
        'weight_kg',
    ]
    return df.dropna(subset=locked_cols).reset_index(drop=True)


def fit_logit(formula: str, data: pd.DataFrame):
    return smf.logit(formula=formula, data=data).fit(
        cov_type='cluster',
        cov_kwds={'groups': data['hospitalization_id']},
        disp=False,
    )


def or_10_to_90(beta: float, x10: float, x90: float) -> float:
    return float(np.exp(beta * (x90 - x10)))


def main():
    df = pd.read_parquet(PARQUET)
    print(f"Loaded {len(df):,} rows from {PARQUET}")

    df = reconstruct_unweighted(df)
    locked = lock_cohort(df)
    n_hosp = locked['hospitalization_id'].nunique()
    print(f"Locked cohort N: {len(locked):,} rows ({n_hosp:,} hospitalizations)\n")

    fits = {}
    for label, prop_terms in SPECS.items():
        formula = (
            f"success_extub_next_day ~ {prop_terms} + "
            f"{FIXED_OTHER_DRUGS} + {COVARIATES}"
        )
        fits[label] = fit_logit(formula, locked)
        print(f"  fit OK: {label}")
    print()

    rows = []
    for label, result in fits.items():
        day_term = PROP_DAY_TERM[label]
        dif_term = PROP_DIF_TERM[label]
        beta_day = result.params.get(day_term, np.nan)
        beta_dif = result.params.get(dif_term, np.nan)
        p_day = result.pvalues.get(day_term, np.nan)
        p_dif = result.pvalues.get(dif_term, np.nan)
        x10_day, x90_day = np.percentile(locked[day_term], [10, 90])
        x10_dif, x90_dif = np.percentile(locked[dif_term], [10, 90])
        rows.append({
            'spec':           label,
            'beta_day':       beta_day,
            'p_day':          p_day,
            'OR_10_90_day':   or_10_to_90(beta_day, x10_day, x90_day),
            'sign_day':       '+' if beta_day > 0 else '-',
            'beta_dif':       beta_dif,
            'p_dif':          p_dif,
            'OR_10_90_dif':   or_10_to_90(beta_dif, x10_dif, x90_dif),
            'sign_dif':       '+' if beta_dif > 0 else '-',
        })
    table = pd.DataFrame(rows).set_index('spec')

    print("=== Daytime-prop + Day-Night-Diff coefficients across parameterizations ===\n")
    fmt = {
        'beta_day':     lambda x: f"{x:>+11.4g}",
        'p_day':        lambda x: f"{x:>7.3f}",
        'OR_10_90_day': lambda x: f"{x:>6.3f}",
        'beta_dif':     lambda x: f"{x:>+11.4g}",
        'p_dif':        lambda x: f"{x:>7.3f}",
        'OR_10_90_dif': lambda x: f"{x:>6.3f}",
    }
    print(table.to_string(formatters=fmt))

    print("\n=== Sanity checks (linear-scaling rule) ===")
    for pair in [('kg_rate', 'kg_amount'), ('mg_rate', 'mg_amount')]:
        a, b = pair
        for term in ['day', 'dif']:
            sa, sb = table.loc[a, f'sign_{term}'], table.loc[b, f'sign_{term}']
            status = 'OK' if sa == sb else 'MISMATCH (numerical instability?)'
            print(f"  {a} vs {b} on prop_{term}: {sa} vs {sb}  →  {status}")

    print("\n=== /kg ablation (the substantive question) ===")
    for term in ['day', 'dif']:
        skg = table.loc['kg_rate', f'sign_{term}']
        smg = table.loc['mg_rate', f'sign_{term}']
        if skg == smg:
            verdict = 'NO FLIP — /kg is not the trigger for this term'
        else:
            verdict = 'FLIP — removing /kg changed the sign for this term'
        print(f"  kg_rate vs mg_rate on prop_{term}: {skg} vs {smg}  →  {verdict}")


if __name__ == "__main__":
    main()
