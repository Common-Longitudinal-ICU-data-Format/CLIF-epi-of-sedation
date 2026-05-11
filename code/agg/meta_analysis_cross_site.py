"""Cross-site meta-analysis: DerSimonian-Laird random-effects pooling.

Reads each site's `output_to_share/{site}/models/models_coeffs.csv` and
produces inverse-variance pooled effect estimates for every configured
(outcome × model_type × spec × predictor) combination. Operates on
SUMMARY STATISTICS only (log_or + se + n) — never touches row-level
patient data. This is the federation-clean entry point for cross-site
inferential pooling.

This script is DATA-ONLY now: it writes `output_to_agg/meta_pooled.csv`
and exits. The pooled estimates are rendered alongside per-site dots
inside the existing forest scripts (forest_night_day_cross_site.py,
forest_daytime_cross_site.py, forest_sbt_sensitivity_cross_site.py),
which import `dl_pool` / `build_pooled_table` / `_stack_per_site_coeffs`
directly from this module so they don't depend on the CSV being written
first (the Makefile glob runs scripts alphabetically, which would put
forest_* before meta_analysis_*).

Pooling math: random-effects DerSimonian-Laird.
  - Fixed-effect weights:   w_i = 1 / σ²_i
  - Q (heterogeneity):      Q = Σ w_i × (β_i − β̂_FE)²
  - C statistic:            C = Σw_i − Σw_i² / Σw_i
  - Between-study variance: τ̂² = max(0, (Q − df) / C)
  - I² statistic:           I² = max(0, (Q − df) / Q) × 100
  - Random-effects weights: w_i,RE = 1 / (σ²_i + τ̂²)
  - Pooled estimate:        β̂_RE = Σ w_i,RE × β_i / Σ w_i,RE
  - Pooled SE:              SE_RE = sqrt(1 / Σ w_i,RE)

Two presentations pooled side-by-side:
  - **per_unit**  — log_or_per_unit + se_per_unit (uniform across sites
    by design; matches a fixed clinical-unit shift like prop +10
    mcg/kg/min). Manuscript primary.
  - **p10_p90**   — log_or_p10_p90 + se_p10_p90 (site-specific cohort
    distribution; "10th→90th percentile shift in one's own site").
    Sensitivity sibling.

Outputs:
  - output_to_agg/meta_pooled.csv
      One row per (outcome, model_type, spec, predictor, presentation).
      Columns: pooled_log_or, pooled_se, pooled_ci_lo, pooled_ci_hi,
               pooled_or, pooled_or_lo, pooled_or_hi, tau2, Q, Q_pval,
               I2, n_studies, n_obs_total, n_events_total.

Usage:
    uv run python code/agg/meta_analysis_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.meta_analysis")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    SHARE_ROOT,
    list_sites,
    load_site_models_coeffs,
    save_agg_csv,
)


# ── Pool scope (matches the manuscript primary spec set) ──────────────────
# Each tuple: (outcome_key, model_type, display_label).
POOL_OUTCOMES: list[tuple[str, str, str]] = [
    ("sbt_done_multiday_next_day", "gee",        "SBT delivered (multiday)"),
    ("success_extub_next_day",     "logit_asym", "Successful extubation"),
]

# Two manuscript linear specs.
POOL_SPECS: list[str] = ["daydose", "daydose_physio"]

# 6 exposure predictors: 3 night-day diffs + 3 daytime continuous rates.
POOL_PREDICTORS: list[tuple[str, str]] = [
    ("prop_dif_mcg_kg_min",  "Δ propofol (per 10 mcg/kg/min)"),
    ("fenteq_dif_mcg_hr",    "Δ fentanyl eq (per 25 mcg/hr)"),
    ("midazeq_dif_mg_hr",    "Δ midazolam eq (per 1 mg/hr)"),
    ("_prop_day_mcg_kg_min", "Daytime propofol (per 10 mcg/kg/min)"),
    ("_fenteq_day_mcg_hr",   "Daytime fentanyl eq (per 25 mcg/hr)"),
    ("_midazeq_day_mg_hr",   "Daytime midazolam eq (per 1 mg/hr)"),
]

# Which presentation (per-unit / per-percentile) maps to which CSV columns.
# Both pooled with the same DL machinery; figures default to per_unit.
PRESENTATIONS: dict[str, dict[str, str]] = {
    "per_unit": {
        "log_or": "log_or_per_unit",
        "se":     "se_per_unit",
        "or":     "or_per_unit",
        "or_lo":  "or_per_unit_lo",
        "or_hi":  "or_per_unit_hi",
    },
    "p10_p90": {
        "log_or": "log_or_p10_p90",
        "se":     "se_p10_p90",
        "or":     "or_p10_p90",
        "or_lo":  "or_p10_p90_lo",
        "or_hi":  "or_p10_p90_hi",
    },
}

OUTCOME_SHORT: dict[str, str] = {
    "sbt_done_multiday_next_day": "sbt_done_multiday",
    "success_extub_next_day":     "success_extub",
    "success_extub_v2_next_day":  "success_extub_v2",
    "sbt_done_v2_next_day":       "sbt_done_v2",
    "sbt_done_prefix_next_day":   "sbt_done_prefix",
    "sbt_done_subira_next_day":   "sbt_done_subira",
    "sbt_done_abc_next_day":      "sbt_done_abc",
}


# ── DerSimonian-Laird pooler ──────────────────────────────────────────────
def dl_pool(effects: np.ndarray, variances: np.ndarray) -> dict:
    """Random-effects DL pool. Returns NaN-only dict if <2 valid studies.

    Filters out NaN and non-positive variance rows before pooling.
    """
    eff = np.asarray(effects, dtype=float)
    var = np.asarray(variances, dtype=float)
    mask = np.isfinite(eff) & np.isfinite(var) & (var > 0)
    eff = eff[mask]
    var = var[mask]
    n = len(eff)
    if n < 2:
        return {
            "pooled_log_or": np.nan, "pooled_se": np.nan,
            "tau2": np.nan, "Q": np.nan, "Q_pval": np.nan, "I2": np.nan,
            "n_studies": n,
        }
    w_fe = 1.0 / var
    beta_fe = float(np.sum(w_fe * eff) / np.sum(w_fe))
    Q = float(np.sum(w_fe * (eff - beta_fe) ** 2))
    df = n - 1
    Q_pval = float(1 - stats.chi2.cdf(Q, df)) if df > 0 else np.nan
    C = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2 = float(max(0.0, (Q - df) / C)) if C > 0 else 0.0
    I2 = float(max(0.0, (Q - df) / Q) * 100) if Q > 0 else 0.0
    w_re = 1.0 / (var + tau2)
    beta_re = float(np.sum(w_re * eff) / np.sum(w_re))
    se_re = float(np.sqrt(1.0 / np.sum(w_re)))
    return {
        "pooled_log_or": beta_re,
        "pooled_se": se_re,
        "tau2": tau2,
        "Q": Q,
        "Q_pval": Q_pval,
        "I2": I2,
        "n_studies": n,
    }


def _stack_per_site_coeffs() -> pd.DataFrame:
    """Concat each site's models_coeffs.csv (exposure rows only) with site col."""
    sites = list_sites()
    if not sites:
        logger.info("No sites under output_to_share/. Nothing to pool.")
        return pd.DataFrame()
    logger.info(f"Discovered sites: {sites}")
    frames = []
    for s in sites:
        path = SHARE_ROOT / s / "models" / "models_coeffs.csv"
        if not path.exists():
            logger.info(f"  SKIP {s}: {path} missing")
            continue
        df = load_site_models_coeffs(s)
        df = df[df["row_type"] == "exposure"].copy()
        df["site"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Pooled-results table builder ──────────────────────────────────────────
def build_pooled_table(df_all: pd.DataFrame) -> pd.DataFrame:
    """Run DL pool over every (outcome × model_type × spec × predictor ×
    presentation) cell. Returns long-format DataFrame."""
    rows = []
    for outcome, model_type, _ in POOL_OUTCOMES:
        for spec in POOL_SPECS:
            for predictor, _ in POOL_PREDICTORS:
                cell = df_all[
                    (df_all["outcome"] == outcome)
                    & (df_all["model_type"] == model_type)
                    & (df_all["spec"] == spec)
                    & (df_all["predictor"] == predictor)
                ]
                n_obs_total = int(cell["n_obs"].sum()) if not cell.empty else 0
                n_events_total = int(cell["n_events"].sum()) if not cell.empty else 0
                for pres_name, cols in PRESENTATIONS.items():
                    res = dl_pool(
                        cell[cols["log_or"]].to_numpy(),
                        cell[cols["se"]].to_numpy() ** 2,
                    )
                    pooled_lo = (res["pooled_log_or"] - 1.96 * res["pooled_se"])
                    pooled_hi = (res["pooled_log_or"] + 1.96 * res["pooled_se"])
                    rows.append({
                        "outcome": outcome,
                        "model_type": model_type,
                        "spec": spec,
                        "predictor": predictor,
                        "presentation": pres_name,
                        "pooled_log_or": res["pooled_log_or"],
                        "pooled_se": res["pooled_se"],
                        "pooled_ci_lo": pooled_lo,
                        "pooled_ci_hi": pooled_hi,
                        "pooled_or": np.exp(res["pooled_log_or"]),
                        "pooled_or_lo": np.exp(pooled_lo),
                        "pooled_or_hi": np.exp(pooled_hi),
                        "tau2": res["tau2"],
                        "Q": res["Q"],
                        "Q_pval": res["Q_pval"],
                        "I2": res["I2"],
                        "n_studies": res["n_studies"],
                        "n_obs_total": n_obs_total,
                        "n_events_total": n_events_total,
                    })
    return pd.DataFrame(rows)



# ── Driver ────────────────────────────────────────────────────────────────
def main() -> None:
    df_all = _stack_per_site_coeffs()
    if df_all.empty:
        logger.info("No models_coeffs.csv files found; nothing to pool.")
        return

    # Validate that each site has the expected (outcome, mt, spec, predictor)
    # cells. Missing combos are warned but do not abort — DL handles k<2 gracefully.
    needed = {
        (o, mt, sp, pr)
        for o, mt, _ in POOL_OUTCOMES
        for sp in POOL_SPECS
        for pr, _ in POOL_PREDICTORS
    }
    have = set(map(tuple, df_all[["outcome", "model_type", "spec", "predictor"]].values))
    missing = needed - have
    if missing:
        logger.info(
            f"  WARN: {len(missing)} (outcome, model_type, spec, predictor) "
            "combos absent from at least one site's models_coeffs.csv."
        )

    pooled = build_pooled_table(df_all)
    save_agg_csv(pooled, "meta_pooled")
    # NO figure rendering here. The pooled estimates are drawn as the
    # bottom "Pooled" row in each panel of forest_night_day_cross_site.py,
    # forest_daytime_cross_site.py, and forest_sbt_sensitivity_cross_site.py.
    # Those scripts import dl_pool / build_pooled_table / _stack_per_site_coeffs
    # from THIS module so they can render even before meta_pooled.csv is
    # (re)written.


if __name__ == "__main__":
    main()
