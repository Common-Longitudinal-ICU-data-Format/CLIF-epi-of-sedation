"""Single source of truth for Table 1 variables, bin edges, and display rules.

Both the per-site Table 1 emitter (`code/06_table1.py`) and the cross-site
pooler (`code/agg/pool_table1.py`) import from here so the variable
inventory, hardcoded bin grids, and binary-display rules stay in lockstep.

The bin grids must be IDENTICAL across sites (MIMIC + UCMC + any future
site) — that's the prerequisite for valid bin-by-bin summation at agg
time. Edges are pre-agreed and clinical-range-bounded.
"""

from __future__ import annotations

import numpy as np


# ── Variable inventory ────────────────────────────────────────────────────

# All 7 continuous variables — one row each in table1_continuous.csv.
# Note: cci_score / sofa_1st24h / n_days_full_24h are integers but go
# through the same continuous-summary path (mean/SD computed for the
# pooler's lossless pooling primitive even though the display format
# is median [Q1, Q3]).
CONTINUOUS_VARS: list[str] = [
    "age",
    "weight_kg",          # admission weight (replaces bmi; height often missing)
    "cci_score",
    "sofa_1st24h",
    "pf_1st24h_min",
    "imv_dur_hrs",
    "n_days_full_24h",
]

# Reported as mean (SD) in the formatted Table 1.
NORMAL_VARS: list[str] = ["age"]

# Reported as median [Q1, Q3] in the formatted Table 1.
NONNORMAL_VARS: list[str] = [v for v in CONTINUOUS_VARS if v not in NORMAL_VARS]

# Categorical, per-stay denominator (one observation per hospitalization).
# `ever_sbt_done_multiday` uses the multi-day SBT definition that's the
# manuscript primary outcome (08_models.py:410); plain `sbt_done` is the
# OG definition still used as the cohort eligibility filter but not as a
# reported outcome.
CATEGORICAL_VARS_PER_STAY: list[str] = [
    "sex_category",
    "icu_type",
    "ever_pressor",
    "sepsis_ase",
    "exit_mechanism",
    "ever_sbt_done_multiday",  # NEW per-stay rollup (added in 04_covariates.py)
    "successful_extubation",   # already on cohort_meta_by_id; just not displayed today
    # In-hospital mortality (derived from discharge_category in 06_table1.py).
    # `discharge_category` carries the full disposition breakdown so reviewers
    # can reconstruct any alternative mortality definition post-hoc. Strict
    # (`died_in_hospital`) matches the existing `died_on_imv` predicate
    # (04_covariates.py:1292); composite (`died_or_hospice`) is the
    # sensitivity definition.
    "discharge_category",
    "died_in_hospital",
    "died_or_hospice",
]

# Categorical, per-patient-day denominator. Different denominator from the
# rest of Table 1 — the cross-site formatter notes this in the row label.
# Numerator/denominator restricted to the modeling cohort's full-24h
# day-1..7 rows (matches the descriptive figures from the prior session).
CATEGORICAL_VARS_PER_PATIENT_DAY: list[str] = [
    "sbt_done_multiday_per_full24h_day",
]

# Which level to show for binary categoricals; the other level is suppressed
# in the formatted Pooled output (it's redundant with N − Yes).
BINARY_DISPLAY_LEVEL: dict[str, str] = {
    "ever_pressor":                      "Yes",
    "sepsis_ase":                        "Yes",
    "ever_sbt_done_multiday":            "Yes",
    "successful_extubation":             "Yes",
    "sbt_done_multiday_per_full24h_day": "Yes",
    "died_in_hospital":                  "Yes",
    "died_or_hospice":                   "Yes",
}

# Display ordering for multi-level categoricals. Per-site CSV emits all
# levels regardless of order; the pooler uses this only to order rows
# in the formatted output.
CATEGORICAL_ORDER: dict[str, list[str]] = {
    "exit_mechanism": [
        "tracheostomy",
        "died_on_imv",
        "palliative_extubation",
        "failed_extubation",
        "successful_extubation",
        "discharge_on_imv",
        "unknown",
    ],
    "sex_category": ["Female", "Male"],
}


# ── Histogram bin edges ──────────────────────────────────────────────────
#
# Per-variable hardcoded edges chosen for clinical bounds + reporting
# precision. For integer variables, 1-unit bins → exact pooled median
# (the bin edge IS the median when cumulative count crosses N/2). For
# continuous variables, ~0.5–1 unit bin width → median precision well
# below the 1-decimal-place reporting precision of a Table 1.
#
# Out-of-range handling: np.histogram silently drops values outside
# the leftmost / rightmost edge. The per-site emitter logs a warning
# if any values fall outside the bin range so a cohort surprise (e.g.,
# a 250-day IMV stay) doesn't get silently lost.
#
# `imv_dur_hrs` uses adaptive bins — 1-hour bins up to 200h (clinical
# decision-relevant range), then 12-hour bins out to 5000h to capture
# the long tail of extreme IMV stays without exploding the bin count.

# Integer-valued variables — quantile interpolation should NOT linear-
# interpolate within a 1-unit bin (all values in [N, N+1) are exactly N
# for integer data). The pooler returns bin_left for these, matching what
# `np.percentile(values, q)` would return on the raw integer sample.
INTEGER_VARS: set[str] = {"cci_score", "sofa_1st24h", "n_days_full_24h"}


BIN_EDGES: dict[str, np.ndarray] = {
    "age":             np.arange(0, 120.5, 0.5),
    "weight_kg":       np.arange(0, 300.5, 0.5),
    "cci_score":       np.arange(0, 51, 1),
    "sofa_1st24h":     np.arange(0, 26, 1),
    "n_days_full_24h": np.arange(0, 201, 1),
    "pf_1st24h_min":   np.arange(0, 801, 1),
    "imv_dur_hrs":     np.concatenate([
        np.arange(0, 200, 1, dtype=float),
        np.arange(200, 5001, 12, dtype=float),
    ]),
}
