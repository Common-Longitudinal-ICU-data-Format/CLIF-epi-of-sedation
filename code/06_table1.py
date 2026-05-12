# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "duckdb>=1.4.1",
#     "pandas>=2.3.1",
#     "numpy>=1.26",
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
    logger = get_logger("epi_sedation.table1")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 06 Table One — federation-friendly long-format outputs

    Emits FOUR CSVs into `output_to_share/{site}/models/`:

      - `cohort_stats.csv` — site, n_hospitalizations, n_unique_patients
      - `table1_continuous.csv` — one row per continuous variable carrying
        n, n_missing, mean, sd, sum, sum_sq, min, max, median, q1, q3.
        `sum` and `sum_sq` are the lossless primitives for cross-site
        master-cohort mean/SD pooling at agg time.
      - `table1_categorical.csv` — one row per (variable, category)
        carrying n, n_missing, total_n, pct, denominator_unit. Per-stay
        and per-patient-day rows coexist via the denominator_unit flag.
      - `table1_histograms.csv` — long-format bin counts per non-normal
        variable, on shared bin edges (see code/_table1_schema.py). Lets
        the cross-site pooler compute master-cohort median/Q1/Q3 by
        summing bin counts across sites and inverse-CDF interpolating.

    No `tableone` dependency. Source-of-truth = cohort_meta_by_id.parquet
    (built in 04_covariates.py) which already carries every per-stay
    variable. sex_category is folded in from model_input_by_id_imvday
    (Day-1 row). The patient-day SBT-multiday rate uses the full-24h
    day-1..7 rows of model_input — same modeling-cohort definition the
    cross-site descriptive figures use.
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    from clifpy.utils import apply_outlier_handling
    from clifpy import Hospitalization, setup_logging
    import pandas as pd
    import numpy as np
    import duckdb
    import json

    from _table1_schema import (
        BIN_EDGES,
        BINARY_DISPLAY_LEVEL,
        CATEGORICAL_VARS_PER_PATIENT_DAY,
        CATEGORICAL_VARS_PER_STAY,
        CONTINUOUS_VARS,
    )

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    os.makedirs(f"output_to_share/{SITE_NAME}/models", exist_ok=True)
    # Per-site dual log files (pyCLIF integration guide rule 1).
    setup_logging(output_directory=f"output_to_share/{SITE_NAME}")
    logger.info(f"Site: {SITE_NAME}")
    return (
        BIN_EDGES,
        BINARY_DISPLAY_LEVEL,
        CATEGORICAL_VARS_PER_PATIENT_DAY,
        CATEGORICAL_VARS_PER_STAY,
        CONFIG_PATH,
        CONTINUOUS_VARS,
        SITE_NAME,
        apply_outlier_handling,
        duckdb,
        np,
        pd,
    )


@app.cell
def _(SITE_NAME, pd):
    # Eligibility cohort = Phase-4 modeling filter applied to model_input.
    # The same filter 08_models.py uses (line 78), so Table 1 matches
    # the population the model was fit on. Day-1 collapse to one row per
    # hospitalization happens after we have the eligibility hosp_id set.
    _model_input = pd.read_parquet(
        f"output/{SITE_NAME}/model_input_by_id_imvday.parquet"
    )
    _eligible = _model_input.loc[
        (_model_input["_nth_day"] > 0)
        & _model_input["sbt_done_next_day"].notna()
        & _model_input["success_extub_next_day"].notna()
    ]
    eligible_hosp_ids = sorted(_eligible["hospitalization_id"].unique().tolist())
    # Day-1 rows of the eligibility cohort — used to fold in `sex_category`
    # which is per-stay metadata not on cohort_meta_by_id today. (One row
    # per hospitalization; Day-1 sex_category equals stay-level sex.)
    day1_df = (
        _eligible.loc[_eligible["_nth_day"] == 1, ["hospitalization_id", "sex_category"]]
        .drop_duplicates("hospitalization_id")
    )
    logger.info(
        f"Eligibility cohort: {len(eligible_hosp_ids)} hospitalizations "
        f"({len(_eligible)} patient-days)"
    )
    return day1_df, eligible_hosp_ids


@app.cell
def _(SITE_NAME, day1_df, eligible_hosp_ids, pd):
    # Per-stay frame = one row per hospitalization with all the baseline
    # characteristics + per-stay outcome rollups. cohort_meta_by_id.parquet
    # already carries every variable we need (built in 04_covariates.py),
    # so this is a single read + filter + join with sex_category.
    _cm = pd.read_parquet(f"output/{SITE_NAME}/cohort_meta_by_id.parquet")
    _cm = _cm.loc[_cm["hospitalization_id"].isin(eligible_hosp_ids)].copy()
    table1_df = _cm.merge(day1_df, on="hospitalization_id", how="left")

    # Convert boolean per-stay outcomes to Yes/No so they read like clinical
    # paper conventions in the formatted output. Keeps the per-site
    # categorical CSV consistent across vars (Yes/No for binaries; named
    # levels for sex_category / icu_type / exit_mechanism).
    for _binary_col in ["ever_pressor", "sepsis_ase",
                        "successful_extubation", "ever_sbt_done_multiday"]:
        if _binary_col in table1_df.columns:
            table1_df[_binary_col] = table1_df[_binary_col].map(
                lambda v: "Yes" if v in (1, True) else ("No" if v in (0, False) else None)
            )

    logger.info(
        f"Table 1 frame: {len(table1_df)} hospitalizations × "
        f"{table1_df.shape[1]} columns"
    )
    return (table1_df,)


@app.cell
def _(CONFIG_PATH, apply_outlier_handling):
    hosp = Hospitalization.from_file(
        config_path=CONFIG_PATH,
        columns=['patient_id', 'hospitalization_id'],
    )
    apply_outlier_handling(hosp, outlier_config_path='config/outlier_config.yaml')
    hosp_df = hosp.df
    return (hosp_df,)


@app.cell
def _(SITE_NAME, eligible_hosp_ids, hosp_df, pd):
    # cohort_stats.csv (unchanged behavior — just simpler: no shift-level
    # join needed since we only count distinct hospitalizations + patients).
    _eligible_hosp = pd.DataFrame({"hospitalization_id": eligible_hosp_ids})
    _stats_df = _eligible_hosp.merge(
        hosp_df[["hospitalization_id", "patient_id"]],
        on="hospitalization_id", how="left",
    )
    n_hospitalizations = _stats_df["hospitalization_id"].nunique()
    n_unique_patients = _stats_df["patient_id"].nunique()
    pd.DataFrame({
        "site": [SITE_NAME],
        "n_hospitalizations": [n_hospitalizations],
        "n_unique_patients": [n_unique_patients],
    }).to_csv(f"output_to_share/{SITE_NAME}/models/cohort_stats.csv", index=False)
    logger.info(
        f"Cohort: {n_hospitalizations} hospitalizations from "
        f"{n_unique_patients} unique patients"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Continuous variables → table1_continuous.csv

    One row per variable. Carries display stats (mean/sd/median/q1/q3/min/max)
    AND lossless pooling primitives (n, sum, sum_sq) so the cross-site
    pooler can recompute master-cohort mean/SD via fixed-effects formulas
    without re-reading PHI.
    """)
    return


@app.cell
def _(CONTINUOUS_VARS, SITE_NAME, np, pd, table1_df):
    _rows = []
    for _var in CONTINUOUS_VARS:
        _vals = pd.to_numeric(table1_df[_var], errors="coerce")
        _ok = _vals.dropna().to_numpy()
        n = int(_ok.size)
        n_missing = int(_vals.isna().sum())
        if n == 0:
            mean = sd = sum_v = sum_sq = median = q1 = q3 = mn = mx = float("nan")
        else:
            mean = float(_ok.mean())
            sd = float(_ok.std(ddof=1)) if n > 1 else float("nan")
            sum_v = float(_ok.sum())
            sum_sq = float((_ok * _ok).sum())
            median = float(np.median(_ok))
            q1 = float(np.percentile(_ok, 25))
            q3 = float(np.percentile(_ok, 75))
            mn = float(_ok.min())
            mx = float(_ok.max())
        _rows.append({
            "variable": _var,
            "n": n, "n_missing": n_missing,
            "mean": mean, "sd": sd,
            "sum": sum_v, "sum_sq": sum_sq,
            "median": median, "q1": q1, "q3": q3,
            "min": mn, "max": mx,
        })
    table1_continuous = pd.DataFrame(_rows)
    _path = f"output_to_share/{SITE_NAME}/models/table1_continuous.csv"
    table1_continuous.to_csv(_path, index=False)
    logger.info(f"Saved {_path}  ({len(table1_continuous)} variables)")
    return (table1_continuous,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Categorical variables → table1_categorical.csv

    Per-stay rows (denominator = patients) + the patient-day SBT rate row
    (denominator = patient-days) coexist via the `denominator_unit` flag.

    Binary vars emit BOTH Yes and No rows so the master-cohort proportion
    is recoverable; the cross-site formatter applies BINARY_DISPLAY_LEVEL
    to suppress the redundant level.
    """)
    return


@app.cell
def _(CATEGORICAL_VARS_PER_STAY, SITE_NAME, pd, table1_df):
    # Per-stay categorical rows. value_counts(dropna=False) keeps NaN
    # visible in counts; we then strip the NaN row to a separate
    # n_missing column so the actual category rows don't contain NaN.
    _per_stay_rows = []
    for _var in CATEGORICAL_VARS_PER_STAY:
        if _var not in table1_df.columns:
            logger.info(f"  WARN: {_var} missing from cohort_meta_by_id; skipping")
            continue
        _counts = table1_df[_var].value_counts(dropna=False)
        _na_count = int(_counts.get(float("nan"), 0)) if any(
            isinstance(_k, float) and _k != _k for _k in _counts.index
        ) else int(table1_df[_var].isna().sum())
        # Drop NaN row; what's left is the named-level rows.
        _counts_clean = _counts[_counts.index.notna()]
        _total = int(_counts_clean.sum())
        for _cat, _cnt in _counts_clean.items():
            _per_stay_rows.append({
                "variable": _var,
                "category": str(_cat),
                "n": int(_cnt),
                "n_missing": _na_count,
                "total_n": _total,
                "pct": (100.0 * int(_cnt) / _total) if _total else float("nan"),
                "denominator_unit": "patients",
            })
    table1_categorical_per_stay = pd.DataFrame(_per_stay_rows)
    return (table1_categorical_per_stay,)


@app.cell
def _(SITE_NAME, pd):
    # Patient-day SBT-multiday rate. Numerator/denominator restricted to
    # the modeling cohort's full-24h day-1..7 rows — matches the descriptive
    # figures' `_is_full_24h_day & _nth_day BETWEEN 1 AND 7` filter.
    # `sbt_done_multiday_next_day` is the manuscript primary SBT outcome
    # (08_models.py:410).
    _model_input = pd.read_parquet(
        f"output/{SITE_NAME}/model_input_by_id_imvday.parquet"
    )
    _full24h = _model_input.loc[
        _model_input["_is_full_24h_day"]
        & _model_input["_nth_day"].between(1, 7)
    ]
    _sbt_col = "_sbt_done_multiday_today"
    _denom = int(len(_full24h))
    _numer_yes = int((_full24h[_sbt_col] == 1).sum())
    _n_missing = int(_full24h[_sbt_col].isna().sum())
    _per_pday_rows = [
        {
            "variable": "sbt_done_multiday_per_full24h_day",
            "category": "Yes",
            "n": _numer_yes,
            "n_missing": _n_missing,
            "total_n": _denom,
            "pct": (100.0 * _numer_yes / _denom) if _denom else float("nan"),
            "denominator_unit": "patient-days",
        },
        {
            "variable": "sbt_done_multiday_per_full24h_day",
            "category": "No",
            "n": _denom - _numer_yes - _n_missing,
            "n_missing": _n_missing,
            "total_n": _denom,
            "pct": (100.0 * (_denom - _numer_yes - _n_missing) / _denom)
                   if _denom else float("nan"),
            "denominator_unit": "patient-days",
        },
    ]
    table1_categorical_per_pday = pd.DataFrame(_per_pday_rows)
    logger.info(
        f"Patient-day SBT-multiday rate: {_numer_yes:,} / {_denom:,} "
        f"({100.0 * _numer_yes / max(_denom, 1):.1f}%, missing {_n_missing})"
    )
    return (table1_categorical_per_pday,)


@app.cell
def _(SITE_NAME, pd, table1_categorical_per_pday, table1_categorical_per_stay):
    table1_categorical = pd.concat(
        [table1_categorical_per_stay, table1_categorical_per_pday],
        ignore_index=True,
    )
    _path = f"output_to_share/{SITE_NAME}/models/table1_categorical.csv"
    table1_categorical.to_csv(_path, index=False)
    logger.info(f"Saved {_path}  ({len(table1_categorical)} rows)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Histograms → table1_histograms.csv

    Long-format bin counts per continuous variable on hardcoded shared
    bin edges (see code/_table1_schema.py). The cross-site pooler sums
    bin counts across sites and inverse-CDF-interpolates the master-cohort
    median/Q1/Q3 — exact for integer vars (1-unit bins), clinically
    precise (≤ 1 unit) for the continuous ones.
    """)
    return


@app.cell
def _(BIN_EDGES, SITE_NAME, np, pd, table1_df):
    # Clip values to [edges[0], edges[-1]] before binning so out-of-range
    # values (e.g., artifactual PF ratios > 800 from FiO2-coding errors) are
    # counted in the last bin rather than silently dropped. This keeps the
    # per-variable cohort N consistent with the continuous CSV's `n` and
    # ensures pooled-median computation is over the full eligibility cohort,
    # not a truncated subset. Clipping is a no-op for medians/Q1/Q3 because
    # those typically fall well inside the bin range; only extreme outliers
    # are affected and they cluster harmlessly in the last bin.
    _rows = []
    for _var, _edges in BIN_EDGES.items():
        _vals = pd.to_numeric(table1_df[_var], errors="coerce").dropna().to_numpy()
        if _vals.size == 0:
            continue
        _n_below = int((_vals < _edges[0]).sum())
        _n_above = int((_vals > _edges[-1]).sum())
        _vals_clipped = np.clip(_vals, _edges[0], _edges[-1])
        _counts, _ = np.histogram(_vals_clipped, bins=_edges)
        if _n_below + _n_above > 0:
            logger.info(
                f"  WARN: {_var}: clipped {_n_below} values below "
                f"{_edges[0]} and {_n_above} above {_edges[-1]} into the "
                f"boundary bins (likely upstream data-quality outliers; "
                f"median/Q1/Q3 unaffected if they fall inside the bin range)"
            )
        for _i, _c in enumerate(_counts):
            _rows.append({
                "variable": _var,
                "bin_left": float(_edges[_i]),
                "bin_right": float(_edges[_i + 1]),
                "count": int(_c),
            })
    table1_histograms = pd.DataFrame(_rows)
    _path = f"output_to_share/{SITE_NAME}/models/table1_histograms.csv"
    table1_histograms.to_csv(_path, index=False)
    logger.info(f"Saved {_path}  ({len(table1_histograms)} rows)")
    return


if __name__ == "__main__":
    app.run()
