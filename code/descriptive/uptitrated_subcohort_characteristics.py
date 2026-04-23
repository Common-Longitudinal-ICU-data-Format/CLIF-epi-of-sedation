"""Table 1-style comparison: up-titrated vs stable patient-days.

Defines an "up-titrated patient-day" as one where ANY of:
  - prop_dif_kgmin > 10 mcg/kg/min   (weight-adjusted)
  - fenteq_dif     > 25 mcg/hr
  - midazeq_dif    >  1 mg/hr

Everything else is "stable/decreased". Outputs a CSV comparing the two groups
on demographics, severity, daytime sedation, and next-day outcomes.

Usage:
    uv run python code/descriptive/uptitrated_subcohort_characteristics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import tableone

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    TABLES_DIR,
    THRESHOLDS,
    ensure_dirs,
    load_analytical,
    prepare_diffs,
)


GROUP_LABEL = {True: "Up-titrated", False: "Stable/decreased"}


def main() -> None:
    df = prepare_diffs(load_analytical())

    above_any = False
    for drug, col in DIFF_COLS.items():
        above_any = above_any | (df[col] > THRESHOLDS[drug])
    df["_uptitrated"] = above_any.map(GROUP_LABEL)

    n_up = int((df["_uptitrated"] == GROUP_LABEL[True]).sum())
    n_st = int((df["_uptitrated"] == GROUP_LABEL[False]).sum())
    print(f"Up-titrated patient-days:   {n_up:,}")
    print(f"Stable/decreased patient-days: {n_st:,}")

    # Binary outcomes rendered as "Yes" so only one row per var shows in the
    # tableone output (follows the 06_table1.py pattern).
    for c in ("_sbt_done_today", "_success_extub_today",
             "sbt_done_next_day", "success_extub_next_day"):
        if c in df.columns:
            df[c] = df[c].map({0: "No", 1: "Yes"})

    continuous_vars = [
        "age",
        "_nth_day",
        "sofa_total",
        "cci_score",
        "elix_score",
        "_prop_day",
        "_fenteq_day",
        "_midazeq_day",
    ]
    # Day counts, severity scores, and dose rates are right-skewed — report
    # median [IQR] rather than mean (SD).
    nonnormal_vars = [
        "_nth_day",
        "sofa_total",
        "cci_score",
        "elix_score",
        "_prop_day",
        "_fenteq_day",
        "_midazeq_day",
    ]
    categorical_vars = [
        "sex_category",
        "icu_type",
        "_sbt_done_today",
        "_success_extub_today",
        "sbt_done_next_day",
        "success_extub_next_day",
    ]
    continuous_vars = [c for c in continuous_vars if c in df.columns]
    nonnormal_vars = [c for c in nonnormal_vars if c in df.columns]
    categorical_vars = [c for c in categorical_vars if c in df.columns]

    order_by_group = [GROUP_LABEL[True], GROUP_LABEL[False]]
    t1 = tableone.TableOne(
        data=df,
        continuous=continuous_vars,
        categorical=categorical_vars,
        nonnormal=nonnormal_vars,
        groupby="_uptitrated",
        order={"_uptitrated": order_by_group},
        pval=True,
    )

    ensure_dirs()
    path = f"{TABLES_DIR}/uptitrated_subcohort_characteristics.csv"
    # tableone's .tableone DataFrame has a 2-level MultiIndex on columns
    # ("Grouped by _uptitrated" / "Missing|Overall|...") that makes CSVs awkward
    # for downstream consumers. Flatten to the inner level (same info, single
    # header row) so 09_report.py can load via pd.read_csv without tweaks.
    out_df = t1.tableone.copy()
    if isinstance(out_df.columns, pd.MultiIndex):
        out_df.columns = out_df.columns.get_level_values(-1)
    out_df.to_csv(path)
    print(t1.tabulate(tablefmt="simple"))
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
