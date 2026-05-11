"""Fixed-effects pooling primitives for Table 1.

Three helpers — all consuming SUMMARY STATISTICS (no raw vectors, no
PHI), matching the federation contract documented in `.dev/CLAUDE.md`:

  - `pooled_mean_sd_from_summary(rows)` — exact pooled mean and SD across
    sites given each site's `(n, sum, sum_sq)`. Replaces the lossy
    Hansen/Snedecor formula that parsed pre-formatted Table 1 strings.

  - `pooled_quantile_from_histograms(rows, q)` — pooled cohort quantile
    via inverse-CDF interpolation on summed bin counts. Works because
    every site emits histograms on the SAME pre-agreed bin edges (see
    `code/_table1_schema.py::BIN_EDGES`); summing bin counts is then
    just `groupby(bin_left, bin_right).sum()`. Median = quantile(0.5),
    Q1 = quantile(0.25), Q3 = quantile(0.75).

  - `pooled_categorical_counts(rows)` — sums per-category counts across
    sites and recomputes pct against the pooled `total_n`. Handles the
    case where ICU type levels differ between sites by treating missing
    categories as 0 at that site.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


def pooled_mean_sd_from_summary(
    rows: Iterable[dict],
) -> tuple[float, float, int]:
    """Pooled mean + sample SD across sites given per-site summary stats.

    Each row in `rows` must carry `n` (int), `sum` (float), `sum_sq` (float).
    Returns `(pooled_mean, pooled_sd, pooled_n)`. NaN-coerced for `n == 0`;
    pooled_sd = NaN if total `n < 2`.

    Fixed-effects formula — algebraically equivalent to recomputing
    mean/SD on the concatenated raw vectors:

        pooled_n    = Σ n
        pooled_mean = Σ sum / Σ n
        pooled_var  = (Σ sum_sq − (Σ sum)² / Σ n) / (Σ n − 1)

    The within/between decomposition (Hansen/Snedecor) used by the legacy
    pooler is mathematically identical when computed from `(n, sum, sum_sq)`
    — this version is just simpler and bypasses the formatted-string parser.
    """
    rows = [r for r in rows if r is not None]
    if not rows:
        return float("nan"), float("nan"), 0
    n_total = int(sum(int(r["n"]) for r in rows))
    if n_total == 0:
        return float("nan"), float("nan"), 0
    s = float(sum(float(r["sum"]) for r in rows))
    s2 = float(sum(float(r["sum_sq"]) for r in rows))
    mean = s / n_total
    if n_total < 2:
        return mean, float("nan"), n_total
    var = max(0.0, (s2 - s * s / n_total) / (n_total - 1))
    return mean, math.sqrt(var), n_total


def pooled_quantile_from_histograms(
    hist_rows: pd.DataFrame, q: float, integer: bool = False,
) -> float:
    """Pooled cohort quantile via inverse-CDF on summed histogram counts.

    `hist_rows` is a long-format DataFrame from one or more sites with
    columns `bin_left, bin_right, count`. Assumes bin edges are
    site-aligned — summed bin counts have well-defined meaning. Caller
    is responsible for filtering to a single `variable` before passing.

    `integer=True` switches to integer-aware mode: returns `bin_left` of
    the containing bin (no within-bin interpolation). Use this for
    integer-valued variables (e.g., `cci_score`, `sofa_1st24h`) where
    all values in `[N, N+1)` are exactly `N` — linear interpolation
    would overshoot. Default `False` uses linear within-bin
    interpolation (correct for continuous variables binned finely).

    Algorithm:
      1. Group by (bin_left, bin_right), sum counts across sites.
      2. Compute cumulative counts on the sorted bin sequence.
      3. Find bin i where cum[i-1] < N·q ≤ cum[i].
      4. Continuous: `quantile = bin_left[i] + ((N·q − cum[i-1]) / count[i]) · (bin_right[i] − bin_left[i])`.
      5. Integer: `quantile = bin_left[i]`.

    Returns NaN if total count is 0 or `q` is out of range.
    """
    if hist_rows.empty or not (0.0 <= q <= 1.0):
        return float("nan")
    grouped = (
        hist_rows.groupby(["bin_left", "bin_right"], as_index=False)["count"]
        .sum()
        .sort_values("bin_left")
        .reset_index(drop=True)
    )
    total = int(grouped["count"].sum())
    if total == 0:
        return float("nan")
    target = q * total
    cum = 0
    for _, row in grouped.iterrows():
        c = int(row["count"])
        if c == 0:
            continue
        new_cum = cum + c
        if new_cum >= target:
            bl = float(row["bin_left"])
            if integer:
                return bl
            br = float(row["bin_right"])
            frac_into_bin = (target - cum) / c if c > 0 else 0.0
            return bl + frac_into_bin * (br - bl)
        cum = new_cum
    # Should never reach here unless rounding; return rightmost edge.
    return float(grouped.iloc[-1]["bin_right"])


def pooled_categorical_counts(
    cat_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Sum per-category counts across sites; recompute pct vs pooled total_n.

    `cat_rows` is a long-format DataFrame with columns `variable, category,
    n, total_n`. May span multiple sites; sums per `(variable, category)`.
    Caller is responsible for filtering to a single variable if desired.

    Returns a frame with `(variable, category, n_pooled, total_n_pooled,
    pct_pooled)`. ICU-type levels that exist at one site but not another
    are handled naturally — the absent site contributes 0 to that level
    (since the per-site CSV only emits rows for observed levels).

    `total_n_pooled` is the sum of per-site `total_n` values divided by
    the number of distinct categories per site (since `total_n` is
    replicated across rows of the same variable in the per-site CSV).
    To avoid double-counting, we recompute `total_n_pooled` per variable
    as `Σ_site (first total_n for that variable)`.
    """
    if cat_rows.empty:
        return cat_rows.assign(n_pooled=[], total_n_pooled=[], pct_pooled=[])

    # First, get the per-site total_n once per variable (it's replicated
    # across rows of the same variable in each per-site CSV). Then sum
    # those one-per-site values to get pooled total_n.
    per_var_totals = (
        cat_rows.drop_duplicates(["variable", "_site", "total_n"])
        .groupby("variable", as_index=False)["total_n"].sum()
        .rename(columns={"total_n": "total_n_pooled"})
    )

    grouped = (
        cat_rows.groupby(["variable", "category"], as_index=False)["n"].sum()
        .rename(columns={"n": "n_pooled"})
    )
    out = grouped.merge(per_var_totals, on="variable", how="left")
    out["pct_pooled"] = np.where(
        out["total_n_pooled"] > 0,
        100.0 * out["n_pooled"] / out["total_n_pooled"],
        np.nan,
    )
    return out


__all__ = [
    "pooled_mean_sd_from_summary",
    "pooled_quantile_from_histograms",
    "pooled_categorical_counts",
]
