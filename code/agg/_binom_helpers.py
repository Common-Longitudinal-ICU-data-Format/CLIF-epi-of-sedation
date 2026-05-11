"""Confidence-interval helpers for cross-site aggregation.

Two helpers, both consuming SUMMARY STATISTICS (not raw vectors), because
cross-site scripts read aggregated per-site CSVs — never row-level data.
This matches the federation contract documented in `.dev/CLAUDE.md`.

  - `wilson_ci(k, n, alpha)` — Wilson 95% interval for a binomial
    proportion. Better small-n behavior than the normal approximation
    (no coverage drop near p = 0 or p = 1) so it works at both per-site
    and pooled-cohort grain.

  - `student_t_ci_from_summary(n, mean, sd, alpha)` — Student-t interval
    for a sample mean given only the summary stats `(n, mean, sd)`.
    Replaces the raw-vector `_ci()` helper that used to live duplicated
    in night_day_diff_mean_cross_site.py and night_day_diff_combined_by_icu_day.py.

Both return `(point, lo, hi)` triples; NaNs for `n < 1` (CI undefined).
"""

from __future__ import annotations

import math

from scipy import stats


def wilson_ci(
    k: int, n: int, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Wilson score interval for k successes out of n trials.

    Returns `(point, lo, hi)` where `point = k / n`. Coverage stays
    near the nominal 1−alpha even at p close to 0 or 1, unlike the
    Wald (normal-approx) interval. Equivalent to
    `statsmodels.stats.proportion.proportion_confint(method="wilson")`
    but written out here so we don't pull statsmodels into the agg path
    just for one helper.

    `n == 0` → all-NaN (undefined). `k == 0` or `k == n` are handled
    correctly by the closed form below (lo or hi pin to 0 or 1).
    """
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    z = float(stats.norm.ppf(1 - alpha / 2))
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return p, lo, hi


def student_t_ci_from_summary(
    n: int, mean: float, sd: float, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Student-t 1−alpha CI for a sample mean given summary statistics.

    `(point, lo, hi)` where `point = mean`. SE computed as `sd / sqrt(n)`;
    half-width uses `t.ppf(1 - alpha/2, df = n - 1)`.

    Edge cases:
      - `n <= 0` → all-NaN
      - `n == 1` → `(mean, mean, mean)` (CI undefined for a single observation)
      - `sd` NaN → `(mean, NaN, NaN)`
    """
    if n is None or n <= 0:
        return float("nan"), float("nan"), float("nan")
    if n == 1:
        return float(mean), float(mean), float(mean)
    if sd is None or (isinstance(sd, float) and math.isnan(sd)):
        return float(mean), float("nan"), float("nan")
    se = float(sd) / math.sqrt(n)
    crit = float(stats.t.ppf(1 - alpha / 2, df=n - 1))
    return float(mean), float(mean) - crit * se, float(mean) + crit * se


__all__ = ["wilson_ci", "student_t_ci_from_summary"]
