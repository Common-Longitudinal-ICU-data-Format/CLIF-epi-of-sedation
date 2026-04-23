"""Mean night-minus-day dose rate ± 95% CI across ICU-day bins.

Tests the clinical hypothesis that the day-night diff is largest early in the
ICU stay (when patients are sickest / most unsettled) and shrinks as they
stabilize. X-axis: _nth_day binned to 1..7, '8+'. Y-axis: mean(`*_dif`) with
t-interval 95% CI. N per bin annotated.

Usage:
    uv run python code/descriptive/night_day_diff_mean_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    apply_style,
    cap_day,
    load_analytical,
    prepare_diffs,
    save_fig,
)


def _ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using Student-t interval. Assumes n > 1."""
    n = len(values)
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf((1 + conf) / 2, df=n - 1))
    return mean, mean - crit * se, mean + crit * se


def main() -> None:
    apply_style()
    df = cap_day(prepare_diffs(load_analytical()), max_day=7)
    bins = list(df["_nth_day_bin"].cat.categories)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        means, lows, highs, ns = [], [], [], []
        for b in bins:
            vals = df.loc[df["_nth_day_bin"] == b, col].dropna().to_numpy()
            m, lo, hi = _ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
            ns.append(len(vals))

        x = np.arange(len(bins))
        color = DRUG_COLORS[drug]
        ax.plot(x, means, marker="o", color=color, linewidth=2)
        ax.fill_between(x, lows, highs, color=color, alpha=0.25, label="95% CI")
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)

        for xi, (m, n) in enumerate(zip(means, ns)):
            ax.annotate(f"n={n:,}", xy=(xi, m), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"Mean {DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_title(DRUG_LABELS[drug])
        ax.legend(loc="upper right")

    fig.suptitle(
        "Night-minus-day dose rate by ICU day (mean ± 95% CI)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_fig(fig, "night_day_diff_mean_by_icu_day")


if __name__ == "__main__":
    main()
