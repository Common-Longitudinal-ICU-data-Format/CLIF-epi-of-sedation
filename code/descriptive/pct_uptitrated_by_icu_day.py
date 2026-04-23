"""Up-titration prevalence by ICU day.

Three panels (propofol / fent-eq / midaz-eq). X = ICU-day bin (1..7, '8+'),
Y = fraction of patient-days in that bin with diff > per-drug threshold.
This is the trajectory-meets-categorical figure: if the hypothesis holds,
bars should step down as ICU day increases.

Usage:
    uv run python code/descriptive/pct_uptitrated_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.proportion import proportion_confint

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUGS,
    THRESHOLDS,
    apply_style,
    cap_day,
    load_analytical,
    save_fig,
    threshold_label,
)


def main() -> None:
    apply_style()
    df = cap_day(load_analytical(), max_day=7)
    bins = list(df["_nth_day_bin"].cat.categories)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        pcts, los, his, ns = [], [], [], []
        for b in bins:
            series = df.loc[df["_nth_day_bin"] == b, col].dropna()
            total = len(series)
            above = int((series > THRESHOLDS[drug]).sum())
            if total == 0:
                pcts.append(0.0)
                los.append(0.0)
                his.append(0.0)
                ns.append(0)
                continue
            lo, hi = proportion_confint(above, total, alpha=0.05, method="wilson")
            pct = above / total
            pcts.append(pct * 100)
            los.append(pct * 100 - lo * 100)
            his.append(hi * 100 - pct * 100)
            ns.append(total)

        x = np.arange(len(bins))
        color = DRUG_COLORS[drug]
        bars = ax.bar(x, pcts, color=color, edgecolor="dimgray", linewidth=0.4, width=0.7)
        ax.errorbar(x, pcts, yerr=[los, his], fmt="none", ecolor="black", capsize=3, linewidth=0.8)

        for xi, (bar, n, pct) in enumerate(zip(bars, ns, pcts)):
            ax.annotate(
                f"{pct:.1f}%\nn={n:,}",
                xy=(xi, bar.get_height() + his[xi] + 0.4),
                ha="center", va="bottom", fontsize=7.5,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(bins)
        ax.set_xlabel("ICU day")
        ax.set_ylabel("% of patient-days above threshold")
        ax.set_title(f"{DRUG_LABELS[drug]}   (threshold {threshold_label(drug)})")
        # Leave headroom for the per-bar annotation
        ymax = max(p + h for p, h in zip(pcts, his)) if pcts else 1.0
        ax.set_ylim(0, ymax * 1.25 + 2)

    fig.suptitle(
        "Nocturnal up-titration prevalence by ICU day",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, "pct_uptitrated_by_icu_day")


if __name__ == "__main__":
    main()
