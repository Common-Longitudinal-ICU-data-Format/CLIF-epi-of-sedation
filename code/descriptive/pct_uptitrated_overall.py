"""Overall percentage of patient-days crossing the clinical up-titration threshold.

Single panel, three bars (propofol / fent-eq / midaz-eq). Height = fraction of
patient-days where night-minus-day rate exceeded the per-drug threshold
(see _shared.THRESHOLDS). Binomial 95% CIs via Wilson's method. Text annotation
shows "N above / total (pct%)".

Usage:
    uv run python code/descriptive/pct_uptitrated_overall.py
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
    load_analytical,
    save_fig,
    threshold_label,
)


def main() -> None:
    apply_style()
    df = load_analytical()

    labels, pcts, los, his, ns_above, ns_total = [], [], [], [], [], []
    for drug in DRUGS:
        col = DIFF_COLS[drug]
        series = df[col].dropna()
        total = len(series)
        above = int((series > THRESHOLDS[drug]).sum())
        lo, hi = proportion_confint(above, total, alpha=0.05, method="wilson")
        pct = above / total if total else 0.0

        labels.append(f"{DRUG_LABELS[drug]}\n({threshold_label(drug)})")
        pcts.append(pct * 100)
        los.append(pct * 100 - lo * 100)
        his.append(hi * 100 - pct * 100)
        ns_above.append(above)
        ns_total.append(total)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(DRUGS))
    colors = [DRUG_COLORS[d] for d in DRUGS]

    bars = ax.bar(x, pcts, color=colors, edgecolor="dimgray", linewidth=0.5, width=0.55)
    ax.errorbar(x, pcts, yerr=[los, his], fmt="none", ecolor="black", capsize=4, linewidth=1.0)

    for xi, (bar, above, total, pct) in enumerate(zip(bars, ns_above, ns_total, pcts)):
        y = bar.get_height() + his[xi] + 0.7
        ax.annotate(
            f"{above:,} / {total:,}\n({pct:.1f}%)",
            xy=(xi, y), ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of patient-days above threshold")
    ax.set_title(
        "Overall nocturnal up-titration prevalence\n"
        "(share of patient-days where night rate exceeded day rate by ≥ threshold)",
        fontsize=12,
    )
    ax.set_ylim(0, max(pcts) + max(his) + 5)
    fig.tight_layout()
    save_fig(fig, "pct_uptitrated_overall")


if __name__ == "__main__":
    main()
