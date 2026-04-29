"""Cumulative contribution of tail patient-days to the gross night-minus-day sum.

For each drug, patient-days are ranked by the signed night-minus-day diff and
split into 5% quantile bins. Two stacked bars are drawn per drug:

  - Left bar (positive tail, Night > Day): cumulative |sum| contributed by
    the top X% of positive-diff patient-days as X grows (5%, 10%, 25%, 50%,
    100%).
  - Right bar (negative tail, Day > Night): same on the negative side.

The annotation on each bar reports what fraction of the gross positive-sum
(or negative-sum) comes from the top X% of days. When the top 5% carries
the majority of the |sum| on BOTH sides, the cohort mean is the tiny residual
of two balanced tails — that's the source of the apparent night-vs-day
"paradox" (mean near zero, but heavy activity at both extremes).

Glossary: a distribution's "tails" are its extreme ends — the small fraction
of observations far from the center. "Fat tails" means those extremes carry
a disproportionately large share of the gross sum.

Usage:
    uv run python code/descriptive/diff_tail_contribution.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DAY_COLS,
    DIFF_COLS,
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    NIGHT_COLS,
    THRESHOLDS,
    apply_style,
    load_exposure,
    prepare_diffs,
    save_fig,
)


QUANTILE_CUTS = (0.05, 0.10, 0.25, 0.50, 1.00)
BAR_COLORS = ["#b2182b", "#ef8a62", "#fddbc7", "#d1e5f0", "#2166ac"]


def _on_drug_slice(df, drug):
    """Return patient-days with nonzero dose on `drug` in either shift.

    Uses the unit-suffixed column names from _shared.DAY_COLS / NIGHT_COLS so
    the rename to `_prop_day_mcg_kg_min` (etc.) flows through automatically.
    """
    day_col, night_col = DAY_COLS[drug], NIGHT_COLS[drug]
    return df[(df[day_col] > 0) | (df[night_col] > 0)]


def _tail_fractions(values: np.ndarray, quantile_cuts=QUANTILE_CUTS) -> list[float]:
    """Given a sorted-descending positive array, return cumulative |sum|
    fractions contributed by the top q share for q in quantile_cuts."""
    if len(values) == 0:
        return [0.0] * len(quantile_cuts)
    total = values.sum()
    if total == 0:
        return [0.0] * len(quantile_cuts)
    n = len(values)
    fracs: list[float] = []
    for q in quantile_cuts:
        k = max(1, int(np.ceil(n * q)))
        fracs.append(values[:k].sum() / total)
    return fracs


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())

    # Tall figure — bottom region reserved for the inline footnote.
    fig, axes = plt.subplots(1, len(DRUGS), figsize=(15, 11.0), sharey=True)

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        thr = THRESHOLDS[drug]
        on = _on_drug_slice(df, drug)
        s = on[col].dropna().to_numpy()

        pos = np.sort(s[s > 0])[::-1]
        neg = np.sort(-s[s < 0])[::-1]  # magnitudes of negatives, descending

        pos_fracs = _tail_fractions(pos)
        neg_fracs = _tail_fractions(neg)

        # Two grouped bar clusters: positive-tail and negative-tail, each stacked
        # into 5 contribution bands.
        def stacked(ax, x0, fracs, label_prefix):
            prev = 0.0
            for i, (f, color) in enumerate(zip(fracs, BAR_COLORS)):
                seg = f - prev
                ax.bar(
                    x0, seg * 100, bottom=prev * 100, width=0.55,
                    color=color, edgecolor="white", linewidth=0.6,
                )
                if seg >= 0.03:
                    ax.text(
                        x0, (prev + seg / 2) * 100, f"{seg * 100:.0f}%",
                        ha="center", va="center", fontsize=8,
                        color="white" if i in (0, 4) else "black",
                    )
                prev = f
            ax.text(
                x0, 102, label_prefix, ha="center", va="bottom",
                fontsize=9, color="dimgray",
            )

        stacked(ax, 0, pos_fracs, f"POS tail\n(Night > Day)\nn = {len(pos):,}")
        stacked(ax, 1, neg_fracs, f"NEG tail\n(Day > Night)\nn = {len(neg):,}")

        pos_sum = pos.sum()
        neg_sum = neg.sum()
        net = pos_sum - neg_sum
        mean_all = s.mean()
        median_all = float(np.median(s))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Night > Day\ntail", "Day > Night\ntail"])
        ax.set_ylim(0, 115)
        ax.set_ylabel("% of gross |sum| within tail")
        ax.set_title(
            f"{DRUG_LABELS[drug]}  (T = {thr} {DRUG_UNITS[drug]})\n"
            f"mean={mean_all:+.3f}, median={median_all:+.3f}\n"
            f"POS sum={pos_sum:,.0f}, NEG sum={neg_sum:,.0f}, net={net:+,.0f} {DRUG_UNITS[drug]}",
            fontsize=10, color=DRUG_COLORS[drug],
        )

    # Shared legend for quantile bands.
    legend_labels = [f"Top {int(q * 100)}%" for q in QUANTILE_CUTS]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white")
        for c in BAR_COLORS
    ]
    # Reorder so legend reads big→small top-down (matches stack visual).
    fig.legend(
        handles, legend_labels,
        title="Cumulative quantile band",
        loc="lower center", bbox_to_anchor=(0.5, -0.06),
        ncol=len(QUANTILE_CUTS), frameon=False,
    )

    fig.suptitle(
        "Tail contribution to gross night-minus-day diff (per sedative)\n"
        "Each stacked bar = how the gross |sum| on one side of 0 is built up by the top 5/10/25/50/100% "
        "of patient-days on that side. Fat-tail check: top-5% band carrying ≫ 5% of bar = fat tail.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.02,
        "Per-panel title: cohort mean and median (over on-sedative patient-days), POS sum (Σ diff for diff>0), "
        "NEG sum (Σ |diff| for diff<0), net (= mean × n). Balanced POS ≈ NEG sums with fat tails on both sides "
        "is the paradox mechanism. Glossary: docs/descriptive_figures.md §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "diff_tail_contribution")


if __name__ == "__main__":
    main()
