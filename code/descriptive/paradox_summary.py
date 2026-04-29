"""Three-panel paradox summary, one row per drug.

Headline figure for the night-vs-day diurnal paradox: at the population
level the cohort mean of (night − day dose) is *positive* for all three
sedatives, yet on the per-patient-day distribution well over half the days
have `diff < 0` (or, equivalently, the median is < 0). This figure puts the
mean, median, sign-split, and tail balance in one view per drug so the
mean-vs-median tension is readable at a glance.

Three panels per drug (3 rows × 3 cols):

  Panel A — Distribution shape.
      Histogram of `*_dif_*` clipped to 1st–99th percentile, with three
      vertical lines: mean (solid green), median (dashed orange), zero
      (dashed black). Top-right annotation reports the numerical
      `mean = X.XX | median = Y.YY` so the right-skew or left-skew is
      readable in one glance.

  Panel B — Sign distribution.
      Single horizontal stacked bar with three segments: % Day > Night
      (blue), % Equal (gray), % Night > Day (red). Each segment annotated
      with its percentage. Directly answers "what fraction of patient-days
      had a higher dose at night vs day".

  Panel C — Gross-sum tail balance.
      Two side-by-side bars: total of all positive diffs (red, "Night > Day
      gross sum") vs absolute total of all negative diffs (blue, "Day >
      Night gross sum"), normalized to per-patient-day. The visual gap
      between the bars *is* the cohort mean; the bar heights *are* the
      gross tail magnitudes. Re-uses the framing of `diff_tail_contribution`
      but at the cohort level rather than per-quantile.

Glossary (printed in the figure footer): a distribution's "tails" are its
extreme ends. "Fat tails" means those extremes carry a disproportionately
large share of the gross sum — when both tails are fat and roughly balanced,
the mean is the tiny residual of two big opposing forces.

Reads `exposure_dataset.parquet` (full hospital-stay coverage including
day 0 and last day). Saves to
`output_to_share/{site}/figures/paradox_summary.png`.

Usage:
    uv run python code/descriptive/paradox_summary.py
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
    apply_style,
    load_exposure,
    prepare_diffs,
    save_fig,
)


def _on_drug_slice(df, drug):
    """Patient-days with nonzero dose on `drug` in either shift."""
    day_col, night_col = DAY_COLS[drug], NIGHT_COLS[drug]
    return df[(df[day_col].fillna(0) > 0) | (df[night_col].fillna(0) > 0)]


def _panel_a_histogram(ax, values: np.ndarray, drug: str) -> None:
    color = DRUG_COLORS[drug]
    if len(values) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    lo, hi = np.percentile(values, [1, 99])
    clipped = values[(values >= lo) & (values <= hi)]
    ax.hist(clipped, bins=60, color=color, edgecolor="white", linewidth=0.3)

    mean = float(np.mean(values))
    median = float(np.median(values))
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(mean, color="green", linestyle="-", linewidth=1.6,
               label=f"Mean = {mean:+.3f}")
    ax.axvline(median, color="darkorange", linestyle="--", linewidth=1.4,
               label=f"Median = {median:+.3f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel(f"Night − day rate ({DRUG_UNITS[drug]})")
    ax.set_ylabel("Patient-days")
    ax.set_title(f"{DRUG_LABELS[drug]} — distribution shape", fontsize=10)


def _panel_b_sign_distribution(ax, values: np.ndarray, drug: str) -> None:
    if len(values) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    finite = values[np.isfinite(values)]
    n = len(finite)
    pct_day_higher = 100.0 * (finite < 0).sum() / n
    pct_equal = 100.0 * (finite == 0).sum() / n
    pct_night_higher = 100.0 * (finite > 0).sum() / n

    segments = [
        ("Day > Night", pct_day_higher, "#2166ac"),  # blue
        ("Equal",       pct_equal,        "#bdbdbd"),  # gray
        ("Night > Day", pct_night_higher, "#b2182b"),  # red
    ]
    cum = 0.0
    for label, pct, color in segments:
        ax.barh(0, pct, left=cum, color=color, edgecolor="white", linewidth=1.5,
                height=0.55, label=label)
        if pct >= 4.0:
            ax.text(cum + pct / 2, 0, f"{label}\n{pct:.1f}%",
                    ha="center", va="center",
                    color="white" if color in ("#2166ac", "#b2182b") else "black",
                    fontsize=9)
        cum += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("% of patient-days")
    ax.set_yticks([])
    ax.set_title(f"{DRUG_LABELS[drug]} — sign split", fontsize=10)


def _panel_c_tail_balance(ax, values: np.ndarray, drug: str) -> None:
    if len(values) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return
    finite = values[np.isfinite(values)]
    n = len(finite)
    pos_sum_per_pd = float(finite[finite > 0].sum() / n) if n else 0.0
    neg_sum_per_pd = float(-finite[finite < 0].sum() / n) if n else 0.0
    net = pos_sum_per_pd - neg_sum_per_pd

    bars = ax.bar(
        [0, 1], [pos_sum_per_pd, neg_sum_per_pd],
        color=["#b2182b", "#2166ac"],
        edgecolor="white", linewidth=0.6, width=0.55,
    )
    for bar, val in zip(bars, (pos_sum_per_pd, neg_sum_per_pd)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Night > Day\ngross sum", "Day > Night\ngross sum"])
    ax.set_ylabel(f"Per-patient-day ({DRUG_UNITS[drug]})")
    ax.set_title(
        f"{DRUG_LABELS[drug]} — tail balance (net = mean = {net:+.3f})",
        fontsize=10,
    )


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())

    # Figure is taller than the panel grid alone needs because we reserve
    # the bottom ~30% for the inline interpretation footnote (avoids
    # forcing the user to flip back to a separate doc).
    fig, axes = plt.subplots(
        len(DRUGS), 3, figsize=(15, 4.0 * len(DRUGS) + 5.0),
    )

    for r, drug in enumerate(DRUGS):
        on = _on_drug_slice(df, drug)
        col = DIFF_COLS[drug]
        values = on[col].dropna().to_numpy()

        _panel_a_histogram(axes[r, 0], values, drug)
        _panel_b_sign_distribution(axes[r, 1], values, drug)
        _panel_c_tail_balance(axes[r, 2], values, drug)

    fig.suptitle(
        "Paradox summary: distribution shape, sign split, and tail balance per drug",
        fontsize=13, y=1.00,
    )
    fig.tight_layout()
    # Reserve the bottom ~30% for the interpretation footnote.
    fig.subplots_adjust(bottom=0.30)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Each row is one drug. Every patient-day on the drug contributes exactly one diff value:\n"
        "    diff = (per-hour rate during night-shift hours) − (per-hour rate during day-shift hours)\n"
        "Units are mcg/kg/min for propofol, mcg/hr for fentanyl-eq, mg/hr for midazolam-eq. Only patient-days\n"
        "where the patient was on the drug in at least one shift are included (off-drug days are excluded here).\n"
        "\n"
        "Panel A (left)  — distribution shape. Histogram of diff, clipped to 1st–99th percentile so a few\n"
        "                  outliers don't squash the visible range. Three vertical lines: solid green = the\n"
        "                  cohort mean of diff; dashed orange = the cohort median of diff; dashed black = 0\n"
        "                  (the equal-shifts reference). Both numerical values printed in the legend.\n"
        "                    → If green and orange are both > 0, the diff distribution genuinely sits above 0\n"
        "                      (night > day on average AND on the typical patient-day).\n"
        "                    → If green > 0 but orange ≤ 0 (or vice versa), the mean and median disagree on\n"
        "                      sign — that is the apparent \"paradox\".\n"
        "\n"
        "Panel B (mid)   — sign split. One horizontal stacked bar showing what fraction of patient-days fall\n"
        "                  into each direction: %Day > Night (blue, diff < 0), %Equal (gray, diff = 0),\n"
        "                  %Night > Day (red, diff > 0). Sums to 100%.\n"
        "                    → Tells you the directional plurality independent of magnitude. A cohort can have\n"
        "                      mean > 0 (Panel A) but >50% of days with diff < 0 (Panel B) — that is the\n"
        "                      mean-vs-majority tension the \"paradox\" framing points to.\n"
        "\n"
        "Panel C (right) — tail balance. \"Gross sum\" of positive diffs = sum over patient-days where\n"
        "                  diff > 0 of (diff value); divided by the total number of patient-days in the slice\n"
        "                  to get a per-patient-day average. Same on the negative side using |diff|.\n"
        "                    Red bar height = per-patient-day average over Night > Day rows of diff.\n"
        "                    Blue bar height = per-patient-day average over Day > Night rows of |diff|.\n"
        "                    The visible gap between the bar heights = the cohort mean (= red − blue).\n"
        "                    → Both bars tall AND of similar height = balanced fat tails: lots of activity\n"
        "                      on both sides, but the cohort mean is the small residual.\n"
        "                    → One bar tall, the other short = unbalanced; the cohort mean reflects a real\n"
        "                      directional drift.\n"
        "\n"
        "GLOSSARY\n"
        "  tail            — the extreme end of a distribution: patient-days far from 0 (rightmost = positive\n"
        "                     tail = night >> day; leftmost = negative tail = day >> night).\n"
        "  gross sum       — arithmetic sum of all values on one side of zero (e.g., sum of all diff values\n"
        "                     where diff > 0). NOT a count — it weights large diffs proportionally more.\n"
        "  fat tails       — informal: the tails carry a disproportionately large share of the gross sum.\n"
        "                     Concretely, a single 80-mcg/kg/min day contributes as much to the positive\n"
        "                     gross sum as eighty 1-mcg/kg/min days. See `diff_tail_contribution.png` for\n"
        "                     the per-quantile breakdown.\n"
    )
    fig.text(
        0.04, 0.001, footnote,
        ha="left", va="bottom", fontsize=8, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, "paradox_summary")


if __name__ == "__main__":
    main()
