"""Cohort-size-by-ICU-day stacked bar with 6-group internal composition.

Replaces the deleted `pct_night_vs_day_by_icu_day.py` with redesigned
semantics: instead of a 100%-normalized stacked bar (which hid cohort
attrition), the y-axis here is in ABSOLUTE patient-day counts. Bars shrink
left-to-right as patients exit the cohort via extubation/death/transfer —
the eye picks up cohort attrition AND group composition simultaneously.

Within each bar, segments are sized proportionally to the 6-group
composition for that day, colored by `DOSE_PATTERN_COLORS` (dark red =
"Markedly higher at night", dark blue = "Markedly higher at day", per the
user's color convention). Segments large enough to fit text get a "X%"
label showing their share of the bar's total.

X-axis: `0 (intub), 1, 2, …, 7, 8+, last (exit)` — same boundary-day
treatment as `night_day_diff_mean_by_icu_day.py`.

Usage:
    uv run python code/descriptive/dose_pattern_6group_count_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    COUNT_BAR_STACK_ORDER,
    DAY_COLS,
    DIFF_COLS,
    DOSE_PATTERN_COLORS,
    DOSE_PATTERN_LABELS,
    DRUG_LABELS,
    DRUGS,
    NIGHT_COLS,
    THRESHOLDS,
    apply_style,
    categorize_diff_6way,
    load_exposure,
    prepare_diffs,
    save_fig,
)


MAX_MID_DAY = 7
MID_DAY_BINS = [str(i) for i in range(1, MAX_MID_DAY + 1)] + [f"{MAX_MID_DAY + 1}+"]
ALL_BIN_LABELS = ["0\n(intub)"] + MID_DAY_BINS + ["last\n(exit)"]
SEGMENT_LABEL_THRESHOLD_FRAC = 0.04  # only label segments ≥ 4% of bar's total


def _classify_bin(row: pd.Series) -> str:
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    d = int(row["_nth_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    for col in ["_is_first_day", "_is_last_day", "_single_shift_day"]:
        if col not in df.columns:
            raise RuntimeError(
                f"exposure_dataset.parquet missing column {col!r} — "
                f"re-run code/05_modeling_dataset.py against the current site."
            )
    df = df.copy()
    df["_x_bin"] = df.apply(_classify_bin, axis=1)
    df["_x_bin"] = pd.Categorical(df["_x_bin"], categories=ALL_BIN_LABELS, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.0), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))

    for ax, drug in zip(axes, DRUGS):
        d = df.copy()
        d["_pattern"] = categorize_diff_6way(
            d[DIFF_COLS[drug]], d[DAY_COLS[drug]], d[NIGHT_COLS[drug]],
            THRESHOLDS[drug],
        )
        # Counts per (x_bin, group)
        counts = (
            d.groupby(["_x_bin", "_pattern"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
            .reindex(index=ALL_BIN_LABELS, fill_value=0)
        )
        bar_totals = counts.sum(axis=1)

        # Stack order: COUNT_BAR_STACK_ORDER puts "Equal, both zero" (drug-
        # holiday) at the TOP of the bar, so total bar height = surviving
        # cohort that ICU day and the on-drug zone (the 5 remaining areas)
        # sits as a contiguous bottom block. Markedly-night ends up just
        # below the drug-holiday cap — still high in the bar, still red,
        # but no longer the literal top. See descriptive_figures.md §6.0.
        cum = np.zeros(len(ALL_BIN_LABELS))
        for label in COUNT_BAR_STACK_ORDER:
            seg = counts[label].to_numpy().astype(float)
            color = DOSE_PATTERN_COLORS[label]
            ax.bar(x_positions, seg, bottom=cum, color=color, edgecolor="white",
                   linewidth=0.4, label=label)
            # Annotate "X%" on segments large enough to read.
            for x, c, total, s in zip(x_positions, cum, bar_totals.to_numpy(), seg):
                if total <= 0:
                    continue
                frac = s / total
                if frac >= SEGMENT_LABEL_THRESHOLD_FRAC and s >= 30:
                    ax.text(
                        x, c + s / 2, f"{frac * 100:.0f}%",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if color in ("#2166ac", "#b2182b") else "black",
                    )
            cum += seg

        # Thin horizontal frame line at the on-drug/off-drug boundary in
        # every bar. The on-drug zone = areas {1,2,4,5,6} = bar_total minus
        # the drug-holiday count. Reads as a quiet ruler across the bars
        # marking where the on-drug fraction ends.
        on_drug_top = (bar_totals - counts["Equal, both zero"]).to_numpy()
        ax.hlines(on_drug_top, x_positions - 0.45, x_positions + 0.45,
                  color="dimgray", linewidth=0.8, zorder=3)

        # Two N annotations per bar: total under the bar (bar height) +
        # on-drug count just above the on-drug-zone frame line.
        for x, total, on_drug in zip(x_positions,
                                       bar_totals.to_numpy(),
                                       on_drug_top):
            ax.text(x, -bar_totals.max() * 0.04, f"n={int(total):,}",
                    ha="center", va="top", fontsize=7, color="dimgray")
            if total > 0:
                ax.text(x, on_drug + bar_totals.max() * 0.012,
                        f"on-drug n={int(on_drug):,}",
                        ha="center", va="bottom", fontsize=6.5, color="dimgray")

        # Subtle gray band on day 0 and "last" so the boundary days are
        # visually distinct from the steady-state middle.
        for bx in (0, len(ALL_BIN_LABELS) - 1):
            ax.axvspan(bx - 0.45, bx + 0.45, color="lightgray", alpha=0.25, zorder=0)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day")
        ax.set_ylabel("Patient-days")
        ax.set_title(f"{DRUG_LABELS[drug]}  (T = ±{THRESHOLDS[drug]})")

    # Single legend across the figure (one entry per group).
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DOSE_PATTERN_COLORS[label], ec="white")
        for label in DOSE_PATTERN_LABELS
    ]
    fig.legend(
        handles, list(DOSE_PATTERN_LABELS),
        loc="lower center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Cohort size by ICU day with 6-group dose-pattern composition\n"
        "Bar HEIGHT = patient-days surviving that ICU day. Bottom 5 areas = on-drug; "
        "top gray cap = drug-holiday. Horizontal line marks the on-drug/off-drug split.",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.005,
        "Single-shift rows (one shift had 0 hours of coverage) are classified via amount-sign fallback in "
        "`categorize_diff_6way` (see docs/descriptive_figures.md §6.0); this figure does not visually "
        "distinguish them. Companion figures `*_hatched.png` and `*_split.png` show the single-shift "
        "contribution explicitly. Per-patient composition is in `dose_pattern_6group_persistence.png` Panel A.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "dose_pattern_6group_count_by_icu_day")


if __name__ == "__main__":
    main()
