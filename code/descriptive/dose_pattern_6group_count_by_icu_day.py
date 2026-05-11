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

X-axis: ICU days 1–7, restricted to full-24h coverage days only. Earlier
versions used `0 (intub), 1, …, 7, 8+, last (exit)` to surface cohort
end-cap dynamics; that's now intentionally dropped (2026-05-08) so each
bar reflects a fully-comparable 12+12 hr coverage population. Days 8+
and partial coverage rows are dropped at the load filter.

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
    load_model_input,
    save_csv,
    save_fig,
)


MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = [str(i) for i in range(MIN_DAY, MAX_DAY + 1)]
SEGMENT_LABEL_THRESHOLD_FRAC = 0.04  # only label segments ≥ 4% of bar's total


def main() -> None:
    apply_style()
    df = load_model_input()
    # Restrict to full-24h ICU days 1..7. Drops day 0 (first_partial),
    # the trajectory-final partial day (last_partial), and any day ≥ 8.
    # Each bar reflects a fully-comparable 12+12 hr coverage population.
    in_range = df["_nth_day"].between(MIN_DAY, MAX_DAY)
    df = df.loc[df["_is_full_24h_day"] & in_range].copy()
    df["_x_bin"] = pd.Categorical(
        df["_nth_day"].astype(int).astype(str),
        categories=DAY_BINS, ordered=True,
    )

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.0), sharex=True)
    x_positions = np.arange(len(DAY_BINS))

    # Long-format rows accumulated across drugs so we can emit a single
    # federated-friendly CSV at the end. Cross-site pooling reads this
    # CSV (never the raw parquet) per the federation contract in
    # .dev/CLAUDE.md "Federation contract" subsection.
    csv_frames: list[pd.DataFrame] = []

    for ax, drug in zip(axes, DRUGS):
        d = df.copy()
        d["_pattern"] = pd.Categorical(
            categorize_diff_6way(
                d[DIFF_COLS[drug]], d[DAY_COLS[drug]], d[NIGHT_COLS[drug]],
                THRESHOLDS[drug],
            ),
            categories=list(DOSE_PATTERN_LABELS), ordered=True,
        )

        counts = (
            d.groupby(["_x_bin", "_pattern"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
            .reindex(index=DAY_BINS, fill_value=0)
        )
        bar_totals = counts.sum(axis=1)

        long = counts.stack().reset_index()
        long.columns = ["nth_day", "pattern_label", "count"]
        long.insert(0, "drug", drug)
        csv_frames.append(long)

        # Stack order from bottom: the 5 measurable-diff colored bands, then
        # the drug-holiday gray cap ("Not receiving that day").
        cum = np.zeros(len(DAY_BINS))
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

        # Thin horizontal frame line marking the boundary between the 5
        # measurable-diff colored bands and the drug-holiday gray cap.
        on_drug_top = (
            bar_totals - counts["Not receiving that day"]
        ).to_numpy()
        ax.hlines(on_drug_top, x_positions - 0.45, x_positions + 0.45,
                  color="dimgray", linewidth=0.8, zorder=3)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(DAY_BINS)
        ax.set_xlabel("ICU day (full-24h coverage only)")
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
        "Cohort size by ICU day (1–7, full-24h coverage) with 6-group dose-pattern composition\n"
        "Bar HEIGHT = patient-days surviving that ICU day. Bottom 5 bands = measurable diff; "
        "gray cap = drug-holiday. Horizontal line marks the boundary between measurable-diff "
        "and drug-holiday segments.",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.005,
        "Cohort: full-24h ICU-day rows (n_hours_day = n_hours_night = 12) from the qualifying "
        "first IMV streak ≥ 24h, restricted to days 1–7. Day 0 (partial intubation day), the "
        "trajectory-final partial day, single-shift rows, and days 8+ are dropped at the load "
        "filter so each bar reflects a fully-comparable 12+12 hr coverage population. "
        "See docs/descriptive_figures.md §6.0. "
        "Per-patient composition is in `dose_pattern_6group_persistence.png` Panel A.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "dose_pattern_6group_count_by_icu_day")

    # Federated-pooling artifact: long-format counts at the (drug, nth_day,
    # pattern_label) grain. Cross-site code reads this CSV (never the raw
    # parquet) and may re-render the y-axis as absolute counts OR per-day
    # proportions on demand. See .dev/CLAUDE.md "Federation contract".
    csv_long = pd.concat(csv_frames, ignore_index=True)
    csv_long["nth_day"] = csv_long["nth_day"].astype(int)
    save_csv(csv_long, "dose_pattern_6group_count_by_icu_day")


if __name__ == "__main__":
    main()
