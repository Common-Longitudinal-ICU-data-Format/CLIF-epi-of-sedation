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
treatment as `night_day_diff_combined_by_icu_day.py`.

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

# 7th category specific to this figure: rows with `_single_shift_day = True`
# stay IN the bar so day 0 / "last (exit)" totals reflect the real cohort N,
# but they're a distinct visual layer rather than being amount-sign-fallback-
# classified into one of the 6 buckets. White fill + diagonal hatch + gray
# border reads "data present, classification undefined" without resembling
# any of the 6 saturated/gray semantic colors.
SINGLE_SHIFT_LABEL = "Single-shift (no diff)"
SINGLE_SHIFT_FACE = "white"
SINGLE_SHIFT_HATCH = "///"
SINGLE_SHIFT_EDGE = "#888888"
EXTENDED_LABELS = tuple(DOSE_PATTERN_LABELS) + (SINGLE_SHIFT_LABEL,)
EXTENDED_STACK_ORDER = tuple(COUNT_BAR_STACK_ORDER) + (SINGLE_SHIFT_LABEL,)


def _classify_bin(row: pd.Series) -> str:
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    # Use _rel_day (= _nth_day − MIN(_nth_day) per hosp) for the numeric bin
    # so cohort attrition is monotonic across days. Hospitalizations whose
    # first row is `_nth_day = 1` (streak begins at 7:00 AM exactly) get
    # _rel_day = 0 on that row (caught by `_is_first_day` above) and _rel_day
    # = 1 on their second row.
    d = int(row["_rel_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    for col in ["_is_first_day", "_is_last_day", "_single_shift_day", "_rel_day"]:
        if col not in df.columns:
            raise RuntimeError(
                f"exposure_dataset.parquet missing column {col!r} — "
                f"re-run code/05_modeling_dataset.py against the current site."
            )
    df = df.copy()
    df["_x_bin"] = df.apply(_classify_bin, axis=1)
    df["_x_bin"] = pd.Categorical(df["_x_bin"], categories=ALL_BIN_LABELS, ordered=True)
    is_single = df["_single_shift_day"].astype(bool)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.0), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))

    for ax, drug in zip(axes, DRUGS):
        d = df.copy()
        # Categorize full-coverage rows via the 6-group classifier; assign
        # single-shift rows to the 7th SINGLE_SHIFT_LABEL bucket directly so
        # the rate-diff = NaN issue doesn't pollute the 6 real categories.
        d["_pattern"] = categorize_diff_6way(
            d[DIFF_COLS[drug]], d[DAY_COLS[drug]], d[NIGHT_COLS[drug]],
            THRESHOLDS[drug],
        ).astype(object)
        d.loc[is_single, "_pattern"] = SINGLE_SHIFT_LABEL
        d["_pattern"] = pd.Categorical(d["_pattern"], categories=list(EXTENDED_LABELS), ordered=True)

        counts = (
            d.groupby(["_x_bin", "_pattern"], observed=False)
            .size()
            .unstack(fill_value=0)
            .reindex(columns=list(EXTENDED_LABELS), fill_value=0)
            .reindex(index=ALL_BIN_LABELS, fill_value=0)
        )
        bar_totals = counts.sum(axis=1)

        # Stack order from bottom: the 5 measurable-diff colored bands, then
        # the drug-holiday gray cap ("Not receiving that day"), then the
        # single-shift hatched cap on top.
        cum = np.zeros(len(ALL_BIN_LABELS))
        for label in EXTENDED_STACK_ORDER:
            seg = counts[label].to_numpy().astype(float)
            if label == SINGLE_SHIFT_LABEL:
                ax.bar(x_positions, seg, bottom=cum, color=SINGLE_SHIFT_FACE,
                       edgecolor=SINGLE_SHIFT_EDGE, linewidth=0.5,
                       hatch=SINGLE_SHIFT_HATCH, label=label)
            else:
                color = DOSE_PATTERN_COLORS[label]
                ax.bar(x_positions, seg, bottom=cum, color=color, edgecolor="white",
                       linewidth=0.4, label=label)
                # Annotate "X%" on saturated/gray segments large enough to read.
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

        # Thin horizontal frame line at the boundary between the 5 measurable-
        # diff bands and the 2 not-measurable-diff caps (drug-holiday +
        # single-shift). on_drug_top = bar_total − {Not receiving that day,
        # Single-shift (no diff)} — i.e. the height of the bottom 5 colored
        # bands. Reads as a quiet ruler across the bars.
        on_drug_top = (
            bar_totals
            - counts["Not receiving that day"]
            - counts[SINGLE_SHIFT_LABEL]
        ).to_numpy()
        ax.hlines(on_drug_top, x_positions - 0.45, x_positions + 0.45,
                  color="dimgray", linewidth=0.8, zorder=3)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day")
        ax.set_ylabel("Patient-days")
        ax.set_title(f"{DRUG_LABELS[drug]}  (T = ±{THRESHOLDS[drug]})")

    # Single legend across the figure (one entry per group + one for single-shift).
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DOSE_PATTERN_COLORS[label], ec="white")
        for label in DOSE_PATTERN_LABELS
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor=SINGLE_SHIFT_FACE,
                      edgecolor=SINGLE_SHIFT_EDGE, hatch=SINGLE_SHIFT_HATCH)
    )
    fig.legend(
        handles, list(DOSE_PATTERN_LABELS) + [SINGLE_SHIFT_LABEL],
        loc="lower center", ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Cohort size by ICU day with 6-group dose-pattern composition\n"
        "Bar HEIGHT = all patient-days surviving that ICU day. Bottom 5 bands = measurable diff; "
        "gray cap = drug-holiday; hatched cap = single-shift (rate-diff undefined). "
        "Horizontal line marks the boundary between measurable-diff and not-measurable bands.",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.005,
        "Cohort: all patient-days from the qualifying first IMV streak ≥ 24h. Single-shift days "
        "(one shift had 0 hours of coverage) sit at the top of each bar in a hatched cap — they "
        "contribute to total N but not to the 6-group classification (rate-diff is undefined). "
        "X-bins use _rel_day so cohort attrition is monotonic across days. "
        "See docs/descriptive_figures.md §6.0. "
        "Per-patient composition is in `dose_pattern_6group_persistence.png` Panel A.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "dose_pattern_6group_count_by_icu_day")


if __name__ == "__main__":
    main()
