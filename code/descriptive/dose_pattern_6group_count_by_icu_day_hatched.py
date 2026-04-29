"""Hatched-overlay variant of `dose_pattern_6group_count_by_icu_day`.

Same geometry as the original, but each color segment is split into TWO
stacked sub-segments of the same color: a flat-fill bottom (full-coverage
rows where the rate-diff is real) and a diagonally-hatched top
(single-shift rows where the 6-group classification was assigned via
amount-sign fallback in `categorize_diff_6way`). The hatch density
visually communicates "what fraction of this segment came from rows whose
classification is artifact-prone."

This is one of two comparison presentations of the artifact contribution
(see also `dose_pattern_6group_count_by_icu_day_split.py`). The original
flat-color figure remains as a baseline.

Stack order matches the original: `Equal, both zero` (drug-holiday) caps
the bar, the 5 on-drug areas form a contiguous bottom block, and a thin
horizontal frame line marks the on-drug/off-drug split.

Usage:
    uv run python code/descriptive/dose_pattern_6group_count_by_icu_day_hatched.py
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
    split_full_vs_single,
)


MAX_MID_DAY = 7
MID_DAY_BINS = [str(i) for i in range(1, MAX_MID_DAY + 1)] + [f"{MAX_MID_DAY + 1}+"]
ALL_BIN_LABELS = ["0\n(intub)"] + MID_DAY_BINS + ["last\n(exit)"]
SEGMENT_LABEL_THRESHOLD_FRAC = 0.04


def _classify_bin(row: pd.Series) -> str:
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    d = int(row["_nth_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def _counts_by_xbin_pattern(d: pd.DataFrame, drug: str) -> pd.DataFrame:
    """Categorize and pivot to (x_bin × pattern) count table."""
    d = d.copy()
    d["_pattern"] = categorize_diff_6way(
        d[DIFF_COLS[drug]], d[DAY_COLS[drug]], d[NIGHT_COLS[drug]],
        THRESHOLDS[drug],
    )
    return (
        d.groupby(["_x_bin", "_pattern"], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
        .reindex(index=ALL_BIN_LABELS, fill_value=0)
    )


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

    df_full, df_single = split_full_vs_single(df)

    fig, axes = plt.subplots(1, 3, figsize=(17, 7.0), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))

    for ax, drug in zip(axes, DRUGS):
        counts_full = _counts_by_xbin_pattern(df_full, drug)
        counts_single = _counts_by_xbin_pattern(df_single, drug)
        counts_total = counts_full + counts_single
        bar_totals = counts_total.sum(axis=1)
        single_totals = counts_single.sum(axis=1)

        cum = np.zeros(len(ALL_BIN_LABELS))
        for label in COUNT_BAR_STACK_ORDER:
            seg_full = counts_full[label].to_numpy().astype(float)
            seg_single = counts_single[label].to_numpy().astype(float)
            seg_total = seg_full + seg_single
            color = DOSE_PATTERN_COLORS[label]
            # Flat-fill bottom = full-coverage rows (rate-diff is real).
            ax.bar(x_positions, seg_full, bottom=cum, color=color, edgecolor="white",
                   linewidth=0.4, label=label)
            # Diagonal-hatch top = single-shift rows (amount-sign fallback).
            ax.bar(x_positions, seg_single, bottom=cum + seg_full, color=color,
                   edgecolor="white", linewidth=0.4, hatch="///", alpha=0.95)
            for x, c, total, s in zip(x_positions, cum, bar_totals.to_numpy(), seg_total):
                if total <= 0:
                    continue
                frac = s / total
                if frac >= SEGMENT_LABEL_THRESHOLD_FRAC and s >= 30:
                    ax.text(
                        x, c + s / 2, f"{frac * 100:.0f}%",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if color in ("#2166ac", "#b2182b") else "black",
                    )
            cum += seg_total

        on_drug_top = (bar_totals - counts_total["Equal, both zero"]).to_numpy()
        ax.hlines(on_drug_top, x_positions - 0.45, x_positions + 0.45,
                  color="dimgray", linewidth=0.8, zorder=3)

        for x, total, on_drug, n_single in zip(x_positions,
                                                  bar_totals.to_numpy(),
                                                  on_drug_top,
                                                  single_totals.to_numpy()):
            label = f"n={int(total):,}"
            if int(n_single) > 0:
                label += f"\n(of which {int(n_single):,} hatched)"
            ax.text(x, -bar_totals.max() * 0.04, label,
                    ha="center", va="top", fontsize=7, color="dimgray")
            if total > 0:
                ax.text(x, on_drug + bar_totals.max() * 0.012,
                        f"on-drug n={int(on_drug):,}",
                        ha="center", va="bottom", fontsize=6.5, color="dimgray")

        for bx in (0, len(ALL_BIN_LABELS) - 1):
            ax.axvspan(bx - 0.45, bx + 0.45, color="lightgray", alpha=0.25, zorder=0)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day")
        ax.set_ylabel("Patient-days")
        ax.set_title(f"{DRUG_LABELS[drug]}  (T = ±{THRESHOLDS[drug]})")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=DOSE_PATTERN_COLORS[label], ec="white")
        for label in DOSE_PATTERN_LABELS
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor="white", ec="dimgray", hatch="///")
    )
    fig.legend(
        handles, list(DOSE_PATTERN_LABELS) + ["// = single-shift (amount-sign fallback)"],
        loc="lower center", ncol=4, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Cohort size by ICU day with 6-group dose-pattern composition (hatched variant)\n"
        "Each color segment splits into flat-fill (full-coverage, real diff) + diagonal hatch "
        "(single-shift, amount-sign fallback). Drug-holiday caps the bar.",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.5, 0.005,
        "Hatched portion of each segment = single-shift rows (one shift had 0 hours of coverage); "
        "their 6-group classification is driven by amount-sign fallback in `categorize_diff_6way` "
        "(see docs/descriptive_figures.md §6.0) — it reflects which shift was non-empty, not real "
        "diurnal variation. Day 0 and last (exit) are typically dominated by hatched contribution.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "dose_pattern_6group_count_by_icu_day_hatched")


if __name__ == "__main__":
    main()
