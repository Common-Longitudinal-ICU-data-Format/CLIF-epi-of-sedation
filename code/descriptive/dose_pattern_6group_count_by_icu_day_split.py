"""Side-by-side split variant of `dose_pattern_6group_count_by_icu_day`.

Each ICU-day x-position renders TWO narrow sub-bars:
  - LEFT (muted/desaturated palette): single-shift rows only — 6-group
    classification was assigned via amount-sign fallback in
    `categorize_diff_6way`, so the diff is artifact-prone.
  - RIGHT (full-saturation palette): full-coverage rows only — both
    shifts have nonzero hours, so the diff is real.

Each sub-bar carries its own 6-group composition. Day 0 and last (exit)
are visually dominated by the left (muted) sub-bars; mid-days are
dominated by the right (saturated) sub-bars. Reading them as a pair makes
the artifact-vs-real contribution quantitatively comparable.

This is one of two comparison presentations of the artifact contribution
(see also `dose_pattern_6group_count_by_icu_day_hatched.py`).

Stack order matches the original: `Equal, both zero` (drug-holiday) caps
each sub-bar.

Usage:
    uv run python code/descriptive/dose_pattern_6group_count_by_icu_day_split.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.colors as mcolors
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
SEGMENT_LABEL_THRESHOLD_FRAC = 0.06   # raised slightly — sub-bars are narrower
SUB_BAR_WIDTH = 0.40
SUB_BAR_OFFSET = 0.22
MUTED_ALPHA = 0.45


def _classify_bin(row: pd.Series) -> str:
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    d = int(row["_nth_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def _muted(color: str) -> tuple[float, float, float, float]:
    return mcolors.to_rgba(color, alpha=MUTED_ALPHA)


def _counts_by_xbin_pattern(d: pd.DataFrame, drug: str) -> pd.DataFrame:
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


def _draw_subbar(ax, x_centers, counts, palette_fn, ymax_for_label):
    """Stack one sub-bar (left or right) per x position. Returns the
    on-drug-zone top heights so the caller can draw frame lines."""
    bar_totals = counts.sum(axis=1).to_numpy()
    cum = np.zeros(len(ALL_BIN_LABELS))
    for label in COUNT_BAR_STACK_ORDER:
        seg = counts[label].to_numpy().astype(float)
        color = palette_fn(DOSE_PATTERN_COLORS[label])
        ax.bar(x_centers, seg, bottom=cum, color=color, edgecolor="white",
               linewidth=0.4, width=SUB_BAR_WIDTH)
        for x, c, total, s in zip(x_centers, cum, bar_totals, seg):
            if total <= 0:
                continue
            frac = s / total
            if frac >= SEGMENT_LABEL_THRESHOLD_FRAC and s >= 30:
                ax.text(
                    x, c + s / 2, f"{frac * 100:.0f}%",
                    ha="center", va="center", fontsize=5.5,
                    color="white" if DOSE_PATTERN_COLORS[label] in ("#2166ac", "#b2182b") else "black",
                )
        cum += seg
    on_drug_top = bar_totals - counts["Equal, both zero"].to_numpy()
    return bar_totals, on_drug_top


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

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.0), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))
    x_left = x_positions - SUB_BAR_OFFSET
    x_right = x_positions + SUB_BAR_OFFSET

    for ax, drug in zip(axes, DRUGS):
        counts_full = _counts_by_xbin_pattern(df_full, drug)
        counts_single = _counts_by_xbin_pattern(df_single, drug)
        ymax = max(counts_full.sum(axis=1).max(), counts_single.sum(axis=1).max())

        single_totals, single_on_drug = _draw_subbar(
            ax, x_left, counts_single, _muted, ymax
        )
        full_totals, full_on_drug = _draw_subbar(
            ax, x_right, counts_full, lambda c: c, ymax
        )

        # Frame lines — one per sub-bar.
        ax.hlines(single_on_drug, x_left - SUB_BAR_WIDTH / 2, x_left + SUB_BAR_WIDTH / 2,
                  color="dimgray", linewidth=0.8, zorder=3)
        ax.hlines(full_on_drug, x_right - SUB_BAR_WIDTH / 2, x_right + SUB_BAR_WIDTH / 2,
                  color="dimgray", linewidth=0.8, zorder=3)

        # N annotations under each sub-bar (single | full).
        y_pad = -ymax * 0.05
        for x, n_s in zip(x_left, single_totals):
            ax.text(x, y_pad, f"single\n{int(n_s):,}",
                    ha="center", va="top", fontsize=6, color="dimgray")
        for x, n_f in zip(x_right, full_totals):
            ax.text(x, y_pad, f"full\n{int(n_f):,}",
                    ha="center", va="top", fontsize=6, color="dimgray")

        for bx in (0, len(ALL_BIN_LABELS) - 1):
            ax.axvspan(bx - 0.45, bx + 0.45, color="lightgray", alpha=0.25, zorder=0)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day  (left = single-shift, right = full-coverage)")
        ax.set_ylabel("Patient-days")
        ax.set_title(f"{DRUG_LABELS[drug]}  (T = ±{THRESHOLDS[drug]})")

    handles = []
    for label in DOSE_PATTERN_LABELS:
        handles.append(
            plt.Rectangle((0, 0), 1, 1, color=DOSE_PATTERN_COLORS[label], ec="white")
        )
    fig.legend(
        handles, list(DOSE_PATTERN_LABELS),
        loc="lower center", ncol=3, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle(
        "Cohort size by ICU day with 6-group dose-pattern composition (split variant)\n"
        "Each ICU-day position renders LEFT (muted) = single-shift rows (artifact) and "
        "RIGHT (saturated) = full-coverage rows (real diff). Drug-holiday caps each sub-bar.",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.text(
        0.5, 0.005,
        "Left sub-bar = single-shift rows (one shift had 0 hours of coverage; 6-group classification "
        "via amount-sign fallback); right sub-bar = full-coverage rows. Day 0 and last (exit) are "
        "visually dominated by left sub-bars; mid-days by right. See docs/descriptive_figures.md §6.0.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "dose_pattern_6group_count_by_icu_day_split")


if __name__ == "__main__":
    main()
