"""Conditional magnitude of night-day diffs by ICU day (signed pointranges).

Companion to `dose_pattern_6group_count_by_icu_day.png`. The count figure
answers HOW MANY patients are night-heavier or day-heavier on each ICU
day; this figure answers HOW BIG the diff is for those subsets.

For each (ICU day, drug) cell, partitions surviving rows by sign(_dif_*):
  - Night-heavier subset (`_dif_* > 0`): pointrange ABOVE y=0 in red.
  - Day-heavier subset   (`_dif_* < 0`): pointrange BELOW y=0 in blue.
  - Zero-diff rows (drug-holiday + truly-stable, `_dif_* == 0`) are
    excluded — magnitude is trivially 0.

Single-shift days are dropped per the §6.0 governing rule (rate-diff is
undefined). Per-pointrange `n=` annotations let the reader cross-reference
the count figure's bar segments.

Usage:
    uv run python code/descriptive/night_day_diff_signed_iqr_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    apply_style,
    load_exposure,
    prepare_diffs,
    save_fig,
)


MAX_MID_DAY = 7
MID_DAY_BINS = [str(i) for i in range(1, MAX_MID_DAY + 1)] + [f"{MAX_MID_DAY + 1}+"]
ALL_BIN_LABELS = ["0\n(intub)"] + MID_DAY_BINS + ["last\n(exit)"]
RED = "#b2182b"   # night-heavier
BLUE = "#2166ac"  # day-heavier


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
    # §6.0 governing rule: drop single-shift days — rate-diff is undefined.
    df = df.loc[~df["_single_shift_day"].astype(bool)].copy()
    df["_x_bin"] = df.apply(_classify_bin, axis=1)
    df["_x_bin"] = pd.Categorical(df["_x_bin"], categories=ALL_BIN_LABELS, ordered=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6.0), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))

    for ax, drug in zip(axes, DRUGS):
        diff_col = DIFF_COLS[drug]
        # Track y-extent for annotation padding.
        all_q75_pos: list[float] = []
        all_q25_neg: list[float] = []

        per_day_stats = []
        for x, day_bin in zip(x_positions, ALL_BIN_LABELS):
            sub = df.loc[df["_x_bin"] == day_bin, diff_col].dropna()
            pos = sub[sub > 0]
            neg = sub[sub < 0]
            stat = {"x": x, "n_pos": len(pos), "n_neg": len(neg),
                    "med_pos": np.nan, "q25_pos": np.nan, "q75_pos": np.nan,
                    "med_neg": np.nan, "q25_neg": np.nan, "q75_neg": np.nan}
            if len(pos):
                stat["med_pos"] = float(pos.median())
                stat["q25_pos"] = float(pos.quantile(0.25))
                stat["q75_pos"] = float(pos.quantile(0.75))
                all_q75_pos.append(stat["q75_pos"])
            if len(neg):
                stat["med_neg"] = float(neg.median())
                stat["q25_neg"] = float(neg.quantile(0.25))
                stat["q75_neg"] = float(neg.quantile(0.75))
                all_q25_neg.append(stat["q25_neg"])
            per_day_stats.append(stat)

        ymax_pos = max(all_q75_pos) if all_q75_pos else 1.0
        ymin_neg = min(all_q25_neg) if all_q25_neg else -1.0
        y_range = max(ymax_pos, abs(ymin_neg))
        y_pad = y_range * 0.05

        for s in per_day_stats:
            x = s["x"]
            if s["n_pos"]:
                ax.plot([x, x], [s["q25_pos"], s["q75_pos"]], color=RED, lw=2.0,
                        solid_capstyle="round", zorder=3)
                ax.plot(x, s["med_pos"], "o", color=RED, ms=5, zorder=4)
                ax.text(x, s["q75_pos"] + y_pad, f"n={s['n_pos']:,}",
                        ha="center", va="bottom", fontsize=6.0, color=RED)
            if s["n_neg"]:
                ax.plot([x, x], [s["q25_neg"], s["q75_neg"]], color=BLUE, lw=2.0,
                        solid_capstyle="round", zorder=3)
                ax.plot(x, s["med_neg"], "o", color=BLUE, ms=5, zorder=4)
                ax.text(x, s["q25_neg"] - y_pad, f"n={s['n_neg']:,}",
                        ha="center", va="top", fontsize=6.0, color=BLUE)

        ax.axhline(0, color="dimgray", lw=0.6, alpha=0.5, zorder=1)

        for bx in (0, len(ALL_BIN_LABELS) - 1):
            ax.axvspan(bx - 0.45, bx + 0.45, color="lightgray", alpha=0.20, zorder=0)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"Night − day diff ({DRUG_UNITS[drug]})")
        ax.set_title(f"{DRUG_LABELS[drug]}")

    handles = [
        plt.Line2D([0], [0], marker="o", color=RED, lw=2.0,
                   label="Night-heavier (median + IQR)"),
        plt.Line2D([0], [0], marker="o", color=BLUE, lw=2.0,
                   label="Day-heavier (median + IQR)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        "Conditional magnitude of night-day diffs by ICU day\n"
        "Pointrange = median (dot) + IQR (line). Red above 0 = night-heavier subset; "
        "blue below 0 = day-heavier. Single-shift days dropped (§6.0).",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.005,
        "Companion to `dose_pattern_6group_count_by_icu_day.png`: the count figure tells you HOW MANY "
        "patient-days are in each direction; this figure tells you HOW BIG the diff is for those "
        "patient-days. Zero-diff rows (drug-holiday + truly-stable) are excluded since they would always "
        "sit at 0 and obscure the conditional shape.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_signed_iqr_by_icu_day")


if __name__ == "__main__":
    main()
