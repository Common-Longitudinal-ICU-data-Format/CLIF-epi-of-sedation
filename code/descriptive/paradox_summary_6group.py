"""6-group paradox summary, one row per sedative.

Headline figure for the night-vs-day diurnal paradox. Rebuilt under the Path
B++ refactor to actually use the 6-way `categorize_diff_6way` classification:
the previous version was a cohort-level histogram + sign-bar + gross-sum-bar
that hid the group structure. This rebuild colors every panel by the same
6-group axis used by the rest of the descriptive layer.

Three panels per sedative (3 rows × 3 cols):

  Panel A — Distribution shape, group-stacked.
      Stacked histogram of `*_dif_*` colored by 6-group membership.
      Vertical lines: mean (solid green), median (dashed orange),
      0 (dashed black), and ±T (dashed red) — both threshold sides drawn so
      the 6-group cutpoints are visually anchored. Top-right annotation
      reports `mean = X.XX | median = Y.YY` and the count of single-shift
      rows that contributed via amount-sign fallback.

  Panel B — Sign distribution via 6-way classification.
      Single horizontal stacked bar with all 6 segments colored by
      `DOSE_PATTERN_COLORS`. Replaces the previous 3-segment bar that was
      blind to off-drug holidays vs. truly-stable days. Single-shift
      contribution is shown as a hatched overlay on each segment.

  Panel C — Gross-sum tail balance, decomposed by group contribution.
      Two stacked bars (Night > Day side, Day > Night side). Each bar
      internally stacked by the 2 groups that contribute to that direction
      (e.g. Night > Day = "Slightly higher at night" + "Markedly higher at
      night"). Hatched overlay = single-shift portion. The two `Equal`
      groups don't appear here (they don't contribute to either tail).

Reads `exposure_dataset.parquet` (full hospital-stay coverage including
day 0 and last day). Saves to
`output_to_share/{site}/descriptive/paradox_summary_6group.png`.

Usage:
    uv run python code/descriptive/paradox_summary_6group.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DAY_COLS,
    DIFF_COLS,
    DOSE_PATTERN_COLORS,
    DOSE_PATTERN_LABELS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    NIGHT_COLS,
    THRESHOLDS,
    apply_style,
    categorize_diff_6way,
    load_exposure,
    prepare_diffs,
    save_fig,
)


# Groups that show up in the tail-balance panel (Equal-{both-zero, both-non-zero}
# don't contribute to either gross sum).
_DAY_TAIL_GROUPS = ("Markedly higher at day", "Slightly higher at day")
_NIGHT_TAIL_GROUPS = ("Slightly higher at night", "Markedly higher at night")


def _classify(df: pd.DataFrame, drug: str) -> pd.Series:
    """Apply the 6-way classifier per row using rate-diff with NaN→0 fallback."""
    return pd.Series(
        categorize_diff_6way(
            df[DIFF_COLS[drug]],
            df[DAY_COLS[drug]],
            df[NIGHT_COLS[drug]],
            THRESHOLDS[drug],
        ),
        index=df.index,
    )


def _panel_a_stacked_histogram(ax, df: pd.DataFrame, drug: str) -> None:
    color = "dimgray"
    diff_col = DIFF_COLS[drug]
    thr = THRESHOLDS[drug]
    series = df[diff_col]

    # Drop the "Equal, both zero" rows from the histogram — they would all
    # collapse onto a single x=0 spike that drowns the rest of the shape.
    # Their count is reported in the annotation instead.
    cat = df["_pattern"]
    drawable_mask = cat != "Equal, both zero"
    drawable = series[drawable_mask].dropna()
    if len(drawable) == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    lo, hi = np.percentile(drawable, [1, 99])
    bins = np.linspace(lo, hi, 61)

    # Stacked histogram: list-of-arrays + matching color list, oldest group at
    # the bottom of the stack so the negative tail sits on the left base and
    # the positive tail rises through it.
    stack_arrays = []
    stack_colors = []
    stack_labels = []
    for label in DOSE_PATTERN_LABELS:
        if label == "Equal, both zero":
            continue
        sub = df.loc[drawable_mask & (cat == label), diff_col].dropna()
        sub = sub[(sub >= lo) & (sub <= hi)]
        stack_arrays.append(sub.to_numpy())
        stack_colors.append(DOSE_PATTERN_COLORS[label])
        stack_labels.append(label)
    ax.hist(stack_arrays, bins=bins, color=stack_colors, stacked=True,
            edgecolor="white", linewidth=0.2, label=stack_labels)

    # Reference lines.
    finite = series.dropna().to_numpy()
    mean = float(np.mean(finite)) if len(finite) else float("nan")
    median = float(np.median(finite)) if len(finite) else float("nan")
    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.axvline(mean, color="green", linestyle="-", linewidth=1.6,
               label=f"Mean = {mean:+.3f}")
    ax.axvline(median, color="darkorange", linestyle="--", linewidth=1.4,
               label=f"Median = {median:+.3f}")
    # ±T threshold lines (the 6-group cutpoints visualized).
    if lo <= thr <= hi:
        ax.axvline(thr, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    if lo <= -thr <= hi:
        ax.axvline(-thr, color="red", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.99, 0.95,
            f"T = ±{thr} {DRUG_UNITS[drug]}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="red")

    n_dropped = int((cat == "Equal, both zero").sum())
    n_single = int(df["_single_shift_day"].fillna(False).sum())
    n_total = len(df)
    ax.text(
        0.99, 0.85,
        f"n total = {n_total:,}\n"
        f"n excluded (both-zero) = {n_dropped:,}\n"
        f"n single-shift = {n_single:,}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=7.5, color="dimgray", family="monospace",
    )

    ax.legend(loc="upper left", fontsize=7)
    ax.set_xlabel(f"Night − day rate ({DRUG_UNITS[drug]})")
    ax.set_ylabel("Patient-days")
    ax.set_title(f"{DRUG_LABELS[drug]} — distribution by group", fontsize=10)
    _ = color  # keep linter quiet; reserved for fallback per-drug accent


def _panel_b_sign_distribution(ax, df: pd.DataFrame, drug: str) -> None:
    cat = df["_pattern"]
    n = int(cat.notna().sum())
    if n == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    cum = 0.0
    for label in DOSE_PATTERN_LABELS:
        mask = cat == label
        n_label = int(mask.sum())
        pct = 100.0 * n_label / n
        if pct == 0:
            continue
        n_single_label = int((mask & df["_single_shift_day"].fillna(False)).sum())
        pct_single = 100.0 * n_single_label / n
        color = DOSE_PATTERN_COLORS[label]
        # Solid (non-partial) base segment.
        ax.barh(0, pct - pct_single, left=cum, color=color, edgecolor="white",
                linewidth=1.5, height=0.55)
        # Hatched overlay = single-shift contribution sitting on the right
        # edge of this segment.
        if pct_single > 0:
            ax.barh(0, pct_single, left=cum + (pct - pct_single),
                    color=color, edgecolor="black", linewidth=0.0,
                    height=0.55, hatch="//", alpha=0.85)
        # Label only if segment wide enough to read.
        if pct >= 5.0:
            ax.text(cum + pct / 2, 0, f"{label.split(',')[0]}\n{pct:.1f}%",
                    ha="center", va="center",
                    color="white" if color in ("#2166ac", "#b2182b") else "black",
                    fontsize=7.5)
        cum += pct

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_xlabel("% of patient-days")
    ax.set_yticks([])
    ax.set_title(f"{DRUG_LABELS[drug]} — sign split (6 groups; // = single-shift)",
                 fontsize=10)


def _panel_c_tail_balance(ax, df: pd.DataFrame, drug: str) -> None:
    cat = df["_pattern"]
    diff_col = DIFF_COLS[drug]
    n = int(cat.notna().sum())
    if n == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    # For each tail (positive and negative), stack the per-group |sum|
    # contribution; hatched portion = single-shift slice.
    def _stack(x_pos: float, group_seq: tuple[str, ...]) -> float:
        cum = 0.0
        for label in group_seq:
            mask = cat == label
            if not mask.any():
                continue
            vals = df.loc[mask, diff_col].dropna()
            partial_mask = mask & df["_single_shift_day"].fillna(False)
            partial_vals = df.loc[partial_mask, diff_col].dropna()
            seg = float(np.abs(vals).sum() / n) if n else 0.0
            seg_partial = float(np.abs(partial_vals).sum() / n) if n else 0.0
            color = DOSE_PATTERN_COLORS[label]
            seg_solid = max(seg - seg_partial, 0.0)
            if seg_solid > 0:
                ax.bar(x_pos, seg_solid, bottom=cum, color=color, edgecolor="white",
                       linewidth=0.6, width=0.55)
            if seg_partial > 0:
                ax.bar(x_pos, seg_partial, bottom=cum + seg_solid, color=color,
                       edgecolor="black", linewidth=0.0, width=0.55, hatch="//")
            cum += seg
        return cum

    night_total = _stack(0, _NIGHT_TAIL_GROUPS)
    day_total = _stack(1, _DAY_TAIL_GROUPS)
    net = night_total - day_total

    ax.text(0, night_total, f"{night_total:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(1, day_total, f"{day_total:.2f}", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([
        "Night > Day side\nΣ|diff| / n_total",
        "Day > Night side\nΣ|diff| / n_total",
    ])
    ax.set_ylabel(f"Per-patient-day avg |diff| ({DRUG_UNITS[drug]})")
    ax.set_title(
        f"{DRUG_LABELS[drug]} — directional balance  (net = mean ≈ {net:+.3f})",
        fontsize=10,
    )


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    if "_single_shift_day" not in df.columns:
        raise RuntimeError(
            "exposure_dataset.parquet missing _single_shift_day — "
            "re-run code/05_modeling_dataset.py against the current site."
        )

    fig, axes = plt.subplots(
        len(DRUGS), 3, figsize=(15, 4.0 * len(DRUGS) + 1.0),
    )

    for r, drug in enumerate(DRUGS):
        sub = df.copy()
        sub["_pattern"] = _classify(sub, drug)
        _panel_a_stacked_histogram(axes[r, 0], sub, drug)
        _panel_b_sign_distribution(axes[r, 1], sub, drug)
        _panel_c_tail_balance(axes[r, 2], sub, drug)

    fig.suptitle(
        "Paradox summary (6-group): distribution, sign split, directional balance — per sedative\n"
        "diff = night-rate − day-rate; T = drug-specific clinical threshold; "
        "// hatched = single-shift rows (one shift had 0 hours of coverage). "
        "Panel C bar height = per-patient-day Σ|diff| from one side of 0; net of the two bars ≈ cohort mean.",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.01,
        "Panel A — stacked histogram by 6-group; mean (green) vs median (orange) on opposite sides of 0 = "
        "the paradox-detection mechanism. Panel B — full 6-segment sign distribution; hatched slice within "
        "each segment = single-shift (coverage-artifact) contribution. Panel C — directional balance: "
        "each bar's height = per-patient-day average |diff| contributed by rows on that side of 0; "
        "the gap between the two bars = cohort mean of diff. Both bars tall and equal = fat opposing tails = "
        "paradox mechanism (mean is the small residual). Glossary: docs/descriptive_figures.md §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "paradox_summary_6group")


if __name__ == "__main__":
    main()
