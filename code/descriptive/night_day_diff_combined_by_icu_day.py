"""Combined night-day diff figure: mean + signed-IQR panels stacked.

Replaces the two separate scripts (`night_day_diff_mean_by_icu_day.py` and
`night_day_diff_signed_iqr_by_icu_day.py`) with a single 2×3 figure where
both views sit on the same x-axis for direct visual comparison:

  - **Top row** (one panel per drug): mean diff line + 95% CI ribbon +
    dashed median overlay for mid-stay days; open-circle errorbar for
    boundary days (0 / last). Per-day `N>D / D>N` direction-percent
    annotation under each mid-stay marker.

  - **Bottom row** (one panel per drug): signed-IQR pointranges
    partitioned by sign — red (median + IQR) above y=0 for the
    night-heavier subset (`_dif_* > 0`), blue below y=0 for the
    day-heavier subset (`_dif_* < 0`). Per-day `n=` annotations above
    and below each pointrange.

Both panel families operate on the SAME row set (single-shift days
dropped per the §6.0 governing rule), so the two panels are always
reading the same population. Cohort attrition is monotonic across days
because the x-bins use `_rel_day = _nth_day − MIN(_nth_day) per hosp`.

Usage:
    uv run python code/descriptive/night_day_diff_combined_by_icu_day.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_COLORS,
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


def _ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf((1 + conf) / 2, df=n - 1))
    return mean, mean - crit * se, mean + crit * se


def _classify_bin(row: pd.Series) -> str:
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    d = int(row["_rel_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def _direction_pcts(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    finite = values[np.isfinite(values)]
    n_finite = len(finite)
    if n_finite == 0:
        return 0.0, 0.0
    pct_night_higher = 100.0 * (finite > 0).sum() / n_finite
    pct_day_higher = 100.0 * (finite < 0).sum() / n_finite
    return pct_night_higher, pct_day_higher


def _draw_mean_panel(ax, df, drug, x_positions, boundary_idx_first, boundary_idx_last):
    col = DIFF_COLS[drug]
    color = DRUG_COLORS[drug]

    means, lows, highs, medians, ns = [], [], [], [], []
    pcts_n_higher, pcts_d_higher = [], []
    for b in ALL_BIN_LABELS:
        sub = df[df["_x_bin"] == b]
        vals = sub[col].dropna().to_numpy()
        m, lo, hi = _ci(vals)
        means.append(m); lows.append(lo); highs.append(hi)
        medians.append(float(np.median(vals)) if len(vals) else float("nan"))
        ns.append(len(vals))
        n_pct, d_pct = _direction_pcts(vals)
        pcts_n_higher.append(n_pct); pcts_d_higher.append(d_pct)

    mid_x = x_positions[1:-1]
    mid_means = np.array(means)[1:-1]
    mid_lows = np.array(lows)[1:-1]
    mid_highs = np.array(highs)[1:-1]
    mid_medians = np.array(medians)[1:-1]

    ax.plot(mid_x, mid_means, marker="o", color=color, linewidth=2, label="Mean")
    ax.plot(mid_x, mid_medians, marker="s", color=color, linewidth=1.2,
            linestyle="--", alpha=0.7, label="Median")
    ax.fill_between(mid_x, mid_lows, mid_highs, color=color, alpha=0.25, label="95% CI")

    for bx, label_for_b in [(boundary_idx_first, "0\n(intub)"),
                            (boundary_idx_last, "last\n(exit)")]:
        sub = df[df["_x_bin"] == label_for_b]
        vals = sub[col].dropna().to_numpy()
        if len(vals):
            m_b, lo_b, hi_b = _ci(vals)
            ax.errorbar(
                bx, m_b, yerr=[[m_b - lo_b], [hi_b - m_b]],
                fmt="o", mfc="white", mec=color, mew=1.6, ecolor=color,
                capsize=3, label="Mean (boundary)" if drug == DRUGS[0] else None,
            )

    for xi, m, n_pct, d_pct in zip(mid_x, mid_means, pcts_n_higher[1:-1], pcts_d_higher[1:-1]):
        ax.annotate(
            f"N>D {n_pct:.0f}%\nD>N {d_pct:.0f}%",
            xy=(xi, m), xytext=(0, -22),
            textcoords="offset points", ha="center", va="top",
            fontsize=6.5, color="dimgray",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(f"Mean diff ({DRUG_UNITS[drug]})")
    ax.set_title(DRUG_LABELS[drug])


def _draw_iqr_panel(ax, df, drug, x_positions):
    diff_col = DIFF_COLS[drug]
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
    ax.set_ylabel(f"Signed diff ({DRUG_UNITS[drug]})")
    ax.set_xlabel("ICU day")


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    for col in ["_is_first_day", "_is_last_day", "_single_shift_day", "_rel_day"]:
        if col not in df.columns:
            raise RuntimeError(
                f"exposure_dataset.parquet missing column {col!r} — "
                f"re-run code/05_modeling_dataset.py against the current site."
            )
    # §6.0 governing rule: drop single-shift days — diff undefined.
    df = df.loc[~df["_single_shift_day"].astype(bool)].copy()
    df["_x_bin"] = df.apply(_classify_bin, axis=1)
    df["_x_bin"] = pd.Categorical(df["_x_bin"], categories=ALL_BIN_LABELS, ordered=True)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), sharex="col")
    x_positions = np.arange(len(ALL_BIN_LABELS))
    boundary_idx_first = 0
    boundary_idx_last = len(ALL_BIN_LABELS) - 1

    for col_idx, drug in enumerate(DRUGS):
        _draw_mean_panel(axes[0, col_idx], df, drug, x_positions,
                         boundary_idx_first, boundary_idx_last)
        _draw_iqr_panel(axes[1, col_idx], df, drug, x_positions)

    for ax in axes.flatten():
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)

    # Top row: legend on first subplot only.
    axes[0, 0].legend(loc="upper right", fontsize=7.5)

    # Bottom row: separate proxy legend for the IQR pointranges.
    iqr_handles = [
        plt.Line2D([0], [0], marker="o", color=RED, lw=2.0,
                   label="Night-heavier (median + IQR)"),
        plt.Line2D([0], [0], marker="o", color=BLUE, lw=2.0,
                   label="Day-heavier (median + IQR)"),
    ]
    axes[1, 0].legend(handles=iqr_handles, loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Night-minus-day dose rate by ICU day — mean (top) + signed IQR (bottom)\n"
        "Single-shift days dropped per §6.0; x-bins use _rel_day for monotonic cohort attrition.",
        fontsize=11, y=0.99,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.text(
        0.5, 0.005,
        "TOP: mean ± 95% CI line for mid-stay; open-circle errorbar for boundary days. "
        "N>D / D>N annotations = patient-day percent in each direction. "
        "BOTTOM: median + IQR pointranges partitioned by sign — red above 0 = night-heavier; blue below 0 = day-heavier. "
        "Cohort: patient-days from the qualifying first IMV streak ≥ 24h with both shifts > 0 hours of coverage. "
        "Single-shift days dropped (rate-diff is undefined). See docs/descriptive_figures.md §6.0.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_combined_by_icu_day")


if __name__ == "__main__":
    main()
