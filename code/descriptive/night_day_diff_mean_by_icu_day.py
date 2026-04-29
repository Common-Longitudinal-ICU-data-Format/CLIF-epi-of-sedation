"""Mean night-minus-day dose rate by ICU day, with day 0 + last-day boundary days highlighted.

Shows whether the diurnal dose pattern is concentrated at the boundaries of
the vent course (day 0 = intubation; "last" = day before extubation) or
distributed across mid-stay days. X-axis: `0 (intub), 1, 2, ..., 7, 8+, last (exit)`.

Day 0 and "last" are stratified into two markers each:

  - **Open circle (complete coverage)** — `_single_shift_day = False`, i.e.
    both day-shift and night-shift had nonzero hours of exposure. Honest
    boundary-day signal.
  - **Filled triangle (partial coverage)** — `_single_shift_day = True`,
    i.e. one shift had zero hours of exposure (e.g., intubated after 7 PM
    so day-shift hours = 0 on day 0). The diff in that subset is mechanically
    `night − 0 = full_night_rate`, which is a coverage artifact, NOT a real
    titration. Plotted alpha-reduced with an asterisk footnote so the
    visual distance between the two markers literally shows the size of the
    bias.

Mid-stay days (1..7+) keep the original mean ± 95% CI line + median overlay,
plus a `% Night > Day / % Day > Night` annotation under each marker so the
mean's tug-of-war with the per-day "majority direction" is visible.

A thin gray band shades day 0 and "last" to visually separate the boundary
days from the steady-state middle.

Usage:
    uv run python code/descriptive/night_day_diff_mean_by_icu_day.py
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


# X-axis layout: boundary day 0 at position 0, mid-stay days at positions
# 1..MAX_MID_DAY, "8+" tail at MAX_MID_DAY+1, last-day at MAX_MID_DAY+2.
MAX_MID_DAY = 7
MID_DAY_BINS = [str(i) for i in range(1, MAX_MID_DAY + 1)] + [f"{MAX_MID_DAY + 1}+"]
ALL_BIN_LABELS = ["0\n(intub)"] + MID_DAY_BINS + ["last\n(exit)"]


def _ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using Student-t interval. Assumes n > 1."""
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
    """Map (_nth_day, _is_first_day, _is_last_day) → x-axis bin label.

    Last-day takes precedence over the day-number bin: a row that is BOTH
    `_nth_day = 5` AND `_is_last_day = True` falls in the "last" column,
    not the "5" column. Day 0 takes precedence over last only if the
    patient's stay was exactly one day (intubation + extubation same partial
    period); those rare rows fall in "0".
    """
    if bool(row["_is_first_day"]):
        return "0\n(intub)"
    if bool(row["_is_last_day"]):
        return "last\n(exit)"
    d = int(row["_nth_day"])
    if d > MAX_MID_DAY:
        return f"{MAX_MID_DAY + 1}+"
    return str(d)


def _direction_pcts(values: np.ndarray) -> tuple[float, float]:
    """Return (% Night > Day, % Day > Night) on a finite-only array, percent."""
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


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    # Make sure the flag columns exist (added in 05_modeling_dataset.py's
    # exposure cell). If they don't, the user is on an old build.
    for col in ["_is_first_day", "_is_last_day", "_single_shift_day"]:
        if col not in df.columns:
            raise RuntimeError(
                f"exposure_dataset.parquet missing column {col!r} — "
                f"re-run code/05_modeling_dataset.py against the current site."
            )
    df = df.copy()
    df["_x_bin"] = df.apply(_classify_bin, axis=1)
    df["_x_bin"] = pd.Categorical(df["_x_bin"], categories=ALL_BIN_LABELS, ordered=True)

    # Tall figure — bottom ~32% reserved for the inline footnote.
    fig, axes = plt.subplots(1, 3, figsize=(16, 9.5), sharex=True)
    x_positions = np.arange(len(ALL_BIN_LABELS))
    boundary_idx_first = 0
    boundary_idx_last = len(ALL_BIN_LABELS) - 1

    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        color = DRUG_COLORS[drug]

        # Background band on day 0 and "last" so the boundary days are
        # visually distinct from the stationary mid-stay region.
        for bx in (boundary_idx_first, boundary_idx_last):
            ax.axvspan(bx - 0.45, bx + 0.45, color="lightgray", alpha=0.20, zorder=0)

        means, lows, highs, medians, ns = [], [], [], [], []
        pcts_n_higher, pcts_d_higher = [], []
        for b in ALL_BIN_LABELS:
            sub = df[df["_x_bin"] == b]
            # For mid-stay days, single-shift rows are negligible (they're
            # essentially boundary-day-only artifacts); we still keep them
            # to match the exposure-view N exactly.
            vals = sub[col].dropna().to_numpy()
            m, lo, hi = _ci(vals)
            means.append(m)
            lows.append(lo)
            highs.append(hi)
            medians.append(float(np.median(vals)) if len(vals) else float("nan"))
            ns.append(len(vals))
            n_pct, d_pct = _direction_pcts(vals)
            pcts_n_higher.append(n_pct)
            pcts_d_higher.append(d_pct)

        # Mid-stay (positions 1..len-2): line + 95% CI ribbon + median overlay.
        mid_x = x_positions[1:-1]
        mid_means = np.array(means)[1:-1]
        mid_lows = np.array(lows)[1:-1]
        mid_highs = np.array(highs)[1:-1]
        mid_medians = np.array(medians)[1:-1]

        ax.plot(mid_x, mid_means, marker="o", color=color, linewidth=2, label="Mean")
        ax.plot(mid_x, mid_medians, marker="s", color=color, linewidth=1.2,
                linestyle="--", alpha=0.7, label="Median")
        ax.fill_between(mid_x, mid_lows, mid_highs, color=color, alpha=0.25, label="95% CI")

        # Boundary days (day 0 and "last"): plot complete-coverage and
        # partial-coverage subsets as separate markers at the same x position.
        for bx, label_for_b in [(boundary_idx_first, "0\n(intub)"),
                                (boundary_idx_last, "last\n(exit)")]:
            sub = df[df["_x_bin"] == label_for_b]
            full = sub[~sub["_single_shift_day"].fillna(False)][col].dropna().to_numpy()
            part = sub[sub["_single_shift_day"].fillna(False)][col].dropna().to_numpy()

            if len(full):
                m_full, lo_full, hi_full = _ci(full)
                ax.errorbar(
                    bx, m_full, yerr=[[m_full - lo_full], [hi_full - m_full]],
                    fmt="o", mfc="white", mec=color, mew=1.6, ecolor=color,
                    capsize=3, label="Mean (complete cov.)" if drug == DRUGS[0] else None,
                )
                ax.annotate(
                    f"n={len(full):,}", xy=(bx, m_full), xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=7, color="dimgray",
                )
            if len(part):
                m_part = float(np.mean(part))
                ax.scatter(
                    bx, m_part, marker="^", s=70, color=color, alpha=0.45,
                    edgecolor="dimgray", linewidth=0.8,
                    label="Mean (partial cov.*)" if drug == DRUGS[0] else None,
                )
                ax.annotate(
                    f"n={len(part):,}*", xy=(bx, m_part), xytext=(0, -14),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="dimgray",
                )

        # Mid-stay direction-pct annotation under each marker.
        for xi, m, n_pct, d_pct in zip(mid_x, mid_means, pcts_n_higher[1:-1], pcts_d_higher[1:-1]):
            ax.annotate(
                f"N>D {n_pct:.0f}%\nD>N {d_pct:.0f}%",
                xy=(xi, m), xytext=(0, -28),
                textcoords="offset points", ha="center", va="top",
                fontsize=6.5, color="dimgray",
            )

        # Mid-stay N-per-bin annotation above each marker.
        for xi, m, n in zip(mid_x, mid_means, ns[1:-1]):
            ax.annotate(
                f"n={n:,}", xy=(xi, m), xytext=(0, 8),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=7, color="dimgray",
            )

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(ALL_BIN_LABELS)
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"{DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_title(DRUG_LABELS[drug])
        if drug == DRUGS[0]:
            ax.legend(loc="upper right", fontsize=7.5)

    fig.suptitle(
        "Mean night-minus-day dose rate by ICU day\n"
        "Mid-stay days: mean ± 95% CI line + dashed median. "
        "Boundary days (0 / last): open circle = complete cov.; filled triangle = partial cov. (one shift had 0h coverage).",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()
    fig.text(
        0.5, -0.02,
        "Annotations: N>D / D>N = patient-day percent in each direction (Equal / NaN excluded). "
        "n=N (n=N*) = patient-day count per marker; * marks partial-coverage subset. "
        "Boundary-day partial-cov triangles reflect coverage artifacts (e.g. intubated after 7 PM "
        "→ diff = night − 0 = full_night_rate), NOT a clinically interpretable diurnal pattern. "
        "Glossary in docs/descriptive_figures.md §3.",
        ha="center", va="top", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_mean_by_icu_day")


if __name__ == "__main__":
    main()
