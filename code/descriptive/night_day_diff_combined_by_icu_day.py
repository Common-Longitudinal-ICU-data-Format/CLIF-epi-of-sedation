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
    ON_DRUG_FLAGS,
    apply_style,
    load_model_input,
    save_csv,
    save_fig,
)


MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = [str(i) for i in range(MIN_DAY, MAX_DAY + 1)]
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


def _build_stats_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Long-format per-(drug, nth_day) stats for federated cross-site pooling.

    Reads from the already-filtered full-24h day-1..7 frame. Emits one row
    per (drug, nth_day) with counts + means + sums-of-squares — enough for
    Wilson CIs on proportions, Student-t CIs on the mean, and fixed-effects
    pooling across sites at agg time without re-reading PHI.

    On-drug definition: `_X_any = 1` (hurdle flag from 05_modeling_dataset.py).
    """
    rows: list[dict] = []
    for drug in DRUGS:
        diff_col = DIFF_COLS[drug]
        on_flag = ON_DRUG_FLAGS[drug]
        for day_bin in DAY_BINS:
            cell = df.loc[df["_x_bin"] == day_bin]
            diff_all = cell[diff_col].fillna(0.0).to_numpy()
            on_drug_mask = cell[on_flag].fillna(0).astype(int).to_numpy() == 1
            diff_on = diff_all[on_drug_mask]

            n_full_24h = int(len(diff_all))
            n_on_drug = int(on_drug_mask.sum())
            n_gt0_all = int((diff_all > 0).sum())
            n_gt0_on = int((diff_on > 0).sum())

            mean_all = float(diff_all.mean()) if n_full_24h else float("nan")
            sd_all = float(diff_all.std(ddof=1)) if n_full_24h > 1 else float("nan")
            mean_on = float(diff_on.mean()) if n_on_drug else float("nan")
            sd_on = float(diff_on.std(ddof=1)) if n_on_drug > 1 else float("nan")
            sum_all = float(diff_all.sum())
            sum_sq_all = float((diff_all * diff_all).sum())

            rows.append({
                "drug": drug,
                "nth_day": int(day_bin),
                "n_full_24h": n_full_24h,
                "n_on_drug": n_on_drug,
                "n_diff_gt_0_all": n_gt0_all,
                "n_diff_gt_0_on_drug": n_gt0_on,
                "mean_diff_all": mean_all,
                "sd_diff_all": sd_all,
                "mean_diff_on_drug": mean_on,
                "sd_diff_on_drug": sd_on,
                "sum_diff_all": sum_all,
                "sum_sq_diff_all": sum_sq_all,
            })
    return pd.DataFrame(rows)


def _draw_mean_panel(ax, df, drug, x_positions):
    col = DIFF_COLS[drug]
    color = DRUG_COLORS[drug]

    means, lows, highs, medians, ns = [], [], [], [], []
    pcts_n_higher, pcts_d_higher = [], []
    for b in DAY_BINS:
        sub = df[df["_x_bin"] == b]
        vals = sub[col].dropna().to_numpy()
        m, lo, hi = _ci(vals)
        means.append(m); lows.append(lo); highs.append(hi)
        medians.append(float(np.median(vals)) if len(vals) else float("nan"))
        ns.append(len(vals))
        n_pct, d_pct = _direction_pcts(vals)
        pcts_n_higher.append(n_pct); pcts_d_higher.append(d_pct)

    means_a = np.array(means)
    lows_a = np.array(lows)
    highs_a = np.array(highs)
    medians_a = np.array(medians)

    ax.plot(x_positions, means_a, marker="o", color=color, linewidth=2, label="Mean")
    ax.plot(x_positions, medians_a, marker="s", color=color, linewidth=1.2,
            linestyle="--", alpha=0.7, label="Median")
    ax.fill_between(x_positions, lows_a, highs_a, color=color, alpha=0.25, label="95% CI")

    for xi, m, n_pct, d_pct in zip(x_positions, means_a, pcts_n_higher, pcts_d_higher):
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
    for x, day_bin in zip(x_positions, DAY_BINS):
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
    df = load_model_input()
    # Restrict to full-24h ICU days 1..7. Drops day 0 (first_partial),
    # the trajectory-final partial day, single-shift rows, and days 8+
    # in one filter — replaces the legacy
    # `~_single_shift_day + _classify_bin + boundary-day` complexity.
    in_range = df["_nth_day"].between(MIN_DAY, MAX_DAY)
    df = df.loc[df["_is_full_24h_day"] & in_range].copy()
    df["_x_bin"] = pd.Categorical(
        df["_nth_day"].astype(int).astype(str),
        categories=DAY_BINS, ordered=True,
    )

    # Federated-pooling artifact: long-format per-(drug, nth_day) stats
    # at output_to_share/{site}/descriptive/. Cross-site code reads this
    # CSV (never the raw parquet) and recomputes Wilson / Student-t CIs
    # + fixed-effects pooled estimates at agg time. See .dev/CLAUDE.md
    # "Federation contract".
    stats_df = _build_stats_csv(df)
    save_csv(stats_df, "night_day_dose_stats_by_icu_day")

    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), sharex="col")
    x_positions = np.arange(len(DAY_BINS))

    for col_idx, drug in enumerate(DRUGS):
        _draw_mean_panel(axes[0, col_idx], df, drug, x_positions)
        _draw_iqr_panel(axes[1, col_idx], df, drug, x_positions)

    for ax in axes.flatten():
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DAY_BINS)

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
        "Night-minus-day dose rate by ICU day 1–7 (full-24h coverage) "
        "— mean (top) + signed IQR (bottom)",
        fontsize=11, y=0.99,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.10)
    fig.text(
        0.5, 0.005,
        "TOP: mean ± 95% CI line per ICU day. N>D / D>N annotations = "
        "patient-day percent in each direction. "
        "BOTTOM: median + IQR pointranges partitioned by sign — red above 0 = night-heavier; blue below 0 = day-heavier. "
        "Cohort: full-24h ICU-day rows from `model_input_by_id_imvday.parquet` "
        "(`_is_full_24h_day = TRUE AND _nth_day BETWEEN 1 AND 7`). "
        "Day 0 partial intubation day, trajectory-final partial day, single-shift rows, "
        "and days 8+ are dropped at the load filter so each bin reflects a "
        "fully-comparable 12+12 hr coverage population. "
        "See docs/descriptive_figures.md §6.0.",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "night_day_diff_combined_by_icu_day")


if __name__ == "__main__":
    main()
