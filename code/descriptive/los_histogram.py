"""Per-site LOS histogram + federated bin-counts CSV.

Reads `output/{site}/cohort_meta_by_id.parquet` and emits:

  - `output_to_share/{site}/descriptive/los_histogram_bins.csv` —
        long-format bin counts. Columns: `(metric, stratum,
        exit_mechanism, bin_left, bin_right, count)`. Federated-safe
        (group-level counts only). Two metrics × two strata:

          metric ∈ {n_days_full_24h, imv_duration_days}
          stratum = 'overall'              → one row per bin, exit_mechanism='__all__'
          stratum = 'by_exit_mechanism'    → one row per (mechanism × bin)

        Bin grid: width = 1 day, range [0, 30) + a single `30+` overflow
        bin (bin_left=30, bin_right=inf). 31 bins per stratum row group.
        Both strata share the same grid so cross-site / cross-mechanism
        pooling is just a sum-of-counts.

  - `output_to_share/{site}/descriptive/los_summary_stats.csv` —
        long-format `(metric, stratum, exit_mechanism, n, mean, sd,
        median, q10, q25, q75, q90, min, max)`. Same `(metric, stratum,
        exit_mechanism)` keying as the bins file. Federated-safe; agg
        script uses Hansen pooled-mean/SD for cross-site rollup
        (mirrors `code/agg/pool_table1.py:71-95`).

  - `output_to_share/{site}/descriptive/los_histogram_overall.png` —
        per-site overall LOS histogram for at-a-glance review.
  - `output_to_share/{site}/descriptive/los_histogram_by_exit.png` —
        per-site stacked histogram colored by exit_mechanism.

Usage:
    uv run python code/descriptive/los_histogram.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    SITE_NAME,
    apply_style,
    save_csv,
    save_fig,
)


# ── Bin grid ──────────────────────────────────────────────────────────────
# 31 bins: [0,1), [1,2), ..., [29,30), [30, +inf). The overflow tail
# matters for IMV duration (long-stay patients can hit 60+ days; the
# grid stays compact while preserving the count). bin_left/bin_right
# are stored as floats so the +inf edge round-trips cleanly through CSV.
_BIN_EDGES = list(range(31)) + [float("inf")]  # 32 edges → 31 bins


# ── Mechanism palette ─────────────────────────────────────────────────────
# Diverging colors that read in clinical-narrative order (terminal events
# warm; recovery cool). Matches the row order used in 06_table1.py's
# tableone `order` arg so the histogram and Table 1 show categories in
# the same sequence.
_EXIT_PALETTE: dict[str, str] = {
    "tracheostomy":          "#8c2d04",  # dark orange
    "died_on_imv":           "#cb181d",  # dark red
    "palliative_extubation": "#f4a582",  # peach
    "failed_extubation":     "#fdae61",  # warm yellow-orange
    "successful_extubation": "#4393c3",  # blue
    "discharge_on_imv":      "#9e9e9e",  # gray
    "unknown":               "#000000",  # black (sentinel — should be 0)
}
_EXIT_ORDER = list(_EXIT_PALETTE.keys())


def _bin_series(s: pd.Series) -> pd.Series:
    """Return integer bin index per value (0..30; 30 = overflow `30+`)."""
    # pd.cut with labels=False gives integer bin indices. Treat values
    # >= 30 as bin 30 (the overflow tail). NaN stays NaN and is dropped.
    s = s.dropna().astype(float)
    idx = pd.cut(s, bins=_BIN_EDGES, right=False, labels=False, include_lowest=True)
    return idx.astype("Int64")


def _bins_long(
    df: pd.DataFrame,
    metric: str,
    metric_col: str,
) -> pd.DataFrame:
    """Build the long bins frame for one metric (overall + by_exit_mechanism)."""
    rows: list[dict] = []

    # Strata 1: overall. One row per bin; exit_mechanism = '__all__'.
    bin_idx_all = _bin_series(df[metric_col])
    counts_all = bin_idx_all.value_counts().reindex(range(31), fill_value=0)
    for b, c in counts_all.items():
        bin_left = float(_BIN_EDGES[b])
        bin_right = float(_BIN_EDGES[b + 1])  # +inf for overflow
        rows.append({
            "metric": metric,
            "stratum": "overall",
            "exit_mechanism": "__all__",
            "bin_left": bin_left,
            "bin_right": bin_right,
            "count": int(c),
        })

    # Strata 2: by_exit_mechanism. One row per (mechanism × bin), even
    # when the count is 0 — keeps the row grid stable across sites so
    # the agg sum-pool is just a vector add, never a re-index.
    for mech in _EXIT_ORDER:
        sub = df[df["exit_mechanism"] == mech]
        bin_idx = _bin_series(sub[metric_col])
        counts = bin_idx.value_counts().reindex(range(31), fill_value=0)
        for b, c in counts.items():
            bin_left = float(_BIN_EDGES[b])
            bin_right = float(_BIN_EDGES[b + 1])
            rows.append({
                "metric": metric,
                "stratum": "by_exit_mechanism",
                "exit_mechanism": mech,
                "bin_left": bin_left,
                "bin_right": bin_right,
                "count": int(c),
            })

    return pd.DataFrame(rows)


def _summary_stats_long(
    df: pd.DataFrame,
    metric: str,
    metric_col: str,
) -> pd.DataFrame:
    """Build the long summary-stats frame for one metric."""

    def _stats(values: pd.Series) -> dict[str, float]:
        s = values.dropna().astype(float)
        if len(s) == 0:
            return {
                "n": 0, "mean": float("nan"), "sd": float("nan"),
                "median": float("nan"), "q10": float("nan"),
                "q25": float("nan"), "q75": float("nan"),
                "q90": float("nan"), "min": float("nan"), "max": float("nan"),
            }
        return {
            "n":      int(len(s)),
            # ddof=1 sample SD matches Hansen pooled formulas downstream.
            "mean":   float(s.mean()),
            "sd":     float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
            "median": float(s.median()),
            "q10":    float(s.quantile(0.10)),
            "q25":    float(s.quantile(0.25)),
            "q75":    float(s.quantile(0.75)),
            "q90":    float(s.quantile(0.90)),
            "min":    float(s.min()),
            "max":    float(s.max()),
        }

    rows: list[dict] = []
    rows.append({
        "metric": metric, "stratum": "overall", "exit_mechanism": "__all__",
        **_stats(df[metric_col]),
    })
    for mech in _EXIT_ORDER:
        sub = df[df["exit_mechanism"] == mech]
        rows.append({
            "metric": metric, "stratum": "by_exit_mechanism", "exit_mechanism": mech,
            **_stats(sub[metric_col]),
        })
    return pd.DataFrame(rows)


def _plot_overall(df: pd.DataFrame, metric_col: str, metric_label: str):
    """Per-site overall LOS histogram (single panel, single color)."""
    s = df[metric_col].dropna().clip(upper=30)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(s, bins=range(32), align="left", color="#4393c3", edgecolor="white")
    ax.set_xlabel(f"{metric_label} (days; 30+ overflow at right edge)")
    ax.set_ylabel("Hospitalizations")
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks(range(0, 31, 5))
    ax.set_title(
        f"{SITE_NAME.upper()} — overall LOS distribution "
        f"(N={len(s):,})",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def _plot_by_exit(df: pd.DataFrame, metric_col: str, metric_label: str):
    """Per-site LOS histogram stacked by exit_mechanism."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(31)
    bin_centers = np.arange(31)
    legend_handles = []
    for mech in _EXIT_ORDER:
        sub = df[df["exit_mechanism"] == mech][metric_col].dropna().clip(upper=30)
        counts, _ = np.histogram(sub, bins=range(32))
        # Skip mechanisms with zero count to keep the legend tight.
        if counts.sum() == 0:
            continue
        bars = ax.bar(
            bin_centers, counts, bottom=bottoms, width=0.9,
            color=_EXIT_PALETTE[mech], edgecolor="white", linewidth=0.4,
            label=f"{mech} (n={counts.sum():,})",
        )
        legend_handles.append(bars)
        bottoms = bottoms + counts

    ax.set_xlabel(f"{metric_label} (days; 30+ overflow at right edge)")
    ax.set_ylabel("Hospitalizations")
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks(range(0, 31, 5))
    ax.set_title(
        f"{SITE_NAME.upper()} — LOS distribution by exit mechanism "
        f"(N={int(bottoms.sum()):,})",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()

    meta_path = f"output/{SITE_NAME}/cohort_meta_by_id.parquet"
    meta = pd.read_parquet(meta_path)
    print(f"Loaded {meta_path}: {len(meta):,} hospitalizations")

    # Convert imv_dur_hrs → imv_duration_days for like-units binning
    # alongside n_days_full_24h. Keep both metrics so reviewers can
    # cross-check (full-24h count is a STRICT subset of total IMV time;
    # the gap is the partial intubation/extubation days).
    meta = meta.copy()
    meta["imv_duration_days"] = meta["imv_dur_hrs"] / 24.0

    metrics = [
        ("n_days_full_24h",    "n_days_full_24h",    "Full-24h IMV days"),
        ("imv_duration_days",  "imv_duration_days",  "IMV duration"),
    ]

    bins_parts = []
    stats_parts = []
    for metric_name, metric_col, _label in metrics:
        bins_parts.append(_bins_long(meta, metric_name, metric_col))
        stats_parts.append(_summary_stats_long(meta, metric_name, metric_col))

    bins_df = pd.concat(bins_parts, ignore_index=True)
    stats_df = pd.concat(stats_parts, ignore_index=True)

    save_csv(bins_df, "los_histogram_bins")
    save_csv(stats_df, "los_summary_stats")

    # Render PNGs for n_days_full_24h (the canonical full-24h-day count).
    # imv_duration_days is logged in CSV form for reviewer cross-check;
    # one PNG flavor is enough to keep the descriptive deck compact.
    primary_col, primary_label = "n_days_full_24h", "Full-24h IMV days"
    fig1 = _plot_overall(meta, primary_col, primary_label)
    save_fig(fig1, "los_histogram_overall")
    plt.close(fig1)

    fig2 = _plot_by_exit(meta, primary_col, primary_label)
    save_fig(fig2, "los_histogram_by_exit")
    plt.close(fig2)


if __name__ == "__main__":
    main()
