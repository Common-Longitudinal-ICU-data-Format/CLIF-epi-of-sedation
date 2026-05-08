"""Cross-site LOS histogram pooler.

Reads each site's `output_to_share/{site}/descriptive/los_histogram_bins.csv`
(produced by `code/descriptive/los_histogram.py`) and pools by summing
counts per `(metric, stratum, exit_mechanism, bin)`. Then renders three
PNGs and writes the pooled bins CSV:

  - `output_to_agg/los_histogram_pooled_bins.csv`
  - `output_to_agg/figures/los_histogram_overall_cross_site.png`
        — pooled overall LOS, faceted by site (one panel per site)
  - `output_to_agg/figures/los_histogram_by_exit_cross_site.png`
        — pooled by-exit-mechanism stacked histogram
        (single panel; sum-across-sites by mechanism × bin)
  - `output_to_agg/figures/los_histogram_compare_cross_site.png`
        — side-by-side: pooled overall vs pooled by-exit-mechanism

Federated convention: per-site CSVs carry only group-level counts —
no row-level data, no IDs. Pooling is sum-across-sites so the agg
script never needs PHI access.

Usage:
    uv run python code/agg/los_histogram_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/los_histogram_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    SHARE_ROOT,
    SITE_PALETTE,
    apply_style,
    list_sites,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


_PRIMARY_METRIC = "n_days_full_24h"
_PRIMARY_LABEL = "Full-24h IMV days"

# Mirror code/descriptive/los_histogram.py palette so the per-site and
# cross-site figures read consistently.
_EXIT_PALETTE: dict[str, str] = {
    "tracheostomy":          "#8c2d04",
    "died_on_imv":           "#cb181d",
    "palliative_extubation": "#f4a582",
    "failed_extubation":     "#fdae61",
    "successful_extubation": "#4393c3",
    "discharge_on_imv":      "#9e9e9e",
    "unknown":               "#000000",
}
_EXIT_ORDER = list(_EXIT_PALETTE.keys())


def _stack_per_site() -> pd.DataFrame:
    """Stack each site's bins.csv with a `site` column attached."""
    sites = list_sites()
    if not sites:
        print("No site dirs under output_to_share/. Run per-site first.")
        return pd.DataFrame()

    parts = []
    for site in sites:
        path = SHARE_ROOT / site / "descriptive" / "los_histogram_bins.csv"
        if not path.exists():
            print(f"  WARN: {path} missing — skipping {site}")
            continue
        df = pd.read_csv(path)
        df["site"] = site
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _pool_bins(df_all: pd.DataFrame) -> pd.DataFrame:
    """Sum counts across sites per (metric, stratum, exit_mechanism, bin)."""
    return (
        df_all
        .groupby(
            ["metric", "stratum", "exit_mechanism", "bin_left", "bin_right"],
            as_index=False,
            observed=True,
        )["count"]
        .sum()
        .sort_values(
            ["metric", "stratum", "exit_mechanism", "bin_left"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def _bin_centers(df: pd.DataFrame) -> np.ndarray:
    """Map (bin_left, bin_right) → center on the [0, 30] axis.

    Overflow bin (bin_right=inf) is plotted at x=30 with width 1, so the
    histogram visually communicates "≥ 30" without a gap.
    """
    centers = df["bin_left"].astype(float).copy()
    return centers.to_numpy()


def _figure_overall_per_site(df_all: pd.DataFrame, sites: list[str]):
    """Per-site facet: one small histogram per site, shared x-axis."""
    n = len(sites)
    fig, axes = plt.subplots(
        1, n, figsize=(4.5 * n, 4.5), sharex=True, sharey=False,
    )
    if n == 1:
        axes = np.atleast_1d(axes)

    for ax, site in zip(axes, sites):
        sub = df_all[
            (df_all["site"] == site)
            & (df_all["metric"] == _PRIMARY_METRIC)
            & (df_all["stratum"] == "overall")
        ].sort_values("bin_left")
        if sub.empty:
            ax.text(0.5, 0.5, "no data", ha="center", va="center")
            ax.set_axis_off()
            continue
        n_total = int(sub["count"].sum())
        ax.bar(
            _bin_centers(sub),
            sub["count"].to_numpy(),
            width=0.9, align="edge",
            color=SITE_PALETTE[sites.index(site) % len(SITE_PALETTE)],
            edgecolor="white", linewidth=0.4,
        )
        ax.set_xlim(-0.5, 30.5)
        ax.set_xticks(range(0, 31, 5))
        ax.set_title(f"{site_label(site)} (N={n_total:,})", fontsize=11)
        ax.set_xlabel(f"{_PRIMARY_LABEL} (days; 30+ overflow)")
        if ax is axes[0]:
            ax.set_ylabel("Hospitalizations")

    fig.suptitle(
        f"Overall LOS distribution per site — {_PRIMARY_LABEL}", fontsize=12,
    )
    fig.tight_layout()
    return fig


def _figure_pooled_by_exit(df_pooled: pd.DataFrame):
    """Single panel: pooled-across-sites stacked histogram colored by exit."""
    pooled = df_pooled[
        (df_pooled["metric"] == _PRIMARY_METRIC)
        & (df_pooled["stratum"] == "by_exit_mechanism")
    ].copy()

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottoms = np.zeros(31)
    bin_centers = np.arange(31, dtype=float)

    for mech in _EXIT_ORDER:
        sub = (
            pooled[pooled["exit_mechanism"] == mech]
            .sort_values("bin_left")
        )
        if sub.empty:
            continue
        counts = sub["count"].to_numpy().astype(float)
        if counts.sum() == 0:
            continue
        ax.bar(
            bin_centers, counts, bottom=bottoms, width=0.9,
            color=_EXIT_PALETTE[mech], edgecolor="white", linewidth=0.4,
            label=f"{mech} (n={int(counts.sum()):,})",
        )
        bottoms = bottoms + counts

    ax.set_xlabel(f"{_PRIMARY_LABEL} (days; 30+ overflow at right edge)")
    ax.set_ylabel("Hospitalizations (pooled across sites)")
    ax.set_xlim(-0.5, 30.5)
    ax.set_xticks(range(0, 31, 5))
    ax.set_title(
        f"Cross-site pooled LOS by exit mechanism — N={int(bottoms.sum()):,}",
        fontsize=12,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


def _figure_compare(df_all: pd.DataFrame, df_pooled: pd.DataFrame, sites: list[str]):
    """Side-by-side: pooled overall vs pooled by_exit. Useful sanity panel."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: pooled overall, colored by site (stacked).
    bin_centers = np.arange(31, dtype=float)
    bottoms_l = np.zeros(31)
    for i, site in enumerate(sites):
        sub = (
            df_all[
                (df_all["site"] == site)
                & (df_all["metric"] == _PRIMARY_METRIC)
                & (df_all["stratum"] == "overall")
            ]
            .sort_values("bin_left")
        )
        if sub.empty:
            continue
        counts = sub["count"].to_numpy().astype(float)
        n_total = int(counts.sum())
        ax_l.bar(
            bin_centers, counts, bottom=bottoms_l, width=0.9,
            color=SITE_PALETTE[i % len(SITE_PALETTE)],
            edgecolor="white", linewidth=0.4,
            label=f"{site_label(site)} (N={n_total:,})",
        )
        bottoms_l = bottoms_l + counts
    ax_l.set_title("Overall (stacked by site)", fontsize=11)
    ax_l.set_xlabel(f"{_PRIMARY_LABEL} (days; 30+ overflow)")
    ax_l.set_ylabel("Hospitalizations")
    ax_l.set_xlim(-0.5, 30.5)
    ax_l.set_xticks(range(0, 31, 5))
    ax_l.legend(loc="upper right", fontsize=9, frameon=False)

    # Right: pooled by exit mechanism (stacked).
    pooled = df_pooled[
        (df_pooled["metric"] == _PRIMARY_METRIC)
        & (df_pooled["stratum"] == "by_exit_mechanism")
    ]
    bottoms_r = np.zeros(31)
    for mech in _EXIT_ORDER:
        sub = (
            pooled[pooled["exit_mechanism"] == mech]
            .sort_values("bin_left")
        )
        if sub.empty:
            continue
        counts = sub["count"].to_numpy().astype(float)
        if counts.sum() == 0:
            continue
        ax_r.bar(
            bin_centers, counts, bottom=bottoms_r, width=0.9,
            color=_EXIT_PALETTE[mech], edgecolor="white", linewidth=0.4,
            label=f"{mech} (n={int(counts.sum()):,})",
        )
        bottoms_r = bottoms_r + counts
    ax_r.set_title("By exit mechanism (stacked)", fontsize=11)
    ax_r.set_xlabel(f"{_PRIMARY_LABEL} (days; 30+ overflow)")
    ax_r.set_ylabel("Hospitalizations")
    ax_r.set_xlim(-0.5, 30.5)
    ax_r.set_xticks(range(0, 31, 5))
    ax_r.legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Cross-site LOS pooled views", fontsize=13)
    fig.tight_layout()
    return fig


def main() -> None:
    apply_style()
    df_all = _stack_per_site()
    if df_all.empty:
        return
    sites = sorted(df_all["site"].unique().tolist())

    df_pooled = _pool_bins(df_all)
    save_agg_csv(df_pooled, "los_histogram_pooled_bins")

    fig1 = _figure_overall_per_site(df_all, sites)
    save_agg_fig(fig1, "los_histogram_overall_cross_site")
    plt.close(fig1)

    fig2 = _figure_pooled_by_exit(df_pooled)
    save_agg_fig(fig2, "los_histogram_by_exit_cross_site")
    plt.close(fig2)

    fig3 = _figure_compare(df_all, df_pooled, sites)
    save_agg_fig(fig3, "los_histogram_compare_cross_site")
    plt.close(fig3)


if __name__ == "__main__":
    main()
