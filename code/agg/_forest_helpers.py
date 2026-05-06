"""Shared utilities for the cross-site forest-plot scripts under code/agg/.

Three scripts (forest_night_day_cross_site.py, forest_daytime_cross_site.py,
forest_sbt_sensitivity_cross_site.py) all read each site's
output_to_share/{site}/models/forest_data.csv and render forest plots of
10→90 percentile-shift ORs. The site-stacking loader, the OR-axis formatter
(log scale + sparse major ticks + minor gridlines every 0.05), and the
auto-xlim helper are factored here so the per-figure scripts only own
their own layout + filtering rules.

Leading underscore = NOT picked up by the `make agg` glob in the Makefile
(`code/agg/*.py`, with the per-script `case _*) continue` filter); this
module is for import only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

sys.path.insert(0, str(Path(__file__).parent))

from _shared import SHARE_ROOT, list_sites, load_site_forest_data  # noqa: E402


# ── Loaders ───────────────────────────────────────────────────────────────
def stack_per_site() -> pd.DataFrame:
    """Concat each discovered site's forest_data.csv with a `site` column.

    Skips sites whose `forest_data.csv` is missing rather than failing — a
    new site dropped under `output_to_share/` won't break a figure until
    its per-site pipeline (08_models.py) has been run. The site-discovery
    skip list (qc/, figures/, _*) lives in `_shared.list_sites()`.
    """
    sites = list_sites()
    if not sites:
        print("No sites found under output_to_share/. Nothing to plot.")
        return pd.DataFrame()
    print(f"Discovered sites: {sites}")

    frames: list[pd.DataFrame] = []
    for s in sites:
        path = SHARE_ROOT / s / "models" / "forest_data.csv"
        if not path.exists():
            print(f"  SKIP {s}: {path} missing — re-run 08_models.py for this site.")
            continue
        df = load_site_forest_data(s)
        df["site"] = s
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── X-axis formatter (shared across all 3 figures) ───────────────────────
# Major ticks at sparse round OR values — readable labels, no collisions.
# Selection is greedy with a minimum log-distance gap (`_MIN_LOG_GAP`) so
# wider xlim spans automatically thin out crowded near-1.0 ticks. Anchors
# are listed dense-to-sparse so 1.0 is always preferred when possible.
# 0.05-resolution gridlines come from the minor locator independently of
# whether a position appears as a labeled major.
_MAJOR_CANDIDATES = [
    1.00,                                    # always-prefer anchor
    0.50, 0.70, 1.50, 2.00,                  # next-tier
    0.60, 0.80, 0.90, 1.10, 1.20,            # mid-tier (used if xlim leaves room)
]
# Minimum log10 gap between consecutive major ticks. ~0.115 ≈ factor of 1.30
# between consecutive labeled positions, which is roughly the print-friendly
# label-spacing threshold at typical agg-figure panel widths.
_MIN_LOG_GAP = 0.115


def or_xlim_from_data(df: pd.DataFrame, fallback: tuple[float, float] = (0.85, 1.05)) -> tuple[float, float]:
    """Auto-compute symmetric-ish OR x-range that hugs the CIs and includes 1.

    Returns `fallback` if no finite CIs are available (e.g., script run
    before per-site forest_data.csv has been regenerated).
    """
    los = df["OR_lo"].to_numpy(dtype=float)
    his = df["OR_hi"].to_numpy(dtype=float)
    los = los[np.isfinite(los)]
    his = his[np.isfinite(his)]
    if len(los) == 0 or len(his) == 0:
        return fallback
    lo = float(min(los.min(), 1.0))
    hi = float(max(his.max(), 1.0))
    span = hi - lo
    pad = max(span * 0.08, 0.005)
    return (lo - pad, hi + pad)


def apply_or_xaxis(
    ax: Axes,
    xlim: tuple[float, float],
    *,
    minor_step: float = 0.05,
) -> None:
    """Configure the OR x-axis: log scale, sparse major ticks, dense minor gridlines.

    Major tick positions are picked from `_MAJOR_CANDIDATES` (round OR values
    inside `xlim`); minor ticks land on every `minor_step` (default 0.05)
    inside `xlim` regardless of major positions. Both major and minor
    gridlines render so reviewers have a 0.05-OR visual reference at every
    line, with bolder lines at the labeled major ticks.
    """
    ax.set_xscale("log")
    ax.set_xlim(*xlim)

    # Major ticks: greedy selection by candidate priority + minimum
    # log-spacing. Walks `_MAJOR_CANDIDATES` (priority-ordered: 1.0 first,
    # then anchors, then mid-tier) and accepts a candidate only if it sits
    # inside xlim AND is at least `_MIN_LOG_GAP` away from every already-
    # accepted tick (in log10 space). The result is sparse on wide xlim
    # (e.g., [0.4, 1.5] → 0.5, 0.7, 1.0, 1.5) and dense on narrow xlim
    # (e.g., [0.85, 1.05] → just 1.0; the minor 0.05 grid still gives
    # context).
    accepted: list[float] = []
    for t in _MAJOR_CANDIDATES:
        if not (xlim[0] <= t <= xlim[1]):
            continue
        log_t = np.log10(t)
        if all(abs(log_t - np.log10(a)) >= _MIN_LOG_GAP for a in accepted):
            accepted.append(t)
    major = sorted(accepted) or [1.0]
    ax.xaxis.set_major_locator(FixedLocator(major))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))

    # Minor ticks: every `minor_step` inside xlim, excluding positions already
    # covered by major ticks (avoids drawing two gridlines on top of each other).
    lo_n = int(np.floor(xlim[0] / minor_step))
    hi_n = int(np.ceil(xlim[1] / minor_step))
    minor_all = [round(k * minor_step, 4) for k in range(lo_n, hi_n + 1)]
    minor = [t for t in minor_all if xlim[0] <= t <= xlim[1] and t not in major]
    ax.xaxis.set_minor_locator(FixedLocator(minor))
    # Suppress minor labels — without this, matplotlib's inherited
    # LogFormatterSciNotation renders things like "0.55 × 10⁰" at every
    # minor position and crowds out the major labels.
    ax.xaxis.set_minor_formatter(NullFormatter())

    # Gridlines: minor at very low alpha for the 0.05-resolution reference,
    # major slightly stronger so the labeled positions are still distinguishable.
    ax.grid(True, which="major", axis="x", linewidth=0.5, alpha=0.45, zorder=0)
    ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.18, zorder=0)


# ── Reference line at OR=1 ────────────────────────────────────────────────
def add_or_reference_line(ax: Axes) -> None:
    """Dashed vertical line at OR=1 — the no-effect anchor."""
    ax.axvline(1.0, color="dimgray", linewidth=0.9, linestyle="--", zorder=1)


__all__ = [
    "stack_per_site",
    "or_xlim_from_data",
    "apply_or_xaxis",
    "add_or_reference_line",
]
