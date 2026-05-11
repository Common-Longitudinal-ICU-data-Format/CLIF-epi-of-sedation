"""Cross-site sedation dose: concatenated 7-day timeline (Phase 3 overhaul).

Replaces the prior 3-rate × 5-cohort × per-drug grid (15 panels per PNG,
24-hour x-axis) with a single concatenated 7-day timeline (168 hours,
2 panels per PNG). Per-site lines overlay on each panel.

Reads each site's pre-aggregated
`output_to_share/{site}/descriptive/sed_dose_by_hr_of_stay.csv`
(federation-clean — see `.dev/CLAUDE.md` "Federation contract"). The
per-site DuckDB join + aggregation runs in
`code/descriptive/sed_dose_by_hr_of_stay.py`, so this script never
touches raw parquet under `output/`.

`hour_of_stay` runs 0..167 where hour 0 = `_nth_day=1` 7am (the
patient's first full ICU day's morning) and hour 167 = `_nth_day=7`
6am. Mapping: `hour_of_stay = (_nth_day - 1) * 24 + ((_hr - 7 + 24) % 24)`.

Outputs:
  - `output_to_agg/sed_dose_by_hr_of_day_cross_site.csv` (long-format,
    one row per (site, hour_of_stay))
  - `output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_prop.png`
  - `output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_fenteq.png`
  - `output_to_agg/figures/sed_dose_by_hr_of_day_cross_site_midazeq.png`

Per-PNG layout: figsize=(16, 6), 2 rows × 1 col, sharex.
  - Row 0: avg rate across all IMV patient-hours (drug-specific units)
  - Row 1: % on drug, y-axis fixed [0, 100]
  - X-axis: hour-of-stay 0..167. Time-of-day ticks at every 6h crossing
    labelled "7am" / "12pm" / "7pm" / "12am" (4 marks per 24h window,
    uniform style). The 7am ticks of days 1–7 carry an additional
    "Day N" line below the time label (multi-line tick label). Solid
    gray vertical lines at every 7am and 7pm position frame the
    night-band edges; much dimmer dotted lines at every 12pm and 12am
    serve as intra-day reference marks. Tick labels are shown on every
    panel, not just the bottom one.
  - Lines: one per site, SITE_PALETTE colors, no markers. Site labels
    via site_label() so ANONYMIZE_SITES=1 produces "Site A" / "Site B".

Survivor-bias note (encoded as an annotation): n_imv decreases as
hour_of_stay increases (extubated patients drop out). The figure
displays the n_imv attrition (hour 0 → hour 167) per site as a small
textbox on Row 0 so the caveat is explicit.

Usage:
    uv run python code/agg/sed_dose_by_hr_of_day_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/sed_dose_by_hr_of_day_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.sed_dose_by_hr_of_day")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DRUG_COLORS,
    DRUG_LABELS,
    DRUG_UNITS,
    SITE_PALETTE,
    apply_style,
    list_sites,
    load_site_descriptive_csv,
    save_agg_csv,
    save_agg_fig,
    site_label,
)


# ── Drug spec ─────────────────────────────────────────────────────────────
# Three drugs render as three PNGs. Each tuple: (drug_key, csv_mean_col,
# csv_n_col, y_label_for_avg_panel). The csv_mean_col carries the
# drug-specific avg-rate metric in clinical units; csv_n_col carries the
# "on drug" patient-hour count needed for pct_on_drug computation.
DRUGS: list[tuple[str, str, str, str]] = [
    ("prop",
     "propofol_mcg_kg_min_mean",
     "n_on_drug_propofol",
     f"Propofol avg rate ({DRUG_UNITS['prop']})"),
    ("fenteq",
     "fenteq_mcg_mean",
     "n_on_drug_fenteq",
     f"Fentanyl-eq avg rate ({DRUG_UNITS['fenteq']})"),
    ("midazeq",
     "midazeq_mg_mean",
     "n_on_drug_midazeq",
     f"Midazolam-eq avg rate ({DRUG_UNITS['midazeq']})"),
]


# ── X-axis layout (time-of-day + day-of-stay split into two rows) ────────
# Hour 0 of the timeline = patient's _nth_day=1, 7am (first full ICU
# day's morning). Within each 24h window we render 4 evenly-spaced
# time-of-day marks at hour offsets 0/5/12/17 → "7am" / "12pm" / "7pm"
# / "12am", all styled identically. The day-of-stay ("Day N") sits on
# its own secondary-axis row beneath the time labels (added in
# `_figure_for_drug`, on the bottom panel only).
_DAY_STARTS = list(range(0, 168, 24))  # 0, 24, 48, 72, 96, 120, 144
_TIME_TICKS_7AM  = [h + 0  for h in _DAY_STARTS] + [168]  # include hour-168 endpoint
_TIME_TICKS_12PM = [h + 5  for h in _DAY_STARTS]
_TIME_TICKS_7PM  = [h + 12 for h in _DAY_STARTS]
_TIME_TICKS_12AM = [h + 17 for h in _DAY_STARTS]

_TIME_TICKS = sorted(_TIME_TICKS_7AM + _TIME_TICKS_12PM
                     + _TIME_TICKS_7PM + _TIME_TICKS_12AM)

# Tick labels combine time-of-day with day-of-stay on a single multi-
# line label so the "Day N" row sits one font-line below the "7am"
# label (much tighter than a separate secondary axis). Day labels live
# only on the 7am ticks of days 1–7; all other ticks (12pm/7pm/12am
# and the hour-168 endpoint) get just the time-of-day string.
_TIME_LABEL_BY_TICK: dict[int, str] = {}
for _i, _h in enumerate(_TIME_TICKS_7AM):
    _TIME_LABEL_BY_TICK[_h] = f"7am\nDay {_i + 1}" if _i < 7 else "7am"
for _h in _TIME_TICKS_12PM:
    _TIME_LABEL_BY_TICK[_h] = "12pm"
for _h in _TIME_TICKS_7PM:
    _TIME_LABEL_BY_TICK[_h] = "7pm"
for _h in _TIME_TICKS_12AM:
    _TIME_LABEL_BY_TICK[_h] = "12am"
_TIME_LABELS = [_TIME_LABEL_BY_TICK[h] for h in _TIME_TICKS]

# Vertical reference lines:
#   - Shift boundaries (7am + 7pm marks) → solid lines that frame the
#     night-band edges. Crossing one means a day↔night shift change.
#   - Intra-day midpoints (every 12pm and 12am position) → very dim
#     dotted lines so the 6-hour rhythm is visible but recedes far
#     behind the shift-boundary cue.
_SHIFT_BOUNDARY_LINES = sorted(_DAY_STARTS[1:] + _TIME_TICKS_7PM)  # 12,24,36,...,156
_INTRADAY_LINES       = sorted(_TIME_TICKS_12PM + _TIME_TICKS_12AM)  # 5,17,29,41,...

# Night-shift bands — each (start, end) pair brackets a 12h 7pm→7am
# block on the timeline. Hour-of-stay 12 = first 7pm; the night runs
# 12 hours to the next 7am at hour 24, etc. Seven night blocks cover
# the 7-day timeline. Shaded with a faint gray axvspan so the
# day/night cycle is visible at a glance without competing with data.
_NIGHT_BANDS = [(12, 24), (36, 48), (60, 72), (84, 96),
                (108, 120), (132, 144), (156, 168)]

# Title-color overrides — DRUG_COLORS (skyblue / salmon / mediumseagreen)
# are tuned for filled-bar charts where a pastel reads well; for a
# big-bold title we want a darker, higher-contrast hue that still
# tracks the drug "channel" so readers can flip between PNGs and
# instantly tell which drug they're looking at.
_DRUG_TITLE_COLORS = {
    "prop":    "#1f4e79",  # deep blue
    "fenteq":  "#a61d24",  # firebrick red
    "midazeq": "#1f6e1f",  # forest green
}


def _load_site(site: str) -> pd.DataFrame:
    """Load one site's pre-aggregated 168-row hour-of-stay CSV.

    Reads `output_to_share/{site}/descriptive/sed_dose_by_hr_of_stay.csv`
    (produced by `code/descriptive/sed_dose_by_hr_of_stay.py`). The CSV
    carries the canonical group-level aggregate; this loader just
    appends a `site` column and computes the per-row percentages
    (federation-safe — only count ratios, no row-level data).
    """
    try:
        df = load_site_descriptive_csv(site, "sed_dose_by_hr_of_stay")
    except FileNotFoundError:
        logger.info(
            f"  WARN: missing input for {site} "
            f"(output_to_share/{site}/descriptive/sed_dose_by_hr_of_stay.csv); "
            "skipping — re-run code/descriptive/sed_dose_by_hr_of_stay.py."
        )
        return pd.DataFrame()
    df.insert(0, "site", site)

    for _drug, _, n_col, _ in DRUGS:
        pct_col = n_col.replace("n_on_drug_", "pct_on_drug_")
        df[pct_col] = 100.0 * df[n_col] / df["n_imv"].replace(0, np.nan)
    df["pct_on_drug_any"] = (
        100.0 * df["n_on_drug_any"] / df["n_imv"].replace(0, np.nan)
    )
    return df


def _stack_per_site(sites: list[str]) -> pd.DataFrame:
    parts = []
    for site in sites:
        sub = _load_site(site)
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _apply_x_axis(ax) -> None:
    """Apply the shared 0..168 hour-of-stay tick layout to one axis.

    Renders 4 evenly-spaced time-of-day labels per 24h window, the
    night-shift shading bands, day-boundary dashed lines, and lighter
    intra-day (12pm/12am) reference lines. All gridlines/shading sit
    behind the data via zorder=0/-1.
    """
    ax.set_xlim(0, 168)
    ax.set_xticks(_TIME_TICKS)
    ax.set_xticklabels(_TIME_LABELS, fontsize=9)

    # Night-shift bands (7pm-7am): light-gray rectangles behind data.
    # zorder=-1 keeps them behind both grid lines and data lines.
    for night_start, night_end in _NIGHT_BANDS:
        ax.axvspan(night_start, night_end,
                   facecolor="0.85", alpha=0.45, zorder=-1, linewidth=0)

    # Shift-boundary cutoff lines (7am + 7pm) — solid gray, framing the
    # night-band edges. On top of shading but behind data lines.
    for h in _SHIFT_BOUNDARY_LINES:
        ax.axvline(h, color="0.55", linestyle="-",
                   linewidth=0.8, alpha=0.7, zorder=0)

    # Intra-day midpoint lines (12pm + 12am) — much dimmer dotted style
    # so they recede well behind the shift-boundary cue.
    for h in _INTRADAY_LINES:
        ax.axvline(h, color="0.85", linestyle=":",
                   linewidth=0.5, alpha=0.4, zorder=0)


def _figure_for_drug(
    df: pd.DataFrame,
    sites: list[str],
    drug_key: str,
    mean_col: str,
    n_col: str,
    rate_label: str,
):
    """Render the 4-row PNG for one drug.

    Row 0: avg rate across all on-IMV patient-hours (drug-specific).
    Row 1: % of on-IMV patient-hours with this drug > 0 (drug-specific).
    Row 2: % of on-IMV patient-hours with ANY of {prop, fenteq, midaz}
           > 0 — drug-independent baseline, identical on all 3 PNGs.
    Row 3: N (n_imv per (site, hour_of_stay)) — patient-hour count
           contributing to each point. Drug-independent; identical on
           all 3 PNGs.

    All four panels share the 0..167 hour-of-stay x-axis and inherit
    the night-shift shading + day-boundary dashed lines from
    `_apply_x_axis`.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 2, 2]})
    ax_rate, ax_pct, ax_pct_any, ax_n = axes

    pct_col = n_col.replace("n_on_drug_", "pct_on_drug_")

    for i, site in enumerate(sites):
        sub = (
            df[df["site"] == site]
            .sort_values("hour_of_stay")
        )
        if sub.empty:
            continue
        color = SITE_PALETTE[i % len(SITE_PALETTE)]
        label = site_label(site)

        # Row 0: avg rate (across ALL on-IMV hours, including zeros).
        ax_rate.plot(sub["hour_of_stay"], sub[mean_col],
                     color=color, linewidth=1.4, label=label, zorder=2)
        # Row 1: % on this specific drug.
        ax_pct.plot(sub["hour_of_stay"], sub[pct_col],
                    color=color, linewidth=1.4, zorder=2)
        # Row 2: % on ANY of the 3 sedatives — same data on every PNG.
        ax_pct_any.plot(sub["hour_of_stay"], sub["pct_on_drug_any"],
                        color=color, linewidth=1.4, zorder=2)
        # Row 3: raw N (patient-hour count), thousands.
        ax_n.plot(sub["hour_of_stay"], sub["n_imv"],
                  color=color, linewidth=1.4, zorder=2)

    # Y-axis labels & ranges per panel.
    ax_rate.set_ylabel(rate_label, fontsize=10)
    ax_rate.grid(axis="y", alpha=0.3)

    ax_pct.set_ylabel(
        f"% on {DRUG_LABELS[drug_key]}\n(per-hour cohort)", fontsize=10,
    )
    ax_pct.set_ylim(0, 100)
    ax_pct.grid(axis="y", alpha=0.3)

    ax_pct_any.set_ylabel("% on any sedative\n(prop / fenteq / midaz)",
                          fontsize=10)
    ax_pct_any.set_ylim(0, 100)
    ax_pct_any.grid(axis="y", alpha=0.3)

    ax_n.set_ylabel("N (on-IMV\npatient-hours)", fontsize=10)
    ax_n.grid(axis="y", alpha=0.3)
    # N axis: comma-formatted integers for readability.
    ax_n.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"{int(v):,}")
    )

    for _ax in axes:
        _apply_x_axis(_ax)
        # `sharex=True` hides tick labels on non-bottom panels by
        # default — re-enable so every panel carries its own
        # time-of-day + "Day N" labels (per user request).
        _ax.tick_params(axis="x", labelbottom=True)

    ax_n.set_xlabel("Hour of stay (anchored at first full-ICU-day 7am)",
                    fontsize=10)

    fig.tight_layout()

    # ── Two-tier title: drug name in big colored-bold so each PNG is
    # immediately distinguishable at a glance, with a smaller subtitle
    # below describing the figure. The legend sits above the plot area
    # but below the title block.
    fig.legend(
        loc="upper center", bbox_to_anchor=(0.5, 1.01),
        ncol=len(sites), frameon=False, fontsize=10,
    )
    fig.text(
        0.5, 1.07, DRUG_LABELS[drug_key],
        ha="center", va="bottom",
        fontsize=20, fontweight="bold",
        color=_DRUG_TITLE_COLORS.get(drug_key, "0.15"),
    )
    fig.text(
        0.5, 1.045,
        "dosing trajectory across the first 7 ICU days, by site",
        ha="center", va="bottom", fontsize=11, color="0.30",
    )

    # ── Footnote — journal-style prose. No SQL formulas: the
    # cohort definition and computation rules belong here in plain
    # English so a reader can audit the figure without opening the
    # codebase. Detailed implementation notes live in the per-cell
    # comments in the source.
    fig.text(
        0.5, -0.045,
        "For each site, hourly sedation totals were joined to the canonical "
        "patient-day registry (Phase 1 cohort: post-encounter-stitch dedup "
        "and post-weight-QC) and aggregated by hour-of-stay. Hour 0 is "
        "anchored at 7am of each patient's first complete 24-hour ICU day; "
        "the timeline then runs hour-by-hour through 7 days, with the day "
        "shift (7am–7pm) on white background and the night shift (7pm–7am) "
        "shaded light gray. "
        "First panel — mean dose rate averaged across all on-IMV patient-"
        "hours at that point, including zero-dose hours; this reflects the "
        "average across the cohort rather than among recipients only. "
        "Second/third panels — percentage of on-IMV patient-hours with a "
        "non-zero dose; the third panel is the union across propofol, "
        "fentanyl-equivalent, and midazolam-equivalent and is rendered "
        "identically on all three figures so the drug-specific curve in the "
        "second panel can be read against a common saturation reference. "
        "Bottom panel — N, the number of patients still on invasive "
        "mechanical ventilation at each hour-of-stay (one count per site "
        "per hour); equivalently, the patient-hour count contributing to "
        "each data point. N declines monotonically as patients are "
        "extubated, so late-timeline values reflect a smaller, sicker, "
        "longer-staying subset and should be interpreted with that "
        "selection bias in mind.",
        ha="center", va="top", fontsize=8.5, color="0.30", wrap=True,
    )
    return fig


def main() -> None:
    apply_style()

    sites = list_sites()
    if not sites:
        logger.info("No site dirs under output_to_share/. Run per-site pipelines first.")
        return

    df = _stack_per_site(sites)
    if df.empty:
        logger.info("No per-site data computed; nothing to render.")
        return

    # Persist the long-format CSV (federated-safe — group-level aggregates
    # only, no IDs). 168 rows per site × n_sites total.
    save_agg_csv(df, "sed_dose_by_hr_of_day_cross_site")

    for drug_key, mean_col, n_col, rate_label in DRUGS:
        fig = _figure_for_drug(df, sites, drug_key, mean_col, n_col, rate_label)
        save_agg_fig(fig, f"sed_dose_by_hr_of_day_cross_site_{drug_key}")
        plt.close(fig)


if __name__ == "__main__":
    main()
