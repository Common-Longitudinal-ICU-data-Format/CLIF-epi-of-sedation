"""Cross-site overlay of the mean night-minus-day dose-rate trajectory.

Extends code/descriptive/night_day_diff_mean_by_icu_day.py by drawing one
color-coded line per site on each of the three drug panels. There is no
pooled line — random-effects pooling of per-site means is out of scope for
this descriptive pass (per user).

Three panels (propofol, fent-eq, midaz-eq). X-axis: ICU day bins 1..7 + '8+'.
Y-axis: mean(diff) with Student-t 95% CI shading, colored per site.

**Full 24-hr ICU days only.** Each patient's *last* `_nth_day` record is
dropped before binning — this is the patient's extubation day, which has
partial coverage (typically `night_total = 0`). Including those rows
biases the early-day means strongly negative because the diff formula
`(night_total - day_total) / 12 / 60` at 05_modeling_dataset.py:256
treats partial-coverage days as if they had a zero night dose. Equivalently:
"patient has a Day n+1 record" ⟺ "Day n's night shift was fully covered"
because IMV is continuous between extubations.

Respects ANONYMIZE_SITES for legend labels. Figure lands at
output_to_agg/figures/night_day_diff_mean_trajectory_cross_site.png.

Usage:
    uv run python code/agg/night_day_diff_mean_cross_site.py
    ANONYMIZE_SITES=1 uv run python code/agg/night_day_diff_mean_cross_site.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_COLS,
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    SITE_PALETTE,
    apply_style,
    cap_day,
    list_sites,
    load_site_analytical,
    prepare_diffs,
    save_agg_fig,
    site_label,
)


def _drop_last_day_per_patient(df) -> "pd.DataFrame":
    """For each hospitalization, drop the row with the maximum `_nth_day`.

    With the ≥24h IMV cohort filter, the only partial-coverage day for a
    given patient is the day they extubate. That's always their highest
    `_nth_day` value. Dropping it leaves only "full 24-hr ICU days" — i.e.,
    days where the patient was on IMV through the next 7am crossing.

    Equivalent to filtering by "patient has a Day n+1 record" and to the
    matched-coverage filter `n_hours_day > 0 AND n_hours_night > 0` from
    `sed_dose_daily.parquet`, but doable here without the daily-table
    join.
    """
    is_last = (
        df.groupby("hospitalization_id")["_nth_day"].transform("max")
        == df["_nth_day"]
    )
    return df.loc[~is_last].copy()


def _ci(values: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Return (mean, lo, hi) using Student-t interval. Assumes n > 1.

    Mirrors the helper in code/descriptive/night_day_diff_mean_by_icu_day.py
    (kept local here to avoid a second sys.path trampoline — the helper is
    small and not central to anything else).
    """
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(values))
    if n < 2:
        return mean, mean, mean
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    crit = float(stats.t.ppf((1 + conf) / 2, df=n - 1))
    return mean, mean - crit * se, mean + crit * se


def main() -> None:
    sites = list_sites()
    if not sites:
        print("No sites found under output_to_share/. Nothing to plot.")
        return
    print(f"Discovered sites: {sites}")

    apply_style()

    # Load + filter to full-24hr ICU days + bin per-site data upfront so
    # each panel reuses the same frame. `_drop_last_day_per_patient`
    # removes the per-patient extubation day (always the row with the
    # highest `_nth_day` for that hospitalization). Without this, Day 1's
    # mean diff is biased ~0.6 mcg/kg/min more negative at mimic propofol
    # by short-stay patients whose Day 1 has a partial night shift.
    per_site_df = {
        s: cap_day(
            prepare_diffs(_drop_last_day_per_patient(load_site_analytical(s))),
            max_day=7,
        )
        for s in sites
    }

    # Union of bin categories across sites. In practice every site sees at
    # least day 1..7 + '8+' so they match, but take the union to be safe.
    bin_set: list[str] = []
    for s in sites:
        cats = list(per_site_df[s]["_nth_day_bin"].cat.categories)
        for c in cats:
            if c not in bin_set:
                bin_set.append(c)
    bins = bin_set

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    # Lines plot at integer x positions — the same positions where x-tick
    # labels and gridlines are drawn — so the per-day means visually align
    # with their day labels. Earlier ±0.15 horizontal offsets disambiguated
    # collisions but pulled markers off the labelled ticks; with the alpha
    # 0.18 CI shading below, the visual disambiguation between sites comes
    # from color + transparency rather than offset.
    for ax, drug in zip(axes, DRUGS):
        col = DIFF_COLS[drug]
        for si, s in enumerate(sites):
            df = per_site_df[s]
            color = SITE_PALETTE[si % len(SITE_PALETTE)]
            label = site_label(s)

            means, lows, highs, ns = [], [], [], []
            for b in bins:
                vals = df.loc[df["_nth_day_bin"] == b, col].dropna().to_numpy()
                m, lo, hi = _ci(vals)
                means.append(m)
                lows.append(lo)
                highs.append(hi)
                ns.append(len(vals))

            x = np.arange(len(bins))
            ax.plot(x, means, marker="o", color=color, linewidth=1.8,
                    label=label, markersize=5)
            ax.fill_between(x, lows, highs, color=color, alpha=0.18)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(np.arange(len(bins)))
        ax.set_xticklabels(bins)
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"Mean {DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_title(DRUG_LABELS[drug])

    # Single legend — draw on the last panel since that tends to have the
    # least data overlap with typical y-axis ranges.
    axes[-1].legend(title="Site", loc="upper right", frameon=False)

    fig.suptitle(
        "Night-minus-day dose rate by ICU day, full-24hr days only "
        "— per site (mean ± 95% CI)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_agg_fig(fig, "night_day_diff_mean_trajectory_cross_site")


if __name__ == "__main__":
    main()
