"""Cross-site overlay of the mean night-minus-day dose-rate trajectory.

Federated companion to code/descriptive/night_day_diff_combined_by_icu_day.py.
Reads each site's pre-aggregated
`output_to_share/{site}/descriptive/night_day_dose_stats_by_icu_day.csv`
(no raw-parquet access — see `.dev/CLAUDE.md` "Federation contract") and
draws one color-coded line per site on each of the three drug panels.
There is no pooled line — random-effects pooling of per-site means is
out of scope here; the new `dose_pattern_summary_panels_cross_site.py`
figure carries the master-cohort overlay if you want a pooled view.

Three panels (propofol, fent-eq, midaz-eq). X-axis: ICU days 1..7
(full-24h coverage only — the filter is already applied on the per-site
side before the CSV is emitted). Y-axis: mean(diff) with Student-t 95%
CI shading, colored per site.

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
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.night_day_diff_mean")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DRUG_LABELS,
    DRUG_UNITS,
    DRUGS,
    SITE_PALETTE,
    apply_style,
    list_sites,
    load_site_descriptive_csv,
    save_agg_fig,
    site_label,
    student_t_ci_from_summary,
)


MIN_DAY = 1
MAX_DAY = 7
DAY_BINS = list(range(MIN_DAY, MAX_DAY + 1))


def main() -> None:
    sites = list_sites()
    if not sites:
        logger.info("No sites found under output_to_share/. Nothing to plot.")
        return
    logger.info(f"Discovered sites: {sites}")

    apply_style()

    # Read each site's per-(drug, nth_day) summary stats. The
    # `_is_full_24h_day & _nth_day BETWEEN 1 AND 7` filter is already
    # applied on the per-site side before the CSV is written, so this
    # script never has to touch raw parquet.
    per_site_stats: dict[str, pd.DataFrame] = {}
    for s in sites:
        try:
            per_site_stats[s] = load_site_descriptive_csv(s, "night_day_dose_stats_by_icu_day")
        except FileNotFoundError:
            logger.info(
                f"  SKIP {s}: output_to_share/{s}/descriptive/"
                f"night_day_dose_stats_by_icu_day.csv not found — "
                f"re-run code/descriptive/night_day_diff_combined_by_icu_day.py."
            )

    if not per_site_stats:
        logger.info("No usable per-site stats; nothing to render.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)

    for ax, drug in zip(axes, DRUGS):
        for si, s in enumerate(per_site_stats.keys()):
            stats_df = per_site_stats[s]
            cell = stats_df.loc[stats_df["drug"] == drug].set_index("nth_day")
            color = SITE_PALETTE[si % len(SITE_PALETTE)]
            label = site_label(s)

            means, lows, highs = [], [], []
            for d in DAY_BINS:
                if d in cell.index:
                    row = cell.loc[d]
                    m, lo, hi = student_t_ci_from_summary(
                        int(row["n_full_24h"]),
                        float(row["mean_diff_all"]),
                        float(row["sd_diff_all"]),
                    )
                else:
                    m, lo, hi = (float("nan"),) * 3
                means.append(m)
                lows.append(lo)
                highs.append(hi)

            x = np.arange(len(DAY_BINS))
            ax.plot(x, means, marker="o", color=color, linewidth=1.8,
                    label=label, markersize=5)
            ax.fill_between(x, lows, highs, color=color, alpha=0.18)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(np.arange(len(DAY_BINS)))
        ax.set_xticklabels([str(d) for d in DAY_BINS])
        ax.set_xlabel("ICU day")
        ax.set_ylabel(f"Mean {DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})")
        ax.set_title(DRUG_LABELS[drug])

    # Single legend — draw on the last panel since that tends to have the
    # least data overlap with typical y-axis ranges.
    axes[-1].legend(title="Site", loc="upper right", frameon=False)

    fig.suptitle(
        "Night-minus-day dose rate by ICU day 1–7 (full-24h coverage) "
        "— per site (mean ± 95% CI)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    save_agg_fig(fig, "night_day_diff_mean_trajectory_cross_site")


if __name__ == "__main__":
    main()
