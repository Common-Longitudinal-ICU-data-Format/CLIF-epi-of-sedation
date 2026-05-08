"""Distribution of `n_hours_day` and `n_hours_night` across patient-days.

Reads `output/{site}/seddose_by_id_imvday.parquet` (the pre-filter table — the
modeling dataset drops most single-shift days via its
`sbt_done_next_day IS NOT NULL` filter). Two panels:

  - Bar of the full integer distribution of `n_hours_day` and
    `n_hours_night`, side-by-side.
  - A small inset table reporting what fraction of patient-days has
    full 12h / partial / zero-length shifts, plus any >12h values
    (DST artifacts).

This is a permanent diagnostic: if upstream 01/02 ever regresses and
starts leaking single-shift rows into the modeling dataset, this
figure should immediately flag it.

Usage:
    uv run python code/descriptive/single_shift_diagnostics.py
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
    save_fig,
)


def _summary_row(s: pd.Series, label: str) -> dict:
    s = s.fillna(0).astype(float)
    return {
        "shift": label,
        "n": len(s),
        "full 12h": f"{100 * (s == 12).mean():.1f}%",
        "partial <12h": f"{100 * ((s > 0) & (s < 12)).mean():.1f}%",
        "zero-length": f"{100 * (s == 0).mean():.1f}%",
        ">12h (DST)": f"{100 * (s > 12).mean():.2f}%",
        "median": f"{s.median():.1f}",
        "mean": f"{s.mean():.2f}",
    }


def main() -> None:
    apply_style()

    sd = pd.read_parquet(f"output/{SITE_NAME}/seddose_by_id_imvday.parquet")
    ad = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")

    # Flag kept-vs-dropped from the modeling filter so the plot shows how
    # much single-shift exposure is actually inherited downstream.
    ad["_nth_day"] = ad["_nth_day"].astype(int)
    sd["_nth_day"] = sd["_nth_day"].astype(int)
    kept_ids = ad[["hospitalization_id", "_nth_day"]].assign(_kept=1)
    sd = sd.merge(kept_ids, on=["hospitalization_id", "_nth_day"], how="left")
    sd["_kept"] = sd["_kept"].fillna(0).astype(int)

    # Tall figure — bottom region reserved for the inline footnote.
    fig = plt.figure(figsize=(14, 11.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.5, 1], hspace=0.55, wspace=0.25)

    hours_range = list(range(0, 15))

    for col_i, shift_col, title in [
        (0, "n_hours_day", "n_hours_day"),
        (1, "n_hours_night", "n_hours_night"),
    ]:
        ax = fig.add_subplot(gs[0, col_i])
        pre = sd[shift_col].fillna(0).astype(int)
        kept = sd.loc[sd["_kept"] == 1, shift_col].fillna(0).astype(int)
        bins = np.array(hours_range + [15]) - 0.5
        pre_counts, _ = np.histogram(pre, bins=bins)
        kept_counts, _ = np.histogram(kept, bins=bins)

        ax.bar(hours_range, pre_counts, width=0.85, color="#9ecae1",
               edgecolor="white", label="sed_dose_daily (pre-filter)")
        ax.bar(hours_range, kept_counts, width=0.85, color="#08519c",
               edgecolor="white", label="modeling_dataset (kept)")
        ax.set_xlabel("Hours in shift")
        ax.set_ylabel("Patient-days")
        ax.set_title(f"{title} — {SITE_NAME.upper()}")
        ax.set_xticks(hours_range)
        ax.legend(loc="upper left")
        ax.set_yscale("log")

    # Summary table spanning full width.
    ax_t = fig.add_subplot(gs[1, :])
    ax_t.axis("off")
    rows = []
    for label_suffix, df_in in [("sed_dose_daily (pre)", sd),
                                 ("modeling (kept)", sd[sd["_kept"] == 1])]:
        rows.append({"source": label_suffix, **_summary_row(df_in["n_hours_day"], "day")})
        rows.append({"source": label_suffix, **_summary_row(df_in["n_hours_night"], "night")})
    table_df = pd.DataFrame(rows)
    tbl = ax_t.table(
        cellText=table_df.values,
        colLabels=list(table_df.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.3)
    for (i, _), cell in tbl.get_celld().items():
        if i == 0:
            cell.set_facecolor("#d9d9d9")
            cell.set_text_props(weight="bold")

    fig.suptitle(
        f"Single-shift diagnostics — {SITE_NAME.upper()} (log scale)\n"
        "Light blue = sed_dose_daily.parquet (pre-filter). "
        "Dark blue = subset retained in modeling_dataset.parquet "
        "(after _nth_day>0 AND sbt_done_next_day IS NOT NULL).",
        fontsize=11, y=1.00,
    )
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.5, 0.02,
        "Regression canary: the dark-blue (modeling-cohort) bars should be concentrated at h=12. "
        "An expansion of dark-blue bars at h<12 means single-shift rows are leaking into the modeling cohort "
        "— investigate 04_covariates.py / 05_modeling_dataset.py. Light blue bars typically include some h=0 "
        "rows from intubation-after-7-PM cases on day 0 (expected).",
        ha="center", va="bottom", fontsize=8, color="dimgray", wrap=True,
    )
    save_fig(fig, "single_shift_diagnostics")


if __name__ == "__main__":
    main()
