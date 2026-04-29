"""Distribution of `n_hours_day` and `n_hours_night` across patient-days.

Reads `output/{site}/sed_dose_daily.parquet` (the pre-filter table — the
modeling dataset drops most partial shifts via its
`sbt_done_next_day IS NOT NULL` filter). Two panels:

  - Bar of the full integer distribution of `n_hours_day` and
    `n_hours_night`, side-by-side.
  - A small inset table reporting what fraction of patient-days has
    full 12h / partial / zero-length shifts, plus any >12h values
    (DST artifacts).

This is a permanent diagnostic: if upstream 01/02 ever regresses and
starts leaking partial-shift rows into the modeling dataset, this
figure should immediately flag it.

Usage:
    uv run python code/descriptive/partial_shift_diagnostics.py
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

    sd = pd.read_parquet(f"output/{SITE_NAME}/sed_dose_daily.parquet")
    ad = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")

    # Flag kept-vs-dropped from the modeling filter so the plot shows how
    # much partial-shift exposure is actually inherited downstream.
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
        f"Partial-shift diagnostics — {SITE_NAME.upper()} "
        "(log scale; kept = retained in modeling_dataset.parquet)",
        fontsize=13, y=1.00,
    )
    fig.subplots_adjust(bottom=0.30)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Two bar panels: LEFT = day-shift hours, RIGHT = night-shift hours. X-axis: integer hours\n"
        "(0..14) in that shift. Y-axis: number of patient-days with that many hours of exposure\n"
        "(LOG SCALE — small bars are a small fraction of the cohort).\n"
        "\n"
        "  light blue bars  → ALL rows in `sed_dose_daily.parquet` (pre-filter, the full hospital-stay\n"
        "                      coverage).\n"
        "  dark blue bars   → the subset that survived into `modeling_dataset.parquet` (after the\n"
        "                      `_nth_day > 0 AND sbt_done_next_day IS NOT NULL` filter).\n"
        "\n"
        "  full 12h     — the normal case. A patient who started the shift before 7 AM (resp. 7 PM)\n"
        "                  and ended after the 12-hour mark.\n"
        "  partial <12h — patient was intubated mid-shift, extubated mid-shift, or died/transferred.\n"
        "  zero-length  — `n_hours_<shift> = 0`. The patient had no exposure window at all on that\n"
        "                  shift (e.g., intubated after 7 PM → 0 day-shift hours on day 0).\n"
        "  > 12h (DST)  — daylight-saving artifact; expected to be a few rows.\n"
        "\n"
        "USE THIS FIGURE AS A REGRESSION CANARY\n"
        "  The dark blue bars should be concentrated almost entirely at h = 12. If you see a sudden\n"
        "  expansion of dark blue bars at h < 12, the upstream pipeline is leaking partial-shift rows\n"
        "  into the modeling cohort — investigate `04_covariates.py` / `05_modeling_dataset.py` filters.\n"
        "\n"
        "  The light blue bars should look qualitatively similar between sites (mostly h = 12, with\n"
        "  some bumps at h = 0 from intubation-after-7-PM cases on day 0). The summary table at the\n"
        "  bottom of the figure quantifies the % full-12h vs partial vs zero-length per source.\n"
        "\n"
        "GLOSSARY\n"
        "  modeling_dataset.parquet — the outcome-modeling cohort (filtered, no day 0, no last day).\n"
        "  sed_dose_daily.parquet   — the pre-filter source: all hospitalization-days with sedative\n"
        "                              data, including day 0 / last day / zero-hour shifts.\n"
    )
    fig.text(
        0.04, 0.001, footnote,
        ha="left", va="bottom", fontsize=8, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, "partial_shift_diagnostics")


if __name__ == "__main__":
    main()
