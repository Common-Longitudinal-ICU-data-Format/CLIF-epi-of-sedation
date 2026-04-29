"""6-way dose-pattern subgroup characteristics — per-drug stratified Table 1 + forest plot.

Supersedes a prior 2-arm "night-higher vs not" subcohort table by stratifying
into all six dose-pattern groups around ±T (see below). For each of the
three drugs, all patient-days from `exposure_dataset.parquet` are
classified into six neutral-terminology buckets (see
`_shared.categorize_diff_6way`):

    Markedly higher at day      diff < -T
    Slightly higher at day      -T <= diff < 0
    Equal, both zero            day == 0 AND night == 0     (off-drug holiday)
    Equal, both non-zero        diff == 0 AND day > 0       (truly stable)
    Slightly higher at night    0 < diff <= +T
    Markedly higher at night    diff > +T

The two `diff == 0` sub-cases are split intentionally — the off-drug-both-
shifts case is a drug-holiday day (very common for midazolam) and would
otherwise pollute the "Slightly higher at day" bucket.

Outputs (per drug, three drugs total):

  - `output_to_share/{site}/dose_pattern_subgroup_{drug}.csv` — tableone-
    style stratified Table 1 with one column per group (6 columns) plus a
    leading "Overall" anchor column. Demographics, severity, daytime
    sedation, downstream outcomes. PLUS a parallel `_partial_shift_n` row
    showing how much of each group's N is from coverage-artifact rows.
  - `output_to_share/{site}/figures/dose_pattern_subgroup_smd_{drug}.png`
    — forest plot of group-vs-overall standardized mean differences for
    the headline severity vars (SOFA total, CCI, age, day-shift dose rate).

Privacy: all outputs are aggregate (counts / means / medians / SDs / IQRs).
No row-level data, no IDs.

Usage:
    uv run python code/descriptive/dose_pattern_subgroup_characteristics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tableone

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DAY_COLS,
    DIFF_COLS,
    DOSE_PATTERN_COLORS,
    DOSE_PATTERN_LABELS,
    DRUG_LABELS,
    DRUGS,
    NIGHT_COLS,
    TABLES_DIR,
    THRESHOLDS,
    apply_style,
    categorize_diff_6way,
    ensure_dirs,
    load_exposure,
    prepare_diffs,
    save_fig,
)


# Headline severity / dose vars used by the forest plot. Subset of the
# tableone continuous-vars list so the SMD figure stays readable.
SMD_VARS = ("age", "sofa_total", "cci_score", "elix_score")


def _build_tableone(df: pd.DataFrame, drug: str, group_col: str) -> tableone.TableOne:
    """Build the per-drug Table 1 with the six dose-pattern groups as columns."""
    # Binary outcomes rendered as Yes/No so tableone shows one row per var
    # (matches the 06_table1.py pattern).
    d = df.copy()
    # tableone runs internal pivots that fail when the groupby column is a
    # pandas Categorical (the resulting columns Index becomes Categorical and
    # pandas merge's `df[:]` then raises InvalidIndexError on slice lookups).
    # Pass the group as plain object/string and supply ordering via `order=`.
    if isinstance(d[group_col].dtype, pd.CategoricalDtype):
        d[group_col] = d[group_col].astype(str)
    for c in (
        "_sbt_done_today", "_success_extub_today",
        "sbt_done_next_day", "success_extub_next_day",
    ):
        if c in d.columns:
            d[c] = d[c].map({0: "No", 1: "Yes"})

    continuous_vars = [
        "age", "_nth_day", "sofa_total", "cci_score", "elix_score",
        DAY_COLS[drug], NIGHT_COLS[drug],
    ]
    nonnormal_vars = [
        "_nth_day", "sofa_total", "cci_score", "elix_score",
        DAY_COLS[drug], NIGHT_COLS[drug],
    ]
    categorical_vars = [
        "sex_category", "icu_type",
        "_sbt_done_today", "_success_extub_today",
        "sbt_done_next_day", "success_extub_next_day",
    ]
    continuous_vars = [c for c in continuous_vars if c in d.columns]
    nonnormal_vars = [c for c in nonnormal_vars if c in d.columns]
    categorical_vars = [c for c in categorical_vars if c in d.columns]

    return tableone.TableOne(
        data=d,
        continuous=continuous_vars,
        categorical=categorical_vars,
        nonnormal=nonnormal_vars,
        groupby=group_col,
        order={group_col: list(DOSE_PATTERN_LABELS)},
        pval=True,
    )


def _add_partial_shift_row(out_df: pd.DataFrame, df: pd.DataFrame, group_col: str,
                           ) -> pd.DataFrame:
    """Append a `_partial_shift_n / %` row per group to the flattened tableone CSV.

    Per the plan, the parallel `_partial_shift` row makes coverage-artifact
    contribution visible per group without silently dropping those rows.
    """
    counts_per_group = df.groupby(group_col, observed=False)["_partial_shift_flag"].apply(
        lambda s: int(s.fillna(False).sum())
    ).reindex(list(DOSE_PATTERN_LABELS), fill_value=0)
    totals_per_group = df.groupby(group_col, observed=False).size().reindex(
        list(DOSE_PATTERN_LABELS), fill_value=0
    )
    pcts = (counts_per_group / totals_per_group.replace(0, np.nan) * 100).fillna(0)

    # tableone columns are the group labels (after MultiIndex flatten).
    new_row = {}
    for label in DOSE_PATTERN_LABELS:
        if label in out_df.columns:
            new_row[label] = f"{counts_per_group[label]:,} ({pcts[label]:.1f}%)"
    # Preserve any "Overall" anchor column if present.
    overall_total = int(df["_partial_shift_flag"].fillna(False).sum())
    overall_pct = 100.0 * overall_total / max(len(df), 1)
    for cand in ("Overall", "all", "Grouped by"):
        if cand in out_df.columns:
            new_row[cand] = f"{overall_total:,} ({overall_pct:.1f}%)"

    # Add as a labeled row at the bottom.
    new_row_series = pd.Series(new_row, name="_partial_shift_n (% of group)")
    out_df = pd.concat([out_df, new_row_series.to_frame().T], axis=0)
    return out_df


def _smd(a: pd.Series, b: pd.Series) -> float:
    """Standardized mean difference; pooled-SD denominator. NaN-safe."""
    a = a.dropna().astype(float)
    b = b.dropna().astype(float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled == 0 or not np.isfinite(pooled):
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def _draw_smd_forest(df: pd.DataFrame, drug: str, group_col: str) -> None:
    """Forest plot — per-group SMD vs overall (across the cohort) for SMD_VARS.

    Two-tone bars per group: solid for the all-rows SMD, hatched overlay for
    the SMD recomputed after dropping `_partial_shift_flag = True` rows.
    Same total length per group; the hatched portion shows what fraction of
    the SMD's signal could be attributable to coverage-artifact rows.
    """
    overall = df.copy()
    overall_no_partial = df[~df["_partial_shift_flag"].fillna(False)].copy()

    n_vars = len(SMD_VARS)
    # Tall figure: bottom ~38% reserved for the interpretation footnote.
    fig, axes = plt.subplots(
        1, n_vars, figsize=(3.5 * n_vars, 11.0), sharey=True,
    )
    if n_vars == 1:
        axes = [axes]

    y_positions = np.arange(len(DOSE_PATTERN_LABELS))[::-1]  # top→bottom: market night→day
    labels_top_down = list(DOSE_PATTERN_LABELS)[::-1]

    for ax, var in zip(axes, SMD_VARS):
        if var not in df.columns:
            ax.text(0.5, 0.5, f"{var} not in dataset", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(var)
            continue

        smds_all, smds_nopart = [], []
        for label in labels_top_down:
            grp_all = df.loc[df[group_col] == label, var]
            grp_nopart = overall_no_partial.loc[overall_no_partial[group_col] == label, var]
            smds_all.append(_smd(grp_all, overall[var]))
            smds_nopart.append(_smd(grp_nopart, overall_no_partial[var]))

        for y, label, smd_all, smd_no in zip(
            y_positions, labels_top_down, smds_all, smds_nopart
        ):
            color = DOSE_PATTERN_COLORS[label]
            if not np.isnan(smd_all):
                ax.barh(y, smd_all, color=color, edgecolor="dimgray",
                        height=0.7, alpha=0.85)
            if not np.isnan(smd_no):
                ax.barh(y, smd_no, color="none", edgecolor="black",
                        height=0.7, hatch="//", linewidth=0.6)
            if not np.isnan(smd_all):
                ax.text(smd_all, y, f" {smd_all:+.2f}", va="center",
                        ha="left" if smd_all >= 0 else "right", fontsize=7,
                        color="dimgray")

        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvspan(-0.10, 0.10, color="lightgray", alpha=0.25, zorder=0)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels_top_down, fontsize=8)
        ax.set_xlabel("SMD vs overall")
        ax.set_title(var, fontsize=10)

    fig.suptitle(
        f"{DRUG_LABELS[drug]} — Standardized Mean Differences by Dose-Pattern Group",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.38)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Each panel shows one variable. Each row is one of the six dose-pattern groups (defined below).\n"
        "Bar length = standardized mean difference (SMD) of that group's mean for that variable, vs the\n"
        "cohort overall mean.\n"
        "\n"
        "    SMD = (group mean − cohort overall mean) / pooled standard deviation\n"
        "\n"
        "An SMD of +0.30 on `sofa_total` for the \"Markedly higher at night\" row means: patient-days in\n"
        "that group have SOFA scores that are 0.30 SDs above the cohort-overall mean SOFA. Conventionally,\n"
        "|SMD| < 0.10 (the gray band) is considered ignorable; |SMD| > 0.20 is a meaningful imbalance.\n"
        "\n"
        "BAR TEXTURE\n"
        "  solid colored bar  → SMD computed over ALL patient-days in the group (including partial-shift\n"
        "                       rows, see glossary).\n"
        "  hatched overlay    → SMD recomputed AFTER dropping patient-days where _partial_shift_flag=True.\n"
        "                       The visible distance between the solid bar and the hatched overlay = the\n"
        "                       contribution of partial-shift rows to that group's signal. If the hatched\n"
        "                       bar is much shorter than the solid bar, the group's apparent characteristics\n"
        "                       are driven by coverage-artifact rows; the underlying complete-coverage\n"
        "                       subgroup is far less distinctive.\n"
        "\n"
        "GROUP DEFINITIONS (6-way classification around per-drug threshold T; see categorize_diff_6way)\n"
        "  Markedly higher at day      diff < -T            big day-shift dose excess vs night\n"
        "  Slightly higher at day      -T <= diff < 0       small day-shift dose excess vs night\n"
        "  Equal, both zero            day == 0 AND night == 0   off-drug (drug holiday / not yet started)\n"
        "  Equal, both non-zero        diff == 0 AND day > 0     truly stable, same dose across both shifts\n"
        "  Slightly higher at night    0 < diff <= +T       small night-shift dose excess vs day\n"
        "  Markedly higher at night    diff > +T            big night-shift dose excess vs day\n"
        "\n"
        "GLOSSARY\n"
        "  diff           — (per-hour rate during night-shift hours) − (per-hour rate during day-shift hours).\n"
        "  T (threshold)  — drug-specific clinically meaningful cutoff (10 mcg/kg/min for propofol,\n"
        "                    25 mcg/hr for fentanyl-eq, 1 mg/hr for midazolam-eq).\n"
        "  partial-shift  — _partial_shift_flag=True: one shift had ZERO hours of exposure window\n"
        "                    (e.g., intubation after 7 PM → 0 day-shift hours on day 0). NOT to be confused\n"
        "                    with a short-but-nonzero shift, which is NOT flagged because its per-hour rate\n"
        "                    is already hour-normalized.\n"
    )
    fig.text(
        0.04, 0.001, footnote,
        ha="left", va="bottom", fontsize=7.5, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, f"dose_pattern_subgroup_smd_{drug}")


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    ensure_dirs()
    if "_partial_shift_flag" not in df.columns:
        raise RuntimeError(
            "exposure_dataset.parquet missing _partial_shift_flag — "
            "re-run code/05_modeling_dataset.py against the current site."
        )

    for drug in DRUGS:
        diff_col = DIFF_COLS[drug]
        day_col = DAY_COLS[drug]
        night_col = NIGHT_COLS[drug]
        thr = THRESHOLDS[drug]

        d = df.copy()
        d[f"_pattern_{drug}"] = categorize_diff_6way(
            d[diff_col], d[day_col].fillna(0), d[night_col].fillna(0), thr,
        )
        # Drop rows we couldn't classify (missing diff/day/night).
        d = d.dropna(subset=[f"_pattern_{drug}"])

        # Per-group counts (printed for sanity; also embedded in the table).
        counts = d[f"_pattern_{drug}"].value_counts().reindex(
            list(DOSE_PATTERN_LABELS), fill_value=0
        )
        partial_counts = d.groupby(f"_pattern_{drug}", observed=False)[
            "_partial_shift_flag"
        ].apply(lambda s: int(s.fillna(False).sum())).reindex(
            list(DOSE_PATTERN_LABELS), fill_value=0
        )
        print(f"\n[{drug}] 6-way group counts (T = {thr}):")
        for label in DOSE_PATTERN_LABELS:
            print(f"  {label:35s}  n = {counts[label]:>7,}  "
                  f"(partial-shift: {partial_counts[label]:,})")

        t1 = _build_tableone(d, drug, f"_pattern_{drug}")
        out_df = t1.tableone.copy()
        if isinstance(out_df.columns, pd.MultiIndex):
            out_df.columns = out_df.columns.get_level_values(-1)
        out_df = _add_partial_shift_row(out_df, d, f"_pattern_{drug}")

        path = f"{TABLES_DIR}/dose_pattern_subgroup_{drug}.csv"
        out_df.to_csv(path)
        print(f"Wrote {path}")

        _draw_smd_forest(d, drug, f"_pattern_{drug}")


if __name__ == "__main__":
    main()
