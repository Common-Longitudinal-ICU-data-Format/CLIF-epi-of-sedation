"""Patient-level persistence + day-to-day transition figure for dose-pattern groups.

Answers "do they experience that every day?" — a question the cross-sectional
6-way subgroup table can't answer alone. Two panels per drug (so 3 drugs × 2
panels = 3 rows × 2 cols):

  Panel A — within-patient distribution.
      For each patient, compute the fraction of their patient-days in each
      of the 6 dose-pattern groups (denominator includes off-drug days).
      Plot as a 6-way stacked horizontal bar where each row is one patient,
      sorted by `% Markedly higher at night` descending. Visualizes whether
      night-heavier dosing is a *stable phenotype* (a few patients with
      consistently high % markedly-night) or an oscillating-day-to-day
      pattern (everyone with mixed ratios).
      Inset: % of patients with `>50%` of their days in each group.

      Each segment is drawn in two textures: solid for non-partial-shift
      days, hatched for partial-shift days. Total bar height (% in each
      group) unchanged; the texture decomposes contribution origin.

  Panel B — day-to-day transition matrix.
      For each pair of consecutive days `(t, t+1)` for the same patient,
      count the (group_t → group_t+1) transition. Render as a 6×6 heatmap
      of conditional probabilities `P(group_t+1 | group_t)`. Diagonal-heavy
      = persistent phenotype; off-diagonal-heavy = oscillating.
      A small companion bar reports `% of all transitions touching a partial-
      shift row`. A sensitivity sub-heatmap to the right shows the same
      matrix recomputed without partial-shift transitions.

CSV companion (per drug): `dose_pattern_transitions_{drug}.csv` with raw
counts + conditional probabilities per cell of the 6×6 matrix, for cross-
site pooling under code/agg/.

Privacy: all outputs are aggregate (per-patient fractions binned into a
heatmap, transition counts pooled across patients). No row-level data,
no IDs.

Usage:
    uv run python code/descriptive/dose_pattern_persistence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def _per_patient_fractions(d: pd.DataFrame, drug: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (fractions_df, partial_fractions_df).

    fractions_df: rows = hospitalization_id, cols = DOSE_PATTERN_LABELS,
                  values = per-patient % of days in each group (0..100, sums to 100 per row).
    partial_fractions_df: same shape, but values are % of patient's days in
                          each group that came from `_partial_shift_flag = True` rows.
    """
    pat_col = f"_pattern_{drug}"
    grouped = d.groupby(["hospitalization_id", pat_col], observed=False).size().unstack(
        fill_value=0
    )
    grouped = grouped.reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
    fractions = grouped.div(grouped.sum(axis=1), axis=0).fillna(0) * 100.0

    # Partial-shift sub-counts per (patient × group)
    partial = d[d["_partial_shift_flag"].fillna(False)].groupby(
        ["hospitalization_id", pat_col], observed=False
    ).size().unstack(fill_value=0)
    partial = partial.reindex(columns=list(DOSE_PATTERN_LABELS), fill_value=0)
    partial = partial.reindex(fractions.index, fill_value=0)
    partial_fractions = partial.div(grouped.sum(axis=1), axis=0).fillna(0) * 100.0

    return fractions, partial_fractions


def _draw_panel_a(ax, fractions: pd.DataFrame, partial_fractions: pd.DataFrame,
                  drug: str) -> None:
    """Stacked horizontal bar: one row per patient, sorted by markedly-night %."""
    sort_key = "Markedly higher at night"
    ordered = fractions.sort_values(sort_key, ascending=False)
    ordered_partial = partial_fractions.loc[ordered.index]
    n_patients = len(ordered)
    if n_patients == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        return

    y = np.arange(n_patients)
    cum = np.zeros(n_patients)
    for label in DOSE_PATTERN_LABELS:
        seg = ordered[label].to_numpy()
        seg_partial = ordered_partial[label].to_numpy()
        seg_solid = np.maximum(seg - seg_partial, 0)
        color = DOSE_PATTERN_COLORS[label]
        # Solid (non-partial) portion of each segment.
        ax.barh(y, seg_solid, left=cum, color=color, edgecolor="none", height=1.0,
                label=label)
        # Hatched (partial) portion stacked on top of the solid segment.
        ax.barh(y, seg_partial, left=cum + seg_solid, color="none",
                edgecolor="black", height=1.0, hatch="//", linewidth=0.0)
        cum += seg

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, n_patients - 0.5)
    ax.invert_yaxis()
    ax.set_xlabel("% of patient's days")
    ax.set_yticks([])
    ax.set_ylabel(f"Patients (n = {n_patients:,}; sorted by % markedly higher at night)",
                  fontsize=9)
    ax.set_title(f"{DRUG_LABELS[drug]} — within-patient distribution across 6 groups",
                 fontsize=10)

    # Inset: % of patients with >50% in each group.
    consistent = (fractions > 50).mean(axis=0) * 100.0
    text_lines = [
        f"% patients w/ >50% days in group:",
    ]
    for label in DOSE_PATTERN_LABELS:
        text_lines.append(f"  {label}: {consistent[label]:.1f}%")
    ax.text(
        1.02, 0.98, "\n".join(text_lines),
        transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
        family="monospace", color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="lightgray"),
    )


def _build_transition_matrix(d: pd.DataFrame, drug: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Return (counts_all, conditional_all, conditional_no_partial, partial_share).

    counts_all: 6×6 raw transition counts (group_t → group_t+1).
    conditional_all: row-normalized — P(group_t+1 | group_t) on all transitions.
    conditional_no_partial: same matrix recomputed after dropping any
        transition where either t or t+1 had `_partial_shift_flag = True`.
    partial_share: float in [0, 1], fraction of all transitions that touched a partial shift.
    """
    pat_col = f"_pattern_{drug}"
    cols = ["hospitalization_id", "_nth_day", pat_col, "_partial_shift_flag"]
    sub = d[cols].dropna(subset=[pat_col]).sort_values(["hospitalization_id", "_nth_day"])
    sub["_pattern_next"] = sub.groupby("hospitalization_id", observed=False)[pat_col].shift(-1)
    sub["_partial_next"] = sub.groupby("hospitalization_id", observed=False)[
        "_partial_shift_flag"
    ].shift(-1)
    sub = sub.dropna(subset=["_pattern_next"])

    labels = list(DOSE_PATTERN_LABELS)
    counts = pd.crosstab(
        sub[pat_col].astype("category"),
        sub["_pattern_next"].astype("category"),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0).astype(int)

    no_partial = sub[
        ~sub["_partial_shift_flag"].fillna(False)
        & ~sub["_partial_next"].fillna(False)
    ]
    counts_no_partial = pd.crosstab(
        no_partial[pat_col].astype("category"),
        no_partial["_pattern_next"].astype("category"),
        dropna=False,
    ).reindex(index=labels, columns=labels, fill_value=0).astype(int)

    cond_all = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    cond_no = counts_no_partial.div(
        counts_no_partial.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0)

    n_total = int(counts.sum().sum())
    n_partial_touched = n_total - int(counts_no_partial.sum().sum())
    partial_share = (n_partial_touched / n_total) if n_total > 0 else 0.0

    return counts, cond_all, cond_no, partial_share


def _draw_heatmap(ax, mat: pd.DataFrame, title: str) -> None:
    labels = list(DOSE_PATTERN_LABELS)
    arr = mat.reindex(index=labels, columns=labels).to_numpy()
    im = ax.imshow(arr, cmap="Reds", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("group at t+1")
    ax.set_ylabel("group at t")
    ax.set_title(title, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6.5,
                        color="white" if v > 0.5 else "black")
    return im


def _draw_panel_b(ax_main, ax_no_partial, ax_share,
                  cond_all: pd.DataFrame, cond_no: pd.DataFrame,
                  partial_share: float, drug: str) -> None:
    _draw_heatmap(ax_main, cond_all,
                  f"{DRUG_LABELS[drug]} — P(group_t+1 | group_t), all transitions")
    _draw_heatmap(ax_no_partial, cond_no,
                  "(sensitivity) excluding partial-shift transitions")
    ax_share.barh(0, partial_share * 100, color="#a6611a", edgecolor="white", height=0.6)
    ax_share.barh(0, 100 - partial_share * 100, left=partial_share * 100,
                  color="#dfc27d", edgecolor="white", height=0.6)
    ax_share.text(0.5, 0, f"{partial_share*100:.1f}% of transitions touch a partial-shift row",
                  ha="center", va="center", fontsize=8.5, transform=ax_share.transAxes)
    ax_share.set_xlim(0, 100)
    ax_share.set_ylim(-0.5, 0.5)
    ax_share.set_yticks([])
    ax_share.set_xticks([])
    for spine in ("top", "right", "left", "bottom"):
        ax_share.spines[spine].set_visible(False)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_exposure())
    ensure_dirs()
    if "_partial_shift_flag" not in df.columns:
        raise RuntimeError(
            "exposure_dataset.parquet missing _partial_shift_flag — "
            "re-run code/05_modeling_dataset.py against the current site."
        )

    # Layout: per-drug, 4 axes — Panel A (left, taller), then 3 stacked
    # axes on the right (sensitivity heatmap, share bar, main heatmap).
    # Implement as 3 rows × 4 cols grid; column 0 = Panel A, columns 1-3 =
    # Panel B with the share bar and sensitivity heatmap stacked.
    n_drugs = len(DRUGS)
    # Tall figure — bottom region of the canvas reserved for the inline
    # interpretation footnote (added after tight_layout).
    fig = plt.figure(figsize=(20, 7.0 * n_drugs + 6.5))
    gs = fig.add_gridspec(
        nrows=n_drugs, ncols=3,
        width_ratios=[2.0, 1.4, 1.4], height_ratios=[1.0] * n_drugs,
        hspace=0.55, wspace=0.45,
    )

    # We need a sub-gridspec inside each Panel B cell for the share bar.
    for r, drug in enumerate(DRUGS):
        diff_col = DIFF_COLS[drug]
        day_col = DAY_COLS[drug]
        night_col = NIGHT_COLS[drug]
        thr = THRESHOLDS[drug]

        d = df.copy()
        d[f"_pattern_{drug}"] = categorize_diff_6way(
            d[diff_col], d[day_col].fillna(0), d[night_col].fillna(0), thr,
        )
        d = d.dropna(subset=[f"_pattern_{drug}"]).copy()
        d[f"_pattern_{drug}"] = pd.Categorical(
            d[f"_pattern_{drug}"], categories=list(DOSE_PATTERN_LABELS), ordered=True,
        )

        fractions, partial_fractions = _per_patient_fractions(d, drug)
        ax_a = fig.add_subplot(gs[r, 0])
        _draw_panel_a(ax_a, fractions, partial_fractions, drug)

        # Sub-grid for Panel B: stacked share bar above the main heatmap.
        gs_b_main = gs[r, 1].subgridspec(2, 1, height_ratios=[1, 12], hspace=0.05)
        ax_share = fig.add_subplot(gs_b_main[0])
        ax_b_main = fig.add_subplot(gs_b_main[1])
        ax_b_sens = fig.add_subplot(gs[r, 2])

        counts, cond_all, cond_no, partial_share = _build_transition_matrix(d, drug)
        _draw_panel_b(ax_b_main, ax_b_sens, ax_share,
                      cond_all, cond_no, partial_share, drug)

        # Save the cross-site-pooling-friendly transition CSV (raw counts +
        # conditional probabilities). One row per (group_t, group_t+1, drug).
        rows = []
        for src in DOSE_PATTERN_LABELS:
            for dst in DOSE_PATTERN_LABELS:
                rows.append({
                    "group_t": src,
                    "group_t+1": dst,
                    "n_transitions": int(counts.loc[src, dst]),
                    "p_t+1_given_t_all": float(cond_all.loc[src, dst]),
                    "p_t+1_given_t_no_partial": float(cond_no.loc[src, dst]),
                })
        out_df = pd.DataFrame(rows)
        path = f"{TABLES_DIR}/dose_pattern_transitions_{drug}.csv"
        out_df.to_csv(path, index=False)
        print(f"Wrote {path}  (drug={drug}; partial-share={partial_share*100:.1f}%)")

    fig.suptitle(
        "Dose-pattern persistence and day-to-day transitions across the 6 groups",
        fontsize=13, y=0.995,
    )
    # Reserve bottom region for footnote.
    fig.subplots_adjust(bottom=0.22)

    footnote = (
        "HOW TO READ THIS FIGURE\n"
        "\n"
        "Three rows, one per drug. Each row has THREE plots:\n"
        "\n"
        "(1) PANEL A (left, wide bar) — within-patient distribution\n"
        "    One horizontal bar per patient. Total bar width is always 100% of that patient's days. The bar\n"
        "    is split into 6 colored segments — one per dose-pattern group — sized by the % of THAT PATIENT's\n"
        "    patient-days in each group. Patients are sorted top-to-bottom by their % \"Markedly higher at\n"
        "    night\" (descending), so the most night-heavy patients appear at the top.\n"
        "\n"
        "    Each colored segment is internally split by texture:\n"
        "      solid       → days the patient was fully covered (`_partial_shift_flag = False`).\n"
        "      hatched //  → days the patient had a zero-hour shift (`_partial_shift_flag = True`).\n"
        "    A patient whose entire \"Markedly higher at night\" segment is hatched got that classification\n"
        "    only because of zero-hour-day-shift artifacts (intubation after 7 PM); a patient with mostly\n"
        "    solid in the same segment has real night-heavier dosing across multiple full days.\n"
        "\n"
        "    Inset text reports `% of patients with > 50% of their days in group X` — the consistent\n"
        "    phenotypes. Concentrations there indicate stable patient phenotypes; if all 6 numbers are\n"
        "    small, every patient is mixed (no consistent phenotype).\n"
        "\n"
        "(2) PANEL B-MAIN (middle heatmap) — day-to-day transition matrix, all transitions\n"
        "    Rows = group at day t; columns = group at day t+1. Cell (i, j) = `P(group_t+1 = j | group_t = i)`.\n"
        "    Each row sums to 1.0. Computed by pooling all consecutive day pairs within each patient and\n"
        "    crosstabbing on the 6×6 group axis.\n"
        "      diagonal-heavy   → persistent: same group repeats day after day (stable phenotype).\n"
        "      off-diagonal     → oscillating: groups change day-to-day (unstable phenotype).\n"
        "    The horizontal stripe ABOVE the heatmap reports the % of all transitions where either day t\n"
        "    or day t+1 was a partial-shift row. High % means the matrix's signal is partly coverage-driven.\n"
        "\n"
        "(3) PANEL B-SENSITIVITY (rightmost heatmap) — same matrix, partial-shift transitions excluded\n"
        "    Identical layout to the main heatmap, but recomputed after dropping any (t, t+1) pair where\n"
        "    either day was partial-shift. If the diagonals are very similar between the two heatmaps, the\n"
        "    persistence story is robust to coverage. If they differ, partial-shift rows are doing real\n"
        "    work in the matrix and the sensitivity version is the one to trust.\n"
        "\n"
        "GROUP DEFINITIONS (6-way classification around per-drug threshold T)\n"
        "  Markedly higher at day      diff < -T            big day-shift dose excess vs night\n"
        "  Slightly higher at day      -T <= diff < 0       small day-shift dose excess vs night\n"
        "  Equal, both zero            day == 0 AND night == 0   off-drug both shifts (drug holiday)\n"
        "  Equal, both non-zero        diff == 0 AND day > 0     truly stable, same dose both shifts\n"
        "  Slightly higher at night    0 < diff <= +T       small night-shift dose excess vs day\n"
        "  Markedly higher at night    diff > +T            big night-shift dose excess vs day\n"
        "\n"
        "GLOSSARY\n"
        "  diff           — (per-hour rate during night-shift hours) − (per-hour rate during day-shift hours).\n"
        "  T (threshold)  — drug-specific clinically meaningful cutoff (10 mcg/kg/min for propofol,\n"
        "                    25 mcg/hr for fentanyl-eq, 1 mg/hr for midazolam-eq).\n"
        "  partial-shift  — `_partial_shift_flag = True`: one shift had ZERO hours of exposure window\n"
        "                    (e.g., intubation after 7 PM → 0 day-shift hours on day 0). NOT to be confused\n"
        "                    with a short-but-nonzero shift, which is NOT flagged.\n"
        "  consecutive    — successive days within the SAME hospitalization. Transitions between different\n"
        "                    hospitalizations are not counted.\n"
    )
    fig.text(
        0.02, 0.001, footnote,
        ha="left", va="bottom", fontsize=8, color="black", family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5",
                  edgecolor="#cccccc", linewidth=0.5),
    )
    save_fig(fig, "dose_pattern_persistence")


if __name__ == "__main__":
    main()
