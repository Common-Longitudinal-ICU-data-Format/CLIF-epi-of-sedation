"""Stacked-bar of patient-day distribution around the night-day up-titration threshold.

Three stacks, one per drug (propofol / fent-eq / midaz-eq). Each stack is 100%
tall, split into four diverging categories:
    < −T   (big down-titration)
    −T to 0 (small down / stable)
    0 to T (small up)
    > T    (clinically meaningful up-titration — the "subcohort")

This replaces a single-proportion bar: the four-way split answers
"what fraction went up, stayed flat, or came down" in one image.

Usage:
    uv run python code/descriptive/pct_uptitrated_overall.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    DIFF_BIN_COLORS,
    DIFF_COLS,
    DRUG_LABELS,
    DRUGS,
    DRUG_UNITS,
    THRESHOLDS,
    apply_style,
    categorize_diff,
    load_analytical,
    prepare_diffs,
    save_fig,
)


def main() -> None:
    apply_style()
    df = prepare_diffs(load_analytical())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DRUGS))

    # Category order must match DIFF_BIN_COLORS (bottom-up on the stack):
    # big-down → small-down → small-up → big-up.
    category_order_for_legend: list[str] = []

    for xi, drug in enumerate(DRUGS):
        col = DIFF_COLS[drug]
        thr = THRESHOLDS[drug]
        cat = categorize_diff(df[col].dropna(), thr)
        counts = cat.value_counts().reindex(cat.cat.categories, fill_value=0)
        total = int(counts.sum())
        pcts = (counts / total * 100.0).to_numpy() if total else np.zeros(4)

        # Stack from bottom up using cumulative offsets.
        cum = 0.0
        for seg_idx, (label, pct) in enumerate(zip(counts.index, pcts)):
            ax.bar(xi, pct, bottom=cum, width=0.62,
                   color=DIFF_BIN_COLORS[seg_idx], edgecolor="white", linewidth=0.4,
                   label=str(label) if xi == 0 else None)
            if pct >= 3.0:  # annotate only segments large enough to fit text
                ax.text(xi, cum + pct / 2, f"{pct:.1f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if seg_idx in (0, 3) else "black")
            cum += pct

        if xi == 0:
            category_order_for_legend = list(counts.index)

        # Total N under the bar.
        ax.text(xi, -4, f"n = {total:,}", ha="center", va="top",
                fontsize=8, color="dimgray")

    # Axis + labels.
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{DRUG_LABELS[d]}\n(T = {THRESHOLDS[d]} {DRUG_UNITS[d]})" for d in DRUGS
    ])
    ax.set_ylabel("% of patient-days")
    ax.set_ylim(-8, 105)
    ax.set_title(
        "Distribution of night-minus-day dose-rate diff around clinical threshold T",
        fontsize=12, pad=10,
    )

    # Legend explains the 4 categories (drawn once from the first drug's bars).
    handles, labels = ax.get_legend_handles_labels()
    # Reorder so the legend reads top-to-bottom matching the visual stack.
    order_top_down = list(reversed(category_order_for_legend))
    h_by_label = dict(zip(labels, handles))
    ax.legend(
        [h_by_label[l] for l in order_top_down],
        order_top_down,
        title="Diff category (vs threshold T)",
        loc="upper right", bbox_to_anchor=(1.28, 1.0), frameon=False,
    )

    fig.tight_layout()
    save_fig(fig, "pct_uptitrated_overall")


if __name__ == "__main__":
    main()
