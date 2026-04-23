"""Shared helpers for the nocturnal up-titration descriptive figures.

All scripts under code/descriptive/ are pure-Python and consume
output/analytical_dataset.parquet. This module centralizes:

  - threshold definitions (fent > 25/hr, prop > 10/hr, midaz > 1/hr)
  - drug label + color conventions (matching 07_descriptive.py)
  - day_n bucketing (1..7, "8+")
  - dataset loader and figure saver

Edit here once to propagate across every figure.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ── Paths (project root is CWD by convention; matches 07_descriptive.py) ──
ANALYTICAL_PARQUET = "output/analytical_dataset.parquet"
FIGURES_DIR = "output_to_share/figures"
TABLES_DIR = "output_to_share"


# ── Drug-level constants ──────────────────────────────────────────────────
DRUGS = ("prop", "fenteq", "midazeq")

DIFF_COLS = {
    "prop": "prop_dif",
    "fenteq": "fenteq_dif",
    "midazeq": "midazeq_dif",
}

DAY_COLS = {
    "prop": "_prop_day",
    "fenteq": "_fenteq_day",
    "midazeq": "_midazeq_day",
}

NIGHT_COLS = {
    "prop": "_prop_night",
    "fenteq": "_fenteq_night",
    "midazeq": "_midazeq_night",
}

# User-specified clinical thresholds on the hourly night-minus-day rate.
# Unit is mg/hr for prop + midazeq and mcg/hr for fenteq (matches the
# raw per-hr rates produced by 05_analytical_dataset.py).
THRESHOLDS = {
    "prop": 10,
    "fenteq": 25,
    "midazeq": 1,
}

DRUG_LABELS = {
    "prop": "Propofol",
    "fenteq": "Fentanyl eq.",
    "midazeq": "Midazolam eq.",
}

DRUG_UNITS = {
    "prop": "mg/hr",
    "fenteq": "mcg/hr",
    "midazeq": "mg/hr",
}

# Matches the palette used in 07_descriptive.py's hourly dose bar chart.
DRUG_COLORS = {
    "prop": "skyblue",
    "fenteq": "salmon",
    "midazeq": "mediumseagreen",
}


# ── Loaders + bucketing ───────────────────────────────────────────────────
def load_analytical() -> pd.DataFrame:
    """Load the analytical dataset from the default parquet path."""
    return pd.read_parquet(ANALYTICAL_PARQUET)


def cap_day(df: pd.DataFrame, max_day: int = 7, col: str = "_nth_day") -> pd.DataFrame:
    """Add `_nth_day_bin` column with ordered categorical values 1..max_day, '{max_day+1}+'.

    Days beyond `max_day` are collapsed into a single tail bucket so late-stay
    rows with thin N don't destabilize the trajectory plots.
    """
    tail = f"{max_day + 1}+"
    labels = [str(i) for i in range(1, max_day + 1)] + [tail]
    out = df.copy()
    day = out[col].astype(int)
    binned = day.astype(str)
    binned[day > max_day] = tail
    out["_nth_day_bin"] = pd.Categorical(binned, categories=labels, ordered=True)
    return out


def threshold_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean `<drug>_above` columns based on THRESHOLDS and DIFF_COLS."""
    out = df.copy()
    for drug in DRUGS:
        out[f"{drug}_above"] = out[DIFF_COLS[drug]] > THRESHOLDS[drug]
    return out


# ── Plotting helpers ──────────────────────────────────────────────────────
def apply_style() -> None:
    """Set matplotlib rcParams for consistent styling across figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def ensure_dirs() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)


def save_fig(fig, name: str) -> str:
    """Save `fig` as output_to_share/figures/{name}.png. Returns the path."""
    ensure_dirs()
    path = os.path.join(FIGURES_DIR, f"{name}.png")
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> str:
    """Save `df` as output_to_share/{name}.csv. Returns the path."""
    ensure_dirs()
    path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    print(f"Saved {path}")
    return path


def drug_axis_label(drug: str) -> str:
    """E.g. 'Propofol diff (mg/hr)' — used as per-panel y/x labels."""
    return f"{DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})"


def threshold_label(drug: str) -> str:
    """E.g. '> 10 mg/hr' — used for annotations and legends."""
    return f"> {THRESHOLDS[drug]} {DRUG_UNITS[drug]}"
