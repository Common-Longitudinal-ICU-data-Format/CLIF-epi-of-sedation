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

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Site selection ────────────────────────────────────────────────────────
# Load site name from config/config.json once at import time so downstream
# paths (output/{site}/..., output_to_share/{site}/...) are consistent across
# every descriptive script. Lowercase convention matches the Makefile's
# SITE=mimic/ucmc flag.
def _load_site_name() -> str:
    cfg_path = Path("config/config.json")
    if not cfg_path.exists():
        # Fallback keeps imports from crashing in environments without a config
        # (e.g. docs build). Scripts will still fail gracefully at I/O time.
        return "unknown"
    with cfg_path.open() as f:
        cfg = json.load(f)
    return cfg.get("site_name", "unknown").lower()


SITE_NAME = _load_site_name()

# ── Paths (project root is CWD by convention; matches 07_descriptive.py) ──
# All outputs are site-scoped so multiple sites coexist on disk (see the
# Makefile's SITE= flag). Phase-2 cross-site aggregation reads these dirs.
ANALYTICAL_PARQUET = f"output/{SITE_NAME}/analytical_dataset.parquet"
FIGURES_DIR = f"output_to_share/{SITE_NAME}/figures"
TABLES_DIR = f"output_to_share/{SITE_NAME}"


# ── Drug-level constants ──────────────────────────────────────────────────
DRUGS = ("prop", "fenteq", "midazeq")

# Propofol uses the weight-adjusted diff column that prepare_diffs() computes
# on load (mcg/kg/min). Fentanyl-eq and midazolam-eq use the raw hourly-rate
# diffs from analytical_dataset.parquet directly.
DIFF_COLS = {
    "prop": "prop_dif_kgmin",
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

# Clinically meaningful night-minus-day dose-rate cutoffs. Propofol uses a
# weight-adjusted mcg/kg/min cutoff (matches bedside pump documentation);
# the other two are absolute hourly rates.
THRESHOLDS = {
    "prop": 10,       # mcg/kg/min — weight-adjusted
    "fenteq": 25,     # mcg/hr
    "midazeq": 1,     # mg/hr
}

DRUG_LABELS = {
    "prop": "Propofol",
    "fenteq": "Fentanyl eq.",
    "midazeq": "Midazolam eq.",
}

DRUG_UNITS = {
    "prop": "mcg/kg/min",
    "fenteq": "mcg/hr",
    "midazeq": "mg/hr",
}

# Matches the palette used in 07_descriptive.py's hourly dose bar chart.
DRUG_COLORS = {
    "prop": "skyblue",
    "fenteq": "salmon",
    "midazeq": "mediumseagreen",
}

# Sensitivity knob on the weight upper bound applied inside prepare_diffs().
# outlier_config.yaml already caps weight at 300 kg at ingestion; this env
# var additionally clips at use-time so tightening sensitivity analyses
# (e.g., MAX_WEIGHT_KG=250) don't require rerunning 04/05. Loosening past
# 300 requires editing the yaml and rerunning the upstream pipeline.
MAX_WEIGHT_KG = float(os.getenv("MAX_WEIGHT_KG", "300"))

# Diverging 4-color palette for the up-titration stacked bars.
# Dark blue (big down) → light blue → light red → dark red (big up).
# RdBu-inspired; keeps up-titrated tail in warm red for immediate reading.
DIFF_BIN_COLORS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]


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


def prepare_diffs(df: pd.DataFrame) -> pd.DataFrame:
    """Add `prop_dif_kgmin` to `df` and return it.

    Converts the existing mg/hr night-minus-day propofol rate into the
    weight-adjusted mcg/kg/min convention used clinically at the bedside:

        mg/hr × 1000 mcg/mg / 60 min/hr / weight_kg  ≡  mcg/kg/min

    Weight is clipped at MAX_WEIGHT_KG (env-var override; default 300) so
    we can do tightening-direction sensitivity without re-running upstream.
    Rows with missing weight propagate NaN, which downstream `.dropna()`
    filters drop from the relevant analyses.
    """
    out = df.copy()
    weight = out["weight_kg_asof_day_start"].clip(upper=MAX_WEIGHT_KG)
    out["prop_dif_kgmin"] = out["prop_dif"] * 1000.0 / 60.0 / weight
    return out


def categorize_diff(series: pd.Series, threshold: float) -> pd.Categorical:
    """Return a 4-level ordered Categorical bucketing `series` around ±threshold.

    Buckets (default pd.cut right-inclusive): `(-inf, -T]`, `(-T, 0]`, `(0, T]`, `(T, inf]`.
    Exact zero (very common — patient-days with no propofol change) lands in
    `(-T, 0]` ("small down / no change"). Labels kept short for legend use.
    """
    labels = [
        f"< −{threshold}",
        f"−{threshold} to 0",
        f"0 to {threshold}",
        f"> {threshold}",
    ]
    bins = [-np.inf, -threshold, 0, threshold, np.inf]
    return pd.cut(series, bins=bins, labels=labels, ordered=True)


# ── Plotting helpers ──────────────────────────────────────────────────────
def apply_style() -> None:
    """Set matplotlib rcParams for consistent styling across figures."""
    plt.rcParams.update({
        "figure.dpi": 120,
        # High savefig DPI so PNGs have enough pixels to stay crisp when
        # re-rasterized into 09_report's image pages at 300 DPI.
        "savefig.dpi": 250,
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
