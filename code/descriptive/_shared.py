"""Shared helpers for the diurnal-dose descriptive figures.

All scripts under code/descriptive/ are pure-Python and consume the
exposure dataset (or, for the single-shift audit specifically, the
modeling dataset). This module centralizes:

  - threshold definitions (fent > 25/hr, prop > 10 mcg/kg/min, midaz > 1/hr)
  - drug label + color conventions
  - day_n bucketing (1..7, "8+")
  - dataset loaders (load_exposure() / load_modeling()) and figure saver
  - 4-way and 6-way categorization helpers around ±threshold

Edit here once to propagate across every figure.

Terminology note: this module deliberately avoids the clinical term
"up-titration"/"down-titration" (which implies goal-directed dose change).
What we observe is purely descriptive — a higher dose at one shift than
another, with no information about why. Group labels use neutral wording:
"markedly higher at night" / "slightly higher at day" / etc.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from clifpy import setup_logging
from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.descriptive_shared")


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

# Per-site dual log files (pyCLIF integration guide rule 1). Every
# descriptive script is its own entry-point subprocess and imports this
# module exactly once, so this module-level call fires once per subprocess
# — equivalent to setup_logging-at-entry-point. Guarded against the
# unknown-site fallback above so a missing config doesn't crash imports.
if SITE_NAME != "unknown":
    os.makedirs(f"output/{SITE_NAME}", exist_ok=True)
    setup_logging(output_directory=f"output_to_share/{SITE_NAME}")

# ── Paths (project root is CWD by convention) ──
# All outputs are site-scoped so multiple sites coexist on disk (see the
# Makefile's SITE= flag). Phase-2 cross-site aggregation reads these dirs.
MODELING_PARQUET = f"output/{SITE_NAME}/modeling_dataset.parquet"
EXPOSURE_PARQUET = f"output/{SITE_NAME}/exposure_dataset.parquet"
# Phase 4 consolidated parquet — superset of MODELING + EXPOSURE plus the
# canonical-registry columns (`_is_full_24h_day`, `_is_last_partial_day`,
# `_is_last_full_day`, `n_hrs_day`, `n_hrs_night`, `day_type`, etc.).
# New consumers should prefer this; legacy `load_exposure` / `load_modeling`
# stay pointed at the legacy parquets during Phase 4 cutover.
MODEL_INPUT_PARQUET = f"output/{SITE_NAME}/model_input_by_id_imvday.parquet"
# As of the Path B++ refactor, every descriptive PNG and CSV lands FLAT in
# output_to_share/{site}/descriptive (no figures/ subdir). The thematic
# sibling directory `models/` houses everything written by 06/07/08/09's
# modeling code path.
FIGURES_DIR = f"output_to_share/{SITE_NAME}/descriptive"
TABLES_DIR = f"output_to_share/{SITE_NAME}/descriptive"
MODELS_DIR = f"output_to_share/{SITE_NAME}/models"


# ── Drug-level constants ──────────────────────────────────────────────────
DRUGS = ("prop", "fenteq", "midazeq")

# Phase 2 (2026-04-27): propofol columns are now produced directly in
# mcg/kg/min by 05_modeling_dataset.py — Stage A (clifpy) hands off
# in /kg units thanks to the pre-attached weight column in 02_exposure.py,
# so the descriptive layer no longer needs to divide by weight. Fentanyl-eq
# and midazolam-eq still use raw hourly-rate diffs (no /kg conversion).
# Column-name convention: `_mg_hr` = mg per hour; `_mcg_hr` = mcg per hour;
# `_mcg_kg_min` = mcg per kg per minute (propofol only).
DIFF_COLS = {
    "prop": "prop_dif_mcg_kg_min",
    "fenteq": "fenteq_dif_mcg_hr",
    "midazeq": "midazeq_dif_mg_hr",
}

# `_total` suffix = continuous + intermittent combined, the canonical
# modeling scope produced by 05_modeling_dataset.py. The legacy unsuffixed
# names were retired when the cont-vs-intm scope split was made explicit
# in the per-day registry; `_cont`-suffixed siblings exist for the
# continuous-only sensitivity pipeline (see project_cont_intm_diagnostic.md).
DAY_COLS = {
    "prop": "_prop_day_mcg_kg_min_total",
    "fenteq": "_fenteq_day_mcg_hr_total",
    "midazeq": "_midazeq_day_mg_hr_total",
}

NIGHT_COLS = {
    "prop": "_prop_night_mcg_kg_min_total",
    "fenteq": "_fenteq_night_mcg_hr_total",
    "midazeq": "_midazeq_night_mg_hr_total",
}

# Hurdle flags marking patient-days where the drug was administered at all
# (day_rate > 0 OR night_rate > 0). Produced by code/05_modeling_dataset.py:332-337
# as the canonical "on drug that day" indicator. Used as the bottom-panel
# denominator in cross-site proportion figures and anywhere the hurdle
# decomposition needs to split off-drug rows.
ON_DRUG_FLAGS = {
    "prop": "_prop_any",
    "fenteq": "_fenteq_any",
    "midazeq": "_midazeq_any",
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

DRUG_COLORS = {
    "prop": "skyblue",
    "fenteq": "salmon",
    "midazeq": "mediumseagreen",
}

# Six-group dose-pattern labels (the user-requested split that separates
# off-drug rows from real same-dose rows). Used by §5/§6 figures and the
# paradox panel. Ordered so that the "Markedly higher at day" -> "Markedly
# higher at night" axis reads left-to-right naturally on heatmaps.
DOSE_PATTERN_LABELS = (
    "Markedly higher at day",     # diff < -T
    "Slightly higher at day",     # -T <= diff < 0
    "Not receiving that day",           # day == 0 AND night == 0  (drug-holiday)
    "Receiving and equal",       # diff == 0 AND day > 0    (truly stable)
    "Slightly higher at night",   # 0 < diff <= +T
    "Markedly higher at night",   # diff > +T
)

# Diverging palette tuned to the 6-way axis. Reds for night-higher, blues
# for day-higher, neutral grays for the equal sub-cases (off-drug holiday
# in lighter gray, truly-stable in darker gray since it's the rarer signal).
DOSE_PATTERN_COLORS = {
    "Markedly higher at day":     "#2166ac",  # dark blue
    "Slightly higher at day":     "#92c5de",  # light blue
    "Not receiving that day":           "#dddddd",  # light gray (off-drug)
    "Receiving and equal":       "#9e9e9e",  # mid gray (truly-stable, rare)
    "Slightly higher at night":   "#f4a582",  # light red
    "Markedly higher at night":   "#b2182b",  # dark red
}

# Stack order for count-by-ICU-day bar charts. Differs from
# DOSE_PATTERN_LABELS: the drug-holiday band ("Not receiving that day") is moved
# to the TOP of the bar so total bar height = cohort survivors and the
# on-drug zone (the 5 remaining areas) sits as a contiguous bottom block.
# See descriptive_figures.md §6.0.
COUNT_BAR_STACK_ORDER = (
    "Markedly higher at day",
    "Slightly higher at day",
    "Receiving and equal",       # on-drug, no diff
    "Slightly higher at night",
    "Markedly higher at night",
    "Not receiving that day",           # off-drug — placed last (top of bar)
)

# Sensitivity knob on the weight upper bound applied inside prepare_diffs().
# outlier_config.yaml already caps weight at 300 kg at ingestion; this env
# var additionally clips at use-time so tightening sensitivity analyses
# (e.g., MAX_WEIGHT_KG=250) don't require rerunning 04/05. Loosening past
# 300 requires editing the yaml and rerunning the upstream pipeline.
MAX_WEIGHT_KG = float(os.getenv("MAX_WEIGHT_KG", "300"))

# Diverging 4-color palette for the night-vs-day stacked bars.
# Dark blue (markedly higher at day) → light blue (slightly higher at day)
# → light red (slightly higher at night) → dark red (markedly higher at night).
# RdBu-inspired; keeps the night-higher tail in warm red for immediate reading.
DIFF_BIN_COLORS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]


# ── Loaders + bucketing ───────────────────────────────────────────────────
def load_modeling() -> pd.DataFrame:
    """Load the modeling dataset (production outcome-modeling cohort).

    Phase 4 cutover (2026-05-08): reads the consolidated
    `model_input_by_id_imvday.parquet` and applies the outcome-modeling
    filter inline. Byte-equivalent to the legacy
    `modeling_dataset.parquet` row set on the surviving cohort —
    verified at both sites: 43,119 rows / 9,119 hosps (UCMC),
    48,092 / 11,628 (MIMIC). Filter:

      `_nth_day > 0 AND sbt_done_next_day IS NOT NULL
       AND success_extub_next_day IS NOT NULL`

    Drops day 0 and trajectory-final partial rows, so every surviving
    row has a well-defined next-day outcome. Use this for partial-shift
    audits that mirror what the production models see.
    """
    df = pd.read_parquet(MODEL_INPUT_PARQUET)
    return df.loc[
        (df["_nth_day"] > 0)
        & df["sbt_done_next_day"].notna()
        & df["success_extub_next_day"].notna()
    ].reset_index(drop=True)


def load_exposure() -> pd.DataFrame:
    """Load the exposure dataset (full hospital-stay coverage).

    Legacy loader kept during Phase 4 cutover. Includes day 0 AND last
    day with the wide-semantic `_is_last_day` (= max-`_nth_day` per
    hosp). Carries `_single_shift_day`, `_is_first_day`, `_is_last_day`
    legacy flags. New descriptive figures should prefer
    `load_model_input()` (canonical Phase 4 source with explicit
    `_is_full_24h_day`, `_is_last_partial_day`, `_is_last_full_day`
    flags from the patient-day registry).
    """
    return pd.read_parquet(EXPOSURE_PARQUET)


def load_model_input() -> pd.DataFrame:
    """Load the Phase 4 consolidated per-day modeling input.

    One row per (hospitalization_id, _nth_day), base table = the canonical
    registry `cohort_meta_by_id_imvday`. Carries every per-day source's
    columns plus the registry day-flags (`_is_full_24h_day`,
    `_is_last_partial_day`, `_is_last_full_day`, `n_hrs_day`,
    `n_hrs_night`, `day_type`). Use this when a figure wants explicit
    full-vs-partial coverage filtering — `WHERE _is_full_24h_day` drops
    intubation-day and extubation-day partial rows in one filter.
    """
    return pd.read_parquet(MODEL_INPUT_PARQUET)


def load_analytical() -> pd.DataFrame:
    """DEPRECATED — kept as a thin alias to load_modeling() for any straggler.

    The Apr-2026 refactor split the old `analytical_dataset.parquet` into
    `modeling_dataset.parquet` (this loader) and `exposure_dataset.parquet`
    (the descriptive default; see `load_exposure`). New code should call
    one of those two by name.
    """
    return load_modeling()


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
    """Pass-through for the propofol diff column.

    Phase 2 (2026-04-27): `prop_dif_mcg_kg_min` is now produced directly by
    05_modeling_dataset.py — the upstream pipeline pre-attaches a project-
    controlled weight column on each medication-admin row (skipping clifpy's
    no-fallback ASOF), then converts propofol to mcg/kg/min preferred unit at
    Stage A. The descriptive layer no longer needs to divide by weight.

    The function is kept as a thin pass-through so existing callers
    (`night_day_diff_*`, `pct_night_vs_day_*`, `diff_tail_contribution`, etc.)
    continue to work without changes. `MAX_WEIGHT_KG` is no longer applied
    here; the weight clamp lives upstream in `config/outlier_config.yaml`.
    """
    return df


def categorize_diff_6way(
    diff: pd.Series, day: pd.Series, night: pd.Series, threshold: float,
) -> pd.Categorical:
    """Return a 6-level ordered Categorical splitting the dose-pattern axis.

    Six neutral-terminology groups in fixed order (see DOSE_PATTERN_LABELS):

        Markedly higher at day  (diff < -T)
        Slightly higher at day  (-T <= diff < 0)
        Not receiving that day        (day == 0 AND night == 0)        ← off-drug
        Receiving and equal    (diff == 0 AND day > 0)          ← truly stable
        Slightly higher at night (0 < diff <= +T)
        Markedly higher at night (diff > +T)

    The two `diff == 0` sub-cases are split intentionally: the off-drug-both-
    shifts case is a drug-holiday day (very common for midazolam) and would
    otherwise pollute the "Slightly higher at day" bucket via pd.cut's right-
    inclusive default. Pass actual day/night columns (not just the diff) so
    the split can read whether the equality came from 0=0 or X=X.

    NaN handling: NaN rate on a shift means that shift had zero hours of
    exposure (a single-shift / coverage-artifact row, see
    `_single_shift_day`). Such rows are TREATED AS ZERO-RATE on the missing
    shift and re-classified accordingly, so single-shift rows surface in
    whichever bucket their finite-shift dose falls into rather than being
    silently dropped. The `_single_shift_day` column carries the artifact
    signal into figures' texture/overlay layer.

    Returns NaN only for rows where BOTH day and night are NaN (which
    shouldn't occur in this project's exposure_dataset).
    """
    labels = list(DOSE_PATTERN_LABELS)
    day_f = day.fillna(0).astype(float)
    night_f = night.fillna(0).astype(float)
    # Recompute diff from filled day/night so single-shift rows are finite.
    # Falls back to user-supplied diff when both day and night are present.
    diff_use = night_f - day_f

    out = pd.Series(pd.NA, index=diff_use.index, dtype="object")
    valid = day.notna() | night.notna()  # at least one shift had observable data

    out.loc[valid & (diff_use < -threshold)] = labels[0]                          # markedly day
    out.loc[valid & (diff_use >= -threshold) & (diff_use < 0)] = labels[1]        # slightly day
    out.loc[valid & (day_f == 0) & (night_f == 0)] = labels[2]                    # off-drug
    out.loc[valid & (diff_use == 0) & (day_f > 0)] = labels[3]                    # truly-stable
    out.loc[valid & (diff_use > 0) & (diff_use <= threshold)] = labels[4]         # slightly night
    out.loc[valid & (diff_use > threshold)] = labels[5]                           # markedly night

    return pd.Categorical(out, categories=labels, ordered=True)


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
    logger.info(f"Saved {path}")
    return path


def save_csv(df: pd.DataFrame, name: str, index: bool = False) -> str:
    """Save `df` as output_to_share/{name}.csv. Returns the path."""
    ensure_dirs()
    path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(path, index=index)
    logger.info(f"Saved {path}")
    return path


def drug_axis_label(drug: str) -> str:
    """E.g. 'Propofol diff (mg/hr)' — used as per-panel y/x labels."""
    return f"{DRUG_LABELS[drug]} diff ({DRUG_UNITS[drug]})"


def threshold_label(drug: str) -> str:
    """E.g. '> 10 mg/hr' — used for annotations and legends."""
    return f"> {THRESHOLDS[drug]} {DRUG_UNITS[drug]}"
