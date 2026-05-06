"""Shared helpers for the cross-site descriptive pooling scripts.

All scripts under code/agg/ read each site's output_to_share/{site}/ artifacts
and write pooled/stacked outputs to output_to_agg/. This module centralizes:

  - site discovery (list_sites): scans output_to_share/*/ for real site dirs
  - site labeling (site_label): honors ANONYMIZE_SITES env var for blinded output
  - analytical dataset loader keyed on site name
  - figure/table save helpers targeting output_to_agg/ (figures/ subdir)
  - re-exports of drug constants from code/descriptive/_shared so palettes
    and thresholds stay in lockstep with per-site descriptive figures

Edit here once to propagate across every cross-site figure.

─────────────────────────────────────────────────────────────────────────
HANDOFF NOTES — for team-lead reconciliation

Team-lead should reconcile the following after this task completes:

1. Makefile — existing `agg:` target at lines 102-115 already globs code/agg/*.py
   (excluding nothing); since _shared.py is a module not a script, confirm the
   shebang-skip behavior. Current glob runs every *.py including _shared.py —
   that's harmless (imports + defines functions, main guard not needed) but
   the team-lead may want to add the same `case "$$(basename $$script)" in _*)
   continue ;; esac` filter that _descriptive_scripts uses. Not strictly required.

2. .dev/CLAUDE.md — "Multi-site support" section references Phase-2 aggregation
   as reserved. This task implements Phase-2 descriptive pooling. Status can be
   flipped from "reserved" to "implemented (descriptive only)"; model-coefficient
   forest plots remain deferred (per user).

3. docs/analysis_plan.md — may want a bullet under §6 (or equivalent) noting
   that cross-site pooled Table 1, cross-site cohort stats, cross-site
   cross-site night_day_diff_mean trajectory overlay
   are now produced under output_to_agg/ via `make agg`.

4. Memory — plan_multisite_layout.md currently says "output_to_agg/ + code/agg/
   reserved". After team-lead accepts this task, update that memory entry to
   reflect that the descriptive pooling scaffold is now in place (model
   coefficient forest plots still deferred).
─────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Reuse per-site descriptive constants so the cross-site figures match
# their single-site counterparts (DRUG_COLORS, DIFF_BIN_COLORS, THRESHOLDS,
# cap_day, prepare_diffs). We load code/descriptive/_shared.py under a
# DIFFERENT module name (`_descriptive_shared`) to avoid the circular
# import that would otherwise happen: each agg script does
# `sys.path.insert(0, code/agg/)` and then `from _shared import ...`, which
# binds `_shared` to THIS file. Using importlib with a distinct name
# sidesteps the name collision without requiring agg scripts to know the
# distinction.
import importlib.util as _iu  # noqa: E402

_DESCRIPTIVE_SHARED_PATH = (
    Path(__file__).resolve().parent.parent / "descriptive" / "_shared.py"
)
_spec = _iu.spec_from_file_location("_descriptive_shared", _DESCRIPTIVE_SHARED_PATH)
if _spec is None or _spec.loader is None:
    raise RuntimeError(f"Cannot locate {_DESCRIPTIVE_SHARED_PATH}")
_descriptive_shared = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_descriptive_shared)

DIFF_BIN_COLORS = _descriptive_shared.DIFF_BIN_COLORS
DIFF_COLS = _descriptive_shared.DIFF_COLS
DRUG_COLORS = _descriptive_shared.DRUG_COLORS
DRUG_LABELS = _descriptive_shared.DRUG_LABELS
DRUG_UNITS = _descriptive_shared.DRUG_UNITS
DRUGS = _descriptive_shared.DRUGS
THRESHOLDS = _descriptive_shared.THRESHOLDS
cap_day = _descriptive_shared.cap_day
prepare_diffs = _descriptive_shared.prepare_diffs
# Also expose descriptive's apply_style so agg scripts can import it via
# `from _shared import ...` without knowing the underlying trampoline.
apply_style = _descriptive_shared.apply_style

# Dose-pattern primitives — used by the cross-site stacked-bar figure
# (dose_pattern_6group_count_by_icu_day_cross_site.py). Same trampoline
# pattern as the constants above; kept in lockstep with the per-site
# descriptive figure so colors and category order match.
COUNT_BAR_STACK_ORDER = _descriptive_shared.COUNT_BAR_STACK_ORDER
DAY_COLS = _descriptive_shared.DAY_COLS
NIGHT_COLS = _descriptive_shared.NIGHT_COLS
DOSE_PATTERN_COLORS = _descriptive_shared.DOSE_PATTERN_COLORS
DOSE_PATTERN_LABELS = _descriptive_shared.DOSE_PATTERN_LABELS
categorize_diff_6way = _descriptive_shared.categorize_diff_6way

# ── Site color palette ────────────────────────────────────────────────────
# Qualitative palette with good contrast for ≤6 sites; loops with a warning
# if a consortium ever exceeds that. Chosen from matplotlib's tab10 so the
# colors play nicely with the drug-specific palette used elsewhere (which
# uses skyblue / salmon / mediumseagreen for the per-site single-panel view).
# Indexed by alphabetical site order so each site gets the same color across
# every cross-site figure.
SITE_PALETTE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#8c564b",  # brown
]


# ── Paths ─────────────────────────────────────────────────────────────────
# Project root is CWD by convention (matches Makefile's invocation pattern).
SHARE_ROOT = Path("output_to_share")
AGG_ROOT = Path("output_to_agg")
AGG_FIGURES_DIR = AGG_ROOT / "figures"


# ── Site discovery ────────────────────────────────────────────────────────
_NON_SITE_DIRS = frozenset({
    "figures",  # legacy pre-refactor top-level figures dir
    "qc",       # cross-site QC PNGs/CSVs — not a site, no models/ subdir
})


def list_sites() -> list[str]:
    """Return sorted list of real site directories under output_to_share/.

    Skips:
      - leading-underscore dirs (reserved — e.g., future `_meta/` for cross-
        site helpers; also guards against `__pycache__` or similar).
      - known non-site directories (`figures/`, `qc/`) listed in `_NON_SITE_DIRS`.

    New sites plug in by dropping `output_to_share/<newsite>/` and rerunning
    `make agg` — no code edit required.
    """
    if not SHARE_ROOT.exists():
        return []
    out = []
    for child in SHARE_ROOT.iterdir():
        if not child.is_dir():
            continue
        name = child.name
        if name.startswith("_"):
            continue
        if name in _NON_SITE_DIRS:
            continue
        out.append(name)
    return sorted(out)


# ── Anonymization ─────────────────────────────────────────────────────────
def _anonymize_enabled() -> bool:
    """True when ANONYMIZE_SITES env var is set to a truthy value."""
    val = os.environ.get("ANONYMIZE_SITES", "0").strip().lower()
    return val in ("1", "true", "yes", "on")


def site_label(site: str) -> str:
    """Map a real site name to its display label.

    Behavior:
      - Default (ANONYMIZE_SITES unset or '0'): returns `site` unchanged
        (e.g., 'mimic', 'ucmc').
      - ANONYMIZE_SITES=1: returns 'Site A' / 'Site B' / ... based on
        alphabetical ordering across discovered sites. Mapping is
        deterministic within a run — all 4 artifacts use the same labels.

    The alphabetical mapping is recomputed on each call (cheap) so that
    dropping a new site dir mid-run wouldn't desync labels within one script.
    In practice `make agg` calls each script fresh, so mappings stay stable.
    """
    if not _anonymize_enabled():
        return site
    sites = list_sites()
    if site not in sites:
        # Unknown site name — fall back to showing the raw name rather than
        # silently mislabeling. Shouldn't happen in normal flow.
        return site
    idx = sites.index(site)
    # 26 sites is plenty; if a consortium ever exceeds that we'll switch to
    # "Site AA" scheme — not a concern for this project.
    return f"Site {chr(ord('A') + idx)}"


# ── Loaders ───────────────────────────────────────────────────────────────
def load_site_analytical(site: str) -> pd.DataFrame:
    """Load the per-site modeling dataset parquet.

    Reads `output/{site}/modeling_dataset.parquet` — same path convention
    the per-site descriptive scripts use via code/descriptive/_shared.py.
    Phase-2 aggregation *reads* these PHI-tier parquets but only emits
    aggregate outputs under output_to_agg/ (never row-level).
    """
    return pd.read_parquet(f"output/{site}/modeling_dataset.parquet")


def load_site_table1(site: str) -> pd.DataFrame:
    """Load a per-site Table 1 CSV (string-formatted cells).

    Returned frame preserves the output_to_share/{site}/models/table1.csv
    layout: columns [Unnamed: 0, Unnamed: 1, Missing, Overall]. Values like
    "64.2 (16.2)" or "5309 (43.0)" are strings — pool_table1.py parses them.
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "table1.csv")


def load_site_cohort_stats(site: str) -> pd.DataFrame:
    """Load per-site cohort stats CSV (one row with site, n_hosp, n_patients)."""
    return pd.read_csv(SHARE_ROOT / site / "models" / "cohort_stats.csv")


def load_site_forest_data(site: str) -> pd.DataFrame:
    """Load per-site forest plot data (10→90 percentile ORs).

    Returns the long-format CSV produced by the forest-plot cell in
    `code/08_models.py`. Columns: outcome, model_type, spec, predictor,
    OR, OR_lo, OR_hi. ORs are already rescaled to a 10th→90th percentile
    shift in the predictor's production-cohort distribution, so cross-site
    overlay is just a stack-and-plot job (no refit needed).
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "forest_data.csv")


def load_site_marginal_effects(site: str) -> pd.DataFrame:
    """Load per-site marginal-effects prediction grid (RCS curves).

    Returns the long-format CSV produced by the marginal-effects cell in
    `code/08_models.py`. Columns: outcome, model_type, spec, focal,
    xlabel, panel_row, panel_col, x_actual, x_scaled, prob, ci_lo, ci_hi.
    50 grid points per (outcome × model_type × spec × focal) panel; the
    grid runs over each predictor's per-site 2.5–97.5 percentile range.
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "marginal_effects_grid.csv")


def load_site_exposure(site: str) -> pd.DataFrame:
    """Load PHI-tier exposure dataset for one site (full hospital-stay coverage).

    Site-parameterized counterpart to `code/descriptive/_shared.py:load_exposure`,
    which is locked to a module-level SITE_NAME captured at import time.
    Carries the four flag columns the cross-site stacked-bar figure needs
    (`_is_first_day`, `_is_last_day`, `_single_shift_day`, `_rel_day`),
    plus the per-drug day/night/diff dose columns produced by
    `code/05_modeling_dataset.py`.
    """
    return pd.read_parquet(f"output/{site}/exposure_dataset.parquet")


# ── Save helpers ──────────────────────────────────────────────────────────
def ensure_agg_dirs() -> None:
    AGG_ROOT.mkdir(parents=True, exist_ok=True)
    AGG_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_agg_csv(df: pd.DataFrame, name: str, index: bool = False) -> str:
    """Save `df` as output_to_agg/{name}.csv. Returns the path."""
    ensure_agg_dirs()
    path = AGG_ROOT / f"{name}.csv"
    df.to_csv(path, index=index)
    print(f"Saved {path}")
    return str(path)


def save_agg_fig(fig, name: str) -> str:
    """Save `fig` as output_to_agg/figures/{name}.png. Returns the path."""
    ensure_agg_dirs()
    path = AGG_FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved {path}")
    return str(path)


# ── Re-exports (stable public surface for agg scripts) ────────────────────
__all__ = [
    # paths + discovery + anonymization
    "SHARE_ROOT",
    "AGG_ROOT",
    "AGG_FIGURES_DIR",
    "SITE_PALETTE",
    "list_sites",
    "site_label",
    # loaders
    "load_site_analytical",
    "load_site_table1",
    "load_site_cohort_stats",
    "load_site_forest_data",
    "load_site_marginal_effects",
    "load_site_exposure",
    # save helpers
    "ensure_agg_dirs",
    "save_agg_csv",
    "save_agg_fig",
    # re-exports from code/descriptive/_shared
    "DRUGS",
    "DIFF_COLS",
    "DAY_COLS",
    "NIGHT_COLS",
    "THRESHOLDS",
    "DRUG_LABELS",
    "DRUG_UNITS",
    "DRUG_COLORS",
    "DIFF_BIN_COLORS",
    "DOSE_PATTERN_COLORS",
    "DOSE_PATTERN_LABELS",
    "COUNT_BAR_STACK_ORDER",
    "cap_day",
    "prepare_diffs",
    "apply_style",
    "categorize_diff_6way",
]
