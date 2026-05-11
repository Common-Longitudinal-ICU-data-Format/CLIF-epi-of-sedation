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

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg_shared")

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
ON_DRUG_FLAGS = _descriptive_shared.ON_DRUG_FLAGS
DOSE_PATTERN_COLORS = _descriptive_shared.DOSE_PATTERN_COLORS
DOSE_PATTERN_LABELS = _descriptive_shared.DOSE_PATTERN_LABELS
categorize_diff_6way = _descriptive_shared.categorize_diff_6way

# CI helpers — federation-clean (consume summary stats, not raw vectors).
from _binom_helpers import (  # noqa: E402
    student_t_ci_from_summary,
    wilson_ci,
)

# Table 1 pooling primitives — federation-clean (consume summary stats only).
from _table1_helpers import (  # noqa: E402
    pooled_categorical_counts,
    pooled_mean_sd_from_summary,
    pooled_quantile_from_histograms,
)

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
    """Load the per-site modeling-cohort rows (filtered consolidated parquet).

    Phase 4 cutover (2026-05-08): reads
    `output/{site}/model_input_by_id_imvday.parquet` and applies the
    outcome-modeling filter inline. Byte-equivalent to the legacy
    `modeling_dataset.parquet` row set; verified at both sites.

      `_nth_day > 0 AND sbt_done_next_day IS NOT NULL
       AND success_extub_next_day IS NOT NULL`

    Phase-2 aggregation *reads* these PHI-tier parquets but only emits
    aggregate outputs under output_to_agg/ (never row-level).
    """
    df = pd.read_parquet(f"output/{site}/model_input_by_id_imvday.parquet")
    return df.loc[
        (df["_nth_day"] > 0)
        & df["sbt_done_next_day"].notna()
        & df["success_extub_next_day"].notna()
    ].reset_index(drop=True)


def load_site_descriptive_csv(site: str, name: str) -> pd.DataFrame:
    """Load a per-site aggregated CSV from `output_to_share/{site}/descriptive/`.

    This is the federation-clean entry point for cross-site scripts —
    consumes pre-aggregated counts/sums/means (no IDs, no row-level data)
    written by `code/descriptive/*.py`. Use this in preference to any
    `load_site_*` loader that reads PHI-tier parquets under `output/`.

    `name` is the CSV stem (no `.csv` extension), e.g.
    `dose_pattern_6group_count_by_icu_day`.
    """
    return pd.read_csv(SHARE_ROOT / site / "descriptive" / f"{name}.csv")


def load_site_table1_continuous(site: str) -> pd.DataFrame:
    """Load `output_to_share/{site}/models/table1_continuous.csv`.

    One row per continuous Table 1 variable carrying
    `(variable, n, n_missing, mean, sd, sum, sum_sq, median, q1, q3, min, max)`.
    `sum` and `sum_sq` are the lossless primitives for cross-site
    master-cohort mean/SD pooling via `pooled_mean_sd_from_summary`.
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "table1_continuous.csv")


def load_site_table1_categorical(site: str) -> pd.DataFrame:
    """Load `output_to_share/{site}/models/table1_categorical.csv`.

    Long-format rows per (variable, category) with
    `(n, n_missing, total_n, pct, denominator_unit)`. Per-stay and
    per-patient-day denominators coexist via the `denominator_unit`
    flag (e.g., `sbt_done_multiday_per_full24h_day` uses patient-days).
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "table1_categorical.csv")


def load_site_table1_histograms(site: str) -> pd.DataFrame:
    """Load `output_to_share/{site}/models/table1_histograms.csv`.

    Long-format bin counts per continuous variable on hardcoded shared
    bin edges (see `code/_table1_schema.py::BIN_EDGES`). Lets the
    cross-site pooler compute master-cohort median/Q1/Q3 by summing
    bin counts across sites and inverse-CDF interpolating.
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "table1_histograms.csv")


def load_site_cohort_stats(site: str) -> pd.DataFrame:
    """Load per-site cohort stats CSV (one row with site, n_hosp, n_patients)."""
    return pd.read_csv(SHARE_ROOT / site / "models" / "cohort_stats.csv")


def load_site_models_coeffs(site: str) -> pd.DataFrame:
    """Load per-site model coefficients (federated meta-analysis payload).

    Returns the long-format CSV produced by the forest-plot cell in
    `code/08_models.py`. One row per (outcome, model_type, spec, logical
    predictor) covering EVERY coefficient in EVERY fit. Schema:

      outcome, model_type, spec, spec_family ('linear'|'rcs'), predictor,
      row_type ('exposure'|'adjustment_continuous'|
                'adjustment_categorical'|'intercept'),
      log_or, se_log_or,                       # raw / headline
      unit_size, unit_label, x_ref_raw,        # per-unit definitions
      log_or_per_unit, se_per_unit,
      or_per_unit, or_per_unit_lo, or_per_unit_hi,
      x10_raw, x90_raw,                        # cohort percentiles
      log_or_p10_p90, se_p10_p90,
      or_p10_p90, or_p10_p90_lo, or_p10_p90_hi,
      n_obs, n_events, n_clusters

    Both per-unit (uniform across sites by design) and per-percentile
    (site-specific cohort distribution) presentations are emitted, so the
    cross-site agg layer can choose either without per-site re-runs.
    """
    return pd.read_csv(SHARE_ROOT / site / "models" / "models_coeffs.csv")


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

    Legacy loader kept during Phase 4 cutover. Reads the wide-semantic
    `exposure_dataset.parquet` whose `_is_last_day` flag fires on
    max-`_nth_day` per hospitalization (= partial extubation day OR last
    full day, whichever exists). New consumers should prefer
    `load_site_model_input` and filter on the registry's narrower
    `_is_last_partial_day` / `_is_last_full_day` columns plus
    `_is_full_24h_day` for explicit 12+12 hr coverage filtering.

    Carries `_is_first_day`, `_is_last_day`, `_single_shift_day` flags
    (legacy wide semantics) plus per-drug day/night/diff dose columns
    produced by `code/05_modeling_dataset.py`.
    """
    return pd.read_parquet(f"output/{site}/exposure_dataset.parquet")


def load_site_model_input(site: str) -> pd.DataFrame:
    """Load PHI-tier consolidated per-day modeling input for one site.

    Phase 4 superset of `exposure_dataset.parquet` keyed off the
    canonical registry. Use when a cross-site figure wants explicit
    full-vs-partial coverage filtering — `_is_full_24h_day = True`
    drops intubation- and extubation-day partial rows in one filter.
    Cross-site equivalent of `code/descriptive/_shared.py:load_model_input`.
    """
    return pd.read_parquet(f"output/{site}/model_input_by_id_imvday.parquet")


# ── Save helpers ──────────────────────────────────────────────────────────
def ensure_agg_dirs() -> None:
    AGG_ROOT.mkdir(parents=True, exist_ok=True)
    AGG_FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def save_agg_csv(df: pd.DataFrame, name: str, index: bool = False) -> str:
    """Save `df` as output_to_agg/{name}.csv. Returns the path."""
    ensure_agg_dirs()
    path = AGG_ROOT / f"{name}.csv"
    df.to_csv(path, index=index)
    logger.info(f"Saved {path}")
    return str(path)


def save_agg_fig(fig, name: str) -> str:
    """Save `fig` as output_to_agg/figures/{name}.png. Returns the path."""
    ensure_agg_dirs()
    path = AGG_FIGURES_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    logger.info(f"Saved {path}")
    return str(path)


# ── Salient three-tier headline ──────────────────────────────────────────
def add_salient_headline(
    fig,
    title: str,
    subtitle: str,
    *,
    units_line: str | None = None,
    title_color: str = "#1f1f1f",
    y_title: float = 1.075,
) -> None:
    """Three-tier headline mirroring `sed_dose_by_hr_of_day_cross_site`.

    Bold large title + muted provenance subtitle + (optional) italic
    units / scope line. Replaces plain `fig.suptitle(...)` so cross-site
    figures share one consistent salient header.

    The subtitle should literally cite the disambiguating columns from
    the underlying CSV (e.g. `outcome=...`, `spec=...`, `model_type=...`,
    `k=N sites`) so each figure self-documents and sibling figures (per_unit
    vs p10_p90; primary vs audit) are easy to tell apart at a glance.
    """
    fig.text(0.5, y_title, title,
             transform=fig.transFigure, ha="center", va="bottom",
             fontsize=20, fontweight="bold", color=title_color)
    fig.text(0.5, y_title - 0.025, subtitle,
             transform=fig.transFigure, ha="center", va="bottom",
             fontsize=11, color="0.30")
    if units_line:
        fig.text(0.5, y_title - 0.046, units_line,
                 transform=fig.transFigure, ha="center", va="bottom",
                 fontsize=10, color="0.40", style="italic")


# ── Visual marker for internal/QAQC figures ──────────────────────────────
def add_audit_badge(fig, *, x: float | None = None, y: float = 0.985,
                    ha: str = "right", byline: str | None = None) -> None:
    """Draw an 'AUDIT VIEW' pill in the top-{ha} corner of `fig`.

    Companion to the `*_audit_cross_site.png` filename suffix that
    distinguishes internal/QAQC figures from `_primary_*` versions —
    the suffix is invisible once a PNG is dropped into a slide deck,
    so the badge surfaces the distinction visually.

    `ha` defaults to "right" for back-compat. Pass `ha="left"` to relocate
    when a column-title or other right-edge artifact collides with the
    badge.
    """
    if x is None:
        x = 0.985 if ha == "right" else 0.015
    fig.text(x, y, "AUDIT VIEW",
             transform=fig.transFigure,
             ha=ha, va="top",
             fontsize=10, fontweight="bold", color="white",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#d97706", edgecolor="none"))
    if byline:
        fig.text(x, y - 0.025, byline,
                 transform=fig.transFigure,
                 ha=ha, va="top",
                 fontsize=8, color="#d97706", style="italic")


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
    "load_site_descriptive_csv",
    "load_site_table1_continuous",
    "load_site_table1_categorical",
    "load_site_table1_histograms",
    "load_site_cohort_stats",
    "load_site_models_coeffs",
    "load_site_marginal_effects",
    "load_site_exposure",
    "load_site_model_input",
    # save helpers
    "ensure_agg_dirs",
    "save_agg_csv",
    "save_agg_fig",
    "add_audit_badge",
    "add_salient_headline",
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
    "ON_DRUG_FLAGS",
    "cap_day",
    "prepare_diffs",
    "apply_style",
    "categorize_diff_6way",
    # CI helpers
    "wilson_ci",
    "student_t_ci_from_summary",
    # Table 1 pooling primitives
    "pooled_mean_sd_from_summary",
    "pooled_quantile_from_histograms",
    "pooled_categorical_counts",
]
