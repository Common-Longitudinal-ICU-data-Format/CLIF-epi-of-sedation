"""Shared helpers for the single-patient trajectory QC dashboard.

Centralizes per-patient data access so the Dash app stays thin:

- `list_sites()` discovers which sites have pipeline outputs available.
- `build_site_config(site)` returns the per-site JSON config (used by clifpy).
- `get_wide_df(site, hosp_id)` returns the wide dataset for ONE patient, loading
  only that patient's rows from raw CLIF tables. LRU-cached so flipping between
  recent IDs is instant.
- `get_enrichment(site, hosp_id)` reads per-patient slices of the pipeline
  artifacts (analytical dataset, sed dose by hour, SBT outcomes, covariates,
  IMV streaks) without loading the full parquet into memory.
- `extract_events(...)` builds a dataframe of clinical event markers
  (intubation / SBT / extubation / tracheostomy / death) with absolute
  timestamps for drawing as vertical lines on the timeline.

All file paths resolve through `output/{site}/...` and `output_to_share/{site}/...`
per the project's multi-site layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

# Reuse the drug color palette from the descriptive figures so the dashboard
# reads consistently with the static figures in output_to_share/<site>/figures.
# Imported via importlib to avoid a name collision — both packages have a
# `_shared.py` and neither is a real package.
import importlib.util as _ilu

_DESC_SHARED = Path(__file__).parent.parent / "descriptive" / "_shared.py"
_spec = _ilu.spec_from_file_location("desc_shared", _DESC_SHARED)
_mod = _ilu.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]
DRUG_COLORS = _mod.DRUG_COLORS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
CONFIG_DIR = PROJECT_ROOT / "config"


# ── Style constants ─────────────────────────────────────────────────────

# Column keys match the 2026-04-24 unit-suffix convention in
# output/{site}/sed_dose_by_hr.parquet: after SUM-per-hour, values are total
# mg (or mcg) delivered in that hour, so the suffix reads `_mg_hr_cont` /
# `_mcg_hr_cont`. Update 02_exposure.py's rename maps if renaming further.
SEDATIVE_COLORS = {
    "propofol_mg_hr_cont": DRUG_COLORS["prop"],
    "fentanyl_mcg_hr_cont": DRUG_COLORS["fenteq"],
    "midazolam_mg_hr_cont": DRUG_COLORS["midazeq"],
    "hydromorphone_mg_hr_cont": "#bcbddc",
    "lorazepam_mg_hr_cont": "#9e9ac8",
}

PRESSOR_COLORS = {
    "norepinephrine": "#d62728",
    "epinephrine": "#ff9896",
    "vasopressin": "#8c564b",
    "nee": "#111111",
}

ASSESSMENT_COLORS = {
    "rass": "#1f77b4",
    "gcs_total": "#2ca02c",
}

RESP_COLORS = {
    "fio2_set": "#e377c2",
    "peep_set": "#17becf",
}

VITAL_COLORS = {
    "heart_rate": "#d62728",
    "map": "#9467bd",
    "spo2": "#17becf",
}

# Device-category ribbon colors. The device ribbon along the bottom edge of
# the resp panel encodes BOTH device and mode at once: IMV blocks are
# sub-colored by mode_category (deep→pale blue palette below); non-IMV
# devices get their single device color since mode doesn't apply.
DEVICE_CATEGORY_COLORS = {
    "imv": "#08519c",  # base blue — actual segments shaded by mode within IMV
    "nippv": "#ff7f0e",
    "cpap": "#ff7f0e",
    "bipap": "#ff7f0e",
    "high_flow_nc": "#17becf",
    "nasal_cannula": "#9edae5",
    "room_air": "#ffffff",
    "trach_mask": "#bcbd22",
    "t_piece": "#bcbd22",
    "other": "#d9d9d9",
}

# Sub-coloring for IMV blocks by mode_category. Sequential blue palette so
# they read as variants of "IMV" but the mode is also distinguishable.
# Anything not in this dict falls back to the generic IMV color.
MODE_IN_IMV_COLORS = {
    "Assist Control-Volume Control": "#08306b",  # darkest — most-supported
    "Pressure Control":              "#2171b5",
    "SIMV":                          "#4292c6",
    "Pressure Support/CPAP":         "#9ecae1",  # lightest — most weaning
    "Other":                         "#deebf7",
}

# Kept for backward compat with any downstream import; full-panel mode-band
# rendering is no longer used (mode now lives in the device ribbon).
MODE_CATEGORY_BAND_COLORS = {
    k: f"rgba(0,0,0,0)" for k in MODE_IN_IMV_COLORS
}

EVENT_COLORS = {
    "intubation": "#b2182b",
    "extubation": "#2166ac",
    # Primary SBT (spec-literal) and its 4 sensitivity siblings. Variants are
    # rendered as distinct shades of the same purple family so they read as
    # related-but-different markers; see docs/intub_extub_specs.md "Sensitivity
    # siblings" for what each variant operationalizes. The contrast between
    # `sbt` (primary) and `sbt_prefix` (pre-fix every-row reproduction) is
    # the most informative side-by-side: anywhere `sbt_prefix` fires but
    # `sbt` does not is a row that the spec-literal correctly excluded
    # (e.g., LAG-mode-not-controlled, or row-1 streak start).
    "sbt": "#7a5195",            # primary — spec-literal
    "sbt_anyprior": "#a87bd7",   # drops controlled-mode whitelist
    "sbt_imv6h":    "#5d3d7a",   # ≥6h continuous IMV before flip
    "sbt_prefix":   "#c4a8e0",   # pre-fix every-row baseline reproduction
    "sbt_2min":     "#9b6cb8",   # 2-min sustained-duration variant
    "tracheostomy": "#d97706",
    "withdrawal": "#525252",
    # Discharge event color-coded by outcome bucket
    "discharge_home": "#2ca02c",
    "discharge_snf": "#7f7f7f",
    "discharge_other": "#7f7f7f",
    "discharge_hospice": "#8c564b",
    "discharge_death": "#d62728",
}

# Day/night shift boundaries (7 AM / 7 PM) — matches the upstream
# `_shift` assignment (see code/01_cohort.py:380 and _utils.add_day_shift_id).
DAY_START_HOUR = 7
NIGHT_START_HOUR = 19

# Colors for the three cohort-region shading layers (§3–§4 of the QC plan).
COHORT_BAND_COLOR = "#cce8e3"        # pale teal — analytical cohort window
EXCLUDED_ZONE_COLOR = "#ffe4b5"      # moccasin — day-0 and last-day excluded slices
NIGHT_SHIFT_COLOR = "#e8e8e8"        # light grey — 7 PM → 7 AM windows

# How many minutes around a hovered timestamp to include in the linked table.
# Widened from 30 to 240 (±4h) so clinicians can see a meaningful window of
# surrounding rows without manually re-hovering.
HOVER_WINDOW_MIN = 240


# ── Site discovery / config ─────────────────────────────────────────────

def list_sites() -> list[str]:
    """Return lowercase site names that have an analytical_dataset.parquet."""
    if not OUTPUT_DIR.exists():
        return []
    sites: list[str] = []
    for child in sorted(OUTPUT_DIR.iterdir()):
        if not child.is_dir():
            continue
        if child.name.startswith("_"):
            continue
        if (child / "analytical_dataset.parquet").exists():
            sites.append(child.name)
    return sites


def _site_config_path(site: str) -> Path:
    """Map a site name to its config file. Falls back to config.json."""
    candidate = CONFIG_DIR / f"{site}_config.json"
    if candidate.exists():
        return candidate
    return CONFIG_DIR / "config.json"


def load_site_config(site: str) -> dict[str, Any]:
    path = _site_config_path(site)
    with path.open() as f:
        return json.load(f)


# ── ClifOrchestrator cache (per site) ──────────────────────────────────

@lru_cache(maxsize=4)
def _get_orchestrator(site: str):
    """Return a ClifOrchestrator bound to the site's config.

    Heavy: constructing the orchestrator pays a small upfront cost, but
    `load_table` is only called later with a per-patient filter so no full
    cohort is ever materialized. Cached per site so flipping between MIMIC
    and UCMC within one session doesn't re-instantiate.
    """
    from clifpy import ClifOrchestrator  # lazy import keeps app startup snappy

    cfg_path = _site_config_path(site)
    return ClifOrchestrator(config_path=str(cfg_path))


_WIDE_TABLES: dict[str, list[str] | None] = {
    "vitals": ["heart_rate", "sbp", "map", "spo2", "temp_c", "respiratory_rate"],
    "medication_admin_continuous": [
        "propofol", "fentanyl", "midazolam", "lorazepam", "hydromorphone",
        "norepinephrine", "epinephrine", "vasopressin",
    ],
    "respiratory_support": [
        "device_category", "mode_category", "fio2_set", "peep_set",
        "pressure_support_set",
    ],
    "patient_assessments": ["gcs_total", "rass"],
}


@lru_cache(maxsize=32)
def get_wide_df(site: str, hosp_id: str) -> pd.DataFrame:
    """Build the wide timeline for ONE hospitalization_id.

    Loads only that patient's rows from each raw CLIF table. LRU-cached on
    (site, id) so repeated clicks on the same ID are free.
    """
    from clifpy.utils.wide_dataset import create_wide_dataset

    co = _get_orchestrator(site)
    # Scope the raw loads to this patient only. ClifOrchestrator's `load_table`
    # accepts a filter dict, so we bypass any prior full-cohort load.
    hosp_filter = {"hospitalization_id": [hosp_id]}
    for t in [
        "hospitalization", "adt",
        "vitals", "medication_admin_continuous",
        "respiratory_support", "patient_assessments",
    ]:
        co.load_table(t, filters=hosp_filter)
    # patient table has no hospitalization_id — load unfiltered but it's small.
    co.load_table("patient")

    wide_df = create_wide_dataset(
        clif_instance=co,
        category_filters=_WIDE_TABLES,
        hospitalization_ids=[hosp_id],
        output_format="dataframe",
        show_progress=False,
    )
    if wide_df is None or wide_df.empty:
        return pd.DataFrame()

    # Normalize the time column name so callers can rely on `event_time`.
    time_cols = [c for c in wide_df.columns if "time" in c.lower() or "dttm" in c.lower()]
    if time_cols:
        wide_df = wide_df.rename(columns={time_cols[0]: "event_time"})
        wide_df = wide_df.sort_values("event_time").reset_index(drop=True)
    return wide_df


# ── Enrichment artifacts (per-patient slices) ──────────────────────────

@dataclass
class PatientEnrichment:
    analytical: pd.DataFrame
    sed_hourly: pd.DataFrame
    covariates: pd.DataFrame
    sbt_daily: pd.DataFrame
    sbt_rows: pd.DataFrame                 # Per-row ONSET events (filtered) for vline drawing
    sbt_audit: pd.DataFrame                # Per-row FULL sbt_outcomes for linked-table merge
    imv_streaks: pd.DataFrame              # First qualifying streak only (cohort)
    all_imv_streaks: pd.DataFrame          # Every streak, incl. < 24h and post-cohort
    first_icu_dttm: pd.Timestamp | None
    cohort_start: pd.Timestamp | None      # First qualifying streak's _start_dttm
    cohort_end: pd.Timestamp | None        # First qualifying streak's _end_dttm
    discharge: pd.Series | None            # 1-row Series: discharge_dttm + _category
    intm: pd.DataFrame                     # Long-form medication_admin_intermittent


def _read_filtered_parquet(path: Path, hosp_id: str) -> pd.DataFrame:
    """Read a parquet filtered to one hospitalization_id (pyarrow pushdown)."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path, filters=[("hospitalization_id", "=", hosp_id)])


def _read_sbt_onset_rows(site_dir: Path, hosp_id: str) -> pd.DataFrame:
    """Load row-level SBT onset events for one hospitalization.

    Reads the per-row `sbt_outcomes.parquet` (NOT the daily aggregate
    `sbt_outcomes_daily.parquet`) so the dashboard can place SBT vlines at
    the *actual* `event_dttm` of each onset row. The daily aggregate has
    no timestamp column and forces an admit-time-anchored guess that can
    be off by up to 24 h.

    Filters at parquet-read time to (a) the patient's rows and (b) rows
    where any of the 5 flag columns = 1.

    NOTE on flag semantics: the four variants with LAG checks (`sbt_done`,
    `sbt_done_anyprior`, `sbt_done_imv6h`, `sbt_done_2min`) are 1 on a
    single onset row per qualifying block. `sbt_done_prefix` has NO LAG
    check by design, so it's 1 on *every* row of a qualifying ≥30-min
    PS-with-low block — collapsing those to one event per block (via
    `_block_id` in `extract_events`) is the dashboard's responsibility,
    NOT this loader's.

    Includes supporting columns alongside each flag so the auditor can
    verify *why* each variant fired (e.g., `_prior_mode_controlled = 1`
    for the primary `sbt_done`).
    """
    path = site_dir / "sbt_outcomes.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "hospitalization_id", "event_dttm", "_block_id",
        "sbt_done", "sbt_done_anyprior", "sbt_done_imv6h",
        "sbt_done_prefix", "sbt_done_2min",
        # Supporting columns for "why did this fire" audit.
        "_block_duration_mins", "_lag_sbt_state",
        "_prior_mode_controlled", "_lag_imv_streak_minutes",
        "mode_category", "device_category",
    ]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("hospitalization_id", "=", hosp_id)],
    )
    if df.empty:
        return df
    # NOTE: `event_dttm` in `sbt_outcomes.parquet` is tagged `America/Chicago`
    # (DuckDB stamps `TIMESTAMPTZ` columns with the session tz at write time,
    # and 03_outcomes.py runs in the user's local tz). Re-tag to the config's
    # `US/Eastern` so the wall-clock matches `wide_df.event_time` in the
    # dashboard's linked table. Same UTC instant, just the display label.
    cfg = load_site_config(site_dir.name)
    target_tz = cfg.get("timezone")
    if target_tz and pd.api.types.is_datetime64_any_dtype(df["event_dttm"]):
        if df["event_dttm"].dt.tz is not None:
            df["event_dttm"] = df["event_dttm"].dt.tz_convert(target_tz)
    flag_cols = ["sbt_done", "sbt_done_anyprior", "sbt_done_imv6h",
                 "sbt_done_prefix", "sbt_done_2min"]
    onset_mask = (df[flag_cols].fillna(0).astype(int) == 1).any(axis=1)
    return df.loc[onset_mask].sort_values("event_dttm").reset_index(drop=True)


def _read_sbt_audit_rows(site_dir: Path, hosp_id: str) -> pd.DataFrame:
    """Load the FULL per-row sbt_outcomes data for one hospitalization.

    Sibling of `_read_sbt_onset_rows` but returns *every* row for the patient
    (no onset-mask filter). Used by the dashboard's linked table to surface
    the row-level SBT computation context — `_block_id`, `_block_duration_mins`,
    `_lag_sbt_state`, `_prior_mode_controlled`, `_imv_streak_minutes`,
    `_lag_imv_streak_minutes`, `pressure_support_set`, the post-processed
    `mode_category` / `device_category`, and the 5 flag columns — alongside
    the wide_df rows.

    These are the columns that *determine* whether each variant flag fires
    (or doesn't) per `code/03_outcomes.py`. Without them, the auditor sees
    only the onset moment via the vline but has no view into WHY each
    variant agreed or disagreed at that moment.

    `event_dttm` is tz-converted to the site's configured timezone so
    timestamps align with `wide_df.event_time` for the merge in
    `_linked_table_source`.
    """
    path = site_dir / "sbt_outcomes.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "hospitalization_id", "event_dttm",
        # Block / LAG-check context for SBT flag computation
        "_block_id", "_block_duration_mins",
        "_lag_sbt_state", "_prior_mode_controlled",
        "_imv_streak_minutes", "_lag_imv_streak_minutes",
        # All resp settings as the BACKFILLED versions (resp_processed_bf
        # waterfall) — these are what `code/03_outcomes.py` evaluates
        # against. Compare with the raw versions in wide_df to spot
        # backfill differences.
        "fio2_set", "peep_set", "pressure_support_set",
        "mode_category", "mode_name",
        "device_category", "device_name",
        # All 5 SBT variant flags
        "sbt_done", "sbt_done_anyprior", "sbt_done_imv6h",
        "sbt_done_prefix", "sbt_done_2min",
        # Extubation outcome flags (row-level, drives the extub GEE/logit
        # outcomes in 08_models.py)
        "_intub", "_extub_1st", "_fail_extub", "_success_extub",
        "_trach_1st",
    ]
    df = pd.read_parquet(
        path,
        columns=cols,
        filters=[("hospitalization_id", "=", hosp_id)],
    )
    if df.empty:
        return df
    cfg = load_site_config(site_dir.name)
    target_tz = cfg.get("timezone")
    if target_tz and pd.api.types.is_datetime64_any_dtype(df["event_dttm"]):
        if df["event_dttm"].dt.tz is not None:
            df["event_dttm"] = df["event_dttm"].dt.tz_convert(target_tz)
    return df.sort_values("event_dttm").reset_index(drop=True)


@lru_cache(maxsize=32)
def get_enrichment(site: str, hosp_id: str) -> PatientEnrichment:
    site_dir = OUTPUT_DIR / site
    analytical = _read_filtered_parquet(site_dir / "analytical_dataset.parquet", hosp_id)
    sed_hourly = _read_filtered_parquet(site_dir / "sed_dose_by_hr.parquet", hosp_id)
    covariates = _read_filtered_parquet(site_dir / "covariates_daily.parquet", hosp_id)
    sbt_daily = _read_filtered_parquet(site_dir / "sbt_outcomes_daily.parquet", hosp_id)
    sbt_rows = _read_sbt_onset_rows(site_dir, hosp_id)
    sbt_audit = _read_sbt_audit_rows(site_dir, hosp_id)
    imv_streaks = _read_filtered_parquet(site_dir / "cohort_imv_streaks.parquet", hosp_id)
    all_streaks = get_all_imv_streaks(site, hosp_id)

    first_icu = None
    if not analytical.empty and "_first_icu_dttm" in analytical.columns:
        first_icu = analytical["_first_icu_dttm"].iloc[0]
        if pd.notna(first_icu):
            first_icu = pd.Timestamp(first_icu)
        else:
            first_icu = None

    cohort_start = cohort_end = None
    if not imv_streaks.empty:
        s = imv_streaks.iloc[0]
        if pd.notna(s.get("_start_dttm")):
            cohort_start = pd.Timestamp(s["_start_dttm"])
        if pd.notna(s.get("_end_dttm")):
            cohort_end = pd.Timestamp(s["_end_dttm"])

    discharge = get_discharge_info(site, hosp_id)
    intm = get_intm_med(site, hosp_id)

    return PatientEnrichment(
        analytical=analytical,
        sed_hourly=sed_hourly,
        covariates=covariates,
        sbt_daily=sbt_daily,
        sbt_rows=sbt_rows,
        sbt_audit=sbt_audit,
        imv_streaks=imv_streaks,
        all_imv_streaks=all_streaks,
        first_icu_dttm=first_icu,
        cohort_start=cohort_start,
        cohort_end=cohort_end,
        discharge=discharge,
        intm=intm,
    )


# ── Intermittent-admin loader (DIY, bypasses create_wide_dataset) ─────

@lru_cache(maxsize=32)
def get_intm_med(site: str, hosp_id: str) -> pd.DataFrame:
    """Long-format `medication_admin_intermittent` rows for one hosp_id.

    Used for: (a) panel markers (one diamond/X per bolus event), and
    (b) the linked audit table (raw bolus dose visible alongside cont
    rates). Bypasses clifpy's `create_wide_dataset` since we want the
    raw event stream, not a category-pivoted wide frame.
    """
    try:
        co = _get_orchestrator(site)
        co.load_table("medication_admin_intermittent",
                      filters={"hospitalization_id": [hosp_id]})
        df = co.medication_admin_intermittent.df
    except Exception:  # noqa: BLE001 — table missing → return empty
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    keep_cats = ["propofol", "fentanyl", "midazolam", "lorazepam", "hydromorphone",
                 "norepinephrine", "epinephrine", "vasopressin"]
    sub = df[df["med_category"].isin(keep_cats)].copy()
    if sub.empty:
        return pd.DataFrame()
    cols = [c for c in ("admin_dttm", "med_category", "med_dose", "med_dose_unit")
            if c in sub.columns]
    return sub[cols].sort_values("admin_dttm").reset_index(drop=True)


# ── All-streaks (QC-only, on-the-fly) ──────────────────────────────────

@lru_cache(maxsize=32)
def get_all_imv_streaks(site: str, hosp_id: str) -> pd.DataFrame:
    """Compute the full IMV-episode list (not just the cohort-qualifying one).

    Lifts the gap-island SQL from code/01_cohort.py:190-265, dropping the
    `_streak_id == 1` filter so re-intubations post-cohort are retained.
    Runs on `output/{site}/resp_processed_bf.parquet` filtered to a single
    hospitalization — small enough that per-patient on-the-fly is faster
    than materializing a separate parquet (and it stays in-sync with
    upstream edits automatically).

    Returns a DataFrame with one row per contiguous IMV episode:
    `_streak_id`, `_start_dttm`, `_end_dttm`, `_duration_hrs`, `_at_least_24h`.
    Empty if no IMV rows or the upstream parquet is missing.
    """
    import duckdb  # lazy import — avoids paying cost at module load

    path = OUTPUT_DIR / site / "resp_processed_bf.parquet"
    if not path.exists():
        return pd.DataFrame()
    resp_p = pd.read_parquet(path, filters=[("hospitalization_id", "=", hosp_id)])
    if resp_p.empty or "recorded_dttm" not in resp_p.columns:
        return pd.DataFrame()

    streaks = duckdb.query("""
        -- Gap-island: detect transitions in/out of IMV → assign _streak_id →
        -- aggregate to per-streak start/end, mirroring 01_cohort.py:190-265.
        WITH t1 AS (
            FROM resp_p
            SELECT hospitalization_id
                , event_dttm: recorded_dttm
                , _on_imv: CASE WHEN device_category = 'imv' THEN 1 ELSE 0 END
                , _chg_imv: CASE
                    WHEN (_on_imv = 0 AND LAG(_on_imv) OVER w = 1)
                    OR (_on_imv = 1 AND _on_imv IS DISTINCT FROM LAG(_on_imv) OVER w)
                    THEN 1 ELSE 0 END
            WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        ), t2 AS (
            FROM t1
            SELECT *
                , _streak_id: SUM(_chg_imv) OVER w
            WINDOW w AS (PARTITION BY hospitalization_id ORDER BY event_dttm)
        ), agg AS (
            FROM t2
            SELECT hospitalization_id
                , _streak_id
                , _start_dttm: MIN(event_dttm)
                , _last_observed_dttm: MAX(event_dttm)
                , _on_imv: MAX(_on_imv)
            GROUP BY hospitalization_id, _streak_id
        )
        FROM agg
        SELECT _streak_id
            , _start_dttm
            , _end_dttm: COALESCE(LEAD(_start_dttm) OVER w, _last_observed_dttm)
            , _duration_hrs: date_diff('minute', _start_dttm, COALESCE(LEAD(_start_dttm) OVER w, _last_observed_dttm)) / 60.0
            , _at_least_24h: CASE WHEN date_diff('minute', _start_dttm, COALESCE(LEAD(_start_dttm) OVER w, _last_observed_dttm)) / 60.0 >= 24 THEN 1 ELSE 0 END
        WHERE _on_imv = 1
        WINDOW w AS (ORDER BY _streak_id)
        ORDER BY _streak_id
    """).df()
    return streaks


# ── Discharge info (for the discharge vline + linked summary) ──────────

@lru_cache(maxsize=32)
def get_discharge_info(site: str, hosp_id: str) -> pd.Series | None:
    """Return a 1-row Series with discharge_dttm + discharge_category.

    Mirrors the loader pattern in code/06_table1.py:78-84. Returns None if
    clifpy can't resolve the hospitalization or the fields are absent.
    """
    try:
        from clifpy import Hospitalization
        from clifpy.utils import apply_outlier_handling
    except Exception:  # noqa: BLE001 — clifpy missing → skip discharge overlay
        return None

    cfg_path = _site_config_path(site)
    hosp = Hospitalization.from_file(
        config_path=str(cfg_path),
        columns=["hospitalization_id", "discharge_dttm", "discharge_category"],
        filters={"hospitalization_id": [hosp_id]},
    )
    try:
        apply_outlier_handling(hosp, outlier_config_path="config/outlier_config.yaml")
    except Exception:  # noqa: BLE001 — config path unknown → tolerate
        pass
    df = hosp.df
    if df is None or df.empty:
        return None
    return df.iloc[0]


def categorize_discharge(category: object) -> str:
    """Map a mCIDE discharge_category value to an EVENT_COLORS key."""
    if category is None or (isinstance(category, float) and pd.isna(category)):
        return "discharge_other"
    c = str(category).lower()
    if any(k in c for k in ("expired", "death", "died", "deceased")):
        return "discharge_death"
    if "hospice" in c:
        return "discharge_hospice"
    if "home" in c:
        return "discharge_home"
    if any(k in c for k in ("snf", "rehab", "skilled", "nursing")):
        return "discharge_snf"
    return "discharge_other"


# ── Shift / cohort-zone geometry helpers ───────────────────────────────

def night_windows_in_range(
    start: pd.Timestamp | None, end: pd.Timestamp | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (night_start, night_end) pairs covering [start, end].

    Each night = 19:00 on day N through 07:00 on day N+1. Edge windows are
    clipped to the range so the shading never extends beyond the plot.
    """
    if start is None or end is None or pd.isna(start) or pd.isna(end) or end <= start:
        return []
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    # Begin one day before `start` to catch a night window that started
    # before the cohort but still intersects the range.
    day = start.normalize() - pd.Timedelta(days=1)
    while day <= end:
        n_start = day + pd.Timedelta(hours=NIGHT_START_HOUR)
        n_end = day + pd.Timedelta(days=1, hours=DAY_START_HOUR)
        clipped_start = max(n_start, start)
        clipped_end = min(n_end, end)
        if clipped_end > clipped_start:
            windows.append((clipped_start, clipped_end))
        day = day + pd.Timedelta(days=1)
    return windows


def cohort_excluded_zones(
    cohort_start: pd.Timestamp | None, cohort_end: pd.Timestamp | None,
) -> list[tuple[pd.Timestamp, pd.Timestamp, str]]:
    """Return [(start, end, label), ...] for day-0 and last-day zones.

    These are the patient-day types that analytical_dataset.parquet drops
    via its `_nth_day > 0 AND next_day_outcome IS NOT NULL` filter (see
    §4 of the plan / docs/uptitration_paradox_investigation.md §0). Making
    them visible is the whole point of the paradox-investigation viewer.
    """
    if cohort_start is None or cohort_end is None:
        return []
    # Day-0 zone: cohort_start → first 7 AM after cohort_start
    first_7am = cohort_start.normalize() + pd.Timedelta(hours=DAY_START_HOUR)
    if first_7am <= cohort_start:
        first_7am = first_7am + pd.Timedelta(days=1)

    zones: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
    if first_7am > cohort_start:
        zones.append((cohort_start, min(first_7am, cohort_end), "day 0"))

    # Last-day zone: last 7 AM on or before cohort_end → cohort_end
    last_7am = cohort_end.normalize() + pd.Timedelta(hours=DAY_START_HOUR)
    if last_7am >= cohort_end:
        last_7am = last_7am - pd.Timedelta(days=1)
    if last_7am > cohort_start and last_7am < cohort_end:
        zones.append((last_7am, cohort_end, "last day"))
    return zones


def day_labels_for_cohort(
    cohort_start: pd.Timestamp | None, cohort_end: pd.Timestamp | None,
) -> list[tuple[pd.Timestamp, int]]:
    """Return [(center_timestamp, nth_day), ...] for labeling each day in cohort.

    Day 0 is partial (cohort_start → first 7 AM). Days 1..N-1 are full
    12h-day + 12h-night windows (7 AM → 7 AM). Last day is partial again.
    Labels are placed at the time-midpoint of each day window so they
    don't collide with 7 AM boundary markers.
    """
    if cohort_start is None or cohort_end is None or cohort_end <= cohort_start:
        return []
    first_7am = cohort_start.normalize() + pd.Timedelta(hours=DAY_START_HOUR)
    if first_7am <= cohort_start:
        first_7am = first_7am + pd.Timedelta(days=1)

    labels: list[tuple[pd.Timestamp, int]] = []
    # Day 0 (partial)
    day0_end = min(first_7am, cohort_end)
    labels.append((cohort_start + (day0_end - cohort_start) / 2, 0))

    # Full days
    t = first_7am
    idx = 1
    while t + pd.Timedelta(days=1) <= cohort_end:
        labels.append((t + pd.Timedelta(hours=12), idx))
        idx += 1
        t = t + pd.Timedelta(days=1)

    # Last partial day (if there's any slice after the last full-day 7 AM)
    if t < cohort_end:
        labels.append((t + (cohort_end - t) / 2, idx))
    return labels


# ── Event extraction ───────────────────────────────────────────────────

def extract_events(enr: PatientEnrichment) -> pd.DataFrame:
    """Return a DataFrame with columns [time, kind, label] for every clinical event.

    Intubation/extubation come from `all_imv_streaks` (every IMV episode,
    incl. post-cohort re-intubations — the cohort uses only the first
    qualifying streak, but QC review wants the whole course).
    SBT, tracheostomy, withdrawal, and death-after-extub come from the daily
    SBT rollup and are re-anchored to wall-clock by adding `_nth_day * 24h`
    to `_first_icu_dttm`. Discharge event comes from the hospitalization
    table with color-coded bucketing.
    """
    rows: list[dict[str, Any]] = []

    # Intubation / extubation — every streak, not just the cohort-qualifying one.
    # Label downgrades to `re-intub #N` / `re-extub #N` for streak_id > 1 so the
    # viewer can spot reintubation patterns at a glance.
    streaks = enr.all_imv_streaks if not enr.all_imv_streaks.empty else enr.imv_streaks
    for _, s in streaks.iterrows():
        sid = int(s.get("_streak_id", 0) or 0)
        prefix_in = "Intubation" if sid <= 1 else f"Re-intub #{sid}"
        prefix_out = "Extubation" if sid <= 1 else f"Re-extub #{sid}"
        if pd.notna(s.get("_start_dttm")):
            rows.append({
                "time": pd.Timestamp(s["_start_dttm"]),
                "kind": "intubation",
                "label": f"{prefix_in} (streak {sid})",
            })
        if pd.notna(s.get("_end_dttm")):
            rows.append({
                "time": pd.Timestamp(s["_end_dttm"]),
                "kind": "extubation",
                "label": f"{prefix_out} (streak {sid})",
            })

    # SBT events — read from the per-row `sbt_outcomes.parquet` (via
    # `enr.sbt_rows`) so the vline x-position is the *actual* onset
    # `event_dttm`, not an admit-time-anchored guess derived from the
    # daily aggregate's `_nth_day`. Pre-2026-04-26 the dashboard used the
    # daily aggregate, which has no timestamp column — that's why the
    # vlines never lined up with the resp panel's mode-ribbon transitions.
    #
    # Collapse to ONE event per `_block_id` per variant. The four LAG-
    # checked variants (sbt_done, sbt_done_anyprior, sbt_done_imv6h,
    # sbt_done_2min) are already 1 on a single onset row per block, so
    # the groupby is a no-op for them. `sbt_done_prefix` has no LAG
    # check by design — it's 1 on every row of a qualifying ≥30-min
    # PS-with-low block — so the groupby collapses each block to its
    # first row's event_dttm (the moment the block began).
    sbt_flag_to_kind = {
        "sbt_done":          ("sbt",          "SBT (primary)"),
        "sbt_done_anyprior": ("sbt_anyprior", "SBT (anyprior)"),
        "sbt_done_imv6h":    ("sbt_imv6h",    "SBT (imv6h)"),
        "sbt_done_prefix":   ("sbt_prefix",   "SBT (prefix)"),
        "sbt_done_2min":     ("sbt_2min",     "SBT (2min)"),
    }
    if not enr.sbt_rows.empty:
        sorted_rows = enr.sbt_rows.sort_values("event_dttm")
        for col, (kind, label) in sbt_flag_to_kind.items():
            flagged = sorted_rows[sorted_rows[col].fillna(0).astype(int) == 1]
            if flagged.empty:
                continue
            # First row per `_block_id` ⇒ block start. `keep="first"` after
            # the sort_values above gives the earliest event_dttm per block.
            onsets = flagged.drop_duplicates(subset=["_block_id"], keep="first")
            for _, r in onsets.iterrows():
                rows.append({
                    "time": pd.Timestamp(r["event_dttm"]),
                    "kind": kind,
                    "label": label,
                })

    # Trach / withdrawal — patient-level flags from the daily aggregate.
    # These don't have a precise row-level onset moment in the same way
    # SBT events do, so the daily anchor is appropriate. (The same admit-
    # time anchoring caveat applies but is far less visually distracting
    # since these fire at most once per patient.)
    if not enr.sbt_daily.empty and enr.first_icu_dttm is not None:
        daily_flag_map = {
            "_trach_1st":     ("tracheostomy", "Tracheostomy"),
            "_withdrawl_lst": ("withdrawal",   "Withdrawal of care"),
        }
        for _, d in enr.sbt_daily.iterrows():
            day_idx = d.get("_nth_day")
            if pd.isna(day_idx):
                continue
            t = enr.first_icu_dttm + pd.Timedelta(days=int(day_idx))
            for col, (kind, label) in daily_flag_map.items():
                if int(d.get(col, 0) or 0) == 1:
                    rows.append({"time": t, "kind": kind, "label": f"{label} (day {int(day_idx)})"})

    # Discharge event (one per patient), color-coded by outcome bucket.
    if enr.discharge is not None:
        dttm = enr.discharge.get("discharge_dttm")
        if dttm is not None and pd.notna(dttm):
            cat = enr.discharge.get("discharge_category")
            kind = categorize_discharge(cat)
            rows.append({
                "time": pd.Timestamp(dttm),
                "kind": kind,
                "label": f"Discharge ({cat})" if cat else "Discharge",
            })

    if not rows:
        return pd.DataFrame(columns=["time", "kind", "label"])
    out = pd.DataFrame(rows).sort_values("time").reset_index(drop=True)
    # Make sure timestamps are timezone-aware-ish for later comparison with
    # wide_df.event_time. If either side is naive, we compare as naive.
    return out


# ── Wide-dataset column helpers ────────────────────────────────────────

def match_columns(df: pd.DataFrame, needles: list[str]) -> list[str]:
    """Return wide_df columns whose lowercase name contains any of `needles`."""
    lower = {c: c.lower() for c in df.columns}
    hits = [c for c, lc in lower.items() if any(n in lc for n in needles)]
    return hits


def compute_nee(wide_df: pd.DataFrame) -> pd.Series:
    """Norepinephrine-equivalents (NEE) per clifpy-style conversion.

    Formula (matches the covariates pipeline):
        NEE = norepinephrine + epinephrine + vasopressin * 2.5

    Any missing component is treated as 0 for the sum. If no component
    columns are present at all, returns an empty series aligned to `wide_df`.
    """
    components = {
        "norepinephrine": 1.0,
        "epinephrine": 1.0,
        "vasopressin": 2.5,
    }
    nee = pd.Series(0.0, index=wide_df.index)
    found = False
    for col, mult in components.items():
        if col in wide_df.columns:
            nee = nee + wide_df[col].fillna(0.0) * mult
            found = True
    if not found:
        return pd.Series(dtype=float)
    # Mask hours where every component is NaN so the line breaks visibly.
    any_present = pd.Series(False, index=wide_df.index)
    for col in components:
        if col in wide_df.columns:
            any_present = any_present | wide_df[col].notna()
    nee[~any_present] = pd.NA
    return nee
