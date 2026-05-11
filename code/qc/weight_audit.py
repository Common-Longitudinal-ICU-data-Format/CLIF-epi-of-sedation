"""Weight QC DIAGNOSTIC — federated audit CSVs + figure.

B3 refactor (2026-05): the same three drop criteria (zero-weight / jump /
range) are now computed inline in `01_cohort.py` via
`_utils.compute_weight_qc_exclusions`. This script is no longer required
by `make run` and no longer produces the load-bearing drop list — it
remains as a DIAGNOSTIC tool for federated cross-site QA:

  - `weight_qc_summary.csv` — section-by-section metrics (raw availability,
    clamp impact, jump/range CDFs, per-criterion drop counts).
  - `weight_qc_exclusions.csv` — per-criterion drop counts + thresholds.
  - `weight_impact_comparison.csv` — section (h) impact analysis on
    prop_dif_mcg_kg_min across 4 weight-attachment strategies.
  - `weight_audit.png` — multi-panel figure.

The legacy `weight_qc_drop_list.parquet` and `weight_audit_examples.csv`
outputs are still written for backward compatibility but are NOT consumed
by the pipeline anymore.

Federated-safe: outputs document drop-rule SEVERITY at each site without
exposing hospitalization-level rows. Off-site reviewers compare summaries
cross-site without ever needing patient-level access.

Sections (each → one panel of weight_audit.png + rows in the summary CSV):
  (a) Raw availability & outlier-clamp impact
  (b) Stage A: clifpy per-admin ASOF success rate
  (c) Stage A pivot fall-through (replicated, since post-pivot parquet
      already strips failed-conversion columns)
  (d) Stage B: per-day ASOF + admission-fallback characterization
  (e) Cross-stage consistency (Stage A weight vs Stage B weight)
  (f) Within-patient stability + cutoff-defensibility CDFs
  (g) Drop-list characterization (no longer feeds the pipeline)
  (h) Impact analysis on prop_dif_mcg_kg_min (4 weight strategies)

Usage:
  make weight-diagnostic SITE=mimic
  # or directly:
  uv run python code/qc/weight_audit.py
  SITE=ucmc uv run python code/qc/weight_audit.py
  WEIGHT_QC_MAX_JUMP_KG=15 uv run python code/qc/weight_audit.py
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clifpy import setup_logging
from clifpy.utils.logging_config import get_logger

logger = get_logger("epi_sedation.weight_audit")

# ── Site / paths ─────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
OUTLIER_CONFIG = PROJECT_ROOT / "config" / "outlier_config.yaml"


def _load_site_name() -> str:
    if not CONFIG_PATH.exists():
        return "unknown"
    with CONFIG_PATH.open() as f:
        return json.load(f).get("site_name", "unknown").lower()


SITE = os.getenv("SITE", _load_site_name())
SITE_OUT = PROJECT_ROOT / "output" / SITE
SITE_QC = SITE_OUT / "qc"
SHARE_QC = PROJECT_ROOT / "output_to_share" / SITE / "qc"
# Path B++ refactor: the legacy `figures/` subdir was retired. Weight-audit
# is itself a QC artifact, so its PNG belongs in the existing `qc/` flat dir
# alongside the weight_qc_* CSVs.
SHARE_FIG = SHARE_QC

# Per-site dual log files at output/{site}/logs/clifpy_all.log +
# clifpy_errors.log (pyCLIF integration guide rule 1). weight_audit is
# its own entry-point subprocess so it must call setup_logging itself.
SITE_OUT.mkdir(parents=True, exist_ok=True)
setup_logging(output_directory=str(SITE_OUT))

# ── Drop-policy thresholds (env-var configurable) ────────────────────────
WEIGHT_QC_MAX_JUMP_KG = float(os.getenv("WEIGHT_QC_MAX_JUMP_KG", "20"))
WEIGHT_QC_MAX_JUMP_HOURS = float(os.getenv("WEIGHT_QC_MAX_JUMP_HOURS", "24"))
WEIGHT_QC_MAX_RANGE_KG = float(os.getenv("WEIGHT_QC_MAX_RANGE_KG", "30"))
WEIGHT_QC_RANGE_RULE_ON = os.getenv("WEIGHT_QC_RANGE_RULE_ON", "0") == "1"

# Outlier clamp from outlier_config.yaml — replicated here for audit-side use
# (the pre-clamp distribution is the *whole point* of section (a), so we can't
# reuse the clifpy clamp directly).
WEIGHT_CLAMP_LO = 30.0
WEIGHT_CLAMP_HI = 300.0

# Drugs whose dose conversion depends on weight (preferred unit unweighted)
SED_PREFERRED_UNITS = {
    "propofol": "mg/min",
    "midazolam": "mg/min",
    "fentanyl": "mcg/min",
    "hydromorphone": "mg/min",
    "lorazepam": "mg/min",
}


# ── Data structures ──────────────────────────────────────────────────────
@dataclass
class AuditSummary:
    """Append-only one-row-per-metric collector. Becomes weight_qc_summary.csv."""

    rows: list[dict] = field(default_factory=list)

    def add(self, section: str, metric: str, value, unit: str = "", note: str = ""):
        self.rows.append({
            "section": section, "metric": metric,
            "value": value, "unit": unit, "note": note,
        })

    def to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ── Loaders ──────────────────────────────────────────────────────────────
def _clifpy_commit() -> str:
    """Capture pyCLIF commit hash so federated meta-analysis can verify like-with-like."""
    try:
        import clifpy
        clifpy_dir = Path(clifpy.__file__).resolve().parent.parent
        out = subprocess.run(
            ["git", "-C", str(clifpy_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip()[:12] if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_cohort_hosp_ids() -> list[str]:
    """Cohort = hospitalizations in the modeling dataset."""
    # Phase 4 cutover (2026-05-08): read consolidated parquet + apply
    # outcome-modeling filter inline to recover the modeling-cohort hosp set.
    df = pd.read_parquet(
        SITE_OUT / "model_input_by_id_imvday.parquet",
        columns=["hospitalization_id", "_nth_day",
                 "sbt_done_next_day", "success_extub_next_day"],
    )
    df = df.loc[
        (df["_nth_day"] > 0)
        & df["sbt_done_next_day"].notna()
        & df["success_extub_next_day"].notna()
    ]
    return df["hospitalization_id"].drop_duplicates().tolist()


def load_raw_weight_events(hosp_ids: list[str]) -> pd.DataFrame:
    """Load raw weight_kg events WITHOUT outlier handling.

    The audit's section (a) needs the pre-clamp distribution to count what
    `apply_outlier_handling` silently drops. This bypasses clifpy's normal
    apply_outlier_handling step.
    """
    from clifpy import Vitals

    v = Vitals.from_file(
        config_path=str(CONFIG_PATH),
        columns=["hospitalization_id", "recorded_dttm", "vital_value", "vital_category"],
        filters={"vital_category": ["weight_kg"], "hospitalization_id": hosp_ids},
    )
    df = v.df
    df = df[df["vital_value"].notna() & (df["vital_value"] > 0)].copy()
    df = df.rename(columns={"vital_value": "weight_kg_raw"})
    return df[["hospitalization_id", "recorded_dttm", "weight_kg_raw"]].sort_values(
        ["hospitalization_id", "recorded_dttm"]
    ).reset_index(drop=True)


def load_med_admins(hosp_ids: list[str]) -> pd.DataFrame:
    """Load continuous sedative admin rows (matches 02_exposure.py)."""
    from clifpy import MedicationAdminContinuous

    m = MedicationAdminContinuous.from_file(
        config_path=str(CONFIG_PATH),
        columns=[
            "hospitalization_id", "admin_dttm", "med_category",
            "med_dose", "med_dose_unit", "mar_action_category",
        ],
        filters={
            "med_category": list(SED_PREFERRED_UNITS.keys()),
            "hospitalization_id": hosp_ids,
        },
    )
    return m.df


def load_weight_daily() -> pd.DataFrame:
    """Per-day weight built by 04_covariates.py (Stage B's weight)."""
    return pd.read_parquet(SITE_OUT / "weight_by_id_imvday.parquet")


def load_analytical() -> pd.DataFrame:
    """For impact analysis (section h) — needs prop_dif_mg_hr + weight cols."""
    # Phase 4 cutover (2026-05-08): consolidated parquet + outcome filter.
    df = pd.read_parquet(SITE_OUT / "model_input_by_id_imvday.parquet")
    return df.loc[
        (df["_nth_day"] > 0)
        & df["sbt_done_next_day"].notna()
        & df["success_extub_next_day"].notna()
    ].reset_index(drop=True)


def _to_utc(s: pd.Series) -> pd.Series:
    """Force a datetime Series into tz-aware UTC.

    DuckDB ASOF round-trips can yield tz-naive or session-tz-aware columns
    inconsistently; subtraction across the pair raises. Normalize to UTC
    before any timedelta arithmetic.
    """
    s = pd.to_datetime(s)
    if s.dt.tz is None:
        s = s.dt.tz_localize("UTC")
    else:
        s = s.dt.tz_convert("UTC")
    return s


# ── Section (a) ──────────────────────────────────────────────────────────
def section_a_raw_availability(
    raw_w: pd.DataFrame, hosp_ids: list[str], smry: AuditSummary
) -> dict:
    """Pre-clamp distribution + outlier-row count + round-number clustering."""
    n_total_rows = len(raw_w)
    out_of_clamp = ~raw_w["weight_kg_raw"].between(WEIGHT_CLAMP_LO, WEIGHT_CLAMP_HI)
    n_out_of_clamp = int(out_of_clamp.sum())
    top20_high = raw_w.loc[raw_w["weight_kg_raw"] > WEIGHT_CLAMP_HI, "weight_kg_raw"].nlargest(20).tolist()

    # Hospitalizations with ≥1 raw weight (post-clamp, since clamp is what the
    # downstream pipeline actually sees).
    clamped = raw_w[~out_of_clamp]
    n_hosp_with_weight = clamped["hospitalization_id"].nunique()
    n_hosp_zero_weight = len(set(hosp_ids)) - n_hosp_with_weight

    # Percentile distribution (clamped values).
    pcts = {p: float(np.nanpercentile(clamped["weight_kg_raw"], p)) for p in (1, 5, 25, 50, 75, 95, 99)}

    # Round-number clustering — fraction at exact integer kg.
    vals = clamped["weight_kg_raw"]
    pct_integer = float((vals == vals.round()).mean()) if len(vals) else 0.0
    pct_mod5 = float((vals.round() % 5 == 0).mean()) if len(vals) else 0.0
    pct_mod10 = float((vals.round() % 10 == 0).mean()) if len(vals) else 0.0

    smry.add("a", "n_raw_weight_rows", n_total_rows)
    smry.add("a", "n_rows_outside_clamp", n_out_of_clamp, note=f"clamp [{WEIGHT_CLAMP_LO},{WEIGHT_CLAMP_HI}]")
    smry.add("a", "n_hosp_with_weight", n_hosp_with_weight)
    smry.add("a", "n_hosp_zero_weight", n_hosp_zero_weight)
    smry.add("a", "n_cohort_hosp", len(hosp_ids))
    for p, v in pcts.items():
        smry.add("a", f"weight_p{p}", round(v, 2), unit="kg")
    smry.add("a", "pct_integer_kg", round(pct_integer, 4))
    smry.add("a", "pct_mod5_kg", round(pct_mod5, 4))
    smry.add("a", "pct_mod10_kg", round(pct_mod10, 4))

    return {
        "clamped_values": clamped["weight_kg_raw"].to_numpy(),
        "n_hosp_zero_weight": n_hosp_zero_weight,
        "top20_high": top20_high,
        "n_total_rows": n_total_rows,
        "n_out_of_clamp": n_out_of_clamp,
        "pcts": pcts,
    }


# ── Section (b) ──────────────────────────────────────────────────────────
def _clamp_vitals(raw_w: pd.DataFrame) -> pd.DataFrame:
    """Replicate apply_outlier_handling's [30, 300] clamp (drop, not winsorize)."""
    return raw_w[raw_w["weight_kg_raw"].between(WEIGHT_CLAMP_LO, WEIGHT_CLAMP_HI)].copy()


def section_b_stage_a_asof(
    med: pd.DataFrame, raw_w_clamped: pd.DataFrame, smry: AuditSummary
) -> dict:
    """Replicate clifpy's find_most_recent_weight per-admin ASOF."""
    if med.empty:
        smry.add("b", "n_admins", 0)
        return {"asof_df": pd.DataFrame()}

    # Replicate clifpy's exact SQL: ASOF backward join admin → most-recent weight.
    weights = raw_w_clamped.rename(columns={"weight_kg_raw": "weight_kg"})
    asof = duckdb.sql("""
        FROM med m
        ASOF LEFT JOIN weights v
          ON m.hospitalization_id = v.hospitalization_id
          AND v.recorded_dttm <= m.admin_dttm
        SELECT m.hospitalization_id
            , m.admin_dttm
            , m.med_category
            , m.med_dose_unit
            , m.med_dose
            , v.weight_kg
            , v.recorded_dttm AS _weight_recorded_dttm
    """).df()

    n_admins = len(asof)
    n_null_weight = int(asof["weight_kg"].isna().sum())

    # Weighted-input share by drug.
    asof["_input_weighted"] = asof["med_dose_unit"].astype(str).str.contains(
        r"/\s*kg|/\s*lb", case=False, regex=True, na=False
    )
    weighted_admins = asof[asof["_input_weighted"]]
    n_weighted_admins = len(weighted_admins)
    n_weighted_null = int(weighted_admins["weight_kg"].isna().sum())

    # Convert-failure count = weighted-input AND NULL weight.
    smry.add("b", "n_admins", n_admins)
    smry.add("b", "n_admins_null_weight", n_null_weight)
    smry.add("b", "pct_admins_null_weight", round(n_null_weight / n_admins, 4) if n_admins else 0)
    smry.add("b", "n_weighted_input_admins", n_weighted_admins)
    smry.add("b", "n_weighted_input_AND_null_weight", n_weighted_null,
             note="silent-conversion-failure count")

    # By drug.
    by_drug = asof.groupby("med_category").agg(
        n_admins=("admin_dttm", "size"),
        n_null=("weight_kg", lambda s: int(s.isna().sum())),
        n_weighted=("_input_weighted", "sum"),
        n_weighted_null=("weight_kg",
                         lambda s: int((s.isna() & asof.loc[s.index, "_input_weighted"]).sum())),
    ).reset_index()
    for _, r in by_drug.iterrows():
        smry.add("b", f"convert_fail_{r['med_category']}",
                 int(r["n_weighted_null"]),
                 note=f"of {int(r['n_weighted'])} weighted-input admins")

    # ASOF staleness (only where weight is non-null).
    have = asof[asof["weight_kg"].notna()].copy()
    have["_staleness_hr"] = (
        (_to_utc(have["admin_dttm"]) - _to_utc(have["_weight_recorded_dttm"]))
        .dt.total_seconds() / 3600.0
    )
    s = have["_staleness_hr"]
    bins = {
        "0-24h": int(((s >= 0) & (s < 24)).sum()),
        "24-72h": int(((s >= 24) & (s < 72)).sum()),
        "72h-7d": int(((s >= 72) & (s < 168)).sum()),
        ">7d": int((s >= 168).sum()),
    }
    for k, v in bins.items():
        smry.add("b", f"stage_a_staleness_{k}", v)

    return {"asof_df": asof, "by_drug": by_drug, "staleness_bins": bins}


# ── Section (c) ──────────────────────────────────────────────────────────
def section_c_conversion_outcome(med: pd.DataFrame, raw_w_clamped: pd.DataFrame,
                                 smry: AuditSummary) -> dict:
    """Run clifpy's convert_dose_units_by_med_category on a fresh load and
    count rows whose med_dose_unit_converted ≠ preferred unit.

    This is the *direct* analog of "look at cont_sed_w pivot fall-through" —
    we replicate the conversion ourselves so we count the failure regardless
    of whether downstream code filters the columns out before saving.
    """
    if med.empty:
        smry.add("c", "n_admins_input", 0)
        return {"by_drug": pd.DataFrame()}

    from clifpy.utils.unit_converter import convert_dose_units_by_med_category

    weights = raw_w_clamped.rename(columns={"weight_kg_raw": "weight_kg"})
    weights = weights.assign(vital_category="weight_kg", vital_value=weights["weight_kg"])

    # Pull `med_dose=0` for stop/not_given to mirror 02_exposure.py.
    med_in = med.copy()
    med_in.loc[
        med_in["mar_action_category"].isin(["stop", "not_given"]), "med_dose"
    ] = 0
    converted, _ = convert_dose_units_by_med_category(
        med_in, vitals_df=weights, preferred_units=SED_PREFERRED_UNITS, override=True,
    )

    # CRITICAL FINDING (clifpy bug, observed in 0.4.9): when input is /kg
    # and weight is NULL, clifpy returns _convert_status='success' with the
    # preferred (unweighted) unit string AND a numerically-converted dose
    # that has SILENTLY DROPPED the /kg factor. Example: 200 mcg/kg/min
    # (weight NULL) → 0.2 mg/min, success. For an 80-kg patient the actual
    # delivered dose is 16 mg/min — off by 80x.
    #
    # We flag these by joining input-unit-was-weighted with weight=NULL.
    converted["_input_weighted"] = converted["med_dose_unit"].astype(str).str.contains(
        r"/\s*kg|/\s*lb", case=False, regex=True, na=False
    )
    converted["_silent_bug"] = (
        converted["_input_weighted"]
        & converted["weight_kg"].isna()
        & (converted["_convert_status"] == "success")
    )
    pref_match = (
        converted["med_dose_unit_converted"].astype(str)
        == converted["med_category"].map(SED_PREFERRED_UNITS).astype(str)
    )
    dose_nonnull = converted["med_dose_converted"].notna()
    # "ok" requires status=success AND not a silent bug.
    converted["_converted_ok"] = (
        pref_match & dose_nonnull & ~converted["_silent_bug"]
    )

    # Non-silent failures: rows that aren't OK and aren't the silent-weight
    # bug. These split into three diagnostic categories with very different
    # implications:
    #
    #   1. input_nan_dose: med_dose was NaN in the source (typically
    #      mar_action_category='other' / 'start' / paused). NOT a clifpy
    #      conversion failure — just upstream missing data. Most common at
    #      UCMC. These propagate as NaN through downstream aggregation,
    #      which is fine *if* the downstream forward-fill logic in
    #      02_exposure.py treats them correctly.
    #
    #   2. unit_mismatch: med_dose is finite but med_dose_unit_converted
    #      doesn't match the preferred unit. These are the genuine
    #      "clifpy can't convert this unit string" cases (volume↔mass,
    #      unrecognized unit strings, etc.). Worth investigating.
    #
    #   3. unexpected_nan_output: status=success and unit matches preferred
    #      but med_dose_converted is NaN despite med_dose being finite.
    #      Should be ~0 — non-zero indicates a clifpy edge case.
    #
    # Federated-safe: aggregates over anonymous string keys; no IDs surfaced.
    non_silent_fail = ~converted["_converted_ok"] & ~converted["_silent_bug"]

    nsf_input_nan = non_silent_fail & converted["med_dose"].isna()
    nsf_unit_fail = (
        non_silent_fail & converted["med_dose"].notna() & ~pref_match
    )
    nsf_unexpected = (
        non_silent_fail & converted["med_dose"].notna() & pref_match & ~dose_nonnull
    )

    breakdown_rows = []
    # (1) input_nan_dose: group by (drug, mar_action) so we can spot which
    # MAR actions UCMC charts as NaN. mar_action_category is an enum-like
    # string column from CLIF.
    if nsf_input_nan.any():
        nan_in = converted.loc[nsf_input_nan]
        for (drug, action), sub in nan_in.groupby(
            ["med_category", "mar_action_category"], dropna=False
        ):
            breakdown_rows.append({
                "failure_category": "input_nan_dose",
                "key": f"mar_action={action}",
                "med_dose_unit": "(input was NaN)",
                "med_dose_unit_converted": "(propagated NaN)",
                "n_admins": int(len(sub)),
                "drugs_affected": str(drug),
            })
    # (2) unit_mismatch: group by (input, output) unit pair.
    if nsf_unit_fail.any():
        uf = converted.loc[nsf_unit_fail]
        for (in_u, out_u), sub in uf.groupby(
            ["med_dose_unit", "med_dose_unit_converted"], dropna=False
        ):
            breakdown_rows.append({
                "failure_category": "unit_mismatch",
                "key": f"{in_u} → {out_u}",
                "med_dose_unit": str(in_u),
                "med_dose_unit_converted": str(out_u),
                "n_admins": int(len(sub)),
                "drugs_affected": ",".join(sorted(sub["med_category"].unique())),
            })
    # (3) unexpected_nan_output: should be ~0; row exists for visibility.
    if nsf_unexpected.any():
        unx = converted.loc[nsf_unexpected]
        for drug, sub in unx.groupby("med_category"):
            breakdown_rows.append({
                "failure_category": "unexpected_nan_output",
                "key": f"drug={drug}",
                "med_dose_unit": str(sub["med_dose_unit"].mode().iat[0]),
                "med_dose_unit_converted": str(sub["med_dose_unit_converted"].mode().iat[0]),
                "n_admins": int(len(sub)),
                "drugs_affected": str(drug),
            })

    breakdown = (
        pd.DataFrame(breakdown_rows).sort_values(
            ["failure_category", "n_admins"], ascending=[True, False]
        ).reset_index(drop=True)
        if breakdown_rows
        else pd.DataFrame(columns=[
            "failure_category", "key", "med_dose_unit",
            "med_dose_unit_converted", "n_admins", "drugs_affected",
        ])
    )

    by_drug = converted.groupby("med_category").agg(
        n_admins=("admin_dttm", "size"),
        n_converted_ok=("_converted_ok", "sum"),
        n_silent_bug=("_silent_bug", "sum"),
    ).reset_index()
    by_drug["n_failed"] = by_drug["n_admins"] - by_drug["n_converted_ok"]
    by_drug["pct_failed"] = (by_drug["n_failed"] / by_drug["n_admins"]).round(4)

    smry.add("c", "n_admins_input", int(by_drug["n_admins"].sum()))
    smry.add("c", "n_admins_converted_ok", int(by_drug["n_converted_ok"].sum()))
    smry.add("c", "n_admins_failed", int(by_drug["n_failed"].sum()))
    smry.add("c", "n_clifpy_silent_bugs", int(by_drug["n_silent_bug"].sum()),
             note="weighted input + NULL weight + clifpy says success "
                  "(/kg factor silently dropped from dose)")
    smry.add("c", "n_input_nan_dose", int(nsf_input_nan.sum()),
             note="med_dose NaN in source (typically mar_action='other'); "
                  "propagates as NaN downstream — not a clifpy conversion failure")
    smry.add("c", "n_unit_mismatch", int(nsf_unit_fail.sum()),
             note="finite input dose but converted-unit ≠ preferred-unit; "
                  "see weight_qc_unit_failure_breakdown.csv")
    smry.add("c", "n_unexpected_nan_output", int(nsf_unexpected.sum()),
             note="finite input + matching unit + NaN output dose; should be 0")
    for _, r in by_drug.iterrows():
        smry.add("c", f"silent_bug_{r['med_category']}",
                 int(r["n_silent_bug"]),
                 note=f"of {int(r['n_admins'])} admins")
    for _, r in by_drug.iterrows():
        smry.add("c", f"conv_fail_pct_{r['med_category']}",
                 float(r["pct_failed"]),
                 note=f"{int(r['n_failed'])} of {int(r['n_admins'])}")

    # Failure → bucket of WHY.
    failed = converted[~converted["_converted_ok"]].copy()
    why = failed["_convert_status"].value_counts().to_dict() if "_convert_status" in failed.columns else {}
    for k, v in why.items():
        smry.add("c", f"why_{k[:60]}", int(v))

    return {"by_drug": by_drug, "breakdown": breakdown}


# ── Section (d) ──────────────────────────────────────────────────────────
def section_d_stage_b(weight_daily: pd.DataFrame, raw_w_clamped: pd.DataFrame,
                      smry: AuditSummary) -> dict:
    """Re-derive staleness + admission-fallback rate from weight_daily + raw."""
    n_pd_total = len(weight_daily)
    n_pd_nonnull = int(weight_daily["weight_kg_asof_day_start"].notna().sum())
    smry.add("d", "n_patient_days_total", n_pd_total)
    smry.add("d", "n_patient_days_nonnull_weight", n_pd_nonnull)
    smry.add("d", "pct_patient_days_nonnull_weight",
             round(n_pd_nonnull / n_pd_total, 4) if n_pd_total else 0)

    # Re-derive Stage B's ASOF (without the COALESCE) to identify which days
    # used the admission-weight fallback. Then subtract from the COALESCE'd
    # column → those rows where the COALESCE fell back to admission weight.
    co = duckdb.connect()
    co.register("weight_daily", weight_daily)
    co.register("raw_w", raw_w_clamped.rename(columns={"weight_kg_raw": "weight_kg"}))

    # Need the daily 7am event_dttm — reconstruct from weight_daily + analytical.
    # weight_daily has hospitalization_id + _nth_day. We need event_dttm.
    # 04_covariates.py's day_starts is filtered from cohort_shift_change_grids;
    # the simplest faithful reconstruction is to load that parquet.
    grid_path = SITE_OUT / "cohort_meta_by_id_imvhr.parquet"
    if grid_path.exists():
        grids = pd.read_parquet(
            grid_path, columns=["hospitalization_id", "_nth_day", "_hr", "event_dttm"]
        )
        day_starts = grids[(grids["_hr"] == 7) & (grids["_nth_day"] > 0)]
        co.register("day_starts", day_starts)
        asof_only = co.sql("""
            FROM day_starts d
            ASOF LEFT JOIN raw_w w
              ON d.hospitalization_id = w.hospitalization_id
              AND w.recorded_dttm <= d.event_dttm
            SELECT d.hospitalization_id, d._nth_day, d.event_dttm
                , w.weight_kg AS weight_asof
                , w.recorded_dttm AS _weight_recorded_dttm
        """).df()
        joined = weight_daily.merge(
            asof_only, on=["hospitalization_id", "_nth_day"], how="left"
        )
        had_nonnull = joined["weight_kg_asof_day_start"].notna()
        used_fallback = had_nonnull & joined["weight_asof"].isna()
        n_fallback = int(used_fallback.sum())
        smry.add("d", "n_patient_days_used_admission_fallback", n_fallback)
        smry.add("d", "pct_nonnull_using_fallback",
                 round(n_fallback / n_pd_nonnull, 4) if n_pd_nonnull else 0)

        have = joined[joined["weight_asof"].notna()].copy()
        have["_staleness_hr"] = (
            (_to_utc(have["event_dttm"]) - _to_utc(have["_weight_recorded_dttm"]))
            .dt.total_seconds() / 3600.0
        )
        s = have["_staleness_hr"]
        bins = {
            "0-24h": int(((s >= 0) & (s < 24)).sum()),
            "24-72h": int(((s >= 24) & (s < 72)).sum()),
            "72h-7d": int(((s >= 72) & (s < 168)).sum()),
            ">7d": int((s >= 168).sum()),
        }
        for k, v in bins.items():
            smry.add("d", f"stage_b_staleness_{k}", v)
        return {"joined": joined, "staleness_bins": bins}
    else:
        smry.add("d", "ERROR", "cohort_meta_by_id_imvhr.parquet not found",
                 note="cannot re-derive Stage B ASOF without grid")
        return {"joined": pd.DataFrame(), "staleness_bins": {}}


# ── Section (e) ──────────────────────────────────────────────────────────
def section_e_cross_stage(asof_df: pd.DataFrame, joined_b: pd.DataFrame,
                          smry: AuditSummary) -> dict:
    """For each admin, compare Stage A weight to Stage B day-start weight."""
    if asof_df.empty or joined_b.empty:
        smry.add("e", "n_compared", 0)
        return {}

    # Map each admin to its day_start by floor(admin_dttm) - then look up Stage B.
    a = asof_df[["hospitalization_id", "admin_dttm", "weight_kg"]].rename(
        columns={"weight_kg": "weight_a"}
    )
    b = joined_b[["hospitalization_id", "event_dttm", "weight_kg_asof_day_start"]].rename(
        columns={"event_dttm": "_day_start_dttm", "weight_kg_asof_day_start": "weight_b"}
    )
    # ASOF backward-join admin → most-recent day_start.
    co = duckdb.connect()
    co.register("a", a)
    co.register("b", b)
    merged = co.sql("""
        FROM a
        ASOF LEFT JOIN b
          ON a.hospitalization_id = b.hospitalization_id
          AND b._day_start_dttm <= a.admin_dttm
        SELECT a.hospitalization_id, a.admin_dttm
            , a.weight_a, b.weight_b
    """).df()
    n_compared = int((merged["weight_a"].notna() & merged["weight_b"].notna()).sum())
    n_a_only = int((merged["weight_a"].notna() & merged["weight_b"].isna()).sum())
    n_b_only_rescue = int((merged["weight_a"].isna() & merged["weight_b"].notna()).sum())

    diff = (merged["weight_a"] - merged["weight_b"]).abs()
    p50 = float(np.nanpercentile(diff, 50)) if n_compared else 0.0
    p95 = float(np.nanpercentile(diff, 95)) if n_compared else 0.0
    p99 = float(np.nanpercentile(diff, 99)) if n_compared else 0.0
    n_diverge = int((diff > 5).sum())

    smry.add("e", "n_compared_both_nonnull", n_compared)
    smry.add("e", "n_only_stage_a_has_weight", n_a_only)
    smry.add("e", "n_only_stage_b_rescue", n_b_only_rescue,
             note="admin where Stage A NULL but Stage B fell back to admission")
    smry.add("e", "abs_diff_p50_kg", round(p50, 2))
    smry.add("e", "abs_diff_p95_kg", round(p95, 2))
    smry.add("e", "abs_diff_p99_kg", round(p99, 2))
    smry.add("e", "n_diverge_gt_5kg", n_diverge)

    return {"merged": merged, "diff": diff}


# ── Section (f) ──────────────────────────────────────────────────────────
def section_f_stability(raw_w_clamped: pd.DataFrame, smry: AuditSummary) -> dict:
    """Per-patient stability + cutoff-defensibility CDFs."""
    g = raw_w_clamped.groupby("hospitalization_id")
    per_patient = g.agg(
        n=("weight_kg_raw", "size"),
        wmin=("weight_kg_raw", "min"),
        wmax=("weight_kg_raw", "max"),
        wsd=("weight_kg_raw", "std"),
    )
    per_patient["range_kg"] = per_patient["wmax"] - per_patient["wmin"]
    multi = per_patient[per_patient["n"] >= 2].copy()

    # Per-pair consecutive jump + time gap. The drop rule per the plan is
    # raw_jump > X kg AND dt_hr < Y hours (NOT a normalized kg/24h rate —
    # normalizing inflates short-gap recordings and misclassifies them).
    raw_sorted = raw_w_clamped.sort_values(["hospitalization_id", "recorded_dttm"]).copy()
    raw_sorted["_prev_w"] = raw_sorted.groupby("hospitalization_id")["weight_kg_raw"].shift()
    raw_sorted["_prev_t"] = raw_sorted.groupby("hospitalization_id")["recorded_dttm"].shift()
    _t_now = _to_utc(raw_sorted["recorded_dttm"])
    _t_prev = _to_utc(raw_sorted["_prev_t"])
    raw_sorted["_dt_hr"] = (_t_now - _t_prev).dt.total_seconds() / 3600.0
    raw_sorted["_jump_kg"] = (raw_sorted["weight_kg_raw"] - raw_sorted["_prev_w"]).abs()

    # For the rule itself: only consider pairs within the configured window.
    in_window = raw_sorted[
        raw_sorted["_dt_hr"].notna() & (raw_sorted["_dt_hr"] < WEIGHT_QC_MAX_JUMP_HOURS)
    ].copy()
    max_jump_in_window = in_window.groupby("hospitalization_id")["_jump_kg"].max()
    multi = multi.join(max_jump_in_window.rename("max_jump_in_window_kg"), how="left")

    # CDF inputs: range and (windowed) raw jump.
    range_cuts = [10, 15, 20, 25, 30, 40, 50, 75, 100]
    jump_cuts = [10, 15, 20, 25, 30, 50, 75, 100]
    n_total = len(per_patient)  # patients with ≥1 weight
    range_excl = {c: int((multi["range_kg"] > c).sum()) for c in range_cuts}
    jump_excl = {
        c: int((multi["max_jump_in_window_kg"] > c).sum()) for c in jump_cuts
    }

    smry.add("f", "n_patients_with_2plus_weights", len(multi))
    smry.add("f", "range_kg_p50",
             round(float(multi["range_kg"].median()), 2) if len(multi) else 0)
    smry.add("f", "range_kg_p95",
             round(float(multi["range_kg"].quantile(0.95)), 2) if len(multi) else 0)
    smry.add("f", "max_jump_in_window_p50",
             round(float(multi["max_jump_in_window_kg"].median()), 2)
             if len(multi) and multi["max_jump_in_window_kg"].notna().any() else 0,
             note=f"jumps within < {WEIGHT_QC_MAX_JUMP_HOURS}h")
    smry.add("f", "max_jump_in_window_p95",
             round(float(multi["max_jump_in_window_kg"].quantile(0.95)), 2)
             if len(multi) and multi["max_jump_in_window_kg"].notna().any() else 0,
             note=f"jumps within < {WEIGHT_QC_MAX_JUMP_HOURS}h")
    for c, n in range_excl.items():
        smry.add("f", f"range_cutoff_{c}kg_excludes_n", n,
                 note=f"{round(n / n_total * 100, 2)}% of cohort" if n_total else "")
    for c, n in jump_excl.items():
        smry.add("f", f"jump_cutoff_{c}kg_in_window_excludes_n", n,
                 note=f"{round(n / n_total * 100, 2)}% of cohort"
                      f" (window < {WEIGHT_QC_MAX_JUMP_HOURS}h)" if n_total else "")

    return {
        "per_patient": per_patient, "multi": multi,
        "range_excl": range_excl, "jump_excl": jump_excl,
        "range_cuts": range_cuts, "jump_cuts": jump_cuts,
    }


# ── Section (g) ──────────────────────────────────────────────────────────
def section_g_drop_list(
    raw_w_clamped: pd.DataFrame, hosp_ids: list[str], stab: dict,
    weight_daily: pd.DataFrame, smry: AuditSummary,
) -> dict:
    """Build the drop list with criterion-incremental counts."""
    cohort = set(hosp_ids)
    multi = stab["multi"]

    # Criterion 1: zero raw weights in current admission.
    has_weight = set(raw_w_clamped["hospitalization_id"].unique())
    zero_weight = cohort - has_weight
    crit1 = pd.DataFrame({
        "hospitalization_id": sorted(zero_weight),
        "_drop_reason": "zero_weight_in_admission",
    })

    # Criterion 2: implausible jump = ANY consecutive pair with jump > X kg
    # AND dt_hr < Y hours. Excluded incrementally — only if not in crit1.
    jump_violators = set(
        multi[multi["max_jump_in_window_kg"] > WEIGHT_QC_MAX_JUMP_KG].index
    )
    crit2_ids = jump_violators - zero_weight
    crit2 = pd.DataFrame({
        "hospitalization_id": sorted(crit2_ids),
        "_drop_reason": (
            f"jump_gt_{int(WEIGHT_QC_MAX_JUMP_KG)}kg_within_"
            f"{int(WEIGHT_QC_MAX_JUMP_HOURS)}h"
        ),
    })

    # Criterion 3: range rule (opt-in).
    crit3_ids = set()
    if WEIGHT_QC_RANGE_RULE_ON:
        range_violators = set(multi[multi["range_kg"] > WEIGHT_QC_MAX_RANGE_KG].index)
        crit3_ids = range_violators - zero_weight - crit2_ids
    crit3 = pd.DataFrame({
        "hospitalization_id": sorted(crit3_ids),
        "_drop_reason": f"range_gt_{int(WEIGHT_QC_MAX_RANGE_KG)}kg",
    })

    drop_list = pd.concat([crit1, crit2, crit3], ignore_index=True)

    smry.add("g", "drop_zero_weight", len(crit1))
    smry.add("g", "drop_jump", len(crit2),
             note=f"threshold {WEIGHT_QC_MAX_JUMP_KG} kg/24h")
    smry.add("g", "drop_range", len(crit3),
             note=f"threshold {WEIGHT_QC_MAX_RANGE_KG} kg" if WEIGHT_QC_RANGE_RULE_ON else "rule disabled")
    smry.add("g", "drop_total_unique_hosp", drop_list["hospitalization_id"].nunique())
    smry.add("g", "cohort_n", len(cohort))
    smry.add("g", "drop_pct_of_cohort",
             round(drop_list["hospitalization_id"].nunique() / len(cohort) * 100, 2))

    # Patient-day impact of drop.
    dropped_set = set(drop_list["hospitalization_id"])
    pd_dropped = int(weight_daily["hospitalization_id"].isin(dropped_set).sum())
    pd_total = len(weight_daily)
    smry.add("g", "patient_days_dropped", pd_dropped)
    smry.add("g", "patient_days_total", pd_total)
    smry.add("g", "patient_days_drop_pct",
             round(pd_dropped / pd_total * 100, 2) if pd_total else 0)

    # Federated-safe per-criterion exclusion table (no IDs).
    excl_rows = [
        {
            "criterion": "zero_weight_in_admission",
            "threshold": "n/a",
            "n_hosp_dropped": len(crit1),
            "n_patient_days_dropped": int(weight_daily["hospitalization_id"].isin(set(crit1["hospitalization_id"])).sum()),
            "pct_cohort": round(len(crit1) / len(cohort) * 100, 2) if cohort else 0,
        },
        {
            "criterion": "jump_per_24h",
            "threshold": f"{WEIGHT_QC_MAX_JUMP_KG} kg / {WEIGHT_QC_MAX_JUMP_HOURS}h",
            "n_hosp_dropped": len(crit2),
            "n_patient_days_dropped": int(weight_daily["hospitalization_id"].isin(set(crit2["hospitalization_id"])).sum()),
            "pct_cohort": round(len(crit2) / len(cohort) * 100, 2) if cohort else 0,
        },
        {
            "criterion": "range_within_stay",
            "threshold": f"{WEIGHT_QC_MAX_RANGE_KG} kg" if WEIGHT_QC_RANGE_RULE_ON else "DISABLED",
            "n_hosp_dropped": len(crit3),
            "n_patient_days_dropped": int(weight_daily["hospitalization_id"].isin(set(crit3["hospitalization_id"])).sum()),
            "pct_cohort": round(len(crit3) / len(cohort) * 100, 2) if cohort else 0,
        },
    ]
    return {"drop_list": drop_list, "excl_rows": excl_rows, "dropped_set": dropped_set}


# ── Section (h) ──────────────────────────────────────────────────────────
def section_h_impact(weight_daily: pd.DataFrame, raw_w_clamped: pd.DataFrame,
                     joined_b: pd.DataFrame, smry: AuditSummary) -> dict:
    """Recompute prop_dif_mcg_kg_min four ways and compare.

    Phase-2 detection: if the analytical dataset already carries
    `prop_dif_mcg_kg_min` directly (post-Phase-2 schema), the
    multi-strategy comparison is moot — Stage A's pre-attached weight is
    the only weight in play. We emit a single `phase_2_pipeline` row with
    the descriptive's actual stats and skip the strategies. Pre-Phase-2
    datasets fall through to the original 4-strategy logic.
    """
    df = load_analytical()

    # Phase 2 schema: prop_dif_mcg_kg_min is produced upstream.
    if "prop_dif_mcg_kg_min" in df.columns and "prop_dif_mg_hr" not in df.columns:
        s = df["prop_dif_mcg_kg_min"].dropna()
        ab = s.abs()
        row = {
            "strategy": "phase_2_pipeline",
            "n": int(len(ab)),
            "mean_abs": float(ab.mean()) if len(ab) else 0.0,
            "p50_abs": float(np.percentile(ab, 50)) if len(ab) else 0.0,
            "p95_abs": float(np.percentile(ab, 95)) if len(ab) else 0.0,
        }
        smry.add("h", "phase_2_n_pd", row["n"])
        smry.add("h", "phase_2_mean_abs", round(row["mean_abs"], 4),
                 unit="mcg/kg/min",
                 note="Phase-2 pipeline produces prop_dif_mcg_kg_min directly; "
                      "multi-strategy comparison skipped")
        smry.add("h", "phase_2_p95_abs", round(row["p95_abs"], 4),
                 unit="mcg/kg/min")
        return {"compare_df": pd.DataFrame([row])}

    # Pre-Phase-2: original 4-strategy comparison.
    if "prop_dif_mg_hr" not in df.columns:
        smry.add("h", "ERROR",
                 "neither prop_dif_mg_hr nor prop_dif_mcg_kg_min in modeling_dataset",
                 note="skipping impact")
        return {"compare_df": pd.DataFrame()}

    # Strategy 1: status quo
    w_sq = df["weight_kg_asof_day_start"].clip(upper=300)
    s1 = df["prop_dif_mg_hr"] * 1000.0 / 60.0 / w_sq

    # Strategy 2: fresh-ASOF only. Drop patient-days whose Stage B ASOF is
    # >72h stale or used the admission-weight fallback (recorded_dttm NaT).
    # joined_b carries the per-day _weight_recorded_dttm from section (d).
    if not joined_b.empty and "_weight_recorded_dttm" in joined_b.columns:
        jb = joined_b[["hospitalization_id", "_nth_day",
                       "event_dttm", "_weight_recorded_dttm"]].copy()
        jb["_staleness_hr"] = (
            (_to_utc(jb["event_dttm"]) - _to_utc(jb["_weight_recorded_dttm"]))
            .dt.total_seconds() / 3600.0
        )
        df2 = df.merge(jb[["hospitalization_id", "_nth_day", "_staleness_hr"]],
                       on=["hospitalization_id", "_nth_day"], how="left")
        is_fresh = df2["_staleness_hr"].notna() & (df2["_staleness_hr"] < 72)
        w_fresh = df2["weight_kg_asof_day_start"].clip(upper=300).where(is_fresh, np.nan)
        s2 = df2["prop_dif_mg_hr"] * 1000.0 / 60.0 / w_fresh
    else:
        s2 = pd.Series(dtype=float)

    # Strategy 3: patient-median weight
    pmed = raw_w_clamped.groupby("hospitalization_id")["weight_kg_raw"].median()
    df3 = df.merge(pmed.rename("_w_median").reset_index(), on="hospitalization_id", how="left")
    w_med = df3["_w_median"].clip(upper=300)
    s3 = df3["prop_dif_mg_hr"] * 1000.0 / 60.0 / w_med

    # Strategy 4: winsorize at empirical p1/p99
    p1 = float(np.nanpercentile(raw_w_clamped["weight_kg_raw"], 1))
    p99 = float(np.nanpercentile(raw_w_clamped["weight_kg_raw"], 99))
    w_wins = df["weight_kg_asof_day_start"].clip(lower=p1, upper=p99)
    s4 = df["prop_dif_mg_hr"] * 1000.0 / 60.0 / w_wins

    def _summary(name, s):
        s = s.dropna()
        if not len(s):
            return {"strategy": name, "n": 0, "mean_abs": 0, "p50_abs": 0, "p95_abs": 0}
        ab = s.abs()
        return {
            "strategy": name,
            "n": len(ab),
            "mean_abs": float(ab.mean()),
            "p50_abs": float(np.percentile(ab, 50)),
            "p95_abs": float(np.percentile(ab, 95)),
        }

    rows = [
        _summary("status_quo", s1),
        _summary("fresh_only", s2),
        _summary("patient_median", s3),
        _summary("winsorized", s4),
    ]
    cmp_df = pd.DataFrame(rows)
    for r in rows:
        smry.add("h", f"{r['strategy']}_n_pd", r["n"])
        smry.add("h", f"{r['strategy']}_mean_abs", round(r["mean_abs"], 4), unit="mcg/kg/min")
        smry.add("h", f"{r['strategy']}_p95_abs", round(r["p95_abs"], 4), unit="mcg/kg/min")
    return {"compare_df": cmp_df, "p1": p1, "p99": p99}


# ── Plotting ─────────────────────────────────────────────────────────────
def render_panel_png(sec_a: dict, sec_b: dict, sec_d: dict, sec_e: dict,
                     sec_f: dict, sec_h: dict, out_path: Path) -> None:
    """Multi-panel summary figure — federated-safe (histograms, no IDs)."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    # (a) raw weight distribution + clamp
    ax = axes[0, 0]
    vals = sec_a["clamped_values"]
    if len(vals):
        ax.hist(vals, bins=60, color="#4a7faa", edgecolor="white")
    ax.axvline(WEIGHT_CLAMP_LO, ls="--", color="#b2182b", alpha=0.6)
    ax.axvline(WEIGHT_CLAMP_HI, ls="--", color="#b2182b", alpha=0.6)
    ax.set_title(f"(a) Raw weight (post-clamp)\nclamped {sec_a['n_out_of_clamp']} of {sec_a['n_total_rows']} rows")
    ax.set_xlabel("kg")
    ax.set_ylabel("rows")

    # (b) Stage A staleness bins
    ax = axes[0, 1]
    bins = sec_b.get("staleness_bins", {})
    if bins:
        ax.bar(list(bins.keys()), list(bins.values()), color="#88a")
    ax.set_title("(b) Stage A ASOF staleness\n(per-admin)")
    ax.set_ylabel("admins")

    # (c) per-drug conversion failure (uses sec_b's by_drug for n_weighted_null)
    ax = axes[0, 2]
    bd = sec_b.get("by_drug")
    if bd is not None and not bd.empty:
        ax.barh(bd["med_category"], bd["n_weighted_null"], color="#b2182b")
        ax.set_xlabel("# weighted-input admins with NULL weight")
    ax.set_title("(c) Convert-failure count by drug")

    # (d) Stage B staleness
    ax = axes[1, 0]
    bins_b = sec_d.get("staleness_bins", {})
    if bins_b:
        ax.bar(list(bins_b.keys()), list(bins_b.values()), color="#5a9")
    ax.set_title("(d) Stage B ASOF staleness\n(per-day)")
    ax.set_ylabel("patient-days")

    # (e) cross-stage |diff| distribution
    ax = axes[1, 1]
    diff = sec_e.get("diff")
    if diff is not None and len(diff):
        d = diff.dropna()
        d = d[d <= 30]
        ax.hist(d, bins=40, color="#999")
    ax.axvline(5, ls="--", color="#b2182b")
    ax.set_title("(e) |Stage A − Stage B| weight\n(red: 5 kg threshold)")
    ax.set_xlabel("|Δ| kg")

    # (f1) range histogram with cutoff lines
    ax = axes[1, 2]
    multi = sec_f["multi"]
    if len(multi):
        ax.hist(multi["range_kg"].clip(upper=80), bins=40, color="#a64")
    for c in [10, 20, 30]:
        ax.axvline(c, ls=":", color="#b2182b", alpha=0.5)
    ax.set_title("(f1) Within-stay weight range")
    ax.set_xlabel("max − min (kg, cap 80)")

    # (f2) cohort-exclusion-vs-cutoff CDF
    ax = axes[2, 0]
    cuts = sec_f["range_cuts"]
    n_total = max(1, len(sec_f["per_patient"]))
    range_pct = [sec_f["range_excl"][c] / n_total * 100 for c in cuts]
    jcuts = sec_f["jump_cuts"]
    jump_pct = [sec_f["jump_excl"][c] / n_total * 100 for c in jcuts]
    ax.plot(cuts, range_pct, marker="o",
            label="range cutoff (kg, full stay)", color="#a64")
    ax.plot(jcuts, jump_pct, marker="s",
            label=f"jump cutoff (kg, gap < {WEIGHT_QC_MAX_JUMP_HOURS}h)",
            color="#46a")
    ax.axvline(WEIGHT_QC_MAX_JUMP_KG, ls="--", color="#46a", alpha=0.5)
    if WEIGHT_QC_RANGE_RULE_ON:
        ax.axvline(WEIGHT_QC_MAX_RANGE_KG, ls="--", color="#a64", alpha=0.5)
    ax.set_title("(f2) Cohort-exclusion-vs-cutoff\n(elbow → defensible threshold)")
    ax.set_xlabel("cutoff (kg)")
    ax.set_ylabel("% of cohort excluded")
    ax.legend(fontsize=8)

    # (f3) max in-window jump (raw kg, gap < jump-window)
    ax = axes[2, 1]
    if len(multi):
        mj = multi["max_jump_in_window_kg"].dropna().clip(upper=80)
        ax.hist(mj, bins=40, color="#46a")
    ax.axvline(WEIGHT_QC_MAX_JUMP_KG, ls="--", color="#b2182b")
    ax.set_title(
        f"(f3) Max consecutive jump within < {WEIGHT_QC_MAX_JUMP_HOURS}h"
    )
    ax.set_xlabel("kg (cap 80)")

    # (h) impact comparison
    ax = axes[2, 2]
    cmp_df = sec_h.get("compare_df")
    if cmp_df is not None and not cmp_df.empty:
        x = np.arange(len(cmp_df))
        ax.bar(x - 0.2, cmp_df["mean_abs"], 0.4, label="mean |Δ|", color="#a64")
        ax.bar(x + 0.2, cmp_df["p95_abs"], 0.4, label="p95 |Δ|", color="#46a")
        ax.set_xticks(x)
        ax.set_xticklabels(cmp_df["strategy"], rotation=20, ha="right")
        ax.set_ylabel("mcg/kg/min")
        ax.legend(fontsize=8)
    ax.set_title("(h) prop_dif_mcg_kg_min by weight strategy")

    fig.suptitle(
        f"Weight-QC audit — site={SITE} | clifpy={_clifpy_commit()} | "
        f"jump_cutoff={WEIGHT_QC_MAX_JUMP_KG} kg/{WEIGHT_QC_MAX_JUMP_HOURS}h | "
        f"range_rule={'ON ' + str(WEIGHT_QC_MAX_RANGE_KG) + ' kg' if WEIGHT_QC_RANGE_RULE_ON else 'OFF'}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ── Output writing ───────────────────────────────────────────────────────
def _csv_with_header(path: Path, df: pd.DataFrame, header_lines: list[str]) -> None:
    """Write a CSV preceded by `# `-prefixed header lines (run-time provenance)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False)


def write_outputs(smry: AuditSummary, sec_c: dict, sec_g: dict, sec_h: dict) -> None:
    """Two-tier output: federated under output_to_share/, PHI under output/."""
    SITE_QC.mkdir(parents=True, exist_ok=True)
    SHARE_QC.mkdir(parents=True, exist_ok=True)
    SHARE_FIG.mkdir(parents=True, exist_ok=True)

    header = [
        f"site={SITE}",
        f"clifpy_commit={_clifpy_commit()}",
        f"jump_threshold_kg={WEIGHT_QC_MAX_JUMP_KG}",
        f"jump_threshold_hours={WEIGHT_QC_MAX_JUMP_HOURS}",
        f"range_threshold_kg={WEIGHT_QC_MAX_RANGE_KG} (active={WEIGHT_QC_RANGE_RULE_ON})",
        f"raw_clamp_kg=[{WEIGHT_CLAMP_LO},{WEIGHT_CLAMP_HI}]",
    ]

    # Federated.
    _csv_with_header(
        SHARE_QC / "weight_qc_summary.csv", smry.to_df(), header,
    )
    _csv_with_header(
        SHARE_QC / "weight_qc_exclusions.csv", pd.DataFrame(sec_g["excl_rows"]), header,
    )
    cmp_df = sec_h.get("compare_df", pd.DataFrame())
    if not cmp_df.empty:
        _csv_with_header(SHARE_QC / "weight_impact_comparison.csv", cmp_df, header)
    breakdown = sec_c.get("breakdown", pd.DataFrame())
    _csv_with_header(
        SHARE_QC / "weight_qc_unit_failure_breakdown.csv", breakdown, header,
    )

    # PHI-internal: drop list parquet (not CSV — preserves integer admin counts etc.)
    drop_list = sec_g["drop_list"]
    drop_list.to_parquet(SITE_QC / "weight_qc_drop_list.parquet", index=False)

    # PHI-internal: examples CSV with patient-level details for local eyeballing.
    # (Empty for now — pipeline only flags hospitalizations as a unit.)
    # We at least dump the drop_list as a CSV mirror for terminal eyeballing.
    drop_list.to_csv(SITE_QC / "weight_audit_examples.csv", index=False)


# ── Orchestrator ─────────────────────────────────────────────────────────
def main() -> None:
    logger.info(f"\n=== Weight QC audit — site={SITE} ===")
    logger.info(f"clifpy commit: {_clifpy_commit()}")
    logger.info(f"jump threshold: {WEIGHT_QC_MAX_JUMP_KG} kg / {WEIGHT_QC_MAX_JUMP_HOURS} h")
    logger.info(f"range rule: {'ON ' + str(WEIGHT_QC_MAX_RANGE_KG) + ' kg' if WEIGHT_QC_RANGE_RULE_ON else 'OFF'}\n")

    smry = AuditSummary()
    hosp_ids = load_cohort_hosp_ids()
    logger.info(f"cohort hospitalizations: {len(hosp_ids)}")
    raw_w = load_raw_weight_events(hosp_ids)
    logger.info(f"raw weight rows (>0): {len(raw_w)}")

    sec_a = section_a_raw_availability(raw_w, hosp_ids, smry)
    raw_w_clamped = _clamp_vitals(raw_w)
    logger.info(f"clamped weight rows: {len(raw_w_clamped)}")

    med = load_med_admins(hosp_ids)
    logger.info(f"sedative admin rows: {len(med)}")

    sec_b = section_b_stage_a_asof(med, raw_w_clamped, smry)
    sec_c = section_c_conversion_outcome(med, raw_w_clamped, smry)
    weight_daily = load_weight_daily()
    sec_d = section_d_stage_b(weight_daily, raw_w_clamped, smry)
    sec_e = section_e_cross_stage(sec_b.get("asof_df", pd.DataFrame()),
                                   sec_d.get("joined", pd.DataFrame()), smry)
    sec_f = section_f_stability(raw_w_clamped, smry)
    sec_g = section_g_drop_list(raw_w_clamped, hosp_ids, sec_f, weight_daily, smry)
    sec_h = section_h_impact(weight_daily, raw_w_clamped,
                              sec_d.get("joined", pd.DataFrame()), smry)

    logger.info("\n--- Summary preview ---")
    logger.info(smry.to_df().to_string(index=False, max_colwidth=80))

    render_panel_png(sec_a, sec_b, sec_d, sec_e, sec_f, sec_h,
                     SHARE_FIG / "weight_audit.png")
    write_outputs(smry, sec_c, sec_g, sec_h)

    logger.info(f"\nWrote:")
    logger.info(f"  {SHARE_QC}/weight_qc_summary.csv")
    logger.info(f"  {SHARE_QC}/weight_qc_exclusions.csv")
    logger.info(f"  {SHARE_QC}/weight_impact_comparison.csv")
    logger.info(f"  {SHARE_QC}/weight_qc_unit_failure_breakdown.csv")
    logger.info(f"  {SHARE_FIG}/weight_audit.png")
    logger.info(f"  {SITE_QC}/weight_qc_drop_list.parquet")
    logger.info(f"  {SITE_QC}/weight_audit_examples.csv")


if __name__ == "__main__":
    main()
