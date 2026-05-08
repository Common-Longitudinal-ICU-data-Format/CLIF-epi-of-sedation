"""08b — Liberation cascade modeling.

Decomposes the sedation→liberation pathway into 4 sequential conditional
stages, each with its own cohort × outcome × 7 specs × 2 methods:

  Stage 0: All IMV-days       → sbt_elig_next_day             (Q: predicting eligibility)
  Stage 1: SBT-eligible       → sbt_done_v2_next_day          (Q: given eligible, was SBT done?)
  Stage 2: SBT performed      → extub_event_v2_next_day       (Q: given SBT, did it escalate to extub?)
  Stage 3: Extub-event        → success_extub_v2_next_day     (Q: given extub, was it successful?)

Outputs land in `output_to_share/{site}/models/` with `cascade_` prefix —
flat (no subdir) per the user's project convention.

Constants and helpers below are COPIED from `code/08_models.py` rather
than imported, to avoid the marimo-cell-return entanglement. Source of
truth is 08; if HURDLE_INDICATORS / VAR_DISPLAY / formulas change there,
update here too. TODO: extract to `code/_models_common.py` in a later
refactor round.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from clifpy.utils.config import get_config_or_params
from patsy import dmatrix

sys.path.insert(0, str(Path(__file__).parent))


# ── Site config ─────────────────────────────────────────────────────────
CONFIG_PATH = "config/config.json"
cfg = get_config_or_params(CONFIG_PATH)
SITE_NAME = cfg["site_name"].lower()
OUT_DIR = f"output_to_share/{SITE_NAME}/models"
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Site: {SITE_NAME}")


# ── Constants copied from 08_models.py (source of truth = 08) ──────────
# If you change these here, mirror in 08_models.py.
VAR_DISPLAY = {
    "prop_dif_mcg_kg_min":  {"scale": 10,  "label": "Δ propofol (per 10 mcg/kg/min)"},
    "fenteq_dif_mcg_hr":    {"scale": 10,  "label": "Δ fentanyl eq (per 10 mcg/hr)"},
    "midazeq_dif_mg_hr":    {"scale": 0.1, "label": "Δ midazolam eq (per 0.1 mg/hr)"},
    "_prop_day_mcg_kg_min": {"scale": 10,  "label": "Daytime propofol (per 10 mcg/kg/min)"},
    "_fenteq_day_mcg_hr":   {"scale": 10,  "label": "Daytime fentanyl eq (per 10 mcg/hr)"},
    "_midazeq_day_mg_hr":   {"scale": 0.1, "label": "Daytime midazolam eq (per 0.1 mg/hr)"},
    "_prop_any":   {"scale": 1, "label": "Any propofol use (day or night, yes/no)"},
    "_fenteq_any": {"scale": 1, "label": "Any fentanyl eq use (day or night, yes/no)"},
    "_midazeq_any": {"scale": 1, "label": "Any midazolam eq use (day or night, yes/no)"},
    "age":        {"scale": 1,   "label": "Age (per year)"},
    "_nth_day":   {"scale": 1,   "label": "Day on IMV (per day)"},
    "cci_score":  {"scale": 1,   "label": "Charlson CCI (per point)"},
    "sofa_total": {"scale": 1,   "label": "SOFA total (per point)"},
    "nee_7am":    {"scale": 0.1, "label": "NEE 7am (per 0.1 mcg/kg/min)"},
    "nee_7pm":    {"scale": 0.1, "label": "NEE 7pm (per 0.1 mcg/kg/min)"},
    "weight_kg":  {"scale": 10,  "label": "Weight (per 10 kg)"},
    "bmi":        {"scale": 5,   "label": "BMI (per 5 kg/m²)"},
}

HURDLE_INDICATORS = {
    "_prop_day_mcg_kg_min": "_prop_any",
    "_fenteq_day_mcg_hr":   "_fenteq_any",
    "_midazeq_day_mg_hr":   "_midazeq_any",
}

_RCS_VARS = [
    "prop_dif_mcg_kg_min", "fenteq_dif_mcg_hr", "midazeq_dif_mg_hr",
    "_prop_day_mcg_kg_min", "_fenteq_day_mcg_hr", "_midazeq_day_mg_hr",
]

BASELINE = (
    "{{outcome}} ~ prop_dif_mcg_kg_min + fenteq_dif_mcg_hr + midazeq_dif_mg_hr + "
    "age + _nth_day + C(sex_category) + C(icu_type) + cci_score"
)
DAYDOSE = BASELINE + (
    " + _prop_any + _fenteq_any + _midazeq_any"
    " + _prop_day_mcg_kg_min + _midazeq_day_mg_hr + _fenteq_day_mcg_hr"
)
SOFA = DAYDOSE + " + sofa_total"
CLINICAL = DAYDOSE + (
    " + ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm"
)
SOFA_WEIGHT = SOFA + " + weight_kg"
SOFA_BMI    = SOFA + " + bmi"

FOREST_PREDICTORS = [
    ("prop_dif_mcg_kg_min",  "Δ propofol (mcg/kg/min)"),
    ("fenteq_dif_mcg_hr",    "Δ fentanyl eq (mcg/hr)"),
    ("midazeq_dif_mg_hr",    "Δ midazolam eq (mg/hr)"),
    ("_prop_day_mcg_kg_min", "Daytime propofol (mcg/kg/min)"),
    ("_fenteq_day_mcg_hr",   "Daytime fentanyl eq (mcg/hr)"),
    ("_midazeq_day_mg_hr",   "Daytime midazolam eq (mg/hr)"),
    ("_prop_any",            "Any propofol use (24h, yes/no)"),
    ("_fenteq_any",          "Any fentanyl eq use (24h, yes/no)"),
    ("_midazeq_any",         "Any midazolam eq use (24h, yes/no)"),
]
SPEC_ORDER = ["baseline", "daydose", "sofa", "clinical", "sofa_rcs", "sofa_weight", "sofa_bmi"]
SPEC_COLORS = {
    "baseline":    "#5e3c99",
    "daydose":     "#1f77b4",
    "sofa":        "#2ca02c",
    "clinical":    "#ff7f0e",
    "sofa_rcs":    "#d62728",
    "sofa_weight": "#17becf",
    "sofa_bmi":    "#bcbd22",
}


# ── Load modeling dataset and derive extub_event_v2_next_day ───────────
df_full = pd.read_parquet(f"output/{SITE_NAME}/modeling_dataset.parquet")
print(f"Modeling dataset: {len(df_full)} rows")

if "extub_event_v2_next_day" not in df_full.columns:
    daily = pd.read_parquet(f"output/{SITE_NAME}/outcomes_by_id_imvday.parquet")
    daily["_extub_event_v2"] = (
        (daily["_success_extub_v2"] == 1) | (daily["_fail_extub_v2"] == 1)
    ).astype(int)
    daily = daily.sort_values(["hospitalization_id", "_nth_day"])
    daily["extub_event_v2_next_day"] = (
        daily.groupby("hospitalization_id")["_extub_event_v2"].shift(-1)
    )
    df_full = df_full.merge(
        daily[["hospitalization_id", "_nth_day", "extub_event_v2_next_day"]],
        on=["hospitalization_id", "_nth_day"],
        how="left",
    )
    print(
        f"Derived extub_event_v2_next_day: "
        f"{int((df_full['extub_event_v2_next_day'] == 1).sum())} positive rows of {len(df_full)}"
    )


# ── Cascade stages ─────────────────────────────────────────────────────
# Each stage: (label, cohort_filter_lambda, outcome_col, human_title)
STAGES = [
    {
        "label": "stage0",
        "filter": lambda d: d.copy(),
        "outcome": "sbt_elig_next_day",
        "title": "Stage 0: All IMV-days → SBT-eligible next day",
    },
    {
        "label": "stage1",
        "filter": lambda d: d[d["sbt_elig_next_day"] == 1].copy(),
        "outcome": "sbt_done_v2_next_day",
        "title": "Stage 1: SBT-eligible → SBT performed next day",
    },
    {
        "label": "stage2",
        "filter": lambda d: d[d["sbt_done_v2_next_day"] == 1].copy(),
        "outcome": "extub_event_v2_next_day",
        "title": "Stage 2: SBT performed → Extubation next day",
    },
    {
        # Mehta-Model-2 analogue: among SBT-performed days, did the patient
        # successfully liberate? Decomposes as P(extub | sbt) × P(success | extub).
        # Larger N than Stage 3 (because non-extub-after-SBT days count as 0)
        # but mixes two decision points (escalation + success) into one outcome.
        "label": "stage2b",
        "filter": lambda d: d[d["sbt_done_v2_next_day"] == 1].copy(),
        "outcome": "success_extub_v2_next_day",
        "title": "Stage 2b: SBT performed → Successful extub next day",
    },
    {
        "label": "stage3",
        "filter": lambda d: d[d["extub_event_v2_next_day"] == 1].copy(),
        "outcome": "success_extub_v2_next_day",
        "title": "Stage 3: Extubation → Successful extub",
    },
]


# ── Helpers (copies of 08's logic) ─────────────────────────────────────

def _scale_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, info in VAR_DISPLAY.items():
        if col in out.columns and info["scale"] != 1:
            out[col] = out[col] / info["scale"]
    return out


def _nz_quartile_knots(s: pd.Series):
    nz = s[s != 0].dropna().to_numpy()
    if len(nz) < 100:
        return None
    return [round(float(k), 6) for k in np.percentile(nz, [25, 50, 75])]


def _build_specs_for_cohort(df_scaled: pd.DataFrame) -> list[dict]:
    """Build the 7 covariate specs, with cohort-specific RCS knots.

    Knots are computed from the NON-ZERO subset of the cohort's predictor
    distribution (mirrors 08's logic). Falls back to df=4 if a predictor
    has fewer than 100 non-zero observations in the cohort.
    """
    knots_by_var = {v: _nz_quartile_knots(df_scaled[v]) for v in _RCS_VARS}

    def _cr_term(v):
        k = knots_by_var[v]
        base = f"cr({v}, df=4)" if k is None else f"cr({v}, knots={k})"
        ind = HURDLE_INDICATORS.get(v)
        return f"{base}:{ind}" if ind else base

    sofa_rcs = (
        "{{outcome}} ~ "
        + " + ".join(_cr_term(v) for v in _RCS_VARS)
        + " + _prop_any + _fenteq_any + _midazeq_any + "
        + "age + _nth_day + C(sex_category) + C(icu_type) + cci_score + sofa_total"
    )
    return [
        {"label": "baseline",    "formula": BASELINE},
        {"label": "daydose",     "formula": DAYDOSE},
        {"label": "sofa",        "formula": SOFA},
        {"label": "clinical",    "formula": CLINICAL},
        {"label": "sofa_rcs",    "formula": sofa_rcs},
        {"label": "sofa_weight", "formula": SOFA_WEIGHT},
        {"label": "sofa_bmi",    "formula": SOFA_BMI},
    ]


def _fit_gee(formula: str, data: pd.DataFrame):
    m = smf.gee(formula=formula, groups="hospitalization_id",
                data=data, family=sm.families.Binomial())
    return m.fit(maxiter=100)


def _fit_logit(formula: str, data: pd.DataFrame):
    _names = [c for c in data.columns if c in formula]
    if "hospitalization_id" not in _names:
        _names.append("hospitalization_id")
    _d = data.dropna(subset=_names)
    m = smf.logit(formula=formula, data=_d)
    return m.fit(cov_type="cluster",
                 cov_kwds={"groups": _d["hospitalization_id"]},
                 maxiter=100)


def _build_reference_row(df_scaled: pd.DataFrame) -> dict:
    ref = df_scaled.median(numeric_only=True).to_dict()
    for col in df_scaled.select_dtypes(include=["object", "category"]).columns:
        ref[col] = df_scaled[col].mode().iloc[0]
    return ref


def _percentile_ref(df_full_cohort: pd.DataFrame) -> dict:
    """Build PERCENTILE_REF for the 9 forest predictors. Non-zero subset for
    hurdle-paired daytime predictors; full distribution for others."""
    out = {}
    for pred, _ in FOREST_PREDICTORS:
        if pred not in df_full_cohort.columns:
            continue
        vals = df_full_cohort[pred].dropna().to_numpy()
        if len(vals) == 0:
            continue
        if pred in HURDLE_INDICATORS:
            vals = vals[vals != 0]
            if len(vals) == 0:
                continue
        x10, x90 = np.percentile(vals, 10), np.percentile(vals, 90)
        scale = VAR_DISPLAY.get(pred, {}).get("scale", 1)
        out[pred] = {
            "x10_raw": x10, "x90_raw": x90,
            "x10_scaled": x10 / scale, "x90_scaled": x90 / scale,
            "delta_scaled": (x90 - x10) / scale,
            "subset": "non-zero" if pred in HURDLE_INDICATORS else "all",
        }
    return out


def _or_10_to_90(fit, predictor: str, percentile_ref: dict, ref_row: dict):
    info = percentile_ref.get(predictor)
    if info is None or info["x10_raw"] == info["x90_raw"]:
        return (np.nan, np.nan, np.nan)
    nd_x10 = pd.DataFrame([ref_row])
    nd_x90 = pd.DataFrame([ref_row])
    nd_x10[predictor] = info["x10_scaled"]
    nd_x90[predictor] = info["x90_scaled"]
    ind = HURDLE_INDICATORS.get(predictor)
    if ind is not None:
        nd_x10[ind] = 1
        nd_x90[ind] = 1
    try:
        di = fit.model.data.design_info
        X10 = np.asarray(dmatrix(di, nd_x10, return_type="matrix"))[0]
        X90 = np.asarray(dmatrix(di, nd_x90, return_type="matrix"))[0]
    except Exception:
        return (np.nan, np.nan, np.nan)
    beta = fit.params.values
    V = fit.cov_params().values
    contrast = X90 - X10
    if np.allclose(contrast, 0):
        return (np.nan, np.nan, np.nan)
    log_or = float(contrast @ beta)
    var_log_or = float(contrast @ V @ contrast)
    if var_log_or < 0:
        return (np.nan, np.nan, np.nan)
    se = np.sqrt(var_log_or)
    return (
        float(np.exp(log_or)),
        float(np.exp(log_or - 1.96 * se)),
        float(np.exp(log_or + 1.96 * se)),
    )


# ── Fit all stages ──────────────────────────────────────────────────────
all_fits = {}  # keyed by (stage_label, model_type, spec_label)
cohort_summaries = []
forest_rows = []

# Reference dataset for percentile calculations: use Stage 0 (full) cohort.
# This way all 4 stages report ORs against the same population-level percentile
# anchors (10th/90th percentile of the FULL IMV-day distribution, not per-stage).
_df_scaled_full = _scale_df(df_full)
PERCENTILE_REF = _percentile_ref(df_full)
REF_ROW = _build_reference_row(_df_scaled_full)
print()
print("PERCENTILE_REF (raw clinical units, anchored to full IMV-day distribution):")
for pred, info in PERCENTILE_REF.items():
    tag = f"  [{info['subset']}]" if info["subset"] == "non-zero" else ""
    print(f"  {pred:<24s}: x10={info['x10_raw']:>+8.3f}, x90={info['x90_raw']:>+8.3f}{tag}")
print()

for stage in STAGES:
    cohort = stage["filter"](df_full)
    cohort_scaled = _scale_df(cohort)
    n_rows = len(cohort)
    n_pat = cohort["hospitalization_id"].nunique()
    out_col = stage["outcome"]
    if out_col not in cohort.columns:
        print(f"  WARN {stage['label']}: outcome column '{out_col}' not in cohort, skipping")
        continue
    out_rate = cohort[out_col].mean()
    rows_per_pat = cohort.groupby("hospitalization_id").size()
    cohort_summaries.append({
        "stage": stage["label"],
        "title": stage["title"],
        "outcome": out_col,
        "n_rows": n_rows,
        "n_patients": n_pat,
        "outcome_positive_rate": float(out_rate) if pd.notna(out_rate) else np.nan,
        "rows_per_patient_median": float(rows_per_pat.median()),
        "rows_per_patient_p90": float(rows_per_pat.quantile(0.90)),
    })
    print(f"=== {stage['title']} ===")
    print(
        f"  cohort: n_rows={n_rows}, n_patients={n_pat}, "
        f"outcome_rate={out_rate:.3f}"
    )

    specs = _build_specs_for_cohort(cohort_scaled)
    # Drop outcome-NaN rows so the fit's data length matches its grouping arg.
    cohort_scaled_fit = cohort_scaled.dropna(subset=[out_col]).copy()
    cohort_scaled_fit[out_col] = cohort_scaled_fit[out_col].astype(int)

    for spec in specs:
        formula = spec["formula"].replace("{{outcome}}", out_col)
        for mt, fit_fn in [("gee", _fit_gee), ("logit", _fit_logit)]:
            try:
                result = fit_fn(formula, cohort_scaled_fit)
                all_fits[(stage["label"], mt, spec["label"])] = result
                print(f"  OK: {spec['label']} / {mt}")
                # extract forest cells
                for pred, _ in FOREST_PREDICTORS:
                    or_, lo, hi = _or_10_to_90(result, pred, PERCENTILE_REF, REF_ROW)
                    forest_rows.append({
                        "stage": stage["label"],
                        "outcome": out_col,
                        "model_type": mt,
                        "spec": spec["label"],
                        "predictor": pred,
                        "OR": or_, "OR_lo": lo, "OR_hi": hi,
                    })
            except Exception as e:
                print(f"  FAIL: {spec['label']} / {mt}: {e}")
    print()

forest_df = pd.DataFrame(forest_rows)
forest_csv = f"{OUT_DIR}/cascade_forest_data.csv"
forest_df.to_csv(forest_csv, index=False)
print(f"Saved {forest_csv} ({len(forest_df)} rows)")

cohort_summary_df = pd.DataFrame(cohort_summaries)
summary_csv = f"{OUT_DIR}/cascade_cohort_summary.csv"
cohort_summary_df.to_csv(summary_csv, index=False)
print(f"Saved {summary_csv}")
print()
print(cohort_summary_df.to_string(index=False))


# ── Forest plots: 1 per (stage, method) ───────────────────────────────
def plot_forest(rows_df, stage_label, model_type, title, predictors, percentile_ref, out_path):
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    n_specs = len(SPEC_ORDER)
    jitter = np.linspace(-0.20, 0.20, n_specs)
    ymin, ymax = -0.6, len(predictors) - 0.4
    for i, (pred, pred_label) in enumerate(predictors):
        y_base = len(predictors) - 1 - i
        for j, spec in enumerate(SPEC_ORDER):
            row = rows_df[
                (rows_df["stage"] == stage_label)
                & (rows_df["model_type"] == model_type)
                & (rows_df["spec"] == spec)
                & (rows_df["predictor"] == pred)
            ]
            if row.empty:
                continue
            r = row.iloc[0]
            if not (np.isfinite(r["OR"]) and np.isfinite(r["OR_lo"]) and np.isfinite(r["OR_hi"])):
                continue
            y = y_base + jitter[j]
            ax.errorbar(
                r["OR"], y,
                xerr=[[r["OR"] - r["OR_lo"]], [r["OR_hi"] - r["OR"]]],
                fmt="o", color=SPEC_COLORS[spec], markersize=4,
                capsize=2, elinewidth=1.0, label=spec if i == 0 else None,
            )
    ytick_labels = []
    ytick_pos = []
    for i, (pred, pred_label) in enumerate(predictors):
        y_base = len(predictors) - 1 - i
        info = percentile_ref.get(pred, {})
        x10 = info.get("x10_raw", np.nan)
        x90 = info.get("x90_raw", np.nan)
        ytick_labels.append(f"{pred_label}\nx10={x10:+.2f}, x90={x90:+.2f}")
        ytick_pos.append(y_base)
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)
    ax.set_ylim(ymin, ymax)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2.0)
    ax.axvline(1.0, color="dimgray", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xlabel(
        "Odds ratio (10th → 90th percentile shift, log scale, clipped to [0.5, 2.0])",
        fontsize=9,
    )
    ax.set_title(f"{title} — {SITE_NAME} ({model_type.upper()})", fontsize=11)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.4, zorder=0)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    if dedup:
        ax.legend(
            [h for h, _ in dedup], [l for _, l in dedup],
            loc="upper center", bbox_to_anchor=(0.5, 1.10),
            ncol=len(dedup), fontsize=8, frameon=False,
        )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


for stage in STAGES:
    for mt in ["gee", "logit"]:
        out_path = f"{OUT_DIR}/cascade_{stage['label']}_forest_{mt}.png"
        plot_forest(
            forest_df, stage["label"], mt, stage["title"],
            FOREST_PREDICTORS, PERCENTILE_REF, out_path,
        )
        print(f"Saved: {out_path}")


# ── Marginal effects: 1 per (stage, method) at sofa_rcs ───────────────
FOCAL_VARS = [
    [("_prop_day_mcg_kg_min",  "Mean Daytime Propofol Rate (mcg/kg/min)"),
     ("_fenteq_day_mcg_hr",    "Mean Daytime Fentanyl Eq Rate (mcg/hr)"),
     ("_midazeq_day_mg_hr",    "Mean Daytime Midazolam Eq Rate (mg/hr)")],
    [("prop_dif_mcg_kg_min",   "Day-to-Night Δ Propofol Rate (mcg/kg/min)"),
     ("fenteq_dif_mcg_hr",     "Day-to-Night Δ Fentanyl Eq Rate (mcg/hr)"),
     ("midazeq_dif_mg_hr",     "Day-to-Night Δ Midazolam Eq Rate (mg/hr)")],
]


def _ggplot_ax(ax):
    ax.set_facecolor("#ebebeb")
    ax.grid(color="white", linewidth=0.8, which="major", zorder=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="#4d4d4d", labelsize=8)
    ax.xaxis.label.set_color("#4d4d4d")
    ax.yaxis.label.set_color("#4d4d4d")


def _marginal_prediction(result, ref_row, focal_col, scaled_grid):
    new_data = pd.DataFrame([ref_row] * len(scaled_grid))
    new_data[focal_col] = scaled_grid
    ind = HURDLE_INDICATORS.get(focal_col)
    if ind is not None:
        new_data[ind] = 1
    pred = result.get_prediction(new_data)
    sf = pred.summary_frame()
    if "mean" in sf.columns:
        return np.asarray(sf["mean"]), np.asarray(sf["mean_ci_lower"]), np.asarray(sf["mean_ci_upper"])
    elif "predicted" in sf.columns:
        return np.asarray(sf["predicted"]), np.asarray(sf["ci_lower"]), np.asarray(sf["ci_upper"])
    raise ValueError(f"Unexpected summary_frame columns: {list(sf.columns)}")


def plot_marginal_effects(result, stage_label, model_type, title, cohort_full, out_path):
    df_scaled = _scale_df(cohort_full)
    ref_row = _build_reference_row(df_scaled)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    fig.patch.set_facecolor("white")
    panel_letters = ["A", "B", "C", "D", "E", "F"]
    pidx = 0
    for r in range(2):
        for c in range(3):
            focal, xlabel = FOCAL_VARS[r][c]
            ax = axes[r, c]
            _ggplot_ax(ax)
            raw = cohort_full[focal].dropna()
            if len(raw) == 0:
                pidx += 1
                continue
            q_lo, q_hi = np.percentile(raw, [2.5, 97.5])
            actual_grid = np.linspace(q_lo, q_hi, 50)
            scale = VAR_DISPLAY.get(focal, {}).get("scale", 1)
            scaled_grid = actual_grid / scale
            try:
                prob, ci_lo, ci_hi = _marginal_prediction(result, ref_row, focal, scaled_grid)
                ax.fill_between(actual_grid, ci_lo, ci_hi, color="#808080", alpha=0.35, zorder=2)
                ax.plot(actual_grid, prob, color="black", linewidth=1.5, zorder=3)
            except Exception as e:
                ax.text(0.5, 0.5, f"failed: {e}", transform=ax.transAxes, ha="center")
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel("Predicted Probability", fontsize=9)
            ax.set_ylim(0, 1)
            ax.text(-0.12, 1.08, panel_letters[pidx], transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="top", ha="left")
            pidx += 1
    fig.suptitle(f"{title}\n(sofa_rcs spec, {model_type.upper()})", fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)


for stage in STAGES:
    cohort = stage["filter"](df_full)
    for mt in ["gee", "logit"]:
        key = (stage["label"], mt, "sofa_rcs")
        if key not in all_fits:
            continue
        out_path = f"{OUT_DIR}/cascade_{stage['label']}_marginal_effects_{mt}_sofa_rcs.png"
        plot_marginal_effects(all_fits[key], stage["label"], mt, stage["title"], cohort, out_path)
        print(f"Saved: {out_path}")


# ── Per-stage model_comparison CSVs ────────────────────────────────────
def _pretty_label(v):
    if v in VAR_DISPLAY:
        return VAR_DISPLAY[v]["label"]
    return v


def _fmt_or(or_, lo, hi, p=None):
    if not (np.isfinite(or_) and np.isfinite(lo) and np.isfinite(hi)):
        return "—"
    star = ""
    if p is not None and np.isfinite(p):
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{or_:.2f} ({lo:.2f}, {hi:.2f}){star}"


def build_wide_table(stage_label, model_type, all_fits_dict):
    """Per-spec wide table: rows = predictors, cols = specs."""
    relevant = {
        sp: all_fits_dict[(stage_label, model_type, sp)]
        for sp in SPEC_ORDER
        if (stage_label, model_type, sp) in all_fits_dict and sp != "sofa_rcs"
    }
    if not relevant:
        return None
    all_vars = []
    for r in relevant.values():
        for v in r.params.index:
            if v not in all_vars and v != "Intercept":
                all_vars.append(v)
    cols = list(relevant.keys()) + ["N"]
    tbl = pd.DataFrame(index=all_vars + ["N"], columns=cols)
    for label, r in relevant.items():
        for v in all_vars:
            if v in r.params.index:
                or_ = float(np.exp(r.params[v]))
                lo = float(np.exp(r.conf_int().loc[v, 0]))
                hi = float(np.exp(r.conf_int().loc[v, 1]))
                p = float(r.pvalues[v]) if v in r.pvalues.index else np.nan
                tbl.loc[v, label] = _fmt_or(or_, lo, hi, p)
            else:
                tbl.loc[v, label] = "—"
        tbl.loc["N", label] = f"{int(r.nobs):,}"
    tbl = tbl.rename(index={v: _pretty_label(v) for v in all_vars})
    tbl.index.name = "Variable"
    return tbl


for stage in STAGES:
    for mt in ["gee", "logit"]:
        wide = build_wide_table(stage["label"], mt, all_fits)
        if wide is None:
            continue
        out_csv = f"{OUT_DIR}/cascade_model_comparison_{stage['label']}_{mt}.csv"
        wide.to_csv(out_csv)
        print(f"Saved {out_csv} ({len(wide)} rows x {wide.shape[1]} cols)")


# ── Cascade flow diagram ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 1.5 * len(cohort_summaries) + 1))
ax.set_facecolor("#fafafa")
n_stages = len(cohort_summaries)
y_positions = list(range(n_stages - 1, -1, -1))
palette = ["#5e3c99", "#1f77b4", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
labels = [
    f"{s['stage'].upper()}\n{s['title']}\nn_rows = {s['n_rows']:,}\noutcome rate = {s['outcome_positive_rate']:.2%}"
    for s in cohort_summaries
]
for i, (lab, y) in enumerate(zip(labels, y_positions)):
    color = palette[i % len(palette)]
    ax.barh(y, 1.0, color=color, alpha=0.30, edgecolor=color, linewidth=1.5, height=0.7)
    ax.text(0.05, y, lab, va="center", fontsize=9)
    if i < n_stages - 1:
        ax.annotate(
            "", xy=(0.5, y_positions[i + 1] + 0.4),
            xytext=(0.5, y - 0.4),
            arrowprops=dict(arrowstyle="->", color="dimgray", lw=1.0),
        )
ax.set_xlim(0, 1.05)
ax.set_ylim(-0.7, n_stages - 0.3)
ax.set_xticks([])
ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title(f"Liberation cascade — {SITE_NAME}", fontsize=12, pad=10)
fig.tight_layout()
diag_path = f"{OUT_DIR}/cascade_flow_diagram.png"
fig.savefig(diag_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved {diag_path}")

print()
print("Done.")
