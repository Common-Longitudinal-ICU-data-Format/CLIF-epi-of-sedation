# Weight QC Audit — How to Interpret the Outputs

`make weight-audit SITE={mimic|ucmc|...}` runs `code/qc/weight_audit.py`, which characterizes how the per-kg weight column flows through the sedative dose pipeline and emits a hospitalization-level **drop list** plus a **federated-safe summary** that off-site collaborators can compare cross-site without inspecting data.

This file explains every output and how to use it for QC review and meta-analysis.

---

## 1. Why this audit exists

The propofol descriptive in this project applies weight **twice** with two *different* per-day weights:

- **Stage A — clifpy upstream**: in `02_exposure.py`, `convert_dose_units_by_med_category` does an ASOF backward join from each `admin_dttm` to the most recent `weight_kg` in the patient's vitals, with **no staleness limit and no admission-weight fallback**. Used to convert e.g. propofol `mcg/kg/min` → `mg/min`.

- **Stage B — descriptive downstream**: in `code/descriptive/_shared.py`, divides the `mg/hr` daily aggregate by `weight_kg_asof_day_start` (per-day ASOF *with* admission-weight fallback) to produce `prop_dif_mcg_kg_min`.

If the two stages used the same weight, the round trip would cancel and the result would equal the bedside pump's mcg/kg/min. Because the strategies differ, the pipeline injects noise plus bias in the direction of the admission weight. The audit also discovered a **silent clifpy bug** that turns missing-weight rows into wrong-by-a-factor-of-weight doses without raising any flag (see Section 4).

---

## 2. Output file map

Two output tiers — federated-safe (numeric summaries; no IDs) and PHI-internal (drop lists with IDs).

### Federated-safe — `output_to_share/{site}/qc/`

| file | purpose |
|---|---|
| `weight_qc_summary.csv` | One row per metric across sections (a)–(h). The exhaustive numeric trace of the audit. |
| `weight_qc_exclusions.csv` | One row per drop criterion: counts and percentages. The CONSORT-ready summary. |
| `weight_impact_comparison.csv` | One row per weight-handling strategy (status quo / fresh-only / patient-median / winsorized) showing how cohort-level `prop_dif_mcg_kg_min` summary stats change. |

### Federated-safe — `output_to_share/{site}/figures/`

| file | purpose |
|---|---|
| `weight_audit.png` | 9-panel summary figure mirroring the CSV. The "first thing to look at" for visual review. |

### PHI-internal — `output/{site}/qc/`

| file | purpose |
|---|---|
| `weight_qc_drop_list.parquet` | List of `hospitalization_id`s that should be excluded and the reason. Phase 2 of this work will read this and apply it as a CONSORT exclusion in `01_cohort.py`. |
| `weight_audit_examples.csv` | Same drop list, in CSV form, for terminal eyeballing. |

**Run-time provenance**: every CSV's first lines (commented `#`) record the site, clifpy commit, threshold values, and outlier clamp at run time. Cross-site meta-analysis uses these to verify like-with-like comparisons.

---

## 3. The 9-panel figure (`weight_audit.png`)

| panel | shows | how to read |
|---|---|---|
| **(a)** Raw weight distribution | Histogram of `weight_kg` after the [30, 300] outlier clamp, with vertical dashed lines at the clamp bounds and a title showing how many rows were *dropped* by the clamp. | A unimodal distribution near 70–90 kg is normal. Heavy spikes at exactly 70.0 or 80.0 kg suggest default-value entry (placeholder weights). The clamp-rejected count + raw values >300 in `weight_qc_summary.csv` flag lb↔kg unit-entry errors. |
| **(b)** Stage A staleness | Number of medication admins by ASOF staleness bucket (gap from `admin_dttm` to its joined `_weight_recorded_dttm`). | The `>7d` bucket is the worry zone: large counts here mean clifpy is using week-old weights to convert. |
| **(c)** Convert-failure count by drug | Horizontal bars of weighted-input admins where Stage A's ASOF returned NULL. | **The headline panel for the silent-bug finding.** Any non-zero bar is a count of admins whose `/kg` factor was silently dropped by clifpy (see Section 4). Almost always propofol. |
| **(d)** Stage B staleness | Same as (b) but per-day, with the admission-fallback rows excluded. | Useful to compare against (b) — Stage B should generally have *fresher* weights because it has the admission-weight fallback to fall back on. |
| **(e)** Cross-stage \|Δ\| | Histogram of \|Stage A weight − Stage B weight\| for admins where both stages return non-null weights. Red line at 5 kg. | A spike at 0 with a thin tail = the two stages mostly agree (good). Mass past the 5 kg line = patients whose weight was recharted mid-stay; the round-trip noise will be largest for them. |
| **(f1)** Within-stay weight range | Histogram of per-patient `(max − min)` weight in their full stay. Capped at 80 kg for visibility. | Most patients should be ≤10 kg. The 30+ kg tail is the candidate population for the (opt-in) range-rule exclusion. |
| **(f2)** Cohort-exclusion-vs-cutoff CDF | Both the range cutoff (kg, full stay) and the jump cutoff (kg, gap < jump-window) plotted as %-of-cohort excluded vs. cutoff value. | **The defensibility panel.** Look for the elbow — the cutoff above which loosening saves few patients but tightening costs many. That's the threshold to cite to reviewers. |
| **(f3)** Max consecutive jump | Histogram of per-patient max raw \|consecutive jump\| where the time gap is < `WEIGHT_QC_MAX_JUMP_HOURS` (default 24 h). Vertical red line at the active threshold. | Mass to the right of the red line = patients flagged for unit-entry-error suspicion. |
| **(h)** Impact analysis | Grouped bars: mean and p95 of \|prop_dif_mcg_kg_min\| under four weight-handling strategies (see Section 4 for what each one means). | If all four bars are within a few % of each other, current handling is robust. If they spread, the choice of strategy is materially affecting downstream results. |

---

## 4. The four weight-handling strategies in panel (h) / `weight_impact_comparison.csv`

Section (h) asks: *if we changed the weight-handling rule, how much would the propofol descriptive move?* It recomputes `prop_dif_mcg_kg_min` four ways for the same patient-days and compares cohort-level summaries (n, mean, p50, p95 of `|prop_dif_mcg_kg_min|`).

The four strategies:

### Strategy 1 — `status_quo`

What `code/descriptive/_shared.py` does today: divide `prop_dif_mg_hr` by `weight_kg_asof_day_start.clip(upper=300)`, where `weight_kg_asof_day_start` itself is `COALESCE(per_day_7am_ASOF_weight, admission_first_weight)`.

This is the baseline. Every other strategy is a sensitivity probe.

### Strategy 2 — `fresh_only`

Same as status quo, but **drop patient-days whose Stage B ASOF is more than 72 h stale, OR whose weight came from the admission fallback**. In code, days where `_weight_recorded_dttm` is NaT (fallback) or where `event_dttm − _weight_recorded_dttm > 72h` are removed entirely from the descriptive (their `prop_dif_mcg_kg_min` becomes NaN and falls out of summary stats).

**Tests:** "How much do *stale* and *fallback-rescued* weights inflate or deflate the descriptive?"

If `fresh_only` differs materially from `status_quo`, the admission-fallback population is meaningfully shifting the mean — staleness is a real concern.

### Strategy 3 — `patient_median`

Replace the per-day weight with each patient's **within-stay median** of all *raw* `weight_kg` recordings (one single value per patient, applied uniformly to every patient-day they have).

**Tests:** "What if we ignored within-stay weight changes entirely and used a single representative value per patient?"

If `patient_median` matches `status_quo`, then within-stay weight drift isn't moving the cohort summary much. If it differs, patients with mid-stay weight changes are getting noisier per-day denominators in status quo than they would under a single-value approach.

### Strategy 4 — `winsorized`

Replace the hard `[?, 300]` clamp with `[p1, p99]` of the cohort's empirical raw-weight distribution — same denominator as status quo, but with a tighter upper cap and a non-zero lower cap derived from data, not a literature constant.

**Tests:** "Does the descriptive depend on the choice of clamp threshold?"

If `winsorized` matches `status_quo`, the hard 300-kg cap isn't load-bearing — values near 300 are too rare to affect the summary. (For MIMIC the empirical p99 is 165 kg; the 300-kg cap is well above the data.)

### How to use the comparison

Three readings of the table:

- **All four within ~5% of each other (MIMIC's case)** → current weight handling is robust. The choice of weight-cleaning rule isn't a methodologically critical decision and doesn't need a sensitivity-analysis caveat in the manuscript.

- **`fresh_only` differs ≥5% from the rest** → admission-weight fallback is leaking bias into stale patient-days. Phase 2's pre-attached weight column with the same drop list will fix this; in the meantime the manuscript should report `fresh_only` alongside `status_quo` as a sensitivity row.

- **`patient_median` and `winsorized` both differ ≥5%** → there's something structural about the cohort's weight distribution (heavy tails, lots of outliers, lots of within-stay drift) that the current pipeline isn't handling cleanly. Investigate site-level data quality before trusting per-kg summaries.

---

## 5. The silent clifpy bug (read this carefully)


In clifpy 0.4.9 (and earlier), when:

- the **input** dose unit is weighted (e.g. `mcg/kg/min`),

- the **preferred** dose unit is unweighted (e.g. `mg/min`), and

- `weight_kg` is NULL after the per-admin ASOF (patient had no charted weight before that admin),

clifpy returns `_convert_status='success'` while *silently dropping* the `/kg` factor. Example from MIMIC:

| input | clifpy output | what the dose actually represents |
|---|---|---|
| `200 mcg/kg/min`, weight=NULL | `0.2 mg/min`, status=success | for an 80-kg patient, real delivered dose is ~16 mg/min — clifpy is off by ~80× |

**There is no log warning. No NaN. The unit string changes, the dose value drops by a factor of weight, and downstream code has no way to tell.**

The audit detects these via section (b) (`n_weighted_input_AND_null_weight`) and section (c) (`n_clifpy_silent_bugs`). The two values must match — sanity invariant #5.

**Why it doesn't matter at MIMIC**: the footprint is 43 admins out of 345,657 (0.012%) for propofol, and 0 for the other sedatives. It's small enough that downstream summaries don't move appreciably, but the bug is real.

**How Phase 2 will avoid it**: by pre-attaching a project-controlled `weight_kg` column on `cont_sed_deduped` *before* `convert_dose_units_by_med_category`, clifpy never has to do its own ASOF and never sees NULL. The handful of patients with truly zero charted weights are dropped at cohort time using the drop list this audit produces.

---

## 6. The drop list (`weight_qc_drop_list.parquet`)

Built from up to four exclusion criteria. Tunable thresholds via env vars (defaults shown):

| criterion | granularity | default | env var |
|---|---|---|---|
| zero raw weight in current admission | drop hospitalization | n/a | n/a |
| outlier raw weight | drop **row** (already in `outlier_config.yaml`) | outside `[30, 300]` kg | (in `outlier_config.yaml`) |
| implausible jump within window | drop hospitalization | jump > 20 kg AND time gap < 24 h | `WEIGHT_QC_MAX_JUMP_KG`, `WEIGHT_QC_MAX_JUMP_HOURS` |
| implausible total range (opt-in) | drop hospitalization | (max − min) > 30 kg | `WEIGHT_QC_MAX_RANGE_KG`, `WEIGHT_QC_RANGE_RULE_ON=1` |

**Important: the criteria are applied incrementally.** A hospitalization assigned `_drop_reason='zero_weight_in_admission'` is removed from the pool before the jump rule is evaluated, etc. So `weight_qc_exclusions.csv`'s per-criterion counts add up to the total — there's no double-counting.

---

## 7. How to use the outputs

### Local site review (you, post-run)

1. Open `weight_audit.png`. Scan panel (c) for any non-zero bars (silent-bug count). Scan panel (f2) for whether the elbow is at or near the configured threshold.

2. Open `weight_qc_summary.csv`. Confirm sanity invariants from the README — they're also asserted programmatically in the audit's verification step:

   - section (a) `n_hosp_with_weight + n_hosp_zero_weight = n_cohort_hosp`
   - section (b) `n_weighted_input_AND_null_weight` ≤ section (a) `n_hosp_zero_weight × admins-per-hosp`
   - section (d) `pct_nonnull_using_fallback ≥ section (b) pct_admins_null_weight`
   - section (g) `drop_total_unique_hosp = drop_zero_weight + drop_jump + drop_range`
   - section (c) `n_clifpy_silent_bugs == section (b) n_weighted_input_AND_null_weight`

3. Open `weight_impact_comparison.csv`. If the four strategies' means are within ≤5% of each other, current handling is robust at this site. If the spread is larger (≥5%), surface the table to the user as a methods decision point.

4. (Optional) Inspect `output/{site}/qc/weight_audit_examples.csv` to eyeball the IDs being dropped — useful to spot whether the drop list is concentrated in some unusual subgroup (very-short-LOS admits, palliative-care diagnoses, etc.).

### Cross-site meta-analysis (consortium-side)

`output_to_share/{site}/qc/weight_qc_exclusions.csv` is the artifact to pool. For each criterion, compare `n_hosp_dropped` and `pct_cohort` across sites. Big differences point to differences in charting practice, not data quality per se — and the comment header in each CSV records exactly which thresholds the site used.

`output_to_share/{site}/qc/weight_qc_summary.csv`'s section (a) percentiles (p1, p25, p50, p75, p99) give a federated-safe view of each site's weight distribution. Compare the round-number-clustering metrics (`pct_integer_kg`, `pct_mod5_kg`, `pct_mod10_kg`) across sites to detect whether a site is heavily using default placeholder weights.

### Phase 2 wiring (when ready)

- `01_cohort.py` reads `output/{site}/qc/weight_qc_drop_list.parquet` and excludes those `hospitalization_id`s as a new explicit cohort step. CONSORT artifacts (`consort.json`, the flowchart PNG, the inline markdown table) are atomically updated with one entry per criterion, counts sourced from `weight_qc_exclusions.csv`.

- `02_exposure.py` changes `preferred_units['propofol']` from `'mg/min'` to `'mcg/kg/min'` and pre-attaches a `weight_kg` column on `cont_sed_deduped` (per-day-start ASOF + admission fallback — both tiers, since the cohort-level drop has already removed patients with neither).

- `code/descriptive/_shared.py` stops dividing by weight at the descriptive layer (the data is already in /kg/min).

---

## 8. Configuring thresholds

All threshold env vars are picked up at run time and recorded in the CSV header for federated comparability. Common knob settings:

```bash
# Tighter — exclude any patient with a 15-kg jump in <24h
WEIGHT_QC_MAX_JUMP_KG=15 make weight-audit SITE=mimic

# Looser — only drop on really egregious jumps
WEIGHT_QC_MAX_JUMP_KG=30 make weight-audit SITE=mimic

# Turn the (opt-in) total-range rule on
WEIGHT_QC_RANGE_RULE_ON=1 make weight-audit SITE=mimic

# Re-run with a larger time window for the jump rule
WEIGHT_QC_MAX_JUMP_HOURS=48 make weight-audit SITE=mimic
```

After picking a threshold, rerun once to lock it in, then commit the CSVs / PNG / drop list together. The header line in every output records the active threshold values so downstream readers can verify the run's parameters without leaving the file.

---

## 9. Sanity invariants (auto-checked)

The audit's verification step asserts the following hold; if they don't, treat the audit run as suspect:

1. `a.n_hosp_with_weight + a.n_hosp_zero_weight == a.n_cohort_hosp`
2. `b.n_admins_null_weight` ≤ a.n_hosp_zero_weight × per-hosp admin counts
3. `d.pct_nonnull_using_fallback ≥ b.pct_admins_null_weight`
4. `g.drop_total_unique_hosp == g.drop_zero_weight + g.drop_jump + g.drop_range`
5. `c.n_clifpy_silent_bugs == b.n_weighted_input_AND_null_weight`

Run the snippet at the end of `code/qc/weight_audit.py`'s development log (or re-paste from the verification step in the audit's output) to verify these post-run.

---

## 10. Next steps

Phase 1 (this audit) is **observational only** — it characterizes the problem and produces a drop list, but does not modify the cohort or the dose pipeline. The intended sequence from here:

### Step 1 — Review Phase 1 outputs at MIMIC

(this just happened; numbers are above)

- Confirm thresholds in `weight_audit.png` panel (f2) align with the elbow in the CDF.

- Confirm the `weight_impact_comparison.csv` spread is small (≤5%) — if not, talk through which strategy to advance.

- Confirm the silent-bug count in panel (c) and the cross-stage divergence in panel (e) are within the expected scale.

### Step 2 — Run Phase 1 on UCMC for parity

```bash
make weight-audit SITE=ucmc
```

Compare each panel head-to-head against MIMIC. Differences in:

- charting frequency (Stage A staleness in panel (b)),

- weight-clamp-rejection rate in panel (a),

- round-number clustering (`pct_integer_kg` in `weight_qc_summary.csv`),

are expected; investigate if drop-list size diverges by more than ~10% from MIMIC's, since that suggests a site-specific data-quality issue worth flagging.

### Step 3 — Lock thresholds

Once both sites' CDFs (panel f2) have been inspected, decide whether to keep `WEIGHT_QC_MAX_JUMP_KG=20` (default) or shift. The decision should be reviewer-defensible: cite the elbow location and the percent-of-cohort excluded.

If the range rule (`WEIGHT_QC_RANGE_RULE_ON=1`) is desired, decide a threshold from the same CDF and lock it in.

### Step 4 — Phase 2 implementation (separate plan)

Phase 2 wires the audit's outputs into the upstream pipeline. Estimated 4–6 file edits across:

1. **`code/01_cohort.py`** — load `output/{site}/qc/weight_qc_drop_list.parquet`, filter `cohort_hosp_ids` to exclude listed hospitalizations, emit one CONSORT exclusion entry per criterion.

2. **`output_to_share/{site}/consort.json`** + flowchart PNG + inline markdown table — atomic refresh with the new exclusions (per the project's [CONSORT visuals required] rule).

3. **`code/02_exposure.py`** —

   - change `preferred_units['propofol']` from `'mg/min'` to `'mcg/kg/min'` (eliminates Stage A weight involvement for the dominant /kg-charted form),

   - pre-attach a `weight_kg` column on `cont_sed_deduped` using the same per-day-7am-ASOF + admission-fallback logic as `weight_kg_asof_day_start` (skips clifpy's no-fallback per-admin ASOF).

4. **`code/descriptive/_shared.py`** — stop dividing by weight at the descriptive layer; `prop_dif_mcg_kg_min` becomes a direct read of the now-already-/kg dose column from `analytical_dataset.parquet`. Update `DIFF_COLS` accordingly.

5. **`Makefile`** — chain `weight-audit` ahead of `run` so the drop list is current when the cohort is rebuilt.

6. **Re-run the entire pipeline** end-to-end on both sites and verify:

   - `weight_qc_summary.csv` section (c) `n_clifpy_silent_bugs` drops to 0 (no NULL weights reach clifpy now),

   - cohort-level CONSORT entries match the audit's federated `weight_qc_exclusions.csv`,

   - propofol-related model coefficients in `08_models.py`'s output don't change direction (a sign change would indicate the bias was load-bearing, which would be its own story to investigate).

### Step 5 — Federated rollout

Before sites other than MIMIC/UCMC run the pipeline, change the clifpy spec in `pyproject.toml` from `file:///Users/wliao0504/code/clif/pyCLIF` (local-only) to a tagged GitHub release: `git+https://github.com/Common-Longitudinal-ICU-data-Format/pyCLIF.git@<tag>`. Other sites cannot install from a sibling local clone.

### Step 6 (longer-term) — Upstream the clifpy bug fix

The silent `_convert_status='success'` bug in `clifpy/utils/unit_converter.py:875-905` is real and worth filing as a clifpy issue + PR. Suggested fix: when `_weighted = 1` (input weighted) and `weight_kg IS NULL`, emit `_convert_status='cannot convert weighted input without weight'` instead of falling through to the success branch. The current behavior was likely a side-effect of focusing the status check only on the *preferred* unit's weighting.

The Phase 2 override (pre-attached weight column) makes this fix non-blocking for *this* project — but other clifpy users in the consortium will hit the same silent bug if they leave clifpy to do its own ASOF.
