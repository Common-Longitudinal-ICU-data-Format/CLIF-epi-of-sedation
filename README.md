# CLIF Epidemiology of Sedation

## CLIF VERSION

2.1.0

## Objective

Multi-site analysis of diurnal variation in sedation practices in mechanically ventilated ICU patients. Aggregates sedation exposure by day (7am–7pm) and night (7pm–7am) shifts throughout each patient's first qualifying IMV course, and estimates associations between night-minus-day dose differences and next-day **SBT delivery** and **successful extubation**, adjusting for severity (SOFA), comorbidity (Charlson), vasopressor support, age, sex, and ICU type. 

## Quickstart

1. prerequisites: you have `mar_action_category`  mapped in `medication_admin_intermittent` (having only `mar_action_name` in `medication_admin_continuous`  is sufficient although `mar_action_category` would of course still be preferred).
2. Install `[uv](https://docs.astral.sh/uv/)` if not already:
  - macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
  - Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
3. rename `config/config_template.json` to `config/config.json` and fill in `site_name`, `data_directory`, `filetype`, `timezone`
  1. if you did not make significant updates to your `respiratory_support` table since last run in Novemember and you still have a copy of the waterfall-processed table from that run (default location should be `output/intermediate/resp_processed.parquet`), you can point the `path_to_waterfall_processed_resp_table` config towards it to bypass the waterfall step.
  2. likewise if you have a relatively large cohort you can set `enable_v2_outcomes: false` to skip some computationally intensive sensitivity analyses.
4. run `make run` from the command line
5. Upload `output_to_share/<site_name>/` to the project's Box folder (include `logs/clifpy_all.log` + `clifpy_errors.log` if crashed)



## Required CLIF tables and fields

Site CLIF parquets must comply with the [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html) (all `_category` columns lowercase per spec). For sites whose loader leaves mixed case, the pipeline now normalizes at load (B4 audit fix), but the underlying parquets should still follow the spec.

1. **patient**: `patient_id`, `sex_category`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `discharge_category`, `age_at_admission`
3. **adt**: `hospitalization_id`, `hospital_id`, `in_dttm`, `out_dttm`, `location_category`, `location_type`
  - `location_category` = 'icu' (cohort entry)
4. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
  - `vital_category` = 'weight_kg' (used for weight-based dose conversions AND for the in-cohort weight-QC step in `01_cohort.py`)
5. **labs**: `hospitalization_id`, `lab_order_dttm`, `lab_result_dttm`, `lab_category`, `lab_value_numeric`
  - `lab_category` = 'ph_arterial', 'ph_venous', 'po2_arterial' (for P/F ratio + pH covariates)
6. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_name`, `mar_action_category` (preferred; see contract below)
  - Sedatives: `med_category` ∈ {propofol, midazolam, fentanyl, lorazepam, hydromorphone}
  - Vasopressors (for NEE + ever-pressor): `med_category` ∈ {norepinephrine, epinephrine, phenylephrine, dopamine, vasopressin, angiotensin}
  - **NMB (for cohort exclusion — required)**: `med_category` ∈ {cisatracurium, vecuronium, rocuronium}. Hospitalizations with >1h of any NMB on any patient-day are excluded entirely (B3 cohort step).
  - `**mar_action_category` contract**: optional for continuous. If absent, the pipeline falls back to `mar_action_name` regex (B5 audit fix) — FFILL semantics tolerate this. If present (recommended), it is the preferred path.
7. **medication_admin_intermittent**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_name`, `**mar_action_category` (REQUIRED)**
  - Sedatives: `med_category` ∈ {propofol, midazolam, fentanyl, lorazepam, hydromorphone}
  - `**mar_action_category` contract**: REQUIRED for intermittent. Each bolus is a discrete event and the 'not_given' zeroing is critical for accurate dose accounting; regex on free-text `mar_action_name` is too lossy. Pipeline raises `RuntimeError` and ERROR-logs to `clifpy_errors.log` if absent.
8. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_name`, `device_category`, `mode_name`, `mode_category`, `fio2_set`, `peep_set`, `pressure_support_set`, `tracheostomy`, `resp_rate_set`, `tidal_volume_set`, `peak_inspiratory_pressure_set`
  - Drives IMV-streak identification, SBT detection, and P/F ratio.
9. **code_status**: `patient_id`, `start_dttm`, `code_status_category`
  - Used to classify withdrawal of life-sustaining treatment vs successful extubation.
10. **hospital_diagnosis** (optional): `hospitalization_id`, `diagnosis_code`, `diagnosis_code_format` — feeds Charlson + Elixhauser scores via clifpy utilities.

## Expected outputs

All outputs are site-scoped under one of two directories (per the federation contract):

```
output/{site}/                       — PHI-tier intermediate parquets (NEVER share)
  cohort_meta_by_id_imvday.parquet
  seddose_by_id_imvhr.parquet
  outcomes_by_id_imvday.parquet
  model_input_by_id_imvday.parquet
  sofa_by_id_imvday.parquet
  ...
  logs/clifpy_all.log                — every INFO+ line from every script
  logs/clifpy_errors.log             — only WARNING+ lines (upload BOTH on crash)
  qc/                                — weight-QC diagnostic outputs (run separately)

output_to_share/{site}/              — federated outputs (safe to ship)
  sedation_report.pdf                — compiled multi-page PDF
  consort_inclusion.{json,png}       — CONSORT flow
  models/                            — Table 1 long-format CSVs, model coefficients,
                                       forest plots, marginal-effects PNGs
  descriptive/                       — per-figure CSVs + PNGs
  qc/                                — weight QC summary + figure
```

### What to upload

After `make run` completes, upload the entire `output_to_share/{site}/` folder to the project's Box folder. The federated agg pipeline (`code/agg/`) reads only this directory and never touches `output/{site}/`.

If `make run` crashes at any step, also include both log files:

- `output_to_share/{site}/logs/clifpy_all.log`
- `output_to_share/{site}/logs/clifpy_errors.log`

Both files are produced per-script per-subprocess (pyCLIF logging integration guide rule 1), so even a mid-pipeline crash captures the failing script's trace.

## Pipeline shape

Single-pass orchestration after the B3 refactor. No more 2-pass weight-audit dance.

```
01_cohort.py → 02_exposure.py → 03_outcomes.py → 04_covariates.py
            → 05_modeling_dataset.py → 06_table1.py → 08_models.py
            → code/descriptive/*.py
```

`09_report.py` (PDF compilation) is intentionally shelved from `make run` — invoke `make report SITE=...` explicitly when the bundled PDF is wanted. Same opt-in pattern as `08b_models_cascade.py`.

Per-script roles:


| Script                   | Reads                                                                                             | Writes                                                                                                                                    | Headline operations                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| `01_cohort.py`           | clifpy: adt, hospitalization, respiratory_support, vitals, medication_admin_continuous (NMB only) | `cohort_meta_by_id_imvhr.parquet`, `cohort_meta_by_id_imvday.parquet`, `cohort_resp_processed_bf.parquet`, `consort_inclusion.{json,png}` | First IMV streak ≥24h per hospitalization; weight-QC drop (in-cohort); NMB exclusion; CONSORT |
| `02_exposure.py`         | clifpy: vitals (weight), medication_admin_continuous/intermittent                                 | `seddose_by_id_imvhr.parquet`, `seddose_by_id_imvday.parquet`                                                                             | Hourly sedation rates → fentanyl/midazolam equivalencies → day/night per-shift averages       |
| `03_outcomes.py`         | `cohort_resp_processed_bf.parquet`, clifpy: hospitalization, code_status, patient                 | `outcomes_by_id_imvday.parquet`                                                                                                           | SBT detection (multiday primary + 6 sensitivity variants), extubation classification          |
| `04_covariates.py`       | clifpy: labs, vitals, medication_admin_continuous                                                 | `sofa_by_id_imvday.parquet`, `covariates_*.parquet`, `weight_by_id_imvday.parquet`                                                        | pH, P/F, NEE at shift-changes; daily SOFA via `_sofa.py`; CCI + Elixhauser; ASE sepsis flag   |
| `05_modeling_dataset.py` | every per-day parquet above + Patient.sex                                                         | `model_input_by_id_imvday.parquet`                                                                                                        | LEFT-join everything onto the canonical registry; LEAD for next-day outcomes                  |
| `06_table1.py`           | `model_input_by_id_imvday.parquet`, `cohort_meta_by_id.parquet`                                   | `table1_{continuous,categorical,histograms}.csv`, `cohort_stats.csv`                                                                      | Federation-friendly long-format Table 1                                                       |
| `08_models.py`           | `model_input_by_id_imvday.parquet`                                                                | `models_coeffs.csv`, `forest_*.png`, `marginal_effects_*.png`                                                                             | GEE + cluster-robust logit; 5 nested specs × 9 outcome/method configs; forest plots; RCS      |
| `code/descriptive/*.py`  | `model_input_by_id_imvday.parquet`, `seddose_by_id_imvhr.parquet`                                 | per-figure PNG + CSV under `descriptive/`                                                                                                 | Diurnal-dose figures (hour-of-stay, 6-group dose-pattern, night-day diff distributions)       |
| `09_report.py`           | every shareable CSV/PNG above                                                                     | `sedation_report.pdf`                                                                                                                     | Multi-page PDF compilation                                                                    |


`08b_models_cascade.py` is intentionally shelved from `make run` (4-stage liberation cascade deferred for this round). Run on demand via `make cascade SITE=...`.

## Configuration

1. Copy `config/config_template.json` → `config/<site>_config.json`. Required fields:
  - `site_name` — e.g. `"mimic"`, `"ucmc"`, `"your-site-id"`. Lowercase by convention.
  - `data_directory` — path to the directory containing your CLIF parquets.
  - `filetype` — usually `"parquet"`.
  - `timezone` — e.g. `"US/Central"`.
  - `reintub_window_hrs` — reintubation classification window (48 per the published ABC-trial / Esteban / Thille literature; 24 also defensible).
2. **Cache controls (optional config keys, default off / null):**

  | Config key                               | Default | Effect when set                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  | ---------------------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `rerun_waterfall`                        | `false` | `true` → force waterfall recompute (re-invalidates `cohort_resp_processed_bf.parquet`, the slowest single step)                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
  | `rerun_sofa_24h`                         | `false` | `true` → force SOFA recompute (re-invalidates `sofa_first_24h.parquet`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
  | `rerun_ase`                              | `false` | `true` → force ASE recompute (re-invalidates `covariates_ase.parquet`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
  | `path_to_waterfall_processed_resp_table` | `null`  | Set to an absolute path of a pre-waterfall'd `respiratory_support` parquet. When the file exists, the project loads from it (filtered to your cohort via Polars predicate pushdown) and skips the internal waterfall entirely. Both whole-CLIF-system tables and cohort-scoped tables are accepted.                                                                                                                                                                                                                                                                                     |
  | `enable_v2_outcomes`                     | `true`  | Set to `false` to skip the v2 sensitivity outcome family (`success_extub_v2`, `sbt_done_v2`, `_trach_v2`). Saves ~7 min per run at typical site scale, ~110 min at 200K-cohort scale (the pandas state-machine in `03_outcomes.py::count_intubations_v2` is the most expensive non-cached step). Manuscript primaries (`success_extub_next_day`, `sbt_done_multiday`) are unaffected. Recommended for sites with ≥50K cohort hospitalizations. v2-suffix output columns become constant zero; `08_models.py` skips v2 outcome fits and records `SKIPPED_V2` in `model_fit_summary.csv`. |

   To force a one-shot rebuild without editing config, delete the cache file directly: `rm output/<site>/cohort_resp_processed_bf.parquet && make run SITE=<site>`.
3. **Other env-var knobs (still env vars — different ergonomic profile):**

  | Env var                    | Default | Effect when set to `1`                                                                                                                                                                                                                                  |
  | -------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
  | `WEIGHT_QC_MAX_JUMP_KG`    | 20      | Weight-QC jump threshold (kg)                                                                                                                                                                                                                           |
  | `WEIGHT_QC_MAX_JUMP_HOURS` | 24      | Weight-QC jump window (hours)                                                                                                                                                                                                                           |
  | `WEIGHT_QC_MAX_RANGE_KG`   | 30      | Weight-QC range threshold (kg)                                                                                                                                                                                                                          |
  | `WEIGHT_QC_RANGE_RULE_ON`  | 0       | Enable the range rule (off by default)                                                                                                                                                                                                                  |
  | `SEDDOSE_CLAMP`            | 1       | Per-hour clinical-ceiling clamp on sedation rates (M1). Set to `0` to disable (pass-through). `seddose_by_id_imvhr_raw.parquet` is always written alongside the canonical clamp-aware `seddose_by_id_imvhr.parquet` so you can diff without re-running. |
  | `ANONYMIZE_SITES`          | 0       | (agg only) Relabel sites as "Site A"/"Site B"/… in cross-site outputs                                                                                                                                                                                   |

   **Compare clamped vs unclamped without rerunning the pipeline:** `seddose_by_id_imvhr_raw.parquet` (always written) is the pre-clamp version of `seddose_by_id_imvhr.parquet`. Diff with `duckdb -c "FROM read_parquet('output/{site}/seddose_by_id_imvhr.parquet') c JOIN read_parquet('output/{site}/seddose_by_id_imvhr_raw.parquet') r USING (hospitalization_id, event_dttm) WHERE c.prop_mcg_kg_min_total <> r.prop_mcg_kg_min_total SELECT COUNT(*)"`. **Compare downstream models/figures end-to-end:** run the pipeline twice — once with default `SEDDOSE_CLAMP=1`, once with `SEDDOSE_CLAMP=0` — and manually preserve the `output/{site}/` and `output_to_share/{site}/` directories between runs.
4. **Outlier config**: `config/outlier_config.yaml` carries numeric range validation per CLIF table (weight 30–300 kg, propofol 0–200 mcg/kg/min, etc.). Shared across sites; customize only if your CLIF parquets have a known data-entry artifact.

## Running

### Setup (uv recommended)

`make run` invokes `uv sync` automatically, so the only one-time setup is having `uv` installed — see the [Quickstart](#quickstart) above for the install one-liner. Run `uv sync` manually only when iterating on marimo notebooks or ad-hoc scripts outside `make`.

### Run the pipeline

```bash
make run                     # uses current config/config.json
make run SITE=mimic          # swap to MIMIC config first
make run SITE=ucmc           # swap to UCMC config first
make run SITE=<your-site>    # swap to your <site>_config.json
```

Other targets:

```bash
make tables       # fast refresh of Table 1 + descriptive (skips 01, 03, 08; no PDF)
make table1       # even faster: just 04 + 05 + 06
make report       # PDF compilation from cached CSVs/PNGs (no compute)
make descriptive  # run all code/descriptive/*.py (no PDF)
make cascade      # run 08b_models_cascade.py (shelved; only if reviewing the 4-stage)
make weight-diagnostic  # federated weight-availability audit CSVs/PNG
make trach-funnel       # federated trach-bucket diagnostic (only if exit_mechanism='tracheostomy' looks empty)
make agg          # Phase-2 cross-site pooling (coordinator-side; reads output_to_share/<site>/)
```

`make report` is the only target that runs `09_report.py`. The main `make run` pipeline ends at the descriptive scripts; the PDF is an explicit opt-in (rationale: presentation layer over already-written CSVs/PNGs; ~30-page matplotlib render is wasteful when iterating on upstream stages).

The `SITE=` flag works on every target (via the shared `_switch` prerequisite).

### Runtime expectations (anchor measurements on MIMIC, 2026-05-11)

Reference numbers on the MIMIC site (~85k hospitalizations pre-cohort filter, ~13k post-cohort). Use as a sanity check; other sites scale roughly with cohort size.

**Warm-cache `make run`** (resp_processed_bf, sofa_24h, ase cached):


| Script                   | Wall-clock          | Notes                                                                                    |
| ------------------------ | ------------------- | ---------------------------------------------------------------------------------------- |
| `01_cohort.py`           | ~7 s                | Reuses cached `cohort_resp_processed_bf.parquet`; the in-cohort weight-QC step adds <1 s |
| `02_exposure.py`         | ~30 s               | Includes M1 clamp + raw-sibling parquet write                                            |
| `03_outcomes.py`         | ~70 s               | Gap-and-islands SBT detection over 9 M resp rows is the bottleneck                       |
| `04_covariates.py`       | ~15 s               | SOFA + ASE both from cache                                                               |
| `05_modeling_dataset.py` | ~5 s                | Consolidated LEFT-JOIN onto registry                                                     |
| `06_table1.py`           | ~5 s                |                                                                                          |
| `08_models.py`           | ~80 s               | GEE fits dominate; ~50 fits across nested specs × outcomes × methods                     |
| `code/descriptive/*.py`  | ~20 s               | 7 scripts                                                                                |
| `09_report.py`           | ~9 s                | PDF compilation                                                                          |
| **Total**                | **~238 s (≈4 min)** | Peak RSS ~16 GB (08_models.py)                                                           |


**Cold-cache `make run`** (all three config `rerun_*` flags set to `true`) — measured 2026-05-11 on MIMIC:


| Script                   | Wall-clock               | Notes                                                                                          |
| ------------------------ | ------------------------ | ---------------------------------------------------------------------------------------------- |
| `01_cohort.py`           | **~192 s (3m12s)**       | Waterfall recompute over 296,497 mode_name_id blocks (pandas ffill/bfill) is the dominant cost |
| `02_exposure.py`         | ~6 s                     |                                                                                                |
| `03_outcomes.py`         | ~59 s                    |                                                                                                |
| `04_covariates.py`       | ~34 s                    | Cold SOFA + ASE recompute (faster than expected on MIMIC)                                      |
| `05_modeling_dataset.py` | ~2 s                     |                                                                                                |
| `06_table1.py`           | ~1 s                     |                                                                                                |
| `08_models.py`           | ~122 s                   | GEE fits — variance with warm is run-to-run noise on this stage                                |
| `code/descriptive/*.py`  | ~6 s                     | 7 scripts in series                                                                            |
| `09_report.py`           | ~14 s                    | PDF compilation                                                                                |
| **Total**                | **~436 s (≈7 min 16 s)** | Peak RSS ~18.9 GB (cold-path materialization in waterfall)                                     |


**Cold tax = +198 s** (+83% over warm), concentrated in `01_cohort.py` (waterfall) and `04_covariates.py` (SOFA/ASE). Every other script is largely cache-independent because they consume already-processed parquets, not raw CLIF tables.

Re-runs after a successful first run automatically benefit from the cached `cohort_resp_processed_bf.parquet`, `sofa_first_24h.parquet`, and `covariates_ase.parquet`. Only force re-invalidation by flipping the per-site `rerun`_* config keys to `true` if upstream logic has changed (e.g., a clifpy version bump or a CLIF spec update).

### What sites should upload

After a successful `make run`:

- `output_to_share/{site}/` — everything in this directory (PDF + federated CSVs/PNGs).

If `make run` crashed:

- `output_to_share/{site}/` (whatever was produced before the crash).
- `output_to_share/{site}/logs/clifpy_all.log` and `output_to_share/{site}/logs/clifpy_errors.log`.

The two log files together let the coordinator diagnose the failure from the Box upload without needing PHI access.

## Cross-site federation (coordinator)

`make agg` pools every site's `output_to_share/<site>/` into `output_to_agg/`. Reads aggregate CSVs only — never PHI. Outputs include pooled Table 1, cross-site cohort stats, cross-site descriptive figures, and DerSimonian–Laird random-effects meta-analysis of model coefficients. Set `ANONYMIZE_SITES=1` before `make agg` to relabel sites as "Site A"/"Site B"/… in figures.

Adding a site post-hoc is zero work at the coordinator beyond dropping `output_to_share/<new-site>/` into the directory and re-running `make agg`.

## Troubleshooting

**Pipeline crashes at `02_exposure.py` with a SQL error about `mar_action_category`.**
Your `medication_admin_intermittent` parquet is missing `mar_action_category`. The column is required per the B5 contract (bolus dose accounting depends on the structured `'not_given'` zeroing). Re-run your CLIF ETL to include it.

**Cohort is empty / zero IMV streaks.**
Most likely cause: a `_category` column is mixed-case (e.g., `'IMV'`, `'Pressure Support/CPAP'`). The pipeline now normalizes at load (B4 audit fix); a fresh ERROR-log line in `clifpy_errors.log` will name the cause. Double-check `device_category`, `mode_category`, `location_category` in the parquets directly.

`**04_covariates.py` runs out of memory.**
The SOFA / ASE caches are usually the bottleneck. On a clean machine, let the first run populate them; on subsequent runs they're skipped. If you need to invalidate them after a code fix, set `rerun_sofa_24h: true` and/or `rerun_ase: true` in the per-site config (or `rm output/<site>/sofa_first_24h.parquet output/<site>/covariates_ase.parquet`) and rerun.

**Site delivers fewer than expected cohort hospitalizations.**
Check the CONSORT JSON at `output_to_share/{site}/consort_inclusion.json` for per-step exclusion counts. Weight-QC drops (B3) appear as their own CONSORT step now and should be small (typically <2% at properly-curated sites).

## Architecture documents

- `vc_proj_patterns.md` (project root) — general CLIF project conventions.
- `vc_sa_patterns.md` (project root) — sensitivity-analysis patterns.
- `pyCLIF/docs/logging_integration_guide.md` — clifpy logging contract.
- `pyCLIF/docs/duckdb_perf_guide.md` — DuckDB performance / materialization rules used throughout this codebase.
- `docs/audit_tracker.md` — running tracker for the 2026-05 pre-conference audit (audit + fixes).
- `docs/analysis_plan.md` — analytical definitions + audit remediation log (peer-review 2026-03-31).
- `docs/outcomes_specs.md` — source-of-truth for SBT detection + extubation classification flags.
- `docs/ats2026_outline.md` — oral presentation script for the ATS 2026 talk.

## QC dashboard (optional)

`make qc` launches a per-patient interactive Plotly Dash trajectory viewer at [http://localhost:8050](http://localhost:8050). Pick a site + a `hospitalization_id` and inspect a 5-panel timeline (sedatives, pressors, assessments, resp, vitals) with clinical event overlays. Loads only the selected patient's data — full-cohort memory is never materialized.