# Analysis Plan: Epidemiology of Sedation in Mechanical Ventilation

Master tracker for all analytical definitions and their implementation status.

**Abstract:** Liao Z, Guleria S, Chhikara K, et al. *Diurnal and Institutional Variation in Sedation and Extubation in Mechanical Ventilation: A Multicenter Study of Six Hospital Systems.* ATS 2025.

**Main notebook:** `code/sedation_sbt.py` (marimo)

---

## 1. Cohort Definition

### Inclusion


| spec                                                         | status | code reference                                                                                                                                                                                                                                          |
| ------------------------------------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Adult ICU patients (any ICU type)                            | DONE   | `sedation_sbt.py` line 196-204: ADT filtered to `location_category = 'icu'`                                                                                                                                                                             |
| First IMV streak (via endotracheal tube) lasting >= 24 hours | DONE   | `cohort_id.sql`: gaps-and-islands on `device_category = 'imv'`, cumulative `_chg_imv` flags → `_streak_id`, duration computed as `date_diff('minute', _start_dttm, _end_dttm) / 60`, filtered to `_at_least_24h = 1 AND _on_imv = 1 AND _streak_id = 1` |
| Respiratory support preprocessed with waterfall backfill     | DONE   | `sedation_sbt.py` lines 236-279: `RespiratorySupport.from_file()` → `apply_outlier_handling()` → `.waterfall(bfill=True)` → cached to parquet                                                                                                           |


### Exclusion


| spec                                                                                   | status | code reference                                                                                                                                                                                          |
| -------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Tracheostomy patients — exclude all rows after tracheostomy placement                  | DONE   | `sbt.sql` line 158: `WHERE (tracheostomy = 0 OR _trach_1st = 1)`. `cohort_id.sql`: streak end time is `COALESCE(_trach_dttm, _next_start_dttm, _last_observed_dttm)`                                    |
| NMB exclusion. Agents: cisatracurium, vecuronium, rocuronium (all verified in mCIDE, med_group: paralytics). Pancuronium excluded — not in mCIDE. Original spec: patient-day level (>1h NMB on a given day). **Changed to hospitalization-level**: exclude entire hospitalization if patient ever received >1h NMB on any day. Rationale: NMB patients are a fundamentally different population (severe ARDS, deep sedation for ventilator synchrony). Patient-day code preserved as comment in `05_analytical_dataset.py` for reversion. | DONE | `01_cohort.py`: NMB loaded, duration via LEAD + SUM per patient-day, flagged days saved to `output/nmb_excluded.parquet`. `05_analytical_dataset.py`: ANTI JOIN on `hospitalization_id` only (hosp-level). |


### Encounter Stitching


| spec                                                           | status              | code reference                                                                                                                                                                                                    |
| -------------------------------------------------------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Stitch encounters within 12-hour gap via `stitch_encounters()` | COMPUTED_BUT_UNUSED | `sedation_sbt.py` lines 208-233: `stitch_encounters(hosp_w_icu_stays.df, adt_w_icu_stays.df, time_interval=12)` produces `hosp_stitched`, `adt_stitched`, `encounter_mapping` but cell returns nothing downstream |


---

## 2. Observation Window


| spec                                                              | status | code reference                                                                                                        |
| ----------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------------------------- |
| Hourly time grid from start to end of first qualifying IMV streak | DONE   | `sedation_sbt.py` lines 325-338: `generate_series(_start_hr, _end_hr, INTERVAL '1 hour')` from `cohort_imv_streaks_f` |
| Day shift: 7:00-19:00; Night shift: 19:00-7:00                    | DONE   | `sedation_sbt.py` lines 104-131: `add_day_shift_id()` — `CASE WHEN _hr >= 7 AND _hr < 19 THEN 'day' ELSE 'night'`     |
| `_nth_day` numbering: cumulative count of 7am boundaries          | DONE   | `add_day_shift_id()`: `_is_day_start` flag at `_hr = 7`, cumulative `SUM` → `_nth_day`                                |
| `_day_shift` label: e.g. `day1_day`, `day1_night`                 | DONE   | `add_day_shift_id()`: `'day' || _nth_day::INT::TEXT || '_' || _shift`                                                 |
| Shift-change grids at 7am and 7pm only (for covariate sampling)   | DONE   | `sedation_sbt.py` lines 348-351: `cohort_shift_change_grids = cohort_hrly_grids_f[_hr.isin([7, 19])]`                 |


---

## 3. Exposure Definition

### Sedation Drugs


| drug            | route        | preferred unit | included | status                   | code reference                                                                               |
| --------------- | ------------ | -------------- | -------- | ------------------------ | -------------------------------------------------------------------------------------------- |
| propofol        | continuous   | mg/min         | yes      | DONE                     | `sedation_sbt.py` line 1017-1018                                                             |
| propofol        | intermittent | mg             | yes      | DONE                     | `sedation_sbt.py` line 1082-1083                                                             |
| fentanyl        | continuous   | mcg/min        | yes      | DONE                     | `sedation_sbt.py` line 1018                                                                  |
| fentanyl        | intermittent | mcg            | yes      | DONE                     | `sedation_sbt.py` line 1083                                                                  |
| hydromorphone   | continuous   | mg/min         | yes      | DONE                     | `sedation_sbt.py` line 1018                                                                  |
| hydromorphone   | intermittent | mg             | yes      | DONE                     | `sedation_sbt.py` line 1083                                                                  |
| midazolam       | continuous   | mg/min         | yes      | DONE                     | `sedation_sbt.py` line 1018                                                                  |
| midazolam       | intermittent | mg             | yes      | DONE                     | `sedation_sbt.py` line 1083                                                                  |
| lorazepam       | continuous   | mg/min         | yes      | DONE                     | `sedation_sbt.py` line 1018                                                                  |
| lorazepam       | intermittent | mg             | yes      | DONE                     | `sedation_sbt.py` line 1083                                                                  |
| dexmedetomidine | continuous   | —              | no       | EXCLUDED (intentionally) | not loaded                                                                                   |
| ketamine        | continuous   | —              | no       | EXCLUDED (intentionally) | not loaded                                                                                   |
| morphine        | any          | —              | no       | EXCLUDED (intentionally) | not loaded; equivalency formula references morphine (10mg = 100mcg fentanyl) but not tracked |


### Dose Processing Pipeline


| spec                                                                                                                                                                    | status | code reference                                                                                      |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------------------------------------------------------------------------------- |
| MAR deduplication: prioritize by `mar_action_category` (or `mar_action_name` fallback) — deprioritize verify/not_given/stop, prefer non-zero doses, prefer larger doses | DONE   | `sedation_sbt.py` lines 134-181: `remove_meds_duplicates()` with `QUALIFY ROW_NUMBER()`             |
| Unit conversion via `clifpy.utils.unit_converter.convert_dose_units_by_med_category`                                                                                    | DONE   | `sedation_sbt.py` line 1022 (cont), line 1087 (intm)                                                |
| Outlier handling via `config/outlier_config.yaml`                                                                                                                       | DONE   | `sedation_sbt.py` line 1033 (cont), line 1091 (intm)                                                |
| Pivot to wide format with unit-aware column names (e.g. `propofol_mg_min_cont`)                                                                                         | DONE   | `sedation_sbt.py` lines 1039-1061 (cont), lines 1097-1119 (intm)                                    |
| Intermittent: zero out doses where `mar_action_category = 'not_given'`                                                                                                  | DONE   | `sedation_sbt.py` line 1106: `CASE WHEN mar_action_category = 'not_given' THEN 0 ELSE med_dose END` |


### Hourly Dose Calculation (Continuous)


| spec                                                                                             | status | code reference                                                     |
| ------------------------------------------------------------------------------------------------ | ------ | ------------------------------------------------------------------ |
| Forward-fill medication columns within each hospitalization using `LAST_VALUE(... IGNORE NULLS)` | DONE   | `cont_sed_dose_by_hr.sql` CTE `t1`                                 |
| Calculate duration (minutes) to next event: `LEAD(event_dttm) - event_dttm` in seconds / 60      | DONE   | `cont_sed_dose_by_hr.sql` CTE `t1`: `_duration`                    |
| Multiply dose rate × duration to get cumulative dose per interval                                | DONE   | `cont_sed_dose_by_hr.sql` CTE `t3`: `COLUMNS('_cont') * _duration` |
| Aggregate by hour: `SUM` grouped by `(hospitalization_id, _dh, _hr)`                             | DONE   | `cont_sed_dose_by_hr.sql` CTE `t4`                                 |


### Hourly Dose Calculation (Intermittent)


| spec                                                                                                 | status | code reference                    |
| ---------------------------------------------------------------------------------------------------- | ------ | --------------------------------- |
| Sum bolus doses by hour: `SUM(COALESCE(COLUMNS('_intm'), 0))` grouped by `(hospitalization_id, _dh)` | DONE   | `sedation_sbt.py` lines 1170-1182 |


### Equivalency Formulas


| formula                                                  | conversion factor                   | status      | code reference                      |
| -------------------------------------------------------- | ----------------------------------- | ----------- | ----------------------------------- |
| fenteq (mcg) = hydromorphone_mg × 50 + fentanyl_mcg | 2mg hydromorphone = 100mcg fentanyl | DONE        | `sedation_sbt.py` line 1204         |
| midazeq (mg) = lorazepam_mg × 2 + midazolam_mg      | 1mg midazolam = 0.5mg lorazepam     | DONE        | `sedation_sbt.py` line 1203         |
| morphine equivalency: 10mg morphine = 100mcg fentanyl    | —                                   | NOT_TRACKED | morphine not loaded as a medication |


### Day/Night Aggregation


| spec                                                                                                 | status | code reference                                                                                   |
| ---------------------------------------------------------------------------------------------------- | ------ | ------------------------------------------------------------------------------------------------ |
| Sum hourly doses by `(hospitalization_id, _nth_day, _shift)` for propofol, fenteq, midazeq | DONE   | `sedation_sbt.py` lines 1377-1405: `sed_dose_agg` → `sed_dose_daily` via pandas pivot            |
| Exposure variable: `diff = night - day` for each drug class                                          | DONE   | `sedation_sbt.py` lines 1450-1452: `prop_dif`, `fenteq_dif`, `midazeq_diff`      |
| Weight-adjusted propofol (mcg/kg/min × weight × time)                                                | FUTURE | Original plan specifies weight-adjusted; code uses absolute mg. Abstract reports in 100mg units. |


---

## 4. Covariates

### Currently Implemented


| covariate                           | measurement                                                                                                                                                                                          | status | code reference                               |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------------------------------------------- |
| Age                                 | `age_at_admission` from hospitalization table                                                                                                                                                        | DONE   | `sedation_sbt.py` line 1454                  |
| pH                                  | Arterial preferred; venous + 0.05 as fallback. ASOF LEFT JOIN on `lab_order_dttm <= event_dttm`, within 12 hours. Categorized: `ph_lt72`, `ph_72_73`, `ph_73_74`, `ph_74_745`, `ph_ge745`, `missing` | DONE   | `sedation_sbt.py` lines 670-699              |
| P/F ratio                           | `po2_arterial / fio2_set`. ASOF LEFT JOIN on resp + labs. Categorized: `pf_lt100`, `pf_100_200`, `pf_200_300`, `pf_ge300`, `missing`                                                                 | DONE   | `sedation_sbt.py` lines 724-752              |
| Norepinephrine equivalent (NEE)     | Formula: `NE + epi + phenylephrine/10 + dopamine/100 + vasopressin×2.5 + angiotensin×10` (all in mcg/kg/min except vasopressin in u/min). ASOF LEFT JOIN per vasopressor.                            | DONE   | `sedation_sbt.py` lines 862-936              |
| Covariates sampled at shift changes | pH, P/F, NEE measured at 7am and 7pm (`ph_level_7am/7pm`, `pf_level_7am/7pm`, `nee_7am/7pm`)                                                                                                         | DONE   | `sedation_sbt.py` lines 953-990              |
| Daytime sedation doses              | `_prop_day`, `_fenteq_day`, `_midazeq_day` included as regression covariates                                                                                                           | DONE   | `sedation_sbt.py` lines 1444-1449, 1616-1617 |

### Planned but Not Yet Implemented


| covariate                  | source table                                               | status |
| -------------------------- | ---------------------------------------------------------- | ------ |
| Sex                        | hospitalization or patient                                 | FUTURE |
| SOFA score                 | can use `clifpy.calculate_sofa2_daily()`                   | FUTURE |
| Surgical vs medical ICU    | ADT `location_category` or admission diagnosis             | FUTURE |
| Charlson comorbidity index | can use `clifpy.utils.cci`                                 | FUTURE |
| Sepsis flag                | requires operationalization (e.g. Sepsis-3 criteria)       | FUTURE |
| RASS (sedation depth)      | patient_assessments table (`assessment_category = 'rass'`) | FUTURE |
| GCS at intubation          | patient_assessments table                                  | FUTURE |
| ICU type / location        | ADT table                                                  | FUTURE |
| Position (prone/not prone) | position table                                             | FUTURE |

### Vasopressor Unit Conversion


| vasopressor    | preferred unit | status |
| -------------- | -------------- | ------ |
| norepinephrine | mcg/kg/min     | DONE   |
| epinephrine    | mcg/kg/min     | DONE   |
| phenylephrine  | mcg/kg/min     | DONE   |
| dopamine       | mcg/kg/min     | DONE   |
| vasopressin    | u/min          | DONE   |
| angiotensin    | mcg/kg/min     | DONE   |



---

## 5. Outcomes

### Primary Outcomes (Current Implementation)

#### Outcome 1: SBT Done (next day)


| spec                                                                                                                                          | status | code reference                  |
| --------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------------------------------- |
| SBT state: `mode_category IN ('pressure support/cpap') AND peep_set <= 8 AND pressure_support_set <= 8`, OR T-piece (regex: `t[\s_-]?piece`) | DONE   | `sbt.sql` lines 16-19           |
| Gaps-and-islands to identify contiguous SBT blocks (`_block_id` via cumsum of `_chg_sbt_state`)                                               | DONE   | `sbt.sql` lines 36-57           |
| SBT duration: block end = start of next block or last observed time; `_duration_mins = date_diff('minute', _start_dttm, _end_dttm)`           | DONE   | `sbt.sql` lines 94-101          |
| `sbt_done = 1` if block duration >= 30 minutes AND `_sbt_state = 1`                                                                           | DONE   | `sbt.sql` lines 125-128         |
| Aggregated to daily level: `COALESCE(MAX(sbt_done), 0)` per `(hospitalization_id, _nth_day)`                                                  | DONE   | `sedation_sbt.py` lines 553-567 |
| Modeled as next-day outcome: `sbt_done_next_day = LEAD(sbt_done) OVER w`                                                                      | DONE   | `sedation_sbt.py` line 1441     |


#### Outcome 2: Successful Extubation (next day)


| spec                                                                                                                                                      | status | code reference              |
| --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | --------------------------- |
| Intubation event: transition into IMV from any other state (including NULL → IMV via `IS DISTINCT FROM`)                                                  | DONE   | `sbt.sql` lines 21-23       |
| Extubation event: transition from IMV to any non-IMV state                                                                                                | DONE   | `sbt.sql` lines 25-28       |
| First extubation only: `_extub_cum = SUM(_extub) OVER w`, `_extub_1st = 1` when `_extub_cum = 1`                                                          | DONE   | `sbt.sql` lines 43-45       |
| Failed extubation: reintubation within 24 hours of first extubation (correlated subquery)                                                                 | DONE   | `sbt.sql` lines 59-67       |
| Withdrawal of life-sustaining treatment: first extubation AND `code_status_category != 'full'` AND `discharge_category IN ('hospice', 'expired')`         | DONE   | `sbt.sql` lines 138-142     |
| `_success_extub = 1` if `_extub_1st = 1 AND _withdrawl_lst = 0 AND _fail_extub = 0`                                                                       | DONE   | `sbt.sql` lines 143-147     |
| Death after extubation without reintubation: `_extub_1st = 1 AND _last_vitals_within_24h = 1 AND _fail_extub = 0 AND discharge IN ('hospice', 'expired')` | DONE   | `sbt.sql` lines 148-154     |
| Modeled as next-day outcome: `success_extub_next_day = LEAD(success_extub) OVER w`                                                                        | DONE   | `sedation_sbt.py` line 1442 |
| Post-tracheostomy rows excluded: `WHERE (tracheostomy = 0 OR _trach_1st = 1)`                                                                             | DONE   | `sbt.sql` line 158          |


### SBT Eligibility (Simplified for Abstract)

The original analysis plan specified detailed SBT eligibility criteria. These were simplified for the ATS abstract — only `sbt_done` is tracked, not `sbt_eligible`. Documented here for future reference.

**Per ABC trial (Lancet 2008; 371: 126-34) and JC/Snigdha/Vaishvik definition:**


| eligibility criterion                                                              | CLIF data source                                     | status                  |
| ---------------------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------- |
| SpO2 >= 88%                                                                        | vitals (`vital_category = 'spo2'`)                   | SIMPLIFIED_FOR_ABSTRACT |
| FiO2 <= 50%                                                                        | respiratory_support (`fio2_set`)                     | SIMPLIFIED_FOR_ABSTRACT |
| PEEP <= 8 cm H2O                                                                   | respiratory_support (`peep_set`)                     | SIMPLIFIED_FOR_ABSTRACT |
| No agitation: mean RASS >= -2 over 6-hour window                                   | patient_assessments (`assessment_category = 'rass'`) | SIMPLIFIED_FOR_ABSTRACT |
| Hemodynamic stability: NEE < 0.2 AND dobutamine < 0.5 mcg/kg/min AND milrinone = 0 | medication_admin_continuous                          | SIMPLIFIED_FOR_ABSTRACT |
| No tracheostomy                                                                    | respiratory_support (`tracheostomy`)                 | SIMPLIFIED_FOR_ABSTRACT |
| Patient on IMV in ICU for 6 cumulative hours between 10 PM and 6/8 AM              | ADT + respiratory_support                            | SIMPLIFIED_FOR_ABSTRACT |
| No evidence of myocardial ischemia in previous 24h                                 | not available in CLIF                                | NOT_APPLICABLE          |
| No evidence of increased ICP > 15                                                  | not available in CLIF                                | NOT_APPLICABLE          |


### SBT Success/Failure (Future — Original ABC Trial Criteria)

Intentionally pivoted to extubation success for the abstract. Documented here for potential future implementation.

**Per ABC trial, SBT failure (`sbt_fail = 1`) if any of:**


| failure criterion                             | CLIF data source                                          | status         |
| --------------------------------------------- | --------------------------------------------------------- | -------------- |
| Respiratory rate > 35 breaths/min             | vitals (`vital_category = 'respiratory_rate'`)            | FUTURE         |
| Respiratory rate < 8 breaths/min for >= 5 min | vitals (time-series check)                                | FUTURE         |
| SpO2 < 88% for >= 5 min                       | vitals (`vital_category = 'spo2'`)                        | FUTURE         |
| Abrupt mental status change (GCS <= 8)        | patient_assessments (`assessment_category = 'gcs_total'`) | FUTURE         |
| Acute cardiac arrhythmia                      | not directly available in CLIF                            | NOT_APPLICABLE |
| Tachycardia > 130 bpm                         | vitals (`vital_category = 'heart_rate'`)                  | FUTURE         |
| Bradycardia < 60 bpm                          | vitals (`vital_category = 'heart_rate'`)                  | FUTURE         |
| Tidal volume < 250 cc (pass criterion)        | respiratory_support (`tidal_volume_set`)                  | FUTURE         |


`**sbt_success = 1` if `sbt_done = 1 AND sbt_fail = 0`**

### Other Outcomes (Future)


| outcome              | status |
| -------------------- | ------ |
| ICU length of stay   | FUTURE |
| Ventilator-free days | FUTURE |
| Mortality            | FUTURE |


---

## 6. Statistical Analysis

### Current Implementation


| analysis                                                                                                                                                                                                                                                                       | status | code reference                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ | ------------------------------------------------------ |
| GEE: `sbt_done_next_day ~ prop_dif + fenteq_dif + midazeq_diff + _prop_day + _midazeq_day + _fenteq_day + ph_level_7am + ph_level_7pm + pf_level_7am + pf_level_7pm + nee_7am + nee_7pm + age`, groups = `hospitalization_id`, family = Binomial | DONE   | `sedation_sbt.py` lines 1611-1631                      |
| Logistic regression with clustered SEs: `success_extub_next_day ~ [same formula]`, clustered by `hospitalization_id`                                                                                                                                                           | DONE   | `sedation_sbt.py` lines 1634-1653                      |
| TableOne: overall (day 1 characteristics)                                                                                                                                                                                                                                      | DONE   | `sedation_sbt.py` lines 1556-1562                      |
| TableOne: by shift (day vs night comparison)                                                                                                                                                                                                                                   | DONE   | `sedation_sbt.py` lines 1573-1580                      |
| Day vs night t-tests (Welch's) on propofol, fenteq, midazeq                                                                                                                                                                                                          | DONE   | `sedation_sbt.py` lines 1242-1262                      |
| Pairwise Pearson correlation matrix of continuous variables                                                                                                                                                                                                                    | DONE   | `sedation_sbt.py` lines 1591-1608                      |
| Hourly sedation dose bar chart (reordered 7am-6am, with 7pm cutoff line)                                                                                                                                                                                                       | DONE   | `sedation_sbt.py` lines 1290-1366                      |
| Meta-analysis across sites: DerSimonian-Laird random-effects                                                                                                                                                                                                                   | DONE   | per ATS abstract; implemented in `meta_analysis.ipynb` |
| Analytical dataset filtered: `_nth_day > 0 AND sbt_done_next_day IS NOT NULL AND success_extub_next_day IS NOT NULL`                                                                                                                                                           | DONE   | `sedation_sbt.py` lines 1463-1472                      |


### Planned but Not Yet Implemented


| analysis                                                                                                                         | reference              | status |
| -------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------ |
| Restricted cubic splines for non-linear exposure-outcome relationships; remove when p for nonlinearity > 0.2                     | Seymour et al.         | FUTURE |
| Rate-based exposure: model average hourly rate change (day-to-night) instead of cumulative dose. Report 10th-to-90th percentile. | Seymour et al.         | FUTURE |
| Adjusted dose-response curves: probability of outcome (y-axis) vs day-to-night dose change, with 95% CI, adjusted for covariates | Seymour et al. Fig 2-4 | FUTURE |
| Latent class analysis                                                                                                            | Sedative Exposure Plan | FUTURE |


---

## 7. Prior Studies Reference


| study                      | design                                  | cohort                                | exposure                                                                          | outcomes                               | key method                                                                                                  |
| -------------------------- | --------------------------------------- | ------------------------------------- | --------------------------------------------------------------------------------- | -------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| Mehta et al. (Lancet 2008) | Secondary analysis of ABC trial (n=423) | MV > 48h + continuous opioid/sedative | Fentanyl-eq and midazolam-eq, night vs day (19:00-7:00 vs 7:00-19:00)             | SBT conducted, SBT success, extubation | GEE logistic regression; cumulative dose approach                                                           |
| Seymour et al.             | Nested cohort in ABC trial (n=140)      | MV > 12h                              | Hourly benzo and propofol doses, day (7am-11pm) vs night (11pm-7am), first 4 days | Delirium, coma, delayed liberation     | GEE; **rate-based** (avg hourly dose change); restricted cubic splines; 10th-90th percentile interpretation |
| Wongtangman et al.         | Retrospective cohort (n=102,204)        | MV >= 24h in 20 ICUs                  | Proportion of time in deep sedation (RASS -3 to -5) in first week                 | Loss of independent living             | Modified Poisson regression; mediation analysis (mobilization level)                                        |


### Equivalency Conversions Reference


| conversion               | factor                                | source       |
| ------------------------ | ------------------------------------- | ------------ |
| Morphine → Fentanyl      | 10 mg morphine = 100 mcg fentanyl     | Mehta et al. |
| Hydromorphone → Fentanyl | 2 mg hydromorphone = 100 mcg fentanyl | Mehta et al. |
| Lorazepam → Midazolam    | 0.5 mg lorazepam = 1 mg midazolam     | Mehta et al. |


---

## 8. Candidates for Future Expansion

Items for discussion with collaborators and mentors on which to prioritize for the full paper, now that the simplified abstract version is complete.

### A. Plan-vs-Code Gaps

Items specified in the analysis plan documents but not yet implemented.


| #   | item                                     | description                                                                                                                       | requires                                                                                                        | effort                 |
| --- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------- |
| A1  | SBT eligibility screening                | ABC trial criteria: SpO2>=88%, FiO2<=50%, PEEP<=8, RASS>=-2, NEE<0.2, no dobutamine>=5, no milrinone, no vasopressin, no ICP>15   | patient_assessments (RASS), vitals (SpO2), respiratory_support, additional vasopressors (dobutamine, milrinone) | HIGH                   |
| A2  | SBT success/failure detection            | ABC trial failure criteria: RR>35, RR<8 for 5min, SpO2<88% for 5min, GCS<=8, tachycardia>130, bradycardia<60, tidal volume>=250cc | vitals (RR, HR, SpO2), patient_assessments (GCS), respiratory_support (TV)                                      | HIGH                   |
| A3  | ~~NMB exclusion~~                        | ~~DONE. Patient-days with >1h NMB (cisatracurium, vecuronium, rocuronium) now excluded via ANTI JOIN in `cohort_merged_final`~~    | —                                                                                                               | —                      |
| A4  | Weight-adjusted propofol                 | Use mcg/kg/min x weight x time instead of absolute mg                                                                             | admission weight from vitals (already loaded)                                                                   | LOW                    |
| A5  | Morphine tracking                        | Add morphine to fentanyl equivalency (10mg morphine = 100mcg fentanyl)                                                            | load morphine from medication_admin_continuous/intermittent                                                     | LOW                    |
| A6  | Additional covariates                    | Sex, SOFA (`calculate_sofa2_daily`), surgical vs medical, Charlson comorbidity index (`clifpy.utils.cci`), sepsis                 | hospitalization, patient, labs, vitals, meds, hospital_diagnosis                                                | MEDIUM                 |
| A7  | RASS / sedation depth                    | Categorize as light (RASS >= -2) vs deep (RASS -3 to -5). Time spent in each category                                             | patient_assessments (`assessment_category = 'rass'`)                                                            | MEDIUM                 |
| A8  | GCS at intubation                        | GCS value at or near time of first intubation event                                                                               | patient_assessments (`assessment_category = 'gcs_total'`)                                                       | LOW                    |
| A9  | ICU type / location                      | Which ICU the patient was in (medical, surgical, neuro, etc.)                                                                     | ADT table (already loaded)                                                                                      | LOW                    |
| A10 | Position (prone/not prone)               | Prone positioning as variable of interest                                                                                         | position table                                                                                                  | LOW                    |
| A11 | Dobutamine/milrinone for SBT eligibility | Required for the full SBT eligibility check (A1)                                                                                  | medication_admin_continuous                                                                                     | LOW (if A1 is pursued) |


### B. Code-Level Gaps

Items in the code that are incomplete or unused.


| #   | item                       | description                                                                                                              | effort |
| --- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------ | ------ |
| B1  | Encounter stitching unused | `stitch_encounters()` called at lines 208-233 but stitched results not connected downstream. Decide: use it or remove it | LOW    |
| B2  | Intermittent MAR handling  | `not_given` zeroing only in the SQL pivot step; not systematically validated pre-pivot                                   | LOW    |
| B3  | Outlier config coverage    | Some medications may lack outlier ranges for converted units (post unit-conversion ranges)                               | LOW    |


### C. Analysis Methodology Expansion

From prior studies (Mehta, Seymour, Wongtangman) and the original analysis plan.


| #   | item                             | description                                                                                                                                        | reference              | effort |
| --- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------- | ------ |
| C1  | Rate-based exposure              | Model average hourly rate change (day-to-night) instead of cumulative dose. Report 10th-to-90th percentile effect. More interpretable for propofol | Seymour et al.         | MEDIUM |
| C2  | Restricted cubic splines         | Handle non-linear exposure-outcome relationships. Remove non-linear terms when p for nonlinearity > 0.2                                            | Seymour et al.         | MEDIUM |
| C3  | Adjusted dose-response curves    | Graphs: probability of outcome (y-axis) vs day-to-night dose change, adjusted for covariates, with 95% CI                                          | Seymour et al. Fig 2-4 | MEDIUM |
| C4  | Proportion of nocturnal increase | Descriptive stat: what % of patient-days had higher night vs day sedation, per drug class                                                          | Mehta et al. Table 2   | LOW    |
| C5  | Drug receipt by study day        | Table: # patients receiving each drug (benzo, propofol, fentanyl) by study day 1-5                                                                 | Seymour et al. Table 2 | LOW    |
| C6  | Nocturnal change breakdown       | Categorize each patient-day as increased, decreased, or no change in each drug class                                                               | Mehta et al.           | LOW    |


### D. Additional Analysis Suggestions


| #   | item                               | description                                                                                                 | effort |
| --- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------ |
| D1  | Subgroup analyses                  | By institution, medical vs surgical ICU, age group, severity strata (SOFA quartiles)                        | MEDIUM |
| D2  | Time-varying exposure              | More granular windows (e.g., 6-hour blocks) instead of 12h day/night                                        | MEDIUM |
| D3  | Dose escalation trajectories       | Track whether doses increase, decrease, or stabilize over successive days per patient                       | MEDIUM |
| D4  | Hour 24 and 72 characterization    | Original plan: characterize variables at hour 24 and 72 of MV specifically, remove expired/extubated by 72h | LOW    |
| D5  | Interaction terms                  | Test interactions between sedation class (opioid vs hypnotic vs benzo) and outcomes                         | LOW    |
| D6  | Competing risks                    | Tracheostomy, death, and withdrawal are competing events for extubation. Consider Fine-Gray models          | HIGH   |
| D7  | ICU LOS, vent-free days, mortality | Back-burner outcomes from original plan                                                                     | MEDIUM |
| D8  | Latent class analysis              | Identify distinct sedation practice phenotypes across institutions                                          | HIGH   |


---

## 9. Audit Remediation

Tracker for resolving findings from the peer review code audit (`docs/code_audit.md`, 2026-03-31). Each row links to the audit finding ID for full details.

**Status key:** `OPEN` = not started, `IN_PROGRESS` = actively being fixed, `FIXED` = code updated, `WONT_FIX` = accepted risk (with rationale), `DEFERRED` = postponed to future work

### Critical

| ID | finding | files | status | resolution notes |
|----|---------|-------|--------|------------------|
| C1 | T-piece regex has stray `1` — misses real T-piece devices | `03_outcomes.py:185`, `analysis_plan.md:185` | FIXED | Removed `1` from regex, added case-insensitive flag `'i'`. Spec updated in §5. |
| C2 | `mode_category` case mismatch with CLIF mCIDE (`'pressure support/cpap'` vs `'Pressure Support/CPAP'`) | `03_outcomes.py:184` | WONT_FIX | clifpy waterfall processing lowercases all mode_category values; no mismatch in practice |
| C3 | Column name mismatch: `_success_extub` vs `success_extub`; stale `n_hrs` reference | `03_outcomes.py:409`, `05_analytical_dataset.py:118-120` | FIXED | 05 now correctly references `o._success_extub`; `n_hrs` removed from SELECT |
| C4 | Vasopressor ASOF join has no staleness window (pH has 12h, vasopressors have none) | `04_covariates.py:360-405` | WONT_FIX | Intentional — ICU vasopressor infusions run for days; ASOF carry-forward reflects clinical reality |
| C5 | P/F ratio ASOF join has no staleness window (inconsistent with pH) | `04_covariates.py:206-233` | DEFERRED | Add to sensitivity analysis (similar to SOFA2 approach); not blocking for current analysis |

### High

| ID | finding | files | status | resolution notes |
|----|---------|-------|--------|------------------|
| H1 | Continuous dose forward-fill persists past infusion stop if stop event has non-zero dose | `02_exposure.py:189-195` | FIXED | Enforced `med_dose = 0` for `mar_action_category IN ('stop', 'not_given')` in continuous pivot step, matching intermittent pattern |
| H2 | FULL JOIN introduces out-of-window medication events | `02_exposure.py:212-225` | OPEN | Change to LEFT JOIN |
| H3 | NMB duration = inter-admin interval, not pharmacological effect | `01_cohort.py:368-414` | OPEN | Review whether NMB is mostly continuous (OK) or bolus (needs fix) |
| H4 | NULL PEEP/PS silently excludes valid SBTs | `03_outcomes.py:183-187` | OPEN | Add COALESCE or explicit NULL handling |
| H5 | Failed extubation check doesn't exclude tracheostomy transitions (diverges from spec) | `03_outcomes.py:235-242` | OPEN | Add `AND tracheostomy = 0` to EXISTS subquery |
| H6 | Table 1 describes Day 1 only; models use all patient-days | `06_table1.py:96-103` | OPEN | Add full-population Table 1 or report day distribution |
| H7 | T-test on hourly data violates independence | `07_analysis.py:104-121` | OPEN | Aggregate to patient-day level before t-test |

### Medium

| ID | finding | files | status | resolution notes |
|----|---------|-------|--------|------------------|
| M1 | Partial-day bias in dose measurement (Day 1 may have unequal shift hours) | `05_analytical_dataset.py:158` | OPEN | Normalize by hours observed or exclude partial days |
| M2 | `COALESCE(dose, 0)` conflates no-drug with not-observed | `05_analytical_dataset.py:123-128` | OPEN | Distinguish true zeros from structural missingness |
| M3 | Missing key covariates (sex, SOFA, RASS, Charlson, ICU type) | `05_analytical_dataset.py:133` | OPEN | Overlaps with §8 items A6, A7, A9. Prioritize sex + SOFA. |
| M4 | Inconsistent GEE vs logit modeling across outcomes | `07_analysis.py:230-274` | OPEN | Use GEE for both; see audit discussion |
| M5 | GEE working correlation structure unspecified (defaults to independence) | `07_analysis.py:238` | OPEN | Specify exchangeable; report QIC comparison |
| M6 | Categorical covariate reference levels uncontrolled | `07_analysis.py:234-236` | OPEN | Set explicit reference levels (e.g. `ph_73_74`, `pf_200_300`) |
| M7 | Venous pH +0.05 adjustment uncited | `04_covariates.py:151` | OPEN | Add citation; consider sensitivity analysis |
| M8 | Encounter stitching computed but unused | `01_cohort.py:106-111` | OPEN | Overlaps with §8 item B1. Decide: wire in or remove. |
| M9 | `_end_mode` computed same as `_start_mode` (identical ORDER BY) | `03_outcomes.py:264-265` | OPEN | Fix ORDER BY or remove unused column |

### Low

| ID | finding | files | status | resolution notes |
|----|---------|-------|--------|------------------|
| L1 | CONSORT flow incomplete (missing downstream exclusions) | `01_cohort.py:441-464` | OPEN | Add steps for null-age, day-0, null-outcome exclusions |
| L2 | Typo `_withdrawl_lst` throughout | `03_outcomes.py`, `05_analytical_dataset.py` | OPEN | Rename to `_withdrawal_lst` |
| L3 | Dexmedetomidine/ketamine excluded without justification | `analysis_plan.md:73-74` | OPEN | Document rationale in manuscript methods |
| L4 | Morphine not in fentanyl equivalency | `analysis_plan.md:75, 116` | OPEN | Overlaps with §8 item A5. Add morphine or report prevalence. |
| L5 | No sensitivity analyses on key thresholds (24h IMV, 60min NMB, 30min SBT) | — | OPEN | Add at least one SA (e.g. SBT >= 20min or >= 45min) |
| L6 | Hourly grid extends 1 hour past IMV streak end | `01_cohort.py:310` | OPEN | Remove `+ INTERVAL '1 hour'` or document rationale |

### Plan-vs-Code Concordance

| item | source | status | resolution notes |
|------|--------|--------|------------------|
| Bolus inclusion contradicts "continuous drip only" spec | `Sedative Exposure Plan.md:59` | OPEN | Document as intentional scope expansion or revert |
| Analysis design changed from hour-24/72 to daily aggregation | `Sedative Exposure Plan.md:81-84` | OPEN | Document when/why the pivot was made |
| `analysis_plan.md` line references stale (point to old `sedation_sbt.py`) | `analysis_plan.md` throughout | OPEN | Update all code references to `01_cohort.py`–`07_analysis.py` |
| Competing risks not addressed for extubation outcome | `code_audit.md` concordance | OPEN | Overlaps with §8 item D6. Discuss in limitations at minimum. |
