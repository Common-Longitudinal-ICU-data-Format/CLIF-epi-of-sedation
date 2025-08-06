# CLIF Epidemiology of Sedation

## CLIF VERSION 

2.0.0

## Objective

The primary objective of this project is to describe the epidemiology of sedation in mechanically ventilated ICU patients across the CLIF consortium. This analysis examines sedative medication usage patterns and investigates whether there is diurnal variation in sedation practices, specifically exploring if higher sedation doses are administered during night shifts compared to day shifts for patients with similar clinical characteristics. The study analyzes sedative doses over 24 and 72-hour windows from initial intubation and explores associations with patient demographics, illness severity (SOFA scores), and time of day.

## Required CLIF tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields.

The following tables are required:
1. **patient**: `patient_id`, `race_category`, `ethnicity_category`, `sex_category`, `death_dttm`
2. **hospitalization**: `patient_id`, `hospitalization_id`, `admission_dttm`, `discharge_dttm`, `age_at_admission`
3. **adt**: `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category` 
   - Used to identify ICU stays and create unique ICU stay identifiers
4. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = 'heart_rate', 'resp_rate', 'sbp', 'dbp', 'map', 'spo2', 'weight_kg', 'height_cm'
5. **labs**: `hospitalization_id`, `lab_collect_dttm`, `lab_category`, `lab_value_numeric`
   - `lab_category` = 'creatinine', 'platelet_count', 'po2_arterial', 'bilirubin_total'
6. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_category`, `med_dose`, `med_dose_unit`
   - Sedatives: "propofol", "midazolam", "fentanyl", "dexmedetomidine", "lorazepam", "hydromorphone", "ketamine"
   - Vasopressors: "norepinephrine", "epinephrine", "phenylephrine", "vasopressin", "dopamine", "angiotensin", "dobutamine", "milrinone"
7. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_category`, `mode_category`, `fio2_set`, `peep_set`, `lpm_set`, `tracheostomy`, `resp_rate_set`, `tidal_volume_set`, `resp_rate_obs`
   - Used to identify mechanical ventilation periods and calculate P/F ratios
8. **patient_assessments**: `hospitalization_id`, `recorded_dttm`, `assessment_category`, `numerical_value`
   - `assessment_category` = 'gcs_total', 'rass', 'RASS'
9. **crrt_therapy**: `hospitalization_id`, `recorded_dttm` (optional table for SOFA renal scoring)

## Cohort identification

The cohort consists of adult patients admitted to an ICU with continuous mechanical ventilation for 24 hours or more via endotracheal tube.

Inclusion criteria:
- On continuous invasive mechanical ventilation (device_category = 'IMV') via endotracheal tube for ≥24 hours
- Age ≥ 18 years at admission
- Admitted to an ICU (identified through ADT location_category = 'icu')

Exclusion criteria:
- Patients with tracheostomy (using tracheostomy column in respiratory_support table)
- Re-intubations within 72 hours of first intubation

## Expected Results

The analysis produces the following outputs in the [`output/final`](output/README.md) directory:
- Table 1 descriptive statistics stratified by:
  - Demographics (age, sex, race, ethnicity)  
  - Time windows (hour 24 and hour 72 of mechanical ventilation)
  - Shift times (day shift vs night shift) to explore diurnal variation
- Enumeration of medication dosing units to ensure coverage
- Regression analyses examining:
  - Associations between sedative doses and patient characteristics
  - Day vs night shift differences in sedation dosing
  - Relationships with illness severity (SOFA scores)
- SOFA score calculations using most recent values within specified time windows

## Configuration

1. Navigate to the `config/` directory
2. Rename `config_template.json` to `config.json`
3. Update the `config.json` with site-specific settings including:
   - Site name
   - Data table paths
   - Timezone information

## Environment setup and project execution

The environment setup code is provided in the `run_project.sh` file for macOS/Linux and `run_project.bat` for Windows.

**For macOS/Linux:**

1. Make the script executable: 
```bash
chmod +x run_project.sh
```

2. Run the script:
```bash
./run_project.sh
```

3. Restart your IDE to load the new virtual environment and select the `Python (sedation)` kernel in the Jupyter notebook.

**For Windows:**

1. Run the script in the command prompt:
```bat
run_project.bat
```

## Manual Setup and Execution

If you prefer to run the analysis manually, follow these steps:

1. **Setup Python Environment**
   - Create a virtual environment: `python3 -m venv .sedation`
   - Activate the environment:
     - Windows: `.sedation\Scripts\activate`
     - macOS/Linux: `source .sedation/bin/activate`
   - Install dependencies: 
     ```bash
     pip install -r requirements.txt
     pip install jupyter ipykernel
     ```
   - Register kernel: `python -m ipykernel install --user --name=.sedation --display-name="Python (sedation)"`
   - Restart your IDE to load the new virtual environment and select the `Python (sedation)` kernel in the Jupyter notebook.

2. **Run Analysis**
   Run the following files from the code directory:
   1. Run the [01_cohort_id.ipynb](code/01_cohort_id.ipynb) notebook to identify the study cohort and calculate sedative doses
   2. Upload results from [output/final](output/final/) to the project shared folder
