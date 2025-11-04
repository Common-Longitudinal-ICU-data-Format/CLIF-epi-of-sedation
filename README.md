# CLIF Epidemiology of Sedation

## CLIF VERSION 

2.1.0

## Objective

This project investigates diurnal variation in sedation practices by comparing sedative doses administered during day shifts (7am-7pm) versus night shifts (7pm-7am). The study aggregates sedation exposure by day and shift throughout each patient's mechanical ventilation course and explores associations between day-night sedation differences and subsequent Spontaneous Breathing Trial (SBT) completion and successful extubation, while controlling for respiratory function (pH, P/F ratio), vasopressor support, and patient age.

## Required CLIF tables and fields

Please refer to the online [CLIF data dictionary](https://clif-consortium.github.io/website/data-dictionary.html), [ETL tools](https://github.com/clif-consortium/CLIF/tree/main/etl-to-clif-resources), and [specific table contacts](https://github.com/clif-consortium/CLIF?tab=readme-ov-file#relational-clif) for more information on constructing the required tables and fields.

The following tables are required:

1. **patient**: `patient_id`

2. **hospitalization**: `patient_id`, `hospitalization_id`, `discharge_dttm`, `discharge_category`, `age_at_admission`

3. **adt**: `hospitalization_id`, `in_dttm`, `out_dttm`, `location_category`

4. **vitals**: `hospitalization_id`, `recorded_dttm`, `vital_category`, `vital_value`
   - `vital_category` = 'weight_kg' (used for weight-based dose conversions)

5. **labs**: `hospitalization_id`, `lab_order_dttm`, `lab_result_dttm`, `lab_category`, `lab_value_numeric`
   - `lab_category` = 'ph_arterial', 'ph_venous', 'po2_arterial' (for P/F ratio calculation)

6. **medication_admin_continuous**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_name`, `mar_action_category`
   - Sedatives: `med_category` = "propofol", "midazolam", "fentanyl", "lorazepam", "hydromorphone"
   - Vasopressors: `med_category` = "norepinephrine", "epinephrine", "phenylephrine", "dopamine", "vasopressin", "angiotensin"

7. **medication_admin_intermittent**: `hospitalization_id`, `admin_dttm`, `med_name`, `med_category`, `med_dose`, `med_dose_unit`, `mar_action_name`, `mar_action_category`
   - Sedatives: "propofol", "midazolam", "fentanyl", "lorazepam", "hydromorphone"

8. **respiratory_support**: `hospitalization_id`, `recorded_dttm`, `device_name`, `device_category`, `mode_name`, `mode_category`, `fio2_set`, `peep_set`, `pressure_support_set`, `tracheostomy`, `resp_rate_set`, `tidal_volume_set`, `peak_inspiratory_pressure_set`
   - Used to identify mechanical ventilation periods, calculate P/F ratios, and identify SBT events

9. **code_status**: `patient_id`, `start_dttm`, `code_status_category`
   - Used to identify withdrawal of life-sustaining treatment


## Expected Results

The analysis produces the following outputs in the [`output/final`](output/README.md) directory:

- **Table 1 (Day 1 Overall)**: Descriptive statistics for the first day of mechanical ventilation, including sedation doses, pH levels, P/F ratios, vasopressor support (norepinephrine equivalents), and SBT outcomes

- **Table 1 (Day 1 by Shift)**: Descriptive statistics stratified by day shift (7am-7pm) versus night shift (7pm-7am), with statistical tests comparing sedation dosing patterns between shifts

- **Sedation by Hour of Day Visualization**: Three-panel bar chart showing mean hourly doses of propofol, fentanyl equivalents, and midazolam equivalents across the 24-hour cycle

- **Regression Analyses**:
  - GEE (Generalized Estimating Equations) examining associations between day-night sedation dose differences and next-day SBT completion, controlling for baseline sedation, pH levels, P/F ratios, vasopressor support, and age
  - Logistic regression examining associations between day-night sedation dose differences and next-day successful extubation (defined as extubation without reintubation within 24 hours and without withdrawal of life-sustaining treatment)

## Configuration

1. Navigate to the `config/` directory

2. Rename `config_template.json` to `config.json`

3. Update the `config.json` with site-specific settings including:
   - Site name
   - Data table paths
   - Timezone information

4. The analysis uses `config/outlier_config.yaml` for outlier detection and handling on vitals, labs, respiratory support parameters, and medications. This configuration file is provided and should work across sites, but can be customized if needed.

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
   1. Run the [sedation_sbt.ipynb](code/sedation_sbt.ipynb) notebook to identify the study cohort, characterize sedation patterns, and analyze Spontaneous Breathing Trial (SBT) outcomes.
   2. Upload results from [output/final](output/final/) to the project shared folder
