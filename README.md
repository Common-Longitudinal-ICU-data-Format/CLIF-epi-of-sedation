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

## Run the project

### Configuration

1. Navigate to the `config/` directory

2. Rename `config_template.json` to `config.json`

3. Update the `config.json` with site-specific settings including:
   - Site name
   - Data table paths
   - Timezone information

4. The analysis uses `config/outlier_config.yaml` for outlier detection and handling on vitals, labs, respiratory support parameters, and medications. This configuration file is provided and should work across sites, but can be customized if needed.

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package manager that simplifies dependency management.

1. Install uv (if not already installed):

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Sync dependencies and set up the environment:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   # macOS/Linux
   source .venv/bin/activate

   # Windows (Command Prompt)
   .venv\Scripts\activate.bat

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

   **Note for Windows PowerShell users**: If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. Open your IDE, select the `.venv` Python interpreter, and open `code/sedation_sbt.ipynb` to run the analysis interactively.

5. Upload results from [output/final](output/final/) to the project shared folder

### Option 2: Using traditional Python virtual environment (Manual setup)

1. Create a virtual environment:

   ```bash
   # macOS/Linux
   python3 -m venv .venv

   # Windows
   python -m venv .venv
   ```

2. Activate the environment:

   ```bash
   # macOS/Linux
   source .venv/bin/activate

   # Windows (Command Prompt)
   .venv\Scripts\activate.bat

   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1
   ```

   **Note for Windows PowerShell users**: If you get an execution policy error, run:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   pip install jupyter ipykernel
   ```

4. Register Jupyter kernel:

   ```bash
   python -m ipykernel install --user --name=.venv --display-name="Python (sedation)"
   ```

5. Open your IDE, select the `.venv` Python interpreter, and open `code/sedation_sbt.ipynb` to run the analysis interactively.

6. Upload results from [output/final](output/final/) to the project shared folder

### Option 3: Using automated setup scripts (not robustly tested)

The easiest way to set up the project is using the provided setup scripts that handle all environment configuration automatically.

**For macOS/Linux:**
```bash
./run_project.sh
```

**For Windows (Command Prompt or PowerShell):**
```cmd
run_project.bat
```

The scripts will:

- Create a `.venv` virtual environment

- Install all required dependencies

- Register the Jupyter kernel

- Validate your configuration

After the script completes, open your IDE (VS Code, PyCharm, etc.), select the `.venv` Python interpreter, and open `code/sedation_sbt.ipynb` to run the analysis interactively.
