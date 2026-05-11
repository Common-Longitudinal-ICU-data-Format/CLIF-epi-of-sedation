## Configuration

1. Rename  `config_template.json` to `config.json`.
2. Update the `config.json` with site-specific settings. You can add or remove attributes based on project requirements.

Note: the `.gitignore` file in this directory ensures that the information in the config file is not pushed to github remote repository.

## Config keys

| Key | Example | Notes |
|---|---|---|
| `site_name` | `"MIMIC"` | Drives the `output/{site_name}/` and `output_to_share/{site_name}/` subdirectories. |
| `data_directory` | `"/path/to/clif-tables"` | Absolute path to the CLIF table parquet/CSV files. |
| `filetype` | `"parquet"` | One of `parquet`, `csv`, `fst`. Matches the layout in `data_directory`. |
| `timezone` | `"US/Central"` | Site-local IANA timezone. Used only for display/shift derivation; all on-disk timestamps are stored in UTC (see `docs/timezone_audit.md`). |
| `id_name` | `"hospitalization_id"` | Encounter ID column expected on every table. |
| `reintub_window_hrs` | `48` | Re-intubation lookback window for the "successful extubation" outcome. |

## Skipping the expensive waterfall step on reruns

The cohort-building step in [`code/01_cohort.py`](../code/01_cohort.py) calls `clifpy.RespiratorySupport.waterfall(bfill=True)` to forward-fill respiratory-support parameters within continuous IMV episodes. This is the single most expensive operation in the pipeline (often tens of minutes to over an hour depending on site size), and its output — `output/{site_name}/cohort_resp_processed_bf.parquet` — is deterministic given the same input `respiratory_support` table and cohort definition.

You can avoid recomputing it on subsequent runs.

**How to skip:** place a previously generated `cohort_resp_processed_bf.parquet` at:

```
output/{site_name}/cohort_resp_processed_bf.parquet
```

…before running `make run` (or the cohort step directly). `code/01_cohort.py` checks for this file (`code/01_cohort.py:441`) and skips the `waterfall(bfill=True)` call when it exists. No config key is needed — the file's presence is the signal.

**When NOT to reuse a cached parquet:**

- The underlying `respiratory_support` table changed (new ETL, additional encounters, fixed outlier values, etc.).

- The cohort definition in `code/01_cohort.py` changed (different inclusion criteria, NMB exclusion logic, encounter stitching).

- `clifpy` was upgraded and the waterfall implementation may have changed.

- The vendored DuckDB outlier handler (`code/_outlier_handler.py`) or `config/outlier_config.yaml` changed.

When in doubt, delete the parquet and let it regenerate.

**Force a re-run** even if the parquet exists: set `RERUN_WATERFALL = True` at [`code/01_cohort.py:24`](../code/01_cohort.py). This is an advanced override kept as a script-level constant rather than a config key on purpose — flipping it should be a deliberate action, not part of routine configuration.
