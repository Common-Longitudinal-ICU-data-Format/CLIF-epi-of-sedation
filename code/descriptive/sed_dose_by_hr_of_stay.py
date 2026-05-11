"""Per-site hour-of-stay aggregate CSV for federated cross-site pooling.

Computes the 168-row aggregate that powers
`code/agg/sed_dose_by_hr_of_day_cross_site.py` — extracted to the
per-site side so the cross-site script can consume aggregated CSVs
only (federation rule, see `.dev/CLAUDE.md`).

Per-site computation: DuckDB query joins
`output/{SITE}/seddose_by_id_imvhr.parquet` to
`output/{SITE}/cohort_meta_by_id_imvday.parquet` on
`(hospitalization_id, _nth_day)`. The registry join inherits the
canonical post-stitch / post-weight-QC cohort definition (Phase 1
deliverable). Restricts to `_nth_day >= 1` so the first_partial
intubation-day rows (which would shift `hour_of_stay` negative) are
excluded.

`hour_of_stay` runs 0..167 where hour 0 = `_nth_day=1` 7am (the
patient's first full ICU day's morning) and hour 167 = `_nth_day=7`
6am. Mapping: `hour_of_stay = (_nth_day - 1) * 24 + ((_hr - 7 + 24) % 24)`.

Output: `output_to_share/{SITE}/descriptive/sed_dose_by_hr_of_stay.csv`.
168 rows, one per `hour_of_stay`. Columns:

  - `hour_of_stay` — 0..167
  - `propofol_mcg_kg_min_mean`, `fenteq_mcg_mean`, `midazeq_mg_mean` —
    per-hour mean rate across all on-IMV patient-hours (zero-filled).
  - `n_on_drug_{propofol,fenteq,midazeq}` — patient-hour counts where
    that drug's rate > 0.
  - `n_on_drug_any` — patient-hours with ANY of the three drugs running.
  - `n_imv` — patient-hours on IMV at that hour-of-stay.

Federation-safe: every row is a group-level aggregate; no IDs.

Usage:
    uv run python code/descriptive/sed_dose_by_hr_of_stay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.descriptive.sed_dose_by_hr_of_stay")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import SITE_NAME, TABLES_DIR, ensure_dirs, save_csv  # noqa: E402


def _compute() -> pd.DataFrame:
    """Run the per-site DuckDB aggregation. See module docstring."""
    seddose_path = Path(f"output/{SITE_NAME}/seddose_by_id_imvhr.parquet")
    meta_path = Path(f"output/{SITE_NAME}/cohort_meta_by_id_imvday.parquet")
    if not seddose_path.exists() or not meta_path.exists():
        logger.info(
            f"Missing inputs: {seddose_path} or {meta_path} — "
            "rerun upstream pipeline (02_exposure.py + 01_cohort.py)."
        )
        return pd.DataFrame()

    # Per-hour-of-stay aggregation (168 hour bins). The `_nth_day BETWEEN 1
    # AND 7` cap already excludes first_partial (`_nth_day=0` per the
    # registry's 7am-crossing convention in `code/_utils.py:425`); the M2
    # `_is_full_24h_day` filter is intentionally NOT applied here because
    # dropping last_partial rows causes patients to drop out in 24h-boundary
    # batches instead of trickling out at their actual extubation hour,
    # producing bumpy trajectories. Per-DAY scripts (e.g.
    # `dose_pattern_6group_count_by_icu_day.py`) correctly use
    # `_is_full_24h_day` because each patient contributes one row per
    # day-N bin there.
    sql = f"""
        WITH hours AS (
            FROM read_parquet('{seddose_path}') seddose_by_id_imvhr
            JOIN read_parquet('{meta_path}') USING (hospitalization_id, _nth_day)
            -- Phase-3 column names (rates with explicit unit suffix). The
            -- upstream parquet now stores per-hour avg rates directly, so
            -- there's no /60.0 conversion needed on this side.
            SELECT
                seddose_by_id_imvhr.hospitalization_id
                , seddose_by_id_imvhr._nth_day
                , seddose_by_id_imvhr._hr
                , seddose_by_id_imvhr.prop_mcg_kg_min_total
                , seddose_by_id_imvhr.fenteq_mcg_hr_total
                , seddose_by_id_imvhr.midazeq_mg_hr_total
                , hour_of_stay: (
                    (seddose_by_id_imvhr._nth_day - 1) * 24
                    + ((seddose_by_id_imvhr._hr - 7 + 24) % 24)
                )::INT
            WHERE seddose_by_id_imvhr._nth_day BETWEEN 1 AND 7
        )
        FROM hours
        SELECT
            hour_of_stay
            -- CSV output column names preserved for back-compat with the
            -- cross-site figure-rendering layer; values are rate-mean
            -- directly (no /60.0 conversion).
            , propofol_mcg_kg_min_mean: AVG(COALESCE(prop_mcg_kg_min_total, 0))
            , fenteq_mcg_mean:          AVG(COALESCE(fenteq_mcg_hr_total, 0))
            , midazeq_mg_mean:          AVG(COALESCE(midazeq_mg_hr_total, 0))
            , n_on_drug_propofol: COUNT(*) FILTER (WHERE prop_mcg_kg_min_total > 0)
            , n_on_drug_fenteq:   COUNT(*) FILTER (WHERE fenteq_mcg_hr_total   > 0)
            , n_on_drug_midazeq:  COUNT(*) FILTER (WHERE midazeq_mg_hr_total   > 0)
            -- Per-hour union: patient-hours with ANY of the three
            -- sedative drugs running. Drug-independent baseline
            -- rendered identically on all 3 PNGs so reviewers can
            -- compare each drug's curve to the larger pie.
            , n_on_drug_any: COUNT(*) FILTER (
                WHERE prop_mcg_kg_min_total > 0
                   OR fenteq_mcg_hr_total > 0
                   OR midazeq_mg_hr_total > 0
            )
            , n_imv: COUNT(*)
        WHERE hour_of_stay BETWEEN 0 AND 167
        GROUP BY hour_of_stay
        ORDER BY hour_of_stay
    """
    return duckdb.sql(sql).df()


def main() -> None:
    ensure_dirs()
    df = _compute()
    if df.empty:
        logger.info("No per-site data computed; nothing to write.")
        return
    save_csv(df, "sed_dose_by_hr_of_stay")
    logger.info(
        f"Wrote {len(df)} rows × {df.shape[1]} cols → "
        f"{TABLES_DIR}/sed_dose_by_hr_of_stay.csv"
    )


if __name__ == "__main__":
    main()
