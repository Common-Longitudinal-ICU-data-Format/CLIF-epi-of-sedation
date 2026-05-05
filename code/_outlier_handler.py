"""DuckDB-native outlier handler for the sedation pipeline.

Mirrors clifpy.utils.outlier_handler's semantics (range-based NULL replacement
driven by a YAML config) but operates on a ``DuckDBPyRelation`` and stays lazy.

Vendored locally (rather than imported from clifpy) so the full lazy chain
``load_data(return_rel=True) → ... → apply_outlier_handling_duckdb(...) → ...
→ .to_parquet()`` avoids the pandas round-trip clifpy's version forces via
``table_obj.df`` + ``polars.from_pandas`` + ``.to_pandas()``.

Schema-tracked: keep config keys aligned with
``clifpy/schemas/outlier_config.yaml``. If clifpy adds a new (table, column)
pair, mirror it here.

Long-term: upstream this DuckDB version into clifpy so other CLIF projects
can drop their pandas pinch.
"""
from __future__ import annotations

import logging
from typing import Any

import duckdb
import yaml
from clifpy.utils.logging_config import get_logger

logger = get_logger("epi_sedation.outlier")


_CATEGORY_DEPENDENT_TABLES = {
    'vitals': ('vital_value', 'vital_category'),
    'labs': ('lab_value_numeric', 'lab_category'),
    'patient_assessments': ('numerical_value', 'assessment_category'),
}
_MEDICATION_TABLES = (
    'medication_admin_continuous',
    'medication_admin_intermittent',
)


def apply_outlier_handling_duckdb(
    rel: duckdb.DuckDBPyRelation,
    table_name: str,
    outlier_config_path: str,
) -> duckdb.DuckDBPyRelation:
    """Apply YAML-driven outlier handling lazily to a ``DuckDBPyRelation``.

    Returns a NEW lazy relation with out-of-range values replaced by NULL.
    The original relation is untouched. Per-(category, unit) drop counts
    are surfaced at DEBUG via ``epi_sedation.outlier`` (gated to avoid
    paying for the count query when DEBUG is off).

    Supported column shapes (mirrors clifpy/schemas/outlier_config.yaml):

    - Category-dependent (vitals.vital_value, labs.lab_value_numeric,
      patient_assessments.numerical_value): nested
      ``{category: {min, max}}`` config.

    - Medication (medication_admin_*.med_dose): nested
      ``{med_category: {unit: {min, max}}}`` config.

    - Simple range (respiratory_support.fio2_set, hospitalization.age_at_admission,
      etc.): flat ``{min, max}`` config.

    Caller contract: ``rel`` is lazy; columns referenced by the YAML keys
    must already exist on the relation (load `med_category` + `med_dose_unit`
    alongside `med_dose`, etc.).
    """
    with open(outlier_config_path) as f:
        config = yaml.safe_load(f)

    table_config = config.get('tables', {}).get(table_name, {})
    if not table_config:
        logger.info(f"No outlier config for {table_name}; passthrough.")
        return rel

    rel_columns = set(rel.columns)
    for col, col_config in table_config.items():
        if col not in rel_columns:
            continue
        case_sql = _build_case_sql(table_name, col, col_config, rel_columns)
        if case_sql is None:
            continue

        if logger.isEnabledFor(logging.DEBUG):
            _log_drop_counts(rel, table_name, col, col_config, case_sql)

        # SELECT * REPLACE wraps the relation lazily — the new column
        # shadows the original; type stays the same (NULL is range-compatible).
        rel = duckdb.sql(
            f"FROM rel SELECT * REPLACE ({case_sql} AS {col})"
        )
    return rel


def _build_case_sql(
    table_name: str,
    col: str,
    col_config: Any,
    rel_columns: set,
) -> str | None:
    """Construct a DuckDB ``CASE WHEN`` expression for one column.

    Returns ``None`` when the column shape doesn't match any supported
    template (logs a warning) or when the YAML config has no usable bounds.
    """
    if (table_name in _CATEGORY_DEPENDENT_TABLES
            and col == _CATEGORY_DEPENDENT_TABLES[table_name][0]):
        cat_col = _CATEGORY_DEPENDENT_TABLES[table_name][1]
        if cat_col not in rel_columns:
            logger.warning(
                f"{cat_col} not in relation; skipping outlier handling for {col}."
            )
            return None
        whens = []
        for category, rng in col_config.items():
            if not (isinstance(rng, dict) and 'min' in rng and 'max' in rng):
                continue
            whens.append(
                f"WHEN LOWER({cat_col}) = '{_esc(category).lower()}' "
                f"AND ({col} < {rng['min']} OR {col} > {rng['max']}) THEN NULL"
            )
        return f"CASE {' '.join(whens)} ELSE {col} END" if whens else None

    if table_name in _MEDICATION_TABLES and col == 'med_dose':
        for required in ('med_category', 'med_dose_unit'):
            if required not in rel_columns:
                logger.warning(
                    f"{required} not in relation; skipping outlier handling for med_dose."
                )
                return None
        whens = []
        for med_cat, unit_cfgs in col_config.items():
            for unit, rng in (unit_cfgs or {}).items():
                if not (isinstance(rng, dict) and 'min' in rng and 'max' in rng):
                    continue
                whens.append(
                    f"WHEN LOWER(med_category) = '{_esc(med_cat).lower()}' "
                    f"AND LOWER(med_dose_unit) = '{_esc(unit).lower()}' "
                    f"AND (med_dose < {rng['min']} OR med_dose > {rng['max']}) THEN NULL"
                )
        return f"CASE {' '.join(whens)} ELSE med_dose END" if whens else None

    if isinstance(col_config, dict) and 'min' in col_config and 'max' in col_config:
        return (
            f"CASE WHEN {col} < {col_config['min']} OR {col} > {col_config['max']} "
            f"THEN NULL ELSE {col} END"
        )
    return None


def _esc(s: str) -> str:
    """Escape single quotes for embedding in a SQL string literal."""
    return s.replace("'", "''")


def _log_drop_counts(
    rel: duckdb.DuckDBPyRelation,
    table_name: str,
    col: str,
    col_config: Any,
    case_sql: str,
) -> None:
    """Emit per-(category[, unit]) drop counts at DEBUG.

    Runs a small scalar query that compares the case_sql result to the input
    column. Lazy on the input relation — the count query is independent and
    doesn't break the outer DAG.
    """
    try:
        if (table_name in _CATEGORY_DEPENDENT_TABLES
                and col == _CATEGORY_DEPENDENT_TABLES[table_name][0]):
            cat_col = _CATEGORY_DEPENDENT_TABLES[table_name][1]
            rows = duckdb.sql(f"""
                FROM rel
                SELECT {cat_col} AS category
                    , SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS n_before
                    , SUM(CASE WHEN ({case_sql}) IS NOT NULL THEN 1 ELSE 0 END) AS n_after
                GROUP BY {cat_col}
                ORDER BY n_before DESC
            """).fetchall()
            for category, n_before, n_after in rows:
                nullified = (n_before or 0) - (n_after or 0)
                if n_before:
                    pct = nullified / n_before * 100
                    logger.debug(
                        f"  {table_name}.{col} [{category}]: "
                        f"{n_before:,} → {nullified:,} nullified ({pct:.1f}%)"
                    )

        elif table_name in _MEDICATION_TABLES and col == 'med_dose':
            rows = duckdb.sql(f"""
                FROM rel
                SELECT med_category, med_dose_unit
                    , SUM(CASE WHEN med_dose IS NOT NULL THEN 1 ELSE 0 END) AS n_before
                    , SUM(CASE WHEN ({case_sql}) IS NOT NULL THEN 1 ELSE 0 END) AS n_after
                GROUP BY med_category, med_dose_unit
                ORDER BY n_before DESC
            """).fetchall()
            for med_cat, unit, n_before, n_after in rows:
                nullified = (n_before or 0) - (n_after or 0)
                if n_before:
                    pct = nullified / n_before * 100
                    logger.debug(
                        f"  {table_name}.med_dose [{med_cat} ({unit})]: "
                        f"{n_before:,} → {nullified:,} nullified ({pct:.1f}%)"
                    )

        else:
            row = duckdb.sql(f"""
                FROM rel
                SELECT
                    SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) AS n_before
                    , SUM(CASE WHEN ({case_sql}) IS NOT NULL THEN 1 ELSE 0 END) AS n_after
            """).fetchone()
            n_before, n_after = row or (0, 0)
            nullified = (n_before or 0) - (n_after or 0)
            if n_before:
                pct = nullified / n_before * 100
                logger.debug(
                    f"  {table_name}.{col}: "
                    f"{n_before:,} → {nullified:,} nullified ({pct:.1f}%)"
                )
    except Exception as e:
        # DEBUG-only diagnostic; never raise during normal pipeline runs.
        logger.debug(f"Outlier drop-count diagnostic failed for {table_name}.{col}: {e}")
