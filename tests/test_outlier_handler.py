"""Equivalence tests for the vendored DuckDB outlier handler.

The handler at ``code/_outlier_handler.py`` mirrors clifpy's
``apply_outlier_handling`` semantics but stays lazy on a DuckDBPyRelation.
This test fixture exercises each column shape (category-dependent,
medication, simple range) on tiny in-memory frames and asserts the
expected null pattern.
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import duckdb
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from _outlier_handler import apply_outlier_handling_duckdb  # noqa: E402


def _values_match(actual_series: pd.Series, expected: list) -> None:
    """Compare a numeric series against an expected list, treating None
    in the expected list as NaN. SQL NULL on float columns reads back as
    pandas NaN, so we can't `==` them directly."""
    assert len(actual_series) == len(expected), \
        f"length mismatch: got {len(actual_series)}, expected {len(expected)}"
    for i, (a, e) in enumerate(zip(actual_series.tolist(), expected)):
        if e is None:
            assert pd.isna(a), f"index {i}: expected NaN, got {a!r}"
        else:
            assert not pd.isna(a) and a == e, f"index {i}: expected {e!r}, got {a!r}"


@pytest.fixture
def config_path(tmp_path: Path) -> str:
    """Minimal outlier config covering all three column shapes."""
    p = tmp_path / "outlier_config.yaml"
    p.write_text(textwrap.dedent("""\
        tables:
          vitals:
            vital_value:
              heart_rate:
                min: 0
                max: 300
              spo2:
                min: 50
                max: 100
          medication_admin_continuous:
            med_dose:
              propofol:
                "mcg/kg/min":
                  min: 0.0
                  max: 200.0
                "mg/hr":
                  min: 0.0
                  max: 400.0
              fentanyl:
                "mcg/min":
                  min: 0.0
                  max: 8.5
          respiratory_support:
            fio2_set:
              min: 0.21
              max: 1.0
            peep_set:
              min: 0.0
              max: 30.0
    """))
    return str(p)


def test_category_dependent_vitals(config_path: str) -> None:
    vitals_df = pd.DataFrame({
        'hospitalization_id': ['H1'] * 6,
        'vital_category': ['heart_rate', 'heart_rate', 'spo2', 'spo2', 'heart_rate', 'spo2'],
        'vital_value':    [80.0,         400.0,         95.0,   30.0,   -5.0,         100.0],
    })
    rel = duckdb.sql("FROM vitals_df SELECT *")
    out = apply_outlier_handling_duckdb(rel, 'vitals', config_path).df()
    # heart_rate: 80 ok, 400 > 300 nullified, -5 < 0 nullified.
    # spo2: 95 ok, 30 < 50 nullified, 100 ok (max inclusive).
    _values_match(out['vital_value'], [80.0, None, 95.0, None, None, 100.0])


def test_medication_continuous(config_path: str) -> None:
    med_df = pd.DataFrame({
        'hospitalization_id': ['H1'] * 5,
        'med_category':       ['propofol', 'propofol',    'propofol', 'fentanyl', 'fentanyl'],
        'med_dose_unit':      ['mcg/kg/min', 'mcg/kg/min', 'mg/hr',    'mcg/min',  'mcg/min'],
        'med_dose':           [50.0,         500.0,         100.0,     2.5,        20.0],
    })
    rel = duckdb.sql("FROM med_df SELECT *")
    out = apply_outlier_handling_duckdb(
        rel, 'medication_admin_continuous', config_path,
    ).df()
    # propofol mcg/kg/min: 50 ok, 500 > 200 nullified.
    # propofol mg/hr: 100 ok.
    # fentanyl mcg/min: 2.5 ok, 20 > 8.5 nullified.
    _values_match(out['med_dose'], [50.0, None, 100.0, 2.5, None])


def test_simple_range_resp_support(config_path: str) -> None:
    resp_df = pd.DataFrame({
        'hospitalization_id': ['H1'] * 4,
        'fio2_set':           [0.4,   1.5,   0.21,  0.10],
        'peep_set':           [5.0,   45.0,  0.0,   8.0],
    })
    rel = duckdb.sql("FROM resp_df SELECT *")
    out = apply_outlier_handling_duckdb(
        rel, 'respiratory_support', config_path,
    ).df()
    # fio2_set: 0.4 ok, 1.5 > 1.0 nullified, 0.21 ok (min inclusive),
    #           0.10 < 0.21 nullified.
    _values_match(out['fio2_set'], [0.4, None, 0.21, None])
    # peep_set: 5 ok, 45 > 30 nullified, 0 ok, 8 ok.
    _values_match(out['peep_set'], [5.0, None, 0.0, 8.0])


def test_passthrough_for_unconfigured_table(config_path: str) -> None:
    df = pd.DataFrame({'col': [1, 2, 3]})
    rel = duckdb.sql("FROM df SELECT *")
    out = apply_outlier_handling_duckdb(rel, 'not_a_real_table', config_path).df()
    pd.testing.assert_frame_equal(out, df)


def test_lazy_chain_preserved(config_path: str) -> None:
    """The handler must return a relation, not a materialized DataFrame."""
    df = pd.DataFrame({
        'hospitalization_id': ['H1'],
        'vital_category':     ['heart_rate'],
        'vital_value':        [80.0],
    })
    rel = duckdb.sql("FROM df SELECT *")
    out = apply_outlier_handling_duckdb(rel, 'vitals', config_path)
    assert isinstance(out, duckdb.DuckDBPyRelation), \
        f"expected DuckDBPyRelation, got {type(out)}"


def test_missing_category_column_skips_with_warning(config_path: str, caplog) -> None:
    """If the relation is missing the category column, skip not crash."""
    import logging
    df = pd.DataFrame({'vital_value': [80.0, 999.0]})  # no vital_category
    rel = duckdb.sql("FROM df SELECT *")
    with caplog.at_level(logging.WARNING, logger="clifpy.epi_sedation.outlier"):
        out = apply_outlier_handling_duckdb(rel, 'vitals', config_path).df()
    # No nullification — values pass through because the category column is absent.
    assert out['vital_value'].tolist() == [80.0, 999.0]


def test_extra_columns_preserved(config_path: str) -> None:
    """SELECT * REPLACE must preserve all non-target columns unchanged."""
    df = pd.DataFrame({
        'hospitalization_id': ['H1', 'H2'],
        'recorded_dttm':      pd.to_datetime(['2024-01-01 10:00', '2024-01-01 11:00']),
        'vital_category':     ['heart_rate', 'heart_rate'],
        'vital_value':        [80.0, 999.0],
        '_extra_col':         ['a', 'b'],
    })
    rel = duckdb.sql("FROM df SELECT *")
    out = apply_outlier_handling_duckdb(rel, 'vitals', config_path).df()
    # All columns survive; only vital_value is mutated.
    assert set(out.columns) == set(df.columns)
    assert out['_extra_col'].tolist() == ['a', 'b']
    assert out['hospitalization_id'].tolist() == ['H1', 'H2']
    _values_match(out['vital_value'], [80.0, None])


def test_string_quote_escaping(tmp_path: Path) -> None:
    """Categories with apostrophes (rare but possible in real data) must
    not break the SQL literal."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(textwrap.dedent("""\
        tables:
          vitals:
            vital_value:
              "patient's heart rate":
                min: 0
                max: 300
    """))
    df = pd.DataFrame({
        'vital_category': ["patient's heart rate", "patient's heart rate"],
        'vital_value':    [80.0, 400.0],
    })
    rel = duckdb.sql("FROM df SELECT *")
    out = apply_outlier_handling_duckdb(rel, 'vitals', str(cfg)).df()
    _values_match(out['vital_value'], [80.0, None])
