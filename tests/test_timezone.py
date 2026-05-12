"""Timezone invariants for the day/night shift derivation pipeline.

Goals:
1. `add_day_shift_id` returns the local-hour interpretation, not UTC.
2. Result is invariant under `SET TimeZone = ...` on DuckDB's global default
   connection (no session-tz dependency).
3. DST boundaries are handled correctly (spring-forward + fall-back).
4. Cross-site comparability: same UTC instant produces site-correct local hour.
5. Legacy default (site_tz='UTC') reproduces pre-fix behavior — staging primitive
   for landing _utils.py before flipping call sites.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

# Make the project's _utils.py importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))
from _utils import (  # noqa: E402
    add_day_shift_id,
    add_dh_hr,
    coerce_dttm_to_utc,
    to_utc,
)


@pytest.fixture
def utc_grid():
    """UTC tz-aware fixture covering DST boundaries + non-DST midyear.

    Hand-picked to anchor on Chicago wall-clock 7:00 and 19:00 (the
    project's shift boundaries) at multiple DST states. The
    `_hr ≠ 7`-between-day-starts rows (3-10 23:00, 6-01 18:00) are
    necessary because the LAG-based `_is_day_start` logic only fires on
    transitions from non-7 → 7. In a real dense hourly grid every hour is
    present, so this is automatic; the sparse fixture must include
    intermediate non-7 rows to mirror that.
    """
    return pd.DataFrame({
        "hospitalization_id": ["H1"] * 9,
        "event_dttm": pd.to_datetime([
            # Pre-DST 2024 (Chicago = CST = UTC-6)
            "2024-03-09 13:00:00+00:00",  # = 07:00 Chicago, day-start #1
            "2024-03-09 23:00:00+00:00",  # = 17:00 Chicago, still day
            "2024-03-10 01:00:00+00:00",  # = 19:00 Chicago, becomes night
            # Spring-forward at 2024-03-10 02:00 CST (= 08:00 UTC); after, Chicago = CDT = UTC-5
            "2024-03-10 12:00:00+00:00",  # = 07:00 Chicago (CDT), day-start #2
            "2024-03-10 23:00:00+00:00",  # = 18:00 Chicago, _hr=18 (breaks consecutive-7 LAG)
            # Mid-year DST (Chicago = CDT = UTC-5)
            "2024-06-01 12:00:00+00:00",  # = 07:00 Chicago, day-start #3
            "2024-06-01 23:00:00+00:00",  # = 18:00 Chicago, still day
            "2024-06-02 00:00:00+00:00",  # = 19:00 Chicago, becomes night
            # Fall-back boundary 2024-11-03 (post-fall-back, CST = UTC-6)
            "2024-11-03 12:00:00+00:00",  # = 06:00 Chicago, _hr=6 (no new day)
        ]),
    })


def test_local_hour_correct(utc_grid):
    """_hr is the local Chicago hour, not the UTC hour."""
    out = add_day_shift_id(utc_grid, site_tz="America/Chicago")
    expected = [7, 17, 19, 7, 18, 7, 18, 19, 6]
    assert out["_hr"].tolist() == expected, (
        f"got {out['_hr'].tolist()}, expected {expected}"
    )


def test_shift_matches_local_hour(utc_grid):
    """_shift = 'day' iff local _hr in [7, 19)."""
    out = add_day_shift_id(utc_grid, site_tz="America/Chicago")
    expected = ["day", "day", "night", "day", "day", "day", "day", "night", "night"]
    assert out["_shift"].tolist() == expected


def test_nth_day_counts_local_7am_crossings(utc_grid):
    """_nth_day increments at each local 7am crossing per hospitalization.

    Three crossings in the fixture:
    - 2024-03-09 13:00 UTC = 07:00 Chicago (CST)  → _nth_day flips 0→1
    - 2024-03-10 12:00 UTC = 07:00 Chicago (CDT)  → _nth_day flips 1→2
    - 2024-06-01 12:00 UTC = 07:00 Chicago (CDT)  → _nth_day flips 2→3
    The fall-back row (2024-11-03 06:00 Chicago) is < 7 so no new crossing.
    """
    out = add_day_shift_id(utc_grid, site_tz="America/Chicago")
    expected = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    assert out["_nth_day"].tolist() == expected


@pytest.mark.parametrize(
    "session_tz",
    ["UTC", "America/Chicago", "America/New_York", "Asia/Tokyo"],
)
def test_invariant_under_session_timezone(utc_grid, session_tz):
    """add_day_shift_id must produce the same result regardless of DuckDB's
    session timezone.

    This guards against the SET TimeZone fragility — clifpy's
    load_parquet_with_tz (pyCLIF/clifpy/utils/io.py:364) explicitly does
    `SET timezone = 'UTC'` on every parquet load via the global default
    connection, and any future helper might mutate it differently. We
    require correctness despite arbitrary session-tz state.
    """
    base = add_day_shift_id(utc_grid, site_tz="America/Chicago")
    duckdb.execute(f"SET TimeZone = '{session_tz}'")
    try:
        out = add_day_shift_id(utc_grid, site_tz="America/Chicago")
        pd.testing.assert_frame_equal(
            base.reset_index(drop=True),
            out.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    finally:
        # Restore the default to avoid leaking session state into other tests.
        duckdb.execute("SET TimeZone = 'UTC'")


def test_cross_site_same_instant_different_local_hour(utc_grid):
    """Same UTC fixture under UCMC vs MIMIC tz: each site sees its own local
    hour for the same physical instant.

    Mid-year row (index 5, 2024-06-01 12:00 UTC):
    - UCMC (Chicago, CDT = UTC-5) → 07:00 → _hr = 7  → _shift = 'day'
    - MIMIC (New York, EDT = UTC-4) → 08:00 → _hr = 8 → _shift = 'day'
    """
    ucmc = add_day_shift_id(utc_grid, site_tz="America/Chicago")
    mimic = add_day_shift_id(utc_grid, site_tz="America/New_York")
    assert ucmc.loc[5, "_hr"] == 7
    assert mimic.loc[5, "_hr"] == 8
    assert ucmc.loc[5, "_shift"] == "day"
    assert mimic.loc[5, "_shift"] == "day"


def test_site_tz_is_required(utc_grid):
    """Calling without site_tz raises TypeError (post-ea911a9: default removed).

    The pre-fix UTC-hour behavior is still reachable by passing
    ``site_tz="UTC"`` explicitly — that escape hatch is preserved here so
    the legacy reproduction guarantee remains testable. What's gone is the
    silent fallback: a forgotten kwarg now fails fast at call time instead
    of producing UTC-hour shifts.
    """
    with pytest.raises(TypeError):
        add_day_shift_id(utc_grid)  # no site_tz → TypeError

    # Explicit site_tz="UTC" still reproduces pre-fix UTC-hour output.
    out = add_day_shift_id(utc_grid, site_tz="UTC")
    expected_utc = [13, 23, 1, 12, 23, 12, 23, 0, 12]
    assert out["_hr"].tolist() == expected_utc


def _load_qc_shared():
    """Import code/qc/_shared.py via importlib (it's not in a regular package
    — the qc dir has no __init__.py and `_shared` would collide with
    descriptive/_shared.py if added to sys.path)."""
    import importlib.util
    qc_path = Path(__file__).resolve().parent.parent / "code" / "qc" / "_shared.py"
    spec = importlib.util.spec_from_file_location("qc_shared", qc_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_qc_night_windows_dst_correct():
    """`night_windows_in_range` produces correct site-local boundaries on
    DST transition days.

    Pre-fix, the helper used `pd.Timedelta(hours=19)` on a tz-aware
    timestamp at local-midnight — `pd.Timedelta` is absolute (always 19 *
    3600 s), so on Chicago's spring-forward day (2024-03-10) the result
    landed at 20:00 CDT instead of 19:00 CDT, and on fall-back
    (2024-11-03) at 18:00 CST instead of 19:00 CST.

    The fix swaps to `_local_boundary` + `pd.tseries.offsets.Day(1)`
    (calendar-aware), so the wall-clock is 19:00 / 07:00 site-local on
    every day, including DST boundaries.
    """
    qc = _load_qc_shared()
    site_tz = "America/Chicago"
    # Cohort spanning both 2024 DST transitions.
    start = pd.Timestamp("2024-03-08 12:00", tz=site_tz)
    end = pd.Timestamp("2024-11-05 12:00", tz=site_tz)

    windows = qc.night_windows_in_range(start, end)
    assert len(windows) > 0, "expected non-empty night-window list"

    # Skip first/last windows — those are clipped to the [start, end]
    # range and don't necessarily land on 19:00 / 07:00 boundaries.
    for n_start, n_end in windows[1:-1]:
        assert n_start.tz is not None, f"tz-naive returned: {n_start}"
        assert n_end.tz is not None, f"tz-naive returned: {n_end}"
        assert n_start.tz_convert(site_tz).hour == 19, (
            f"night_start drift on {n_start.date()}: got hour "
            f"{n_start.tz_convert(site_tz).hour}, expected 19"
        )
        assert n_end.tz_convert(site_tz).hour == 7, (
            f"night_end drift on {n_end.date()}: got hour "
            f"{n_end.tz_convert(site_tz).hour}, expected 7"
        )

    # Specifically validate the DST boundary days are present and correct.
    boundary_dates = {
        pd.Timestamp("2024-03-10", tz=site_tz).date(),  # spring forward
        pd.Timestamp("2024-11-03", tz=site_tz).date(),  # fall back
    }
    found = {n_start.date() for n_start, _ in windows}
    assert boundary_dates <= found, (
        f"expected DST boundary dates in night windows; missing "
        f"{boundary_dates - found}"
    )


# ── to_utc unit tests ────────────────────────────────────────────────────
#
# `to_utc` is the consolidated tz-normalization helper used at every
# parquet-write and clifpy-load boundary. These tests pin the four
# behavioral contracts that downstream code relies on:
# (1) idempotent on already-UTC input, (2) tz-aware → UTC is metadata-only,
# (3) naive + naive_means → localize-then-convert handles DST edge cases,
# (4) naive without naive_means raises (defensive: silent localization to
#     a wrong tz would shift every downstream UTC instant by 4-6 hours).


def test_to_utc_idempotent_on_already_utc():
    """Already-UTC columns are unchanged."""
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(
            ["2024-06-01 12:00", "2024-11-03 06:00"], utc=True
        )
    })
    out = to_utc(df, "event_dttm")
    assert str(out["event_dttm"].dt.tz) == "UTC"
    pd.testing.assert_series_equal(out["event_dttm"], df["event_dttm"])


def test_to_utc_tz_aware_site_local_metadata_flip():
    """tz-aware US/Eastern → UTC preserves the absolute instant.

    12:00 EDT (June, UTC-4) on a US/Eastern-tagged column reads back as
    16:00 UTC after the metadata flip.
    """
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2024-06-01 12:00"]).tz_localize(
            "US/Eastern"
        )
    })
    out = to_utc(df, "event_dttm")
    assert str(out["event_dttm"].dt.tz) == "UTC"
    assert out["event_dttm"].iloc[0].hour == 16


def test_to_utc_naive_means_utc():
    """Naive timestamp claimed to be UTC: tag added, value unchanged."""
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2024-06-01 12:00"])
    })
    out = to_utc(df, "event_dttm", naive_means="UTC")
    assert str(out["event_dttm"].dt.tz) == "UTC"
    assert out["event_dttm"].iloc[0].hour == 12


def test_to_utc_naive_means_site_tz_dst_fallback():
    """Naive site-local with a fall-back-ambiguous timestamp resolves.

    UCMC's 2019-11-03 01:30 Central exists twice (once before fall-back as
    CDT, once after as CST). to_utc must resolve deterministically (post-
    fall-back / standard-time, matching DuckDB's `AT TIME ZONE` convention)
    rather than raise AmbiguousTimeError.
    """
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2019-11-03 01:30", "2024-06-01 12:00"])
    })
    out = to_utc(df, "event_dttm", naive_means="US/Central")
    assert str(out["event_dttm"].dt.tz) == "UTC"
    # 01:30 CST (post-fall-back, UTC-6) → 07:30 UTC.
    assert out["event_dttm"].iloc[0].hour == 7
    assert out["event_dttm"].iloc[0].minute == 30
    # 12:00 CDT (June, UTC-5) → 17:00 UTC.
    assert out["event_dttm"].iloc[1].hour == 17


def test_to_utc_naive_without_naive_means_raises():
    """Defensive: naive input without `naive_means` raises ValueError.

    Silent localization to a default tz (e.g., always UTC, or session tz)
    would shift every downstream UTC instant by 4-6 hours when clifpy
    returns naive site-local. The defensive raise forces callers to
    declare intent at every load boundary.
    """
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2024-06-01 12:00"])
    })
    with pytest.raises(ValueError, match="naive"):
        to_utc(df, "event_dttm")


def test_to_utc_missing_column_silently_skipped():
    """Missing column names are silently ignored.

    This mirrors the pre-consolidation `retag_to_local_tz` behavior and is
    relied on at sites where the column set varies per CLIF site (e.g.,
    `compute_ase` emits more event-types at UCMC than MIMIC).
    """
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2024-06-01 12:00"], utc=True)
    })
    out = to_utc(df, ["event_dttm", "nonexistent_dttm"])
    assert "nonexistent_dttm" not in out.columns
    assert str(out["event_dttm"].dt.tz) == "UTC"


def test_to_utc_string_columns_arg_accepted():
    """A bare string for `columns` is accepted (not just a list).

    Ergonomic for the common single-column case.
    """
    df = pd.DataFrame({
        "event_dttm": pd.to_datetime(["2024-06-01 12:00"], utc=True)
    })
    out_str = to_utc(df, "event_dttm")
    out_list = to_utc(df, ["event_dttm"])
    pd.testing.assert_frame_equal(out_str, out_list)


# ── Integration-style schema audit ──────────────────────────────────────


def test_event_dttm_is_utc_on_disk_after_pipeline():
    """Every `*_dttm` column written under `output/{site}/` is UTC.

    Project convention (post-2026-05-10 UTC-everywhere migration): every
    `*_dttm` column on disk is `datetime64[us, UTC]`. This guards against
    regressions where a future writer forgets the to_utc call and lets a
    SITE_TZ-tagged or naive column slip onto disk.

    Skipped if the pipeline hasn't been run yet (no output directory).
    """
    output_root = Path(__file__).resolve().parent.parent / "output"
    if not output_root.is_dir():
        pytest.skip("no output/ directory — pipeline not run yet")

    parquet_files = sorted(output_root.glob("*/*.parquet"))
    if not parquet_files:
        pytest.skip("no per-site parquet outputs — pipeline not run yet")

    violations: list[str] = []
    for path in parquet_files:
        df = pd.read_parquet(path)
        for c in df.columns:
            if not c.endswith("_dttm"):
                continue
            if not pd.api.types.is_datetime64_any_dtype(df[c]):
                continue
            tz = df[c].dt.tz
            if tz is None or str(tz) != "UTC":
                rel = path.relative_to(output_root.parent)
                violations.append(f"{rel}.{c} tz={tz}")

    assert not violations, (
        "non-UTC *_dttm columns found on disk:\n  "
        + "\n  ".join(violations)
    )


# ── add_dh_hr unit tests ────────────────────────────────────────────────
#
# `add_dh_hr` is the polymorphic helper extracted from `add_day_shift_id`.
# It's the single source of truth for `_dh` (local-tz hour-floor) and
# `_hr` (local hour-of-day) derivation across the project. These tests
# pin polymorphism, correctness, and session-tz invariance at the
# add_dh_hr layer so failures point precisely there rather than to the
# shift-id assignment composed on top by add_day_shift_id.


def test_add_dh_hr_pandas_in_pandas_out(utc_grid):
    """pandas DataFrame in → pandas DataFrame out."""
    out = add_dh_hr(utc_grid, "event_dttm", site_tz="America/Chicago")
    assert isinstance(out, pd.DataFrame), f"got {type(out).__name__}"
    assert "_dh" in out.columns and "_hr" in out.columns
    assert len(out) == len(utc_grid)


def test_add_dh_hr_relation_in_relation_out(utc_grid):
    """DuckDBPyRelation in → DuckDBPyRelation out (lazy)."""
    rel = duckdb.sql("FROM utc_grid SELECT *")
    out = add_dh_hr(rel, "event_dttm", site_tz="America/Chicago")
    assert not isinstance(out, pd.DataFrame), f"got DataFrame; expected lazy relation"
    cols = [d[0] for d in out.description]
    assert "_dh" in cols and "_hr" in cols


def test_add_dh_hr_local_hour_correct(utc_grid):
    """_hr is the local Chicago hour, not UTC. Same fixture as
    test_local_hour_correct but at the add_dh_hr layer so failures
    point precisely to _dh/_hr derivation rather than the shift-id
    assignment composed on top.
    """
    out = add_dh_hr(utc_grid, "event_dttm", site_tz="America/Chicago")
    expected = [7, 17, 19, 7, 18, 7, 18, 19, 6]
    assert out["_hr"].tolist() == expected, (
        f"got {out['_hr'].tolist()}, expected {expected}"
    )


def test_add_dh_hr_session_tz_invariant(utc_grid):
    """Result is identical regardless of DuckDB's session timezone.

    Guards against the same SET TimeZone fragility tested for
    add_day_shift_id (test_invariant_under_session_timezone) but at the
    add_dh_hr layer — the parent test would still pass if shift-id
    assignment masked an upstream session-tz leak.
    """
    base = add_dh_hr(utc_grid, "event_dttm", site_tz="America/Chicago")
    for session_tz in ["UTC", "America/Chicago", "Asia/Tokyo"]:
        duckdb.execute(f"SET TimeZone = '{session_tz}'")
        try:
            out = add_dh_hr(utc_grid, "event_dttm", site_tz="America/Chicago")
            pd.testing.assert_frame_equal(
                base.reset_index(drop=True),
                out.reset_index(drop=True),
                check_dtype=True,
                check_exact=True,
            )
        finally:
            duckdb.execute("SET TimeZone = 'UTC'")


# ── coerce_dttm_to_utc — DuckDB-native load-boundary tz coercion ──────────
#
# coerce_dttm_to_utc is the lazy DuckDBPyRelation sibling of to_utc. The
# three tests below pin behavior for the three source-parquet shapes a new
# CLIF site may deliver: UTC tz-aware (current MIMIC/UCMC), local-tz
# tz-aware, and naive site-local. All three should normalize to identical
# UTC instants downstream when site_tz is set correctly.


def _coerce_value(rel, col):
    """Pull the single dttm value from a 1-row relation in UTC."""
    duckdb.execute("SET TimeZone = 'UTC'")
    return rel.df()[col].iloc[0]


def test_coerce_dttm_to_utc_timestamptz_passthrough():
    """TIMESTAMPTZ input (UTC tag) is unchanged by the helper.

    Shape 1 of the cross-site source-tz robustness story: existing MIMIC +
    UCMC parquets ship `recorded_dttm` as `datetime64[us, UTC]`. The
    helper must be a strict no-op so MIMIC/UCMC outputs stay byte-
    equivalent after the migration.
    """
    rel = duckdb.sql(
        "SELECT TIMESTAMPTZ '2024-06-01 12:00:00+00' AS event_dttm"
    )
    out = coerce_dttm_to_utc(rel, ["event_dttm"], "US/Eastern")
    val = _coerce_value(out, "event_dttm")
    assert val == pd.Timestamp("2024-06-01 12:00", tz="UTC")
    assert "TIMESTAMP" in str(out.types[0]).upper()


def test_coerce_dttm_to_utc_local_tagged_same_utc_instant_as_utc_tagged():
    """TIMESTAMPTZ from a local-tagged source carries the correct UTC instant.

    Shape 2: a site whose parquet stored `recorded_dttm` as
    `datetime64[us, US/Eastern]`. DuckDB's TIMESTAMPTZ stores the UTC
    instant regardless of source tag, so 12:00 EDT (= 16:00 UTC) on the
    local-tagged source must equal the 16:00-UTC value loaded from a
    UTC-tagged source.
    """
    rel_utc_tagged = duckdb.sql(
        "SELECT TIMESTAMPTZ '2024-06-01 16:00:00+00' AS event_dttm"
    )
    rel_local_tagged = duckdb.sql(
        "SELECT TIMESTAMPTZ '2024-06-01 12:00:00-04' AS event_dttm"
    )
    out_utc = coerce_dttm_to_utc(rel_utc_tagged, ["event_dttm"], "US/Eastern")
    out_local = coerce_dttm_to_utc(rel_local_tagged, ["event_dttm"], "US/Eastern")
    assert _coerce_value(out_utc, "event_dttm") == _coerce_value(
        out_local, "event_dttm"
    )


def test_coerce_dttm_to_utc_naive_localizes_to_site_tz():
    """TIMESTAMP (naive) input is reinterpreted as site_tz-local.

    Shape 3: a site whose parquet stored `recorded_dttm` as naive
    `datetime64[us]` with the wall-clock IN site-local time. The helper
    must wrap the column in `timezone('<site_tz>', <col>)` so the value
    rebases to the correct UTC instant. For 12:00 US/Eastern (June, EDT,
    UTC-4) → 16:00 UTC.
    """
    rel = duckdb.sql("SELECT TIMESTAMP '2024-06-01 12:00:00' AS event_dttm")
    out = coerce_dttm_to_utc(rel, ["event_dttm"], "US/Eastern")
    val = _coerce_value(out, "event_dttm")
    assert val == pd.Timestamp("2024-06-01 16:00", tz="UTC")
    # The output dtype is TIMESTAMPTZ regardless of input shape.
    assert "TIMESTAMP" in str(out.types[0]).upper()


def test_coerce_dttm_to_utc_three_shapes_yield_identical_instant():
    """All three source shapes converge to the same UTC instant.

    This is the project-wide invariant: every `*_dttm` column flowing past
    the load boundary is `TIMESTAMPTZ` (UTC), and the underlying instant
    is consistent regardless of how the source parquet happened to tag
    (or not tag) its wall-clock.
    """
    rel_shape1 = duckdb.sql(
        "SELECT TIMESTAMPTZ '2024-06-01 16:00:00+00' AS event_dttm"
    )
    rel_shape2 = duckdb.sql(
        "SELECT TIMESTAMPTZ '2024-06-01 12:00:00-04' AS event_dttm"
    )
    rel_shape3 = duckdb.sql(
        "SELECT TIMESTAMP '2024-06-01 12:00:00' AS event_dttm"
    )
    out1 = _coerce_value(
        coerce_dttm_to_utc(rel_shape1, ["event_dttm"], "US/Eastern"), "event_dttm"
    )
    out2 = _coerce_value(
        coerce_dttm_to_utc(rel_shape2, ["event_dttm"], "US/Eastern"), "event_dttm"
    )
    out3 = _coerce_value(
        coerce_dttm_to_utc(rel_shape3, ["event_dttm"], "US/Eastern"), "event_dttm"
    )
    assert out1 == out2 == out3 == pd.Timestamp("2024-06-01 16:00", tz="UTC")


def test_coerce_dttm_to_utc_missing_column_skipped():
    """Names not present in the relation are silently skipped.

    Mirrors `to_utc`'s missing-column tolerance — covariate / event tables
    have heterogeneous column sets across sites, so the helper must not
    fail loudly when an optional dttm column is absent.
    """
    rel = duckdb.sql(
        "SELECT TIMESTAMP '2024-06-01 12:00:00' AS event_dttm, 1 AS x"
    )
    out = coerce_dttm_to_utc(
        rel, ["event_dttm", "nonexistent_dttm"], "US/Eastern"
    )
    assert set(out.columns) == {"event_dttm", "x"}
