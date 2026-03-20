"""Shared utility functions for the sedation pipeline."""

import duckdb
import pandas as pd


def add_day_shift_id(
    df: pd.DataFrame, timestamp_name: str = "event_dttm"
) -> pd.DataFrame:
    """Add day/shift columns (_dh, _hr, _shift, _nth_day, _day_shift) to a DataFrame.

    Day shift: 7:00-19:00, Night shift: 19:00-7:00.
    _nth_day increments at each 7am boundary.
    """
    df["_dh"] = df[timestamp_name].dt.floor("h", ambiguous="NaT")
    df["_hr"] = df[timestamp_name].dt.hour
    _q = """
    WITH day_starts AS (
        FROM df
        SELECT *
            , _shift: CASE WHEN _hr >= 7 AND _hr < 19 THEN 'day' ELSE 'night' END
            , _is_day_start: CASE
                WHEN _hr = 7 AND COALESCE(LAG(_hr) OVER w, -1) != 7 THEN 1
                ELSE 0 END
        WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _dh)
    )
    FROM day_starts
    SELECT *
        , _nth_day: SUM(_is_day_start) OVER w
        , _day_shift: 'day' || _nth_day::INT::TEXT || '_' || _shift
    WINDOW w AS (PARTITION BY hospitalization_id ORDER BY _dh)
    ORDER BY hospitalization_id, _dh
    """
    return duckdb.sql(_q).df()


def remove_meds_duplicates(meds_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate medication records by (hospitalization_id, admin_dttm, med_category).

    Priority: prefer actionable MAR actions > non-zero doses > larger doses.
    Falls back to mar_action_name if mar_action_category is unavailable.
    """
    if 'mar_action_category' not in meds_df.columns:
        print('mar_action_category not available, deduping by mar_action_name instead')
        _q = """
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY
                CASE WHEN mar_action_name IS NULL THEN 10
                    WHEN regexp_matches(mar_action_name, 'verify', 'i') THEN 9
                    WHEN regexp_matches(mar_action_name, '(stopped)|(held)|(paused)|(completed)', 'i') THEN 8
                    ELSE 1 END,
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                med_dose DESC
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
    else:
        _q = """
        SELECT *
        FROM meds_df
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY hospitalization_id, admin_dttm, med_category
            ORDER BY
                CASE WHEN mar_action_category IS NULL THEN 10
                    WHEN mar_action_category in ('verify', 'not_given') THEN 9
                    WHEN mar_action_category = 'stop' THEN 8
                    WHEN mar_action_category = 'going' THEN 7
                    ELSE 1 END,
                CASE WHEN med_dose > 0 THEN 1
                    ELSE 2 END,
                med_dose DESC
        ) = 1
        ORDER BY hospitalization_id, med_category, admin_dttm;
        """
    return duckdb.sql(_q).to_df()
