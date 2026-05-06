"""Shared utility functions for the sedation pipeline."""

import duckdb
import pandas as pd
from pathlib import Path


def localize_naive_to_site_tz(s: pd.Series, site_tz: str) -> pd.Series:
    """Attach a site-tz tag to a naive datetime Series.

    Sibling to ``retag_to_local_tz`` for the OTHER tz-fix direction:
    ``retag`` is for tz-aware columns (relabels a UTC instant); this is
    for naive columns whose wall-clock is *already* in site-local time
    (clifpy's ``*.from_file`` convention) and just needs the tz tag.

    Handles DST edge cases that ``Series.dt.tz_localize`` would otherwise
    raise on:

    - **Fall-back ambiguity** (e.g., 2024-11-03 01:30 Central exists twice):
      tries ``ambiguous='infer'`` first (pandas uses neighboring timestamp
      ordering to disambiguate); falls back to ``ambiguous=False`` (= the
      LATER, post-fall-back / standard-time occurrence) if infer can't
      resolve isolated ambiguous values. The fallback is deterministic
      and matches the convention used elsewhere (e.g., DuckDB's
      ``AT TIME ZONE`` resolves fall-back ambiguity to standard time).

    - **Spring-forward gap** (e.g., 2024-03-10 02:30 Central doesn't exist):
      shifts forward to the next valid minute via ``nonexistent='shift_forward'``.

    No-op if the column is already tz-aware.
    """
    if not pd.api.types.is_datetime64_any_dtype(s) or s.dt.tz is not None:
        return s
    try:
        return s.dt.tz_localize(site_tz, ambiguous='infer', nonexistent='shift_forward')
    except Exception:
        return s.dt.tz_localize(site_tz, ambiguous=False, nonexistent='shift_forward')


def retag_to_local_tz(
    df: pd.DataFrame, columns: list[str], site_tz: str
) -> pd.DataFrame:
    """Re-tag UTC tz-aware columns as site-local tz-aware (preserves the tz
    metadata on disk; downstream consumers read pre-converted local-tz
    values without an extra ``tz_convert`` call).

    pandas ``dt.tz_convert`` is a metadata-only operation — same UTC instants,
    different display tag. pyarrow round-trips the tag through parquet, so a
    column written here as ``datetime64[us, America/Chicago]`` reads back the
    same way. Naive columns are left untouched.

    Used at parquet-write boundaries in ``01_cohort.py`` (and any other
    script that persists ``*_dttm`` columns) so ``output/{site}/*.parquet``
    files self-document their tz convention.
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None:
            df[col] = s.dt.tz_convert(site_tz)
    return df


def add_day_shift_id(
    df: pd.DataFrame,
    timestamp_name: str = "event_dttm",
    *,
    site_tz: str,
) -> pd.DataFrame:
    """Add day/shift columns (_dh, _hr, _shift, _is_day_start, _nth_day,
    _day_shift) to a DataFrame.

    Day shift: 7:00-19:00 site-local, Night shift: 19:00-7:00 site-local.
    _nth_day increments at each local 7am boundary.

    Caller contract: ``df[timestamp_name]`` is UTC tz-aware
    (``datetime64[*, UTC]``). ``site_tz`` is REQUIRED (keyword-only) — pass
    ``cfg['timezone']`` from config for clinical site-local semantics. To
    reproduce pre-`ea911a9` (UTC-hour) outputs, pass ``site_tz="UTC"``
    explicitly.

    All local-tz interpretation is done with explicit ``AT TIME ZONE`` in
    SQL, so the result is invariant under DuckDB's session timezone (any
    upstream library doing ``SET TimeZone = '...'`` cannot affect us).

    Output columns:

    - ``_dh``: naive local TIMESTAMP at the local hour-floor (derived
      bucket label — its tz is implied by the column name).
    - ``_hr``: INT, local hour 0-23.
    - ``_shift``: 'day' if ``_hr`` in [7, 19), else 'night'.
    - ``_is_day_start``: 1 at the row where ``_hr`` first crosses to 7.
    - ``_nth_day``: running count of local 7am crossings per hospitalization.
    - ``_day_shift``: e.g., ``'day1_day'`` / ``'day2_night'``.
    """
    result = duckdb.sql(f"""
        WITH local_ts AS (
            FROM df
            SELECT *
                , _dh: date_trunc('hour', {timestamp_name} AT TIME ZONE '{site_tz}')
                , _hr: extract('hour' FROM {timestamp_name} AT TIME ZONE '{site_tz}')::INT
        )
        , day_starts AS (
            FROM local_ts
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
    """).df()
    # Re-anchor the input timestamp's display tz to UTC, so the function's
    # output dtype is deterministic regardless of DuckDB's session tz at the
    # time of `.df()` (when session tz != UTC, the TIMESTAMPTZ → pandas
    # conversion tags the column with whatever session tz was set; that
    # leaks session state into the caller's frame).
    s = result[timestamp_name]
    if pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None:
        result[timestamp_name] = s.dt.tz_convert("UTC")
    return result


def remove_meds_duplicates(meds_df):
    """Deduplicate medication records by (hospitalization_id, admin_dttm, med_category).

    Priority: prefer actionable MAR actions > non-zero doses > larger doses.
    Falls back to mar_action_name if mar_action_category is unavailable.

    Accepts either a pandas ``DataFrame`` or a ``DuckDBPyRelation`` and
    returns the same type. The lazy-relation form is preferred in new code
    so callers can keep the chain unmaterialized; the pandas form is kept
    for backward compatibility with existing callers (e.g.,
    ``04_covariates.py``'s ``cont_veso_deduped`` cell).
    """
    is_relation = isinstance(meds_df, duckdb.DuckDBPyRelation)
    if 'mar_action_category' not in meds_df.columns:
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
        ORDER BY hospitalization_id, med_category, admin_dttm
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
        ORDER BY hospitalization_id, med_category, admin_dttm
        """
    rel = duckdb.sql(_q)
    return rel if is_relation else rel.to_df()


def consort_to_markdown(consort_json: dict) -> str:
    """Convert a CONSORT flow JSON to a markdown table.

    Expected JSON structure:
        {"site": "...", "steps": [{"step": N, "description": "...", "n_remaining": N, ...}, ...]}
    """
    rows = []
    rows.append("| Step | Description | Remaining | Excluded | Reason |")
    rows.append("|------|-------------|-----------|----------|--------|")
    for s in consort_json["steps"]:
        n_remaining = f"{s.get('n_remaining', ''):,}" if 'n_remaining' in s else ""
        n_excluded = f"{s.get('n_excluded', ''):,}" if 'n_excluded' in s else ""
        if 'n_patient_days_excluded' in s:
            n_excluded = f"{s['n_patient_days_excluded']:,} patient-days ({s.get('n_hospitalizations_affected', '?')} hosp)"
        reason = s.get("exclusion_reason", "")
        rows.append(f"| {s['step']} | {s['description']} | {n_remaining} | {n_excluded} | {reason} |")
    return "\n".join(rows)


def plot_consort(consort_json: dict, output_path: Path) -> None:
    """Draw a vertical CONSORT flowchart and save as PNG.

    Expected JSON structure:
        {"site": "...", "steps": [{"step": N, "description": "...", "n_remaining": N,
         "n_excluded": N, "exclusion_reason": "..."}, ...]}
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    steps = consort_json["steps"]
    n_steps = len(steps)
    fig_height = max(8, n_steps * 2.0)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_x = 0.38
    excl_x = 0.78
    box_w = 0.32
    box_h = 0.055
    y_top = 0.92
    y_bottom = 0.05
    y_spacing = (y_top - y_bottom) / max(n_steps - 1, 1)

    # Light publication-friendly palette: pale fills, dark edges, serif text
    main_color = "#e8f1fb"   # pale blue fill for main flow boxes
    main_edge = "#2874a6"    # medium blue edge
    excl_color = "#fdecea"   # pale red fill for exclusion boxes
    excl_edge = "#922b21"    # dark red edge
    start_color = "#eafaf1"  # pale green fill for starting box
    start_edge = "#1e8449"   # dark green edge
    arrow_color = "#424242"  # dark gray for arrows
    text_color = "#212121"   # near-black text

    def draw_box(x_center, y_center, w, h, text, facecolor, edgecolor, fontsize=9):
        x0 = x_center - w / 2
        y0 = y_center - h / 2
        box = mpatches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=0.008",
            facecolor=facecolor, edgecolor=edgecolor, linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x_center, y_center, text, ha="center", va="center",
                fontsize=fontsize, fontweight="normal", color=text_color,
                family="serif", wrap=True)

    def draw_arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=arrow_color,
                                    lw=1.2, mutation_scale=12))

    def _fmt(n):
        return f"{n:,}" if isinstance(n, (int, float)) else str(n)

    # Step 0
    s0 = steps[0]
    y0 = y_top
    draw_box(main_x, y0, box_w, box_h,
             f"{s0['description']}\n(n = {_fmt(s0.get('n_remaining', '?'))})",
             start_color, start_edge, fontsize=10)
    prev_y = y0

    # Steps 1+
    for i, s in enumerate(steps[1:], start=1):
        y = y_top - i * y_spacing
        draw_arrow(main_x, prev_y - box_h / 2, main_x, y + box_h / 2)

        n_rem = s.get('n_remaining')
        rem_label = f"\n(n = {_fmt(n_rem)})" if n_rem is not None else ""
        draw_box(main_x, y, box_w, box_h,
                 f"{s['description']}{rem_label}",
                 main_color, main_edge)

        n_excl = s.get('n_excluded', s.get('n_patient_days_excluded'))
        reason = s.get('exclusion_reason', '')
        if n_excl is not None:
            excl_y = (prev_y + y) / 2
            draw_arrow(main_x + box_w / 2, excl_y, excl_x - box_w / 2, excl_y)
            excl_label = "patient-days" if 'n_patient_days_excluded' in s else ""
            draw_box(excl_x, excl_y, box_w, box_h,
                     f"Excluded: {reason}\n(n = {_fmt(n_excl)} {excl_label})",
                     excl_color, excl_edge)

        prev_y = y

    site = consort_json.get("site", "")
    if site:
        ax.text(0.5, 0.985, f"CONSORT — {site}", ha="center", va="top",
                fontsize=13, fontweight="bold", color=text_color, family="serif",
                transform=ax.transAxes)

    fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
