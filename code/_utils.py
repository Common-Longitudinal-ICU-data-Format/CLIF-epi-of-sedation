"""Shared utility functions for the sedation pipeline."""

import duckdb
import pandas as pd
from pathlib import Path


def to_utc(
    df: pd.DataFrame,
    columns: "list[str] | str",
    *,
    naive_means: "str | None" = None,
) -> pd.DataFrame:
    """Ensure named columns are UTC tz-aware on the returned frame.

    Single canonical helper for tz normalization at every boundary in the
    pipeline. Replaces both legacy helpers (``retag_to_local_tz`` for the
    tz-aware → UTC metadata flip and ``localize_naive_to_site_tz`` + a
    follow-on ``tz_convert`` for the naive site-local → UTC two-step).

    Used at parquet-write boundaries (project convention: every ``*_dttm``
    column on disk is ``datetime64[us, UTC]``) and at clifpy-load boundaries
    (where ``from_file`` returns naive site-local that needs both a tz tag
    and a UTC normalization).

    Parameters
    ----------
    df : pd.DataFrame
        Returned frame is a shallow copy with the named columns replaced.
    columns : str | list[str]
        Column name(s) to normalize. Missing columns are silently skipped
        so the same call works across sites with slightly different schemas
        (e.g., ``compute_ase`` emits more event-types at UCMC than MIMIC).
    naive_means : str, optional
        REQUIRED if any named column is naive (no tz tag); names the tz
        the wall-clock is *already* in. Common values:

        - ``"UTC"`` — the naive timestamp IS UTC, just missing the tag.
        - site_tz (e.g., ``"US/Eastern"``) — clifpy's ``from_file``
          convention, where naive wall-clock is in site-local time.

        Raises ``ValueError`` if a column is naive and ``naive_means`` is
        None (defensive: silent localization to a wrong tz would shift
        every downstream UTC instant by 4-6 hours).

    DST handling for the naive→aware step (when ``naive_means`` is set):

    - **Fall-back ambiguity** (e.g., ``2024-11-03 01:30 Central`` exists
      twice): tries ``ambiguous='infer'`` first (pandas uses neighboring
      timestamp ordering to disambiguate); falls back to
      ``ambiguous=False`` (= post-fall-back / standard time) if isolated
      ambiguous values defeat ``infer``. Deterministic; matches DuckDB's
      ``AT TIME ZONE`` fall-back resolution.

    - **Spring-forward gap** (e.g., ``2024-03-10 02:30 Central`` doesn't
      exist): shifts forward to the next valid minute via
      ``nonexistent='shift_forward'``.

    Idempotent: already-UTC columns are unchanged. Non-datetime columns
    are silently skipped.
    """
    if isinstance(columns, str):
        columns = [columns]
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            continue
        if s.dt.tz is None:
            if naive_means is None:
                raise ValueError(
                    f"Column {col!r} is naive (no tz tag); pass "
                    f"naive_means=<tz> to disambiguate "
                    f"(e.g., 'UTC' or site_tz)"
                )
            try:
                s = s.dt.tz_localize(
                    naive_means,
                    ambiguous='infer',
                    nonexistent='shift_forward',
                )
            except Exception:
                s = s.dt.tz_localize(
                    naive_means,
                    ambiguous=False,
                    nonexistent='shift_forward',
                )
        if str(s.dt.tz) != 'UTC':
            s = s.dt.tz_convert('UTC')
        df[col] = s
    return df


def compute_weight_qc_exclusions(
    weight_rel,
    hosp_ids,
    *,
    clamp_lo: float = 30.0,
    clamp_hi: float = 300.0,
    max_jump_kg: float = 20.0,
    max_jump_hours: float = 24.0,
    max_range_kg: float = 30.0,
    range_rule_on: bool = False,
) -> dict:
    """Compute weight-QC exclusion sets in-memory (no parquet round-trip).

    Replaces the 2-pass `make run`-`make weight-audit`-`make run` dance with
    a single-pass in-cohort computation. Same three criteria as the original
    `code/qc/weight_audit.py` `section_g_drop_list`:

    1. **zero_weight** — cohort hosps with no weight rows surviving the
       [clamp_lo, clamp_hi] clamp. Catches both "no weight ever charted"
       AND "all weights are clearly garbage" (e.g., lb-vs-kg unit errors).
    2. **jump** — hosps with any consecutive pair of weight readings
       differing by > ``max_jump_kg`` within ``max_jump_hours``. Raw jump
       (not normalized rate) — normalizing inflates short-gap recordings
       and misclassifies them as outliers.
    3. **range** (opt-in via ``range_rule_on``) — hosps whose min-max
       weight spread within the stay exceeds ``max_range_kg``.

    Criteria are incremental: ``jump`` excludes hosps already in
    ``zero_weight``; ``range`` excludes hosps in either prior set.

    Implementation is DuckDB-native (avoids materializing the full
    cohort-wide weight frame to pandas). Defaults match the env-var
    defaults in ``code/qc/weight_audit.py``.

    Parameters
    ----------
    weight_rel : duckdb.DuckDBPyRelation
        Raw weight events with columns ``hospitalization_id``,
        ``recorded_dttm``, ``weight_kg_raw``. Caller is responsible for
        the source filter (e.g., ``vital_category = 'weight_kg'``).
    hosp_ids : iterable of str
        Cohort hospitalization IDs. ``zero_weight`` is computed against
        this set.

    Returns
    -------
    dict with keys:
        ``zero_weight``, ``jump``, ``range`` — sets of hospitalization_id;
        ``jump_threshold_str``, ``range_threshold_str`` — reason strings
        suitable for CONSORT label / parquet ``_drop_reason`` columns
        (range_threshold_str is None when ``range_rule_on=False``).
    """
    duckdb.execute(f"""
        CREATE OR REPLACE TEMP TABLE _wqc_clamped AS
        FROM weight_rel
        SELECT hospitalization_id, recorded_dttm, weight_kg_raw
        WHERE weight_kg_raw BETWEEN {clamp_lo} AND {clamp_hi}
    """)
    _has_weight_ids = {
        r[0]
        for r in duckdb.sql(
            "FROM _wqc_clamped SELECT DISTINCT hospitalization_id"
        ).fetchall()
    }
    zero_weight = set(hosp_ids) - _has_weight_ids

    _jump_rows = duckdb.sql(f"""
        WITH ordered AS (
            FROM _wqc_clamped
            SELECT hospitalization_id, recorded_dttm, weight_kg_raw
                , prev_w: LAG(weight_kg_raw) OVER (
                    PARTITION BY hospitalization_id ORDER BY recorded_dttm)
                , prev_t: LAG(recorded_dttm) OVER (
                    PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        )
        , pairs AS (
            FROM ordered
            SELECT hospitalization_id
                , jump_kg: abs(weight_kg_raw - prev_w)
                , dt_hr: epoch(recorded_dttm - prev_t) / 3600.0
            WHERE prev_w IS NOT NULL
        )
        , per_hosp AS (
            FROM pairs
            SELECT hospitalization_id, max_jump: MAX(jump_kg)
            WHERE dt_hr < {max_jump_hours}
            GROUP BY hospitalization_id
        )
        FROM per_hosp
        SELECT hospitalization_id
        WHERE max_jump > {max_jump_kg}
    """).fetchall()
    jump = {r[0] for r in _jump_rows} - zero_weight

    range_set: set = set()
    if range_rule_on:
        _range_rows = duckdb.sql(f"""
            FROM _wqc_clamped
            SELECT hospitalization_id
            GROUP BY hospitalization_id
            HAVING MAX(weight_kg_raw) - MIN(weight_kg_raw) > {max_range_kg}
        """).fetchall()
        range_set = {r[0] for r in _range_rows} - zero_weight - jump

    duckdb.execute("DROP TABLE IF EXISTS _wqc_clamped")

    return {
        'zero_weight': zero_weight,
        'jump': jump,
        'range': range_set,
        'jump_threshold_str': (
            f"jump_gt_{int(max_jump_kg)}kg_within_{int(max_jump_hours)}h"
        ),
        'range_threshold_str': (
            f"range_gt_{int(max_range_kg)}kg" if range_rule_on else None
        ),
    }


def mar_action_zero_dose_sql(has_category: bool) -> str:
    """Return the SQL fragment that zeros out doses for stop/not_given MAR actions.

    Used in pivot CASE expressions like::

        , med_dose: CASE WHEN {mar_action_zero_dose_sql(...)} THEN 0 ELSE med_dose END

    Asymmetric contract — see B5 in the audit plan:
    - `medication_admin_continuous` MAY lack ``mar_action_category``; this
      helper provides a ``mar_action_name`` regex fallback. FFILL semantics
      are forgiving so a missed stop event over-FFILLs by minutes at worst.
    - `medication_admin_intermittent` MUST have ``mar_action_category`` —
      bolus dose accounting is too sensitive to the ``not_given`` zeroing
      for the regex fallback to be safe. Callers should assert presence at
      load time rather than using this helper for intermittent data.

    Parameters
    ----------
    has_category : bool
        True if ``mar_action_category`` exists on the source table.
    """
    if has_category:
        return "mar_action_category IN ('stop', 'not_given')"
    # Regex fallback over mar_action_name. Matches the dedup priority in
    # remove_meds_duplicates() (regex on mar_action_name when category is
    # missing). `[\s_-]*` allows variants like 'not_given', 'not given',
    # 'not-given'. COALESCE ensures NULL mar_action_name doesn't blow the
    # regex up.
    return (
        "regexp_matches(COALESCE(mar_action_name, ''), "
        "'(stopped)|(held)|(paused)|(not[\\s_-]*given)', 'i')"
    )


def mar_action_not_given_filter_sql(has_category: bool) -> str:
    """Return the WHERE fragment that EXCLUDES not_given rows from a load.

    Used in inline WHERE clauses like::

        WHERE ... AND {mar_action_not_given_filter_sql(...)}

    Same asymmetric contract as ``mar_action_zero_dose_sql``: regex fallback
    for continuous when ``mar_action_category`` is absent; intermittent
    callers should assert column presence rather than relying on this.
    """
    if has_category:
        return "mar_action_category != 'not_given'"
    return (
        "NOT regexp_matches(COALESCE(mar_action_name, ''), "
        "'not[\\s_-]*given', 'i')"
    )


def normalize_categories(data, columns):
    """Lowercase + strip-whitespace the named CLIF ``_category`` columns.

    Single canonical helper for category normalization at every CLIF load
    boundary. UCMC and MIMIC happen to deliver categories already
    lowercased today, but a new CLIF site whose loader leaves
    ``device_category = 'IMV'`` or ``mode_category = 'Pressure
    Support/CPAP'`` would silently return zero IMV streaks / zero SBTs
    from every downstream SQL that hardcodes lowercase literals (e.g.,
    ``device_category = 'imv'``). Calling this at load makes the
    downstream SQL deterministic across sites.

    Polymorphic: pandas DataFrame in → pandas DataFrame out;
    ``DuckDBPyRelation`` in → ``DuckDBPyRelation`` out (lazy). Missing
    columns are silently skipped so the same call works across sites
    with slightly different schemas (e.g., ``mar_action_category`` is
    optional on ``medication_admin_continuous``).

    Mirrors the canonical pattern at
    ``CLIF-eligibility-for-mobilization/code/01_cohort_identification.py:340-341``
    and ``sofa_score.py:365-366``.

    Parameters
    ----------
    data : pd.DataFrame or duckdb.DuckDBPyRelation
        Input table. Returned frame/relation is the same type.
    columns : str or list[str]
        Column name(s) to normalize. Missing columns are silently
        skipped (no error). Each named column is lowercased AND has
        leading/trailing whitespace stripped — both fixes for real-world
        CLIF data quirks.
    """
    if isinstance(columns, str):
        columns = [columns]
    is_relation = isinstance(data, duckdb.DuckDBPyRelation)
    if is_relation:
        present = [c for c in columns if c in data.columns]
        if not present:
            return data
        replace_sql = ", ".join(
            f"TRIM(LOWER({c})) AS {c}" for c in present
        )
        return duckdb.sql(f"FROM data SELECT * REPLACE ({replace_sql})")
    # pandas path. NB: don't promote to nullable StringDtype here — that
    # turns NaN into pd.NA, which propagates through downstream
    # `.eq(...).to_numpy()` chains as an object array, breaking any
    # subsequent `.any()` / `np.argmax` etc. with "boolean value of NA is
    # ambiguous." See 03_outcomes.py:433 (count_intubations_v2). Operate
    # on the existing dtype so NaN stays NaN.
    df = data.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = df[col]
        if s.dtype == "object" or pd.api.types.is_string_dtype(s):
            df[col] = s.str.strip().str.lower()
    return df


def add_dh_hr(
    data,
    timestamp_col: str = "event_dttm",
    *,
    site_tz: str,
):
    """Add `_dh` (local-tz hour-floor) and `_hr` (local hour-of-day, 0-23)
    columns to ``data`` and return the same container type.

    Single source of truth for `_dh` / `_hr` derivation across the
    pipeline. Used by ``add_day_shift_id`` for the cohort grid and
    directly by admin-side cells in ``02_exposure.py`` that need the same
    columns without the shift-id assignment.

    Polymorphic: pandas DataFrame in → pandas DataFrame out;
    ``DuckDBPyRelation`` in → ``DuckDBPyRelation`` out (lazy). DuckDB's
    replacement scan resolves ``data`` in Python scope, so the SQL
    references the input by name without an explicit register call.

    Caller contract: ``data[timestamp_col]`` is UTC tz-aware
    (``datetime64[*, UTC]`` or TIMESTAMPTZ tagged UTC). ``site_tz`` is
    REQUIRED (keyword-only) — pass ``cfg['timezone']`` from config for
    clinical site-local semantics. To reproduce pre-`ea911a9` (UTC-hour)
    outputs, pass ``site_tz="UTC"`` explicitly.

    All local-tz interpretation is done with explicit ``AT TIME ZONE`` in
    SQL, so the result is invariant under DuckDB's session timezone (any
    upstream library doing ``SET TimeZone = '...'`` cannot affect us).
    """
    rel = duckdb.sql(f"""
        FROM data
        SELECT *
            , _dh: date_trunc('hour', {timestamp_col} AT TIME ZONE '{site_tz}')
            , _hr: extract('hour' FROM {timestamp_col} AT TIME ZONE '{site_tz}')::INT
    """)
    if not isinstance(data, pd.DataFrame):
        return rel
    # Re-anchor the timestamp column's display tz to UTC, so the function's
    # output dtype is deterministic regardless of DuckDB's session tz at
    # the `.df()` boundary (when session tz != UTC, the TIMESTAMPTZ →
    # pandas conversion tags the column with whatever session tz was set;
    # that leaks session state into the caller's frame).
    result = rel.df()
    s = result[timestamp_col]
    if pd.api.types.is_datetime64_any_dtype(s) and s.dt.tz is not None:
        result[timestamp_col] = s.dt.tz_convert("UTC")
    return result


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

    Composes ``add_dh_hr`` for the `_dh`/`_hr` step and then assigns the
    shift-id columns on top via window functions. The `_dh` / `_hr`
    formula lives in ``add_dh_hr`` and only there.

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
    with_dh = add_dh_hr(df, timestamp_name, site_tz=site_tz)
    result = duckdb.sql(f"""
        WITH day_starts AS (
            FROM with_dh
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
