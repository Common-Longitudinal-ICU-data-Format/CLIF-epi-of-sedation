"""Shared utility functions for the sedation pipeline."""

import duckdb
import pandas as pd
from pathlib import Path


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
