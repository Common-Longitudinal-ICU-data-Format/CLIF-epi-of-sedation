"""Single-patient trajectory QC dashboard (Plotly Dash) — Phase 2 (debugged).

Interactive viewer for reviewing one hospitalization_id at a time with the
full hospitalization in view, including events outside the cohort window
(re-intubations, discharge). Panels:

    1. Sedatives    (exposure: propofol / fent-eq / midaz-eq)
    2. Resp support (outcome:  FiO2, PEEP + device ribbon — IMV blocks are
                     sub-colored by ventilator mode_category)
    3. Pressors     (confounder: NEE + components)
    4. Assessments  (contextual: RASS + GCS) — off by default
    5. Vitals       (contextual: HR, MAP, SpO2)

All trace y-values are normalized per-trace into [0, 1] of their panel so
panels with traces of very different absolute scales (FiO2 vs PEEP, NEE vs
raw norepi) are immediately readable. Y-axis tick labels are hidden;
hovering a point reveals the original raw value.

Boundaries between cohort hours and the excluded day-0 / last-day windows
are marked with dashed vertical lines (no colored zones, to keep the
night-shift grey shading dominant).

Run:
    make qc SITE=mimic
    # or: uv run python code/qc/trajectory_viewer.py
"""
from __future__ import annotations

import random
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import ALL, ctx, Input, Output, State, dash_table, dcc, html, no_update
from plotly.subplots import make_subplots

# Local helpers.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).parent))
from _shared import (  # noqa: E402
    ASSESSMENT_COLORS,
    DAY_START_HOUR,
    DEVICE_CATEGORY_COLORS,
    EVENT_COLORS,
    HOVER_WINDOW_MIN,
    MODE_IN_IMV_COLORS,
    NIGHT_SHIFT_COLOR,
    PRESSOR_COLORS,
    RESP_COLORS,
    SEDATIVE_COLORS,
    VITAL_COLORS,
    PatientEnrichment,
    cohort_excluded_zones,
    compute_nee,
    day_labels_for_cohort,
    extract_events,
    get_enrichment,
    get_wide_df,
    list_sites,
    night_windows_in_range,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"


# ── Panel metadata ─────────────────────────────────────────────────────
PANELS = [
    {"id": "sedatives",   "title": "Sedatives",   "subtitle": "(mg/hr or mcg/kg/min)", "height": 0.22, "default": True},
    {"id": "resp",        "title": "Resp support", "subtitle": "(FiO₂, PEEP, mode/device)", "height": 0.22, "default": True},
    {"id": "pressors",    "title": "Pressors",     "subtitle": "(NEE, mcg/kg/min)", "height": 0.18, "default": True},
    {"id": "assessments", "title": "Assessments",  "subtitle": "(RASS, GCS)", "height": 0.18, "default": False},
    {"id": "vitals",      "title": "Vitals",       "subtitle": "(HR, MAP, SpO₂)", "height": 0.20, "default": True},
]
PANEL_BY_ID = {p["id"]: p for p in PANELS}


# ── Cohort pool discovery ───────────────────────────────────────────────

def cohort_ids_for_site(site: str) -> list[str]:
    path = OUTPUT_DIR / site / "analytical_dataset.parquet"
    if not path.exists():
        return []
    hids = pd.read_parquet(path, columns=["hospitalization_id"])["hospitalization_id"]
    return sorted(hids.unique().tolist())


# ── Dash app scaffold ──────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="CLIF Sedation — Patient QC",
    suppress_callback_exceptions=True,
)

_SITES = list_sites()
_DEFAULT_SITE = _SITES[0] if _SITES else None


def _panel_checklist() -> dcc.Checklist:
    return dcc.Checklist(
        id="panel-checklist",
        options=[{"label": f" {p['title']}", "value": p["id"]} for p in PANELS],
        value=[p["id"] for p in PANELS if p["default"]],
        inline=True,
        persistence=True,
        persistence_type="local",
        labelStyle={"marginRight": "16px", "fontSize": "13px"},
        className="mb-2",
    )


# Bottom drawer: always-visible thin tab + collapsible body. Ctrl+` toggles.
_BOTTOM_DRAWER_STYLE = {
    "position": "fixed", "bottom": "0", "left": "0", "right": "0",
    "backgroundColor": "white", "borderTop": "1px solid #ccc",
    "zIndex": 1030, "boxShadow": "0 -2px 8px rgba(0,0,0,0.06)",
}


app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H4("CLIF Sedation — Patient Trajectory QC", className="mt-2 mb-0"), width=12),
        dbc.Col(html.Small(
            "Pick a site, paste a hospitalization_id (or sample 5), pick panels, then click Load. "
            "Scroll to zoom · drag to pan · Ctrl+` toggles the table drawer.",
            className="text-muted",
        ), width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Label("Site", className="fw-semibold"),
            dcc.Dropdown(id="site-dropdown",
                options=[{"label": s, "value": s} for s in _SITES],
                value=_DEFAULT_SITE, clearable=False,
                style={"width": "160px"}),
        ], width="auto"),
        dbc.Col([
            html.Label("Hospitalization ID", className="fw-semibold"),
            dcc.Input(id="hosp-id-input", type="text",
                placeholder="type or click a sampled ID",
                style={"width": "260px"}, debounce=True),
        ], width="auto"),
        dbc.Col([
            html.Label(".", className="fw-semibold text-white"), html.Br(),
            dbc.Button("Sample random 5", id="sample-btn",
                color="secondary", outline=True, size="sm"),
        ], width="auto"),
        dbc.Col([
            html.Label(".", className="fw-semibold text-white"), html.Br(),
            dbc.Button("Load", id="load-btn", color="primary", size="sm"),
        ], width="auto"),
        dbc.Col([
            html.Label("Sampled IDs", className="fw-semibold"),
            html.Div(id="sample-chips", style={"fontFamily": "monospace", "fontSize": "12px"}),
        ], width=True),
    ], className="my-3 align-items-end g-2"),

    html.Div(id="summary-strip", className="mb-2"),

    dbc.Row([
        dbc.Col([html.Small("Show panels:", className="fw-semibold text-muted me-2")], width="auto"),
        dbc.Col(_panel_checklist(), width=True),
    ], className="g-2 align-items-center"),

    dbc.Row([
        dbc.Col(html.Small(
            id="pin-status",
            children="Hover the plot to filter the table · click to pin a cursor",
            className="text-muted",
        ), width=True),
    ], className="g-2 align-items-center mb-1"),

    dcc.Loading(
        id="plot-loading", type="default",
        children=dcc.Graph(
            id="timeline-graph",
            config={"displaylogo": False, "scrollZoom": True},
            style={"height": "780px"},
        ),
    ),

    # Spacer so the fixed bottom drawer never covers content above.
    html.Div(style={"height": "60px"}),

    # Fixed-bottom drawer (VS Code-style). Tab strip is always visible.
    html.Div([
        # Always-visible tab strip
        dbc.Button(
            [
                html.Span("▸ ", id="collapse-caret", style={"display": "inline-block", "width": "14px"}),
                html.Span("Linked rows (±4 h)", className="me-2"),
                html.Small("Ctrl+` to toggle · hover plot to filter",
                           className="text-muted"),
            ],
            id="collapse-btn",
            color="link", size="sm",
            className="text-decoration-none w-100 text-start py-2 px-3",
            style={"borderRadius": 0, "fontSize": "13px"},
        ),
        dbc.Collapse(
            html.Div(
                dash_table.DataTable(
                    id="linked-table",
                    page_size=50,
                    page_action="native",
                    # NOTE: virtualization + fixed_rows render zero rows when
                    # the parent dbc.Collapse is hidden on first paint (the
                    # inner table measures height as 0 and never recovers).
                    # Native pagination + vertical scroll is more robust.
                    style_table={"overflowX": "auto", "overflowY": "auto", "maxHeight": "40vh"},
                    style_cell={"fontFamily": "monospace", "fontSize": "12px", "padding": "4px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#f5f5f5",
                                  "position": "sticky", "top": 0, "zIndex": 1},
                ),
                style={"padding": "0 8px 8px 8px"},
            ),
            id="linked-table-collapse",
            is_open=False,
        ),
    ], style=_BOTTOM_DRAWER_STYLE),

    # Hidden state + clientside-callback plumbing
    dcc.Store(id="loaded-state"),
    # Pinned-cursor state: when the user clicks the plot (or a table row),
    # the vertical cursor "freezes" at that timestamp and the table stops
    # updating on hover. None = unpinned, hover-driven (default).
    dcc.Store(id="pinned-cursor", data=None),
    dcc.Store(id="keyboard-init-store"),
    html.Div(id="keyboard-init-output", style={"display": "none"}),
], fluid=True)


# ── Sampling callbacks ─────────────────────────────────────────────────

@app.callback(
    Output("sample-chips", "children"),
    Output("hosp-id-input", "value"),
    Input("sample-btn", "n_clicks"),
    State("site-dropdown", "value"),
    prevent_initial_call=True,
)
def sample_ids(_n_clicks, site):
    if not site:
        return "no site available", no_update
    pool = cohort_ids_for_site(site)
    if not pool:
        return f"no cohort found for {site}", no_update
    picks = random.sample(pool, min(5, len(pool)))
    chips = [
        dbc.Badge(
            pid, id={"type": "sample-chip", "idx": i}, n_clicks=0,
            color="light", text_color="dark", pill=True,
            className="me-1", style={"cursor": "pointer", "border": "1px solid #ddd"},
        )
        for i, pid in enumerate(picks)
    ]
    return chips, picks[0]


@app.callback(
    Output("hosp-id-input", "value", allow_duplicate=True),
    Input({"type": "sample-chip", "idx": ALL}, "n_clicks"),
    State({"type": "sample-chip", "idx": ALL}, "children"),
    prevent_initial_call=True,
)
def pick_sampled_chip(n_clicks_list, chip_children):
    if not any(n_clicks_list or []):
        return no_update
    triggered = ctx.triggered_id
    if triggered is None:
        return no_update
    return chip_children[triggered["idx"]]


# ── Core load callback ────────────────────────────────────────────────

@app.callback(
    Output("timeline-graph", "figure"),
    Output("summary-strip", "children"),
    Output("loaded-state", "data"),
    Output("linked-table", "data"),
    Output("linked-table", "columns"),
    Output("linked-table-collapse", "is_open"),
    Output("collapse-caret", "children"),
    Output("pinned-cursor", "data"),
    Input("load-btn", "n_clicks"),
    Input("hosp-id-input", "n_submit"),
    Input("panel-checklist", "value"),
    State("site-dropdown", "value"),
    State("hosp-id-input", "value"),
    State("loaded-state", "data"),
    prevent_initial_call=True,
)
def load_patient(_n_load, _n_submit, visible_panels, site, hosp_id, prior_state):
    trigger = ctx.triggered_id
    if trigger == "panel-checklist":
        if not prior_state or not prior_state.get("hosp_id"):
            return (no_update,) * 8
        site = prior_state.get("site", site)
        hosp_id = prior_state.get("hosp_id", hosp_id)

    if not site or not hosp_id:
        return _empty_fig("Select a site and enter a hospitalization_id."), None, None, [], [], False, "▸ ", None

    hosp_id = str(hosp_id).strip()
    pool = cohort_ids_for_site(site)
    if hosp_id not in set(pool):
        msg = f"ID {hosp_id} is not in the {site} analytical cohort."
        return _empty_fig(msg), dbc.Alert(msg, color="warning", className="py-1"), None, [], [], False, "▸ ", None

    enr = get_enrichment(site, hosp_id)
    try:
        wide_df = get_wide_df(site, hosp_id)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load wide dataset: {exc}"
        return _empty_fig(msg), dbc.Alert(msg, color="danger", className="py-1"), None, [], [], False, "▸ ", None

    if wide_df.empty:
        msg = f"No wide-dataset rows returned for {hosp_id}."
        return _empty_fig(msg), dbc.Alert(msg, color="warning", className="py-1"), None, [], [], False, "▸ ", None

    events = extract_events(enr)
    fig = build_timeline(wide_df, enr, events, visible_panels or [])
    summary = build_summary_strip(enr, wide_df)

    table_df = _linked_table_source(wide_df, enr)
    columns, records = _table_spec(table_df)
    state = {"site": site, "hosp_id": hosp_id}
    # Collapse the drawer + clear any pinned cursor on fresh patient load.
    return fig, summary, state, records, columns, False, "▸ ", None


# ── Drawer toggle (button click) + hover auto-expand ───────────────────

@app.callback(
    Output("linked-table-collapse", "is_open", allow_duplicate=True),
    Output("collapse-caret", "children", allow_duplicate=True),
    Input("collapse-btn", "n_clicks"),
    State("linked-table-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_collapse(_n, is_open):
    new_state = not is_open
    return new_state, ("▾ " if new_state else "▸ ")


def _ts_to_naive(x: object) -> pd.Timestamp | None:
    """Robustly parse an x-value (from hover/click/store) to a naive ts."""
    if x is None:
        return None
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts


def _table_window_subset(table_df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Return rows whose event_time is within ±HOVER_WINDOW_MIN of `ts`."""
    if "event_time" not in table_df.columns or table_df.empty:
        return table_df.iloc[0:0]
    time_col = pd.to_datetime(table_df["event_time"])
    if time_col.dt.tz is not None:
        time_col = time_col.dt.tz_localize(None)
    window = pd.Timedelta(minutes=HOVER_WINDOW_MIN)
    mask = (time_col >= ts - window) & (time_col <= ts + window)
    return table_df.loc[mask]


@app.callback(
    Output("linked-table", "style_data_conditional"),
    Output("linked-table", "page_current"),
    Output("pin-status", "children"),
    Input("pinned-cursor", "data"),
    State("linked-table", "data"),
    State("linked-table", "page_size"),
    prevent_initial_call=True,
)
def update_pin_state(pinned_ts, table_records, page_size):
    """When the cursor is pinned, highlight the closest row in the
    (always-full) table and jump the table to that row's page so it's
    visible. When unpinned, clear styling.

    The table data itself is rendered once at patient load and never
    changes — this is an audit view. Hovering the plot only moves the
    visual cursor; it doesn't filter rows anymore.
    """
    if not pinned_ts:
        return [], no_update, "Hover the plot to read values · click to pin a cursor"

    ts = _ts_to_naive(pinned_ts)
    if ts is None or not table_records:
        return [], no_update, no_update

    # Find the table row whose event_time is closest to the pin.
    diffs = []
    for i, rec in enumerate(table_records):
        et = rec.get("event_time")
        if not et:
            continue
        try:
            t = pd.Timestamp(et)
            if t.tzinfo is not None:
                t = t.tz_localize(None)
            diffs.append((i, abs((t - ts).total_seconds())))
        except (ValueError, TypeError):
            continue
    if not diffs:
        return [], no_update, no_update
    best_idx, _ = min(diffs, key=lambda x: x[1])

    style_cond = [{
        "if": {"row_index": best_idx},
        "backgroundColor": "#fff3e0",
        "fontWeight": "bold",
    }]
    page_current = best_idx // (page_size or 50)

    status = html.Span([
        html.Strong("📌 PINNED ", style={"color": "#e6550d"}),
        f"at {ts.strftime('%Y-%m-%d %H:%M')} · ",
        html.Span("click the plot again to unpin · table jumped to closest row",
                  className="text-muted"),
    ])
    return style_cond, page_current, status


# NOTE: The previous "auto-open drawer on first hover" callback was removed
# (2026-04-26) at user request — it kept popping the table over the plot
# whenever the cursor passed over a panel, covering half the screen.
# The drawer now only opens via the visible toggle button or Ctrl+` hotkey.


# ── Click-to-pin: clicking the plot freezes the cursor at that x ──────

@app.callback(
    Output("pinned-cursor", "data", allow_duplicate=True),
    Input("timeline-graph", "clickData"),
    State("pinned-cursor", "data"),
    prevent_initial_call=True,
)
def toggle_pin_on_plot_click(click_data, current_pin):
    """Click the plot to pin the cursor. Click again (or unpin via the
    table-row click) to clear. If you click at the *same* x that's
    already pinned, treat it as a toggle-off."""
    if not click_data:
        return no_update
    try:
        x = click_data["points"][0]["x"]
    except (KeyError, IndexError):
        return no_update
    if current_pin == x:
        return None
    return x


# ── Table-row click: pin the cursor at that row's event_time ──────────

@app.callback(
    Output("pinned-cursor", "data", allow_duplicate=True),
    Input("linked-table", "active_cell"),
    State("linked-table", "data"),
    State("linked-table", "page_current"),
    State("linked-table", "page_size"),
    prevent_initial_call=True,
)
def pin_from_table_click(active_cell, table_data, page_current, page_size):
    if not active_cell or not table_data:
        return no_update
    page_current = page_current or 0
    page_size = page_size or 50
    # Row-index from active_cell is page-relative when paginated
    row_idx_local = active_cell.get("row", 0)
    row_idx = row_idx_local + page_current * page_size
    if row_idx >= len(table_data):
        return no_update
    et = table_data[row_idx].get("event_time")
    return et or no_update


# ── Pin marker: draw / remove the pinned vline on the figure ──────────

@app.callback(
    Output("timeline-graph", "figure", allow_duplicate=True),
    Input("pinned-cursor", "data"),
    State("timeline-graph", "figure"),
    prevent_initial_call=True,
)
def update_pin_marker(pinned_ts, fig):
    """Add or remove the pinned-cursor shape on the plot. We tag the
    shape with `name='__pinned_cursor'` so it can be replaced atomically
    without touching the other (~270) figure shapes."""
    if not fig:
        return no_update
    layout = fig.setdefault("layout", {})
    shapes = [s for s in (layout.get("shapes") or [])
              if (s.get("name") if isinstance(s, dict) else None) != "__pinned_cursor"]
    if pinned_ts:
        shapes.append({
            "type": "line",
            "xref": "x", "yref": "paper",
            "x0": pinned_ts, "x1": pinned_ts,
            "y0": 0, "y1": 1,
            "line": {"color": "#e6550d", "width": 2.5},
            "layer": "above",
            "name": "__pinned_cursor",
        })
    layout["shapes"] = shapes
    return fig


# ── Ctrl+` clientside hotkey ──────────────────────────────────────────
# Registers a single document-level keydown listener on first load. When
# Ctrl+` (Backquote) fires, programmatically clicks the drawer toggle.
app.clientside_callback(
    """
    function(_) {
        if (!window._qcKeyInit) {
            window._qcKeyInit = true;
            document.addEventListener('keydown', function(e) {
                if (e.ctrlKey && (e.key === '`' || e.code === 'Backquote')) {
                    e.preventDefault();
                    const btn = document.getElementById('collapse-btn');
                    if (btn) btn.click();
                }
            });
        }
        return '';
    }
    """,
    Output("keyboard-init-output", "children"),
    Input("keyboard-init-store", "data"),
    prevent_initial_call=False,
)


# ── Trace normalization helper ────────────────────────────────────────

def _normalize(y: pd.Series) -> tuple[pd.Series, pd.Series, float]:
    """Return (y_normalized_to_[0,1], y_raw_for_hover, max_value).

    Each trace is divided by its own absolute max so curves with very
    different scales (FiO2 0–1 vs PEEP 5–20; NEE 0.05 vs raw norepi) all
    live in the same vertical band of their panel.
    """
    raw = y
    if raw.empty or not raw.notna().any():
        return raw, raw, 0.0
    m = float(np.nanmax(np.abs(raw.to_numpy())))
    if m == 0 or not np.isfinite(m):
        return raw, raw, 0.0
    return raw / m, raw, m


# ── Figure construction ───────────────────────────────────────────────

def build_timeline(
    wide_df: pd.DataFrame,
    enr: PatientEnrichment,
    events: pd.DataFrame,
    visible_panels: list[str],
) -> go.Figure:
    ordered = [p for p in PANELS if p["id"] in set(visible_panels)]
    if not ordered:
        return _empty_fig("All panels hidden. Check at least one above.")

    row_heights = [p["height"] for p in ordered]
    total = sum(row_heights) or 1.0
    row_heights = [h / total for h in row_heights]
    panel_to_row = {p["id"]: i + 1 for i, p in enumerate(ordered)}

    fig = make_subplots(
        rows=len(ordered), cols=1,
        shared_xaxes=True, vertical_spacing=0.035,
        row_heights=row_heights,
    )

    cohort_start, cohort_end = enr.cohort_start, enr.cohort_end

    # ── Background: night-shift grey, drawn per visible panel so each
    # panel's shape stays bound to its own xref/yref. Higher opacity
    # (0.55) so the grey actually reads against simple_white's bg.
    # NOTE: use `add_shape` (not `add_vrect`) — `add_vrect` with
    # `row=N, col=N` is silently a no-op in current Plotly versions.
    plot_start = wide_df["event_time"].min() if "event_time" in wide_df.columns else cohort_start
    plot_end = wide_df["event_time"].max() if "event_time" in wide_df.columns else cohort_end
    night_pairs = night_windows_in_range(plot_start, plot_end)
    for r in range(1, len(ordered) + 1):
        for n_start, n_end in night_pairs:
            fig.add_shape(
                type="rect", xref="x", yref="y domain",
                x0=n_start, x1=n_end, y0=0, y1=1,
                fillcolor=NIGHT_SHIFT_COLOR, opacity=0.55,
                layer="below", line_width=0,
                row=r, col=1,
            )

    # ── Filter-boundary dashed vlines: cohort_start, day-0→day-1, last
    # 7 AM, cohort_end. These mark the analytical-filter boundaries
    # without colored shading. cohort_start/end already have
    # intubation/extubation event vlines so we only emit the two
    # internal boundaries (day 0 → day 1 and N-1 → last day).
    for z_start, z_end, label in cohort_excluded_zones(cohort_start, cohort_end):
        # The "interior" boundary is the end of the day-0 zone (= first
        # 7 AM after intub) and the start of the last-day zone (= last
        # 7 AM before extub). Drawing both edges gives the reviewer a
        # clear "this is where the analytical filter starts/ends keeping
        # data" cue.
        if label == "day 0":
            x = z_end
        else:  # "last day"
            x = z_start
        for r in range(1, len(ordered) + 1):
            fig.add_shape(
                type="line", xref="x", yref="y domain",
                x0=x, x1=x, y0=0, y1=1,
                line={"color": "#b97700", "width": 1.0, "dash": "dot"},
                layer="above",
                row=r, col=1,
            )
        # Annotate top edge of plot once per boundary (yref=paper).
        fig.add_annotation(
            x=x, y=1.005, xref="x", yref="paper",
            text=("d0|d1" if label == "day 0" else "lastday"),
            showarrow=False,
            font={"size": 8, "color": "#b97700"},
            xanchor="center",
        )

    # ── Panel traces ──────────────────────────────────────────────────
    if "sedatives" in panel_to_row:
        _draw_sedatives(fig, wide_df, enr, row=panel_to_row["sedatives"])
    if "resp" in panel_to_row:
        _draw_resp(fig, wide_df, row=panel_to_row["resp"])
    if "pressors" in panel_to_row:
        _draw_pressors(fig, wide_df, row=panel_to_row["pressors"])
    if "assessments" in panel_to_row:
        _draw_assessments(fig, wide_df, row=panel_to_row["assessments"])
    if "vitals" in panel_to_row:
        _draw_vitals(fig, wide_df, row=panel_to_row["vitals"])

    _draw_event_vlines(fig, events, n_rows=len(ordered))

    # _nth_day labels along top edge
    for ts_center, nth_day in day_labels_for_cohort(cohort_start, cohort_end):
        fig.add_annotation(
            x=ts_center, y=1.025, xref="x", yref="paper",
            text=f"d{nth_day}", showarrow=False,
            font={"size": 9, "color": "dimgray"},
        )

    # Night-shift legend entry (single row, since cohort/excluded shading
    # is now line-based and doesn't need a swatch).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker={"size": 10, "color": NIGHT_SHIFT_COLOR, "symbol": "square",
                "line": {"color": "#bbb", "width": 0.5}},
        name="night shift (7 PM – 7 AM)",
        legendgroup="zones", legendgrouptitle_text="Zones",
        showlegend=True,
    ))

    # ── Layout: hide y-axis tick labels; set rotated row titles via
    # annotations on the left margin. Per-trace normalization makes [0, 1]
    # the natural range for every panel.
    #
    # `fixedrange=True` on every y-axis is the canonical Plotly idiom for
    # "horizontal-only zoom" — it disables scroll-zoom, box-zoom, and
    # double-click rescale on y while leaving x fully interactive. Without
    # this, scroll-wheel zoom would silently rescale y to fit data and
    # break the lower-bound-at-0 invariant on the sedative panel (and
    # every other normalized panel).
    for i, p in enumerate(ordered, start=1):
        fig.update_yaxes(
            showticklabels=False,
            showgrid=False, zeroline=False,
            range=[0, 1.05] if p["id"] != "assessments" else None,
            fixedrange=True,
            row=i, col=1,
        )
        # Compact left-margin label (panel title + subtitle on two lines).
        # Positioned via annotation rather than `title_text` so we have
        # full control of placement.
        yaxis_id = "yaxis" if i == 1 else f"yaxis{i}"
        ydomain = fig.layout[yaxis_id].domain or (0, 1)
        y_center = (ydomain[0] + ydomain[1]) / 2
        fig.add_annotation(
            x=-0.005, y=y_center, xref="paper", yref="paper",
            text=f"<b>{p['title']}</b><br><span style='font-size:9px;color:#888'>{p['subtitle']}</span>",
            showarrow=False, xanchor="right", yanchor="middle",
            font={"size": 11},
        )

    # Rangeselector on the TOP rendered panel
    rangeselector_cfg = {
        "buttons": [
            {"count": 6, "label": "6h", "step": "hour", "stepmode": "backward"},
            {"count": 12, "label": "12h", "step": "hour", "stepmode": "backward"},
            {"count": 1, "label": "1d", "step": "day", "stepmode": "backward"},
            {"count": 3, "label": "3d", "step": "day", "stepmode": "backward"},
            {"label": "Cohort", "step": "all"},
            {"label": "All stay", "step": "all"},
        ],
        "font": {"size": 10},
        "x": 0, "y": 1.06, "xanchor": "left", "yanchor": "bottom",
        "bgcolor": "#f5f5f5",
    }
    fig.update_xaxes(rangeselector=rangeselector_cfg, row=1, col=1)

    # Rangeslider on the BOTTOM rendered panel
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.04},
        row=len(ordered), col=1,
    )

    if cohort_start is not None and cohort_end is not None:
        fig.update_xaxes(range=[cohort_start, cohort_end], row=1, col=1)

    fig.update_layout(
        hovermode="x unified",
        margin={"t": 80, "b": 40, "l": 130, "r": 20},
        legend={
            "orientation": "v", "x": 1.01, "y": 1.0,
            "groupclick": "toggleitem", "font": {"size": 10},
        },
        showlegend=True,
        template="simple_white",
        dragmode="pan",
    )
    return fig


# ── Per-panel trace drawers (all pass through _normalize) ─────────────

def _add_normed_line(
    fig: go.Figure, x: pd.Series, y_raw: pd.Series, *,
    name: str, color: str, row: int,
    legendgroup: str, legendtitle: str | None = None,
    width: float = 1.5, dash: str | None = None,
    mode: str = "lines",
    step: bool = False,
    carry_forward: bool = False,
    unit: str | None = None,
) -> None:
    """Add a line trace, normalized to [0, 1] in-panel.

    - `step=True`         → render as step-function (`line_shape="hv"`),
      so the value visually holds until the next observation. Use for
      anything that has a "set value persists until changed" semantic
      (FiO₂, PEEP, continuous infusion rates, RASS targets, etc.).
    - `carry_forward=True` → forward-fill `y_raw` before normalization
      so sparse joins with other wide-dataset tables don't break the
      step rendering with NaN gaps. Only the densified non-null span
      [first_valid_index → last_valid_index] is filled — leading and
      trailing NaN stay NaN.
    """
    if carry_forward:
        y_raw = y_raw.copy()
        first = y_raw.first_valid_index()
        last = y_raw.last_valid_index()
        if first is not None and last is not None:
            y_raw.loc[first:last] = y_raw.loc[first:last].ffill()
    y_norm, y_raw_kept, m = _normalize(y_raw)
    if m == 0:
        return
    line_kwargs: dict = {"color": color, "width": width}
    if dash:
        line_kwargs["dash"] = dash
    if step:
        line_kwargs["shape"] = "hv"
    # Append unit to the hover label when one was supplied. The `name`
    # itself stays clean for the legend; only the per-row tooltip carries
    # the unit. With `hovermode="x unified"` the x-axis timestamp is
    # printed once at the top, so we deliberately omit it from the
    # per-trace template.
    unit_suffix = f" {unit}" if unit else ""
    trace_kwargs: dict = {
        "x": x, "y": y_norm, "mode": mode,
        "name": name, "line": line_kwargs,
        "customdata": y_raw_kept,
        "hovertemplate": f"{name}: %{{customdata:.3g}}{unit_suffix}<extra></extra>",
        "legendgroup": legendgroup,
    }
    if legendtitle:
        trace_kwargs["legendgrouptitle_text"] = legendtitle
    if mode == "lines+markers":
        trace_kwargs["marker"] = {"size": 4}
    fig.add_trace(go.Scatter(**trace_kwargs), row=row, col=1)


_SED_DRUGS = ("propofol", "fentanyl", "midazolam", "lorazepam", "hydromorphone")
_SED_LABELS = {
    "propofol": "prop", "fentanyl": "fent", "midazolam": "midaz",
    "lorazepam": "loraz", "hydromorphone": "hydromorph",
}
# clifpy preferred raw units (set in 02_exposure.py). Cont rates carry
# the time denominator; intm boluses are unit-less of time.
_SED_CONT_UNIT = {
    "propofol": "mg/min", "fentanyl": "mcg/min", "midazolam": "mg/min",
    "lorazepam": "mg/min", "hydromorphone": "mg/min",
}
_SED_INTM_UNIT = {
    "propofol": "mg", "fentanyl": "mcg", "midazolam": "mg",
    "lorazepam": "mg", "hydromorphone": "mg",
}
# Marker palette for cont colors comes from existing DRUG_COLORS via
# `code/descriptive/_shared.py`; rebuild a flat dict here for the panel.
from _shared import DRUG_COLORS as _DRUG_COLORS  # noqa: E402
_SED_COLOR = {
    "propofol": _DRUG_COLORS["prop"],
    "fentanyl": _DRUG_COLORS["fenteq"],
    "midazolam": _DRUG_COLORS["midazeq"],
    "lorazepam": "#9e9ac8",
    "hydromorphone": "#bcbddc",
}


def _draw_sedatives(
    fig: go.Figure, wide_df: pd.DataFrame, enr: PatientEnrichment, row: int,
) -> None:
    """Sedatives panel — RAW data, audit-friendly.

    Two trace types per drug:
      - Continuous: step-function line from `wide_df[<drug>]` (raw
        clifpy mg/min or mcg/min charted rate, persists until next change).
      - Intermittent: marker-only trace from `enr.intm` (raw bolus dose
        at admin_dttm). Each bolus drawn as a diamond at its raw value
        normalized into [0, 1] of its own trace.

    Per-trace normalization (unchanged) keeps both visible despite very
    different magnitudes (cont 0.2–5 mg/min vs. intm 50–200 mg bolus).
    """
    t = wide_df["event_time"] if "event_time" in wide_df.columns else pd.Series(dtype="datetime64[ns]")
    intm_df = enr.intm
    seen_drugs: list[str] = []
    seen_legend = False

    for drug in _SED_DRUGS:
        color = _SED_COLOR[drug]
        label = _SED_LABELS[drug]

        # CONT — step-function from wide_df
        if drug in wide_df.columns and wide_df[drug].notna().any():
            _add_normed_line(
                fig, t, wide_df[drug],
                name=f"{label} cont", color=color, row=row,
                legendgroup="sedatives",
                legendtitle="Sedatives" if not seen_legend else None,
                width=1.5, step=True, carry_forward=True,
                unit=_SED_CONT_UNIT[drug],
            )
            seen_legend = True
            seen_drugs.append(drug)

        # INTM — diamond markers per bolus event
        if not intm_df.empty:
            sub = intm_df[intm_df["med_category"] == drug]
            if not sub.empty and sub["med_dose"].notna().any():
                # Reuse the normalized-line helper but mode="markers" gives
                # us no line; the helper computes its own normalization.
                _add_normed_line(
                    fig, sub["admin_dttm"], sub["med_dose"],
                    name=f"{label} intm", color=color, row=row,
                    legendgroup="sedatives",
                    legendtitle="Sedatives" if not seen_legend else None,
                    width=0, mode="markers",
                    unit=_SED_INTM_UNIT[drug],
                )
                # Override the marker symbol → diamond, clearly distinct
                # from the cont step line. Last trace just added.
                fig.data[-1].marker = {"size": 8, "color": color, "symbol": "diamond",
                                        "line": {"color": "#222", "width": 0.6}}
                seen_legend = True


_PRESSOR_UNITS = {
    "norepinephrine": "mcg/kg/min",
    "epinephrine": "mcg/kg/min",
    "vasopressin": "units/min",
    "nee": "mcg/kg/min",
}
_RESP_UNITS = {"fio2_set": "(fraction)", "peep_set": "cmH₂O"}
_VITAL_UNITS = {"heart_rate": "bpm", "map": "mmHg", "spo2": "%", "respiratory_rate": "/min"}
_ASSESS_UNITS = {"rass": "(score)", "gcs_total": "(score)"}


def _draw_pressors(fig: go.Figure, wide_df: pd.DataFrame, row: int) -> None:
    t = wide_df["event_time"] if "event_time" in wide_df.columns else pd.Series(dtype="datetime64[ns]")
    nee = compute_nee(wide_df)
    seen = False
    if not nee.empty and nee.notna().any():
        _add_normed_line(
            fig, t, nee,
            name="NEE", color=PRESSOR_COLORS["nee"], row=row,
            legendgroup="pressors", legendtitle="Pressors",
            width=2.2, step=True, carry_forward=True,
            unit=_PRESSOR_UNITS["nee"],
        )
        seen = True
    for col, color in PRESSOR_COLORS.items():
        if col == "nee":
            continue
        if col in wide_df.columns and wide_df[col].notna().any():
            _add_normed_line(
                fig, t, wide_df[col],
                name=col, color=color, row=row,
                legendgroup="pressors",
                legendtitle="Pressors" if not seen else None,
                width=1.0, dash="dot",
                step=True, carry_forward=True,
                unit=_PRESSOR_UNITS.get(col, ""),
            )
            seen = True


def _draw_assessments(fig: go.Figure, wide_df: pd.DataFrame, row: int) -> None:
    t = wide_df["event_time"] if "event_time" in wide_df.columns else pd.Series(dtype="datetime64[ns]")
    seen = False
    for col, color in ASSESSMENT_COLORS.items():
        if col in wide_df.columns and wide_df[col].notna().any():
            _add_normed_line(
                fig, t, wide_df[col],
                name=col.upper(), color=color, row=row,
                legendgroup="assessments",
                legendtitle="Assessments" if not seen else None,
                width=1.0, mode="lines+markers",
                unit=_ASSESS_UNITS.get(col, ""),
            )
            seen = True


def _draw_resp(fig: go.Figure, wide_df: pd.DataFrame, row: int) -> None:
    t = wide_df["event_time"] if "event_time" in wide_df.columns else pd.Series(dtype="datetime64[ns]")
    _draw_device_ribbon(fig, wide_df, row=row)
    seen = False
    for col, color in RESP_COLORS.items():
        if col in wide_df.columns and wide_df[col].notna().any():
            _add_normed_line(
                fig, t, wide_df[col],
                name=col, color=color, row=row,
                legendgroup="resp",
                legendtitle="Resp" if not seen else None,
                width=1.4,
                step=True, carry_forward=True,
                unit=_RESP_UNITS.get(col, ""),
            )
            seen = True


def _draw_vitals(fig: go.Figure, wide_df: pd.DataFrame, row: int) -> None:
    t = wide_df["event_time"] if "event_time" in wide_df.columns else pd.Series(dtype="datetime64[ns]")
    seen = False
    for col, color in VITAL_COLORS.items():
        if col in wide_df.columns and wide_df[col].notna().any():
            _add_normed_line(
                fig, t, wide_df[col],
                name=col, color=color, row=row,
                legendgroup="vitals",
                legendtitle="Vitals" if not seen else None,
                width=1.1, mode="lines+markers",
                unit=_VITAL_UNITS.get(col, ""),
            )
            seen = True


# ── Resp-panel device ribbon (mode-encoded inside IMV) ────────────────

def _draw_device_ribbon(fig: go.Figure, wide_df: pd.DataFrame, row: int) -> None:
    """Narrow ribbon at the bottom of the resp panel.

    For non-IMV devices, each contiguous segment is filled with the
    device's color. For IMV segments, the ribbon is sub-divided by
    `mode_category` and each sub-segment is filled with a variant blue
    from `MODE_IN_IMV_COLORS`. This puts both device and mode on a
    single layer (no full-height mode background) so they don't compete
    visually with the FiO₂ / PEEP lines.
    """
    needed = {"event_time", "device_category"}
    if not needed.issubset(wide_df.columns):
        return
    df = wide_df[["event_time", "device_category", "mode_category"]].copy()
    df = df.dropna(subset=["event_time"])
    if df.empty:
        return
    df = df.sort_values("event_time").reset_index(drop=True)
    # Carry-forward device + mode between observed records. The upstream
    # resp_processed_bf is already waterfall-ffilled, but the wide-dataset
    # join with vitals/meds re-introduces NaN rows. Forward-filling here
    # collapses those gaps so segments are clinically contiguous.
    # Only fill between first_valid → last_valid so leading/trailing
    # NaN (truly outside the resp record window) stay NaN.
    for col in ("device_category", "mode_category"):
        first = df[col].first_valid_index()
        last = df[col].last_valid_index()
        if first is not None and last is not None:
            df.loc[first:last, col] = df.loc[first:last, col].ffill()
    # Drop rows where device is still NaN (outside the resp record span).
    df = df.dropna(subset=["device_category"]).reset_index(drop=True)
    if df.empty:
        return

    # Compute panel y-domain in paper coordinates so the ribbon hugs the
    # bottom 6% of the panel regardless of FiO₂/PEEP scale.
    yaxis_id = "yaxis" if row == 1 else f"yaxis{row}"
    panel_domain = fig.layout[yaxis_id].domain if fig.layout[yaxis_id] is not None else (0, 1)
    y_bottom = panel_domain[0]
    y_top = panel_domain[0] + (panel_domain[1] - panel_domain[0]) * 0.07

    # Build (start, end, device, mode) segments via gap-island. Devices
    # change less frequently than modes, so we segment by device first
    # then within each IMV segment, sub-segment by mode.
    seen_devices: set[str] = set()
    seen_modes: set[str] = set()

    # First, collapse consecutive rows where (device, mode) are identical.
    df["__key"] = df["device_category"].astype(str) + "|" + df["mode_category"].fillna("?").astype(str)
    df["__chg"] = (df["__key"] != df["__key"].shift()).astype(int)
    df["__seg"] = df["__chg"].cumsum()
    segments = df.groupby("__seg").agg(
        start=("event_time", "first"),
        last_in_seg=("event_time", "last"),
        device=("device_category", "first"),
        mode=("mode_category", "first"),
    ).reset_index(drop=True)
    # Make segments contiguous by extending each one's end to the start
    # of the next segment (the time at which the device/mode actually
    # changed). The final segment ends at its own last observation.
    segments["end"] = segments["start"].shift(-1).fillna(segments["last_in_seg"])

    for _, seg in segments.iterrows():
        start = pd.Timestamp(seg["start"])
        end = pd.Timestamp(seg["end"])
        if end <= start:
            end = start + pd.Timedelta(minutes=15)
        device = str(seg["device"])
        mode = seg["mode"] if pd.notna(seg["mode"]) else None

        # Determine fill color: IMV → mode-specific; else device color.
        if device.lower() == "imv" and mode is not None:
            color = MODE_IN_IMV_COLORS.get(str(mode), DEVICE_CATEGORY_COLORS["imv"])
            label = f"{device} / {mode}"
            legend_key = f"imv: {mode}"
        else:
            color = DEVICE_CATEGORY_COLORS.get(device.lower(), DEVICE_CATEGORY_COLORS["other"])
            label = device
            legend_key = f"device: {device}"

        fig.add_shape(
            type="rect", xref="x", yref="paper",
            x0=start, x1=end, y0=y_bottom, y1=y_top,
            fillcolor=color, line={"color": "#666", "width": 0.3},
            layer="above",
        )
        # Inline label at segment midpoint
        mid = start + (end - start) / 2
        fig.add_annotation(
            x=mid, y=y_top, xref="x", yref="paper",
            text=label, showarrow=False,
            font={"size": 7, "color": "#333"}, yanchor="bottom",
        )

        # Legend entry once per unique device or device-mode combo
        if legend_key not in seen_devices.union(seen_modes):
            fig.add_trace(go.Scatter(
                x=[start], y=[None], mode="markers",
                marker={"size": 8, "color": color, "symbol": "square",
                        "line": {"color": "#666", "width": 0.5}},
                name=legend_key,
                legendgroup="resp", showlegend=True,
            ), row=row, col=1)
            (seen_modes if "imv:" in legend_key else seen_devices).add(legend_key)


# ── Events ────────────────────────────────────────────────────────────

def _draw_event_vlines(fig: go.Figure, events: pd.DataFrame, n_rows: int) -> None:
    if events.empty:
        return
    # Per-row vlines so each is bound to its panel's domain
    for _, e in events.iterrows():
        t_iso = pd.Timestamp(e["time"]).isoformat()
        color = EVENT_COLORS.get(e["kind"], "#444")
        for r in range(1, n_rows + 1):
            fig.add_shape(
                type="line", xref="x", yref="y domain",
                x0=t_iso, x1=t_iso, y0=0, y1=1,
                line={"color": color, "width": 1.2, "dash": "dash"},
                layer="above",
                row=r, col=1,
            )
        fig.add_annotation(
            x=t_iso, y=1.005, xref="x", yref="paper",
            text=e["kind"].replace("discharge_", "disch:"),
            showarrow=False,
            font={"size": 8, "color": color}, xanchor="center",
        )

    for kind in sorted(events["kind"].unique()):
        color = EVENT_COLORS.get(kind, "#444")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line={"color": color, "dash": "dash", "width": 1.5},
            name=f"event: {kind}",
            legendgroup="events", legendgrouptitle_text="Events",
            showlegend=True,
        ))


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        annotations=[{
            "text": msg, "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5, "showarrow": False,
            "font": {"size": 14, "color": "#888"},
        }],
        xaxis={"visible": False}, yaxis={"visible": False},
        template="simple_white",
    )
    return fig


# ── Summary strip + linked table ──────────────────────────────────────

def build_summary_strip(enr: PatientEnrichment, wide_df: pd.DataFrame) -> dbc.Alert:
    a = enr.analytical
    if a.empty:
        return dbc.Alert("No analytical row for this hospitalization.", color="warning", className="py-2")
    row = a.iloc[0]
    los_days = int(a["_nth_day"].max()) if "_nth_day" in a.columns and not a["_nth_day"].isna().all() else None

    # Ever-extub: source of truth is sbt_outcomes_daily._success_extub.
    # The analytical_dataset's `_success_extub_today` is always 0 due to
    # the filter that drops the last day (where extubation actually
    # happens). See docs/uptitration_paradox_investigation.md §0.
    ever_extub = "N"
    if not enr.sbt_daily.empty and "_success_extub" in enr.sbt_daily.columns:
        if enr.sbt_daily["_success_extub"].fillna(0).max() > 0:
            ever_extub = "Y"

    all_streaks = enr.all_imv_streaks
    reintub_n = max(0, len(all_streaks) - 1) if not all_streaks.empty else 0
    dsp = "?"
    if enr.discharge is not None and pd.notna(enr.discharge.get("discharge_category")):
        dsp = str(enr.discharge["discharge_category"])

    # Per-variant SBT counts. Reads sbt_outcomes_daily directly so the
    # numbers reflect raw upstream flags (analytical_dataset's
    # `sbt_done_today` is filter-trimmed and therefore an undercount).
    # Built as `primary | anyprior | imv6h | prefix | 2min` so the auditor
    # can spot a patient where the variants diverge — e.g., prefix=8 with
    # primary=2 says the spec-literal correctly excluded 6 days that the
    # pre-fix every-row baseline would have over-counted.
    sbt_counts_str = "—"
    if not enr.sbt_daily.empty:
        _sd = enr.sbt_daily
        _cols = ['sbt_done', 'sbt_done_anyprior', 'sbt_done_imv6h',
                 'sbt_done_prefix', 'sbt_done_2min']
        _shorts = ['prim', 'anyprior', 'imv6h', 'prefix', '2min']
        _vals = [
            int(_sd[c].fillna(0).sum()) if c in _sd.columns else 0
            for c in _cols
        ]
        sbt_counts_str = ' | '.join(f"{s}={v}" for s, v in zip(_shorts, _vals))

    pieces = [
        f"Age {int(row.get('age'))}" if pd.notna(row.get("age")) else "Age ?",
        f"Sex {row.get('sex_category', '?')}",
        f"ICU {row.get('icu_type', '?')}",
        f"SOFA d1 {int(row.get('sofa_1st24h'))}" if pd.notna(row.get("sofa_1st24h")) else "SOFA d1 ?",
        f"LOS {los_days}d" if los_days is not None else "LOS ?",
        f"IMV {row.get('imv_duration_hrs', '?')}h" if pd.notna(row.get("imv_duration_hrs")) else "IMV ?",
        f"Ever extub: {ever_extub}",
        f"Reintubs: {reintub_n}",
        f"Discharge: {dsp}",
        f"SBT-days [{sbt_counts_str}]",
    ]
    return dbc.Alert(
        " · ".join(pieces),
        color="light", className="py-2 my-1",
        style={"fontFamily": "monospace", "fontSize": "13px"},
    )


# Column rename → snake_case with units only on meds (rates use time-
# denominator suffix; boluses don't). Vitals/resp get bare names since
# their units are universal in clinical context.
_TABLE_COLUMN_LABELS = {
    "propofol":            "prop_mg_min",
    "propofol_intm":       "prop_mg",
    "fentanyl":            "fent_mcg_min",
    "fentanyl_intm":       "fent_mcg",
    "midazolam":           "midaz_mg_min",
    "midazolam_intm":      "midaz_mg",
    "lorazepam":           "loraz_mg_min",
    "lorazepam_intm":      "loraz_mg",
    "hydromorphone":       "hydromorph_mg_min",
    "hydromorphone_intm":  "hydromorph_mg",
    "fio2_set":            "fio2",
    "peep_set":            "peep",
    "mode_category":       "mode",
    "device_category":     "device",
    "norepinephrine":      "norepi_mcg_kg_min",
    "epinephrine":         "epi_mcg_kg_min",
    "vasopressin":         "vaso_units_min",
    "heart_rate":          "hr",
    "map":                 "map",
    "spo2":                "spo2",
    "respiratory_rate":    "rr",
}

# Final L→R column order in the linked table (matches panel order:
# sedatives → resp → pressors → vitals; assessments dropped).
_TABLE_COLUMN_ORDER = [
    "event_time",
    # Sedatives (cont/intm pairs per drug)
    "prop_mg_min", "prop_mg",
    "fent_mcg_min", "fent_mcg",
    "midaz_mg_min", "midaz_mg",
    "loraz_mg_min", "loraz_mg",
    "hydromorph_mg_min", "hydromorph_mg",
    # Resp
    "fio2", "peep", "mode", "device",
    # Pressors
    "norepi_mcg_kg_min", "epi_mcg_kg_min", "vaso_units_min",
    # Vitals
    "hr", "map", "spo2", "rr",
]


def _linked_table_source(wide_df: pd.DataFrame, enr: PatientEnrichment) -> pd.DataFrame:
    """Raw-data audit table.

    Pulls cont sedatives + pressors + vitals + resp settings from
    `wide_df` (clifpy-pivoted raw events) and outer-merges intermittent
    boluses from `enr.intm` (DIY-loaded raw `medication_admin_intermittent`)
    so each bolus appears as its own row with cont columns NaN at that
    timestamp and an intm column populated.

    Reads RAW values exclusively — no aggregation, no time-window
    filtering. The whole point is to spot-check downstream processing
    against the source data.
    """
    keep_from_wide = [
        "event_time",
        "propofol", "fentanyl", "midazolam", "lorazepam", "hydromorphone",
        "fio2_set", "peep_set", "mode_category", "device_category",
        "norepinephrine", "epinephrine", "vasopressin",
        "heart_rate", "map", "spo2", "respiratory_rate",
    ]
    cols = [c for c in keep_from_wide if c in wide_df.columns]
    out = wide_df[cols].copy()

    # Outer-merge the long-form intm events as new rows. Pivot intm
    # long-form to wide-form keyed on admin_dttm, then concat.
    intm_df = enr.intm
    if not intm_df.empty:
        intm_pivot = (
            intm_df.pivot_table(
                index="admin_dttm",
                columns="med_category",
                values="med_dose",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"admin_dttm": "event_time"})
        )
        # Tag every intm-derived column with `_intm` suffix so it doesn't
        # collide with cont columns of the same name in `wide_df`.
        intm_pivot.columns = [
            "event_time" if c == "event_time" else f"{c}_intm"
            for c in intm_pivot.columns
        ]
        # Align dtypes for the concat (event_time tz). Use the wide_df's
        # tz so the merge is timezone-consistent.
        if pd.api.types.is_datetime64_any_dtype(out["event_time"]):
            target_tz = out["event_time"].dt.tz
            it = pd.to_datetime(intm_pivot["event_time"])
            if target_tz is not None and it.dt.tz is None:
                it = it.dt.tz_localize(target_tz)
            elif target_tz is None and it.dt.tz is not None:
                it = it.dt.tz_localize(None)
            intm_pivot["event_time"] = it
        out = pd.concat([out, intm_pivot], ignore_index=True, sort=False)
        out = out.sort_values("event_time").reset_index(drop=True)

    # Apply rename → snake_case with unit suffixes.
    out = out.rename(columns={k: v for k, v in _TABLE_COLUMN_LABELS.items()
                              if k in out.columns})

    # Reorder columns to canonical L→R; drop anything not in the spec.
    final_cols = [c for c in _TABLE_COLUMN_ORDER if c in out.columns]
    return out[final_cols]


def _records_for_display(df: pd.DataFrame) -> list[dict]:
    """Convert a table-source DataFrame to DataTable-ready records.

    Stringifies `event_time` and rounds floats. Idempotent on already-
    string event_time columns.
    """
    if df.empty:
        return []
    out = df.copy()
    if "event_time" in out.columns and pd.api.types.is_datetime64_any_dtype(out["event_time"]):
        out["event_time"] = out["event_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(2)
    return out.to_dict("records")


def _table_spec(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """DataTable column-spec + records (records routed through display helper)."""
    if df.empty:
        return [], []
    columns = [{"name": c, "id": c} for c in df.columns]
    return columns, _records_for_display(df)


# ── Entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _SITES:
        print("WARNING: no site directories found under output/. Did you run `make run SITE=...`?")
    else:
        print(f"Sites discovered: {_SITES}")
    app.run(debug=False, host="127.0.0.1", port=8050)
