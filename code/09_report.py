# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas>=2.3.1",
#     "matplotlib",
#     "seaborn",
#     "numpy",
# ]
# ///

import marimo

__generated_with = "0.21.0"
app = marimo.App(sql_output="native")

with app.setup:
    import marimo as mo
    import os
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 09 Unified PDF Report

    Compiles all outputs from the pipeline into a single multi-page PDF for
    sharing with coauthors. Uses `matplotlib.backends.backend_pdf.PdfPages` —
    no extra dependencies.

    **Pages (in order):**

    1. Title page

    2. CONSORT flow diagram

    3. Table 1 (baseline characteristics)

    4. Dose by Shift (paired + unpaired day-vs-night hourly rates)

    5. Correlation matrix heatmap

    6. Hourly dose visualization

    7. Night-minus-day diff distribution (histograms)

    8. Mean night-day diff by ICU day (trajectory)

    9. Spread of night-day diff by ICU day (violin)

    10. Overall % of patient-days up-titrated (bar)

    11. % of patient-days up-titrated by ICU day (bar panel)

    12. Up-titrated vs stable subcohort characteristics (Table)

    13. Model comparison — SBT Done Next Day (GEE)

    14. Model comparison — Successful Extubation (GEE)

    15. Model comparison — Successful Extubation (Logit)

    16. Marginal effects (Linear) — SBT Done Next Day (GEE)

    17. Marginal effects (Linear) — Successful Extubation (GEE)

    18. Marginal effects (Linear) — Successful Extubation (Logit)

    19. Marginal effects (RCS) — SBT Done Next Day (GEE)

    20. Marginal effects (RCS) — Successful Extubation (GEE)

    21. Marginal effects (RCS) — Successful Extubation (Logit)
    """)
    return


@app.cell
def _():
    from clifpy.utils.config import get_config_or_params
    import pandas as pd
    from datetime import datetime

    CONFIG_PATH = "config/config.json"
    cfg = get_config_or_params(CONFIG_PATH)
    SITE_NAME = cfg['site_name'].lower()

    # Site-scoped output dir (see Makefile SITE= flag).
    os.makedirs(f"output_to_share/{SITE_NAME}/figures", exist_ok=True)
    print(f"Site: {SITE_NAME}")
    return SITE_NAME, datetime, pd


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Helper functions
    """)
    return


@app.cell
def _():
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import seaborn as sns
    import numpy as np

    LETTER = (11, 8.5)  # landscape letter size

    def add_text_page(pdf, title, lines, fontsize=10):
        """Add a page with a title and monospace text body."""
        fig, ax = plt.subplots(figsize=LETTER)
        ax.axis('off')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)
        body = '\n'.join(lines)
        ax.text(0.02, 0.95, body, family='monospace', fontsize=fontsize,
                verticalalignment='top', transform=ax.transAxes)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_table_page(pdf, df, title, fontsize=7, col_widths=None):
        """Render a DataFrame as a matplotlib table on a full page."""
        fig, ax = plt.subplots(figsize=LETTER)
        ax.axis('off')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)

        # Reset index to get the row labels as the first column
        df_reset = df.reset_index()
        col_labels = list(df_reset.columns)
        cell_text = df_reset.astype(str).values.tolist()

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc='center',
            cellLoc='left',
            colWidths=col_widths,
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(fontsize)
        tbl.scale(1.0, 1.4)
        # Bold header row
        for _j in range(len(col_labels)):
            tbl[(0, _j)].set_text_props(fontweight='bold')
            tbl[(0, _j)].set_facecolor('#d0d0d0')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_image_page(pdf, image_path, title, dpi=300):
        """Embed a PNG as a page at high resolution.

        Matplotlib's default savefig DPI is 100, which downsamples upstream PNGs
        (generated at ~160-200 DPI) and produces visibly blurry images in the PDF.
        We set both the figure DPI and the pdf.savefig DPI to 300, and use
        interpolation='none' so imshow doesn't add its own anti-aliasing blur
        when the source PNG pixel grid doesn't align exactly with the axes.
        """
        fig, ax = plt.subplots(figsize=LETTER, dpi=dpi)
        img = imread(image_path)
        ax.imshow(img, interpolation='none')
        ax.axis('off')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)
        pdf.savefig(fig, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    def add_heatmap_page(pdf, corr_df, title):
        """Render a correlation DataFrame as a heatmap page."""
        fig, ax = plt.subplots(figsize=LETTER)
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='vlag',
                    linewidths=0.5, cbar_kws={'label': 'Pearson r'},
                    ax=ax, annot_kws={'size': 6})
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def add_stargazer_table(pdf, df, title, notes=None, fontsize=9,
                            label_col_width=0.42):
        """Render a DataFrame as a publication-style (stargazer-like) table.

        Features:
        - Three horizontal rules: top (lw=1.2), mid below header (lw=0.5),
          bottom (lw=1.2). Optional thin separator above the `N` row if present.
        - Serif font throughout; italic bold column headers; left-aligned
          row labels; center-aligned data cells.
        - Dynamic font sizing: scales down to fit many rows on one page
          (floor: 6pt).
        - Notes line at bottom (e.g., significance legend), italic + smaller.
        - White background; no vertical gridlines.

        Args:
            pdf: PdfPages instance
            df: DataFrame with row labels as index; columns become table cols
            title: Page title (bold serif, centered)
            notes: Optional string for the footnote line
            fontsize: Max font size (will scale down if needed)
            label_col_width: Fraction of page width for the row-label column
        """
        fig, ax = plt.subplots(figsize=LETTER)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Title (centered, serif, bold)
        ax.text(0.5, 0.96, title, ha='center', va='top',
                fontsize=14, fontweight='bold', family='serif')

        # ── Column layout ─────────────────────────────────────────────
        n_data_cols = len(df.columns)
        data_col_width = (1.0 - label_col_width - 0.02) / n_data_cols  # 0.02 for right margin
        left_margin = 0.01
        col_lefts = [left_margin]
        col_lefts.append(left_margin + label_col_width)
        for _ in range(n_data_cols - 1):
            col_lefts.append(col_lefts[-1] + data_col_width)
        col_rights = [col_lefts[1]]
        for i in range(n_data_cols):
            col_rights.append(col_lefts[i + 1] + data_col_width)
        col_centers = [(col_lefts[i] + col_rights[i]) / 2 for i in range(n_data_cols + 1)]
        table_left = col_lefts[0]
        table_right = col_rights[-1]

        # ── Vertical layout ───────────────────────────────────────────
        n_rows = len(df)
        y_top_rule = 0.90
        y_bottom_margin = 0.10  # reserve for bottom rule + notes
        has_n_row = 'N' in df.index
        # Rows needed: header + data rows (+ 1 for N separator if present)
        effective_rows = n_rows + 1 + (1 if has_n_row else 0)
        available_h = y_top_rule - y_bottom_margin
        line_h = available_h / effective_rows
        # Scale font down if rows are cramped (floor: 6pt)
        # Heuristic: line_h in axes units * 170 ≈ readable pt for 8.5" height
        fontsize_actual = max(6, min(fontsize, int(line_h * 170)))
        notes_fontsize = max(5, int(fontsize_actual * 0.85))

        # ── Top rule ──────────────────────────────────────────────────
        ax.plot([table_left, table_right], [y_top_rule, y_top_rule],
                color='black', lw=1.2)

        # ── Header row ────────────────────────────────────────────────
        y_header = y_top_rule - line_h * 0.55
        # First column: index name (or blank), left-aligned
        ax.text(col_lefts[0] + 0.003, y_header, df.index.name or '',
                ha='left', va='center',
                fontsize=fontsize_actual, fontweight='bold', family='serif')
        # Data columns: italic bold, centered
        for i, col in enumerate(df.columns):
            ax.text(col_centers[i + 1], y_header, str(col),
                    ha='center', va='center',
                    fontsize=fontsize_actual, fontweight='bold',
                    fontstyle='italic', family='serif')

        # ── Mid rule ──────────────────────────────────────────────────
        y_mid_rule = y_top_rule - line_h * 1.1
        ax.plot([table_left, table_right], [y_mid_rule, y_mid_rule],
                color='black', lw=0.5)

        # ── Data rows ─────────────────────────────────────────────────
        n_row_index = df.index.get_loc('N') if has_n_row else -1
        row_y_start = y_mid_rule - line_h * 0.55
        for j, (idx, row) in enumerate(df.iterrows()):
            y = row_y_start - j * line_h
            # If this is the N row, draw a thin separator just above it
            if has_n_row and j == n_row_index:
                sep_y = y + line_h * 0.5
                ax.plot([table_left, table_right], [sep_y, sep_y],
                        color='black', lw=0.5)
            # Row label, left-aligned
            ax.text(col_lefts[0] + 0.003, y, str(idx),
                    ha='left', va='center',
                    fontsize=fontsize_actual, family='serif')
            # Data cells, center-aligned
            for i, val in enumerate(row.values):
                ax.text(col_centers[i + 1], y, str(val),
                        ha='center', va='center',
                        fontsize=fontsize_actual, family='serif')

        # ── Bottom rule ───────────────────────────────────────────────
        y_bottom_rule = row_y_start - (n_rows - 1) * line_h - line_h * 0.55
        ax.plot([table_left, table_right], [y_bottom_rule, y_bottom_rule],
                color='black', lw=1.2)

        # ── Notes ─────────────────────────────────────────────────────
        if notes:
            y_notes = y_bottom_rule - 0.025
            ax.text(table_left, y_notes, notes,
                    ha='left', va='top',
                    fontsize=notes_fontsize, fontstyle='italic', family='serif')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return (
        PdfPages,
        add_heatmap_page,
        add_image_page,
        add_stargazer_table,
        add_table_page,
        add_text_page,
        plt,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Load all source outputs
    """)
    return


@app.cell
def _(SITE_NAME, pd):
    # Cohort stats (from 06_table1.py)
    try:
        cohort_stats = pd.read_csv(f"output_to_share/{SITE_NAME}/cohort_stats.csv").iloc[0].to_dict()
    except FileNotFoundError:
        cohort_stats = {'site': 'unknown', 'n_hospitalizations': 'n/a', 'n_unique_patients': 'n/a'}
    print(f"Cohort stats: {cohort_stats}")
    return (cohort_stats,)


@app.cell
def _(SITE_NAME, pd):
    # Table 1 (from 06_table1.py) — tableone CSV has hierarchical 2-col format
    table1_df = pd.read_csv(f"output_to_share/{SITE_NAME}/table1.csv")
    # Replace NaN with empty string for cleaner display
    table1_df = table1_df.fillna('')
    print(f"Table 1: {len(table1_df)} rows, {len(table1_df.columns)} cols")
    return (table1_df,)


@app.cell
def _(SITE_NAME, pd):
    # Dose by Shift (from 07_descriptive.py) — paired + unpaired day-vs-night
    # sedation rates, long/tidy format (6 rows × 6 columns).
    dose_by_shift_df = pd.read_csv(f"output_to_share/{SITE_NAME}/sed_dose_by_shift.csv")
    print(f"Dose by Shift: {dose_by_shift_df.shape}")
    return (dose_by_shift_df,)


@app.cell
def _(SITE_NAME, pd):
    # Correlation matrix (from 07_descriptive.py)
    corr_df = pd.read_csv(f"output_to_share/{SITE_NAME}/pairwise_corr_matrix.csv", index_col=0)
    print(f"Correlation matrix: {corr_df.shape}")
    return (corr_df,)


@app.cell
def _(SITE_NAME, pd):
    # Model comparison tables (from 08_models.py)
    sbt_gee_df = pd.read_csv(f"output_to_share/{SITE_NAME}/model_comparison_sbt_gee.csv", index_col=0)
    extub_gee_df = pd.read_csv(f"output_to_share/{SITE_NAME}/model_comparison_extub_gee.csv", index_col=0)
    extub_logit_df = pd.read_csv(f"output_to_share/{SITE_NAME}/model_comparison_extub_logit.csv", index_col=0)
    print(f"SBT GEE: {sbt_gee_df.shape}, Extub GEE: {extub_gee_df.shape}, Extub Logit: {extub_logit_df.shape}")
    return extub_gee_df, extub_logit_df, sbt_gee_df


@app.cell
def _(SITE_NAME, pd):
    # Up-titrated subcohort characteristics table (from
    # code/descriptive/uptitrated_subcohort_characteristics.py).
    # tableone writes a 2-col index (variable, level); MultiIndex header has
    # already been flattened upstream.
    try:
        subcohort_df = pd.read_csv(
            f"output_to_share/{SITE_NAME}/uptitrated_subcohort_characteristics.csv"
        ).fillna('')
    except FileNotFoundError:
        subcohort_df = None
    if subcohort_df is not None:
        print(f"Subcohort table: {subcohort_df.shape}")
    return (subcohort_df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Compile PDF
    """)
    return


@app.cell
def _(
    PdfPages,
    SITE_NAME,
    add_heatmap_page,
    add_image_page,
    add_stargazer_table,
    add_text_page,
    cohort_stats,
    corr_df,
    datetime,
    dose_by_shift_df,
    extub_gee_df,
    extub_logit_df,
    plt,
    sbt_gee_df,
    subcohort_df,
    table1_df,
):
    _pdf_path = f"output_to_share/{SITE_NAME}/figures/sedation_report.pdf"

    # Paths to pre-rendered images from earlier scripts
    _consort_png = f"output_to_share/{SITE_NAME}/consort_inclusion.png"
    _hourly_png = f"output_to_share/{SITE_NAME}/figures/sed_dose_by_hr_of_day.png"

    # Build cohort summary lines (handle numeric vs string values gracefully)
    def _fmt_n(val):
        return f"{val:,}" if isinstance(val, (int, float)) else str(val)

    _title_lines = [
        "",
        f"Site:                   {cohort_stats.get('site', SITE_NAME)}",
        f"Hospitalizations:       {_fmt_n(cohort_stats.get('n_hospitalizations', 'n/a'))}",
        f"Unique patients:        {_fmt_n(cohort_stats.get('n_unique_patients', 'n/a'))}",
        f"Report generated:       {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Contents:",
        "  1. Title",
        "  2. CONSORT flow",
        "  3. Table 1: Baseline characteristics",
        "  4. Dose by Shift: Day vs Night hourly sedation rates",
        "  5. Correlation matrix",
        "  6. Hourly dose distribution",
        "  7. Night-minus-day diff distribution",
        "  8. Mean night-day diff by ICU day",
        "  9. Spread of night-day diff by ICU day",
        " 10. Overall % of patient-days up-titrated",
        " 11. % of patient-days up-titrated by ICU day",
        " 12. Up-titrated subcohort characteristics",
        " 13. Model comparison — SBT Done Next Day (GEE)",
        " 14. Model comparison — Successful Extubation (GEE)",
        " 15. Model comparison — Successful Extubation (Logit)",
        " 16. Marginal effects (Linear) — SBT Done Next Day (GEE)",
        " 17. Marginal effects (Linear) — Successful Extubation (GEE)",
        " 18. Marginal effects (Linear) — Successful Extubation (Logit)",
        " 19. Marginal effects (RCS) — SBT Done Next Day (GEE)",
        " 20. Marginal effects (RCS) — Successful Extubation (GEE)",
        " 21. Marginal effects (RCS) — Successful Extubation (Logit)",
    ]

    # Significance legend shown as footnote on every model comparison table
    _model_notes = (
        "Cells report odds ratios with 95% confidence intervals. "
        "— indicates variable not included in model.  "
        "* p < 0.05, ** p < 0.01, *** p < 0.001"
    )

    with PdfPages(_pdf_path) as _pdf:
        # Page 1: Title page
        add_text_page(
            _pdf,
            f"Sedation Epidemiology Study — {SITE_NAME.upper()}",
            _title_lines,
            fontsize=11,
        )

        # Page 2: CONSORT flow
        if os.path.exists(_consort_png):
            add_image_page(_pdf, _consort_png, "CONSORT Flow")
        else:
            add_text_page(_pdf, "CONSORT Flow",
                          [f"[Image not found: {_consort_png}]",
                           "Run 01_cohort.py to generate."])

        # Page 3: Table 1 — tableone CSV has a 2-col structure (variable, value)
        add_stargazer_table(
            _pdf,
            table1_df.set_index(table1_df.columns[0]),
            "Table 1: Baseline Characteristics",
            fontsize=9,
        )

        # Page 4: Dose by Shift (NEW) — paired + unpaired day-vs-night rates.
        # Compose compound row labels ("<variable> — paired/unpaired") to
        # collapse the tidy 6-row × 6-col CSV into a 6-row × 4-col
        # publication layout matching the user's original spec.
        _dose_by_shift_display = dose_by_shift_df.copy()
        _dose_by_shift_display['_row_label'] = (
            _dose_by_shift_display['Variable'] + ' — ' + _dose_by_shift_display['spec']
        )
        _dose_by_shift_display = (
            _dose_by_shift_display
            .set_index('_row_label')
            .drop(columns=['Variable', 'spec'])
        )
        _dose_by_shift_display.index.name = ''
        add_stargazer_table(
            _pdf,
            _dose_by_shift_display,
            "Dose by Shift: Day vs Night Hourly Sedation Rates",
            notes=(
                "Per-hospitalization weighted-mean hourly dose rate "
                "(sum of dose / sum of hours on IMV within each shift). "
                "Paired t-test is the primary analysis; unpaired Welch's "
                "t-test shown as a robustness check. Two-sided p-values; "
                "95% CI for mean difference."
            ),
            fontsize=9,
        )

        # Page 5: Correlation matrix
        add_heatmap_page(_pdf, corr_df, "Pairwise Pearson Correlation Matrix")

        # Page 6: Hourly dose distribution
        if os.path.exists(_hourly_png):
            add_image_page(_pdf, _hourly_png, "Hourly Dose Distribution")
        else:
            add_text_page(_pdf, "Hourly Dose Distribution",
                          [f"[Image not found: {_hourly_png}]",
                           "Run 07_descriptive.py to generate."])

        # Pages 7-12: Nocturnal up-titration descriptives.
        # Generated by the standalone scripts under code/descriptive/ — each
        # script writes one PNG (or CSV) and is re-runnable on its own. They
        # live here (pre-models) because they motivate the exposure-of-interest
        # visually before the reader sees any coefficients.
        _uptitration_pages = [
            (f"output_to_share/{SITE_NAME}/figures/night_day_diff_distribution.png",
             "Night-minus-day Dose Rate: Distribution per Patient-day"),
            (f"output_to_share/{SITE_NAME}/figures/night_day_diff_mean_by_icu_day.png",
             "Mean Night-minus-day Dose Rate by ICU Day (±95% CI)"),
            (f"output_to_share/{SITE_NAME}/figures/night_day_diff_spread_by_icu_day.png",
             "Spread of Night-minus-day Dose Rate by ICU Day"),
            (f"output_to_share/{SITE_NAME}/figures/pct_uptitrated_overall.png",
             "Overall Nocturnal Up-titration Prevalence"),
            (f"output_to_share/{SITE_NAME}/figures/pct_uptitrated_by_icu_day.png",
             "Nocturnal Up-titration Prevalence by ICU Day"),
        ]
        for _path, _title in _uptitration_pages:
            if os.path.exists(_path):
                add_image_page(_pdf, _path, _title)
            else:
                add_text_page(_pdf, _title,
                              [f"[Image not found: {_path}]",
                               "Run the corresponding script under code/descriptive/ to generate."])

        # Page 12: Up-titrated vs stable subcohort characteristics (tableone).
        # Thresholds: prop > 10 mcg/kg/min, fenteq > 25 mcg/hr, midazeq > 1 mg/hr
        # (a day is "up-titrated" if ANY drug crosses its threshold).
        if subcohort_df is not None:
            add_stargazer_table(
                _pdf,
                subcohort_df.set_index(subcohort_df.columns[0]),
                "Up-titrated vs Stable Patient-days: Characteristics",
                notes=(
                    "Up-titrated = any of prop > 10 mcg/kg/min, fenteq > 25 mcg/hr, "
                    "midazeq > 1 mg/hr night-minus-day rate. Continuous vars "
                    "report median [Q1,Q3] unless otherwise noted. "
                    "P-values: Welch's t-test / chi-sq / Kruskal–Wallis."
                ),
                fontsize=8,
            )
        else:
            add_text_page(
                _pdf,
                "Up-titrated vs Stable Patient-days: Characteristics",
                [f"[CSV not found: output_to_share/{SITE_NAME}/uptitrated_subcohort_characteristics.csv]",
                 "Run code/descriptive/uptitrated_subcohort_characteristics.py to generate."],
            )

        # Pages 13-15: Model comparison tables (stargazer-style)
        add_stargazer_table(
            _pdf, sbt_gee_df,
            "Model Comparison: SBT Done Next Day (GEE)",
            notes=_model_notes, fontsize=9,
        )
        add_stargazer_table(
            _pdf, extub_gee_df,
            "Model Comparison: Successful Extubation Next Day (GEE)",
            notes=_model_notes, fontsize=9,
        )
        add_stargazer_table(
            _pdf, extub_logit_df,
            "Model Comparison: Successful Extubation Next Day (Logit + clustered SE)",
            notes=_model_notes, fontsize=9,
        )

        # Pages 16-18: Linear marginal effect plots (sofa spec).
        # PNGs are generated by 08_models.py from the `sofa` spec (linear
        # exposure terms); we just embed them here as regular image pages.
        for _label, _outcome_short, _mt in [
            ('SBT Done Next Day (GEE)', 'sbt', 'gee'),
            ('Successful Extubation Next Day (GEE)', 'extub', 'gee'),
            ('Successful Extubation Next Day (Logit)', 'extub', 'logit'),
        ]:
            _me_path = (
                f"output_to_share/{SITE_NAME}/figures/"
                f"marginal_effects_{_outcome_short}_{_mt}_sofa.png"
            )
            if os.path.exists(_me_path):
                add_image_page(_pdf, _me_path, f"Marginal Effects (Linear) — {_label}")
            else:
                add_text_page(
                    _pdf,
                    f"Marginal Effects (Linear) — {_label}",
                    [
                        f"[Image not found: {_me_path}]",
                        "Run 08_models.py to generate.",
                    ],
                )

        # Pages 19-21: RCS marginal effect plots (sofa_rcs spec).
        # Same 6 exposures wrapped in patsy's cr(x, df=4) so the fitted curves
        # can bend — reveals dose-response shape that the linear plots cannot
        # capture. See plan_rcs_exposures.md memory for the decision rationale.
        for _label, _outcome_short, _mt in [
            ('SBT Done Next Day (GEE)', 'sbt', 'gee'),
            ('Successful Extubation Next Day (GEE)', 'extub', 'gee'),
            ('Successful Extubation Next Day (Logit)', 'extub', 'logit'),
        ]:
            _me_path = (
                f"output_to_share/{SITE_NAME}/figures/"
                f"marginal_effects_{_outcome_short}_{_mt}_sofa_rcs.png"
            )
            if os.path.exists(_me_path):
                add_image_page(_pdf, _me_path, f"Marginal Effects (RCS) — {_label}")
            else:
                add_text_page(
                    _pdf,
                    f"Marginal Effects (RCS) — {_label}",
                    [
                        f"[Image not found: {_me_path}]",
                        "Run 08_models.py to generate.",
                    ],
                )

    print(f"Saved unified PDF report: {_pdf_path}")
    return


if __name__ == "__main__":
    app.run()
