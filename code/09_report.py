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

    4. Correlation matrix heatmap

    5. Hourly dose visualization

    6. Model comparison — SBT Done Next Day (GEE)

    7. Model comparison — Successful Extubation (GEE)

    8. Model comparison — Successful Extubation (Logit)
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

    os.makedirs("output_to_share/figures", exist_ok=True)
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

    def add_image_page(pdf, image_path, title):
        """Embed a PNG as a page."""
        fig, ax = plt.subplots(figsize=LETTER)
        img = imread(image_path)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(title, loc='left', fontweight='bold', fontsize=14, pad=20)
        pdf.savefig(fig, bbox_inches='tight')
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

    return (
        PdfPages,
        add_heatmap_page,
        add_image_page,
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
def _(pd):
    # Cohort stats (from 06_table1.py)
    try:
        cohort_stats = pd.read_csv("output_to_share/cohort_stats.csv").iloc[0].to_dict()
    except FileNotFoundError:
        cohort_stats = {'site': 'unknown', 'n_hospitalizations': 'n/a', 'n_unique_patients': 'n/a'}
    print(f"Cohort stats: {cohort_stats}")
    return (cohort_stats,)


@app.cell
def _(pd):
    # Table 1 (from 06_table1.py) — tableone CSV has hierarchical 2-col format
    table1_df = pd.read_csv("output_to_share/table1.csv")
    # Replace NaN with empty string for cleaner display
    table1_df = table1_df.fillna('')
    print(f"Table 1: {len(table1_df)} rows, {len(table1_df.columns)} cols")
    return (table1_df,)


@app.cell
def _(pd):
    # Correlation matrix (from 07_descriptive.py)
    corr_df = pd.read_csv("output_to_share/pairwise_corr_matrix.csv", index_col=0)
    print(f"Correlation matrix: {corr_df.shape}")
    return (corr_df,)


@app.cell
def _(pd):
    # Model comparison tables (from 08_models.py)
    sbt_gee_df = pd.read_csv("output_to_share/model_comparison_sbt_gee.csv", index_col=0)
    extub_gee_df = pd.read_csv("output_to_share/model_comparison_extub_gee.csv", index_col=0)
    extub_logit_df = pd.read_csv("output_to_share/model_comparison_extub_logit.csv", index_col=0)
    print(f"SBT GEE: {sbt_gee_df.shape}, Extub GEE: {extub_gee_df.shape}, Extub Logit: {extub_logit_df.shape}")
    return extub_gee_df, extub_logit_df, sbt_gee_df


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
    add_table_page,
    add_text_page,
    cohort_stats,
    corr_df,
    datetime,
    extub_gee_df,
    extub_logit_df,
    plt,
    sbt_gee_df,
    table1_df,
):
    _pdf_path = f"output_to_share/figures/sedation_report_{SITE_NAME}.pdf"

    # Paths to pre-rendered images from earlier scripts
    _consort_png = "output_to_share/consort_inclusion.png"
    _hourly_png = "output_to_share/figures/sed_dose_by_hr_of_day.png"

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
        "  4. Correlation matrix",
        "  5. Hourly dose distribution",
        "  6. Model comparison — SBT Done Next Day (GEE)",
        "  7. Model comparison — Successful Extubation (GEE)",
        "  8. Model comparison — Successful Extubation (Logit)",
    ]

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

        # Page 3: Table 1
        add_table_page(_pdf, table1_df.set_index(table1_df.columns[0]),
                       "Table 1: Baseline Characteristics", fontsize=7)

        # Page 4: Correlation matrix
        add_heatmap_page(_pdf, corr_df, "Pairwise Pearson Correlation Matrix")

        # Page 5: Hourly dose distribution
        if os.path.exists(_hourly_png):
            add_image_page(_pdf, _hourly_png, "Hourly Dose Distribution")
        else:
            add_text_page(_pdf, "Hourly Dose Distribution",
                          [f"[Image not found: {_hourly_png}]",
                           "Run 07_descriptive.py to generate."])

        # Pages 6-8: Model comparison tables
        add_table_page(_pdf, sbt_gee_df,
                       "Model Comparison: SBT Done Next Day (GEE)", fontsize=7)
        add_table_page(_pdf, extub_gee_df,
                       "Model Comparison: Successful Extubation Next Day (GEE)", fontsize=7)
        add_table_page(_pdf, extub_logit_df,
                       "Model Comparison: Successful Extubation Next Day (Logit + clustered SE)", fontsize=7)

    print(f"Saved unified PDF report: {_pdf_path}")
    return


if __name__ == "__main__":
    app.run()
