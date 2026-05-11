"""Side-by-side cohort stats across all discovered sites.

Concatenates each output_to_share/{site}/cohort_stats.csv row into one
CSV at output_to_agg/cohort_stats_cross_site.csv, appending a 'Total' row
with column-wise sums of numeric columns.

Respects ANONYMIZE_SITES via site_label() — the emitted 'site' column uses
the display label (e.g., 'Site A') when anonymization is enabled.

Usage:
    uv run python code/agg/cross_site_cohort_stats.py
    ANONYMIZE_SITES=1 uv run python code/agg/cross_site_cohort_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from clifpy.utils.logging_config import get_logger
logger = get_logger("epi_sedation.agg.cross_site_cohort_stats")

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    list_sites,
    load_site_cohort_stats,
    save_agg_csv,
    site_label,
)


def main() -> None:
    sites = list_sites()
    if not sites:
        logger.info("No sites found under output_to_share/. Nothing to combine.")
        return
    logger.info(f"Discovered sites: {sites}")

    frames: list[pd.DataFrame] = []
    for s in sites:
        df = load_site_cohort_stats(s)
        # Overwrite the per-site 'site' column with the (possibly anonymized)
        # display label. The underlying file already stores the real name, so
        # replacing preserves data integrity when anonymization is off.
        if "site" in df.columns:
            df = df.copy()
            df["site"] = site_label(s)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Append a Total row that sums numeric columns. Non-numeric cells (only
    # the 'site' column in the current schema) get the literal 'Total'.
    numeric_cols = combined.select_dtypes(include="number").columns.tolist()
    total_row = {c: "Total" for c in combined.columns}
    for c in numeric_cols:
        total_row[c] = int(combined[c].sum())
    combined = pd.concat([combined, pd.DataFrame([total_row])], ignore_index=True)

    save_agg_csv(combined, "cohort_stats_cross_site")
    logger.info(combined.to_string(index=False))


if __name__ == "__main__":
    main()
