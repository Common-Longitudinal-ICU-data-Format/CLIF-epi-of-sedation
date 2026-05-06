"""Pooled Table 1 across all discovered sites.

Reads each output_to_share/{site}/table1.csv and produces a single
side-by-side CSV at output_to_agg/table1_by_site.csv with columns:

    (row_label, category, Site1, Site2, ..., Pooled)

Pooling rules (lifted / adapted from archive/meta_analysis.py):
  - `mean (SD)` rows: weighted-mean + pooled-variance formula. Uses the 'n'
    row at the top of each site's Table 1 as the sample size (same N across
    continuous rows within a site, matching how tableone emits them).
  - `n (%)` rows: sum counts across sites, recompute % against the per-
    variable summed denominator (per-site n is the effective denom in a
    categorical row since every subject is represented in some level).
  - `median [Q1,Q3]` rows: median/IQR across sites cannot be validly
    pooled from summary statistics alone. Pooled column shows "—" with
    a note in output_to_agg/README.md.
  - The 'n' row is special-cased: pooled value = sum of per-site n's.

Site column headers respect ANONYMIZE_SITES (via site_label()).

Usage:
    uv run python code/agg/pool_table1.py
    ANONYMIZE_SITES=1 uv run python code/agg/pool_table1.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from _shared import (  # noqa: E402
    list_sites,
    load_site_table1,
    save_agg_csv,
    site_label,
)


# ── Parsers for the string-formatted cells tableone emits ─────────────────
_MSD_RE = re.compile(r"([-]?\d+\.?\d*)\s*\(\s*([-]?\d+\.?\d*)\s*\)")
_COUNT_PCT_RE = re.compile(r"([-]?\d+)\s*\(\s*([-]?\d+\.?\d*)\s*\)")


def _parse_mean_sd(s: str) -> tuple[float, float] | tuple[None, None]:
    """Parse 'mean (SD)' → (mean, sd); return (None, None) on unparseable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (None, None)
    m = _MSD_RE.search(str(s))
    if not m:
        return (None, None)
    return float(m.group(1)), float(m.group(2))


def _parse_count_pct(s: str) -> tuple[int, float] | tuple[None, None]:
    """Parse 'count (pct)' → (count, pct); return (None, None) on unparseable."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return (None, None)
    m = _COUNT_PCT_RE.search(str(s))
    if not m:
        return (None, None)
    return int(m.group(1)), float(m.group(2))


def _pooled_mean_sd(
    means: list[float], sds: list[float], ns: list[int]
) -> tuple[float, float]:
    """Hansen/Snedecor pooled mean + SD across independent groups.

    Lifted verbatim (with cleanup) from archive/meta_analysis.py:
      pooled mean = Σ(mean_i · n_i) / Σn_i
      pooled var  = [Σ(n_i − 1)·SD_i² + Σ n_i · (mean_i − pooled_mean)²] / (N − 1)

    Second term captures between-group variance lost when you only average
    means; drop it and you systematically under-estimate SD.
    """
    ms = np.asarray(means, dtype=float)
    ss = np.asarray(sds, dtype=float)
    ns_arr = np.asarray(ns, dtype=float)
    mask = ~(np.isnan(ms) | np.isnan(ss) | np.isnan(ns_arr))
    ms, ss, ns_arr = ms[mask], ss[mask], ns_arr[mask]
    N = ns_arr.sum()
    if N <= 1 or len(ms) == 0:
        return (float("nan"), float("nan"))
    pooled_mean = (ms * ns_arr).sum() / N
    between = (ns_arr * (ms - pooled_mean) ** 2).sum()
    within = ((ns_arr - 1) * ss**2).sum()
    pooled_var = (within + between) / (N - 1)
    return float(pooled_mean), float(np.sqrt(pooled_var))


def main() -> None:
    sites = list_sites()
    if not sites:
        print("No sites found under output_to_share/. Nothing to pool.")
        return
    print(f"Discovered sites: {sites}")

    # Load all per-site tables and stack their "Overall" columns side by side.
    # We key rows on (Unnamed: 0, Unnamed: 1) so rows stay aligned even if
    # the exact n_rows differs between sites (e.g., different icu_type levels).
    per_site: dict[str, pd.DataFrame] = {s: load_site_table1(s) for s in sites}

    # Extract per-site n (first row where Unnamed: 0 == 'n').
    site_n: dict[str, int] = {}
    for s, df in per_site.items():
        n_row = df[df["Unnamed: 0"] == "n"]
        if n_row.empty:
            raise RuntimeError(f"Site {s}: no 'n' row in table1.csv")
        site_n[s] = int(n_row.iloc[0]["Overall"])

    # Collect the union of (row_label, category) pairs across sites, keeping
    # first-site order as the canonical ordering (tableone is deterministic
    # within a project so this is stable in practice).
    seen: set[tuple[str, str]] = set()
    row_order: list[tuple[str, str]] = []
    for s in sites:
        df = per_site[s]
        for _, r in df.iterrows():
            key = (str(r["Unnamed: 0"]), "" if pd.isna(r["Unnamed: 1"]) else str(r["Unnamed: 1"]))
            if key not in seen:
                seen.add(key)
                row_order.append(key)

    # Build per-site (row_label, category) → Overall-cell lookup.
    site_cell: dict[str, dict[tuple[str, str], str]] = {}
    for s, df in per_site.items():
        m: dict[tuple[str, str], str] = {}
        for _, r in df.iterrows():
            key = (str(r["Unnamed: 0"]), "" if pd.isna(r["Unnamed: 1"]) else str(r["Unnamed: 1"]))
            m[key] = str(r["Overall"])
        site_cell[s] = m

    # Assemble the pooled column row-by-row using the appropriate rule.
    pooled_col: list[str] = []
    for row_label, category in row_order:
        # Case: the 'n' header row.
        if row_label == "n":
            pooled_col.append(str(sum(site_n.values())))
            continue

        is_mean_sd = "mean (SD)" in row_label
        is_count_pct = "n (%)" in row_label
        is_median = "median" in row_label.lower()

        if is_mean_sd:
            means, sds, ns = [], [], []
            for s in sites:
                cell = site_cell[s].get((row_label, category))
                if cell is None:
                    continue
                m, sd = _parse_mean_sd(cell)
                if m is None:
                    continue
                means.append(m)
                sds.append(sd)
                ns.append(site_n[s])
            if not means:
                pooled_col.append("")
            else:
                pm, psd = _pooled_mean_sd(means, sds, ns)
                pooled_col.append(f"{pm:.1f} ({psd:.1f})")

        elif is_count_pct:
            total_count = 0
            denom = 0
            any_parsed = False
            for s in sites:
                cell = site_cell[s].get((row_label, category))
                if cell is None:
                    continue
                cnt, _pct = _parse_count_pct(cell)
                if cnt is None:
                    continue
                total_count += cnt
                denom += site_n[s]
                any_parsed = True
            if not any_parsed or denom == 0:
                pooled_col.append("")
            else:
                pct = total_count / denom * 100.0
                pooled_col.append(f"{total_count} ({pct:.1f})")

        elif is_median:
            # Cannot validly pool medians from summary stats across sites;
            # surfacing a dash is better than a misleading number.
            pooled_col.append("—")

        else:
            # Unknown row format — leave blank rather than guess.
            pooled_col.append("")

    # Assemble the output frame: row_label, category, one col per site, Pooled.
    labels_used = [site_label(s) for s in sites]
    # Guard against accidental label collisions (e.g., two sites with the
    # same anonymized letter would mask data — shouldn't happen with sorted
    # list_sites() but be defensive).
    if len(set(labels_used)) != len(labels_used):
        raise RuntimeError(
            f"Site label collision among {labels_used} — check site_label()"
        )

    out = pd.DataFrame({
        "row_label": [k[0] for k in row_order],
        "category": [k[1] for k in row_order],
    })
    for s, lab in zip(sites, labels_used):
        out[lab] = [site_cell[s].get(k, "") for k in row_order]
    out["Pooled"] = pooled_col

    save_agg_csv(out, "table1_by_site")
    print(f"Sites pooled: {labels_used}  (totals: {site_n})")


if __name__ == "__main__":
    main()
