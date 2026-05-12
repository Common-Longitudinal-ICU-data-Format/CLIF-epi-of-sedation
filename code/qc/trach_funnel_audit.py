"""Trach Funnel DIAGNOSTIC — federated audit CSVs for triangulating an empty
`exit_mechanism = 'tracheostomy'` bucket.

Independent of the main pipeline. Run AFTER `make run SITE=<site>` (the
pipeline's outputs must exist on disk). Reads:

  - {data_directory}/clif_respiratory_support.parquet  (raw CLIF source)
  - output/{site}/cohort_resp_processed_bf.parquet     (post-waterfall resp_p)
  - output/{site}/outcomes_by_id_imvday.parquet        (per-day _trach_1st + _trach_v2)
  - output/{site}/cohort_meta_by_id.parquet            (per-hosp canonical exit_mechanism)
  - output/{site}/cohort_imv_streaks.parquet           (IMV streak boundaries)

Writes (federation-safe — aggregates only, no IDs surfaced):

  - output_to_share/{site}/qc/trach_funnel.csv          (6 stages × per-site row)
  - output_to_share/{site}/qc/trach_table1_variants.csv (canonical vs v2-based exit_mechanism
                                                         — if H2/H2b is right, this CSV IS the
                                                         corrected Table 1 trach row)

Hypotheses tested (see docs/audit_tracker.md F13 for full rationale):

  H1   — source parquet has all-zero or NULL tracheostomy column
  H2a  — patient already trached on first resp row → LAG(tracheostomy) is NULL
  H2b  — source charts tracheostomy only when 1 (NULL otherwise) + clifpy's
         waterfall.ffill() + scaffold rows → NULL→1 instead of 0→1 → no flip
  H3   — waterfall smears trach=1 backward (essentially impossible against
         today's clifpy — `tracheostomy` is ffill-only at
         clifpy/utils/waterfall.py:354-355)
  H4   — trach events occur after the IMV streak ends
  H5   — Mode A producer stripped the trach column upstream

The decisive readout is **Stage D v1 vs v2**: if v1=0 while v2>0, the NULL-LAG
bug (H2/H2b) is confirmed. The trach_table1_variants.csv then carries the
corrected count under the v2-based detector.

Usage:
    make trach-funnel SITE=mimic
    # or directly:
    uv run python code/qc/trach_funnel_audit.py
    SITE=ucmc uv run python code/qc/trach_funnel_audit.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd
from clifpy import setup_logging
from clifpy.utils.logging_config import get_logger

logger = get_logger("epi_sedation.trach_funnel")

# ── Paths / site config ─────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"


def _load_config() -> dict:
    if not CONFIG_PATH.exists():
        logger.error(f"Missing config: {CONFIG_PATH}")
        sys.exit(1)
    with CONFIG_PATH.open() as f:
        return json.load(f)


_cfg = _load_config()
SITE = os.getenv("SITE", _cfg.get("site_name", "unknown").lower())
DATA_DIR = Path(_cfg["data_directory"])
FILETYPE = _cfg.get("filetype", "parquet")
SITE_OUT = PROJECT_ROOT / "output" / SITE
SHARE_QC = PROJECT_ROOT / "output_to_share" / SITE / "qc"
SHARE_QC.mkdir(parents=True, exist_ok=True)

# Co-locate logs with the federation bundle (matches weight_audit.py pattern).
setup_logging(output_directory=str(PROJECT_ROOT / "output_to_share" / SITE))


# ── Helpers ─────────────────────────────────────────────────────────────
def _safe_pct(num: int, denom: int) -> float:
    return (num / denom) if denom else 0.0


def _trach_dist_map(series: pd.Series) -> dict[str, int]:
    """Return a {value-str: count} map with 0/1/null keys, handling both
    integer-dtype (no NaN possible) and object/float-dtype (NaN possible)
    columns uniformly."""
    vc = series.value_counts(dropna=False)
    out = {'0': 0, '1': 0, 'null': 0, 'other': 0}
    for k, v in vc.items():
        if (isinstance(k, float) and pd.isna(k)) or k is None:
            out['null'] += int(v)
        else:
            try:
                ki = int(k)
                if ki == 0:
                    out['0'] += int(v)
                elif ki == 1:
                    out['1'] += int(v)
                else:
                    out['other'] += int(v)
            except (TypeError, ValueError):
                out['other'] += int(v)
    return out


# ── Stage B: pre-waterfall raw distribution ─────────────────────────────
def stage_b_pre_waterfall() -> dict:
    """Read the raw clif_respiratory_support.parquet directly. This is the
    data BEFORE clifpy's waterfall.ffill() and scaffold-row insertion. The
    0/1/null split is the single most informative cross-site comparator
    for H2b: MIMIC charts dense 0s + sparse 1s; a sparse-charting site
    shows near-zero `tracheostomy = 0` rows.

    Loaded via DuckDB scan over the raw parquet — no clifpy round-trip,
    no tz conversion, just the raw column.
    """
    raw_p = DATA_DIR / f"clif_respiratory_support.{FILETYPE}"
    if not raw_p.exists():
        logger.warning(
            f"Stage B raw source not found at {raw_p}; skipping pre-waterfall stage"
        )
        return {'observable': False}
    if FILETYPE != 'parquet':
        logger.warning(
            f"Stage B currently only supports parquet sources; got filetype={FILETYPE}"
        )
        return {'observable': False}
    rel = duckdb.sql(f"FROM '{raw_p}' SELECT hospitalization_id, tracheostomy")
    dist_rows = duckdb.sql(
        "FROM rel SELECT tracheostomy, COUNT(*) AS n GROUP BY 1"
    ).df()
    out = {'observable': True, 'rows_total': 0, 'rows_trach1': 0,
           'rows_trach0': 0, 'rows_null': 0, 'hosp_total': 0, 'hosp_trach1': 0}
    for _, row in dist_rows.iterrows():
        n = int(row['n'])
        out['rows_total'] += n
        if pd.isna(row['tracheostomy']):
            out['rows_null'] += n
        elif int(row['tracheostomy']) == 1:
            out['rows_trach1'] += n
        elif int(row['tracheostomy']) == 0:
            out['rows_trach0'] += n
    out['hosp_total'] = int(duckdb.sql(
        "FROM rel SELECT COUNT(DISTINCT hospitalization_id)"
    ).fetchone()[0])
    out['hosp_trach1'] = int(duckdb.sql(
        "FROM rel SELECT COUNT(DISTINCT hospitalization_id) WHERE tracheostomy = 1"
    ).fetchone()[0])
    return out


# ── Stage A: post-waterfall resp_p distribution ─────────────────────────
def stage_a_resp_p() -> tuple[pd.DataFrame, dict]:
    """Read cohort_resp_processed_bf.parquet (the post-waterfall resp_p
    that 03_outcomes.py consumes). The F5 two-branch dtype coercion in
    03_outcomes.py is a no-op for clean int64 inputs, so this measurement
    is effectively post-Stage-C as well.

    H2a + H2b cross-tabs surface here as well:
      - "first-row trach=1 hosps" → H2a (already-trached at row 1)
      - "NULL-LAG on first-trach-1 hosps" → H2b (sparse charting)
    """
    p = SITE_OUT / "cohort_resp_processed_bf.parquet"
    if not p.exists():
        logger.error(
            f"Missing {p} — run `make run SITE={SITE}` first"
        )
        sys.exit(1)
    resp_p = pd.read_parquet(p)

    dist = _trach_dist_map(resp_p['tracheostomy'])
    out = {
        'rows_total': int(len(resp_p)),
        'rows_trach1': dist['1'],
        'rows_trach0': dist['0'],
        'rows_null': dist['null'],
        'hosp_total': int(resp_p['hospitalization_id'].nunique()),
        'hosp_trach1': int(
            resp_p.loc[resp_p['tracheostomy'] == 1, 'hospitalization_id'].nunique()
        ),
        'h2a_first_row_trach1': 0,
        'h2b_null_lag_on_first_trach1': 0,
    }
    if out['hosp_trach1'] > 0:
        _ordered = resp_p.sort_values(['hospitalization_id', 'recorded_dttm'])
        _ordered['_prev_trach'] = (
            _ordered.groupby('hospitalization_id')['tracheostomy'].shift(1)
        )
        _first_t1 = (
            _ordered[_ordered['tracheostomy'] == 1]
            .groupby('hospitalization_id').head(1)
        )
        out['h2b_null_lag_on_first_trach1'] = int(
            _first_t1['_prev_trach'].isna().sum()
        )
        _first_rows = (
            _ordered.groupby('hospitalization_id').head(1)
            [['hospitalization_id', 'tracheostomy']]
        )
        out['h2a_first_row_trach1'] = int((_first_rows['tracheostomy'] == 1).sum())
    return resp_p, out


# ── Stage D: row-level flip detection (v1) + episode detection (v2) ────
def stage_d(resp_p: pd.DataFrame) -> dict:
    """Re-run the v1 LAG-based flip detector via DuckDB on the same resp_p
    that 03_outcomes.py sees. For v2 (episode-event detector), use the
    per-hosp `_trach_v2` column from outcomes_by_id_imvday.parquet — the
    row-level pandas state machine in 03_outcomes.py is too coupled to
    its host file to import cleanly, and the per-hosp rollup is what
    actually matters for the funnel readout.
    """
    duckdb.register("_resp_p", resp_p)
    _d_v1 = duckdb.sql("""
        WITH flipped AS (
            FROM _resp_p
            SELECT
                hospitalization_id, recorded_dttm, tracheostomy
                , _flip: CASE
                    WHEN LAG(tracheostomy) OVER w = 0
                        AND tracheostomy = 1 THEN 1 ELSE 0 END
            WINDOW w AS (PARTITION BY hospitalization_id ORDER BY recorded_dttm)
        )
        FROM flipped
        SELECT
            n_rows: COUNT(*)
            , n_flip: SUM(_flip)
            , n_hosp_flip: COUNT(DISTINCT hospitalization_id) FILTER (WHERE _flip = 1)
            , n_hosp_trach1: COUNT(DISTINCT hospitalization_id) FILTER (WHERE tracheostomy = 1)
    """).fetchone()
    out = {
        'v1_n_rows': int(_d_v1[0] or 0),
        'v1_n_flip': int(_d_v1[1] or 0),
        'v1_n_hosp_flip': int(_d_v1[2] or 0),
        'v1_n_hosp_trach1': int(_d_v1[3] or 0),
    }

    # v2 — read per-hosp episode-event count from the daily parquet's
    # `_trach_v2` column. Per-hosp granularity is what the funnel needs;
    # row-level v2 (the "_trach_event_v2 fired on N rows" metric) is
    # redundant with the per-hosp count (there's at most one event per
    # hosp by the `_trach_cum_v2 = 1` gating).
    daily_p = SITE_OUT / "outcomes_by_id_imvday.parquet"
    if daily_p.exists():
        daily = pd.read_parquet(daily_p, columns=['hospitalization_id', '_trach_v2'])
        v2_hosps = daily.loc[daily['_trach_v2'] == 1, 'hospitalization_id'].nunique()
        out['v2_n_hosp_event'] = int(v2_hosps)
    else:
        out['v2_n_hosp_event'] = None
    return out


# ── Stage E: daily-grain rollup ────────────────────────────────────────
def stage_e() -> dict:
    """Read outcomes_by_id_imvday.parquet. `_trach_1st` (v1, LAG-based)
    and `_trach_v2` (v2, episode-event) are both already columns there.
    """
    p = SITE_OUT / "outcomes_by_id_imvday.parquet"
    if not p.exists():
        logger.error(f"Missing {p}")
        sys.exit(1)
    df = pd.read_parquet(
        p, columns=['hospitalization_id', '_trach_1st', '_trach_v2']
    )
    return {
        'rows_total': int(len(df)),
        'v1_rows': int((df['_trach_1st'] == 1).sum()),
        'v1_hosps': int(
            df.loc[df['_trach_1st'] == 1, 'hospitalization_id'].nunique()
        ),
        'v2_rows': int((df['_trach_v2'] == 1).sum()),
        'v2_hosps': int(
            df.loc[df['_trach_v2'] == 1, 'hospitalization_id'].nunique()
        ),
        'hosp_total': int(df['hospitalization_id'].nunique()),
    }


# ── Stage F: per-hosp ever_trach + canonical + v2-based exit_mechanism ──
def stage_f() -> tuple[dict, pd.DataFrame]:
    """Re-derive Table 1's exit_mechanism row under both detectors.

    Reads cohort_meta_by_id.parquet (already has canonical exit_mechanism)
    AND outcomes_by_id_imvday.parquet (per-day _trach_v2). Computes
    ever_trach_v2 per hosp by MAX(_trach_v2) over days. Then runs the
    same mutually-exclusive CASE block as 04_covariates.py:1284-1293 but
    with the trach branch swapped to ever_trach_v2 — yields the v2-based
    variant. Both variant counts go into trach_table1_variants.csv.
    """
    meta_p = SITE_OUT / "cohort_meta_by_id.parquet"
    daily_p = SITE_OUT / "outcomes_by_id_imvday.parquet"
    if not meta_p.exists() or not daily_p.exists():
        logger.error(f"Missing {meta_p} or {daily_p}")
        sys.exit(1)

    meta = pd.read_parquet(meta_p)
    # Per-hosp v2 detector + the other outcome flags we need to reproduce
    # the exit_mechanism CASE block.
    daily = pd.read_parquet(
        daily_p,
        columns=[
            'hospitalization_id', '_trach_v2',
            '_extub_1st', '_success_extub', '_fail_extub', '_withdrawl_lst',
        ],
    )
    per_hosp = daily.groupby('hospitalization_id').agg(
        ever_trach_v2=('_trach_v2', 'max'),
        ever_extubated=('_extub_1st', 'max'),
        ever_success_extub=('_success_extub', 'max'),
        ever_failed_extub=('_fail_extub', 'max'),
        ever_withdrawl=('_withdrawl_lst', 'max'),
    ).fillna(0).astype(int).reset_index()

    # Pull discharge_category from meta for the death-on-IMV branch.
    df = meta[[
        'hospitalization_id', 'discharge_category', 'exit_mechanism',
    ]].merge(per_hosp, on='hospitalization_id', how='left')

    # Re-derive exit_mechanism_v2_based with the same CASE order as
    # 04_covariates.py:1284-1293.
    def _classify_v2(row) -> str:
        if row['ever_trach_v2'] == 1:
            return 'tracheostomy'
        ever_extub = row.get('ever_extubated') or 0
        dc = str(row.get('discharge_category') or '').strip().lower()
        if ever_extub == 0 and dc == 'expired':
            return 'died_on_imv'
        if row.get('ever_withdrawl') == 1:
            return 'palliative_extubation'
        if row.get('ever_failed_extub') == 1:
            return 'failed_extubation'
        if row.get('ever_success_extub') == 1:
            return 'successful_extubation'
        if ever_extub == 0:
            return 'discharge_on_imv'
        return 'unknown'

    df['exit_mechanism_v2_based'] = df.apply(_classify_v2, axis=1)

    n_hosps = int(len(df))
    canon_counts = (
        df['exit_mechanism'].value_counts(dropna=False).to_dict()
    )
    v2_counts = (
        df['exit_mechanism_v2_based'].value_counts(dropna=False).to_dict()
    )
    all_values = sorted(set(canon_counts.keys()) | set(v2_counts.keys()))
    variants = pd.DataFrame([
        {
            'site_name': SITE,
            'exit_mechanism_value': v,
            'n_canonical': int(canon_counts.get(v, 0)),
            'pct_canonical': _safe_pct(int(canon_counts.get(v, 0)), n_hosps),
            'n_v2_based': int(v2_counts.get(v, 0)),
            'pct_v2_based': _safe_pct(int(v2_counts.get(v, 0)), n_hosps),
            'delta': int(v2_counts.get(v, 0)) - int(canon_counts.get(v, 0)),
        }
        for v in all_values
    ])

    out = {
        'n_hosps': n_hosps,
        'canon_trach': int(canon_counts.get('tracheostomy', 0)),
        'v2_trach': int(v2_counts.get('tracheostomy', 0)),
    }
    out['delta_trach'] = out['v2_trach'] - out['canon_trach']
    return out, variants


# ── Main orchestration ─────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 70)
    logger.info(f"Trach funnel audit — site: {SITE}")
    logger.info("=" * 70)

    # Stage A — post-waterfall resp_p
    resp_p, a = stage_a_resp_p()
    logger.info(
        f"[trach funnel A] resp_p: {a['rows_total']:,} rows "
        f"| trach=0:{a['rows_trach0']:,} trach=1:{a['rows_trach1']:,} "
        f"trach=NaN:{a['rows_null']:,} "
        f"| hosps w/ any trach=1: {a['hosp_trach1']}/{a['hosp_total']} "
        f"| NULL-LAG on first-trach-1 hosps: {a['h2b_null_lag_on_first_trach1']} (H2b sig) "
        f"| first-row trach=1 hosps: {a['h2a_first_row_trach1']} (H2a sig)"
    )

    # Stage B — pre vs post-waterfall (raw CLIF)
    b = stage_b_pre_waterfall()
    if b['observable']:
        ffill_lift = a['rows_trach1'] - b['rows_trach1']
        logger.info(
            f"[trach funnel B pre] raw clif_respiratory_support: "
            f"trach=0:{b['rows_trach0']:,} trach=1:{b['rows_trach1']:,} "
            f"trach=NaN:{b['rows_null']:,} "
            f"| hosps w/ any trach=1: {b['hosp_trach1']}/{b['hosp_total']}"
        )
        logger.info(
            f"[trach funnel B post] ffill+scaffold lift trach=1 rows: "
            f"+{ffill_lift:,} (post {a['rows_trach1']:,} − pre {b['rows_trach1']:,})"
        )
    else:
        logger.info(
            "[trach funnel B] skipped — raw clif_respiratory_support not "
            "observable (Mode A external waterfall, missing source, or "
            "non-parquet filetype)"
        )

    # Stage D — row-level flip (v1) on the UNCOHORTED resp_p universe
    # (different scope from Stage E/F, which use the cohort-filtered daily
    # parquet). The decisive cohort-filtered v1↔v2 comparison lives at
    # Stage F; Stage D's value is the "hosps with trach=1 but no flip"
    # metric — direct H2/H2b indicator at the source granularity.
    d = stage_d(resp_p)
    v2_msg = (
        str(d['v2_n_hosp_event']) if d['v2_n_hosp_event'] is not None
        else "n/a (outcomes parquet missing)"
    )
    logger.info(
        f"[trach funnel D v1] (uncohorted resp_p) flips detected: {d['v1_n_flip']:,} rows "
        f"across {d['v1_n_hosp_flip']} hosps "
        f"| hosps w/ any trach=1: {d['v1_n_hosp_trach1']} "
        f"| hosps w/ trach=1 but no flip: {d['v1_n_hosp_trach1'] - d['v1_n_hosp_flip']} "
        f"(direct H2/H2b indicator at source grain)"
    )
    logger.info(
        f"[trach funnel D v2] (cohort-filtered daily parquet) _trach_v2=1: "
        f"{v2_msg} hosps. NB: this is cohort-filtered; the cleanest v1↔v2 "
        f"comparison is at Stage F (both detectors on the same cohort)."
    )

    # Stage E — daily rollup
    e = stage_e()
    logger.info(
        f"[trach funnel E] daily _trach_1st=1: {e['v1_rows']} hosp-days/{e['v1_hosps']} hosps "
        f"| daily _trach_v2=1: {e['v2_rows']} hosp-days/{e['v2_hosps']} hosps "
        f"| n_hosps_in_daily: {e['hosp_total']}"
    )

    # Stage F — per-hosp + final exit_mechanism (both variants)
    f, variants = stage_f()
    logger.info(
        f"[trach funnel F] exit_mechanism='tracheostomy' (canonical): {f['canon_trach']} hosps "
        f"| exit_mechanism_v2_based='tracheostomy': {f['v2_trach']} hosps "
        f"| candidate-fix delta: {f['delta_trach']:+d} hosps "
        f"(positive → v2-based detector recovers patients the v1 LAG missed)"
    )

    # ── Write CSVs ──────────────────────────────────────────────────────
    funnel_rows = [
        {
            'site_name': SITE,
            'stage': 'A_resp_p_post_waterfall',
            'n_rows_total': a['rows_total'],
            'n_rows_trach1': a['rows_trach1'],
            'n_rows_trach0': a['rows_trach0'],
            'n_rows_trach_null': a['rows_null'],
            'n_distinct_hosp_total': a['hosp_total'],
            'n_distinct_hosp_trach1': a['hosp_trach1'],
            'n_rows_trach1_v2': None,
            'n_distinct_hosp_trach1_v2': None,
            'pct_rows_trach1': _safe_pct(a['rows_trach1'], a['rows_total']),
            'pct_hosp_trach1': _safe_pct(a['hosp_trach1'], a['hosp_total']),
            'notes': (
                f"H2a first-row trach=1 hosps: {a['h2a_first_row_trach1']}; "
                f"H2b null-LAG on first-trach-1 hosps: {a['h2b_null_lag_on_first_trach1']}"
            ),
        },
    ]
    if b['observable']:
        funnel_rows.append({
            'site_name': SITE,
            'stage': 'B_raw_clif_pre_waterfall',
            'n_rows_total': b['rows_total'],
            'n_rows_trach1': b['rows_trach1'],
            'n_rows_trach0': b['rows_trach0'],
            'n_rows_trach_null': b['rows_null'],
            'n_distinct_hosp_total': b['hosp_total'],
            'n_distinct_hosp_trach1': b['hosp_trach1'],
            'n_rows_trach1_v2': None,
            'n_distinct_hosp_trach1_v2': None,
            'pct_rows_trach1': _safe_pct(b['rows_trach1'], b['rows_total']),
            'pct_hosp_trach1': _safe_pct(b['hosp_trach1'], b['hosp_total']),
            'notes': (
                'raw CLIF source; pre clifpy waterfall.ffill() + scaffold. '
                'Near-zero `trach=0` here is the H2b sparse-charting signature.'
            ),
        })
    funnel_rows.append({
        'site_name': SITE,
        'stage': 'D_flip_detected',
        'n_rows_total': d['v1_n_rows'],
        'n_rows_trach1': d['v1_n_flip'],
        'n_rows_trach0': None,
        'n_rows_trach_null': None,
        'n_distinct_hosp_total': None,
        'n_distinct_hosp_trach1': d['v1_n_hosp_flip'],
        'n_rows_trach1_v2': None,
        'n_distinct_hosp_trach1_v2': d['v2_n_hosp_event'],
        'pct_rows_trach1': _safe_pct(d['v1_n_flip'], d['v1_n_rows']),
        'pct_hosp_trach1': None,
        'notes': (
            f"v1 hosps w/ trach=1 but no flip: "
            f"{d['v1_n_hosp_trach1'] - d['v1_n_hosp_flip']} "
            f"(v1=0 + v2>0 → H2/H2b NULL-LAG bug)"
        ),
    })
    funnel_rows.append({
        'site_name': SITE,
        'stage': 'E_trach_1st_daily',
        'n_rows_total': e['rows_total'],
        'n_rows_trach1': e['v1_rows'],
        'n_rows_trach0': None,
        'n_rows_trach_null': None,
        'n_distinct_hosp_total': e['hosp_total'],
        'n_distinct_hosp_trach1': e['v1_hosps'],
        'n_rows_trach1_v2': e['v2_rows'],
        'n_distinct_hosp_trach1_v2': e['v2_hosps'],
        'pct_rows_trach1': _safe_pct(e['v1_rows'], e['rows_total']),
        'pct_hosp_trach1': _safe_pct(e['v1_hosps'], e['hosp_total']),
        'notes': 'per-day rollup; one =1 row per hosp by _flip_cum=1 / _cum_v2=1 gating',
    })
    funnel_rows.append({
        'site_name': SITE,
        'stage': 'F_ever_trach_per_hosp',
        'n_rows_total': f['n_hosps'],
        'n_rows_trach1': f['canon_trach'],
        'n_rows_trach0': None,
        'n_rows_trach_null': None,
        'n_distinct_hosp_total': f['n_hosps'],
        'n_distinct_hosp_trach1': f['canon_trach'],
        'n_rows_trach1_v2': f['v2_trach'],
        'n_distinct_hosp_trach1_v2': f['v2_trach'],
        'pct_rows_trach1': _safe_pct(f['canon_trach'], f['n_hosps']),
        'pct_hosp_trach1': _safe_pct(f['canon_trach'], f['n_hosps']),
        'notes': (
            f"canonical exit_mechanism='tracheostomy': {f['canon_trach']}; "
            f"v2_based: {f['v2_trach']}; delta: {f['delta_trach']:+d}"
        ),
    })

    funnel_csv = SHARE_QC / "trach_funnel.csv"
    pd.DataFrame(funnel_rows).to_csv(funnel_csv, index=False)
    logger.info(f"Saved: {funnel_csv} ({len(funnel_rows)} stages)")

    variants_csv = SHARE_QC / "trach_table1_variants.csv"
    variants.to_csv(variants_csv, index=False)
    logger.info(f"Saved: {variants_csv}")

    logger.info(
        "Funnel readout guide (lower-right cell of trach_funnel.csv is the bug stage):"
        "\n  - Stage A trach1=0 → H1 (no trach in source)"
        "\n  - Stage B raw trach0≈0 → H2b (sparse-charting site)"
        "\n  - Stage D v1=0, v2>0 → NULL-LAG bug confirmed (H2a or H2b)"
        "\n  - Stage F canonical=0, v2_based>0 → ship trach_table1_variants.csv's"
        " v2-based count into the manuscript Table 1 trach row"
    )


if __name__ == "__main__":
    main()
