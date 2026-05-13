.PHONY: mo run table1 mortality tables report descriptive cascade qc weight-audit weight-diagnostic trach-funnel agg agg-local clean-legacy _switch _descriptive_scripts _agg_run

# ── Site selection ───────────────────────────────────────────────────
# Usage:
#   make run                   — use current config/config.json as-is
#                                 (default flow for collaborators at other sites)
#   make run SITE=mimic        — swap config/mimic_config.json → config/config.json, then run
#   make run SITE=ucmc         — swap config/ucmc_config.json  → config/config.json, then run
#
# Same SITE=... flag works on run / tables / table1 / report (via the
# shared _switch prerequisite). All three of config/{config,mimic_config,
# ucmc_config}.json are gitignored, so copy-swap causes no repo churn.
#
# Outputs are written under per-site subdirectories so multiple sites can
# coexist on disk without collision:
#   output/{site}/                  — pipeline intermediates (PHI-tier)
#   output_to_share/{site}/         — shareable CSVs + compiled PDF
#   output_to_share/{site}/figures/ — PNGs
#   output_to_agg/descriptive/      — Phase-2 cross-site descriptive outputs
#                                     (CSVs + figures side-by-side)
#   output_to_agg/models/           — Phase-2 cross-site model outputs
#                                     (forests, pooled coeffs, RCS overlays)
SITE ?=
site ?=

# ── Interactive editing ──────────────────────────────────────────────
mo:
	uv run marimo edit --watch

# Shared prerequisite: optionally swap active config based on SITE.
# Accepts SITE=... (canonical) or site=... (lowercase fallback, since
# Make variables are case-sensitive and a silent miss here previously
# let runs target the wrong site). Errors loudly if both are set to
# different values, and always prints the active site so a typo can't
# masquerade as a silent run against the prior config.
_switch:
	@_active="$(SITE)"; \
	if [ -z "$$_active" ] && [ -n "$(site)" ]; then \
		_active="$(site)"; \
		echo "→ Note: 'site=$(site)' picked up — Make variables are case-sensitive; prefer SITE=$(site)"; \
	fi; \
	if [ -n "$(SITE)" ] && [ -n "$(site)" ] && [ "$(SITE)" != "$(site)" ]; then \
		echo "ERROR: both SITE=$(SITE) and site=$(site) set with different values" >&2; exit 1; \
	fi; \
	if [ -n "$$_active" ]; then \
		if [ ! -f "config/$${_active}_config.json" ]; then \
			echo "ERROR: config/$${_active}_config.json not found" >&2; exit 1; \
		fi; \
		cp "config/$${_active}_config.json" config/config.json; \
		echo "→ Using config/$${_active}_config.json (active config now matches SITE=$$_active)"; \
	else \
		_cur=$$(grep -o '"site_name"[^,}]*' config/config.json 2>/dev/null | head -1); \
		echo "→ No SITE flag passed; using existing config/config.json ($$_cur)"; \
	fi

# ── Phase 1: per-site pipeline ───────────────────────────────────────
# Pipeline: 01 → 08 + descriptive scripts. PDF report generation (09)
# is intentionally EXCLUDED — invoke `make report` explicitly when the
# bundled PDF is wanted. The rationale: the report is a presentation
# layer over already-written CSVs/PNGs and slow (matplotlib renders 30+
# pages); running it on every `make run` is wasteful when a site is
# iterating on upstream stages.
#
# B3 refactor (2026-05): weight-QC value checks are now computed INSIDE
# 01_cohort.py via `_utils.compute_weight_qc_exclusions`. The prior 2-pass
# `make run` → `make weight-audit` → `make run` round-trip is gone — saves
# roughly half the wall-clock per site. `make weight-diagnostic` (alias of
# the old `make weight-audit`) is preserved for the federated audit CSV /
# PNG, but is no longer required for cohort definition.
#
# Cache controls live in the per-site config (`config/<site>_config.json`):
#   rerun_waterfall    bool  force waterfall recompute (default false)
#   rerun_sofa_24h     bool  force SOFA recompute      (default false)
#   rerun_ase          bool  force ASE recompute       (default false)
#   path_to_waterfall_processed_resp_table   str|null
#                            if set + file exists, use it directly instead
#                            of running the waterfall (predicate-pushdown
#                            filter to cohort). Bypass by setting to null.
# To force a one-shot rebuild without editing config, delete the cache file:
#   rm output/<site>/cohort_resp_processed_bf.parquet  &&  make run SITE=...
#
# Other env-var knobs (kept as env vars — different ergonomics):
#   WEIGHT_QC_MAX_JUMP_KG=15 / WEIGHT_QC_MAX_JUMP_HOURS=24
#   WEIGHT_QC_MAX_RANGE_KG=30 / WEIGHT_QC_RANGE_RULE_ON=1
#   SEDDOSE_CLAMP=0     disable the per-hour clinical-ceiling clamp on
#                       sedation rates (M1). Default is on (cap to ceiling,
#                       not NULL). seddose_by_id_imvhr_raw.parquet is
#                       ALWAYS written alongside the canonical clamp-aware
#                       parquet so users can diff without rerunning.
run: _switch
	uv sync
	uv run python code/01_cohort.py
	uv run python code/02_exposure.py
	uv run python code/03_outcomes.py
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py
	uv run python code/08_models.py
	# 08b_models_cascade.py is INTENTIONALLY shelved from `make run` —
	# the 4-stage cascade is deferred analysis. Run on demand via
	# `make cascade SITE=...` if a re-investigation is needed.
	$(MAKE) _descriptive_scripts
	# 09_report.py is INTENTIONALLY shelved from `make run` — the
	# bundled PDF is a presentation layer over already-written CSVs/PNGs.
	# Run on demand via `make report SITE=...` when the PDF is wanted.

# Fast Table 1-only refresh — skip 01–03 since their outputs don't change here
table1: _switch
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py

# Mortality addendum — for sites that have already run the full pipeline and
# just need to refresh Table 1 with the new in-hospital mortality variables
# (`discharge_category`, `died_in_hospital`, `died_or_hospice`). Skips all
# upstream re-derivation since `discharge_category` already lives on
# output/{site}/cohort_meta_by_id.parquet from the prior `make run`.
# Overwrites output_to_share/{site}/models/table1_*.csv; only
# table1_categorical.csv has new content (the other three are deterministic).
# Site shares the updated table1_categorical.csv.
mortality: _switch
	uv run python code/06_table1.py

# ── Liberation cascade (4-stage modeling sibling to 08) ──────────────
# Decomposes the sedation→liberation pathway into 4 conditional stages
# (eligibility → SBT → extubation → success). Reads modeling_dataset.parquet
# + sbt_outcomes_daily.parquet. Outputs cascade_*.{png,csv} under
# output_to_share/{site}/models/ flat with prefix.
cascade: _switch
	uv run python code/08b_models_cascade.py

# Refresh all tables + descriptive outputs.
# (common iterate-and-share workflow.)
# Includes 02_exposure because we track n_hours columns that flow into 05/07.
# PDF report generation is INTENTIONALLY EXCLUDED — invoke `make report`
# explicitly when the bundled PDF is wanted (same rationale as `make run`).
tables: _switch
	uv run python code/02_exposure.py
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py
	$(MAKE) _descriptive_scripts

# Regenerate just the PDF from existing CSVs/PNGs (fastest, no compute)
report: _switch
	uv run python code/09_report.py

# Run only the per-figure scripts under code/descriptive/.
# Assumes 05_modeling_dataset.py has already run for the active site — each
# descriptive script reads output/{site}/exposure_dataset.parquet (and some
# read modeling_dataset.parquet). Skips the
# expensive 01–08 compute. Useful for iterating on the nocturnal up-titration
# figures (histograms, stacked bars, subcohort table) without re-running models.
# PDF report generation is INTENTIONALLY EXCLUDED — invoke `make report`
# explicitly when the bundled PDF is wanted.
descriptive: _switch _descriptive_scripts

# Internal: glob every *.py under code/descriptive/ (skipping _shared.py and
# other _-prefixed helpers) and run them in turn. Exits on first failure.
# Reused by `run`, `tables`, and the public `descriptive` target — so adding
# a new figure under code/descriptive/ is auto-picked-up without Makefile edits.
_descriptive_scripts:
	@for script in code/descriptive/*.py; do \
		case "$$(basename $$script)" in _*) continue ;; esac; \
		echo "→ $$script"; \
		uv run python $$script || exit 1; \
	done

# ── Phase 2: cross-site aggregation (coordinator-side) ───────────────
# Follows the VC convention from vc_proj_patterns.md §6 — aggregation
# scripts live in code/agg/ and write to output_to_agg/{descriptive,models}/.
# Reads each site's
# <input_root>/<site>/ folder read-only; never runs pipeline scripts
# (01–09). Phase 1 (per-site runs) is complete by the time this runs.
#
# Input source (controlled by the AGG_INPUT_DIR env var, plumbed through
# code/agg/_shared.py:SHARE_ROOT):
#   make agg          → reads from the shared Box folder (default; coordinator
#                       workflow — assumes each site has uploaded their
#                       output_to_share/<site>/ bundle to AGG_BOX_DIR).
#   make agg-local    → reads from the repo-local output_to_share/ tree
#                       (use when iterating on agg scripts against a local
#                       copy without round-tripping through Box).
# Override the Box path on the command line if needed:
#   make agg AGG_BOX_DIR=/some/other/mount
#
# Both targets write pooled outputs under output_to_agg/ in the repo,
# split by output type via the `category=` kwarg on the save helpers:
#   output_to_agg/descriptive/  → cohort stats, dose patterns, LOS, Table 1,
#                                 sed-dose-by-hour, night-day-diff trajectory
#   output_to_agg/models/       → meta-analysis pooled coeffs, forest plots
#                                 (night-day / daytime / SBT-sensitivity),
#                                 RCS marginal-effects overlays
#
# Implemented: pooled Table 1, cross-site cohort stats, cross-site stacked-
# bar prevalence, cross-site trajectory overlay, cross-site forest plots
# (night-day diff / daytime / SBT sensitivity), cross-site marginal-effects
# RCS overlays, AND DerSimonian-Laird random-effects meta-analysis pooling
# (meta_analysis_cross_site.py reads each site's models_coeffs.csv).
#
# Anonymization: set ANONYMIZE_SITES=1 to relabel sites as "Site A"/"Site B"/…
# in all outputs. Default is real names (mimic, UCMC, ...). Orthogonal to the
# input-source switch — composes with both `make agg` and `make agg-local`.
AGG_BOX_DIR := /Users/wliao0504/Library/CloudStorage/Box-Box/CLIF/Projects/CLIF-epi-of-sedation

agg:
	@AGG_INPUT_DIR="$(AGG_BOX_DIR)" $(MAKE) _agg_run

agg-local:
	@$(MAKE) _agg_run

# Shared loop body — recursing through $(MAKE) propagates AGG_INPUT_DIR to
# each `uv run python` subprocess without export-ing it into the user's shell.
_agg_run:
	@echo "→ Aggregating from: $${AGG_INPUT_DIR:-output_to_share}"
	@echo "→ Writing to:       output_to_agg/{descriptive,models}/"
	@for script in code/agg/*.py; do \
		case "$$(basename $$script)" in _*) continue ;; esac; \
		echo "→ $$script"; \
		uv run python $$script || exit 1; \
	done
	@echo ""
	@echo "Done. New artifacts:"
	@echo "  output_to_agg/descriptive/  ($$(ls output_to_agg/descriptive 2>/dev/null | wc -l | tr -d ' ') files)"
	@echo "  output_to_agg/models/       ($$(ls output_to_agg/models 2>/dev/null | wc -l | tr -d ' ') files)"
	@if [ -d output_to_agg/figures ] || ls output_to_agg/*.csv >/dev/null 2>&1; then \
		echo ""; \
		echo "Note: legacy artifacts still present at output_to_agg/figures/ and output_to_agg/*.csv —"; \
		echo "      run 'make clean-legacy' to list them, 'make clean-legacy FORCE=1' to delete."; \
	fi

# ── QC tool: per-patient interactive trajectory dashboard ───────────
# Launches a Plotly Dash app on http://localhost:8050. Pick a site + a
# hospitalization_id (or random-sample) and inspect the full 5-panel
# timeline (sedatives, pressors, assessments, resp, vitals) with clinical
# event overlays. Per-patient wide-dataset load; LRU-cached so repeat
# views are instant. Loads only the selected patient's data so full-cohort
# memory is never materialized.
qc:
	uv run python code/qc/trajectory_viewer.py

# ── QC: weight-availability diagnostic (federated audit CSVs/PNG) ────
# Federated-safe audit of the per-kg weight used to convert sedative doses.
# Characterizes the same three drop criteria 01_cohort.py applies, so site
# QC reviewers can compare counts/distributions across sites without
# re-running the pipeline.
#
# B3 refactor (2026-05): no longer produces weight_qc_drop_list.parquet —
# 01_cohort.py now computes the same drop sets in-memory via
# `_utils.compute_weight_qc_exclusions`. `make run` no longer depends on
# this target.
#
# Outputs (all federated-safe — no row-level PHI):
#   output_to_share/{site}/qc/weight_qc_summary.csv
#   output_to_share/{site}/qc/weight_qc_exclusions.csv
#   output_to_share/{site}/qc/weight_impact_comparison.csv
#   output_to_share/{site}/qc/weight_audit.png
#
# Tunable thresholds via env vars (defaults match 01_cohort.py):
#   WEIGHT_QC_MAX_JUMP_KG=20              (jump rule, kg per 24h)
#   WEIGHT_QC_MAX_JUMP_HOURS=24
#   WEIGHT_QC_MAX_RANGE_KG=30             (range rule)
#   WEIGHT_QC_RANGE_RULE_ON=1             (off by default)
weight-diagnostic weight-audit: _switch
	uv run python code/qc/weight_audit.py

# ── Trach funnel diagnostic (optional, run only on sites with issue) ──
# Standalone diagnostic for triangulating an empty
# `exit_mechanism = 'tracheostomy'` bucket in Table 1. Reads pipeline
# outputs + raw CLIF respiratory_support; writes federation-safe CSVs to
# output_to_share/{site}/qc/. Independent of `make run`. Sites without
# the issue do not need to run it. See docs/audit_tracker.md F13.
#
# Outputs (all federated-safe — aggregates only, no row-level PHI):
#   output_to_share/{site}/qc/trach_funnel.csv          (6-stage funnel)
#   output_to_share/{site}/qc/trach_table1_variants.csv (canonical vs v2-based
#                                                        exit_mechanism preview)
trach-funnel: _switch
	uv run python code/qc/trach_funnel_audit.py

# ── One-time cleanup of legacy flat outputs (pre-site-subdir) ────────
# Lists (and optionally deletes) artifacts left at the top level of
# output/ and output_to_share/ from before the site-subdir refactor.
# Per-site subdirs (mimic/, ucmc/, etc.) are preserved.
clean-legacy:
	@echo "Legacy un-scoped outputs (top-level files in output/ and output_to_share/):"
	@find output output_to_share -maxdepth 1 -type f \
		! -name '.gitignore' ! -name '.DS_Store' ! -name '*.yaml' ! -name 'README*' -print 2>/dev/null || true
	@find output_to_share/figures -maxdepth 1 -type f ! -name 'README*' -print 2>/dev/null || true
	@echo ""
	@echo "Legacy cross-site outputs (pre-descriptive/models restructure):"
	@find output_to_agg -maxdepth 1 -type f ! -name '.gitignore' ! -name 'README*' -print 2>/dev/null || true
	@find output_to_agg/figures -maxdepth 1 -type f ! -name 'README*' -print 2>/dev/null || true
	@echo ""
	@echo "Pass FORCE=1 to delete them."
	@if [ "$(FORCE)" = "1" ]; then \
		find output output_to_share -maxdepth 1 -type f \
			! -name '.gitignore' ! -name '.DS_Store' ! -name '*.yaml' ! -name 'README*' -delete 2>/dev/null; \
		find output_to_share/figures -maxdepth 1 -type f ! -name 'README*' -delete 2>/dev/null; \
		find output_to_agg -maxdepth 1 -type f ! -name '.gitignore' ! -name 'README*' -delete 2>/dev/null; \
		find output_to_agg/figures -maxdepth 1 -type f ! -name 'README*' -delete 2>/dev/null; \
		rmdir output_to_agg/figures 2>/dev/null || true; \
		echo "Deleted."; \
	fi
