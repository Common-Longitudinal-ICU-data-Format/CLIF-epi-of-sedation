.PHONY: mo run table1 tables report descriptive cascade qc weight-audit agg clean-legacy _switch _descriptive_scripts

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
#   output_to_agg/                  — Phase-2 cross-site pooled outputs
#                                     (produced by code/agg/ scripts; not
#                                     yet implemented — see `make agg`)
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
# Full pipeline: 01 → 09 (slowest; first run includes compute_ase + SOFA recompute).
#
# Weight-QC bootstrap (2-pass): 01_cohort.py reads the weight-QC drop list
# from output/{site}/qc/weight_qc_drop_list.parquet if it exists. On a fresh
# repo the file doesn't exist, so the FIRST `make run` skips weight QC and
# completes the whole pipeline. Then `make weight-audit SITE=...` produces
# the drop list (it consumes 05's modeling_dataset.parquet). A SECOND
# `make run` then applies the drop. After the first round-trip, both targets
# stay in sync.
run: _switch
	uv sync
	uv run python code/01_cohort.py
	uv run python code/02_exposure.py
	uv run python code/03_outcomes.py
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	$(MAKE) weight-audit
	uv run python code/01_cohort.py
	uv run python code/02_exposure.py
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py
	uv run python code/07_descriptive.py
	uv run python code/08_models.py
	uv run python code/08b_models_cascade.py
	$(MAKE) _descriptive_scripts
	uv run python code/09_report.py

# Fast Table 1-only refresh — skip 01–03 since their outputs don't change here
table1: _switch
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py

# ── Liberation cascade (4-stage modeling sibling to 08) ──────────────
# Decomposes the sedation→liberation pathway into 4 conditional stages
# (eligibility → SBT → extubation → success). Reads modeling_dataset.parquet
# + sbt_outcomes_daily.parquet. Outputs cascade_*.{png,csv} under
# output_to_share/{site}/models/ flat with prefix.
cascade: _switch
	uv run python code/08b_models_cascade.py

# Refresh all tables + descriptive outputs + compiled PDF
# (common iterate-and-share workflow).
# Includes 02_exposure because we track n_hours columns that flow into 05/07.
tables: _switch
	uv run python code/02_exposure.py
	uv run python code/04_covariates.py
	uv run python code/05_modeling_dataset.py
	uv run python code/06_table1.py
	uv run python code/07_descriptive.py
	$(MAKE) _descriptive_scripts
	uv run python code/09_report.py

# Regenerate just the PDF from existing CSVs/PNGs (fastest, no compute)
report: _switch
	uv run python code/09_report.py

# Run only the per-figure scripts under code/descriptive/ and rebuild the PDF.
# Assumes 05_modeling_dataset.py has already run for the active site — each
# descriptive script reads output/{site}/exposure_dataset.parquet (and some
# read modeling_dataset.parquet). Skips the
# expensive 01–08 compute. Useful for iterating on the nocturnal up-titration
# figures (histograms, stacked bars, subcohort table) without re-running models.
descriptive: _switch _descriptive_scripts
	uv run python code/09_report.py

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
# scripts live in code/agg/ and write to output_to_agg/. Reads each site's
# output_to_share/<site>/ folder read-only; never runs pipeline scripts
# (01–09). Phase 1 (per-site runs) is complete by the time this runs.
#
# Currently implemented (descriptive-only): pooled Table 1, cross-site cohort
# stats, cross-site stacked-bar prevalence, cross-site trajectory overlay.
# Model-coefficient forest plots deferred to a later pass.
#
# Anonymization: set ANONYMIZE_SITES=1 to relabel sites as "Site A"/"Site B"/…
# in all outputs. Default is real names (mimic, UCMC, ...).
agg:
	@for script in code/agg/*.py; do \
		case "$$(basename $$script)" in _*) continue ;; esac; \
		echo "→ $$script"; \
		uv run python $$script || exit 1; \
	done

# ── QC tool: per-patient interactive trajectory dashboard ───────────
# Launches a Plotly Dash app on http://localhost:8050. Pick a site + a
# hospitalization_id (or random-sample) and inspect the full 5-panel
# timeline (sedatives, pressors, assessments, resp, vitals) with clinical
# event overlays. Per-patient wide-dataset load; LRU-cached so repeat
# views are instant. Loads only the selected patient's data so full-cohort
# memory is never materialized.
qc:
	uv run python code/qc/trajectory_viewer.py

# ── QC: weight-availability audit + drop-list generation ─────────────
# Federated-safe audit of the per-kg weight used to convert sedative doses.
# Replicates clifpy's per-admin ASOF, characterizes admission-fallback rate,
# compares Stage A vs Stage B weights, and generates a hospitalization-level
# drop list per the configured exclusion criteria. Phase 1 of the weight-QC
# work (audit-only — drops are applied in Phase 2 via 01_cohort.py).
#
# Outputs:
#   output_to_share/{site}/qc/weight_qc_summary.csv      (federated)
#   output_to_share/{site}/qc/weight_qc_exclusions.csv   (federated)
#   output_to_share/{site}/qc/weight_impact_comparison.csv (federated)
#   output_to_share/{site}/figures/weight_audit.png      (federated)
#   output/{site}/qc/weight_qc_drop_list.parquet         (PHI-internal)
#   output/{site}/qc/weight_audit_examples.csv           (PHI-internal)
#
# Tunable thresholds via env vars:
#   WEIGHT_QC_MAX_JUMP_KG=20              (jump rule, kg per 24h)
#   WEIGHT_QC_MAX_JUMP_HOURS=24
#   WEIGHT_QC_MAX_RANGE_KG=30             (range rule)
#   WEIGHT_QC_RANGE_RULE_ON=1             (off by default)
weight-audit: _switch
	uv run python code/qc/weight_audit.py

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
	@echo "Pass FORCE=1 to delete them."
	@if [ "$(FORCE)" = "1" ]; then \
		find output output_to_share -maxdepth 1 -type f \
			! -name '.gitignore' ! -name '.DS_Store' ! -name '*.yaml' ! -name 'README*' -delete 2>/dev/null; \
		find output_to_share/figures -maxdepth 1 -type f ! -name 'README*' -delete 2>/dev/null; \
		echo "Deleted."; \
	fi
