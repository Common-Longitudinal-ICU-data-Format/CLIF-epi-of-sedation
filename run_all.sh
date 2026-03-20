#!/usr/bin/env bash
set -euo pipefail

LOGFILE="output/run_all_$(date '+%Y-%m-%d_%H%M%S').log"
mkdir -p output output_to_share/figures
exec > >(tee "$LOGFILE") 2>&1

SCRIPT_START=$SECONDS
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

run_step() {
    local name="$1"; shift
    log "START  $name"
    local step_start=$SECONDS
    "$@"
    local elapsed=$(( SECONDS - step_start ))
    log "DONE   $name  (${elapsed}s)"
}

log "Pipeline started"
run_step "uv sync"                        uv sync
run_step "01_cohort.py"                   uv run python code/01_cohort.py
run_step "02_exposure.py"                 uv run python code/02_exposure.py
run_step "03_outcomes.py"                 uv run python code/03_outcomes.py
run_step "04_covariates.py"               uv run python code/04_covariates.py
run_step "05_analytical_dataset.py"       uv run python code/05_analytical_dataset.py
run_step "06_table1.py"                   uv run python code/06_table1.py
run_step "07_analysis.py"                 uv run python code/07_analysis.py

TOTAL=$(( SECONDS - SCRIPT_START ))
log "Pipeline finished  (total ${TOTAL}s)"
log "Log saved to $LOGFILE"
