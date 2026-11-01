#!/usr/bin/env bash
# Full reproducible analysis chain (all 23 participants, all 20 protocol sessions).
# Detector priority is frozen in config.yaml (dev split); the primary cohort run
# uses leave-one-subject-out priorities so no participant's data influences
# the detector chosen for it.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
JOBS=${JOBS:-23}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
$PY run_method_selection.py --jobs "$JOBS"
$PY run_method_selection.py --split all --jobs "$JOBS" \
    --output paper_results/method_selection_all.csv \
    --summary paper_results/tables/method_selection_all_summary.csv
$PY run_all_participants.py --data-root Dataset --jobs "$JOBS" --force \
    --loso paper_results/method_selection_all.csv
$PY run_stats.py
$PY run_ablation.py --jobs "$JOBS"
$PY run_window_analysis.py --jobs "$JOBS"
$PY run_fiducial_study.py --jobs "$JOBS"
$PY -W ignore run_baselines.py --jobs "$JOBS"
$PY -W ignore -m analysis.vendor_control --jobs "$JOBS"
$PY generate_figures.py --rebuild-cache --jobs "$JOBS"
# Manuscript numbers, tables and figures (only when the paper folder is next to this repository)
PAPER=${PAPER:-../Assessment-of-PPG-Pipline-Paper}
if [ -d "$PAPER" ]; then
    $PY -W ignore -m analysis.paper_assets --paper-dir "$PAPER"
fi
