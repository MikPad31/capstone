#!/usr/bin/env bash
# Full Phase 2 grid, resumable: cells whose predictions exist are skipped.
#   main grid      : OLS, EN, RF x L0/L1/IC/L2 x h=1/3/12, all 341 predictors, common sample
#   predictor sets : L2 with the 40 within-issuer-varying predictors (all models), and with the
#                    301 issuer-constant ones (OLS, RF) as the pipeline check
# Then builds the tables. Run from the repo root: bash scripts/run_phase2.sh
set -euo pipefail
PY=${PY:-.venv/bin/python}
mkdir -p output/logs

$PY -m exploratory.phase2_grid --models ols,enet,rf 2>&1 | tee -a output/logs/phase2_main.log
$PY -m exploratory.phase2_grid --models ols,enet,rf --levels L2 --predictor-set within 2>&1 | tee -a output/logs/phase2_within.log
$PY -m exploratory.phase2_grid --models ols,rf --levels L2 --predictor-set constant 2>&1 | tee -a output/logs/phase2_constant.log
$PY -m exploratory.phase2_results 2>&1 | tee -a output/logs/phase2_results.log
