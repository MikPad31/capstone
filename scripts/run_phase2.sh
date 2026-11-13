#!/usr/bin/env bash
# Full Phase 2 pipeline, resumable: grid cells whose predictions exist are skipped.
#   predictor split: which predictors vary within issuer-month (the grid reads this file)
#   main grid      : OLS, EN, RF x L0/L1/IC/L2 x h=1/3/12, all 341 predictors, common sample
#   predictor sets : L2 with the 40 within-issuer-varying predictors (all models), and with the
#                    301 issuer-constant ones (OLS, RF) as the pipeline check
# Then builds the tables and the data summary. Run from the repo root: bash scripts/run_phase2.sh
# Compare against the reported numbers afterwards: python scripts/check_results.py
set -euo pipefail
PY=${PY:-.venv/bin/python}
mkdir -p output/logs

$PY -m exploratory.within_issuer_predictors 2>&1 | tee -a output/logs/within_issuer_predictors.log
$PY -m exploratory.phase2_grid --models ols,enet,rf 2>&1 | tee -a output/logs/phase2_main.log
$PY -m exploratory.phase2_grid --models ols,enet,rf --levels L2 --predictor-set within 2>&1 | tee -a output/logs/phase2_within.log
$PY -m exploratory.phase2_grid --models ols,rf --levels L2 --predictor-set constant 2>&1 | tee -a output/logs/phase2_constant.log
$PY -m exploratory.phase2_results 2>&1 | tee -a output/logs/phase2_results.log
$PY -m exploratory.dm_weighting_check 2>&1 | tee -a output/logs/dm_weighting_check.log
$PY -m exploratory.data_summary 2>&1 | tee -a output/logs/data_summary.log
