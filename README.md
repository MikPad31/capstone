# Decomposing Machine-Learning Predictability in Corporate Bond Returns

DSE4101 Capstone Project in Data Science and Economics I, Semester 1 AY2026/27. Group 1:
Avaneesh Reddy, Gan Zhi Yu Charlene, Padua Mikhail Anthony Gagni, Tan Shyan Q'yan Kate,
Vanisha Muthu, Wang Yutian.

**Question.** How do the economic components of machine-learning predictability in corporate
bond returns vary across forecast horizons?

**Approach.** Each bond's return is split into nested targets, and each target is forecast with
OLS, Elastic Net and Random Forest at 1, 3 and 12 months under one out-of-sample framework
(expanding window from 2015-01, refit yearly, horizon embargo). Forecasts are compared with
out-of-sample R² and modified Diebold–Mariano tests.

**Primary specification** (following the Phase 2 proposal feedback): every level and every
horizon is scored on the same 385,119 bond-months (2015-01 to 2021-11), and L2 forecasts are
demeaned within issuer-month before scoring (IC forecasts averaged within it), since the targets
satisfy those constraints by construction. Each horizon's own sample with raw forecasts is kept
as an appendix. IC is fitted and scored on bond rows, so issuers with more bonds weigh more.

| Level | Target |
|---|---|
| **L0** | total return in excess of the risk-free rate (`retxrf`) |
| **L1** | return in excess of a duration-matched Treasury (`retx`) |
| **IC** | issuer-common: equal-weighted mean of L1 across the issuer's bonds that month |
| **L2** | bond-specific: L1 − IC, so L1 = IC + L2 exactly |

**Main result.** Only the bond-specific component is reliably forecastable: at 1 month its R²
is 1.60% (OLS), 1.42% (Elastic Net) and 1.29% (Random Forest), each significant against a zero
forecast (DM p < 0.01), and no model significantly beats another. No other component is
significant at any horizon. Full tables, primary specification first:
[`results/phase2/tables.md`](results/phase2/tables.md).

---

## Contents

1. [Data](#data)
2. [Setup](#setup)
3. [Reproducing the results](#reproducing-the-results)
4. [Where each reported number comes from](#where-each-reported-number-comes-from)
5. [Repository structure](#repository-structure)
6. [Conventions](#conventions)

---

## Data

Two public sources from the Open Source Bond Asset Pricing project (Dickerson, Nozawa and
Robotti, <https://openbondassetpricing.com>). Neither is committed here; `scripts/get_data.py`
downloads and verifies both.

| File | Source | Size | Contents |
|---|---|---|---|
| `OSBAP_ML_Panel_Oct_2024.pkl` | [OSBAP ML panel (2024)](https://openbondassetpricing.com/wp-content/uploads/2024/10/OSBAP_ML_Panel_Oct_2024.zip) | 886 MB zip, 3.3 GB unpacked | 1,102,569 bond-months × 381 columns: 19,768 bonds, 2,030 issuers, 2002-07 to 2022-11; 341 predictors rank-scaled to [−1, 1] with no missing values; identifiers (`date`, `cusip`, `gvkey`, …) |
| `dnr_ml_predictions/predictions.parquet` | [ML forecasts and realized returns](https://openbondassetpricing.com/wp-content/uploads/2026/04/dnr_ml_predictions.zip) | 201 MB zip | Realized next-month returns (`retd`, `retx`, `retxrf`); `retx` and `retxrf` are joined onto the panel as targets, plus the authors' own forecasts, used as an external benchmark. Its `README.txt` documents the columns. |
| `OSBAP_ML_Panel_columns.txt` | this repo, `scripts/` | 5 KB | Column positions of the pickle, which `src/data.py` uses to load only the columns it needs. It does not ship with the panel; `get_data.py` copies it into place. |

```bash
python scripts/get_data.py            # downloads what is missing into data/raw/, then verifies
python scripts/get_data.py --verify   # verifies an existing copy only (~10 s)
```

Verification checks exact file sizes against the published archives, then loads the panel and
checks its row count, bonds, months and the target join. A truncated download fails here rather
than producing wrong numbers later.

Already have the data elsewhere? Point the code at it instead of making a second 3 GB copy:

```bash
export DSE4101_DATA_DIR=/path/to/folder/with/the/files      # Windows: $env:DSE4101_DATA_DIR = "..."
```

---

## Setup

Python 3.11 or later (developed on 3.13). A 16 GB machine is enough: the pipeline peaks at about
9 GB of memory.

```bash
git clone https://github.com/MikPad31/capstone.git
cd capstone
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/get_data.py
pytest test/                                          # 151 tests, no data needed, ~5 s
```

`pandas` is pinned below 3.0 because the column-selective loader in `src/data.py` hooks pandas
2.x pickle internals.
`requirements-lock.txt` lists the exact versions the reported results were produced with
(`pip install -r requirements-lock.txt` to match them).

---

## Reproducing the results

Run everything from the repo root. Regenerated files go to `output/` (gitignored); the numbers
we report are committed in `results/`, and `scripts/check_results.py` compares the two.

| Step | Command | Time (M1 Pro, 16 GB) | Writes |
|---|---|---|---|
| 1. Data | `python scripts/get_data.py` | download-bound | `data/raw/` |
| 2. Phase 2 pipeline | `bash scripts/run_phase2.sh` (Windows: `powershell -ExecutionPolicy Bypass -File scripts\run_phase2.ps1`) | ~2.5–3 h of compute | `output/phase2/`, `output/data_summary/`, `output/within_issuer_predictors.csv` |
| 3. Earlier, separate replication of Bali et al. (optional; the reported replication is in step 2) | `python -m exploratory.bali_replication` then `python -m exploratory.bali_oos` | see `exploratory/README.md` | `output/replication/` |
| 4. Compare | `python scripts/check_results.py` | seconds | — |

Step 2 runs, in order: the within-issuer predictor split; the 36-cell main grid (3 models × 4
targets × 3 horizons); L2 with only the 40 within-issuer predictors; L2 with only the 301
issuer-constant predictors (a pipeline check whose R² must be zero); the result tables and
Diebold–Mariano tests; and the data summary. It is resumable: a cell whose predictions already
exist is skipped, so an interrupted run picks up where it stopped. Logs go to `output/logs/`.

**Quicker check of the headline result.** Instead of the full step 2, fit only the 1-month L2
cells (a few minutes for OLS, longer for Random Forest), then compare; `check_results.py`
compares just the cells that were rerun:

```bash
python -m exploratory.within_issuer_predictors
python -m exploratory.phase2_grid --models ols,rf --levels L2 --horizons 1
python -m exploratory.data_summary
python scripts/check_results.py
```

The machine must stay awake. A laptop that sleeps pauses the run; it does not fail, it just takes
longer in wall-clock time.

---

## Where each reported number comes from

| Reported item | File in `results/` | Produced by |
|---|---|---|
| Data description, summary statistics, variance split, monthly return figure | `data_summary/` | `exploratory/data_summary.py` |
| 40 within-issuer vs 301 issuer-constant predictors | `within_issuer_predictors.csv` | `exploratory/within_issuer_predictors.py` |
| **Primary:** R² and DM test vs zero for every model, target and horizon on the common sample | `phase2/primary_scores.csv`, `phase2/tables.md` | `exploratory/phase2_results.py` (from the forecasts of `phase2_grid.py`) |
| **Primary:** DM tests model vs model, and 341 vs 40 predictors | `phase2/primary_dm_tests.csv` | `exploratory/phase2_results.py` |
| **Primary:** sub-periods (pre-COVID, 2020, 2021) | `phase2/primary_subperiods.csv` | `exploratory/phase2_results.py` |
| Replication inside the pipeline (OLS / EN / RF, L0, 1 month, three samples) | `phase2/primary_replication.csv` | `exploratory/phase2_results.py` |
| Appendix: each horizon's own sample, raw forecasts (R², DM tests, sub-periods) | `phase2/scores.csv`, `phase2/dm_tests.csv`, `phase2/subperiods.csv` | `exploratory/phase2_grid.py`, `exploratory/phase2_results.py` |
| Chosen hyperparameters per refit (e.g. RF tree depth) | `phase2/diagnostics/` | `exploratory/phase2_grid.py` |
| DM tests with bond-month weights instead of month-equal weights (no 5% verdict changes) | `phase2/dm_weighting_check.csv` | `exploratory/dm_weighting_check.py` |
| DNR authors' forecasts on our test rows | `phase2/dnr_benchmark.csv` | `exploratory/phase2_results.py` |
| Earlier, separate replication of Bali et al. (monthly refits from 2007-07, full panel: OLS 0.31%, RF 0.55%) | `replication/` | `exploratory/bali_replication.py`, `exploratory/bali_oos.py` |

The forecasts themselves (`output/phase2/preds/`, 534 MB) are regenerated by step 2 and are not
committed.

---

## Repository structure

```
capstone/
├── README.md
├── requirements.txt               # version ranges
├── requirements-lock.txt          # exact versions used for the reported results
├── scripts/
│   ├── get_data.py                  # download + verify both data sources
│   ├── OSBAP_ML_Panel_columns.txt   # column-position sidecar for the panel pickle
│   ├── run_phase2.sh / .ps1         # the full pipeline (step 2)
│   └── check_results.py             # compare output/ against results/
├── src/                             # importable code, all unit-tested
│   ├── config.py                    # every path, date and setting: data dir, OOS start, sub-periods
│   ├── data.py                      # column-selective panel loader, target join
│   ├── targets.py                   # 1-, 3-, 12-month forward returns
│   ├── decompose.py                 # issuer-month statistics, within-issuer demeaning
│   ├── splits.py                    # expanding-window folds with the horizon embargo
│   ├── metrics.py                   # OOS R², winsorization
│   ├── harness.py                   # targets L0/L1/IC/L2, common sample, fit-predict-score per fold
│   ├── models.py                    # Elastic Net (coordinate descent), Random Forest (LightGBM)
│   ├── pipeline.py                  # OLS
│   ├── inference.py                 # modified Diebold–Mariano test
│   └── runlog.py                    # [verify] / [report] logging
├── exploratory/                     # runnable analyses; README.md: one entry per script
├── test/                            # unit tests for src/, synthetic data only
├── results/                         # committed: the numbers we report
├── data/                            # gitignored: raw data
└── output/                          # gitignored: everything a run regenerates
```

`exploratory/README.md` records, for every script, the question it asked and what it found,
including the earlier midterm and prototype scripts that the current pipeline grew out of.

---

## Conventions

- **No data in the repo.** `data/` and `output/` are gitignored. `results/` holds only the small
  CSVs behind reported numbers.
- **No absolute paths.** Every path comes from `src/config.py`.
- **Nothing enters `src/` untested.** One `test/test_<module>.py` per module, runnable without
  the data.
- **It has to run from a clean checkout.**
