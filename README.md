# Capstone — corporate bond return predictability

DSE4101

> **Project description: TBD.** To be written once the conceptual framework and
> methodology are settled.

**Levels of decomposition** — working nomenclature, used throughout the code and subject
to change:

| Level | Target |
|---|---|
| **L0** | total return in excess of the risk-free rate (`retxrf`) |
| **L1** | credit return, excess over duration-matched Treasury (`retx`) |
| **L2** | `retx` demeaned within issuer-month |

---

## Data

Two sources, both from the Open Source Bond Asset Pricing project
(Dickerson–Nozawa–Robotti, <https://openbondassetpricing.com>). Neither is in this repo
and neither ever will be — see [Conventions](#conventions).

**1. OSBAP ML panel** — the predictor panel.

- `OSBAP_ML_Panel_Oct_2024.pkl` — 1,102,569 rows × 381 columns, ~3.11 GiB on disk.
  Bond-month observations, 19,768 bonds × 245 months, 2002-07 to 2022-11. Holds 341
  rank-rescaled predictors, identifiers (`date`, `cusip`, `issuer_cusip`, `ID`, `gvkey`,
  `permno`, `RATING_NUM`, …), and 27 shipped ML forecast columns.
- `OSBAP_ML_Panel_columns.txt` — position → column-name sidecar. **Do not discard it.**
  It is how we select columns without materializing the whole ~5.1 GB frame.

URL: [OSBAP Panel Data (2024)](https://openbondassetpricing.com/wp-content/uploads/2024/10/OSBAP_ML_Panel_Oct_2024.zip)

**2. DNR ML predictions** — realized returns and the authors' own model forecasts, used as
the replication benchmark.

- `predictions.parquet` — long format: one row per (`model_key`, `target`, bond, month).
  9 model keys × 3 targets. `signal_date` is forecast month *t*, `realized_return_date`
  is *t+1*; no additional lead/lag is needed.
- `README.txt` — ships alongside it and documents the model keys and target definitions.
  Authoritative; read it before assuming what a column means.

URL: [ML Forecasts and realized returns](https://openbondassetpricing.com/wp-content/uploads/2026/04/dnr_ml_predictions.zip)

### Getting the data

```bash
python scripts/get_data.py
```

> **Not written yet.** `scripts/get_data.py` is the intended entry point: it downloads
> both sources into `data/raw/`, creating the layout below, and verifies each file against
> its expected size and row count so a truncated download fails loudly instead of silently
> producing wrong numbers. Until it exists, download manually into this layout.

```
data/raw/
├── OSBAP_ML_Panel_Oct_2024.pkl
├── OSBAP_ML_Panel_columns.txt
└── dnr_ml_predictions/
    ├── predictions.parquet
    └── README.txt
```

If you already keep the panel somewhere else (it is 3 GB — you probably do not want a
second copy), set `DSE4101_DATA_DIR` to that folder and everything reads from there
instead:

```bash
export DSE4101_DATA_DIR=/path/to/your/existing/copy
```

### Verify your download

Before trusting any result, confirm the panel you loaded is the panel you expected:

- shape `(1102569, 381)`
- `date` spans 2002-07 to 2022-11, 245 unique months
- 19,768 unique `ID`

A file short by a few thousand rows will still load, still run, and still give you numbers.

---

## Setup

```bash
git clone https://github.com/MikPad31/capstone.git
cd capstone
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/get_data.py
```

Python 3.11+. The full panel costs roughly 5 GB as a materialized DataFrame, plus
overhead; the loader in `src/data.py` exists so you do not have to pay that.

---

## Repository structure

*Proposed structure only (WIP).*

```
capstone/
├── README.md
├── requirements.txt
├── .gitignore                  # data/, output/, __pycache__/, .ipynb_checkpoints/
├── scripts/
│   └── get_data.py             # downloads both sources into data/raw/
├── data/                       # gitignored. Never committed, no exceptions.
│   └── raw/
├── src/                        # importable, reusable code only
│   ├── __init__.py
│   ├── config.py               # the one place a path, a key, or a seed is written down
│   ├── data.py                 # load the panel and the predictions
│   ├── targets.py              # 1-, 3-, 12-month forward returns; overlap handling
│   ├── splits.py               # expanding-window train/test index generator
│   └── metrics.py              # OOS R² (both denominator conventions), winsorization
├── exploratory/                # EDA, replication, verification, Midterm work
│   └── README.md               # one line per script: what it asked, what it found
├── test/                       # tests for src/ only
│   ├── README.md               # one line per test file: what it guards
│   └── conftest.py             # puts the repo root on sys.path
└── output/                     # gitignored: figures and tables, all regenerable
```

**The `src/` vs `exploratory/` split is the important.** Scripts in `exploratory/`
are meant for exploratory work and producing quick results and can remain stale. Anything in `src/` is imported by something else and has
to keep working. A finding that starts as an exploratory script and eventually moves into `src/` is a normal workflow, do not skip the first step as much as possible.

`exploratory/` is where replication of published results, verification of our own data
checks, and other miscellaneous work.

`test/` is the third step: `exploratory/` → `src/` → `test/`. Verification in
`exploratory/` is print-based `[verify]` blocks you read yourself. Promotion to `src/`
is not done until those blocks are rewritten as assertions in `test/test_<module>.py`.

```bash
pytest test/
```

Tests read no data — the panel is not in the repo, so anything needing it cannot run on a
clean checkout. Use synthetic inputs: a hand-written month list, a ten-row frame. The suite
finishes in seconds. If a function cannot be tested without the panel, split the pure part
out of it.

---

## Conventions for working on repo

For everyone working in this repo:

- **No data in the repo.** `data/` and `output/` are gitignored.
- **No absolute paths anywhere.** Every path comes from `src/config.py`. If you find
  yourself typing `/Users/…` or `C:\…` into a script, that belongs in config instead.
- **Notebooks live in `exploratory/` only**, with outputs cleared before commit. Anything
  that will be rerun, or that a other people's work depends on, becomes a `.py`.
- **Nothing enters `src/` untested.** One `test/test_<module>.py` per `src/` module, and
  it runs without `data/`.
- **It has to run from a clean checkout.** No cells that only work after deleted cells, no
  state living in your kernel, no file that exists only on your machine.
