# test/

Tests for `src/`. Nothing else.

One file per `src/` module that runs without the data, named `test_<module>.py`: 8 files, 151
tests. `data.py` and `pipeline.py` need the real panel and are checked by the `[verify]` checks
in `exploratory/` instead; `runlog.py` only logs. Any working directory:

```bash
pytest test/                    # everything
pytest test/test_<module>.py    # one module
```

`__init__.py` and `conftest.py` both put the repo root on `sys.path` so `from src import ...`
resolves. Keep both: `__init__.py` also keeps test module names unique across directories,
and `conftest.py` is the one that survives `--import-mode=importlib`.

- **No data.** Synthetic inputs only — the panel is not in the repo and a test that needs
  it cannot run on a clean checkout.
- **Fast.** Seconds, not minutes. No fitting, no downloads.
- **Assert, do not print.** `[verify]` prints belong in `exploratory/`; here they are
  assertions.

| File | Guards | What it asserts |
|---|---|---|
| `test_config.py` | `src/config.py` | `OOS_START` is pinned at 2015-01 and gives valid embargoed folds with at least ten years of training at every horizon; `SUBPERIODS` tile `OOS_START`…`PANEL_END` with no gap or overlap, under unique names |
| `test_decompose.py` | `src/decompose.py` | Between/within sums of squares against hand-computed figures; the ANOVA identity off both a zero and a non-zero grand mean; residuals are exactly zero-mean; float group keys are cast to int64 and missing ones raise rather than being silently dropped |
| `test_harness.py` | `src/harness.py` | L1 = IC + L2 row by row; the common sample is the same mask for every level; the three R² benchmarks against hand arithmetic; `score_by_period` equals `score` on each slice whatever the date dtype, includes both bounds, scores an empty period as NaN, and leaves rows outside every period unscored |
| `test_inference.py` | `src/inference.py` | The Newey–West standard error against statsmodels at several lag lengths; the lag rule's floor and its overlap minimum of `h−1`; the loss differential collapses to one mean per month; the DM statistic is positive when the second forecast passed to `dm_test` is the better one; NaN inputs raise |
| `test_metrics.py` | `src/metrics.py` | Clip bounds against quantiles read off a five-row group; row order and the Series return type survive, both of which a per-group `apply` loses; pooling predictions across models is not the same as keying on the model; OOS R² against arithmetic done on paper and against the `2ck − k²` identity; misaligned, missing and zero-denominator inputs raise rather than scoring |
| `test_models.py` | `src/models.py` | The cross-product (Gram) elastic net against scikit-learn's `ElasticNet` across penalties and mixing ratios; accumulated moments add up to the full-sample ones, including in chunks with masks; the inner tuning split keeps the `h`-month embargo and refuses a window too short to split; the tuned Elastic Net recovers a planted signal, and the tuned Random Forest fits and predicts on a small synthetic panel |
| `test_splits.py` | `src/splits.py` | One fold checked against arithmetic done on paper; that the naive `s <= T-1` window leaks exactly `h-1` calendar months per fold while the embargoed one leaks none; the embargo gap is `h` exactly, not merely at least `h`; train windows expand and test blocks partition the whole usable OOS range; `h=1, refit_freq=1` reproduces the plain expanding window; impossible splits raise; `rows_for_months` matches on month whatever the date dtype and refuses an all-False mask |
| `test_targets.py` | `src/targets.py` | At `h=1` the forward return reproduces each input row exactly; longer horizons compound consecutive months, keep a negative gross return, and never compound across a gap or across bonds; the tail loses `h−1` months, not `h`; row order does not matter; duplicate bond-months, missing columns and non-positive horizons raise; label coverage counts bond-months |

`naive_folds` in `test_splits.py` is a deliberately wrong splitter, kept so the leakage
assertions can be watched firing. It never moves into `src/`.

