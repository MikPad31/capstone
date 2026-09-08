# test/

Tests for `src/`. Nothing else.

One file per `src/` module, named `test_<module>.py`. Any working directory:

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
| `test_decompose.py` | `src/decompose.py` | Between/within sums of squares against hand-computed figures; the ANOVA identity off both a zero and a non-zero grand mean; residuals are exactly zero-mean; float group keys are cast to int64 and missing ones raise rather than being silently dropped |
| `test_splits.py` | `src/splits.py` | One fold checked against arithmetic done on paper; that the naive `s <= T-1` window leaks exactly `h-1` calendar months per fold while the embargoed one leaks none; the embargo gap is `h` exactly, not merely at least `h`; train windows expand and test blocks partition the whole usable OOS range; `h=1, refit_freq=1` reproduces the plain expanding window; impossible splits raise; `rows_for_months` matches on month whatever the date dtype and refuses an all-False mask |

`naive_folds` in `test_splits.py` is a deliberately wrong splitter, kept so the leakage
assertions can be watched firing. It never moves into `src/`.

